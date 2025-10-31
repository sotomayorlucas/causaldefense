"""
Simple Detection Demo

Este script demuestra detección de APT usando el modelo entrenado
de forma directa, sin usar el pipeline complejo.
"""

import torch
import torch.nn.functional as F
import json
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data

# Import detector
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from causaldefend.models.spatiotemporal_detector import APTDetector
except ImportError:
    from models.spatiotemporal_detector import APTDetector


def create_sample_attack_graph():
    """Crea un grafo de ataque sintético"""
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    # Simular un ataque APT
    # 1. Initial Access
    G.add_edge(0, 1, type="process_spawn", label="phishing_email")  # Usuario abre email malicioso
    
    # 2. Execution
    G.add_edge(1, 2, type="process_spawn", label="malware_download")  # Descarga malware
    
    # 3. Persistence
    G.add_edge(2, 3, type="file_write", label="registry_mod")  # Modifica registro para persistencia
    
    # 4. Privilege Escalation
    G.add_edge(3, 4, type="process_spawn", label="exploit")  # Explota vulnerabilidad
    
    # 5. Defense Evasion
    G.add_edge(4, 5, type="process_modify", label="disable_av")  # Desactiva antivirus
    
    # 6. Credential Access
    G.add_edge(5, 6, type="file_read", label="credential_dump")  # Roba credenciales
    
    # 7. Discovery
    G.add_edge(6, 7, type="network_scan", label="network_enum")  # Escanea red
    
    # 8. Lateral Movement
    G.add_edge(7, 8, type="remote_exec", label="lateral_move")  # Se mueve lateralmente
    
    # 9. Collection
    G.add_edge(8, 9, type="file_read", label="data_collect")  # Recolecta datos
    
    # 10. Exfiltration
    G.add_edge(9, 10, type="network_conn", label="exfiltrate")  # Exfiltra datos
    
    # Agregar algunos nodos/aristas normales para hacer más realista
    for i in range(11, 20):
        G.add_edge(i, i+1, type="normal_activity", label="benign")
    
    return G


def create_normal_graph():
    """Crea un grafo de actividad normal"""
    G = nx.DiGraph()
    
    # Actividad normal de un usuario
    for i in range(15):
        if i % 3 == 0:
            G.add_edge(i, i+1, type="process_spawn", label="normal_app")
        elif i % 3 == 1:
            G.add_edge(i, i+1, type="file_read", label="document_open")
        else:
            G.add_edge(i, i+1, type="network_conn", label="web_browsing")
    
    return G


def graph_to_pyg_data(G, num_features=64):
    """Convierte un networkx graph a PyG Data"""
    # Crear mapeo de nodos
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Edge index
    edge_index = torch.tensor([
        [node_mapping[src], node_mapping[dst]]
        for src, dst in G.edges()
    ], dtype=torch.long).t().contiguous()
    
    # Node features (aleatorias por ahora)
    x = torch.randn(len(G.nodes()), num_features)
    
    # Edge features (una característica por tipo de arista)
    edge_types = [G[src][dst].get('type', 'unknown') for src, dst in G.edges()]
    edge_attr = torch.randn(len(edge_types), 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_detector(checkpoint_path):
    """Carga el detector desde checkpoint"""
    print(f"Cargando detector desde: {checkpoint_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extraer hiperparámetros
    hparams = checkpoint.get("hyper_parameters", {})
    print(f"Hiperparámetros: {hparams}")
    
    # Agregar gru_hidden_dim si no existe (para compatibilidad con checkpoints antiguos)
    if 'gru_hidden_dim' not in hparams:
        # Inferir gru_hidden_dim del estado del modelo
        gru_weight_shape = checkpoint["state_dict"]["detector.temporal_encoder.gru.weight_ih_l0"].shape
        gru_hidden_dim = gru_weight_shape[0] // 3  # Dividir por 3 (update, reset, new gates)
        hparams['gru_hidden_dim'] = gru_hidden_dim
        print(f"  → Inferido gru_hidden_dim={gru_hidden_dim} desde checkpoint")
    
    # Crear modelo con hiperparámetros
    model = APTDetector(**hparams)
    
    # Cargar state_dict (removiendo prefijo "detector." si existe)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("detector.", "")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print("✓ Modelo cargado exitosamente")
    return model


def detect_attack(model, graph_data, threshold=12.4):
    """Detecta si un grafo es un ataque"""
    with torch.no_grad():
        # Forward pass
        graph_embedding, edge_probs = model(
            graph_data.x,
            graph_data.edge_index,
            batch=None  # Single graph
        )
        
        # Reconstruir features
        x_recon = model.feature_decoder(graph_embedding)
        
        # Calcular anomaly score (error de reconstrucción)
        anomaly_score = F.mse_loss(
            x_recon, 
            graph_data.x.mean(0, keepdim=True), 
            reduction='mean'
        )
        
        is_attack = anomaly_score.item() > threshold
        
        return {
            "is_attack": is_attack,
            "anomaly_score": anomaly_score.item(),
            "edge_probs_mean": edge_probs.mean().item(),
            "edge_probs_std": edge_probs.std().item()
        }


def main():
    print("="*80)
    print("CAUSALDEFEND - DEMO SIMPLE DE DETECCIÓN")
    print("="*80)
    print()
    
    # Cargar modelo
    checkpoint_path = Path("models/detector.ckpt")
    if not checkpoint_path.exists():
        print(f"❌ No se encontró el checkpoint en: {checkpoint_path}")
        return
    
    model = load_detector(checkpoint_path)
    print()
    
    # Crear grafos de ejemplo
    print("Creando grafos de ejemplo...")
    attack_graph = create_sample_attack_graph()
    normal_graph = create_normal_graph()
    
    attack_data = graph_to_pyg_data(attack_graph)
    normal_data = graph_to_pyg_data(normal_graph)
    
    print(f"  - Grafo de ataque: {len(attack_graph.nodes())} nodos, {len(attack_graph.edges())} aristas")
    print(f"  - Grafo normal: {len(normal_graph.nodes())} nodos, {len(normal_graph.edges())} aristas")
    print()
    
    # Detectar ataque
    print("-"*80)
    print("DETECCIÓN DE ATAQUE SINTÉTICO")
    print("-"*80)
    
    attack_result = detect_attack(model, attack_data)
    print(f"🚨 Anomaly Score: {attack_result['anomaly_score']:.2f}")
    print(f"   Edge Prob (mean): {attack_result['edge_probs_mean']:.4f}")
    print(f"   Edge Prob (std): {attack_result['edge_probs_std']:.4f}")
    
    if attack_result['is_attack']:
        print(f"   ✅ ATAQUE DETECTADO")
    else:
        print(f"   ❌ NO DETECTADO (threshold=12.4)")
    print()
    
    # Detectar actividad normal
    print("-"*80)
    print("DETECCIÓN DE ACTIVIDAD NORMAL")
    print("-"*80)
    
    normal_result = detect_attack(model, normal_data)
    print(f"✓ Anomaly Score: {normal_result['anomaly_score']:.2f}")
    print(f"   Edge Prob (mean): {normal_result['edge_probs_mean']:.4f}")
    print(f"   Edge Prob (std): {normal_result['edge_probs_std']:.4f}")
    
    if normal_result['is_attack']:
        print(f"   ⚠️ FALSO POSITIVO")
    else:
        print(f"   ✅ CORRECTAMENTE CLASIFICADO COMO NORMAL")
    print()
    
    print("="*80)
    print("DEMO COMPLETADA")
    print("="*80)
    print()
    print("💡 Próximos pasos:")
    print("   1. Usa 'python examples/test_detector_advanced.py' para evaluación completa")
    print("   2. Usa 'python examples/dashboard.py' para ver métricas del modelo")
    print("   3. Usa 'python examples/compare_apt_detection.py' para comparar diferentes ataques")


if __name__ == "__main__":
    main()
