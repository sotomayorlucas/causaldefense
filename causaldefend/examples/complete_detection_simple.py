"""
Complete Detection Example (Simplified)

Demonstrates APT detection pipeline with synthetic graph data.
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for stdout (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from causaldefend.models.spatiotemporal_detector import APTDetector
except ImportError:
    from models.spatiotemporal_detector import APTDetector


def create_attack_graph():
    """Crea un grafo de ataque APT realista"""
    G = nx.DiGraph()
    
    # Simular cadena de ataque MITRE ATT&CK
    # Initial Access
    G.add_edge(0, 1, type="phishing", label="email_open")
    
    # Execution
    G.add_edge(1, 2, type="malware_exec", label="payload_download")
    G.add_edge(2, 3, type="code_exec", label="reverse_shell")
    
    # Persistence
    G.add_edge(3, 4, type="registry_mod", label="autorun_key")
    
    # Privilege Escalation
    G.add_edge(4, 5, type="exploit", label="kernel_exploit")
    
    # Defense Evasion
    G.add_edge(5, 6, type="process_injection", label="hide_process")
    
    # Credential Access
    G.add_edge(6, 7, type="credential_dump", label="lsass_dump")
    
    # Discovery
    G.add_edge(7, 8, type="network_scan", label="subnet_scan")
    
    # Lateral Movement
    G.add_edge(8, 9, type="remote_exec", label="psexec")
    
    # Collection
    G.add_edge(9, 10, type="file_copy", label="sensitive_data")
    
    # Exfiltration
    G.add_edge(10, 11, type="exfil", label="c2_upload")
    
    # Agregar nodos normales
    for i in range(12, 20):
        G.add_edge(i, i+1, type="normal", label="benign_activity")
    
    return G


def create_normal_graph():
    """Crea un grafo de actividad normal"""
    G = nx.DiGraph()
    
    # Actividad normal de usuario
    for i in range(25):
        if i % 4 == 0:
            G.add_edge(i, i+1, type="file_access", label="document")
        elif i % 4 == 1:
            G.add_edge(i, i+1, type="web_browse", label="https")
        elif i % 4 == 2:
            G.add_edge(i, i+1, type="app_launch", label="office")
        else:
            G.add_edge(i, i+1, type="network", label="email_sync")
    
    return G


def graph_to_pyg(G, num_features=64):
    """Convierte networkx graph a PyG Data"""
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    
    edge_index = torch.tensor([
        [node_mapping[src], node_mapping[dst]]
        for src, dst in G.edges()
    ], dtype=torch.long).t().contiguous()
    
    x = torch.randn(len(G.nodes()), num_features)
    edge_attr = torch.randn(len(G.edges()), 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def load_model(checkpoint_path):
    """Carga el modelo con configuración automática"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint.get("hyper_parameters", {})
    
    # Inferir gru_hidden_dim si no existe
    if 'gru_hidden_dim' not in hparams:
        gru_key = None
        for key in checkpoint["state_dict"].keys():
            if "temporal_encoder.gru.weight_ih_l0" in key:
                gru_key = key
                break
        
        if gru_key:
            gru_weight_shape = checkpoint["state_dict"][gru_key].shape
            hparams['gru_hidden_dim'] = gru_weight_shape[0] // 3
    
    model = APTDetector(**hparams)
    
    # Cargar state_dict (removiendo prefijo "detector.")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("detector.", "")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model, hparams


def detect(model, graph_data, threshold=12.4):
    """Detecta anomalías en el grafo"""
    with torch.no_grad():
        graph_embedding, edge_probs = model(
            graph_data.x,
            graph_data.edge_index,
            batch=None
        )
        
        x_recon = model.feature_decoder(graph_embedding)
        anomaly_score = F.mse_loss(
            x_recon,
            graph_data.x.mean(0, keepdim=True),
            reduction='mean'
        )
        
        return {
            "anomaly_score": anomaly_score.item(),
            "is_attack": anomaly_score.item() > threshold,
            "edge_probs_mean": edge_probs.mean().item(),
            "edge_probs_std": edge_probs.std().item(),
            "confidence": (anomaly_score.item() / threshold) if anomaly_score.item() > threshold else 1.0
        }


def main():
    print("="*80)
    print("CAUSALDEFEND - COMPLETE APT DETECTION PIPELINE")
    print("="*80)
    print()
    
    # Step 1: Cargar modelo
    print("Step 1: Loading APT Detector Model")
    print("-"*80)
    
    checkpoint_path = Path("models/detector.ckpt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    model, hparams = load_model(checkpoint_path)
    print(f"OK Model loaded successfully")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Hidden channels: {hparams.get('hidden_channels', 'N/A')}")
    print(f"   - GRU hidden dim: {hparams.get('gru_hidden_dim', 'N/A')}")
    print()
    
    # Step 2: Crear grafos de prueba
    print("Step 2: Creating Test Provenance Graphs")
    print("-"*80)
    
    attack_nx = create_attack_graph()
    normal_nx = create_normal_graph()
    
    attack_pyg = graph_to_pyg(attack_nx)
    normal_pyg = graph_to_pyg(normal_nx)
    
    print(f"OK Created synthetic graphs")
    print(f"   - Attack graph: {len(attack_nx.nodes())} nodes, {len(attack_nx.edges())} edges")
    print(f"   - Normal graph: {len(normal_nx.nodes())} nodes, {len(normal_nx.edges())} edges")
    print()
    
    # Step 3: Detectar ataque
    print("Step 3: Detecting APT Attack Pattern")
    print("-"*80)
    
    attack_result = detect(model, attack_pyg)
    
    print(f"Anomaly Score: {attack_result['anomaly_score']:.4f}")
    print(f"Detection: {'ATTACK DETECTED' if attack_result['is_attack'] else 'NOT DETECTED'}")
    print(f"Confidence: {attack_result['confidence']:.2%}")
    print(f"Edge Reconstruction: {attack_result['edge_probs_mean']:.4f} ± {attack_result['edge_probs_std']:.4f}")
    print()
    
    # Step 4: Detectar actividad normal
    print("Step 4: Analyzing Normal Activity")
    print("-"*80)
    
    normal_result = detect(model, normal_pyg)
    
    print(f"Anomaly Score: {normal_result['anomaly_score']:.4f}")
    print(f"Detection: {'FALSE POSITIVE' if normal_result['is_attack'] else 'CORRECTLY CLASSIFIED AS NORMAL'}")
    print(f"Confidence: {normal_result['confidence']:.2%}")
    print(f"Edge Reconstruction: {normal_result['edge_probs_mean']:.4f} ± {normal_result['edge_probs_std']:.4f}")
    print()
    
    # Step 5: Resumen
    print("="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print()
    print(f"Threshold: 12.4066")
    print(f"Attack Score: {attack_result['anomaly_score']:.4f} - {'DETECTED' if attack_result['is_attack'] else 'MISSED'}")
    print(f"Normal Score: {normal_result['anomaly_score']:.4f} - {'FALSE ALARM' if normal_result['is_attack'] else 'CORRECT'}")
    print()
    
    if attack_result['is_attack'] and not normal_result['is_attack']:
        print("RESULT: Perfect detection - Attack identified, normal activity not flagged")
    elif attack_result['is_attack'] and normal_result['is_attack']:
        print("RESULT: High sensitivity - Attack detected but with false positive")
    elif not attack_result['is_attack'] and not normal_result['is_attack']:
        print("RESULT: Low sensitivity - Attack missed (threshold too high)")
    else:
        print("RESULT: Inverted detection - Critical error")
    
    print()
    print("="*80)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("="*80)
    print()
    print("Next Steps:")
    print("  1. Run 'python examples/test_detector_advanced.py' for full evaluation")
    print("  2. Run 'python examples/dashboard.py' for performance metrics")
    print("  3. Run 'python examples/compare_apt_detection.py' for attack comparisons")
    print()


if __name__ == "__main__":
    main()
