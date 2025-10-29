"""
Prueba Avanzada del Detector APT

Carga los modelos entrenados y los prueba con:
1. Grafos del test set
2. Grafos sintéticos de ataques conocidos
3. Evaluación de métricas (Precision, Recall, F1)
4. Visualización de resultados
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from causaldefend.models.spatiotemporal_detector import APTDetector
except ImportError:
    from models.spatiotemporal_detector import APTDetector


def load_detector(checkpoint_path: str = "models/detector.ckpt") -> APTDetector:
    """Cargar detector desde checkpoint"""
    print(f"\n{'='*80}")
    print("CARGANDO DETECTOR APT")
    print(f"{'='*80}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extraer hiperparámetros
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"\n✓ Hiperparámetros encontrados:")
        for k, v in hparams.items():
            print(f"  - {k}: {v}")
    else:
        hparams = {
            'in_channels': 64,
            'hidden_channels': 128,
            'embedding_dim': 64,
            'num_heads': 8,
            'num_layers': 3,
        }
        print(f"\n⚠️  Usando hiperparámetros por defecto")
    
    # Crear modelo con TODOS los parámetros del checkpoint
    model = APTDetector(
        in_channels=hparams.get('in_channels', 64),
        hidden_channels=hparams.get('hidden_channels', 128),
        embedding_dim=hparams.get('embedding_dim', 64),
        gru_hidden_dim=hparams.get('gru_hidden_dim', hparams.get('embedding_dim', 64)),  # ← IMPORTANTE
        num_heads=hparams.get('num_heads', 8),
        num_layers=hparams.get('num_layers', 3),
        gru_layers=hparams.get('gru_layers', 2),
        dropout=hparams.get('dropout', 0.3),
    )
    
    # Cargar pesos
    if 'state_dict' in checkpoint:
        # PyTorch Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remover prefijo 'detector.' si existe
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('detector.'):
                new_state_dict[k.replace('detector.', '')] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"\n✓ Modelo cargado exitosamente")
    print(f"✓ Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_test_graphs(data_dir: str = "data/processed/test") -> List[Dict]:
    """Cargar grafos del test set"""
    print(f"\n{'='*80}")
    print("CARGANDO TEST SET")
    print(f"{'='*80}")
    
    data_path = Path(data_dir)
    graph_files = sorted(list(data_path.glob("graph_*.pkl")))
    
    if len(graph_files) == 0:
        print(f"\n⚠️  No se encontraron grafos en {data_dir}")
        return []
    
    graphs = []
    for graph_file in graph_files:
        graph_id = graph_file.stem.replace("graph_", "")
        
        # Cargar grafo
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        
        # Cargar features
        feature_file = data_path / f"features_{graph_id}.npy"
        features = np.load(feature_file)
        
        # Cargar label
        label_file = data_path / f"label_{graph_id}.json"
        with open(label_file, "r") as f:
            label_data = json.load(f)
        
        # Convertir a PyG Data
        edge_list = list(graph.edges())
        if len(edge_list) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        pyg_data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index
        )
        
        graphs.append({
            'data': pyg_data,
            'label': label_data['is_attack'],
            'graph_id': graph_id,
            'attack_type': label_data.get('attack_type', 'unknown'),
        })
    
    print(f"\n✓ Cargados {len(graphs)} grafos del test set")
    
    # Estadísticas
    num_attacks = sum(1 for g in graphs if g['label'])
    num_normal = len(graphs) - num_attacks
    print(f"  - Ataques: {num_attacks}")
    print(f"  - Normales: {num_normal}")
    
    return graphs


def evaluate_detector(model: APTDetector, graphs: List[Dict]) -> Dict:
    """Evaluar detector en grafos"""
    print(f"\n{'='*80}")
    print("EVALUANDO DETECTOR")
    print(f"{'='*80}")
    
    predictions = []
    labels = []
    scores = []
    
    with torch.no_grad():
        for i, graph_dict in enumerate(graphs):
            data = graph_dict['data']
            label = graph_dict['label']
            
            # Forward pass
            graph_embedding, edge_probs = model(data.x, data.edge_index)
            
            # Reconstruir features
            x_recon = model.feature_decoder(graph_embedding)
            
            # Compute anomaly score
            recon_error = F.mse_loss(x_recon, data.x.mean(0, keepdim=True), reduction='mean')
            anomaly_score = recon_error.item()
            
            predictions.append(anomaly_score)
            labels.append(1 if label else 0)
            scores.append(anomaly_score)
            
            # Mostrar algunos ejemplos
            if i < 5 or (i < 10 and label):
                status = "🚨 ATAQUE" if label else "✓ Normal"
                print(f"\nGrafo {graph_dict['graph_id']}: {status}")
                print(f"  Tipo: {graph_dict['attack_type']}")
                print(f"  Nodos: {data.x.size(0)}")
                print(f"  Aristas: {data.edge_index.size(1)}")
                print(f"  Anomaly Score: {anomaly_score:.4f}")
    
    # Calcular threshold óptimo (usando el mediano)
    threshold = np.median(predictions)
    
    # Clasificar
    binary_preds = [1 if score > threshold else 0 for score in predictions]
    
    # Calcular métricas
    tp = sum(1 for pred, label in zip(binary_preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(binary_preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(binary_preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(binary_preds, labels) if pred == 0 and label == 1)
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*80}")
    print("RESULTADOS DE EVALUACIÓN")
    print(f"{'='*80}")
    print(f"\n📊 Métricas (threshold={threshold:.4f}):")
    print(f"  • Accuracy:  {accuracy*100:.2f}%")
    print(f"  • Precision: {precision*100:.2f}%")
    print(f"  • Recall:    {recall*100:.2f}%")
    print(f"  • F1 Score:  {f1*100:.2f}%")
    
    print(f"\n📈 Matriz de Confusión:")
    print(f"                 Predicho")
    print(f"                 Neg    Pos")
    print(f"         Neg  │  {tn:3d}    {fp:3d}")
    print(f"  Real   Pos  │  {fn:3d}    {tp:3d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'scores': scores,
        'labels': labels,
    }


def create_synthetic_attack() -> Data:
    """Crear un grafo sintético de ataque para prueba"""
    print(f"\n{'='*80}")
    print("CREANDO ATAQUE SINTÉTICO")
    print(f"{'='*80}")
    
    # Simular un ataque de lateral movement
    num_nodes = 15
    
    # Features aleatorias con patrón de ataque
    features = np.random.randn(num_nodes, 64).astype(np.float32)
    
    # Añadir "señales" de ataque en algunos nodos
    attack_nodes = [3, 7, 10, 12]
    for node in attack_nodes:
        features[node] += 2.0  # Anomalía en features
    
    # Crear cadena de ataque
    edges = [
        (0, 1), (1, 2), (2, 3),  # Initial access
        (3, 4), (3, 5), (3, 6),  # Spread
        (6, 7), (7, 8),          # Lateral movement
        (8, 9), (9, 10),         # Privilege escalation
        (10, 11), (10, 12),      # Data access
        (12, 13), (13, 14),      # Exfiltration
    ]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index
    )
    
    print(f"\n✓ Ataque sintético creado:")
    print(f"  • Nodos: {num_nodes}")
    print(f"  • Aristas: {len(edges)}")
    print(f"  • Nodos comprometidos: {len(attack_nodes)}")
    print(f"  • Patrón: Initial Access → Lateral Movement → Exfiltration")
    
    return data


def test_synthetic_attack(model: APTDetector, data: Data):
    """Probar detector con ataque sintético"""
    print(f"\n{'='*80}")
    print("DETECTANDO ATAQUE SINTÉTICO")
    print(f"{'='*80}")
    
    with torch.no_grad():
        # Forward pass
        graph_embedding, edge_probs = model(data.x, data.edge_index)
        
        # Reconstruir features
        x_recon = model.feature_decoder(graph_embedding)
        
        # Anomaly score
        recon_error = F.mse_loss(x_recon, data.x.mean(0, keepdim=True), reduction='mean')
        anomaly_score = recon_error.item()
        
        # Analizar edge probabilities
        edge_prob_mean = edge_probs.mean().item()
        edge_prob_std = edge_probs.std().item()
        
        print(f"\n🔍 Análisis del Ataque:")
        print(f"  • Anomaly Score: {anomaly_score:.4f}")
        print(f"  • Edge Prob Media: {edge_prob_mean:.4f}")
        print(f"  • Edge Prob Std: {edge_prob_std:.4f}")
        
        # Determinar si es ataque
        threshold = 0.5  # Ajustar según necesidad
        is_attack = anomaly_score > threshold
        
        print(f"\n{'🚨 ATAQUE DETECTADO' if is_attack else '✓ Normal'}")
        print(f"  Confianza: {abs(anomaly_score - threshold) / threshold * 100:.1f}%")
        
        # Análisis de nodos más sospechosos
        node_scores = (data.x - x_recon.expand_as(data.x)).pow(2).mean(1)
        top_suspicious = torch.topk(node_scores, k=min(5, len(node_scores)))
        
        print(f"\n🎯 Top 5 Nodos Sospechosos:")
        for rank, (score, idx) in enumerate(zip(top_suspicious.values, top_suspicious.indices), 1):
            print(f"  {rank}. Nodo {idx.item()}: score={score.item():.4f}")


def main():
    """Función principal"""
    print("\n" + "="*80)
    print(" "*20 + "PRUEBA AVANZADA DEL DETECTOR APT")
    print("="*80)
    
    # 1. Cargar modelo
    try:
        model = load_detector("models/detector.ckpt")
    except Exception as e:
        print(f"\n❌ Error cargando modelo: {e}")
        print("\n💡 Asegúrate de haber entrenado el modelo primero:")
        print("   python scripts/train_detector.py --epochs 10")
        return
    
    # 2. Cargar y evaluar en test set
    test_graphs = load_test_graphs("data/processed/test")
    
    if len(test_graphs) > 0:
        results = evaluate_detector(model, test_graphs)
        
        # Guardar resultados
        results_file = Path("models/evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'threshold': results['threshold'],
                'confusion_matrix': results['confusion_matrix'],
            }, f, indent=2)
        print(f"\n✓ Resultados guardados en: {results_file}")
    
    # 3. Probar con ataque sintético
    synthetic_attack = create_synthetic_attack()
    test_synthetic_attack(model, synthetic_attack)
    
    print(f"\n{'='*80}")
    print(" "*25 + "PRUEBA COMPLETADA")
    print(f"{'='*80}")
    print("\n📝 Próximos pasos:")
    print("  1. Ajustar threshold basado en resultados")
    print("  2. Probar con datos reales de auditoría")
    print("  3. Integrar con pipeline de detección completo")
    print("  4. Implementar explicaciones causales")
    print()


if __name__ == "__main__":
    main()
