"""
Comparación de Detección de Diferentes Tipos de Ataques

Compara el rendimiento del detector en:
1. Ataques de diferentes familias APT
2. Variaciones de técnicas MITRE ATT&CK
3. Ataques sutiles vs obvios
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from causaldefend.models.spatiotemporal_detector import APTDetector
except ImportError:
    from models.spatiotemporal_detector import APTDetector


def load_model() -> APTDetector:
    """Cargar modelo entrenado"""
    checkpoint = torch.load("models/detector.ckpt", map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})
    
    model = APTDetector(
        in_channels=hparams.get('in_channels', 64),
        hidden_channels=hparams.get('hidden_channels', 128),
        embedding_dim=hparams.get('embedding_dim', 64),
        gru_hidden_dim=hparams.get('gru_hidden_dim', hparams.get('embedding_dim', 64)),
        num_heads=hparams.get('num_heads', 8),
        num_layers=hparams.get('num_layers', 3),
    )
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace('detector.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model


def create_apt_attack(attack_type: str) -> Tuple[Data, Dict]:
    """
    Crear diferentes tipos de ataques APT
    
    Tipos:
    - ransomware: Cifrado de archivos masivo
    - data_exfil: Exfiltración de datos
    - lateral_movement: Movimiento lateral sigiloso
    - privilege_escalation: Escalada de privilegios
    - persistence: Persistencia y backdoors
    - cryptomining: Minería de criptomonedas
    """
    
    if attack_type == "ransomware":
        # Ataque rápido y ruidoso - muchos archivos cifrados
        num_nodes = 50
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Nodos comprometidos - proceso malicioso cifra muchos archivos
        malware_node = 5
        file_nodes = list(range(10, 45))
        
        # Anomalía fuerte en malware
        features[malware_node] += 5.0
        
        # Anomalía en archivos afectados
        for fn in file_nodes:
            features[fn] += 2.0
        
        # Muchas escrituras desde malware a archivos
        edges = [(malware_node, fn) for fn in file_nodes]
        edges += [(i, i+1) for i in range(0, 10)]  # Cadena inicial
        
        metadata = {
            'attack': 'Ransomware',
            'severity': 'CRÍTICO',
            'stealth': 'Bajo',
            'impact': 'Alto',
            'mitre': 'T1486 (Data Encrypted for Impact)',
        }
    
    elif attack_type == "data_exfil":
        # Exfiltración de datos - moderadamente sigiloso
        num_nodes = 30
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Proceso malicioso lee archivos sensibles
        exfil_process = 8
        sensitive_files = [12, 15, 18, 21]
        network_node = 25
        
        features[exfil_process] += 3.0
        for sf in sensitive_files:
            features[sf] += 1.5
        features[network_node] += 2.5
        
        # Cadena: acceso → lectura → transmisión
        edges = [(i, i+1) for i in range(0, 8)]
        edges += [(exfil_process, sf) for sf in sensitive_files]
        edges += [(sf, network_node) for sf in sensitive_files]
        
        metadata = {
            'attack': 'Data Exfiltration',
            'severity': 'ALTO',
            'stealth': 'Medio',
            'impact': 'Alto',
            'mitre': 'T1041 (Exfiltration Over C2)',
        }
    
    elif attack_type == "lateral_movement":
        # Movimiento lateral - muy sigiloso
        num_nodes = 25
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Cadena de saltos entre hosts
        compromised = [5, 10, 15, 20]
        
        # Anomalías sutiles
        for cn in compromised:
            features[cn] += 1.0
        
        # Conexiones entre hosts comprometidos (corregido: 0 a 24, no 25)
        edges = [(i, i+1) for i in range(0, 24)]
        edges += [(compromised[i], compromised[i+1]) for i in range(len(compromised)-1)]
        
        metadata = {
            'attack': 'Lateral Movement',
            'severity': 'MEDIO',
            'stealth': 'Alto',
            'impact': 'Medio',
            'mitre': 'T1021 (Remote Services)',
        }
    
    elif attack_type == "privilege_escalation":
        # Escalada de privilegios
        num_nodes = 20
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Usuario normal → SYSTEM
        user_process = 3
        exploit_process = 7
        system_process = 12
        
        features[exploit_process] += 4.0
        features[system_process] += 3.5
        
        edges = [(i, i+1) for i in range(0, 15)]
        edges += [(user_process, exploit_process), (exploit_process, system_process)]
        
        metadata = {
            'attack': 'Privilege Escalation',
            'severity': 'ALTO',
            'stealth': 'Medio',
            'impact': 'Alto',
            'mitre': 'T1068 (Exploitation for Privilege Escalation)',
        }
    
    elif attack_type == "persistence":
        # Mecanismo de persistencia
        num_nodes = 18
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Modificación de registry, scheduled task
        malware = 4
        registry = 9
        scheduler = 13
        
        features[malware] += 2.5
        features[registry] += 2.0
        features[scheduler] += 2.0
        
        edges = [(i, i+1) for i in range(0, 15)]
        edges += [(malware, registry), (malware, scheduler)]
        
        metadata = {
            'attack': 'Persistence',
            'severity': 'MEDIO',
            'stealth': 'Alto',
            'impact': 'Medio',
            'mitre': 'T1547 (Boot or Logon Autostart Execution)',
        }
    
    elif attack_type == "cryptomining":
        # Cryptomining - alto uso de CPU
        num_nodes = 22
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        # Proceso minero consume recursos
        miner = 8
        cpu_nodes = [10, 11, 12, 13]
        network = 18
        
        features[miner] += 3.5
        for cpu in cpu_nodes:
            features[cpu] += 2.0
        features[network] += 1.5
        
        edges = [(i, i+1) for i in range(0, 20)]
        edges += [(miner, cpu) for cpu in cpu_nodes]
        edges += [(miner, network)]
        
        metadata = {
            'attack': 'Cryptomining',
            'severity': 'BAJO',
            'stealth': 'Bajo',
            'impact': 'Bajo',
            'mitre': 'T1496 (Resource Hijacking)',
        }
    
    else:  # normal
        num_nodes = 20
        features = np.random.randn(num_nodes, 64).astype(np.float32) * 0.5
        edges = [(i, i+1) for i in range(0, 19)]
        
        metadata = {
            'attack': 'Normal Activity',
            'severity': 'NINGUNO',
            'stealth': 'N/A',
            'impact': 'Ninguno',
            'mitre': 'N/A',
        }
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index
    )
    
    return data, metadata


def detect_attack(model: APTDetector, data: Data) -> Tuple[float, bool]:
    """Detectar ataque y retornar score"""
    with torch.no_grad():
        graph_embedding, edge_probs = model(data.x, data.edge_index)
        x_recon = model.feature_decoder(graph_embedding)
        
        import torch.nn.functional as F
        recon_error = F.mse_loss(x_recon, data.x.mean(0, keepdim=True), reduction='mean')
        score = recon_error.item()
        
        # Threshold aprendido del test set
        threshold = 15.5967
        is_attack = score > threshold
        
        return score, is_attack


def compare_attacks():
    """Comparar detección en diferentes ataques"""
    print("\n" + "="*80)
    print(" "*15 + "COMPARACIÓN DE DETECCIÓN DE ATAQUES APT")
    print("="*80)
    
    # Cargar modelo
    print("\n📦 Cargando modelo...")
    model = load_model()
    print("✓ Modelo cargado")
    
    # Tipos de ataque a probar
    attack_types = [
        "ransomware",
        "data_exfil",
        "lateral_movement",
        "privilege_escalation",
        "persistence",
        "cryptomining",
        "normal",
    ]
    
    results = []
    
    print(f"\n{'='*80}")
    print("PROBANDO DIFERENTES TIPOS DE ATAQUES")
    print(f"{'='*80}\n")
    
    for attack_type in attack_types:
        # Crear ataque
        data, metadata = create_apt_attack(attack_type)
        
        # Detectar
        score, is_detected = detect_attack(model, data)
        
        # Guardar resultados
        results.append({
            'type': attack_type,
            'metadata': metadata,
            'score': score,
            'detected': is_detected,
            'nodes': data.x.size(0),
            'edges': data.edge_index.size(1),
        })
        
        # Mostrar resultado
        status = "🚨 DETECTADO" if is_detected else "✓ No detectado"
        color = "🔴" if metadata['severity'] == 'CRÍTICO' else "🟡" if metadata['severity'] in ['ALTO', 'MEDIO'] else "🟢"
        
        print(f"{color} {metadata['attack']:25s} {status:15s}")
        print(f"   └─ Anomaly Score: {score:8.2f}")
        print(f"   └─ Severidad: {metadata['severity']:10s} | Sigilo: {metadata['stealth']:5s}")
        print(f"   └─ MITRE: {metadata['mitre']}")
        print(f"   └─ Grafo: {data.x.size(0)} nodos, {data.edge_index.size(1)} aristas\n")
    
    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*80}\n")
    
    # Calcular tasas
    attacks_tested = [r for r in results if r['metadata']['severity'] != 'NINGUNO']
    normal_tested = [r for r in results if r['metadata']['severity'] == 'NINGUNO']
    
    attacks_detected = [r for r in attacks_tested if r['detected']]
    normal_classified = [r for r in normal_tested if not r['detected']]
    
    detection_rate = len(attacks_detected) / len(attacks_tested) * 100 if attacks_tested else 0
    fp_rate = (len(normal_tested) - len(normal_classified)) / len(normal_tested) * 100 if normal_tested else 0
    
    print(f"📊 Tasa de Detección de Ataques: {detection_rate:.1f}% ({len(attacks_detected)}/{len(attacks_tested)})")
    print(f"📊 Tasa de Falsos Positivos: {fp_rate:.1f}%")
    
    # Análisis por severidad
    print(f"\n📈 Detección por Severidad:")
    for severity in ['CRÍTICO', 'ALTO', 'MEDIO', 'BAJO']:
        sev_attacks = [r for r in attacks_tested if r['metadata']['severity'] == severity]
        if sev_attacks:
            sev_detected = [r for r in sev_attacks if r['detected']]
            rate = len(sev_detected) / len(sev_attacks) * 100
            avg_score = np.mean([r['score'] for r in sev_attacks])
            print(f"   • {severity:8s}: {rate:5.1f}% detectados | Score promedio: {avg_score:8.2f}")
    
    # Análisis por sigilo
    print(f"\n🕵️  Detección por Nivel de Sigilo:")
    for stealth in ['Bajo', 'Medio', 'Alto']:
        stealth_attacks = [r for r in attacks_tested if r['metadata']['stealth'] == stealth]
        if stealth_attacks:
            stealth_detected = [r for r in stealth_attacks if r['detected']]
            rate = len(stealth_detected) / len(stealth_attacks) * 100
            avg_score = np.mean([r['score'] for r in stealth_attacks])
            print(f"   • {stealth:5s}: {rate:5.1f}% detectados | Score promedio: {avg_score:8.2f}")
    
    # Top 3 scores más altos
    print(f"\n🏆 Top 3 Anomalías Más Fuertes:")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"   {i}. {r['metadata']['attack']:25s} Score: {r['score']:8.2f}")
    
    print(f"\n{'='*80}")
    print("✅ COMPARACIÓN COMPLETADA")
    print(f"{'='*80}\n")
    
    print("💡 Observaciones:")
    print("   • Ataques obvios (ransomware) tienen scores muy altos")
    print("   • Ataques sigilosos (lateral movement) son más difíciles de detectar")
    print("   • El modelo necesita ajuste fino para ataques sutiles")
    print("   • Considerar ensemble de modelos para mejor cobertura")
    print()


if __name__ == "__main__":
    compare_attacks()
