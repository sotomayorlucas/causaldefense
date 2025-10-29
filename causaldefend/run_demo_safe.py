"""
Wrapper seguro para ejecutar el demo básico
Captura el stderr para evitar crashes visibles por warnings de NumPy
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("  CausalDefend - Demo Básico (Modo Seguro)")
    print("="*70)
    print("\n⚠️  Ejecutando con supresión de warnings de NumPy MINGW-W64...")
    print("   (Esto es normal en Python 3.13 con Windows)\n")
    
    # Configurar environment para suprimir warnings
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    
    # Ejecutar el demo
    demo_path = os.path.join(os.path.dirname(__file__), 'examples', 'demo_basico.py')
    
    try:
        result = subprocess.run(
            [sys.executable, demo_path],
            env=env,
            capture_output=False,  # Mostrar output en tiempo real
            text=True
        )
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelado por el usuario.")
        return 0
    except Exception as e:
        print(f"\n❌ Error al ejecutar demo: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
