"""
Fix NumPy Installation Issues on Windows

Reinstala NumPy con una versión estable compatible.
"""

import subprocess
import sys


def fix_numpy():
    """Reinstala NumPy con versión estable"""
    print("="*70)
    print("  Fixing NumPy Installation for Windows")
    print("="*70)
    print()
    
    print("Paso 1: Desinstalando NumPy actual...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                      check=False)
        print("✓ NumPy desinstalado")
    except Exception as e:
        print(f"⚠ Advertencia: {e}")
    
    print("\nPaso 2: Instalando NumPy 1.26.4 (versión estable)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], 
                      check=True)
        print("✓ NumPy 1.26.4 instalado correctamente")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\nPaso 3: Verificando instalación...")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} funcionando correctamente")
        
        # Test básico
        arr = np.array([1, 2, 3])
        print(f"✓ Test básico: {arr.sum()} = 6")
        
        return True
    except Exception as e:
        print(f"✗ Error en verificación: {e}")
        return False


def main():
    print("\n🔧 CausalDefend - NumPy Fix Tool\n")
    
    # Verificar si hay problema
    try:
        import numpy as np
        print(f"NumPy actual: {np.__version__}")
        
        # Test si funciona
        test = np.array([1, 2, 3])
        test.sum()
        
        print("✓ NumPy parece funcionar correctamente")
        print("\nSi estás viendo warnings sobre MINGW-W64, continúa con el fix.")
        response = input("\n¿Quieres reinstalar NumPy de todas formas? (s/n): ")
        
        if response.lower() != 's':
            print("\nSaliendo sin cambios.")
            return
    except ImportError:
        print("NumPy no está instalado. Instalando...")
    except Exception as e:
        print(f"⚠ Problema detectado con NumPy: {e}")
        print("\nContinuando con el fix...")
    
    print()
    
    if fix_numpy():
        print("\n" + "="*70)
        print("✅ NumPy instalado correctamente!")
        print("="*70)
        print("\nAhora puedes ejecutar:")
        print("  python quick_start.py")
        print("  python examples/basic_usage.py")
    else:
        print("\n" + "="*70)
        print("❌ Hubo problemas con la instalación")
        print("="*70)
        print("\nIntentos alternativos:")
        print("\n1. Reinstalar manualmente:")
        print("   pip uninstall numpy")
        print("   pip install numpy==1.26.4")
        print("\n2. Usar conda (si tienes Anaconda/Miniconda):")
        print("   conda install numpy=1.26.4")
        print("\n3. Descargar wheel precompilado desde:")
        print("   https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso cancelado por el usuario.")
        sys.exit(0)
