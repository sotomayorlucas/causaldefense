"""
Script Simple para Descargar y Preparar StreamSpot

Este script descarga el dataset StreamSpot y lo prepara para uso con CausalDefend.
"""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

def download_with_progress(url: str, output_path: Path):
    """Descarga archivo con barra de progreso"""
    print(f"\n📥 Descargando desde: {url}")
    print(f"📁 Guardando en: {output_path}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
    
    urlretrieve(url, output_path, reporthook)
    print("\n✅ Descarga completada!")

def extract_zip(zip_path: Path, extract_to: Path):
    """Extrae archivo ZIP"""
    print(f"\n📦 Extrayendo archivo...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraído en: {extract_to}")

def main():
    print("="*80)
    print("  StreamSpot Dataset Downloader")
    print("="*80)
    
    # Configuración
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "external" / "streamspot"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = data_dir / "streamspot-master.zip"
    
    # URL del dataset
    streamspot_url = "https://github.com/sbustreamspot/sbustreamspot-data/archive/refs/heads/master.zip"
    
    # Verificar si ya existe
    if zip_file.exists():
        print(f"\n⚠️  El archivo ya existe: {zip_file}")
        response = input("¿Descargar nuevamente? (s/N): ")
        if response.lower() != 's':
            print("ℹ️  Usando archivo existente")
        else:
            zip_file.unlink()
            download_with_progress(streamspot_url, zip_file)
    else:
        download_with_progress(streamspot_url, zip_file)
    
    # Extraer
    extract_zip(zip_file, data_dir)
    
    # Mostrar instrucciones
    print("\n" + "="*80)
    print("✅ StreamSpot descargado exitosamente!")
    print("="*80)
    print("\n📋 Próximos pasos:")
    print("\n1. Importar al formato de CausalDefend:")
    print(f"   python scripts\\import_external_dataset.py \\")
    print(f"     --dataset streamspot \\")
    print(f"     --input {data_dir} \\")
    print(f"     --output data\\processed\\streamspot \\")
    print(f"     --max-graphs 100")
    
    print("\n2. Dividir en train/val/test:")
    print(f"   python scripts\\split_dataset.py \\")
    print(f"     --input data\\processed\\streamspot \\")
    print(f"     --output data\\processed\\streamspot_split")
    
    print("\n3. Entrenar modelo:")
    print(f"   python scripts\\train_detector.py \\")
    print(f"     --data data\\processed\\streamspot_split \\")
    print(f"     --epochs 20 \\")
    print(f"     --output models\\streamspot_detector.ckpt")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
