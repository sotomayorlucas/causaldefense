"""
Script Rápido para Entrenar Detector APT - Versión Simplificada

Usa configuración mínima para entrenamiento rápido.
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from causaldefend.models.spatiotemporal_detector import APTDetector
except ImportError:
    from models.spatiotemporal_detector import APTDetector


def train_quick():
    """Entrenamiento rápido y simple"""
    print("="*80)
    print("Entrenamiento Rápido del Detector APT")
    print("="*80)
    
    # Crear modelo
    print("\n1. Creando modelo...")
    model = APTDetector(
        in_channels=64,
        hidden_channels=64,  # Reducido para velocidad
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        learning_rate=0.001,
    )
    
    print(f"✓ Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Guardar modelo inicial
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    print("\n2. Guardando checkpoint inicial...")
    checkpoint_path = output_dir / "detector.ckpt"
    
    # Guardar con PyTorch Lightning format
    torch.save({
        'state_dict': model.state_dict(),
        'hyper_parameters': model.hparams,
    }, checkpoint_path)
    
    print(f"✓ Modelo guardado en: {checkpoint_path}")
    print(f"✓ Tamaño: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO SIMULADO COMPLETO")
    print("="*80)
    print("\n📝 Nota: Este es un modelo inicializado (sin entrenar)")
    print("   Para entrenamiento completo, usa: python scripts/train_detector.py")
    print("\n🎯 Próximos pasos:")
    print("   1. python scripts/train_ci_tester_quick.py  # Generar CI tester")
    print("   2. python examples/demo_basico.py            # Probar sistema")
    
    return checkpoint_path


if __name__ == "__main__":
    train_quick()
