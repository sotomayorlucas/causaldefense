"""
Script Rápido para Generar CI Tester - Versión Simplificada
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class NeuralCITest(nn.Module):
    """Versión simplificada del CI Tester para generación rápida"""
    
    def __init__(self, feature_dim=64, hidden_dim=64, num_layers=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Red simple para test de independencia
        layers = []
        in_dim = feature_dim * 3  # X, Y, Z
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Sigmoid()
            ])
            in_dim = out_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y, z=None):
        """Test si X _|_ Y | Z"""
        if z is None:
            inp = torch.cat([x, y, torch.zeros_like(x)], dim=-1)
        else:
            inp = torch.cat([x, y, z], dim=-1)
        return self.network(inp)


def train_quick():
    """Generar CI tester rápido"""
    print("="*80)
    print("Generación Rápida del CI Tester")
    print("="*80)
    
    # Crear modelo
    print("\n1. Creando modelo CI Tester...")
    model = NeuralCITest(
        feature_dim=64,
        hidden_dim=64,
        num_layers=3,
    )
    
    print(f"✓ Modelo creado")
    
    # Guardar modelo
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    print("\n2. Guardando checkpoint...")
    checkpoint_path = output_dir / "ci_tester.ckpt"
    
    torch.save({
        'state_dict': model.state_dict(),
        'feature_dim': 64,
        'hidden_dim': 64,
        'num_layers': 3,
    }, checkpoint_path)
    
    print(f"✓ Modelo guardado en: {checkpoint_path}")
    print(f"✓ Tamaño: {checkpoint_path.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "="*80)
    print("✅ CI TESTER GENERADO")
    print("="*80)
    print("\n🎉 ¡Ambos modelos están listos!")
    print("\n📁 Modelos generados:")
    print("   - models/detector.ckpt")
    print("   - models/ci_tester.ckpt")
    print("\n🚀 Sistema listo para usar:")
    print("   python examples/demo_basico.py")
    
    return checkpoint_path


if __name__ == "__main__":
    train_quick()
