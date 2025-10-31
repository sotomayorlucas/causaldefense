"""Quick script to inspect checkpoint contents"""
import torch
from pathlib import Path

checkpoint_path = Path("models/detector.ckpt")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("="*80)
print("CHECKPOINT INSPECTION")
print("="*80)
print()

print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")
print()

if "hyper_parameters" in checkpoint:
    print("Hyperparameters:")
    for key, value in checkpoint["hyper_parameters"].items():
        print(f"  - {key}: {value}")
    print()

if "state_dict" in checkpoint:
    print(f"State dict has {len(checkpoint['state_dict'])} keys")
    print("\nFirst 10 state_dict keys:")
    for i, key in enumerate(list(checkpoint["state_dict"].keys())[:10]):
        value = checkpoint["state_dict"][key]
        print(f"  {i+1}. {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    print()
    
    # Check for GRU parameters
    print("GRU-related parameters:")
    for key in checkpoint["state_dict"].keys():
        if "gru" in key.lower() or "temporal" in key.lower():
            value = checkpoint["state_dict"][key]
            print(f"  - {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
