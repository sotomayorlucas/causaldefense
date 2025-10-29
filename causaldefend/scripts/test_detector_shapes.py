"""
Test rápido para validar shapes del APTDetectorTrainer
"""
from pathlib import Path
import torch
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import APTDetectorTrainer from scripts/train_detector.py by path (not a package)
from importlib.machinery import SourceFileLoader
train_detector_path = str(Path(__file__).parent / "train_detector.py")
loader = SourceFileLoader("train_detector", train_detector_path)
mod = loader.load_module()
APTDetectorTrainer = mod.APTDetectorTrainer
from torch_geometric.data import Data, Batch


def make_graph(num_nodes=5, feat_dim=64):
    x = torch.randn((num_nodes, feat_dim))
    # create a simple chain edges
    edge_index = torch.tensor([[i for i in range(num_nodes-1)] + [i+1 for i in range(num_nodes-1)],
                               [i+1 for i in range(num_nodes-1)] + [i for i in range(num_nodes-1)]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def run_test():
    trainer = APTDetectorTrainer(in_channels=64, hidden_channels=128, embedding_dim=64)
    g1 = make_graph(6, 64)
    g2 = make_graph(4, 64)
    batch = Batch.from_data_list([g1, g2])
    labels = torch.tensor([[0.0],[1.0]])
    batch_dict = {"data": batch, "labels": labels}

    loss = trainer.training_step(batch_dict, 0)
    print("Training step returned loss:", loss)

if __name__ == '__main__':
    run_test()
