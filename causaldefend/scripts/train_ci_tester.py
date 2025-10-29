"""
Training Script for Neural CI Tester

Trains the neural conditional independence tester for causal discovery.

Usage:
    python scripts/train_ci_tester.py --data data/processed --epochs 50
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger
from tqdm import tqdm

# Add src to path
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from causaldefend.causal.neural_ci_test import NeuralCITest
    from causaldefend.data.provenance_graph import ProvenanceGraph
except ImportError:
    from causal.neural_ci_test import NeuralCITest
    from data.provenance_graph import ProvenanceGraph


class CITestDataset(Dataset):
    """
    Dataset for training conditional independence tests.
    
    Creates triplets (X, Y, Z) where:
    - X, Y are independent given Z (positive samples)
    - X, Y are dependent given Z (negative samples)
    """
    
    def __init__(
        self,
        data_dir: Path,
        num_samples: int = 10000,
        feature_dim: int = 64
    ):
        """
        Initialize CI test dataset.
        
        Args:
            data_dir: Directory containing processed graphs
            num_samples: Number of (X, Y, Z) triplets to generate
            feature_dim: Feature dimensionality
        """
        self.data_dir = Path(data_dir)
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
        # Load all graphs and features
        logger.info(f"Loading graphs from {data_dir}...")
        self.graphs = []
        self.features = []
        
        graph_files = sorted(list(self.data_dir.glob("graph_*.pkl")))
        
        for graph_file in tqdm(graph_files[:100], desc="Loading graphs"):  # Limit for speed
            graph_id = graph_file.stem.replace("graph_", "")
            
            # Load graph (networkx)
            with open(graph_file, "rb") as f:
                graph = pickle.load(f)
            
            # Load features
            feature_file = self.data_dir / f"features_{graph_id}.npy"
            features = np.load(feature_file)
            
            if len(graph.nodes) >= 3:  # Need at least 3 nodes for (X, Y, Z)
                self.graphs.append(graph)
                self.features.append(features)
        
        logger.info(f"Loaded {len(self.graphs)} graphs")
        
        # Pre-generate samples
        logger.info(f"Generating {num_samples} CI test samples...")
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Generate (X, Y, Z, label) samples"""
        samples = []
        
        for _ in tqdm(range(self.num_samples), desc="Generating samples"):
            # Random graph
            graph_idx = random.randint(0, len(self.graphs) - 1)
            graph = self.graphs[graph_idx]
            features = self.features[graph_idx]
            
            num_nodes = len(graph.nodes)
            
            if num_nodes < 3:
                continue
            
            # Sample 3 different nodes
            node_indices = random.sample(range(num_nodes), 3)
            x_idx, y_idx, z_idx = node_indices
            
            # Get features
            x_feat = features[x_idx]
            y_feat = features[y_idx]
            z_feat = features[z_idx]
            
            # Determine if X and Y are independent given Z
            # Simplified: check if they are d-separated in the graph
            is_independent = self._check_independence(graph, x_idx, y_idx, z_idx)
            
            label = 1 if is_independent else 0
            
            samples.append((x_feat, y_feat, z_feat, label))
        
        return samples
    
    def _check_independence(
        self,
        graph: "nx.DiGraph",
        x_id: int,
        y_id: int,
        z_id: int
    ) -> bool:
        """
        Check if X and Y are independent given Z.
        
        Simplified version: X and Y are independent if:
        - No direct edge between them
        - All paths go through Z
        """
        # Check direct connection
        if graph.has_edge(x_id, y_id) or graph.has_edge(y_id, x_id):
            return False
        
        # Simplified: if no direct edge, consider independent
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        x, y, z, label = self.samples[idx]
        
        return {
            "x": torch.tensor(x, dtype=torch.float),
            "y": torch.tensor(y, dtype=torch.float),
            "z": torch.tensor(z, dtype=torch.float),
            "label": torch.tensor([label], dtype=torch.float),
        }


class CITesterTrainer(pl.LightningModule):
    """Lightning module for training CI tester"""
    
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize CI tester
        self.ci_tester = NeuralCITest(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Metrics
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
    
    def forward(self, x, y, z):
        """Forward pass"""
        # Compute test statistic
        test_stat = self.ci_tester.test_independence(x, y, z)
        
        # Convert to probability (higher stat = more dependent)
        # Use sigmoid to map to [0, 1]
        prob_dependent = torch.sigmoid(test_stat / 10.0)  # Scale factor
        
        return prob_dependent
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x = batch["x"]
        y = batch["y"]
        z = batch["z"]
        labels = batch["label"]
        
        # Forward pass
        probs = self(x, y, z)
        
        # Loss (label=1 means independent, so prob_dependent should be low)
        # Flip labels: independent -> dependent for loss computation
        target = 1.0 - labels
        loss = self.criterion(probs, target)
        
        # Accuracy
        preds = (probs > 0.5).float()
        correct = (preds == target).sum().item()
        self.train_correct += correct
        self.train_total += labels.size(0)
        
        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training accuracy"""
        if self.train_total > 0:
            acc = self.train_correct / self.train_total
            self.log("train_acc", acc, prog_bar=True)
            self.train_correct = 0
            self.train_total = 0
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x = batch["x"]
        y = batch["y"]
        z = batch["z"]
        labels = batch["label"]
        
        # Forward pass
        probs = self(x, y, z)
        
        # Loss
        target = 1.0 - labels
        loss = self.criterion(probs, target)
        
        # Accuracy
        preds = (probs > 0.5).float()
        correct = (preds == target).sum().item()
        self.val_correct += correct
        self.val_total += labels.size(0)
        
        # Log
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log validation accuracy"""
        if self.val_total > 0:
            acc = self.val_correct / self.val_total
            self.log("val_acc", acc, prog_bar=True)
            self.val_correct = 0
            self.val_total = 0
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description="Train Neural CI Tester")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed"),
        help="Data directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of CI test samples to generate"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to use"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Training Neural CI Tester")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CI samples: {args.num_samples}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = CITestDataset(
        args.data / "train",
        num_samples=args.num_samples,
    )
    
    val_dataset = CITestDataset(
        args.data / "val",
        num_samples=args.num_samples // 5,  # Smaller val set
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    logger.info(f"✓ Train dataset: {len(train_dataset)} samples")
    logger.info(f"✓ Val dataset: {len(val_dataset)} samples")
    
    # Initialize model
    logger.info("\nInitializing model...")
    model = CITesterTrainer(
        feature_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output,
        filename="ci_tester-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    
    # Logger
    tb_logger = TensorBoardLogger("logs", name="ci_tester")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,  # 1 for CPU
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        log_every_n_steps=10,
    )
    
    # Train
    logger.info("\nStarting training...")
    logger.info("="*80)
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = args.output / "ci_tester.ckpt"
    trainer.save_checkpoint(final_path)
    
    logger.info("="*80)
    logger.info(f"✓ Training complete!")
    logger.info(f"✓ Best model saved to: {checkpoint_callback.best_model_path}")
    logger.info(f"✓ Final model saved to: {final_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test full pipeline: python examples/complete_detection.py")
    logger.info(f"  2. Run API server: uvicorn causaldefend.api.main:app")


if __name__ == "__main__":
    main()
