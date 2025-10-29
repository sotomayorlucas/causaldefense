"""
Training Script for APT Detector (GAT+GRU)

Trains the spatiotemporal detector on provenance graphs.

Usage:
    python scripts/train_detector.py --data data/processed --epochs 100
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
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
    from causaldefend.models.spatiotemporal_detector import APTDetector
    from causaldefend.data.provenance_graph import ProvenanceGraph
except ImportError:
    from models.spatiotemporal_detector import APTDetector
    from data.provenance_graph import ProvenanceGraph


class ProvenanceDataset(Dataset):
    """Dataset for provenance graphs"""
    
    def __init__(self, data_dir: Path, feature_dim: int = 64):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing processed graphs
            feature_dim: Node feature dimensionality
        """
        self.data_dir = Path(data_dir)
        self.feature_dim = feature_dim
        
        # Find all graph files
        self.graph_files = sorted(list(self.data_dir.glob("graph_*.pkl")))
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")
        
        logger.info(f"Found {len(self.graph_files)} graphs in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.graph_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a single graph"""
        graph_file = self.graph_files[idx]
        graph_id = graph_file.stem.replace("graph_", "")
        
        # Load graph (networkx DiGraph)
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)
        
        # Load features
        feature_file = self.data_dir / f"features_{graph_id}.npy"
        features = np.load(feature_file)
        
        # Load label
        label_file = self.data_dir / f"label_{graph_id}.json"
        with open(label_file, "r") as f:
            label_data = json.load(f)
        
        # Convert to PyTorch Geometric format
        pyg_data = self._networkx_to_pytorch_geometric(graph, features)
        
        return {
            "data": pyg_data,
            "label": torch.tensor([1.0 if label_data["is_attack"] else 0.0], dtype=torch.float),
            "graph_id": graph_id,
        }
    
    def _networkx_to_pytorch_geometric(
        self,
        graph: "nx.DiGraph",
        features: np.ndarray
    ) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data"""
        import networkx as nx
        
        # Node features
        x = torch.tensor(features, dtype=torch.float)
        
        # Edge index from networkx
        edge_list = list(graph.edges())
        
        if len(edge_list) == 0:
            # Empty graph - create self-loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return Data(x=x, edge_index=edge_index)


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching graphs"""
    
    # Batch PyG data
    data_list = [item["data"] for item in batch]
    batched_data = Batch.from_data_list(data_list)
    
    # Stack labels
    labels = torch.stack([item["label"] for item in batch])
    
    return {
        "data": batched_data,
        "labels": labels,
    }


class APTDetectorTrainer(pl.LightningModule):
    """Lightning module for training APT detector"""
    
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize detector
        self.detector = APTDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            # Ensure GRU hidden size matches embedding when using single-snapshot flows
            gru_hidden_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            learning_rate=learning_rate,
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
    
    def forward(self, x, edge_index, batch=None):
        return self.detector(x, edge_index, batch)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        data = batch["data"]
        labels = batch["labels"]
        
        # Forward pass
        graph_embedding, edge_probs = self(
            data.x,
            data.edge_index,
            data.batch if hasattr(data, 'batch') else None
        )
        
        # Compute anomaly scores
        anomaly_scores = []
        batch_size = labels.size(0)
        
        if hasattr(data, 'batch'):
            # Multiple graphs in batch
            for i in range(batch_size):
                mask = data.batch == i
                graph_x = data.x[mask]
                
                # Reconstruct features
                x_recon = self.detector.feature_decoder(graph_embedding[i:i+1])
                
                # Compute score (simplified - just use reconstruction error)
                recon_error = torch.mean((graph_x.mean(0) - x_recon) ** 2)
                anomaly_scores.append(recon_error)
        else:
            # Single graph
            x_recon = self.detector.feature_decoder(graph_embedding)
            recon_error = torch.mean((data.x.mean(0) - x_recon) ** 2)
            anomaly_scores.append(recon_error)
        
        anomaly_scores = torch.stack(anomaly_scores).unsqueeze(1)
        
        # Compute loss
        loss = self.criterion(anomaly_scores, labels)
        
        # Compute accuracy
        preds = (torch.sigmoid(anomaly_scores) > 0.5).float()
        correct = (preds == labels).sum().item()
        self.train_correct += correct
        self.train_total += labels.size(0)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training accuracy at epoch end"""
        if self.train_total > 0:
            acc = self.train_correct / self.train_total
            self.log("train_acc", acc, prog_bar=True)
            self.train_correct = 0
            self.train_total = 0
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        data = batch["data"]
        labels = batch["labels"]
        
        # Forward pass
        graph_embedding, edge_probs = self(
            data.x,
            data.edge_index,
            data.batch if hasattr(data, 'batch') else None
        )
        
        # Compute anomaly scores (same as training)
        anomaly_scores = []
        batch_size = labels.size(0)
        
        if hasattr(data, 'batch'):
            for i in range(batch_size):
                mask = data.batch == i
                graph_x = data.x[mask]
                x_recon = self.detector.feature_decoder(graph_embedding[i:i+1])
                recon_error = torch.mean((graph_x.mean(0) - x_recon) ** 2)
                anomaly_scores.append(recon_error)
        else:
            x_recon = self.detector.feature_decoder(graph_embedding)
            recon_error = torch.mean((data.x.mean(0) - x_recon) ** 2)
            anomaly_scores.append(recon_error)
        
        anomaly_scores = torch.stack(anomaly_scores).unsqueeze(1)
        
        # Compute loss
        loss = self.criterion(anomaly_scores, labels)
        
        # Compute accuracy
        preds = (torch.sigmoid(anomaly_scores) > 0.5).float()
        correct = (preds == labels).sum().item()
        self.val_correct += correct
        self.val_total += labels.size(0)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log validation accuracy at epoch end"""
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
    parser = argparse.ArgumentParser(description="Train APT Detector")
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
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Hidden channels in GAT"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GAT layers"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to use"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Training APT Detector")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Hidden channels: {args.hidden_channels}")
    logger.info(f"Attention heads: {args.num_heads}")
    logger.info(f"GAT layers: {args.num_layers}")
    
    # Create datasets
    logger.info("\nLoading datasets...")
    train_dataset = ProvenanceDataset(args.data / "train")
    val_dataset = ProvenanceDataset(args.data / "val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    logger.info(f"✓ Train dataset: {len(train_dataset)} graphs")
    logger.info(f"✓ Val dataset: {len(val_dataset)} graphs")
    
    # Initialize model
    logger.info("\nInitializing model...")
    model = APTDetectorTrainer(
        in_channels=64,
        hidden_channels=args.hidden_channels,
        embedding_dim=64,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output,
        filename="detector-{epoch:02d}-{val_loss:.4f}",
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
    tb_logger = TensorBoardLogger("logs", name="apt_detector")
    
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
    final_path = args.output / "detector.ckpt"
    trainer.save_checkpoint(final_path)
    
    logger.info("="*80)
    logger.info(f"✓ Training complete!")
    logger.info(f"✓ Best model saved to: {checkpoint_callback.best_model_path}")
    logger.info(f"✓ Final model saved to: {final_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Train CI tester: python scripts/train_ci_tester.py")
    logger.info(f"  2. Test detector: python examples/complete_detection.py")


if __name__ == "__main__":
    main()
