"""
Neural Conditional Independence Testing (Tier 2)

Implements amortized neural CI tests for scalable causal discovery.
Based on LCIT (Learning Conditional Independence Tests) and DeepBET.

References:
- LCIT: https://arxiv.org/abs/2110.06701
- DeepBET: https://arxiv.org/abs/2305.04630
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """Residual block for deep encoder networks"""
    
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class ConditionalEncoder(nn.Module):
    """
    Deep neural network encoder for conditional independence testing.
    
    Maps (X, Z) → latent representation such that X ⊥ Y | Z can be
    tested via correlation in latent space.
    
    Architecture: MLP with residual connections and batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        dropout: float = 0.2,
        use_residual: bool = True,
    ) -> None:
        """
        Initialize conditional encoder.
        
        Args:
            input_dim: Dimension of input features (X + Z concatenated)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent representation
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers with optional residual connections
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode (X, Z) to latent representation.
        
        Args:
            x: Variable X [batch_size, x_dim]
            z: Conditioning set Z [batch_size, z_dim] (optional)
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        if z is not None:
            # Concatenate X and Z
            inputs = torch.cat([x, z], dim=-1)
        else:
            inputs = x
        
        return self.encoder(inputs)


class NeuralCITest(nn.Module):
    """
    Neural Conditional Independence Test using learned embeddings.
    
    Tests H0: X ⊥ Y | Z by:
    1. Encoding ϕ_θ(X|Z) and ϕ_θ(Y|Z)
    2. Computing correlation ρ in latent space
    3. Comparing ρ to threshold τ
    
    If ρ < τ: Accept H0 (independent)
    If ρ ≥ τ: Reject H0 (dependent)
    """
    
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int = 0,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize Neural CI Test.
        
        Args:
            x_dim: Dimension of variable X
            y_dim: Dimension of variable Y
            z_dim: Dimension of conditioning set Z
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent representation dimension
            dropout: Dropout probability
            device: Device for computation
        """
        super().__init__()
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Encoder for X given Z
        self.encoder_x = ConditionalEncoder(
            input_dim=x_dim + z_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        
        # Encoder for Y given Z
        self.encoder_y = ConditionalEncoder(
            input_dim=y_dim + z_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        
        # Learnable threshold (calibrated during training)
        self.log_threshold = nn.Parameter(torch.tensor(0.0))
        
        self.to(device)
    
    def encode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode X and Y given Z to latent representations.
        
        Args:
            x: Variable X [batch_size, x_dim]
            y: Variable Y [batch_size, y_dim]
            z: Conditioning set Z [batch_size, z_dim]
            
        Returns:
            Latent X [batch_size, latent_dim]
            Latent Y [batch_size, latent_dim]
        """
        latent_x = self.encoder_x(x, z)
        latent_y = self.encoder_y(y, z)
        
        return latent_x, latent_y
    
    def compute_correlation(
        self,
        latent_x: torch.Tensor,
        latent_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation between latent representations.
        
        Uses Hilbert-Schmidt Independence Criterion (HSIC) for
        robust correlation measurement.
        
        Args:
            latent_x: Latent X [batch_size, latent_dim]
            latent_y: Latent Y [batch_size, latent_dim]
            
        Returns:
            Correlation score (scalar)
        """
        # Normalize
        latent_x = F.normalize(latent_x, dim=-1)
        latent_y = F.normalize(latent_y, dim=-1)
        
        # Compute Gram matrices
        K_x = torch.mm(latent_x, latent_x.t())
        K_y = torch.mm(latent_y, latent_y.t())
        
        # HSIC statistic
        n = K_x.size(0)
        H = torch.eye(n, device=K_x.device) - torch.ones(n, n, device=K_x.device) / n
        
        hsic = torch.trace(torch.mm(torch.mm(K_x, H), torch.mm(K_y, H))) / (n ** 2)
        
        return hsic
    
    def test_independence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        significance_level: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Test conditional independence X ⊥ Y | Z.
        
        Args:
            x: Variable X [batch_size, x_dim]
            y: Variable Y [batch_size, y_dim]
            z: Conditioning set Z [batch_size, z_dim]
            significance_level: Significance level α
            
        Returns:
            is_independent: True if X ⊥ Y | Z (accept H0)
            correlation: Computed correlation score
            threshold: Decision threshold
        """
        self.eval()
        
        with torch.no_grad():
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)
            if z is not None:
                z = z.to(self.device)
            
            # Encode
            latent_x, latent_y = self.encode(x, y, z)
            
            # Compute correlation
            correlation = self.compute_correlation(latent_x, latent_y).item()
            
            # Threshold (learned during training)
            threshold = torch.exp(self.log_threshold).item()
            
            # Decision: if correlation < threshold, accept independence
            is_independent = correlation < threshold
        
        return is_independent, correlation, threshold
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: compute correlation for training.
        
        Returns:
            Correlation score
        """
        latent_x, latent_y = self.encode(x, y, z)
        return self.compute_correlation(latent_x, latent_y)
    
    def pretrain(
        self,
        dataset: TensorDataset,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Pretrain encoders on dataset of (X, Y, Z, label) samples.
        
        Label = 1 if X ⊥ Y | Z, else 0
        
        Args:
            dataset: TensorDataset of (X, Y, Z, label)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        # Split into train/val
        n = len(dataset)
        indices = np.arange(n)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Train
            self.train()
            train_losses = []
            
            for batch in train_loader:
                if len(batch) == 4:
                    x, y, z, labels = batch
                else:
                    x, y, labels = batch
                    z = None
                
                x = x.to(self.device)
                y = y.to(self.device)
                if z is not None:
                    z = z.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward
                correlation = self.forward(x, y, z)
                threshold = torch.exp(self.log_threshold)
                
                # Loss: binary cross-entropy
                # Predict independent (1) if correlation < threshold
                pred = (correlation < threshold).float()
                loss = F.binary_cross_entropy(pred.unsqueeze(0), labels.unsqueeze(0))
                
                # Add regularization to encourage meaningful representations
                loss += 0.01 * correlation  # Encourage low correlation for independent
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validate
            self.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        x, y, z, labels = batch
                    else:
                        x, y, labels = batch
                        z = None
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if z is not None:
                        z = z.to(self.device)
                    labels = labels.to(self.device).float()
                    
                    correlation = self.forward(x, y, z)
                    threshold = torch.exp(self.log_threshold)
                    
                    pred = (correlation < threshold).float()
                    loss = F.binary_cross_entropy(pred.unsqueeze(0), labels.unsqueeze(0))
                    
                    val_losses.append(loss.item())
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
            
            # Record history
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Step scheduler
            scheduler.step(val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.3f}, "
                      f"threshold={threshold.item():.4f}")
        
        return history


class BatchCITester:
    """
    Batch conditional independence testing for parallel processing.
    
    Efficiently tests multiple CI queries using a pretrained NeuralCITest.
    """
    
    def __init__(
        self,
        ci_test: NeuralCITest,
        batch_size: int = 1024
    ) -> None:
        """
        Initialize batch tester.
        
        Args:
            ci_test: Pretrained NeuralCITest model
            batch_size: Batch size for parallel testing
        """
        self.ci_test = ci_test
        self.batch_size = batch_size
    
    def batch_test(
        self,
        test_list: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        significance_level: float = 0.05
    ) -> Dict[int, Tuple[bool, float]]:
        """
        Test multiple CI queries in parallel.
        
        Args:
            test_list: List of (X, Y, Z) tuples to test
            significance_level: Significance level
            
        Returns:
            Dictionary mapping test_id → (is_independent, correlation)
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(test_list), self.batch_size):
            batch = test_list[i:i + self.batch_size]
            
            # Stack batch
            x_batch = torch.stack([item[0] for item in batch])
            y_batch = torch.stack([item[1] for item in batch])
            
            if batch[0][2] is not None:
                z_batch = torch.stack([item[2] for item in batch])
            else:
                z_batch = None
            
            # Test
            is_indep, corr, _ = self.ci_test.test_independence(
                x_batch,
                y_batch,
                z_batch,
                significance_level
            )
            
            # Store results
            for j, test_id in enumerate(range(i, min(i + self.batch_size, len(test_list)))):
                results[test_id] = (is_indep, corr)
        
        return results
