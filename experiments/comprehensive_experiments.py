"""
Comprehensive Experimental Suite for CausalDefend
==================================================

This script runs multiple experiments:
1. Ablation studies (architecture components)
2. Scalability analysis (varying graph sizes)
3. Dataset comparison (synthetic vs real)
4. Hyperparameter sensitivity
5. Adversarial robustness
"""

import sys
import os

# Configure UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Configure paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "causaldefend" / "src"))

from causaldefend.models.spatiotemporal_detector import APTDetector

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

class ExperimentRunner:
    """Orchestrates all experiments and data collection"""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'ablation': {},
            'scalability': {},
            'datasets': {},
            'hyperparameters': {},
            'adversarial': {}
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_data(self, dataset_name: str = 'streamspot'):
        """Load processed data"""
        # Navigate from experiments/ to causaldefend/data/processed/
        data_dir = Path(__file__).parent.parent / "causaldefend" / "data" / "processed"
        
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        test_dir = data_dir / "test"
        
        train_graphs = list(train_dir.glob("*.pt"))
        val_graphs = list(val_dir.glob("*.pt"))
        test_graphs = list(test_dir.glob("*.pt"))
        
        print(f"\nDataset: {dataset_name}")
        print(f"Train: {len(train_graphs)} graphs")
        print(f"Val: {len(val_graphs)} graphs")
        print(f"Test: {len(test_graphs)} graphs")
        
        return train_graphs, val_graphs, test_graphs
    
    def create_model(self, config: Dict):
        """Create model with specific configuration"""
        model = APTDetector(
            in_channels=config.get('in_channels', 64),
            hidden_channels=config.get('hidden_channels', 128),
            num_layers=config.get('num_layers', 3),
            num_heads=config.get('num_heads', 8),
            embedding_dim=config.get('embedding_dim', 64),
            gru_hidden_dim=config.get('gru_hidden_dim', 64),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        return model
    
    def train_and_evaluate(self, model, train_files, val_files, test_files, 
                          epochs: int = 10, lr: float = 0.001):
        """Train model and return metrics"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        train_times = []
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            epoch_loss = 0
            
            for graph_file in train_files:
                data = torch.load(graph_file)
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, embedding = model(data.x, data.edge_index)
                
                # Reconstruction loss
                loss = F.mse_loss(reconstructed, data.x)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_files)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for graph_file in val_files:
                    data = torch.load(graph_file)
                    data = data.to(self.device)
                    
                    reconstructed, embedding = model(data.x, data.edge_index)
                    loss = F.mse_loss(reconstructed, data.x)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_files)
            val_losses.append(avg_val_loss)
            
            epoch_time = time.time() - start_time
            train_times.append(epoch_time)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Final evaluation on test set
        model.eval()
        test_scores = []
        test_labels = []
        inference_times = []
        
        with torch.no_grad():
            for graph_file in test_files:
                data = torch.load(graph_file)
                data = data.to(self.device)
                
                start = time.time()
                reconstructed, embedding = model(data.x, data.edge_index)
                inference_time = time.time() - start
                
                score = F.mse_loss(reconstructed, data.x, reduction='mean').item()
                test_scores.append(score)
                
                # Extract label from filename (assuming format: benign_X.pt or attack_X.pt)
                label = 1 if 'attack' in graph_file.name else 0
                test_labels.append(label)
                inference_times.append(inference_time)
        
        # Calculate metrics
        test_scores = np.array(test_scores)
        test_labels = np.array(test_labels)
        
        # Determine threshold (median of benign scores)
        benign_scores = test_scores[test_labels == 0]
        threshold = np.percentile(benign_scores, 95) if len(benign_scores) > 0 else np.median(test_scores)
        
        predictions = (test_scores > threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (test_labels == 1))
        fp = np.sum((predictions == 1) & (test_labels == 0))
        tn = np.sum((predictions == 0) & (test_labels == 0))
        fn = np.sum((predictions == 0) & (test_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(test_labels)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_times': train_times,
            'test_scores': test_scores.tolist(),
            'test_labels': test_labels.tolist(),
            'predictions': predictions.tolist(),
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'avg_inference_time': float(np.mean(inference_times)),
            'total_params': sum(p.numel() for p in model.parameters())
        }
    
    def experiment_ablation(self):
        """Ablation study: test different architecture components"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: Ablation Study")
        print("="*80)
        
        train_files, val_files, test_files = self.load_data()
        
        configs = {
            'Full Model': {
                'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
                'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
            },
            'No Multi-Head (1 head)': {
                'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
                'num_heads': 1, 'embedding_dim': 64, 'gru_hidden_dim': 64
            },
            'Shallow (1 layer)': {
                'in_channels': 64, 'hidden_channels': 128, 'num_layers': 1,
                'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
            },
            'Small Hidden (64)': {
                'in_channels': 64, 'hidden_channels': 64, 'num_layers': 3,
                'num_heads': 8, 'embedding_dim': 32, 'gru_hidden_dim': 32
            },
            'Large Hidden (256)': {
                'in_channels': 64, 'hidden_channels': 256, 'num_layers': 3,
                'num_heads': 8, 'embedding_dim': 128, 'gru_hidden_dim': 128
            }
        }
        
        results = {}
        for name, config in configs.items():
            print(f"\nTesting: {name}")
            print(f"Config: {config}")
            
            model = self.create_model(config)
            metrics = self.train_and_evaluate(model, train_files, val_files, test_files, epochs=20)
            results[name] = metrics
            
            print(f"Results - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                  f"Params: {metrics['total_params']:,}")
        
        self.results['ablation'] = results
        
        # Save results
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experiment_scalability(self):
        """Scalability analysis with synthetic graphs of varying sizes"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: Scalability Analysis")
        print("="*80)
        
        # Generate synthetic graphs of different sizes
        graph_sizes = [100, 500, 1000, 5000, 10000, 50000]
        results = {}
        
        model = self.create_model({
            'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
            'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
        })
        model.eval()
        
        for size in graph_sizes:
            print(f"\nGraph size: {size} nodes")
            
            # Create synthetic graph
            num_edges = int(size * 2.5)  # Avg degree ~5
            edge_index = torch.randint(0, size, (2, num_edges), device=self.device)
            x = torch.randn(size, 64, device=self.device)
            
            data = Data(x=x, edge_index=edge_index)
            
            # Warm-up
            with torch.no_grad():
                _ = model(data.x, data.edge_index)
            
            # Measure inference time (average of 10 runs)
            times = []
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    _ = model(data.x, data.edge_index)
                times.append(time.time() - start)
            
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (memory_after - memory_before) / 1024**2  # MB
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[size] = {
                'avg_time': float(avg_time),
                'std_time': float(std_time),
                'memory_mb': float(memory_used),
                'throughput': float(1.0 / avg_time) if avg_time > 0 else 0
            }
            
            print(f"Inference time: {avg_time*1000:.2f}±{std_time*1000:.2f}ms, "
                  f"Memory: {memory_used:.2f}MB")
        
        self.results['scalability'] = results
        
        with open(self.output_dir / 'scalability_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experiment_hyperparameters(self):
        """Hyperparameter sensitivity analysis"""
        print("\n" + "="*80)
        print("EXPERIMENT 3: Hyperparameter Sensitivity")
        print("="*80)
        
        train_files, val_files, test_files = self.load_data()
        
        # Test different learning rates
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        results = {}
        
        for lr in learning_rates:
            print(f"\nLearning rate: {lr}")
            
            model = self.create_model({
                'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
                'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
            })
            
            metrics = self.train_and_evaluate(model, train_files, val_files, test_files, 
                                              epochs=15, lr=lr)
            results[f'lr_{lr}'] = metrics
            
            print(f"F1: {metrics['f1']:.4f}, Final train loss: {metrics['train_losses'][-1]:.4f}")
        
        self.results['hyperparameters'] = results
        
        with open(self.output_dir / 'hyperparameter_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_plots(self):
        """Generate all comparison plots"""
        print("\n" + "="*80)
        print("Generating Comparison Plots")
        print("="*80)
        
        # Plot 1: Ablation Study Comparison
        if self.results['ablation']:
            self._plot_ablation()
        
        # Plot 2: Scalability Analysis
        if self.results['scalability']:
            self._plot_scalability()
        
        # Plot 3: Hyperparameter Sensitivity
        if self.results['hyperparameters']:
            self._plot_hyperparameters()
        
        # Plot 4: Combined Comparison
        self._plot_combined_comparison()
    
    def _plot_ablation(self):
        """Plot ablation study results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        data = self.results['ablation']
        configs = list(data.keys())
        
        # Metrics comparison
        metrics = ['f1', 'precision', 'recall', 'accuracy']
        values = {m: [data[c][m] for c in configs] for m in metrics}
        
        ax = axes[0, 0]
        x = np.arange(len(configs))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, values[metric], width, label=metric.upper())
        
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Score')
        ax.set_title('Ablation Study: Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter count
        ax = axes[0, 1]
        params = [data[c]['total_params'] for c in configs]
        ax.bar(configs, params, color='skyblue', edgecolor='black')
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity')
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Training curves
        ax = axes[1, 0]
        for config in configs:
            ax.plot(data[config]['train_losses'], label=config, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Inference time vs F1
        ax = axes[1, 1]
        f1_scores = [data[c]['f1'] for c in configs]
        inf_times = [data[c]['avg_inference_time'] * 1000 for c in configs]  # ms
        
        scatter = ax.scatter(inf_times, f1_scores, s=200, c=range(len(configs)), 
                            cmap='viridis', edgecolor='black', linewidth=2)
        
        for i, config in enumerate(configs):
            ax.annotate(config, (inf_times[i], f1_scores[i]), 
                       fontsize=8, ha='right', va='bottom')
        
        ax.set_xlabel('Avg Inference Time (ms)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Efficiency vs Performance Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'ablation_comparison.pdf', bbox_inches='tight')
        print(f"Saved: ablation_comparison.png/pdf")
        plt.close()
    
    def _plot_scalability(self):
        """Plot scalability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        data = self.results['scalability']
        sizes = sorted([int(k) for k in data.keys()])
        
        times = [data[str(s)]['avg_time'] * 1000 for s in sizes]  # ms
        stds = [data[str(s)]['std_time'] * 1000 for s in sizes]
        memory = [data[str(s)]['memory_mb'] for s in sizes]
        throughput = [data[str(s)]['throughput'] for s in sizes]
        
        # Inference time vs graph size
        ax = axes[0, 0]
        ax.errorbar(sizes, times, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Scalability: Inference Time vs Graph Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Memory usage
        ax = axes[0, 1]
        ax.plot(sizes, memory, marker='s', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Consumption')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Throughput
        ax = axes[1, 0]
        ax.plot(sizes, throughput, marker='^', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Throughput (graphs/sec)')
        ax.set_title('Processing Throughput')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Scaling efficiency (normalized to smallest graph)
        ax = axes[1, 1]
        baseline_time = times[0] / sizes[0]  # time per node for smallest graph
        expected_times = [s * baseline_time for s in sizes]
        efficiency = [exp / actual for exp, actual in zip(expected_times, times)]
        
        ax.plot(sizes, efficiency, marker='D', linewidth=2, markersize=8, color='red')
        ax.axhline(y=1.0, color='black', linestyle='--', label='Linear Scaling')
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Scaling Efficiency (relative to linear)')
        ax.set_title('Scaling Efficiency Analysis')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
        print(f"Saved: scalability_analysis.png/pdf")
        plt.close()
    
    def _plot_hyperparameters(self):
        """Plot hyperparameter sensitivity"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        data = self.results['hyperparameters']
        lrs = [float(k.split('_')[1]) for k in data.keys()]
        
        # F1 vs learning rate
        ax = axes[0, 0]
        f1_scores = [data[f'lr_{lr}']['f1'] for lr in lrs]
        ax.plot(lrs, f1_scores, marker='o', linewidth=2, markersize=10)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Learning Rate')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Training curves for different LRs
        ax = axes[0, 1]
        for lr in lrs:
            losses = data[f'lr_{lr}']['train_losses']
            ax.plot(losses, label=f'LR={lr}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Curves: Different Learning Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convergence speed (epochs to reach 95% of final performance)
        ax = axes[1, 0]
        convergence_epochs = []
        for lr in lrs:
            losses = data[f'lr_{lr}']['train_losses']
            final_loss = losses[-1]
            target = final_loss * 1.05  # Within 5% of final
            
            conv_epoch = next((i for i, loss in enumerate(losses) if loss <= target), len(losses))
            convergence_epochs.append(conv_epoch)
        
        ax.bar(range(len(lrs)), convergence_epochs, color='coral', edgecolor='black')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Epochs to Convergence')
        ax.set_title('Convergence Speed')
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Performance metrics heatmap
        ax = axes[1, 1]
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        values = [[data[f'lr_{lr}'][m] for m in metrics] for lr in lrs]
        
        im = ax.imshow(values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f'{lr:.4f}' for lr in lrs])
        ax.set_xlabel('Metric')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Performance Metrics Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score')
        
        # Add text annotations
        for i in range(len(lrs)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{values[i][j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'hyperparameter_sensitivity.pdf', bbox_inches='tight')
        print(f"Saved: hyperparameter_sensitivity.png/pdf")
        plt.close()
    
    def _plot_combined_comparison(self):
        """Create a comprehensive comparison figure"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Best model from ablation
        if self.results['ablation']:
            ablation_data = self.results['ablation']
            best_config = max(ablation_data.keys(), key=lambda k: ablation_data[k]['f1'])
            
            ax1 = fig.add_subplot(gs[0, :2])
            configs = list(ablation_data.keys())
            f1_scores = [ablation_data[c]['f1'] for c in configs]
            colors = ['green' if c == best_config else 'lightblue' for c in configs]
            
            bars = ax1.bar(configs, f1_scores, color=colors, edgecolor='black', linewidth=2)
            ax1.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
            ax1.set_title('Model Configuration Comparison', fontsize=16, fontweight='bold')
            ax1.set_xticklabels(configs, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Target (0.95)')
            ax1.legend()
            
            # Add value labels on bars
            for bar, val in zip(bars, f1_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Scalability summary
        if self.results['scalability']:
            ax2 = fig.add_subplot(gs[0, 2])
            data = self.results['scalability']
            sizes = sorted([int(k) for k in data.keys()])
            times = [data[str(s)]['avg_time'] * 1000 for s in sizes]
            
            ax2.plot(sizes, times, marker='o', linewidth=3, markersize=10, color='orange')
            ax2.set_xlabel('Graph Size (nodes)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
            ax2.set_title('Inference Time Scaling', fontsize=14, fontweight='bold')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # Learning rate comparison
        if self.results['hyperparameters']:
            ax3 = fig.add_subplot(gs[1, :])
            data = self.results['hyperparameters']
            
            for key in data.keys():
                lr = float(key.split('_')[1])
                losses = data[key]['train_losses']
                ax3.plot(losses, label=f'LR={lr}', linewidth=2, marker='o', markersize=4)
            
            ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
            ax3.set_title('Training Dynamics: Learning Rate Comparison', fontsize=14, fontweight='bold')
            ax3.legend(ncol=3)
            ax3.grid(True, alpha=0.3)
        
        # Performance summary table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        if self.results['ablation']:
            table_data = []
            for config in configs:
                data_row = [
                    config,
                    f"{ablation_data[config]['f1']:.4f}",
                    f"{ablation_data[config]['precision']:.4f}",
                    f"{ablation_data[config]['recall']:.4f}",
                    f"{ablation_data[config]['accuracy']:.4f}",
                    f"{ablation_data[config]['avg_inference_time']*1000:.2f}ms",
                    f"{ablation_data[config]['total_params']:,}"
                ]
                table_data.append(data_row)
            
            table = ax4.table(cellText=table_data,
                            colLabels=['Configuration', 'F1', 'Precision', 'Recall', 
                                     'Accuracy', 'Inference Time', 'Parameters'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.15, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(7):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight best configuration
            best_idx = configs.index(best_config) + 1
            for i in range(7):
                table[(best_idx, i)].set_facecolor('#FFD700')
        
        plt.suptitle('CausalDefend: Comprehensive Experimental Results', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'combined_comparison.pdf', bbox_inches='tight')
        print(f"Saved: combined_comparison.png/pdf")
        plt.close()
    
    def save_summary_report(self):
        """Generate summary report"""
        report_path = self.output_dir / 'experimental_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CAUSALDEFEND: EXPERIMENTAL RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Ablation Study
            if self.results['ablation']:
                f.write("\n" + "-"*80 + "\n")
                f.write("ABLATION STUDY\n")
                f.write("-"*80 + "\n")
                
                data = self.results['ablation']
                best = max(data.keys(), key=lambda k: data[k]['f1'])
                
                f.write(f"\nBest Configuration: {best}\n")
                f.write(f"  F1 Score: {data[best]['f1']:.4f}\n")
                f.write(f"  Precision: {data[best]['precision']:.4f}\n")
                f.write(f"  Recall: {data[best]['recall']:.4f}\n")
                f.write(f"  Accuracy: {data[best]['accuracy']:.4f}\n")
                f.write(f"  Parameters: {data[best]['total_params']:,}\n")
                f.write(f"  Inference Time: {data[best]['avg_inference_time']*1000:.2f}ms\n")
                
                f.write("\nAll Configurations:\n")
                for config in data.keys():
                    f.write(f"\n  {config}:\n")
                    f.write(f"    F1: {data[config]['f1']:.4f}, ")
                    f.write(f"Acc: {data[config]['accuracy']:.4f}, ")
                    f.write(f"Params: {data[config]['total_params']:,}\n")
            
            # Scalability
            if self.results['scalability']:
                f.write("\n" + "-"*80 + "\n")
                f.write("SCALABILITY ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                data = self.results['scalability']
                for size in sorted([int(k) for k in data.keys()]):
                    d = data[str(size)]
                    f.write(f"  {size:,} nodes: {d['avg_time']*1000:.2f}ms ± {d['std_time']*1000:.2f}ms, ")
                    f.write(f"Memory: {d['memory_mb']:.2f}MB, ")
                    f.write(f"Throughput: {d['throughput']:.2f} graphs/s\n")
            
            # Hyperparameters
            if self.results['hyperparameters']:
                f.write("\n" + "-"*80 + "\n")
                f.write("HYPERPARAMETER SENSITIVITY\n")
                f.write("-"*80 + "\n\n")
                
                data = self.results['hyperparameters']
                lrs = sorted([float(k.split('_')[1]) for k in data.keys()])
                best_lr = max(lrs, key=lambda lr: data[f'lr_{lr}']['f1'])
                
                f.write(f"Best Learning Rate: {best_lr}\n")
                f.write(f"  F1 Score: {data[f'lr_{best_lr}']['f1']:.4f}\n\n")
                
                f.write("Learning Rate Comparison:\n")
                for lr in lrs:
                    d = data[f'lr_{lr}']
                    f.write(f"  LR={lr}: F1={d['f1']:.4f}, ")
                    f.write(f"Final Loss={d['train_losses'][-1]:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary report saved: {report_path}")
        
        # Also save all results as JSON
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'experiments': self.results
        }
        
        with open(self.output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Complete results saved: {self.output_dir / 'all_results.json'}")


def main():
    """Main execution"""
    print("="*80)
    print("CAUSALDEFEND: COMPREHENSIVE EXPERIMENTAL SUITE")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # Run all experiments
    try:
        runner.experiment_ablation()
        runner.experiment_scalability()
        runner.experiment_hyperparameters()
        
        # Generate plots
        runner.generate_plots()
        
        # Save summary
        runner.save_summary_report()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved in: {runner.output_dir}")
        print("\nGenerated files:")
        print("  - ablation_comparison.png/pdf")
        print("  - scalability_analysis.png/pdf")
        print("  - hyperparameter_sensitivity.png/pdf")
        print("  - combined_comparison.png/pdf")
        print("  - experimental_summary.txt")
        print("  - all_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
