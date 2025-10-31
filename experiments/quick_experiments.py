"""
Quick Experimental Suite for CausalDefend (Fast Demo Version)
==============================================================
Runs experiments with reduced epochs for quick demonstration
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Configure paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "causaldefend" / "src"))

from causaldefend.models.spatiotemporal_detector import APTDetector

warnings.filterwarnings('ignore')

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

print("="*80)
print("CAUSALDEFEND: QUICK EXPERIMENTAL DEMO")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create output directory
output_dir = Path("experiments/results")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
data_dir = project_root / "causaldefend" / "data" / "processed"
train_files = sorted(list((data_dir / "train").glob("*.pt")))[:50]  # Limit for speed
val_files = sorted(list((data_dir / "val").glob("*.pt")))[:15]
test_files = sorted(list((data_dir / "test").glob("*.pt")))[:15]

print(f"Using {len(train_files)} train, {len(val_files)} val, {len(test_files)} test graphs\n")

# =============================================================================
# EXPERIMENT 1: Ablation Study
# =============================================================================
print("EXPERIMENT 1: Ablation Study")
print("-"*80)

ablation_configs = {
    'Full Model (baseline)': {
        'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
        'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
    },
    'Single-Head Attention': {
        'in_channels': 64, 'hidden_channels': 128, 'num_layers': 3,
        'num_heads': 1, 'embedding_dim': 64, 'gru_hidden_dim': 64
    },
    'Shallow (2 layers)': {
        'in_channels': 64, 'hidden_channels': 128, 'num_layers': 2,
        'num_heads': 8, 'embedding_dim': 64, 'gru_hidden_dim': 64
    },
    'Smaller Hidden (64)': {
        'in_channels': 64, 'hidden_channels': 64, 'num_layers': 3,
        'num_heads': 8, 'embedding_dim': 32, 'gru_hidden_dim': 32
    }
}

ablation_results = {}

for name, config in ablation_configs.items():
    print(f"\nTesting: {name}")
    model = APTDetector(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training (5 epochs)
    train_losses = []
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for graph_file in train_files:
            data = torch.load(graph_file).to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_files))
    
    # Evaluation
    model.eval()
    test_scores = []
    test_labels = []
    inference_times = []
    
    with torch.no_grad():
        for graph_file in test_files:
            data = torch.load(graph_file).to(device)
            start = time.time()
            reconstructed, _ = model(data.x, data.edge_index)
            inference_times.append(time.time() - start)
            
            score = F.mse_loss(reconstructed, data.x).item()
            test_scores.append(score)
            label = 1 if 'attack' in graph_file.name else 0
            test_labels.append(label)
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    benign_scores = test_scores[test_labels == 0]
    threshold = np.percentile(benign_scores, 90) if len(benign_scores) > 0 else np.median(test_scores)
    predictions = (test_scores > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_labels)
    
    ablation_results[name] = {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'avg_inference_time': float(np.mean(inference_times)),
        'total_params': sum(p.numel() for p in model.parameters()),
        'train_losses': train_losses
    }
    
    print(f"  F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Params: {ablation_results[name]['total_params']:,}")

# =============================================================================
# EXPERIMENT 2: Scalability Analysis
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: Scalability Analysis")
print("-"*80)

model = APTDetector(in_channels=64, hidden_channels=128, num_layers=3,
                   num_heads=8, embedding_dim=64, gru_hidden_dim=64).to(device)
model.eval()

graph_sizes = [100, 500, 1000, 5000, 10000]
scalability_results = {}

for size in graph_sizes:
    num_edges = int(size * 3)
    edge_index = torch.randint(0, size, (2, num_edges), device=device)
    x = torch.randn(size, 64, device=device)
    data = Data(x=x, edge_index=edge_index)
    
    # Warm-up
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
    
    # Measure
    times = []
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            _ = model(data.x, data.edge_index)
        times.append(time.time() - start)
    
    scalability_results[size] = {
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times))
    }
    print(f"  {size:,} nodes: {scalability_results[size]['avg_time']*1000:.2f}ms")

# =============================================================================
# EXPERIMENT 3: Learning Rate Comparison
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: Learning Rate Sensitivity")
print("-"*80)

learning_rates = [0.0001, 0.0005, 0.001, 0.005]
lr_results = {}

for lr in learning_rates:
    print(f"\nLR = {lr}")
    model = APTDetector(in_channels=64, hidden_channels=128, num_layers=3,
                       num_heads=8, embedding_dim=64, gru_hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for graph_file in train_files:
            data = torch.load(graph_file).to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_files))
    
    lr_results[lr] = {'train_losses': train_losses}
    print(f"  Final loss: {train_losses[-1]:.4f}")

# =============================================================================
# GENERATE PLOTS
# =============================================================================
print("\n" + "="*80)
print("Generating Comparison Plots")
print("-"*80)

# Plot 1: Ablation Study
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

configs = list(ablation_results.keys())
metrics = ['f1', 'precision', 'recall', 'accuracy']

ax = axes[0, 0]
x = np.arange(len(configs))
width = 0.2
for i, metric in enumerate(metrics):
    values = [ablation_results[c][metric] for c in configs]
    ax.bar(x + i*width, values, width, label=metric.upper())

ax.set_xlabel('Configuration', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Ablation Study: Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(configs, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Parameters
ax = axes[0, 1]
params = [ablation_results[c]['total_params'] for c in configs]
bars = ax.bar(configs, params, color='skyblue', edgecolor='black', linewidth=2)
for bar, param in zip(bars, params):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{param/1e6:.2f}M', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Parameters (millions)', fontweight='bold')
ax.set_title('Model Complexity', fontsize=14, fontweight='bold')
ax.set_xticklabels(configs, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Training curves
ax = axes[1, 0]
for config in configs:
    ax.plot(ablation_results[config]['train_losses'], marker='o', linewidth=2, label=config)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Training Loss', fontweight='bold')
ax.set_title('Training Convergence', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Efficiency Trade-off
ax = axes[1, 1]
f1_scores = [ablation_results[c]['f1'] for c in configs]
inf_times = [ablation_results[c]['avg_inference_time'] * 1000 for c in configs]
scatter = ax.scatter(inf_times, f1_scores, s=300, c=range(len(configs)), 
                    cmap='viridis', edgecolor='black', linewidth=2)
for i, config in enumerate(configs):
    ax.annotate(config, (inf_times[i], f1_scores[i]), 
               fontsize=8, ha='center', va='bottom')
ax.set_xlabel('Inference Time (ms)', fontweight='bold')
ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Efficiency vs Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'ablation_study.pdf', bbox_inches='tight')
print("✓ Saved: ablation_study.png/pdf")

# Plot 2: Scalability
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sizes = sorted(scalability_results.keys())
times = [scalability_results[s]['avg_time'] * 1000 for s in sizes]
stds = [scalability_results[s]['std_time'] * 1000 for s in sizes]

ax = axes[0]
ax.errorbar(sizes, times, yerr=stds, marker='o', capsize=5, linewidth=3, markersize=10,
           color='darkblue', ecolor='red')
ax.set_xlabel('Graph Size (nodes)', fontsize=14, fontweight='bold')
ax.set_ylabel('Inference Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Scalability Analysis', fontsize=16, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(np.log(sizes), np.log(times), 1)
p = np.poly1d(z)
ax.plot(sizes, np.exp(p(np.log(sizes))), "r--", linewidth=2, 
       label=f'Trend: O(n^{z[0]:.2f})')
ax.legend(fontsize=12)

ax = axes[1]
throughput = [1.0 / (scalability_results[s]['avg_time']) for s in sizes]
ax.plot(sizes, throughput, marker='s', linewidth=3, markersize=10, color='green')
ax.set_xlabel('Graph Size (nodes)', fontsize=14, fontweight='bold')
ax.set_ylabel('Throughput (graphs/sec)', fontsize=14, fontweight='bold')
ax.set_title('Processing Throughput', fontsize=16, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
print("✓ Saved: scalability_analysis.png/pdf")

# Plot 3: Learning Rate
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for lr in sorted(lr_results.keys()):
    ax.plot(lr_results[lr]['train_losses'], marker='o', linewidth=3, 
           markersize=8, label=f'LR = {lr}')

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
ax.set_title('Learning Rate Sensitivity Analysis', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'learning_rate_comparison.pdf', bbox_inches='tight')
print("✓ Saved: learning_rate_comparison.png/pdf")

# Plot 4: Combined Summary
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Ablation bars
ax1 = fig.add_subplot(gs[0, :2])
f1_values = [ablation_results[c]['f1'] for c in configs]
colors = ['#2ecc71' if f1 == max(f1_values) else '#3498db' for f1 in f1_values]
bars = ax1.bar(configs, f1_values, color=colors, edgecolor='black', linewidth=2)
for bar, val in zip(bars, f1_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax1.set_title('Model Configuration Comparison', fontsize=15, fontweight='bold')
ax1.set_xticklabels(configs, rotation=45, ha='right')
ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (0.90)')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Scalability
ax2 = fig.add_subplot(gs[0, 2])
ax2.loglog(sizes, times, 'o-', linewidth=3, markersize=10, color='orange')
ax2.set_xlabel('Graph Size', fontsize=11, fontweight='bold')
ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Scalability', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Training curves
ax3 = fig.add_subplot(gs[1, :])
for config in configs:
    ax3.plot(ablation_results[config]['train_losses'], marker='o', 
            linewidth=2, label=config, markersize=6)
ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax3.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax3.set_title('Training Dynamics', fontsize=15, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# Summary Table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

table_data = []
for config in configs:
    row = [
        config,
        f"{ablation_results[config]['f1']:.4f}",
        f"{ablation_results[config]['precision']:.4f}",
        f"{ablation_results[config]['recall']:.4f}",
        f"{ablation_results[config]['accuracy']:.4f}",
        f"{ablation_results[config]['avg_inference_time']*1000:.2f}ms",
        f"{ablation_results[config]['total_params']/1e6:.2f}M"
    ]
    table_data.append(row)

table = ax4.table(cellText=table_data,
                 colLabels=['Configuration', 'F1', 'Precision', 'Recall', 
                          'Accuracy', 'Inference', 'Parameters'],
                 cellLoc='center', loc='center',
                 colWidths=[0.3, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(7):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

best_idx = f1_values.index(max(f1_values)) + 1
for i in range(7):
    table[(best_idx, i)].set_facecolor('#f1c40f')
    table[(best_idx, i)].set_text_props(weight='bold')

plt.suptitle('CausalDefend: Comprehensive Experimental Results', 
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig(output_dir / 'comprehensive_results.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'comprehensive_results.pdf', bbox_inches='tight')
print("✓ Saved: comprehensive_results.png/pdf")

# Save JSON results
all_results = {
    'timestamp': datetime.now().isoformat(),
    'ablation': ablation_results,
    'scalability': {str(k): v for k, v in scalability_results.items()},
    'learning_rates': {str(k): v for k, v in lr_results.items()}
}

with open(output_dir / 'experimental_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("✓ Saved: experimental_results.json")

# Summary Report
print("\n" + "="*80)
print("EXPERIMENTAL SUMMARY")
print("="*80)
print("\nAblation Study:")
best_config = max(ablation_results.keys(), key=lambda k: ablation_results[k]['f1'])
print(f"  Best configuration: {best_config}")
print(f"    F1: {ablation_results[best_config]['f1']:.4f}")
print(f"    Precision: {ablation_results[best_config]['precision']:.4f}")
print(f"    Recall: {ablation_results[best_config]['recall']:.4f}")
print(f"    Accuracy: {ablation_results[best_config]['accuracy']:.4f}")

print("\nScalability:")
print(f"  10,000 nodes: {scalability_results[10000]['avg_time']*1000:.2f}ms")

print("\n" + "="*80)
print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nResults saved in: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • ablation_study.png/pdf")
print("  • scalability_analysis.png/pdf")
print("  • learning_rate_comparison.png/pdf")
print("  • comprehensive_results.png/pdf")
print("  • experimental_results.json")
