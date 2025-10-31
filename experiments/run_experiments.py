"""
Synthetic Experimental Suite for CausalDefend
==============================================
Generates synthetic data and runs comprehensive experiments
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
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

print("="*80)
print("CAUSALDEFEND: SYNTHETIC EXPERIMENTAL SUITE")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Create output directory
output_dir = Path("experiments/results")
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Generate Synthetic Dataset
# =============================================================================
print("Generating Synthetic Provenance Graphs...")
print("-"*80)

def generate_synthetic_graph(num_nodes, avg_degree=5, is_malicious=False):
    """Generate synthetic provenance graph"""
    num_edges = int(num_nodes * avg_degree / 2)
    
    # Create edge index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create node features (64-dimensional)
    x = torch.randn(num_nodes, 64)
    
    if is_malicious:
        # Add anomalous patterns for malicious graphs
        anomaly_nodes = torch.randint(0, num_nodes, (max(1, num_nodes // 10),))
        x[anomaly_nodes] += torch.randn(len(anomaly_nodes), 64) * 2.0
    
    return Data(x=x, edge_index=edge_index, y=torch.tensor([1 if is_malicious else 0]))

# Generate datasets
train_data = []
val_data = []
test_data = []

print("Generating training data...")
for i in range(100):
    train_data.append(generate_synthetic_graph(200, is_malicious=(i % 4 == 0)))

print("Generating validation data...")
for i in range(30):
    val_data.append(generate_synthetic_graph(200, is_malicious=(i % 4 == 0)))

print("Generating test data...")
for i in range(30):
    test_data.append(generate_synthetic_graph(200, is_malicious=(i % 4 == 0)))

print(f"✓ Generated {len(train_data)} train, {len(val_data)} val, {len(test_data)} test graphs\n")

# =============================================================================
# EXPERIMENT 1: Ablation Study
# =============================================================================
print("="*80)
print("EXPERIMENT 1: Ablation Study")
print("-"*80)

ablation_configs = {
    'Full Model (Baseline)': {
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
    },
    'Larger Hidden (256)': {
        'in_channels': 64, 'hidden_channels': 256, 'num_layers': 3,
        'num_heads': 8, 'embedding_dim': 128, 'gru_hidden_dim': 128
    }
}

ablation_results = {}

for name, config in ablation_configs.items():
    print(f"\nTesting: {name}")
    model = APTDetector(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    train_losses = []
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_data))
    
    # Evaluation
    model.eval()
    test_scores = []
    test_labels = []
    inference_times = []
    
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            start = time.time()
            reconstructed, _ = model(data.x, data.edge_index)
            inference_times.append(time.time() - start)
            
            score = F.mse_loss(reconstructed, data.x).item()
            test_scores.append(score)
            test_labels.append(data.y.item())
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    # Determine threshold
    benign_scores = test_scores[test_labels == 0]
    threshold = np.percentile(benign_scores, 90) if len(benign_scores) > 0 else np.median(test_scores)
    predictions = (test_scores > threshold).astype(int)
    
    # Calculate metrics
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
        'train_losses': train_losses,
        'final_loss': train_losses[-1]
    }
    
    print(f"  F1: {f1:.4f}, Accuracy: {accuracy:.4f}, "
          f"Params: {ablation_results[name]['total_params']:,}, "
          f"Inference: {ablation_results[name]['avg_inference_time']*1000:.2f}ms")

# =============================================================================
# EXPERIMENT 2: Scalability Analysis
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: Scalability Analysis")
print("-"*80)

model = APTDetector(in_channels=64, hidden_channels=128, num_layers=3,
                   num_heads=8, embedding_dim=64, gru_hidden_dim=64).to(device)
model.eval()

graph_sizes = [100, 500, 1000, 5000, 10000, 50000]
scalability_results = {}

for size in graph_sizes:
    print(f"\nTesting {size:,} nodes...")
    num_edges = int(size * 3)
    edge_index = torch.randint(0, size, (2, num_edges), device=device)
    x = torch.randn(size, 64, device=device)
    data = Data(x=x, edge_index=edge_index)
    
    # Warm-up
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
    
    # Measure inference time
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(data.x, data.edge_index)
        times.append(time.time() - start)
    
    scalability_results[size] = {
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'throughput': float(1.0 / np.mean(times))
    }
    print(f"  Inference: {scalability_results[size]['avg_time']*1000:.2f}ms ± "
          f"{scalability_results[size]['std_time']*1000:.2f}ms, "
          f"Throughput: {scalability_results[size]['throughput']:.2f} graphs/s")

# =============================================================================
# EXPERIMENT 3: Learning Rate Comparison
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: Learning Rate Sensitivity")
print("-"*80)

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
lr_results = {}

for lr in learning_rates:
    print(f"\nLR = {lr}")
    model = APTDetector(in_channels=64, hidden_channels=128, num_layers=3,
                       num_heads=8, embedding_dim=64, gru_hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(10):
        # Training
        model.train()
        epoch_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_data))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_data:
                data = data.to(device)
                reconstructed, _ = model(data.x, data.edge_index)
                loss = F.mse_loss(reconstructed, data.x)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_data))
    
    # Evaluate on test set
    model.eval()
    test_scores = []
    test_labels = []
    
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            reconstructed, _ = model(data.x, data.edge_index)
            score = F.mse_loss(reconstructed, data.x).item()
            test_scores.append(score)
            test_labels.append(data.y.item())
    
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
    
    lr_results[lr] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    print(f"  Final train loss: {train_losses[-1]:.4f}, "
          f"Val loss: {val_losses[-1]:.4f}, F1: {f1:.4f}")

# =============================================================================
# EXPERIMENT 4: Dataset Size Impact
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 4: Dataset Size Impact")
print("-"*80)

dataset_sizes = [20, 50, 100, 200]
dataset_results = {}

for n_samples in dataset_sizes:
    print(f"\nTraining with {n_samples} samples...")
    
    # Create smaller dataset
    small_train = train_data[:n_samples]
    
    model = APTDetector(in_channels=64, hidden_channels=128, num_layers=3,
                       num_heads=8, embedding_dim=64, gru_hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training
    for epoch in range(10):
        model.train()
        for data in small_train:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    test_scores = []
    test_labels = []
    
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            reconstructed, _ = model(data.x, data.edge_index)
            score = F.mse_loss(reconstructed, data.x).item()
            test_scores.append(score)
            test_labels.append(data.y.item())
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    benign_scores = test_scores[test_labels == 0]
    threshold = np.percentile(benign_scores, 90) if len(benign_scores) > 0 else np.median(test_scores)
    predictions = (test_scores > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    f1 = 2*tp / (2*tp + fp + np.sum((predictions == 0) & (test_labels == 1))) if (2*tp + fp + np.sum((predictions == 0) & (test_labels == 1))) > 0 else 0
    
    dataset_results[n_samples] = {'f1': float(f1)}
    print(f"  F1 Score: {f1:.4f}")

# =============================================================================
# GENERATE PLOTS
# =============================================================================
print("\n" + "="*80)
print("Generating Comparison Plots")
print("-"*80)

# PLOT 1: Ablation Study
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

configs = list(ablation_results.keys())
metrics = ['f1', 'precision', 'recall', 'accuracy']

ax = axes[0, 0]
x = np.arange(len(configs))
width = 0.2
colors_metric = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics):
    values = [ablation_results[c][metric] for c in configs]
    ax.bar(x + i*width, values, width, label=metric.upper(), color=colors_metric[i], edgecolor='black')

ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Ablation Study: Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([c.replace(' ', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Parameters bar chart
ax = axes[0, 1]
params = [ablation_results[c]['total_params'] / 1e6 for c in configs]  # In millions
colors_param = ['#3498db' if p == min(params) else ('#2ecc71' if p == max(params) else '#95a5a6') for p in params]
bars = ax.bar(configs, params, color=colors_param, edgecolor='black', linewidth=2)

for bar, param in zip(bars, params):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{param:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Parameters (millions)', fontweight='bold', fontsize=12)
ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
ax.set_xticklabels([c.replace(' ', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Training curves
ax = axes[1, 0]
for i, config in enumerate(configs):
    ax.plot(ablation_results[config]['train_losses'], marker='o', linewidth=2.5, 
           label=config, markersize=4)

ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
ax.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Efficiency vs Performance scatter
ax = axes[1, 1]
f1_scores = [ablation_results[c]['f1'] for c in configs]
inf_times = [ablation_results[c]['avg_inference_time'] * 1000 for c in configs]  # ms
params_normalized = [(ablation_results[c]['total_params'] / 1e6) * 100 for c in configs]  # Size based on params

scatter = ax.scatter(inf_times, f1_scores, s=params_normalized, c=range(len(configs)), 
                    cmap='viridis', edgecolor='black', linewidth=2, alpha=0.7)

for i, config in enumerate(configs):
    ax.annotate(config.split('(')[0].strip(), (inf_times[i], f1_scores[i]), 
               fontsize=8, ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Inference Time (ms)', fontweight='bold', fontsize=12)
ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
ax.set_title('Efficiency vs Performance Trade-off\n(Bubble size = Model parameters)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'ablation_study.pdf', bbox_inches='tight')
print("✓ Saved: ablation_study.png/pdf")
plt.close()

# PLOT 2: Scalability Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sizes = sorted(scalability_results.keys())
times = [scalability_results[s]['avg_time'] * 1000 for s in sizes]
stds = [scalability_results[s]['std_time'] * 1000 for s in sizes]
throughput = [scalability_results[s]['throughput'] for s in sizes]

# Time vs Size
ax = axes[0, 0]
ax.errorbar(sizes, times, yerr=stds, marker='o', capsize=5, linewidth=3, markersize=10,
           color='#e74c3c', ecolor='#95a5a6', elinewidth=2)
ax.set_xlabel('Graph Size (nodes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Inference Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('Inference Time vs Graph Size', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Add polynomial fit
if len(sizes) > 2:
    z = np.polyfit(np.log(sizes), np.log(times), 1)
    p = np.poly1d(z)
    ax.plot(sizes, np.exp(p(np.log(sizes))), "b--", linewidth=2, 
           label=f'Complexity: O(n^{z[0]:.2f})')
    ax.legend(fontsize=11)

# Throughput
ax = axes[0, 1]
ax.plot(sizes, throughput, marker='s', linewidth=3, markersize=10, color='#2ecc71')
ax.set_xlabel('Graph Size (nodes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Throughput (graphs/sec)', fontsize=13, fontweight='bold')
ax.set_title('Processing Throughput', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Scaling efficiency
ax = axes[1, 0]
baseline_time = times[0] / sizes[0]
expected_times = [s * baseline_time for s in sizes]
efficiency = [exp / actual for exp, actual in zip(expected_times, times)]

ax.plot(sizes, efficiency, marker='D', linewidth=3, markersize=10, color='#9b59b6')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Linear Scaling (ideal)')
ax.set_xlabel('Graph Size (nodes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Scaling Efficiency\n(relative to linear)', fontsize=13, fontweight='bold')
ax.set_title('Scaling Efficiency Analysis', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Summary table
ax = axes[1, 1]
ax.axis('off')

table_data = [[f"{s:,}", f"{scalability_results[s]['avg_time']*1000:.2f}", 
              f"{scalability_results[s]['throughput']:.2f}"] 
             for s in sizes]

table = ax.table(cellText=table_data,
                colLabels=['Graph Size\n(nodes)', 'Inference Time\n(ms)', 'Throughput\n(graphs/s)'],
                cellLoc='center', loc='center',
                colWidths=[0.35, 0.35, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(3):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(sizes) + 1):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.tight_layout()
plt.savefig(output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
print("✓ Saved: scalability_analysis.png/pdf")
plt.close()

# PLOT 3: Learning Rate Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

lrs = sorted(lr_results.keys())

# Training curves
ax = axes[0, 0]
for lr in lrs:
    ax.plot(lr_results[lr]['train_losses'], marker='o', linewidth=2.5, 
           markersize=5, label=f'LR = {lr}')

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax.set_title('Training Dynamics: Learning Rate Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Validation curves
ax = axes[0, 1]
for lr in lrs:
    ax.plot(lr_results[lr]['val_losses'], marker='s', linewidth=2.5, 
           markersize=5, label=f'LR = {lr}')

ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
ax.set_title('Validation Performance', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# F1 vs LR
ax = axes[1, 0]
f1_by_lr = [lr_results[lr]['f1'] for lr in lrs]
ax.plot(lrs, f1_by_lr, marker='o', linewidth=3, markersize=12, color='#e74c3c')
ax.set_xlabel('Learning Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_title('F1 Score vs Learning Rate', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Highlight best LR
best_lr_idx = np.argmax(f1_by_lr)
ax.scatter([lrs[best_lr_idx]], [f1_by_lr[best_lr_idx]], 
          s=300, color='gold', edgecolor='black', linewidth=3, zorder=5)
ax.annotate(f'Best: LR={lrs[best_lr_idx]}\nF1={f1_by_lr[best_lr_idx]:.4f}',
           (lrs[best_lr_idx], f1_by_lr[best_lr_idx]),
           fontsize=11, fontweight='bold', ha='left', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Convergence speed
ax = axes[1, 1]
convergence_epochs = []
for lr in lrs:
    losses = lr_results[lr]['train_losses']
    final_loss = losses[-1]
    target = final_loss * 1.1
    conv_epoch = next((i for i, loss in enumerate(losses) if loss <= target), len(losses))
    convergence_epochs.append(conv_epoch)

bars = ax.bar(range(len(lrs)), convergence_epochs, color='#3498db', edgecolor='black', linewidth=2)
ax.set_xlabel('Learning Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('Epochs to Convergence', fontsize=13, fontweight='bold')
ax.set_title('Convergence Speed Analysis', fontsize=15, fontweight='bold')
ax.set_xticks(range(len(lrs)))
ax.set_xticklabels([f'{lr:.4f}' for lr in lrs], rotation=45)
ax.grid(True, alpha=0.3, axis='y')

for bar, epochs in zip(bars, convergence_epochs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{epochs}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'learning_rate_comparison.pdf', bbox_inches='tight')
print("✓ Saved: learning_rate_comparison.png/pdf")
plt.close()

# PLOT 4: Dataset Size Impact
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ds_sizes = sorted(dataset_results.keys())
f1_by_dataset = [dataset_results[s]['f1'] for s in ds_sizes]

ax.plot(ds_sizes, f1_by_dataset, marker='o', linewidth=3, markersize=12, color='#9b59b6')
ax.set_xlabel('Training Dataset Size (samples)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Impact of Training Dataset Size on Performance', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotations
for x, y in zip(ds_sizes, f1_by_dataset):
    ax.annotate(f'{y:.3f}', (x, y), fontsize=10, ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'dataset_size_impact.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'dataset_size_impact.pdf', bbox_inches='tight')
print("✓ Saved: dataset_size_impact.png/pdf")
plt.close()

# PLOT 5: Comprehensive Summary Dashboard
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

# Row 1: Ablation Comparison
ax1 = fig.add_subplot(gs[0, :2])
f1_values = [ablation_results[c]['f1'] for c in configs]
colors_ablation = ['#2ecc71' if f1 == max(f1_values) else '#3498db' for f1 in f1_values]
bars = ax1.bar(configs, f1_values, color=colors_ablation, edgecolor='black', linewidth=2)

for bar, val in zip(bars, f1_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax1.set_title('Model Configuration Performance', fontsize=15, fontweight='bold')
ax1.set_xticklabels([c.replace(' ', '\n') for c in configs], rotation=0, ha='center', fontsize=9)
ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (0.90)', alpha=0.7)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Row 1, Col 3: Scalability summary
ax2 = fig.add_subplot(gs[0, 2])
ax2.loglog(sizes, times, 'o-', linewidth=3, markersize=10, color='#e74c3c')
ax2.set_xlabel('Graph Size (nodes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Scalability', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Row 2: Training dynamics
ax3 = fig.add_subplot(gs[1, :])
for config in configs:
    ax3.plot(ablation_results[config]['train_losses'], marker='o', 
            linewidth=2.5, label=config, markersize=5)

ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax3.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax3.set_title('Training Convergence Across Configurations', fontsize=15, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10, ncol=2)
ax3.grid(True, alpha=0.3)

# Row 3: LR comparison
ax4 = fig.add_subplot(gs[2, :2])
for lr in lrs:
    ax4.plot(lr_results[lr]['train_losses'], marker='s', 
            linewidth=2.5, label=f'LR={lr}', markersize=4)

ax4.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax4.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax4.set_title('Learning Rate Sensitivity', fontsize=15, fontweight='bold')
ax4.legend(fontsize=10, ncol=3)
ax4.grid(True, alpha=0.3)

# Row 3, Col 3: Dataset size
ax5 = fig.add_subplot(gs[2, 2])
ax5.plot(ds_sizes, f1_by_dataset, 'o-', linewidth=3, markersize=10, color='#9b59b6')
ax5.set_xlabel('Dataset Size', fontsize=11, fontweight='bold')
ax5.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax5.set_title('Data Efficiency', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Row 4: Summary Table
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')

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

table = ax6.table(cellText=table_data,
                 colLabels=['Configuration', 'F1', 'Precision', 'Recall', 
                          'Accuracy', 'Inference Time', 'Parameters'],
                 cellLoc='center', loc='center',
                 colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)

for i in range(7):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

best_idx = f1_values.index(max(f1_values)) + 1
for i in range(7):
    table[(best_idx, i)].set_facecolor('#f1c40f')
    table[(best_idx, i)].set_text_props(weight='bold')

plt.suptitle('CausalDefend: Comprehensive Experimental Results Dashboard', 
            fontsize=20, fontweight='bold', y=0.995)

plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'comprehensive_dashboard.pdf', bbox_inches='tight')
print("✓ Saved: comprehensive_dashboard.png/pdf")
plt.close()

# =============================================================================
# SAVE JSON RESULTS
# =============================================================================
all_results = {
    'timestamp': datetime.now().isoformat(),
    'device': str(device),
    'ablation_study': ablation_results,
    'scalability_analysis': {str(k): v for k, v in scalability_results.items()},
    'learning_rate_comparison': {str(k): v for k, v in lr_results.items()},
    'dataset_size_impact': {str(k): v for k, v in dataset_results.items()}
}

with open(output_dir / 'experimental_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("✓ Saved: experimental_results.json")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENTAL SUMMARY")
print("="*80)

print("\n1. ABLATION STUDY:")
print("-" * 40)
best_config = max(ablation_results.keys(), key=lambda k: ablation_results[k]['f1'])
print(f"   Best Configuration: {best_config}")
print(f"     • F1 Score: {ablation_results[best_config]['f1']:.4f}")
print(f"     • Precision: {ablation_results[best_config]['precision']:.4f}")
print(f"     • Recall: {ablation_results[best_config]['recall']:.4f}")
print(f"     • Accuracy: {ablation_results[best_config]['accuracy']:.4f}")
print(f"     • Parameters: {ablation_results[best_config]['total_params']:,}")
print(f"     • Inference Time: {ablation_results[best_config]['avg_inference_time']*1000:.2f}ms")

print("\n2. SCALABILITY ANALYSIS:")
print("-" * 40)
print(f"   10,000 nodes: {scalability_results[10000]['avg_time']*1000:.2f}ms")
print(f"   50,000 nodes: {scalability_results[50000]['avg_time']*1000:.2f}ms")
print(f"   Scaling: ~O(n^{z[0]:.2f})")

print("\n3. LEARNING RATE SENSITIVITY:")
print("-" * 40)
best_lr = max(lr_results.keys(), key=lambda k: lr_results[k]['f1'])
print(f"   Optimal Learning Rate: {best_lr}")
print(f"     • F1 Score: {lr_results[best_lr]['f1']:.4f}")
print(f"     • Final Train Loss: {lr_results[best_lr]['train_losses'][-1]:.4f}")
print(f"     • Final Val Loss: {lr_results[best_lr]['val_losses'][-1]:.4f}")

print("\n4. DATASET SIZE IMPACT:")
print("-" * 40)
for size in sorted(dataset_results.keys()):
    print(f"   {size:3d} samples → F1: {dataset_results[size]['f1']:.4f}")

print("\n" + "="*80)
print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nResults directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • ablation_study.png/pdf")
print("  • scalability_analysis.png/pdf")
print("  • learning_rate_comparison.png/pdf")
print("  • dataset_size_impact.png/pdf")
print("  • comprehensive_dashboard.png/pdf")
print("  • experimental_results.json")
print("\n" + "="*80)
