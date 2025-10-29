"""
Dataset Preparation Script for CausalDefend - SIMPLIFIED VERSION

Generates synthetic provenance graphs for training when DARPA TC is not available.

Usage:
    python scripts/prepare_dataset_simple.py --output data/processed --num-graphs 200
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import pickle

import networkx as nx
import numpy as np
from tqdm import tqdm

# Simple graph generator that doesn't use complex classes


def generate_simple_dataset(
    num_benign: int = 100,
    num_attack: int = 100,
    avg_nodes: int = 150,
    output_dir: Path = Path("data/processed")
):
    """Generate simple synthetic dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic dataset...")
    print(f"  Benign graphs: {num_benign}")
    print(f"  Attack graphs: {num_attack}")
    print(f"  Total: {num_benign + num_attack}")
    
    all_graphs = []
    
    # Generate benign graphs
    print("\nGenerating benign graphs...")
    for i in tqdm(range(num_benign)):
        G = nx.DiGraph()
        num_nodes = random.randint(avg_nodes - 50, avg_nodes + 50)
        
        # Add nodes
        for j in range(num_nodes):
            G.add_node(j, node_type="process" if j % 3 == 0 else "file")
        
        # Add random edges
        num_edges = num_nodes * 2
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                G.add_edge(src, dst, edge_type="read")
        
        # Create features
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        
        all_graphs.append((G, features, False))  # False = benign
    
    # Generate attack graphs
    print("Generating attack graphs...")
    for i in tqdm(range(num_attack)):
        G = nx.DiGraph()
        num_nodes = random.randint(avg_nodes - 50, avg_nodes + 50)
        
        # Add nodes
        for j in range(num_nodes):
            G.add_node(j, node_type="process" if j % 3 == 0 else "file")
        
        # Add random edges
        num_edges = num_nodes * 2
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                G.add_edge(src, dst, edge_type="read")
        
        # Add attack chain (specific pattern)
        attack_chain_length = 6
        attack_nodes = random.sample(range(num_nodes), attack_chain_length)
        for k in range(len(attack_nodes) - 1):
            G.add_edge(attack_nodes[k], attack_nodes[k+1], edge_type="malicious")
        
        # Create features with anomaly signature
        features = np.random.randn(num_nodes, 64).astype(np.float32)
        for node in attack_nodes:
            features[node, :10] += 2.0  # Anomalous signature
        
        all_graphs.append((G, features, True))  # True = attack
    
    # Shuffle
    random.shuffle(all_graphs)
    
    # Split into train/val/test
    n = len(all_graphs)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    splits = {
        "train": all_graphs[:train_size],
        "val": all_graphs[train_size:train_size + val_size],
        "test": all_graphs[train_size + val_size:],
    }
    
    # Save each split
    print("\nSaving dataset...")
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"  Saving {split_name}: {len(split_data)} graphs...")
        
        for idx, (graph, features, is_attack) in enumerate(tqdm(split_data, desc=f"  {split_name}")):
            # Save graph
            graph_file = split_dir / f"graph_{idx}.pkl"
            with open(graph_file, "wb") as f:
                pickle.dump(graph, f)
            
            # Save features
            feature_file = split_dir / f"features_{idx}.npy"
            np.save(feature_file, features)
            
            # Save label
            label_file = split_dir / f"label_{idx}.json"
            with open(label_file, "w") as f:
                json.dump({
                    "is_attack": is_attack,
                    "attack_nodes": [],
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                }, f)
    
    # Save metadata
    metadata = {
        "total_graphs": n,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "feature_dim": 64,
        "creation_date": datetime.now().isoformat(),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset saved to {output_dir}")
    print(f"  - Train: {len(splits['train'])} graphs")
    print(f"  - Val: {len(splits['val'])} graphs")
    print(f"  - Test: {len(splits['test'])} graphs")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare CausalDefend dataset (simple version)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-benign",
        type=int,
        default=100,
        help="Number of benign graphs"
    )
    parser.add_argument(
        "--num-attack",
        type=int,
        default=100,
        help="Number of attack graphs"
    )
    parser.add_argument(
        "--avg-nodes",
        type=int,
        default=150,
        help="Average nodes per graph"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_simple_dataset(
        num_benign=args.num_benign,
        num_attack=args.num_attack,
        avg_nodes=args.avg_nodes,
        output_dir=args.output
    )
    
    print("\n✓ Dataset preparation complete!")
    print("\nNext steps:")
    print("  1. Train detector: python scripts/train_detector.py")
    print("  2. Train CI tester: python scripts/train_ci_tester.py")


if __name__ == "__main__":
    main()
