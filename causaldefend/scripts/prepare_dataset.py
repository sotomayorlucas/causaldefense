"""
Dataset Preparation Script for CausalDefend

Generates synthetic provenance graphs for training when DARPA TC is not available.
Includes both benign and APT attack patterns.

Usage:
    python scripts/prepare_dataset.py --output data/processed --num-graphs 1000
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle

import networkx as nx
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add src to path
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from causaldefend.data.provenance_graph import ProvenanceGraph, ProvenanceNode, ProvenanceEdge, NodeType, RelationType
except ImportError:
    # Fallback: import directly from src
    from data.provenance_graph import ProvenanceGraph, ProvenanceNode, ProvenanceEdge, NodeType, RelationType


class SyntheticDataGenerator:
    """Generate synthetic provenance graphs with APT attack patterns"""
    
    # MITRE ATT&CK patterns
    ATTACK_PATTERNS = {
        "phishing": {
            "chain": [
                ("email_client", "browser", "process_create"),
                ("browser", "downloader", "process_create"),
                ("downloader", "malware.exe", "file_write"),
                ("malware.exe", "malware_process", "process_create"),
            ],
            "probability": 0.15,
        },
        "credential_dump": {
            "chain": [
                ("attacker_process", "lsass.exe", "process_read"),
                ("lsass.exe", "credentials.txt", "file_write"),
                ("credentials.txt", "exfil_process", "file_read"),
            ],
            "probability": 0.20,
        },
        "lateral_movement": {
            "chain": [
                ("initial_access", "psexec.exe", "process_create"),
                ("psexec.exe", "remote_host", "network_connect"),
                ("remote_host", "admin_share", "file_write"),
            ],
            "probability": 0.18,
        },
        "data_exfiltration": {
            "chain": [
                ("search_process", "sensitive_data.db", "file_read"),
                ("sensitive_data.db", "compress.exe", "process_create"),
                ("compress.exe", "archive.zip", "file_write"),
                ("archive.zip", "c2_server", "network_send"),
            ],
            "probability": 0.22,
        },
        "privilege_escalation": {
            "chain": [
                ("exploit.exe", "kernel_module", "module_load"),
                ("kernel_module", "system_process", "process_create"),
                ("system_process", "admin_token", "token_create"),
            ],
            "probability": 0.25,
        },
    }
    
    # Benign activity patterns
    BENIGN_PATTERNS = [
        ("chrome.exe", "google.com", "network_connect"),
        ("word.exe", "document.docx", "file_write"),
        ("explorer.exe", "Desktop", "file_read"),
        ("svchost.exe", "system_service", "process_create"),
        ("python.exe", "script.py", "file_read"),
        ("git.exe", "github.com", "network_connect"),
        ("vscode.exe", "code.py", "file_write"),
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator"""
        random.seed(seed)
        np.random.seed(seed)
        self.node_counter = 0
        self.timestamp = datetime.now()
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        self.node_counter += 1
        return f"node_{self.node_counter}"
    
    def _generate_timestamp(self, offset_seconds: int = 0) -> str:
        """Generate timestamp"""
        ts = self.timestamp + timedelta(seconds=offset_seconds)
        return ts.isoformat()
    
    def _create_benign_graph(self, num_nodes: int = 100) -> ProvenanceGraph:
        """Create benign provenance graph"""
        graph = ProvenanceGraph()
        
        # Create baseline system activity
        time_offset = 0
        
        for _ in range(num_nodes):
            # Random benign pattern
            pattern = random.choice(self.BENIGN_PATTERNS)
            src_name, dst_name, edge_type = pattern
            
            # Create source node
            src_id = self._generate_node_id()
            src_type = "process" if ".exe" in src_name else "file"
            src_node = ProvenanceNode(
                id=src_id,
                node_type=NodeType(src_type),
                features=np.random.randn(64) * 0.1,  # Random features
                timestamp=datetime.fromisoformat(self._generate_timestamp(time_offset)),
                metadata={
                    "name": src_name,
                    "pid": random.randint(1000, 9999) if src_type == "process" else None,
                }
            )
            graph.add_node(src_node)
            
            # Create destination node
            dst_id = self._generate_node_id()
            if "network" in edge_type:
                dst_type = "network"
            elif ".exe" in dst_name or "process" in dst_name:
                dst_type = "process"
            else:
                dst_type = "file"
            
            dst_node = ProvenanceNode(
                node_id=dst_id,
                node_type=dst_type,
                timestamp=self._generate_timestamp(time_offset + 1),
                properties={
                    "name": dst_name,
                    "address": f"192.168.1.{random.randint(1, 254)}" if dst_type == "network" else None,
                }
            )
            graph.add_node(dst_node)
            
            # Create edge
            edge = ProvenanceEdge(
                source_id=src_id,
                target_id=dst_id,
                edge_type=edge_type,
                timestamp=self._generate_timestamp(time_offset + 1),
                properties={}
            )
            graph.add_edge(edge)
            
            time_offset += random.randint(5, 30)
        
        return graph
    
    def _inject_attack(
        self,
        graph: ProvenanceGraph,
        attack_type: str
    ) -> Tuple[ProvenanceGraph, List[str]]:
        """Inject attack pattern into benign graph"""
        pattern = self.ATTACK_PATTERNS[attack_type]
        attack_nodes = []
        
        time_offset = random.randint(50, 150)
        
        for src_name, dst_name, edge_type in pattern["chain"]:
            # Create attack nodes
            src_id = self._generate_node_id()
            src_type = "process" if "process" in src_name or ".exe" in src_name else "file"
            
            src_node = ProvenanceNode(
                node_id=src_id,
                node_type=src_type,
                timestamp=self._generate_timestamp(time_offset),
                properties={
                    "name": src_name,
                    "pid": random.randint(1000, 9999) if src_type == "process" else None,
                    "suspicious": True,  # Ground truth label
                }
            )
            graph.add_node(src_node)
            attack_nodes.append(src_id)
            
            dst_id = self._generate_node_id()
            if "network" in edge_type or "server" in dst_name:
                dst_type = "network"
            elif "process" in dst_name or ".exe" in dst_name:
                dst_type = "process"
            else:
                dst_type = "file"
            
            dst_node = ProvenanceNode(
                node_id=dst_id,
                node_type=dst_type,
                timestamp=self._generate_timestamp(time_offset + 2),
                properties={
                    "name": dst_name,
                    "suspicious": True,
                }
            )
            graph.add_node(dst_node)
            attack_nodes.append(dst_id)
            
            # Create attack edge
            edge = ProvenanceEdge(
                source_id=src_id,
                target_id=dst_id,
                edge_type=edge_type,
                timestamp=self._generate_timestamp(time_offset + 2),
                properties={"suspicious": True}
            )
            graph.add_edge(edge)
            
            time_offset += random.randint(10, 60)
        
        return graph, attack_nodes
    
    def generate_dataset(
        self,
        num_benign: int = 500,
        num_attack: int = 500,
        avg_nodes_per_graph: int = 150
    ) -> List[Tuple[ProvenanceGraph, bool, List[str]]]:
        """
        Generate complete dataset.
        
        Returns:
            List of (graph, is_attack, attack_node_ids)
        """
        dataset = []
        
        logger.info(f"Generating {num_benign} benign graphs...")
        for _ in tqdm(range(num_benign), desc="Benign graphs"):
            num_nodes = random.randint(avg_nodes_per_graph - 50, avg_nodes_per_graph + 50)
            graph = self._create_benign_graph(num_nodes)
            dataset.append((graph, False, []))
            self.timestamp += timedelta(hours=1)
        
        logger.info(f"Generating {num_attack} attack graphs...")
        for _ in tqdm(range(num_attack), desc="Attack graphs"):
            # Start with benign activity
            num_benign_nodes = random.randint(80, 120)
            graph = self._create_benign_graph(num_benign_nodes)
            
            # Inject attack
            attack_type = random.choice(list(self.ATTACK_PATTERNS.keys()))
            graph, attack_nodes = self._inject_attack(graph, attack_type)
            
            dataset.append((graph, True, attack_nodes))
            self.timestamp += timedelta(hours=1)
        
        # Shuffle
        random.shuffle(dataset)
        
        return dataset


def extract_features(graph: ProvenanceGraph) -> np.ndarray:
    """
    Extract node features from provenance graph.
    
    Features:
        - Node degree (in/out)
        - Node type (one-hot)
        - Temporal features
        - Suspicious flag (if present)
    """
    num_nodes = len(graph.nodes)
    feature_dim = 64  # Match detector input
    
    features = np.zeros((num_nodes, feature_dim))
    
    node_to_idx = {node.node_id: idx for idx, node in enumerate(graph.nodes.values())}
    
    for idx, node in enumerate(graph.nodes.values()):
        # Degree features
        in_degree = len([e for e in graph.edges.values() if e.target_id == node.node_id])
        out_degree = len([e for e in graph.edges.values() if e.source_id == node.node_id])
        features[idx, 0] = in_degree / 10.0  # Normalize
        features[idx, 1] = out_degree / 10.0
        
        # Node type one-hot
        type_mapping = {"process": 2, "file": 3, "network": 4, "registry": 5}
        type_idx = type_mapping.get(node.node_type, 6)
        features[idx, type_idx] = 1.0
        
        # Suspicious flag
        if node.properties.get("suspicious", False):
            features[idx, 7] = 1.0
        
        # Add some random features for diversity
        features[idx, 8:] = np.random.randn(feature_dim - 8) * 0.1
    
    return features


def save_dataset(
    dataset: List[Tuple[ProvenanceGraph, bool, List[str]]],
    output_dir: Path,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
):
    """Save dataset to disk with train/val/test splits"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    n = len(dataset)
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Save each split
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving {split_name} split...")
        
        for idx, (graph, is_attack, attack_nodes) in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            # Save graph
            graph_file = split_dir / f"graph_{idx}.pkl"
            with open(graph_file, "wb") as f:
                pickle.dump(graph, f)
            
            # Extract and save features
            features = extract_features(graph)
            feature_file = split_dir / f"features_{idx}.npy"
            np.save(feature_file, features)
            
            # Save label
            label_file = split_dir / f"label_{idx}.json"
            with open(label_file, "w") as f:
                json.dump({
                    "is_attack": is_attack,
                    "attack_nodes": attack_nodes,
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                }, f)
    
    # Save metadata
    metadata = {
        "total_graphs": n,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "feature_dim": 64,
        "creation_date": datetime.now().isoformat(),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Dataset saved to {output_dir}")
    logger.info(f"  - Train: {len(train_data)} graphs")
    logger.info(f"  - Val: {len(val_data)} graphs")
    logger.info(f"  - Test: {len(test_data)} graphs")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare CausalDefend dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-benign",
        type=int,
        default=500,
        help="Number of benign graphs"
    )
    parser.add_argument(
        "--num-attack",
        type=int,
        default=500,
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
    
    logger.info("Starting dataset generation...")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Benign graphs: {args.num_benign}")
    logger.info(f"  Attack graphs: {args.num_attack}")
    logger.info(f"  Avg nodes/graph: {args.avg_nodes}")
    
    # Generate dataset
    generator = SyntheticDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(
        num_benign=args.num_benign,
        num_attack=args.num_attack,
        avg_nodes_per_graph=args.avg_nodes
    )
    
    # Save dataset
    save_dataset(dataset, args.output)
    
    logger.info("✓ Dataset preparation complete!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Train detector: python scripts/train_detector.py")
    logger.info(f"  2. Train CI tester: python scripts/train_ci_tester.py")


if __name__ == "__main__":
    main()
