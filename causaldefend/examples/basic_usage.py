"""
Example: Basic usage of CausalDefend

This example demonstrates how to:
1. Parse system logs into provenance graphs
2. Train APT detector
3. Detect anomalies
4. Get explanations
"""

from datetime import datetime
import numpy as np
import torch

from causaldefend import ProvenanceGraph, ProvenanceNode, ProvenanceEdge, ProvenanceParser
from causaldefend.data.provenance_graph import NodeType, RelationType
from causaldefend import APTDetector


def create_sample_graph() -> ProvenanceGraph:
    """Create a sample provenance graph for testing"""
    graph = ProvenanceGraph()
    
    # Create nodes
    process1 = ProvenanceNode(
        id="proc_1",
        node_type=NodeType.PROCESS,
        features=np.random.randn(64),
        timestamp=datetime.now(),
        metadata={"pid": "1234", "comm": "bash", "uid": "1000"}
    )
    
    file1 = ProvenanceNode(
        id="file_1",
        node_type=NodeType.FILE,
        features=np.random.randn(64),
        timestamp=datetime.now(),
        metadata={"path": "/etc/passwd", "size": "2048"}
    )
    
    process2 = ProvenanceNode(
        id="proc_2",
        node_type=NodeType.PROCESS,
        features=np.random.randn(64),
        timestamp=datetime.now(),
        metadata={"pid": "5678", "comm": "cat", "uid": "0"}
    )
    
    # Add nodes to graph
    graph.add_node(process1)
    graph.add_node(file1)
    graph.add_node(process2)
    
    # Create edges
    edge1 = ProvenanceEdge(
        source="proc_1",
        target="proc_2",
        relation_type=RelationType.FORK,
        timestamp=datetime.now(),
        metadata={"syscall": "fork"}
    )
    
    edge2 = ProvenanceEdge(
        source="proc_2",
        target="file_1",
        relation_type=RelationType.READ,
        timestamp=datetime.now(),
        metadata={"syscall": "open", "flags": "O_RDONLY"}
    )
    
    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    return graph


def example_graph_creation():
    """Example: Create and manipulate provenance graphs"""
    print("=== Example 1: Graph Creation ===\n")
    
    # Create sample graph
    graph = create_sample_graph()
    
    print(f"Created graph: {graph}")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    
    # Convert to PyTorch Geometric format
    pyg_data = graph.to_pytorch_geometric()
    print(f"\nPyTorch Geometric conversion:")
    print(f"  Node types: {list(pyg_data.node_types)}")
    print(f"  Edge types: {list(pyg_data.edge_types)}")
    
    # Save and load
    graph.save("example_graph.json", format="json")
    loaded_graph = ProvenanceGraph.load("example_graph.json", format="json")
    print(f"\nLoaded graph: {loaded_graph}")


def example_log_parsing():
    """Example: Parse system logs"""
    print("\n=== Example 2: Log Parsing ===\n")
    
    # Create sample JSON log
    sample_log = {
        "nodes": [
            {
                "id": "process_1234",
                "type": "process",
                "timestamp": "2024-10-29T12:00:00",
                "attributes": {
                    "pid": 1234,
                    "comm": "nginx",
                    "uid": 33
                }
            },
            {
                "id": "file_5678",
                "type": "file",
                "timestamp": "2024-10-29T12:00:01",
                "attributes": {
                    "path": "/var/log/nginx/access.log",
                    "size": 102400
                }
            }
        ],
        "edges": [
            {
                "source": "process_1234",
                "target": "file_5678",
                "relation": "write",
                "timestamp": "2024-10-29T12:00:02"
            }
        ]
    }
    
    import json
    with open("sample_log.json", "w") as f:
        json.dump(sample_log, f)
    
    # Parse log
    parser = ProvenanceParser(feature_dim=64)
    graph = parser.parse_logs("sample_log.json", format="json")
    
    print(f"Parsed graph from logs: {graph}")


def example_detection():
    """Example: Train and use APT detector"""
    print("\n=== Example 3: APT Detection ===\n")
    
    # Initialize detector
    detector = APTDetector(
        in_channels=64,
        hidden_channels=128,
        embedding_dim=64,
        num_heads=4,
        num_layers=2
    )
    
    print(f"Detector architecture:\n{detector}\n")
    
    # Create sample graph for detection
    graph = create_sample_graph()
    pyg_data = graph.to_pytorch_geometric()
    
    # Process with detector (inference mode)
    detector.eval()
    with torch.no_grad():
        # Combine all node features (simplified for example)
        all_features = []
        for node_type in pyg_data.node_types:
            if hasattr(pyg_data[node_type], 'x'):
                all_features.append(pyg_data[node_type].x)
        
        if all_features:
            x = torch.cat(all_features, dim=0)
            # Use first edge type (simplified)
            edge_types = list(pyg_data.edge_types)
            if edge_types:
                edge_index = pyg_data[edge_types[0]].edge_index
                
                # Forward pass
                graph_embedding, edge_probs = detector.forward(x, edge_index)
                
                print(f"Graph embedding shape: {graph_embedding.shape}")
                print(f"Edge probabilities shape: {edge_probs.shape}")
                print(f"Sample edge probabilities: {edge_probs[:5]}")


def example_graph_reduction():
    """Example: Graph reduction for scalability"""
    print("\n=== Example 4: Graph Reduction ===\n")
    
    from causaldefend.causal.graph_reduction import GraphDistiller, CriticalAssetManager
    
    # Create larger sample graph
    graph = ProvenanceGraph()
    
    # Add multiple nodes
    for i in range(100):
        node = ProvenanceNode(
            id=f"node_{i}",
            node_type=NodeType.PROCESS if i % 2 == 0 else NodeType.FILE,
            features=np.random.randn(64),
            timestamp=datetime.now(),
            metadata={"index": i}
        )
        graph.add_node(node)
    
    # Add edges
    for i in range(50):
        edge = ProvenanceEdge(
            source=f"node_{i}",
            target=f"node_{i+1}",
            relation_type=RelationType.READ,
            timestamp=datetime.now()
        )
        graph.add_edge(edge)
    
    print(f"Original graph: {len(graph)} nodes")
    
    # Setup critical assets
    assets = CriticalAssetManager()
    assets.add_asset("node_50", "database", criticality=0.9)
    assets.add_asset("node_75", "credential_file", criticality=1.0)
    
    # Reduce graph
    distiller = GraphDistiller(k_hop=2)
    reduced_graph, reduction_ratio = distiller.distill(
        graph=graph,
        alert_nodes=["node_10", "node_20"],
        critical_assets=assets
    )
    
    print(f"Reduced graph: {len(reduced_graph)} nodes")
    print(f"Reduction ratio: {reduction_ratio:.1%}")


def main():
    """Run all examples"""
    print("CausalDefend Examples\n" + "="*50 + "\n")
    
    example_graph_creation()
    example_log_parsing()
    example_detection()
    example_graph_reduction()
    
    print("\n" + "="*50)
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()
