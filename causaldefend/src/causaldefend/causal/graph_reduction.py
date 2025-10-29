"""
Graph Reduction Module (Tier 1)

Implements Security-Aware Graph Distillation (Algorithm 2 from paper).
Reduces provenance graphs by 90-95% while preserving attack-relevant structure.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger

from ..data.provenance_graph import ProvenanceGraph, ProvenanceNode


@dataclass
class CriticalAsset:
    """Represents a critical system asset requiring protection"""
    node_id: str
    asset_type: str  # 'database', 'domain_controller', 'sensitive_file', etc.
    criticality: float  # 0.0 to 1.0
    metadata: Dict = None


class CriticalAssetManager:
    """
    Manages critical asset inventory and criticality scoring.
    
    Critical assets include:
    - Databases and data stores
    - Domain controllers / authentication servers
    - Sensitive files (credentials, keys, PII)
    - Critical services (DNS, DHCP, etc.)
    - High-value user accounts
    """
    
    def __init__(self) -> None:
        self.assets: Dict[str, CriticalAsset] = {}
        
        # Default criticality levels by asset type
        self.default_criticality = {
            'database': 0.9,
            'domain_controller': 1.0,
            'credential_file': 1.0,
            'ssh_key': 0.95,
            'admin_account': 0.85,
            'sensitive_document': 0.7,
            'system_service': 0.6,
        }
    
    def add_asset(
        self,
        node_id: str,
        asset_type: str,
        criticality: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Register a critical asset"""
        if criticality is None:
            criticality = self.default_criticality.get(asset_type, 0.5)
        
        self.assets[node_id] = CriticalAsset(
            node_id=node_id,
            asset_type=asset_type,
            criticality=criticality,
            metadata=metadata or {}
        )
    
    def get_criticality_score(self, node_id: str) -> float:
        """Get criticality score for a node (0.0 if not critical)"""
        asset = self.assets.get(node_id)
        return asset.criticality if asset else 0.0
    
    def is_critical(self, node_id: str) -> bool:
        """Check if node is a critical asset"""
        return node_id in self.assets
    
    def auto_detect_assets(self, graph: ProvenanceGraph) -> None:
        """
        Auto-detect critical assets based on heuristics.
        
        Heuristics:
        - Files in /etc/shadow, /etc/passwd, ~/.ssh/
        - Processes with uid=0 (root)
        - Database connections (port 3306, 5432, etc.)
        - Domain controller connections (port 389, 636, 88)
        """
        for node in graph.nodes.values():
            metadata = node.metadata
            
            # Check file paths for sensitive locations
            if 'path' in metadata:
                path = metadata['path'].lower()
                if any(p in path for p in ['/etc/shadow', '/etc/passwd', '/.ssh/', 'id_rsa']):
                    self.add_asset(node.id, 'credential_file', 1.0)
                elif 'database' in path or '.db' in path:
                    self.add_asset(node.id, 'database', 0.9)
            
            # Check for privileged processes
            if 'uid' in metadata and metadata['uid'] == '0':
                if 'comm' in metadata and metadata['comm'] in ['sshd', 'systemd', 'init']:
                    self.add_asset(node.id, 'system_service', 0.6)
            
            # Check for database/auth server connections
            if 'port' in metadata:
                port = int(metadata.get('port', 0))
                if port in [3306, 5432, 27017]:  # MySQL, PostgreSQL, MongoDB
                    self.add_asset(node.id, 'database', 0.9)
                elif port in [389, 636, 88]:  # LDAP, Kerberos
                    self.add_asset(node.id, 'domain_controller', 1.0)


class GraphDistiller:
    """
    Security-aware graph distillation for provenance graphs.
    
    Implements Algorithm 2:
    1. Alert-driven sampling: Extract k-hop neighborhoods around alerts
    2. Blast-radius scoring: Score nodes by potential impact
    3. Structure preservation: Maintain connectivity and attack paths
    """
    
    def __init__(
        self,
        k_hop: int = 2,
        temporal_window_hours: int = 2,
        blast_radius_threshold: float = 0.1,
    ) -> None:
        """
        Initialize distiller.
        
        Args:
            k_hop: Neighborhood size around alert nodes
            temporal_window_hours: Temporal window for related events
            blast_radius_threshold: Minimum blast radius score to include node
        """
        self.k_hop = k_hop
        self.temporal_window_hours = temporal_window_hours
        self.blast_radius_threshold = blast_radius_threshold
    
    def distill(
        self,
        graph: ProvenanceGraph,
        alert_nodes: List[str],
        critical_assets: CriticalAssetManager,
        target_reduction: float = 0.9
    ) -> Tuple[ProvenanceGraph, float]:
        """
        Distill provenance graph via security-aware sampling.
        
        Args:
            graph: Input provenance graph
            alert_nodes: List of node IDs flagged by detector
            critical_assets: Critical asset manager
            target_reduction: Target reduction ratio (0.9 = 90% reduction)
            
        Returns:
            Reduced graph
            Actual reduction ratio achieved
        """
        logger.info(f"Starting graph distillation: {len(graph)} nodes, {len(graph.edges)} edges")
        
        # Convert to NetworkX for graph algorithms
        G = graph.to_networkx()
        
        # Phase 1: Alert-driven sampling
        sampled_nodes = self._alert_driven_sampling(G, alert_nodes)
        logger.info(f"Phase 1 (Alert-driven): {len(sampled_nodes)} nodes")
        
        # Phase 2: Blast-radius scoring
        blast_scores = self._compute_blast_radius_scores(G, critical_assets)
        high_impact_nodes = {
            node for node, score in blast_scores.items()
            if score >= self.blast_radius_threshold
        }
        logger.info(f"Phase 2 (Blast-radius): {len(high_impact_nodes)} high-impact nodes")
        
        # Combine sampled and high-impact nodes
        selected_nodes = sampled_nodes.union(high_impact_nodes)
        
        # Phase 3: Structure preservation
        # Ensure connectivity between alert nodes and critical assets
        selected_nodes = self._preserve_attack_paths(
            G,
            selected_nodes,
            alert_nodes,
            critical_assets
        )
        logger.info(f"Phase 3 (Structure preservation): {len(selected_nodes)} final nodes")
        
        # Build reduced graph
        reduced_graph = self._induce_subgraph(graph, selected_nodes)
        
        # Compute reduction ratio
        reduction_ratio = 1.0 - (len(reduced_graph) / len(graph))
        
        logger.info(
            f"Distillation complete: {len(reduced_graph)} nodes "
            f"({reduction_ratio:.1%} reduction)"
        )
        
        return reduced_graph, reduction_ratio
    
    def _alert_driven_sampling(
        self,
        G: nx.DiGraph,
        alert_nodes: List[str]
    ) -> Set[str]:
        """
        Phase 1: Extract k-hop neighborhoods around alerts with temporal filtering.
        
        Algorithm 2, lines 5-10
        """
        sampled_nodes = set()
        
        for alert_node in alert_nodes:
            if alert_node not in G:
                logger.warning(f"Alert node {alert_node} not in graph")
                continue
            
            # Extract k-hop neighborhood
            k_hop_neighbors = self._extract_k_hop_neighborhood(G, alert_node, self.k_hop)
            
            # Apply temporal window filtering
            alert_timestamp = G.nodes[alert_node].get('timestamp')
            if alert_timestamp:
                k_hop_neighbors = self._temporal_window_filter(
                    G,
                    k_hop_neighbors,
                    alert_timestamp,
                    self.temporal_window_hours
                )
            
            sampled_nodes.update(k_hop_neighbors)
            
            # Always include the alert node itself
            sampled_nodes.add(alert_node)
        
        return sampled_nodes
    
    def _extract_k_hop_neighborhood(
        self,
        G: nx.DiGraph,
        center_node: str,
        k: int
    ) -> Set[str]:
        """Extract k-hop neighborhood (both forward and backward)"""
        neighbors = {center_node}
        
        # Forward k-hop (descendants)
        try:
            for node in nx.descendants_at_distance(G, center_node, k):
                neighbors.add(node)
        except nx.NetworkXError:
            pass
        
        # Backward k-hop (ancestors)
        try:
            for node in nx.ancestors(G, center_node):
                # Check if within k hops
                try:
                    if nx.shortest_path_length(G, node, center_node) <= k:
                        neighbors.add(node)
                except nx.NetworkXNoPath:
                    pass
        except nx.NetworkXError:
            pass
        
        return neighbors
    
    def _temporal_window_filter(
        self,
        G: nx.DiGraph,
        nodes: Set[str],
        reference_time: datetime,
        window_hours: int
    ) -> Set[str]:
        """Filter nodes within temporal window of reference time"""
        window = timedelta(hours=window_hours)
        
        filtered_nodes = set()
        for node in nodes:
            node_time = G.nodes[node].get('timestamp')
            if node_time:
                if isinstance(node_time, (int, float)):
                    node_time = datetime.fromtimestamp(node_time)
                
                if abs(node_time - reference_time) <= window:
                    filtered_nodes.add(node)
            else:
                # Include nodes without timestamp
                filtered_nodes.add(node)
        
        return filtered_nodes
    
    def _compute_blast_radius_scores(
        self,
        G: nx.DiGraph,
        critical_assets: CriticalAssetManager
    ) -> Dict[str, float]:
        """
        Phase 2: Compute blast radius scores for all nodes.
        
        Score = Σ_{c ∈ CriticalAssets} criticality(c) / distance(node, c)
        
        Algorithm 2, lines 13-16
        """
        scores = {node: 0.0 for node in G.nodes()}
        
        # Get all critical asset nodes
        critical_nodes = [
            node_id for node_id in critical_assets.assets.keys()
            if node_id in G
        ]
        
        if not critical_nodes:
            logger.warning("No critical assets found in graph")
            return scores
        
        # Compute shortest paths from each node to critical assets
        for node in G.nodes():
            score = 0.0
            
            for critical_node in critical_nodes:
                try:
                    # Check both directions (forward and backward in DAG)
                    distance = float('inf')
                    
                    # Try forward path
                    if nx.has_path(G, node, critical_node):
                        distance = min(distance, nx.shortest_path_length(G, node, critical_node))
                    
                    # Try backward path
                    if nx.has_path(G, critical_node, node):
                        distance = min(distance, nx.shortest_path_length(G, critical_node, node))
                    
                    if distance < float('inf'):
                        criticality = critical_assets.get_criticality_score(critical_node)
                        score += criticality / (distance + 1)  # +1 to avoid division by zero
                
                except nx.NetworkXError:
                    continue
            
            scores[node] = score
        
        return scores
    
    def _preserve_attack_paths(
        self,
        G: nx.DiGraph,
        selected_nodes: Set[str],
        alert_nodes: List[str],
        critical_assets: CriticalAssetManager
    ) -> Set[str]:
        """
        Phase 3: Preserve connectivity by including nodes on attack paths.
        
        Ensures paths exist between alert nodes and critical assets.
        """
        critical_nodes = [
            node_id for node_id in critical_assets.assets.keys()
            if node_id in G
        ]
        
        # Find all paths from alerts to critical assets
        path_nodes = set()
        
        for alert_node in alert_nodes:
            if alert_node not in G:
                continue
            
            for critical_node in critical_nodes:
                try:
                    # Find shortest path
                    if nx.has_path(G, alert_node, critical_node):
                        path = nx.shortest_path(G, alert_node, critical_node)
                        path_nodes.update(path)
                except nx.NetworkXError:
                    continue
        
        # Include path nodes in selection
        selected_nodes.update(path_nodes)
        
        return selected_nodes
    
    def _induce_subgraph(
        self,
        graph: ProvenanceGraph,
        selected_nodes: Set[str]
    ) -> ProvenanceGraph:
        """
        Induce subgraph from selected nodes.
        
        Creates new ProvenanceGraph containing only selected nodes and edges between them.
        """
        reduced_graph = ProvenanceGraph()
        
        # Add selected nodes
        for node_id in selected_nodes:
            if node_id in graph.nodes:
                reduced_graph.add_node(graph.nodes[node_id])
        
        # Add edges between selected nodes
        for edge in graph.edges:
            if edge.source in selected_nodes and edge.target in selected_nodes:
                reduced_graph.add_edge(edge)
        
        return reduced_graph
