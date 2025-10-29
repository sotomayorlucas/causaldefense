"""
Causal Discovery Module (Tier 3)

Implements PC-Stable algorithm with temporal constraints and MITRE ATT&CK priors.
Discovers causal graphs from provenance data for explainable APT detection.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from loguru import logger

from .neural_ci_test import NeuralCITest
from ..data.provenance_graph import ProvenanceGraph


@dataclass
class ATTACKTechnique:
    """MITRE ATT&CK technique representation"""
    id: str  # e.g., "T1566"
    name: str  # e.g., "Phishing"
    tactic: str  # e.g., "Initial Access"
    description: str
    typical_sequence_position: int  # Position in typical kill chain (1-10)


class ATTACKKnowledge:
    """
    MITRE ATT&CK framework knowledge base.
    
    Provides:
    - Valid technique transitions
    - Typical kill chains
    - Attack sequence penalties
    """
    
    def __init__(self) -> None:
        self.techniques: Dict[str, ATTACKTechnique] = {}
        self.valid_transitions: Set[Tuple[str, str]] = set()
        self._initialize_attack_knowledge()
    
    def _initialize_attack_knowledge(self) -> None:
        """Initialize MITRE ATT&CK knowledge (simplified)"""
        
        # Define common techniques
        techniques_data = [
            ("T1566", "Phishing", "Initial Access", 1),
            ("T1078", "Valid Accounts", "Initial Access", 1),
            ("T1190", "Exploit Public-Facing Application", "Initial Access", 1),
            ("T1543", "Create or Modify System Process", "Persistence", 2),
            ("T1053", "Scheduled Task/Job", "Persistence", 2),
            ("T1068", "Exploitation for Privilege Escalation", "Privilege Escalation", 3),
            ("T1134", "Access Token Manipulation", "Privilege Escalation", 3),
            ("T1552", "Unsecured Credentials", "Credential Access", 4),
            ("T1003", "OS Credential Dumping", "Credential Access", 4),
            ("T1021", "Remote Services", "Lateral Movement", 5),
            ("T1570", "Lateral Tool Transfer", "Lateral Movement", 5),
            ("T1071", "Application Layer Protocol", "Command and Control", 6),
            ("T1573", "Encrypted Channel", "Command and Control", 6),
            ("T1005", "Data from Local System", "Collection", 7),
            ("T1560", "Archive Collected Data", "Collection", 7),
            ("T1041", "Exfiltration Over C2 Channel", "Exfiltration", 8),
            ("T1048", "Exfiltration Over Alternative Protocol", "Exfiltration", 8),
        ]
        
        for tid, name, tactic, position in techniques_data:
            self.techniques[tid] = ATTACKTechnique(
                id=tid,
                name=name,
                tactic=tactic,
                description=f"{tactic}: {name}",
                typical_sequence_position=position
            )
        
        # Define valid transitions (simplified kill chain)
        # Initial Access → Persistence → Privilege Escalation → ...
        self._build_valid_transitions()
    
    def _build_valid_transitions(self) -> None:
        """Build valid technique transitions based on tactics"""
        tactic_order = [
            "Initial Access",
            "Persistence",
            "Privilege Escalation",
            "Credential Access",
            "Lateral Movement",
            "Command and Control",
            "Collection",
            "Exfiltration"
        ]
        
        # Group techniques by tactic
        by_tactic = defaultdict(list)
        for tech in self.techniques.values():
            by_tactic[tech.tactic].append(tech.id)
        
        # Allow transitions within same tactic or to next tactic
        for i, tactic in enumerate(tactic_order):
            current_techs = by_tactic[tactic]
            
            # Same tactic transitions
            for t1 in current_techs:
                for t2 in current_techs:
                    self.valid_transitions.add((t1, t2))
            
            # Next tactic transitions
            if i < len(tactic_order) - 1:
                next_tactic = tactic_order[i + 1]
                next_techs = by_tactic[next_tactic]
                
                for t1 in current_techs:
                    for t2 in next_techs:
                        self.valid_transitions.add((t1, t2))
    
    def get_valid_transitions(self, technique1: str, technique2: str) -> bool:
        """Check if transition from technique1 to technique2 is valid"""
        return (technique1, technique2) in self.valid_transitions
    
    def compute_attack_penalty(self, causal_graph: nx.DiGraph) -> float:
        """
        Compute penalty for causal graphs violating typical attack sequences.
        
        Penalizes:
        - Invalid technique transitions
        - Out-of-order sequences (e.g., Exfiltration before Persistence)
        
        Returns:
            Penalty score (0 = perfect alignment with ATT&CK)
        """
        penalty = 0.0
        
        for edge in causal_graph.edges():
            source, target = edge
            
            # Get techniques (if mapped)
            source_tech = causal_graph.nodes[source].get('attack_technique')
            target_tech = causal_graph.nodes[target].get('attack_technique')
            
            if source_tech and target_tech:
                # Check if transition is valid
                if not self.get_valid_transitions(source_tech, target_tech):
                    penalty += 1.0
                
                # Check sequence order
                source_pos = self.techniques[source_tech].typical_sequence_position
                target_pos = self.techniques[target_tech].typical_sequence_position
                
                if target_pos < source_pos:
                    # Backwards transition (e.g., Exfiltration → Persistence)
                    penalty += 2.0
        
        return penalty
    
    def get_typical_kill_chains(self) -> List[List[str]]:
        """
        Get typical APT kill chains.
        
        Returns:
            List of technique sequences
        """
        return [
            # Classic APT kill chain
            ["T1566", "T1543", "T1068", "T1003", "T1021", "T1071", "T1005", "T1041"],
            # Credential theft variant
            ["T1078", "T1552", "T1021", "T1570", "T1041"],
            # Exploit-based variant
            ["T1190", "T1543", "T1134", "T1071", "T1048"],
        ]


class CausalGraph:
    """
    Causal Directed Acyclic Graph (DAG) with attack semantics.
    
    Represents causal relationships between provenance events,
    mapped to MITRE ATT&CK techniques.
    """
    
    def __init__(self, graph: nx.DiGraph) -> None:
        """
        Initialize causal graph.
        
        Args:
            graph: NetworkX DiGraph
        """
        self.graph = graph
    
    def get_parents(self, node: str) -> List[str]:
        """Get causal parents of a node"""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get causal children of a node"""
        return list(self.graph.successors(node))
    
    def extract_attack_chains(
        self,
        min_length: int = 3,
        max_length: int = 10
    ) -> List[List[str]]:
        """
        Extract attack chains as directed paths through causal graph.
        
        Args:
            min_length: Minimum chain length
            max_length: Maximum chain length
            
        Returns:
            List of attack chains (node sequences)
        """
        chains = []
        
        # Find all source nodes (no incoming edges)
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        # Find all sink nodes (no outgoing edges)
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # Extract paths from sources to sinks
        for source in sources:
            for sink in sinks:
                try:
                    # Find all simple paths
                    paths = nx.all_simple_paths(
                        self.graph,
                        source,
                        sink,
                        cutoff=max_length
                    )
                    
                    for path in paths:
                        if min_length <= len(path) <= max_length:
                            chains.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return chains
    
    def rank_chains(
        self,
        chains: List[List[str]]
    ) -> List[Tuple[List[str], float]]:
        """
        Rank attack chains by causal strength.
        
        Strength = Π edge_weights
        
        Args:
            chains: List of attack chains
            
        Returns:
            List of (chain, score) tuples, sorted by score descending
        """
        ranked = []
        
        for chain in chains:
            score = 1.0
            
            # Multiply edge weights along path
            for i in range(len(chain) - 1):
                source, target = chain[i], chain[i+1]
                edge_data = self.graph.edges.get((source, target), {})
                weight = edge_data.get('weight', 0.5)
                score *= weight
            
            ranked.append((chain, score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def generate_narrative(
        self,
        chain: List[str],
        attack_knowledge: ATTACKKnowledge
    ) -> str:
        """
        Generate natural language narrative for attack chain.
        
        Args:
            chain: Attack chain (node IDs)
            attack_knowledge: MITRE ATT&CK knowledge
            
        Returns:
            Narrative string
        """
        narrative_parts = []
        
        for i, node in enumerate(chain):
            # Get MITRE technique
            technique_id = self.graph.nodes[node].get('attack_technique')
            
            if technique_id and technique_id in attack_knowledge.techniques:
                tech = attack_knowledge.techniques[technique_id]
                
                if i == 0:
                    narrative_parts.append(f"Attack initiated via {tech.tactic}: {tech.name} ({tech.id})")
                elif i == len(chain) - 1:
                    narrative_parts.append(f"achieved {tech.tactic}: {tech.name} ({tech.id})")
                else:
                    narrative_parts.append(f"executed {tech.tactic}: {tech.name} ({tech.id})")
            else:
                # Fallback to node metadata
                node_type = self.graph.nodes[node].get('type', 'unknown')
                narrative_parts.append(f"performed {node_type} action")
        
        return ", ".join(narrative_parts) + "."


class TemporalPCStable:
    """
    PC-Stable algorithm with temporal constraints and ATT&CK priors.
    
    Implements Algorithm 3 from the paper:
    1. Skeleton discovery with temporal constraints
    2. V-structure orientation
    3. ATT&CK-guided edge orientation
    """
    
    def __init__(
        self,
        ci_test: NeuralCITest,
        attack_knowledge: Optional[ATTACKKnowledge] = None,
        max_degree: int = 3,
        significance_level: float = 0.05,
        attack_penalty_weight: float = 0.1,
    ) -> None:
        """
        Initialize Temporal PC-Stable.
        
        Args:
            ci_test: Neural CI test for independence testing
            attack_knowledge: MITRE ATT&CK knowledge base
            max_degree: Maximum node degree for CI tests
            significance_level: Significance level for CI tests
            attack_penalty_weight: Weight for ATT&CK penalty in scoring
        """
        self.ci_test = ci_test
        self.attack_knowledge = attack_knowledge or ATTACKKnowledge()
        self.max_degree = max_degree
        self.significance_level = significance_level
        self.attack_penalty_weight = attack_penalty_weight
    
    def discover_causal_graph(
        self,
        graph: ProvenanceGraph,
        verbose: bool = True
    ) -> CausalGraph:
        """
        Discover causal DAG from provenance graph.
        
        Args:
            graph: Input provenance graph
            verbose: Whether to print progress
            
        Returns:
            Discovered causal graph
        """
        logger.info(f"Starting causal discovery on graph with {len(graph)} nodes")
        
        # Convert to NetworkX
        G = graph.to_networkx()
        
        # Initialize with complete undirected graph
        skeleton = self._initialize_complete_graph(G)
        
        # Phase 1: Skeleton discovery with temporal constraints
        skeleton = self._discover_skeleton(skeleton, G, verbose)
        
        # Phase 2: Orient v-structures
        dag = self._orient_v_structures(skeleton, G)
        
        # Phase 3: Orient with ATT&CK constraints
        dag = self._orient_attack_constraints(dag, G)
        
        # Score the final graph
        score = self._score_graph(dag, G)
        logger.info(f"Causal discovery complete. Final score: {score:.3f}")
        
        return CausalGraph(dag)
    
    def _initialize_complete_graph(self, G: nx.DiGraph) -> nx.Graph:
        """Initialize complete undirected graph"""
        skeleton = nx.Graph()
        skeleton.add_nodes_from(G.nodes(data=True))
        
        # Add all possible edges
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                skeleton.add_edge(nodes[i], nodes[j])
        
        return skeleton
    
    def _discover_skeleton(
        self,
        skeleton: nx.Graph,
        G: nx.DiGraph,
        verbose: bool
    ) -> nx.Graph:
        """
        Phase 1: Skeleton discovery with temporal constraints.
        
        Algorithm 3, lines 8-23
        """
        logger.info("Phase 1: Skeleton discovery")
        
        # Separation sets (for v-structure orientation later)
        sep_sets = {}
        
        # Iterate through conditioning set sizes
        for level in range(self.max_degree + 1):
            if verbose:
                logger.info(f"  Testing conditioning sets of size {level}")
            
            # Get edges to test at this level
            edges_to_test = [
                (u, v) for u, v in skeleton.edges()
                if len(list(skeleton.neighbors(u))) + len(list(skeleton.neighbors(v))) > level
            ]
            
            for u, v in edges_to_test:
                # Check temporal constraint
                u_time = G.nodes[u].get('timestamp')
                v_time = G.nodes[v].get('timestamp')
                
                if u_time and v_time:
                    if isinstance(u_time, (int, float)):
                        u_time = datetime.fromtimestamp(u_time)
                    if isinstance(v_time, (int, float)):
                        v_time = datetime.fromtimestamp(v_time)
                    
                    # If u happens after v, they can't have u→v edge
                    # Remove edge if temporal ordering makes it impossible
                    if u_time > v_time:
                        time_diff = (u_time - v_time).total_seconds()
                        if time_diff > 3600:  # 1 hour threshold
                            skeleton.remove_edge(u, v)
                            sep_sets[(u, v)] = set()
                            continue
                
                # Get potential conditioning sets
                neighbors_u = set(skeleton.neighbors(u)) - {v}
                neighbors_v = set(skeleton.neighbors(v)) - {u}
                potential_cond = neighbors_u.union(neighbors_v)
                
                # Test conditional independence with subsets of size 'level'
                from itertools import combinations
                for cond_set in combinations(potential_cond, min(level, len(potential_cond))):
                    cond_set = set(cond_set)
                    
                    # Prepare data for CI test
                    # (Simplified: use node features)
                    x_features = G.nodes[u].get('features')
                    y_features = G.nodes[v].get('features')
                    
                    if x_features is None or y_features is None:
                        continue
                    
                    # Convert to tensors
                    import torch
                    x = torch.FloatTensor(x_features).unsqueeze(0)
                    y = torch.FloatTensor(y_features).unsqueeze(0)
                    
                    # Conditioning set features
                    if cond_set:
                        z_features = []
                        for node in cond_set:
                            node_features = G.nodes[node].get('features')
                            if node_features is not None:
                                z_features.append(node_features)
                        
                        if z_features:
                            z = torch.FloatTensor(np.mean(z_features, axis=0)).unsqueeze(0)
                        else:
                            z = None
                    else:
                        z = None
                    
                    # Test independence
                    is_independent, corr, _ = self.ci_test.test_independence(
                        x, y, z, self.significance_level
                    )
                    
                    if is_independent:
                        # Remove edge
                        if skeleton.has_edge(u, v):
                            skeleton.remove_edge(u, v)
                            sep_sets[(u, v)] = cond_set
                            sep_sets[(v, u)] = cond_set
                        break
        
        logger.info(f"  Skeleton has {skeleton.number_of_edges()} edges")
        return skeleton
    
    def _orient_v_structures(
        self,
        skeleton: nx.Graph,
        G: nx.DiGraph
    ) -> nx.DiGraph:
        """
        Phase 2: Orient v-structures.
        
        For unshielded triple u - v - w where u and w are not adjacent:
        If v not in sep_set(u, w), orient as u → v ← w
        """
        logger.info("Phase 2: Orienting v-structures")
        
        dag = nx.DiGraph()
        dag.add_nodes_from(skeleton.nodes(data=True))
        dag.add_edges_from(skeleton.edges())
        
        # Find and orient v-structures
        for v in skeleton.nodes():
            neighbors = list(skeleton.neighbors(v))
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, w = neighbors[i], neighbors[j]
                    
                    # Check if u and w are not adjacent (unshielded)
                    if not skeleton.has_edge(u, w):
                        # Orient as u → v ← w
                        if dag.has_edge(v, u):
                            dag.remove_edge(v, u)
                        if dag.has_edge(v, w):
                            dag.remove_edge(v, w)
                        
                        dag.add_edge(u, v)
                        dag.add_edge(w, v)
        
        return dag
    
    def _orient_attack_constraints(
        self,
        dag: nx.DiGraph,
        G: nx.DiGraph
    ) -> nx.DiGraph:
        """
        Phase 3: Orient edges using ATT&CK knowledge.
        
        Algorithm 3, line 27
        """
        logger.info("Phase 3: Orienting with ATT&CK constraints")
        
        # For each undirected edge, try to orient based on attack knowledge
        undirected_edges = [
            (u, v) for u, v in dag.edges()
            if dag.has_edge(v, u)  # Both directions exist = undirected
        ]
        
        for u, v in undirected_edges:
            u_tech = G.nodes[u].get('attack_technique')
            v_tech = G.nodes[v].get('attack_technique')
            
            if u_tech and v_tech:
                # Check if u → v is valid transition
                uv_valid = self.attack_knowledge.get_valid_transitions(u_tech, v_tech)
                vu_valid = self.attack_knowledge.get_valid_transitions(v_tech, u_tech)
                
                if uv_valid and not vu_valid:
                    # Orient as u → v
                    dag.remove_edge(v, u)
                elif vu_valid and not uv_valid:
                    # Orient as v → u
                    dag.remove_edge(u, v)
                # If both valid or both invalid, leave undirected
        
        return dag
    
    def _score_graph(self, dag: nx.DiGraph, G: nx.DiGraph) -> float:
        """
        Score causal graph using BIC + ATT&CK penalty.
        
        Equation (11): score = BIC + λ * ATT&CK-penalty
        """
        # BIC approximation: -k/2 * log(n)
        n = dag.number_of_nodes()
        k = dag.number_of_edges()
        
        bic = -k / 2 * np.log(n) if n > 0 else 0
        
        # ATT&CK penalty
        attack_penalty = self.attack_knowledge.compute_attack_penalty(dag)
        
        # Combined score
        score = bic - self.attack_penalty_weight * attack_penalty
        
        return score
