"""
Causal Explanation and Attack Narrative Generation

Generates human-readable explanations of detected attacks using
causal reasoning and MITRE ATT&CK framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch
from jinja2 import Template

from ..causal.causal_discovery import ATTACKKnowledge, ATTACKTechnique, CausalGraph


@dataclass
class InterventionQuery:
    """Interventional query: do(X=x)"""
    
    variable: str
    value: float
    target: str  # Variable to observe after intervention


@dataclass
class CounterfactualQuery:
    """Counterfactual query: Y_x(u) where X had been x"""
    
    factual_scenario: Dict[str, float]
    intervention: Dict[str, float]
    target: str


@dataclass
class CausalExplanation:
    """Complete causal explanation of an attack"""
    
    attack_chain: List[str]  # Sequence of nodes
    attack_techniques: List[ATTACKTechnique]  # MITRE techniques
    causal_effects: Dict[Tuple[str, str], float]  # (cause, effect) -> strength
    narrative: str  # Human-readable description
    counterfactuals: List[str]  # "What if..." scenarios
    critical_nodes: List[str]  # High-impact intervention points
    confidence: float  # Overall explanation confidence


class CausalExplainer:
    """
    Causal explainer for APT attack paths.
    
    Answers interventional and counterfactual queries using
    Pearl's causal hierarchy (equations 13, 15).
    """
    
    def __init__(
        self,
        causal_graph: CausalGraph,
        attack_knowledge: ATTACKKnowledge
    ) -> None:
        """
        Initialize causal explainer.
        
        Args:
            causal_graph: Discovered causal DAG
            attack_knowledge: MITRE ATT&CK knowledge base
        """
        self.causal_graph = causal_graph
        self.attack_knowledge = attack_knowledge
    
    def interventional_effect(
        self,
        query: InterventionQuery
    ) -> float:
        """
        Compute interventional effect: E[Y | do(X=x)]
        
        Uses do-calculus to compute effect of intervention.
        Equation (13): P(Y|do(X=x)) = Σ_Z P(Y|X=x,Z) P(Z)
        
        Args:
            query: Interventional query
            
        Returns:
            Expected value of target after intervention
        """
        # Find parent variables of target (confounders)
        target_node = query.target
        parents = list(self.causal_graph.graph.predecessors(target_node))
        
        # If no confounders, interventional = conditional
        if not parents:
            return self._conditional_expectation(query.target, {
                query.variable: query.value
            })
        
        # Back-door adjustment: adjust for parents
        # E[Y|do(X=x)] = Σ_Z E[Y|X=x,Z] P(Z)
        total_effect = 0.0
        
        # For simplicity, assume uniform distribution over parent configs
        # In practice, would use learned distribution
        parent_configs = self._sample_parent_configs(parents, n_samples=100)
        
        for parent_vals in parent_configs:
            condition = {query.variable: query.value}
            condition.update(parent_vals)
            
            effect = self._conditional_expectation(target_node, condition)
            total_effect += effect
        
        avg_effect = total_effect / len(parent_configs)
        return avg_effect
    
    def counterfactual_reasoning(
        self,
        query: CounterfactualQuery
    ) -> float:
        """
        Answer counterfactual query: Y_x(u)
        
        "What would Y have been if X had been x, given that
        we observed factual scenario?"
        
        Equation (15): Y_x(u) = f_Y(x, PA_Y\X(u), U_Y(u))
        
        Args:
            query: Counterfactual query
            
        Returns:
            Counterfactual value of target
        """
        # Step 1: Abduction - infer latent variables U from factual
        latent_u = self._infer_latent(query.factual_scenario)
        
        # Step 2: Action - apply intervention in modified graph
        intervened_graph = self._apply_intervention(
            self.causal_graph.graph.copy(),
            query.intervention
        )
        
        # Step 3: Prediction - compute target value in intervened graph
        counterfactual_val = self._forward_propagate(
            intervened_graph,
            query.intervention,
            query.target,
            latent_u
        )
        
        return counterfactual_val
    
    def explain_attack(
        self,
        attack_chain: List[str],
        graph_data: Optional[nx.DiGraph] = None
    ) -> CausalExplanation:
        """
        Generate comprehensive causal explanation of attack.
        
        Args:
            attack_chain: Sequence of nodes in attack path
            graph_data: Optional full provenance graph for context
            
        Returns:
            Complete causal explanation
        """
        # Identify MITRE techniques
        techniques = self._identify_techniques(attack_chain)
        
        # Compute causal effects
        causal_effects = self._compute_causal_effects(attack_chain)
        
        # Generate narrative
        narrative = self._generate_narrative(
            attack_chain,
            techniques,
            causal_effects
        )
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(attack_chain)
        
        # Identify critical intervention points
        critical_nodes = self._identify_critical_nodes(
            attack_chain,
            causal_effects
        )
        
        # Compute confidence
        confidence = self._compute_explanation_confidence(
            attack_chain,
            techniques
        )
        
        return CausalExplanation(
            attack_chain=attack_chain,
            attack_techniques=techniques,
            causal_effects=causal_effects,
            narrative=narrative,
            counterfactuals=counterfactuals,
            critical_nodes=critical_nodes,
            confidence=confidence
        )
    
    def _conditional_expectation(
        self,
        target: str,
        conditions: Dict[str, float]
    ) -> float:
        """Compute E[target | conditions]"""
        # Simplified: use edge weights as effect strengths
        # In practice, would use learned conditional distributions
        
        total = 0.0
        count = 0
        
        for var, val in conditions.items():
            if self.causal_graph.graph.has_edge(var, target):
                weight = self.causal_graph.graph[var][target].get('weight', 1.0)
                total += val * weight
                count += 1
        
        return total / count if count > 0 else 0.0
    
    def _sample_parent_configs(
        self,
        parents: List[str],
        n_samples: int = 100
    ) -> List[Dict[str, float]]:
        """Sample parent variable configurations"""
        # Simplified: sample uniform [0, 1]
        # In practice, use learned distributions
        
        import random
        
        configs = []
        for _ in range(n_samples):
            config = {
                parent: random.random()
                for parent in parents
            }
            configs.append(config)
        
        return configs
    
    def _infer_latent(
        self,
        observations: Dict[str, float]
    ) -> Dict[str, float]:
        """Infer latent variables from observations"""
        # Simplified: use observations as proxy
        # In practice, solve structural equations backwards
        return observations.copy()
    
    def _apply_intervention(
        self,
        graph: nx.DiGraph,
        interventions: Dict[str, float]
    ) -> nx.DiGraph:
        """Apply do-operator: remove incoming edges to intervened variables"""
        for var in interventions:
            if var in graph:
                # Remove all incoming edges (break causal influence)
                in_edges = list(graph.in_edges(var))
                graph.remove_edges_from(in_edges)
        
        return graph
    
    def _forward_propagate(
        self,
        graph: nx.DiGraph,
        interventions: Dict[str, float],
        target: str,
        latent: Dict[str, float]
    ) -> float:
        """Forward propagate through intervened graph"""
        # Topological sort for correct evaluation order
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # If cyclic, use arbitrary order
            topo_order = list(graph.nodes())
        
        # Initialize values
        values = interventions.copy()
        
        # Propagate through graph
        for node in topo_order:
            if node in values:
                continue  # Already set by intervention
            
            # Compute based on parents
            parents = list(graph.predecessors(node))
            if parents:
                node_val = sum(
                    values.get(p, latent.get(p, 0.5)) *
                    graph[p][node].get('weight', 1.0)
                    for p in parents
                )
                values[node] = node_val / len(parents)
            else:
                values[node] = latent.get(node, 0.5)
        
        return values.get(target, 0.0)
    
    def _identify_techniques(
        self,
        attack_chain: List[str]
    ) -> List[ATTACKTechnique]:
        """Map attack chain to MITRE ATT&CK techniques"""
        techniques = []
        
        # Extract edges from chain
        for i in range(len(attack_chain) - 1):
            src = attack_chain[i]
            dst = attack_chain[i + 1]
            
            # Check if edge maps to known technique
            for tech in self.attack_knowledge.techniques:
                # Simple heuristic: check node types and edge direction
                if self._matches_technique_pattern(src, dst, tech):
                    techniques.append(tech)
                    break
        
        return techniques
    
    def _matches_technique_pattern(
        self,
        src: str,
        dst: str,
        technique: ATTACKTechnique
    ) -> bool:
        """Check if edge matches technique pattern"""
        # Simplified pattern matching
        # In practice, use learned patterns or rules
        
        # Example: T1059 (Command Execution)
        if technique.tid == "T1059":
            return "shell" in src.lower() or "cmd" in src.lower()
        
        # Example: T1003 (Credential Dumping)
        if technique.tid == "T1003":
            return "lsass" in dst.lower() or "credential" in dst.lower()
        
        # Default: no match
        return False
    
    def _compute_causal_effects(
        self,
        attack_chain: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise causal effect strengths"""
        effects = {}
        
        for i in range(len(attack_chain) - 1):
            src = attack_chain[i]
            dst = attack_chain[i + 1]
            
            # Use edge weight from causal graph
            if self.causal_graph.graph.has_edge(src, dst):
                weight = self.causal_graph.graph[src][dst].get('weight', 1.0)
            else:
                weight = 0.5  # Default
            
            effects[(src, dst)] = weight
        
        return effects
    
    def _generate_narrative(
        self,
        attack_chain: List[str],
        techniques: List[ATTACKTechnique],
        causal_effects: Dict[Tuple[str, str], float]
    ) -> str:
        """Generate human-readable attack narrative"""
        narrative_gen = AttackNarrativeGenerator(self.attack_knowledge)
        return narrative_gen.generate(attack_chain, techniques, causal_effects)
    
    def _generate_counterfactuals(
        self,
        attack_chain: List[str]
    ) -> List[str]:
        """Generate "what if" counterfactual scenarios"""
        counterfactuals = []
        
        # For each node in chain, ask what if it were blocked
        for i, node in enumerate(attack_chain[1:-1], start=1):
            cf = (
                f"If {node} had been blocked, the attack would have been "
                f"prevented at step {i+1}/{len(attack_chain)}."
            )
            counterfactuals.append(cf)
        
        # Chain-level counterfactuals
        if len(attack_chain) >= 3:
            cf = (
                f"If {attack_chain[1]} had been isolated, "
                f"{attack_chain[-1]} would not have been compromised."
            )
            counterfactuals.append(cf)
        
        return counterfactuals
    
    def _identify_critical_nodes(
        self,
        attack_chain: List[str],
        causal_effects: Dict[Tuple[str, str], float]
    ) -> List[str]:
        """Identify high-impact intervention points"""
        # Compute node criticality scores
        criticality = {}
        
        for node in attack_chain:
            # Downstream impact: sum of outgoing effects
            downstream = sum(
                effect for (src, _), effect in causal_effects.items()
                if src == node
            )
            
            # Upstream dependency: sum of incoming effects
            upstream = sum(
                effect for (_, dst), effect in causal_effects.items()
                if dst == node
            )
            
            # Criticality = product (both high upstream and downstream)
            criticality[node] = downstream * upstream
        
        # Return top-3 critical nodes
        sorted_nodes = sorted(
            criticality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [node for node, _ in sorted_nodes[:3]]
    
    def _compute_explanation_confidence(
        self,
        attack_chain: List[str],
        techniques: List[ATTACKTechnique]
    ) -> float:
        """Compute overall confidence in explanation"""
        # Factor 1: Chain length (shorter = more confident)
        length_score = 1.0 / (1.0 + 0.1 * len(attack_chain))
        
        # Factor 2: Technique coverage (more techniques = more confident)
        coverage_score = len(techniques) / max(len(attack_chain) - 1, 1)
        
        # Factor 3: Edge weights (higher = more confident)
        avg_weight = 0.0
        count = 0
        for i in range(len(attack_chain) - 1):
            src = attack_chain[i]
            dst = attack_chain[i + 1]
            if self.causal_graph.graph.has_edge(src, dst):
                avg_weight += self.causal_graph.graph[src][dst].get('weight', 0.5)
                count += 1
        
        weight_score = avg_weight / count if count > 0 else 0.5
        
        # Combine
        confidence = (length_score + coverage_score + weight_score) / 3.0
        
        return min(1.0, max(0.0, confidence))


class AttackNarrativeGenerator:
    """
    Generates natural language narratives of attack chains.
    
    Uses Jinja2 templates and MITRE ATT&CK knowledge.
    """
    
    NARRATIVE_TEMPLATE = """
**Attack Summary**

The attacker initiated {{ tactics[0].tactic }} by {{ techniques[0].description }}.

{% for i in range(1, techniques|length) %}
Next, the attacker performed {{ tactics[i].tactic }} using {{ techniques[i].description }}.
{% endfor %}

**Attack Chain:**
{% for node in attack_chain %}
{{ loop.index }}. {{ node }}
{% endfor %}

**Causal Analysis:**
The attack progressed through {{ attack_chain|length }} stages, with the most critical
transition occurring at step {{ critical_step }}, where {{ critical_node }} enabled
{{ critical_effect }}.

**Impact:**
This attack chain resulted in {{ final_impact }}, affecting {{ num_assets }} critical assets.

**Mitigation Recommendations:**
{% for rec in recommendations %}
- {{ rec }}
{% endfor %}
"""
    
    def __init__(self, attack_knowledge: ATTACKKnowledge):
        """Initialize narrative generator"""
        self.attack_knowledge = attack_knowledge
        self.template = Template(self.NARRATIVE_TEMPLATE)
    
    def generate(
        self,
        attack_chain: List[str],
        techniques: List[ATTACKTechnique],
        causal_effects: Dict[Tuple[str, str], float]
    ) -> str:
        """
        Generate narrative from attack data.
        
        Args:
            attack_chain: Sequence of nodes
            techniques: MITRE techniques
            causal_effects: Edge effect strengths
            
        Returns:
            Natural language narrative
        """
        # Find critical step (highest causal effect)
        critical_step = 1
        critical_effect_val = 0.0
        critical_node = ""
        
        for i, ((src, dst), effect) in enumerate(causal_effects.items(), start=1):
            if effect > critical_effect_val:
                critical_effect_val = effect
                critical_step = i
                critical_node = src
        
        # Determine final impact
        final_node = attack_chain[-1]
        if "admin" in final_node.lower() or "root" in final_node.lower():
            final_impact = "privileged access compromise"
        elif "data" in final_node.lower():
            final_impact = "data exfiltration"
        else:
            final_impact = "system compromise"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(techniques)
        
        # Render template
        narrative = self.template.render(
            attack_chain=attack_chain,
            techniques=techniques,
            tactics=[{"tactic": t.tactic} for t in techniques],
            critical_step=critical_step,
            critical_node=critical_node,
            critical_effect=f"access to {attack_chain[critical_step]}",
            final_impact=final_impact,
            num_assets=len(set(attack_chain)),
            recommendations=recommendations
        )
        
        return narrative.strip()
    
    def _generate_recommendations(
        self,
        techniques: List[ATTACKTechnique]
    ) -> List[str]:
        """Generate mitigation recommendations based on techniques"""
        recommendations = []
        
        tactics_seen = set(t.tactic for t in techniques)
        
        if "Initial Access" in tactics_seen:
            recommendations.append(
                "Implement strict email filtering and user awareness training"
            )
        
        if "Execution" in tactics_seen:
            recommendations.append(
                "Enable application whitelisting and script execution policies"
            )
        
        if "Privilege Escalation" in tactics_seen:
            recommendations.append(
                "Apply principle of least privilege and monitor privilege changes"
            )
        
        if "Credential Access" in tactics_seen:
            recommendations.append(
                "Implement MFA and credential vault solutions"
            )
        
        if "Lateral Movement" in tactics_seen:
            recommendations.append(
                "Segment network and monitor lateral movement patterns"
            )
        
        if "Exfiltration" in tactics_seen:
            recommendations.append(
                "Deploy DLP solutions and monitor outbound traffic"
            )
        
        # Default
        if not recommendations:
            recommendations.append(
                "Review and update security controls based on MITRE ATT&CK framework"
            )
        
        return recommendations
