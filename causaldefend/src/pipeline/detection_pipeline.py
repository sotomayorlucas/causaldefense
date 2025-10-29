"""
End-to-End APT Detection Pipeline

Integrates all CausalDefend components into a unified
detection and explanation system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch

from ..data.provenance_parser import ProvenanceParser
from ..data.provenance_graph import ProvenanceGraph
from ..models.spatiotemporal_detector import APTDetector
from ..causal.graph_reduction import GraphDistiller, CriticalAssetManager
from ..causal.neural_ci_test import NeuralCITest, BatchCITester
from ..causal.causal_discovery import TemporalPCStable, ATTACKKnowledge, CausalGraph
from ..uncertainty.conformal_prediction import UncertaintyQuantifier, AdaptiveConformalPredictor
from ..explanations.causal_explainer import CausalExplainer, CausalExplanation
from ..compliance.eu_ai_act import ComplianceManager, AuditLogger, ModelCard


logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for detection pipeline"""
    
    def __init__(
        self,
        # Model paths
        detector_checkpoint: Path,
        ci_tester_checkpoint: Path,
        
        # Critical assets
        critical_assets: Optional[List[str]] = None,
        
        # Graph reduction
        reduction_target_size: int = 50000,
        reduction_phases: int = 3,
        
        # Causal discovery
        ci_significance: float = 0.05,
        max_conditioning_set: int = 5,
        
        # Uncertainty quantification
        confidence_level: float = 0.95,
        use_adaptive_conformal: bool = True,
        conformal_window_size: int = 1000,
        
        # Escalation
        escalation_threshold: float = 0.8,
        
        # Compliance
        audit_log_dir: Path = Path("./audit_logs"),
        enable_blockchain: bool = False,
        
        # Performance
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.detector_checkpoint = detector_checkpoint
        self.ci_tester_checkpoint = ci_tester_checkpoint
        self.critical_assets = critical_assets or []
        self.reduction_target_size = reduction_target_size
        self.reduction_phases = reduction_phases
        self.ci_significance = ci_significance
        self.max_conditioning_set = max_conditioning_set
        self.confidence_level = confidence_level
        self.use_adaptive_conformal = use_adaptive_conformal
        self.conformal_window_size = conformal_window_size
        self.escalation_threshold = escalation_threshold
        self.audit_log_dir = audit_log_dir
        self.enable_blockchain = enable_blockchain
        self.batch_size = batch_size
        self.device = device


class DetectionResult:
    """Complete detection result"""
    
    def __init__(
        self,
        # Detection
        anomaly_detected: bool,
        anomaly_score: float,
        confidence: float,
        prediction_set: List[int],
        should_escalate: bool,
        
        # Causal analysis
        causal_graph: Optional[CausalGraph],
        attack_chains: List[List[str]],
        
        # Explanations
        explanations: List[CausalExplanation],
        
        # Metadata
        processing_time_ms: float,
        graph_stats: Dict[str, int],
        audit_log_id: str
    ):
        self.anomaly_detected = anomaly_detected
        self.anomaly_score = anomaly_score
        self.confidence = confidence
        self.prediction_set = prediction_set
        self.should_escalate = should_escalate
        self.causal_graph = causal_graph
        self.attack_chains = attack_chains
        self.explanations = explanations
        self.processing_time_ms = processing_time_ms
        self.graph_stats = graph_stats
        self.audit_log_id = audit_log_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'anomaly_detected': self.anomaly_detected,
            'anomaly_score': self.anomaly_score,
            'confidence': self.confidence,
            'prediction_set': self.prediction_set,
            'should_escalate': self.should_escalate,
            'attack_chains': self.attack_chains,
            'explanations': [
                {
                    'chain': exp.attack_chain,
                    'narrative': exp.narrative,
                    'techniques': [
                        {'tid': t.tid, 'name': t.name, 'tactic': t.tactic}
                        for t in exp.attack_techniques
                    ],
                    'critical_nodes': exp.critical_nodes,
                    'confidence': exp.confidence
                }
                for exp in self.explanations
            ],
            'processing_time_ms': self.processing_time_ms,
            'graph_stats': self.graph_stats,
            'audit_log_id': self.audit_log_id
        }


class CausalDefendPipeline:
    """
    Main detection pipeline integrating all components.
    
    Pipeline stages:
    1. Parse logs → Provenance graph
    2. Detect anomalies → APT score
    3. Reduce graph → Distilled graph
    4. Discover causality → Causal DAG
    5. Explain attack → Narratives
    6. Quantify uncertainty → Confidence intervals
    7. Log for compliance → Audit trail
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        model_card: Optional[ModelCard] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            model_card: Model card for compliance
        """
        self.config = config
        
        logger.info("Initializing CausalDefend pipeline...")
        
        # Load components
        self._load_components()
        
        # Initialize compliance
        self.audit_logger = AuditLogger(
            config.audit_log_dir,
            enable_blockchain=config.enable_blockchain
        )
        
        if model_card:
            self.compliance_manager = ComplianceManager(
                model_card,
                self.audit_logger,
                config.escalation_threshold
            )
        else:
            self.compliance_manager = None
        
        logger.info("Pipeline initialized successfully")
    
    def _load_components(self) -> None:
        """Load all ML components"""
        logger.info("Loading detector model...")
        self.detector = APTDetector.load_from_checkpoint(
            str(self.config.detector_checkpoint)
        )
        self.detector.eval()
        self.detector.to(self.config.device)
        
        logger.info("Loading CI tester...")
        self.ci_tester = BatchCITester(device=self.config.device)
        # Load pretrained CI tester if checkpoint exists
        if self.config.ci_tester_checkpoint.exists():
            checkpoint = torch.load(
                self.config.ci_tester_checkpoint,
                map_location=self.config.device
            )
            self.ci_tester.neural_ci.load_state_dict(checkpoint)
        
        logger.info("Initializing graph reduction...")
        self.asset_manager = CriticalAssetManager(self.config.critical_assets)
        self.graph_distiller = GraphDistiller(
            self.config.critical_assets,
            target_size=self.config.reduction_target_size,
            num_phases=self.config.reduction_phases
        )
        
        logger.info("Initializing causal discovery...")
        self.attack_knowledge = ATTACKKnowledge()
        self.causal_discoverer = TemporalPCStable(
            self.ci_tester,
            self.attack_knowledge,
            alpha=self.config.ci_significance,
            max_cond_size=self.config.max_conditioning_set
        )
        
        logger.info("Initializing uncertainty quantification...")
        self.uncertainty_quantifier = UncertaintyQuantifier(
            self.detector.model,
            significance_level=1 - self.config.confidence_level,
            window_size=self.config.conformal_window_size,
            use_adaptive=self.config.use_adaptive_conformal
        )
        
        # Parser (no loading required)
        self.parser = ProvenanceParser()
    
    def process_alert(
        self,
        log_data: str,
        log_format: str = "json",
        user_id: str = "system",
        session_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Process security alert through full pipeline.
        
        Args:
            log_data: Raw log data
            log_format: Log format (auditd, etw, json, darpa_tc)
            user_id: User ID for audit logging
            session_id: Session ID for tracking
            
        Returns:
            Complete detection result
        """
        import time
        
        start_time = time.time()
        
        logger.info(f"Processing alert (format: {log_format})...")
        
        # Stage 1: Parse logs
        logger.info("Stage 1: Parsing logs...")
        provenance_graph = self.parser.parse_logs(log_data, log_format=log_format)
        
        initial_stats = {
            'num_nodes': provenance_graph.graph.number_of_nodes(),
            'num_edges': provenance_graph.graph.number_of_edges()
        }
        
        logger.info(f"Parsed graph: {initial_stats['num_nodes']} nodes, {initial_stats['num_edges']} edges")
        
        # Stage 2: Detect anomalies
        logger.info("Stage 2: Detecting anomalies...")
        pyg_graph = provenance_graph.to_pytorch_geometric()
        pyg_graph = pyg_graph.to(self.config.device)
        
        with torch.no_grad():
            is_anomaly, anomaly_score = self.detector.detect_anomaly(pyg_graph)
        
        logger.info(f"Anomaly score: {anomaly_score:.4f}")
        
        if not is_anomaly:
            # No anomaly detected - log and return
            audit_log_id = self._log_detection(
                user_id,
                session_id,
                provenance_graph,
                prediction=0,
                confidence=1 - anomaly_score,
                prediction_set=[0],
                human_override=False
            )
            
            elapsed = (time.time() - start_time) * 1000
            
            return DetectionResult(
                anomaly_detected=False,
                anomaly_score=anomaly_score,
                confidence=1 - anomaly_score,
                prediction_set=[0],
                should_escalate=False,
                causal_graph=None,
                attack_chains=[],
                explanations=[],
                processing_time_ms=elapsed,
                graph_stats=initial_stats,
                audit_log_id=audit_log_id
            )
        
        # Stage 3: Graph reduction
        logger.info("Stage 3: Reducing graph...")
        reduced_graph = self.graph_distiller.distill(provenance_graph.graph)
        
        reduced_stats = {
            'num_nodes': reduced_graph.number_of_nodes(),
            'num_edges': reduced_graph.number_of_edges(),
            'reduction_ratio': 1 - (reduced_graph.number_of_nodes() / initial_stats['num_nodes'])
        }
        
        logger.info(f"Reduced graph: {reduced_stats['num_nodes']} nodes ({reduced_stats['reduction_ratio']:.2%} reduction)")
        
        # Stage 4: Causal discovery
        logger.info("Stage 4: Discovering causal structure...")
        causal_graph = self.causal_discoverer.discover_causal_graph(
            reduced_graph,
            provenance_graph.node_features
        )
        
        logger.info(f"Causal DAG: {causal_graph.graph.number_of_edges()} causal edges")
        
        # Extract attack chains
        attack_chains = causal_graph.extract_attack_chains(top_k=5)
        logger.info(f"Extracted {len(attack_chains)} attack chains")
        
        # Stage 5: Generate explanations
        logger.info("Stage 5: Generating explanations...")
        explainer = CausalExplainer(causal_graph, self.attack_knowledge)
        
        explanations = []
        for chain in attack_chains:
            explanation = explainer.explain_attack(chain)
            explanations.append(explanation)
        
        # Stage 6: Uncertainty quantification
        logger.info("Stage 6: Quantifying uncertainty...")
        uncertainty_result = self.uncertainty_quantifier.classify_with_uncertainty(
            pyg_graph.x.mean(dim=0)  # Aggregate features
        )
        
        should_escalate = uncertainty_result['should_escalate']
        
        # Stage 7: Compliance logging
        logger.info("Stage 7: Logging for compliance...")
        audit_log_id = self._log_detection(
            user_id,
            session_id,
            provenance_graph,
            prediction=1,
            confidence=uncertainty_result['confidence'],
            prediction_set=uncertainty_result['prediction_set'],
            human_override=False
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(f"Pipeline completed in {elapsed:.2f}ms")
        
        return DetectionResult(
            anomaly_detected=True,
            anomaly_score=anomaly_score,
            confidence=uncertainty_result['confidence'],
            prediction_set=uncertainty_result['prediction_set'],
            should_escalate=should_escalate,
            causal_graph=causal_graph,
            attack_chains=attack_chains,
            explanations=explanations,
            processing_time_ms=elapsed,
            graph_stats={**initial_stats, **reduced_stats},
            audit_log_id=audit_log_id
        )
    
    def process_feedback(
        self,
        audit_log_id: str,
        true_label: int,
        feedback_text: str
    ) -> None:
        """
        Process analyst feedback for continuous learning.
        
        Args:
            audit_log_id: ID of audit log entry
            true_label: True label (0=benign, 1=attack)
            feedback_text: Analyst feedback
        """
        logger.info(f"Processing feedback for {audit_log_id}")
        
        # Update audit log
        self.audit_logger.update_with_feedback(
            audit_log_id,
            true_label,
            feedback_text
        )
        
        # Update adaptive conformal predictor
        if self.config.use_adaptive_conformal:
            # In practice: retrieve original input and update
            pass
    
    def _log_detection(
        self,
        user_id: str,
        session_id: Optional[str],
        graph: ProvenanceGraph,
        prediction: int,
        confidence: float,
        prediction_set: List[int],
        human_override: bool,
        human_decision: Optional[int] = None,
        human_justification: Optional[str] = None
    ) -> str:
        """Log detection to audit system"""
        import hashlib
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = hashlib.md5(
                f"{user_id}_{graph.graph.number_of_nodes()}".encode()
            ).hexdigest()
        
        # Log prediction
        audit_log_id = self.audit_logger.log_prediction(
            user_id=user_id,
            session_id=session_id,
            input_data=graph.to_dict(),
            prediction=prediction,
            confidence=confidence,
            prediction_set=prediction_set,
            human_override=human_override,
            human_decision=human_decision,
            human_justification=human_justification,
            metadata={
                'num_nodes': graph.graph.number_of_nodes(),
                'num_edges': graph.graph.number_of_edges()
            }
        )
        
        return audit_log_id
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get EU AI Act compliance status"""
        if self.compliance_manager:
            return self.compliance_manager.check_compliance()
        else:
            return {'error': 'Compliance manager not initialized'}
    
    def export_compliance_report(self, output_path: Path) -> None:
        """Export compliance report"""
        if self.compliance_manager:
            self.compliance_manager.generate_compliance_report(output_path)
        else:
            logger.warning("Compliance manager not initialized")


def create_default_pipeline(
    detector_checkpoint: Path,
    ci_tester_checkpoint: Path,
    critical_assets: Optional[List[str]] = None
) -> CausalDefendPipeline:
    """
    Create pipeline with default configuration.
    
    Args:
        detector_checkpoint: Path to detector model checkpoint
        ci_tester_checkpoint: Path to CI tester checkpoint
        critical_assets: List of critical asset identifiers
        
    Returns:
        Configured pipeline
    """
    config = PipelineConfig(
        detector_checkpoint=detector_checkpoint,
        ci_tester_checkpoint=ci_tester_checkpoint,
        critical_assets=critical_assets
    )
    
    # Create default model card
    from ..compliance.eu_ai_act import create_default_model_card
    model_card = create_default_model_card("CausalDefend", "1.0.0")
    
    pipeline = CausalDefendPipeline(config, model_card)
    
    return pipeline
