"""
Complete Training Pipeline for CausalDefend

Runs the full training pipeline:
1. Generate synthetic dataset
2. Train APT detector
3. Train CI tester
4. Validate models

Usage:
    python scripts/train_all.py --quick      # Fast training (for testing)
    python scripts/train_all.py --full       # Full training (production)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("="*80)
    logger.info(f"STEP: {description}")
    logger.info("="*80)
    logger.info(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False
        )
        logger.info(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        logger.error(f"Error: {e}\n")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {e}\n")
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Complete CausalDefend Training Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Training mode: quick (testing) or full (production)"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation if already exists"
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Skip detector training"
    )
    parser.add_argument(
        "--skip-ci-tester",
        action="store_true",
        help="Skip CI tester training"
    )
    
    args = parser.parse_args()
    
    # Configuration based on mode
    if args.mode == "quick":
        config = {
            "num_benign": 100,
            "num_attack": 100,
            "detector_epochs": 5,
            "ci_tester_epochs": 3,
            "batch_size": 16,
            "num_ci_samples": 1000,
        }
        logger.info("🚀 QUICK MODE - Fast training for testing")
    else:
        config = {
            "num_benign": 500,
            "num_attack": 500,
            "detector_epochs": 100,
            "ci_tester_epochs": 50,
            "batch_size": 32,
            "num_ci_samples": 10000,
        }
        logger.info("🚀 FULL MODE - Production training")
    
    logger.info("\n" + "="*80)
    logger.info("CausalDefend Complete Training Pipeline")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  - {key}: {value}")
    logger.info("="*80 + "\n")
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Generate dataset
    if not args.skip_dataset:
        cmd = [
            sys.executable,
            "scripts/prepare_dataset_simple.py",
            "--num-benign", str(config["num_benign"]),
            "--num-attack", str(config["num_attack"]),
        ]
        
        if run_command(cmd, "Dataset Generation"):
            success_steps.append("Dataset Generation")
        else:
            failed_steps.append("Dataset Generation")
            logger.error("❌ Cannot continue without dataset. Exiting.")
            return 1
    else:
        logger.info("⏭️  Skipping dataset generation (using existing data)\n")
        success_steps.append("Dataset Generation (skipped)")
    
    # Step 2: Train APT Detector
    if not args.skip_detector:
        cmd = [
            sys.executable,
            "scripts/train_detector.py",
            "--epochs", str(config["detector_epochs"]),
            "--batch-size", str(config["batch_size"]),
        ]
        
        if run_command(cmd, "APT Detector Training"):
            success_steps.append("APT Detector Training")
        else:
            failed_steps.append("APT Detector Training")
    else:
        logger.info("⏭️  Skipping detector training\n")
        success_steps.append("APT Detector Training (skipped)")
    
    # Step 3: Train CI Tester
    if not args.skip_ci_tester:
        cmd = [
            sys.executable,
            "scripts/train_ci_tester.py",
            "--epochs", str(config["ci_tester_epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--num-samples", str(config["num_ci_samples"]),
        ]
        
        if run_command(cmd, "Neural CI Tester Training"):
            success_steps.append("Neural CI Tester Training")
        else:
            failed_steps.append("Neural CI Tester Training")
    else:
        logger.info("⏭️  Skipping CI tester training\n")
        success_steps.append("Neural CI Tester Training (skipped)")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info("="*80)
    
    if success_steps:
        logger.info(f"\n✅ Successful steps ({len(success_steps)}):")
        for step in success_steps:
            logger.info(f"   ✓ {step}")
    
    if failed_steps:
        logger.error(f"\n❌ Failed steps ({len(failed_steps)}):")
        for step in failed_steps:
            logger.error(f"   ✗ {step}")
    
    logger.info("\n" + "="*80)
    
    if not failed_steps:
        logger.info("🎉 ALL TRAINING COMPLETE!")
        logger.info("\n📁 Model checkpoints saved to:")
        logger.info("   - models/detector.ckpt")
        logger.info("   - models/ci_tester.ckpt")
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Test detection: python examples/complete_detection.py")
        logger.info("   2. Run demo: python examples/demo_basico.py")
        logger.info("   3. Start API: uvicorn causaldefend.api.main:app")
        return 0
    else:
        logger.error(f"⚠️  Training completed with {len(failed_steps)} errors")
        logger.error("Please check the logs above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
