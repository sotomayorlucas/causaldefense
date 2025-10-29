"""
CausalDefend Installation Verification Script

Checks that all components are properly installed and configured.
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple


def print_header(text: str) -> None:
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_python_version() -> bool:
    """Check Python version >= 3.10"""
    version = sys.version_info
    required = (3, 10)
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if (version.major, version.minor) >= required:
        print("✓ Python version OK")
        return True
    else:
        print(f"✗ Python {required[0]}.{required[1]}+ required")
        return False


def check_imports() -> List[Tuple[str, bool]]:
    """Check critical package imports"""
    packages = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("networkx", "NetworkX"),
        ("fastapi", "FastAPI"),
        ("celery", "Celery"),
        ("redis", "Redis"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("jinja2", "Jinja2"),
        ("pydantic", "Pydantic"),
    ]
    
    results = []
    
    for package, name in packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:25s} (v{version})")
            results.append((name, True))
        except ImportError:
            print(f"✗ {name:25s} NOT FOUND")
            results.append((name, False))
    
    return results


def check_causaldefend_modules() -> List[Tuple[str, bool]]:
    """Check CausalDefend module imports"""
    modules = [
        ("causaldefend.data.provenance_graph", "Provenance Graph"),
        ("causaldefend.data.provenance_parser", "Provenance Parser"),
        ("causaldefend.models.spatiotemporal_detector", "Detector Model"),
        ("causaldefend.causal.graph_reduction", "Graph Reduction"),
        ("causaldefend.causal.neural_ci_test", "Neural CI Test"),
        ("causaldefend.causal.causal_discovery", "Causal Discovery"),
        ("causaldefend.uncertainty.conformal_prediction", "Conformal Prediction"),
        ("causaldefend.explanations.causal_explainer", "Causal Explainer"),
        ("causaldefend.compliance.eu_ai_act", "EU AI Act Compliance"),
        ("causaldefend.api.main", "FastAPI Server"),
        ("causaldefend.pipeline.detection_pipeline", "Detection Pipeline"),
    ]
    
    results = []
    
    for module, name in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {name}")
            results.append((name, True))
        except ImportError as e:
            print(f"✗ {name} - {str(e)}")
            results.append((name, False))
    
    return results


def check_cuda() -> bool:
    """Check CUDA availability"""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"✓ CUDA available (v{cuda_version})")
            print(f"  Devices: {device_count}")
            print(f"  Device 0: {device_name}")
            return True
        else:
            print("⚠ CUDA not available (CPU-only mode)")
            return False
            
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def check_files() -> List[Tuple[str, bool]]:
    """Check critical files exist"""
    base_path = Path(__file__).parent
    
    files = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "config/train_config.yaml",
        "config/api_config.yaml",
        "docker-compose.yml",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/causal/__init__.py",
        "src/uncertainty/__init__.py",
        "src/explanations/__init__.py",
        "src/compliance/__init__.py",
        "src/api/__init__.py",
        "src/pipeline/__init__.py",
    ]
    
    results = []
    
    for file_path in files:
        full_path = base_path / file_path
        exists = full_path.exists()
        
        if exists:
            print(f"✓ {file_path}")
            results.append((file_path, True))
        else:
            print(f"✗ {file_path} NOT FOUND")
            results.append((file_path, False))
    
    return results


def check_config_files() -> bool:
    """Check configuration files are valid"""
    try:
        import yaml
        
        base_path = Path(__file__).parent
        
        # Check train_config.yaml
        train_config_path = base_path / "config" / "train_config.yaml"
        if train_config_path.exists():
            with open(train_config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ train_config.yaml loaded")
        else:
            print(f"✗ train_config.yaml not found")
            return False
        
        # Check api_config.yaml
        api_config_path = base_path / "config" / "api_config.yaml"
        if api_config_path.exists():
            with open(api_config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ api_config.yaml loaded")
        else:
            print(f"✗ api_config.yaml not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Config file check failed: {e}")
        return False


def main():
    """Run all verification checks"""
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "CAUSALDEFEND VERIFICATION" + " "*20 + "#")
    print("#"*70)
    
    # Python version
    print_header("1. Python Version")
    python_ok = check_python_version()
    
    # Package imports
    print_header("2. Required Packages")
    package_results = check_imports()
    packages_ok = all(result[1] for result in package_results)
    
    # CausalDefend modules
    print_header("3. CausalDefend Modules")
    module_results = check_causaldefend_modules()
    modules_ok = all(result[1] for result in module_results)
    
    # CUDA
    print_header("4. CUDA Support")
    cuda_ok = check_cuda()
    
    # Files
    print_header("5. Project Files")
    file_results = check_files()
    files_ok = all(result[1] for result in file_results)
    
    # Config files
    print_header("6. Configuration Files")
    config_ok = check_config_files()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    total_checks = 6
    passed_checks = sum([
        python_ok,
        packages_ok,
        modules_ok,
        cuda_ok,  # Optional, don't count as failure
        files_ok,
        config_ok
    ])
    
    print(f"Python Version:       {'✓ PASS' if python_ok else '✗ FAIL'}")
    print(f"Required Packages:    {'✓ PASS' if packages_ok else '✗ FAIL'}")
    print(f"CausalDefend Modules: {'✓ PASS' if modules_ok else '✗ FAIL'}")
    print(f"CUDA Support:         {'✓ AVAILABLE' if cuda_ok else '⚠ CPU-ONLY'}")
    print(f"Project Files:        {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Configuration Files:  {'✓ PASS' if config_ok else '✗ FAIL'}")
    
    print("\n" + "-"*70)
    
    # Don't count CUDA as critical
    critical_passed = sum([python_ok, packages_ok, modules_ok, files_ok, config_ok])
    critical_total = 5
    
    if critical_passed == critical_total:
        print("\n🎉 ALL CHECKS PASSED! CausalDefend is ready to use.")
        if not cuda_ok:
            print("⚠️  Running in CPU-only mode (CUDA not available)")
        return 0
    else:
        print(f"\n❌ VERIFICATION FAILED: {critical_passed}/{critical_total} critical checks passed")
        print("\nPlease fix the issues above and run verification again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
