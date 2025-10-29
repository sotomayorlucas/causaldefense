# Contributing to CausalDefend

Thank you for your interest in contributing to CausalDefend! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional. We welcome contributions from everyone.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/causaldefend.git
   cd causaldefend
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy isort
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

We follow these style guidelines:

- **Python**: PEP 8 with Black formatter (100 character line length)
- **Type hints**: Use type annotations for all functions
- **Docstrings**: Google-style docstrings

Format your code before committing:

```bash
# Auto-format with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Testing

All new features must include tests. We aim for >80% code coverage.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=causaldefend --cov-report=html

# Run specific test file
pytest tests/unit/test_provenance_graph.py

# Run specific test
pytest tests/unit/test_provenance_graph.py::test_node_creation
```

### Writing Tests

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test multiple components together
- **Performance tests**: Benchmark critical functions
- **Security tests**: Test adversarial robustness

Example test structure:

```python
import pytest
from causaldefend.data import ProvenanceGraph

def test_node_creation():
    """Test creating a node in provenance graph"""
    graph = ProvenanceGraph()
    node = ProvenanceNode(
        id="test_node",
        node_type=NodeType.PROCESS,
        features=np.zeros(64),
        timestamp=datetime.now()
    )
    graph.add_node(node)
    
    assert len(graph.nodes) == 1
    assert graph.get_node("test_node") == node
```

## Documentation

- Add docstrings to all public functions/classes
- Update README.md if adding new features
- Add examples to notebooks/ for complex features
- Update docs/ for architectural changes

## Pull Request Process

1. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

2. Format and lint your code:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. Update CHANGELOG.md with your changes

4. Commit with descriptive messages:
   ```bash
   git commit -m "feat: Add neural CI test implementation"
   git commit -m "fix: Resolve memory leak in graph reduction"
   git commit -m "docs: Update API documentation"
   ```

   Commit message prefixes:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Maintenance tasks

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub:
   - Provide a clear description of changes
   - Reference any related issues
   - Include test results
   - Add screenshots/examples if applicable

## Review Process

- Maintainers will review your PR within 3-5 business days
- Address review comments promptly
- Ensure CI/CD checks pass
- Squash commits if requested

## Project Areas

We welcome contributions in these areas:

### High Priority
- [ ] Neural CI test implementation (Tier 2)
- [ ] PC-Stable causal discovery (Tier 3)
- [ ] Conformal prediction module
- [ ] API endpoints implementation
- [ ] Frontend UI components

### Medium Priority
- [ ] Additional log parsers (Windows ETW, DARPA TC)
- [ ] Adversarial robustness tests
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Example notebooks

### Research Extensions
- [ ] Novel causal discovery algorithms
- [ ] Alternative uncertainty quantification methods
- [ ] Advanced explanation techniques
- [ ] Multi-modal threat detection

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Email: team@causaldefend.ai

Thank you for contributing to CausalDefend! 🚀
