# CausalDefend Project Structure

```
causaldefend/
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── provenance_graph.py   # Graph data structures
│   │   └── provenance_parser.py  # Log parsers (auditd, ETW, DARPA TC)
│   │
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── spatiotemporal_detector.py  # GAT+GRU detector
│   │
│   ├── causal/                   # Causal discovery
│   │   ├── __init__.py
│   │   ├── graph_reduction.py    # Tier 1: Graph distillation
│   │   ├── neural_ci_test.py     # Tier 2: Neural CI tests
│   │   └── causal_discovery.py   # Tier 3: PC-Stable with ATT&CK
│   │
│   ├── uncertainty/              # Uncertainty quantification
│   │   ├── __init__.py
│   │   └── conformal_prediction.py  # Conformal prediction
│   │
│   ├── explanations/             # Explanation generation
│   │   ├── __init__.py
│   │   └── causal_explainer.py   # Causal explanations & narratives
│   │
│   ├── compliance/               # EU AI Act compliance
│   │   ├── __init__.py
│   │   └── eu_ai_act.py          # Compliance checks & audit logs
│   │
│   ├── api/                      # REST API
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application
│   │   ├── routes/               # API endpoints
│   │   ├── models.py             # Pydantic models
│   │   └── celery_app.py         # Celery tasks
│   │
│   ├── pipeline/                 # End-to-end pipeline
│   │   ├── __init__.py
│   │   └── detection_pipeline.py # Main detection pipeline
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── metrics.py            # Evaluation metrics
│       └── visualization.py      # Visualization utilities
│
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   │   ├── test_provenance_graph.py
│   │   ├── test_graph_reduction.py
│   │   ├── test_causal_discovery.py
│   │   └── test_conformal_prediction.py
│   │
│   ├── integration/              # Integration tests
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   │
│   ├── performance/              # Performance tests
│   │   └── test_scalability.py
│   │
│   └── security/                 # Security tests
│       └── test_adversarial.py
│
├── config/                       # Configuration files
│   ├── train_config.yaml         # Training configuration
│   ├── api_config.yaml           # API configuration
│   └── causal_config.yaml        # Causal discovery configuration
│
├── docker/                       # Docker files
│   ├── Dockerfile.api            # API server
│   ├── Dockerfile.worker         # Celery worker
│   ├── nginx.conf                # Nginx configuration
│   └── prometheus.yml            # Prometheus configuration
│
├── scripts/                      # Utility scripts
│   ├── train_detector.py         # Train GAT+GRU
│   ├── pretrain_ci_test.py       # Pretrain CI test
│   ├── run_pipeline.py           # Run detection pipeline
│   └── serve_api.py              # Start API server
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_detector.ipynb
│   ├── 03_causal_discovery_demo.ipynb
│   ├── 04_uncertainty_quantification.ipynb
│   └── 05_end_to_end_demo.ipynb
│
├── experiments/                  # Experiment scripts
│   ├── run_evaluation.py         # Run benchmarks
│   └── results/                  # Results storage
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── training.md
│   ├── deployment.md
│   └── compliance.md
│
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata
├── docker-compose.yml            # Docker Compose configuration
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── setup.py                      # Setup script
└── README.md                     # Project README
```

## Key Modules

### 1. Data Processing (`src/data/`)
- **ProvenanceGraph**: Heterogeneous temporal graph structure
- **ProvenanceParser**: Parses system logs (auditd, ETW, DARPA TC, JSON)

### 2. Detection Model (`src/models/`)
- **MultiHeadGAT**: Graph Attention Network with multi-head attention
- **TemporalGRU**: Temporal dynamics modeling
- **GraphAutoencoder**: Autoencoder for anomaly detection
- **APTDetector**: Complete detection system

### 3. Causal Discovery (`src/causal/`)
- **GraphDistiller** (Tier 1): Graph reduction via distillation
- **NeuralCITest** (Tier 2): Amortized conditional independence testing
- **TemporalPCStable** (Tier 3): Causal discovery with temporal constraints

### 4. Uncertainty Quantification (`src/uncertainty/`)
- **SplitConformalPredictor**: Conformal prediction with coverage guarantees
- **AdaptiveConformalPredictor**: Adaptive calibration for concept drift

### 5. Explanations (`src/explanations/`)
- **CausalExplainer**: Interventional and counterfactual reasoning
- **AttackNarrativeGenerator**: Natural language attack narratives

### 6. Compliance (`src/compliance/`)
- **ComplianceManager**: EU AI Act compliance verification
- **AuditLogger**: Immutable audit trail

### 7. API (`src/api/`)
- **FastAPI**: REST API endpoints
- **Celery**: Asynchronous task processing
- **Authentication**: JWT-based auth

### 8. Pipeline (`src/pipeline/`)
- **CausalDefendPipeline**: End-to-end detection pipeline

## Development Workflow

1. **Setup**: Run `python setup.py`
2. **Train**: `causaldefend-train --config config/train_config.yaml`
3. **Test**: `pytest tests/`
4. **Serve**: `causaldefend-serve` or `docker-compose up`
5. **Monitor**: Access Grafana at `http://localhost:3000`

## Production Deployment

1. Build Docker images: `docker-compose build`
2. Start services: `docker-compose up -d`
3. Access API: `http://localhost:8000`
4. Access monitoring: `http://localhost:3000` (Grafana)

## Next Steps

After basic setup, you should:

1. **Implement remaining modules**:
   - `neural_ci_test.py`
   - `causal_discovery.py`
   - `conformal_prediction.py`
   - `causal_explainer.py`
   - `eu_ai_act.py`
   - `detection_pipeline.py`
   - API routes

2. **Add comprehensive tests**

3. **Create Jupyter notebooks for demos**

4. **Deploy and validate on real data**

See individual module files for detailed implementation guidance.
