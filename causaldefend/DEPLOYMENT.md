# CausalDefend Deployment Guide

This guide covers production deployment of CausalDefend for Security Operations Centers (SOCs).

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Model Training](#model-training)
4. [API Deployment](#api-deployment)
5. [Monitoring](#monitoring)
6. [EU AI Act Compliance](#eu-ai-act-compliance)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

**Minimum (Development)**
- CPU: 8 cores (Intel Xeon or AMD EPYC)
- RAM: 32 GB
- Storage: 500 GB SSD
- GPU: NVIDIA Tesla T4 (16 GB VRAM)

**Recommended (Production)**
- CPU: 32 cores (Intel Xeon Platinum or AMD EPYC)
- RAM: 128 GB
- Storage: 2 TB NVMe SSD
- GPU: NVIDIA A100 (40-80 GB VRAM)

### Software

- OS: Ubuntu 22.04 LTS or RHEL 8+
- Python: 3.10 or 3.11
- CUDA: 11.8 or 12.1
- Docker: 24.0+
- Kubernetes: 1.27+ (optional, for orchestration)

---

## Infrastructure Setup

### 1. Database Setup (PostgreSQL)

```bash
# Install PostgreSQL
sudo apt-get install postgresql-14 postgresql-client-14

# Create database and user
sudo -u postgres psql
CREATE DATABASE causaldefend;
CREATE USER causaldefend_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE causaldefend TO causaldefend_user;
\q

# Configure connection
export DATABASE_URL="postgresql://causaldefend_user:secure_password@localhost:5432/causaldefend"
```

### 2. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf

# Set:
# maxmemory 4gb
# maxmemory-policy allkeys-lru
# appendonly yes

# Restart Redis
sudo systemctl restart redis
```

### 3. Message Queue (Celery)

```bash
# Celery workers are configured via docker-compose
# See docker-compose.yml for configuration

# Start Celery worker manually:
celery -A src.api.main.celery_app worker -l info --concurrency=4
```

---

## Model Training

### 1. Prepare Training Data

```bash
# Download DARPA TC dataset
cd data/raw
wget https://example.com/darpa-tc-dataset.tar.gz
tar -xzf darpa-tc-dataset.tar.gz

# Preprocess logs
python scripts/preprocess_data.py \
    --input data/raw/darpa-tc \
    --output data/processed \
    --format darpa_tc
```

### 2. Train Detector Model

```bash
# Train GAT+GRU detector
python scripts/train_detector.py \
    --data data/processed/train \
    --config config/train_config.yaml \
    --output models/detector.ckpt \
    --epochs 100 \
    --batch-size 32 \
    --gpus 1

# Expected training time: ~8 hours on A100
```

### 3. Train CI Tester

```bash
# Pretrain neural conditional independence tester
python scripts/train_ci_tester.py \
    --data data/processed/train \
    --output models/ci_tester.ckpt \
    --epochs 50 \
    --batch-size 128

# Expected training time: ~4 hours on A100
```

### 4. Calibrate Conformal Predictor

```bash
# Calibrate on held-out calibration set
python scripts/calibrate_conformal.py \
    --detector models/detector.ckpt \
    --calibration-data data/processed/calibration \
    --output models/conformal_calibration.pkl \
    --confidence-level 0.95
```

---

## API Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Services:
# - api: FastAPI server (port 8000)
# - worker: Celery workers
# - postgres: Database
# - redis: Cache/message broker
# - prometheus: Metrics
# - grafana: Dashboards (port 3000)
# - nginx: Reverse proxy (port 80)

# View logs
docker-compose logs -f api

# Scale workers
docker-compose up -d --scale worker=4
```

### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n causaldefend
kubectl logs -f deployment/causaldefend-api -n causaldefend
```

### Option 3: Systemd Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/causaldefend.service

# Content:
[Unit]
Description=CausalDefend API Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=causaldefend
WorkingDirectory=/opt/causaldefend
Environment="PATH=/opt/causaldefend/venv/bin"
ExecStart=/opt/causaldefend/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable causaldefend
sudo systemctl start causaldefend
sudo systemctl status causaldefend
```

---

## Monitoring

### 1. Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'causaldefend'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

**Key Metrics:**
- `causaldefend_requests_total`: Total API requests
- `causaldefend_detection_latency_seconds`: Detection latency histogram
- `causaldefend_anomaly_score`: Anomaly scores
- `causaldefend_escalation_rate`: Human escalation rate
- `causaldefend_conformal_coverage`: Conformal prediction coverage

### 2. Grafana Dashboards

```bash
# Import dashboard
# See dashboards/causaldefend.json

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

### 3. Log Aggregation

```bash
# Configure log shipping to ELK/Splunk
# Example: Filebeat configuration

filebeat.inputs:
  - type: log
    paths:
      - /var/log/causaldefend/*.log
    fields:
      service: causaldefend
      
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

---

## EU AI Act Compliance

### 1. Model Card Generation

```python
from causaldefend.compliance import create_default_model_card

model_card = create_default_model_card(
    model_name="CausalDefend-Production",
    model_version="1.0.0"
)

# Save model card
model_card.to_json("compliance/model_card.json")
```

### 2. Audit Log Configuration

```python
from causaldefend.compliance import AuditLogger

# Initialize audit logger with blockchain anchoring
audit_logger = AuditLogger(
    log_dir="audit_logs",
    enable_blockchain=True,
    blockchain_endpoint="https://api.chainpoint.org"
)
```

### 3. Compliance Verification

```bash
# Run compliance checks
python scripts/verify_compliance.py \
    --model-card compliance/model_card.json \
    --audit-logs audit_logs/ \
    --output compliance_report.json

# Expected output:
# ✓ Technical Documentation (Article 11)
# ✓ Record-keeping (Article 12)
# ✓ Transparency (Article 13)
# ✓ Human Oversight (Article 14)
# ✓ Accuracy & Robustness (Article 15)
```

### 4. Human-in-the-Loop Setup

Configure escalation thresholds in `config/api_config.yaml`:

```yaml
escalation:
  confidence_threshold: 0.8
  set_size_threshold: 2
  notify_channels:
    - email: soc-team@company.com
    - slack: "#security-alerts"
    - pagerduty: "service-key-here"
```

---

## Troubleshooting

### High Memory Usage

```bash
# Check memory usage
docker stats

# Reduce batch size in config/train_config.yaml
batch_size: 16  # Down from 32

# Enable gradient checkpointing
python scripts/train_detector.py --gradient-checkpointing
```

### Slow Inference

```bash
# Profile inference
python -m cProfile -o profile.stats scripts/run_inference.py

# View hotspots
python -m pstats profile.stats
> sort cumulative
> stats 20

# Common fixes:
# 1. Enable TorchScript compilation
# 2. Use ONNX runtime
# 3. Quantize model to INT8
```

### Database Connection Errors

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
psql -U causaldefend_user -d causaldefend -c "SELECT * FROM pg_stat_activity;"

# Increase connection pool
# In config/api_config.yaml:
database:
  pool_size: 20
  max_overflow: 40
```

### Celery Tasks Stuck

```bash
# Check Celery workers
celery -A src.api.main.celery_app inspect active

# Purge queue
celery -A src.api.main.celery_app purge

# Restart workers
docker-compose restart worker
```

---

## Performance Tuning

### GPU Optimization

```python
# Enable mixed precision training
from pytorch_lightning import Trainer

trainer = Trainer(
    precision='16-mixed',  # Use bfloat16 on A100
    devices=1,
    accelerator='gpu'
)
```

### Database Indexing

```sql
-- Create indexes for common queries
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_session ON audit_logs(session_id);
```

### Caching Strategy

```python
# Use Redis for caching frequently accessed data
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Cache model predictions (TTL: 1 hour)
redis_client.setex(
    f"prediction:{input_hash}",
    3600,
    prediction_json
)
```

---

## Security Hardening

### 1. API Authentication

```bash
# Generate secure JWT secret
python -c "import secrets; print(secrets.token_hex(32))"

# Set in environment
export SECRET_KEY="your-generated-secret-here"
```

### 2. TLS/SSL Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/causaldefend.crt;
    ssl_certificate_key /etc/ssl/private/causaldefend.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    location / {
        proxy_pass http://api:8000;
    }
}
```

### 3. Rate Limiting

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/api/v1/detect")
@app.rate_limit(times=10, seconds=60)  # 10 requests per minute
async def detect_apt(...):
    ...
```

---

## Backup and Recovery

### Database Backup

```bash
# Automated daily backups
0 2 * * * pg_dump -U causaldefend_user causaldefend | gzip > /backups/causaldefend_$(date +\%Y\%m\%d).sql.gz

# Restore from backup
gunzip -c /backups/causaldefend_20240115.sql.gz | psql -U causaldefend_user causaldefend
```

### Model Versioning

```bash
# Use DVC for model versioning
dvc add models/detector.ckpt
dvc push

# Rollback to previous version
git checkout HEAD~1 models/detector.ckpt.dvc
dvc pull
```

---

## Support

- **Documentation**: https://docs.causaldefend.ai
- **Issue Tracker**: https://github.com/causaldefend/causaldefend/issues
- **Email**: support@causaldefend.ai
- **Slack**: causaldefend.slack.com
