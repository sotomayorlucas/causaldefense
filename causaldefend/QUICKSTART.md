"""
Quick start guide for CausalDefend
"""

# Installation
"""
1. Clone repository:
   git clone https://github.com/causaldefend/causaldefend.git
   cd causaldefend

2. Create virtual environment:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate

3. Install dependencies:
   pip install -r requirements.txt
   pip install -e .

4. Download models:
   python -m spacy download en_core_web_sm
"""

# Basic Usage
"""
from causaldefend import ProvenanceGraph, ProvenanceParser, APTDetector

# 1. Parse logs
parser = ProvenanceParser()
graph = parser.parse_logs("system_audit.log", format="auditd")

# 2. Load detector
detector = APTDetector.load_from_checkpoint("models/causaldefend-v1.ckpt")

# 3. Detect anomalies
is_anomalous, score = detector.detect_anomaly(graph)

if is_anomalous:
    print(f"⚠️ APT detected! Anomaly score: {score:.3f}")
"""

# API Usage
"""
import requests

# Start server: causaldefend-serve

# Detect APT
response = requests.post(
    "http://localhost:8000/api/v1/detect",
    files={"graph": open("provenance_graph.json", "rb")}
)

result = response.json()
print(f"Malicious: {result['is_malicious']}")
print(f"Confidence: {result['confidence']}")

# Get explanation
response = requests.post(
    "http://localhost:8000/api/v1/explain",
    json={"graph_id": result['graph_id']}
)

explanation = response.json()
print(f"Attack narrative: {explanation['narrative']}")
print(f"MITRE techniques: {explanation['mitre_techniques']}")
"""

# Docker Usage
"""
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Access API: http://localhost:8000
# Access Grafana: http://localhost:3000
"""

print(__doc__)
