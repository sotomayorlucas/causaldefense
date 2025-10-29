"""
FastAPI REST API for CausalDefend

Provides endpoints for APT detection, causal explanation,
and system management.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
from celery import Celery
import redis

from ..data.provenance_parser import ProvenanceParser
from ..models.spatiotemporal_detector import APTDetector
from ..causal.graph_reduction import GraphDistiller
from ..causal.neural_ci_test import BatchCITester
from ..causal.causal_discovery import TemporalPCStable, ATTACKKnowledge
from ..uncertainty.conformal_prediction import UncertaintyQuantifier
from ..explanations.causal_explainer import CausalExplainer
from ..compliance.eu_ai_act import ComplianceManager, AuditLogger


# Configuration
API_VERSION = "v1"
SECRET_KEY = "your-secret-key-here"  # In production: load from env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Initialize FastAPI app
app = FastAPI(
    title="CausalDefend API",
    description="APT Detection using Causal Graph Neural Networks",
    version=API_VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Celery for async tasks
celery_app = Celery(
    "causaldefend",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)


# ==================== Pydantic Models ====================

class Token(BaseModel):
    """JWT access token"""
    access_token: str
    token_type: str = "bearer"


class UserLogin(BaseModel):
    """User login credentials"""
    username: str
    password: str


class DetectionRequest(BaseModel):
    """APT detection request"""
    log_format: str = Field(..., description="Log format: auditd, etw, json, darpa_tc")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    critical_assets: Optional[List[str]] = None
    confidence_level: float = Field(0.95, ge=0.0, le=1.0)


class DetectionResponse(BaseModel):
    """APT detection response"""
    task_id: str
    status: str
    anomaly_detected: bool
    anomaly_score: float
    causal_chains: List[List[str]]
    explanations: List[str]
    confidence_intervals: List[Dict[str, float]]
    should_escalate: bool
    mitre_techniques: List[str]


class ExplanationRequest(BaseModel):
    """Request for causal explanation"""
    attack_chain: List[str]
    include_counterfactuals: bool = True
    include_interventions: bool = True


class ExplanationResponse(BaseModel):
    """Causal explanation response"""
    narrative: str
    attack_techniques: List[Dict[str, str]]
    causal_effects: Dict[str, float]
    counterfactuals: List[str]
    critical_nodes: List[str]
    confidence: float


class InterventionRequest(BaseModel):
    """Interventional query request"""
    variable: str
    value: float
    target: str


class InterventionResponse(BaseModel):
    """Interventional query response"""
    intervention: str
    expected_effect: float
    confidence_interval: List[float]


class FeedbackRequest(BaseModel):
    """Feedback on prediction"""
    prediction_id: str
    true_label: int
    feedback_text: str
    analyst_id: str


class MetricsResponse(BaseModel):
    """System metrics"""
    total_predictions: int
    avg_latency_ms: float
    accuracy: float
    coverage: float
    escalation_rate: float
    uptime_hours: float


# ==================== Authentication ====================

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


# ==================== Endpoints ====================

@app.post("/api/v1/auth/login", response_model=Token)
async def login(user: UserLogin):
    """
    Authenticate user and return JWT token.
    
    In production: verify against database with hashed passwords.
    """
    # Placeholder authentication
    if user.username == "admin" and user.password == "admin":
        access_token = create_access_token(data={"sub": user.username})
        return Token(access_token=access_token)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_apt(
    request: DetectionRequest,
    log_file: UploadFile = File(...),
    username: str = Depends(verify_token)
):
    """
    Detect APT in uploaded log file.
    
    Returns:
        Detection results with causal chains and explanations
    """
    # Read log file
    log_content = await log_file.read()
    
    # Create async Celery task
    task = detect_apt_task.delay(
        log_content.decode('utf-8'),
        request.log_format,
        request.critical_assets,
        request.confidence_level,
        username
    )
    
    # Return task ID for status polling
    return DetectionResponse(
        task_id=task.id,
        status="processing",
        anomaly_detected=False,
        anomaly_score=0.0,
        causal_chains=[],
        explanations=[],
        confidence_intervals=[],
        should_escalate=False,
        mitre_techniques=[]
    )


@app.get("/api/v1/detect/{task_id}", response_model=DetectionResponse)
async def get_detection_result(
    task_id: str,
    username: str = Depends(verify_token)
):
    """
    Get detection result for async task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Detection results when ready
    """
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return DetectionResponse(
            task_id=task_id,
            status="processing",
            anomaly_detected=False,
            anomaly_score=0.0,
            causal_chains=[],
            explanations=[],
            confidence_intervals=[],
            should_escalate=False,
            mitre_techniques=[]
        )
    elif task.state == 'SUCCESS':
        result = task.result
        return DetectionResponse(**result)
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task failed: {task.info}"
        )


@app.post("/api/v1/explain", response_model=ExplanationResponse)
async def explain_attack(
    request: ExplanationRequest,
    username: str = Depends(verify_token)
):
    """
    Generate causal explanation for attack chain.
    
    Args:
        request: Explanation request with attack chain
        
    Returns:
        Causal explanation with narrative
    """
    # Load components (in production: load from cache/DB)
    attack_knowledge = ATTACKKnowledge()
    
    # Placeholder: create causal graph and explainer
    # In practice, load from previous detection
    from ..causal.causal_discovery import CausalGraph
    import networkx as nx
    
    causal_graph = CausalGraph(nx.DiGraph())
    explainer = CausalExplainer(causal_graph, attack_knowledge)
    
    # Generate explanation
    explanation = explainer.explain_attack(request.attack_chain)
    
    return ExplanationResponse(
        narrative=explanation.narrative,
        attack_techniques=[
            {
                'tid': t.tid,
                'name': t.name,
                'tactic': t.tactic,
                'description': t.description
            }
            for t in explanation.attack_techniques
        ],
        causal_effects={
            f"{src}->{dst}": effect
            for (src, dst), effect in explanation.causal_effects.items()
        },
        counterfactuals=explanation.counterfactuals,
        critical_nodes=explanation.critical_nodes,
        confidence=explanation.confidence
    )


@app.post("/api/v1/interventions", response_model=InterventionResponse)
async def compute_intervention(
    request: InterventionRequest,
    username: str = Depends(verify_token)
):
    """
    Compute effect of interventional query.
    
    Args:
        request: Intervention specification
        
    Returns:
        Expected effect of intervention
    """
    # Placeholder implementation
    # In practice, use CausalExplainer.interventional_effect()
    
    return InterventionResponse(
        intervention=f"do({request.variable} = {request.value})",
        expected_effect=0.75,  # Placeholder
        confidence_interval=[0.65, 0.85]
    )


@app.post("/api/v1/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    username: str = Depends(verify_token)
):
    """
    Submit feedback on prediction for model improvement.
    
    Args:
        request: Feedback with true label
        
    Returns:
        Success confirmation
    """
    # Log feedback to audit system
    # In practice, update model with adaptive learning
    
    return {
        "status": "success",
        "message": "Feedback recorded"
    }


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics(username: str = Depends(verify_token)):
    """
    Get system performance metrics.
    
    Returns:
        Current system metrics
    """
    # Placeholder metrics
    # In practice, query from monitoring system (Prometheus)
    
    return MetricsResponse(
        total_predictions=1000,
        avg_latency_ms=250.0,
        accuracy=0.95,
        coverage=0.96,
        escalation_rate=0.15,
        uptime_hours=720.0
    )


@app.get("/api/v1/audit-log")
async def get_audit_log(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    username: str = Depends(verify_token)
):
    """
    Retrieve audit log entries (EU AI Act compliance).
    
    Args:
        start_date: Filter start date
        end_date: Filter end date
        
    Returns:
        Audit log entries
    """
    # Load audit logger
    from pathlib import Path
    audit_logger = AuditLogger(Path("./audit_logs"))
    
    # Export report
    df = audit_logger.export_report(start_date, end_date)
    
    return df.to_dict(orient='records')


@app.get("/api/v1/compliance")
async def get_compliance_status(username: str = Depends(verify_token)):
    """
    Get EU AI Act compliance status.
    
    Returns:
        Compliance check results
    """
    # Placeholder compliance check
    # In practice, load from ComplianceManager
    
    return {
        "technical_documentation": True,
        "record_keeping": True,
        "transparency": True,
        "human_oversight": True,
        "accuracy_robustness": True,
        "overall_compliant": True
    }


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts.
    
    Streams detection alerts to connected clients.
    """
    await websocket.accept()
    
    try:
        while True:
            # Listen for new alerts from Redis pub/sub
            # In practice, subscribe to alert channel
            
            # Send alert to client
            await websocket.send_json({
                "timestamp": datetime.utcnow().isoformat(),
                "alert_type": "apt_detected",
                "severity": "high",
                "message": "Potential APT detected in System A"
            })
            
            # Keep connection alive
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("Client disconnected")


# ==================== Celery Tasks ====================

@celery_app.task(bind=True)
def detect_apt_task(
    self,
    log_content: str,
    log_format: str,
    critical_assets: Optional[List[str]],
    confidence_level: float,
    username: str
) -> Dict[str, Any]:
    """
    Async task for APT detection.
    
    This runs the full detection pipeline in background.
    """
    import torch
    from pathlib import Path
    
    # Update task state
    self.update_state(state='PROGRESS', meta={'step': 'parsing_logs'})
    
    # Parse logs
    parser = ProvenanceParser()
    graph = parser.parse_logs(log_content, log_format=log_format)
    
    # Update state
    self.update_state(state='PROGRESS', meta={'step': 'detecting_anomalies'})
    
    # Load detector model
    detector = APTDetector.load_from_checkpoint("models/detector.ckpt")
    
    # Convert to PyTorch Geometric
    pyg_graph = graph.to_pytorch_geometric()
    
    # Detect anomalies
    is_anomaly, anomaly_score = detector.detect_anomaly(pyg_graph)
    
    if not is_anomaly:
        return {
            "task_id": self.request.id,
            "status": "completed",
            "anomaly_detected": False,
            "anomaly_score": float(anomaly_score),
            "causal_chains": [],
            "explanations": [],
            "confidence_intervals": [],
            "should_escalate": False,
            "mitre_techniques": []
        }
    
    # Update state
    self.update_state(state='PROGRESS', meta={'step': 'causal_discovery'})
    
    # Graph reduction
    distiller = GraphDistiller(critical_assets or [])
    reduced_graph = distiller.distill(graph.graph)
    
    # Causal discovery
    attack_knowledge = ATTACKKnowledge()
    ci_tester = BatchCITester()
    discoverer = TemporalPCStable(ci_tester, attack_knowledge)
    
    causal_graph = discoverer.discover_causal_graph(
        reduced_graph,
        graph.node_features
    )
    
    # Extract attack chains
    attack_chains = causal_graph.extract_attack_chains(top_k=5)
    
    # Update state
    self.update_state(state='PROGRESS', meta={'step': 'generating_explanations'})
    
    # Generate explanations
    explainer = CausalExplainer(causal_graph, attack_knowledge)
    explanations = []
    techniques = set()
    
    for chain in attack_chains:
        explanation = explainer.explain_attack(chain)
        explanations.append(explanation.narrative)
        techniques.update(t.tid for t in explanation.attack_techniques)
    
    # Uncertainty quantification
    # Placeholder: in practice, use UncertaintyQuantifier
    
    return {
        "task_id": self.request.id,
        "status": "completed",
        "anomaly_detected": True,
        "anomaly_score": float(anomaly_score),
        "causal_chains": attack_chains,
        "explanations": explanations,
        "confidence_intervals": [],
        "should_escalate": float(anomaly_score) > 0.9,
        "mitre_techniques": list(techniques)
    }


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("Starting CausalDefend API...")
    # In practice: load models, connect to DB, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    print("Shutting down CausalDefend API...")
    # In practice: close DB connections, save state, etc.


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
