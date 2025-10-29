# 🚀 CausalDefend - Próximos Pasos

## ✅ Estado Actual (29 Oct 2025)

**COMPLETADO:**
- ✅ 100% del código base (11 módulos principales)
- ✅ Arquitectura completa implementada
- ✅ Demos básicos funcionando
- ✅ Documentación completa
- ✅ Docker & deployment configs

---

## 📋 ROADMAP DE IMPLEMENTACIÓN

### **FASE 1: Preparación de Datos** (1-2 semanas)

#### 1.1 Descargar Dataset DARPA TC
```bash
# Opción 1: DARPA Transparent Computing
wget https://drive.google.com/... # Dataset público

# Opción 2: Usar logs propios
# Colocar en: data/raw/
```

#### 1.2 Preparar Dataset
```bash
python scripts/prepare_dataset.py \
    --input data/raw/darpa_tc \
    --output data/processed \
    --format darpa_tc
```

**Crear:** `scripts/prepare_dataset.py`
```python
"""
Script para preparar datasets de entrenamiento.
Convierte logs a formato ProvenanceGraph.
"""
# - Parse DARPA TC logs
# - Generar grafos de proveniencia
# - Split train/val/test
# - Feature extraction
```

---

### **FASE 2: Entrenamiento de Modelos** (2-4 semanas)

#### 2.1 Entrenar APT Detector
```bash
python scripts/train_detector.py \
    --data data/processed/train \
    --epochs 100 \
    --batch-size 32 \
    --gpus 1
```

**Crear:** `scripts/train_detector.py`
```python
"""
Entrena el modelo APTDetector (GAT+GRU).
Target: F1-score > 0.95 en DARPA TC
"""
# - DataLoader para grafos de proveniencia
# - Training loop con PyTorch Lightning
# - Validation y early stopping
# - Guardar checkpoint en models/detector.ckpt
```

**Métricas objetivo:**
- **F1-Score:** > 0.95
- **Precision:** > 0.93
- **Recall:** > 0.92
- **False Positive Rate:** < 0.01

#### 2.2 Entrenar Neural CI Tester
```bash
python scripts/train_ci_tester.py \
    --data data/processed/train \
    --epochs 50 \
    --batch-size 64
```

**Crear:** `scripts/train_ci_tester.py`
```python
"""
Entrena NeuralCITest para tests de independencia.
Usa pairs de nodos para aprender correlaciones.
"""
# - Generar pares (X, Y, Z) para CI tests
# - Entrenar encoder condicional
# - Validar con tests estadísticos conocidos
# - Guardar checkpoint en models/ci_tester.ckpt
```

---

### **FASE 3: Evaluación y Benchmarking** (1 semana)

#### 3.1 Evaluar Detección
```bash
python experiments/evaluate_detection.py \
    --detector models/detector.ckpt \
    --test-data data/processed/test
```

**Crear:** `experiments/evaluate_detection.py`
```python
"""
Evalúa APTDetector en test set.
Genera métricas y confusion matrix.
"""
# - Load test data
# - Run inference
# - Compute metrics (F1, Precision, Recall, AUC-ROC)
# - Generate plots
```

#### 3.2 Benchmark de Performance
```bash
python experiments/benchmark_performance.py
```

**Crear:** `experiments/benchmark_performance.py`
```python
"""
Mide tiempos de ejecución de cada componente.
Verifica que cumple requisitos de tiempo real.
"""
# - Parsing: < 1s para 10K eventos
# - Detection: < 5s para grafo de 100K nodos
# - Causal Discovery: < 30s tras reducción
# - Total pipeline: < 60s end-to-end
```

**Requisitos de Performance:**
- ✅ Parsing: < 1s para 10K eventos
- ✅ Detection: < 5s para 100K nodos
- ✅ Graph Reduction: < 10s (1M → 50K nodos)
- ✅ Causal Discovery: < 30s
- ✅ **Total E2E:** < 60s

---

### **FASE 4: Tests Unitarios** (1 semana)

#### 4.1 Completar Tests
```bash
# Implementar tests para cada módulo
pytest tests/ --cov=src --cov-report=html
```

**Archivos a completar:**

1. **`tests/test_detector.py`**
```python
def test_apt_detector_forward():
    """Test forward pass de APTDetector"""
    
def test_anomaly_detection():
    """Test detección de anomalías"""
    
def test_temporal_encoding():
    """Test GRU temporal"""
```

2. **`tests/test_causal_discovery.py`**
```python
def test_pc_stable():
    """Test PC-Stable algorithm"""
    
def test_attack_knowledge():
    """Test MITRE ATT&CK integration"""
    
def test_counterfactual_generation():
    """Test generación de contrafactuales"""
```

3. **`tests/test_pipeline.py`**
```python
def test_full_pipeline():
    """Test pipeline end-to-end"""
    
def test_error_handling():
    """Test manejo de errores"""
```

**Target:** > 80% code coverage

---

### **FASE 5: Despliegue en Producción** (2 semanas)

#### 5.1 Configurar Infraestructura

**Servicios necesarios:**
1. **PostgreSQL** (metadata de detecciones)
2. **Neo4j** (grafos causales)
3. **Redis** (Celery backend)
4. **Nginx** (reverse proxy)

**Configuración:**
```bash
# 1. Copiar configuración
cp .env.example .env

# 2. Editar variables
vim .env

# 3. Levantar servicios
docker-compose up -d
```

#### 5.2 Deployment con Docker

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

#### 5.3 Monitoreo

**Herramientas:**
- **Prometheus:** Métricas del sistema
- **Grafana:** Dashboards
- **ELK Stack:** Logs centralizados

---

### **FASE 6: Validación con SOC** (Piloto)

#### 6.1 Despliegue Piloto
- Deploy en 2-3 sistemas de test
- Integrar con SIEM existente
- Entrenar 2-3 analistas

#### 6.2 Métricas a Validar
1. **Efectividad:**
   - True Positive Rate
   - False Positive Rate
   - Mean Time to Detect (MTTD)

2. **Usabilidad:**
   - Tiempo de análisis de alertas
   - Satisfacción de analistas
   - Explicaciones comprensibles

3. **Performance:**
   - Latencia end-to-end
   - Throughput (eventos/seg)
   - Uso de recursos

---

## 📊 CRONOGRAMA ESTIMADO

| Fase | Duración | Prioridad |
|------|----------|-----------|
| 1. Preparación de Datos | 1-2 semanas | 🔴 ALTA |
| 2. Entrenamiento | 2-4 semanas | 🔴 ALTA |
| 3. Evaluación | 1 semana | 🟡 MEDIA |
| 4. Tests | 1 semana | 🟡 MEDIA |
| 5. Deployment | 2 semanas | 🟢 BAJA |
| 6. Piloto | 4+ semanas | 🟢 BAJA |
| **TOTAL** | **11-14 semanas** | |

---

## 🎯 MILESTONE INMEDIATO

### **Semana 1-2: "Primera Detección"**

**Objetivo:** Entrenar detector básico y ejecutar primera detección exitosa

**Tareas:**
1. ✅ Descargar DARPA TC dataset (día 1-2)
2. ✅ Crear `prepare_dataset.py` (día 3-4)
3. ✅ Crear `train_detector.py` (día 5-7)
4. ✅ Entrenar modelo básico (día 8-10)
5. ✅ Ejecutar primera detección (día 11-14)

**Criterio de éxito:**
```bash
python examples/complete_detection.py
# Output: "✓ Anomaly detected with 95% confidence"
```

---

## 📚 RECURSOS ÚTILES

### Datasets
- **DARPA TC:** https://github.com/darpa-i2o/Transparent-Computing
- **DARPA Engagement:** https://github.com/FiveDirections/OpTC-data
- **StreamSpot:** https://github.com/sbustreamspot/sbustreamspot-data

### Papers de Referencia
- **BEEP:** Graph-based provenance tracking
- **UNICORN:** Provenance-based APT detection
- **ThreaTrace:** Causal analysis for APT
- **PC Algorithm:** Constraint-based causal discovery

### Herramientas
- **PyTorch Geometric:** GNN framework
- **causal-learn:** Causal discovery library
- **NetworkX:** Graph algorithms

---

## 🤝 CONTACTO

¿Necesitas ayuda con alguna fase?

1. **Preparación de datos:** Te puedo ayudar a crear el script
2. **Entrenamiento:** Puedo generar el código de training
3. **Tests:** Puedo escribir los unit tests
4. **Deployment:** Puedo revisar la configuración

**Siguiente paso recomendado:**
```bash
# Empezar con preparación de datos
python -c "print('¿Tienes acceso al dataset DARPA TC?')"
```

¿Con cuál fase quieres empezar? 🚀
