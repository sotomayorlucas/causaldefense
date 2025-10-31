# 🚀 Guía Rápida de Ejecución - CausalDefend

## 📋 Pre-requisitos

✅ Dataset importado y procesado (200 grafos)  
✅ Modelo entrenado (`models/detector.ckpt`)  
✅ Ambiente Python configurado

---

## 🎯 Scripts Disponibles

### 1️⃣ Evaluación Avanzada del Detector

**Propósito**: Evaluar el modelo entrenado con el test set completo

```powershell
python examples/test_detector_advanced.py
```

**Salida esperada**:
- Carga del modelo y hiperparámetros
- Evaluación en 30 grafos del test set
- Métricas: Accuracy, Precision, Recall, F1
- Matriz de confusión
- Detección de ataque sintético
- Top 5 nodos sospechosos

**Tiempo estimado**: ~5 segundos

---

### 2️⃣ Dashboard de Métricas

**Propósito**: Visualizar métricas del modelo de forma interactiva

```powershell
python examples/dashboard.py
```

**Salida esperada**:
- Estado del sistema (componentes OK)
- Barras de progreso de métricas (96.7% accuracy, 100% precision)
- Matriz de confusión detallada
- Análisis de riesgo
- Recomendaciones operacionales

**Tiempo estimado**: ~3 segundos

---

### 3️⃣ Comparación de Tipos de Ataques

**Propósito**: Probar el detector con diferentes tipos de ataques APT

```powershell
python examples/compare_apt_detection.py
```

**Salida esperada**:
- Detección de 6 tipos de ataques:
  - Ransomware (Score: ~2560) ✅ DETECTADO
  - Cryptomining (Score: ~277) ✅ DETECTADO
  - Persistence (Score: ~218) ✅ DETECTADO
  - Privilege Escalation (Score: ~202) ✅ DETECTADO
  - Lateral Movement (Score: ~49) ✅ DETECTADO
  - Data Exfiltration (Score: ~21) ✅ DETECTADO
- Tasa de detección: 100%
- Análisis por severidad y sigilo

**Tiempo estimado**: ~5 segundos

---

### 4️⃣ Demo Simple de Detección

**Propósito**: Demostración básica de carga del modelo y detección

```powershell
python examples/simple_detection_demo.py
```

**Salida esperada**:
- Carga automática del modelo
- Inferencia de hiperparámetros
- Creación de grafos sintéticos
- Detección en grafo de ataque y normal
- Anomaly scores

**Tiempo estimado**: ~3 segundos

---

### 5️⃣ Pipeline Completo Simplificado

**Propósito**: Demostración del pipeline completo de detección

```powershell
python examples/complete_detection_simple.py
```

**Salida esperada**:
- Carga del modelo APT Detector
- Creación de grafos de proveniencia
- Detección de patrones de ataque
- Análisis de actividad normal
- Resumen de detección con recomendaciones

**Tiempo estimado**: ~3 segundos

---

## 📊 Resultados Esperados

### Métricas del Modelo
```
Accuracy:   96.67%
Precision: 100.00% ← ¡Sin falsos positivos!
Recall:     93.75%
F1 Score:   96.77%
```

### Matriz de Confusión
```
         Predicho
         Neg  Pos
Neg  │   14    0   ← Perfecto
Pos  │    1   15   ← Solo 1 FN
```

---

## 🐛 Solución de Problemas

### Error: "No module named 'causaldefend'"
```powershell
# Asegúrate de estar en el directorio correcto
cd C:\Users\sotom\OneDrive\Escritorio\causaldefense-20251029T220821Z-1-001\causaldefense\causaldefend
```

### Error: "Checkpoint not found"
```powershell
# Verifica que el modelo esté entrenado
ls models/detector.ckpt

# Si no existe, entrena el modelo
python scripts/train_detector.py
```

### Error: "UnicodeEncodeError"
✅ **Ya está arreglado** - Los scripts ahora usan UTF-8 en Windows

### Caracteres raros en la salida (Ô£ô en lugar de ✓)
✅ **Normal en PowerShell** - El script funciona correctamente, solo es un problema de visualización

---

## 💡 Tips de Uso

1. **Ejecuta los scripts en orden** para ver la progresión completa
2. **Revisa los logs** para entender el flujo de ejecución
3. **Compara los scores** entre ataques y actividad normal
4. **Ajusta el threshold** (12.4066) según tus necesidades

---

## 📈 Interpretación de Resultados

### Anomaly Scores
- **Score > 12.4066**: Clasificado como ATAQUE 🚨
- **Score ≤ 12.4066**: Clasificado como NORMAL ✅

### Confianza de Detección
- **> 90%**: Alta confianza → Escalar inmediatamente
- **70-90%**: Media confianza → Investigar
- **< 70%**: Baja confianza → Monitorear

### Tipos de Ataques (por Score Promedio)
1. **Ransomware**: ~2560 (Muy obvio)
2. **Cryptomining**: ~277 (Obvio)
3. **Persistence**: ~218 (Moderado)
4. **Privilege Escalation**: ~202 (Moderado)
5. **Lateral Movement**: ~49 (Sutil)
6. **Data Exfiltration**: ~21 (Muy sutil)

---

## 🎓 Flujo de Trabajo Recomendado

```
1. Evaluación Inicial
   ↓
   python examples/test_detector_advanced.py

2. Revisar Métricas
   ↓
   python examples/dashboard.py

3. Probar Diferentes Ataques
   ↓
   python examples/compare_apt_detection.py

4. Demo del Pipeline
   ↓
   python examples/complete_detection_simple.py
```

---

## 🔗 Archivos Importantes

- `models/detector.ckpt` - Modelo entrenado
- `models/evaluation_results.json` - Métricas detalladas
- `data/processed/` - Dataset procesado
- `logs/apt_detector/` - Logs de entrenamiento

---

## ✨ ¡Listo para Usar!

Todos los scripts están completamente funcionales y listos para ejecutar.  
¡Disfruta explorando las capacidades de **CausalDefend**! 🛡️🔒
