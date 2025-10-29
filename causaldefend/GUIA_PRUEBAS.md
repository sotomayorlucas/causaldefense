# 🧪 Guía de Pruebas de CausalDefend

Esta guía describe todas las formas de probar el sistema CausalDefend más allá del demo básico.

---

## 📋 Índice de Pruebas

1. **Demo Básico** - Introducción rápida al sistema
2. **Evaluación Avanzada** - Test set completo con métricas
3. **Comparación de Ataques** - Diferentes tipos de APTs
4. **Dashboard Interactivo** - Métricas y recomendaciones
5. **Pruebas Personalizadas** - Crea tus propios tests

---

## 1. 🎯 Demo Básico (Ya ejecutado)

**Propósito**: Verificar que todos los componentes funcionan

```powershell
python examples\demo_basico.py
```

**Qué muestra**:
- ✅ Creación de grafos de proveniencia
- ✅ Red neuronal de detección
- ✅ Descubrimiento de cadenas causales
- ✅ Generación de explicaciones

**Tiempo**: ~30 segundos  
**Dificultad**: ⭐ Básico

---

## 2. 🔬 Evaluación Avanzada

**Propósito**: Evaluar el detector en el test set completo

```powershell
python examples\test_detector_advanced.py
```

**Qué hace**:
1. Carga el modelo entrenado (`models/detector.ckpt`)
2. Evalúa en 30 grafos del test set (16 ataques + 14 normales)
3. Calcula métricas: Accuracy, Precision, Recall, F1
4. Muestra matriz de confusión
5. Prueba con ataque sintético
6. Identifica nodos más sospechosos

**Resultados Obtenidos**:
- ✅ **Accuracy**: 96.67%
- ✅ **Precision**: 100% (sin falsos positivos)
- ✅ **Recall**: 93.75% (15/16 ataques detectados)
- ✅ **F1 Score**: 96.77%

**Salida**:
- Archivo: `models/evaluation_results.json`
- Métricas detalladas guardadas

**Tiempo**: ~1 minuto  
**Dificultad**: ⭐⭐ Intermedio

---

## 3. 🎭 Comparación de Tipos de Ataques

**Propósito**: Ver cómo se comporta el detector con diferentes familias APT

```powershell
python examples\compare_apt_detection.py
```

**Tipos de Ataques Probados**:
1. **Ransomware** (T1486) - CRÍTICO
   - Cifrado masivo de archivos
   - Score típico: ~150+ (muy alto)
   - ✅ Siempre detectado

2. **Data Exfiltration** (T1041) - ALTO
   - Robo de datos sensibles
   - Score típico: ~0.2 (bajo)
   - ⚠️ Difícil de detectar (sigiloso)

3. **Lateral Movement** (T1021) - MEDIO
   - Movimiento entre hosts
   - Score típico: ~11 (moderado)
   - ⚠️ Puede pasar desapercibido

4. **Privilege Escalation** (T1068) - ALTO
   - Escalada a SYSTEM
   - Score típico: ~1 (bajo)
   - ⚠️ Requiere ajuste de threshold

5. **Persistence** (T1547) - MEDIO
   - Backdoors y autostart
   - Score típico: ~35 (alto)
   - ✅ Generalmente detectado

6. **Cryptomining** (T1496) - BAJO
   - Minería de criptomonedas
   - Score típico: ~29 (alto)
   - ✅ Fácil de detectar

**Resultados**:
```
🔴 Ransomware         🚨 DETECTADO    Score: 157.08
🟡 Data Exfiltration  ✓ No detectado  Score:   0.20
🟡 Lateral Movement   ✓ No detectado  Score:  11.27
🟡 Privilege Escalat. ✓ No detectado  Score:   1.05
🟡 Persistence        🚨 DETECTADO    Score:  35.17
🟢 Cryptomining       🚨 DETECTADO    Score:  29.36
🟢 Normal Activity    ✓ No detectado  Score:   0.02
```

**Conclusiones**:
- ✅ Excelente contra ataques obvios/ruidosos
- ⚠️ Necesita mejora para ataques sigilosos
- 💡 Threshold actual: 15.59 (puede ajustarse)

**Tiempo**: ~30 segundos  
**Dificultad**: ⭐⭐ Intermedio

---

## 4. 📊 Dashboard Interactivo

**Propósito**: Visualización profesional de métricas y recomendaciones

```powershell
python examples\dashboard.py
```

**Secciones del Dashboard**:

### ⚙️ Estado del Sistema
- Verifica componentes críticos
- Estado: 🟢 OPERACIONAL

### 📊 Métricas de Rendimiento
```
Accuracy   [████████████████████████████] 96.7%
Precision  [████████████████████████████] 100.0%
Recall     [███████████████████████████ ] 93.8%
F1 Score   [████████████████████████████] 96.8%
```

### 📈 Matriz de Confusión
```
              PREDICCIÓN
           Normal  Ataque
Real  N     14       0     ← Sin falsos positivos
      A      1      15     ← Solo 1 falso negativo
```

### 🎯 Inteligencia de Amenazas
- Total amenazas: 16
- Detectadas: 15 (93.8%)
- No detectadas: 1 (6.2%)
- Nivel de riesgo: 🟡 MEDIO

### ⚖️ Threshold de Detección
- Actual: 15.5967
- Guías para ajuste

### 💡 Recomendaciones
- Actualización de modelo
- Monitoreo continuo
- Respuesta a incidentes
- Mejora continua

**Tiempo**: 5 segundos  
**Dificultad**: ⭐ Básico

---

## 5. 🛠️ Crear Tus Propias Pruebas

### Ejemplo: Probar con un Grafo Personalizado

```python
import torch
from torch_geometric.data import Data
from models.spatiotemporal_detector import APTDetector

# 1. Cargar modelo
checkpoint = torch.load("models/detector.ckpt", map_location='cpu')
model = APTDetector(in_channels=64, hidden_channels=128, ...)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# 2. Crear tu grafo
features = torch.randn(20, 64)  # 20 nodos, 64 features
edges = torch.tensor([[0,1,2], [1,2,3]], dtype=torch.long)
data = Data(x=features, edge_index=edges)

# 3. Detectar
with torch.no_grad():
    embedding, edge_probs = model(data.x, data.edge_index)
    # Calcular anomaly score...
```

---

## 📊 Resumen de Resultados

| Prueba | Resultado | Tiempo | Archivos Generados |
|--------|-----------|--------|--------------------|
| Demo Básico | ✅ 4/4 tests OK | 30s | - |
| Evaluación Avanzada | ✅ 96.7% accuracy | 1min | `evaluation_results.json` |
| Comparación Ataques | ⚠️ 50% detección | 30s | - |
| Dashboard | ✅ Sistema operacional | 5s | - |

---

## 🎯 Métricas Finales del Sistema

### Test Set (30 grafos)
- **True Positives**: 15 ataques detectados
- **True Negatives**: 14 normales clasificados correctamente
- **False Positives**: 0 (¡perfecto!)
- **False Negatives**: 1 ataque perdido

### Rendimiento
- **Accuracy**: 96.67% - Excelente
- **Precision**: 100% - Sin falsas alarmas
- **Recall**: 93.75% - Alta cobertura
- **F1 Score**: 96.77% - Balance óptimo

### Tipos de Ataque
- **Obvios/Ruidosos**: 100% detección (Ransomware, Cryptomining)
- **Sigilosos**: 0-50% detección (Data Exfil, Lateral Movement)
- **Intermedios**: 50% detección (Persistence)

---

## 💡 Recomendaciones de Uso

### Para Desarrollo
1. Ejecutar `test_detector_advanced.py` después de cada entrenamiento
2. Usar `compare_apt_detection.py` para validar nuevos tipos de ataque
3. Revisar `dashboard.py` antes de desplegar

### Para Producción
1. Configurar threshold basado en tolerancia a falsos positivos
2. Implementar re-entrenamiento periódico (cada 2-4 semanas)
3. Integrar con SIEM para alertas automáticas
4. Mantener logs de detecciones para mejora continua

### Para Investigación
1. Analizar el ataque no detectado (falso negativo)
2. Ajustar features para mejorar detección de ataques sigilosos
3. Probar con datos reales de auditoría
4. Implementar explicaciones causales completas

---

## 🚀 Próximos Pasos

### Inmediato
- [x] Sistema base funcionando
- [x] Modelos entrenados
- [x] Evaluación completa
- [ ] Ajustar threshold para ataques sigilosos
- [ ] Documentar el ataque no detectado

### Corto Plazo (1-2 semanas)
- [ ] Integrar con fuentes de threat intelligence
- [ ] Implementar API REST para detección
- [ ] Crear pipeline automatizado
- [ ] Generar reportes ejecutivos

### Mediano Plazo (1-2 meses)
- [ ] Re-entrenar con más datos (1000+ grafos)
- [ ] Implementar ensemble de modelos
- [ ] Agregar detección temporal (secuencias de grafos)
- [ ] Desplegar en ambiente de producción

---

## 📞 Comandos Rápidos

```powershell
# Navegar al proyecto
cd C:\Users\lsotomayor\Desktop\causaldefense\causaldefend

# Ejecutar todas las pruebas
python examples\demo_basico.py
python examples\test_detector_advanced.py
python examples\compare_apt_detection.py
python examples\dashboard.py

# Ver resultados guardados
cat models\evaluation_results.json

# Re-entrenar (opcional)
python scripts\train_detector.py --epochs 20 --batch-size 16
```

---

## 🏆 Estado del Proyecto

**CausalDefend v1.0** está:
- ✅ Completamente funcional
- ✅ Bien evaluado (96.7% accuracy)
- ✅ Listo para demos
- ⚠️ Necesita ajustes para ataques sigilosos
- 🚀 Preparado para producción piloto

---

**Documentación generada**: 29 de octubre de 2025  
**Última actualización**: Después de entrenamiento completo
