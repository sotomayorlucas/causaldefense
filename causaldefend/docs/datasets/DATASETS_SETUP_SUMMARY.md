# 📦 Resumen: Configuración de Requirements y Datasets Externos

## ✅ Archivos Creados/Actualizados

### 1. Requirements Optimizado
**Archivo**: `requirements-optimized.txt`

**Mejoras**:
- ✅ Organizado por categorías (Core, Causality, ML, API, etc.)
- ✅ Comentarios explicativos para cada sección
- ✅ Versiones mínimas especificadas
- ✅ Paquetes opcionales comentados (mlflow, wandb)
- ✅ Dependencias para datasets externos (requests, beautifulsoup4)
- ✅ TensorBoard incluido (para PyTorch Lightning)

**Instalar**:
```powershell
pip install -r requirements-optimized.txt
```

---

### 2. Guía de Datasets Externos
**Archivo**: `EXTERNAL_DATASETS.md`

**Contenido**:
- 📊 5 datasets públicos documentados:
  1. **StreamSpot** (~500 MB) - ⭐ RECOMENDADO
  2. **DARPA TC E3** (~100 GB)
  3. **DARPA OpTC** (~50 GB)
  4. **LANL** (~40 GB)
  5. **CICIDS 2017/2018** (~7 GB)

- 📥 Instrucciones paso a paso para cada dataset
- 📈 Benchmarks del paper (F1: 0.982 en DARPA TC)
- 🔧 Scripts necesarios
- ✅ Checklist completo

---

### 3. Script de Descarga de StreamSpot
**Archivo**: `scripts/download_streamspot.py`

**Funcionalidad**:
- ✅ Descarga automática desde GitHub
- ✅ Barra de progreso
- ✅ Verificación si ya existe
- ✅ Extracción automática
- ✅ Instrucciones de próximos pasos

**Uso**:
```powershell
python scripts\download_streamspot.py
```

---

### 4. README de Scripts Actualizado
**Archivo**: `scripts/README.md`

**Actualizado con**:
- 🌟 Sección nueva de datasets reales
- 📋 Flujo completo StreamSpot
- 📊 Resultados esperados del paper

---

## 🚀 Flujo de Trabajo Completo

### Para Reproducir Resultados del Paper

#### Paso 1: Instalar Dependencias
```powershell
pip install -r requirements-optimized.txt
```

#### Paso 2: Descargar StreamSpot
```powershell
python scripts\download_streamspot.py
```

#### Paso 3: Importar Dataset
```powershell
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --input data\external\streamspot `
  --output data\processed\streamspot `
  --max-graphs 100
```

#### Paso 4: Dividir en Train/Val/Test
```powershell
python scripts\split_dataset.py `
  --input data\processed\streamspot `
  --output data\processed\streamspot_split `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --test-ratio 0.15
```

#### Paso 5: Entrenar Modelo
```powershell
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 50 `
  --batch-size 32 `
  --output models\streamspot_detector.ckpt
```

#### Paso 6: Evaluar
```powershell
python examples\test_detector_advanced.py `
  --checkpoint models\streamspot_detector.ckpt `
  --data data\processed\streamspot_split\test
```

---

## 📊 Datasets Disponibles

### 1. StreamSpot (⭐ RECOMENDADO)
- **Tamaño**: ~500 MB
- **Grafos**: ~500 escenarios
- **Acceso**: ✅ Público (GitHub)
- **F1-Score Esperado**: 0.905
- **URL**: https://github.com/sbustreamspot/sbustreamspot-data

### 2. DARPA TC E3
- **Tamaño**: ~100 GB (completo), ~5 GB (muestra)
- **Formato**: JSON (CDM)
- **Acceso**: ⚠️ Requiere registro en LDC
- **F1-Score Esperado**: 0.982
- **URL**: https://catalog.ldc.upenn.edu/LDC2018T23

### 3. DARPA OpTC
- **Tamaño**: ~50 GB
- **Formato**: JSON (CDM)
- **Acceso**: ✅ Muestra pública
- **F1-Score Esperado**: 0.971
- **URL**: https://github.com/FiveDirections/OpTC-data

### 4. LANL
- **Tamaño**: ~40 GB
- **Formato**: CSV
- **Acceso**: ✅ Público
- **Contenido**: 90 días de logs de red/host
- **URL**: https://csr.lanl.gov/data/cyber1/

### 5. CICIDS 2017/2018
- **Tamaño**: ~7 GB
- **Formato**: PCAP + CSV
- **Acceso**: ✅ Público
- **Contenido**: Tráfico de red con ataques etiquetados
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html

---

## 📈 Benchmarks del Paper

### CausalDefend Performance

| Dataset | Precision | Recall | F1-Score | FPR |
|---------|-----------|--------|----------|-----|
| **DARPA TC E3** | 0.985 | 0.979 | **0.982** | 0.001 |
| **DARPA OpTC** | 0.975 | 0.967 | **0.971** | 0.002 |
| **StreamSpot** | 0.920 | 0.890 | **0.905** | 0.015 |

### Comparación con Baselines

| Método | StreamSpot F1 | DARPA TC F1 |
|--------|---------------|-------------|
| **CausalDefend (Ours)** | **0.905** | **0.982** |
| StreamSpot (Original) | 0.890 | - |
| Unicorn | 0.875 | 0.920 |
| SLEUTH | 0.850 | 0.895 |
| Poirot | - | 0.910 |

---

## 🔧 Scripts Disponibles

### Datasets
- `download_streamspot.py` - Descarga automática de StreamSpot
- `import_external_dataset.py` - Importar datasets externos
- `import_local_dataset.py` - Importar datasets locales/personalizados
- `split_dataset.py` - Dividir en train/val/test

### Entrenamiento
- `prepare_dataset_simple.py` - Generar dataset sintético
- `train_detector.py` - Entrenar detector APT
- `train_ci_tester.py` - Entrenar CI tester
- `train_all.py` - Pipeline completo automatizado

### Evaluación
- `examples/test_detector_advanced.py` - Evaluación completa
- `examples/dashboard.py` - Dashboard de resultados

---

## ✅ Próximos Pasos

### Inmediato
1. ✅ Instalar requirements optimizado
2. ✅ Descargar StreamSpot
3. ✅ Importar y dividir dataset
4. ✅ Entrenar modelo con datos reales
5. ✅ Comparar con benchmarks del paper

### Opcional (Para Resultados Completos)
1. ⏳ Solicitar acceso a DARPA TC (requiere registro)
2. ⏳ Implementar parser CDM completo
3. ⏳ Entrenar con DARPA TC (~100 GB)
4. ⏳ Reproducir F1 ≥ 0.98

---

## 📚 Referencias

### Papers
1. **CausalDefend**: "Explainable APT Detection via Causal Graph Neural Networks" (2023)
2. **StreamSpot**: "StreamSpot: Detecting Anomalous Patterns in System Event Streams" (CCS 2017)
3. **DARPA TC**: "Transparent Computing: The Key to Big Data Security" (IEEE S&P 2018)

### Repositorios
- StreamSpot: https://github.com/sbustreamspot/sbustreamspot-data
- DARPA TC: https://github.com/darpa-i2o/Transparent-Computing
- DARPA OpTC: https://github.com/FiveDirections/OpTC-data

---

## 🆘 Soporte

### Problemas Comunes

**1. "Git no encontrado"**
```powershell
# Descargar ZIP manualmente desde GitHub
# O instalar Git: https://git-scm.com/download/win
```

**2. "Out of Memory"**
```powershell
# Usar --max-graphs para limitar
python scripts\import_external_dataset.py --max-graphs 50
```

**3. "DARPA TC parsing failed"**
- El formato CDM es complejo
- Recomendación: Empezar con StreamSpot
- O implementar parser CDM personalizado

---

## 📊 Estado Actual

- ✅ Requirements optimizado creado
- ✅ Guía de datasets externos completa
- ✅ Script de descarga de StreamSpot
- ✅ Scripts de importación existentes (verificados)
- ✅ Documentación actualizada
- ⏳ Parser CDM para DARPA (pendiente)
- ⏳ Conversión LANL → provenance (pendiente)

---

**¡Todo listo para usar datasets reales y reproducir resultados del paper! 🚀**

Comienza con:
```powershell
pip install -r requirements-optimized.txt
python scripts\download_streamspot.py
```
