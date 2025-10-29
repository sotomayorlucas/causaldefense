# 🚀 CausalDefend Training Scripts

Scripts para entrenar los modelos de CausalDefend desde cero.

## 📋 Descripción

Este directorio contiene los scripts necesarios para:
1. **Generar datasets sintéticos** (para pruebas rápidas)
2. **Importar datasets externos** (StreamSpot, DARPA TC, etc.)
3. **Entrenar el detector APT** (GAT+GRU)
4. **Entrenar el CI tester** (Neural Conditional Independence)
5. **Pipeline completo automatizado**

---

## 🌟 NUEVO: Usar Datasets Reales

### Opción Recomendada: StreamSpot (~500 MB)

```powershell
# 1. Descargar StreamSpot (automático)
python scripts\download_streamspot.py

# 2. Importar al formato de CausalDefend
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --input data\external\streamspot `
  --output data\processed\streamspot `
  --max-graphs 100

# 3. Dividir en train/val/test
python scripts\split_dataset.py `
  --input data\processed\streamspot `
  --output data\processed\streamspot_split

# 4. Entrenar con datos reales
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 20 `
  --output models\streamspot_detector.ckpt

# 5. Evaluar
python examples\test_detector_advanced.py `
  --checkpoint models\streamspot_detector.ckpt `
  --data data\processed\streamspot_split\test
```

**Resultados Esperados** (según paper):
- F1-Score: ~0.90
- Precision: ~0.92
- Recall: ~0.89

📚 **Más información**: Ver [EXTERNAL_DATASETS.md](../docs/datasets/EXTERNAL_DATASETS.md)

---

## 🎯 Quick Start (Modo Rápido - Datos Sintéticos)

Para entrenar rápidamente con un dataset pequeño (ideal para testing):

```bash
# Entrenar TODO (5-10 minutos en CPU)
python scripts/train_all.py --mode quick

# O paso por paso:
python scripts/prepare_dataset.py --num-benign 100 --num-attack 100
python scripts/train_detector.py --epochs 5 --batch-size 16
python scripts/train_ci_tester.py --epochs 3 --batch-size 16
```

**Resultado:**
- ✅ Dataset: 200 grafos (100 benignos, 100 ataques)
- ✅ Detector: 5 epochs (~3 min)
- ✅ CI Tester: 3 epochs (~2 min)
- ✅ Modelos guardados en `models/`

---

## 🏭 Producción (Modo Completo)

Para entrenar con dataset completo (producción):

```bash
# Entrenar TODO (1-2 horas en CPU, 15-30 min en GPU)
python scripts/train_all.py --mode full

# O paso por paso:
python scripts/prepare_dataset.py --num-benign 500 --num-attack 500
python scripts/train_detector.py --epochs 100 --batch-size 32 --gpus 1
python scripts/train_ci_tester.py --epochs 50 --batch-size 64 --gpus 1
```

**Resultado:**
- ✅ Dataset: 1000 grafos (500 benignos, 500 ataques)
- ✅ Detector: 100 epochs (~45 min en GPU)
- ✅ CI Tester: 50 epochs (~30 min en GPU)
- ✅ Métricas objetivo: F1 > 0.95

---

## 📜 Scripts Disponibles

### 1. `prepare_dataset.py`

Genera datasets sintéticos de grafos de proveniencia con patrones de ataque APT.

**Uso:**
```bash
python scripts/prepare_dataset.py \
    --output data/processed \
    --num-benign 500 \
    --num-attack 500 \
    --avg-nodes 150 \
    --seed 42
```

**Argumentos:**
- `--output`: Directorio de salida (default: `data/processed`)
- `--num-benign`: Número de grafos benignos (default: 500)
- `--num-attack`: Número de grafos con ataques (default: 500)
- `--avg-nodes`: Promedio de nodos por grafo (default: 150)
- `--seed`: Semilla aleatoria (default: 42)

**Output:**
```
data/processed/
├── train/
│   ├── graph_0.pkl
│   ├── features_0.npy
│   ├── label_0.json
│   └── ...
├── val/
└── test/
```

**Patrones de Ataque Incluidos:**
1. **Phishing** (15%): Email → Browser → Malware
2. **Credential Dump** (20%): Process → LSASS → Credentials
3. **Lateral Movement** (18%): PSExec → Remote Host
4. **Data Exfiltration** (22%): Search → Compress → C2
5. **Privilege Escalation** (25%): Exploit → Kernel → Admin

---

### 2. `train_detector.py`

Entrena el modelo APTDetector (GAT+GRU) para detección de anomalías.

**Uso:**
```bash
python scripts/train_detector.py \
    --data data/processed \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --hidden-channels 128 \
    --num-heads 8 \
    --num-layers 3 \
    --gpus 1
```

**Argumentos:**
- `--data`: Directorio de datos (default: `data/processed`)
- `--output`: Directorio de checkpoints (default: `models`)
- `--epochs`: Número de epochs (default: 100)
- `--batch-size`: Tamaño de batch (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--hidden-channels`: Canales ocultos GAT (default: 128)
- `--num-heads`: Cabezas de atención (default: 8)
- `--num-layers`: Capas GAT (default: 3)
- `--gpus`: Número de GPUs (default: 0 = CPU)

**Output:**
```
models/
├── detector.ckpt                    # Modelo final
├── detector-epoch=XX-val_loss=Y.ckpt  # Mejores checkpoints
└── ...

logs/
└── apt_detector/
    └── version_0/
        └── events.out.tfevents.*    # TensorBoard logs
```

**Métricas Objetivo:**
- ✅ **F1-Score:** > 0.95
- ✅ **Precision:** > 0.93
- ✅ **Recall:** > 0.92
- ✅ **Val Loss:** < 0.1

**Visualizar Entrenamiento:**
```bash
tensorboard --logdir logs/apt_detector
# Abrir: http://localhost:6006
```

---

### 3. `train_ci_tester.py`

Entrena el Neural CI Tester para tests de independencia condicional.

**Uso:**
```bash
python scripts/train_ci_tester.py \
    --data data/processed \
    --epochs 50 \
    --batch-size 64 \
    --num-samples 10000 \
    --learning-rate 0.001 \
    --hidden-dim 128 \
    --num-layers 4 \
    --gpus 1
```

**Argumentos:**
- `--data`: Directorio de datos (default: `data/processed`)
- `--output`: Directorio de checkpoints (default: `models`)
- `--epochs`: Número de epochs (default: 50)
- `--batch-size`: Tamaño de batch (default: 64)
- `--num-samples`: Triplets (X,Y,Z) a generar (default: 10000)
- `--learning-rate`: Learning rate (default: 0.001)
- `--hidden-dim`: Dimensión oculta (default: 128)
- `--num-layers`: Número de capas (default: 4)
- `--gpus`: Número de GPUs (default: 0 = CPU)

**Output:**
```
models/
├── ci_tester.ckpt                    # Modelo final
├── ci_tester-epoch=XX-val_loss=Y.ckpt  # Mejores checkpoints
└── ...
```

**Métricas Objetivo:**
- ✅ **Accuracy:** > 0.85
- ✅ **Val Loss:** < 0.3

---

### 4. `train_all.py`

Pipeline automatizado que ejecuta todo el entrenamiento secuencialmente.

**Uso:**
```bash
# Modo rápido (testing)
python scripts/train_all.py --mode quick

# Modo completo (producción)
python scripts/train_all.py --mode full

# Con opciones de skip
python scripts/train_all.py --mode quick --skip-dataset  # Usar dataset existente
python scripts/train_all.py --mode full --skip-ci-tester  # Solo entrenar detector
```

**Modos:**

| Modo | Grafos | Detector Epochs | CI Epochs | Tiempo (CPU) | Tiempo (GPU) |
|------|--------|----------------|-----------|--------------|--------------|
| `quick` | 200 | 5 | 3 | ~10 min | ~3 min |
| `full` | 1000 | 100 | 50 | ~2 horas | ~30 min |

**Proceso:**
1. ✅ Genera dataset sintético
2. ✅ Entrena APT Detector
3. ✅ Entrena CI Tester
4. ✅ Valida modelos
5. ✅ Genera reporte

---

## 📊 Validación de Modelos

Después del entrenamiento, validar que los modelos funcionan:

```bash
# Verificar que los checkpoints existen
ls -lh models/

# Debería mostrar:
# detector.ckpt       (~50-100 MB)
# ci_tester.ckpt      (~20-40 MB)

# Ejecutar demo completo
python examples/complete_detection.py

# O demo básico
python examples/demo_basico.py
```

---

## 🐛 Troubleshooting

### Error: "No graphs found"
```bash
# Solución: Generar dataset primero
python scripts/prepare_dataset.py
```

### Error: "CUDA out of memory"
```bash
# Solución 1: Reducir batch size
python scripts/train_detector.py --batch-size 16

# Solución 2: Usar CPU
python scripts/train_detector.py --gpus 0
```

### Error: "NumPy warnings"
```bash
# Solución: Reinstalar NumPy con wheel precompilada
pip uninstall numpy -y
pip install --only-binary :all: numpy
```

### Warning: "Low F1-score"
Si después del entrenamiento el F1 < 0.90:

1. **Aumentar epochs:**
   ```bash
   python scripts/train_detector.py --epochs 200
   ```

2. **Aumentar dataset:**
   ```bash
   python scripts/prepare_dataset.py --num-benign 1000 --num-attack 1000
   ```

3. **Ajustar arquitectura:**
   ```bash
   python scripts/train_detector.py --hidden-channels 256 --num-layers 4
   ```

---

## 📈 Métricas Esperadas

### APT Detector (después de 100 epochs)
```
Epoch 100/100
train_loss: 0.042
train_acc: 0.978
val_loss: 0.055
val_acc: 0.963
✓ F1-Score: 0.95+
```

### CI Tester (después de 50 epochs)
```
Epoch 50/50
train_loss: 0.198
train_acc: 0.891
val_loss: 0.245
val_acc: 0.867
✓ Accuracy: 0.85+
```

---

## 🔄 Re-entrenar Modelos

Si necesitas re-entrenar con nuevos datos:

```bash
# 1. Limpiar modelos anteriores
rm -rf models/*.ckpt logs/*

# 2. Generar nuevo dataset
python scripts/prepare_dataset.py --num-benign 1000 --num-attack 1000

# 3. Re-entrenar
python scripts/train_all.py --mode full
```

---

## 🎓 Siguiente Nivel: Dataset Real (DARPA TC)

Para usar el dataset real DARPA TC en lugar de sintético:

1. **Descargar DARPA TC:**
   ```bash
   # Opción 1: Desde Google Drive
   wget https://drive.google.com/file/d/DARPA_TC_DATASET_ID
   
   # Opción 2: Desde repositorio oficial
   git clone https://github.com/darpa-i2o/Transparent-Computing
   ```

2. **Modificar `prepare_dataset.py`:**
   ```python
   # Agregar soporte para parsear logs DARPA TC
   # Ver: src/data/provenance_parser.py (ya tiene soporte)
   ```

3. **Entrenar con datos reales:**
   ```bash
   python scripts/prepare_dataset_darpa.py --input data/raw/darpa_tc
   python scripts/train_all.py --mode full
   ```

---

## 📚 Referencias

- **Paper Original:** CausalDefend (ver `causaldefend.tex`)
- **Arquitectura:** `docs/ARCHITECTURE.md`
- **API:** `docs/API.md`
- **Deployment:** `DEPLOYMENT.md`

---

## ✅ Checklist de Entrenamiento

- [ ] Dataset generado (`data/processed/`)
- [ ] Detector entrenado (`models/detector.ckpt`)
- [ ] CI Tester entrenado (`models/ci_tester.ckpt`)
- [ ] F1-Score > 0.95
- [ ] CI Accuracy > 0.85
- [ ] Demo funciona sin errores
- [ ] TensorBoard logs generados

¡Una vez completado, tu sistema CausalDefend está listo para detectar APTs! 🎉
