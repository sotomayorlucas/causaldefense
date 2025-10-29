# 🚀 Inicio Rápido: Datasets Externos para CausalDefend

## ⚡ Opción Más Rápida: StreamSpot

### Paso 1: Instalar dependencias adicionales
```powershell
pip install requests beautifulsoup4
```

### Paso 2: Descargar StreamSpot manualmente

**Opción A - Usando Git** (Recomendado):
```powershell
cd data\external
git clone https://github.com/sbustreamspot/sbustreamspot-data.git streamspot
cd ..\..
```

**Opción B - Descarga Manual**:
1. Ir a: https://github.com/sbustreamspot/sbustreamspot-data
2. Click en "Code" → "Download ZIP"
3. Extraer en `data\external\streamspot\`

**Opción C - Script Automático**:
```powershell
python scripts\download_streamspot.py
```

### Paso 3: Verificar la descarga
```powershell
ls data\external\streamspot\
# Deberías ver: sbustreamspot-data-master\ o archivos .txt
```

### Paso 4: Importar al formato de CausalDefend
```powershell
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --output data\processed\streamspot `
  --max-graphs 50
```

### Paso 5: Dividir en train/val/test
```powershell
python scripts\split_dataset.py `
  --input data\processed\streamspot `
  --output data\processed\streamspot_split
```

### Paso 6: Entrenar
```powershell
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 10 `
  --batch-size 16 `
  --output models\streamspot_detector.ckpt
```

### Paso 7: Evaluar
```powershell
python examples\test_detector_advanced.py `
  --checkpoint models\streamspot_detector.ckpt `
  --data data\processed\streamspot_split\test
```

---

## 📊 Resultados Esperados

Según el paper de CausalDefend:
- **Precision**: ~0.92
- **Recall**: ~0.89
- **F1-Score**: ~0.905
- **FPR**: <0.015

---

## 🔍 Ver Más Datasets

Para otros datasets (DARPA TC, OpTC, LANL, etc.), ver:
- **Guía completa**: [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)
- **Documentación existente**: [DATASETS_GUIDE.md](DATASETS_GUIDE.md)

---

## ⚠️ Troubleshooting

### Error: "Git no encontrado"
**Solución**: Usar Opción B (descarga manual) o instalar Git desde https://git-scm.com/

### Error: "Module not found"
**Solución**: 
```powershell
pip install -r requirements-optimized.txt
```

### Error: "Out of memory"
**Solución**: Reducir número de grafos
```powershell
python scripts\import_external_dataset.py --max-graphs 20
```

---

## 📁 Estructura de Datos Esperada

Después de importar, deberías tener:
```
data/
├── external/
│   └── streamspot/
│       └── sbustreamspot-data-master/
│           ├── graph1.txt
│           ├── graph2.txt
│           └── ...
└── processed/
    ├── streamspot/
    │   ├── graph_0.pkl
    │   ├── features_0.npy
    │   ├── label_0.json
    │   └── ...
    └── streamspot_split/
        ├── train/
        ├── val/
        └── test/
```

---

## 🎯 Comandos en Una Línea

```powershell
# Todo en secuencia (después de descargar manualmente)
cd data\external; git clone https://github.com/sbustreamspot/sbustreamspot-data.git streamspot; cd ..\..;
python scripts\import_external_dataset.py --dataset streamspot --output data\processed\streamspot --max-graphs 50;
python scripts\split_dataset.py --input data\processed\streamspot --output data\processed\streamspot_split;
python scripts\train_detector.py --data data\processed\streamspot_split --epochs 10 --output models\streamspot_detector.ckpt
```

---

**¡Listo para entrenar con datos reales! 🎉**
