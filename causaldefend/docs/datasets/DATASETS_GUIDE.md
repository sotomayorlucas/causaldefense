# 📊 Guía Completa: Datasets Externos

Esta guía te muestra cómo importar y usar datasets reales para evaluar CausalDefend.

---

## 📑 Tabla de Contenidos

1. [Datasets Públicos Disponibles](#datasets-públicos)
2. [Importar Dataset Externo](#importar-dataset-externo)
3. [Importar Dataset Local/Personalizado](#importar-dataset-local)
4. [Dividir Dataset](#dividir-dataset)
5. [Entrenar con Dataset Real](#entrenar-con-dataset-real)
6. [Benchmarks del Paper](#benchmarks)

---

## 📦 Datasets Públicos Disponibles

### 1. StreamSpot (~500 MB)
- **Descripción**: Dataset de grafos de proveniencia para detección de APTs
- **Fuente**: Stony Brook University
- **Formato**: Archivos `.txt` con listas de aristas
- **Contenido**: ~500 escenarios (benignos y maliciosos)
- **Tipos de Ataque**: Descarga drive-by, clickbait, iframe injection
- **Acceso**: Público (GitHub)

### 2. DARPA Transparent Computing E3 (~100 GB completo)
- **Descripción**: Grafos de proveniencia de ataques simulados
- **Fuente**: DARPA I2O
- **Formato**: JSON (Common Data Model - CDM)
- **Hosts**: TRACE, CADETS, THEIA, ClearScope, FiveDirections
- **Ataques**: Infiltración, exfiltración, lateral movement
- **Acceso**: Requiere registro en DARPA

### 3. DARPA OpTC (~50 GB)
- **Descripción**: Operational Transparent Computing
- **Fuente**: FiveDirections
- **Formato**: JSON (CDM)
- **Escenarios**: Ataques en entorno operacional
- **Acceso**: Muestra pública en GitHub

---

## 🚀 Importar Dataset Externo

### Listar Datasets Disponibles

```powershell
python scripts\import_external_dataset.py --list
```

**Salida esperada**:
```
📦 StreamSpot
   ID: streamspot
   Tamaño: ~500 MB
   ...
```

### Importar StreamSpot (Recomendado para Empezar)

```powershell
# Descargar e importar StreamSpot
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --output data\external\streamspot `
  --max-graphs 100
```

**Parámetros**:
- `--dataset`: ID del dataset (streamspot, darpa_tc_sample, optc_sample)
- `--output`: Directorio de salida
- `--max-graphs`: Limitar número de grafos (opcional)
- `--force`: Re-descargar si ya existe

**Proceso**:
1. ✅ Descarga archivo ZIP (~500 MB)
2. ✅ Extrae archivos `.txt`
3. ✅ Parsea listas de aristas → NetworkX graphs
4. ✅ Genera features de 64 dimensiones
5. ✅ Asigna etiquetas basadas en nombres de archivo
6. ✅ Guarda en formato compatible:
   - `graph_X.pkl` (NetworkX DiGraph)
   - `features_X.npy` (numpy array 64-dim)
   - `label_X.json` (metadata + is_attack)

**Tiempo estimado**: 5-10 minutos

### Importar DARPA TC (Requiere Registro)

```powershell
# NOTA: Primero debes registrarte en DARPA y descargar manualmente
# URL: https://github.com/darpa-i2o/Transparent-Computing

# Opción 1: Usar script (si URL pública disponible)
python scripts\import_external_dataset.py `
  --dataset darpa_tc_sample `
  --output data\external\darpa_tc

# Opción 2: Importar desde descarga manual (ver siguiente sección)
```

---

## 📂 Importar Dataset Local/Personalizado

Si tienes tus propios logs o grafos, puedes importarlos:

### Formato JSON Personalizado

**Crear archivo `my_graph.json`**:
```json
{
  "nodes": [
    {"id": "proc_1", "type": "process", "name": "chrome.exe"},
    {"id": "file_1", "type": "file", "name": "passwords.txt"},
    {"id": "proc_2", "type": "process", "name": "exfil.exe"}
  ],
  "edges": [
    {"source": "proc_1", "target": "file_1", "type": "write"},
    {"source": "proc_2", "target": "file_1", "type": "read"}
  ],
  "metadata": {
    "is_attack": true,
    "attack_type": "data_exfiltration"
  }
}
```

**Importar**:
```powershell
python scripts\import_local_dataset.py `
  --input my_graph.json `
  --output data\processed\custom
```

### Formato CSV (Lista de Aristas)

**Crear archivo `edges.csv`**:
```csv
source,target,edge_type
process_1,file_1,write
process_2,file_1,read
process_2,network_1,connect
```

**Importar**:
```powershell
python scripts\import_local_dataset.py `
  --input edges.csv `
  --output data\processed\custom `
  --is-attack
```

### Importar Directorio Completo

```powershell
# Importar todos los JSON de un directorio
python scripts\import_local_dataset.py `
  --input "C:\mis_grafos\" `
  --output data\processed\custom `
  --pattern "*.json"
```

---

## ✂️ Dividir Dataset

Una vez importado, divide en train/val/test:

```powershell
python scripts\split_dataset.py `
  --input data\external\streamspot `
  --output data\processed\streamspot_split `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --test-ratio 0.15
```

**Parámetros**:
- `--input`: Directorio con grafos importados
- `--output`: Directorio de salida (crea train/val/test/)
- `--train-ratio`: Proporción para entrenamiento (default: 0.7)
- `--val-ratio`: Proporción para validación (default: 0.15)
- `--test-ratio`: Proporción para prueba (default: 0.15)
- `--no-stratify`: No estratificar por clase
- `--seed`: Semilla aleatoria (default: 42)

**Salida**:
```
data\processed\streamspot_split\
├── train\
│   ├── graph_0.pkl
│   ├── features_0.npy
│   ├── label_0.json
│   └── ...
├── val\
│   └── ...
├── test\
│   └── ...
└── split_info.json
```

---

## 🏋️ Entrenar con Dataset Real

### 1. Entrenar Desde Cero

```powershell
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 20 `
  --batch-size 16 `
  --lr 0.0001 `
  --output models\streamspot_detector.ckpt
```

### 2. Fine-tuning (Transfer Learning)

```powershell
# Partir del modelo pre-entrenado con datos sintéticos
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --checkpoint models\detector.ckpt `
  --epochs 10 `
  --lr 0.00001 `
  --output models\streamspot_finetuned.ckpt
```

### 3. Evaluar en Test Set

```powershell
python examples\test_detector_advanced.py `
  --checkpoint models\streamspot_detector.ckpt `
  --data data\processed\streamspot_split\test
```

---

## 📈 Benchmarks del Paper

### Resultados Reportados en CausalDefend (2023)

| Dataset | Precision | Recall | F1-Score | FPR |
|---------|-----------|--------|----------|-----|
| **DARPA TC E3** | 0.985 | 0.979 | 0.982 | 0.001 |
| **DARPA OpTC** | 0.975 | 0.967 | 0.971 | 0.002 |
| **StreamSpot** | 0.920 | 0.890 | 0.905 | 0.015 |

### Cómo Reproducir

```powershell
# 1. Importar dataset
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --output data\external\streamspot

# 2. Dividir
python scripts\split_dataset.py `
  --input data\external\streamspot `
  --output data\processed\streamspot_split

# 3. Entrenar
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 50 `
  --batch-size 32 `
  --output models\streamspot_benchmark.ckpt

# 4. Evaluar
python examples\test_detector_advanced.py `
  --checkpoint models\streamspot_benchmark.ckpt `
  --data data\processed\streamspot_split\test

# 5. Ver resultados
python examples\dashboard.py
```

**Métricas esperadas** (según paper):
- F1-Score: ~0.90
- Precision: ~0.92
- Recall: ~0.89
- FPR: <0.02

---

## 🔗 URLs de Datasets

### Públicos (Acceso Directo)

**StreamSpot**:
- GitHub: https://github.com/sbustreamspot/sbustreamspot-data
- Descarga directa: https://github.com/sbustreamspot/sbustreamspot-data/archive/refs/heads/master.zip

### Requieren Registro

**DARPA TC**:
- Repositorio: https://github.com/darpa-i2o/Transparent-Computing
- Datos completos: https://catalog.ldc.upenn.edu/LDC2018T23
- Registro: Requiere afiliación académica o gubernamental

**DARPA OpTC**:
- GitHub: https://github.com/FiveDirections/OpTC-data
- Muestra pública disponible

---

## 🛠️ Troubleshooting

### Error: "URL not accessible"

**Problema**: Algunos datasets requieren autenticación.

**Solución**:
```powershell
# Descargar manualmente y luego importar con script local
python scripts\import_local_dataset.py `
  --input "C:\Downloads\darpa_tc_data\" `
  --output data\external\darpa_tc
```

### Error: "Parsing failed"

**Problema**: Formato de datos no reconocido.

**Solución**: Verificar formato esperado y adaptar parser en `scripts/import_external_dataset.py`:

```python
def parse_custom_format(self, file_path: Path) -> nx.DiGraph:
    # Implementar parser personalizado
    G = nx.DiGraph()
    # ... parsing logic
    return G
```

### Error: "Out of memory"

**Problema**: Dataset muy grande.

**Solución**: Usar `--max-graphs` para limitar:

```powershell
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --max-graphs 50 `
  --output data\external\streamspot_sample
```

---

## ✅ Checklist Completo

- [ ] Listar datasets disponibles
- [ ] Importar StreamSpot (dataset más accesible)
- [ ] Verificar grafos importados (graph_*.pkl, features_*.npy)
- [ ] Dividir en train/val/test (70/15/15)
- [ ] Entrenar modelo con datos reales
- [ ] Evaluar en test set
- [ ] Comparar con benchmarks del paper
- [ ] (Opcional) Importar DARPA TC si tienes acceso
- [ ] (Opcional) Importar datasets locales personalizados

---

## 📝 Próximos Pasos

1. **Empezar con StreamSpot**: Es público y fácil de obtener
2. **Validar pipeline completo**: Importar → Dividir → Entrenar → Evaluar
3. **Comparar rendimiento**: Sintético vs. Real
4. **Fine-tuning**: Ajustar hiperparámetros para cada dataset
5. **Solicitar acceso a DARPA TC**: Si quieres reproducir resultados completos del paper

---

## 🆘 Soporte

Si encuentras problemas:
1. Verifica logs en consola
2. Revisa `split_info.json` para estadísticas
3. Consulta documentación del dataset original
4. Abre un issue con detalles del error

---

**¡Listo para usar datasets reales! 🎉**

Comienza con:
```powershell
python scripts\import_external_dataset.py --list
```
