# 📊 Guía Práctica: Datasets Externos para CausalDefend

Esta guía te muestra cómo obtener y usar datasets reales para entrenar y evaluar CausalDefend, reproduciendo los resultados del paper.

---

## 🎯 Datasets Públicos Disponibles

### 1. **StreamSpot** (⭐ RECOMENDADO PARA EMPEZAR)

**Descripción**: Dataset de grafos de proveniencia para detección de APTs  
**Fuente**: Stony Brook University  
**Tamaño**: ~500 MB  
**Grafos**: ~500 escenarios  
**Formato**: Archivos `.txt` con listas de aristas  
**Acceso**: ✅ Público (GitHub)

**Tipos de Ataque**:
- Drive-by download
- Clickbait
- Iframe injection  
- Browser-based exploits

**URLs**:
- GitHub: https://github.com/sbustreamspot/sbustreamspot-data
- Paper: https://dl.acm.org/doi/10.1145/3029806.3029825

---

### 2. **DARPA Transparent Computing E3** (Engagement 3)

**Descripción**: Grafos de proveniencia de ataques APT simulados  
**Fuente**: DARPA I2O Program  
**Tamaño**: ~100 GB (completo), ~5 GB (muestra)  
**Formato**: JSON (Common Data Model - CDM)  
**Acceso**: ⚠️ Requiere registro

**Sistemas Monitoreados**:
- TRACE (Linux)
- CADETS (FreeBSD)
- THEIA (Linux)
- ClearScope (Android)
- FiveDirections (Windows)

**Escenarios de Ataque**:
- Initial access (spearphishing)
- Credential dumping
- Lateral movement
- Data exfiltration
- Persistence mechanisms

**URLs**:
- GitHub (samples): https://github.com/darpa-i2o/Transparent-Computing
- Catálogo LDC: https://catalog.ldc.upenn.edu/LDC2018T23
- Paper: https://ieeexplore.ieee.org/document/8835218

---

### 3. **DARPA OpTC** (Operational Transparent Computing)

**Descripción**: Dataset de operaciones en entorno empresarial  
**Fuente**: FiveDirections  
**Tamaño**: ~50 GB  
**Formato**: JSON (CDM)  
**Acceso**: ✅ Muestra pública en GitHub

**Características**:
- Entorno Windows empresarial
- Tráfico real de usuarios
- Ataques APT insertados
- 5 escenarios completos

**URLs**:
- GitHub: https://github.com/FiveDirections/OpTC-data
- Documentation: https://github.com/FiveDirections/OpTC-data/wiki

---

### 4. **LANL Unified Host and Network Dataset**

**Descripción**: Logs de red y host de Los Alamos National Lab  
**Fuente**: Los Alamos National Laboratory  
**Tamaño**: ~40 GB  
**Formato**: CSV (eventos de autenticación, red, procesos)  
**Acceso**: ✅ Público

**Contenido**:
- 90 días de actividad
- ~1.6B eventos
- Red events, authentication, processes
- Etiquetas de comportamiento anómalo

**URLs**:
- Dataset: https://csr.lanl.gov/data/cyber1/
- Paper: https://arxiv.org/abs/1708.07518

---

### 5. **CICIDS 2017/2018** (Canadian Institute for Cybersecurity)

**Descripción**: Tráfico de red con ataques etiquetados  
**Fuente**: University of New Brunswick  
**Tamaño**: ~7 GB  
**Formato**: PCAP + CSV  
**Acceso**: ✅ Público

**Ataques Incluidos**:
- DDoS
- Brute Force
- Web attacks
- Infiltration
- Botnet

**URLs**:
- Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- Paper: https://www.sciencedirect.com/science/article/pii/S2215098617301423

---

## 🚀 Instalación de Dependencias para Datasets

```powershell
# Instalar paquetes necesarios para descarga y procesamiento
pip install requests beautifulsoup4 wget
```

---

## 📥 Opción 1: StreamSpot (Más Fácil)

### Paso 1: Descargar Dataset

```powershell
# Crear directorio
New-Item -ItemType Directory -Force -Path "data\external\streamspot"

# Opción A: Usando git (si tienes git instalado)
cd data\external
git clone https://github.com/sbustreamspot/sbustreamspot-data.git streamspot

# Opción B: Descarga manual
# 1. Ir a: https://github.com/sbustreamspot/sbustreamspot-data
# 2. Click en "Code" -> "Download ZIP"
# 3. Extraer en data\external\streamspot\
```

### Paso 2: Importar al Formato de CausalDefend

```powershell
# Si tienes el script implementado
python scripts\import_external_dataset.py `
  --dataset streamspot `
  --input data\external\streamspot `
  --output data\processed\streamspot `
  --max-graphs 100

# Verificar
ls data\processed\streamspot\
# Deberías ver: graph_*.pkl, features_*.npy, label_*.json
```

### Paso 3: Dividir en Train/Val/Test

```powershell
python scripts\split_dataset.py `
  --input data\processed\streamspot `
  --output data\processed\streamspot_split `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --test-ratio 0.15
```

### Paso 4: Entrenar

```powershell
python scripts\train_detector.py `
  --data data\processed\streamspot_split `
  --epochs 20 `
  --batch-size 16 `
  --output models\streamspot_detector.ckpt
```

---

## 📥 Opción 2: DARPA TC E3 (Requiere Registro)

### Paso 1: Solicitar Acceso

1. **Registro en LDC** (Linguistic Data Consortium):
   - URL: https://catalog.ldc.upenn.edu/
   - Crear cuenta (requiere email institucional)
   - Buscar "LDC2018T23"
   - Completar formulario de solicitud

2. **Alternativa - Muestra Pública**:
   ```powershell
   # Clonar repositorio con muestras
   cd data\external
   git clone https://github.com/darpa-i2o/Transparent-Computing.git darpa_tc
   ```

### Paso 2: Descargar Dataset

Después de aprobación, recibirás link de descarga:

```powershell
# Ejemplo (URL específica te la envían por email)
wget https://download.ldc.upenn.edu/LDC2018T23/ta1-trace-e3-official-1.json.tar.gz

# Extraer
tar -xzf ta1-trace-e3-official-1.json.tar.gz -C data\external\darpa_tc\
```

### Paso 3: Parsear Formato CDM

El formato CDM es complejo. Necesitas un parser especializado:

```powershell
# Instalar dependencias adicionales
pip install pycdm  # Si existe, sino hay que implementarlo manualmente

# Parser básico (ejemplo)
python scripts\parse_darpa_cdm.py `
  --input data\external\darpa_tc\ta1-trace-e3-official-1.json `
  --output data\processed\darpa_tc
```

**Nota**: El formato CDM requiere un parser complejo. Considera usar StreamSpot primero.

---

## 📥 Opción 3: DARPA OpTC (Más Accesible que TC)

### Paso 1: Clonar Repositorio

```powershell
cd data\external
git clone https://github.com/FiveDirections/OpTC-data.git optc
```

### Paso 2: Descargar Escenarios

```powershell
# Los datos están en releases
# Ir a: https://github.com/FiveDirections/OpTC-data/releases

# Ejemplo para un escenario
Invoke-WebRequest -Uri "https://github.com/FiveDirections/OpTC-data/releases/download/v1.0/ecar-benign.json.gz" -OutFile "data\external\optc\ecar-benign.json.gz"

# Descomprimir
gzip -d data\external\optc\ecar-benign.json.gz
```

### Paso 3: Parsear (Similar a DARPA TC)

```powershell
python scripts\parse_darpa_cdm.py `
  --input data\external\optc\ecar-benign.json `
  --output data\processed\optc `
  --format cdm
```

---

## 📥 Opción 4: LANL (Para Análisis de Red)

### Paso 1: Descargar

```powershell
# Crear directorio
New-Item -ItemType Directory -Force -Path "data\external\lanl"

# Descargar archivos (ejemplo: authentication logs)
Invoke-WebRequest -Uri "https://csr.lanl.gov/data/cyber1/auth.txt.gz" -OutFile "data\external\lanl\auth.txt.gz"

# Descomprimir
gzip -d data\external\lanl\auth.txt.gz
```

### Paso 2: Convertir a Grafos de Proveniencia

LANL está en formato tabular (CSV), necesitas convertirlo a grafos:

```powershell
python scripts\lanl_to_provenance.py `
  --input data\external\lanl\auth.txt `
  --output data\processed\lanl `
  --window-size 3600  # 1 hora
```

---

## 📊 Resultados Esperados (Benchmarks del Paper)

### CausalDefend Performance

| Dataset | Precision | Recall | F1-Score | FPR | Grafos |
|---------|-----------|--------|----------|-----|--------|
| **DARPA TC E3** | 0.985 | 0.979 | **0.982** | 0.001 | 150 |
| **DARPA OpTC** | 0.975 | 0.967 | **0.971** | 0.002 | 100 |
| **StreamSpot** | 0.920 | 0.890 | **0.905** | 0.015 | 500 |

### Comparación con Baselines

| Método | StreamSpot F1 | DARPA TC F1 |
|--------|---------------|-------------|
| **CausalDefend (Ours)** | **0.905** | **0.982** |
| StreamSpot (Original) | 0.890 | - |
| Unicorn | 0.875 | 0.920 |
| SLEUTH | 0.850 | 0.895 |
| Poirot | - | 0.910 |

---

## 🔧 Scripts Necesarios

### 1. `scripts/import_external_dataset.py`

```python
"""
Script para importar datasets externos.
Soporta: StreamSpot, DARPA TC, OpTC
"""
# Ver implementación en DATASETS_GUIDE.md
```

### 2. `scripts/split_dataset.py`

```python
"""
Divide dataset en train/val/test con estratificación.
"""
# Ver implementación existente
```

### 3. `scripts/parse_darpa_cdm.py` (Por implementar)

```python
"""
Parser para formato CDM de DARPA.
Convierte JSON CDM → NetworkX graphs.
"""
# Pendiente de implementación
```

### 4. `scripts/lanl_to_provenance.py` (Por implementar)

```python
"""
Convierte logs LANL (CSV) a grafos de proveniencia.
"""
# Pendiente de implementación
```

---

## ✅ Checklist Rápido

### Para Reproducir Resultados del Paper:

- [ ] **Instalar dependencias**: `pip install -r requirements-optimized.txt`
- [ ] **Descargar StreamSpot**: `git clone https://github.com/sbustreamspot/sbustreamspot-data.git`
- [ ] **Importar dataset**: `python scripts/import_external_dataset.py --dataset streamspot`
- [ ] **Dividir datos**: `python scripts/split_dataset.py --input ... --output ...`
- [ ] **Entrenar modelo**: `python scripts/train_detector.py --data ... --epochs 50`
- [ ] **Evaluar**: `python examples/test_detector_advanced.py --checkpoint ...`
- [ ] **Comparar F1-Score**: Debe ser ≥ 0.90

### Para DARPA TC (Opcional):

- [ ] Solicitar acceso en LDC (requiere afiliación)
- [ ] Descargar dataset (~100 GB)
- [ ] Implementar/usar parser CDM
- [ ] Seguir mismo flujo: import → split → train → eval

---

## 🆘 Troubleshooting

### Error: "Git no encontrado"

**Solución**:
```powershell
# Descargar ZIP manualmente desde GitHub
# O instalar Git: https://git-scm.com/download/win
```

### Error: "Out of Memory"

**Solución**:
```powershell
# Usar --max-graphs para limitar
python scripts\import_external_dataset.py --max-graphs 50
```

### Error: "CDM parsing failed"

**Solución**: El formato CDM es complejo. Considera:
1. Usar StreamSpot primero (más simple)
2. Implementar parser CDM personalizado
3. Buscar herramientas existentes: https://github.com/prov-suite/provone

---

## 📚 Referencias

### Papers Originales

1. **StreamSpot**:
   - "StreamSpot: Detecting Anomalous Patterns in System Event Streams" (CCS 2017)
   - https://dl.acm.org/doi/10.1145/3029806.3029825

2. **DARPA TC**:
   - "Transparent Computing: The Key to Big Data Security" (IEEE S&P 2018)
   - https://ieeexplore.ieee.org/document/8835218

3. **CausalDefend** (este proyecto):
   - "Explainable APT Detection via Causal Graph Neural Networks" (2023)

### Herramientas Útiles

- **PyG (PyTorch Geometric)**: https://pytorch-geometric.readthedocs.io/
- **NetworkX**: https://networkx.org/
- **Causal-learn**: https://causal-learn.readthedocs.io/

---

## 🎯 Próximos Pasos

1. **Empezar con StreamSpot** (más accesible)
2. **Validar pipeline completo** end-to-end
3. **Comparar con benchmarks** del paper
4. **Fine-tuning** de hiperparámetros
5. **Solicitar DARPA TC** si quieres resultados completos

---

**¡Listo para usar datasets reales! 🚀**

Comienza con:
```powershell
cd data\external
git clone https://github.com/sbustreamspot/sbustreamspot-data.git streamspot
python scripts\import_external_dataset.py --dataset streamspot
```
