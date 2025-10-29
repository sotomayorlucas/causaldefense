# 📚 Índice de Documentación: Requirements y Datasets

## 🎯 ¿Por dónde empezar?

### Para Usar Datos Reales (RECOMENDADO)
1. **Inicio Rápido**: [QUICKSTART_DATASETS.md](QUICKSTART_DATASETS.md)
   - Instrucciones paso a paso para StreamSpot
   - Comandos listos para copiar/pegar
   - ~1 hora de setup

2. **Guía Completa**: [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)
   - 5 datasets públicos documentados
   - Benchmarks del paper
   - Instrucciones detalladas

### Para Entender Todo el Setup
3. **Resumen de Configuración**: [DATASETS_SETUP_SUMMARY.md](DATASETS_SETUP_SUMMARY.md)
   - Vista general completa
   - Flujo de trabajo end-to-end
   - Referencias y troubleshooting

---

## 📦 Archivos de Configuración

### Requirements
- **`requirements-optimized.txt`** - ⭐ USAR ESTE
  - Organizado por categorías
  - Versiones específicas
  - Comentarios explicativos
  
- **`requirements.txt`** - Original
  - Muchas dependencias comentadas
  - Puede tener conflictos

- **`requirements-minimal.txt`** - Mínimo
  - Solo lo esencial
  - Para pruebas rápidas

### Instalación
```powershell
pip install -r requirements-optimized.txt
```

---

## 📊 Documentación de Datasets

### Guías Principales
| Archivo | Propósito | Nivel |
|---------|-----------|-------|
| [QUICKSTART_DATASETS.md](QUICKSTART_DATASETS.md) | Inicio rápido StreamSpot | Principiante |
| [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md) | Guía completa de datasets | Intermedio |
| [DATASETS_GUIDE.md](DATASETS_GUIDE.md) | Documentación técnica | Avanzado |
| [DATASETS_STATUS.md](DATASETS_STATUS.md) | Estado de implementación | Referencia |

### Resúmenes
| Archivo | Contenido |
|---------|-----------|
| [DATASETS_SETUP_SUMMARY.md](DATASETS_SETUP_SUMMARY.md) | Resumen completo de setup |
| [../status/ENTRENAMIENTO_COMPLETADO.md](../status/ENTRENAMIENTO_COMPLETADO.md) | Estado del entrenamiento |
| [REFERENCES.md](REFERENCES.md) | Bibliografía y papers |

---

## 🔧 Scripts Disponibles

### Datasets
| Script | Función | Ubicación |
|--------|---------|-----------|
| `download_streamspot.py` | Descarga StreamSpot | `scripts/` |
| `import_external_dataset.py` | Importa datasets externos | `scripts/` |
| `import_local_dataset.py` | Importa datasets locales | `scripts/` |
| `split_dataset.py` | Divide train/val/test | `scripts/` |
| `prepare_dataset_simple.py` | Genera datos sintéticos | `scripts/` |

### Entrenamiento
| Script | Función | Ubicación |
|--------|---------|-----------|
| `train_detector.py` | Entrena detector APT | `scripts/` |
| `train_ci_tester.py` | Entrena CI tester | `scripts/` |
| `train_all.py` | Pipeline completo | `scripts/` |

### Evaluación
| Script | Función | Ubicación |
|--------|---------|-----------|
| `test_detector_advanced.py` | Evaluación completa | `examples/` |
| `dashboard.py` | Dashboard de resultados | `examples/` |
| `demo_basico.py` | Demo funcional | `examples/` |

---

## 📊 Datasets Disponibles

### Públicos (Acceso Directo)

#### 1. StreamSpot ⭐ RECOMENDADO
- **Tamaño**: ~500 MB
- **Grafos**: ~500 escenarios
- **F1-Score**: 0.905
- **URL**: https://github.com/sbustreamspot/sbustreamspot-data
- **Inicio Rápido**: [QUICKSTART_DATASETS.md](QUICKSTART_DATASETS.md)

#### 2. LANL
- **Tamaño**: ~40 GB
- **Contenido**: 90 días de logs
- **URL**: https://csr.lanl.gov/data/cyber1/

#### 3. CICIDS 2017/2018
- **Tamaño**: ~7 GB
- **Formato**: PCAP + CSV
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html

### Requieren Registro

#### 4. DARPA TC E3
- **Tamaño**: ~100 GB
- **F1-Score**: 0.982
- **Acceso**: Registro en LDC
- **URL**: https://catalog.ldc.upenn.edu/LDC2018T23

#### 5. DARPA OpTC
- **Tamaño**: ~50 GB
- **F1-Score**: 0.971
- **Muestra**: GitHub público
- **URL**: https://github.com/FiveDirections/OpTC-data

---

## 🚀 Flujos de Trabajo

### Opción 1: Datos Sintéticos (Rápido)
```powershell
# 1. Generar datos
python scripts\prepare_dataset_simple.py --num-graphs 200

# 2. Entrenar
python scripts\train_detector.py --epochs 10

# 3. Probar
python examples\demo_basico.py
```
**Tiempo**: ~10 minutos

### Opción 2: StreamSpot (Datos Reales)
```powershell
# 1. Descargar
cd data\external
git clone https://github.com/sbustreamspot/sbustreamspot-data.git streamspot
cd ..\..

# 2. Importar
python scripts\import_external_dataset.py --dataset streamspot --output data\processed\streamspot

# 3. Dividir
python scripts\split_dataset.py --input data\processed\streamspot --output data\processed\streamspot_split

# 4. Entrenar
python scripts\train_detector.py --data data\processed\streamspot_split --epochs 20

# 5. Evaluar
python examples\test_detector_advanced.py --checkpoint models\streamspot_detector.ckpt
```
**Tiempo**: ~2 horas (descarga + procesamiento + entrenamiento)

### Opción 3: DARPA TC (Paper Completo)
Ver: [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md) - Sección DARPA TC

---

## 📈 Benchmarks del Paper

| Dataset | F1-Score | Precision | Recall |
|---------|----------|-----------|--------|
| DARPA TC E3 | **0.982** | 0.985 | 0.979 |
| DARPA OpTC | **0.971** | 0.975 | 0.967 |
| StreamSpot | **0.905** | 0.920 | 0.890 |

---

## ❓ FAQ

### ¿Qué dataset usar primero?
**StreamSpot** - Es público, pequeño (~500 MB), y tiene buen F1-Score (0.905).

### ¿Necesito DARPA TC?
No para empezar. StreamSpot es suficiente para validar el sistema.

### ¿Cómo reproduzco los resultados del paper?
1. Usar StreamSpot: F1 ≥ 0.90
2. Usar DARPA TC (si tienes acceso): F1 ≥ 0.98

### ¿Puedo usar mis propios datos?
Sí, ver: `scripts/import_local_dataset.py`

---

## 🆘 Problemas Comunes

| Problema | Solución |
|----------|----------|
| Git no encontrado | Descargar ZIP manualmente |
| Out of memory | Usar `--max-graphs 20` |
| Parsing failed | Ver [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md) - Troubleshooting |
| Import error | `pip install -r requirements-optimized.txt` |

---

## 📞 Soporte

1. Ver documentación en este índice
2. Revisar troubleshooting en [EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)
3. Consultar ejemplos en `examples/`
4. Revisar logs de errores

---

## ✅ Checklist Rápido

### Setup Inicial
- [ ] Instalar requirements: `pip install -r requirements-optimized.txt`
- [ ] Verificar instalación: `python examples\demo_basico.py`

### Datasets Reales
- [ ] Descargar StreamSpot
- [ ] Importar con `import_external_dataset.py`
- [ ] Dividir con `split_dataset.py`
- [ ] Entrenar con `train_detector.py`
- [ ] Evaluar y comparar con benchmarks

### Opcional (Para Paper Completo)
- [ ] Solicitar acceso a DARPA TC
- [ ] Implementar parser CDM
- [ ] Reproducir F1 ≥ 0.98

---

**Última actualización**: 29 de octubre de 2025  
**Versión**: 1.0  
**Estado**: ✅ Completo y funcional
