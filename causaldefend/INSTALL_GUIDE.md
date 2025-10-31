# Guía de Instalación de CausalDefend

## Problema Común: Versiones de PyTorch

Si recibes el error `ERROR: Could not find a version that satisfies the requirement torch==2.1.0`, es porque las versiones antiguas ya no están disponibles en PyPI.

## Solución: Instalación Paso a Paso

### Opción 1: Instalación Mínima (Recomendada para empezar)

```powershell
# 1. Asegúrate de estar en el entorno virtual
cd C:\Users\lsotomayor\Desktop\causaldefense\causaldefend
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Actualiza pip
python -m pip install --upgrade pip setuptools wheel

# 3. Instala solo las dependencias esenciales
pip install -r requirements-minimal.txt
```

### Opción 2: Instalación Completa

```powershell
# 1. Activa el entorno virtual
.\venv\Scripts\Activate.ps1

# 2. Actualiza pip
python -m pip install --upgrade pip setuptools wheel

# 3. Instala PyTorch primero (separado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instala PyTorch Geometric y extensiones
pip install torch-geometric

# 5. Instala el resto de dependencias
pip install -r requirements.txt
```

### Opción 3: Instalación por Categorías

Si tienes problemas, instala por categorías:

```powershell
# Activa el entorno
.\venv\Scripts\Activate.ps1

# 1. Core ML (PyTorch)
pip install torch torchvision pytorch-lightning

# 2. Graph Neural Networks
pip install torch-geometric networkx

# 3. Causal Discovery
pip install causal-learn pgmpy

# 4. API Framework
pip install fastapi uvicorn pydantic

# 5. Utilities
pip install numpy pandas scikit-learn scipy
pip install pyyaml jinja2 tqdm rich click

# 6. Security & Auth
pip install python-jose passlib pyjwt cryptography

# 7. Database & Cache
pip install sqlalchemy redis celery

# 8. Testing
pip install pytest pytest-asyncio
## 🚨 Documento movido

Esta guía de instalación ahora vive en `docs/INSTALL_GUIDE.md` junto al resto de la documentación.

👉 Abre la versión actualizada con:

```powershell
type docs\INSTALL_GUIDE.md
```

o visítala en tu editor para ver los pasos más recientes.
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
