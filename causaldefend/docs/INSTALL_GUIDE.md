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

# 9. Code Quality
pip install black isort
```

## Verificar Instalación

```powershell
# Verificar instalación básica
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

# O ejecutar script de verificación completo
python verify_installation.py
```

## Problemas Comunes y Soluciones

### 1. Error con torch-scatter, torch-sparse, torch-cluster

**Problema**: Estos paquetes necesitan compilación y pueden fallar en Windows.

**Solución**:
```powershell
# Usar wheels precompilados
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
```

### 2. Error con psycopg2-binary

**Problema**: Necesita compiladores C en Windows.

**Solución**:
```powershell
# Usar la versión binary (ya especificada en requirements)
pip install psycopg2-binary
```

### 3. Error con python-jose[cryptography]

**Problema**: Dependencias de cryptography pueden fallar.

**Solución**:
```powershell
# Instalar cryptography primero
pip install cryptography
pip install python-jose[cryptography]
```

### 4. NumPy 2.0 incompatibilidad

**Problema**: Algunos paquetes aún no son compatibles con NumPy 2.0.

**Solución**: Ya está especificado en requirements: `numpy>=1.26.0,<2.0.0`

### 5. Error "Microsoft Visual C++ 14.0 is required"

**Problema**: Algunos paquetes necesitan compiladores en Windows.

**Solución**:
```powershell
# Opción A: Instalar Build Tools
# Descargar: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Opción B: Usar solo paquetes con wheels precompilados
# (requirements-minimal.txt ya está configurado para esto)
```

## Instalación Solo para CPU (Sin GPU)

Si no tienes GPU NVIDIA o quieres instalación más rápida:

```powershell
# Instalar PyTorch CPU-only
pip install torch torchvision torchaudio

# Luego continuar con requirements-minimal.txt
pip install -r requirements-minimal.txt
```

## Actualizar Dependencias

```powershell
# Ver paquetes desactualizados
pip list --outdated

# Actualizar todos los paquetes
pip install --upgrade -r requirements-minimal.txt

# O actualizar paquetes específicos
pip install --upgrade torch pytorch-lightning fastapi
```

## Crear requirements.txt desde entorno

Si instalaste manualmente y quieres guardar las versiones exactas:

```powershell
# Exportar todas las dependencias instaladas
pip freeze > requirements-frozen.txt

# Luego puedes reinstalar exactamente con:
pip install -r requirements-frozen.txt
```

## Desinstalar Todo y Empezar de Nuevo

```powershell
# Desactivar entorno
deactivate

# Eliminar entorno virtual
Remove-Item -Recurse -Force venv

# Crear nuevo entorno
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar de nuevo
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

## Siguiente Paso Después de Instalación

Una vez instaladas las dependencias:

```powershell
# 1. Verificar instalación
python verify_installation.py

# 2. Ejecutar quick start
python quick_start.py

# 3. O ejecutar ejemplo básico
python examples/basic_usage.py
```

## Soporte

Si sigues teniendo problemas:
1. Verifica la versión de Python: `python --version` (debe ser 3.10+)
2. Verifica pip: `pip --version`
3. Revisa los logs de error completos
4. Consulta la documentación de PyTorch: https://pytorch.org/get-started/locally/
