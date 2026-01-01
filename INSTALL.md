# Kinetra Installation Guide

**CRITICAL**: This is a production financial trading system. All dependencies are LOCKED to specific versions to prevent failures that could lead to financial losses.

## System Requirements

- Python 3.10, 3.11, or 3.12 (3.13+ not supported)
- 16GB+ RAM recommended
- GPU with ROCm 6.2+ (AMD) or CUDA 12.4+ (NVIDIA) for optimal performance
- Linux (Ubuntu 22.04+ or similar) recommended for production

## Installation Methods

### Method 1: pip (Recommended for Production)

#### 1. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

#### 3. Install PyTorch (Choose your GPU)

**AMD ROCm (Recommended for this system):**
```bash
pip install torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 \
    --index-url https://download.pytorch.org/whl/rocm6.2
```

**NVIDIA CUDA:**
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

**CPU Only (NOT recommended for production):**
```bash
pip install torch==2.5.1 torchvision==0.20.1
```

#### 4. Install Kinetra Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'ROCm Available: {torch.version.hip is not None}')"
python -c "import numba; print(f'Numba: {numba.__version__}')"
python -c "import metaapi_cloud_sdk; print('MetaAPI: OK')"
```

### Method 2: Poetry

#### 1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### 2. Install Dependencies
```bash
poetry install
```

#### 3. Install PyTorch (in Poetry shell)
```bash
poetry shell

# AMD ROCm
pip install torch==2.5.1+rocm6.2 torchvision==0.20.1+rocm6.2 \
    --index-url https://download.pytorch.org/whl/rocm6.2

# OR NVIDIA CUDA
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

## Dependency Lock Philosophy

**WHY LOCKED VERSIONS?**

This is a FINANCIAL TRADING SYSTEM. Unpredictable behavior from dependency updates could cause:
- Trading logic bugs leading to financial losses
- Monte Carlo test failures masking real issues
- Non-deterministic backtest results
- Production crashes during live trading

**All versions are locked using `==` not `>=`**

## Critical Dependencies

### Core (Must be exact versions)
- **numpy==1.26.4** - Financial calculations must be deterministic
- **pandas==2.2.3** - Data integrity for market data
- **torch==2.5.1** - RL agent behavior must be reproducible
- **numba==0.60.0** - JIT compilation for performance-critical code
- **metaapi-cloud-sdk==31.1.0** - PRIMARY MT5 connection

### RL Framework (Must be exact versions)
- **gymnasium==1.0.0** - Environment behavior must be consistent
- **stable-baselines3==2.7.1** - Agent training must be reproducible

### Performance (Must be exact versions)
- **numba==0.60.0** - Physics calculations JIT compiled
- **llvmlite==0.43.0** - Numba dependency

## Updating Dependencies

**DO NOT update dependencies without:**

1. Full test suite passing (100% tests)
2. Monte Carlo validation passing
3. End-to-end integration tests passing
4. Backtest result comparison (before/after)
5. Documentation of why update is necessary
6. Version locked in pyproject.toml AND requirements.txt

## Verification Checklist

After installation, run:

```bash
# 1. Check all imports work
python scripts/testing/test_imports.py

# 2. Run unit tests
pytest tests/ -v

# 3. Run integration test
python scripts/testing/test_p0_p5_integration.py

# 4. Verify GPU availability
python -c "import torch; print(torch.cuda.is_available() or (torch.version.hip is not None))"

# 5. Test MetaAPI connection (if configured)
python scripts/testing/test_metaapi_auth.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'numba'"
```bash
pip install numba==0.60.0 llvmlite==0.43.0
```

### "ModuleNotFoundError: No module named 'metaapi_cloud_sdk'"
```bash
pip install metaapi-cloud-sdk==31.1.0 inquirer==3.4.0
```

### PyTorch not detecting GPU
```bash
# Check ROCm
rocm-smi

# Check CUDA
nvidia-smi

# Reinstall correct PyTorch version (see step 3 above)
```

### Poetry lock file conflicts
```bash
poetry lock --no-update
poetry install
```

## Production Deployment

For production deployment:

1. **Use Docker** - Pin base image to exact Python version
2. **Copy exact requirements.txt** - No `>=` operators
3. **Disable auto-updates** - No `pip install -U`
4. **Verify checksums** - Use `pip hash` for critical packages
5. **Test in staging** - Identical environment to production
6. **Monitor versions** - Alert on any dependency changes

## Support

If installation fails:
1. Check Python version: `python --version` (must be 3.10-3.12)
2. Check pip version: `pip --version` (must be 23.0+)
3. Check virtual environment is active
4. Check GPU drivers (ROCm/CUDA versions)
5. Review error messages for specific missing packages
6. Create issue at: https://github.com/sandman9988/Kinetra/issues
