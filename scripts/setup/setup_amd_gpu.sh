#!/bin/bash
# ============================================================
# Kinetra AMD GPU Setup for Pop!_OS / Ubuntu
# AMD Radeon 7700 XT (RDNA 3)
# ============================================================

set -e

echo "========================================"
echo "KINETRA AMD GPU SETUP - Pop!_OS/Ubuntu"
echo "========================================"

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Step 1: Install ROCm
echo ""
echo "[1/5] Installing ROCm for AMD GPU..."
echo "This requires sudo access."

# For Pop!_OS/Ubuntu 22.04+
if ! command -v rocminfo &> /dev/null; then
    echo "Installing ROCm packages..."

    # Install prerequisites
    sudo apt update
    sudo apt install -y wget gnupg2

    # Add AMD ROCm repository for Ubuntu
    # Pop!_OS 22.04 is based on Ubuntu 22.04 (jammy)
    sudo mkdir -p --mode=0755 /etc/apt/keyrings

    # Download and install the repository key
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

    # Add the repo based on Ubuntu version
    UBUNTU_VERSION=$(lsb_release -rs)
    if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
        CODENAME="jammy"
    elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        CODENAME="noble"
    else
        CODENAME="jammy"  # Default fallback
    fi

    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 ${CODENAME} main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list

    # Set package priority
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | \
        sudo tee /etc/apt/preferences.d/rocm-pin-600

    # Install ROCm
    sudo apt update
    sudo apt install -y rocm-dev rocm-libs

    # Add user to video and render groups
    sudo usermod -aG video $USER
    sudo usermod -aG render $USER

    echo "ROCm installed. You may need to log out and back in for group changes."
else
    echo "ROCm already installed"
    rocminfo 2>/dev/null | grep "Marketing Name" | head -1 || echo "Checking GPU..."
fi

# Step 2: Create virtual environment
echo ""
echo "[2/5] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created"
else
    echo "Virtual environment exists"
fi

# Step 3: Activate and install PyTorch with ROCm
echo ""
echo "[3/5] Installing PyTorch with ROCm support..."
source .venv/bin/activate

pip install --upgrade pip

# Remove CPU-only PyTorch if present
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch with ROCm 6.2 (for RDNA 3)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Step 4: Install other dependencies
echo ""
echo "[4/5] Installing project dependencies..."
pip install pandas numpy prometheus-client

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying GPU detection..."
python -c "
import torch
print()
print('=' * 50)
print('PYTORCH GPU CHECK')
print('=' * 50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f'GPU: {device_name}')

    # Quick performance test
    print()
    print('Running GPU benchmark...')
    import time
    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'Matrix multiply (100x 4096x4096): {elapsed:.2f}s')
    print()
    print('GPU READY FOR TRAINING!')
else:
    print()
    print('WARNING: GPU not detected!')
    print('Try:')
    print('  1. Log out and log back in')
    print('  2. Run: rocminfo')
    print('  3. Check AMD drivers are installed')
print('=' * 50)
"

echo ""
echo "========================================"
echo "SETUP COMPLETE"
echo "========================================"
echo ""
echo "To train with GPU:"
echo "  cd $PROJECT_ROOT"
echo "  source .venv/bin/activate"
echo "  python scripts/train_berserker.py"
echo ""
