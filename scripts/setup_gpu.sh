#!/bin/bash
# ============================================================
# Kinetra GPU Setup Script for AMD Radeon 7700 XT
# ============================================================
#
# Run this script on your LOCAL machine with the GPU
# (Not in the Claude Code sandbox which has proxy restrictions)
#
# Requirements:
# - Ubuntu 22.04 or 24.04
# - AMD Radeon RX 7000 series (RDNA 3)
# - sudo access for ROCm installation
#
# Usage: ./scripts/setup_gpu.sh
# ============================================================

set -e

echo "========================================"
echo "KINETRA AMD GPU SETUP"
echo "========================================"

# Check if running as root for system packages
if [ "$EUID" -eq 0 ]; then
    echo "Do not run as root - we'll ask for sudo when needed"
    exit 1
fi

# 1. Install ROCm (if not already installed)
echo ""
echo "[1/4] Checking ROCm installation..."
if ! command -v rocm-smi &> /dev/null; then
    echo "ROCm not found. Installing..."

    # Add AMD ROCm repository
    sudo mkdir -p /etc/apt/keyrings
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
    echo 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600

    sudo apt update
    sudo apt install -y rocm-dev

    # Add user to render and video groups
    sudo usermod -aG render $USER
    sudo usermod -aG video $USER

    echo "ROCm installed. You may need to log out and back in."
else
    echo "ROCm already installed:"
    rocm-smi --showproductname 2>/dev/null | head -5 || echo "GPU info unavailable"
fi

# 2. Create/activate virtual environment
echo ""
echo "[2/4] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install PyTorch with ROCm
echo ""
echo "[3/4] Installing PyTorch with ROCm 6.2..."
pip install --upgrade pip

# Uninstall CPU PyTorch if present
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Verify GPU detection
echo ""
echo "[4/4] Verifying GPU detection..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = x @ y
    print('GPU computation test: PASSED')
else:
    print('WARNING: GPU not detected. Check ROCm installation.')
"

# 5. Install other dependencies
echo ""
echo "[5/5] Installing remaining dependencies..."
pip install pandas numpy prometheus-client

echo ""
echo "========================================"
echo "SETUP COMPLETE"
echo "========================================"
echo ""
echo "To use GPU in training:"
echo "  source .venv/bin/activate"
echo "  python scripts/train_berserker.py"
echo ""
echo "If GPU not detected, try:"
echo "  1. Log out and back in (for group membership)"
echo "  2. Reboot if needed"
echo "  3. Check: rocm-smi"
echo ""
