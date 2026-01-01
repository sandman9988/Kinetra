#!/bin/bash
# Kinetra ROCm Setup for AMD Radeon 7700 XT
# Run this on your local machine to enable GPU acceleration

echo "================================================"
echo "KINETRA ROCm Setup for AMD GPU"
echo "================================================"

# Check if ROCm is installed
if ! command -v rocm-smi &> /dev/null; then
    echo ""
    echo "ROCm not found. Please install ROCm first:"
    echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    echo ""
    echo "For Ubuntu/Debian:"
    echo "  wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb"
    echo "  sudo apt install ./amdgpu-install_6.2.60200-1_all.deb"
    echo "  sudo amdgpu-install --usecase=rocm"
    echo ""
    exit 1
fi

echo "ROCm detected!"
rocm-smi --showid

echo ""
echo "Installing PyTorch with ROCm support..."

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y 2>/dev/null

# Install PyTorch with ROCm 6.2 (for RX 7700 XT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: ROCm not detected by PyTorch')
    print('Make sure ROCm is properly installed and your GPU is supported')
"

echo ""
echo "================================================"
echo "Setup complete!"
echo "Run: python scripts/train_with_metrics.py"
echo "View: http://localhost:8000/metrics"
echo "Or open: monitoring/dashboard.html"
echo "================================================"
