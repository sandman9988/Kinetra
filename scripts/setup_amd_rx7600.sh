#!/bin/bash
# Kinetra ROCm Setup for AMD Radeon RX 7600
# Optimized for AMD Ryzen 9 5950X (32 threads) + 128GB RAM + RX 7600 (8GB VRAM)

echo "================================================"
echo "KINETRA AMD RX 7600 GPU Setup"
echo "================================================"
echo ""
echo "System specs:"
echo "  CPU: AMD Ryzen 9 5950X (16 cores / 32 threads)"
echo "  RAM: 128GB"
echo "  GPU: AMD Radeon RX 7600 (8GB VRAM)"
echo ""

# Check if ROCm is installed
if ! command -v rocm-smi &> /dev/null; then
    echo "ROCm not found. Installing ROCm 6.2..."
    echo ""

    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    fi

    if [ "$OS" = "ubuntu" ]; then
        echo "Installing ROCm on Ubuntu $VER..."

        # Add ROCm repository
        wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
        sudo apt-get update
        sudo apt-get install -y ./amdgpu-install_6.2.60200-1_all.deb
        rm amdgpu-install_6.2.60200-1_all.deb

        # Install ROCm
        sudo amdgpu-install --usecase=rocm -y

        # Add user to render and video groups
        sudo usermod -a -G render,video $USER

        echo ""
        echo "ROCm installed! Please log out and back in for group changes to take effect."
        echo "Then run this script again."
        exit 0
    else
        echo "Unsupported OS: $OS"
        echo "Please install ROCm manually:"
        echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        exit 1
    fi
fi

# Verify ROCm installation
echo "Checking ROCm installation..."
rocm-smi --showid

# Check GPU
if ! rocm-smi | grep -q "7600"; then
    echo ""
    echo "WARNING: RX 7600 not detected. Checking for compatible GPUs..."
    rocm-smi
fi

echo ""
echo "Installing PyTorch with ROCm support..."

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio -y 2>/dev/null

# Install PyTorch with ROCm 6.2 (RX 7600 is RDNA 3 - gfx1100)
echo "Installing PyTorch for ROCm 6.2 (RDNA 3)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install additional dependencies for GPU-accelerated training
pip install hmmlearn scikit-learn

# Set environment variables for optimal ROCm performance
echo ""
echo "Setting ROCm environment variables..."

# Add to current session
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RDNA 3 architecture
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export GPU_MAX_HW_QUEUES=8
export HIP_VISIBLE_DEVICES=0

# Add to shell profile for persistence
SHELL_RC="$HOME/.bashrc"
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "HSA_OVERRIDE_GFX_VERSION" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# ROCm environment variables for Kinetra" >> "$SHELL_RC"
    echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7600 (RDNA 3)" >> "$SHELL_RC"
    echo "export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128" >> "$SHELL_RC"
    echo "export GPU_MAX_HW_QUEUES=8" >> "$SHELL_RC"
    echo "export HIP_VISIBLE_DEVICES=0" >> "$SHELL_RC"
    echo "Added ROCm variables to $SHELL_RC"
fi

# Verify installation
echo ""
echo "Verifying PyTorch + ROCm installation..."
python3 -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'VRAM: {vram_gb:.1f} GB')
    print(f'PyTorch HIP version: {torch.version.hip}')
    print('')
    print('✅ GPU acceleration ready!')
    print('')
    print('Recommended settings for RX 7600 (8GB VRAM):')
    print('  - Batch size: 512-1024 (RL training)')
    print('  - Physics batch size: 10000-20000 bars')
    print('  - Parallel workers: 30 (leave 2 cores for system)')
else:
    print('')
    print('❌ WARNING: ROCm not detected by PyTorch')
    print('Troubleshooting:')
    print('  1. Verify ROCm installation: rocm-smi')
    print('  2. Check GPU architecture: rocm-smi --showid')
    print('  3. Verify groups: groups (should include render, video)')
    print('  4. May need to log out and back in')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Setup complete!"
    echo "================================================"
    echo ""
    echo "Hardware Configuration:"
    echo "  CPU: 32 threads (30 workers recommended)"
    echo "  RAM: 128GB (use up to 100GB for parallel processing)"
    echo "  GPU: RX 7600 8GB VRAM (ROCm accelerated)"
    echo ""
    echo "Quick Start:"
    echo "  1. Data preparation (parallel): python scripts/prepare_data.py"
    echo "  2. Train with GPU: python scripts/explore_universal.py"
    echo ""
    echo "Performance Tips:"
    echo "  - Data prep uses all 32 threads automatically"
    echo "  - Physics computation uses GPU when available"
    echo "  - RL training uses GPU for neural networks"
    echo ""
else
    echo ""
    echo "Setup failed. Please check error messages above."
    exit 1
fi
