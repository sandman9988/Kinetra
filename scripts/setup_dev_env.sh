#!/bin/bash
# Kinetra Development Environment Setup for Pop!_OS
# Run with: chmod +x setup_dev_env.sh && ./setup_dev_env.sh

set -e

echo "=========================================="
echo "  Kinetra Dev Environment Setup"
echo "  Pop!_OS / Ubuntu 22.04+"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ==========================================
# STEP 1: System Update & Base Dependencies
# ==========================================
echo ""
echo "Step 1: Updating system and installing base dependencies..."

sudo apt update && sudo apt upgrade -y

# Essential build tools
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

print_status "Base dependencies installed"

# ==========================================
# STEP 2: Python Development Environment
# ==========================================
echo ""
echo "Step 2: Setting up Python development environment..."

# Install Python 3.11+ and dev tools
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-setuptools \
    python3-wheel

# Verify Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Python $PYTHON_VERSION installed"

# ==========================================
# STEP 3: Create Virtual Environment
# ==========================================
echo ""
echo "Step 3: Creating Python virtual environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv .venv
    print_status "Virtual environment created at $VENV_DIR"
else
    print_warning "Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

print_status "Virtual environment activated and pip upgraded"

# ==========================================
# STEP 4: Install Project Dependencies
# ==========================================
echo ""
echo "Step 4: Installing project dependencies..."

# Install PyTorch with CUDA support (if available)
# Detect if NVIDIA GPU is present
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project requirements
pip install -r requirements.txt

print_status "Project dependencies installed"

# ==========================================
# STEP 5: Install Additional Dev Tools
# ==========================================
echo ""
echo "Step 5: Installing additional development tools..."

pip install \
    ipython \
    jupyter \
    jupyterlab \
    notebook

print_status "Development tools installed"

# ==========================================
# STEP 6: Verify Installation
# ==========================================
echo ""
echo "Step 6: Verifying installation..."

python -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import pandas as pd
print(f'Pandas: {pd.__version__}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

# Test kinetra import
import kinetra
print('Kinetra: Import successful')
"

print_status "All packages verified"

# ==========================================
# DONE
# ==========================================
echo ""
echo "=========================================="
echo -e "${GREEN}  Development environment ready!${NC}"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "Next: Run ./scripts/setup_mt5_wine.sh to install MetaTrader 5"
echo ""
