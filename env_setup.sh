#!/bin/bash
# Kinetra Environment Setup
# Source this file: source env_setup.sh

# Set workspace
export KINETRA_WORKSPACE="/workspace"

# Add to PYTHONPATH
export PYTHONPATH="${KINETRA_WORKSPACE}:${PYTHONPATH}"

# Activate virtual environment if it exists
if [ -d "/workspace/.venv" ]; then
    source "/workspace/.venv/bin/activate"
fi

# Add local bin to PATH
export PATH="${KINETRA_WORKSPACE}/scripts:$PATH"

# GPU environment variables (AMD ROCm)
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
export GPU_MAX_HW_QUEUES="8"

# Python optimizations
export PYTHONOPTIMIZE="1"
export PYTHONDONTWRITEBYTECODE="1"

echo "Kinetra environment loaded"
echo "Python: $(which python3)"
echo "Workspace: $KINETRA_WORKSPACE"
