# Setup Scripts

Environment setup and configuration scripts for various platforms.

## Development Environment

- **`setup_dev_env.sh`** - Complete development environment setup
- **`check_gpu.py`** - Check GPU availability and configuration

## GPU Setup

- **`setup_gpu.sh`** - Generic GPU setup
- **`setup_amd_gpu.sh`** - AMD GPU setup
- **`setup_amd_rx7600.sh`** - AMD RX 7600 specific setup
- **`setup_rocm.sh`** - ROCm setup for AMD GPUs

## MT5 Bridge Setup

- **`setup_mt5_wine.sh`** - MT5 via Wine on Linux
- **`setup_mt5_bridge.bat`** - Windows MT5 bridge setup
- **`setup_mt5_bridge.ps1`** - PowerShell MT5 bridge setup

## Automation

- **`setup_weekly_cron.sh`** - Weekly cron job setup

## Quick Start

```bash
# Full development environment setup (recommended)
./scripts/setup/setup_dev_env.sh

# Check GPU availability
python scripts/setup/check_gpu.py

# Setup AMD GPU (if applicable)
./scripts/setup/setup_amd_gpu.sh

# Setup MT5 bridge on Linux
./scripts/setup/setup_mt5_wine.sh
```

## Platform-Specific Notes

### Pop!_OS / Ubuntu
Use `setup_dev_env.sh` for full automated setup.

### AMD GPU Users
1. Run `setup_amd_gpu.sh` or `setup_rocm.sh`
2. For RX 7600 specifically, use `setup_amd_rx7600.sh`
3. Verify with `check_gpu.py`

### Windows MT5 Users
1. Use `setup_mt5_bridge.bat` (Command Prompt)
2. Or `setup_mt5_bridge.ps1` (PowerShell)

### Linux MT5 Users
Use `setup_mt5_wine.sh` to setup MT5 via Wine.
