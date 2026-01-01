#!/usr/bin/env python3
"""GPU Detection Script for ROCm/CUDA"""

import sys

print("=" * 60)
print("GPU DETECTION (PyTorch ROCm/CUDA)")
print("=" * 60)

try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")

    # CUDA/ROCm availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")

        # Memory info
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)
        print(f"Total memory: {total_mem:.1f} GB")

        # ROCm detection
        config = torch.__config__.show()
        is_rocm = "rocm" in config.lower() or "hip" in config.lower()
        print(f"ROCm backend: {is_rocm}")

        # Quick compute test
        print("\nRunning quick GPU compute test...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"  Matrix multiply (1000x1000): OK")

        # Physics-like batch test
        prices = torch.randn(32, 10000, device="cuda")  # 32 instruments, 10k bars
        velocity = torch.diff(prices, dim=1)
        momentum = velocity * prices[:, 1:]
        torch.cuda.synchronize()
        print(f"  Physics batch (32x10000): OK")

        print("\n[OK] GPU ready for physics computation!")
    else:
        print("\n[WARN] No GPU available - will use CPU fallback")
        print("  Check ROCm/CUDA installation")

except ImportError as e:
    print(f"\n[ERROR] PyTorch not installed: {e}")
    print("  Install with: pip install torch")
    sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] GPU test failed: {e}")
    sys.exit(1)

print("=" * 60)
