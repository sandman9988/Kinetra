#!/usr/bin/env python3
"""
GPU-Accelerated RL Training

Trains across multiple instruments/timeframes.
Uses ROCm (AMD) or CUDA (NVIDIA) for acceleration.

Let RL discover:
- Fat candle probability
- Continuation vs reversal
- Optimal exit timing

NO hardcoded rules - pure feature learning.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.rl_gpu_trainer import (
    train_across_instruments,
    TrainingConfig,
    TORCH_AVAILABLE,
)

if not TORCH_AVAILABLE:
    print("PyTorch not available!")
    print("Install with: pip install torch --index-url https://download.pytorch.org/whl/rocm5.6")
    sys.exit(1)

import torch


def main():
    # Check GPU
    print("=" * 60)
    print("GPU STATUS")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected - using CPU")
        print("For AMD GPUs, install ROCm: https://rocm.docs.amd.com/")

    # Find data files
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*.csv"))

    if not csv_files:
        print("\nNo CSV files found in project root")
        return

    print(f"\nFound {len(csv_files)} data files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Training config
    config = TrainingConfig(
        hidden_sizes=(128, 64, 32),
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=50000,
        target_update_freq=100,
        n_episodes=100,
        device='auto',
    )

    print("\n" + "=" * 60)
    print("TRAINING CONFIG")
    print("=" * 60)
    print(f"  Hidden layers: {config.hidden_sizes}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Batch size: {config.batch_size}")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()

    trainer, stats = train_across_instruments(
        [str(f) for f in csv_files],
        config,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")

    # Final stats
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if stats:
        final = stats[-1]
        print(f"  Total trades: {final.get('trades', 0)}")
        print(f"  Win rate: {final.get('avg_win_rate', 0):.1f}%")
        print(f"  Avg P&L: {final.get('avg_pnl', 0):.4f}%")
        print(f"  Final epsilon: {final.get('epsilon', 0):.3f}")

    # Learning curve
    print("\n  LEARNING CURVE (every 20 episodes):")
    for i, s in enumerate(stats):
        if (i + 1) % 20 == 0:
            print(f"    Episode {i+1}: WR={s.get('avg_win_rate', 0):.1f}%, PnL={s.get('avg_pnl', 0):.4f}%")

    print("\n" + "=" * 60)
    print("RL DISCOVERY SUMMARY")
    print("=" * 60)
    print("""
  The RL agent has learned from ALL physics features:
  - Energy, Damping, Entropy (core physics)
  - Jerk, Impulse, Acceleration (derivatives)
  - Reynolds, Viscosity (flow dynamics)
  - Angular Momentum, Potential Energy (rotational)
  - Buying Pressure, Liquidity (order flow)
  - Range Position, Flow Consistency (context)

  Feature importance shows what RL found most useful.
  The learned policy encodes:
  - When to enter (fat candle probability)
  - Which direction (continuation vs reversal)
  - When to exit (energy recovery)

  No hardcoded rules - pure learned behavior.
""")


if __name__ == "__main__":
    main()
