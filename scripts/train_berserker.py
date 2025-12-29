#!/usr/bin/env python3
"""
Berserker Entry Training - GPU Accelerated

NEW ARCHITECTURE:
- Train ONE instrument at a time
- 4 PARALLEL timeframe streams (M15, M30, H1, H4)
- Analyze & summarize after each instrument completes
- Move to next instrument

Features:
- GPU acceleration (AMD ROCm / NVIDIA CUDA)
- Parallel timeframe training per instrument
- Atomic model saving (no corruption on interrupt)
- Run-based data management
- Comprehensive logging

Usage:
    python scripts/train_berserker.py --run berserker_run1
    python scripts/train_berserker.py --new-run  # Creates new run
"""

import sys
import os

# Configure ROCm BEFORE importing torch - must be set before any CUDA/HIP initialization
# This avoids hipBLASLt warnings on RDNA3 GPUs (RX 7600, etc.)
os.environ.setdefault('TORCH_BLAS_PREFER_HIPBLASLT', '0')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse
import logging
import threading
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetra.mt5_connector import load_csv_data
from kinetra.metrics_server import start_metrics_server, RLMetrics
from kinetra.rl_gpu_trainer import DQN, ReplayBuffer, TrainingConfig, TradingEnv, PhysicsFeatureComputer
from kinetra.data_manager import DataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thread-safe logging lock
log_lock = threading.Lock()

TIMEFRAMES = ['M15', 'M30', 'H1', 'H4']


class PowerManager:
    """Manage CPU/GPU power settings for maximum training performance."""

    def __init__(self):
        self.original_gpu_power = None
        self.original_cpu_governor = None

    def set_high_performance(self):
        """Set CPU and GPU to high performance mode."""
        logger.info("Setting power to HIGH PERFORMANCE mode...")

        # GPU: Set AMD GPU to high performance
        try:
            # Read current setting
            gpu_power_file = "/sys/class/drm/card1/device/power_dpm_force_performance_level"
            if os.path.exists(gpu_power_file):
                with open(gpu_power_file, 'r') as f:
                    self.original_gpu_power = f.read().strip()

                # Set to high (requires root or appropriate permissions)
                result = subprocess.run(
                    ['sudo', 'tee', gpu_power_file],
                    input=b'high',
                    capture_output=True
                )
                if result.returncode == 0:
                    logger.info("  GPU: Set to 'high' performance")
                else:
                    logger.warning("  GPU: Could not set power (may need sudo)")
        except Exception as e:
            logger.warning(f"  GPU power setting failed: {e}")

        # CPU: Set governor to performance
        try:
            result = subprocess.run(
                ['sudo', 'cpupower', 'frequency-set', '-g', 'performance'],
                capture_output=True
            )
            if result.returncode == 0:
                self.original_cpu_governor = 'powersave'  # Assume default
                logger.info("  CPU: Set to 'performance' governor")
            else:
                logger.warning("  CPU: Could not set governor (may need cpupower installed)")
        except FileNotFoundError:
            logger.warning("  CPU: cpupower not installed (sudo apt install linux-tools-common)")
        except Exception as e:
            logger.warning(f"  CPU governor setting failed: {e}")

        # Disable screen blanking during training
        try:
            subprocess.run(['xset', 's', 'off'], capture_output=True)
            subprocess.run(['xset', '-dpms'], capture_output=True)
            logger.info("  Display: Screen blanking disabled")
        except Exception:
            pass  # May not have display

    def restore(self):
        """Restore original power settings."""
        logger.info("Restoring power settings...")

        # Restore GPU
        if self.original_gpu_power:
            try:
                gpu_power_file = "/sys/class/drm/card1/device/power_dpm_force_performance_level"
                subprocess.run(
                    ['sudo', 'tee', gpu_power_file],
                    input=self.original_gpu_power.encode(),
                    capture_output=True
                )
                logger.info(f"  GPU: Restored to '{self.original_gpu_power}'")
            except Exception as e:
                logger.warning(f"  GPU restore failed: {e}")

        # Restore CPU
        if self.original_cpu_governor:
            try:
                subprocess.run(
                    ['sudo', 'cpupower', 'frequency-set', '-g', self.original_cpu_governor],
                    capture_output=True
                )
                logger.info(f"  CPU: Restored to '{self.original_cpu_governor}'")
            except Exception:
                pass

        # Re-enable screen blanking
        try:
            subprocess.run(['xset', 's', 'on'], capture_output=True)
            subprocess.run(['xset', '+dpms'], capture_output=True)
        except Exception:
            pass

    def __enter__(self):
        self.set_high_performance()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()


def estimate_vram_usage() -> Dict:
    """Estimate VRAM usage per model and calculate parallelism capacity.

    RX 7600 = 8GB VRAM
    Each DQN model (256, 128, 64) with 47 inputs ≈ 0.5MB parameters
    Replay buffer (50k samples) ≈ 200MB per timeframe
    PyTorch overhead ≈ 500MB

    Returns capacity estimates.
    """
    if not torch.cuda.is_available():
        return {'parallel_instruments': 1, 'reason': 'CPU only'}

    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        reserved = 0.5  # GB for PyTorch overhead

        # Per timeframe estimates
        model_size = 0.002  # GB (small DQN)
        buffer_size = 0.2   # GB (50k samples)
        per_timeframe = model_size + buffer_size

        # Per instrument = 4 timeframes
        per_instrument = per_timeframe * 4

        available = total_vram - reserved
        max_parallel = int(available / per_instrument)
        max_parallel = max(1, min(max_parallel, 4))  # Cap at 4

        return {
            'total_vram_gb': round(total_vram, 1),
            'per_instrument_gb': round(per_instrument, 2),
            'parallel_instruments': max_parallel,
            'parallel_timeframes': 4,
            'total_parallel_models': max_parallel * 4,
        }
    except Exception as e:
        return {'parallel_instruments': 1, 'error': str(e)}


def setup_gpu_optimizations():
    """Configure GPU for maximum training speed."""
    if not torch.cuda.is_available():
        return

    # Enable cuDNN auto-tuner - finds fastest algorithms for your hardware
    torch.backends.cudnn.benchmark = True

    # Enable TF32 for faster matrix ops on Ampere+ GPUs (also helps on some AMD)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set default tensor type to CUDA
    if torch.cuda.is_available():
        torch.set_default_device('cuda')

    logger.info("GPU optimizations enabled: cudnn.benchmark, TF32")


def compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Compile model with torch.compile for 2x+ speedup.

    NOTE: Disabled on ROCm/AMD GPUs due to compatibility issues.
    Uses 'reduce-overhead' mode for small models like DQN.
    Falls back gracefully if compilation fails.
    """
    if not torch.cuda.is_available():
        return model

    # Check if ROCm - torch.compile has issues with AMD GPUs
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if is_rocm:
        # ROCm doesn't fully support torch.compile yet
        logger.info("  Skipping torch.compile (not fully supported on ROCm)")
        return model

    try:
        # PyTorch 2.0+ torch.compile
        # 'reduce-overhead' is best for small models with many iterations
        compiled = torch.compile(
            model,
            mode='reduce-overhead',  # Fast compile, good for RL
            fullgraph=False,         # Allow graph breaks
        )
        logger.info("  Model compiled with torch.compile (reduce-overhead mode)")
        return compiled
    except Exception as e:
        logger.warning(f"  torch.compile failed, using eager mode: {e}")
        return model


class AtomicSaver:
    """Atomic model saving to prevent corruption."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        episode: int,
        metrics: Dict,
        filename: str = "checkpoint.pt"
    ) -> Path:
        """Atomically save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        with self._lock:
            target_path = self.models_dir / filename
            temp_path = self.models_dir / f".{filename}.tmp"
            torch.save(checkpoint, temp_path)
            shutil.move(str(temp_path), str(target_path))

        return target_path

    def save_best(self, model: nn.Module, metrics: Dict, filename: str = "best_model.pt") -> Path:
        """Save best model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        with self._lock:
            target_path = self.models_dir / filename
            temp_path = self.models_dir / f".{filename}.tmp"
            torch.save(checkpoint, temp_path)
            shutil.move(str(temp_path), str(target_path))

        return target_path

    def load_checkpoint(self, filename: str = "checkpoint.pt") -> Optional[Dict]:
        """Load checkpoint if it exists."""
        path = self.models_dir / filename
        if path.exists():
            return torch.load(path)
        return None


class RunLogger:
    """Logs training results to file."""

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.logs_dir / f"results_{timestamp}.jsonl"
        self.summary_file = self.logs_dir / "summary.json"

    def log_episode(self, instrument: str, timeframe: str, episode: int, metrics: Dict):
        """Log single episode results."""
        entry = {
            'instrument': instrument,
            'timeframe': timeframe,
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        with self._lock:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def log_instrument_summary(self, instrument: str, summary: Dict):
        """Log instrument completion summary."""
        summary_file = self.logs_dir / f"summary_{instrument}.json"
        summary['completed_at'] = datetime.now().isoformat()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def log_summary(self, summary: Dict):
        """Log final summary."""
        summary['completed_at'] = datetime.now().isoformat()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def configure_rocm_backend():
    """Configure ROCm to use hipBLAS instead of hipBLASLt for unsupported architectures."""
    import os
    # Force hipBLAS backend for RDNA3 (gfx1100/gfx1102) - avoids hipBLASLt warnings
    os.environ.setdefault('TORCH_BLAS_PREFER_HIPBLASLT', '0')
    # Ensure GFX version override is set for RX 7600 (gfx1102 -> gfx1100)
    os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')


def detect_device() -> torch.device:
    """Detect best available device - AMD ROCm or NVIDIA CUDA."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

        if is_rocm:
            configure_rocm_backend()
            logger.info(f"Using AMD GPU (ROCm): {gpu_name}")
            logger.info(f"  hipBLAS backend configured (TORCH_BLAS_PREFER_HIPBLASLT=0)")
        else:
            logger.info(f"Using NVIDIA GPU (CUDA): {gpu_name}")

        return device
    else:
        logger.error("=" * 60)
        logger.error("NO GPU DETECTED - Training will be 100x slower!")
        logger.error("=" * 60)
        logger.error("For AMD GPUs: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        logger.error("export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RDNA3")
        logger.error("=" * 60)
        return torch.device('cpu')


def parse_instrument_timeframe(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse instrument and timeframe from filename.

    Expected format: BTCUSD_M30_20240101_20241231.csv
    Returns: (instrument, timeframe) or (None, None) if parse fails
    """
    parts = filename.replace('.csv', '').split('_')
    if len(parts) >= 2:
        instrument = parts[0]
        timeframe = parts[1]
        return instrument, timeframe
    return None, None


def group_files_by_instrument(data_files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """Group data files by instrument, then by timeframe.

    Returns: {instrument: {timeframe: path, ...}, ...}
    """
    grouped = defaultdict(dict)

    for path in data_files:
        instrument, timeframe = parse_instrument_timeframe(path.name)
        if instrument and timeframe:
            grouped[instrument][timeframe] = path

    return dict(grouped)


def train_timeframe(
    instrument: str,
    timeframe: str,
    data_path: Path,
    device: torch.device,
    saver: AtomicSaver,
    run_logger: RunLogger,
    n_episodes: int = 100,
    config: TrainingConfig = None,
) -> Dict:
    """Train single timeframe for an instrument.

    Returns summary metrics for this timeframe.
    """
    thread_id = f"{instrument}_{timeframe}"

    with log_lock:
        logger.info(f"  [{thread_id}] Starting training...")

    # Load data
    data = load_csv_data(str(data_path))
    fc = PhysicsFeatureComputer()
    features = fc.compute(data)
    env = TradingEnv(data, features)

    with log_lock:
        logger.info(f"  [{thread_id}] Loaded {len(data)} bars, {env.state_dim} features")

    state_dim = env.state_dim
    action_dim = env.action_dim

    if config is None:
        config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.95,
            epsilon_start=0.8,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=50000,
            n_episodes=n_episodes,
        )

    # Initialize networks with GPU optimizations
    hidden_sizes = (256, 128, 64)
    q_net = DQN(state_dim, action_dim, hidden_sizes).to(device)
    target_net = DQN(state_dim, action_dim, hidden_sizes).to(device)
    target_net.load_state_dict(q_net.state_dict())

    # Compile models for 2x+ speedup (PyTorch 2.0+)
    use_compile = device.type == 'cuda'
    if use_compile:
        q_net = compile_model(q_net, device)
        target_net = compile_model(target_net, device)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size, device)

    # Mixed precision training (FP16) - 2x memory efficiency, faster math
    # Note: AMP has issues on ROCm, disable it for AMD GPUs
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    use_amp = device.type == 'cuda' and not is_rocm
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if is_rocm:
        logger.info("  AMP disabled (compatibility issues on ROCm)")

    epsilon = config.epsilon_start
    best_pnl = float('-inf')
    total_steps = 0
    episode_start_time = None
    heartbeat_interval = 1000  # Log every N steps

    all_episode_metrics = []

    for episode in range(n_episodes):
        episode_stats = {
            'trades': 0, 'wins': 0, 'pnl': 0,
            'mfe': [], 'mae': [],
            'entry_energy': [], 'move_capture': [], 'mfe_first': [],
        }
        losses = []
        episode_steps = 0
        episode_start_time = time.time()

        state = env.reset()
        done = False

        while not done:
            # Action selection
            if np.random.random() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q = q_net(state_t)
                    action = q.argmax().item()

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            # Track trades
            if info.get('trade_pnl') is not None:
                pnl = info['trade_pnl']
                episode_stats['trades'] += 1
                episode_stats['pnl'] += pnl
                if pnl > 0:
                    episode_stats['wins'] += 1
                episode_stats['mfe'].append(info.get('mfe', 0))
                episode_stats['mae'].append(info.get('mae', 0))
                episode_stats['entry_energy'].append(info.get('entry_energy', 0.5))
                episode_stats['move_capture'].append(info.get('move_capture', 0))
                episode_stats['mfe_first'].append(info.get('mfe_first', False))

            # Training step with AMP (Automatic Mixed Precision)
            if len(buffer) >= config.batch_size:
                batch = buffer.sample(config.batch_size)
                states_t, actions_t, rewards_t, next_states_t, dones_t = batch

                if isinstance(states_t, torch.Tensor):
                    states_t = torch.nan_to_num(states_t, nan=0.5, posinf=1.0, neginf=0.0)
                    next_states_t = torch.nan_to_num(next_states_t, nan=0.5, posinf=1.0, neginf=0.0)
                    states_t = torch.clamp(states_t, -10, 10)
                    next_states_t = torch.clamp(next_states_t, -10, 10)
                    rewards_t = torch.clamp(rewards_t, -10, 10)
                else:
                    states_np = np.nan_to_num(np.array(states_t), nan=0.5, posinf=1.0, neginf=0.0)
                    next_states_np = np.nan_to_num(np.array(next_states_t), nan=0.5, posinf=1.0, neginf=0.0)
                    states_t = torch.FloatTensor(np.clip(states_np, -10, 10)).to(device)
                    actions_t = torch.LongTensor(actions_t).to(device)
                    rewards_t = torch.FloatTensor(np.clip(rewards_t, -10, 10)).to(device)
                    next_states_t = torch.FloatTensor(np.clip(next_states_np, -10, 10)).to(device)
                    dones_t = torch.FloatTensor(dones_t).to(device)

                # Compute targets (no grad needed for target network)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                    targets = rewards_t + config.gamma * next_q * (1 - dones_t)
                    targets = torch.clamp(targets, -100, 100)

                # Forward pass with AMP autocast
                optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
                        loss = F.huber_loss(current_q, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
                    loss = F.huber_loss(current_q, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                    optimizer.step()

                losses.append(loss.item())

                # Soft target update (Polyak averaging) - more stable than hard updates
                tau = 0.005
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            state = next_state
            total_steps += 1
            episode_steps += 1

            # Heartbeat - show progress during long episodes
            if episode_steps % heartbeat_interval == 0:
                elapsed = time.time() - episode_start_time
                steps_per_sec = episode_steps / elapsed if elapsed > 0 else 0
                with log_lock:
                    print(f"    [{thread_id}] ♥ Step {episode_steps:,} | {steps_per_sec:.0f} steps/s | Trades: {episode_stats['trades']}        ", end='\r', flush=True)

        # Clear heartbeat line
        print(" " * 80, end='\r', flush=True)

        # Epsilon decay
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        # Episode metrics
        n_trades = episode_stats['trades']
        win_rate = (episode_stats['wins'] / n_trades * 100) if n_trades > 0 else 0
        total_pnl = episode_stats['pnl']
        avg_loss = np.mean(losses) if losses else 0
        avg_mfe = np.mean(episode_stats['mfe']) if episode_stats['mfe'] else 0
        avg_mae = np.mean(episode_stats['mae']) if episode_stats['mae'] else 0
        mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

        episode_metrics = {
            'trades': n_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_loss': avg_loss,
            'mfe_mae_ratio': mfe_mae_ratio,
            'epsilon': epsilon,
        }
        all_episode_metrics.append(episode_metrics)

        # Log to file
        run_logger.log_episode(instrument, timeframe, episode + 1, episode_metrics)

        # Save best model
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            saver.save_best(q_net, episode_metrics, f"best_{instrument}_{timeframe}.pt")

        # Progress log every 10 episodes
        if (episode + 1) % 10 == 0:
            with log_lock:
                logger.info(
                    f"  [{thread_id}] Ep {episode+1:3d}/{n_episodes} | "
                    f"Trades: {n_trades:3d} | WR: {win_rate:5.1f}% | "
                    f"PnL: {total_pnl:+.2f}% | Loss: {avg_loss:.4f}"
                )
            saver.save_checkpoint(
                q_net, optimizer, episode + 1,
                {'best_pnl': best_pnl, **episode_metrics},
                f"checkpoint_{instrument}_{timeframe}.pt"
            )

    # Final save
    saver.save_checkpoint(
        q_net, optimizer, n_episodes,
        {'best_pnl': best_pnl},
        f"final_{instrument}_{timeframe}.pt"
    )

    # Summary for this timeframe
    summary = {
        'instrument': instrument,
        'timeframe': timeframe,
        'bars': len(data),
        'episodes': n_episodes,
        'best_pnl': best_pnl,
        'final_epsilon': epsilon,
        'total_steps': total_steps,
        'avg_trades_per_episode': np.mean([m['trades'] for m in all_episode_metrics]),
        'avg_win_rate': np.mean([m['win_rate'] for m in all_episode_metrics]),
        'avg_pnl': np.mean([m['total_pnl'] for m in all_episode_metrics]),
    }

    with log_lock:
        logger.info(f"  [{thread_id}] ✓ Complete! Best PnL: {best_pnl:+.2f}%")

    return summary


def analyze_instrument_results(instrument: str, timeframe_results: Dict[str, Dict]) -> Dict:
    """Analyze and summarize results across all timeframes for an instrument."""

    logger.info(f"\n{'='*60}")
    logger.info(f"INSTRUMENT ANALYSIS: {instrument}")
    logger.info(f"{'='*60}")

    summary = {
        'instrument': instrument,
        'timeframes': {},
        'best_timeframe': None,
        'best_pnl': float('-inf'),
        'total_trades': 0,
        'avg_win_rate': 0,
    }

    win_rates = []
    pnls = []

    for tf, results in timeframe_results.items():
        summary['timeframes'][tf] = results

        logger.info(f"  {tf:4s} | PnL: {results['best_pnl']:+8.2f}% | "
                   f"WR: {results['avg_win_rate']:5.1f}% | "
                   f"Trades/Ep: {results['avg_trades_per_episode']:.1f}")

        if results['best_pnl'] > summary['best_pnl']:
            summary['best_pnl'] = results['best_pnl']
            summary['best_timeframe'] = tf

        win_rates.append(results['avg_win_rate'])
        pnls.append(results['best_pnl'])

    summary['avg_win_rate'] = np.mean(win_rates) if win_rates else 0
    summary['avg_pnl'] = np.mean(pnls) if pnls else 0

    logger.info(f"\n  BEST: {summary['best_timeframe']} with {summary['best_pnl']:+.2f}% PnL")
    logger.info(f"  AVG across timeframes: WR={summary['avg_win_rate']:.1f}%, PnL={summary['avg_pnl']:+.2f}%")
    logger.info(f"{'='*60}\n")

    return summary


def train_instrument_parallel(
    instrument: str,
    timeframe_files: Dict[str, Path],
    device: torch.device,
    saver: AtomicSaver,
    run_logger: RunLogger,
    n_episodes: int = 100,
) -> Dict:
    """Train all timeframes for an instrument in parallel.

    Returns instrument summary after all timeframes complete.
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"# TRAINING INSTRUMENT: {instrument}")
    logger.info(f"# Timeframes: {', '.join(timeframe_files.keys())}")
    logger.info(f"{'#'*60}\n")

    timeframe_results = {}

    # Run timeframes in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(timeframe_files)) as executor:
        futures = {}

        for timeframe, data_path in timeframe_files.items():
            future = executor.submit(
                train_timeframe,
                instrument=instrument,
                timeframe=timeframe,
                data_path=data_path,
                device=device,
                saver=saver,
                run_logger=run_logger,
                n_episodes=n_episodes,
            )
            futures[future] = timeframe

        # Collect results as they complete
        for future in as_completed(futures):
            timeframe = futures[future]
            try:
                result = future.result()
                timeframe_results[timeframe] = result
            except Exception as e:
                logger.error(f"  [{instrument}_{timeframe}] FAILED: {e}")
                timeframe_results[timeframe] = {'error': str(e)}

    # Analyze results for this instrument
    summary = analyze_instrument_results(instrument, timeframe_results)

    # Log instrument summary
    run_logger.log_instrument_summary(instrument, summary)

    return summary


def train_berserker(
    run_dir: Path,
    n_episodes: int = 100,
    metrics_port: int = 8001,
    parallel_instruments: int = 0,  # 0 = auto-detect based on VRAM
):
    """Train Berserker - parallel instruments, each with parallel timeframes.

    Architecture:
    1. Group files by instrument
    2. Train N instruments in parallel (based on VRAM)
    3. Each instrument has 4 parallel timeframe threads
    4. Each model is SEPARATE (no shared weights)
    5. Analyze & summarize after each instrument batch
    """

    logger.info("=" * 70)
    logger.info("KINETRA BERSERKER TRAINING")
    logger.info("=" * 70)

    # Setup paths
    data_dir = run_dir / "data"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"

    # Initialize components
    device = detect_device()
    setup_gpu_optimizations()  # Enable cudnn.benchmark, TF32, etc.

    # Estimate VRAM and parallelism capacity
    vram_info = estimate_vram_usage()
    if parallel_instruments <= 0:
        parallel_instruments = vram_info.get('parallel_instruments', 1)

    logger.info(f"VRAM: {vram_info.get('total_vram_gb', '?')} GB")
    logger.info(f"Per instrument: ~{vram_info.get('per_instrument_gb', '?')} GB (4 timeframes)")
    logger.info(f"Parallel capacity: {parallel_instruments} instruments × 4 timeframes = {parallel_instruments * 4} models")
    logger.info("-" * 70)

    saver = AtomicSaver(models_dir)
    run_logger = RunLogger(logs_dir)
    metrics = start_metrics_server(metrics_port)

    # Load and group data files
    data_files = sorted(data_dir.glob("*.csv"))
    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        return

    logger.info(f"Found {len(data_files)} data files")

    # Group by instrument
    grouped = group_files_by_instrument(data_files)

    logger.info(f"Instruments: {list(grouped.keys())}")
    for inst, tfs in grouped.items():
        logger.info(f"  {inst}: {list(tfs.keys())}")

    # Train instruments with power management
    all_results = {}
    instruments_list = list(grouped.items())

    with PowerManager():
        # Process instruments in batches based on parallel capacity
        for batch_start in range(0, len(instruments_list), parallel_instruments):
            batch = instruments_list[batch_start:batch_start + parallel_instruments]
            batch_num = batch_start // parallel_instruments + 1
            total_batches = (len(instruments_list) + parallel_instruments - 1) // parallel_instruments

            if len(batch) > 1:
                logger.info(f"\n{'#'*60}")
                logger.info(f"# BATCH {batch_num}/{total_batches}: Training {len(batch)} instruments in parallel")
                logger.info(f"# {[inst for inst, _ in batch]}")
                logger.info(f"{'#'*60}")

                # Train batch of instruments in parallel
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    futures = {}
                    for instrument, timeframe_files in batch:
                        future = executor.submit(
                            train_instrument_parallel,
                            instrument=instrument,
                            timeframe_files=timeframe_files,
                            device=device,
                            saver=saver,
                            run_logger=run_logger,
                            n_episodes=n_episodes,
                        )
                        futures[future] = instrument

                    for future in as_completed(futures):
                        instrument = futures[future]
                        try:
                            result = future.result()
                            all_results[instrument] = result
                        except Exception as e:
                            logger.error(f"  [{instrument}] FAILED: {e}")
                            all_results[instrument] = {'error': str(e)}
            else:
                # Single instrument - no extra parallelism needed
                instrument, timeframe_files = batch[0]
                logger.info(f"\n[{batch_start + 1}/{len(instruments_list)}] Processing {instrument}...")
                result = train_instrument_parallel(
                    instrument=instrument,
                    timeframe_files=timeframe_files,
                    device=device,
                    saver=saver,
                    run_logger=run_logger,
                    n_episodes=n_episodes,
                )
                all_results[instrument] = result

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE - FINAL SUMMARY")
    logger.info("=" * 70)

    best_overall = {'instrument': None, 'timeframe': None, 'pnl': float('-inf')}

    for instrument, result in all_results.items():
        if result.get('best_pnl', float('-inf')) > best_overall['pnl']:
            best_overall['instrument'] = instrument
            best_overall['timeframe'] = result.get('best_timeframe')
            best_overall['pnl'] = result.get('best_pnl', 0)

        logger.info(f"  {instrument}: Best={result.get('best_timeframe')} @ {result.get('best_pnl', 0):+.2f}%")

    logger.info(f"\n  OVERALL BEST: {best_overall['instrument']} / {best_overall['timeframe']} "
               f"with {best_overall['pnl']:+.2f}% PnL")
    logger.info("=" * 70)

    # Save final summary
    run_logger.log_summary({
        'instruments': list(all_results.keys()),
        'results': all_results,
        'best_overall': best_overall,
        'episodes_per_timeframe': n_episodes,
    })


def main():
    parser = argparse.ArgumentParser(description="Berserker Entry Training")
    parser.add_argument("--run", type=str, help="Run name (e.g., berserker_run1)")
    parser.add_argument("--new-run", action="store_true", help="Create new run")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per timeframe")
    parser.add_argument("--port", type=int, default=8001, help="Metrics port")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Parallel instruments (0=auto based on VRAM)")
    args = parser.parse_args()

    dm = DataManager()

    # Get or create run
    if args.new_run:
        run_dir = dm.create_run(strategy="berserker")
    elif args.run:
        run_dir = dm.get_run(args.run)
        if run_dir is None:
            logger.error(f"Run not found: {args.run}")
            logger.info("Available runs:")
            for run in dm.list_runs():
                logger.info(f"  - {run['name']}")
            return
    else:
        runs = dm.list_runs()
        if runs:
            run_name = runs[-1]['name']
            run_dir = dm.get_run(run_name)
            logger.info(f"Using existing run: {run_name}")
        else:
            run_dir = dm.create_run(strategy="berserker")

    train_berserker(
        run_dir=run_dir,
        n_episodes=args.episodes,
        metrics_port=args.port,
        parallel_instruments=args.parallel,
    )


if __name__ == "__main__":
    main()
