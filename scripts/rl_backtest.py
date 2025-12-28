#!/usr/bin/env python3
"""
RL-Integrated Backtester

Runs backtests in VIRTUAL mode (no gates) and collects experience for RL training.

Features:
- Physics-based state representation
- Friction-aware reward shaping
- Adaptive regime classification
- Atomic checkpointing
- Experience replay buffer collection

Usage:
    python scripts/rl_backtest.py [data/*.csv]
    python scripts/rl_backtest.py --synthetic  # Generate test data
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinetra import (
    # Physics v7
    PhysicsEngineV7,
    PhysicsState,
    AgentType,
    # Strategies
    BerserkerStrategy,
    SniperStrategy,
    # Friction
    FrictionModel,
    TradingMode,
    get_symbol_spec,
    compute_friction_series,
    # Health/Reward
    RewardShaper,
    compute_reward_from_trade,
    # Persistence
    AtomicCheckpointer,
    CheckpointType,
)

from kinetra.physics_v7 import (
    compute_oscillator_state,
    compute_fractal_dimension_katz,
    compute_sample_entropy,
    compute_ftle_fast,
    classify_regime_adaptive,
)


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int  # 0=hold, 1=buy, 2=sell, 3=close
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    is_long: bool
    pnl_pct: float
    friction_pct: float
    net_pnl_pct: float
    duration_bars: int
    regime_entry: str
    physics_state_entry: Dict
    physics_state_exit: Dict


def generate_synthetic_data(
    n_bars: int = 2000,
    symbol: str = "EURUSD",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic characteristics.

    Includes:
    - Trend periods
    - Range periods
    - Volatility clusters
    - Volume patterns
    """
    np.random.seed(seed)

    # Base parameters
    base_price = 1.1000
    base_vol = 0.0008  # 8 pips typical hourly volatility

    # Generate regime switches
    regime_length = 100  # Average bars per regime
    regimes = []
    current_regime = 'trend_up'
    bar = 0
    while bar < n_bars:
        length = int(np.random.exponential(regime_length))
        length = max(20, min(200, length))  # Clamp
        regimes.extend([current_regime] * length)
        bar += length
        # Switch regime
        if current_regime == 'trend_up':
            current_regime = np.random.choice(['trend_down', 'range'])
        elif current_regime == 'trend_down':
            current_regime = np.random.choice(['trend_up', 'range'])
        else:
            current_regime = np.random.choice(['trend_up', 'trend_down'])
    regimes = regimes[:n_bars]

    # Generate returns based on regime
    returns = np.zeros(n_bars)
    volatility = np.zeros(n_bars)

    for i in range(n_bars):
        regime = regimes[i]

        # Volatility clustering (GARCH-like)
        if i > 0:
            vol = 0.9 * volatility[i-1] + 0.1 * base_vol + 0.05 * abs(returns[i-1])
        else:
            vol = base_vol
        volatility[i] = vol

        # Returns based on regime
        if regime == 'trend_up':
            drift = 0.0002  # Slight upward drift
            returns[i] = drift + np.random.normal(0, vol)
        elif regime == 'trend_down':
            drift = -0.0002  # Slight downward drift
            returns[i] = drift + np.random.normal(0, vol)
        else:  # range
            # Mean-reverting
            returns[i] = np.random.normal(0, vol * 0.7)

    # Build price series
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.0003, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0003, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume (higher during trends, lower in ranges)
    base_volume = 1000
    volume = np.zeros(n_bars)
    for i in range(n_bars):
        regime = regimes[i]
        if regime in ['trend_up', 'trend_down']:
            volume[i] = base_volume * (1 + np.random.exponential(0.5))
        else:
            volume[i] = base_volume * (0.5 + np.random.exponential(0.3))

    # Create DataFrame
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_bars,
        freq='h'
    )

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

    return df


def compute_state_vector(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 50
) -> np.ndarray:
    """
    Compute physics-based state vector for RL.

    Returns normalized feature vector suitable for neural network input.
    """
    start = max(0, idx - lookback + 1)
    window = df.iloc[start:idx+1]

    if len(window) < 10:
        # Not enough data - return zeros
        return np.zeros(20)

    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    volume = window['Volume'].values

    # Damped oscillator state
    osc = compute_oscillator_state(
        high, low, close, volume,
        lookback=min(20, len(window))
    )

    # Get latest values
    mass = osc['mass'][-1] if len(osc['mass']) > 0 else 0
    force = osc['force'][-1] if len(osc['force']) > 0 else 0
    accel = osc['acceleration'][-1] if len(osc['acceleration']) > 0 else 0
    velocity = osc['velocity'][-1] if len(osc['velocity']) > 0 else 0
    displacement = osc['displacement'][-1] if len(osc['displacement']) > 0 else 0
    symc = osc['symc'][-1] if len(osc['symc']) > 0 else 1.0

    # Fractal dimension
    fd = compute_fractal_dimension_katz(close)[-1] if len(close) >= 10 else 1.5

    # Sample entropy
    se = compute_sample_entropy(close, m=2)[-1] if len(close) >= 20 else 0.5

    # FTLE
    ftle = compute_ftle_fast(close, window=min(20, len(close)))[-1] if len(close) >= 20 else 0

    # Price features
    returns = np.diff(close) / close[:-1] if len(close) > 1 else [0]
    ret_mean = np.mean(returns)
    ret_std = np.std(returns) + 1e-10
    ret_skew = np.mean((returns - ret_mean)**3) / (ret_std**3) if ret_std > 0 else 0

    # Volume features
    vol_ratio = volume[-1] / np.mean(volume) if np.mean(volume) > 0 else 1

    # Normalize and clip
    state = np.array([
        np.clip(mass / 1e6, -5, 5),  # Mass (normalized)
        np.clip(force / 1e3, -5, 5),  # Force (normalized)
        np.clip(accel * 100, -5, 5),  # Acceleration
        np.clip(velocity * 100, -5, 5),  # Velocity
        np.clip(displacement * 10, -5, 5),  # Displacement
        np.clip(symc, 0, 5),  # SymC ratio
        np.clip(fd - 1.5, -1, 1),  # Fractal dimension (centered)
        np.clip(se, 0, 3),  # Sample entropy
        np.clip(ftle * 10, -2, 2),  # FTLE
        np.clip(ret_mean * 1000, -5, 5),  # Return mean
        np.clip(ret_std * 100, 0, 5),  # Return std
        np.clip(ret_skew, -3, 3),  # Return skew
        np.clip(vol_ratio - 1, -2, 2),  # Volume ratio
        # Regime one-hot (will be filled based on symc)
        1.0 if symc < 0.8 else 0.0,  # Underdamped
        1.0 if 0.8 <= symc <= 1.2 else 0.0,  # Critical
        1.0 if symc > 1.2 else 0.0,  # Overdamped
        # Recent momentum
        np.clip(np.sum(returns[-5:]) * 100 if len(returns) >= 5 else 0, -5, 5),
        np.clip(np.sum(returns[-10:]) * 100 if len(returns) >= 10 else 0, -5, 5),
        # ATR normalized
        np.clip((np.mean(high - low) / close[-1]) * 100, 0, 5),
        # Position in range
        np.clip((close[-1] - np.min(low)) / (np.max(high) - np.min(low) + 1e-10), 0, 1),
    ])

    return state


def run_backtest_episode(
    df: pd.DataFrame,
    symbol: str = "EURUSD",
    initial_capital: float = 10000,
    position_size_pct: float = 0.02,
    max_holding_bars: int = 48,
) -> Tuple[List[Experience], List[TradeResult], Dict]:
    """
    Run a single backtest episode and collect experiences.

    Uses VIRTUAL mode - no gates, explore freely.
    """
    # Setup
    spec = get_symbol_spec(symbol)
    friction_model = FrictionModel(spec, mode=TradingMode.VIRTUAL)
    reward_shaper = RewardShaper()

    # Compute friction series
    friction_df = compute_friction_series(df, symbol)

    # State
    capital = initial_capital
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0
    entry_bar = 0
    entry_state = None

    experiences: List[Experience] = []
    trades: List[TradeResult] = []

    # Metrics
    total_pnl = 0
    total_friction = 0
    n_trades = 0

    warmup = 100  # Bars needed for indicators

    for i in range(warmup, len(df) - 1):
        # Current state
        state = compute_state_vector(df, i)
        friction = friction_df.iloc[i]
        close = df['Close'].iloc[i]
        next_close = df['Close'].iloc[i + 1]

        # Determine regime from state
        symc = state[5]  # SymC is at index 5
        if symc < 0.8:
            regime = 'underdamped'
        elif symc > 1.2:
            regime = 'overdamped'
        else:
            regime = 'critical'

        # Simple policy for data collection (random with bias)
        # In real RL, this would be the policy network
        if position == 0:
            # Flat - decide to enter or stay flat
            if regime == 'underdamped' and state[3] > 0.5:  # Velocity up
                action = 1  # Buy
            elif regime == 'underdamped' and state[3] < -0.5:  # Velocity down
                action = 2  # Sell
            elif regime == 'overdamped':
                # Mean reversion - fade the move
                if state[4] > 0.3:  # Displaced up
                    action = 2  # Sell
                elif state[4] < -0.3:  # Displaced down
                    action = 1  # Buy
                else:
                    action = 0  # Hold
            else:
                action = 0 if np.random.random() > 0.1 else np.random.choice([1, 2])
        else:
            # In position - decide to hold or close
            bars_held = i - entry_bar
            pnl_pct = ((close / entry_price) - 1) * 100 * position

            # Exit conditions
            should_exit = (
                bars_held >= max_holding_bars or  # Max holding
                pnl_pct > 0.5 or  # Take profit
                pnl_pct < -0.3 or  # Stop loss
                (regime == 'critical' and bars_held > 10)  # Exit in critical regime
            )

            action = 3 if should_exit else 0

        # Execute action
        reward = 0
        done = False

        if action == 1 and position == 0:  # Buy
            position = 1
            entry_price = close
            entry_bar = i
            entry_state = state.copy()
            # Entry friction
            reward -= friction['total_friction_pct'] / 2  # Half on entry

        elif action == 2 and position == 0:  # Sell
            position = -1
            entry_price = close
            entry_bar = i
            entry_state = state.copy()
            reward -= friction['total_friction_pct'] / 2

        elif action == 3 and position != 0:  # Close
            # Calculate PnL
            pnl_pct = ((close / entry_price) - 1) * 100 * position
            exit_friction = friction['total_friction_pct'] / 2
            net_pnl = pnl_pct - exit_friction

            # Reward shaping
            duration = i - entry_bar
            reward = net_pnl  # Base reward

            # Physics-aligned bonus
            if regime == 'underdamped' and pnl_pct > 0:
                reward *= 1.2  # Bonus for trend capture
            elif regime == 'overdamped' and pnl_pct > 0:
                reward *= 1.1  # Small bonus for mean-reversion

            # Duration penalty
            if duration > max_holding_bars / 2:
                reward *= 0.9  # Slight penalty for long holds

            # Record trade
            trades.append(TradeResult(
                entry_bar=entry_bar,
                exit_bar=i,
                entry_price=entry_price,
                exit_price=close,
                is_long=(position == 1),
                pnl_pct=pnl_pct,
                friction_pct=friction['total_friction_pct'],
                net_pnl_pct=net_pnl,
                duration_bars=duration,
                regime_entry=regime,
                physics_state_entry={'symc': float(entry_state[5]) if entry_state is not None else 0},
                physics_state_exit={'symc': float(state[5])},
            ))

            total_pnl += net_pnl
            total_friction += friction['total_friction_pct']
            n_trades += 1

            position = 0
            entry_price = 0
            entry_bar = 0
            entry_state = None

        elif action == 0 and position != 0:  # Hold
            # Unrealized PnL change as reward signal
            unrealized_change = ((next_close / close) - 1) * 100 * position
            reward = unrealized_change * 0.1  # Scaled down

        # Next state
        next_state = compute_state_vector(df, i + 1)

        # Store experience
        experiences.append(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={'bar': i, 'regime': regime, 'position': position}
        ))

    # Summary metrics
    metrics = {
        'total_bars': len(df) - warmup,
        'n_trades': n_trades,
        'total_pnl_pct': total_pnl,
        'total_friction_pct': total_friction,
        'net_pnl_pct': total_pnl,
        'avg_pnl_per_trade': total_pnl / n_trades if n_trades > 0 else 0,
        'win_rate': sum(1 for t in trades if t.net_pnl_pct > 0) / n_trades if n_trades > 0 else 0,
        'n_experiences': len(experiences),
    }

    return experiences, trades, metrics


def main():
    parser = argparse.ArgumentParser(description='RL-Integrated Backtester')
    parser.add_argument('files', nargs='*', help='CSV data files')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--bars', type=int, default=2000, help='Bars for synthetic data')
    parser.add_argument('--symbol', default='EURUSD', help='Symbol name')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    args = parser.parse_args()

    print("=" * 60)
    print("KINETRA RL BACKTESTER")
    print("VIRTUAL MODE - No gates, exploring freely")
    print("=" * 60)

    # Setup checkpointer
    checkpointer = AtomicCheckpointer(args.checkpoint_dir)

    # Load or generate data
    if args.synthetic or not args.files:
        print(f"\nGenerating {args.bars} bars of synthetic data...")
        df = generate_synthetic_data(n_bars=args.bars, symbol=args.symbol)
        print(f"  Price range: {df['Close'].min():.5f} - {df['Close'].max():.5f}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    else:
        print(f"\nLoading {len(args.files)} data files...")
        dfs = []
        for f in args.files:
            df = pd.read_csv(f, parse_dates=['time'], index_col='time')
            dfs.append(df)
        df = pd.concat(dfs).sort_index()

    # Run backtest
    print(f"\nRunning backtest episode...")
    print(f"  Symbol: {args.symbol}")
    print(f"  Bars: {len(df)}")

    experiences, trades, metrics = run_backtest_episode(df, symbol=args.symbol)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total bars:      {metrics['total_bars']}")
    print(f"  Total trades:    {metrics['n_trades']}")
    print(f"  Total PnL:       {metrics['total_pnl_pct']:.2f}%")
    print(f"  Total friction:  {metrics['total_friction_pct']:.2f}%")
    print(f"  Net PnL:         {metrics['net_pnl_pct']:.2f}%")
    print(f"  Avg per trade:   {metrics['avg_pnl_per_trade']:.3f}%")
    print(f"  Win rate:        {metrics['win_rate']:.1%}")
    print(f"  Experiences:     {metrics['n_experiences']}")

    # Regime distribution
    if trades:
        regime_counts = {}
        for t in trades:
            regime_counts[t.regime_entry] = regime_counts.get(t.regime_entry, 0) + 1
        print("\nTrades by regime:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / len(trades) * 100
            regime_trades = [t for t in trades if t.regime_entry == regime]
            avg_pnl = np.mean([t.net_pnl_pct for t in regime_trades])
            print(f"  {regime:12s}: {count:3d} ({pct:5.1f}%) - Avg PnL: {avg_pnl:+.3f}%")

    # Save experiences to replay buffer
    print("\nSaving replay buffer...")
    buffer_data = {
        'experiences': [(asdict(e) if hasattr(e, '__dataclass_fields__') else e) for e in experiences],
        'trades': [asdict(t) for t in trades],
        'metrics': metrics,
    }

    # Convert numpy arrays to lists for serialization
    for exp in buffer_data['experiences']:
        exp['state'] = exp['state'].tolist() if isinstance(exp['state'], np.ndarray) else exp['state']
        exp['next_state'] = exp['next_state'].tolist() if isinstance(exp['next_state'], np.ndarray) else exp['next_state']

    version = checkpointer.save(CheckpointType.REPLAY_BUFFER, buffer_data)
    print(f"  Saved as version {version}")

    # Save metrics
    checkpointer.save_metrics(metrics)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
