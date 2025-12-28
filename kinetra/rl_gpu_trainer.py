"""
GPU-Accelerated RL Trainer for Physics-Based Trading

Uses PyTorch with ROCm (AMD GPU) or CUDA (NVIDIA).
Trains across multiple timeframes and instruments.

Let RL discover:
- What predicts fat candles (magnitude)
- What predicts continuation vs reversal (direction)
- Optimal exit timing (energy recovery)

No hardcoded rules - pure feature learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Network
    hidden_sizes: Tuple[int, ...] = (128, 64, 32)
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Training
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 100
    n_episodes: int = 200

    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'


class PhysicsFeatureComputer:
    """Compute ALL physics features for RL - no hardcoded rules."""

    def __init__(self):
        # No hardcoded lookback - computed from data
        pass

    def _fast_rolling_percentile(self, series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
        """
        FAST rolling percentile using vectorized operations.

        Replaces slow lambda-based approach:
          series.rolling(w).apply(lambda x: (x.iloc[-1] > x.iloc[:-1]).mean())

        This is O(n) vs O(n * window) for the lambda version.
        Uses pandas rank() which is implemented in C/Cython.
        """
        # Rolling rank approach: for each window, what's the rank of current value?
        # rank / window_size = percentile

        def rolling_rank(arr, w):
            """Compute rolling rank using numpy operations."""
            n = len(arr)
            result = np.full(n, 0.5)

            # Use stride tricks for efficient windowing
            if n < w:
                return result

            for i in range(w - 1, n):
                start = max(0, i - w + 1)
                window_vals = arr[start:i+1]
                current = arr[i]
                # Percentile = proportion of values less than current
                result[i] = np.mean(window_vals[:-1] < current) if len(window_vals) > 1 else 0.5

            return result

        # Use numba if available, otherwise numpy
        try:
            from numba import jit

            @jit(nopython=True, cache=True)
            def fast_rolling_pct(arr, w):
                n = len(arr)
                result = np.full(n, 0.5)
                for i in range(w - 1, n):
                    start = i - w + 1
                    window_vals = arr[start:i+1]
                    current = arr[i]
                    count = 0
                    for j in range(len(window_vals) - 1):
                        if window_vals[j] < current:
                            count += 1
                    result[i] = count / (len(window_vals) - 1) if len(window_vals) > 1 else 0.5
                return result

            result = fast_rolling_pct(series.values.astype(np.float64), window)
        except ImportError:
            # Fallback to pandas rolling rank (still faster than lambda)
            result = series.rolling(window, min_periods=min_periods).rank(pct=True).fillna(0.5).values

        return pd.Series(result, index=series.index)

    def _compute_dominant_periods(self, series: pd.Series) -> Tuple[int, int]:
        """
        Use Fourier Transform (DSP) to find dominant periods in the data.

        Physics: FFT decomposes the signal into frequency components.
        The dominant frequencies tell us the natural cycles/windows.

        Returns (short_period, long_period) - the two most significant cycles.
        """

        # Clean series
        clean = series.dropna().values
        n = len(clean)

        if n < 50:
            # Not enough data
            return (max(5, n // 10), max(20, n // 3))

        # Remove trend (detrend) - we want cycles, not drift
        detrended = clean - np.linspace(clean[0], clean[-1], n)

        # Apply FFT
        fft_vals = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(n)

        # Power spectrum (magnitude squared)
        power = np.abs(fft_vals) ** 2

        # Only look at positive frequencies, skip DC component (index 0)
        # and very high frequencies (noise)
        # Heuristic bounds for detectable periods (in samples):
        # - MIN_MEANINGFUL_PERIOD_SAMPLES: require several samples per cycle
        #   (>= 2 per Nyquist; we use 5 to reduce very high-frequency noise).
        # - MAX_PERIOD_FRACTION_OF_WINDOW: require that a cycle repeats
        #   multiple times within the analysis window (here ~3 times).
        MIN_MEANINGFUL_PERIOD_SAMPLES = 5
        MAX_PERIOD_FRACTION_OF_WINDOW = 1.0 / 3.0

        min_period = MIN_MEANINGFUL_PERIOD_SAMPLES
        max_period = max(int(n * MAX_PERIOD_FRACTION_OF_WINDOW), min_period + 1)
        # Convert frequency to period, find peaks
        periods = []
        powers = []

        for i in range(1, n // 2):
            freq = abs(freqs[i])
            if freq > 0:
                period = int(1 / freq)
                if min_period <= period <= max_period:
                    periods.append(period)
                    powers.append(power[i])

        if not periods:
            return (max(5, n // 20), max(50, n // 5))

        # Find top 2 dominant periods
        sorted_idx = np.argsort(powers)[::-1]

        # Get unique periods (avoid duplicates from harmonics)
        seen_periods = set()
        dominant = []

        for idx in sorted_idx:
            p = periods[idx]
            # Skip if too close to already seen period (harmonic)
            if not any(abs(p - sp) < p * 0.3 for sp in seen_periods):
                dominant.append(p)
                seen_periods.add(p)
            if len(dominant) >= 2:
                break

        # Ensure we have two periods
        if len(dominant) == 0:
            dominant = [max(5, n // 20), max(50, n // 5)]
        elif len(dominant) == 1:
            dominant.append(dominant[0] * 5)

        # Sort: short first, long second
        dominant.sort()

        return (dominant[0], dominant[1])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all physics features as percentiles."""
        result = df.copy()

        # === ADAPTIVE WINDOW SIZES VIA DSP ===
        # Use Fourier Transform to find dominant cycles in the data
        # No hardcoded windows - derived from spectral analysis

        velocity = df['close'].pct_change()

        # FFT-based dominant period detection
        short_period, long_period = self._compute_dominant_periods(velocity)

        # Store for reference
        self.short_period = short_period
        self.long_period = long_period

        # Use DSP-derived windows
        lookback = short_period
        window = long_period

        # === SPECTRAL FEATURES (DSP) ===
        # Add the dominant periods as features - let RL know the current cycle structure
        result['dominant_short_period'] = float(short_period)
        result['dominant_long_period'] = float(long_period)
        result['period_ratio'] = float(long_period) / float(short_period)  # Dimensionless

        # === CORE PHYSICS ===
        result['energy'] = 0.5 * (velocity ** 2)

        vol = velocity.rolling(lookback).std()
        mean_abs = velocity.abs().rolling(lookback).mean()
        result['damping'] = vol / (mean_abs + 1e-10)

        # Entropy (return dispersion)
        result['entropy'] = velocity.rolling(lookback).std() / (velocity.rolling(lookback).mean().abs() + 1e-10)

        # === DERIVATIVES ===
        result['acceleration'] = velocity.diff()
        result['jerk'] = result['acceleration'].diff()

        momentum = df['close'].pct_change(lookback)
        result['impulse'] = momentum.diff(5)

        # === ORDER FLOW ===
        bar_range = (df['high'] - df['low']).clip(lower=1e-10)
        result['liquidity'] = df['volume'] / (bar_range * df['close'] / 100)

        bp = (df['close'] - df['low']) / bar_range
        result['buying_pressure'] = bp.rolling(5).mean()

        # === FLOW DYNAMICS ===
        bar_range_pct = bar_range / df['close']
        volatility = velocity.rolling(lookback).std().clip(lower=1e-10)
        volume_norm = df['volume'] / df['volume'].rolling(lookback).mean().clip(lower=1e-10)

        # Reynolds number
        result['reynolds'] = ((velocity.abs() * bar_range_pct * volume_norm) / volatility).rolling(lookback).mean()

        # Viscosity
        result['viscosity'] = (bar_range_pct / volume_norm.clip(lower=1e-10)).rolling(lookback).mean()

        # === ROTATIONAL/CYCLICAL ===
        price_mean = df['close'].rolling(lookback).mean()
        price_std = df['close'].rolling(lookback).std().clip(lower=1e-10)
        angular_pos = (df['close'] - price_mean) / price_std
        angular_vel = angular_pos.diff()
        result['angular_momentum'] = angular_pos * angular_vel

        # === POTENTIAL ENERGY ===
        avg_range = bar_range.rolling(lookback).mean()
        range_compression = (1 - bar_range / avg_range.clip(lower=1e-10)).clip(lower=0)
        result['potential_energy'] = range_compression * volatility

        # === MARKET TORQUE (bull/bear tension creating rotational force) ===
        # From physics: torque = imbalance × acceleration
        # High imbalance + high acceleration = price trajectory "bends" sharply
        imbalance = (result['buying_pressure'] - 0.5) * 2  # Map 0-1 to -1 to +1
        result['torque'] = imbalance * result['acceleration'] * 1000  # Scale for visibility

        # === MARKET REYNOLDS NUMBER (alternative formulation) ===
        # Re = momentum / (viscosity × friction)
        # Low Re = laminar (trend), High Re = turbulent (chaos)
        result['market_reynolds'] = result['energy'] / (result['viscosity'] * result['damping'] + 1e-10)

        # === CONTEXT ===
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()
        result['range_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

        return_sign = np.sign(velocity)
        result['flow_consistency'] = return_sign.rolling(5).apply(
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
        )

        result['roc'] = df['close'].pct_change(lookback)

        # Momentum direction (let RL learn what to do with it)
        result['momentum_dir'] = np.sign(df['close'].pct_change(5))

        # === ENERGY RELEASE DETECTION (Entry Timing) ===

        # Volume compression (declining volume = coiling)
        result['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(lookback).mean().clip(lower=1e-10)

        # === TRUE PHYSICS COMPRESSION (not just Bollinger Bands) ===

        # 1. PHASE SPACE CONFINEMENT
        # Compression = both price AND momentum confined to small region
        price_norm = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std().clip(lower=1e-10)
        mom_norm = (velocity - velocity.rolling(50).mean()) / velocity.rolling(50).std().clip(lower=1e-10)

        # Bounding box area in phase space
        price_range = price_norm.rolling(lookback).max() - price_norm.rolling(lookback).min()
        mom_range = mom_norm.rolling(lookback).max() - mom_norm.rolling(lookback).min()
        phase_area = (price_range * mom_range).clip(lower=1e-10)
        result['phase_compression'] = 1 / (1 + phase_area)  # Higher = more compressed

        # 2. SUPPRESSION RATIO (Hidden Force Imbalance)
        # High volume + low energy = energy being absorbed, not expressed
        # Using physics energy instead of ATR
        result['suppression_ratio'] = df['volume'] / (result['energy'].clip(lower=1e-10) * df['close'] * 10000)
        # Normalize: high ratio = lots of volume, little movement = compressed

        # 3. APPROXIMATE ENTROPY (Entropy Collapse)
        # Low entropy = price path is predictable, repetitive, confined
        # Using simplified rolling entropy approximation
        returns_abs = velocity.abs()
        returns_std = returns_abs.rolling(lookback).std().clip(lower=1e-10)
        returns_mean = returns_abs.rolling(lookback).mean().clip(lower=1e-10)
        # Coefficient of variation as entropy proxy (lower = more uniform = compressed)
        result['entropy_proxy'] = returns_std / returns_mean

        # 4. SPRING STIFFNESS (k = Force / Displacement)
        # High k = deep book, price doesn't move easily
        # k = volume / price_change → high = stiff market
        result['spring_stiffness'] = df['volume'] / (velocity.abs().clip(lower=1e-10) * df['close'])

        # NO COMPOSITE RULES - RL discovers patterns from raw percentiles
        # All physics compression metrics are provided as percentiles
        # Neural network learns what combinations matter

        # 2. TRIGGER METRICS (Release Begins)
        # Range breakout - close beyond N-day high/low
        result['at_range_high'] = (df['close'] >= rolling_high.shift(1)).astype(float)
        result['at_range_low'] = (df['close'] <= rolling_low.shift(1)).astype(float)

        # Volume spike - current vs average
        result['volume_spike'] = df['volume'] / df['volume'].rolling(lookback).mean().clip(lower=1e-10)

        # Candle body ratio (strong candle = body > 70% of range)
        body = (df['close'] - df['open']).abs()
        result['body_ratio'] = body / bar_range.clip(lower=1e-10)

        # Liquidity sweep (wick beyond range then close inside or beyond)
        result['swept_high'] = ((df['high'] > rolling_high.shift(1)) & (df['close'] > df['open'])).astype(float)
        result['swept_low'] = ((df['low'] < rolling_low.shift(1)) & (df['close'] < df['open'])).astype(float)

        # 3. ACCELERATION METRICS (Momentum is Real)
        # ROC acceleration - is momentum increasing?
        roc_5 = df['close'].pct_change(5)
        result['roc_5'] = roc_5
        result['roc_accel'] = roc_5 - roc_5.shift(1)  # Positive = accelerating

        # Volume-weighted ROC (move has fuel)
        result['vol_weighted_roc'] = roc_5 * volume_norm

        # NO STATIC ENERGY RELEASE RULES
        # RL learns from: compression percentiles + trigger features + acceleration
        # Combinations are discovered, not prescribed

        # === PARETO ANALYSIS: FAT CANDLE DETECTION ===
        # "Fat" is RELATIVE to this instrument's distribution at this time
        # What's fat for crypto is different from gold is different from forex
        # Summer volatility differs from winter
        # Solution: Rolling percentile of candle magnitude

        # Candle magnitude (absolute move size)
        candle_magnitude = bar_range / df['close']  # Normalized by price
        result['candle_magnitude'] = candle_magnitude

        # Where does THIS candle sit in the instrument's recent distribution?
        # This adapts to the instrument AND the current volatility regime
        result['candle_magnitude_pct'] = self._fast_rolling_percentile(candle_magnitude, window, lookback)

        # Fat candle = top 20% of distribution (Pareto: 20% of candles = 80% of gains)
        # This is a FACT about the distribution, not a trading rule
        result['is_fat_candle'] = (result['candle_magnitude_pct'] >= 0.8).astype(float)

        # Energy magnitude (velocity squared, captures directional energy)
        energy_magnitude = velocity.abs()
        result['energy_magnitude_pct'] = self._fast_rolling_percentile(energy_magnitude, window, lookback)

        # === PRE-FAT CANDLE PHYSICS STATE ===
        # What did physics look like BEFORE fat candles?
        # Shift compression metrics forward to align with fat candle occurrence

        # Was there compression before this bar?
        result['prior_phase_compression'] = result['phase_compression'].shift(1)
        result['prior_suppression'] = result['suppression_ratio'].shift(1)
        result['prior_entropy'] = result['entropy_proxy'].shift(1)
        result['prior_stiffness'] = result['spring_stiffness'].shift(1)

        # Rolling compression buildup (was compression increasing over last N bars?)
        result['compression_buildup'] = result['phase_compression'].rolling(5).apply(
            lambda x: (x.iloc[-1] > x.iloc[0]) if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # === SEASONALITY / REGIME ADAPTATION ===
        # Use multiple rolling windows to capture different time scales
        # Short window = recent regime, Long window = seasonal baseline

        # Short-term volatility regime (recent 20 bars)
        short_vol = velocity.abs().rolling(lookback).mean()
        # Long-term volatility baseline (500 bars ~ captures seasonality)
        long_vol = velocity.abs().rolling(window).mean()

        # Volatility regime ratio: is current vol above/below seasonal baseline?
        result['vol_regime_ratio'] = short_vol / long_vol.clip(lower=1e-10)

        # Where in the seasonal distribution is current volatility?
        result['vol_regime_pct'] = self._fast_rolling_percentile(result['vol_regime_ratio'], window, lookback)

        # === CONVERT TO PERCENTILES ===
        feature_cols = [
            # Core physics
            'energy', 'damping', 'entropy', 'acceleration', 'jerk', 'impulse',
            'liquidity', 'buying_pressure', 'reynolds', 'viscosity',
            'angular_momentum', 'potential_energy', 'torque', 'market_reynolds',
            'range_position', 'flow_consistency', 'roc',
            # TRUE physics compression
            'phase_compression', 'suppression_ratio', 'entropy_proxy', 'spring_stiffness',
            # Energy release detection
            'volume_trend', 'volume_spike', 'body_ratio',
            'roc_5', 'roc_accel', 'vol_weighted_roc',
            # Pareto / Fat candle analysis
            'candle_magnitude', 'vol_regime_ratio',
            # Prior state (for learning pre-fat patterns)
            'prior_phase_compression', 'prior_suppression', 'prior_entropy', 'prior_stiffness',
        ]

        # FAST: Use vectorized percentile instead of slow lambda
        for col in feature_cols:
            result[f'{col}_pct'] = self._fast_rolling_percentile(result[col], window, lookback)

        # Keep momentum_dir as-is (not percentile)
        result['momentum_dir_pct'] = (result['momentum_dir'] + 1) / 2  # Map -1,0,1 to 0,0.5,1

        # Binary observations - keep as 0/1 (factual, not thresholds)
        result['at_range_high_pct'] = result['at_range_high']  # Fact: at N-day high
        result['at_range_low_pct'] = result['at_range_low']    # Fact: at N-day low
        result['swept_high_pct'] = result['swept_high']        # Fact: wick swept high
        result['swept_low_pct'] = result['swept_low']          # Fact: wick swept low

        # Pareto/fat candle features (already computed as percentiles)
        result['is_fat_candle_pct'] = result['is_fat_candle']  # Binary: top 20% of distribution
        result['compression_buildup_pct'] = result['compression_buildup']  # Was compression increasing?
        # candle_magnitude_pct, energy_magnitude_pct, vol_regime_pct already computed above

        # FRICTION/LIQUIDITY METRICS (from symbol_info if available, else estimate)
        # These replace "time of day" filters - we use market friction instead
        if 'spread' in df.columns:
            result['spread'] = df['spread']
            result['spread_pct'] = self._fast_rolling_percentile(result['spread'], window, lookback)
        else:
            # Estimate spread from bar range during low-volume periods
            # High range / low volume = wide effective spread
            vol_low = df['volume'] < df['volume'].rolling(lookback).quantile(0.3)
            result['spread_pct'] = vol_low.astype(float).rolling(5).mean().fillna(0.5)

        # LIQUIDITY / FRICTION - raw percentiles for RL
        # NO composite "fragile regime" rules - RL learns from raw features
        result['liquidity_pct'] = self._fast_rolling_percentile(df['volume'], window, lookback)

        return result.fillna(0.5)


if TORCH_AVAILABLE:

    class DQN(nn.Module):
        """Deep Q-Network for physics-based trading."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 64, 32)):
            super().__init__()

            layers = []
            prev_size = state_dim

            for size in hidden_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_size = size

            layers.append(nn.Linear(prev_size, action_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class ReplayBuffer:
        """Experience replay buffer."""

        def __init__(self, capacity: int, device: torch.device):
            self.capacity = capacity
            self.device = device
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size: int):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]

            states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)


    class TradingEnv:
        """Trading environment with all physics features."""

        # Actions: HOLD, LONG, SHORT, EXIT
        HOLD = 0
        LONG = 1
        SHORT = 2
        EXIT = 3

        def __init__(self, df: pd.DataFrame, feature_df: pd.DataFrame):
            self.df = df
            self.feature_df = feature_df
            self.feature_cols = [c for c in feature_df.columns if c.endswith('_pct')]

            self.state_dim = len(self.feature_cols) + 4  # + position info
            self.action_dim = 4

            # DIAGNOSTIC: Track action distribution
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            self.reset()

        def reset(self) -> np.ndarray:
            self.idx = 100
            self.position = 0  # -1 short, 0 flat, 1 long
            self.entry_price = 0.0
            self.entry_bar = 0
            self.mfe = 0.0
            self.mae = 0.0
            self.trades = []
            # DIAGNOSTIC: Reset action counts
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            # DIAGNOSTIC: Track entry quality
            self.entry_energy_pct = 0.0
            self.entry_candle_position = 0.0  # 0=low, 1=high of entry candle
            self.move_high = 0.0  # Highest point after entry
            self.move_low = 0.0   # Lowest point after entry

            # POTENTIAL-BASED SHAPING: Track previous state for delta rewards
            self.prev_unrealized_pnl = 0.0
            self.prev_mfe = 0.0

            return self._get_state()

        def _get_state(self) -> np.ndarray:
            if self.idx >= len(self.feature_df):
                return np.zeros(self.state_dim)

            features = self.feature_df.iloc[self.idx][self.feature_cols].values.astype(float)

            # Position info
            pos_dir = float(self.position)
            bars_held = float(self.idx - self.entry_bar) / 10 if self.position != 0 else 0

            if self.position != 0:
                current_price = self.df.iloc[self.idx]['close']
                pnl = (current_price - self.entry_price) / self.entry_price * self.position
            else:
                pnl = 0.0

            mfe_ratio = self.mfe / (self.mae + 1e-6) if self.mae > 0 else 0

            return np.concatenate([features, [pos_dir, bars_held, pnl, mfe_ratio]])

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
            reward = 0.0
            info = {}

            # DIAGNOSTIC: Track action distribution
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

            current = self.df.iloc[self.idx]
            high, low, close = current['high'], current['low'], current['close']
            candle_range = high - low if high > low else 0.0001

            # === POTENTIAL-BASED REWARD SHAPING ===
            # Compute current unrealized PnL for delta reward
            if self.position != 0:
                current_unrealized = (close - self.entry_price) / self.entry_price * 100 * self.position
            else:
                current_unrealized = 0.0

            # Update MFE/MAE and move tracking if in position
            prev_mfe = self.mfe  # Store before update
            if self.position != 0:
                self.move_high = max(self.move_high, high)
                self.move_low = min(self.move_low, low)

                if self.position == 1:
                    self.mfe = max(self.mfe, (high - self.entry_price) / self.entry_price * 100)
                    self.mae = max(self.mae, (self.entry_price - low) / self.entry_price * 100)
                else:
                    self.mfe = max(self.mfe, (self.entry_price - low) / self.entry_price * 100)
                    self.mae = max(self.mae, (high - self.entry_price) / self.entry_price * 100)

            # Execute action
            if action == self.LONG and self.position == 0:
                self.position = 1
                self.entry_price = close
                self.entry_bar = self.idx
                self.mfe = 0.0
                self.mae = 0.0
                # DIAGNOSTIC: Entry quality
                self.entry_energy_pct = self.feature_df.iloc[self.idx].get('energy_pct', 0.5)
                self.entry_candle_position = (close - low) / candle_range  # Where in candle we entered
                self.move_high = high
                self.move_low = low
                # Reset potential tracking for new trade
                self.prev_unrealized_pnl = 0.0
                self.prev_mfe = 0.0

            elif action == self.SHORT and self.position == 0:
                self.position = -1
                self.entry_price = close
                self.entry_bar = self.idx
                self.mfe = 0.0
                self.mae = 0.0
                # DIAGNOSTIC: Entry quality
                self.entry_energy_pct = self.feature_df.iloc[self.idx].get('energy_pct', 0.5)
                self.entry_candle_position = (high - close) / candle_range  # For short, high is bad
                self.move_high = high
                self.move_low = low
                # Reset potential tracking for new trade
                self.prev_unrealized_pnl = 0.0
                self.prev_mfe = 0.0

            elif action == self.EXIT and self.position != 0:
                pnl = (close - self.entry_price) / self.entry_price * 100 * self.position

                # DIAGNOSTIC: Calculate move capture %
                total_move = self.move_high - self.move_low
                if self.position == 1:
                    captured_move = close - self.entry_price
                    potential_move = self.move_high - self.entry_price
                else:
                    captured_move = self.entry_price - close
                    potential_move = self.entry_price - self.move_low
                move_capture_pct = (captured_move / potential_move * 100) if potential_move > 0 else 0

                # MFE/MAE efficiency (dimensionless ratio)
                mfe_capture = pnl / self.mfe if self.mfe > 0.01 else 0
                efficiency = self.mfe / (self.mae + 1e-6)

                # DIAGNOSTIC: MFE came first or MAE came first?
                mfe_first = self.mfe > self.mae

                # === OMEGA REWARD (Pythagorean Path Efficiency) ===
                #
                # Goal: Maximum displacement from entry via shortest path
                #
                # Physics: The agent traveled through (MFE, MAE) space.
                # Total excursion = sqrt(MFE² + MAE²) [Pythagorean hypotenuse]
                # Final displacement = |PnL|
                #
                # Path efficiency = displacement / total_excursion
                #
                # ω = PnL × (|PnL| / sqrt(MFE² + MAE²))
                #   = PnL² / sqrt(MFE² + MAE²)  [preserving sign via PnL]
                #
                # Interpretation:
                # - Clean move (MFE only): omega = pnl² / MFE ≈ MFE (high reward)
                # - Whipsaw (high MAE): omega = pnl² / large → small reward
                # - Loss with excursion: omega = negative / large → penalty
                #
                # No static weights - pure geometry

                total_excursion = np.sqrt(self.mfe**2 + self.mae**2 + 1e-12)
                path_efficiency = abs(pnl) / total_excursion

                # Signed omega: PnL determines direction, efficiency scales magnitude
                omega = pnl * path_efficiency

                reward = omega

                trade_info = {
                    'pnl': pnl,
                    'mfe': self.mfe,
                    'mae': self.mae,
                    'mfe_capture': mfe_capture * 100,
                    'efficiency': efficiency,
                    'bars': self.idx - self.entry_bar,
                    # NEW DIAGNOSTICS
                    'entry_energy_pct': self.entry_energy_pct,
                    'entry_candle_pos': self.entry_candle_position,
                    'move_capture_pct': move_capture_pct,
                    'mfe_first': mfe_first,
                    'direction': 'long' if self.position == 1 else 'short',
                }
                self.trades.append(trade_info)

                # Return trade info for logging
                info['trade_pnl'] = pnl
                info['mfe'] = self.mfe
                info['mae'] = self.mae
                info['entry_energy'] = self.entry_energy_pct
                info['move_capture'] = move_capture_pct
                info['mfe_first'] = mfe_first

                self.position = 0

            elif action == self.HOLD and self.position != 0:
                # === POTENTIAL-BASED SHAPING (Dense Reward Signal) ===
                #
                # Problem: Sparse rewards (only on EXIT) → slow learning
                # Solution: Reward = change in potential function Φ
                #
                # Φ(s) = unrealized_pnl + mfe_bonus
                # Shaping reward = Φ(s') - Φ(s) = Δunrealized + Δmfe
                #
                # This gives immediate feedback without changing optimal policy
                # (Ng et al. 1999: potential-based shaping is policy invariant)

                # Delta unrealized PnL: moving in right direction?
                delta_pnl = current_unrealized - self.prev_unrealized_pnl

                # Delta MFE: reaching new profit highs?
                delta_mfe = self.mfe - prev_mfe

                # Combined shaping reward (no static weights - pure deltas)
                # Both are in same units (% of price), naturally scaled
                reward = delta_pnl + delta_mfe

                # Update previous state for next step
                self.prev_unrealized_pnl = current_unrealized
                self.prev_mfe = self.mfe

            # Move to next bar
            self.idx += 1
            done = self.idx >= len(self.df) - 1

            # Force close at end
            if done and self.position != 0:
                pnl = (self.df.iloc[self.idx]['close'] - self.entry_price) / self.entry_price * 100 * self.position
                reward = pnl
                self.trades.append({
                    'pnl': pnl, 'mfe': self.mfe, 'mae': self.mae,
                    'entry_energy_pct': self.entry_energy_pct,
                    'forced_close': True
                })
                info['trade_pnl'] = pnl
                info['mfe'] = self.mfe
                info['mae'] = self.mae
                self.position = 0

            return self._get_state(), reward, done, info

        def get_stats(self) -> Dict:
            if not self.trades:
                return {'trades': 0}

            pnls = [t['pnl'] for t in self.trades]
            mfes = [t['mfe'] for t in self.trades]
            maes = [t['mae'] for t in self.trades]

            stats = {
                'trades': len(self.trades),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'avg_mfe': np.mean(mfes),
                'avg_mae': np.mean(maes),
                'mfe_mae_ratio': np.mean(mfes) / (np.mean(maes) + 0.01),
            }

            # NEW DIAGNOSTICS
            # Entry energy - are we entering on high or low energy candles?
            entry_energies = [t.get('entry_energy_pct', 0.5) for t in self.trades if 'entry_energy_pct' in t]
            if entry_energies:
                stats['avg_entry_energy'] = np.mean(entry_energies)

            # Move capture - what % of the move after entry did we capture?
            move_captures = [t.get('move_capture_pct', 0) for t in self.trades if 'move_capture_pct' in t]
            if move_captures:
                stats['avg_move_capture'] = np.mean(move_captures)

            # MFE first ratio - how often did profit come before drawdown?
            mfe_firsts = [t.get('mfe_first', False) for t in self.trades if 'mfe_first' in t]
            if mfe_firsts:
                stats['mfe_first_ratio'] = sum(mfe_firsts) / len(mfe_firsts) * 100

            # Entry candle position - are we chasing (entering late in candle)?
            entry_positions = [t.get('entry_candle_pos', 0.5) for t in self.trades if 'entry_candle_pos' in t]
            if entry_positions:
                stats['avg_entry_candle_pos'] = np.mean(entry_positions)

            # Action distribution
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                stats['hold_pct'] = self.action_counts[0] / total_actions * 100
                stats['long_pct'] = self.action_counts[1] / total_actions * 100
                stats['short_pct'] = self.action_counts[2] / total_actions * 100
                stats['exit_pct'] = self.action_counts[3] / total_actions * 100

            # Winning vs losing trade diagnostics
            winning = [t for t in self.trades if t['pnl'] > 0]
            losing = [t for t in self.trades if t['pnl'] <= 0]

            if winning:
                stats['winners_avg_mfe'] = np.mean([t['mfe'] for t in winning])
                stats['winners_avg_mae'] = np.mean([t['mae'] for t in winning])
                stats['winners_avg_bars'] = np.mean([t.get('bars', 0) for t in winning])
            if losing:
                stats['losers_avg_mfe'] = np.mean([t['mfe'] for t in losing])
                stats['losers_avg_mae'] = np.mean([t['mae'] for t in losing])
                stats['losers_avg_bars'] = np.mean([t.get('bars', 0) for t in losing])

            return stats


    class GPUTrainer:
        """GPU-accelerated DQN trainer."""

        def __init__(self, config: TrainingConfig = None):
            self.config = config or TrainingConfig()

            # Detect device
            if self.config.device == 'auto':
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device('cpu')
                    print("Using CPU")
            else:
                self.device = torch.device(self.config.device)

            self.policy_net = None
            self.target_net = None
            self.optimizer = None
            self.buffer = None
            self.epsilon = self.config.epsilon_start

        def init_networks(self, state_dim: int, action_dim: int):
            """Initialize networks for given dimensions."""
            self.policy_net = DQN(state_dim, action_dim, self.config.hidden_sizes).to(self.device)
            self.target_net = DQN(state_dim, action_dim, self.config.hidden_sizes).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
            self.buffer = ReplayBuffer(self.config.buffer_size, self.device)

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            if training and np.random.random() < self.epsilon:
                return np.random.randint(4)

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

        def train_step(self) -> float:
            if len(self.buffer) < self.config.batch_size:
                return 0.0

            states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

            # Current Q values
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Target Q values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + self.config.gamma * next_q * (1 - dones)

            # Loss
            loss = F.mse_loss(q_values.squeeze(), target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()

        def update_target(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        def train(self, envs: List[TradingEnv], n_episodes: int = None) -> List[Dict]:
            """Train on multiple environments (instruments/timeframes)."""
            n_episodes = n_episodes or self.config.n_episodes

            # Initialize networks from first env
            self.init_networks(envs[0].state_dim, envs[0].action_dim)

            stats = []
            step_count = 0

            for episode in range(n_episodes):
                episode_reward = 0
                episode_loss = 0
                n_updates = 0

                # Train on each environment
                for env in envs:
                    state = env.reset()

                    while True:
                        action = self.select_action(state)
                        next_state, reward, done, _ = env.step(action)

                        self.buffer.push(state, action, reward, next_state, done)

                        loss = self.train_step()
                        if loss > 0:
                            episode_loss += loss
                            n_updates += 1

                        step_count += 1
                        if step_count % self.config.target_update_freq == 0:
                            self.update_target()

                        state = next_state
                        episode_reward += reward

                        if done:
                            break

                self.decay_epsilon()

                # Aggregate stats across envs
                avg_stats = {
                    'episode': episode,
                    'total_reward': episode_reward,
                    'avg_loss': episode_loss / n_updates if n_updates > 0 else 0,
                    'epsilon': self.epsilon,
                    'trades': sum(e.get_stats().get('trades', 0) for e in envs),
                    'avg_pnl': np.mean([e.get_stats().get('avg_pnl', 0) for e in envs if e.get_stats().get('trades', 0) > 0]),
                    'avg_win_rate': np.mean([e.get_stats().get('win_rate', 50) for e in envs if e.get_stats().get('trades', 0) > 0]),
                }
                stats.append(avg_stats)

                if (episode + 1) % 20 == 0:
                    print(f"Episode {episode + 1}/{n_episodes}: "
                          f"Trades={avg_stats['trades']}, "
                          f"WinRate={avg_stats['avg_win_rate']:.1f}%, "
                          f"PnL={avg_stats['avg_pnl']:.4f}%, "
                          f"ε={self.epsilon:.3f}")

            return stats

        def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
            """Get feature importance from first layer weights."""
            if self.policy_net is None:
                return pd.DataFrame()

            with torch.no_grad():
                weights = self.policy_net.network[0].weight.cpu().numpy()

            importance = np.abs(weights).mean(axis=0)
            importance = importance / importance.sum()

            return pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)


def train_across_instruments(
    data_paths: List[str],
    config: TrainingConfig = None,
) -> Tuple['GPUTrainer', List[Dict]]:
    """
    Train RL across multiple instruments/timeframes.

    Let RL discover:
    - Fat candle probability
    - Continuation vs reversal
    - Optimal exit timing
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    config = config or TrainingConfig()
    feature_computer = PhysicsFeatureComputer()

    # Load and prepare all environments
    envs = []
    for path in data_paths:
        print(f"Loading: {Path(path).name}")

        # Load data (assuming CSV format)
        df = pd.read_csv(path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        # Compute features
        feature_df = feature_computer.compute(df)

        # Create environment
        env = TradingEnv(df, feature_df)
        envs.append(env)

    print(f"\nTraining on {len(envs)} instruments/timeframes")
    print(f"State dimension: {envs[0].state_dim}")
    print(f"Action dimension: {envs[0].action_dim}")

    # Train
    trainer = GPUTrainer(config)
    stats = trainer.train(envs, config.n_episodes)

    # Feature importance
    feature_names = [c.replace('_pct', '') for c in envs[0].feature_cols] + ['pos_dir', 'bars_held', 'pnl', 'mfe_ratio']
    importance = trainer.get_feature_importance(feature_names)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (What RL learned)")
    print("=" * 60)
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:>20}: {row['importance']:.1%}")

    return trainer, stats


if __name__ == "__main__":
    # Test with BTCUSD data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))

    if csv_files:
        trainer, stats = train_across_instruments(
            [str(f) for f in csv_files],
            TrainingConfig(n_episodes=50)
        )
    else:
        print("No CSV files found")
