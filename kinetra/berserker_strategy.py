"""
Berserker Strategy: Fat Candle Hunter

Berserker is a FAT CANDLE HUNTER - not a trend follower or mean reverter.
It predicts WHEN fat candles occur, then determines direction separately.

PHASE 1: FAT CANDLE PROBABILITY (Berserker Detection)
- High energy + low damping = high probability of fat candle
- Enhanced by: jerk (1.37x), impulse (1.30x), liquidity (1.34x)
- Does NOT predict direction

PHASE 2: DIRECTION PREDICTION (Continuation vs Reversal)
- Laminar flow → CONTINUATION (trend will continue in fat candle)
- Turbulent flow → REVERSAL (trend will reverse in fat candle)
- Buying pressure confirms direction edge

PHASE 3: TRADE MANAGEMENT
- Let fat candles RUN - wide initial stop, trail after profit
- Don't exit prematurely - fat candles need room to develop
- Calibrated from MFE/MAE analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .physics_engine import PhysicsEngine


class Direction(Enum):
    NEUTRAL = "NEUTRAL"
    BUY = "BUY"
    SELL = "SELL"


class FlowRegime(Enum):
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


@dataclass
class TriggerPrediction:
    """Prediction output with magnitude, direction, and trade management."""
    # Magnitude
    probability: float  # 0-1 probability of fat candle
    magnitude_lift: float  # Expected lift vs baseline

    # Direction
    direction: Direction
    direction_confidence: float  # 0-1

    # Trade management
    flow_regime: FlowRegime
    trailing_stop_pct: float  # Adaptive trailing stop %
    initial_stop_pct: float  # Initial stop loss %
    target_pct: float  # Take profit target %

    # Meta
    confidence: str  # LOW, MEDIUM, HIGH, BERSERKER
    conditions_met: List[str]
    contributing_factors: Dict[str, float]
    message: str


@dataclass
class TradeExit:
    """Exit decision from adaptive trailing stop."""
    should_exit: bool
    reason: str  # 'trailing_stop', 'target', 'stop_loss', 'regime_change', 'hold'
    exit_price: Optional[float]
    pnl_pct: float
    mfe_captured_pct: float  # How much of MFE we captured


class PhysicsFeatures:
    """Compute all physics features for prediction."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.engine = PhysicsEngine(lookback=lookback)
        self.percentile_window = 500

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all physics features as adaptive percentiles."""
        result = df.copy()
        window = min(self.percentile_window, len(df))

        # === CORE PHYSICS ===
        physics = self.engine.compute_physics_state(
            df['close'], df['volume'], include_percentiles=True
        )
        result['energy'] = physics['energy']
        result['damping'] = physics['damping']
        result['entropy'] = physics['entropy']
        result['energy_pct'] = physics['energy_pct']
        result['damping_pct'] = physics['damping_pct']
        result['entropy_pct'] = physics['entropy_pct']

        # === DERIVATIVES ===
        # Acceleration
        velocity = df['close'].pct_change()
        result['acceleration'] = velocity.diff()

        # Jerk (1.37x fat candle lift)
        result['jerk'] = result['acceleration'].diff()
        result['jerk_pct'] = result['jerk'].abs().rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # Impulse (1.30x lift)
        momentum = df['close'].pct_change(self.lookback)
        result['impulse'] = momentum.diff(5)
        result['impulse_pct'] = result['impulse'].abs().rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # === ORDER FLOW ===
        # Liquidity (1.34x lift at berserker)
        bar_range = (df['high'] - df['low']).clip(lower=1e-10)
        result['liquidity'] = df['volume'] / (bar_range * df['close'] / 100)
        result['liquidity_pct'] = result['liquidity'].rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # Buying pressure (+12% direction edge)
        bp = (df['close'] - df['low']) / bar_range
        result['buying_pressure'] = bp.rolling(5).mean().fillna(0.5)

        # === FLOW REGIME (Reynolds) ===
        # Low Reynolds = laminar (smooth) = continuation
        # High Reynolds = turbulent (chaotic) = reversal
        volatility = velocity.rolling(self.lookback).std().clip(lower=1e-10)
        bar_range_pct = bar_range / df['close']
        volume_norm = df['volume'] / df['volume'].rolling(self.lookback).mean().clip(lower=1e-10)
        reynolds = (velocity.abs() * bar_range_pct * volume_norm) / volatility
        result['reynolds'] = reynolds.rolling(self.lookback).mean().fillna(1.0)
        result['reynolds_pct'] = result['reynolds'].rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # Viscosity
        avg_volume = df['volume'].rolling(self.lookback).mean().clip(lower=1e-10)
        volume_norm_v = df['volume'] / avg_volume
        result['viscosity'] = bar_range_pct / volume_norm_v.clip(lower=1e-10)
        result['viscosity'] = result['viscosity'].rolling(self.lookback).mean().fillna(1.0)
        result['viscosity_pct'] = result['viscosity'].rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # === CONTEXT ===
        # Flow consistency (laminar indicator)
        return_sign = np.sign(velocity)
        result['flow_consistency'] = return_sign.rolling(5).apply(
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
        ).fillna(0.5)

        # Momentum (for direction)
        result['momentum_5'] = df['close'].pct_change(5)

        # Inertia (bars same direction)
        direction = np.sign(velocity)
        counts = []
        count = 1
        for i in range(len(direction)):
            if i == 0:
                counts.append(1)
            elif direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
                count += 1
                counts.append(count)
            else:
                count = 1
                counts.append(count)
        result['inertia'] = pd.Series(counts, index=df.index)

        return result.fillna(0.5)


class CompositePredictor:
    """
    Composite trigger predictor using ALL empirically-validated signals.
    """

    # Empirical lift factors
    LIFTS = {
        'berserker_90_10': 1.83,
        'berserker_75_25': 1.54,
        'underdamped': 1.87,
        'high_jerk': 1.37,
        'strong_impulse': 1.30,
        'high_liquidity_berserker': 1.34,
        'energy_peaked': 1.43,
        'turbulent_berserker': 1.25,  # Turbulent = bigger but unpredictable
    }

    # Log-weights for multiplicative combination
    WEIGHTS = {k: np.log(v) for k, v in LIFTS.items()}

    # Direction confidence levels
    DIRECTION_CONF = {
        'base_berserker': 0.537,
        'laminar_berserker': 0.56,
        'high_bp_down': 0.62,  # High buying pressure → DOWN
        'low_bp_up': 0.57,    # Low buying pressure → UP
    }

    def __init__(self):
        self.features = PhysicsFeatures()

    def predict(
        self,
        energy_pct: float,
        damping_pct: float,
        entropy_pct: float,
        jerk_pct: float = 0.5,
        impulse_pct: float = 0.5,
        liquidity_pct: float = 0.5,
        reynolds_pct: float = 0.5,
        buying_pressure: float = 0.5,
        momentum: float = 0.0,
        flow_consistency: float = 0.5,
        inertia: int = 1,
    ) -> TriggerPrediction:
        """
        Generate composite prediction.

        All percentile inputs are [0, 1] - adaptive, no fixed thresholds.
        """
        conditions_met = []
        contributing_factors = {}
        composite_score = 0.0

        # === BERSERKER CONDITIONS ===
        is_berserker_90_10 = (energy_pct > 0.90) and (damping_pct < 0.10)
        is_berserker_75_25 = (energy_pct > 0.75) and (damping_pct < 0.25)

        if is_berserker_90_10:
            conditions_met.append('BERSERKER_90_10')
            weight = self.WEIGHTS['berserker_90_10']
            contributing_factors['berserker_90_10'] = weight
            composite_score += weight
        elif is_berserker_75_25:
            conditions_met.append('BERSERKER_75_25')
            weight = self.WEIGHTS['berserker_75_25']
            contributing_factors['berserker_75_25'] = weight
            composite_score += weight

        # === DERIVATIVE SIGNALS ===
        # High jerk (1.37x lift)
        if jerk_pct > 0.85:
            conditions_met.append('HIGH_JERK')
            weight = self.WEIGHTS['high_jerk']
            contributing_factors['high_jerk'] = weight
            composite_score += weight

        # Strong impulse (1.30x lift)
        if impulse_pct > 0.80:
            conditions_met.append('STRONG_IMPULSE')
            weight = self.WEIGHTS['strong_impulse']
            contributing_factors['strong_impulse'] = weight
            composite_score += weight

        # === ORDER FLOW ===
        # High liquidity at berserker (1.34x lift)
        if liquidity_pct > 0.70 and (is_berserker_90_10 or is_berserker_75_25):
            conditions_met.append('HIGH_LIQUIDITY_BERSERKER')
            weight = self.WEIGHTS['high_liquidity_berserker']
            contributing_factors['high_liquidity_berserker'] = weight
            composite_score += weight

        # === FLOW REGIME ===
        if reynolds_pct < 0.25:
            flow_regime = FlowRegime.LAMINAR
            conditions_met.append('LAMINAR_FLOW')
        elif reynolds_pct > 0.75:
            flow_regime = FlowRegime.TURBULENT
            conditions_met.append('TURBULENT_FLOW')
            if is_berserker_75_25:
                # Turbulent berserker = bigger moves
                weight = self.WEIGHTS['turbulent_berserker']
                contributing_factors['turbulent_berserker'] = weight
                composite_score += weight
        else:
            flow_regime = FlowRegime.TRANSITIONAL

        # === CONVERT TO PROBABILITY ===
        base_prob = 0.22  # Baseline fat candle rate
        if composite_score > 0:
            magnitude_lift = np.exp(composite_score)
            probability = min(0.85, base_prob * magnitude_lift)
        else:
            magnitude_lift = 1.0
            probability = base_prob * 0.5

        # === DIRECTION ===
        # Primary signal: BUYING PRESSURE (empirically validated +12% edge)
        # - High BP (>0.6) → exhausted buyers → SELL
        # - Low BP (<0.4) → exhausted sellers → BUY
        #
        # Flow regime modulates confidence:
        # - Transitional = best (buying pressure works)
        # - Laminar/Turbulent = less reliable
        direction_confidence = 0.50

        if is_berserker_90_10 or is_berserker_75_25:
            # BUYING PRESSURE is primary direction signal
            if buying_pressure > 0.60:
                direction = Direction.SELL
                direction_confidence = self.DIRECTION_CONF['high_bp_down']  # 62%
                conditions_met.append('FADE_HIGH_BP')
            elif buying_pressure < 0.40:
                direction = Direction.BUY
                direction_confidence = self.DIRECTION_CONF['low_bp_up']  # 57%
                conditions_met.append('FADE_LOW_BP')
            else:
                # Neutral BP - use momentum as tiebreaker (fade it)
                if momentum > 0.001:
                    direction = Direction.SELL
                    direction_confidence = self.DIRECTION_CONF['base_berserker']  # 53.7%
                    conditions_met.append('FADE_UP_MOMENTUM')
                elif momentum < -0.001:
                    direction = Direction.BUY
                    direction_confidence = self.DIRECTION_CONF['base_berserker']
                    conditions_met.append('FADE_DOWN_MOMENTUM')
                else:
                    direction = Direction.NEUTRAL

            # Boost confidence in transitional regime (where BP works best)
            if flow_regime == FlowRegime.TRANSITIONAL and direction != Direction.NEUTRAL:
                direction_confidence = min(0.65, direction_confidence + 0.05)
                conditions_met.append('TRANS_BOOST')
        else:
            # Non-berserker: no trade
            direction = Direction.NEUTRAL

        # === ADAPTIVE TRAILING STOP ===
        trailing_stop_pct, initial_stop_pct, target_pct = self._compute_stops(
            flow_regime, is_berserker_75_25, magnitude_lift
        )

        # === CONFIDENCE LEVEL ===
        if is_berserker_90_10:
            confidence = 'BERSERKER'
        elif is_berserker_75_25:
            confidence = 'HIGH'
        elif probability >= 0.35:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # === MESSAGE ===
        prob_pct = int(probability * 100)
        dir_conf_pct = int(direction_confidence * 100)

        if confidence in ['BERSERKER', 'HIGH']:
            message = (
                f"{confidence}: {prob_pct}% fat candle, {dir_conf_pct}% {direction.value}. "
                f"Flow: {flow_regime.value}. "
                f"Stops: TS={trailing_stop_pct:.2f}%, SL={initial_stop_pct:.2f}%, TP={target_pct:.2f}%"
            )
        else:
            message = f"{confidence}: {prob_pct}% probability. Monitoring."

        return TriggerPrediction(
            probability=probability,
            magnitude_lift=magnitude_lift,
            direction=direction,
            direction_confidence=direction_confidence,
            flow_regime=flow_regime,
            trailing_stop_pct=trailing_stop_pct,
            initial_stop_pct=initial_stop_pct,
            target_pct=target_pct,
            confidence=confidence,
            conditions_met=conditions_met,
            contributing_factors=contributing_factors,
            message=message,
        )

    def _compute_stops(
        self,
        flow_regime: FlowRegime,
        is_berserker: bool,
        magnitude_lift: float,
    ) -> Tuple[float, float, float]:
        """
        No fixed stops - return placeholders.
        Exit is based on energy recovery ratio.
        """
        # Placeholders - actual exit via EnergyRecoveryExit
        return 0.0, 0.0, 0.0


@dataclass
class HistoricalStats:
    """Rolling historical statistics for adaptive thresholds."""
    avg_mfe: float = 0.5
    avg_mae: float = 0.3
    mfe_mae_ratio: float = 1.5
    avg_bars_to_mfe: float = 3.0
    p75_mfe: float = 0.8
    p25_mae: float = 0.2


class EnergyRecoveryExit:
    """
    Energy-based exit logic - no fixed stops.

    Exit when:
    1. ENERGY DEPLETED: Energy drops from peak (move exhausted)
    2. EFFICIENCY RATIO: MAE/MFE efficiency threshold (dynamic from history)
    3. REGIME CHANGE: Flow regime shifts unfavorably

    All thresholds are ADAPTIVE from rolling history - no fixed values.
    """

    def __init__(
        self,
        entry_price: float,
        direction: int,  # 1 = long, -1 = short
        entry_energy_pct: float,  # Energy percentile at entry
        magnitude_lift: float,  # Expected lift from predictor
        flow_regime: FlowRegime,
        historical_stats: HistoricalStats,  # Dynamic thresholds from history
    ):
        self.entry_price = entry_price
        self.direction = direction
        self.entry_energy_pct = entry_energy_pct
        self.magnitude_lift = magnitude_lift
        self.flow_regime = flow_regime
        self.hist = historical_stats

        # Expected move from history * lift
        self.expected_mfe = self.hist.avg_mfe * magnitude_lift
        self.expected_mae = self.hist.avg_mae

        # Track trade state
        self.best_price = entry_price
        self.worst_price = entry_price
        self.mfe = 0.0
        self.mae = 0.0
        self.bars_held = 0
        self.peak_energy_pct = entry_energy_pct
        self.in_profit = False

    def update(
        self,
        high: float,
        low: float,
        close: float,
        current_energy_pct: float,
        current_damping_pct: float,
        current_reynolds_pct: float,
    ) -> TradeExit:
        """
        Check if should exit based on energy recovery.

        Args:
            high, low, close: Current bar prices
            current_energy_pct: Current energy percentile
            current_damping_pct: Current damping percentile
            current_reynolds_pct: Current Reynolds percentile
        """
        self.bars_held += 1

        # Update price extremes and MFE/MAE
        if self.direction == 1:  # Long
            self.best_price = max(self.best_price, high)
            self.worst_price = min(self.worst_price, low)
            self.mfe = max(self.mfe, (high - self.entry_price) / self.entry_price * 100)
            self.mae = max(self.mae, (self.entry_price - low) / self.entry_price * 100)
            current_pnl = (close - self.entry_price) / self.entry_price * 100
        else:  # Short
            self.best_price = min(self.best_price, low)
            self.worst_price = max(self.worst_price, high)
            self.mfe = max(self.mfe, (self.entry_price - low) / self.entry_price * 100)
            self.mae = max(self.mae, (high - self.entry_price) / self.entry_price * 100)
            current_pnl = (self.entry_price - close) / self.entry_price * 100

        self.in_profit = current_pnl > 0
        self.peak_energy_pct = max(self.peak_energy_pct, current_energy_pct)

        # === EFFICIENCY METRICS (Dynamic from history) ===
        # Current efficiency = MFE / MAE (higher = better trade quality)
        current_efficiency = self.mfe / self.mae if self.mae > 0.01 else float('inf')
        historical_efficiency = self.hist.mfe_mae_ratio

        # Recovery ratio vs expected (from rolling history)
        mfe_vs_expected = self.mfe / self.expected_mfe if self.expected_mfe > 0 else 0
        mae_vs_expected = self.mae / self.expected_mae if self.expected_mae > 0 else 0

        # Energy ratio
        energy_ratio = current_energy_pct / self.entry_energy_pct if self.entry_energy_pct > 0 else 1

        # === EXIT CONDITIONS (All adaptive from history) ===

        # 1. EFFICIENCY THRESHOLD: MAE exceeds historical tolerance
        if mae_vs_expected > 1.5 and not self.in_profit:
            # MAE 50% worse than historical average - exit
            return TradeExit(
                should_exit=True,
                reason='mae_exceeded',
                exit_price=close,
                pnl_pct=current_pnl,
                mfe_captured_pct=self._calc_mfe_captured(close),
            )

        # 2. MFE ACHIEVED + ENERGY DEPLETING: Captured expected move, energy fading
        if mfe_vs_expected >= 0.8 and energy_ratio < 0.6:
            # Hit 80% of expected MFE, energy dropped 40% - take profits
            return TradeExit(
                should_exit=True,
                reason='mfe_target_energy_fade',
                exit_price=close,
                pnl_pct=current_pnl,
                mfe_captured_pct=self._calc_mfe_captured(close),
            )

        # 3. EFFICIENCY DETERIORATING: Trade quality worsening
        if self.bars_held >= 2 and current_efficiency < historical_efficiency * 0.5:
            # Efficiency dropped to half of historical average
            if self.in_profit:
                return TradeExit(
                    should_exit=True,
                    reason='efficiency_deteriorating',
                    exit_price=close,
                    pnl_pct=current_pnl,
                    mfe_captured_pct=self._calc_mfe_captured(close),
                )

        # 4. ENERGY EXHAUSTED: Energy dropped from peak (berserker done)
        if self.peak_energy_pct > 0:
            energy_from_peak = current_energy_pct / self.peak_energy_pct
            # Dynamic threshold based on historical bars to MFE
            exhaustion_threshold = 0.4 if self.bars_held >= self.hist.avg_bars_to_mfe else 0.3
            if energy_from_peak < exhaustion_threshold and self.bars_held >= 2:
                if current_pnl > -self.hist.p25_mae:  # Allow small loss up to p25 MAE
                    return TradeExit(
                        should_exit=True,
                        reason='energy_exhausted',
                        exit_price=close,
                        pnl_pct=current_pnl,
                        mfe_captured_pct=self._calc_mfe_captured(close),
                    )

        # 5. MFE PROTECTION: Captured good move, now protect it
        if self.mfe > self.hist.p75_mfe:
            # Above 75th percentile MFE - protect gains
            mfe_captured_ratio = current_pnl / self.mfe if self.mfe > 0 else 0
            if mfe_captured_ratio < 0.25:  # Given back 75% of peak gains
                return TradeExit(
                    should_exit=True,
                    reason='mfe_protection',
                    exit_price=close,
                    pnl_pct=current_pnl,
                    mfe_captured_pct=mfe_captured_ratio * 100,
                )

        # 6. DAMPING SURGE: Friction building (regime shift)
        if current_damping_pct > 0.80 and self.in_profit and energy_ratio < 0.7:
            return TradeExit(
                should_exit=True,
                reason='damping_surge',
                exit_price=close,
                pnl_pct=current_pnl,
                mfe_captured_pct=self._calc_mfe_captured(close),
            )

        # 7. MAX BARS (based on historical avg bars to MFE * 2)
        max_bars = int(self.hist.avg_bars_to_mfe * 3)
        if self.bars_held >= max_bars:
            return TradeExit(
                should_exit=True,
                reason='max_bars',
                exit_price=close,
                pnl_pct=current_pnl,
                mfe_captured_pct=self._calc_mfe_captured(close),
            )

        # Hold
        return TradeExit(
            should_exit=False,
            reason='hold',
            exit_price=None,
            pnl_pct=current_pnl,
            mfe_captured_pct=self._calc_mfe_captured(close),
        )

    def _calc_mfe_captured(self, exit_price: float) -> float:
        """Calculate what % of MFE we captured."""
        if self.mfe <= 0:
            return 0.0

        if self.direction == 1:
            captured = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            captured = (self.entry_price - exit_price) / self.entry_price * 100

        return max(0, captured / self.mfe * 100)


class HistoricalStatsTracker:
    """
    Track rolling historical statistics from past trades.
    Updates dynamically to provide adaptive thresholds.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.mfe_history: List[float] = []
        self.mae_history: List[float] = []
        self.bars_to_mfe_history: List[float] = []

    def update(self, mfe: float, mae: float, bars_to_mfe: float):
        """Add new trade results to history."""
        self.mfe_history.append(mfe)
        self.mae_history.append(mae)
        self.bars_to_mfe_history.append(bars_to_mfe)

        # Keep window size
        if len(self.mfe_history) > self.window:
            self.mfe_history.pop(0)
            self.mae_history.pop(0)
            self.bars_to_mfe_history.pop(0)

    def get_stats(self) -> HistoricalStats:
        """Get current rolling statistics."""
        if len(self.mfe_history) < 5:
            # Not enough history, use defaults
            return HistoricalStats()

        mfe_arr = np.array(self.mfe_history)
        mae_arr = np.array(self.mae_history)
        bars_arr = np.array(self.bars_to_mfe_history)

        avg_mfe = np.mean(mfe_arr)
        avg_mae = np.mean(mae_arr)
        ratio = avg_mfe / avg_mae if avg_mae > 0.01 else 1.5

        return HistoricalStats(
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            mfe_mae_ratio=ratio,
            avg_bars_to_mfe=np.mean(bars_arr),
            p75_mfe=np.percentile(mfe_arr, 75),
            p25_mae=np.percentile(mae_arr, 25),
        )


def backtest_strategy(
    df: pd.DataFrame,
    min_confidence: str = 'HIGH',
) -> Dict:
    """
    Backtest the complete berserker strategy with energy recovery exit.

    Uses dynamic thresholds from rolling historical stats.
    """
    features = PhysicsFeatures()
    predictor = CompositePredictor()
    stats_tracker = HistoricalStatsTracker(window=100)

    # Compute features
    feature_df = features.compute(df)

    trades = []
    i = 100  # Skip warmup

    while i < len(feature_df) - 15:
        row = feature_df.iloc[i]

        # Generate prediction
        pred = predictor.predict(
            energy_pct=row['energy_pct'],
            damping_pct=row['damping_pct'],
            entropy_pct=row['entropy_pct'],
            jerk_pct=row['jerk_pct'],
            impulse_pct=row['impulse_pct'],
            liquidity_pct=row['liquidity_pct'],
            reynolds_pct=row['reynolds_pct'],
            buying_pressure=row['buying_pressure'],
            momentum=row['momentum_5'],
            flow_consistency=row['flow_consistency'],
            inertia=int(row['inertia']),
        )

        # Check if should trade
        conf_levels = ['LOW', 'MEDIUM', 'HIGH', 'BERSERKER']
        if conf_levels.index(pred.confidence) < conf_levels.index(min_confidence):
            i += 1
            continue

        if pred.direction == Direction.NEUTRAL:
            i += 1
            continue

        # Get dynamic historical stats
        hist_stats = stats_tracker.get_stats()

        # Enter trade
        entry_price = df.iloc[i]['close']
        direction = 1 if pred.direction == Direction.BUY else -1

        energy_exit = EnergyRecoveryExit(
            entry_price=entry_price,
            direction=direction,
            entry_energy_pct=row['energy_pct'],
            magnitude_lift=pred.magnitude_lift,
            flow_regime=pred.flow_regime,
            historical_stats=hist_stats,
        )

        # Simulate trade
        max_bars = int(hist_stats.avg_bars_to_mfe * 3) if hist_stats.avg_bars_to_mfe > 0 else 10
        exit_result = None
        bars_to_mfe = 1

        for j in range(1, max_bars + 1):
            bar_idx = i + j
            if bar_idx >= len(feature_df):
                break

            bar = df.iloc[bar_idx]
            feature_row = feature_df.iloc[bar_idx]

            exit_result = energy_exit.update(
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                current_energy_pct=feature_row['energy_pct'],
                current_damping_pct=feature_row['damping_pct'],
                current_reynolds_pct=feature_row['reynolds_pct'],
            )

            # Track bars to MFE
            if energy_exit.mfe > 0 and j < bars_to_mfe:
                bars_to_mfe = j

            if exit_result.should_exit:
                break

        # Force exit if still holding
        if exit_result is None or not exit_result.should_exit:
            exit_bar = min(i + max_bars, len(df) - 1)
            exit_price = df.iloc[exit_bar]['close']
            if direction == 1:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            exit_reason = 'max_bars'
            mfe_captured = energy_exit._calc_mfe_captured(exit_price)
        else:
            pnl_pct = exit_result.pnl_pct
            exit_reason = exit_result.reason
            mfe_captured = exit_result.mfe_captured_pct

        # Update historical stats tracker
        stats_tracker.update(
            mfe=energy_exit.mfe,
            mae=energy_exit.mae,
            bars_to_mfe=bars_to_mfe,
        )

        trades.append({
            'entry_bar': i,
            'direction': direction,
            'confidence': pred.confidence,
            'flow_regime': pred.flow_regime.value,
            'pnl_pct': pnl_pct,
            'mfe': energy_exit.mfe,
            'mae': energy_exit.mae,
            'mfe_mae_ratio': energy_exit.mfe / energy_exit.mae if energy_exit.mae > 0.01 else 0,
            'mfe_captured_pct': mfe_captured,
            'bars_held': energy_exit.bars_held,
            'exit_reason': exit_reason,
            'direction_pred': pred.direction.value,
            'direction_conf': pred.direction_confidence,
            'hist_mfe_mae_ratio': hist_stats.mfe_mae_ratio,
        })

        # Skip ahead
        i += energy_exit.bars_held + 1

    if not trades:
        return {'trades': 0, 'message': 'No trades generated'}

    trades_df = pd.DataFrame(trades)

    # Statistics
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)

    return {
        'trades': total,
        'win_rate': wins / total * 100,
        'total_pnl': trades_df['pnl_pct'].sum(),
        'avg_pnl': trades_df['pnl_pct'].mean(),
        'avg_mfe': trades_df['mfe'].mean(),
        'avg_mae': trades_df['mae'].mean(),
        'avg_mfe_mae_ratio': trades_df['mfe_mae_ratio'].mean(),
        'avg_mfe_captured': trades_df['mfe_captured_pct'].mean(),
        'profit_factor': (
            trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() /
            abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum() or 1)
        ),
        'avg_bars_held': trades_df['bars_held'].mean(),
        'by_exit_reason': trades_df['exit_reason'].value_counts().to_dict(),
        'by_flow_regime': trades_df.groupby('flow_regime')['pnl_pct'].agg(['count', 'mean', 'sum']).to_dict(),
        'efficiency_trend': trades_df['mfe_mae_ratio'].rolling(20).mean().iloc[-1] if len(trades_df) > 20 else 0,
    }
