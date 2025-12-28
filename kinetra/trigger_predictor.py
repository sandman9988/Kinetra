"""
Composite Trigger Predictor

Probabilistic predictor for berserker mode entry.
Uses ONLY empirically-validated signals with adaptive percentiles.

Validated Signals (from BTCUSD M30 17k bars):

MAGNITUDE (when to expect big move):
- Energy > 90th pct + Damping < 10th pct = 1.83x lift
- Underdamped regime = 1.87x lift
- Energy building (peaked) = 1.43x lift
- Volume amplification = 1.24x when high volume

DIRECTION (which way the move goes):
- Berserker bars show MEAN REVERSION (53.7% counter-trend)
- LAMINAR flow + Berserker = 56% reversal (+12% edge)
- High flow consistency + Berserker = 56.4% reversal (+6% edge)
- Trade AGAINST momentum on berserker bars

Physics interpretation:
- Laminar flow = smooth consistent trend (low turbulence)
- Berserker in laminar flow = trend exhaustion point
- High energy + low damping = energy about to release (reverse)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from .physics_engine import PhysicsEngine


class Direction(Enum):
    NEUTRAL = "NEUTRAL"
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TriggerPrediction:
    """Prediction output for trigger event."""
    probability: float  # 0-1 probability of high-energy release
    direction: Direction  # BUY, SELL, or NEUTRAL (counter-trend on berserker)
    horizon_bars: int  # Forecast horizon
    confidence: str  # LOW, MEDIUM, HIGH, BERSERKER
    signal_strength: float  # Combined feature score
    conditions_met: List[str]  # List of triggered conditions
    contributing_factors: Dict[str, float]  # Factor contributions
    direction_confidence: float  # 0-1 confidence in direction
    is_laminar: bool  # Laminar flow state
    message: str  # Human-readable prediction

    def __str__(self) -> str:
        return self.message


class TriggerPredictor:
    """
    Composite trigger predictor using empirically-validated signals.

    MAGNITUDE: All features are adaptive percentiles - no fixed thresholds.
    DIRECTION: Counter-trend (mean reversion) on berserker bars.
    """

    # Empirically validated lift factors
    VALIDATED_LIFTS = {
        'high_energy': 1.71,         # energy_pct > 0.75
        'underdamped': 1.87,         # damping_pct < 0.25 (low damping)
        'berserker_90_10': 1.83,     # energy_pct > 0.90 AND damping_pct < 0.10
        'berserker_75_25': 1.54,     # energy_pct > 0.75 AND damping_pct < 0.25
        'energy_peaked': 1.43,       # high energy + was rising
        'volume_amplify': 1.24,      # high energy + high volume
        'low_entropy': 1.06,         # entropy_pct < 0.50
    }

    # Direction confidence based on empirical testing
    DIRECTION_CONFIDENCE = {
        'base_berserker': 0.537,     # 53.7% counter-trend accuracy
        'laminar_berserker': 0.56,   # 56% with laminar flow
        'high_flow_berserker': 0.564, # 56.4% with high flow consistency
    }

    # Convert lifts to log-weights for combining
    FACTOR_WEIGHTS = {k: np.log(v) for k, v in VALIDATED_LIFTS.items()}

    def __init__(self, lookback: int = 20, horizon: int = 2):
        """
        Initialize predictor.

        Args:
            lookback: Physics engine lookback period
            horizon: Forecast horizon in bars
        """
        self.lookback = lookback
        self.horizon = horizon
        self.engine = PhysicsEngine(lookback=lookback)

    def predict(
        self,
        energy_pct: float,
        damping_pct: float,
        entropy_pct: float,
        volume_pct: float = 0.5,
        energy_vel_pct: float = 0.5,
        prev_energy_vel_pct: float = 0.5,
        momentum: float = 0.0,
        flow_consistency: float = 0.5,
        laminar_score_pct: float = 0.5,
    ) -> TriggerPrediction:
        """
        Generate composite trigger prediction.

        All inputs are adaptive percentiles [0, 1] - no fixed thresholds.

        Args:
            energy_pct: Energy percentile (0-1)
            damping_pct: Damping percentile (0-1) - LOW = underdamped
            entropy_pct: Entropy percentile (0-1) - LOW = cleaner signal
            volume_pct: Volume percentile (0-1)
            energy_vel_pct: Energy velocity percentile (0-1)
            prev_energy_vel_pct: Previous bar energy velocity percentile
            momentum: Price momentum (positive = up trend, negative = down)
            flow_consistency: How consistent recent direction was (0-1)
            laminar_score_pct: Laminar flow score percentile (0-1)

        Returns:
            TriggerPrediction with probability, direction, confidence
        """
        conditions_met = []
        contributing_factors = {}
        composite_score = 0.0

        # === BERSERKER CONDITION (strongest signal) ===
        is_berserker_90_10 = (energy_pct > 0.90) and (damping_pct < 0.10)
        is_berserker_75_25 = (energy_pct > 0.75) and (damping_pct < 0.25)

        if is_berserker_90_10:
            conditions_met.append('BERSERKER_90_10')
            weight = self.FACTOR_WEIGHTS['berserker_90_10']
            contributing_factors['berserker_90_10'] = weight
            composite_score += weight
        elif is_berserker_75_25:
            conditions_met.append('BERSERKER_75_25')
            weight = self.FACTOR_WEIGHTS['berserker_75_25']
            contributing_factors['berserker_75_25'] = weight
            composite_score += weight

        # === UNDERDAMPED REGIME ===
        is_underdamped = damping_pct < 0.25
        if is_underdamped and not is_berserker_90_10 and not is_berserker_75_25:
            conditions_met.append('UNDERDAMPED')
            weight = self.FACTOR_WEIGHTS['underdamped'] * 0.5
            contributing_factors['underdamped'] = weight
            composite_score += weight

        # === HIGH ENERGY ===
        is_high_energy = energy_pct > 0.75
        if is_high_energy and not is_berserker_90_10 and not is_berserker_75_25:
            conditions_met.append('HIGH_ENERGY')
            weight = self.FACTOR_WEIGHTS['high_energy'] * 0.5
            contributing_factors['high_energy'] = weight
            composite_score += weight

        # === ENERGY PEAKED ===
        energy_was_rising = prev_energy_vel_pct > 0.5
        energy_now_high = energy_pct > 0.70
        is_peaked = energy_was_rising and energy_now_high
        if is_peaked:
            conditions_met.append('ENERGY_PEAKED')
            weight = self.FACTOR_WEIGHTS['energy_peaked']
            contributing_factors['energy_peaked'] = weight
            composite_score += weight

        # === VOLUME AMPLIFICATION ===
        is_high_volume = volume_pct > 0.75
        if is_high_volume and is_high_energy:
            conditions_met.append('VOLUME_AMPLIFY')
            weight = self.FACTOR_WEIGHTS['volume_amplify']
            contributing_factors['volume_amplify'] = weight
            composite_score += weight

        # === LOW ENTROPY ===
        is_low_entropy = entropy_pct < 0.50
        if is_low_entropy and len(conditions_met) > 0:
            conditions_met.append('CLEAN_SIGNAL')
            weight = self.FACTOR_WEIGHTS['low_entropy']
            contributing_factors['low_entropy'] = weight
            composite_score += weight

        # === LAMINAR FLOW DETECTION ===
        is_laminar = laminar_score_pct > 0.6 or flow_consistency > 0.7
        if is_laminar:
            conditions_met.append('LAMINAR_FLOW')

        # === CONVERT SCORE TO PROBABILITY ===
        base_prob = 0.22
        if composite_score > 0:
            combined_lift = np.exp(composite_score)
            probability = min(0.85, base_prob * combined_lift)
        else:
            probability = base_prob * 0.5

        # === DETERMINE DIRECTION (COUNTER-TREND ON BERSERKER) ===
        # Key insight: Berserker bars show mean reversion
        # Laminar flow + berserker = trend exhaustion = 56% reversal

        if is_berserker_90_10 or is_berserker_75_25:
            # COUNTER-TREND: Trade against momentum
            if momentum > 0.0001:
                # Momentum is UP -> predict DOWN (reversal)
                direction = Direction.SELL
                conditions_met.append('FADE_UP_MOMENTUM')
            elif momentum < -0.0001:
                # Momentum is DOWN -> predict UP (reversal)
                direction = Direction.BUY
                conditions_met.append('FADE_DOWN_MOMENTUM')
            else:
                direction = Direction.NEUTRAL

            # Direction confidence based on flow state
            if is_laminar:
                direction_confidence = self.DIRECTION_CONFIDENCE['laminar_berserker']
            elif flow_consistency > 0.7:
                direction_confidence = self.DIRECTION_CONFIDENCE['high_flow_berserker']
            else:
                direction_confidence = self.DIRECTION_CONFIDENCE['base_berserker']
        else:
            # Non-berserker: neutral or follow momentum weakly
            if momentum > 0.001:
                direction = Direction.BUY
            elif momentum < -0.001:
                direction = Direction.SELL
            else:
                direction = Direction.NEUTRAL
            direction_confidence = 0.50  # No edge outside berserker

        # === DETERMINE CONFIDENCE LEVEL ===
        if is_berserker_90_10:
            confidence = 'BERSERKER'
        elif is_berserker_75_25:
            confidence = 'HIGH'
        elif probability >= 0.35:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # === GENERATE MESSAGE ===
        prob_pct = int(probability * 100)
        dir_str = direction.value
        dir_conf_pct = int(direction_confidence * 100)

        if confidence == 'BERSERKER':
            flow_state = "LAMINAR (trend exhaustion)" if is_laminar else "turbulent"
            message = (
                f"BERSERKER: {prob_pct}% magnitude, {dir_conf_pct}% {dir_str}. "
                f"Flow: {flow_state}. Counter-trend entry."
            )
        elif confidence == 'HIGH':
            message = (
                f"HIGH: {prob_pct}% move expected. Direction: {dir_str} ({dir_conf_pct}%)"
            )
        elif confidence == 'MEDIUM':
            message = f"MEDIUM: {prob_pct}% probability. {', '.join(conditions_met[:2])}"
        else:
            message = f"LOW: {prob_pct}% - monitoring"

        return TriggerPrediction(
            probability=probability,
            direction=direction,
            horizon_bars=self.horizon,
            confidence=confidence,
            signal_strength=composite_score,
            conditions_met=conditions_met,
            contributing_factors=contributing_factors,
            direction_confidence=direction_confidence,
            is_laminar=is_laminar,
            message=message,
        )

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> TriggerPrediction:
        """
        Generate prediction from pre-computed DataFrame.

        Expected columns (all adaptive percentiles):
        - energy_pct, damping_pct, entropy_pct
        - volume_pct (optional, defaults to 0.5)
        - energy_vel_pct (optional)
        - flow_consistency (optional) - computed if missing
        - laminar_score_pct (optional)
        - close (for momentum)
        """
        row = df.iloc[bar_idx]
        prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else row

        # Get momentum
        if 'momentum_5' in df.columns:
            momentum = row['momentum_5']
        elif 'close' in df.columns and bar_idx > 4:
            momentum = (row['close'] - df.iloc[bar_idx - 5]['close']) / df.iloc[bar_idx - 5]['close']
        else:
            momentum = 0.0

        # Flow consistency (direction consistency over last 5 bars)
        if 'flow_consistency' in df.columns:
            flow_consistency = row['flow_consistency']
        elif 'returns' in df.columns and bar_idx >= 5:
            signs = np.sign(df['returns'].iloc[bar_idx-4:bar_idx+1])
            flow_consistency = (signs == signs.iloc[-1]).mean()
        else:
            flow_consistency = 0.5

        return self.predict(
            energy_pct=row.get('energy_pct', 0.5),
            damping_pct=row.get('damping_pct', 0.5),
            entropy_pct=row.get('entropy_pct', 0.5),
            volume_pct=row.get('volume_pct', 0.5),
            energy_vel_pct=row.get('energy_vel_pct', 0.5),
            prev_energy_vel_pct=prev_row.get('energy_vel_pct', 0.5),
            momentum=momentum,
            flow_consistency=flow_consistency,
            laminar_score_pct=row.get('laminar_score_pct', 0.5),
        )

    def backtest(
        self,
        df: pd.DataFrame,
        min_confidence: str = 'HIGH',
    ) -> Dict:
        """
        Backtest the predictor on historical data.

        Args:
            df: DataFrame with physics percentiles
            min_confidence: Minimum confidence level to count as signal

        Returns:
            Dict with backtest statistics including direction accuracy
        """
        confidence_order = ['LOW', 'MEDIUM', 'HIGH', 'BERSERKER']
        min_level = confidence_order.index(min_confidence)

        signals = []
        for i in range(max(self.lookback, 5), len(df) - self.horizon):
            pred = self.predict_from_dataframe(df, i)
            level = confidence_order.index(pred.confidence)

            if level >= min_level:
                fwd_return = (
                    df.iloc[i + self.horizon]['close'] - df.iloc[i]['close']
                ) / df.iloc[i]['close']

                # Direction correctness
                if pred.direction == Direction.BUY:
                    dir_correct = fwd_return > 0
                elif pred.direction == Direction.SELL:
                    dir_correct = fwd_return < 0
                else:
                    dir_correct = None

                signals.append({
                    'bar': i,
                    'probability': pred.probability,
                    'direction': pred.direction.value,
                    'direction_confidence': pred.direction_confidence,
                    'confidence': pred.confidence,
                    'is_laminar': pred.is_laminar,
                    'fwd_return': fwd_return,
                    'fwd_abs_return': abs(fwd_return),
                    'dir_correct': dir_correct,
                    'conditions': pred.conditions_met,
                })

        if not signals:
            return {'signals': 0, 'message': 'No signals generated'}

        signals_df = pd.DataFrame(signals)

        # Magnitude stats
        avg_abs_return = signals_df['fwd_abs_return'].mean() * 100
        baseline_abs_return = df['close'].pct_change(self.horizon).abs().mean() * 100

        # Direction stats (exclude NEUTRAL)
        dir_signals = signals_df[signals_df['dir_correct'].notna()]
        if len(dir_signals) > 0:
            dir_accuracy = dir_signals['dir_correct'].mean() * 100
        else:
            dir_accuracy = 0

        # Laminar vs non-laminar
        laminar_signals = signals_df[signals_df['is_laminar']]
        if len(laminar_signals) > 0:
            laminar_dir_acc = laminar_signals[laminar_signals['dir_correct'].notna()]['dir_correct'].mean() * 100
        else:
            laminar_dir_acc = 0

        return {
            'signals': len(signals_df),
            'avg_abs_move': avg_abs_return,
            'baseline_abs_move': baseline_abs_return,
            'magnitude_lift': avg_abs_return / baseline_abs_return if baseline_abs_return > 0 else 0,
            'direction_accuracy': dir_accuracy,
            'direction_edge': dir_accuracy - 50,
            'laminar_signals': len(laminar_signals),
            'laminar_dir_accuracy': laminar_dir_acc,
        }


def validate_predictor(data_path: str = None):
    """Validate the composite predictor on actual data."""
    from pathlib import Path

    if data_path is None:
        project_root = Path(__file__).parent.parent
        csv_files = list(project_root.glob("*BTCUSD*.csv"))
        if not csv_files:
            print("No BTCUSD CSV file found")
            return
        data_path = str(csv_files[0])

    from .mt5_connector import load_csv_data

    print(f"Loading: {Path(data_path).name}")
    data = load_csv_data(data_path)
    print(f"Loaded {len(data)} bars")

    # Compute physics with percentiles
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']
    df['entropy_pct'] = physics['entropy_pct']
    df['returns'] = df['close'].pct_change()
    df['momentum_5'] = df['close'].pct_change(5)

    # Volume percentile
    window = min(500, len(df))
    df['volume_pct'] = df['volume'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # Energy velocity percentile
    df['energy_vel'] = physics['energy'].diff()
    df['energy_vel_pct'] = df['energy_vel'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # Flow consistency
    df['return_sign'] = np.sign(df['returns'])
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Laminar score
    df['trend_5'] = df['close'].pct_change(5)
    df['vol_5'] = df['returns'].rolling(5).std()
    df['smoothness'] = df['trend_5'].abs() / (df['vol_5'] + 1e-10)
    df['laminar_score'] = df['flow_consistency'] * (1 - df['entropy_pct']) * df['smoothness'].clip(0, 5) / 5
    df['laminar_score_pct'] = df['laminar_score'].rolling(200, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Backtest
    predictor = TriggerPredictor(horizon=2)

    print("\n" + "=" * 70)
    print("COMPOSITE PREDICTOR VALIDATION (Magnitude + Direction)")
    print("=" * 70)

    for min_conf in ['MEDIUM', 'HIGH', 'BERSERKER']:
        results = predictor.backtest(df, min_confidence=min_conf)
        print(f"\n{min_conf} threshold:")
        print(f"  Signals: {results.get('signals', 0)}")
        if results.get('signals', 0) > 0:
            print(f"  Magnitude:")
            print(f"    Avg move: {results['avg_abs_move']:.3f}%")
            print(f"    Baseline: {results['baseline_abs_move']:.3f}%")
            print(f"    Lift: {results['magnitude_lift']:.2f}x")
            print(f"  Direction:")
            print(f"    Accuracy: {results['direction_accuracy']:.1f}%")
            print(f"    Edge: {results['direction_edge']:+.1f}%")
            if results.get('laminar_signals', 0) > 10:
                print(f"  Laminar subset ({results['laminar_signals']} signals):")
                print(f"    Direction accuracy: {results['laminar_dir_accuracy']:.1f}%")


if __name__ == "__main__":
    validate_predictor()
