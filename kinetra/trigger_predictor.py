"""
Composite Trigger Predictor

Probabilistic predictor for berserker mode entry.
Uses ONLY empirically-validated signals with adaptive percentiles.

Validated Signals (from BTCUSD M30 17k bars):
- Energy > 90th pct + Damping < 10th pct = 1.83x lift
- Underdamped regime = 1.87x lift
- Energy building (peaked) = 1.43x lift
- Volume amplification = 1.24x when high volume
- Low entropy improves quality = 1.06x

NOT USED (empirically not supported):
- Body ratio (0.93x - worse than baseline)
- Fixed MAs (lagging)
- Fixed ROC periods
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
    direction: Direction  # BUY, SELL, or NEUTRAL
    horizon_bars: int  # Forecast horizon
    confidence: str  # LOW, MEDIUM, HIGH, BERSERKER
    signal_strength: float  # Combined feature score
    conditions_met: List[str]  # List of triggered conditions
    contributing_factors: Dict[str, float]  # Factor contributions
    message: str  # Human-readable prediction

    def __str__(self) -> str:
        return self.message


class TriggerPredictor:
    """
    Composite trigger predictor using empirically-validated signals.

    All features are adaptive percentiles - no fixed thresholds.
    Weights are derived from empirical lift measurements.
    """

    # Empirically validated lift factors (from validate_thesis.py)
    VALIDATED_LIFTS = {
        'high_energy': 1.71,         # energy_pct > 0.75
        'underdamped': 1.87,         # damping_pct < 0.25 (low damping)
        'berserker_90_10': 1.83,     # energy_pct > 0.90 AND damping_pct < 0.10
        'berserker_75_25': 1.54,     # energy_pct > 0.75 AND damping_pct < 0.25
        'energy_peaked': 1.43,       # high energy + was rising
        'volume_amplify': 1.24,      # high energy + high volume
        'low_entropy': 1.06,         # entropy_pct < 0.50
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
        price_direction: float = 0.0,
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
            price_direction: Price return direction for trade bias

        Returns:
            TriggerPrediction with probability, direction, confidence
        """
        conditions_met = []
        contributing_factors = {}
        composite_score = 0.0

        # === BERSERKER CONDITION (strongest signal) ===
        # Energy > 90th percentile + Damping < 10th percentile = 1.83x lift
        is_berserker_90_10 = (energy_pct > 0.90) and (damping_pct < 0.10)
        if is_berserker_90_10:
            conditions_met.append('BERSERKER_90_10')
            weight = self.FACTOR_WEIGHTS['berserker_90_10']
            contributing_factors['berserker_90_10'] = weight
            composite_score += weight

        # Energy > 75th percentile + Damping < 25th percentile = 1.54x lift
        is_berserker_75_25 = (energy_pct > 0.75) and (damping_pct < 0.25)
        if is_berserker_75_25 and not is_berserker_90_10:
            conditions_met.append('BERSERKER_75_25')
            weight = self.FACTOR_WEIGHTS['berserker_75_25']
            contributing_factors['berserker_75_25'] = weight
            composite_score += weight

        # === UNDERDAMPED REGIME (1.87x lift) ===
        # Low damping = underdamped = high energy transfer potential
        is_underdamped = damping_pct < 0.25
        if is_underdamped and not is_berserker_90_10 and not is_berserker_75_25:
            conditions_met.append('UNDERDAMPED')
            weight = self.FACTOR_WEIGHTS['underdamped'] * 0.5  # Partial credit
            contributing_factors['underdamped'] = weight
            composite_score += weight

        # === HIGH ENERGY (1.71x lift) ===
        is_high_energy = energy_pct > 0.75
        if is_high_energy and not is_berserker_90_10 and not is_berserker_75_25:
            conditions_met.append('HIGH_ENERGY')
            weight = self.FACTOR_WEIGHTS['high_energy'] * 0.5  # Partial credit
            contributing_factors['high_energy'] = weight
            composite_score += weight

        # === ENERGY PEAKED (1.43x lift) ===
        # Energy was building (rising) and is now high
        energy_was_rising = prev_energy_vel_pct > 0.5
        energy_now_high = energy_pct > 0.70
        is_peaked = energy_was_rising and energy_now_high
        if is_peaked:
            conditions_met.append('ENERGY_PEAKED')
            weight = self.FACTOR_WEIGHTS['energy_peaked']
            contributing_factors['energy_peaked'] = weight
            composite_score += weight

        # === VOLUME AMPLIFICATION (1.24x) ===
        # High volume amplifies energy signals
        is_high_volume = volume_pct > 0.75
        if is_high_volume and is_high_energy:
            conditions_met.append('VOLUME_AMPLIFY')
            weight = self.FACTOR_WEIGHTS['volume_amplify']
            contributing_factors['volume_amplify'] = weight
            composite_score += weight

        # === LOW ENTROPY (1.06x - quality filter) ===
        # Lower entropy = cleaner signal
        is_low_entropy = entropy_pct < 0.50
        if is_low_entropy and len(conditions_met) > 0:
            conditions_met.append('CLEAN_SIGNAL')
            weight = self.FACTOR_WEIGHTS['low_entropy']
            contributing_factors['low_entropy'] = weight
            composite_score += weight

        # === CONVERT SCORE TO PROBABILITY ===
        # Use sigmoid mapping calibrated to empirical hit rates:
        # - Berserker 90/10 condition: ~40% hit rate (1.83x on 22% base)
        # - Berserker 75/25 condition: ~34% hit rate (1.54x on 22% base)
        # - Baseline: ~22% probability

        # Sigmoid: p = 1 / (1 + exp(-k*(score - threshold)))
        # Calibrated so score of 0.6 (berserker_90_10) -> ~40%
        base_prob = 0.22  # Baseline probability from data
        if composite_score > 0:
            # Scale probability based on combined log-lifts
            # exp(score) gives combined lift, multiply by base
            combined_lift = np.exp(composite_score)
            probability = min(0.85, base_prob * combined_lift)
        else:
            probability = base_prob * 0.5  # Below baseline

        # === DETERMINE DIRECTION ===
        if price_direction > 0.0005:
            direction = Direction.BUY
        elif price_direction < -0.0005:
            direction = Direction.SELL
        else:
            # Use energy velocity as tiebreaker
            if energy_vel_pct > 0.6:
                direction = Direction.BUY
            elif energy_vel_pct < 0.4:
                direction = Direction.SELL
            else:
                direction = Direction.NEUTRAL

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

        if confidence == 'BERSERKER':
            message = (
                f"BERSERKER: {prob_pct}% probability of {dir_str} energy release "
                f"in next {self.horizon} bars. "
                f"[E>{int(energy_pct*100)}pct, D<{int(damping_pct*100)}pct]"
            )
        elif confidence == 'HIGH':
            message = (
                f"HIGH: {prob_pct}% chance of {dir_str} release. "
                f"Conditions: {', '.join(conditions_met[:3])}"
            )
        elif confidence == 'MEDIUM':
            message = (
                f"MEDIUM: {prob_pct}% probability. {', '.join(conditions_met[:2])}"
            )
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
        - close (for direction)

        Args:
            df: DataFrame with physics percentile features
            bar_idx: Current bar index

        Returns:
            TriggerPrediction
        """
        row = df.iloc[bar_idx]
        prev_row = df.iloc[bar_idx - 1] if bar_idx > 0 else row

        # Get price direction
        if 'close' in df.columns and bar_idx > 0:
            price_direction = (row['close'] - prev_row['close']) / prev_row['close']
        else:
            price_direction = 0.0

        return self.predict(
            energy_pct=row.get('energy_pct', 0.5),
            damping_pct=row.get('damping_pct', 0.5),
            entropy_pct=row.get('entropy_pct', 0.5),
            volume_pct=row.get('volume_pct', 0.5),
            energy_vel_pct=row.get('energy_vel_pct', 0.5),
            prev_energy_vel_pct=prev_row.get('energy_vel_pct', 0.5),
            price_direction=price_direction,
        )

    def backtest(
        self,
        df: pd.DataFrame,
        min_confidence: str = 'HIGH',
    ) -> Dict:
        """
        Backtest the predictor on historical data.

        Args:
            df: DataFrame with physics percentiles and forward returns
            min_confidence: Minimum confidence level to count as signal

        Returns:
            Dict with backtest statistics
        """
        confidence_order = ['LOW', 'MEDIUM', 'HIGH', 'BERSERKER']
        min_level = confidence_order.index(min_confidence)

        signals = []
        for i in range(self.lookback, len(df) - self.horizon):
            pred = self.predict_from_dataframe(df, i)
            level = confidence_order.index(pred.confidence)

            if level >= min_level:
                # Get forward return
                fwd_return = (
                    df.iloc[i + self.horizon]['close'] - df.iloc[i]['close']
                ) / df.iloc[i]['close']

                signals.append({
                    'bar': i,
                    'probability': pred.probability,
                    'direction': pred.direction.value,
                    'confidence': pred.confidence,
                    'fwd_return': fwd_return,
                    'fwd_abs_return': abs(fwd_return),
                    'conditions': pred.conditions_met,
                })

        if not signals:
            return {'signals': 0, 'message': 'No signals generated'}

        signals_df = pd.DataFrame(signals)

        # Calculate statistics
        avg_abs_return = signals_df['fwd_abs_return'].mean() * 100
        baseline_abs_return = df['close'].pct_change(self.horizon).abs().mean() * 100

        # Direction accuracy
        signals_df['correct_dir'] = (
            ((signals_df['direction'] == 'BUY') & (signals_df['fwd_return'] > 0)) |
            ((signals_df['direction'] == 'SELL') & (signals_df['fwd_return'] < 0))
        )
        dir_accuracy = signals_df['correct_dir'].mean() * 100

        return {
            'signals': len(signals_df),
            'avg_abs_move': avg_abs_return,
            'baseline_abs_move': baseline_abs_return,
            'lift': avg_abs_return / baseline_abs_return if baseline_abs_return > 0 else 0,
            'direction_accuracy': dir_accuracy,
            'by_confidence': signals_df.groupby('confidence').agg({
                'fwd_abs_return': ['count', 'mean'],
                'correct_dir': 'mean',
            }).to_dict(),
        }


def validate_predictor(data_path: str = None):
    """Validate the composite predictor on actual data."""
    import sys
    from pathlib import Path

    # Find data
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

    # Add volume percentile
    window = min(500, len(df))
    df['volume_pct'] = df['volume'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # Add energy velocity percentile
    df['energy_vel'] = physics['energy'].diff()
    df['energy_vel_pct'] = df['energy_vel'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Initialize predictor
    predictor = TriggerPredictor(horizon=2)

    # Backtest
    print("\n" + "=" * 70)
    print("COMPOSITE PREDICTOR VALIDATION")
    print("=" * 70)

    for min_conf in ['LOW', 'MEDIUM', 'HIGH', 'BERSERKER']:
        results = predictor.backtest(df, min_confidence=min_conf)
        print(f"\n{min_conf} confidence threshold:")
        print(f"  Signals: {results.get('signals', 0)}")
        if results.get('signals', 0) > 0:
            print(f"  Avg move: {results['avg_abs_move']:.3f}%")
            print(f"  Baseline: {results['baseline_abs_move']:.3f}%")
            print(f"  Lift: {results['lift']:.2f}x")
            print(f"  Direction accuracy: {results['direction_accuracy']:.1f}%")


if __name__ == "__main__":
    validate_predictor()
