#!/usr/bin/env python3
"""
Analyze what physics features predict DIRECTION of energy release.

We know high energy + low damping predicts MAGNITUDE (1.83x lift).
Now find what predicts whether the move is UP or DOWN.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_direction_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute features that might predict direction."""
    result = df.copy()

    # Use actual PhysicsEngine
    engine = PhysicsEngine(lookback=lookback)
    physics = engine.compute_physics_state(
        df['close'], volume=df['volume'], include_percentiles=True
    )

    result['energy'] = physics['energy']
    result['damping'] = physics['damping']
    result['entropy'] = physics['entropy']
    result['energy_pct'] = physics['energy_pct']
    result['damping_pct'] = physics['damping_pct']
    result['entropy_pct'] = physics['entropy_pct']

    # === DIRECTIONAL PHYSICS FEATURES ===

    # 1. Price velocity and momentum
    result['returns'] = df['close'].pct_change()
    result['price_vel'] = df['close'].diff()
    result['price_vel_sign'] = np.sign(result['price_vel'])

    # 2. Momentum over different windows
    for w in [3, 5, 10, 20]:
        result[f'momentum_{w}'] = df['close'].pct_change(w)

    # 3. Energy on UP bars vs DOWN bars (directional energy)
    result['bar_direction'] = np.sign(df['close'] - df['open'])
    result['up_energy'] = result['energy'] * (result['bar_direction'] > 0)
    result['down_energy'] = result['energy'] * (result['bar_direction'] < 0)

    # Rolling sum of directional energy
    window = min(10, len(result))
    result['up_energy_sum'] = result['up_energy'].rolling(window).sum()
    result['down_energy_sum'] = result['down_energy'].rolling(window).sum()
    result['energy_imbalance'] = (
        (result['up_energy_sum'] - result['down_energy_sum']) /
        (result['up_energy_sum'] + result['down_energy_sum'] + 1e-10)
    )

    # 4. Volume on UP vs DOWN bars (accumulation/distribution)
    result['up_volume'] = df['volume'] * (result['bar_direction'] > 0)
    result['down_volume'] = df['volume'] * (result['bar_direction'] < 0)
    result['up_vol_sum'] = result['up_volume'].rolling(window).sum()
    result['down_vol_sum'] = result['down_volume'].rolling(window).sum()
    result['volume_imbalance'] = (
        (result['up_vol_sum'] - result['down_vol_sum']) /
        (result['up_vol_sum'] + result['down_vol_sum'] + 1e-10)
    )

    # 5. Price position relative to recent range
    result['price_position'] = (
        (df['close'] - df['low'].rolling(20).min()) /
        (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    )

    # 6. Higher highs / lower lows momentum
    result['hh'] = (df['high'] > df['high'].shift(1)).astype(float).rolling(5).mean()
    result['ll'] = (df['low'] < df['low'].shift(1)).astype(float).rolling(5).mean()
    result['hh_ll_ratio'] = result['hh'] - result['ll']

    # 7. Close position in bar (bullish = close near high)
    bar_range = df['high'] - df['low'] + 1e-10
    result['close_position'] = (df['close'] - df['low']) / bar_range

    # 8. Recent price velocity percentile (adaptive)
    window_pct = min(200, len(result))
    result['price_vel_pct'] = result['price_vel'].rolling(window_pct).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # 9. Momentum percentile
    result['momentum_pct'] = result['momentum_5'].rolling(window_pct).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # 10. Energy imbalance percentile
    result['energy_imb_pct'] = result['energy_imbalance'].rolling(window_pct).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # Forward returns for testing
    result['fwd_return_1'] = df['close'].pct_change().shift(-1)
    result['fwd_return_2'] = df['close'].pct_change(2).shift(-2)
    result['fwd_direction'] = np.sign(result['fwd_return_1'])

    return result.dropna()


def test_direction_predictor(df: pd.DataFrame, feature: str, threshold_high: float, threshold_low: float) -> dict:
    """
    Test if a feature predicts direction.

    Args:
        feature: Column name to test
        threshold_high: Above this = predict UP
        threshold_low: Below this = predict DOWN
    """
    # Only look at high-energy bars (berserker candidates)
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)

    # Predictions
    predict_up = berserker & (df[feature] > threshold_high)
    predict_down = berserker & (df[feature] < threshold_low)

    # Accuracy
    up_correct = (predict_up & (df['fwd_direction'] > 0)).sum()
    up_total = predict_up.sum()
    up_accuracy = up_correct / up_total * 100 if up_total > 0 else 0

    down_correct = (predict_down & (df['fwd_direction'] < 0)).sum()
    down_total = predict_down.sum()
    down_accuracy = down_correct / down_total * 100 if down_total > 0 else 0

    # Combined accuracy
    total_correct = up_correct + down_correct
    total_predictions = up_total + down_total
    combined_accuracy = total_correct / total_predictions * 100 if total_predictions > 0 else 0

    return {
        'feature': feature,
        'up_signals': int(up_total),
        'up_accuracy': up_accuracy,
        'down_signals': int(down_total),
        'down_accuracy': down_accuracy,
        'total_signals': int(total_predictions),
        'combined_accuracy': combined_accuracy,
        'edge': combined_accuracy - 50,  # Edge over random
    }


def test_continuous_correlation(df: pd.DataFrame, feature: str) -> dict:
    """Test correlation between feature and forward direction."""
    # Only berserker bars
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    df_berk = df[berserker].copy()

    if len(df_berk) < 50:
        return {'feature': feature, 'correlation': 0, 'samples': 0}

    corr = df_berk[feature].corr(df_berk['fwd_return_1'])

    # Directional accuracy if we use feature sign
    if feature in ['energy_imbalance', 'volume_imbalance', 'hh_ll_ratio', 'momentum_5']:
        # Sign-based prediction
        predict = np.sign(df_berk[feature])
        actual = df_berk['fwd_direction']
        accuracy = (predict == actual).mean() * 100
    else:
        # Threshold-based (above/below 0.5 for percentiles)
        predict = np.where(df_berk[feature] > 0.5, 1, -1)
        actual = df_berk['fwd_direction']
        accuracy = (predict == actual).mean() * 100

    return {
        'feature': feature,
        'correlation': corr,
        'accuracy': accuracy,
        'samples': len(df_berk),
        'edge': accuracy - 50,
    }


def main():
    # Find data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return
    data_path = csv_files[0]
    print(f"Using: {data_path.name}")

    # Load and compute
    print("\nLoading data...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    print("\nComputing direction features...")
    df = compute_direction_features(data)
    print(f"Computed {len(df)} bars with all features")

    # Count berserker bars
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    print(f"\nBerserker bars (E>75%, D<25%): {berserker.sum()}")

    # Test directional features
    print("\n" + "=" * 70)
    print("DIRECTION PREDICTION ANALYSIS")
    print("(Testing which features predict UP vs DOWN on berserker bars)")
    print("=" * 70)

    # Features to test
    features = [
        'energy_imbalance',    # Up vs down energy
        'volume_imbalance',    # Accumulation/distribution
        'hh_ll_ratio',         # Higher highs vs lower lows
        'close_position',      # Close near high = bullish
        'price_vel_pct',       # Price velocity percentile
        'momentum_pct',        # Momentum percentile
        'energy_imb_pct',      # Energy imbalance percentile
        'momentum_3',          # Short momentum
        'momentum_5',          # Medium momentum
        'momentum_10',         # Longer momentum
        'price_position',      # Price in range
    ]

    print("\n--- Correlation with Forward Returns ---")
    results = []
    for feature in features:
        result = test_continuous_correlation(df, feature)
        results.append(result)

    results = sorted(results, key=lambda x: abs(x.get('edge', 0)), reverse=True)

    for r in results:
        print(f"\n  {r['feature']}:")
        print(f"    Correlation: {r['correlation']:.4f}")
        print(f"    Direction accuracy: {r['accuracy']:.1f}%")
        print(f"    Edge over random: {r['edge']:+.1f}%")
        print(f"    Samples: {r['samples']}")

    # Find best combination
    print("\n" + "=" * 70)
    print("COMBINED DIRECTION SIGNALS")
    print("=" * 70)

    # Test combinations
    combinations = [
        ('momentum_5 > 0', lambda x: x['momentum_5'] > 0),
        ('momentum_10 > 0', lambda x: x['momentum_10'] > 0),
        ('energy_imbalance > 0', lambda x: x['energy_imbalance'] > 0),
        ('volume_imbalance > 0', lambda x: x['volume_imbalance'] > 0),
        ('price_vel_pct > 0.5', lambda x: x['price_vel_pct'] > 0.5),
        ('momentum_pct > 0.6', lambda x: x['momentum_pct'] > 0.6),
        ('hh_ll_ratio > 0', lambda x: x['hh_ll_ratio'] > 0),
        ('close_position > 0.6', lambda x: x['close_position'] > 0.6),
        # Combos
        ('momentum_5 > 0 AND energy_imb > 0',
         lambda x: (x['momentum_5'] > 0) & (x['energy_imbalance'] > 0)),
        ('momentum_5 > 0 AND volume_imb > 0',
         lambda x: (x['momentum_5'] > 0) & (x['volume_imbalance'] > 0)),
        ('momentum_pct > 0.6 AND energy_imb_pct > 0.6',
         lambda x: (x['momentum_pct'] > 0.6) & (x['energy_imb_pct'] > 0.6)),
        ('ALL: mom>0, E_imb>0, V_imb>0',
         lambda x: (x['momentum_5'] > 0) & (x['energy_imbalance'] > 0) & (x['volume_imbalance'] > 0)),
    ]

    df_berk = df[(df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)].copy()

    print("\nPredicting UP when condition is True (on berserker bars):\n")
    combo_results = []
    for name, condition in combinations:
        try:
            mask = condition(df_berk)
            if mask.sum() > 20:
                accuracy = (df_berk.loc[mask, 'fwd_direction'] > 0).mean() * 100
                combo_results.append({
                    'condition': name,
                    'signals': int(mask.sum()),
                    'accuracy': accuracy,
                    'edge': accuracy - 50,
                })
        except Exception as e:
            continue

    combo_results = sorted(combo_results, key=lambda x: x['edge'], reverse=True)

    for r in combo_results:
        edge_str = f"+{r['edge']:.1f}%" if r['edge'] > 0 else f"{r['edge']:.1f}%"
        print(f"  {r['condition']}")
        print(f"    Signals: {r['signals']}, UP accuracy: {r['accuracy']:.1f}%, Edge: {edge_str}")

    # Best directional signals
    print("\n" + "=" * 70)
    print("BEST DIRECTIONAL PHYSICS SIGNALS")
    print("=" * 70)

    if combo_results:
        best = combo_results[0]
        print(f"\n  Best condition: {best['condition']}")
        print(f"  Direction accuracy: {best['accuracy']:.1f}%")
        print(f"  Edge over random: {best['edge']:+.1f}%")
        print(f"  Signals: {best['signals']}")

    # Asymmetry check - do UP and DOWN moves have different physics?
    print("\n" + "=" * 70)
    print("UP vs DOWN MOVE PHYSICS ASYMMETRY")
    print("=" * 70)

    up_moves = df_berk[df_berk['fwd_direction'] > 0]
    down_moves = df_berk[df_berk['fwd_direction'] < 0]

    print(f"\n  UP moves: {len(up_moves)}, DOWN moves: {len(down_moves)}")

    check_features = ['energy_pct', 'damping_pct', 'entropy_pct',
                      'energy_imbalance', 'volume_imbalance', 'momentum_5']

    print("\n  Feature means before UP vs DOWN moves:")
    for feat in check_features:
        up_mean = up_moves[feat].mean()
        down_mean = down_moves[feat].mean()
        diff = up_mean - down_mean
        print(f"    {feat}: UP={up_mean:.4f}, DOWN={down_mean:.4f}, diff={diff:+.4f}")


if __name__ == "__main__":
    main()
