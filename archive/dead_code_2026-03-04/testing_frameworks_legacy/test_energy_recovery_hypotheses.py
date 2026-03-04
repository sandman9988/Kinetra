#!/usr/bin/env python3
"""
Empirical Test of Energy Recovery Hypotheses from Theorem Proofs

Hypotheses to test:
1. Energy extraction efficiency η = 68% achievable
2. ARS reward components: optimal α (MFE), β (MAE), γ (Time)
3. ATR normalization improves regime-invariance
4. Time penalty reduces overholding losses

ARS Formula from theorem:
R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


@pytest.fixture
def df():
    """Load test market data with physics features."""
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))

    if not csv_files:
        # Return synthetic data if no real data available
        dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
        data = pd.DataFrame({
            'open': [50000 + i*10 + np.random.randn()*100 for i in range(1000)],
            'high': [50100 + i*10 + np.random.randn()*100 for i in range(1000)],
            'low': [49900 + i*10 + np.random.randn()*100 for i in range(1000)],
            'close': [50000 + i*10 + np.random.randn()*100 for i in range(1000)],
            'volume': [1000 + i + np.random.randn()*50 for i in range(1000)]
        }, index=dates)
    else:
        data = load_csv_data(str(csv_files[0]))

    # Compute physics
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    data['energy'] = compute_energy(data['close'], lookback=20)
    data['energy_pct'] = physics['energy_pct']
    data['damping_pct'] = physics['damping_pct']
    data['atr'] = compute_atr(data, period=14)
    data['momentum_5'] = data['close'].pct_change(5)
    data['counter_direction'] = -np.sign(data['momentum_5'])

    # Flow consistency
    data['return_sign'] = np.sign(data['close'].pct_change())
    data['flow_consistency'] = data['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Volume percentile
    vol_window = min(500, len(data))
    data['volume_pct'] = data['volume'].rolling(vol_window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    return data.dropna()


@pytest.fixture
def signals(df):
    """Generate berserker+ signals for testing."""
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)
    return df[berserker_plus].index


@pytest.fixture
def direction_col():
    """Direction column name for tests."""
    return 'counter_direction'


def compute_energy(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Compute kinetic energy: E_t = (1/2) * m * v²"""
    velocity = close.pct_change()
    energy = 0.5 * (velocity ** 2)
    # Rolling sum as "mass" proxy
    return energy.rolling(lookback).mean() * lookback


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for normalization."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate_trade_with_ars(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    max_bars: int = 10,
    alpha: float = 1.0,  # MFE reward weight
    beta: float = 1.5,   # MAE penalty weight
    gamma: float = 0.01, # Time decay per bar
):
    """
    Simulate trade and compute ARS reward at each bar.

    Returns per-bar rewards and final metrics.
    """
    entry_price = df.iloc[entry_idx]['close']
    entry_energy = df.iloc[entry_idx]['energy'] if 'energy' in df.columns else 0.001
    entry_atr = df.iloc[entry_idx]['atr'] if 'atr' in df.columns else 0.01

    rewards = []
    running_mfe = 0
    running_mae = 0

    for bar in range(1, max_bars + 1):
        if entry_idx + bar >= len(df):
            break

        high = df.iloc[entry_idx + bar]['high']
        low = df.iloc[entry_idx + bar]['low']
        close = df.iloc[entry_idx + bar]['close']
        atr = df.iloc[entry_idx + bar]['atr'] if 'atr' in df.columns else entry_atr

        if direction == 1:
            pnl = (close - entry_price) / entry_price * 100
            bar_mfe = (high - entry_price) / entry_price * 100
            bar_mae = (entry_price - low) / entry_price * 100
        else:
            pnl = (entry_price - close) / entry_price * 100
            bar_mfe = (entry_price - low) / entry_price * 100
            bar_mae = (high - entry_price) / entry_price * 100

        running_mfe = max(running_mfe, bar_mfe)
        running_mae = max(running_mae, bar_mae)

        # Normalize by ATR (convert % to ATR units)
        atr_pct = atr / entry_price * 100 if entry_price > 0 else 0.01
        mfe_atr = running_mfe / atr_pct if atr_pct > 0 else 0
        mae_atr = running_mae / atr_pct if atr_pct > 0 else 0

        # Energy-normalized PnL
        energy_pnl = pnl / (entry_energy * 1000 + 0.001) if entry_energy > 0 else pnl

        # ARS reward: R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
        ars_reward = energy_pnl + alpha * mfe_atr - beta * mae_atr - gamma * bar

        rewards.append({
            'bar': bar,
            'pnl': pnl,
            'mfe': running_mfe,
            'mae': running_mae,
            'mfe_atr': mfe_atr,
            'mae_atr': mae_atr,
            'energy_pnl': energy_pnl,
            'ars_reward': ars_reward,
        })

    return rewards


def test_hypothesis_1_energy_efficiency(df, signals, direction_col):
    """
    Test Hypothesis 1: Energy extraction efficiency η = 68% achievable

    η = |PnL| / (k · E_t)
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1: Energy Extraction Efficiency (η = 68% claimed)")
    print("=" * 80)

    efficiencies = []

    for bar in signals:
        idx = df.index.get_loc(bar)
        direction = int(df.loc[bar, direction_col])
        if direction == 0 or idx + 10 >= len(df):
            continue

        entry_price = df.iloc[idx]['close']
        entry_energy = df.iloc[idx]['energy']

        # Compute P&L at various exit points
        for exit_bar in [1, 3, 5, 10]:
            if idx + exit_bar >= len(df):
                continue

            exit_price = df.iloc[idx + exit_bar]['close']
            if direction == 1:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            # Energy efficiency
            if entry_energy > 0:
                eta = abs(pnl) / (entry_energy * 1000)  # k = 1000 (normalization)
                efficiencies.append({'exit_bar': exit_bar, 'eta': eta, 'pnl': pnl * 100})

    # Group by exit bar
    for exit_bar in [1, 3, 5, 10]:
        subset = [e for e in efficiencies if e['exit_bar'] == exit_bar]
        if subset:
            avg_eta = np.mean([e['eta'] for e in subset])
            avg_pnl = np.mean([e['pnl'] for e in subset])
            print(f"\n  Exit bar +{exit_bar}:")
            print(f"    Avg η (efficiency): {avg_eta:.2f} ({avg_eta*100:.0f}%)")
            print(f"    Avg PnL: {avg_pnl:+.4f}%")

    overall_eta = np.mean([e['eta'] for e in efficiencies]) if efficiencies else 0
    print(f"\n  RESULT: Mean η = {overall_eta:.2f} ({overall_eta*100:.0f}%)")
    print(f"  Claimed: 68%")
    print(f"  Status: {'SUPPORTED' if 0.5 < overall_eta < 0.9 else 'NEEDS CALIBRATION'}")


def test_hypothesis_2_ars_weights(df, signals, direction_col):
    """
    Test Hypothesis 2: Optimal ARS weights (α, β, γ)

    Find weights that maximize Omega ratio.
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 2: Optimal ARS Weights (α, β, γ)")
    print("=" * 80)

    # Grid search over α, β, γ
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [1.0, 1.5, 2.0, 2.5]
    gammas = [0.0, 0.01, 0.02, 0.05]

    results = []

    for alpha, beta, gamma in product(alphas, betas, gammas):
        all_final_pnl = []
        all_final_ars = []

        for bar in signals:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, direction_col])
            if direction == 0 or idx + 10 >= len(df):
                continue

            rewards = simulate_trade_with_ars(df, idx, direction, max_bars=5, alpha=alpha, beta=beta, gamma=gamma)
            if rewards:
                final = rewards[-1]
                all_final_pnl.append(final['pnl'])
                all_final_ars.append(final['ars_reward'])

        if all_final_pnl:
            # Compute Omega ratio for ARS-optimized exits
            # (Would need to optimize exit based on ARS, but for now just measure correlation)
            avg_pnl = np.mean(all_final_pnl)
            avg_ars = np.mean(all_final_ars)
            win_rate = sum(1 for p in all_final_pnl if p > 0) / len(all_final_pnl) * 100
            gains = sum(p for p in all_final_pnl if p > 0)
            losses = abs(sum(p for p in all_final_pnl if p < 0))
            omega = gains / losses if losses > 0 else float('inf')

            # Also compute correlation between ARS and PnL
            correlation = np.corrcoef(all_final_pnl, all_final_ars)[0, 1] if len(all_final_pnl) > 1 else 0

            results.append({
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
                'avg_pnl': avg_pnl, 'avg_ars': avg_ars,
                'win_rate': win_rate, 'omega': omega,
                'correlation': correlation,
            })

    # Find best by Omega
    results = sorted(results, key=lambda x: x['omega'] if x['omega'] != float('inf') else 0, reverse=True)

    print(f"\n  Top 5 configurations by Omega ratio:\n")
    print(f"  {'α':>5} {'β':>5} {'γ':>5} │ {'Omega':>8} │ {'Win%':>7} │ {'AvgPnL':>9} │ {'Corr':>6}")
    print("  " + "─" * 65)

    for r in results[:5]:
        omega_str = f"{r['omega']:.2f}" if r['omega'] != float('inf') else "inf"
        print(f"  {r['alpha']:>5.1f} {r['beta']:>5.1f} {r['gamma']:>5.2f} │ {omega_str:>8} │ "
              f"{r['win_rate']:>6.1f}% │ {r['avg_pnl']:>+8.4f}% │ {r['correlation']:>6.3f}")

    if results:
        best = results[0]
        print(f"\n  OPTIMAL WEIGHTS:")
        print(f"    α (MFE reward):   {best['alpha']}")
        print(f"    β (MAE penalty):  {best['beta']}")
        print(f"    γ (Time decay):   {best['gamma']}")
        print(f"    Omega ratio:      {best['omega']:.2f}" if best['omega'] != float('inf') else "    Omega ratio:      inf")


def test_hypothesis_3_atr_normalization(df, signals, direction_col):
    """
    Test Hypothesis 3: ATR normalization improves regime-invariance
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 3: ATR Normalization for Regime-Invariance")
    print("=" * 80)

    # Split data into high-vol and low-vol regimes
    df['atr_pct'] = df['atr'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    high_vol = df['atr_pct'] > 0.7
    low_vol = df['atr_pct'] < 0.3

    for regime_name, regime_mask in [('HIGH VOL', high_vol), ('LOW VOL', low_vol)]:
        # Filter signals by regime
        regime_signals = [s for s in signals if s in df.index and df.loc[s, 'atr_pct'] > 0.7] if regime_name == 'HIGH VOL' else [s for s in signals if s in df.index and df.loc[s, 'atr_pct'] < 0.3]

        if len(regime_signals) < 30:
            print(f"\n  {regime_name}: <30 signals, skipping")
            continue

        raw_mfes = []
        norm_mfes = []

        for bar in regime_signals:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, direction_col])
            if direction == 0 or idx + 5 >= len(df):
                continue

            entry_price = df.iloc[idx]['close']
            atr = df.iloc[idx]['atr']

            # 5-bar MFE
            if direction == 1:
                mfe = max((df.iloc[idx + i]['high'] - entry_price) / entry_price * 100 for i in range(1, 6))
            else:
                mfe = max((entry_price - df.iloc[idx + i]['low']) / entry_price * 100 for i in range(1, 6))

            raw_mfes.append(mfe)
            norm_mfes.append(mfe / (atr / entry_price * 100) if atr > 0 else mfe)

        print(f"\n  {regime_name} ({len(raw_mfes)} signals):")
        print(f"    Raw MFE:        mean={np.mean(raw_mfes):.4f}%, std={np.std(raw_mfes):.4f}%")
        print(f"    ATR-Norm MFE:   mean={np.mean(norm_mfes):.2f}x,  std={np.std(norm_mfes):.2f}x")

    # Check if normalized has lower variance
    all_norm = []
    all_raw = []
    for bar in list(signals):
        idx = df.index.get_loc(bar)
        direction = int(df.loc[bar, direction_col])
        if direction == 0 or idx + 5 >= len(df):
            continue

        entry_price = df.iloc[idx]['close']
        atr = df.iloc[idx]['atr']

        if direction == 1:
            mfe = max((df.iloc[idx + i]['high'] - entry_price) / entry_price * 100 for i in range(1, 6))
        else:
            mfe = max((entry_price - df.iloc[idx + i]['low']) / entry_price * 100 for i in range(1, 6))

        all_raw.append(mfe)
        all_norm.append(mfe / (atr / entry_price * 100) if atr > 0 else mfe)

    cv_raw = np.std(all_raw) / np.mean(all_raw) if np.mean(all_raw) > 0 else 0
    cv_norm = np.std(all_norm) / np.mean(all_norm) if np.mean(all_norm) > 0 else 0

    print(f"\n  Coefficient of Variation (lower = more consistent):")
    print(f"    Raw:        CV = {cv_raw:.3f}")
    print(f"    ATR-Norm:   CV = {cv_norm:.3f}")
    print(f"\n  RESULT: ATR normalization {'REDUCES' if cv_norm < cv_raw else 'DOES NOT REDUCE'} variance")
    print(f"  Variance reduction: {(1 - cv_norm/cv_raw)*100:.1f}%" if cv_raw > 0 else "")


def test_hypothesis_4_time_penalty(df, signals, direction_col):
    """
    Test Hypothesis 4: Time penalty reduces overholding losses
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS 4: Time Penalty Reduces Overholding Losses")
    print("=" * 80)

    # Test with different gamma values
    gammas = [0, 0.01, 0.02, 0.05, 0.1]

    for gamma in gammas:
        optimal_exits = []
        pnls_at_optimal = []

        for bar in signals:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, direction_col])
            if direction == 0 or idx + 10 >= len(df):
                continue

            entry_price = df.iloc[idx]['close']
            best_bar = 1
            best_score = float('-inf')
            best_pnl = 0.0  # Initialize before the inner loop

            for exit_bar in range(1, 11):
                if idx + exit_bar >= len(df):
                    break

                close = df.iloc[idx + exit_bar]['close']
                if direction == 1:
                    pnl = (close - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - close) / entry_price * 100

                # Score with time penalty
                score = pnl - gamma * exit_bar

                if score > best_score:
                    best_score = score
                    best_bar = exit_bar
                    best_pnl = pnl

            optimal_exits.append(best_bar)
            pnls_at_optimal.append(best_pnl)

        avg_exit = np.mean(optimal_exits)
        avg_pnl = np.mean(pnls_at_optimal)
        win_rate = sum(1 for p in pnls_at_optimal if p > 0) / len(pnls_at_optimal) * 100

        print(f"\n  γ = {gamma}:")
        print(f"    Avg optimal exit: bar +{avg_exit:.1f}")
        print(f"    Avg P&L at exit:  {avg_pnl:+.4f}%")
        print(f"    Win rate:         {win_rate:.1f}%")


def main():
    # Load data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return

    data = load_csv_data(str(csv_files[0]))
    print(f"Loaded {len(data)} bars")

    # Compute physics
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy'] = compute_energy(df['close'], lookback=20)
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']
    df['atr'] = compute_atr(df, period=14)
    df['momentum_5'] = df['close'].pct_change(5)
    df['counter_direction'] = -np.sign(df['momentum_5'])

    # Flow consistency
    df['return_sign'] = np.sign(df['close'].pct_change())
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Volume percentile
    vol_window = min(500, len(df))
    df['volume_pct'] = df['volume'].rolling(vol_window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Signals
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print(f"\nBerserker+ signals: {berserker_plus.sum()}")

    print("\n" + "=" * 80)
    print("EMPIRICAL TESTS OF ENERGY RECOVERY HYPOTHESES")
    print("From theorem_proofs.md")
    print("=" * 80)

    # Run hypothesis tests
    test_hypothesis_1_energy_efficiency(df, df[berserker_plus].index, 'counter_direction')
    test_hypothesis_2_ars_weights(df, df[berserker_plus].index, 'counter_direction')
    test_hypothesis_3_atr_normalization(df, df[berserker_plus].index, 'counter_direction')
    test_hypothesis_4_time_penalty(df, df[berserker_plus].index, 'counter_direction')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ENERGY RECOVERY HYPOTHESES")
    print("=" * 80)
    print("""
  From theorem_proofs.md, tested empirically:

  1. ENERGY EFFICIENCY (η = 68%):
     - Need calibration of normalization constant k
     - Concept valid, absolute value depends on scale

  2. ARS WEIGHTS (α, β, γ):
     - α (MFE reward): Higher = reward capturing opportunity
     - β (MAE penalty): Higher = punish adverse excursion
     - γ (Time decay): Small positive value reduces overholding

  3. ATR NORMALIZATION:
     - Tested for regime-invariance across high/low vol
     - Reduces coefficient of variation if successful

  4. TIME PENALTY:
     - Higher γ = earlier optimal exit
     - Balances capture vs time-decay of edge

  ARS FORMULA:
  R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
""")


if __name__ == "__main__":
    main()
