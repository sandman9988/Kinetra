#!/usr/bin/env python3
"""
Quick test of physics calculations and strategies.

Generates synthetic OHLCV data and tests all physics measures.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from backtesting import Backtest

# Import physics modules
from kinetra.physics_v7 import (
    compute_fractal_dimension_katz,
    compute_sample_entropy_fast,
    compute_center_of_mass,
    compute_com_divergence,
    compute_ftle_fast,
    compute_symc_ratio,
    compute_oscillator_state,
    compute_rl_state_vector,
    compute_action_mask,
    compute_vpin_proxy,
    compute_kurtosis_rolling,
    wasserstein_distance_1d,
    compute_regime_shift_wasserstein,
    OnlineMetricsTracker,
    OnlineWassersteinTracker,
    calculate_omega_ratio,
)
from kinetra.strategies_v7 import (
    BerserkerStrategy,
    SniperStrategy,
    MultiAgentV7Strategy,
)


def generate_synthetic_ohlcv(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with trending and ranging regimes."""
    np.random.seed(seed)

    # Start price
    price = 100.0
    prices = [price]

    # Generate price series with regime changes
    regime = 'trend'  # Start trending
    trend_dir = 1

    for i in range(n_bars - 1):
        # Regime change every ~200 bars
        if np.random.random() < 0.005:
            regime = 'range' if regime == 'trend' else 'trend'
            trend_dir = np.random.choice([-1, 1])

        if regime == 'trend':
            # Trending: drift + small noise
            change = trend_dir * 0.001 + np.random.normal(0, 0.005)
        else:
            # Ranging: mean reversion
            mean_price = np.mean(prices[-20:]) if len(prices) > 20 else price
            change = -0.1 * (price - mean_price) / mean_price + np.random.normal(0, 0.008)

        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    # Generate OHLC from close prices
    df = pd.DataFrame({
        'Close': prices,
    })

    # Generate Open, High, Low
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])

    # High/Low based on volatility
    volatility = df['Close'].rolling(20).std().fillna(0.5)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(np.random.normal(0, 1, n_bars)) * volatility
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(np.random.normal(0, 1, n_bars)) * volatility

    # Volume with regime-dependent patterns
    base_volume = 1000000
    df['Volume'] = base_volume * (1 + np.random.exponential(0.5, n_bars))

    # Add datetime index
    df.index = pd.date_range(start='2023-01-01', periods=n_bars, freq='h')

    return df


def test_physics_measures():
    """Test all physics measures."""
    print("=" * 60)
    print(" TESTING PHYSICS MEASURES")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_ohlcv(500)
    print(f"\nGenerated {len(df)} bars of synthetic data")
    print(f"Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

    close = df['Close'].values
    volume = df['Volume'].values
    high = df['High'].values
    low = df['Low'].values
    returns = df['Close'].pct_change().fillna(0).values

    # Test Fractal Dimension
    print("\n1. Fractal Dimension (Katz):")
    fd = compute_fractal_dimension_katz(close, window=50)
    print(f"   Mean: {np.mean(fd[50:]):.3f}")
    print(f"   Range: [{np.min(fd[50:]):.3f}, {np.max(fd[50:]):.3f}]")
    print(f"   Trend bars (FD < 1.25): {np.sum(fd < 1.25)}")
    print(f"   Chop bars (FD > 1.50): {np.sum(fd > 1.50)}")

    # Test Sample Entropy
    print("\n2. Sample Entropy (Fast):")
    se = compute_sample_entropy_fast(close, window=50)
    print(f"   Mean: {np.mean(se[50:]):.4f}")
    print(f"   Range: [{np.min(se[50:]):.4f}, {np.max(se[50:]):.4f}]")

    # Test Center of Mass
    print("\n3. Center of Mass:")
    com = compute_center_of_mass(close, volume, window=20)
    com_div = compute_com_divergence(close, volume, window=20)
    print(f"   CoM Mean: {np.mean(com):.2f}")
    print(f"   Divergence Range: [{np.min(com_div):.3f}, {np.max(com_div):.3f}]")

    # Test FTLE
    print("\n4. FTLE (Chaos Detector):")
    ftle = compute_ftle_fast(close, window=50)
    print(f"   Mean: {np.mean(ftle[50:]):.4f}")
    print(f"   Range: [{np.min(ftle[50:]):.4f}, {np.max(ftle[50:]):.4f}]")
    print(f"   High chaos bars (FTLE > 0.1): {np.sum(np.abs(ftle) > 0.1)}")

    # Test SymC Ratio
    print("\n5. SymC Ratio (Damping):")
    symc = compute_symc_ratio(close, volume, lookback=20)
    print(f"   Mean: {np.mean(symc[20:]):.3f}")
    print(f"   Underdamped (χ < 0.8): {np.sum(symc < 0.8)} bars")
    print(f"   Overdamped (χ > 1.2): {np.sum(symc > 1.2)} bars")
    print(f"   Critical (0.8-1.2): {np.sum((symc >= 0.8) & (symc <= 1.2))} bars")

    # Test VPIN
    print("\n6. VPIN (Toxicity):")
    vpin = compute_vpin_proxy(close, volume, window=50)
    print(f"   Mean: {np.mean(vpin[50:]):.3f}")
    print(f"   High toxicity bars (VPIN > 0.7): {np.sum(vpin > 0.7)}")

    # Test Kurtosis
    print("\n7. Rolling Kurtosis:")
    kurt = compute_kurtosis_rolling(returns, window=50)
    print(f"   Mean: {np.mean(kurt[50:]):.3f}")
    print(f"   Fat tail bars (kurt > 3): {np.sum(kurt > 3)}")

    # Test Wasserstein
    print("\n8. Wasserstein Distance:")
    w_dist = compute_regime_shift_wasserstein(returns, window=50, reference_window=100)
    print(f"   Mean: {np.mean(w_dist[150:]):.6f}")
    print(f"   Regime shift bars (W > 0.005): {np.sum(w_dist > 0.005)}")

    # Test RL State Vector
    print("\n9. RL State Vector:")
    state = compute_rl_state_vector(high, low, close, volume, window=50)
    print(f"   Features: {list(state.keys())}")
    print(f"   Trend regime: {np.sum(state['regime_trend'])} bars")
    print(f"   Range regime: {np.sum(state['regime_range'])} bars")
    print(f"   Chaos regime: {np.sum(state['regime_chaos'])} bars")

    # Test Action Mask
    print("\n10. Action Masking:")
    mask = compute_action_mask(state, 200)
    print(f"   Allowed at bar 200: {mask}")

    return df


def test_online_trackers():
    """Test online tracking algorithms."""
    print("\n" + "=" * 60)
    print(" TESTING ONLINE TRACKERS")
    print("=" * 60)

    # Generate returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 200)

    # Test OnlineMetricsTracker
    print("\n1. OnlineMetricsTracker:")
    tracker = OnlineMetricsTracker(initial_equity=100000)

    for i, ret in enumerate(returns):
        pnl = 100000 * ret  # Simulate trade PnL
        tracker.record_trade(pnl)

    metrics = tracker.get_metrics()
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"   Omega Ratio: {metrics['omega_ratio']:.3f}")
    print(f"   Ulcer Index: {metrics['ulcer_index']:.3f}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")

    # Test OnlineWassersteinTracker
    print("\n2. OnlineWassersteinTracker:")
    w_tracker = OnlineWassersteinTracker(current_size=50, historical_size=100)

    for i, ret in enumerate(returns):
        w_dist = w_tracker.update(ret)
        if i == 150:
            print(f"   At bar 150: W = {w_dist:.6f}, Regime shift: {w_tracker.is_regime_shift}")

    print(f"   Final W distance: {w_tracker.distance:.6f}")


def test_strategies():
    """Test backtesting strategies."""
    print("\n" + "=" * 60)
    print(" TESTING STRATEGIES")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_ohlcv(1000, seed=123)
    print(f"\nData: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    strategies = {
        'Berserker': BerserkerStrategy,
        'Sniper': SniperStrategy,
        'MultiAgent': MultiAgentV7Strategy,
    }

    results = []
    for name, strategy_class in strategies.items():
        print(f"\n{name}:")
        try:
            bt = Backtest(
                df,
                strategy_class,
                cash=100000,
                commission=0.001,
                trade_on_close=True,
                exclusive_orders=True
            )
            stats = bt.run()

            ret = float(stats['Return [%]'])
            trades = int(stats['# Trades'])
            sharpe = float(stats['Sharpe Ratio']) if not np.isnan(stats['Sharpe Ratio']) else 0
            win_rate = float(stats['Win Rate [%]']) if not np.isnan(stats['Win Rate [%]']) else 0
            max_dd = float(stats['Max. Drawdown [%]'])

            print(f"   Return: {ret:+.2f}%")
            print(f"   Trades: {trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Sharpe: {sharpe:.3f}")
            print(f"   Max DD: {max_dd:.2f}%")

            # Calculate Omega
            if hasattr(stats, '_trades') and len(stats._trades) > 0:
                trade_returns = stats._trades['ReturnPct'].values / 100
                omega = calculate_omega_ratio(pd.Series(trade_returns))
                print(f"   Omega: {omega:.3f}")

            results.append({
                'strategy': name,
                'return': ret,
                'trades': trades,
                'win_rate': win_rate,
                'sharpe': sharpe,
                'max_dd': max_dd,
            })

        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print("\n" + "-" * 40)
        print("SUMMARY:")
        best = max(results, key=lambda x: x['return'])
        print(f"Best strategy: {best['strategy']} with {best['return']:+.2f}%")

    return results


def main():
    print("\n" + "=" * 60)
    print(" KINETRA PHYSICS BACKTESTER - TEST SUITE")
    print(" Energy-Transfer Trading Theorem v7.0")
    print("=" * 60)

    # Test physics measures
    df = test_physics_measures()

    # Test online trackers
    test_online_trackers()

    # Test strategies
    test_strategies()

    print("\n" + "=" * 60)
    print(" ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
