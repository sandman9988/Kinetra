"""
Batch backtesting script for multiple instruments.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from kinetra.physics_engine import PhysicsEngine
from kinetra.backtest_engine import BacktestEngine


def main():
    """Run batch backtests."""
    parser = argparse.ArgumentParser(description='Kinetra Batch Backtest')
    parser.add_argument('--instrument', type=str, default='BTCUSD', help='Instrument to backtest')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    parser.add_argument('--runs', type=int, default=1, help='Number of Monte Carlo runs')
    
    args = parser.parse_args()
    
    print(f"🚀 Kinetra Batch Backtest")
    print(f"   Instrument: {args.instrument}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Monte Carlo Runs: {args.runs}")
    print("=" * 60)
    
    # Generate synthetic data (replace with real data loading)
    print("Loading data...")
    np.random.seed(42)
    prices = pd.Series(np.random.randn(1000).cumsum() + 100)
    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.randn(1000) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(1000) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(1000) * 0.002)),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Compute physics state
    print("Computing physics state...")
    physics = PhysicsEngine(lookback=20)
    state = physics.compute_physics_state(data['close'])
    
    # Run backtest
    print(f"Running {args.runs} backtest(s)...")
    engine = BacktestEngine()
    
    if args.runs == 1:
        result = engine.run_backtest(data, agent=None)
        print("\n📊 Results:")
        print(f"   Omega Ratio: {result['omega_ratio']:.2f}")
        print(f"   Z-Factor: {result['z_factor']:.2f}")
        print(f"   Energy Captured: {result['energy_captured_pct']:.1%}")
        print(f"   Final Equity: ${result['final_equity']:,.2f}")
    else:
        results = engine.monte_carlo_validation(data, agent=None, n_runs=args.runs)
        print("\n📊 Monte Carlo Results:")
        print(f"   Omega Ratio: {results['omega_ratio'].mean():.2f} ± {results['omega_ratio'].std():.2f}")
        print(f"   Z-Factor: {results['z_factor'].mean():.2f} ± {results['z_factor'].std():.2f}")
        print(f"   Energy Captured: {results['energy_captured_pct'].mean():.1%}")
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
