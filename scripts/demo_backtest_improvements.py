"""
Demo script showing BacktestEngine improvements.

Demonstrates:
1. Timeframe-aware metric calculations
2. Margin level tracking
3. Safe math operations
4. MT5-style logging (optional)
5. Comprehensive data validation
"""

import numpy as np
import pandas as pd

from kinetra.backtest_engine import BacktestEngine
from kinetra.symbol_spec import SymbolSpec, CommissionSpec, CommissionType


def create_eurusd_spec():
    """Create EURUSD symbol specification."""
    return SymbolSpec(
        symbol="EURUSD",
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100000,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=2.0,
        commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
        slippage_avg=0.5,
    )


def create_trending_data(n_bars=200, trend_strength=0.0001):
    """Create trending price data for demonstration."""
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")
    
    # Strong uptrend + noise
    trend = np.arange(n_bars) * trend_strength
    noise = np.random.randn(n_bars) * 0.00005
    prices = 1.08 + trend + noise.cumsum()
    
    return pd.DataFrame({
        "time": dates,
        "open": prices,
        "high": prices + np.abs(np.random.randn(n_bars) * 0.0002),
        "low": prices - np.abs(np.random.randn(n_bars) * 0.0002),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_bars),
    })


def simple_trend_signal(row, physics_state, bar_index):
    """Simple moving average crossover signal."""
    if bar_index < 20:
        return 0  # Not enough data
    
    # Get recent closes
    lookback = 20
    if bar_index >= len(physics_state.get("close", [])):
        return 0
    
    # Simple: buy if price is above entry, sell if below
    if bar_index % 30 == 10:  # Buy every 30 bars
        return 1
    elif bar_index % 30 == 25:  # Sell 15 bars later
        return -1
    
    return 0  # Hold


def main():
    """Run backtest demonstration."""
    print("=" * 80)
    print("BacktestEngine Improvements Demonstration")
    print("=" * 80)
    print()
    
    # Create symbol spec
    eurusd_spec = create_eurusd_spec()
    print(f"Symbol: {eurusd_spec.symbol}")
    print(f"Tick size: {eurusd_spec.tick_size}")
    print(f"Contract size: {eurusd_spec.contract_size}")
    print(f"Spread: {eurusd_spec.spread_points} points")
    print()
    
    # Create engine with new features
    print("Creating BacktestEngine with new features:")
    print("- Timeframe: H1 (for correct annualization)")
    print("- Leverage: 100:1 (for margin calculations)")
    print("- Logging: Disabled (can be enabled with enable_logging=True)")
    print()
    
    engine = BacktestEngine(
        initial_capital=10000.0,
        risk_per_trade=0.02,  # 2% risk per trade
        max_positions=1,
        timeframe="H1",  # NEW: Timeframe parameter
        leverage=100.0,   # NEW: Leverage parameter
        enable_logging=False,  # NEW: Optional MT5-style logging
    )
    
    # Create data
    print("Generating trending market data (200 bars)...")
    data = create_trending_data(200, trend_strength=0.0001)
    print(f"Price range: {data['close'].min():.5f} - {data['close'].max():.5f}")
    print()
    
    # Run backtest
    print("Running backtest with simple trend-following strategy...")
    result = engine.run_backtest(
        data=data,
        symbol_spec=eurusd_spec,
        signal_func=simple_trend_signal,  # Use custom signal function
    )
    print()
    
    # Display results
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print()
    
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate * 100:.2f}%")
    print()
    
    print(f"Gross P&L: ${result.total_gross_pnl:.2f}")
    print(f"Total Costs: ${result.total_costs:.2f}")
    print(f"Net P&L: ${result.total_net_pnl:.2f}")
    print()
    
    print("Cost Breakdown:")
    print(f"  Spread: ${result.total_spread_cost:.2f}")
    print(f"  Commission: ${result.total_commission:.2f}")
    print(f"  Slippage: ${result.total_slippage:.2f}")
    print(f"  Swap: ${result.total_swap_cost:.2f}")
    print()
    
    print(f"Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f} (annualized for H1 timeframe)")
    print(f"Sortino Ratio: {result.sortino_ratio:.3f}")
    print(f"Omega Ratio: {result.omega_ratio:.3f}")
    print(f"Z-Factor: {result.z_factor:.3f}")
    print()
    
    # NEW: Margin tracking
    print("=" * 80)
    print("NEW FEATURE: Margin Level Tracking")
    print("=" * 80)
    print(f"Minimum Margin Level: {result.min_margin_level:.2f}%")
    if result.min_margin_level < float("inf"):
        print("✓ Margin levels were tracked throughout backtest")
        if result.min_margin_level < 100:
            print("⚠ Warning: Margin call occurred during backtest!")
    else:
        print("(No positions opened, so margin level was always infinite)")
    print()
    
    # Physics metrics
    print("=" * 80)
    print("PHYSICS-BASED METRICS")
    print("=" * 80)
    print(f"Energy Captured: {result.energy_captured_pct * 100:.2f}%")
    print(f"MFE Capture: {result.mfe_capture_pct * 100:.2f}%")
    print()
    
    # Show a few trades
    if result.trades:
        print("=" * 80)
        print(f"SAMPLE TRADES (showing first {min(3, len(result.trades))})")
        print("=" * 80)
        for i, trade in enumerate(result.trades[:3]):
            print(f"\nTrade #{trade.trade_id}:")
            print(f"  Direction: {trade.direction.value}")
            print(f"  Entry: {trade.entry_price:.5f} @ {trade.entry_time}")
            print(f"  Exit: {trade.exit_price:.5f} @ {trade.exit_time}")
            print(f"  Volume: {trade.lots:.2f} lots")
            print(f"  Gross P&L: ${trade.gross_pnl:.2f}")
            print(f"  Costs: ${trade.total_cost:.2f}")
            print(f"  Net P&L: ${trade.net_pnl:.2f}")
            print(f"  MFE: {trade.mfe * 10000:.1f} pips")
            print(f"  MAE: {trade.mae * 10000:.1f} pips")
            print(f"  Regime: {trade.regime_at_entry}")
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key improvements demonstrated:")
    print("✓ Timeframe parameter for correct metric annualization")
    print("✓ Margin level tracking (min margin level: {:.2f}%)".format(result.min_margin_level))
    print("✓ Safe math operations (no crashes on edge cases)")
    print("✓ Comprehensive input validation (NaN/Inf checks)")
    print("✓ Defensive programming throughout")
    print()


if __name__ == "__main__":
    main()
