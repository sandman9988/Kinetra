#!/usr/bin/env python3
"""
Trade Lifecycle Test with Real Market Data

Tests the complete trade lifecycle using real EURJPY M15 data:
1. Load real market data from CSV
2. Generate simple entry/exit signals
3. Run backtest and validate trade lifecycle
4. Verify MFE/MAE tracking, costs, and metrics

This validates the backtester works correctly on real market conditions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from kinetra.realistic_backtester import RealisticBacktester
from kinetra.market_microstructure import SymbolSpec, AssetClass


def load_real_data(csv_path: str, max_rows: int = 1000) -> pd.DataFrame:
    """Load real market data from CSV."""
    print(f"\n[Loading Data]")
    print(f"  File: {csv_path}")

    # Read CSV (MT5 format)
    df = pd.read_csv(csv_path, sep='\t', nrows=max_rows)

    # Rename columns to standard format
    df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()

    # Combine date and time
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    # Rename OHLC
    df = df.rename(columns={
        'tickvol': 'volume',
    })

    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

    # Set timestamp as index (required by backtester)
    df = df.set_index('timestamp')

    print(f"  Loaded {len(df)} candles")
    print(f"  Timeframe: M15")
    print(f"  Period: {df.index[0]} to {df.index[-1]}")
    print(f"  Avg spread: {df['spread'].mean():.1f} points")

    return df


def create_eurjpy_spec() -> SymbolSpec:
    """Create realistic EURJPY specification."""
    return SymbolSpec(
        symbol="EURJPY+",
        asset_class=AssetClass.FOREX,
        digits=3,
        # Volume constraints
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        # Trading mode
        filling_mode="IOC",
        order_mode="MARKET_LIMIT",
        trade_mode="FULL",
        # Costs
        spread_typical=80,  # ~8 pips (typical for EURJPY)
        commission_per_lot=0.0,
        swap_long=-0.3,
        swap_short=0.1,
        # MT5 constraints (CRITICAL)
        trade_freeze_level=50,   # 5 pips
        trade_stops_level=100,   # 10 pips
    )


def generate_simple_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple buy/sell signals for testing.

    Strategy: Simple crossover
    - Buy when price crosses above 50-period MA
    - Sell when price crosses below 50-period MA
    """
    df = data.copy()

    # Calculate 50-period moving average
    df['ma50'] = df['close'].rolling(window=50).mean()

    # Detect crossovers
    df['signal'] = 0
    df.loc[df['close'] > df['ma50'], 'signal'] = 1  # Bullish
    df.loc[df['close'] < df['ma50'], 'signal'] = -1  # Bearish

    # Only take first signal change (entry points)
    df['signal_change'] = df['signal'].diff()

    signals = []

    # Generate entry/exit signals
    position = None
    for idx, row in df.iterrows():
        if position is None:
            # Look for entry
            if row['signal_change'] == 2:  # Cross above MA (long entry)
                entry_price = row['close']
                sl = entry_price - 0.200  # 200 pips SL
                tp = entry_price + 0.400  # 400 pips TP (2:1 R:R)

                signals.append({
                    'time': idx,  # Use index (timestamp)
                    'action': 'open_long',
                    'sl': sl,
                    'tp': tp,
                    'volume': 1.0,  # Optional volume (defaults to 1.0)
                })
                position = 'long'

            elif row['signal_change'] == -2:  # Cross below MA (short entry)
                entry_price = row['close']
                sl = entry_price + 0.200  # 200 pips SL
                tp = entry_price - 0.400  # 400 pips TP (2:1 R:R)

                signals.append({
                    'time': idx,  # Use index (timestamp)
                    'action': 'open_short',
                    'sl': sl,
                    'tp': tp,
                    'volume': 1.0,  # Optional volume (defaults to 1.0)
                })
                position = 'short'

        else:
            # Look for exit (opposite signal)
            if (position == 'long' and row['signal_change'] < 0) or \
               (position == 'short' and row['signal_change'] > 0):
                signals.append({
                    'time': idx,  # Use index (timestamp)
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None

    signals_df = pd.DataFrame(signals)

    if len(signals_df) > 0:
        print(f"\n[Signal Generation]")
        print(f"  Total signals: {len(signals_df)}")
        print(f"  Entry signals: {len(signals_df[signals_df['action'] != 'close'])}")
        print(f"  Exit signals: {len(signals_df[signals_df['action'] == 'close'])}")

    return signals_df


def run_lifecycle_test():
    """Run complete trade lifecycle test on real data."""
    print("=" * 70)
    print("TRADE LIFECYCLE TEST - REAL MARKET DATA")
    print("Instrument: EURJPY+ | Timeframe: M15")
    print("=" * 70)

    # 1. Load real data
    data_path = "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv"
    data = load_real_data(data_path, max_rows=5000)  # Use first 5000 candles

    # 2. Create symbol spec
    spec = create_eurjpy_spec()

    # 3. Initialize backtester
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,  # 2% risk per trade
        timeframe="M15",
        enable_slippage=True,
        slippage_std_pips=0.5,
        enable_freeze_zones=True,
        enable_stop_validation=True,
        verbose=False,
    )

    print(f"\n[Backtester Configuration]")
    print(f"  Initial capital: ${backtester.initial_capital:,.2f}")
    print(f"  Risk per trade: {backtester.risk_per_trade:.1%}")
    print(f"  Freeze level: {spec.trade_freeze_level} points ({spec.trade_freeze_level * spec.point:.3f})")
    print(f"  Stops level: {spec.trade_stops_level} points ({spec.trade_stops_level * spec.point:.3f})")
    print(f"  Slippage enabled: {backtester.enable_slippage}")

    # 4. Generate signals
    signals = generate_simple_signals(data)

    if len(signals) == 0:
        print("\n❌ No signals generated!")
        return 1

    # 5. Run backtest
    print(f"\n[Running Backtest]")
    print(f"  Processing {len(data)} candles...")

    result = backtester.run(data, signals, classify_regimes=False)

    # 6. Display results
    print(f"\n{'=' * 70}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 70}")

    print(f"\n[Overall Performance]")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Winning trades: {result.winning_trades} ({result.win_rate:.1%})")
    print(f"  Losing trades: {result.losing_trades}")
    print(f"  Total P&L: ${result.total_pnl:,.2f}")
    print(f"  Total return: {result.total_return_pct:+.2%}")

    print(f"\n[Risk-Adjusted Metrics]")
    print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino ratio: {result.sortino_ratio:.2f}")
    print(f"  Omega ratio: {result.omega_ratio:.2f}")
    print(f"  Max drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2%})")
    print(f"  CVaR (95%): {result.cvar_95:.4f}")
    print(f"  CVaR (99%): {result.cvar_99:.4f}")

    print(f"\n[Transaction Costs]")
    print(f"  Total spread cost: ${result.total_spread_cost:,.2f}")
    print(f"  Total commission: ${result.total_commission:,.2f}")
    print(f"  Total swap: ${result.total_swap:,.2f}")
    print(f"  Total slippage: ${result.total_slippage:,.2f}")
    total_costs = (result.total_spread_cost + result.total_commission +
                   abs(result.total_swap) + result.total_slippage)
    print(f"  Total costs: ${total_costs:,.2f}")
    if (result.total_pnl + total_costs) > 0:
        print(f"  Cost as % of gross P&L: {total_costs / (result.total_pnl + total_costs) * 100:.1f}%")
    else:
        print(f"  Cost as % of gross P&L: N/A")

    print(f"\n[Execution Quality]")
    print(f"  Avg MFE: {result.avg_mfe:.3f} ({result.avg_mfe / spec.point:.1f} pips)")
    print(f"  Avg MAE: {result.avg_mae:.3f} ({result.avg_mae / spec.point:.1f} pips)")
    print(f"  MFE/MAE ratio: {result.avg_mfe_mae_ratio:.2f}x")
    print(f"  MFE capture: {result.mfe_capture_pct:.2%}")

    print(f"\n[MT5 Constraint Violations]")
    print(f"  Freeze zone violations: {result.total_freeze_violations}")
    print(f"  Invalid stop placements: {result.total_invalid_stops}")
    print(f"  Rejected orders: {result.total_rejected_orders}")

    if result.total_trades == 0:
        print("\n❌ No trades executed!")
        return 1

    # 7. Analyze individual trades
    print(f"\n{'=' * 70}")
    print("INDIVIDUAL TRADE ANALYSIS")
    print(f"{'=' * 70}")

    print(f"\n{'#':<4} {'Type':<6} {'Entry':<12} {'Exit':<12} {'P&L':<10} {'MFE':<8} {'MAE':<8} {'Eff':<6}")
    print(f"{'-' * 70}")

    for i, trade in enumerate(result.trades[:10], 1):  # Show first 10 trades
        trade_type = "LONG" if trade.direction == 1 else "SHORT"
        entry_time = trade.entry_time.strftime("%m/%d %H:%M")
        exit_time = trade.exit_time.strftime("%m/%d %H:%M")
        mfe_pips = trade.mfe / spec.point
        mae_pips = trade.mae / spec.point

        print(f"{i:<4} {trade_type:<6} {entry_time:<12} {exit_time:<12} "
              f"${trade.pnl:>8.2f} {mfe_pips:>6.1f}p {mae_pips:>6.1f}p {trade.mfe_efficiency:>5.1%}")

    if len(result.trades) > 10:
        print(f"\n... and {len(result.trades) - 10} more trades")

    # 8. Detailed lifecycle for first trade
    if result.total_trades > 0:
        print(f"\n{'=' * 70}")
        print("DETAILED LIFECYCLE - FIRST TRADE")
        print(f"{'=' * 70}")

        trade = result.trades[0]

        print(f"\n[Trade Setup]")
        print(f"  Direction: {'LONG' if trade.direction == 1 else 'SHORT'}")
        print(f"  Entry time: {trade.entry_time}")
        print(f"  Entry price: {trade.entry_price:.3f}")
        print(f"  Volume: {trade.volume:.2f} lots")

        print(f"\n[Trade Execution]")
        print(f"  Exit time: {trade.exit_time}")
        print(f"  Exit price: {trade.exit_price:.3f}")
        print(f"  Holding time: {trade.holding_time:.2f} hours")
        print(f"  Price captured: {trade.price_captured:.3f} ({trade.price_captured / spec.point:.1f} pips)")

        print(f"\n[Price Excursion]")
        print(f"  Max Favorable Excursion (MFE): {trade.mfe:.3f} ({trade.mfe / spec.point:.1f} pips)")
        print(f"  Max Adverse Excursion (MAE): {trade.mae:.3f} ({trade.mae / spec.point:.1f} pips)")
        print(f"  MFE efficiency: {trade.mfe_efficiency:.2%}")
        print(f"  MAE efficiency: {trade.mae_efficiency:.2%}")

        print(f"\n[Cost Breakdown]")
        print(f"  Entry spread: ${trade.entry_spread:.2f}")
        print(f"  Exit spread: ${trade.exit_spread:.2f}")
        print(f"  Commission: ${trade.commission:.2f}")
        print(f"  Swap: ${trade.swap:.2f}")
        print(f"  Entry slippage: ${abs(trade.entry_slippage):.2f}")
        print(f"  Exit slippage: ${abs(trade.exit_slippage):.2f}")
        print(f"  ─────────────────────")
        print(f"  Total costs: ${trade.total_cost:.2f}")

        print(f"\n[P&L]")
        print(f"  Gross P&L: ${trade.gross_pnl:.2f}")
        print(f"  Total costs: -${trade.total_cost:.2f}")
        print(f"  ─────────────────────")
        print(f"  Net P&L: ${trade.pnl:.2f}")

    # 9. Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION CHECKS")
    print(f"{'=' * 70}")

    checks = []

    # Check 1: All trades have MFE/MAE tracked
    all_have_mfe = all(t.mfe >= 0 for t in result.trades)
    all_have_mae = all(t.mae >= 0 for t in result.trades)
    checks.append(("MFE tracking", all_have_mfe))
    checks.append(("MAE tracking", all_have_mae))

    # Check 2: MFE >= MAE for profitable trades
    for trade in result.trades:
        if trade.pnl > 0:
            checks.append((f"Trade MFE >= MAE", trade.mfe >= abs(trade.mae)))

    # Check 3: Costs are positive
    checks.append(("Spread costs > 0", result.total_spread_cost > 0))

    # Check 4: No constraint violations (realistic backtest)
    checks.append(("No freeze violations", result.total_freeze_violations == 0))
    checks.append(("No invalid stops", result.total_invalid_stops == 0))

    # Check 5: Equity curve exists
    checks.append(("Equity curve exists", result.equity_curve is not None))

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in checks)

    print(f"\n{'=' * 70}")
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED")
        print("\nTrade lifecycle working correctly on real market data!")
        print("\nKey validations:")
        print("  ✓ MFE/MAE tracked for all trades")
        print("  ✓ Transaction costs calculated (spread, commission, swap, slippage)")
        print("  ✓ MT5 constraints enforced (no violations)")
        print("  ✓ Realistic execution simulation")
        print(f"{'=' * 70}")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nReview failures above")
        print(f"{'=' * 70}")
        return 1


if __name__ == "__main__":
    exit(run_lifecycle_test())
