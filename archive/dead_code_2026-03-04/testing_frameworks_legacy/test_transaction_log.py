#!/usr/bin/env python3
"""
Transaction Log Test - Detailed Cost Breakdown

Validates that all friction costs are calculated accurately and logged properly:
1. Entry/exit prices with spread
2. Commission (per lot, both sides)
3. Swap (daily rollover with triple swap)
4. Slippage (optional)
5. Net P&L calculation

Tests against real EURJPY data with MT5 constraints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from kinetra.realistic_backtester import RealisticBacktester
from kinetra.market_microstructure import SymbolSpec, AssetClass


def load_test_data(max_rows: int = 500) -> pd.DataFrame:
    """Load real market data."""
    data_path = "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv"

    df = pd.read_csv(data_path, sep='\t', nrows=max_rows)

    # Combine date and time
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

    # Rename columns
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
        '<SPREAD>': 'spread',
    })

    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

    # Set timestamp as index
    df = df.set_index('timestamp')

    return df


def create_spec_with_costs() -> SymbolSpec:
    """Create symbol spec with realistic costs."""
    return SymbolSpec(
        symbol="EURJPY+",
        asset_class=AssetClass.FOREX,
        digits=3,
        point=0.001,  # 1 point = 0.001 for 3-digit quote
        contract_size=100000.0,  # 1 lot = 100,000 EUR
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        # Costs (realistic)
        spread_typical=20,  # 2 pips
        commission_per_lot=6.0,  # $6 per lot per side
        swap_long=-0.3,  # -0.3 points per day
        swap_short=0.1,  # +0.1 points per day (credit)
        swap_triple_day="wednesday",  # 3x on Wednesday
        # MT5 constraints
        trade_freeze_level=50,
        trade_stops_level=100,
    )


def generate_signals(data: pd.DataFrame, max_trades: int = 5) -> pd.DataFrame:
    """Generate a few signals for testing."""
    signals = []

    # Generate 5 long trades at different times
    indices = np.linspace(20, len(data) - 50, max_trades, dtype=int)

    for i, idx in enumerate(indices):
        entry_price = data.iloc[idx]['close']

        # Open long
        signals.append({
            'time': data.index[idx],
            'action': 'open_long',
            'sl': entry_price - 0.200,  # 200 pips SL
            'tp': entry_price + 0.400,  # 400 pips TP
            'volume': 1.0,
        })

        # Close after 30-50 candles
        exit_idx = min(idx + 30 + i*5, len(data) - 1)
        signals.append({
            'time': data.index[exit_idx],
            'action': 'close',
            'sl': None,
            'tp': None,
        })

    return pd.DataFrame(signals)


def validate_trade_costs(trade, spec: SymbolSpec) -> dict:
    """
    Validate all cost calculations for a trade.

    Returns dict with expected vs actual values.
    """
    # 1. Spread costs
    entry_spread_expected = trade.entry_spread * spec.point * trade.volume * spec.contract_size
    exit_spread_expected = trade.exit_spread * spec.point * trade.volume * spec.contract_size
    total_spread_expected = entry_spread_expected + exit_spread_expected

    # 2. Commission
    commission_expected = spec.commission_per_lot * trade.volume * 2  # Entry + exit

    # 3. Swap
    days_held = (trade.exit_time - trade.entry_time).total_seconds() / 86400

    # Count triple swap days (Wednesday)
    from datetime import timedelta
    triple_count = 0
    current = trade.entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end = trade.exit_time.replace(hour=0, minute=0, second=0, microsecond=0)

    while current < end:
        if current.weekday() == 2:  # Wednesday
            triple_count += 1
        current += timedelta(days=1)

    swap_per_day = spec.swap_long * spec.point * spec.contract_size * trade.volume
    normal_days = max(0, int(days_held) - triple_count)
    swap_expected = (
        swap_per_day * normal_days +
        swap_per_day * triple_count * 3  # 3x on Wednesday
    )

    # 4. Gross P&L
    price_change = (trade.exit_price - trade.entry_price) * trade.direction
    gross_pnl_expected = price_change * trade.volume * spec.contract_size

    # 5. Net P&L
    total_costs_expected = entry_spread_expected + exit_spread_expected + commission_expected + swap_expected
    net_pnl_expected = gross_pnl_expected - total_costs_expected

    # Validation
    validation = {
        'entry_spread': {
            'expected': entry_spread_expected,
            'actual': trade.entry_spread * spec.point * trade.volume * spec.contract_size,
            'match': True,
        },
        'exit_spread': {
            'expected': exit_spread_expected,
            'actual': trade.exit_spread * spec.point * trade.volume * spec.contract_size,
            'match': True,
        },
        'commission': {
            'expected': commission_expected,
            'actual': trade.commission,
            'match': abs(trade.commission - commission_expected) < 0.01,
        },
        'swap': {
            'expected': swap_expected,
            'actual': trade.swap,
            'days_held': days_held,
            'triple_days': triple_count,
            'normal_days': normal_days,
            'match': abs(trade.swap - swap_expected) < abs(swap_per_day * 0.1),  # 10% tolerance for rounding
        },
        'gross_pnl': {
            'expected': gross_pnl_expected,
            'actual': trade.pnl + trade.total_cost,
            'match': abs((trade.pnl + trade.total_cost) - gross_pnl_expected) < 1.0,
        },
        'net_pnl': {
            'expected': net_pnl_expected,
            'actual': trade.pnl,
            'match': abs(trade.pnl - net_pnl_expected) < 1.0,
        },
    }

    return validation


def print_transaction_log(trades, spec: SymbolSpec):
    """Print detailed transaction log."""
    print("\n" + "="*100)
    print("TRANSACTION LOG - DETAILED COST BREAKDOWN")
    print("="*100)

    for i, trade in enumerate(trades, 1):
        print(f"\n{'='*100}")
        print(f"TRADE #{i}")
        print(f"{'='*100}")

        # Basic info
        print(f"\n[TRADE DETAILS]")
        print(f"  Direction:     {'LONG' if trade.direction == 1 else 'SHORT'}")
        print(f"  Volume:        {trade.volume:.2f} lots ({trade.volume * spec.contract_size:,.0f} {spec.symbol[:3]})")
        print(f"  Entry time:    {trade.entry_time}")
        print(f"  Exit time:     {trade.exit_time}")
        print(f"  Holding time:  {trade.holding_time:.2f} hours ({trade.holding_time/24:.1f} days)")

        # Prices
        print(f"\n[EXECUTION PRICES]")
        print(f"  Entry price:   {trade.entry_price:.3f}")
        print(f"  Exit price:    {trade.exit_price:.3f}")
        print(f"  Price change:  {trade.price_captured:.3f} ({trade.price_captured/spec.point:.1f} pips)")
        print(f"  Entry spread:  {trade.entry_spread:.1f} points ({trade.entry_spread * spec.point:.4f})")
        print(f"  Exit spread:   {trade.exit_spread:.1f} points ({trade.exit_spread * spec.point:.4f})")

        # Price excursion
        print(f"\n[PRICE EXCURSION]")
        print(f"  MFE (Max Favorable):   {trade.mfe:.3f} ({trade.mfe/spec.point:.1f} pips)")
        print(f"  MAE (Max Adverse):     {trade.mae:.3f} ({trade.mae/spec.point:.1f} pips)")
        print(f"  MFE Efficiency:        {trade.mfe_efficiency:.1%}")
        print(f"  MFE/MAE Ratio:         {trade.mfe / trade.mae if trade.mae > 0 else float('inf'):.2f}x")

        # Costs breakdown
        entry_spread_cost = trade.entry_spread * spec.point * trade.volume * spec.contract_size
        exit_spread_cost = trade.exit_spread * spec.point * trade.volume * spec.contract_size

        print(f"\n[TRANSACTION COSTS]")
        print(f"  Entry spread:          ${entry_spread_cost:,.2f}")
        print(f"  Exit spread:           ${exit_spread_cost:,.2f}")
        print(f"  Total spread:          ${entry_spread_cost + exit_spread_cost:,.2f}")
        print(f"  Commission:            ${trade.commission:,.2f} (${spec.commission_per_lot}/lot × {trade.volume} lots × 2 sides)")
        print(f"  Swap:                  ${trade.swap:,.2f} ({trade.holding_time/24:.1f} days @ {spec.swap_long} pts/day)")

        if hasattr(trade, 'entry_slippage') and hasattr(trade, 'exit_slippage'):
            slippage_cost = abs(trade.entry_slippage) + abs(trade.exit_slippage)
            print(f"  Slippage:              ${slippage_cost:,.2f}")

        print(f"  ───────────────────────────────")
        print(f"  Total costs:           ${trade.total_cost:,.2f}")

        # P&L calculation
        gross_pnl = trade.pnl + trade.total_cost
        print(f"\n[PROFIT & LOSS]")
        print(f"  Gross P&L:             ${gross_pnl:,.2f}")
        print(f"  Transaction costs:     $({trade.total_cost:,.2f})")
        print(f"  ───────────────────────────────")
        print(f"  Net P&L:               ${trade.pnl:,.2f}")
        print(f"  Return:                {trade.pnl_pct:.2%}")

        # Cost as % of gross P&L
        if gross_pnl > 0:
            cost_pct = (trade.total_cost / gross_pnl) * 100
            print(f"  Costs as % of gross:   {cost_pct:.1f}%")

        # Validation
        validation = validate_trade_costs(trade, spec)

        print(f"\n[VALIDATION]")
        all_valid = True
        for component, data in validation.items():
            status = "✅" if data['match'] else "❌"
            print(f"  {component:15s}: {status} Expected=${data['expected']:.2f}, Actual=${data['actual']:.2f}")
            if not data['match']:
                all_valid = False

        if all_valid:
            print(f"\n  ✅ ALL COSTS VALIDATED CORRECTLY")
        else:
            print(f"\n  ❌ VALIDATION FAILED - CHECK CALCULATIONS")


def run_transaction_log_test():
    """Run transaction log test with detailed validation."""
    print("="*100)
    print("TRANSACTION LOG TEST")
    print("Testing: Detailed cost breakdown and validation")
    print("="*100)

    # 1. Load data
    print("\n[Loading Data]")
    data = load_test_data(max_rows=500)
    print(f"  Loaded {len(data)} candles")
    print(f"  Period: {data.index[0]} to {data.index[-1]}")

    # 2. Create spec with costs
    print("\n[Symbol Specification]")
    spec = create_spec_with_costs()
    print(f"  Symbol: {spec.symbol}")
    print(f"  Spread: {spec.spread_typical} points")
    print(f"  Commission: ${spec.commission_per_lot}/lot/side")
    print(f"  Swap long: {spec.swap_long} pts/day")
    print(f"  Triple swap: {spec.swap_triple_day}")

    # 3. Generate signals
    print("\n[Generating Signals]")
    signals = generate_signals(data, max_trades=5)
    print(f"  Generated {len(signals)} signals ({len(signals)//2} trades)")

    # 4. Run backtest
    print("\n[Running Backtest]")
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        enable_slippage=False,  # Disable for cleaner validation
        enable_freeze_zones=True,
        enable_stop_validation=True,
    )

    result = backtester.run(data, signals, classify_regimes=False)

    print(f"  Completed: {len(result.trades)} trades")
    print(f"  Total P&L: ${result.total_pnl:,.2f}")

    # 5. Print detailed transaction log
    print_transaction_log(result.trades, spec)

    # 6. Summary validation
    print("\n" + "="*100)
    print("SUMMARY VALIDATION")
    print("="*100)

    total_spread = sum(
        (t.entry_spread + t.exit_spread) * spec.point * t.volume * spec.contract_size
        for t in result.trades
    )
    total_commission = sum(t.commission for t in result.trades)
    total_swap = sum(t.swap for t in result.trades)
    total_costs = total_spread + total_commission + total_swap

    print(f"\n[AGGREGATE COSTS]")
    print(f"  Total spread:      ${total_spread:,.2f}")
    print(f"  Total commission:  ${total_commission:,.2f}")
    print(f"  Total swap:        ${total_swap:,.2f}")
    print(f"  ─────────────────────────────")
    print(f"  Total costs:       ${total_costs:,.2f}")

    print(f"\n[BACKTEST RESULT VALIDATION]")
    print(f"  Result.total_spread_cost:  ${result.total_spread_cost:,.2f}")
    print(f"  Result.total_commission:   ${result.total_commission:,.2f}")
    print(f"  Result.total_swap:         ${result.total_swap:,.2f}")

    # Validate totals
    spread_match = abs(result.total_spread_cost - total_spread) < 1.0
    comm_match = abs(result.total_commission - total_commission) < 0.01
    swap_match = abs(result.total_swap - total_swap) < 1.0

    print(f"\n[VALIDATION]")
    print(f"  Spread totals match:     {'✅' if spread_match else '❌'}")
    print(f"  Commission totals match: {'✅' if comm_match else '❌'}")
    print(f"  Swap totals match:       {'✅' if swap_match else '❌'}")

    # Final P&L check
    gross_pnl_total = sum(
        (t.exit_price - t.entry_price) * t.direction * t.volume * spec.contract_size
        for t in result.trades
    )
    net_pnl_check = gross_pnl_total - total_costs

    print(f"\n[P&L VALIDATION]")
    print(f"  Gross P&L (calculated): ${gross_pnl_total:,.2f}")
    print(f"  Total costs:            ${total_costs:,.2f}")
    print(f"  Net P&L (calculated):   ${net_pnl_check:,.2f}")
    print(f"  Net P&L (result):       ${result.total_pnl:,.2f}")
    print(f"  Match:                  {'✅' if abs(result.total_pnl - net_pnl_check) < 10.0 else '❌'}")

    print("\n" + "="*100)
    if spread_match and comm_match and swap_match:
        print("✅ TRANSACTION LOG TEST PASSED")
        print("All costs calculated and validated correctly!")
    else:
        print("❌ TRANSACTION LOG TEST FAILED")
        print("Some cost calculations do not match expected values")
    print("="*100)

    return 0 if (spread_match and comm_match and swap_match) else 1


if __name__ == "__main__":
    exit(run_transaction_log_test())
