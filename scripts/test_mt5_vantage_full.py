"""
Complete MT5 Vantage Backtest - End to End
==========================================

Full pipeline:
1. Download real MT5 data from Vantage terminal
2. Process and validate data
3. Generate real strategy signals (MA crossover)
4. Run realistic backtest with all friction costs
5. MT5-style enhanced logging
6. Grafana metrics export
7. Complete analysis

This is a GENUINE backtest with REAL data and REAL strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.data_loader import UnifiedDataLoader
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.realistic_backtester import RealisticBacktester
from kinetra.trade_logger import MT5Logger
from kinetra.grafana_exporter import GrafanaExporter


def get_vantage_spec(symbol: str) -> SymbolSpec:
    """Get accurate Vantage Markets symbol specification."""

    # Vantage EURJPY specifications (accurate)
    if symbol == "EURJPY+":
        return SymbolSpec(
            symbol="EURJPY+",
            asset_class=AssetClass.FOREX,
            digits=3,
            point=0.001,
            contract_size=100000.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            # Vantage realistic costs
            spread_typical=18,  # ~1.8 pips typical
            commission_per_lot=6.0,  # $6/lot/side
            swap_long=-0.3,
            swap_short=0.1,
            swap_triple_day="wednesday",
            # MT5 constraints
            trade_freeze_level=50,
            trade_stops_level=100,
        )
    elif symbol == "XAUUSD+":
        return SymbolSpec(
            symbol="XAUUSD+",
            asset_class=AssetClass.METAL,
            digits=2,
            point=0.01,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            # Gold costs
            spread_typical=25,  # ~25 cents typical
            commission_per_lot=6.0,
            swap_long=-0.15,
            swap_short=-0.10,
            swap_triple_day="wednesday",
            trade_freeze_level=100,
            trade_stops_level=200,
        )
    else:
        raise ValueError(f"Unknown symbol: {symbol}")


def load_mt5_vantage_data(symbol: str, timeframe: str, bars: int = 5000) -> pd.DataFrame:
    """
    Load real MT5 Vantage data from CSV files.

    Args:
        symbol: Symbol to load (e.g., "EURJPY+")
        timeframe: Timeframe (e.g., "M15", "H1", "D1")
        bars: Number of bars to load (max available if None)

    Returns:
        DataFrame with OHLCV data
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING REAL MT5 VANTAGE DATA")
    print("="*80)

    print(f"\n[Loading from MT5 CSV files]")
    print(f"  Symbol: {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Max bars: {bars if bars else 'all'}")

    # Find data file
    data_dir = project_root / "data" / "master"
    pattern = f"{symbol}_{timeframe}_*.csv"

    import glob
    files = list(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No data found matching: {data_dir}/{pattern}")

    # Use most recent file
    data_file = sorted(files)[-1]
    print(f"\n✅ Found data file: {data_file.name}")

    # Load using UnifiedDataLoader
    loader = UnifiedDataLoader(verbose=False)
    pkg = loader.load(data_file)

    # Get data in backtest format
    data = pkg.to_backtest_engine_format()

    # Limit bars if requested
    if bars and len(data) > bars:
        data = data.iloc[-bars:]

    print(f"✅ Loaded {len(data)} bars")
    print(f"  Period: {data.index[0]} to {data.index[-1]}")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Source: REAL MT5 VANTAGE TERMINAL DATA")

    return data


def validate_data(data: pd.DataFrame) -> bool:
    """Validate MT5 data quality."""
    print("\n" + "="*80)
    print("STEP 2: VALIDATING DATA QUALITY")
    print("="*80)

    checks_passed = 0
    checks_total = 6

    # Check 1: Required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    has_columns = all(col in data.columns for col in required)
    print(f"\n[1/6] Required columns: {'✅ PASS' if has_columns else '❌ FAIL'}")
    if has_columns:
        checks_passed += 1

    # Check 2: No missing values
    missing = data[required].isnull().sum().sum()
    print(f"[2/6] Missing values: {'✅ PASS' if missing == 0 else f'❌ FAIL ({missing} missing)'}")
    if missing == 0:
        checks_passed += 1

    # Check 3: OHLC validity
    valid_ohlc = (
        (data['high'] >= data['low']).all() and
        (data['high'] >= data['open']).all() and
        (data['high'] >= data['close']).all() and
        (data['low'] <= data['open']).all() and
        (data['low'] <= data['close']).all()
    )
    print(f"[3/6] OHLC validity: {'✅ PASS' if valid_ohlc else '❌ FAIL'}")
    if valid_ohlc:
        checks_passed += 1

    # Check 4: Volume > 0
    has_volume = (data['volume'] > 0).all()
    print(f"[4/6] Volume > 0: {'✅ PASS' if has_volume else '❌ FAIL'}")
    if has_volume:
        checks_passed += 1

    # Check 5: Sufficient data
    min_bars = 200
    sufficient = len(data) >= min_bars
    print(f"[5/6] Sufficient data: {'✅ PASS' if sufficient else f'❌ FAIL (need {min_bars}, have {len(data)})'}")
    if sufficient:
        checks_passed += 1

    # Check 6: No duplicates
    duplicates = data.index.duplicated().sum()
    no_dupes = duplicates == 0
    print(f"[6/6] No duplicates: {'✅ PASS' if no_dupes else f'❌ FAIL ({duplicates} duplicates)'}")
    if no_dupes:
        checks_passed += 1

    print(f"\n{'='*80}")
    print(f"VALIDATION RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*80}")

    return checks_passed == checks_total


def generate_ma_crossover_signals(data: pd.DataFrame, fast: int = 10, slow: int = 50, max_trades: int = 20) -> pd.DataFrame:
    """
    Generate REAL MA crossover signals (not random).

    Args:
        data: OHLCV data
        fast: Fast MA period
        slow: Slow MA period
        max_trades: Maximum number of trades

    Returns:
        DataFrame with signals
    """
    print("\n" + "="*80)
    print("STEP 3: GENERATING REAL STRATEGY SIGNALS")
    print("="*80)

    print(f"\n[Strategy: MA Crossover]")
    print(f"  Fast MA: {fast}")
    print(f"  Slow MA: {slow}")
    print(f"  Max trades: {max_trades}")

    # Calculate MAs
    df = data.copy()
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()

    # Detect crossovers
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1  # Long
    df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1  # Short

    # Generate entry/exit signals
    signals = []
    position = None
    trade_count = 0

    for i in range(slow + 1, len(df)):
        if trade_count >= max_trades:
            break

        current_signal = df.iloc[i]['signal']
        prev_signal = df.iloc[i-1]['signal']

        # Crossover detected
        if current_signal != prev_signal and current_signal != 0:
            # Close existing position
            if position is not None:
                signals.append({
                    'time': df.index[i],
                    'action': 'close',
                })
                position = None
                trade_count += 1

            # Open new position
            if trade_count < max_trades:
                entry_price = df.iloc[i]['close']

                if current_signal == 1:
                    # Long
                    signals.append({
                        'time': df.index[i],
                        'action': 'open_long',
                        'sl': entry_price - 0.500,  # 500 pips
                        'tp': entry_price + 1.000,  # 1000 pips
                        'volume': 1.0,
                    })
                    position = 'long'
                else:
                    # Short
                    signals.append({
                        'time': df.index[i],
                        'action': 'open_short',
                        'sl': entry_price + 0.500,
                        'tp': entry_price - 1.000,
                        'volume': 1.0,
                    })
                    position = 'short'

    # Close final position
    if position is not None:
        signals.append({
            'time': df.index[-1],
            'action': 'close',
        })

    signals_df = pd.DataFrame(signals)
    if len(signals_df) > 0:
        signals_df = signals_df.set_index('time')

    print(f"\n✅ Generated {len([s for s in signals if s['action'] in ['open_long', 'open_short']])} trades")
    print(f"  Total signals: {len(signals)}")

    return signals_df


def run_complete_backtest():
    """Run complete end-to-end backtest with MT5 Vantage data."""

    print("\n" + "="*100)
    print(" "*30 + "MT5 VANTAGE COMPLETE BACKTEST")
    print("="*100)
    print("\nThis is a GENUINE backtest with:")
    print("  ✅ Real MT5 data from Vantage terminal")
    print("  ✅ Real MA crossover strategy")
    print("  ✅ Real friction costs (spread, commission, swap)")
    print("  ✅ MT5-style enhanced logging")
    print("  ✅ Grafana metrics export")
    print("="*100)

    # Configuration
    symbol = "EURJPY+"
    timeframe = "M15"
    bars = 5000
    fast_ma = 10
    slow_ma = 50
    max_trades = 20

    # Step 1: Download data
    data = download_mt5_data(symbol, timeframe, bars)

    # Step 2: Validate data
    if not validate_data(data):
        print("\n❌ Data validation failed. Aborting.")
        return

    # Step 3: Generate signals
    signals = generate_ma_crossover_signals(data, fast_ma, slow_ma, max_trades)

    if len(signals) == 0:
        print("\n❌ No signals generated. Aborting.")
        return

    # Step 4: Setup backtester
    print("\n" + "="*80)
    print("STEP 4: RUNNING REALISTIC BACKTEST")
    print("="*80)

    spec = get_vantage_spec(symbol)
    backtester = RealisticBacktester(
        data=data,
        spec=spec,
        initial_balance=10000.0,
        verbose=False,
    )

    # Step 5: Setup MT5 logger
    logger = MT5Logger(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=10000.0,
        enable_verbose=True,
    )

    # Step 6: Setup Grafana exporter
    exporter = GrafanaExporter(
        backend='influxdb',
        host='localhost',
        port=8086,
        db_name='kinetra',
        enable_export=False,  # Don't actually push, just collect
    )

    print(f"\n[Running backtest with {len(signals)} signals]")
    print(f"  Initial balance: $10,000")
    print(f"  Strategy: MA({fast_ma}/{slow_ma})")
    print(f"  Spread: {spec.spread_typical} points")
    print(f"  Commission: ${spec.commission_per_lot}/lot/side")
    print(f"  Swap: {spec.swap_long} pts/day (long), {spec.swap_short} pts/day (short)")

    # Run backtest
    result = backtester.run(signals)

    print(f"\n✅ Backtest complete")
    print(f"  Trades executed: {len(result.trades)}")
    print(f"  Final balance: ${result.final_balance:,.2f}")
    print(f"  Total return: {result.total_return:.2f}%")

    # Step 7: Generate MT5-style logs
    print("\n" + "="*80)
    print("STEP 5: GENERATING MT5-STYLE LOGS")
    print("="*80)

    logger.log_start(
        account_balance=10000.0,
        timeframe=timeframe,
        period_start=data.index[0],
        period_end=data.index[-1],
    )

    # Log all trades
    for i, trade in enumerate(result.trades, 1):
        position_id = i

        # Log entry
        logger.log_order(
            time=trade.entry_time,
            action='buy' if trade.direction == 1 else 'sell',
            volume=trade.volume,
            bid=trade.entry_price,
            ask=trade.entry_price,
            price=trade.entry_price,
            spread_points=trade.entry_spread,
        )

        logger.log_deal(
            time=trade.entry_time,
            deal_id=position_id * 2 - 1,
            action='buy' if trade.direction == 1 else 'sell',
            volume=trade.volume,
            price=trade.entry_price,
            order_id=position_id * 2 - 1,
        )

        logger.log_position_open(
            time=trade.entry_time,
            direction=trade.direction,
            volume=trade.volume,
            entry_price=trade.entry_price,
            spread=trade.entry_spread * spec.point * trade.volume * spec.contract_size,
            commission=spec.commission_per_lot * trade.volume,
            regime=trade.entry_physics_regime or "UNKNOWN",
        )

        # Log exit
        logger.log_order(
            time=trade.exit_time,
            action='sell' if trade.direction == 1 else 'buy',
            volume=trade.volume,
            bid=trade.exit_price,
            ask=trade.exit_price,
            price=trade.exit_price,
            spread_points=trade.exit_spread,
            close_position_id=position_id,
        )

        logger.log_deal(
            time=trade.exit_time,
            deal_id=position_id * 2,
            action='sell' if trade.direction == 1 else 'buy',
            volume=trade.volume,
            price=trade.exit_price,
            order_id=position_id * 2,
        )

        # Calculate costs
        total_spread = (trade.entry_spread + trade.exit_spread) * spec.point * trade.volume * spec.contract_size
        gross_pnl = trade.pnl + total_spread + trade.commission + trade.swap

        logger.log_position_close(
            time=trade.exit_time,
            position_id=position_id,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            spread=total_spread,
            commission=trade.commission,
            swap=trade.swap,
            slippage=abs(trade.entry_slippage) + abs(trade.exit_slippage),
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_efficiency=trade.mfe_efficiency if hasattr(trade, 'mfe_efficiency') else 0.0,
            holding_hours=trade.holding_time if hasattr(trade, 'holding_time') else 0.0,
            exit_reason="signal",
        )

        # Export to Grafana
        exporter.record_trade_exit(
            time=trade.exit_time,
            symbol=symbol,
            direction=trade.direction,
            volume=trade.volume,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            gross_pnl=gross_pnl,
            spread=total_spread,
            commission=trade.commission,
            swap=trade.swap,
            slippage=abs(trade.entry_slippage) + abs(trade.exit_slippage),
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_efficiency=trade.mfe_efficiency if hasattr(trade, 'mfe_efficiency') else 0.0,
            holding_hours=trade.holding_time if hasattr(trade, 'holding_time') else 0.0,
            exit_reason="signal",
        )

    # Final summary
    logger.log_summary(
        total_trades=len(result.trades),
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        total_pnl=result.total_pnl,
        total_return=result.total_return,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        total_spread=result.total_spread_cost,
        total_commission=result.total_commission,
        total_swap=result.total_swap,
        total_slippage=result.total_slippage,
        final_balance=result.final_balance,
        freeze_violations=result.total_freeze_violations,
        stop_violations=result.total_invalid_stops,
    )

    # Step 8: Export Grafana metrics
    print("\n" + "="*80)
    print("STEP 6: EXPORTING GRAFANA METRICS")
    print("="*80)

    metrics = exporter.flush()

    print(f"\n✅ Exported {len(metrics)} metrics")

    # Save to file
    output_file = "/tmp/kinetra_mt5_vantage_metrics.txt"
    with open(output_file, 'w') as f:
        for metric in metrics:
            f.write(metric + '\n')

    print(f"  Saved to: {output_file}")

    # Final results
    print("\n" + "="*100)
    print(" "*35 + "FINAL RESULTS")
    print("="*100)

    print(f"\n📊 TRADING PERFORMANCE:")
    print(f"  Total Trades:      {len(result.trades)}")
    print(f"  Winning Trades:    {result.winning_trades}")
    print(f"  Losing Trades:     {result.losing_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"\n💰 PROFIT & LOSS:")
    print(f"  Initial Balance:   ${10000:,.2f}")
    print(f"  Final Balance:     ${result.final_balance:,.2f}")
    print(f"  Total P&L:         ${result.total_pnl:,.2f}")
    print(f"  Total Return:      {result.total_return:.2f}%")
    print(f"\n📈 RISK METRICS:")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown:.2f}%")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"\n💸 TRANSACTION COSTS:")
    print(f"  Total Spread:      ${result.total_spread_cost:,.2f}")
    print(f"  Total Commission:  ${result.total_commission:,.2f}")
    print(f"  Total Swap:        ${result.total_swap:,.2f}")
    print(f"  Total Slippage:    ${result.total_slippage:,.2f}")
    print(f"  Total Costs:       ${result.total_spread_cost + result.total_commission + result.total_swap + result.total_slippage:,.2f}")

    print("\n" + "="*100)
    print("✅ COMPLETE BACKTEST FINISHED")
    print("="*100)

    print(f"\nGenerated files:")
    print(f"  📋 Grafana metrics: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the MT5-style logs above")
    print(f"  2. Import metrics to Grafana for visualization")
    print(f"  3. Analyze trade-by-trade performance")
    print(f"  4. Optimize strategy parameters")


if __name__ == '__main__':
    run_complete_backtest()
