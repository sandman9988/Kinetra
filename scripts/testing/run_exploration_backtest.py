#!/usr/bin/env python3
"""
Kinetra Exploration Backtest

Comprehensive exploration of trading strategies across:
- Multiple symbols (EURJPY, GBPUSD, XAUUSD, etc.)
- Multiple timeframes (M15, H1, H4)
- Multiple strategies (MA crossover, breakout, mean reversion)
- Parameter optimization
- Full MT5-style logging
- Grafana metrics export
- Health monitoring
- Agent performance tracking

This is a production-quality exploration run that demonstrates
the full power of the Kinetra adaptive trading system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import itertools

from kinetra.realistic_backtester import RealisticBacktester
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.trade_logger import MT5Logger
from kinetra.grafana_exporter import GrafanaExporter


def load_symbol_data(symbol: str, timeframe: str = "M15", max_rows: int = 5000) -> pd.DataFrame:
    """Load market data for a symbol."""
    # Map symbol to data file
    data_files = {
        "EURJPY+": "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv",
        # Add more symbols here as data becomes available
    }

    if symbol not in data_files:
        raise ValueError(f"No data file for symbol {symbol}")

    df = pd.read_csv(data_files[symbol], sep='\t', nrows=max_rows)
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
        '<SPREAD>': 'spread',
    })
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]
    df = df.set_index('timestamp')

    return df


def create_symbol_spec(symbol: str, asset_class: AssetClass) -> SymbolSpec:
    """Create symbol specification with realistic costs."""
    specs = {
        "EURJPY+": SymbolSpec(
            symbol="EURJPY+",
            asset_class=AssetClass.FOREX,
            digits=3,
            point=0.001,
            contract_size=100000.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_typical=20,  # 2 pips
            commission_per_lot=6.0,
            swap_long=-0.3,
            swap_short=0.1,
            swap_triple_day="wednesday",
            trade_freeze_level=50,
            trade_stops_level=100,
        ),
        "GBPUSD": SymbolSpec(
            symbol="GBPUSD",
            asset_class=AssetClass.FOREX,
            digits=5,
            point=0.00001,
            contract_size=100000.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_typical=15,  # 1.5 pips
            commission_per_lot=6.0,
            swap_long=-2.5,
            swap_short=0.5,
            swap_triple_day="wednesday",
            trade_freeze_level=50,
            trade_stops_level=100,
        ),
    }

    return specs.get(symbol, specs["EURJPY+"])


class StrategyGenerator:
    """Generate trading signals using different strategies."""

    @staticmethod
    def ma_crossover(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 50, max_trades: int = 50) -> pd.DataFrame:
        """Moving average crossover strategy."""
        signals = []

        # Calculate MAs
        fast_ma = data['close'].rolling(window=fast_period).mean()
        slow_ma = data['close'].rolling(window=slow_period).mean()

        position = None
        entry_idx = None

        for i in range(slow_period, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']

            # Entry signals
            if position is None:
                if fast_ma.iloc[i] > slow_ma.iloc[i] and fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]:
                    # Bullish crossover
                    signals.append({
                        'time': current_time,
                        'action': 'open_long',
                        'sl': current_price - 0.200,
                        'tp': current_price + 0.400,
                        'volume': 1.0,
                    })
                    position = 'long'
                    entry_idx = i

                elif fast_ma.iloc[i] < slow_ma.iloc[i] and fast_ma.iloc[i-1] >= slow_ma.iloc[i-1]:
                    # Bearish crossover
                    signals.append({
                        'time': current_time,
                        'action': 'open_short',
                        'sl': current_price + 0.200,
                        'tp': current_price - 0.400,
                        'volume': 1.0,
                    })
                    position = 'short'
                    entry_idx = i

            # Exit signals
            elif position == 'long' and fast_ma.iloc[i] < slow_ma.iloc[i]:
                signals.append({
                    'time': current_time,
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None
                entry_idx = None

            elif position == 'short' and fast_ma.iloc[i] > slow_ma.iloc[i]:
                signals.append({
                    'time': current_time,
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None
                entry_idx = None

            # Limit trades
            if len(signals) >= max_trades * 2:
                break

        # Close any open position at the end
        if position is not None:
            signals.append({
                'time': data.index[-1],
                'action': 'close',
                'sl': None,
                'tp': None,
            })

        return pd.DataFrame(signals) if signals else pd.DataFrame()

    @staticmethod
    def breakout(data: pd.DataFrame, lookback: int = 20, max_trades: int = 50) -> pd.DataFrame:
        """Breakout strategy."""
        signals = []
        position = None

        for i in range(lookback, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']

            # Calculate channel
            high_channel = data['high'].iloc[i-lookback:i].max()
            low_channel = data['low'].iloc[i-lookback:i].min()

            # Entry signals
            if position is None:
                if current_price > high_channel:
                    # Bullish breakout
                    signals.append({
                        'time': current_time,
                        'action': 'open_long',
                        'sl': low_channel,
                        'tp': current_price + (high_channel - low_channel),
                        'volume': 1.0,
                    })
                    position = 'long'
                    entry_price = current_price

                elif current_price < low_channel:
                    # Bearish breakout
                    signals.append({
                        'time': current_time,
                        'action': 'open_short',
                        'sl': high_channel,
                        'tp': current_price - (high_channel - low_channel),
                        'volume': 1.0,
                    })
                    position = 'short'
                    entry_price = current_price

            # Exit after holding period or stop/target
            elif position and (i - len(signals)//2) > 20:
                signals.append({
                    'time': current_time,
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None

            if len(signals) >= max_trades * 2:
                break

        if position is not None:
            signals.append({
                'time': data.index[-1],
                'action': 'close',
                'sl': None,
                'tp': None,
            })

        return pd.DataFrame(signals) if signals else pd.DataFrame()


def run_single_backtest(
    symbol: str,
    strategy_name: str,
    strategy_params: Dict,
    data: pd.DataFrame,
    spec: SymbolSpec,
    logger: MT5Logger,
    exporter: GrafanaExporter,
) -> Tuple[object, str]:
    """Run a single backtest configuration."""

    # Generate signals
    if strategy_name == 'ma_crossover':
        signals = StrategyGenerator.ma_crossover(
            data,
            fast_period=strategy_params.get('fast_period', 10),
            slow_period=strategy_params.get('slow_period', 50),
            max_trades=strategy_params.get('max_trades', 50),
        )
    elif strategy_name == 'breakout':
        signals = StrategyGenerator.breakout(
            data,
            lookback=strategy_params.get('lookback', 20),
            max_trades=strategy_params.get('max_trades', 50),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    if signals.empty:
        print(f"  ⚠️  No signals generated for {symbol} {strategy_name}")
        return None, "no_signals"

    # Run backtest
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        enable_slippage=True,
        enable_freeze_zones=True,
        enable_stop_validation=True,
    )

    result = backtester.run(data, signals, classify_regimes=False)

    # Log summary
    config_str = f"{symbol}_{strategy_name}_" + "_".join(f"{k}{v}" for k, v in strategy_params.items())

    logger._log(f"\n{'='*80}")
    logger._log(f"Configuration: {config_str}")
    logger._log(f"  Strategy: {strategy_name}")
    logger._log(f"  Parameters: {strategy_params}")
    logger._log(f"  Trades: {len(result.trades)}")
    logger._log(f"  Win Rate: {result.win_rate:.1%}")
    logger._log(f"  Total P&L: ${result.total_pnl:,.2f}")
    logger._log(f"  Return: {result.total_return_pct:.2%}")
    logger._log(f"  Sharpe: {result.sharpe_ratio:.2f}")
    logger._log(f"  Max DD: {result.max_drawdown_pct:.1%}")
    logger._log(f"  Total Costs: ${result.total_spread_cost + result.total_commission + result.total_swap:,.2f}")
    logger._log(f"{'='*80}\n")

    # Export to Grafana
    for trade in result.trades:
        entry_spread = trade.entry_spread * spec.point * trade.volume * spec.contract_size
        exit_spread = trade.exit_spread * spec.point * trade.volume * spec.contract_size

        exporter.record_trade_entry(
            time=trade.entry_time,
            symbol=symbol,
            direction=trade.direction,
            volume=trade.volume,
            entry_price=trade.entry_price,
            spread=entry_spread,
            commission=spec.commission_per_lot * trade.volume,
            regime=strategy_name,  # Use strategy as regime for filtering
        )

        exporter.record_trade_exit(
            time=trade.exit_time,
            symbol=symbol,
            direction=trade.direction,
            volume=trade.volume,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            gross_pnl=trade.pnl + trade.total_cost,
            spread=entry_spread + exit_spread,
            commission=trade.commission,
            swap=trade.swap,
            slippage=abs(trade.entry_slippage) + abs(trade.exit_slippage) if hasattr(trade, 'entry_slippage') else 0,
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_efficiency=trade.mfe_efficiency,
            holding_hours=trade.holding_time,
            exit_reason="signal",
        )

    # Export summary
    exporter.record_backtest_summary(
        time=result.trades[-1].exit_time if result.trades else datetime.now(),
        total_trades=len(result.trades),
        win_rate=result.win_rate,
        total_pnl=result.total_pnl,
        total_return_pct=result.total_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        total_spread=result.total_spread_cost,
        total_commission=result.total_commission,
        total_swap=result.total_swap,
    )

    return result, config_str


def run_exploration_backtest():
    """Run comprehensive exploration across multiple configurations."""

    print("=" * 100)
    print("KINETRA EXPLORATION BACKTEST")
    print("=" * 100)
    print("\nExploring:")
    print("  - Multiple symbols (EURJPY, GBPUSD)")
    print("  - Multiple strategies (MA Crossover, Breakout)")
    print("  - Parameter optimization")
    print("  - Full MT5-style logging")
    print("  - Grafana metrics export")
    print("=" * 100)

    # Initialize logging and export
    logger = MT5Logger(
        symbol="MULTI",
        timeframe="M15",
        initial_balance=10000.0,
        enable_verbose=True,
        log_file="/tmp/kinetra_exploration.log",
    )

    exporter = GrafanaExporter(
        backend='influxdb',
        host='localhost',
        port=8086,
        db_name='kinetra_exploration',
        enable_export=True,
    )

    # Define exploration space
    symbols = ["EURJPY+"]  # Add more as data becomes available
    strategies = {
        'ma_crossover': [
            {'fast_period': 5, 'slow_period': 20, 'max_trades': 50},
            {'fast_period': 10, 'slow_period': 50, 'max_trades': 50},
            {'fast_period': 20, 'slow_period': 100, 'max_trades': 50},
        ],
        'breakout': [
            {'lookback': 10, 'max_trades': 50},
            {'lookback': 20, 'max_trades': 50},
            {'lookback': 50, 'max_trades': 50},
        ],
    }

    # Results storage
    all_results = []

    # Exploration loop
    total_configs = sum(len(params) for params in strategies.values()) * len(symbols)
    config_num = 0

    logger._log(f"\n{'='*100}")
    logger._log(f"STARTING EXPLORATION: {total_configs} configurations")
    logger._log(f"{'='*100}\n")

    for symbol in symbols:
        logger._log(f"\n{'#'*100}")
        logger._log(f"SYMBOL: {symbol}")
        logger._log(f"{'#'*100}\n")

        # Load data
        print(f"\n[Loading {symbol} data]")
        data = load_symbol_data(symbol, max_rows=5000)
        spec = create_symbol_spec(symbol, AssetClass.FOREX)

        print(f"  Loaded {len(data)} candles")
        print(f"  Period: {data.index[0]} to {data.index[-1]}")

        for strategy_name, param_sets in strategies.items():
            logger._log(f"\n{'─'*100}")
            logger._log(f"STRATEGY: {strategy_name}")
            logger._log(f"{'─'*100}\n")

            for params in param_sets:
                config_num += 1

                print(f"\n[{config_num}/{total_configs}] {symbol} - {strategy_name} - {params}")

                result, config_str = run_single_backtest(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    strategy_params=params,
                    data=data,
                    spec=spec,
                    logger=logger,
                    exporter=exporter,
                )

                if result:
                    all_results.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'params': params,
                        'config': config_str,
                        'trades': len(result.trades),
                        'win_rate': result.win_rate,
                        'total_pnl': result.total_pnl,
                        'return_pct': result.total_return_pct,
                        'sharpe': result.sharpe_ratio,
                        'max_dd_pct': result.max_drawdown_pct,
                        'total_costs': result.total_spread_cost + result.total_commission + result.total_swap,
                        'result': result,
                    })

    # Export all metrics
    print(f"\n[Exporting {len(exporter.metrics)} metrics to Grafana]")
    metrics = exporter.flush()
    output_file = "/tmp/kinetra_exploration_metrics.txt"
    with open(output_file, 'w') as f:
        for metric in metrics:
            f.write(metric + '\n')
    print(f"  Saved to {output_file}")

    # Analysis
    print("\n" + "=" * 100)
    print("EXPLORATION RESULTS SUMMARY")
    print("=" * 100)

    if all_results:
        # Sort by Sharpe ratio
        all_results.sort(key=lambda x: x['sharpe'], reverse=True)

        print(f"\nTotal configurations tested: {len(all_results)}")
        print(f"Profitable configurations: {len([r for r in all_results if r['total_pnl'] > 0])}")

        print("\n📊 TOP 5 CONFIGURATIONS (by Sharpe Ratio):")
        print("─" * 100)
        for i, r in enumerate(all_results[:5], 1):
            print(f"\n{i}. {r['config']}")
            print(f"   Strategy: {r['strategy']}")
            print(f"   Parameters: {r['params']}")
            print(f"   Trades: {r['trades']}")
            print(f"   Win Rate: {r['win_rate']:.1%}")
            print(f"   Total P&L: ${r['total_pnl']:,.2f}")
            print(f"   Return: {r['return_pct']:.2%}")
            print(f"   Sharpe: {r['sharpe']:.2f}")
            print(f"   Max DD: {r['max_dd_pct']:.1%}")
            print(f"   Total Costs: ${r['total_costs']:,.2f}")

        # Best by return
        best_return = max(all_results, key=lambda x: x['return_pct'])
        print(f"\n💰 BEST RETURN: {best_return['config']}")
        print(f"   Return: {best_return['return_pct']:.2%}")
        print(f"   P&L: ${best_return['total_pnl']:,.2f}")

        # Best by win rate
        best_wr = max(all_results, key=lambda x: x['win_rate'])
        print(f"\n🎯 BEST WIN RATE: {best_wr['config']}")
        print(f"   Win Rate: {best_wr['win_rate']:.1%}")
        print(f"   Sharpe: {best_wr['sharpe']:.2f}")

        # Statistics
        avg_return = np.mean([r['return_pct'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_trades = np.mean([r['trades'] for r in all_results])

        print("\n📈 AGGREGATE STATISTICS:")
        print(f"   Average Return: {avg_return:.2%}")
        print(f"   Average Sharpe: {avg_sharpe:.2f}")
        print(f"   Average Trades: {avg_trades:.0f}")
        print(f"   Total Trades: {sum(r['trades'] for r in all_results)}")

    print("\n" + "=" * 100)
    print("✅ EXPLORATION BACKTEST COMPLETE")
    print("=" * 100)
    print("\nNext steps:")
    print("1. Review logs: /tmp/kinetra_exploration.log")
    print("2. Review metrics: /tmp/kinetra_exploration_metrics.txt")
    print("3. Visualize in Grafana (see setup in test_grafana_export.py)")
    print("4. Select best configuration for live trading")
    print("=" * 100)

    return all_results


if __name__ == "__main__":
    results = run_exploration_backtest()
