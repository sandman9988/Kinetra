#!/usr/bin/env python3
"""
Test multi-instrument backtest functionality.
Uses two instruments from available data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kinetra.backtest_engine import BacktestEngine
from kinetra.symbol_spec import SymbolSpec


def main():
    # Find available data files
    data_dir = Path("data/master")
    available_files = list(data_dir.glob("*_H1_*.csv"))
    print(f"Available H1 files: {len(available_files)}")
    for f in available_files[:5]:
        print(f"  {f.name}")

    # Pick two instruments
    symbols_to_test = []
    for f in available_files:
        symbol = f.name.split("_")[0]
        if symbol not in [s[0] for s in symbols_to_test]:
            symbols_to_test.append((symbol, f))
        if len(symbols_to_test) >= 2:
            break

    print(f"\nTesting with: {[s[0] for s in symbols_to_test]}")

    # Load data (MT5 format with <DATE>, <TIME> columns)
    data_dict = {}
    for symbol, filepath in symbols_to_test:
        df = pd.read_csv(filepath, sep="\t")
        # Combine DATE and TIME into datetime index
        df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
        df = df.set_index("time")
        # Rename columns to lowercase
        df = df.rename(
            columns={
                "<OPEN>": "open",
                "<HIGH>": "high",
                "<LOW>": "low",
                "<CLOSE>": "close",
                "<TICKVOL>": "tick_volume",
                "<VOL>": "volume",
                "<SPREAD>": "spread",
            }
        )
        df = df.drop(columns=["<DATE>", "<TIME>"], errors="ignore")
        # Take last 1000 bars for quick test
        df = df.tail(1000)
        data_dict[symbol] = df
        print(f"  {symbol}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Create symbol specs using the correct SymbolSpec class (from symbol_spec.py)
    symbol_specs = {}
    for symbol, _ in symbols_to_test:
        if "BTC" in symbol or "ETH" in symbol:
            symbol_specs[symbol] = SymbolSpec(
                symbol=symbol,
                contract_size=1.0,  # 1 BTC per lot
                tick_size=0.01,
                tick_value=0.01,
                digits=2,
                margin_initial=0.10,  # 10% margin (10:1 leverage)
                spread_points=50.0,
            )
        elif "XAU" in symbol or "GOLD" in symbol:
            symbol_specs[symbol] = SymbolSpec(
                symbol=symbol,
                contract_size=100.0,  # 100 oz per lot
                tick_size=0.01,
                tick_value=1.0,
                digits=2,
                margin_initial=0.01,  # 1% margin (100:1 leverage)
                spread_points=30.0,
            )
        else:
            # Generic spec for indices
            symbol_specs[symbol] = SymbolSpec(
                symbol=symbol,
                contract_size=1.0,  # 1 contract
                tick_size=1.0,
                tick_value=1.0,
                digits=0,
                margin_initial=0.05,  # 5% margin (20:1 leverage)
                spread_points=5.0,
            )
        print(
            f"  Spec for {symbol}: contract_size={symbol_specs[symbol].contract_size}, digits={symbol_specs[symbol].digits}"
        )

    # Track signal counts for debugging
    signal_counts = {"long": 0, "short": 0, "flat": 0}
    physics_cols_printed = [False]

    # Simple signal function: price vs rolling mean
    def signal_func(symbol: str, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
        """Simple signal: enter based on physics state at bar_idx."""
        if bar_idx < 20:
            return 0  # Need warmup

        # Physics is a DataFrame - extract row at bar_idx
        if physics is None or bar_idx >= len(physics):
            return 0

        phys_row = physics.iloc[bar_idx]

        # Debug: print physics columns once
        if not physics_cols_printed[0]:
            print(f"  Physics columns: {list(physics.columns)}")
            physics_cols_printed[0] = True

        # Use physics data if available - use energy_pct and regime
        energy_pct = phys_row.get("energy_pct", 0.5)
        regime = phys_row.get("regime", "unknown")

        # Simple signal: high energy + favorable regime
        # Use row close vs previous close for direction
        if bar_idx > 0:
            prev_close = (
                physics.iloc[bar_idx - 1].get("close", row["close"])
                if "close" in physics.columns
                else 0
            )
            curr_close = row["close"]
            price_up = curr_close > prev_close if prev_close > 0 else False
        else:
            price_up = True

        # Signal based on energy and price direction
        if energy_pct > 0.6:
            if price_up:
                signal_counts["long"] += 1
                return 1  # Long when energy high and price up
            else:
                signal_counts["short"] += 1
                return -1  # Short when energy high and price down
        signal_counts["flat"] += 1
        return 0

    # Create engine and run multi-instrument backtest
    engine = BacktestEngine(
        initial_capital=10000.0,
        max_positions=2,  # Allow 2 total positions
        max_positions_per_symbol=1,  # 1 per symbol
        portfolio_stop_out_level=0.5,
    )

    print("\nRunning multi-instrument backtest...")
    try:
        result = engine.run_multi_instrument_backtest(
            data_dict=data_dict,
            symbol_specs=symbol_specs,
            signal_func=signal_func,
        )

        print("\n=== Multi-Instrument Results ===")
        print(f"Instruments tested: {list(result['instrument_results'].keys())}")
        equity_curve = result["portfolio_equity_curve"]
        print(f"Equity curve length: {len(equity_curve)}")
        if len(equity_curve) > 0:
            print(f"Portfolio final equity: ${equity_curve.iloc[-1]:,.2f}")
        else:
            print(f"Portfolio final equity: ${engine.initial_capital:,.2f} (no trades)")
        print(f"Portfolio Sharpe: {result['portfolio_sharpe']:.3f}")
        print(f"Portfolio max drawdown: {result['portfolio_max_drawdown_pct']:.2f}%")

        print("\n--- Per-Instrument Results ---")
        for symbol, inst_result in result["instrument_results"].items():
            print(f"\n{symbol}:")
            print(f"  Total trades: {inst_result.total_trades}")
            print(f"  Win rate: {inst_result.win_rate:.1%}")
            print(f"  Total PnL: ${inst_result.total_net_pnl:,.2f}")
            pf = (
                inst_result.gross_profit / abs(inst_result.gross_loss)
                if inst_result.gross_loss != 0
                else 0.0
            )
            print(f"  Profit factor: {pf:.2f}")

        print(f"\nSignal counts: {signal_counts}")
        print("\n[OK] Multi-instrument backtest completed successfully!")

    except Exception as e:
        import traceback

        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
