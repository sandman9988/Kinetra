#!/usr/bin/env python3
"""
QUICK RESULTS - Get Numbers on Screen NOW
==========================================

Uses existing CSV data directly - NO preparation needed.
Shows actual trading performance numbers immediately.
"""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

# =============================================================================
# SIMPLE DATA LOADER - Direct from CSV
# =============================================================================


def load_csv_direct(filepath: Path) -> pd.DataFrame:
    """Load MT5 CSV directly - no preprocessing needed."""
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("datetime", inplace=True)
    df = df.rename(columns={"tickvol": "volume"})
    return df[["open", "high", "low", "close", "volume", "spread"]]


# =============================================================================
# SIMPLE AGENTS - No Dependencies
# =============================================================================


class SimpleAgent:
    """Base agent - random baseline."""

    name = "Random"

    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0  # -1, 0, 1
        self.entry_price = 0
        self.trades = []

    def act(self, state: dict) -> int:
        """Return action: 0=hold, 1=buy, 2=sell, 3=close."""
        return np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])


class MomentumAgent(SimpleAgent):
    """Simple momentum - buy on up, sell on down."""

    name = "Momentum"

    def __init__(self, lookback: int = 10):
        super().__init__()
        self.lookback = lookback
        self.prices = []

    def act(self, state: dict) -> int:
        self.prices.append(state["close"])
        if len(self.prices) < self.lookback:
            return 0

        # Simple momentum
        returns = (self.prices[-1] - self.prices[-self.lookback]) / self.prices[-self.lookback]

        if self.position == 0:
            if returns > 0.001:  # Up momentum
                return 1  # Buy
            elif returns < -0.001:  # Down momentum
                return 2  # Sell
        elif self.position != 0 and abs(returns) < 0.0005:
            return 3  # Close on momentum fade

        return 0


class MeanReversionAgent(SimpleAgent):
    """Simple mean reversion - fade extremes."""

    name = "MeanReversion"

    def __init__(self, lookback: int = 20, threshold: float = 2.0):
        super().__init__()
        self.lookback = lookback
        self.threshold = threshold
        self.prices = []

    def act(self, state: dict) -> int:
        self.prices.append(state["close"])
        if len(self.prices) < self.lookback:
            return 0

        # Z-score
        window = self.prices[-self.lookback :]
        mean = np.mean(window)
        std = np.std(window) + 1e-10
        zscore = (self.prices[-1] - mean) / std

        if self.position == 0:
            if zscore < -self.threshold:  # Oversold
                return 1  # Buy
            elif zscore > self.threshold:  # Overbought
                return 2  # Sell
        elif self.position == 1 and zscore > 0:
            return 3  # Close long at mean
        elif self.position == -1 and zscore < 0:
            return 3  # Close short at mean

        return 0


class BreakoutAgent(SimpleAgent):
    """Simple breakout - trade range breaks."""

    name = "Breakout"

    def __init__(self, lookback: int = 50):
        super().__init__()
        self.lookback = lookback
        self.highs = []
        self.lows = []

    def act(self, state: dict) -> int:
        self.highs.append(state["high"])
        self.lows.append(state["low"])

        if len(self.highs) < self.lookback:
            return 0

        highest = max(self.highs[-self.lookback : -1])
        lowest = min(self.lows[-self.lookback : -1])
        current = state["close"]

        if self.position == 0:
            if current > highest:  # Breakout up
                return 1
            elif current < lowest:  # Breakout down
                return 2
        elif self.position != 0:
            # Trail stop at middle of range
            mid = (highest + lowest) / 2
            if self.position == 1 and current < mid:
                return 3
            elif self.position == -1 and current > mid:
                return 3

        return 0


# =============================================================================
# BACKTEST ENGINE - Simple & Fast
# =============================================================================


def backtest_agent(agent, data: pd.DataFrame, symbol: str, spread_cost: float = 0.0001):
    """
    Run simple backtest.

    Returns dict with performance metrics.
    """
    agent.reset()

    balance = 10000.0
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [balance]

    # Vectorized state preparation
    for i, (open_val, high_val, low_val, close_val, volume_val, spread_val) in enumerate(
        zip(
            data["open"].values,
            data["high"].values,
            data["low"].values,
            data["close"].values,
            data["volume"].values,
            data["spread"].values,
        )
    ):
        state = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
            "spread": spread_val,
            "bar": i,
        }

        action = agent.act(state)
        price = close_val

        # Execute action
        if action == 1 and position == 0:  # Buy
            position = 1
            entry_price = price * (1 + spread_cost)  # Pay spread

        elif action == 2 and position == 0:  # Sell
            position = -1
            entry_price = price * (1 - spread_cost)  # Pay spread

        elif action == 3 and position != 0:  # Close
            if position == 1:
                pnl = (price - entry_price) / entry_price
            else:
                pnl = (entry_price - price) / entry_price

            pnl_dollars = balance * 0.1 * pnl  # 10% position size
            balance += pnl_dollars
            trades.append(
                {
                    "entry": entry_price,
                    "exit": price,
                    "pnl": pnl_dollars,
                    "direction": "long" if position == 1 else "short",
                    "bars_held": i - len(trades),
                }
            )
            position = 0
            entry_price = 0

        # Track equity
        if position == 1:
            unrealized = balance * 0.1 * ((price - entry_price) / entry_price)
        elif position == -1:
            unrealized = balance * 0.1 * ((entry_price - price) / entry_price)
        else:
            unrealized = 0
        equity_curve.append(balance + unrealized)

    # Close any open position at end
    if position != 0:
        price = data.iloc[-1]["close"]
        if position == 1:
            pnl = (price - entry_price) / entry_price
        else:
            pnl = (entry_price - price) / entry_price
        pnl_dollars = balance * 0.1 * pnl
        balance += pnl_dollars
        trades.append({"pnl": pnl_dollars})

    # Calculate metrics
    if not trades:
        return {"error": "No trades"}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak
    max_dd = np.max(drawdown) * 100

    total_return = ((balance - 10000) / 10000) * 100

    return {
        "symbol": symbol,
        "agent": agent.name,
        "total_return": total_return,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": abs(sum(wins) / sum(losses))
        if losses and sum(losses) != 0
        else float("inf"),
        "max_drawdown": max_dd,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "final_balance": balance,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("QUICK RESULTS - Numbers on Screen NOW")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    # Find data files
    data_dir = Path("/workspace/data/runs/berserker_run3/data")
    if not data_dir.exists():
        data_dir = Path("/workspace/data/master")

    # Test configs - 3 symbols, multiple agents
    test_files = [
        ("XAUUSD", "XAUUSD+_H1_202401020100_202512262300.csv"),
        ("BTCUSD", "BTCUSD_H1_202401020000_202512282200.csv"),
        ("GBPUSD", "GBPUSD+_H1_202401020000_202512262300.csv"),
    ]

    agents = [
        SimpleAgent(),
        MomentumAgent(lookback=10),
        MeanReversionAgent(lookback=20),
        BreakoutAgent(lookback=50),
    ]

    all_results = []

    for symbol, filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"⚠ File not found: {filename}")
            continue

        print(f"\n{'─' * 70}")
        print(f"Loading {symbol}...")
        data = load_csv_direct(filepath)

        # Use last 5000 bars for speed
        test_data = data.iloc[-5000:]
        print(f"  {len(test_data)} bars from {test_data.index[0]} to {test_data.index[-1]}")

        print(f"\n  Testing agents...")

        for agent in agents:
            start = time.time()
            result = backtest_agent(agent, test_data, symbol)
            elapsed = time.time() - start

            if "error" not in result:
                print(
                    f"    {agent.name:15} | Return: {result['total_return']:>7.2f}% | "
                    f"Trades: {result['trades']:>4} | Win: {result['win_rate']:>5.1f}% | "
                    f"DD: {result['max_drawdown']:>5.1f}% | {elapsed:.1f}s"
                )
                all_results.append(result)
            else:
                print(f"    {agent.name:15} | {result['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY AGENT")
    print("=" * 70)

    if all_results:
        df = pd.DataFrame(all_results)

        # Group by agent
        summary = (
            df.groupby("agent")
            .agg(
                {
                    "total_return": "mean",
                    "trades": "sum",
                    "win_rate": "mean",
                    "max_drawdown": "mean",
                    "profit_factor": "mean",
                }
            )
            .round(2)
        )

        print(
            f"\n{'Agent':<15} {'Avg Return':>12} {'Total Trades':>14} {'Avg Win%':>10} {'Avg DD%':>10} {'Avg PF':>10}"
        )
        print("-" * 70)
        # Vectorized summary printing
        for agent_name, total_ret, trades, win_rt, max_dd, pf_val in zip(
            summary.index,
            summary["total_return"].values,
            summary["trades"].values,
            summary["win_rate"].values,
            summary["max_drawdown"].values,
            summary["profit_factor"].values,
        ):
            pf = f"{pf_val:.2f}" if pf_val < 100 else "∞"
            print(
                f"{agent_name:<15} {total_ret:>11.2f}% {int(trades):>14} "
                f"{win_rt:>9.1f}% {max_dd:>9.1f}% {pf:>10}"
            )

        # Best performer
        best = df.loc[df["total_return"].idxmax()]
        print(
            f"\n🏆 Best: {best['agent']} on {best['symbol']} with {best['total_return']:.2f}% return"
        )

    print(f"\n✅ Completed at {datetime.now()}")
    print("\nNEXT: Use these baselines to compare against RL agents")


if __name__ == "__main__":
    main()
