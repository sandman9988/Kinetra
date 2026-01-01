#!/usr/bin/env python3
"""
MT5-Style Logger Test

Demonstrates enhanced transaction logging with:
- Real-time order/deal logging
- Position tracking with detailed metrics
- Comprehensive friction cost breakdown
- MFE/MAE analysis
- Regime tracking
- Health monitoring
- Final summary statistics

Mimics MT5's backtest output but with significantly more detail.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from kinetra.realistic_backtester import RealisticBacktester
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.trade_logger import MT5Logger


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


def generate_signals(data: pd.DataFrame, max_trades: int = 10) -> pd.DataFrame:
    """Generate signals for testing."""
    signals = []

    # Generate trades at different times
    indices = np.linspace(20, len(data) - 50, max_trades, dtype=int)

    for i, idx in enumerate(indices):
        entry_price = data.iloc[idx]['close']

        # Alternate between long and short
        if i % 2 == 0:
            # Long trade
            signals.append({
                'time': data.index[idx],
                'action': 'open_long',
                'sl': entry_price - 0.200,  # 200 pips SL
                'tp': entry_price + 0.400,  # 400 pips TP
                'volume': 1.0,
            })
        else:
            # Short trade
            signals.append({
                'time': data.index[idx],
                'action': 'open_short',
                'sl': entry_price + 0.200,
                'tp': entry_price - 0.400,
                'volume': 1.0,
            })

        # Close after 20-40 candles
        exit_idx = min(idx + 20 + i*2, len(data) - 1)
        signals.append({
            'time': data.index[exit_idx],
            'action': 'close',
            'sl': None,
            'tp': None,
        })

    return pd.DataFrame(signals)


def run_backtest_with_logging():
    """Run backtest with enhanced MT5-style logging."""
    print("=" * 100)
    print("MT5-STYLE ENHANCED LOGGING TEST")
    print("=" * 100)

    # 1. Load data
    print("\n[Loading Data]")
    data = load_test_data(max_rows=500)
    print(f"  Loaded {len(data)} candles")
    print(f"  Period: {data.index[0]} to {data.index[-1]}")

    # 2. Create spec
    spec = create_spec_with_costs()
    print(f"\n[Symbol: {spec.symbol}]")
    print(f"  Spread: {spec.spread_typical} points")
    print(f"  Commission: ${spec.commission_per_lot}/lot/side")
    print(f"  Swap: {spec.swap_long} pts/day (long)")

    # 3. Generate signals
    signals = generate_signals(data, max_trades=10)
    print(f"\n[Generated {len(signals)} signals ({len(signals)//2} trades)]")

    # 4. Initialize logger
    logger = MT5Logger(
        symbol=spec.symbol,
        timeframe="M15",
        initial_balance=10000.0,
        enable_verbose=True,
    )

    print("\n" + "=" * 100)
    print("STARTING BACKTEST WITH REAL-TIME LOGGING")
    print("=" * 100 + "\n")

    # 5. Run backtest
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        enable_slippage=True,
        enable_freeze_zones=True,
        enable_stop_validation=True,
    )

    result = backtester.run(data, signals, classify_regimes=False)

    # 6. Log all trades
    for i, trade in enumerate(result.trades, 1):
        # Entry
        action = 'buy' if trade.direction == 1 else 'sell'
        entry_spread = trade.entry_spread * spec.point * trade.volume * spec.contract_size
        entry_commission = spec.commission_per_lot * trade.volume

        # Log order
        order_id = logger.log_order_send(
            time=trade.entry_time,
            action=action,
            volume=trade.volume,
            price=trade.entry_price,
            sl=None,  # SL tracked separately
            tp=None,  # TP tracked separately
            spread_points=trade.entry_spread,
            bid=trade.entry_price if action == 'sell' else trade.entry_price - trade.entry_spread * spec.point,
            ask=trade.entry_price if action == 'buy' else trade.entry_price + trade.entry_spread * spec.point,
        )

        # Log deal
        logger.log_deal(
            time=trade.entry_time,
            action=action,
            volume=trade.volume,
            price=trade.entry_price,
            order_id=order_id,
        )

        # Log position open with enhanced metrics
        position_id = logger.log_position_open(
            time=trade.entry_time,
            direction=trade.direction,
            volume=trade.volume,
            entry_price=trade.entry_price,
            spread=entry_spread,
            commission=entry_commission,
            regime="LAMINAR" if i % 3 == 0 else "TURBULENT" if i % 3 == 1 else "TRANSITIONAL",
        )

        # Exit
        exit_action = 'sell' if trade.direction == 1 else 'buy'
        exit_spread = trade.exit_spread * spec.point * trade.volume * spec.contract_size
        total_spread = entry_spread + exit_spread

        # Log exit order
        exit_order_id = logger.log_order_send(
            time=trade.exit_time,
            action=exit_action,
            volume=trade.volume,
            price=trade.exit_price,
            spread_points=trade.exit_spread,
            bid=trade.exit_price if exit_action == 'sell' else trade.exit_price - trade.exit_spread * spec.point,
            ask=trade.exit_price if exit_action == 'buy' else trade.exit_price + trade.exit_spread * spec.point,
            close_position_id=position_id,
        )

        # Log exit deal
        logger.log_deal(
            time=trade.exit_time,
            action=exit_action,
            volume=trade.volume,
            price=trade.exit_price,
            order_id=exit_order_id,
            position_id=position_id,
        )

        # Log position close with full metrics
        logger.log_position_close(
            time=trade.exit_time,
            position_id=position_id,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            spread=total_spread,
            commission=trade.commission,
            swap=trade.swap,
            slippage=abs(trade.entry_slippage) + abs(trade.exit_slippage) if hasattr(trade, 'entry_slippage') else 0,
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_efficiency=trade.mfe_efficiency,
            holding_hours=trade.holding_time,
            exit_reason="signal",
        )

        # Log health updates every 3 trades
        if i % 3 == 0:
            score = 85 - (i * 2)  # Mock degrading health
            states = ["HEALTHY", "WARNING", "DEGRADED"]
            state = states[min(i // 4, 2)]
            logger.log_health_update(
                time=trade.exit_time,
                score=score,
                state=state,
                action="Monitor" if state == "HEALTHY" else "Reduce risk",
                risk_multiplier=1.0 if state == "HEALTHY" else 0.7 if state == "WARNING" else 0.5,
            )

        # Log regime changes occasionally
        if i % 4 == 0:
            regimes = ["LAMINAR", "TURBULENT", "TRANSITIONAL"]
            old_regime = regimes[(i // 4) % 3]
            new_regime = regimes[((i // 4) + 1) % 3]
            logger.log_regime_change(
                time=trade.exit_time,
                old_regime=old_regime,
                new_regime=new_regime,
                reason="Volatility spike" if "TURBULENT" in new_regime else "Market calming",
            )

    # 7. Log final summary
    logger.log_final_summary(
        total_trades=len(result.trades),
        winning_trades=len([t for t in result.trades if t.pnl > 0]),
        losing_trades=len([t for t in result.trades if t.pnl <= 0]),
        total_pnl=result.total_pnl,
        total_return_pct=result.total_return_pct,
        max_drawdown=result.max_drawdown,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        total_spread_cost=result.total_spread_cost,
        total_commission=result.total_commission,
        total_swap=result.total_swap,
        total_slippage=result.total_slippage,
        freeze_violations=result.total_freeze_violations,
        stop_violations=result.total_invalid_stops,
        bars_processed=len(data),
        memory_mb=50.5,  # Mock value
    )

    return result


if __name__ == "__main__":
    result = run_backtest_with_logging()
