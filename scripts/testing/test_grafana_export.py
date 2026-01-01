#!/usr/bin/env python3
"""
Grafana Export Test

Demonstrates exporting all Kinetra metrics to Grafana for visualization:
- Real-time trade metrics
- Friction costs breakdown
- Execution quality (MFE/MAE)
- Portfolio health monitoring
- Market regime tracking
- Agent performance

Supports multiple backends:
- Prometheus (pull-based scraping)
- InfluxDB (push-based)
- Graphite (push-based)
- JSON (debugging)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from kinetra.realistic_backtester import RealisticBacktester
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.grafana_exporter import GrafanaExporter, create_grafana_dashboards


def load_test_data(max_rows: int = 500) -> pd.DataFrame:
    """Load real market data."""
    data_path = "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv"

    df = pd.read_csv(data_path, sep='\t', nrows=max_rows)
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


def create_spec_with_costs() -> SymbolSpec:
    """Create symbol spec with realistic costs."""
    return SymbolSpec(
        symbol="EURJPY+",
        asset_class=AssetClass.FOREX,
        digits=3,
        point=0.001,
        contract_size=100000.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_typical=20,
        commission_per_lot=6.0,
        swap_long=-0.3,
        swap_short=0.1,
        swap_triple_day="wednesday",
        trade_freeze_level=50,
        trade_stops_level=100,
    )


def generate_signals(data: pd.DataFrame, max_trades: int = 10) -> pd.DataFrame:
    """Generate signals for testing."""
    signals = []
    indices = np.linspace(20, len(data) - 50, max_trades, dtype=int)

    for i, idx in enumerate(indices):
        entry_price = data.iloc[idx]['close']
        signals.append({
            'time': data.index[idx],
            'action': 'open_long' if i % 2 == 0 else 'open_short',
            'sl': entry_price - 0.200 if i % 2 == 0 else entry_price + 0.200,
            'tp': entry_price + 0.400 if i % 2 == 0 else entry_price - 0.400,
            'volume': 1.0,
        })
        exit_idx = min(idx + 20 + i*2, len(data) - 1)
        signals.append({
            'time': data.index[exit_idx],
            'action': 'close',
            'sl': None,
            'tp': None,
        })

    return pd.DataFrame(signals)


def run_backtest_with_grafana_export():
    """Run backtest and export metrics to Grafana."""
    print("=" * 100)
    print("GRAFANA METRICS EXPORT TEST")
    print("=" * 100)

    # Load data
    print("\n[Loading Data]")
    data = load_test_data(max_rows=500)
    print(f"  Loaded {len(data)} candles")

    # Create spec
    spec = create_spec_with_costs()
    print(f"\n[Symbol: {spec.symbol}]")

    # Generate signals
    signals = generate_signals(data, max_trades=10)
    print(f"\n[Generated {len(signals)//2} trades]")

    # Initialize Grafana exporter
    print("\n[Initializing Grafana Exporter]")
    exporter = GrafanaExporter(
        backend='influxdb',  # Change to 'prometheus', 'graphite', or 'json'
        host='localhost',
        port=8086,  # InfluxDB port
        db_name='kinetra',
        enable_export=True,
    )
    print(f"  Backend: {exporter.backend}")
    print(f"  Target: {exporter.host}:{exporter.port}")

    # Run backtest
    print("\n[Running Backtest with Grafana Export]")
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        enable_slippage=True,
    )

    result = backtester.run(data, signals, classify_regimes=False)

    # Export all trades to Grafana
    print(f"\n[Exporting {len(result.trades)} trades to Grafana]")

    regimes = ["LAMINAR", "TURBULENT", "TRANSITIONAL"]

    for i, trade in enumerate(result.trades, 1):
        # Record entry
        entry_spread = trade.entry_spread * spec.point * trade.volume * spec.contract_size
        entry_commission = spec.commission_per_lot * trade.volume

        regime = regimes[i % 3]

        exporter.record_trade_entry(
            time=trade.entry_time,
            symbol=spec.symbol,
            direction=trade.direction,
            volume=trade.volume,
            entry_price=trade.entry_price,
            spread=entry_spread,
            commission=entry_commission,
            regime=regime,
        )

        # Record exit
        exit_spread = trade.exit_spread * spec.point * trade.volume * spec.contract_size
        total_spread = entry_spread + exit_spread
        gross_pnl = trade.pnl + trade.total_cost

        exporter.record_trade_exit(
            time=trade.exit_time,
            symbol=spec.symbol,
            direction=trade.direction,
            volume=trade.volume,
            exit_price=trade.exit_price,
            pnl=trade.pnl,
            gross_pnl=gross_pnl,
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

        # Record health updates every 3 trades
        if i % 3 == 0:
            score = 85 - (i * 2)
            state = "HEALTHY" if score > 80 else "WARNING" if score > 60 else "DEGRADED"

            exporter.record_health_update(
                time=trade.exit_time,
                score=score,
                state=state,
                risk_multiplier=1.0 if state == "HEALTHY" else 0.7,
                return_efficiency=100.0,
                downside_risk=30.0,
                structural_stability=75.0,
                behavioral_health=70.0,
            )

        # Record regime changes
        if i % 4 == 0:
            old_regime = regimes[(i // 4) % 3]
            new_regime = regimes[((i // 4) + 1) % 3]

            exporter.record_regime_change(
                time=trade.exit_time,
                old_regime=old_regime,
                new_regime=new_regime,
                volatility=0.015 if new_regime == "TURBULENT" else 0.005,
                trend_strength=0.7 if new_regime == "LAMINAR" else 0.3,
            )

        # Record agent events
        if i % 5 == 0:
            exporter.record_agent_event(
                time=trade.exit_time,
                event_type='drift_detected',
                agent_name='live',
                metric_value=0.15,  # 15% drift
                details={'severity': 'medium'},
            )

    # Record final summary
    print("\n[Exporting Summary Metrics]")
    exporter.record_backtest_summary(
        time=result.trades[-1].exit_time,
        total_trades=len(result.trades),
        win_rate=len([t for t in result.trades if t.pnl > 0]) / len(result.trades),
        total_pnl=result.total_pnl,
        total_return_pct=result.total_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        total_spread=result.total_spread_cost,
        total_commission=result.total_commission,
        total_swap=result.total_swap,
    )

    # Flush and show metrics
    print("\n[Flushing Metrics]")
    metrics = exporter.flush()
    print(f"  Total metrics: {len(metrics)}")

    # Show sample metrics
    print("\n[Sample Metrics (first 10)]")
    for metric in metrics[:10]:
        print(f"  {metric}")

    # Save to file for debugging
    output_file = "/tmp/kinetra_metrics.txt"
    with open(output_file, 'w') as f:
        for metric in metrics:
            f.write(metric + '\n')
    print(f"\n[Saved {len(metrics)} metrics to {output_file}]")

    # Show dashboard configs
    print("\n[Grafana Dashboard Configurations]")
    dashboards = create_grafana_dashboards()
    for name, config in dashboards.items():
        print(f"\n  Dashboard: {config['title']}")
        print(f"    Panels: {len(config['panels'])}")
        for panel in config['panels']:
            print(f"      - {panel['title']} ({panel['type']})")

    print("\n" + "=" * 100)
    print("METRICS EXPORT COMPLETE")
    print("=" * 100)
    print("\nTo visualize in Grafana:")
    print("1. Set up InfluxDB:")
    print("   docker run -p 8086:8086 influxdb:1.8")
    print("\n2. Create database:")
    print("   curl -XPOST 'http://localhost:8086/query' --data-urlencode 'q=CREATE DATABASE kinetra'")
    print("\n3. Run this script to push metrics")
    print("\n4. Set up Grafana:")
    print("   docker run -p 3000:3000 grafana/grafana")
    print("\n5. Add InfluxDB data source in Grafana (http://localhost:8086)")
    print("\n6. Import dashboard JSON from create_grafana_dashboards()")
    print("\nAll metrics are now available for real-time visualization!")

    return result, exporter


if __name__ == "__main__":
    result, exporter = run_backtest_with_grafana_export()
