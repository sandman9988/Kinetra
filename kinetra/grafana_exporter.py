"""
Grafana Metrics Exporter for Kinetra

Exports all trading metrics to Grafana for real-time visualization:
- Trade execution metrics
- Friction costs (spread, commission, swap, slippage)
- P&L breakdown (gross, net, costs)
- Execution quality (MFE, MAE, efficiency)
- Portfolio health scores
- Market regime transitions
- Agent performance and drift
- MT5 constraint violations

Supports:
- Prometheus metrics endpoint
- InfluxDB line protocol
- Graphite plaintext protocol
- JSON over HTTP

Usage:
    exporter = GrafanaExporter(backend='prometheus', port=9090)
    exporter.record_trade(trade, spec)
    exporter.record_health_update(score, state)
    exporter.record_regime_change(old, new)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None

    def to_influx(self) -> str:
        """Convert to InfluxDB line protocol."""
        tags_str = ','.join(f"{k}={v}" for k, v in (self.tags or {}).items())
        timestamp_ns = int(self.timestamp.timestamp() * 1e9)
        if tags_str:
            return f"{self.name},{tags_str} value={self.value} {timestamp_ns}"
        return f"{self.name} value={self.value} {timestamp_ns}"

    def to_graphite(self) -> str:
        """Convert to Graphite plaintext."""
        timestamp_s = int(self.timestamp.timestamp())
        tags_str = '.' + '.'.join(f"{k}_{v}" for k, v in (self.tags or {}).items())
        return f"kinetra.{self.name}{tags_str} {self.value} {timestamp_s}"

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        tags_str = ','.join(f'{k}="{v}"' for k, v in (self.tags or {}).items())
        if tags_str:
            return f"kinetra_{self.name}{{{tags_str}}} {self.value}"
        return f"kinetra_{self.name} {self.value}"


class GrafanaExporter:
    """
    Export Kinetra metrics to Grafana.

    Supports multiple backends:
    - prometheus: Prometheus metrics endpoint (pull)
    - influxdb: InfluxDB line protocol (push)
    - graphite: Graphite plaintext protocol (push)
    - json: JSON over HTTP (push)
    """

    def __init__(
        self,
        backend: str = 'prometheus',
        host: str = 'localhost',
        port: int = 9090,
        db_name: str = 'kinetra',
        enable_export: bool = True,
    ):
        """
        Initialize exporter.

        Args:
            backend: 'prometheus', 'influxdb', 'graphite', or 'json'
            host: Target host
            port: Target port
            db_name: Database/measurement name
            enable_export: Enable metric export
        """
        self.backend = backend
        self.host = host
        self.port = port
        self.db_name = db_name
        self.enable_export = enable_export

        # Metric buffers
        self.metrics: List[MetricPoint] = []
        self.trade_counter = 0

        # Cumulative stats
        self.cumulative_pnl = 0.0
        self.cumulative_costs = 0.0
        self.cumulative_spread = 0.0
        self.cumulative_commission = 0.0
        self.cumulative_swap = 0.0

    def record_trade_entry(
        self,
        time: datetime,
        symbol: str,
        direction: int,
        volume: float,
        entry_price: float,
        spread: float,
        commission: float,
        regime: Optional[str] = None,
    ):
        """Record trade entry metrics."""
        if not self.enable_export:
            return

        tags = {
            'symbol': symbol,
            'direction': 'long' if direction == 1 else 'short',
            'regime': regime or 'unknown',
        }

        # Entry metrics
        self.metrics.append(MetricPoint('trade_entry', 1.0, time, tags))
        self.metrics.append(MetricPoint('entry_price', entry_price, time, tags))
        self.metrics.append(MetricPoint('entry_volume', volume, time, tags))
        self.metrics.append(MetricPoint('entry_spread_cost', spread, time, tags))
        self.metrics.append(MetricPoint('entry_commission', commission, time, tags))

        # Cumulative
        self.cumulative_spread += spread
        self.cumulative_commission += commission
        self.metrics.append(MetricPoint('cumulative_spread', self.cumulative_spread, time))
        self.metrics.append(MetricPoint('cumulative_commission', self.cumulative_commission, time))

    def record_trade_exit(
        self,
        time: datetime,
        symbol: str,
        direction: int,
        volume: float,
        exit_price: float,
        pnl: float,
        gross_pnl: float,
        spread: float,
        commission: float,
        swap: float,
        slippage: float,
        mfe: float,
        mae: float,
        mfe_efficiency: float,
        holding_hours: float,
        exit_reason: str,
    ):
        """Record trade exit metrics."""
        if not self.enable_export:
            return

        self.trade_counter += 1

        tags = {
            'symbol': symbol,
            'direction': 'long' if direction == 1 else 'short',
            'exit_reason': exit_reason,
        }

        # Exit metrics
        self.metrics.append(MetricPoint('trade_exit', 1.0, time, tags))
        self.metrics.append(MetricPoint('exit_price', exit_price, time, tags))
        self.metrics.append(MetricPoint('trade_pnl', pnl, time, tags))
        self.metrics.append(MetricPoint('trade_gross_pnl', gross_pnl, time, tags))
        self.metrics.append(MetricPoint('trade_holding_hours', holding_hours, time, tags))

        # Costs
        total_costs = spread + commission + swap + slippage
        self.metrics.append(MetricPoint('exit_spread_cost', spread, time, tags))
        self.metrics.append(MetricPoint('total_commission', commission, time, tags))
        self.metrics.append(MetricPoint('total_swap', swap, time, tags))
        self.metrics.append(MetricPoint('total_slippage', slippage, time, tags))
        self.metrics.append(MetricPoint('total_trade_costs', total_costs, time, tags))

        # Execution quality
        self.metrics.append(MetricPoint('mfe', mfe, time, tags))
        self.metrics.append(MetricPoint('mae', mae, time, tags))
        self.metrics.append(MetricPoint('mfe_efficiency', mfe_efficiency, time, tags))
        if mae > 0:
            self.metrics.append(MetricPoint('mfe_mae_ratio', mfe / mae, time, tags))

        # Cost analysis
        if gross_pnl != 0:
            cost_pct = (total_costs / abs(gross_pnl)) * 100
            self.metrics.append(MetricPoint('cost_impact_pct', cost_pct, time, tags))

        # Cumulative
        self.cumulative_pnl += pnl
        self.cumulative_costs += total_costs
        self.cumulative_swap += swap

        self.metrics.append(MetricPoint('cumulative_pnl', self.cumulative_pnl, time))
        self.metrics.append(MetricPoint('cumulative_costs', self.cumulative_costs, time))
        self.metrics.append(MetricPoint('cumulative_swap', self.cumulative_swap, time))
        self.metrics.append(MetricPoint('total_trades', float(self.trade_counter), time))

    def record_health_update(
        self,
        time: datetime,
        score: float,
        state: str,
        risk_multiplier: float,
        return_efficiency: float = 0,
        downside_risk: float = 0,
        structural_stability: float = 0,
        behavioral_health: float = 0,
    ):
        """Record portfolio health metrics."""
        if not self.enable_export:
            return

        tags = {'state': state}

        # Overall health
        self.metrics.append(MetricPoint('health_score', score, time, tags))
        self.metrics.append(MetricPoint('risk_multiplier', risk_multiplier, time, tags))

        # 4 pillars
        self.metrics.append(MetricPoint('health_return_efficiency', return_efficiency, time))
        self.metrics.append(MetricPoint('health_downside_risk', downside_risk, time))
        self.metrics.append(MetricPoint('health_structural_stability', structural_stability, time))
        self.metrics.append(MetricPoint('health_behavioral', behavioral_health, time))

    def record_regime_change(
        self,
        time: datetime,
        old_regime: str,
        new_regime: str,
        volatility: float = 0,
        trend_strength: float = 0,
    ):
        """Record market regime transitions."""
        if not self.enable_export:
            return

        tags = {
            'from_regime': old_regime,
            'to_regime': new_regime,
        }

        self.metrics.append(MetricPoint('regime_change', 1.0, time, tags))
        self.metrics.append(MetricPoint('regime_volatility', volatility, time, {'regime': new_regime}))
        self.metrics.append(MetricPoint('regime_trend', trend_strength, time, {'regime': new_regime}))

    def record_agent_event(
        self,
        time: datetime,
        event_type: str,
        agent_name: str,
        metric_value: float,
        details: Optional[Dict[str, str]] = None,
    ):
        """Record agent-related events (drift, promotion, etc.)."""
        if not self.enable_export:
            return

        tags = {
            'event': event_type,
            'agent': agent_name,
            **(details or {}),
        }

        self.metrics.append(MetricPoint('agent_event', 1.0, time, tags))
        self.metrics.append(MetricPoint('agent_metric', metric_value, time, tags))

    def record_constraint_violation(
        self,
        time: datetime,
        violation_type: str,
        symbol: str,
        severity: str = 'warning',
    ):
        """Record MT5 constraint violations."""
        if not self.enable_export:
            return

        tags = {
            'type': violation_type,
            'symbol': symbol,
            'severity': severity,
        }

        self.metrics.append(MetricPoint('constraint_violation', 1.0, time, tags))

    def record_backtest_summary(
        self,
        time: datetime,
        total_trades: int,
        win_rate: float,
        total_pnl: float,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        total_spread: float,
        total_commission: float,
        total_swap: float,
    ):
        """Record final backtest summary metrics."""
        if not self.enable_export:
            return

        # Summary metrics
        self.metrics.append(MetricPoint('summary_total_trades', float(total_trades), time))
        self.metrics.append(MetricPoint('summary_win_rate', win_rate, time))
        self.metrics.append(MetricPoint('summary_total_pnl', total_pnl, time))
        self.metrics.append(MetricPoint('summary_return_pct', total_return_pct, time))
        self.metrics.append(MetricPoint('summary_sharpe', sharpe_ratio, time))
        self.metrics.append(MetricPoint('summary_max_dd_pct', max_drawdown_pct, time))
        self.metrics.append(MetricPoint('summary_total_spread', total_spread, time))
        self.metrics.append(MetricPoint('summary_total_commission', total_commission, time))
        self.metrics.append(MetricPoint('summary_total_swap', total_swap, time))

    def flush(self) -> List[str]:
        """
        Flush metrics buffer and return formatted strings.

        Returns:
            List of formatted metric strings for the selected backend
        """
        if not self.metrics:
            return []

        if self.backend == 'influxdb':
            output = [m.to_influx() for m in self.metrics]
        elif self.backend == 'graphite':
            output = [m.to_graphite() for m in self.metrics]
        elif self.backend == 'prometheus':
            output = [m.to_prometheus() for m in self.metrics]
        elif self.backend == 'json':
            output = [json.dumps({
                'name': m.name,
                'value': m.value,
                'timestamp': m.timestamp.isoformat(),
                'tags': m.tags or {},
            }) for m in self.metrics]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.metrics.clear()
        return output

    def get_prometheus_metrics(self) -> str:
        """Get all metrics in Prometheus format (for scraping endpoint)."""
        lines = self.flush()
        return '\n'.join(lines) + '\n'

    def export_to_file(self, filepath: str):
        """Export metrics to file (for debugging)."""
        lines = self.flush()
        with open(filepath, 'a') as f:
            for line in lines:
                f.write(line + '\n')

    def export_to_influxdb(self):
        """Push metrics to InfluxDB (requires requests library)."""
        try:
            import requests
            lines = self.flush()
            if not lines:
                return

            url = f"http://{self.host}:{self.port}/write?db={self.db_name}"
            response = requests.post(url, data='\n'.join(lines))
            response.raise_for_status()
        except ImportError:
            print("Warning: requests library not available for InfluxDB export")
        except Exception as e:
            print(f"Error exporting to InfluxDB: {e}")

    def export_to_graphite(self):
        """Push metrics to Graphite (requires socket)."""
        try:
            import socket
            lines = self.flush()
            if not lines:
                return

            sock = socket.socket()
            sock.connect((self.host, self.port))
            sock.sendall(('\n'.join(lines) + '\n').encode())
            sock.close()
        except Exception as e:
            print(f"Error exporting to Graphite: {e}")


def create_grafana_dashboards() -> Dict[str, Any]:
    """
    Create Grafana dashboard JSON configurations.

    Returns:
        Dictionary of dashboard configs for different views
    """
    return {
        'trading_overview': {
            'title': 'Kinetra Trading Overview',
            'panels': [
                {
                    'title': 'Cumulative P&L',
                    'targets': ['cumulative_pnl'],
                    'type': 'graph',
                },
                {
                    'title': 'Trade Count',
                    'targets': ['total_trades'],
                    'type': 'stat',
                },
                {
                    'title': 'Win Rate',
                    'targets': ['summary_win_rate'],
                    'type': 'gauge',
                    'thresholds': [0.3, 0.5, 0.7],
                },
                {
                    'title': 'Health Score',
                    'targets': ['health_score'],
                    'type': 'gauge',
                    'thresholds': [40, 60, 80],
                },
            ],
        },
        'friction_costs': {
            'title': 'Friction Costs Analysis',
            'panels': [
                {
                    'title': 'Cumulative Costs',
                    'targets': [
                        'cumulative_spread',
                        'cumulative_commission',
                        'cumulative_swap',
                    ],
                    'type': 'graph',
                    'stack': True,
                },
                {
                    'title': 'Cost per Trade',
                    'targets': [
                        'exit_spread_cost',
                        'total_commission',
                        'total_swap',
                        'total_slippage',
                    ],
                    'type': 'bargauge',
                },
                {
                    'title': 'Cost Impact %',
                    'targets': ['cost_impact_pct'],
                    'type': 'graph',
                },
            ],
        },
        'execution_quality': {
            'title': 'Execution Quality',
            'panels': [
                {
                    'title': 'MFE vs MAE',
                    'targets': ['mfe', 'mae'],
                    'type': 'graph',
                },
                {
                    'title': 'MFE Efficiency',
                    'targets': ['mfe_efficiency'],
                    'type': 'graph',
                },
                {
                    'title': 'MFE/MAE Ratio',
                    'targets': ['mfe_mae_ratio'],
                    'type': 'stat',
                },
            ],
        },
        'health_monitoring': {
            'title': 'Portfolio Health',
            'panels': [
                {
                    'title': 'Overall Health',
                    'targets': ['health_score'],
                    'type': 'graph',
                },
                {
                    'title': '4 Pillars',
                    'targets': [
                        'health_return_efficiency',
                        'health_downside_risk',
                        'health_structural_stability',
                        'health_behavioral',
                    ],
                    'type': 'graph',
                },
                {
                    'title': 'Risk Multiplier',
                    'targets': ['risk_multiplier'],
                    'type': 'graph',
                },
            ],
        },
        'regime_analysis': {
            'title': 'Market Regime Analysis',
            'panels': [
                {
                    'title': 'Regime Changes',
                    'targets': ['regime_change'],
                    'type': 'timeseries',
                },
                {
                    'title': 'Regime Volatility',
                    'targets': ['regime_volatility'],
                    'type': 'graph',
                },
                {
                    'title': 'Regime Trend Strength',
                    'targets': ['regime_trend'],
                    'type': 'graph',
                },
            ],
        },
    }
