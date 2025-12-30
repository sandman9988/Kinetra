"""
Grafana Integration for Kinetra Backtesting

Provides:
- Prometheus metrics exporter
- Pre-built dashboard templates (JSON export for Grafana)
- Drill-down support from portfolio to individual trades
- Data source classes for backtest metrics
"""

from .dashboards import DashboardGenerator
from .datasource import (
    BacktestStreamManager,
    GrafanaDataSource,
    StreamingUpdate,
)
from .metrics import BacktestMetrics, MetricsExporter

__all__ = [
    "GrafanaDataSource",
    "MetricsExporter",
    "BacktestMetrics",
    "DashboardGenerator",
    "BacktestStreamManager",
    "StreamingUpdate",
]
