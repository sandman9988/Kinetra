"""
Metrics Exporter for Prometheus/Grafana

Exports backtest metrics in Prometheus format for scraping.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..backtest_engine import BacktestResult, Trade
from ..portfolio_backtest import InstrumentClass, PortfolioBacktestResult
from ..reporting import PortfolioReport, ReportingEngine


@dataclass
class BacktestMetrics:
    """Container for all backtest metrics in Prometheus-compatible format."""

    # Metadata
    backtest_id: str = ""
    backtest_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Portfolio metrics
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0  # Added: worst 1% expected loss

    # Physics metrics (legacy)
    avg_energy_at_entry: float = 0.0
    energy_capture_efficiency: float = 0.0
    z_factor: float = 0.0

    # Layer-1 Physics sensors (new)
    avg_ke_pct: float = 0.0  # Kinetic Energy percentile
    avg_re_m_pct: float = 0.0  # Reynolds number percentile
    avg_zeta_pct: float = 0.0  # Damping percentile
    avg_hs_pct: float = 0.0  # Entropy percentile
    avg_pe_pct: float = 0.0  # Potential Energy percentile
    avg_eta_pct: float = 0.0  # Efficiency percentile

    # Per-instrument metrics
    instrument_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-timeframe metrics
    timeframe_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-regime metrics (new - from GMM clustering)
    regime_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Time series data
    equity_series: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    margin_series: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # Layer-1 physics time series (new)
    ke_pct_series: List[float] = field(default_factory=list)
    re_m_pct_series: List[float] = field(default_factory=list)
    zeta_pct_series: List[float] = field(default_factory=list)
    regime_series: List[str] = field(default_factory=list)

    # Trade-level data for drill-down
    trades: List[Dict[str, Any]] = field(default_factory=list)


class MetricsExporter:
    """
    Exports backtest results to various formats for Grafana consumption.

    Supports:
    - Prometheus text format
    - JSON for SimpleJSON datasource
    - InfluxDB line protocol
    """

    def __init__(self, namespace: str = "kinetra"):
        self.namespace = namespace
        self.reporting = ReportingEngine()

    def extract_metrics(
        self,
        result: PortfolioBacktestResult,
        backtest_id: str = "",
        backtest_name: str = "",
    ) -> BacktestMetrics:
        """
        Extract all metrics from a backtest result.

        Args:
            result: Portfolio backtest result
            backtest_id: Unique identifier for this backtest
            backtest_name: Human-readable name

        Returns:
            BacktestMetrics object with all data
        """
        report = self.reporting.generate_report(result)

        metrics = BacktestMetrics(
            backtest_id=backtest_id or f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            backtest_name=backtest_name or "Backtest",
            initial_capital=report.initial_capital,
            final_equity=report.final_capital,
        )

        # Calculate return
        if report.initial_capital > 0:
            metrics.total_return_pct = (
                (report.final_capital - report.initial_capital) / report.initial_capital
            ) * 100

        # Trade analysis
        if report.trade_analysis:
            ta = report.trade_analysis
            metrics.total_trades = ta.total_trades
            metrics.winning_trades = ta.winning_trades
            metrics.losing_trades = ta.losing_trades
            metrics.win_rate = ta.win_rate
            metrics.profit_factor = ta.profit_factor

        # Risk metrics
        if report.risk_metrics:
            rm = report.risk_metrics
            metrics.sharpe_ratio = rm.sharpe_ratio
            metrics.sortino_ratio = rm.sortino_ratio
            metrics.calmar_ratio = rm.calmar_ratio
            metrics.max_drawdown = rm.max_drawdown
            metrics.max_drawdown_pct = rm.max_drawdown_pct
            metrics.var_95 = rm.var_95
            metrics.cvar_95 = rm.cvar_95
            if hasattr(rm, "cvar_99"):
                metrics.cvar_99 = rm.cvar_99

        # Physics metrics (legacy)
        if report.physics_metrics:
            pm = report.physics_metrics
            metrics.avg_energy_at_entry = pm.avg_energy_at_entry
            metrics.energy_capture_efficiency = pm.energy_capture_efficiency
            metrics.z_factor = pm.z_factor

        # Layer-1 physics metrics (from trades)
        if result.trades:
            ke_vals = [
                t.ke_pct_at_entry
                for t in result.trades
                if hasattr(t, "ke_pct_at_entry") and t.ke_pct_at_entry is not None
            ]
            re_vals = [
                t.re_m_pct_at_entry
                for t in result.trades
                if hasattr(t, "re_m_pct_at_entry") and t.re_m_pct_at_entry is not None
            ]
            zeta_vals = [
                t.zeta_pct_at_entry
                for t in result.trades
                if hasattr(t, "zeta_pct_at_entry") and t.zeta_pct_at_entry is not None
            ]
            hs_vals = [
                t.hs_pct_at_entry
                for t in result.trades
                if hasattr(t, "hs_pct_at_entry") and t.hs_pct_at_entry is not None
            ]
            pe_vals = [
                t.pe_pct_at_entry
                for t in result.trades
                if hasattr(t, "pe_pct_at_entry") and t.pe_pct_at_entry is not None
            ]
            eta_vals = [
                t.eta_pct_at_entry
                for t in result.trades
                if hasattr(t, "eta_pct_at_entry") and t.eta_pct_at_entry is not None
            ]

            if ke_vals:
                metrics.avg_ke_pct = sum(ke_vals) / len(ke_vals)
            if re_vals:
                metrics.avg_re_m_pct = sum(re_vals) / len(re_vals)
            if zeta_vals:
                metrics.avg_zeta_pct = sum(zeta_vals) / len(zeta_vals)
            if hs_vals:
                metrics.avg_hs_pct = sum(hs_vals) / len(hs_vals)
            if pe_vals:
                metrics.avg_pe_pct = sum(pe_vals) / len(pe_vals)
            if eta_vals:
                metrics.avg_eta_pct = sum(eta_vals) / len(eta_vals)

            # Per-regime metrics
            regime_trades: Dict[str, list] = {}
            for t in result.trades:
                regime = getattr(t, "regime_at_entry", "unknown") or "unknown"
                if regime not in regime_trades:
                    regime_trades[regime] = []
                regime_trades[regime].append(t)

            for regime, trades in regime_trades.items():
                wins = len([t for t in trades if t.net_pnl > 0])
                total_pnl = sum(t.net_pnl for t in trades)
                metrics.regime_metrics[regime] = {
                    "total_trades": len(trades),
                    "win_rate": wins / len(trades) if trades else 0,
                    "net_profit": total_pnl,
                    "avg_pnl": total_pnl / len(trades) if trades else 0,
                }

        # Per-instrument metrics
        for symbol, inst_report in report.instrument_reports.items():
            if inst_report.trade_analysis:
                ta = inst_report.trade_analysis
                metrics.instrument_metrics[symbol] = {
                    "total_trades": ta.total_trades,
                    "win_rate": ta.win_rate,
                    "net_profit": ta.net_profit,
                    "profit_factor": ta.profit_factor,
                    "avg_trade": ta.avg_trade,
                    "largest_win": ta.largest_win,
                    "largest_loss": ta.largest_loss,
                }

        # Per-timeframe metrics
        for tf, tf_analysis in report.timeframe_reports.items():
            metrics.timeframe_metrics[tf] = {
                "total_trades": tf_analysis.total_trades,
                "win_rate": tf_analysis.win_rate,
                "net_profit": tf_analysis.net_profit,
                "profit_factor": tf_analysis.profit_factor,
            }

        # Per-class metrics
        for cls, cls_analysis in report.class_reports.items():
            metrics.class_metrics[cls.value] = {
                "total_trades": cls_analysis.total_trades,
                "win_rate": cls_analysis.win_rate,
                "net_profit": cls_analysis.net_profit,
                "profit_factor": cls_analysis.profit_factor,
            }

        # Time series
        if report.equity_curve is not None:
            metrics.equity_series = report.equity_curve.tolist()
        if report.drawdown_curve is not None:
            metrics.drawdown_series = report.drawdown_curve.tolist()

        # Trade details for drill-down
        for t in result.trades:
            trade_dict = {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "direction": t.direction.value,
                "lots": t.lots,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "total_cost": t.total_cost,
                "mfe": t.mfe,
                "mae": t.mae,
                "energy_at_entry": t.energy_at_entry,
                "regime_at_entry": t.regime_at_entry,
            }

            # Add portfolio trade fields if available
            if hasattr(t, "timeframe"):
                trade_dict["timeframe"] = t.timeframe
            if hasattr(t, "instrument_class"):
                trade_dict["instrument_class"] = t.instrument_class.value

            metrics.trades.append(trade_dict)

        return metrics

    def to_prometheus(self, metrics: BacktestMetrics) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format
        """
        lines = []
        ns = self.namespace
        bt_id = metrics.backtest_id

        # Helper to add metric
        def add_metric(
            name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = ""
        ):
            if help_text:
                lines.append(f"# HELP {ns}_{name} {help_text}")
                lines.append(f"# TYPE {ns}_{name} gauge")

            label_str = ""
            if labels:
                label_parts = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = "{" + ",".join(label_parts) + "}"

            lines.append(f"{ns}_{name}{label_str} {value}")

        base_labels = {"backtest_id": bt_id, "backtest_name": metrics.backtest_name}

        # Portfolio metrics
        add_metric("initial_capital", metrics.initial_capital, base_labels, "Initial capital")
        add_metric("final_equity", metrics.final_equity, base_labels, "Final equity")
        add_metric(
            "total_return_pct", metrics.total_return_pct, base_labels, "Total return percentage"
        )
        add_metric("total_trades", metrics.total_trades, base_labels, "Total number of trades")
        add_metric(
            "winning_trades", metrics.winning_trades, base_labels, "Number of winning trades"
        )
        add_metric("losing_trades", metrics.losing_trades, base_labels, "Number of losing trades")
        add_metric("win_rate", metrics.win_rate, base_labels, "Win rate")
        add_metric("profit_factor", metrics.profit_factor, base_labels, "Profit factor")

        # Risk metrics
        add_metric("sharpe_ratio", metrics.sharpe_ratio, base_labels, "Sharpe ratio")
        add_metric("sortino_ratio", metrics.sortino_ratio, base_labels, "Sortino ratio")
        add_metric("calmar_ratio", metrics.calmar_ratio, base_labels, "Calmar ratio")
        add_metric("max_drawdown", metrics.max_drawdown, base_labels, "Maximum drawdown")
        add_metric(
            "max_drawdown_pct", metrics.max_drawdown_pct, base_labels, "Maximum drawdown percentage"
        )
        add_metric("var_95", metrics.var_95, base_labels, "Value at Risk 95%")
        add_metric("cvar_95", metrics.cvar_95, base_labels, "Conditional VaR 95%")

        # Physics metrics (legacy)
        add_metric(
            "avg_energy_at_entry",
            metrics.avg_energy_at_entry,
            base_labels,
            "Average energy at entry",
        )
        add_metric(
            "energy_capture_efficiency",
            metrics.energy_capture_efficiency,
            base_labels,
            "Energy capture efficiency",
        )
        add_metric("z_factor", metrics.z_factor, base_labels, "Z-factor statistical edge")

        # Layer-1 Physics sensors
        add_metric(
            "avg_ke_pct", metrics.avg_ke_pct, base_labels, "Avg kinetic energy percentile at entry"
        )
        add_metric(
            "avg_re_m_pct",
            metrics.avg_re_m_pct,
            base_labels,
            "Avg Reynolds number percentile at entry",
        )
        add_metric(
            "avg_zeta_pct", metrics.avg_zeta_pct, base_labels, "Avg damping percentile at entry"
        )
        add_metric("avg_hs_pct", metrics.avg_hs_pct, base_labels, "Avg entropy percentile at entry")
        add_metric(
            "avg_pe_pct",
            metrics.avg_pe_pct,
            base_labels,
            "Avg potential energy percentile at entry",
        )
        add_metric(
            "avg_eta_pct", metrics.avg_eta_pct, base_labels, "Avg efficiency percentile at entry"
        )
        add_metric("cvar_99", metrics.cvar_99, base_labels, "Conditional VaR 99%")

        # Per-regime metrics
        for regime, regime_metrics in metrics.regime_metrics.items():
            regime_labels = {**base_labels, "regime": regime}
            for metric_name, value in regime_metrics.items():
                add_metric(f"regime_{metric_name}", value, regime_labels)

        # Per-instrument metrics
        for symbol, inst_metrics in metrics.instrument_metrics.items():
            inst_labels = {**base_labels, "symbol": symbol}
            for metric_name, value in inst_metrics.items():
                add_metric(f"instrument_{metric_name}", value, inst_labels)

        # Per-timeframe metrics
        for tf, tf_metrics in metrics.timeframe_metrics.items():
            tf_labels = {**base_labels, "timeframe": tf}
            for metric_name, value in tf_metrics.items():
                add_metric(f"timeframe_{metric_name}", value, tf_labels)

        # Per-class metrics
        for cls, cls_metrics in metrics.class_metrics.items():
            cls_labels = {**base_labels, "instrument_class": cls}
            for metric_name, value in cls_metrics.items():
                add_metric(f"class_{metric_name}", value, cls_labels)

        return "\n".join(lines)

    def to_influxdb(self, metrics: BacktestMetrics) -> str:
        """
        Export metrics in InfluxDB line protocol format.

        Returns:
            String in InfluxDB line protocol
        """
        lines = []
        timestamp = int(datetime.now().timestamp() * 1e9)  # nanoseconds

        # Portfolio metrics
        tags = f"backtest_id={metrics.backtest_id},backtest_name={metrics.backtest_name.replace(' ', '_')}"

        fields = [
            f"initial_capital={metrics.initial_capital}",
            f"final_equity={metrics.final_equity}",
            f"total_return_pct={metrics.total_return_pct}",
            f"total_trades={metrics.total_trades}i",
            f"win_rate={metrics.win_rate}",
            f"sharpe_ratio={metrics.sharpe_ratio}",
            f"max_drawdown_pct={metrics.max_drawdown_pct}",
        ]

        lines.append(f"{self.namespace}_portfolio,{tags} {','.join(fields)} {timestamp}")

        # Per-instrument
        for symbol, inst_metrics in metrics.instrument_metrics.items():
            inst_tags = f"{tags},symbol={symbol}"
            inst_fields = [f"{k}={v}" for k, v in inst_metrics.items()]
            lines.append(
                f"{self.namespace}_instrument,{inst_tags} {','.join(inst_fields)} {timestamp}"
            )

        return "\n".join(lines)

    def to_json(self, metrics: BacktestMetrics) -> str:
        """
        Export metrics as JSON for Grafana SimpleJSON datasource.

        Returns:
            JSON string
        """
        return json.dumps(
            {
                "backtest_id": metrics.backtest_id,
                "backtest_name": metrics.backtest_name,
                "portfolio": {
                    "initial_capital": metrics.initial_capital,
                    "final_equity": metrics.final_equity,
                    "total_return_pct": metrics.total_return_pct,
                    "total_trades": metrics.total_trades,
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                },
                "risk": {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "calmar_ratio": metrics.calmar_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "var_95": metrics.var_95,
                    "cvar_95": metrics.cvar_95,
                    "cvar_99": metrics.cvar_99,
                },
                "physics": {
                    "avg_energy_at_entry": metrics.avg_energy_at_entry,
                    "energy_capture_efficiency": metrics.energy_capture_efficiency,
                    "z_factor": metrics.z_factor,
                },
                "layer1_physics": {
                    "avg_ke_pct": metrics.avg_ke_pct,
                    "avg_re_m_pct": metrics.avg_re_m_pct,
                    "avg_zeta_pct": metrics.avg_zeta_pct,
                    "avg_hs_pct": metrics.avg_hs_pct,
                    "avg_pe_pct": metrics.avg_pe_pct,
                    "avg_eta_pct": metrics.avg_eta_pct,
                },
                "instruments": metrics.instrument_metrics,
                "timeframes": metrics.timeframe_metrics,
                "classes": metrics.class_metrics,
                "regimes": metrics.regime_metrics,
                "equity_series": metrics.equity_series,
                "drawdown_series": metrics.drawdown_series,
                "ke_pct_series": metrics.ke_pct_series,
                "regime_series": metrics.regime_series,
                "trades": metrics.trades,
            },
            indent=2,
            default=str,
        )
