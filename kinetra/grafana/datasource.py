"""
Grafana SimpleJSON Data Source Backend

Provides REST API endpoints compatible with Grafana's SimpleJSON plugin.
Supports drill-down from portfolio -> class -> instrument -> timeframe -> trade

Real-time streaming via Server-Sent Events (SSE) for live backtest updates.
"""

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional

from .metrics import BacktestMetrics, MetricsExporter


@dataclass
class QueryTarget:
    """Represents a Grafana query target."""

    target: str
    ref_id: str = "A"
    type: str = "timeserie"
    filters: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeRange:
    """Time range for queries."""

    from_time: datetime
    to_time: datetime


@dataclass
class StreamingUpdate:
    """A single update for streaming."""

    backtest_id: str
    update_type: str  # "trade", "equity", "metrics", "complete"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BacktestStreamManager:
    """
    Manages real-time streaming for live backtests.

    Supports multiple concurrent subscribers per backtest.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[queue.Queue]] = {}
        self._lock = threading.Lock()
        self._active_backtests: Dict[str, bool] = {}

    def start_backtest(self, backtest_id: str):
        """Register a new live backtest for streaming."""
        with self._lock:
            self._subscribers[backtest_id] = []
            self._active_backtests[backtest_id] = True

    def end_backtest(self, backtest_id: str):
        """Mark a backtest as complete and notify subscribers."""
        with self._lock:
            self._active_backtests[backtest_id] = False
            if backtest_id in self._subscribers:
                update = StreamingUpdate(
                    backtest_id=backtest_id,
                    update_type="complete",
                    data={"status": "completed"},
                )
                for q in self._subscribers[backtest_id]:
                    try:
                        q.put_nowait(update)
                    except queue.Full:
                        pass

    def subscribe(self, backtest_id: str) -> queue.Queue:
        """Subscribe to updates for a backtest."""
        with self._lock:
            if backtest_id not in self._subscribers:
                self._subscribers[backtest_id] = []
            q: queue.Queue[StreamingUpdate] = queue.Queue(maxsize=1000)
            self._subscribers[backtest_id].append(q)
            return q

    def unsubscribe(self, backtest_id: str, q: queue.Queue):
        """Unsubscribe from a backtest."""
        with self._lock:
            if backtest_id in self._subscribers:
                try:
                    self._subscribers[backtest_id].remove(q)
                except ValueError:
                    pass

    def publish(self, update: StreamingUpdate):
        """Publish an update to all subscribers of a backtest."""
        with self._lock:
            if update.backtest_id in self._subscribers:
                for q in self._subscribers[update.backtest_id]:
                    try:
                        q.put_nowait(update)
                    except queue.Full:
                        # Drop oldest if full
                        try:
                            q.get_nowait()
                            q.put_nowait(update)
                        except queue.Empty:
                            pass

    def is_active(self, backtest_id: str) -> bool:
        """Check if a backtest is still active."""
        return self._active_backtests.get(backtest_id, False)

    def get_active_backtests(self) -> List[str]:
        """Get list of active backtest IDs."""
        with self._lock:
            return [k for k, v in self._active_backtests.items() if v]


class GrafanaDataSource:
    """
    Grafana SimpleJSON compatible data source.

    Endpoints:
    - / : Health check
    - /search : List available metrics
    - /query : Return metric data
    - /annotations : Return annotations (trade entries/exits)
    - /tag-keys : Return available filter keys
    - /tag-values : Return filter values for a key
    - /stream/<backtest_id> : SSE endpoint for real-time updates

    Usage:
        datasource = GrafanaDataSource()
        datasource.add_backtest(metrics)

        # Start server
        start_grafana_server(datasource, port=8080)

        # For live streaming:
        datasource.start_live_backtest("bt_001")
        datasource.stream_trade(backtest_id, trade_data)
        datasource.stream_equity(backtest_id, equity, timestamp)
        datasource.end_live_backtest("bt_001")
    """

    def __init__(self):
        self.backtests: Dict[str, BacktestMetrics] = {}
        self.current_backtest_id: Optional[str] = None
        self._lock = threading.Lock()
        self.stream_manager = BacktestStreamManager()

    def add_backtest(self, metrics: BacktestMetrics) -> str:
        """
        Add a backtest result to the data source.

        Args:
            metrics: BacktestMetrics object

        Returns:
            Backtest ID
        """
        with self._lock:
            self.backtests[metrics.backtest_id] = metrics
            self.current_backtest_id = metrics.backtest_id
        return metrics.backtest_id

    def remove_backtest(self, backtest_id: str):
        """Remove a backtest from the data source."""
        with self._lock:
            if backtest_id in self.backtests:
                del self.backtests[backtest_id]

    def get_backtest(self, backtest_id: Optional[str] = None) -> Optional[BacktestMetrics]:
        """Get a backtest by ID, or the current one."""
        bt_id = backtest_id or self.current_backtest_id
        if bt_id is None:
            return None
        return self.backtests.get(bt_id)

    # === Live Streaming Methods ===

    def start_live_backtest(
        self,
        backtest_id: str,
        initial_capital: float = 10000.0,
        backtest_name: str = "",
    ) -> BacktestMetrics:
        """
        Start a new live streaming backtest.

        Args:
            backtest_id: Unique identifier for this backtest
            initial_capital: Starting capital
            backtest_name: Human-readable name

        Returns:
            Empty BacktestMetrics object to be filled during streaming
        """
        metrics = BacktestMetrics(
            backtest_id=backtest_id,
            backtest_name=backtest_name or backtest_id,
            initial_capital=initial_capital,
            start_time=datetime.now(),
        )

        with self._lock:
            self.backtests[backtest_id] = metrics
            self.current_backtest_id = backtest_id

        self.stream_manager.start_backtest(backtest_id)
        return metrics

    def end_live_backtest(self, backtest_id: str):
        """
        Mark a live backtest as complete.

        Args:
            backtest_id: Backtest to end
        """
        with self._lock:
            if backtest_id in self.backtests:
                self.backtests[backtest_id].end_time = datetime.now()

        self.stream_manager.end_backtest(backtest_id)

    def stream_trade(self, backtest_id: str, trade: Dict[str, Any]):
        """
        Stream a new trade update.

        Args:
            backtest_id: Backtest ID
            trade: Trade data dictionary
        """
        with self._lock:
            if backtest_id in self.backtests:
                self.backtests[backtest_id].trades.append(trade)

                # Update trade counts
                bt = self.backtests[backtest_id]
                bt.total_trades = len(bt.trades)
                bt.winning_trades = len([t for t in bt.trades if t.get("net_pnl", 0) > 0])
                bt.losing_trades = len([t for t in bt.trades if t.get("net_pnl", 0) < 0])
                if bt.total_trades > 0:
                    bt.win_rate = bt.winning_trades / bt.total_trades

        update = StreamingUpdate(
            backtest_id=backtest_id,
            update_type="trade",
            data=trade,
        )
        self.stream_manager.publish(update)

    def stream_equity(
        self,
        backtest_id: str,
        equity: float,
        timestamp: Optional[datetime] = None,
        margin: float = 0.0,
    ):
        """
        Stream an equity update.

        Args:
            backtest_id: Backtest ID
            equity: Current equity value
            timestamp: Time of update
            margin: Current margin usage
        """
        ts = timestamp or datetime.now()

        with self._lock:
            if backtest_id in self.backtests:
                bt = self.backtests[backtest_id]
                bt.equity_series.append(equity)
                bt.timestamps.append(ts)
                bt.final_equity = equity

                if margin > 0:
                    bt.margin_series.append(margin)

                # Calculate drawdown
                if bt.equity_series:
                    peak = max(bt.equity_series)
                    dd = (peak - equity) / peak if peak > 0 else 0
                    bt.drawdown_series.append(dd)
                    bt.max_drawdown_pct = max(bt.drawdown_series)

        update = StreamingUpdate(
            backtest_id=backtest_id,
            update_type="equity",
            data={
                "equity": equity,
                "timestamp": ts.isoformat(),
                "margin": margin,
            },
        )
        self.stream_manager.publish(update)

    def stream_metrics(self, backtest_id: str, metrics: Dict[str, float]):
        """
        Stream updated metrics (sharpe, sortino, etc).

        Args:
            backtest_id: Backtest ID
            metrics: Dictionary of metric updates
        """
        with self._lock:
            if backtest_id in self.backtests:
                bt = self.backtests[backtest_id]
                for key, value in metrics.items():
                    if hasattr(bt, key):
                        setattr(bt, key, value)

        update = StreamingUpdate(
            backtest_id=backtest_id,
            update_type="metrics",
            data=metrics,
        )
        self.stream_manager.publish(update)

    def get_stream(self, backtest_id: str) -> Generator[str, None, None]:
        """
        Get SSE stream for a backtest.

        Yields:
            Server-Sent Event formatted strings
        """
        q = self.stream_manager.subscribe(backtest_id)

        try:
            # Send initial state
            bt = self.get_backtest(backtest_id)
            if bt:
                yield f"event: init\ndata: {json.dumps({'backtest_id': backtest_id, 'total_trades': bt.total_trades, 'equity': bt.final_equity}, default=str)}\n\n"

            # Stream updates
            while True:
                try:
                    update = q.get(timeout=30)  # 30s heartbeat timeout

                    if update.update_type == "complete":
                        yield f"event: complete\ndata: {json.dumps(update.data, default=str)}\n\n"
                        break

                    yield f"event: {update.update_type}\ndata: {json.dumps(update.data, default=str)}\n\n"

                except queue.Empty:
                    # Send heartbeat
                    yield f"event: heartbeat\ndata: {json.dumps({'time': datetime.now().isoformat()})}\n\n"

                    # Check if still active
                    if not self.stream_manager.is_active(backtest_id):
                        yield f"event: complete\ndata: {json.dumps({'status': 'ended'})}\n\n"
                        break

        finally:
            self.stream_manager.unsubscribe(backtest_id, q)

    # === Grafana SimpleJSON Endpoints ===

    def health_check(self) -> Dict:
        """GET / - Health check."""
        return {"status": "ok", "backtests": len(self.backtests)}

    def search(self, target: str = "") -> List[str]:
        """
        POST /search - List available metrics.

        Returns list of queryable metric names.
        """
        metrics = [
            # Portfolio level
            "portfolio.equity",
            "portfolio.drawdown",
            "portfolio.margin",
            "portfolio.balance",
            "portfolio.return_pct",
            "portfolio.win_rate",
            "portfolio.sharpe",
            "portfolio.sortino",
            "portfolio.profit_factor",
            # Aggregate metrics
            "summary.total_trades",
            "summary.winning_trades",
            "summary.losing_trades",
            "summary.max_drawdown_pct",
            "summary.z_factor",
            # Risk metrics
            "risk.cvar_95",
            "risk.cvar_99",
            "risk.var_95",
            # Physics metrics (legacy)
            "physics.energy",
            "physics.efficiency",
            "physics.regime_performance",
            "physics.regime_timeline",
            "physics.energy_captured_pct",
            "physics.mfe_capture_pct",
            # Layer-1 Physics sensors
            "layer1.ke_pct",
            "layer1.re_m_pct",
            "layer1.zeta_pct",
            "layer1.hs_pct",
            "layer1.pe_pct",
            "layer1.eta_pct",
            "layer1.ke_pct_series",
            "layer1.re_m_pct_series",
            "layer1.zeta_pct_series",
            "layer1.hs_pct_series",
            # Regime metrics
            "regime.distribution",
            "regime.timeline",
            "regime.transition_matrix",
            "regime.age_frac",
            "regime.sharpe_by_regime",
            "regime.winrate_by_regime",
            # RL Training metrics
            "rl.episode",
            "rl.total_trades",
            "rl.win_rate",
            "rl.total_pnl",
            "rl.epsilon",
            "rl.loss",
            "rl.mfe_mae_ratio",
            "rl.mfe_captured",
            "rl.episode_reward",
            "rl.cumulative_pnl",
            "rl.loss_history",
            "rl.epsilon_history",
            "rl.feature_importance",
            "rl.action_distribution",
            "rl.mfe_history",
            "rl.mae_history",
            "rl.recent_trades",
            "rl.physics.ke_pct",
            "rl.physics.re_m_pct",
            "rl.physics.zeta_pct",
            "rl.physics.hs_pct",
            "rl.physics.bp",
            "rl.physics.regime",
            # Drill-down tables
            "table.instruments",
            "table.timeframes",
            "table.classes",
            "table.trades",
            "table.trades_by_instrument",
            "table.trades_by_regime",
            "table.regime_performance",
            "table.regime_sensors",
            # Distributions
            "distribution.pnl",
            "distribution.mfe",
            "distribution.mae",
            "distribution.holding_time",
            "distribution.pnl_by_regime",
        ]

        # Filter by target if provided
        if target:
            metrics = [m for m in metrics if target.lower() in m.lower()]

        return metrics

    def query(self, body: Dict) -> List[Dict]:
        """
        POST /query - Return metric data.

        Supports both timeseries and table responses.
        """
        results = []

        # Parse time range
        time_range = self._parse_time_range(body)

        # Parse targets
        targets = body.get("targets", [])

        for target_data in targets:
            target = target_data.get("target", "")
            target_type = target_data.get("type", "timeserie")

            # Get filters from adhoc filters or target data
            filters = {}
            for f in body.get("adhocFilters", []):
                filters[f.get("key")] = f.get("value")

            # Also check scopedVars for template variables
            scoped_vars = body.get("scopedVars", {})
            if "backtest_id" in scoped_vars:
                filters["backtest_id"] = scoped_vars["backtest_id"].get("value")

            result = self._query_target(target, target_type, time_range, filters)
            if result:
                results.append(result)

        return results

    def annotations(self, body: Dict) -> List[Dict]:
        """
        POST /annotations - Return trade entry/exit annotations.
        """
        annotations: List[Dict[str, Any]] = []

        bt = self.get_backtest()
        if not bt:
            return annotations

        for trade in bt.trades:
            # Entry annotation
            if trade.get("entry_time"):
                annotations.append(
                    {
                        "annotation": {"name": "Trade Entry"},
                        "time": self._to_ms(trade["entry_time"]),
                        "title": f"{trade['direction'].upper()} {trade['symbol']}",
                        "text": f"Entry @ {trade['entry_price']:.5f}, {trade['lots']} lots",
                        "tags": [trade["symbol"], trade["direction"], "entry"],
                    }
                )

            # Exit annotation
            if trade.get("exit_time"):
                pnl = trade.get("net_pnl", 0)
                color = "green" if pnl > 0 else "red"
                annotations.append(
                    {
                        "annotation": {"name": "Trade Exit"},
                        "time": self._to_ms(trade["exit_time"]),
                        "title": f"Close {trade['symbol']} {'WIN' if pnl > 0 else 'LOSS'}",
                        "text": f"Exit @ {trade['exit_price']:.5f}, PnL: ${pnl:.2f}",
                        "tags": [trade["symbol"], "exit", "win" if pnl > 0 else "loss"],
                    }
                )

        return annotations

    def tag_keys(self) -> List[Dict]:
        """
        POST /tag-keys - Return available filter keys.
        """
        return [
            {"type": "string", "text": "backtest_id"},
            {"type": "string", "text": "symbol"},
            {"type": "string", "text": "timeframe"},
            {"type": "string", "text": "instrument_class"},
            {"type": "string", "text": "direction"},
            {"type": "string", "text": "regime"},
        ]

    def tag_values(self, key: str) -> List[Dict]:
        """
        POST /tag-values - Return filter values for a key.
        """
        values = set()

        for bt in self.backtests.values():
            if key == "backtest_id":
                values.add(bt.backtest_id)
            elif key == "symbol":
                values.update(bt.instrument_metrics.keys())
            elif key == "timeframe":
                values.update(bt.timeframe_metrics.keys())
            elif key == "instrument_class":
                values.update(bt.class_metrics.keys())
            elif key == "direction":
                values.update(["long", "short"])
            elif key == "regime":
                values.update(["underdamped", "critical", "overdamped"])

        return [{"text": v} for v in sorted(values)]

    # === Query Handlers ===

    def _query_target(
        self,
        target: str,
        target_type: str,
        time_range: TimeRange,
        filters: Dict[str, str],
    ) -> Optional[Dict]:
        """Handle a single query target."""

        bt_id: Optional[str] = filters.get("backtest_id")
        bt = self.get_backtest(bt_id)
        if not bt:
            return None

        # Timeseries queries
        if target.startswith("portfolio."):
            return self._query_portfolio_series(target, bt, target_type)

        # Table queries
        if target.startswith("table."):
            return self._query_table(target, bt, filters)

        # Summary queries
        if target.startswith("summary."):
            return self._query_summary(target, bt)

        # Risk queries (CVaR, VaR)
        if target.startswith("risk."):
            return self._query_risk(target, bt)

        # Physics queries
        if target.startswith("physics."):
            return self._query_physics(target, bt)

        # Layer-1 physics queries
        if target.startswith("layer1."):
            return self._query_layer1(target, bt)

        # Regime queries
        if target.startswith("regime."):
            return self._query_regime(target, bt)

        # RL Training queries
        if target.startswith("rl."):
            return self._query_rl(target, bt)

        # Distribution queries
        if target.startswith("distribution."):
            return self._query_distribution(target, bt, filters)

        return None

    def _query_portfolio_series(
        self,
        target: str,
        bt: BacktestMetrics,
        target_type: str,
    ) -> Dict:
        """Query portfolio time series data."""

        metric = target.replace("portfolio.", "")

        if metric == "equity":
            datapoints = self._to_datapoints(bt.equity_series, bt.timestamps)
        elif metric == "drawdown":
            datapoints = self._to_datapoints(bt.drawdown_series, bt.timestamps)
        elif metric == "margin":
            datapoints = self._to_datapoints(bt.margin_series, bt.timestamps)
        elif metric == "balance":
            # Calculate running balance from trades
            balance = [bt.initial_capital]
            for trade in bt.trades:
                if trade.get("net_pnl"):
                    balance.append(balance[-1] + trade["net_pnl"])
            datapoints = self._to_datapoints(balance, bt.timestamps[: len(balance)])
        else:
            datapoints = []

        return {
            "target": target,
            "datapoints": datapoints,
        }

    def _query_table(
        self,
        target: str,
        bt: BacktestMetrics,
        filters: Dict,
    ) -> Dict:
        """Query table data for drill-down."""

        table_type = target.replace("table.", "")

        if table_type == "instruments":
            return self._instruments_table(bt)
        elif table_type == "timeframes":
            return self._timeframes_table(bt)
        elif table_type == "classes":
            return self._classes_table(bt)
        elif table_type == "trades":
            return self._trades_table(bt, filters)
        elif table_type == "trades_by_instrument":
            return self._trades_by_instrument_table(bt, filters)
        elif table_type == "trades_by_regime":
            return self._trades_by_regime_table(bt)

        return {"columns": [], "rows": [], "type": "table"}

    def _instruments_table(self, bt: BacktestMetrics) -> Dict:
        """Generate instruments drill-down table."""
        columns = [
            {"text": "Symbol", "type": "string"},
            {"text": "Trades", "type": "number"},
            {"text": "Win Rate", "type": "number"},
            {"text": "Net Profit", "type": "number"},
            {"text": "Profit Factor", "type": "number"},
            {"text": "Avg Trade", "type": "number"},
            {"text": "Largest Win", "type": "number"},
            {"text": "Largest Loss", "type": "number"},
        ]

        rows = []
        for symbol, metrics in bt.instrument_metrics.items():
            rows.append(
                [
                    symbol,
                    metrics.get("total_trades", 0),
                    round(metrics.get("win_rate", 0) * 100, 2),
                    round(metrics.get("net_profit", 0), 2),
                    round(metrics.get("profit_factor", 0), 2),
                    round(metrics.get("avg_trade", 0), 2),
                    round(metrics.get("largest_win", 0), 2),
                    round(metrics.get("largest_loss", 0), 2),
                ]
            )

        return {"columns": columns, "rows": rows, "type": "table"}

    def _timeframes_table(self, bt: BacktestMetrics) -> Dict:
        """Generate timeframes drill-down table."""
        columns = [
            {"text": "Timeframe", "type": "string"},
            {"text": "Trades", "type": "number"},
            {"text": "Win Rate", "type": "number"},
            {"text": "Net Profit", "type": "number"},
            {"text": "Profit Factor", "type": "number"},
        ]

        rows = []
        for tf, metrics in bt.timeframe_metrics.items():
            rows.append(
                [
                    tf,
                    metrics.get("total_trades", 0),
                    round(metrics.get("win_rate", 0) * 100, 2),
                    round(metrics.get("net_profit", 0), 2),
                    round(metrics.get("profit_factor", 0), 2),
                ]
            )

        return {"columns": columns, "rows": rows, "type": "table"}

    def _classes_table(self, bt: BacktestMetrics) -> Dict:
        """Generate instrument classes drill-down table."""
        columns = [
            {"text": "Class", "type": "string"},
            {"text": "Trades", "type": "number"},
            {"text": "Win Rate", "type": "number"},
            {"text": "Net Profit", "type": "number"},
            {"text": "Profit Factor", "type": "number"},
        ]

        rows = []
        for cls, metrics in bt.class_metrics.items():
            rows.append(
                [
                    cls.upper(),
                    metrics.get("total_trades", 0),
                    round(metrics.get("win_rate", 0) * 100, 2),
                    round(metrics.get("net_profit", 0), 2),
                    round(metrics.get("profit_factor", 0), 2),
                ]
            )

        return {"columns": columns, "rows": rows, "type": "table"}

    def _trades_table(self, bt: BacktestMetrics, filters: Dict) -> Dict:
        """Generate trades table with filtering."""
        columns = [
            {"text": "ID", "type": "number"},
            {"text": "Symbol", "type": "string"},
            {"text": "Direction", "type": "string"},
            {"text": "Entry Time", "type": "time"},
            {"text": "Exit Time", "type": "time"},
            {"text": "Entry Price", "type": "number"},
            {"text": "Exit Price", "type": "number"},
            {"text": "Lots", "type": "number"},
            {"text": "Gross PnL", "type": "number"},
            {"text": "Net PnL", "type": "number"},
            {"text": "Costs", "type": "number"},
            {"text": "MFE", "type": "number"},
            {"text": "MAE", "type": "number"},
            {"text": "Regime", "type": "string"},
            {"text": "Energy", "type": "number"},
        ]

        rows = []
        for trade in bt.trades:
            # Apply filters
            if filters.get("symbol") and trade.get("symbol") != filters["symbol"]:
                continue
            if filters.get("direction") and trade.get("direction") != filters["direction"]:
                continue
            if filters.get("regime") and trade.get("regime_at_entry") != filters["regime"]:
                continue

            rows.append(
                [
                    trade.get("trade_id"),
                    trade.get("symbol"),
                    trade.get("direction", "").upper(),
                    trade.get("entry_time"),
                    trade.get("exit_time"),
                    round(trade.get("entry_price", 0), 5),
                    round(trade.get("exit_price", 0), 5) if trade.get("exit_price") else None,
                    trade.get("lots"),
                    round(trade.get("gross_pnl", 0), 2),
                    round(trade.get("net_pnl", 0), 2),
                    round(trade.get("total_cost", 0), 2),
                    round(trade.get("mfe", 0), 5),
                    round(trade.get("mae", 0), 5),
                    trade.get("regime_at_entry", ""),
                    round(trade.get("energy_at_entry", 0), 4),
                ]
            )

        return {"columns": columns, "rows": rows, "type": "table"}

    def _trades_by_instrument_table(self, bt: BacktestMetrics, filters: Dict) -> Dict:
        """Generate trades grouped by instrument."""
        symbol = filters.get("symbol")
        if not symbol:
            return {"columns": [], "rows": [], "type": "table"}

        # Filter trades for this symbol
        filtered_metrics = BacktestMetrics(
            backtest_id=bt.backtest_id, trades=[t for t in bt.trades if t.get("symbol") == symbol]
        )

        return self._trades_table(filtered_metrics, {})

    def _trades_by_regime_table(self, bt: BacktestMetrics) -> Dict:
        """Generate trades summary by regime."""
        columns = [
            {"text": "Regime", "type": "string"},
            {"text": "Trades", "type": "number"},
            {"text": "Winners", "type": "number"},
            {"text": "Losers", "type": "number"},
            {"text": "Win Rate", "type": "number"},
            {"text": "Net PnL", "type": "number"},
            {"text": "Avg Energy", "type": "number"},
        ]

        regimes: Dict[str, List[Dict[str, Any]]] = {
            "underdamped": [],
            "critical": [],
            "overdamped": [],
            "unknown": [],
        }

        for trade in bt.trades:
            regime = trade.get("regime_at_entry", "unknown").lower()
            if regime not in regimes:
                regime = "unknown"
            regimes[regime].append(trade)

        rows = []
        for regime, trades in regimes.items():
            if not trades:
                continue

            winners = len([t for t in trades if t.get("net_pnl", 0) > 0])
            losers = len([t for t in trades if t.get("net_pnl", 0) < 0])
            net_pnl = sum(t.get("net_pnl", 0) for t in trades)
            avg_energy = sum(t.get("energy_at_entry", 0) for t in trades) / len(trades)

            rows.append(
                [
                    regime.upper(),
                    len(trades),
                    winners,
                    losers,
                    round(winners / len(trades) * 100, 2) if trades else 0,
                    round(net_pnl, 2),
                    round(avg_energy, 4),
                ]
            )

        return {"columns": columns, "rows": rows, "type": "table"}

    def _query_summary(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query summary metrics."""
        metric = target.replace("summary.", "")

        value: float = 0.0
        if metric == "total_trades":
            value = float(bt.total_trades)
        elif metric == "winning_trades":
            value = float(bt.winning_trades)
        elif metric == "losing_trades":
            value = float(bt.losing_trades)
        elif metric == "max_drawdown_pct":
            value = bt.max_drawdown_pct
        elif metric == "z_factor":
            value = bt.z_factor

        return {
            "target": target,
            "datapoints": [[value, int(datetime.now().timestamp() * 1000)]],
        }

    def _query_physics(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query physics metrics."""
        metric = target.replace("physics.", "")

        if metric == "regime_performance":
            # Return as table
            return self._trades_by_regime_table(bt)

        if metric == "regime_timeline":
            # Return regime time series
            return {
                "target": target,
                "datapoints": self._to_datapoints_string(bt.regime_series, bt.timestamps),
            }

        value: float = 0.0
        if metric == "energy":
            value = bt.avg_energy_at_entry
        elif metric == "efficiency":
            value = bt.energy_capture_efficiency
        elif metric == "energy_captured_pct":
            value = bt.energy_capture_efficiency
        elif metric == "mfe_capture_pct":
            value = getattr(bt, "mfe_capture_pct", 0.0)

        return {
            "target": target,
            "datapoints": [[value, int(datetime.now().timestamp() * 1000)]],
        }

    def _query_risk(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query risk metrics (CVaR, VaR)."""
        metric = target.replace("risk.", "")

        value: float = 0.0
        if metric == "cvar_95":
            value = bt.cvar_95
        elif metric == "cvar_99":
            value = bt.cvar_99
        elif metric == "var_95":
            value = bt.var_95

        return {
            "target": target,
            "datapoints": [[value, int(datetime.now().timestamp() * 1000)]],
        }

    def _query_layer1(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query Layer-1 physics sensor metrics."""
        metric = target.replace("layer1.", "")

        # Time series queries
        if metric == "ke_pct_series":
            return {
                "target": target,
                "datapoints": self._to_datapoints(bt.ke_pct_series, bt.timestamps),
            }
        elif metric == "re_m_pct_series":
            return {
                "target": target,
                "datapoints": self._to_datapoints(bt.re_m_pct_series, bt.timestamps),
            }
        elif metric == "zeta_pct_series":
            return {
                "target": target,
                "datapoints": self._to_datapoints(bt.zeta_pct_series, bt.timestamps),
            }
        elif metric == "hs_pct_series":
            return {
                "target": target,
                "datapoints": self._to_datapoints(getattr(bt, "hs_pct_series", []), bt.timestamps),
            }

        # Current value queries
        value: float = 0.0
        if metric == "ke_pct":
            value = bt.avg_ke_pct
        elif metric == "re_m_pct":
            value = bt.avg_re_m_pct
        elif metric == "zeta_pct":
            value = bt.avg_zeta_pct
        elif metric == "hs_pct":
            value = bt.avg_hs_pct
        elif metric == "pe_pct":
            value = bt.avg_pe_pct
        elif metric == "eta_pct":
            value = bt.avg_eta_pct

        return {
            "target": target,
            "datapoints": [[value, int(datetime.now().timestamp() * 1000)]],
        }

    def _query_regime(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query regime-related metrics."""
        metric = target.replace("regime.", "")

        if metric == "distribution":
            # Regime distribution bar chart
            regime_counts: Dict[str, int] = {}
            for regime in bt.regime_series:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            columns = [
                {"text": "Regime", "type": "string"},
                {"text": "Count", "type": "number"},
            ]
            rows = [[r, c] for r, c in sorted(regime_counts.items())]
            return {"columns": columns, "rows": rows, "type": "table"}

        elif metric == "timeline":
            return {
                "target": target,
                "datapoints": self._to_datapoints_string(bt.regime_series, bt.timestamps),
            }

        elif metric == "sharpe_by_regime" or metric == "winrate_by_regime":
            # Get from regime_metrics
            columns = [
                {"text": "Regime", "type": "string"},
                {"text": "Value", "type": "number"},
            ]
            rows = []
            for regime, metrics in bt.regime_metrics.items():
                if metric == "sharpe_by_regime":
                    val = metrics.get("sharpe", 0)
                else:
                    val = metrics.get("win_rate", 0) * 100
                rows.append([regime, val])
            return {"columns": columns, "rows": rows, "type": "table"}

        return {"target": target, "datapoints": []}

    def _query_rl(self, target: str, bt: BacktestMetrics) -> Dict:
        """Query RL training metrics (from streaming data)."""
        # RL metrics come from live streaming - return last known values
        # These are populated by stream_metrics() calls during training
        metric = target.replace("rl.", "")

        # For physics sub-metrics
        if metric.startswith("physics."):
            physics_metric = metric.replace("physics.", "")
            # Return last known physics state
            if physics_metric == "regime" and bt.regime_series:
                return {
                    "target": target,
                    "datapoints": [[bt.regime_series[-1], int(datetime.now().timestamp() * 1000)]],
                }
            elif physics_metric == "ke_pct" and bt.ke_pct_series:
                return {
                    "target": target,
                    "datapoints": [[bt.ke_pct_series[-1], int(datetime.now().timestamp() * 1000)]],
                }
            # Add other physics metrics as needed
            return {"target": target, "datapoints": [[0.5, int(datetime.now().timestamp() * 1000)]]}

        # Basic RL stats
        value: float = 0.0
        if metric == "episode":
            value = float(getattr(bt, "rl_episode", 0))
        elif metric == "total_trades":
            value = float(bt.total_trades)
        elif metric == "win_rate":
            value = bt.win_rate * 100
        elif metric == "total_pnl":
            value = bt.total_return_pct
        elif metric == "epsilon":
            value = getattr(bt, "rl_epsilon", 1.0)
        elif metric == "loss":
            value = float(getattr(bt, "rl_loss", 0.0))
        elif metric == "mfe_mae_ratio":
            value = getattr(bt, "rl_mfe_mae_ratio", 1.0)
        elif metric == "mfe_captured":
            value = float(getattr(bt, "rl_mfe_captured", 0.0))

        # Time series metrics
        if metric in ["episode_reward", "cumulative_pnl", "loss_history", "epsilon_history"]:
            series = getattr(bt, f"rl_{metric}", [])
            ts = bt.timestamps[: len(series)] if bt.timestamps else []
            return {
                "target": target,
                "datapoints": self._to_datapoints(series, ts),
            }

        # Table metrics
        if metric == "feature_importance":
            importance = getattr(bt, "rl_feature_importance", {})
            columns = [
                {"text": "Feature", "type": "string"},
                {"text": "Importance", "type": "number"},
            ]
            rows = [[f, v] for f, v in sorted(importance.items(), key=lambda x: -x[1])]
            return {"columns": columns, "rows": rows, "type": "table"}

        if metric == "action_distribution":
            actions = getattr(bt, "rl_action_distribution", {"flat": 0, "long": 0, "short": 0})
            columns = [
                {"text": "Action", "type": "string"},
                {"text": "Count", "type": "number"},
            ]
            rows = [[a, c] for a, c in actions.items()]
            return {"columns": columns, "rows": rows, "type": "table"}

        if metric == "recent_trades":
            # Return last 20 trades
            recent = bt.trades[-20:] if bt.trades else []
            columns = [
                {"text": "Time", "type": "time"},
                {"text": "Action", "type": "string"},
                {"text": "Entry", "type": "number"},
                {"text": "Exit", "type": "number"},
                {"text": "PnL", "type": "number"},
                {"text": "MFE", "type": "number"},
                {"text": "MAE", "type": "number"},
                {"text": "Regime", "type": "string"},
            ]
            rows = []
            for t in recent:
                rows.append(
                    [
                        t.get("entry_time"),
                        t.get("direction", ""),
                        t.get("entry_price", 0),
                        t.get("exit_price", 0),
                        t.get("net_pnl", 0),
                        t.get("mfe", 0),
                        t.get("mae", 0),
                        t.get("regime_at_entry", ""),
                    ]
                )
            return {"columns": columns, "rows": rows, "type": "table"}

        return {
            "target": target,
            "datapoints": [[value, int(datetime.now().timestamp() * 1000)]],
        }

    def _to_datapoints_string(
        self,
        values: List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[List]:
        """Convert string values to Grafana datapoints for state timeline."""
        if not values:
            return []

        if timestamps and len(timestamps) >= len(values):
            return [[v, self._to_ms(timestamps[i])] for i, v in enumerate(values)]
        else:
            now = datetime.now()
            return [
                [v, int((now - timedelta(minutes=len(values) - i)).timestamp() * 1000)]
                for i, v in enumerate(values)
            ]

    def _query_distribution(
        self,
        target: str,
        bt: BacktestMetrics,
        filters: Dict,
    ) -> Dict:
        """Query distribution data for histograms."""
        dist_type = target.replace("distribution.", "")

        values = []
        for trade in bt.trades:
            if dist_type == "pnl":
                values.append(trade.get("net_pnl", 0))
            elif dist_type == "mfe":
                values.append(trade.get("mfe", 0))
            elif dist_type == "mae":
                values.append(trade.get("mae", 0))

        # Create histogram buckets
        if values:
            import numpy as np

            hist, bins = np.histogram(values, bins=20)

            columns = [
                {"text": "Bucket", "type": "string"},
                {"text": "Count", "type": "number"},
            ]

            rows = []
            for i in range(len(hist)):
                bucket_label = f"{bins[i]:.2f} - {bins[i + 1]:.2f}"
                rows.append([bucket_label, int(hist[i])])

            return {"columns": columns, "rows": rows, "type": "table"}

        return {"columns": [], "rows": [], "type": "table"}

    # === Helpers ===

    def _parse_time_range(self, body: Dict) -> TimeRange:
        """Parse time range from request body."""
        range_data = body.get("range", {})

        from_str = range_data.get("from", "")
        to_str = range_data.get("to", "")

        try:
            from_time = datetime.fromisoformat(from_str.replace("Z", "+00:00"))
            to_time = datetime.fromisoformat(to_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            from_time = datetime.now() - timedelta(days=30)
            to_time = datetime.now()

        return TimeRange(from_time=from_time, to_time=to_time)

    def _to_datapoints(
        self,
        values: List[float],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[List]:
        """Convert values to Grafana datapoints format [[value, timestamp_ms], ...]."""
        if not values:
            return []

        if timestamps and len(timestamps) >= len(values):
            return [[v, self._to_ms(timestamps[i])] for i, v in enumerate(values)]
        else:
            # Generate synthetic timestamps
            now = datetime.now()
            return [
                [v, int((now - timedelta(minutes=len(values) - i)).timestamp() * 1000)]
                for i, v in enumerate(values)
            ]

    def _to_ms(self, dt: Any) -> int:
        """Convert datetime to milliseconds timestamp."""
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            except ValueError:
                return 0

        if isinstance(dt, datetime):
            return int(dt.timestamp() * 1000)

        return 0
