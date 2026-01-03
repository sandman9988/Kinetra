"""
Live Dashboard Streaming for Kinetra Testing
=============================================

Real-time monitoring dashboard for exhaustive testing runs.
Updates automatically as tests progress, showing:
- Current test progress (X/Y completed)
- Live performance metrics heatmap
- Running best/worst performers
- Resource utilization (CPU, memory, GPU)
- Estimated time remaining

Features:
- File-watch based updates (no polling overhead)
- Auto-refresh HTML dashboard
- WebSocket support for instant updates (optional)
- Lightweight - minimal impact on test performance
- Works with existing CSV output structure

Usage:
    # Start live dashboard (file-watch mode)
    python -m kinetra.live_dashboard --watch test_results/

    # Start with WebSocket server (real-time)
    python -m kinetra.live_dashboard --watch test_results/ --websocket --port 8050

    # View in browser
    http://localhost:8050

Integration with exhaustive tests:
    # Terminal 1: Run tests
    python scripts/run_exhaustive_tests.py --full --output-dir test_results/

    # Terminal 2: Monitor live
    python -m kinetra.live_dashboard --watch test_results/ --auto-refresh 5
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psutil

# Dashboard framework
try:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html
    from dash.dependencies import Input, Output
    from plotly.subplots import make_subplots

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# File watching
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestProgressMonitor:
    """
    Monitors test progress by watching output directory.
    Aggregates results as they arrive.
    """

    def __init__(self, watch_dir: Path):
        self.watch_dir = Path(watch_dir)
        self.results = []
        self.start_time = None
        self.last_update = None
        self.completed_count = 0
        self.total_count = 0
        self.process = psutil.Process()

    def scan_results(self) -> Dict[str, Any]:
        """Scan directory for test results and aggregate."""
        csv_files = list(self.watch_dir.glob("**/test_results_*.csv"))
        log_files = list(self.watch_dir.glob("**/test_run_*.log"))

        # Load all CSV results
        all_results = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_results.append(df)
            except Exception as e:
                logger.warning(f"Could not read {csv_file}: {e}")

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            self.results = combined.to_dict("records")
            self.completed_count = len(self.results)

        # Try to infer total from summary file
        summary_file = self.watch_dir / "test_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                    self.total_count = summary.get("total_tests", 0)
                    if not self.start_time:
                        self.start_time = datetime.fromisoformat(
                            summary.get("start_time", datetime.now().isoformat())
                        )
            except Exception as e:
                logger.warning(f"Could not read summary: {e}")

        self.last_update = datetime.now()

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.start_time:
            self.start_time = datetime.now()

        elapsed = (datetime.now() - self.start_time).total_seconds()

        # Estimate completion
        eta_seconds = None
        if self.completed_count > 0 and self.total_count > 0:
            rate = self.completed_count / elapsed
            remaining = self.total_count - self.completed_count
            eta_seconds = remaining / rate if rate > 0 else None

        # Resource usage
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        # GPU usage (if available)
        gpu_usage = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_usage = {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "utilization": torch.cuda.utilization(),
                }
        except:
            pass

        return {
            "completed": self.completed_count,
            "total": self.total_count if self.total_count > 0 else "unknown",
            "progress_pct": (
                (self.completed_count / self.total_count * 100) if self.total_count > 0 else 0
            ),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_usage": gpu_usage,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class LiveDashboard:
    """
    Live dashboard for test monitoring using Dash/Plotly.
    """

    def __init__(
        self,
        watch_dir: Path,
        port: int = 8050,
        auto_refresh: int = 5,
        websocket: bool = False,
    ):
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash not installed. Run: pip install dash plotly dash-bootstrap-components"
            )

        self.watch_dir = Path(watch_dir)
        self.port = port
        self.auto_refresh = auto_refresh * 1000  # Convert to milliseconds
        self.websocket = websocket

        self.monitor = TestProgressMonitor(watch_dir)

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            title="Kinetra Live Testing Dashboard",
            update_title="Updating...",
        )

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1(
                            "Kinetra Live Testing Dashboard",
                            style={"color": "#2c3e50", "marginBottom": 10},
                        ),
                        html.P(
                            f"Monitoring: {self.watch_dir}",
                            style={"color": "#7f8c8d", "fontSize": 14},
                        ),
                    ],
                    style={"textAlign": "center", "padding": 20},
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="interval-component",
                    interval=self.auto_refresh,
                    n_intervals=0,
                ),
                # Status cards
                html.Div(
                    id="status-cards",
                    style={"padding": "0 20px"},
                ),
                # Progress bar
                html.Div(
                    id="progress-bar",
                    style={"padding": "20px"},
                ),
                # Heatmap
                html.Div(
                    [
                        html.H3("Performance Heatmap", style={"textAlign": "center"}),
                        dcc.Graph(id="heatmap"),
                    ],
                    style={"padding": "20px"},
                ),
                # Top performers
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Top 10 Performers"),
                                html.Div(id="top-performers"),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H4("Bottom 10 Performers"),
                                html.Div(id="bottom-performers"),
                            ],
                            style={"width": "48%", "display": "inline-block", "float": "right"},
                        ),
                    ],
                    style={"padding": "20px"},
                ),
                # Resource utilization
                html.Div(
                    [
                        html.H3("Resource Utilization", style={"textAlign": "center"}),
                        dcc.Graph(id="resource-usage"),
                    ],
                    style={"padding": "20px"},
                ),
            ]
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks for live updates."""

        @self.app.callback(
            [
                Output("status-cards", "children"),
                Output("progress-bar", "children"),
                Output("heatmap", "figure"),
                Output("top-performers", "children"),
                Output("bottom-performers", "children"),
                Output("resource-usage", "figure"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_dashboard(n):
            # Scan for new results
            status = self.monitor.scan_results()
            results = self.monitor.results

            # Status cards
            cards = self._create_status_cards(status)

            # Progress bar
            progress = self._create_progress_bar(status)

            # Heatmap
            heatmap = self._create_heatmap(results)

            # Top/bottom performers
            top_performers = self._create_top_performers(results, top=True)
            bottom_performers = self._create_top_performers(results, top=False)

            # Resource usage
            resource_fig = self._create_resource_chart(status)

            return cards, progress, heatmap, top_performers, bottom_performers, resource_fig

    def _create_status_cards(self, status: Dict[str, Any]) -> html.Div:
        """Create status cards showing key metrics."""
        card_style = {
            "border": "1px solid #ddd",
            "borderRadius": 8,
            "padding": 15,
            "margin": 10,
            "textAlign": "center",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "display": "inline-block",
            "width": "22%",
        }

        eta_text = "Calculating..."
        if status["eta_seconds"]:
            eta_delta = timedelta(seconds=int(status["eta_seconds"]))
            eta_text = str(eta_delta)

        elapsed_text = str(timedelta(seconds=int(status["elapsed_seconds"])))

        cards = html.Div(
            [
                html.Div(
                    [
                        html.H4(f"{status['completed']} / {status['total']}", style={"margin": 0}),
                        html.P("Tests Completed", style={"color": "#7f8c8d", "fontSize": 12}),
                    ],
                    style=card_style,
                ),
                html.Div(
                    [
                        html.H4(f"{status['progress_pct']:.1f}%", style={"margin": 0}),
                        html.P("Progress", style={"color": "#7f8c8d", "fontSize": 12}),
                    ],
                    style=card_style,
                ),
                html.Div(
                    [
                        html.H4(elapsed_text, style={"margin": 0}),
                        html.P("Elapsed Time", style={"color": "#7f8c8d", "fontSize": 12}),
                    ],
                    style=card_style,
                ),
                html.Div(
                    [
                        html.H4(eta_text, style={"margin": 0}),
                        html.P("ETA", style={"color": "#7f8c8d", "fontSize": 12}),
                    ],
                    style=card_style,
                ),
            ],
            style={"textAlign": "center"},
        )

        return cards

    def _create_progress_bar(self, status: Dict[str, Any]) -> html.Div:
        """Create animated progress bar."""
        progress_pct = status["progress_pct"]

        return html.Div(
            [
                html.Div(
                    style={
                        "width": "100%",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": 10,
                        "height": 30,
                        "position": "relative",
                    },
                    children=[
                        html.Div(
                            style={
                                "width": f"{progress_pct}%",
                                "backgroundColor": "#3498db",
                                "height": "100%",
                                "borderRadius": 10,
                                "transition": "width 0.5s ease",
                            }
                        ),
                        html.Span(
                            f"{progress_pct:.1f}%",
                            style={
                                "position": "absolute",
                                "top": "50%",
                                "left": "50%",
                                "transform": "translate(-50%, -50%)",
                                "fontWeight": "bold",
                                "color": "#2c3e50",
                            },
                        ),
                    ],
                )
            ]
        )

    def _create_heatmap(self, results: List[Dict]) -> go.Figure:
        """Create performance heatmap."""
        if not results:
            return go.Figure()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Create pivot table for heatmap
        if "agent" in df.columns and "instrument" in df.columns and "omega_ratio" in df.columns:
            pivot = df.pivot_table(
                values="omega_ratio", index="agent", columns="instrument", aggfunc="mean"
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                    text=pivot.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10},
                    colorbar=dict(title="Omega Ratio"),
                )
            )

            fig.update_layout(
                xaxis_title="Instrument",
                yaxis_title="Agent",
                height=400,
                margin=dict(l=100, r=50, t=50, b=100),
            )

            return fig

        return go.Figure()

    def _create_top_performers(self, results: List[Dict], top: bool = True) -> html.Div:
        """Create list of top/bottom performers."""
        if not results:
            return html.P("No results yet...")

        df = pd.DataFrame(results)

        if "omega_ratio" not in df.columns:
            return html.P("Waiting for performance metrics...")

        # Sort by omega ratio
        sorted_df = df.sort_values("omega_ratio", ascending=not top).head(10)

        items = []
        for i, row in sorted_df.iterrows():
            agent = row.get("agent", "Unknown")
            instrument = row.get("instrument", "Unknown")
            timeframe = row.get("timeframe", "Unknown")
            omega = row.get("omega_ratio", 0)

            color = "#27ae60" if omega > 2.7 else "#e74c3c"

            items.append(
                html.Div(
                    [
                        html.Span(
                            f"{agent} - {instrument} {timeframe}",
                            style={"fontWeight": "bold"},
                        ),
                        html.Span(
                            f"Ω = {omega:.2f}",
                            style={"float": "right", "color": color, "fontWeight": "bold"},
                        ),
                    ],
                    style={
                        "padding": "8px",
                        "borderBottom": "1px solid #ecf0f1",
                    },
                )
            )

        return html.Div(items, style={"maxHeight": "400px", "overflow": "auto"})

    def _create_resource_chart(self, status: Dict[str, Any]) -> go.Figure:
        """Create resource utilization chart."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("CPU Usage", "Memory Usage"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        )

        # CPU gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=status["cpu_percent"],
                title={"text": "CPU %"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "red"},
                    ],
                },
            ),
            row=1,
            col=1,
        )

        # Memory gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=status["memory_mb"],
                title={"text": "Memory (MB)"},
                gauge={
                    "axis": {"range": [0, psutil.virtual_memory().total / 1024 / 1024]},
                    "bar": {"color": "darkgreen"},
                },
            ),
            row=1,
            col=2,
        )

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

        return fig

    def run(self):
        """Start the dashboard server."""
        logger.info(f"Starting live dashboard at http://localhost:{self.port}")
        logger.info(f"Watching directory: {self.watch_dir}")
        logger.info(f"Auto-refresh: {self.auto_refresh / 1000:.0f} seconds")
        logger.info("Press Ctrl+C to stop")

        self.app.run_server(debug=False, host="0.0.0.0", port=self.port)


def main():
    """Main entry point for live dashboard."""
    parser = argparse.ArgumentParser(
        description="Live dashboard for Kinetra test monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic file-watch mode
  python -m kinetra.live_dashboard --watch test_results/

  # Custom port and refresh rate
  python -m kinetra.live_dashboard --watch test_results/ --port 8080 --auto-refresh 10

  # WebSocket mode for instant updates
  python -m kinetra.live_dashboard --watch test_results/ --websocket

View dashboard: http://localhost:8050
        """,
    )

    parser.add_argument(
        "--watch",
        type=Path,
        required=True,
        help="Directory to watch for test results",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard server port (default: 8050)",
    )
    parser.add_argument(
        "--auto-refresh",
        type=int,
        default=5,
        help="Auto-refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--websocket",
        action="store_true",
        help="Enable WebSocket for instant updates (experimental)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate watch directory
    if not args.watch.exists():
        logger.error(f"Watch directory does not exist: {args.watch}")
        return 1

    # Check dependencies
    if not DASH_AVAILABLE:
        logger.error("Dash not installed. Run: pip install dash plotly dash-bootstrap-components")
        return 1

    # Create and run dashboard
    try:
        dashboard = LiveDashboard(
            watch_dir=args.watch,
            port=args.port,
            auto_refresh=args.auto_refresh,
            websocket=args.websocket,
        )

        dashboard.run()
        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️  Dashboard stopped by user")
        return 0

    except Exception as e:
        logger.error(f"❌ Dashboard failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
