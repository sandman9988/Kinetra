"""
Interactive Test Dashboard for Kinetra
========================================

Real-time monitoring and visualization of exhaustive test results.

Features:
- Live test progress tracking
- Interactive heatmaps (agent × regime × timeframe)
- Performance metrics visualization
- Agent comparison charts
- Export to static HTML for CI artifacts

Usage:
    # Launch dashboard
    python -m kinetra.test_dashboard

    # Or programmatically
    from kinetra.test_dashboard import TestDashboard
    dashboard = TestDashboard()
    dashboard.launch(port=8050)

    # Generate static report
    dashboard.generate_static_report('test_report.html')

Philosophy: Make test results VISUAL and INTERACTIVE for rapid insights.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: dash not available. Install with: pip install dash dash-bootstrap-components")

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

METRICS = {
    "omega_ratio": {"target": 2.7, "higher_better": True, "format": ".2f"},
    "z_factor": {"target": 2.5, "higher_better": True, "format": ".2f"},
    "chs": {"target": 0.90, "higher_better": True, "format": ".3f"},
    "ror": {"target": 0.05, "higher_better": False, "format": ".3f"},
    "p_value": {"target": 0.01, "higher_better": False, "format": ".4f"},
}

COLORS = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "danger": "#d62728",
    "info": "#17a2b8",
    "background": "#f8f9fa",
}


# =============================================================================
# DATA LOADER
# =============================================================================


class TestResultsLoader:
    """Load and parse test results from CSV files."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_latest_results(self, test_type: Optional[str] = None) -> pd.DataFrame:
        """Load latest test results."""
        pattern = (
            f"exhaustive_results_{test_type}_*.csv" if test_type else "exhaustive_results_*.csv"
        )
        files = sorted(self.data_dir.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            logger.warning(f"No test results found in {self.data_dir}")
            return pd.DataFrame()

        # Load most recent file
        df = pd.read_csv(files[0])
        logger.info(f"Loaded {len(df)} results from {files[0].name}")
        return df

    def load_all_results(self) -> Dict[str, pd.DataFrame]:
        """Load all test types."""
        results = {}
        for test_type in ["unit", "integration", "monte_carlo", "walk_forward"]:
            df = self.load_latest_results(test_type)
            if not df.empty:
                results[test_type] = df
        return results

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if df.empty:
            return {}

        return {
            "total_tests": len(df),
            "passed": int(df["valid"].sum()),
            "failed": int((~df["valid"]).sum()),
            "pass_rate": float(df["valid"].mean()),
            "mean_omega": float(df["omega"].mean()) if "omega" in df.columns else None,
            "mean_z_factor": float(df["z_factor"].mean()) if "z_factor" in df.columns else None,
            "mean_chs": float(df["chs"].mean()) if "chs" in df.columns else None,
            "mean_ror": float(df["ror"].mean()) if "ror" in df.columns else None,
        }


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================


class Visualizer:
    """Create interactive visualizations."""

    @staticmethod
    def create_heatmap(df: pd.DataFrame, metric: str = "valid") -> go.Figure:
        """Create interactive heatmap."""
        if df.empty:
            return go.Figure()

        # Create pivot table
        df["combo"] = df["instrument"] + "_" + df["timeframe"]
        df["agent_regime"] = df["agent_type"] + "_" + df["regime"]

        pivot = df.pivot_table(values=metric, index="combo", columns="agent_regime", aggfunc="mean")

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn" if metric == "valid" else "Viridis",
                text=np.round(pivot.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=f"{metric.upper()} Heatmap",
            xaxis_title="Agent × Regime",
            yaxis_title="Instrument × Timeframe",
            height=max(400, len(pivot.index) * 20),
        )

        return fig

    @staticmethod
    def create_agent_comparison(df: pd.DataFrame) -> go.Figure:
        """Compare agents across metrics."""
        if df.empty or "agent_type" not in df.columns:
            return go.Figure()

        # Aggregate by agent type
        agent_stats = (
            df.groupby("agent_type")
            .agg(
                {
                    "valid": "mean",
                    "omega": "mean",
                    "z_factor": "mean",
                    "chs": "mean",
                    "ror": "mean",
                }
            )
            .reset_index()
        )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Pass Rate", "Omega Ratio", "Z-Factor", "CHS"),
        )

        # Pass rate
        fig.add_trace(
            go.Bar(x=agent_stats["agent_type"], y=agent_stats["valid"], name="Pass Rate"),
            row=1,
            col=1,
        )

        # Omega ratio
        fig.add_trace(
            go.Bar(x=agent_stats["agent_type"], y=agent_stats["omega"], name="Omega"), row=1, col=2
        )

        # Z-factor
        fig.add_trace(
            go.Bar(x=agent_stats["agent_type"], y=agent_stats["z_factor"], name="Z-Factor"),
            row=2,
            col=1,
        )

        # CHS
        fig.add_trace(
            go.Bar(x=agent_stats["agent_type"], y=agent_stats["chs"], name="CHS"), row=2, col=2
        )

        fig.update_layout(
            title="Agent Performance Comparison",
            height=600,
            showlegend=False,
        )

        return fig

    @staticmethod
    def create_regime_comparison(df: pd.DataFrame) -> go.Figure:
        """Compare regimes across metrics."""
        if df.empty or "regime" not in df.columns:
            return go.Figure()

        regime_stats = (
            df.groupby("regime")
            .agg(
                {
                    "valid": "mean",
                    "omega": "mean",
                    "z_factor": "mean",
                }
            )
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=regime_stats["regime"],
                y=regime_stats["valid"],
                name="Pass Rate",
                marker_color=COLORS["success"],
            )
        )

        fig.update_layout(
            title="Regime Performance",
            xaxis_title="Regime",
            yaxis_title="Pass Rate",
            height=400,
        )

        return fig

    @staticmethod
    def create_metrics_distribution(df: pd.DataFrame, metric: str) -> go.Figure:
        """Create distribution plot for a metric."""
        if df.empty or metric not in df.columns:
            return go.Figure()

        data = df[metric].dropna()

        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name="Distribution",
                marker_color=COLORS["primary"],
            )
        )

        # Add target line if applicable
        if metric in METRICS:
            target = METRICS[metric]["target"]
            fig.add_vline(
                x=target,
                line_dash="dash",
                line_color=COLORS["danger"],
                annotation_text=f"Target: {target}",
            )

        fig.update_layout(
            title=f"{metric.upper()} Distribution",
            xaxis_title=metric,
            yaxis_title="Count",
            height=400,
        )

        return fig


# =============================================================================
# DASHBOARD
# =============================================================================


class TestDashboard:
    """Interactive test dashboard."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.loader = TestResultsLoader(data_dir)
        self.visualizer = Visualizer()
        self.app = None

        if DASH_AVAILABLE:
            self._create_app()

    def _create_app(self):
        """Create Dash app."""
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Kinetra Test Dashboard"
        )

        self.app.layout = self._create_layout()
        self._register_callbacks()

    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Kinetra Exhaustive Test Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.P(
                                    "Real-time monitoring of exhaustive test results across agents, regimes, and timeframes",
                                    className="text-center text-muted",
                                ),
                            ],
                            width=12,
                        )
                    ]
                ),
                # Controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Test Type:"),
                                dcc.Dropdown(
                                    id="test-type-dropdown",
                                    options=[
                                        {"label": "Unit", "value": "unit"},
                                        {"label": "Integration", "value": "integration"},
                                        {"label": "Monte Carlo", "value": "monte_carlo"},
                                        {"label": "Walk-Forward", "value": "walk_forward"},
                                    ],
                                    value="unit",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Metric for Heatmap:"),
                                dcc.Dropdown(
                                    id="metric-dropdown",
                                    options=[
                                        {"label": "Valid", "value": "valid"},
                                        {"label": "Omega Ratio", "value": "omega"},
                                        {"label": "Z-Factor", "value": "z_factor"},
                                        {"label": "CHS", "value": "chs"},
                                        {"label": "RoR", "value": "ror"},
                                    ],
                                    value="valid",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Auto-Refresh:"),
                                dbc.Switch(
                                    id="auto-refresh-switch",
                                    label="Enable",
                                    value=False,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button(
                                    "Refresh Now",
                                    id="refresh-button",
                                    color="primary",
                                    className="w-100",
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-4",
                ),
                # Summary stats
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(id="total-tests", className="text-center"),
                                                html.P(
                                                    "Total Tests",
                                                    className="text-center text-muted",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    id="passed-tests",
                                                    className="text-center text-success",
                                                ),
                                                html.P(
                                                    "Passed", className="text-center text-muted"
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    id="failed-tests",
                                                    className="text-center text-danger",
                                                ),
                                                html.P(
                                                    "Failed", className="text-center text-muted"
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    id="pass-rate",
                                                    className="text-center text-info",
                                                ),
                                                html.P(
                                                    "Pass Rate", className="text-center text-muted"
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                            width=3,
                        ),
                    ],
                    className="mb-4",
                ),
                # Main visualizations
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="heatmap"),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="agent-comparison"),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(id="regime-comparison"),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="metric-distribution"),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                # Auto-refresh interval
                dcc.Interval(
                    id="interval-component",
                    interval=10 * 1000,  # 10 seconds
                    n_intervals=0,
                    disabled=True,
                ),
                # Store
                dcc.Store(id="data-store"),
            ],
            fluid=True,
        )

    def _register_callbacks(self):
        """Register Dash callbacks."""

        @self.app.callback(
            Output("interval-component", "disabled"), Input("auto-refresh-switch", "value")
        )
        def toggle_auto_refresh(enabled):
            return not enabled

        @self.app.callback(
            Output("data-store", "data"),
            [
                Input("test-type-dropdown", "value"),
                Input("refresh-button", "n_clicks"),
                Input("interval-component", "n_intervals"),
            ],
        )
        def load_data(test_type, n_clicks, n_intervals):
            df = self.loader.load_latest_results(test_type)
            if df.empty:
                return {}
            return df.to_json(date_format="iso", orient="split")

        @self.app.callback(
            [
                Output("total-tests", "children"),
                Output("passed-tests", "children"),
                Output("failed-tests", "children"),
                Output("pass-rate", "children"),
            ],
            Input("data-store", "data"),
        )
        def update_stats(data_json):
            if not data_json:
                return "0", "0", "0", "0%"

            df = pd.read_json(data_json, orient="split")
            stats = self.loader.get_summary_stats(df)

            return (
                str(stats.get("total_tests", 0)),
                str(stats.get("passed", 0)),
                str(stats.get("failed", 0)),
                f"{stats.get('pass_rate', 0) * 100:.1f}%",
            )

        @self.app.callback(
            Output("heatmap", "figure"),
            [Input("data-store", "data"), Input("metric-dropdown", "value")],
        )
        def update_heatmap(data_json, metric):
            if not data_json:
                return go.Figure()
            df = pd.read_json(data_json, orient="split")
            return self.visualizer.create_heatmap(df, metric)

        @self.app.callback(Output("agent-comparison", "figure"), Input("data-store", "data"))
        def update_agent_comparison(data_json):
            if not data_json:
                return go.Figure()
            df = pd.read_json(data_json, orient="split")
            return self.visualizer.create_agent_comparison(df)

        @self.app.callback(Output("regime-comparison", "figure"), Input("data-store", "data"))
        def update_regime_comparison(data_json):
            if not data_json:
                return go.Figure()
            df = pd.read_json(data_json, orient="split")
            return self.visualizer.create_regime_comparison(df)

        @self.app.callback(Output("metric-distribution", "figure"), Input("data-store", "data"))
        def update_metric_distribution(data_json):
            if not data_json:
                return go.Figure()
            df = pd.read_json(data_json, orient="split")
            return self.visualizer.create_metrics_distribution(df, "omega")

    def launch(self, port: int = 8050, debug: bool = True):
        """Launch dashboard server."""
        if not DASH_AVAILABLE:
            print(
                "ERROR: Dash not available. Install with: pip install dash dash-bootstrap-components"
            )
            return

        print(f"\n🚀 Launching Kinetra Test Dashboard at http://localhost:{port}")
        print("Press Ctrl+C to stop\n")

        self.app.run_server(debug=debug, port=port)

    def generate_static_report(self, output_path: str = "test_report.html"):
        """Generate static HTML report."""
        if not PLOTLY_AVAILABLE:
            print("ERROR: Plotly not available. Install with: pip install plotly")
            return

        # Load all test results
        all_results = self.loader.load_all_results()

        if not all_results:
            print("No test results found")
            return

        # Create HTML report
        html_parts = [
            "<html>",
            "<head>",
            "<title>Kinetra Test Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #1f77b4; }",
            "h2 { color: #2ca02c; margin-top: 40px; }",
            ".summary { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }",
            ".metric { display: inline-block; margin: 10px 20px; }",
            ".metric-value { font-size: 24px; font-weight: bold; }",
            ".metric-label { font-size: 14px; color: #666; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Kinetra Exhaustive Test Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        for test_type, df in all_results.items():
            stats = self.loader.get_summary_stats(df)

            html_parts.append(f"<h2>{test_type.upper()} Tests</h2>")
            html_parts.append('<div class="summary">')
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{stats["total_tests"]}</div><div class="metric-label">Total Tests</div></div>'
            )
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{stats["passed"]}</div><div class="metric-label">Passed</div></div>'
            )
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{stats["failed"]}</div><div class="metric-label">Failed</div></div>'
            )
            html_parts.append(
                f'<div class="metric"><div class="metric-value">{stats["pass_rate"] * 100:.1f}%</div><div class="metric-label">Pass Rate</div></div>'
            )
            html_parts.append("</div>")

            # Add charts
            heatmap = self.visualizer.create_heatmap(df, "valid")
            html_parts.append(heatmap.to_html(include_plotlyjs="cdn", full_html=False))

            agent_comp = self.visualizer.create_agent_comparison(df)
            html_parts.append(agent_comp.to_html(include_plotlyjs=False, full_html=False))

        html_parts.append("</body></html>")

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(html_parts))

        print(f"✅ Static report generated: {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinetra Test Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--data-dir", default="data", help="Test results directory")
    parser.add_argument("--static", action="store_true", help="Generate static HTML report")
    parser.add_argument("--output", default="test_report.html", help="Static report output path")

    args = parser.parse_args()

    dashboard = TestDashboard(data_dir=args.data_dir)

    if args.static:
        dashboard.generate_static_report(args.output)
    else:
        dashboard.launch(port=args.port)
