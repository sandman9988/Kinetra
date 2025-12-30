"""
Grafana Dashboard JSON Templates

Provides pre-built dashboard configurations for:
- Portfolio Overview: Equity curve, drawdown, key metrics
- Instrument Drill-down: Per-instrument performance
- Physics Analysis: Regime performance, energy metrics
- Trade Analysis: Individual trade details
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PanelConfig:
    """Configuration for a Grafana panel."""

    id: int
    title: str
    type: str
    gridPos: Dict[str, int]
    targets: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    fieldConfig: Dict[str, Any] = field(default_factory=dict)


class DashboardGenerator:
    """
    Generates Grafana dashboard JSON configurations.

    Usage:
        generator = DashboardGenerator(datasource_name="Kinetra Backtest")

        # Get individual dashboards
        portfolio_json = generator.portfolio_overview()
        instrument_json = generator.instrument_drilldown()

        # Export all dashboards
        generator.export_all("/path/to/grafana/dashboards/")
    """

    def __init__(
        self,
        datasource_name: str = "Kinetra Backtest",
        datasource_uid: str = "kinetra-backtest",
    ):
        self.datasource_name = datasource_name
        self.datasource_uid = datasource_uid

    def _base_dashboard(
        self,
        title: str,
        uid: str,
        tags: Optional[List[str]] = None,
    ) -> Dict:
        """Create base dashboard structure."""
        return {
            "id": None,
            "uid": uid,
            "title": title,
            "tags": tags or ["kinetra", "backtest"],
            "timezone": "browser",
            "schemaVersion": 38,
            "version": 1,
            "refresh": "",
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,  # Shared crosshair
            "links": [],
            "liveNow": False,
            "panels": [],
            "templating": {"list": []},
            "time": {"from": "now-30d", "to": "now"},
            "timepicker": {},
            "weekStart": "",
        }

    def _datasource_ref(self) -> Dict:
        """Create datasource reference."""
        return {
            "type": "simplejson",
            "uid": self.datasource_uid,
        }

    def _stat_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
        unit: str = "none",
        color_mode: str = "value",
        thresholds: Optional[List[Dict]] = None,
    ) -> Dict:
        """Create a stat panel."""
        panel = {
            "id": id,
            "title": title,
            "type": "stat",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "timeserie",
                }
            ],
            "options": {
                "colorMode": color_mode,
                "graphMode": "none",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "textMode": "auto",
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds
                        or [
                            {"color": "red", "value": None},
                            {"color": "green", "value": 0},
                        ],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
        }
        return panel

    def _timeseries_panel(
        self,
        id: int,
        title: str,
        targets: List[str],
        gridPos: Dict[str, int],
        unit: str = "none",
        fill_opacity: int = 10,
        show_legend: bool = True,
    ) -> Dict:
        """Create a time series panel."""
        panel = {
            "id": id,
            "title": title,
            "type": "timeseries",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": chr(65 + i),  # A, B, C...
                    "target": t,
                    "type": "timeserie",
                }
                for i, t in enumerate(targets)
            ],
            "options": {
                "legend": {
                    "calcs": ["mean", "max", "min", "lastNotNull"],
                    "displayMode": "table",
                    "placement": "bottom",
                    "showLegend": show_legend,
                },
                "tooltip": {
                    "mode": "multi",
                    "sort": "desc",
                },
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": fill_opacity,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": False,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "off"},
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
        }
        return panel

    def _table_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
        column_widths: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """Create a table panel."""
        overrides = []
        if column_widths:
            for col, width in column_widths.items():
                overrides.append(
                    {
                        "matcher": {"id": "byName", "options": col},
                        "properties": [{"id": "custom.width", "value": width}],
                    }
                )

        panel = {
            "id": id,
            "title": title,
            "type": "table",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "table",
                }
            ],
            "options": {
                "showHeader": True,
                "cellHeight": "sm",
                "footer": {"show": False, "reducer": ["sum"], "fields": ""},
                "sortBy": [],
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "custom": {
                        "align": "auto",
                        "cellOptions": {"type": "auto"},
                        "inspect": False,
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                },
                "overrides": overrides,
            },
        }
        return panel

    def _gauge_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
        min_val: float = 0,
        max_val: float = 100,
        unit: str = "percent",
        thresholds: Optional[List[Dict]] = None,
    ) -> Dict:
        """Create a gauge panel."""
        panel = {
            "id": id,
            "title": title,
            "type": "gauge",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "timeserie",
                }
            ],
            "options": {
                "minVizHeight": 75,
                "minVizWidth": 75,
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "max": max_val,
                    "min": min_val,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds
                        or [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 40},
                            {"color": "green", "value": 60},
                        ],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
        }
        return panel

    def _bar_chart_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
        orientation: str = "horizontal",
    ) -> Dict:
        """Create a bar chart panel."""
        panel = {
            "id": id,
            "title": title,
            "type": "barchart",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "table",
                }
            ],
            "options": {
                "barRadius": 0,
                "barWidth": 0.8,
                "fullHighlight": False,
                "groupWidth": 0.7,
                "legend": {"displayMode": "list", "placement": "right", "showLegend": True},
                "orientation": orientation,
                "showValue": "auto",
                "stacking": "none",
                "tooltip": {"mode": "single", "sort": "none"},
                "xTickLabelRotation": 0,
                "xTickLabelSpacing": 0,
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "fillOpacity": 80,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineWidth": 1,
                        "scaleDistribution": {"type": "linear"},
                        "thresholdsStyle": {"mode": "off"},
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                },
                "overrides": [],
            },
        }
        return panel

    def _row_panel(self, id: int, title: str, y_pos: int, collapsed: bool = False) -> Dict:
        """Create a row panel for grouping."""
        return {
            "id": id,
            "title": title,
            "type": "row",
            "collapsed": collapsed,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
            "panels": [],
        }

    def _bargauge_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
        min_val: float = 0,
        max_val: float = 1,
        unit: str = "percentunit",
        orientation: str = "horizontal",
        thresholds: Optional[List[Dict]] = None,
    ) -> Dict:
        """Create a bar gauge panel - ideal for Layer-1 physics percentiles."""
        panel = {
            "id": id,
            "title": title,
            "type": "bargauge",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "timeserie",
                }
            ],
            "options": {
                "displayMode": "gradient",
                "minVizHeight": 10,
                "minVizWidth": 0,
                "orientation": orientation,
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "showUnfilled": True,
                "valueMode": "color",
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "continuous-BlYlRd"},
                    "mappings": [],
                    "max": max_val,
                    "min": min_val,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds
                        or [
                            {"color": "blue", "value": None},
                            {"color": "green", "value": 0.3},
                            {"color": "yellow", "value": 0.6},
                            {"color": "red", "value": 0.8},
                        ],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
        }
        return panel

    def _heatmap_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
    ) -> Dict:
        """Create a heatmap panel - ideal for regime transitions."""
        panel = {
            "id": id,
            "title": title,
            "type": "heatmap",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "timeserie",
                }
            ],
            "options": {
                "calculate": False,
                "cellGap": 1,
                "color": {
                    "exponent": 0.5,
                    "fill": "dark-orange",
                    "mode": "scheme",
                    "reverse": False,
                    "scale": "exponential",
                    "scheme": "Oranges",
                    "steps": 64,
                },
                "exemplars": {"color": "rgba(255,0,255,0.7)"},
                "filterValues": {"le": 1e-9},
                "legend": {"show": True},
                "rowsFrame": {"layout": "auto"},
                "tooltip": {"show": True, "yHistogram": False},
                "yAxis": {"axisPlacement": "left", "reverse": False},
            },
        }
        return panel

    def _state_timeline_panel(
        self,
        id: int,
        title: str,
        target: str,
        gridPos: Dict[str, int],
    ) -> Dict:
        """Create a state timeline panel - ideal for regime visualization."""
        panel = {
            "id": id,
            "title": title,
            "type": "state-timeline",
            "datasource": self._datasource_ref(),
            "gridPos": gridPos,
            "targets": [
                {
                    "datasource": self._datasource_ref(),
                    "refId": "A",
                    "target": target,
                    "type": "timeserie",
                }
            ],
            "options": {
                "alignValue": "center",
                "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                "mergeValues": True,
                "rowHeight": 0.9,
                "showValue": "auto",
                "tooltip": {"mode": "single", "sort": "none"},
            },
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "fillOpacity": 70,
                        "lineWidth": 0,
                        "spanNulls": False,
                    },
                    "mappings": [
                        {
                            "options": {"underdamped": {"color": "green", "index": 0}},
                            "type": "value",
                        },
                        {"options": {"overdamped": {"color": "red", "index": 1}}, "type": "value"},
                        {"options": {"critical": {"color": "yellow", "index": 2}}, "type": "value"},
                        {"options": {"laminar": {"color": "blue", "index": 3}}, "type": "value"},
                        {"options": {"breakout": {"color": "purple", "index": 4}}, "type": "value"},
                    ],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                },
                "overrides": [],
            },
        }
        return panel

    def _template_variable(
        self,
        name: str,
        label: str,
        query: str,
        multi: bool = False,
        include_all: bool = False,
    ) -> Dict:
        """Create a template variable."""
        return {
            "current": {
                "selected": False,
                "text": "All" if include_all else "",
                "value": "$__all" if include_all else "",
            },
            "datasource": self._datasource_ref(),
            "definition": query,
            "hide": 0,
            "includeAll": include_all,
            "label": label,
            "multi": multi,
            "name": name,
            "options": [],
            "query": {"query": query, "refId": "StandardVariableQuery"},
            "refresh": 1,
            "regex": "",
            "skipUrlSync": False,
            "sort": 1,
            "type": "query",
        }

    # === Dashboard Generators ===

    def portfolio_overview(self) -> Dict:
        """
        Generate Portfolio Overview dashboard.

        Shows:
        - Equity curve
        - Drawdown chart
        - Key performance metrics (win rate, sharpe, sortino, profit factor)
        - Risk metrics (max drawdown, min margin level)
        - Cost breakdown (spread, commission, slippage, gap slippage, swap)
        - Summary statistics
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Portfolio Overview",
            uid="kinetra-portfolio",
            tags=["kinetra", "backtest", "portfolio"],
        )

        panels = []
        panel_id = 1

        # Row 1: Key Metrics (stat panels) - 8 panels across
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Total Trades",
                target="summary.total_trades",
                gridPos={"h": 4, "w": 3, "x": 0, "y": 0},
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Win Rate",
                target="portfolio.win_rate",
                gridPos={"h": 4, "w": 3, "x": 3, "y": 0},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "green", "value": 0.55},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Sharpe Ratio",
                target="portfolio.sharpe",
                gridPos={"h": 4, "w": 3, "x": 6, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 1.0},
                    {"color": "blue", "value": 2.0},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Sortino Ratio",
                target="portfolio.sortino",
                gridPos={"h": 4, "w": 3, "x": 9, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 1.0},
                    {"color": "blue", "value": 2.0},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Profit Factor",
                target="portfolio.profit_factor",
                gridPos={"h": 4, "w": 3, "x": 12, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1.0},
                    {"color": "green", "value": 1.5},
                    {"color": "blue", "value": 2.0},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Omega Ratio",
                target="portfolio.omega",
                gridPos={"h": 4, "w": 3, "x": 15, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1.0},
                    {"color": "green", "value": 1.5},
                    {"color": "blue", "value": 2.0},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Max Drawdown",
                target="summary.max_drawdown_pct",
                gridPos={"h": 4, "w": 3, "x": 18, "y": 0},
                unit="percentunit",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.1},
                    {"color": "red", "value": 0.2},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Min Margin Level",
                target="risk.min_margin_level",
                gridPos={"h": 4, "w": 3, "x": 21, "y": 0},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1.5},
                    {"color": "green", "value": 3.0},
                ],
            )
        )
        panel_id += 1

        # Row 2: Equity Curve and Margin Level
        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Equity Curve",
                targets=["portfolio.equity"],
                gridPos={"h": 8, "w": 16, "x": 0, "y": 4},
                unit="currencyUSD",
                fill_opacity=20,
            )
        )
        panel_id += 1

        # Margin Level % over time (risk monitoring)
        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Margin Level %",
                targets=["portfolio.margin_level"],
                gridPos={"h": 8, "w": 8, "x": 16, "y": 4},
                unit="percentunit",
                fill_opacity=30,
            )
        )
        panel_id += 1

        # Row 3: Drawdown and PnL Distribution
        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Drawdown",
                targets=["portfolio.drawdown"],
                gridPos={"h": 6, "w": 12, "x": 0, "y": 12},
                unit="percentunit",
                fill_opacity=50,
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="PnL Distribution",
                target="distribution.pnl",
                gridPos={"h": 6, "w": 12, "x": 12, "y": 12},
                orientation="vertical",
            )
        )
        panel_id += 1

        # Row 4: Cost Breakdown
        panels.append(self._row_panel(id=panel_id, title="Cost Analysis", y_pos=18))
        panel_id += 1

        # Cost breakdown stat panels
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Spread Cost",
                target="costs.spread",
                gridPos={"h": 4, "w": 4, "x": 0, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "orange", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Commission",
                target="costs.commission",
                gridPos={"h": 4, "w": 4, "x": 4, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "orange", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Slippage",
                target="costs.slippage",
                gridPos={"h": 4, "w": 4, "x": 8, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "orange", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Gap Slippage",
                target="costs.gap_slippage",
                gridPos={"h": 4, "w": 4, "x": 12, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "red", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Swap Cost",
                target="costs.swap",
                gridPos={"h": 4, "w": 4, "x": 16, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "orange", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Total Costs",
                target="costs.total",
                gridPos={"h": 4, "w": 4, "x": 20, "y": 19},
                unit="currencyUSD",
                color_mode="background",
                thresholds=[{"color": "red", "value": None}],
            )
        )
        panel_id += 1

        # Row 5: Instrument Summary Table
        panels.append(self._row_panel(id=panel_id, title="Instrument Breakdown", y_pos=23))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Performance by Instrument",
                target="table.instruments",
                gridPos={"h": 8, "w": 24, "x": 0, "y": 24},
                column_widths={"Symbol": 120, "Win Rate": 100, "Net Profit": 120},
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def instrument_drilldown(self) -> Dict:
        """
        Generate Instrument Drill-down dashboard.

        Shows detailed performance per instrument with variable selection.
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Instrument Drill-down",
            uid="kinetra-instrument",
            tags=["kinetra", "backtest", "instrument"],
        )

        # Add template variables
        dashboard["templating"]["list"] = [
            self._template_variable(
                name="symbol",
                label="Symbol",
                query="tag_values(symbol)",
                multi=False,
                include_all=True,
            ),
            self._template_variable(
                name="timeframe",
                label="Timeframe",
                query="tag_values(timeframe)",
                multi=True,
                include_all=True,
            ),
        ]

        panels = []
        panel_id = 1

        # Row 1: Instrument selector and summary
        panels.append(
            self._table_panel(
                id=panel_id,
                title="Instruments Overview",
                target="table.instruments",
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0},
            )
        )
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Timeframe Breakdown",
                target="table.timeframes",
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
            )
        )
        panel_id += 1

        # Row 2: Class Performance
        panels.append(self._row_panel(id=panel_id, title="Instrument Class Performance", y_pos=8))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Performance by Asset Class",
                target="table.classes",
                gridPos={"h": 6, "w": 24, "x": 0, "y": 9},
            )
        )
        panel_id += 1

        # Row 3: Trades for selected instrument
        panels.append(self._row_panel(id=panel_id, title="Trade Details", y_pos=15))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Trades (filtered by $symbol)",
                target="table.trades",
                gridPos={"h": 12, "w": 24, "x": 0, "y": 16},
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def physics_analysis(self) -> Dict:
        """
        Generate Physics Analysis dashboard.

        Shows Kinetra-specific physics metrics:
        - Layer-1 sensors (KE, Re, zeta, Hs, PE, eta percentiles)
        - Regime performance (GMM clustering)
        - Energy metrics (kinetic/potential)
        - Potential Energy (PE) analysis - squeeze/compression detection
        - Z-Factor analysis
        - MFE/MAE efficiency
        - CVaR tail risk metrics
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Physics Analysis",
            uid="kinetra-physics",
            tags=["kinetra", "backtest", "physics", "layer1"],
        )

        panels = []
        panel_id = 1

        # Row 1: Layer-1 Physics Sensors (Bar Gauges) - 6 sensors
        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="KE% (Kinetic Energy)",
                target="layer1.ke_pct",
                gridPos={"h": 4, "w": 4, "x": 0, "y": 0},
                thresholds=[
                    {"color": "blue", "value": None},
                    {"color": "green", "value": 0.3},
                    {"color": "yellow", "value": 0.6},
                    {"color": "red", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="Re% (Reynolds - Trend)",
                target="layer1.re_m_pct",
                gridPos={"h": 4, "w": 4, "x": 4, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.6},
                    {"color": "blue", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="ζ% (Damping - Friction)",
                target="layer1.zeta_pct",
                gridPos={"h": 4, "w": 4, "x": 8, "y": 0},
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "orange", "value": 0.6},
                    {"color": "red", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="Hs% (Entropy - Disorder)",
                target="layer1.hs_pct",
                gridPos={"h": 4, "w": 4, "x": 12, "y": 0},
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "orange", "value": 0.6},
                    {"color": "red", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="PE% (Potential Energy)",
                target="layer1.pe_pct",
                gridPos={"h": 4, "w": 4, "x": 16, "y": 0},
                thresholds=[
                    {"color": "blue", "value": None},
                    {"color": "green", "value": 0.3},
                    {"color": "yellow", "value": 0.6},
                    {"color": "red", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="η% (Efficiency)",
                target="layer1.eta_pct",
                gridPos={"h": 4, "w": 4, "x": 20, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.5},
                    {"color": "blue", "value": 0.7},
                ],
            )
        )
        panel_id += 1

        # Row 2: Key Stats + CVaR Risk Metrics
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Z-Factor",
                target="summary.z_factor",
                gridPos={"h": 3, "w": 3, "x": 0, "y": 4},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 1.0},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Energy Captured",
                target="physics.energy_captured_pct",
                gridPos={"h": 3, "w": 3, "x": 3, "y": 4},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="MFE Capture",
                target="physics.mfe_capture_pct",
                gridPos={"h": 3, "w": 3, "x": 6, "y": 4},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "green", "value": 0.6},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="CVaR 95%",
                target="risk.cvar_95",
                gridPos={"h": 3, "w": 3, "x": 9, "y": 4},
                unit="percentunit",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": -0.02},
                    {"color": "red", "value": -0.05},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="CVaR 99%",
                target="risk.cvar_99",
                gridPos={"h": 3, "w": 3, "x": 12, "y": 4},
                unit="percentunit",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": -0.03},
                    {"color": "red", "value": -0.08},
                ],
            )
        )
        panel_id += 1

        # Regime state timeline
        panels.append(
            self._state_timeline_panel(
                id=panel_id,
                title="Regime Timeline",
                target="physics.regime_timeline",
                gridPos={"h": 3, "w": 9, "x": 15, "y": 4},
            )
        )
        panel_id += 1

        # Row 3: Regime Performance Table
        panels.append(
            self._row_panel(id=panel_id, title="Regime Analysis (GMM Clustering)", y_pos=7)
        )
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Performance by Market Regime",
                target="table.trades_by_regime",
                gridPos={"h": 7, "w": 16, "x": 0, "y": 8},
                column_widths={
                    "Regime": 120,
                    "Trades": 80,
                    "Win Rate": 100,
                    "Net PnL": 100,
                    "Sharpe": 80,
                    "Avg KE%": 80,
                    "Avg ζ%": 80,
                },
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="PnL by Regime",
                target="distribution.pnl_by_regime",
                gridPos={"h": 7, "w": 8, "x": 16, "y": 8},
                orientation="horizontal",
            )
        )
        panel_id += 1

        # Row 4: Layer-1 Time Series
        panels.append(self._row_panel(id=panel_id, title="Layer-1 Sensor History", y_pos=15))
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="KE% & Re% (Energy & Trend)",
                targets=["layer1.ke_pct_series", "layer1.re_m_pct_series"],
                gridPos={"h": 6, "w": 12, "x": 0, "y": 16},
                unit="percentunit",
                fill_opacity=20,
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="ζ% & Hs% (Damping & Entropy)",
                targets=["layer1.zeta_pct_series", "layer1.hs_pct_series"],
                gridPos={"h": 6, "w": 12, "x": 12, "y": 16},
                unit="percentunit",
                fill_opacity=20,
            )
        )
        panel_id += 1

        # Row 5: PE Segment Analysis
        panels.append(self._row_panel(id=panel_id, title="Potential Energy Analysis", y_pos=22))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Performance by PE Segment",
                target="table.trades_by_pe",
                gridPos={"h": 6, "w": 12, "x": 0, "y": 23},
                column_widths={
                    "PE Segment": 100,
                    "Trades": 80,
                    "Win Rate": 100,
                    "Avg PnL": 100,
                    "PE Exploitation": 120,
                },
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="PE Distribution at Entry",
                target="distribution.pe_at_entry",
                gridPos={"h": 6, "w": 12, "x": 12, "y": 14},
                orientation="vertical",
            )
        )
        panel_id += 1

        # Row 4: MFE/MAE Analysis
        panels.append(self._row_panel(id=panel_id, title="Trade Efficiency (MFE/MAE)", y_pos=20))
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="MFE Distribution",
                target="distribution.mfe",
                gridPos={"h": 8, "w": 8, "x": 0, "y": 21},
                orientation="vertical",
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="MAE Distribution",
                target="distribution.mae",
                gridPos={"h": 8, "w": 8, "x": 8, "y": 21},
                orientation="vertical",
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Trade Efficiency Distribution",
                target="distribution.trade_efficiency",
                gridPos={"h": 8, "w": 8, "x": 16, "y": 21},
                orientation="vertical",
            )
        )
        panel_id += 1

        # Row 5: Pareto Analysis (80/20 Rule)
        panels.append(self._row_panel(id=panel_id, title="Pareto Analysis (80/20)", y_pos=29))
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Top 20% Trades Contribution",
                target="pareto.top_20_pct_contribution",
                gridPos={"h": 4, "w": 6, "x": 0, "y": 30},
                unit="percentunit",
                thresholds=[
                    {"color": "blue", "value": None},
                    {"color": "green", "value": 0.5},
                    {"color": "yellow", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Trades for 80% Profit",
                target="pareto.trades_for_80_pct",
                gridPos={"h": 4, "w": 6, "x": 6, "y": 30},
                thresholds=[{"color": "green", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Cumulative PnL Distribution",
                targets=["pareto.cumulative_pnl"],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 30},
                unit="percentunit",
                fill_opacity=30,
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def trade_analysis(self) -> Dict:
        """
        Generate Trade Analysis dashboard.

        Shows individual trade details with filtering and annotations:
        - Trade statistics and efficiency metrics
        - MFE/MAE analysis
        - PE exploitation metrics
        - R-multiple distribution
        - Full trade log with all metrics
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Trade Analysis",
            uid="kinetra-trades",
            tags=["kinetra", "backtest", "trades"],
        )

        # Enable annotations
        dashboard["annotations"] = {
            "list": [
                {
                    "builtIn": 0,
                    "datasource": self._datasource_ref(),
                    "enable": True,
                    "hide": False,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Trade Entries",
                    "target": {"limit": 100, "matchAny": False, "tags": ["entry"], "type": "tags"},
                    "type": "dashboard",
                },
                {
                    "builtIn": 0,
                    "datasource": self._datasource_ref(),
                    "enable": True,
                    "hide": False,
                    "iconColor": "rgba(255, 96, 96, 1)",
                    "name": "Trade Exits",
                    "target": {"limit": 100, "matchAny": False, "tags": ["exit"], "type": "tags"},
                    "type": "dashboard",
                },
            ]
        }

        # Template variables for filtering
        dashboard["templating"]["list"] = [
            self._template_variable(
                name="symbol",
                label="Symbol",
                query="tag_values(symbol)",
                multi=True,
                include_all=True,
            ),
            self._template_variable(
                name="direction",
                label="Direction",
                query="tag_values(direction)",
                multi=False,
                include_all=True,
            ),
            self._template_variable(
                name="regime",
                label="Regime",
                query="tag_values(regime)",
                multi=True,
                include_all=True,
            ),
        ]

        panels = []
        panel_id = 1

        # Row 1: Trade Statistics - 8 compact panels
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Total Trades",
                target="summary.total_trades",
                gridPos={"h": 3, "w": 3, "x": 0, "y": 0},
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Winners",
                target="summary.winning_trades",
                gridPos={"h": 3, "w": 3, "x": 3, "y": 0},
                thresholds=[{"color": "green", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Losers",
                target="summary.losing_trades",
                gridPos={"h": 3, "w": 3, "x": 6, "y": 0},
                thresholds=[{"color": "red", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Avg R-Multiple",
                target="trades.avg_r_multiple",
                gridPos={"h": 3, "w": 3, "x": 9, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0},
                    {"color": "green", "value": 0.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._gauge_panel(
                id=panel_id,
                title="Win Rate",
                target="portfolio.win_rate",
                gridPos={"h": 6, "w": 4, "x": 12, "y": 0},
                min_val=0,
                max_val=1,
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "green", "value": 0.55},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._gauge_panel(
                id=panel_id,
                title="Profit Factor",
                target="portfolio.profit_factor",
                gridPos={"h": 6, "w": 4, "x": 16, "y": 0},
                min_val=0,
                max_val=3,
                unit="none",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1.0},
                    {"color": "green", "value": 1.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._gauge_panel(
                id=panel_id,
                title="MFE Capture",
                target="trades.avg_mfe_capture",
                gridPos={"h": 6, "w": 4, "x": 20, "y": 0},
                min_val=0,
                max_val=1,
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.4},
                    {"color": "green", "value": 0.6},
                ],
            )
        )
        panel_id += 1

        # Row 1 second line: Efficiency metrics
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Avg Trade Efficiency",
                target="trades.avg_efficiency",
                gridPos={"h": 3, "w": 3, "x": 0, "y": 3},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Avg MFE",
                target="trades.avg_mfe",
                gridPos={"h": 3, "w": 3, "x": 3, "y": 3},
                unit="currencyUSD",
                thresholds=[{"color": "green", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Avg MAE",
                target="trades.avg_mae",
                gridPos={"h": 3, "w": 3, "x": 6, "y": 3},
                unit="currencyUSD",
                thresholds=[{"color": "red", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Avg PE Exploitation",
                target="trades.avg_pe_exploitation",
                gridPos={"h": 3, "w": 3, "x": 9, "y": 3},
                unit="percentunit",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.3},
                    {"color": "green", "value": 0.5},
                ],
            )
        )
        panel_id += 1

        # Equity with trade annotations
        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Equity with Trade Markers",
                targets=["portfolio.equity"],
                gridPos={"h": 8, "w": 16, "x": 0, "y": 6},
                unit="currencyUSD",
            )
        )
        panel_id += 1

        # R-Multiple Distribution
        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="R-Multiple Distribution",
                target="distribution.r_multiple",
                gridPos={"h": 8, "w": 8, "x": 16, "y": 6},
                orientation="vertical",
            )
        )
        panel_id += 1

        # Row 2: Trade efficiency scatter
        panels.append(self._row_panel(id=panel_id, title="Trade Quality Analysis", y_pos=14))
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Trade Efficiency by Exit Type",
                target="trades.efficiency_by_exit",
                gridPos={"h": 6, "w": 8, "x": 0, "y": 15},
                orientation="horizontal",
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Win Rate by Regime",
                target="trades.winrate_by_regime",
                gridPos={"h": 6, "w": 8, "x": 8, "y": 15},
                orientation="horizontal",
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Avg PnL by Direction",
                target="trades.pnl_by_direction",
                gridPos={"h": 6, "w": 8, "x": 16, "y": 15},
                orientation="horizontal",
            )
        )
        panel_id += 1

        # Row 3: Full trade list
        panels.append(self._row_panel(id=panel_id, title="All Trades", y_pos=21))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Trade List (filtered)",
                target="table.trades",
                gridPos={"h": 12, "w": 24, "x": 0, "y": 22},
                column_widths={
                    "Entry Time": 150,
                    "Exit Time": 150,
                    "Direction": 80,
                    "Entry Price": 100,
                    "Exit Price": 100,
                    "Net PnL": 100,
                    "R-Multiple": 80,
                    "MFE": 80,
                    "MAE": 80,
                    "Efficiency": 80,
                    "PE at Entry": 80,
                    "Exit Reason": 100,
                },
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def walk_forward_analysis(self) -> Dict:
        """
        Generate Walk-Forward Analysis dashboard.

        Shows in-sample vs out-of-sample performance comparison.
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Walk-Forward Analysis",
            uid="kinetra-walkforward",
            tags=["kinetra", "backtest", "walk-forward"],
        )

        panels = []
        panel_id = 1

        # Header row
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="IS Sharpe",
                target="walkforward.is_sharpe",
                gridPos={"h": 4, "w": 4, "x": 0, "y": 0},
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="OOS Sharpe",
                target="walkforward.oos_sharpe",
                gridPos={"h": 4, "w": 4, "x": 4, "y": 0},
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Robustness Ratio",
                target="walkforward.robustness_ratio",
                gridPos={"h": 4, "w": 4, "x": 8, "y": 0},
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0.5},
                    {"color": "green", "value": 0.8},
                ],
            )
        )
        panel_id += 1

        # Comparison chart placeholder
        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="IS vs OOS Performance",
                targets=["walkforward.is_equity", "walkforward.oos_equity"],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 4},
                unit="currencyUSD",
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def rl_training(self) -> Dict:
        """
        Generate RL Training dashboard.

        Shows real-time RL agent training metrics:
        - Episode progress and rewards
        - Win rate and P&L tracking
        - Physics state gauges (Layer-1 sensors)
        - Feature importance from learned policy
        - MFE/MAE trade quality metrics
        - Loss curves and exploration rate
        """
        dashboard = self._base_dashboard(
            title="Kinetra - RL Training",
            uid="kinetra-rl-training",
            tags=["kinetra", "rl", "training", "live"],
        )

        # Enable auto-refresh for live training
        dashboard["refresh"] = "1s"
        dashboard["time"] = {"from": "now-1h", "to": "now"}

        panels = []
        panel_id = 1

        # Row 1: Training Progress Stats
        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Episode",
                target="rl.episode",
                gridPos={"h": 4, "w": 3, "x": 0, "y": 0},
                thresholds=[{"color": "blue", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Total Trades",
                target="rl.total_trades",
                gridPos={"h": 4, "w": 3, "x": 3, "y": 0},
                thresholds=[{"color": "blue", "value": None}],
            )
        )
        panel_id += 1

        panels.append(
            self._gauge_panel(
                id=panel_id,
                title="Win Rate",
                target="rl.win_rate",
                gridPos={"h": 4, "w": 3, "x": 6, "y": 0},
                min_val=0,
                max_val=100,
                unit="percent",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 40},
                    {"color": "green", "value": 55},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Total P&L",
                target="rl.total_pnl",
                gridPos={"h": 4, "w": 3, "x": 9, "y": 0},
                unit="percent",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 0},
                    {"color": "green", "value": 5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Epsilon (Exploration)",
                target="rl.epsilon",
                gridPos={"h": 4, "w": 3, "x": 12, "y": 0},
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.1},
                    {"color": "red", "value": 0.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="Loss",
                target="rl.loss",
                gridPos={"h": 4, "w": 3, "x": 15, "y": 0},
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.01},
                    {"color": "red", "value": 0.1},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._gauge_panel(
                id=panel_id,
                title="MFE/MAE Ratio",
                target="rl.mfe_mae_ratio",
                gridPos={"h": 4, "w": 3, "x": 18, "y": 0},
                min_val=0,
                max_val=3,
                unit="none",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 1.0},
                    {"color": "green", "value": 1.5},
                ],
            )
        )
        panel_id += 1

        panels.append(
            self._stat_panel(
                id=panel_id,
                title="MFE Captured",
                target="rl.mfe_captured",
                gridPos={"h": 4, "w": 3, "x": 21, "y": 0},
                unit="percent",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 30},
                    {"color": "green", "value": 50},
                ],
            )
        )
        panel_id += 1

        # Row 2: Physics State Gauges
        panels.append(self._row_panel(id=panel_id, title="Current Physics State", y_pos=4))
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="KE% (Energy)",
                target="rl.physics.ke_pct",
                gridPos={"h": 4, "w": 4, "x": 0, "y": 5},
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="Re% (Trend)",
                target="rl.physics.re_m_pct",
                gridPos={"h": 4, "w": 4, "x": 4, "y": 5},
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="ζ% (Damping)",
                target="rl.physics.zeta_pct",
                gridPos={"h": 4, "w": 4, "x": 8, "y": 5},
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="Hs% (Entropy)",
                target="rl.physics.hs_pct",
                gridPos={"h": 4, "w": 4, "x": 12, "y": 5},
            )
        )
        panel_id += 1

        panels.append(
            self._bargauge_panel(
                id=panel_id,
                title="Buying Pressure",
                target="rl.physics.bp",
                gridPos={"h": 4, "w": 4, "x": 16, "y": 5},
            )
        )
        panel_id += 1

        panels.append(
            self._state_timeline_panel(
                id=panel_id,
                title="Regime",
                target="rl.physics.regime",
                gridPos={"h": 4, "w": 4, "x": 20, "y": 5},
            )
        )
        panel_id += 1

        # Row 3: Training Curves
        panels.append(self._row_panel(id=panel_id, title="Training Curves", y_pos=9))
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Episode Reward",
                targets=["rl.episode_reward"],
                gridPos={"h": 6, "w": 8, "x": 0, "y": 10},
                fill_opacity=30,
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Cumulative P&L",
                targets=["rl.cumulative_pnl"],
                gridPos={"h": 6, "w": 8, "x": 8, "y": 10},
                unit="percent",
                fill_opacity=30,
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Loss & Epsilon",
                targets=["rl.loss_history", "rl.epsilon_history"],
                gridPos={"h": 6, "w": 8, "x": 16, "y": 10},
                fill_opacity=20,
            )
        )
        panel_id += 1

        # Row 4: Feature Importance & Trade Quality
        panels.append(self._row_panel(id=panel_id, title="Learned Policy Analysis", y_pos=16))
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Feature Importance (Learned)",
                target="rl.feature_importance",
                gridPos={"h": 8, "w": 12, "x": 0, "y": 17},
                orientation="horizontal",
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="MFE vs MAE per Trade",
                targets=["rl.mfe_history", "rl.mae_history"],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 17},
                fill_opacity=20,
            )
        )
        panel_id += 1

        # Row 5: Action Distribution
        panels.append(self._row_panel(id=panel_id, title="Action Analysis", y_pos=25))
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Action Distribution",
                target="rl.action_distribution",
                gridPos={"h": 6, "w": 8, "x": 0, "y": 26},
                orientation="vertical",
            )
        )
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Recent Trades",
                target="rl.recent_trades",
                gridPos={"h": 6, "w": 16, "x": 8, "y": 26},
                column_widths={
                    "Time": 150,
                    "Action": 80,
                    "Entry": 100,
                    "Exit": 100,
                    "PnL": 80,
                    "MFE": 80,
                    "MAE": 80,
                    "Regime": 100,
                },
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def regime_analysis(self) -> Dict:
        """
        Generate Regime Analysis dashboard.

        Dedicated dashboard for GMM regime clustering analysis:
        - Regime distribution and transitions
        - Performance breakdown by regime
        - Sensor correlations within regimes
        - Regime prediction accuracy
        """
        dashboard = self._base_dashboard(
            title="Kinetra - Regime Analysis",
            uid="kinetra-regimes",
            tags=["kinetra", "backtest", "regimes", "gmm"],
        )

        panels = []
        panel_id = 1

        # Row 1: Regime Distribution
        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Regime Distribution",
                target="regime.distribution",
                gridPos={"h": 6, "w": 8, "x": 0, "y": 0},
                orientation="vertical",
            )
        )
        panel_id += 1

        panels.append(
            self._state_timeline_panel(
                id=panel_id,
                title="Regime Timeline",
                target="regime.timeline",
                gridPos={"h": 6, "w": 16, "x": 8, "y": 0},
            )
        )
        panel_id += 1

        # Row 2: Performance by Regime
        panels.append(self._row_panel(id=panel_id, title="Performance by Regime", y_pos=6))
        panel_id += 1

        panels.append(
            self._table_panel(
                id=panel_id,
                title="Regime Performance Summary",
                target="table.regime_performance",
                gridPos={"h": 8, "w": 14, "x": 0, "y": 7},
                column_widths={
                    "Regime": 120,
                    "Bars": 80,
                    "% Time": 80,
                    "Trades": 80,
                    "Win Rate": 100,
                    "Net PnL": 100,
                    "Sharpe": 80,
                    "Avg Duration": 100,
                },
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Sharpe by Regime",
                target="regime.sharpe_by_regime",
                gridPos={"h": 8, "w": 5, "x": 14, "y": 7},
                orientation="horizontal",
            )
        )
        panel_id += 1

        panels.append(
            self._bar_chart_panel(
                id=panel_id,
                title="Win Rate by Regime",
                target="regime.winrate_by_regime",
                gridPos={"h": 8, "w": 5, "x": 19, "y": 7},
                orientation="horizontal",
            )
        )
        panel_id += 1

        # Row 3: Regime Sensor Profiles
        panels.append(self._row_panel(id=panel_id, title="Regime Sensor Profiles", y_pos=15))
        panel_id += 1

        # Sensor averages per regime
        panels.append(
            self._table_panel(
                id=panel_id,
                title="Average Sensors by Regime",
                target="table.regime_sensors",
                gridPos={"h": 6, "w": 24, "x": 0, "y": 16},
                column_widths={
                    "Regime": 120,
                    "Avg KE%": 80,
                    "Avg Re%": 80,
                    "Avg ζ%": 80,
                    "Avg Hs%": 80,
                    "Avg PE%": 80,
                    "Avg η%": 80,
                    "Avg BP": 80,
                },
            )
        )
        panel_id += 1

        # Row 4: Regime Transitions
        panels.append(self._row_panel(id=panel_id, title="Regime Transitions", y_pos=22))
        panel_id += 1

        panels.append(
            self._heatmap_panel(
                id=panel_id,
                title="Transition Matrix",
                target="regime.transition_matrix",
                gridPos={"h": 8, "w": 12, "x": 0, "y": 23},
            )
        )
        panel_id += 1

        panels.append(
            self._timeseries_panel(
                id=panel_id,
                title="Regime Age (Time in Current Regime)",
                targets=["regime.age_frac"],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 23},
                unit="percentunit",
                fill_opacity=30,
            )
        )
        panel_id += 1

        dashboard["panels"] = panels
        return dashboard

    def export_all(self, output_dir: str) -> List[str]:
        """
        Export all dashboards to JSON files.

        Args:
            output_dir: Directory to save dashboard JSON files

        Returns:
            List of created file paths
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        dashboards = [
            ("portfolio_overview.json", self.portfolio_overview()),
            ("instrument_drilldown.json", self.instrument_drilldown()),
            ("physics_analysis.json", self.physics_analysis()),
            ("trade_analysis.json", self.trade_analysis()),
            ("walk_forward_analysis.json", self.walk_forward_analysis()),
            ("rl_training.json", self.rl_training()),
            ("regime_analysis.json", self.regime_analysis()),
        ]

        created_files = []
        for filename, dashboard in dashboards:
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(dashboard, f, indent=2, default=str)
            created_files.append(filepath)

        return created_files

    def get_all_dashboards(self) -> Dict[str, Dict]:
        """Get all dashboard configurations as a dictionary."""
        return {
            "portfolio_overview": self.portfolio_overview(),
            "instrument_drilldown": self.instrument_drilldown(),
            "physics_analysis": self.physics_analysis(),
            "trade_analysis": self.trade_analysis(),
            "walk_forward_analysis": self.walk_forward_analysis(),
            "rl_training": self.rl_training(),
            "regime_analysis": self.regime_analysis(),
        }

    def to_json(self, dashboard: Dict, pretty: bool = True) -> str:
        """Convert dashboard to JSON string."""
        if pretty:
            return json.dumps(dashboard, indent=2, default=str)
        return json.dumps(dashboard, default=str)
