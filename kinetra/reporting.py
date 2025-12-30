"""
Comprehensive Reporting Engine for Backtesting

Generates detailed statistics and distributions:
- Per instrument analysis
- Per timeframe analysis
- Per instrument class analysis
- Portfolio-level aggregates
- Distribution analysis
- Kinetra physics metrics
- AI-consumable big data outputs
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .backtest_engine import BacktestResult, Trade
from .portfolio_backtest import (
    InstrumentClass,
    PortfolioBacktestResult,
    PortfolioTrade,
)


@dataclass
class DistributionStats:
    """Statistical distribution analysis."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0

    # Percentiles
    p5: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0

    # Shape
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Normality test
    shapiro_stat: float = 0.0
    shapiro_pvalue: float = 0.0
    is_normal: bool = False

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "p5": self.p5,
            "p10": self.p10,
            "p25": self.p25,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "is_normal": self.is_normal,
        }


@dataclass
class TradeAnalysis:
    """Detailed trade analysis."""

    # Basic counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win/loss
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Average trade
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Largest trades
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0

    # Holding time
    avg_holding_bars: float = 0.0
    avg_winning_holding_bars: float = 0.0
    avg_losing_holding_bars: float = 0.0

    # MFE/MAE analysis
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_capture_ratio: float = 0.0  # How much of MFE was captured
    mae_recovery_ratio: float = 0.0  # How much MAE was recovered

    # Distributions
    pnl_distribution: Optional[DistributionStats] = None
    mfe_distribution: Optional[DistributionStats] = None
    mae_distribution: Optional[DistributionStats] = None
    holding_time_distribution: Optional[DistributionStats] = None

    def to_dict(self) -> Dict:
        result = {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "avg_trade": self.avg_trade,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_mfe": self.avg_mfe,
            "avg_mae": self.avg_mae,
            "mfe_capture_ratio": self.mfe_capture_ratio,
        }

        if self.pnl_distribution:
            result["pnl_distribution"] = self.pnl_distribution.to_dict()
        if self.mfe_distribution:
            result["mfe_distribution"] = self.mfe_distribution.to_dict()
        if self.mae_distribution:
            result["mae_distribution"] = self.mae_distribution.to_dict()

        return result


@dataclass
class RiskMetrics:
    """Risk and performance metrics."""

    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Volatility
    daily_volatility: float = 0.0
    annualized_volatility: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    avg_drawdown: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Tail risk
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Omega ratio
    omega_ratio: float = 0.0

    # Recovery
    recovery_factor: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "daily_volatility": self.daily_volatility,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "omega_ratio": self.omega_ratio,
            "recovery_factor": self.recovery_factor,
        }


@dataclass
class PhysicsMetrics:
    """Kinetra physics-based metrics."""

    # Energy metrics
    avg_energy_at_entry: float = 0.0
    avg_energy_winners: float = 0.0
    avg_energy_losers: float = 0.0
    energy_edge: float = 0.0  # Winners vs losers energy difference

    # Regime performance
    trades_underdamped: int = 0
    trades_critical: int = 0
    trades_overdamped: int = 0
    win_rate_underdamped: float = 0.0
    win_rate_critical: float = 0.0
    win_rate_overdamped: float = 0.0
    pnl_underdamped: float = 0.0
    pnl_critical: float = 0.0
    pnl_overdamped: float = 0.0

    # Energy capture
    total_energy_captured: float = 0.0
    energy_capture_efficiency: float = 0.0

    # Z-factor (statistical edge)
    z_factor: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "avg_energy_at_entry": self.avg_energy_at_entry,
            "avg_energy_winners": self.avg_energy_winners,
            "avg_energy_losers": self.avg_energy_losers,
            "energy_edge": self.energy_edge,
            "trades_by_regime": {
                "underdamped": self.trades_underdamped,
                "critical": self.trades_critical,
                "overdamped": self.trades_overdamped,
            },
            "win_rate_by_regime": {
                "underdamped": self.win_rate_underdamped,
                "critical": self.win_rate_critical,
                "overdamped": self.win_rate_overdamped,
            },
            "pnl_by_regime": {
                "underdamped": self.pnl_underdamped,
                "critical": self.pnl_critical,
                "overdamped": self.pnl_overdamped,
            },
            "energy_capture_efficiency": self.energy_capture_efficiency,
            "z_factor": self.z_factor,
        }


@dataclass
class InstrumentReport:
    """Complete report for a single instrument."""

    symbol: str
    instrument_class: InstrumentClass = InstrumentClass.OTHER
    timeframe: str = ""

    trade_analysis: Optional[TradeAnalysis] = None
    risk_metrics: Optional[RiskMetrics] = None
    physics_metrics: Optional[PhysicsMetrics] = None

    # Time-based analysis
    performance_by_hour: Optional[Dict[int, float]] = None
    performance_by_day: Optional[Dict[int, float]] = None  # 0=Mon, 6=Sun
    performance_by_month: Optional[Dict[int, float]] = None

    def to_dict(self) -> Dict:
        result = {
            "symbol": self.symbol,
            "instrument_class": self.instrument_class.value,
            "timeframe": self.timeframe,
        }

        if self.trade_analysis:
            result["trade_analysis"] = self.trade_analysis.to_dict()
        if self.risk_metrics:
            result["risk_metrics"] = self.risk_metrics.to_dict()
        if self.physics_metrics:
            result["physics_metrics"] = self.physics_metrics.to_dict()
        if self.performance_by_hour:
            result["performance_by_hour"] = self.performance_by_hour
        if self.performance_by_day:
            result["performance_by_day"] = self.performance_by_day

        return result


@dataclass
class PortfolioReport:
    """Complete portfolio report."""

    # Summary
    initial_capital: float = 0.0
    final_capital: float = 0.0

    # Aggregate metrics
    trade_analysis: Optional[TradeAnalysis] = None
    risk_metrics: Optional[RiskMetrics] = None
    physics_metrics: Optional[PhysicsMetrics] = None

    # Per-instrument reports
    instrument_reports: Dict[str, InstrumentReport] = field(default_factory=dict)

    # Per-timeframe reports
    timeframe_reports: Dict[str, TradeAnalysis] = field(default_factory=dict)

    # Per-class reports
    class_reports: Dict[InstrumentClass, TradeAnalysis] = field(default_factory=dict)

    # Correlation analysis
    instrument_correlations: Optional[pd.DataFrame] = None

    # Equity curve data
    equity_curve: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        result = {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_capital": self.final_capital,
                "total_return_pct": ((self.final_capital / self.initial_capital) - 1) * 100
                if self.initial_capital > 0
                else 0,
            }
        }

        if self.trade_analysis:
            result["trade_analysis"] = self.trade_analysis.to_dict()
        if self.risk_metrics:
            result["risk_metrics"] = self.risk_metrics.to_dict()
        if self.physics_metrics:
            result["physics_metrics"] = self.physics_metrics.to_dict()

        result["instruments"] = {k: v.to_dict() for k, v in self.instrument_reports.items()}
        result["timeframes"] = {k: v.to_dict() for k, v in self.timeframe_reports.items()}
        result["classes"] = {k.value: v.to_dict() for k, v in self.class_reports.items()}

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to flat DataFrame for AI consumption."""
        rows = []

        # Portfolio level
        row = {"level": "portfolio", "symbol": "ALL", "timeframe": "ALL", "class": "ALL"}
        if self.trade_analysis:
            row.update(self.trade_analysis.to_dict())
        if self.risk_metrics:
            row.update(self.risk_metrics.to_dict())
        if self.physics_metrics:
            for k, v in self.physics_metrics.to_dict().items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        row[f"{k}_{kk}"] = vv
                else:
                    row[k] = v
        rows.append(row)

        # Per instrument
        for symbol, report in self.instrument_reports.items():
            row = {
                "level": "instrument",
                "symbol": symbol,
                "timeframe": report.timeframe,
                "class": report.instrument_class.value,
            }
            if report.trade_analysis:
                row.update(report.trade_analysis.to_dict())
            if report.risk_metrics:
                row.update(report.risk_metrics.to_dict())
            rows.append(row)

        return pd.DataFrame(rows)


class ReportingEngine:
    """
    Comprehensive reporting engine for backtest analysis.

    Generates detailed statistics, distributions, and AI-consumable outputs.
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def analyze_distribution(self, values: List[float]) -> DistributionStats:
        """Analyze statistical distribution of values."""
        if not values or len(values) < 2:
            return DistributionStats()

        arr = np.array(values)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 2:
            return DistributionStats()

        dist = DistributionStats(
            count=len(arr),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            p5=float(np.percentile(arr, 5)),
            p10=float(np.percentile(arr, 10)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            skewness=float(stats.skew(arr)) if len(arr) > 2 else 0,
            kurtosis=float(stats.kurtosis(arr)) if len(arr) > 2 else 0,
        )

        # Normality test (only for reasonable sample sizes)
        if 3 <= len(arr) <= 5000:
            try:
                stat, pvalue = stats.shapiro(arr)
                dist.shapiro_stat = float(stat)
                dist.shapiro_pvalue = float(pvalue)
                dist.is_normal = pvalue > 0.05
            except Exception:
                pass

        return dist

    def analyze_trades(self, trades: List[Trade]) -> TradeAnalysis:
        """Analyze a list of trades."""
        if not trades:
            return TradeAnalysis()

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return TradeAnalysis()

        winners = [t for t in closed_trades if t.net_pnl > 0]
        losers = [t for t in closed_trades if t.net_pnl < 0]
        breakeven = [t for t in closed_trades if t.net_pnl == 0]

        analysis = TradeAnalysis(
            total_trades=len(closed_trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            breakeven_trades=len(breakeven),
            win_rate=len(winners) / len(closed_trades) if closed_trades else 0,
            loss_rate=len(losers) / len(closed_trades) if closed_trades else 0,
            gross_profit=sum(t.gross_pnl for t in winners),
            gross_loss=abs(sum(t.gross_pnl for t in losers)),
            net_profit=sum(t.net_pnl for t in closed_trades),
        )

        # Profit factor
        if analysis.gross_loss > 0:
            analysis.profit_factor = analysis.gross_profit / analysis.gross_loss
        else:
            analysis.profit_factor = float("inf") if analysis.gross_profit > 0 else 0

        # Averages
        analysis.avg_trade = np.mean([t.net_pnl for t in closed_trades])
        analysis.avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
        analysis.avg_loss = abs(np.mean([t.net_pnl for t in losers])) if losers else 0

        if analysis.avg_loss > 0:
            analysis.avg_win_loss_ratio = analysis.avg_win / analysis.avg_loss

        # Largest trades
        pnls = [t.net_pnl for t in closed_trades]
        analysis.largest_win = max(pnls) if pnls else 0
        analysis.largest_loss = min(pnls) if pnls else 0

        # Streaks
        analysis.max_consecutive_wins, analysis.max_consecutive_losses = self._calculate_streaks(
            [t.net_pnl > 0 for t in closed_trades]
        )

        # MFE/MAE
        mfes = [t.mfe for t in closed_trades if t.mfe > 0]
        maes = [t.mae for t in closed_trades if t.mae > 0]

        analysis.avg_mfe = np.mean(mfes) if mfes else 0
        analysis.avg_mae = np.mean(maes) if maes else 0

        # MFE capture ratio
        total_mfe = sum(t.mfe for t in closed_trades)
        total_captured = sum(max(0, t.gross_pnl) for t in closed_trades)
        analysis.mfe_capture_ratio = total_captured / total_mfe if total_mfe > 0 else 0

        # Distributions
        analysis.pnl_distribution = self.analyze_distribution(pnls)
        analysis.mfe_distribution = self.analyze_distribution(mfes)
        analysis.mae_distribution = self.analyze_distribution(maes)

        return analysis

    def analyze_risk(
        self,
        equity_curve: pd.Series,
        initial_capital: float,
        risk_free_rate: float = 0.02,
    ) -> RiskMetrics:
        """Analyze risk metrics from equity curve."""
        if len(equity_curve) < 2:
            return RiskMetrics()

        metrics = RiskMetrics()

        # Returns
        metrics.total_return = equity_curve.iloc[-1] - initial_capital
        metrics.total_return_pct = (metrics.total_return / initial_capital) * 100

        # Daily returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return metrics

        # Volatility
        metrics.daily_volatility = returns.std()
        metrics.annualized_volatility = metrics.daily_volatility * np.sqrt(252)

        # Annualized return (assuming 252 trading days)
        n_periods = len(returns)
        total_return_factor = equity_curve.iloc[-1] / initial_capital
        metrics.annualized_return = (total_return_factor ** (252 / n_periods) - 1) * 100

        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve - rolling_max
        metrics.max_drawdown = abs(drawdown.min())

        if rolling_max[drawdown.idxmin()] > 0:
            metrics.max_drawdown_pct = abs(drawdown.min() / rolling_max[drawdown.idxmin()]) * 100

        metrics.avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0

        # Sharpe ratio
        excess_return = returns.mean() - (risk_free_rate / 252)
        if returns.std() > 0:
            metrics.sharpe_ratio = (excess_return / returns.std()) * np.sqrt(252)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_ratio = (excess_return / downside_returns.std()) * np.sqrt(252)

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

        # VaR and CVaR
        metrics.var_95 = abs(np.percentile(returns, 5)) * initial_capital
        metrics.var_99 = abs(np.percentile(returns, 1)) * initial_capital

        tail_returns = returns[returns <= np.percentile(returns, 5)]
        metrics.cvar_95 = abs(tail_returns.mean()) * initial_capital if len(tail_returns) > 0 else 0

        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        metrics.omega_ratio = gains / losses if losses > 0 else float("inf")

        # Recovery factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

        return metrics

    def analyze_physics(self, trades: List[Trade]) -> PhysicsMetrics:
        """Analyze Kinetra physics metrics."""
        if not trades:
            return PhysicsMetrics()

        closed_trades = [t for t in trades if t.is_closed]
        if not closed_trades:
            return PhysicsMetrics()

        winners = [t for t in closed_trades if t.net_pnl > 0]
        losers = [t for t in closed_trades if t.net_pnl < 0]

        metrics = PhysicsMetrics()

        # Energy analysis
        energies = [t.energy_at_entry for t in closed_trades if t.energy_at_entry > 0]
        winner_energies = [t.energy_at_entry for t in winners if t.energy_at_entry > 0]
        loser_energies = [t.energy_at_entry for t in losers if t.energy_at_entry > 0]

        metrics.avg_energy_at_entry = np.mean(energies) if energies else 0
        metrics.avg_energy_winners = np.mean(winner_energies) if winner_energies else 0
        metrics.avg_energy_losers = np.mean(loser_energies) if loser_energies else 0
        metrics.energy_edge = metrics.avg_energy_winners - metrics.avg_energy_losers

        # Regime analysis
        regimes = {"underdamped": [], "critical": [], "overdamped": []}

        for t in closed_trades:
            regime = t.regime_at_entry.lower() if t.regime_at_entry else "unknown"
            if regime in regimes:
                regimes[regime].append(t)

        metrics.trades_underdamped = len(regimes["underdamped"])
        metrics.trades_critical = len(regimes["critical"])
        metrics.trades_overdamped = len(regimes["overdamped"])

        for regime, regime_trades in regimes.items():
            if regime_trades:
                wins = len([t for t in regime_trades if t.net_pnl > 0])
                pnl = sum(t.net_pnl for t in regime_trades)

                if regime == "underdamped":
                    metrics.win_rate_underdamped = wins / len(regime_trades)
                    metrics.pnl_underdamped = pnl
                elif regime == "critical":
                    metrics.win_rate_critical = wins / len(regime_trades)
                    metrics.pnl_critical = pnl
                elif regime == "overdamped":
                    metrics.win_rate_overdamped = wins / len(regime_trades)
                    metrics.pnl_overdamped = pnl

        # Energy capture
        total_energy = sum(t.energy_at_entry for t in closed_trades)
        winning_energy = sum(t.energy_at_entry for t in winners)
        metrics.total_energy_captured = winning_energy
        metrics.energy_capture_efficiency = winning_energy / total_energy if total_energy > 0 else 0

        # Z-factor
        if len(closed_trades) > 1 and winners and losers:
            win_rate = len(winners) / len(closed_trades)
            avg_win = np.mean([t.net_pnl for t in winners])
            avg_loss = abs(np.mean([t.net_pnl for t in losers]))

            if avg_loss > 0:
                metrics.z_factor = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss

        return metrics

    def _calculate_streaks(self, results: List[bool]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        if not results:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for is_win in results:
            if is_win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def generate_report(
        self,
        result: Union[BacktestResult, PortfolioBacktestResult],
        initial_capital: float = None,
    ) -> PortfolioReport:
        """
        Generate comprehensive report from backtest result.

        Args:
            result: Backtest result (single or portfolio)
            initial_capital: Initial capital (uses self.initial_capital if None)

        Returns:
            PortfolioReport with all analysis
        """
        if initial_capital is None:
            initial_capital = self.initial_capital

        report = PortfolioReport(initial_capital=initial_capital)

        if isinstance(result, PortfolioBacktestResult):
            return self._generate_portfolio_report(result, report)
        else:
            return self._generate_single_report(result, report)

    def _generate_single_report(
        self,
        result: BacktestResult,
        report: PortfolioReport,
    ) -> PortfolioReport:
        """Generate report for single-instrument backtest."""

        report.trade_analysis = self.analyze_trades(result.trades)

        if result.equity_curve is not None:
            report.equity_curve = result.equity_curve
            report.risk_metrics = self.analyze_risk(
                result.equity_curve,
                report.initial_capital,
            )
            report.final_capital = result.equity_curve.iloc[-1]

            # Drawdown curve
            rolling_max = result.equity_curve.expanding().max()
            report.drawdown_curve = result.equity_curve - rolling_max

        report.physics_metrics = self.analyze_physics(result.trades)

        return report

    def _generate_portfolio_report(
        self,
        result: PortfolioBacktestResult,
        report: PortfolioReport,
    ) -> PortfolioReport:
        """Generate report for portfolio backtest."""

        # Portfolio-level analysis
        report.trade_analysis = self.analyze_trades(result.trades)

        if result.equity_curve is not None:
            report.equity_curve = result.equity_curve
            report.risk_metrics = self.analyze_risk(
                result.equity_curve,
                report.initial_capital,
            )
            report.final_capital = result.equity_curve.iloc[-1]

            rolling_max = result.equity_curve.expanding().max()
            report.drawdown_curve = result.equity_curve - rolling_max

        report.physics_metrics = self.analyze_physics(result.trades)

        # Per-instrument reports
        for symbol, inst_result in result.instrument_results.items():
            inst_report = InstrumentReport(symbol=symbol)
            inst_report.trade_analysis = self.analyze_trades(inst_result.trades)

            # Find instrument class and timeframe from trades
            inst_trades = [t for t in result.trades if t.symbol == symbol]
            if inst_trades and isinstance(inst_trades[0], PortfolioTrade):
                inst_report.instrument_class = inst_trades[0].instrument_class
                inst_report.timeframe = inst_trades[0].timeframe

            report.instrument_reports[symbol] = inst_report

        # Per-timeframe analysis
        for tf, tf_result in result.timeframe_results.items():
            report.timeframe_reports[tf] = self.analyze_trades(tf_result.trades)

        # Per-class analysis
        for cls, cls_result in result.class_results.items():
            report.class_reports[cls] = self.analyze_trades(cls_result.trades)

        return report

    def export_to_json(self, report: PortfolioReport) -> str:
        """Export report to JSON string."""
        import json

        return json.dumps(report.to_dict(), indent=2, default=str)

    def export_to_dataframe(self, report: PortfolioReport) -> pd.DataFrame:
        """Export report to flat DataFrame for AI/ML consumption."""
        return report.to_dataframe()

    def export_trades_dataframe(
        self,
        trades: List[Trade],
        include_physics: bool = True,
    ) -> pd.DataFrame:
        """Export all trades to detailed DataFrame."""
        rows = []

        for t in trades:
            row = {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "direction": t.direction.value,
                "lots": t.lots,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "spread_cost": t.spread_cost,
                "commission": t.commission,
                "slippage": t.slippage,
                "swap_cost": t.swap_cost,
                "total_cost": t.total_cost,
                "mfe": t.mfe,
                "mae": t.mae,
                "is_winner": t.net_pnl > 0,
            }

            if include_physics:
                row["energy_at_entry"] = t.energy_at_entry
                row["regime_at_entry"] = t.regime_at_entry

            if isinstance(t, PortfolioTrade):
                row["timeframe"] = t.timeframe
                row["instrument_class"] = t.instrument_class.value
                row["portfolio_equity_at_entry"] = t.portfolio_equity_at_entry

            rows.append(row)

        return pd.DataFrame(rows)
