"""
Stress Testing Framework

Run backtests under stressed market conditions:
- High volatility
- Flash crash scenarios
- Liquidity crises
- Black swan events

Applies realistic cost adjustments based on historical event data.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine, BacktestResult, Trade
from .market_events import MARKET_EVENT_CALENDAR, EventType, MarketEvent, MarketEventCalendar
from .symbol_spec import SymbolSpec


@dataclass
class StressTestScenario:
    """
    A stress test scenario with cost multipliers.

    Defines how costs are amplified during stressed conditions.
    """

    name: str
    description: str = ""

    # Cost multipliers
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    commission_multiplier: float = 1.0

    # Market condition modifiers
    volatility_multiplier: float = 1.0
    liquidity_factor: float = 1.0  # 1.0 = normal, 0.5 = reduced fills

    # Gap injection
    inject_gaps: bool = False
    gap_frequency: float = 0.0  # Probability per bar
    gap_size_atr: float = 2.0  # Gap size in ATR units

    # Partial fills (for larger positions)
    max_fill_pct: float = 1.0  # 1.0 = 100% fill, 0.5 = 50% max fill

    def apply_to_trade(self, trade: Trade, base_spec: SymbolSpec) -> Trade:
        """Apply scenario costs to a trade."""
        stressed_trade = copy.copy(trade)

        # Amplify costs
        if trade.spread_cost:
            stressed_trade.spread_cost = trade.spread_cost * self.spread_multiplier
        if trade.slippage:
            stressed_trade.slippage = trade.slippage * self.slippage_multiplier
        if trade.commission:
            stressed_trade.commission = trade.commission * self.commission_multiplier

        # Recalculate net P&L (total_cost is a property, calculated automatically)
        stressed_trade.net_pnl = (stressed_trade.gross_pnl or 0) - stressed_trade.total_cost

        return stressed_trade


# === Pre-defined Stress Scenarios ===

STRESS_SCENARIOS = {
    "normal": StressTestScenario(
        name="Normal",
        description="Baseline conditions, no stress applied",
        spread_multiplier=1.0,
        slippage_multiplier=1.0,
        volatility_multiplier=1.0,
        liquidity_factor=1.0,
    ),
    "high_volatility": StressTestScenario(
        name="High Volatility",
        description="VIX-like spike, 2x normal volatility",
        spread_multiplier=2.0,
        slippage_multiplier=2.0,
        volatility_multiplier=2.0,
        liquidity_factor=0.9,
    ),
    "low_liquidity": StressTestScenario(
        name="Low Liquidity",
        description="Holiday/off-hours thin market",
        spread_multiplier=3.0,
        slippage_multiplier=2.5,
        volatility_multiplier=1.2,
        liquidity_factor=0.5,
        max_fill_pct=0.7,
    ),
    "flash_crash": StressTestScenario(
        name="Flash Crash",
        description="Rapid market dislocation, extreme spreads",
        spread_multiplier=10.0,
        slippage_multiplier=5.0,
        volatility_multiplier=5.0,
        liquidity_factor=0.3,
        inject_gaps=True,
        gap_frequency=0.05,  # 5% of bars have gaps
        gap_size_atr=3.0,
        max_fill_pct=0.5,
    ),
    "liquidity_crisis": StressTestScenario(
        name="Liquidity Crisis",
        description="Credit event, widespread deleveraging",
        spread_multiplier=5.0,
        slippage_multiplier=3.0,
        volatility_multiplier=3.0,
        liquidity_factor=0.4,
        inject_gaps=True,
        gap_frequency=0.02,
        gap_size_atr=2.0,
        max_fill_pct=0.6,
    ),
    "black_swan": StressTestScenario(
        name="Black Swan",
        description="CHF 2015 / COVID-like extreme event",
        spread_multiplier=20.0,
        slippage_multiplier=10.0,
        volatility_multiplier=10.0,
        liquidity_factor=0.1,
        inject_gaps=True,
        gap_frequency=0.10,
        gap_size_atr=5.0,
        max_fill_pct=0.3,
    ),
    "news_spike": StressTestScenario(
        name="News Spike",
        description="Major news release (NFP, FOMC)",
        spread_multiplier=3.0,
        slippage_multiplier=2.0,
        volatility_multiplier=2.5,
        liquidity_factor=0.7,
        inject_gaps=True,
        gap_frequency=0.01,
        gap_size_atr=1.5,
    ),
    "weekend_gap": StressTestScenario(
        name="Weekend Gap",
        description="Gap risk over weekend/holiday",
        spread_multiplier=1.5,
        slippage_multiplier=1.5,
        volatility_multiplier=1.0,
        liquidity_factor=1.0,
        inject_gaps=True,
        gap_frequency=0.005,  # Rare but large
        gap_size_atr=4.0,
    ),
}


@dataclass
class StressTestResult:
    """Results from a stress test run."""

    scenario: StressTestScenario
    backtest_result: BacktestResult

    # Comparison metrics
    baseline_pnl: float = 0.0
    stressed_pnl: float = 0.0
    pnl_degradation: float = 0.0  # How much worse under stress

    # Cost impact
    baseline_costs: float = 0.0
    stressed_costs: float = 0.0
    cost_increase_pct: float = 0.0

    # Gap impact
    gaps_hit: int = 0
    gap_losses: float = 0.0

    # Survival metrics
    max_drawdown: float = 0.0
    survived: bool = True  # Did not blow up


@dataclass
class StressTestReport:
    """Comprehensive stress test report across all scenarios."""

    symbol: str = ""
    baseline_result: Optional[BacktestResult] = None
    scenario_results: Dict[str, StressTestResult] = field(default_factory=dict)

    # Aggregate metrics
    worst_case_pnl: float = 0.0
    worst_case_scenario: str = ""
    avg_pnl_degradation: float = 0.0
    survival_rate: float = 0.0  # % of scenarios survived

    # Robustness score (0-100)
    robustness_score: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.baseline_result is not None:
            baseline_pnl_str = f"  Net P&L: ${self.baseline_result.total_net_pnl:.2f}"
            baseline_trades_str = f"  Trades: {self.baseline_result.total_trades}"
        else:
            baseline_pnl_str = "  N/A"
            baseline_trades_str = "  N/A"

        lines = [
            f"Stress Test Report: {self.symbol}",
            f"=" * 50,
            f"",
            f"Baseline Performance:",
            baseline_pnl_str,
            baseline_trades_str,
            f"",
            f"Robustness Score: {self.robustness_score:.1f}/100",
            f"Survival Rate: {self.survival_rate:.1%}",
            f"",
            f"Scenario Results:",
        ]

        for name, result in sorted(
            self.scenario_results.items(),
            key=lambda x: x[1].stressed_pnl,
        ):
            status = "OK" if result.survived else "BLOWN UP"
            lines.append(
                f"  {name}: ${result.stressed_pnl:,.2f} ({result.pnl_degradation:+.1%}) [{status}]"
            )

        lines.extend(
            [
                f"",
                f"Worst Case: {self.worst_case_scenario}",
                f"  P&L: ${self.worst_case_pnl:,.2f}",
            ]
        )

        return "\n".join(lines)


class StressTestEngine:
    """
    Stress testing engine for trading strategies.

    Runs backtests under various stress scenarios and compares
    performance to baseline conditions.

    Usage:
        engine = StressTestEngine()

        # Run all scenarios
        report = engine.run_all_scenarios(
            data=data,
            signal_func=my_strategy,
            symbol_spec=spec,
        )

        # Run specific scenario
        result = engine.run_scenario(
            data=data,
            signal_func=my_strategy,
            symbol_spec=spec,
            scenario="flash_crash",
        )
    """

    def __init__(
        self,
        scenarios: Dict[str, StressTestScenario] = None,
        event_calendar: MarketEventCalendar = None,
    ):
        self.scenarios = scenarios or STRESS_SCENARIOS
        self.event_calendar = event_calendar or MARKET_EVENT_CALENDAR

    def run_scenario(
        self,
        data: pd.DataFrame,
        signal_func: Callable,
        symbol_spec: SymbolSpec,
        scenario: str,
        baseline_result: BacktestResult = None,
        initial_capital: float = 10000.0,
        **backtest_kwargs,
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Args:
            data: OHLCV data
            signal_func: Strategy signal function
            symbol_spec: Instrument specification
            scenario: Scenario name from STRESS_SCENARIOS
            baseline_result: Pre-computed baseline result
            initial_capital: Starting capital
            **backtest_kwargs: Additional backtest engine args

        Returns:
            StressTestResult with comparison metrics
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        stress_scenario = self.scenarios[scenario]

        # Create stressed symbol spec
        stressed_spec = self._apply_scenario_to_spec(symbol_spec, stress_scenario)

        # Inject gaps if configured
        stressed_data = data.copy()
        if stress_scenario.inject_gaps:
            stressed_data = self._inject_gaps(
                stressed_data,
                frequency=stress_scenario.gap_frequency,
                size_atr=stress_scenario.gap_size_atr,
            )

        # Run stressed backtest
        engine = BacktestEngine(
            symbol_spec=stressed_spec,
            physics=backtest_kwargs.get("physics"),
        )

        stressed_result = engine.run_backtest(
            data=stressed_data,
            signal_func=signal_func,
            initial_capital=initial_capital,
        )

        # Run baseline if not provided
        if baseline_result is None:
            baseline_engine = BacktestEngine(
                symbol_spec=symbol_spec,
                physics=backtest_kwargs.get("physics"),
            )
            baseline_result = baseline_engine.run_backtest(
                data=data,
                signal_func=signal_func,
                initial_capital=initial_capital,
            )

        # Calculate comparison metrics
        baseline_pnl = baseline_result.total_net_pnl
        stressed_pnl = stressed_result.total_net_pnl

        pnl_degradation = 0.0
        if baseline_pnl != 0:
            pnl_degradation = (stressed_pnl - baseline_pnl) / abs(baseline_pnl)

        baseline_costs = sum(t.total_cost for t in baseline_result.trades)
        stressed_costs = sum(t.total_cost for t in stressed_result.trades)

        cost_increase = 0.0
        if baseline_costs > 0:
            cost_increase = (stressed_costs - baseline_costs) / baseline_costs

        # Check survival (not blown up)
        survived = (
            stressed_result.equity_history[-1] > 0 if stressed_result.equity_history else True
        )

        return StressTestResult(
            scenario=stress_scenario,
            backtest_result=stressed_result,
            baseline_pnl=baseline_pnl,
            stressed_pnl=stressed_pnl,
            pnl_degradation=pnl_degradation,
            baseline_costs=baseline_costs,
            stressed_costs=stressed_costs,
            cost_increase_pct=cost_increase,
            max_drawdown=stressed_result.max_drawdown_pct,
            survived=survived,
        )

    def run_all_scenarios(
        self,
        data: pd.DataFrame,
        signal_func: Callable,
        symbol_spec: SymbolSpec,
        scenarios: List[str] = None,
        initial_capital: float = 10000.0,
        **backtest_kwargs,
    ) -> StressTestReport:
        """
        Run all stress test scenarios.

        Args:
            data: OHLCV data
            signal_func: Strategy signal function
            symbol_spec: Instrument specification
            scenarios: List of scenario names (None = all)
            initial_capital: Starting capital
            **backtest_kwargs: Additional backtest engine args

        Returns:
            StressTestReport with all scenario results
        """
        scenarios_to_run = scenarios or list(self.scenarios.keys())

        # Run baseline first
        baseline_engine = BacktestEngine(
            symbol_spec=symbol_spec,
            physics=backtest_kwargs.get("physics"),
        )
        baseline_result = baseline_engine.run_backtest(
            data=data,
            signal_func=signal_func,
            initial_capital=initial_capital,
        )

        report = StressTestReport(
            symbol=symbol_spec.symbol,
            baseline_result=baseline_result,
        )

        # Run each scenario
        for scenario_name in scenarios_to_run:
            result = self.run_scenario(
                data=data,
                signal_func=signal_func,
                symbol_spec=symbol_spec,
                scenario=scenario_name,
                baseline_result=baseline_result,
                initial_capital=initial_capital,
                **backtest_kwargs,
            )
            report.scenario_results[scenario_name] = result

        # Calculate aggregate metrics
        self._calculate_report_metrics(report)

        return report

    def run_historical_events(
        self,
        data: pd.DataFrame,
        signal_func: Callable,
        symbol_spec: SymbolSpec,
        initial_capital: float = 10000.0,
        min_severity: float = 7.0,
        **backtest_kwargs,
    ) -> Dict[str, StressTestResult]:
        """
        Run stress tests based on actual historical events.

        Uses MarketEventCalendar to apply realistic costs
        during periods that overlap with historical events.

        Args:
            data: OHLCV data
            signal_func: Strategy signal function
            symbol_spec: Instrument specification
            initial_capital: Starting capital
            min_severity: Minimum event severity to include
            **backtest_kwargs: Additional backtest engine args

        Returns:
            Dict of event name -> StressTestResult
        """
        results = {}

        # Get events for this symbol
        events = self.event_calendar.get_events_for_symbol(symbol_spec.symbol)
        events = [e for e in events if e.severity >= min_severity]

        # Run baseline
        baseline_engine = BacktestEngine(
            symbol_spec=symbol_spec,
            physics=backtest_kwargs.get("physics"),
        )
        baseline_result = baseline_engine.run_backtest(
            data=data,
            signal_func=signal_func,
            initial_capital=initial_capital,
        )

        for event in events:
            # Create scenario from event
            scenario = StressTestScenario(
                name=event.name,
                description=event.description,
                spread_multiplier=event.spread_multiplier,
                slippage_multiplier=event.slippage_multiplier,
                volatility_multiplier=event.volatility_multiplier,
                inject_gaps=event.price_impact_pct > 5.0,
                gap_size_atr=event.price_impact_pct / 10.0,
            )

            # Create stressed spec
            stressed_spec = self._apply_scenario_to_spec(symbol_spec, scenario)

            # Run backtest
            engine = BacktestEngine(
                symbol_spec=stressed_spec,
                physics=backtest_kwargs.get("physics"),
            )

            stressed_result = engine.run_backtest(
                data=data,
                signal_func=signal_func,
                initial_capital=initial_capital,
            )

            # Calculate comparison
            baseline_pnl = baseline_result.net_pnl
            stressed_pnl = stressed_result.net_pnl
            pnl_degradation = 0.0
            if baseline_pnl != 0:
                pnl_degradation = (stressed_pnl - baseline_pnl) / abs(baseline_pnl)

            results[event.name] = StressTestResult(
                scenario=scenario,
                backtest_result=stressed_result,
                baseline_pnl=baseline_pnl,
                stressed_pnl=stressed_pnl,
                pnl_degradation=pnl_degradation,
                max_drawdown=stressed_result.max_drawdown_pct,
                survived=stressed_result.equity_history[-1] > 0,
            )

        return results

    def _apply_scenario_to_spec(
        self,
        spec: SymbolSpec,
        scenario: StressTestScenario,
    ) -> SymbolSpec:
        """Create a stressed version of the symbol spec."""
        stressed_spec = copy.deepcopy(spec)

        # Apply spread multiplier
        stressed_spec.spread_points = int(stressed_spec.spread_points * scenario.spread_multiplier)
        if stressed_spec.spread_min:
            stressed_spec.spread_min = int(stressed_spec.spread_min * scenario.spread_multiplier)
        if stressed_spec.spread_max:
            stressed_spec.spread_max = int(stressed_spec.spread_max * scenario.spread_multiplier)

        # Apply slippage multiplier
        if stressed_spec.slippage_avg:
            stressed_spec.slippage_avg = int(
                stressed_spec.slippage_avg * scenario.slippage_multiplier
            )
        if stressed_spec.slippage_max:
            stressed_spec.slippage_max = int(
                stressed_spec.slippage_max * scenario.slippage_multiplier
            )

        # Apply commission multiplier if needed
        if stressed_spec.commission and scenario.commission_multiplier != 1.0:
            stressed_spec.commission.rate *= scenario.commission_multiplier

        return stressed_spec

    def _inject_gaps(
        self,
        data: pd.DataFrame,
        frequency: float,
        size_atr: float,
    ) -> pd.DataFrame:
        """
        Inject random price gaps into data.

        Simulates flash crash / gap scenarios.
        """
        data = data.copy()

        # Calculate ATR
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        atr = pd.Series(tr).rolling(20).mean().fillna(tr).values

        # Randomly inject gaps
        n_bars = len(data)
        n_gaps = int(n_bars * frequency)

        if n_gaps > 0:
            gap_indices = np.random.choice(
                range(1, n_bars),
                size=min(n_gaps, n_bars - 1),
                replace=False,
            )

            for idx in gap_indices:
                gap_size = atr[idx] * size_atr
                direction = np.random.choice([-1, 1])

                # Apply gap to open
                data.iloc[idx, data.columns.get_loc("open")] += direction * gap_size

                # Adjust high/low to maintain validity
                if direction > 0:
                    data.iloc[idx, data.columns.get_loc("high")] = max(
                        data.iloc[idx]["high"],
                        data.iloc[idx]["open"],
                    )
                else:
                    data.iloc[idx, data.columns.get_loc("low")] = min(
                        data.iloc[idx]["low"],
                        data.iloc[idx]["open"],
                    )

        return data

    def _calculate_report_metrics(self, report: StressTestReport):
        """Calculate aggregate metrics for the report."""
        if not report.scenario_results:
            return

        # Find worst case
        worst_pnl = float("inf")
        worst_scenario = ""

        total_degradation = 0.0
        survived_count = 0

        for name, result in report.scenario_results.items():
            if result.stressed_pnl < worst_pnl:
                worst_pnl = result.stressed_pnl
                worst_scenario = name

            total_degradation += result.pnl_degradation
            if result.survived:
                survived_count += 1

        report.worst_case_pnl = worst_pnl
        report.worst_case_scenario = worst_scenario
        report.avg_pnl_degradation = total_degradation / len(report.scenario_results)
        report.survival_rate = survived_count / len(report.scenario_results)

        # Calculate robustness score
        # Based on: survival, degradation, consistency
        score = 100.0

        # Penalize for non-survival
        score -= (1 - report.survival_rate) * 50

        # Penalize for degradation
        score -= min(50, abs(report.avg_pnl_degradation) * 100)

        # Penalize if worst case is total loss
        if report.worst_case_pnl < 0 and report.baseline_result:
            loss_ratio = abs(report.worst_case_pnl) / report.baseline_result.net_pnl
            score -= min(30, loss_ratio * 10)

        report.robustness_score = max(0, min(100, score))


def quick_stress_test(
    data: pd.DataFrame,
    signal_func: Callable,
    symbol_spec: SymbolSpec,
    scenarios: List[str] = None,
) -> StressTestReport:
    """
    Quick stress test with default settings.

    Usage:
        report = quick_stress_test(data, my_strategy, spec)
        print(report.summary())
    """
    engine = StressTestEngine()
    return engine.run_all_scenarios(
        data=data,
        signal_func=signal_func,
        symbol_spec=symbol_spec,
        scenarios=scenarios or ["normal", "high_volatility", "flash_crash", "black_swan"],
    )
