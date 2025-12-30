"""
Portfolio Backtesting Engine for Multi-Instrument, Multi-Timeframe Trading

Handles:
- Multiple instruments simultaneously
- Multiple timeframes per instrument
- Cumulative margin and equity tracking
- Portfolio-level stop-out
- Correlated position tracking
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest_engine import (
    BacktestResult,
    OpenPosition,
    PortfolioState,
    Trade,
    TradeDirection,
)
from .data_alignment import PointInTimeAligner, TimeframeSpec
from .market_microstructure import AssetClass
from .physics_engine import PhysicsEngine
from .symbol_spec import SymbolSpec


@dataclass
class InstrumentConfig:
    """Configuration for a single instrument in the portfolio."""

    symbol: str
    symbol_spec: SymbolSpec
    instrument_class: AssetClass
    base_timeframe: str  # Primary timeframe for signals
    feature_timeframes: List[str] = field(default_factory=list)  # Higher TFs for context
    max_positions: int = 1
    risk_per_trade: float = 0.01  # 1% risk per trade
    enabled: bool = True


@dataclass
class PortfolioConfig:
    """Configuration for the entire portfolio."""

    instruments: List[InstrumentConfig]
    initial_capital: float = 100000.0
    max_total_positions: int = 10
    max_correlation_exposure: float = 0.5  # Max 50% in correlated positions
    portfolio_stop_out_level: float = 0.3  # 30% margin level triggers stop-out
    max_drawdown_limit: float = 0.25  # Stop trading at 25% drawdown

    def get_instrument(self, symbol: str) -> Optional[InstrumentConfig]:
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        return None


@dataclass
class PortfolioTrade(Trade):
    """Extended trade with portfolio context."""

    timeframe: str = ""
    instrument_class: AssetClass = AssetClass.COMMODITY
    portfolio_equity_at_entry: float = 0.0
    portfolio_margin_at_entry: float = 0.0


@dataclass
class PortfolioBacktestResult:
    """Complete portfolio backtest results."""

    # All trades across instruments
    trades: List[PortfolioTrade]

    # Per-instrument results
    instrument_results: Dict[str, BacktestResult]

    # Per-timeframe results
    timeframe_results: Dict[str, BacktestResult]

    # Per-class results
    class_results: Dict[AssetClass, BacktestResult]

    # Portfolio-level metrics
    total_trades: int = 0
    total_net_pnl: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_sortino: float = 0.0
    portfolio_max_drawdown: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0

    # Margin metrics
    max_margin_used: float = 0.0
    avg_margin_used: float = 0.0
    margin_calls: int = 0
    stop_outs: int = 0

    # Equity curve
    equity_curve: Optional[pd.Series] = None
    margin_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "total_net_pnl": self.total_net_pnl,
            "portfolio_sharpe": self.portfolio_sharpe,
            "portfolio_sortino": self.portfolio_sortino,
            "portfolio_max_drawdown": self.portfolio_max_drawdown,
            "portfolio_max_drawdown_pct": self.portfolio_max_drawdown_pct,
            "max_margin_used": self.max_margin_used,
            "avg_margin_used": self.avg_margin_used,
            "margin_calls": self.margin_calls,
            "stop_outs": self.stop_outs,
            "instruments": {k: v.to_dict() for k, v in self.instrument_results.items()},
            "timeframes": {k: v.to_dict() for k, v in self.timeframe_results.items()},
            "classes": {k.value: v.to_dict() for k, v in self.class_results.items()},
        }


class PortfolioBacktestEngine:
    """
    Multi-instrument portfolio backtesting engine.

    Features:
    - Simultaneous trading across multiple instruments
    - Multi-timeframe signal generation
    - Cumulative margin and equity tracking
    - Portfolio-level risk management
    - Point-in-time data alignment
    """

    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio backtest engine.

        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.physics = PhysicsEngine()
        self.aligner = PointInTimeAligner()

        # State
        self.balance = config.initial_capital
        self.equity = config.initial_capital
        self.open_positions: Dict[str, OpenPosition] = {}  # symbol -> position
        self.all_trades: List[PortfolioTrade] = []
        self.trade_counter = 0

        # History
        self.equity_history: List[float] = [config.initial_capital]
        self.margin_history: List[float] = [0.0]
        self.balance_history: List[float] = [config.initial_capital]

        # Flags
        self.stop_out_triggered = False
        self.max_dd_triggered = False

    def reset(self):
        """Reset engine state."""
        self.balance = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.open_positions = {}
        self.all_trades = []
        self.trade_counter = 0
        self.equity_history = [self.config.initial_capital]
        self.margin_history = [0.0]
        self.balance_history = [self.config.initial_capital]
        self.stop_out_triggered = False
        self.max_dd_triggered = False

    def add_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
    ):
        """
        Add OHLCV data for a symbol/timeframe.

        Args:
            symbol: Instrument symbol
            timeframe: Timeframe string (M15, H1, etc.)
            data: DataFrame with OHLCV columns
        """
        self.aligner.add_data(symbol, timeframe, data)

    def run_backtest(
        self,
        master_data: pd.DataFrame,
        signal_funcs: Dict[str, Callable] = None,
        agents: Dict[str, Any] = None,
    ) -> PortfolioBacktestResult:
        """
        Run portfolio backtest.

        Args:
            master_data: Master timeline DataFrame (usually lowest timeframe)
                        Index should be datetime
            signal_funcs: Dict of symbol -> signal function
            agents: Dict of symbol -> RL agent

        Returns:
            PortfolioBacktestResult with all metrics
        """
        self.reset()

        if signal_funcs is None:
            signal_funcs = {}
        if agents is None:
            agents = {}

        # Pending signals for next bar execution
        pending_entries: Dict[
            str, Tuple[int, float, str]
        ] = {}  # symbol -> (signal, energy, regime)
        pending_exits: Dict[str, bool] = {}  # symbol -> should_exit

        # Pre-compute physics states for each instrument
        physics_states: Dict[str, pd.DataFrame] = {}
        for inst in self.config.instruments:
            if not inst.enabled:
                continue

            # Get base timeframe data
            tf_data = self.aligner.data.get(inst.symbol, {}).get(inst.base_timeframe)
            if tf_data is not None and "close" in tf_data.columns:
                physics_states[inst.symbol] = self.physics.compute_physics_state(tf_data["close"])

        # Main backtest loop
        has_datetime_index = isinstance(master_data.index, pd.DatetimeIndex)

        for i in range(1, len(master_data)):
            current_time = master_data.index[i] if has_datetime_index else datetime.now()

            # === EXECUTION PHASE: Execute pending signals ===
            self._execute_pending_signals(
                current_time,
                pending_entries,
                pending_exits,
            )

            # === UPDATE PHASE: Update open positions ===
            self._update_open_positions(current_time)

            # === RISK CHECK: Portfolio-level stop-out ===
            if self._check_portfolio_stop_out():
                self._close_all_positions(current_time, "Portfolio stop-out")
                self.stop_out_triggered = True

            # === SIGNAL PHASE: Generate signals for next bar ===
            if not self.stop_out_triggered and not self.max_dd_triggered:
                pending_entries, pending_exits = self._generate_signals(
                    current_time,
                    physics_states,
                    signal_funcs,
                    agents,
                )

            # Record portfolio state
            self._record_portfolio_state()

            # Check max drawdown limit
            if self._check_max_drawdown():
                warnings.warn("Max drawdown limit reached, stopping backtest")
                self.max_dd_triggered = True

        # Close remaining positions
        if len(self.open_positions) > 0:
            final_time = master_data.index[-1] if has_datetime_index else datetime.now()
            self._close_all_positions(final_time, "End of backtest")

        return self._calculate_results()

    def _execute_pending_signals(
        self,
        current_time: datetime,
        pending_entries: Dict[str, Tuple[int, float, str]],
        pending_exits: Dict[str, bool],
    ):
        """Execute pending entry/exit signals at current bar open."""

        # Execute exits first
        for symbol, should_exit in pending_exits.items():
            if should_exit and symbol in self.open_positions:
                self._close_position(symbol, current_time)

        pending_exits.clear()

        # Execute entries
        for symbol, (signal, energy, regime) in pending_entries.items():
            if signal == 0:
                continue

            if symbol in self.open_positions:
                continue  # Already have position

            inst = self.config.get_instrument(symbol)
            if inst is None:
                continue

            # Check max positions
            if len(self.open_positions) >= self.config.max_total_positions:
                continue

            # Get execution price
            aligned = self.aligner.get_last_completed_bar(symbol, inst.base_timeframe, current_time)
            if aligned is None:
                continue

            execution_price = aligned.get("open", aligned.get("close", 0))
            if execution_price <= 0:
                continue

            # Check margin availability
            direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
            self._open_position(
                symbol, direction, execution_price, current_time, inst, energy, regime
            )

        pending_entries.clear()

    def _update_open_positions(self, current_time: datetime):
        """Update all open positions with current prices."""

        for symbol, position in list(self.open_positions.items()):
            inst = self.config.get_instrument(symbol)
            if inst is None:
                continue

            # Get current price
            aligned = self.aligner.get_last_completed_bar(symbol, inst.base_timeframe, current_time)
            if aligned is None:
                continue

            current_price = aligned.get("close", position.current_price)
            position.update(current_price)

            # Update MFE/MAE
            if "high" in aligned and "low" in aligned:
                self._update_mfe_mae(position, aligned)

    def _generate_signals(
        self,
        current_time: datetime,
        physics_states: Dict[str, pd.DataFrame],
        signal_funcs: Dict[str, Callable],
        agents: Dict[str, Any],
    ) -> Tuple[Dict, Dict]:
        """Generate signals for each instrument."""

        pending_entries = {}
        pending_exits = {}

        for inst in self.config.instruments:
            if not inst.enabled:
                continue

            symbol = inst.symbol

            # Get aligned data
            aligned = self.aligner.get_aligned_features(
                symbol,
                inst.base_timeframe,
                inst.feature_timeframes,
                current_time,
            )

            base_bar = aligned.get(inst.base_timeframe)
            if base_bar is None:
                continue

            # Get physics state
            physics_state = physics_states.get(symbol)
            if physics_state is None:
                continue

            # Find bar index
            bar_idx = self._find_bar_index(physics_state, current_time, inst)
            if bar_idx < self.physics.lookback:
                continue

            # Get energy and regime
            energy = physics_state["energy"].iloc[bar_idx] if bar_idx < len(physics_state) else 0
            regime = (
                physics_state["regime"].iloc[bar_idx] if bar_idx < len(physics_state) else "unknown"
            )

            # Generate signal
            signal = 0

            if symbol in signal_funcs:
                signal = signal_funcs[symbol](base_bar, physics_state, bar_idx)
            elif symbol in agents:
                state = self._build_agent_state(base_bar, physics_state, bar_idx)
                action = agents[symbol].select_action(state)
                signal = [-1, 0, 1, 0][action] if action < 4 else 0
            else:
                # Default physics-based signal
                signal = self._default_signal(base_bar, physics_state, bar_idx)

            # Check for exit signals
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                if self._should_exit(position, base_bar, physics_state, bar_idx, signal):
                    pending_exits[symbol] = True
            else:
                # Entry signal
                if signal != 0:
                    pending_entries[symbol] = (signal, energy, regime)

        return pending_entries, pending_exits

    def _find_bar_index(
        self,
        physics_state: pd.DataFrame,
        current_time: datetime,
        inst: InstrumentConfig,
    ) -> int:
        """Find the bar index in physics state for current time."""
        tf_spec = TimeframeSpec(inst.base_timeframe)
        completed_time = tf_spec.get_completed_bar_time(current_time)

        # Find index in physics_state
        if hasattr(physics_state, "index"):
            mask = physics_state.index <= completed_time
            if mask.any():
                return mask.sum() - 1

        return 0

    def _open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        time: datetime,
        inst: InstrumentConfig,
        energy: float,
        regime: str,
    ):
        """Open a new position."""

        # Calculate position size
        portfolio_state = self._get_portfolio_state()
        risk_amount = portfolio_state.equity * inst.risk_per_trade

        spec = inst.symbol_spec
        lots = min(
            risk_amount / (spec.spread_points * spec.tick_value * 2),
            spec.volume_max,
        )
        lots = max(lots, spec.volume_min)
        lots = round(lots / spec.volume_step) * spec.volume_step

        # Check margin
        required_margin = spec.calculate_margin(lots, price, direction.value)
        if required_margin > portfolio_state.free_margin:
            # Reduce size to fit margin
            max_lots = spec.max_lots_for_margin(
                portfolio_state.free_margin * 0.9,  # 90% of free margin
                price,
                direction.value,
            )
            if max_lots < spec.volume_min:
                return  # Can't open position
            lots = max_lots

        # Calculate costs
        spread_cost = spec.spread_cost(lots, price)
        commission = spec.commission.calculate_commission(lots, lots * spec.contract_size * price)
        slippage = spec.slippage_avg * spec.tick_value * lots

        self.trade_counter += 1

        trade = PortfolioTrade(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction,
            lots=lots,
            entry_time=time,
            entry_price=price,
            spread_cost=spread_cost,
            commission=commission / 2,
            slippage=slippage / 2,
            energy_at_entry=energy,
            regime_at_entry=regime,
            timeframe=inst.base_timeframe,
            instrument_class=inst.instrument_class,
            portfolio_equity_at_entry=portfolio_state.equity,
            portfolio_margin_at_entry=portfolio_state.used_margin,
        )

        position = OpenPosition(
            trade=trade,
            symbol_spec=spec,
            current_price=price,
        )
        position.update(price)

        self.open_positions[symbol] = position

    def _close_position(self, symbol: str, time: datetime, reason: str = "Signal"):
        """Close a position."""

        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        trade = position.trade
        spec = position.symbol_spec

        exit_price = position.current_price
        trade.exit_price = exit_price
        trade.exit_time = time

        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price

        trade.gross_pnl = (price_diff / spec.tick_size) * spec.tick_value * trade.lots

        # Exit costs
        exit_commission = (
            spec.commission.calculate_commission(
                trade.lots, trade.lots * spec.contract_size * exit_price
            )
            / 2
        )
        exit_slippage = (spec.slippage_avg * spec.tick_value * trade.lots) / 2

        trade.commission += exit_commission
        trade.slippage += exit_slippage

        # Swap costs
        if trade.entry_time and trade.exit_time:
            trade.swap_cost = spec.calculate_swap_cost(
                trade.direction.value,
                trade.lots,
                trade.entry_time,
                trade.exit_time,
                trade.entry_price,
            )

        trade.net_pnl = trade.gross_pnl - trade.total_cost

        # Update balance
        self.balance += trade.net_pnl

        # Record trade
        self.all_trades.append(trade)

        # Remove position
        del self.open_positions[symbol]

    def _close_all_positions(self, time: datetime, reason: str):
        """Close all open positions."""
        for symbol in list(self.open_positions.keys()):
            self._close_position(symbol, time, reason)

    def _get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        state = PortfolioState()
        state.update(self.balance, list(self.open_positions.values()))
        return state

    def _check_portfolio_stop_out(self) -> bool:
        """Check if portfolio margin level triggers stop-out."""
        state = self._get_portfolio_state()
        if state.used_margin <= 0:
            return False
        return state.margin_level < (self.config.portfolio_stop_out_level * 100)

    def _check_max_drawdown(self) -> bool:
        """Check if max drawdown limit is reached."""
        if len(self.equity_history) < 2:
            return False

        peak = max(self.equity_history)
        current = self.equity_history[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0

        return drawdown >= self.config.max_drawdown_limit

    def _record_portfolio_state(self):
        """Record current portfolio state to history."""
        state = self._get_portfolio_state()
        self.equity = state.equity
        self.equity_history.append(state.equity)
        self.margin_history.append(state.used_margin)
        self.balance_history.append(self.balance)

    def _update_mfe_mae(self, position: OpenPosition, bar: pd.Series):
        """Update MFE/MAE for position."""
        trade = position.trade
        high = bar.get("high", position.current_price)
        low = bar.get("low", position.current_price)

        if trade.direction == TradeDirection.LONG:
            favorable = high - trade.entry_price
            adverse = trade.entry_price - low
        else:
            favorable = trade.entry_price - low
            adverse = high - trade.entry_price

        trade.mfe = max(trade.mfe, favorable)
        trade.mae = max(trade.mae, adverse)

    def _should_exit(
        self,
        position: OpenPosition,
        bar: pd.Series,
        physics_state: pd.DataFrame,
        bar_idx: int,
        current_signal: int,
    ) -> bool:
        """Check if position should be closed."""
        trade = position.trade

        # Exit on opposite signal
        if trade.direction == TradeDirection.LONG and current_signal < 0:
            return True
        if trade.direction == TradeDirection.SHORT and current_signal > 0:
            return True

        # Exit on regime change to overdamped
        if bar_idx < len(physics_state.get("regime", [])):
            regime = physics_state["regime"].iloc[bar_idx]
            if regime == "overdamped":
                return True

        return False

    def _default_signal(
        self,
        bar: pd.Series,
        physics_state: pd.DataFrame,
        bar_idx: int,
    ) -> int:
        """Default physics-based signal."""
        if bar_idx >= len(physics_state["energy"]):
            return 0

        regime = physics_state["regime"].iloc[bar_idx] if "regime" in physics_state else None

        if regime == "underdamped":
            energy = physics_state["energy"].iloc[bar_idx]
            if energy > physics_state["energy"].iloc[:bar_idx].quantile(0.75):
                close = bar.get("close", 0)
                sma = physics_state["energy"].iloc[:bar_idx].mean()
                if close > sma:
                    return 1
                elif close < sma:
                    return -1

        return 0

    def _build_agent_state(
        self,
        bar: pd.Series,
        physics_state: pd.DataFrame,
        bar_idx: int,
    ) -> np.ndarray:
        """Build state vector for RL agent."""
        energy = physics_state["energy"].iloc[bar_idx] if bar_idx < len(physics_state) else 0
        damping = physics_state["damping"].iloc[bar_idx] if bar_idx < len(physics_state) else 0
        entropy = physics_state["entropy"].iloc[bar_idx] if bar_idx < len(physics_state) else 0

        return np.array(
            [
                energy,
                damping,
                entropy,
                bar.get("close", 0),
                bar.get("volume", 0),
            ],
            dtype=np.float32,
        )

    def _calculate_results(self) -> PortfolioBacktestResult:
        """Calculate all portfolio metrics."""

        # Group trades by instrument, timeframe, class
        by_instrument: Dict[str, List[PortfolioTrade]] = {}
        by_timeframe: Dict[str, List[PortfolioTrade]] = {}
        by_class: Dict[AssetClass, List[PortfolioTrade]] = {}

        for trade in self.all_trades:
            # By instrument
            if trade.symbol not in by_instrument:
                by_instrument[trade.symbol] = []
            by_instrument[trade.symbol].append(trade)

            # By timeframe
            if trade.timeframe not in by_timeframe:
                by_timeframe[trade.timeframe] = []
            by_timeframe[trade.timeframe].append(trade)

            # By class
            if trade.instrument_class not in by_class:
                by_class[trade.instrument_class] = []
            by_class[trade.instrument_class].append(trade)

        # Calculate per-group results
        instrument_results = {k: self._trades_to_result(v) for k, v in by_instrument.items()}
        timeframe_results = {k: self._trades_to_result(v) for k, v in by_timeframe.items()}
        class_results = {k: self._trades_to_result(v) for k, v in by_class.items()}

        # Portfolio metrics
        equity_curve = pd.Series(self.equity_history)
        margin_curve = pd.Series(self.margin_history)

        total_trades = len(self.all_trades)
        total_pnl = sum(t.net_pnl for t in self.all_trades)

        # Sharpe
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = float("inf") if returns.mean() > 0 else 0

        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve - rolling_max
        max_dd = abs(drawdown.min())
        max_dd_pct = (
            abs(drawdown.min() / rolling_max[drawdown.idxmin()]) * 100
            if rolling_max[drawdown.idxmin()] > 0
            else 0
        )

        return PortfolioBacktestResult(
            trades=self.all_trades,
            instrument_results=instrument_results,
            timeframe_results=timeframe_results,
            class_results=class_results,
            total_trades=total_trades,
            total_net_pnl=total_pnl,
            portfolio_sharpe=sharpe,
            portfolio_sortino=sortino,
            portfolio_max_drawdown=max_dd,
            portfolio_max_drawdown_pct=max_dd_pct,
            max_margin_used=max(self.margin_history),
            avg_margin_used=np.mean(self.margin_history),
            margin_calls=0,  # TODO: Track margin calls
            stop_outs=1 if self.stop_out_triggered else 0,
            equity_curve=equity_curve,
            margin_curve=margin_curve,
        )

    def _trades_to_result(self, trades: List[PortfolioTrade]) -> BacktestResult:
        """Convert trade list to BacktestResult."""
        if not trades:
            return BacktestResult(trades=[])

        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        return BacktestResult(
            trades=trades,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades),
            gross_profit=sum(t.gross_pnl for t in winners),
            gross_loss=sum(t.gross_pnl for t in losers),
            total_gross_pnl=sum(t.gross_pnl for t in trades),
            total_costs=sum(t.total_cost for t in trades),
            total_net_pnl=sum(t.net_pnl for t in trades),
            total_spread_cost=sum(t.spread_cost for t in trades),
            total_commission=sum(t.commission for t in trades),
            total_slippage=sum(t.slippage for t in trades),
            total_swap_cost=sum(t.swap_cost for t in trades),
        )
