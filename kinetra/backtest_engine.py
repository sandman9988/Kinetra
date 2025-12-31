"""
Backtest Engine with Realistic Friction Costs and ML/RL Integration

Complete backtesting framework for theorem validation and agent-based trading:
- Accurate cost modeling (spread, commission, swap, slippage)
- Per-instrument specifications
- Trade-by-trade simulation
- Monte Carlo validation
- Performance metrics (Omega, Sharpe, Z-factor)
- ML/RL agent integration
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from .physics_engine import PhysicsEngine
from .symbol_spec import SymbolSpec

# Try GPU physics
try:
    from .parallel import GPUPhysicsEngine, TORCH_AVAILABLE
    GPU_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False


class TradeDirection(Enum):
    """Trade direction enumeration."""

    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Individual trade record with comprehensive tracking."""

    trade_id: int
    symbol: str
    direction: TradeDirection
    lots: float
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    # Costs
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap_cost: float = 0.0

    # Results
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Physics state at entry
    energy_at_entry: float = 0.0
    regime_at_entry: str = ""

    # Trade quality metrics
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    @property
    def total_cost(self) -> float:
        """Calculate total transaction costs."""
        return self.spread_cost + self.commission + self.slippage + abs(self.swap_cost)

    @property
    def holding_time(self) -> Optional[timedelta]:
        """Calculate holding time for closed trades."""
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

    @property
    def price_captured(self) -> float:
        """Calculate price difference captured."""
        if not self.is_closed:
            return 0.0
        if self.direction == TradeDirection.LONG:
            return self.exit_price - self.entry_price
        else:
            return self.entry_price - self.exit_price

    @property
    def mfe_efficiency(self) -> float:
        """MFE efficiency: how much of MFE was captured as profit (0-1)."""
        if self.mfe > 0:
            return max(0, min(1.0, self.price_captured / self.mfe))
        return 0.0

    @property
    def mae_efficiency(self) -> float:
        """MAE efficiency: how well adverse excursion was limited (0-1)."""
        if self.mfe > 0:
            return max(0, 1 - self.mae / self.mfe)
        return 0.0


@dataclass
class BacktestResult:
    """Complete backtest results with comprehensive metrics."""

    # Trade list
    trades: List[Trade]

    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    total_net_pnl: float = 0.0

    # Cost breakdown
    total_spread_cost: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_swap_cost: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    min_margin_level: float = float("inf")

    # CVaR metrics
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Kinetra-specific metrics
    omega_ratio: float = 0.0
    z_factor: float = 0.0
    energy_captured_pct: float = 0.0
    mfe_capture_pct: float = 0.0

    # Equity curve
    equity_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "total_gross_pnl": self.total_gross_pnl,
            "total_costs": self.total_costs,
            "total_net_pnl": self.total_net_pnl,
            "cost_breakdown": {
                "spread": self.total_spread_cost,
                "commission": self.total_commission,
                "slippage": self.total_slippage,
                "swap": self.total_swap_cost,
            },
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "omega_ratio": self.omega_ratio,
            "z_factor": self.z_factor,
            "energy_captured_pct": self.energy_captured_pct,
        }


class BacktestEngine:
    """
    Realistic backtesting engine with full friction modeling and ML/RL integration.

    Features:
    - Per-bar simulation with OHLC handling
    - Accurate cost modeling per instrument
    - MFE/MAE tracking
    - Physics-aware signal generation
    - ML/RL agent integration
    - Monte Carlo validation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_positions: int = 1,
        use_physics_signals: bool = True,
        use_gpu: bool = True,  # Use GPU for physics if available
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            risk_per_trade: Percentage of equity to risk per trade (0-1)
            max_positions: Maximum number of concurrent positions
            use_physics_signals: Use physics-based signals when no custom signal provided
            use_gpu: Use GPU acceleration for physics (ROCm/CUDA)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.use_physics_signals = use_physics_signals
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Use GPU physics if available
        if self.use_gpu:
            self.gpu_physics = GPUPhysicsEngine(device="auto")
        self.physics = PhysicsEngine()
        self.trades: List[Trade] = []
        self.equity = initial_capital
        self.equity_history: List[float] = [initial_capital]
        self.trade_counter = 0

    def reset(self) -> None:
        """Reset engine state for new backtest."""
        self.trades = []
        self.equity = self.initial_capital
        self.equity_history = [self.initial_capital]
        self.trade_counter = 0

    def run_backtest(
        self,
        data: pd.DataFrame,
        symbol_spec: SymbolSpec,
        signal_func: Optional[Callable] = None,
        agent=None,
    ) -> BacktestResult:
        """
        Run backtest on OHLCV data with optional signal function or RL agent.

        Args:
            data: DataFrame with columns [time, open, high, low, close, volume]
            symbol_spec: Instrument specification with costs
            signal_func: Optional custom signal function(row, physics_state, bar_index) -> int
                         Returns: 1=buy, -1=sell, 0=hold
            agent: Optional RL agent for signal generation

        Returns:
            BacktestResult with all metrics

        Raises:
            ValueError: If data lacks required columns or if equity becomes negative
        """
        self.reset()

        # Ensure required columns
        required = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")

        # Compute physics state for entire series
        physics_state = self.physics.compute_physics_state(
            prices=data["close"],
            volume=data.get("volume"),
            high=data.get("high"),
            low=data.get("low"),
            open_price=data.get("open"),
            include_percentiles=True,
        )

        # Track open position
        open_position: Optional[Trade] = None

        for i in range(1, len(data)):
            row = data.iloc[i]

            # Get physics features at this bar
            if i < len(physics_state["energy"]):
                energy = physics_state["energy"].iloc[i]
                regime = physics_state["regime"].iloc[i]
            else:
                energy = 0.0
                regime = "unknown"

            # Update MFE/MAE for open position
            if open_position is not None:
                self._update_mfe_mae(open_position, row, symbol_spec)

                # Check for exit signals
                should_exit = self._check_exit_signal(
                    open_position, row, physics_state, i, signal_func, agent
                )

                if should_exit:
                    self._close_position(
                        open_position,
                        row["close"],
                        data.iloc[i].name if hasattr(data.iloc[i], "name") else datetime.now(),
                        symbol_spec,
                        i,  # bars held
                    )
                    open_position = None

            # Check for entry signals (only if no open position)
            if open_position is None and i > self.physics.lookback:
                signal = self._get_signal(row, physics_state, i, signal_func, agent)

                if signal != 0:
                    direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT
                    open_position = self._open_position(
                        symbol_spec.symbol,
                        direction,
                        row["close"],
                        data.iloc[i].name if hasattr(data.iloc[i], "name") else datetime.now(),
                        symbol_spec,
                        energy,
                        regime,
                    )

            # Record equity
            mark_to_market = self._calculate_mtm(open_position, row["close"], symbol_spec)
            equity_value = self.equity + mark_to_market
            self.equity_history.append(equity_value)

            # Check for negative equity (stop trading)
            if equity_value < 0:
                warnings.warn(f"Equity became negative at bar {i}, stopping backtest")
                break

        # Close any remaining position
        if open_position is not None:
            self._close_position(
                open_position,
                data.iloc[-1]["close"],
                data.iloc[-1].name if hasattr(data.iloc[-1], "name") else datetime.now(),
                symbol_spec,
                len(data) - 1,
            )

        return self._calculate_results()

    def _get_signal(
        self,
        row: pd.Series,
        physics_state: pd.DataFrame,
        bar_index: int,
        signal_func: Optional[Callable],
        agent,
    ) -> int:
        """
        Get trading signal: 1=buy, -1=sell, 0=hold.

        Args:
            row: Current OHLCV bar
            physics_state: Physics state DataFrame
            bar_index: Current bar index
            signal_func: Custom signal function
            agent: RL agent

        Returns:
            Trading signal (-1, 0, 1)
        """
        # Custom signal function takes precedence
        if signal_func is not None:
            return signal_func(row, physics_state, bar_index)

        # RL agent next
        if agent is not None:
            state = self._build_agent_state(row, physics_state, bar_index)
            action = agent.select_action(state)
            # Map action to signal (depends on agent's action space)
            return [-1, 0, 1, 0][action] if action < 4 else 0

        # Default: physics-based signal
        if bar_index >= len(physics_state["energy"]):
            return 0

        energy = physics_state["energy"].iloc[bar_index]
        regime = physics_state["regime"].iloc[bar_index]

        # Simple momentum + regime filter
        if regime == "underdamped":
            # High energy regime - trend following
            sma = physics_state.get("sma", pd.Series([row["close"]]))
            momentum = row["close"] - sma.iloc[min(bar_index, len(sma) - 1)]
            if momentum > 0:
                return 1
            elif momentum < 0:
                return -1

        return 0

    def _check_exit_signal(
        self,
        position: Trade,
        row: pd.Series,
        physics_state: pd.DataFrame,
        bar_index: int,
        signal_func: Optional[Callable],
        agent,
    ) -> bool:
        """
        Check if position should be closed.

        Args:
            position: Open position
            row: Current OHLCV bar
            physics_state: Physics state DataFrame
            bar_index: Current bar index
            signal_func: Custom signal function
            agent: RL agent

        Returns:
            True if position should be closed
        """
        # Simple exit: opposite signal
        current_signal = self._get_signal(row, physics_state, bar_index, signal_func, agent)

        if position.direction == TradeDirection.LONG and current_signal < 0:
            return True
        if position.direction == TradeDirection.SHORT and current_signal > 0:
            return True

        # Exit on regime change to overdamped
        if bar_index < len(physics_state.get("regime", [])):
            regime = physics_state["regime"].iloc[bar_index]
            if regime == "overdamped":
                return True

        return False

    def _open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        time: datetime,
        spec: SymbolSpec,
        energy: float,
        regime: str,
    ) -> Trade:
        """Open a new position with proper cost calculation."""
        self.trade_counter += 1

        # Calculate position size based on risk
        risk_amount = self.equity * self.risk_per_trade
        # Simplified: risk = X% of equity, size accordingly
        lots = min(
            risk_amount / (spec.spread_points * spec.tick_value * 2),  # 2x spread as stop
            spec.volume_max,
        )
        lots = max(lots, spec.volume_min)
        lots = round(lots / spec.volume_step) * spec.volume_step

        # Calculate entry costs
        spread_cost = spec.spread_cost(lots, price)
        commission = spec.commission.calculate_commission(lots, lots * spec.contract_size * price)
        slippage = spec.slippage_avg * spec.tick_value * lots

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction,
            lots=lots,
            entry_time=time,
            entry_price=price,
            spread_cost=spread_cost,
            commission=commission / 2,  # Half on entry, half on exit
            slippage=slippage / 2,
            energy_at_entry=energy,
            regime_at_entry=regime,
            mfe=0.0,
            mae=0.0,
        )

        return trade

    def _close_position(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        spec: SymbolSpec,
        bars_held: int,
    ) -> None:
        """Close an open position with proper cost calculation."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time

        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price

        # Convert to money
        trade.gross_pnl = price_diff * trade.lots * spec.tick_value / spec.tick_size

        # Exit costs
        exit_commission = spec.commission.calculate_commission(
            trade.lots, trade.lots * spec.contract_size * exit_price
        )
        exit_slippage = spec.slippage_avg * spec.tick_value * trade.lots

        trade.commission += exit_commission / 2
        trade.slippage += exit_slippage / 2

        # Swap costs (simplified: assume 1 swap per day held)
        days_held = max(1, bars_held // 24)  # Rough estimate for H1 data
        trade.swap_cost = spec.holding_cost(trade.direction.value, trade.lots, days_held)

        # Net P&L
        trade.net_pnl = trade.gross_pnl - trade.total_cost

        # Update equity
        self.equity += trade.net_pnl

        # Record trade
        self.trades.append(trade)

    def _update_mfe_mae(self, trade: Trade, row: pd.Series, spec: SymbolSpec) -> None:
        """Update MFE/MAE for open position."""
        if trade.direction == TradeDirection.LONG:
            favorable = row["high"] - trade.entry_price
            adverse = trade.entry_price - row["low"]
        else:
            favorable = trade.entry_price - row["low"]
            adverse = row["high"] - trade.entry_price

        trade.mfe = max(trade.mfe, favorable)
        trade.mae = max(trade.mae, adverse)

    def _calculate_mtm(
        self, position: Optional[Trade], current_price: float, spec: SymbolSpec
    ) -> float:
        """Calculate mark-to-market P&L for open position."""
        if position is None:
            return 0.0

        if position.direction == TradeDirection.LONG:
            price_diff = current_price - position.entry_price
        else:
            price_diff = position.entry_price - current_price

        return price_diff * position.lots * spec.tick_value / spec.tick_size

    def _build_agent_state(
        self, row: pd.Series, physics_state: pd.DataFrame, bar_index: int
    ) -> np.ndarray:
        """
        Build state vector for RL agent.

        Args:
            row: Current OHLCV bar
            physics_state: Physics state DataFrame
            bar_index: Current bar index

        Returns:
            Numpy array with state features
        """
        # Extract physics features
        energy = (
            physics_state["energy"].iloc[bar_index]
            if bar_index < len(physics_state["energy"])
            else 0
        )
        damping = (
            physics_state["damping"].iloc[bar_index]
            if bar_index < len(physics_state["damping"])
            else 0
        )
        entropy = (
            physics_state["entropy"].iloc[bar_index]
            if bar_index < len(physics_state["entropy"])
            else 0
        )

        # Extract percentile features if available
        energy_pct = (
            physics_state["energy_pct"].iloc[bar_index]
            if "energy_pct" in physics_state and bar_index < len(physics_state["energy_pct"])
            else 0.5
        )
        damping_pct = (
            physics_state["damping_pct"].iloc[bar_index]
            if "damping_pct" in physics_state and bar_index < len(physics_state["damping_pct"])
            else 0.5
        )

        # Build state vector
        return np.array(
            [
                energy,
                damping,
                entropy,
                energy_pct,
                damping_pct,
                row["close"],
                row.get("volume", 0),
            ],
            dtype=np.float32,
        )

    def _calculate_results(self) -> BacktestResult:
        """Calculate all backtest metrics from completed trades."""
        if not self.trades:
            return BacktestResult(trades=[], equity_curve=pd.Series(self.equity_history))

        # Basic counts
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.net_pnl > 0]
        losers = [t for t in self.trades if t.net_pnl <= 0]

        # P&L
        gross_profit = sum(t.gross_pnl for t in winners)
        gross_loss = sum(t.gross_pnl for t in losers)
        total_gross = sum(t.gross_pnl for t in self.trades)
        total_costs = sum(t.total_cost for t in self.trades)
        total_net = sum(t.net_pnl for t in self.trades)

        # Cost breakdown
        total_spread = sum(t.spread_cost for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        total_swap = sum(t.swap_cost for t in self.trades)

        # Equity curve
        equity_curve = pd.Series(self.equity_history)

        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve - rolling_max
        max_dd = drawdown.min()
        max_dd_pct = (
            (max_dd / rolling_max[drawdown.idxmin()]) * 100
            if rolling_max[drawdown.idxmin()] > 0
            else 0
        )

        # Returns for ratio calculations
        returns = equity_curve.pct_change().dropna()

        # Determine bars per year based on timeframe for proper scaling
        timeframe_bars_per_year = {
            "M1": 525600,
            "M5": 105120,
            "M15": 35040,
            "M30": 17520,
            "H1": 8760,
            "H4": 2190,
            "D1": 252,
            "W1": 52,
            "MN": 12,
        }
        bars_per_year = timeframe_bars_per_year.get(self.timeframe, 252)
        annualization = np.sqrt(bars_per_year)

        # Sharpe ratio (annualized, timeframe-aware)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * annualization
        else:
            sharpe = 0.0

        # Sortino ratio
        if len(returns) > 1 and returns.std() > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                sortino = (returns.mean() / downside_returns.std()) * annualization
            else:
                sortino = float("inf")
        else:
            sortino = 0.0

        # CVaR (Conditional Value at Risk) - downside tail risk
        if len(returns) > 0:
            q95 = returns.quantile(0.05)
            q99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= q95].mean() if len(returns[returns <= q95]) > 0 else 0.0
            cvar_99 = returns[returns <= q99].mean() if len(returns[returns <= q99]) > 0 else 0.0
        else:
            cvar_95 = 0.0
            cvar_99 = 0.0

        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega = gains / losses if losses > 0 else float("inf")

        # Z-factor (statistical edge)
        if total_trades > 1:
            win_rate = len(winners) / total_trades
            avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
            avg_loss = np.mean([abs(t.net_pnl) for t in losers]) if losers else 0
            if avg_loss > 0:
                z_factor = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
            else:
                z_factor = float("inf") if avg_win > 0 else 0.0
        else:
            z_factor = 0.0

        # Energy captured
        total_energy_at_entry = sum(t.energy_at_entry for t in self.trades)
        profitable_energy = sum(t.energy_at_entry for t in winners)
        energy_captured = (
            profitable_energy / total_energy_at_entry if total_energy_at_entry > 0 else 0.0
        )

        # MFE capture
        total_mfe = sum(t.mfe for t in self.trades)
        realized_pnl = sum(max(0, t.gross_pnl) for t in self.trades)
        mfe_capture = realized_pnl / total_mfe if total_mfe > 0 else 0.0

        return BacktestResult(
            trades=self.trades,
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / total_trades if total_trades > 0 else 0,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_gross_pnl=total_gross,
            total_costs=total_costs,
            total_net_pnl=total_net,
            total_spread_cost=total_spread,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_swap_cost=total_swap,
            max_drawdown=abs(max_dd),
            max_drawdown_pct=abs(max_dd_pct),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            min_margin_level=self.min_margin_level,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            omega_ratio=omega,
            z_factor=z_factor,
            energy_captured_pct=energy_captured,
            mfe_capture_pct=mfe_capture,
            equity_curve=equity_curve,
        )

    def monte_carlo_validation(
        self,
        data: pd.DataFrame,
        symbol_spec: SymbolSpec,
        n_runs: int = 100,
        shuffle_method: str = "returns",
    ) -> pd.DataFrame:
        """
        Run Monte Carlo validation to assess strategy robustness.

        Args:
            data: Original OHLCV data
            symbol_spec: Instrument specification
            n_runs: Number of simulation runs
            shuffle_method: "returns" or "bootstrap"

        Returns:
            DataFrame with results from each run

        Raises:
            ValueError: If shuffle method is not supported
        """
        # Parallel Monte Carlo - use all available cores
        n_workers = min(mp.cpu_count(), n_runs, 32)  # Cap at 32 for AMD 5950

        def run_single_mc(run_id: int):
            """Single MC run for parallel execution."""
            np.random.seed(run_id)  # Reproducible per-run
            if shuffle_method == "returns":
                shuffled = self._shuffle_returns(data)
            elif shuffle_method == "bootstrap":
                shuffled = self._bootstrap_sample(data)
            else:
                raise ValueError(f"Unsupported shuffle method: {shuffle_method}")
            result = self.run_backtest(shuffled, symbol_spec)
            return result.to_dict()

        results = []

        if n_workers > 1 and n_runs >= 4:
            # Parallel execution for significant workloads
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(run_single_mc, i): i for i in range(n_runs)}
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential for small runs
            for i in range(n_runs):
                results.append(run_single_mc(i))

        return pd.DataFrame(results)

    def _shuffle_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle returns while preserving price structure."""
        returns = data["close"].pct_change().dropna()
        shuffled_returns = returns.sample(frac=1).reset_index(drop=True)

        # Reconstruct prices
        new_close = [data["close"].iloc[0]]
        for r in shuffled_returns:
            new_close.append(new_close[-1] * (1 + r))

        new_data = data.copy()
        new_data["close"] = new_close[: len(data)]

        # Adjust OHLC proportionally
        ratio = new_data["close"] / data["close"]
        new_data["open"] = data["open"] * ratio
        new_data["high"] = data["high"] * ratio
        new_data["low"] = data["low"] * ratio

        return new_data

    def _bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create bootstrap sample of data."""
        indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[indices].reset_index(drop=True)
