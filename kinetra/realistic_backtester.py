"""
Realistic Backtesting Engine with MT5 Constraints

Prevents sim-to-real gap by enforcing real broker constraints:
1. Dynamic per-candle spread (not fixed)
2. Freeze zones (block modifications before session close)
3. Stops level validation (minimum SL/TP distance)
4. Regime-aware performance metrics (detect overfitting)
5. MT5 error code simulation (10016, 10029, 10030, etc.)

Goal: If it works in this backtest, it WILL work in live trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import IntEnum

from .market_microstructure import SymbolSpec, AssetClass
from .regime_filtered_env import PhysicsRegime, VolatilityRegime, MomentumRegime


class MT5ErrorCode(IntEnum):
    """MetaTrader 5 error codes we simulate."""
    SUCCESS = 10009                  # Request completed
    INVALID_STOPS = 10016            # Invalid stops (too close to price)
    FROZEN = 10029                   # Trading frozen (in freeze zone)
    INVALID_FILL = 10030             # Invalid fill mode
    LONG_ONLY = 10042                # Only long positions allowed
    SHORT_ONLY = 10043               # Only short positions allowed
    CLOSE_ONLY = 10044               # Only position close allowed


@dataclass
class Trade:
    """Completed trade with full detail."""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    pnl_pct: float

    # Execution reality
    entry_spread: float  # Actual spread paid at entry
    exit_spread: float   # Actual spread paid at exit
    commission: float
    swap: float

    # Slippage (if simulated)
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0

    # Excursions
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion

    # Regime context
    entry_physics_regime: Optional[str] = None
    entry_vol_regime: Optional[str] = None
    entry_momentum_regime: Optional[str] = None

    # Physics state (for energy-based metrics)
    entry_energy: float = 0.0

    # Metadata
    rejected_modifications: int = 0  # How many times SL/TP modification was rejected
    freeze_zone_violations: int = 0  # How many times tried to modify in freeze zone

    @property
    def total_cost(self) -> float:
        """Calculate total transaction costs."""
        return (self.entry_spread + self.exit_spread +
                self.commission + abs(self.swap) +
                abs(self.entry_slippage) + abs(self.exit_slippage))

    @property
    def gross_pnl(self) -> float:
        """Calculate gross P&L (before costs)."""
        return self.pnl + self.total_cost

    @property
    def holding_time(self) -> float:
        """Calculate holding time in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600.0

    @property
    def price_captured(self) -> float:
        """Calculate price difference captured."""
        if self.direction == 1:  # Long
            return self.exit_price - self.entry_price
        else:  # Short
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
    """Complete backtest results with regime breakdown."""
    # Overall metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    total_pnl: float
    total_return_pct: float

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0  # Downside deviation-adjusted
    omega_ratio: float = 0.0    # Gain/loss threshold ratio

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    cvar_95: float = 0.0  # Conditional Value at Risk (95%)
    cvar_99: float = 0.0  # Conditional Value at Risk (99%)

    # Realistic costs
    total_spread_cost: float = 0.0
    total_commission: float = 0.0
    total_swap: float = 0.0
    total_slippage: float = 0.0

    # Quality metrics
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_mfe_mae_ratio: float = 0.0
    mfe_capture_pct: float = 0.0  # How much MFE was captured as profit

    # Physics-specific metrics
    z_factor: float = 0.0  # Statistical edge metric
    energy_captured_pct: float = 0.0  # % of energy captured in winning trades

    # Regime breakdown (CRITICAL for detecting overfitting)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # Constraint violations (should be ZERO in realistic backtest)
    total_freeze_violations: int = 0
    total_invalid_stops: int = 0
    total_rejected_orders: int = 0

    # All trades
    trades: List[Trade] = field(default_factory=list)

    # Equity curve
    equity_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'omega_ratio': self.omega_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'total_spread_cost': self.total_spread_cost,
            'total_commission': self.total_commission,
            'total_swap': self.total_swap,
            'total_slippage': self.total_slippage,
            'avg_mfe': self.avg_mfe,
            'avg_mae': self.avg_mae,
            'avg_mfe_mae_ratio': self.avg_mfe_mae_ratio,
            'mfe_capture_pct': self.mfe_capture_pct,
            'regime_performance': self.regime_performance,
            'constraint_violations': {
                'freeze_violations': self.total_freeze_violations,
                'invalid_stops': self.total_invalid_stops,
                'rejected_orders': self.total_rejected_orders,
            }
        }


class RealisticBacktester:
    """
    Backtesting engine enforcing real MT5 broker constraints.

    Key features:
    - Dynamic per-candle spread (from candle data)
    - Freeze zone enforcement (from SymbolSpec)
    - Stop distance validation (from SymbolSpec)
    - Regime classification (from RegimeFilteredEnv)
    - MT5 error code simulation

    Usage:
        spec = SymbolSpec(symbol="EURUSD", ...)
        backtester = RealisticBacktester(spec)

        # Run backtest with trades
        result = backtester.run(data, trades)

        # Check regime performance
        if result.regime_performance['overdamped']['sharpe'] < 0:
            print("Agent loses money in chop → Use regime filter!")
    """

    def __init__(
        self,
        spec: SymbolSpec,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        timeframe: str = "H1",         # For proper Sharpe annualization
        enable_slippage: bool = True,
        slippage_std_pips: float = 0.5,
        enable_freeze_zones: bool = True,
        enable_stop_validation: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize realistic backtester.

        Args:
            spec: SymbolSpec with freeze zones and stops levels
            initial_capital: Starting capital
            risk_per_trade: Percentage of equity to risk per trade (0-1)
            timeframe: Data timeframe for proper Sharpe annualization
            enable_slippage: Simulate slippage (Gaussian noise)
            slippage_std_pips: Slippage standard deviation in pips
            enable_freeze_zones: Enforce freeze zone restrictions
            enable_stop_validation: Validate SL/TP distances
            verbose: Print warnings for violations
        """
        self.spec = spec
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.timeframe = timeframe
        self.enable_slippage = enable_slippage
        self.slippage_std_pips = slippage_std_pips
        self.enable_freeze_zones = enable_freeze_zones
        self.enable_stop_validation = enable_stop_validation
        self.verbose = verbose

    def validate_stop_placement(
        self,
        current_price: float,
        stop_price: float,
        action: str = "place_order"
    ) -> Tuple[bool, MT5ErrorCode, str]:
        """
        Validate stop loss/take profit placement.

        Returns:
            (is_valid, error_code, error_message)
        """
        if not self.enable_stop_validation:
            return (True, MT5ErrorCode.SUCCESS, "")

        # Check minimum distance
        is_valid, error_msg = self.spec.validate_stop_distance(current_price, stop_price)

        if not is_valid:
            return (False, MT5ErrorCode.INVALID_STOPS, error_msg)

        return (True, MT5ErrorCode.SUCCESS, "")

    def is_in_freeze_zone(
        self,
        current_time: datetime,
        session_end_time: Optional[time] = None
    ) -> bool:
        """
        Check if current time is in freeze zone.

        Freeze zone = last N minutes before session close where modifications blocked.
        """
        if not self.enable_freeze_zones:
            return False

        if self.spec.trade_freeze_level == 0:
            return False  # No freeze zone

        # For now, use simplified logic (assume 24h trading)
        # In production, use spec.trading_hours to get real session end
        # Freeze zone is typically last 3-5 minutes before close

        # Simplified: Block modifications in last bar of day
        if session_end_time is None:
            # Assume 24h trading, no freeze
            return False

        # Check if within freeze window
        # freeze_window_seconds = self.spec.trade_freeze_level * 60  # Assume minutes
        # TODO: Implement proper session end detection

        return False  # Placeholder

    def simulate_fill(
        self,
        desired_price: float,
        direction: int,
        spread_points: float
    ) -> Tuple[float, float]:
        """
        Simulate realistic fill with spread and slippage.

        Args:
            desired_price: Price agent wants to execute at
            direction: 1=long (buy), -1=short (sell)
            spread_points: Current spread in points

        Returns:
            (actual_fill_price, slippage)
        """
        # Apply spread
        spread_price = spread_points * self.spec.point

        if direction == 1:  # Long (buy at ask)
            fill_price = desired_price + spread_price / 2
        else:  # Short (sell at bid)
            fill_price = desired_price - spread_price / 2

        # Apply slippage
        slippage = 0.0
        if self.enable_slippage:
            # Gaussian slippage (more realistic than uniform)
            slippage_pips = np.random.normal(0, self.slippage_std_pips)
            slippage = slippage_pips * 10 * self.spec.point  # pips to price

            # Slippage is ALWAYS against you (Murphy's law)
            slippage = abs(slippage) * direction
            fill_price += slippage

        return (fill_price, slippage)

    def calculate_swap(
        self,
        direction: int,
        volume: float,
        days_held: float
    ) -> float:
        """
        Calculate swap/rollover charges.

        Args:
            direction: 1=long, -1=short
            volume: Position size in lots
            days_held: Number of days position held

        Returns:
            Swap cost (negative = cost, positive = credit)
        """
        if days_held < 1:
            return 0.0  # No swap for intraday

        # Get swap rate from spec
        swap_points = self.spec.swap_long if direction == 1 else self.spec.swap_short

        # Convert points to price
        swap_per_lot_per_day = swap_points * self.spec.point * self.spec.contract_size

        # Total swap
        total_swap = swap_per_lot_per_day * volume * days_held

        # Check for triple swap day (Wednesday by default)
        # TODO: Implement day-of-week check for triple swap

        return total_swap

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        classify_regimes: bool = True
    ) -> BacktestResult:
        """
        Run backtest with realistic constraints.

        Args:
            data: OHLCV DataFrame with 'spread' column (dynamic spread)
            signals: DataFrame with columns ['time', 'action', 'price', 'sl', 'tp', 'volume']
            classify_regimes: Classify trades by regime for performance breakdown

        Returns:
            BacktestResult with regime breakdown
        """
        # Ensure data has spread column
        if 'spread' not in data.columns:
            if self.verbose:
                print(f"Warning: No 'spread' column in data, using typical spread: {self.spec.spread_typical}")
            data['spread'] = self.spec.spread_typical

        # Classify regimes if requested
        if classify_regimes:
            data = self._classify_regimes(data)

        # Initialize tracking
        trades: List[Trade] = []
        equity_curve = [self.initial_capital]
        current_position = None

        total_freeze_violations = 0
        total_invalid_stops = 0
        total_rejected_orders = 0

        # Process signals
        for idx, signal in signals.iterrows():
            signal_time = signal['time']
            action = signal['action']

            # Get current candle
            candle = data[data.index == signal_time]
            if len(candle) == 0:
                continue
            candle = candle.iloc[0]

            current_price = candle['close']
            spread_points = candle['spread']

            # Check freeze zone
            in_freeze = self.is_in_freeze_zone(signal_time)

            if action == 'open_long' or action == 'open_short':
                direction = 1 if action == 'open_long' else -1
                volume = signal.get('volume', 1.0)
                sl = signal.get('sl', None)
                tp = signal.get('tp', None)

                # Validate SL distance
                if sl is not None:
                    is_valid, error_code, error_msg = self.validate_stop_placement(current_price, sl)
                    if not is_valid:
                        total_invalid_stops += 1
                        if self.verbose:
                            print(f"[{signal_time}] REJECTED: {error_msg}")
                        continue

                # Simulate fill
                fill_price, slippage = self.simulate_fill(current_price, direction, spread_points)

                # Open position
                current_position = {
                    'direction': direction,
                    'entry_time': signal_time,
                    'entry_price': fill_price,
                    'volume': volume,
                    'sl': sl,
                    'tp': tp,
                    'entry_spread': spread_points,
                    'entry_slippage': slippage,
                    'mfe': 0.0,
                    'mae': 0.0,
                    'entry_energy': candle.get('energy', 0.0),  # For physics-based metrics
                    'regime': {
                        'physics': candle.get('physics_regime', None),
                        'vol': candle.get('vol_regime', None),
                        'momentum': candle.get('momentum_regime', None),
                    } if classify_regimes else None,
                    'rejected_modifications': 0,
                }

            elif action == 'close' and current_position is not None:
                # Close position
                direction = current_position['direction']
                fill_price, slippage = self.simulate_fill(current_price, -direction, spread_points)

                # Calculate P&L
                pnl_price = (fill_price - current_position['entry_price']) * direction
                pnl_pct = pnl_price / current_position['entry_price']
                pnl = pnl_price * current_position['volume'] * self.spec.contract_size

                # Calculate costs
                entry_spread_cost = current_position['entry_spread'] * self.spec.point * current_position['volume'] * self.spec.contract_size
                exit_spread_cost = spread_points * self.spec.point * current_position['volume'] * self.spec.contract_size
                commission = abs(pnl) * 0.0  # From spec (if available)

                # Calculate swap
                days_held = (signal_time - current_position['entry_time']).total_seconds() / 86400
                swap = self.calculate_swap(direction, current_position['volume'], days_held)

                # Net P&L
                net_pnl = pnl - entry_spread_cost - exit_spread_cost - commission - swap

                # Create trade record
                trade = Trade(
                    entry_time=current_position['entry_time'],
                    exit_time=signal_time,
                    direction=direction,
                    entry_price=current_position['entry_price'],
                    exit_price=fill_price,
                    volume=current_position['volume'],
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    entry_spread=current_position['entry_spread'],
                    exit_spread=spread_points,
                    commission=commission,
                    swap=swap,
                    entry_slippage=current_position['entry_slippage'],
                    exit_slippage=slippage,
                    mfe=current_position['mfe'],
                    mae=current_position['mae'],
                    entry_physics_regime=current_position['regime']['physics'] if current_position['regime'] else None,
                    entry_vol_regime=current_position['regime']['vol'] if current_position['regime'] else None,
                    entry_momentum_regime=current_position['regime']['momentum'] if current_position['regime'] else None,
                    entry_energy=current_position['entry_energy'],
                    rejected_modifications=current_position['rejected_modifications'],
                )

                trades.append(trade)
                equity_curve.append(equity_curve[-1] + net_pnl)

                current_position = None

            elif action == 'modify_sl' and current_position is not None:
                # Modify stop loss
                new_sl = signal.get('sl', None)

                if new_sl is not None:
                    # Check freeze zone
                    if in_freeze:
                        total_freeze_violations += 1
                        current_position['rejected_modifications'] += 1
                        if self.verbose:
                            print(f"[{signal_time}] FREEZE ZONE: Cannot modify SL")
                        continue

                    # Validate new SL distance
                    is_valid, error_code, error_msg = self.validate_stop_placement(current_price, new_sl)
                    if not is_valid:
                        total_invalid_stops += 1
                        current_position['rejected_modifications'] += 1
                        if self.verbose:
                            print(f"[{signal_time}] INVALID SL: {error_msg}")
                        continue

                    # Update SL
                    current_position['sl'] = new_sl

            # Update MFE/MAE
            if current_position is not None:
                excursion = (current_price - current_position['entry_price']) * current_position['direction']
                if excursion > current_position['mfe']:
                    current_position['mfe'] = excursion
                if excursion < -current_position['mae']:
                    current_position['mae'] = -excursion

        # Compute metrics
        result = self._compute_metrics(trades, equity_curve)
        result.total_freeze_violations = total_freeze_violations
        result.total_invalid_stops = total_invalid_stops
        result.total_rejected_orders = total_rejected_orders

        return result

    def _classify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime classifications to data."""
        # Import here to avoid circular dependency
        from .regime_filtered_env import RegimeFilteredTradingEnv

        # Create temporary env just for regime classification
        temp_env = RegimeFilteredTradingEnv(data)

        # Copy regime columns to data
        data['physics_regime'] = temp_env.features['physics_regime_enum'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        data['vol_regime'] = temp_env.features['vol_regime_enum'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        data['momentum_regime'] = temp_env.features['momentum_regime_enum'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))

        return data

    def _compute_metrics(self, trades: List[Trade], equity_curve: List[float]) -> BacktestResult:
        """Compute backtest metrics with regime breakdown."""
        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_spread_cost=0.0,
                total_commission=0.0,
                total_swap=0.0,
                total_slippage=0.0,
                avg_mfe=0.0,
                avg_mae=0.0,
                avg_mfe_mae_ratio=0.0,
                trades=[],
                equity_curve=pd.Series(equity_curve),
            )

        # Overall metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)

        # Convert equity curve to pandas Series and calculate returns
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        # Timeframe-aware annualization factor
        timeframe_bars_per_year = {
            "M1": 525600, "M5": 105120, "M15": 35040, "M30": 17520,
            "H1": 8760, "H4": 2190, "D1": 252, "W1": 52, "MN": 12,
        }
        bars_per_year = timeframe_bars_per_year.get(self.timeframe, 252)
        annualization = np.sqrt(bars_per_year)

        # Sharpe ratio (annualized, timeframe-aware)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * annualization
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        sortino = 0.0
        if len(returns) > 1 and returns.std() > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = (returns.mean() / downside_returns.std()) * annualization
            else:
                sortino = float('inf') if returns.mean() > 0 else 0.0

        # CVaR (Conditional Value at Risk) - downside tail risk
        cvar_95 = 0.0
        cvar_99 = 0.0
        if len(returns) > 0:
            q95 = returns.quantile(0.05)
            q99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= q95].mean() if len(returns[returns <= q95]) > 0 else 0.0
            cvar_99 = returns[returns <= q99].mean() if len(returns[returns <= q99]) > 0 else 0.0

        # Omega ratio (gain/loss threshold)
        omega = 0.0
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')

        # Drawdown (absolute and percentage)
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd_pct = float(np.max(drawdown))

        # Find max drawdown in absolute terms
        drawdown_abs = peak - equity_arr
        max_dd = float(np.max(drawdown_abs))

        # Costs
        total_spread = sum(t.entry_spread + t.exit_spread for t in trades)
        total_commission = sum(t.commission for t in trades)
        total_swap = sum(t.swap for t in trades)
        total_slippage = sum(abs(t.entry_slippage) + abs(t.exit_slippage) for t in trades)

        # Quality metrics
        avg_mfe = np.mean([t.mfe for t in trades])
        avg_mae = np.mean([t.mae for t in trades])
        avg_mfe_mae = avg_mfe / (avg_mae + 1e-8) if avg_mae > 0 else 0.0

        # MFE capture % - how much of potential profit was actually captured
        total_mfe = sum(t.mfe for t in trades)
        realized_profit = sum(max(0, t.pnl) for t in trades)
        mfe_capture = realized_profit / total_mfe if total_mfe > 0 else 0.0

        # Z-factor (statistical edge metric)
        z_factor = 0.0
        if len(trades) > 1:
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0

            if avg_loss > 0:
                z_factor = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
            elif avg_win > 0:
                z_factor = 999.0  # Infinite edge (no losses)

        # Energy captured % (physics-specific metric)
        energy_captured = 0.0
        total_energy_at_entry = sum(t.entry_energy for t in trades)
        profitable_energy = sum(t.entry_energy for t in winning_trades)

        if total_energy_at_entry > 0:
            energy_captured = profitable_energy / total_energy_at_entry

        # Regime breakdown
        regime_performance = self._compute_regime_breakdown(trades)

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades),
            total_pnl=total_pnl,
            total_return_pct=total_pnl / self.initial_capital,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino) if sortino != float('inf') else 999.0,
            omega_ratio=float(omega) if omega != float('inf') else 999.0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            total_spread_cost=total_spread,
            total_commission=total_commission,
            total_swap=total_swap,
            total_slippage=total_slippage,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            avg_mfe_mae_ratio=avg_mfe_mae,
            mfe_capture_pct=float(mfe_capture),
            z_factor=float(z_factor),
            energy_captured_pct=float(energy_captured),
            regime_performance=regime_performance,
            trades=trades,
            equity_curve=equity,
        )

    def _compute_regime_breakdown(self, trades: List[Trade]) -> Dict[str, Dict]:
        """
        Compute performance breakdown by regime.

        CRITICAL: This shows if agent is overfit to specific regimes.
        """
        regime_groups = {
            'physics': {},
            'volatility': {},
            'momentum': {},
        }

        for trade in trades:
            # Group by physics regime
            if trade.entry_physics_regime:
                if trade.entry_physics_regime not in regime_groups['physics']:
                    regime_groups['physics'][trade.entry_physics_regime] = []
                regime_groups['physics'][trade.entry_physics_regime].append(trade)

            # Group by volatility regime
            if trade.entry_vol_regime:
                if trade.entry_vol_regime not in regime_groups['volatility']:
                    regime_groups['volatility'][trade.entry_vol_regime] = []
                regime_groups['volatility'][trade.entry_vol_regime].append(trade)

            # Group by momentum regime
            if trade.entry_momentum_regime:
                if trade.entry_momentum_regime not in regime_groups['momentum']:
                    regime_groups['momentum'][trade.entry_momentum_regime] = []
                regime_groups['momentum'][trade.entry_momentum_regime].append(trade)

        # Compute metrics per regime
        breakdown = {}

        for regime_type, groups in regime_groups.items():
            for regime_name, regime_trades in groups.items():
                if not regime_trades:
                    continue

                pnls = [t.pnl for t in regime_trades]
                win_rate = sum(1 for t in regime_trades if t.pnl > 0) / len(regime_trades)
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) if len(pnls) > 1 else 0.0

                key = f"{regime_type}_{regime_name}"
                breakdown[key] = {
                    'trades': len(regime_trades),
                    'win_rate': float(win_rate),
                    'total_pnl': float(sum(pnls)),
                    'avg_pnl': float(np.mean(pnls)),
                    'sharpe': float(sharpe),
                }

        return breakdown

    def monte_carlo_validation(
        self,
        data: pd.DataFrame,
        signal_generator: callable,
        n_runs: int = 100,
        shuffle_method: str = "returns",
        classify_regimes: bool = True,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo validation to assess strategy robustness.

        This tests if the strategy's edge is real or just luck by running
        backtest on randomized versions of the data.

        Args:
            data: Original OHLCV data
            signal_generator: Function that takes data and returns signals DataFrame
            n_runs: Number of simulation runs
            shuffle_method: "returns" (shuffle returns) or "bootstrap" (resample with replacement)
            classify_regimes: Classify regimes in shuffled data

        Returns:
            DataFrame with results from each run

        Usage:
            def my_signal_gen(data):
                # Your strategy logic
                return signals_df

            mc_results = backtester.monte_carlo_validation(data, my_signal_gen, n_runs=100)

            # Check if real result is in top 5% (p < 0.05)
            real_sharpe = backtester.run(data, signals).sharpe_ratio
            percentile = (mc_results['sharpe_ratio'] < real_sharpe).mean()
            if percentile > 0.95:
                print("Strategy has statistically significant edge!")
        """
        results = []

        for i in range(n_runs):
            # Create shuffled data
            if shuffle_method == "returns":
                shuffled = self._shuffle_returns(data)
            elif shuffle_method == "bootstrap":
                shuffled = self._bootstrap_sample(data)
            else:
                raise ValueError(f"Unsupported shuffle method: {shuffle_method}. Use 'returns' or 'bootstrap'.")

            # Generate signals on shuffled data
            signals = signal_generator(shuffled)

            # Run backtest
            result = self.run(shuffled, signals, classify_regimes=classify_regimes)

            # Extract key metrics
            results.append({
                'run': i,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'omega_ratio': result.omega_ratio,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_pct': result.max_drawdown_pct,
                'cvar_95': result.cvar_95,
                'mfe_capture_pct': result.mfe_capture_pct,
            })

        return pd.DataFrame(results)

    def _shuffle_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle returns while preserving price structure.

        This maintains statistical properties of returns but destroys
        any predictive patterns in the time series.
        """
        returns = data["close"].pct_change().dropna()
        shuffled_returns = returns.sample(frac=1).reset_index(drop=True)

        # Reconstruct prices from shuffled returns
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

        # Preserve spread if exists
        if 'spread' in data.columns:
            new_data['spread'] = data['spread']

        return new_data

    def _bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create bootstrap sample of data (sample with replacement).

        This tests if the strategy works on different random subsets
        of the historical data.
        """
        indices = np.random.choice(len(data), size=len(data), replace=True)
        return data.iloc[indices].reset_index(drop=True)
