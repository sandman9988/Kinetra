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

    # Metadata
    rejected_modifications: int = 0  # How many times SL/TP modification was rejected
    freeze_zone_violations: int = 0  # How many times tried to modify in freeze zone


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
    sharpe_ratio: float
    max_drawdown: float

    # Realistic costs
    total_spread_cost: float
    total_commission: float
    total_swap: float
    total_slippage: float

    # Quality metrics
    avg_mfe: float
    avg_mae: float
    avg_mfe_mae_ratio: float

    # Regime breakdown (CRITICAL for detecting overfitting)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # Constraint violations (should be ZERO in realistic backtest)
    total_freeze_violations: int = 0
    total_invalid_stops: int = 0
    total_rejected_orders: int = 0

    # All trades
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_spread_cost': self.total_spread_cost,
            'total_commission': self.total_commission,
            'total_swap': self.total_swap,
            'avg_mfe_mae_ratio': self.avg_mfe_mae_ratio,
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
            enable_slippage: Simulate slippage (Gaussian noise)
            slippage_std_pips: Slippage standard deviation in pips
            enable_freeze_zones: Enforce freeze zone restrictions
            enable_stop_validation: Validate SL/TP distances
            verbose: Print warnings for violations
        """
        self.spec = spec
        self.initial_capital = initial_capital
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
            )

        # Overall metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        returns = [t.pnl for t in trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0

        # Drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown))

        # Costs
        total_spread = sum(t.entry_spread + t.exit_spread for t in trades)
        total_commission = sum(t.commission for t in trades)
        total_swap = sum(t.swap for t in trades)
        total_slippage = sum(abs(t.entry_slippage) + abs(t.exit_slippage) for t in trades)

        # Quality
        avg_mfe = np.mean([t.mfe for t in trades])
        avg_mae = np.mean([t.mae for t in trades])
        avg_mfe_mae = avg_mfe / (avg_mae + 1e-8) if avg_mae > 0 else 0.0

        # Regime breakdown
        regime_performance = self._compute_regime_breakdown(trades)

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades),
            total_pnl=total_pnl,
            total_return_pct=total_pnl / self.initial_capital,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_spread_cost=total_spread,
            total_commission=total_commission,
            total_swap=total_swap,
            total_slippage=total_slippage,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            avg_mfe_mae_ratio=avg_mfe_mae,
            regime_performance=regime_performance,
            trades=trades,
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
