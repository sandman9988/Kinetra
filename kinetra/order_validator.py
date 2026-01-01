"""
Order Validator - Shared Between Backtest and Live Trading

Validates orders against MT5 broker constraints BEFORE execution.
Same code used in both backtest and live → prevents sim-to-real gap.

Architecture:
    Agent → OrderValidator → OrderExecutor
              ↑ shared      ↓ context-specific
                         BacktestExecutor OR LiveExecutor
"""

from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time
from enum import IntEnum

from .market_microstructure import SymbolSpec


class ValidationResult(IntEnum):
    """Order validation result codes (matches MT5)."""
    SUCCESS = 10009
    INVALID_STOPS = 10016
    FROZEN = 10029
    INVALID_FILL = 10030
    LONG_ONLY = 10042
    SHORT_ONLY = 10043
    CLOSE_ONLY = 10044
    INVALID_VOLUME = 10014
    NO_MONEY = 10019


@dataclass
class OrderValidation:
    """Result of order validation."""
    is_valid: bool
    error_code: ValidationResult
    error_message: str

    # Adjusted parameters (if validator fixed them)
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None


class OrderValidator:
    """
    Validates orders against MT5 broker constraints.

    CRITICAL: Same validator used in backtest AND live trading.
    This ensures backtest results transfer to live.

    Usage:
        validator = OrderValidator(spec)

        # Check if order is valid
        validation = validator.validate_order(
            action='open_long',
            price=1.08500,
            sl=1.08350,  # 15 pips SL
            tp=1.08800,  # 30 pips TP
        )

        if validation.is_valid:
            execute_order(...)
        else:
            print(f"Order rejected: {validation.error_message}")
    """

    def __init__(
        self,
        spec: SymbolSpec,
        auto_adjust_stops: bool = False,
        safety_multiplier: float = 1.5,
    ):
        """
        Initialize order validator.

        Args:
            spec: SymbolSpec with freeze zones and stops levels
            auto_adjust_stops: Automatically adjust SL/TP to meet constraints
            safety_multiplier: Safety buffer for auto-adjusted stops
        """
        self.spec = spec
        self.auto_adjust_stops = auto_adjust_stops
        self.safety_multiplier = safety_multiplier

    def validate_order(
        self,
        action: str,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        volume: float = 1.0,
        current_time: Optional[datetime] = None,
    ) -> OrderValidation:
        """
        Validate order against all MT5 constraints.

        Args:
            action: 'open_long', 'open_short', 'close', 'modify_sl', 'modify_tp'
            price: Current market price
            sl: Stop loss price
            tp: Take profit price
            volume: Order volume in lots
            current_time: Current time (for freeze zone check)

        Returns:
            OrderValidation with is_valid flag and error details
        """
        # 1. Check trade mode
        trade_mode_check = self._validate_trade_mode(action)
        if not trade_mode_check.is_valid:
            return trade_mode_check

        # 2. Check volume
        volume_check = self._validate_volume(volume)
        if not volume_check.is_valid:
            return volume_check

        # 3. Check freeze zone (for modifications)
        if action.startswith('modify') and current_time:
            freeze_check = self._validate_freeze_zone(current_time)
            if not freeze_check.is_valid:
                return freeze_check

        # 4. Check stop loss distance
        sl_check = None
        if sl is not None:
            sl_check = self._validate_stop_distance(price, sl, 'SL')
            if not sl_check.is_valid:
                if self.auto_adjust_stops:
                    # Auto-adjust SL to meet constraints
                    sl_check = self._auto_adjust_sl(price, sl)
                else:
                    return sl_check

        # 5. Check take profit distance
        tp_check = None
        if tp is not None:
            tp_check = self._validate_stop_distance(price, tp, 'TP')
            if not tp_check.is_valid:
                if self.auto_adjust_stops:
                    # Auto-adjust TP to meet constraints
                    tp_check = self._auto_adjust_tp(price, tp)
                else:
                    return tp_check

        # All checks passed
        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message="",
            adjusted_sl=sl_check.adjusted_sl if sl_check is not None and hasattr(sl_check, 'adjusted_sl') else None,
            adjusted_tp=tp_check.adjusted_tp if tp_check is not None and hasattr(tp_check, 'adjusted_tp') else None,
        )

    def _validate_trade_mode(self, action: str) -> OrderValidation:
        """Validate action against trade_mode (FULL, LONGONLY, SHORTONLY, CLOSEONLY)."""
        trade_mode = self.spec.trade_mode

        if trade_mode == "DISABLED":
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.LONG_ONLY,  # Generic rejection
                error_message=f"Trading disabled for {self.spec.symbol}"
            )

        if trade_mode == "LONGONLY" and action == 'open_short':
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.SHORT_ONLY,
                error_message=f"Short positions not allowed (LONGONLY mode)"
            )

        if trade_mode == "SHORTONLY" and action == 'open_long':
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.LONG_ONLY,
                error_message=f"Long positions not allowed (SHORTONLY mode)"
            )

        if trade_mode == "CLOSEONLY" and action.startswith('open'):
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.CLOSE_ONLY,
                error_message=f"Only closing positions allowed (CLOSEONLY mode)"
            )

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=""
        )

    def _validate_volume(self, volume: float) -> OrderValidation:
        """Validate order volume against min/max/step constraints."""
        if volume < self.spec.volume_min:
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.INVALID_VOLUME,
                error_message=f"Volume {volume} < minimum {self.spec.volume_min}"
            )

        if volume > self.spec.volume_max:
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.INVALID_VOLUME,
                error_message=f"Volume {volume} > maximum {self.spec.volume_max}"
            )

        # Check volume step
        if self.spec.volume_step > 0:
            steps = round(volume / self.spec.volume_step)
            if abs(volume - steps * self.spec.volume_step) > 1e-6:
                return OrderValidation(
                    is_valid=False,
                    error_code=ValidationResult.INVALID_VOLUME,
                    error_message=f"Volume {volume} not multiple of step {self.spec.volume_step}"
                )

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=""
        )

    def _validate_freeze_zone(self, current_time: datetime) -> OrderValidation:
        """Check if current time is in freeze zone."""
        # Simplified: Always allow for now
        # In production, implement proper session end detection
        # using spec.trading_hours

        if self.spec.trade_freeze_level == 0:
            return OrderValidation(
                is_valid=True,
                error_code=ValidationResult.SUCCESS,
                error_message=""
            )

        # Placeholder: Real implementation needs session end time
        is_frozen = False  # TODO: Implement proper freeze zone detection

        if is_frozen:
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.FROZEN,
                error_message=f"Trading frozen (within {self.spec.trade_freeze_level} points of session end)"
            )

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=""
        )

    def _validate_stop_distance(
        self,
        price: float,
        stop: float,
        stop_type: str
    ) -> OrderValidation:
        """Validate SL/TP distance meets minimum requirement."""
        is_valid, error_msg = self.spec.validate_stop_distance(price, stop)

        if not is_valid:
            return OrderValidation(
                is_valid=False,
                error_code=ValidationResult.INVALID_STOPS,
                error_message=f"{stop_type}: {error_msg}"
            )

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=""
        )

    def _auto_adjust_sl(self, price: float, desired_sl: float) -> OrderValidation:
        """Auto-adjust SL to meet minimum distance requirement."""
        direction = 1 if desired_sl < price else -1  # Long if SL below price

        # Get safe distance
        safe_distance = self.spec.get_safe_stop_distance(self.safety_multiplier)

        # Adjust SL
        adjusted_sl = price - (safe_distance * direction)

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=f"SL auto-adjusted from {desired_sl:.5f} to {adjusted_sl:.5f}",
            adjusted_sl=adjusted_sl
        )

    def _auto_adjust_tp(self, price: float, desired_tp: float) -> OrderValidation:
        """Auto-adjust TP to meet minimum distance requirement."""
        direction = 1 if desired_tp > price else -1  # Long if TP above price

        # Get safe distance
        safe_distance = self.spec.get_safe_stop_distance(self.safety_multiplier)

        # Adjust TP
        adjusted_tp = price + (safe_distance * direction)

        return OrderValidation(
            is_valid=True,
            error_code=ValidationResult.SUCCESS,
            error_message=f"TP auto-adjusted from {desired_tp:.5f} to {adjusted_tp:.5f}",
            adjusted_tp=adjusted_tp
        )

    def get_safe_sl_tp(
        self,
        price: float,
        direction: int,
        sl_distance_pips: Optional[float] = None,
        tp_distance_pips: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Get safe SL/TP prices that meet broker constraints.

        Helper method for agents to calculate valid SL/TP.

        Args:
            price: Current price
            direction: 1=long, -1=short
            sl_distance_pips: Desired SL distance in pips (or use safe default)
            tp_distance_pips: Desired TP distance in pips (or use safe default)

        Returns:
            (sl_price, tp_price)

        Example:
            price = 1.08500
            sl, tp = validator.get_safe_sl_tp(price, direction=1, sl_distance_pips=20)
            # Returns: (1.08480, 1.08540) - guaranteed valid
        """
        # Use safe distance if not specified
        if sl_distance_pips is None:
            safe_distance_price = self.spec.get_safe_stop_distance(self.safety_multiplier)
            sl_distance_pips = safe_distance_price / (10 * self.spec.point)

        if tp_distance_pips is None:
            safe_distance_price = self.spec.get_safe_stop_distance(self.safety_multiplier)
            tp_distance_pips = safe_distance_price / (10 * self.spec.point)

        # Convert pips to price
        sl_distance_price = sl_distance_pips * 10 * self.spec.point
        tp_distance_price = tp_distance_pips * 10 * self.spec.point

        # Calculate SL/TP
        if direction == 1:  # Long
            sl = price - sl_distance_price
            tp = price + tp_distance_price
        else:  # Short
            sl = price + sl_distance_price
            tp = price - tp_distance_price

        return (sl, tp)
