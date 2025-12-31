"""
Order Executor - Modular Interface for Backtest and Live Trading

Dependency injection pattern:
- Agent uses OrderExecutor interface (doesn't know if backtest or live)
- BacktestExecutor implements interface for backtesting
- LiveExecutor implements interface for live MT5 trading

This allows SAME agent code to run in both contexts!

Architecture:
    Agent.decide() → OrderExecutor.execute()
                         ↓
              BacktestExecutor OR LiveExecutor
              (same interface, different implementation)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .order_validator import OrderValidator, OrderValidation
from .market_microstructure import SymbolSpec


@dataclass
class OrderResult:
    """Result of order execution."""
    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None

    # Actual execution details
    actual_sl: Optional[float] = None
    actual_tp: Optional[float] = None
    slippage: float = 0.0
    spread_paid: float = 0.0


class OrderExecutor(ABC):
    """
    Abstract interface for order execution.

    Implementations:
    - BacktestExecutor: Simulates execution in backtest
    - LiveExecutor: Executes on real MT5 account

    Agent code is agnostic to implementation!
    """

    def __init__(self, spec: SymbolSpec, validator: OrderValidator):
        """
        Initialize executor.

        Args:
            spec: SymbolSpec for the instrument
            validator: OrderValidator (shared logic)
        """
        self.spec = spec
        self.validator = validator

    @abstractmethod
    def execute_order(
        self,
        action: str,
        volume: float = 1.0,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> OrderResult:
        """
        Execute order (open, close, modify).

        Args:
            action: 'open_long', 'open_short', 'close', 'modify_sl', 'modify_tp'
            volume: Order volume in lots
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def get_current_price(self) -> float:
        """Get current market price."""
        pass

    @abstractmethod
    def get_current_spread(self) -> float:
        """Get current spread in points."""
        pass

    @abstractmethod
    def get_current_time(self) -> datetime:
        """Get current time."""
        pass


class BacktestExecutor(OrderExecutor):
    """
    Backtest implementation of OrderExecutor.

    Simulates order execution with realistic constraints.
    Uses same validator as live trading!
    """

    def __init__(
        self,
        spec: SymbolSpec,
        validator: OrderValidator,
        data: 'pd.DataFrame',  # OHLCV data with 'spread' column
    ):
        """
        Initialize backtest executor.

        Args:
            spec: SymbolSpec
            validator: OrderValidator (shared with live)
            data: Historical data for backtest
        """
        super().__init__(spec, validator)
        self.data = data
        self.current_idx = 0

        # Track open position
        self.position: Optional[Dict[str, Any]] = None

    def execute_order(
        self,
        action: str,
        volume: float = 1.0,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> OrderResult:
        """Execute order in backtest."""
        # Get current market state
        current_price = self.get_current_price()
        current_time = self.get_current_time()

        # Validate order
        validation = self.validator.validate_order(
            action=action,
            price=current_price,
            sl=sl,
            tp=tp,
            volume=volume,
            current_time=current_time,
        )

        if not validation.is_valid:
            return OrderResult(
                success=False,
                error_code=validation.error_code,
                error_message=validation.error_message
            )

        # Use adjusted stops if validator modified them
        actual_sl = validation.adjusted_sl if validation.adjusted_sl else sl
        actual_tp = validation.adjusted_tp if validation.adjusted_tp else tp

        # Simulate fill
        fill_price, slippage, spread_paid = self._simulate_fill(current_price, action)

        # Execute action
        if action == 'open_long':
            self.position = {
                'direction': 1,
                'entry_price': fill_price,
                'volume': volume,
                'sl': actual_sl,
                'tp': actual_tp,
                'entry_time': current_time,
            }
            return OrderResult(
                success=True,
                order_id=f"backtest_{current_time.timestamp()}",
                fill_price=fill_price,
                actual_sl=actual_sl,
                actual_tp=actual_tp,
                slippage=slippage,
                spread_paid=spread_paid,
            )

        elif action == 'open_short':
            self.position = {
                'direction': -1,
                'entry_price': fill_price,
                'volume': volume,
                'sl': actual_sl,
                'tp': actual_tp,
                'entry_time': current_time,
            }
            return OrderResult(
                success=True,
                order_id=f"backtest_{current_time.timestamp()}",
                fill_price=fill_price,
                actual_sl=actual_sl,
                actual_tp=actual_tp,
                slippage=slippage,
                spread_paid=spread_paid,
            )

        elif action == 'close':
            if self.position is None:
                return OrderResult(
                    success=False,
                    error_message="No position to close"
                )

            self.position = None
            return OrderResult(
                success=True,
                fill_price=fill_price,
                slippage=slippage,
                spread_paid=spread_paid,
            )

        elif action == 'modify_sl':
            if self.position is None:
                return OrderResult(
                    success=False,
                    error_message="No position to modify"
                )

            self.position['sl'] = actual_sl
            return OrderResult(
                success=True,
                actual_sl=actual_sl,
            )

        elif action == 'modify_tp':
            if self.position is None:
                return OrderResult(
                    success=False,
                    error_message="No position to modify"
                )

            self.position['tp'] = actual_tp
            return OrderResult(
                success=True,
                actual_tp=actual_tp,
            )

        return OrderResult(
            success=False,
            error_message=f"Unknown action: {action}"
        )

    def _simulate_fill(
        self,
        price: float,
        action: str
    ) -> tuple[float, float, float]:
        """
        Simulate realistic fill with spread and slippage.

        Returns:
            (fill_price, slippage, spread_paid)
        """
        spread_points = self.get_current_spread()
        spread_price = spread_points * self.spec.point

        # Apply spread
        if action in ['open_long', 'close']:
            direction = 1 if action == 'open_long' else -1
            fill_price = price + (spread_price / 2) * direction
        else:
            direction = -1 if action == 'open_short' else 1
            fill_price = price + (spread_price / 2) * direction

        # Simplified slippage (1 point against you)
        slippage = self.spec.point * (1 if action.startswith('open') else -1)
        fill_price += slippage

        return (fill_price, slippage, spread_price)

    def get_current_price(self) -> float:
        """Get current price from backtest data."""
        return self.data.iloc[self.current_idx]['close']

    def get_current_spread(self) -> float:
        """Get current spread from backtest data."""
        if 'spread' in self.data.columns:
            return self.data.iloc[self.current_idx]['spread']
        return self.spec.spread_typical

    def get_current_time(self) -> datetime:
        """Get current timestamp from backtest data."""
        return self.data.index[self.current_idx]

    def step_forward(self):
        """Move to next bar in backtest."""
        self.current_idx += 1


class LiveExecutor(OrderExecutor):
    """
    Live trading implementation of OrderExecutor.

    Executes orders on real MT5 account via MetaApi.
    Uses SAME validator as backtest!

    NOTE: This is a skeleton - requires MetaApi SDK integration.
    """

    def __init__(
        self,
        spec: SymbolSpec,
        validator: OrderValidator,
        mt5_connection,  # MetaApi connection object
    ):
        """
        Initialize live executor.

        Args:
            spec: SymbolSpec
            validator: OrderValidator (SAME as backtest)
            mt5_connection: MetaApi connection
        """
        super().__init__(spec, validator)
        self.mt5 = mt5_connection

    def execute_order(
        self,
        action: str,
        volume: float = 1.0,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> OrderResult:
        """Execute order on live MT5 account."""
        # Get current market state
        current_price = self.get_current_price()
        current_time = self.get_current_time()

        # SAME validation as backtest!
        validation = self.validator.validate_order(
            action=action,
            price=current_price,
            sl=sl,
            tp=tp,
            volume=volume,
            current_time=current_time,
        )

        if not validation.is_valid:
            return OrderResult(
                success=False,
                error_code=validation.error_code,
                error_message=validation.error_message
            )

        # Use adjusted stops if validator modified them
        actual_sl = validation.adjusted_sl if validation.adjusted_sl else sl
        actual_tp = validation.adjusted_tp if validation.adjusted_tp else tp

        # Execute on MT5 via MetaApi
        try:
            if action == 'open_long':
                result = self._execute_mt5_order(
                    symbol=self.spec.symbol,
                    order_type='ORDER_TYPE_BUY',
                    volume=volume,
                    sl=actual_sl,
                    tp=actual_tp,
                    comment=comment or "kinetra_agent"
                )

            elif action == 'open_short':
                result = self._execute_mt5_order(
                    symbol=self.spec.symbol,
                    order_type='ORDER_TYPE_SELL',
                    volume=volume,
                    sl=actual_sl,
                    tp=actual_tp,
                    comment=comment or "kinetra_agent"
                )

            elif action == 'close':
                result = self._close_mt5_position()

            elif action == 'modify_sl':
                result = self._modify_mt5_sl(actual_sl)

            elif action == 'modify_tp':
                result = self._modify_mt5_tp(actual_tp)

            else:
                return OrderResult(
                    success=False,
                    error_message=f"Unknown action: {action}"
                )

            return result

        except Exception as e:
            return OrderResult(
                success=False,
                error_message=f"MT5 execution error: {str(e)}"
            )

    def _execute_mt5_order(self, symbol: str, order_type: str, volume: float,
                          sl: Optional[float], tp: Optional[float], comment: str) -> OrderResult:
        """Execute MT5 order via MetaApi."""
        # Placeholder: Real implementation uses MetaApi SDK
        # Example:
        # result = await self.mt5.trade(symbol=symbol, actionType=order_type, ...)
        raise NotImplementedError("MT5 execution requires MetaApi SDK integration")

    def _close_mt5_position(self) -> OrderResult:
        """Close MT5 position."""
        raise NotImplementedError("MT5 execution requires MetaApi SDK integration")

    def _modify_mt5_sl(self, new_sl: float) -> OrderResult:
        """Modify MT5 stop loss."""
        raise NotImplementedError("MT5 execution requires MetaApi SDK integration")

    def _modify_mt5_tp(self, new_tp: float) -> OrderResult:
        """Modify MT5 take profit."""
        raise NotImplementedError("MT5 execution requires MetaApi SDK integration")

    def get_current_price(self) -> float:
        """Get current price from MT5."""
        # Placeholder: Real implementation queries MT5
        # Example:
        # price_data = await self.mt5.getSymbolPrice(self.spec.symbol)
        # return price_data['bid']
        raise NotImplementedError("MT5 price fetch requires MetaApi SDK integration")

    def get_current_spread(self) -> float:
        """Get current spread from MT5."""
        # Placeholder: Real implementation queries MT5
        raise NotImplementedError("MT5 spread fetch requires MetaApi SDK integration")

    def get_current_time(self) -> datetime:
        """Get current MT5 server time."""
        return datetime.now()


def create_executor(
    spec: SymbolSpec,
    mode: str = 'backtest',
    **kwargs
) -> OrderExecutor:
    """
    Factory function to create appropriate executor.

    Args:
        spec: SymbolSpec
        mode: 'backtest' or 'live'
        **kwargs: Context-specific arguments

    Returns:
        OrderExecutor (backtest or live)

    Example:
        # In backtest
        executor = create_executor(spec, mode='backtest', data=historical_data)

        # In live trading (SAME agent code!)
        executor = create_executor(spec, mode='live', mt5_connection=mt5_conn)

        # Agent code (works in both contexts)
        result = executor.execute_order('open_long', volume=1.0, sl=1.08350)
    """
    # Create validator (shared between backtest and live)
    validator = OrderValidator(spec, auto_adjust_stops=True)

    if mode == 'backtest':
        return BacktestExecutor(
            spec=spec,
            validator=validator,
            data=kwargs['data']
        )

    elif mode == 'live':
        return LiveExecutor(
            spec=spec,
            validator=validator,
            mt5_connection=kwargs['mt5_connection']
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
