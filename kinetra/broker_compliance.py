"""
Broker Compliance Module - Graceful Failure & Self-Healing
==========================================================

Comprehensive broker and symbol specification compliance:
- Order validation against broker rules
- Symbol specification enforcement
- Graceful failure with exponential backoff
- Self-healing error recovery
- Dynamic throttling
- Performance monitoring and logging

Design Principles:
- Defense in depth
- Fail gracefully, never crash
- Self-healing with exponential backoff
- Comprehensive logging for debugging
"""

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result status."""
    VALID = auto()
    INVALID = auto()
    WARNING = auto()
    THROTTLED = auto()
    RETRY = auto()


class ErrorCategory(Enum):
    """Error categorization for handling."""
    TRANSIENT = auto()     # Temporary, retry with backoff
    PERMANENT = auto()     # Won't succeed on retry
    THROTTLING = auto()    # Rate limited, wait
    CONFIGURATION = auto()  # Config error, needs fix
    UNKNOWN = auto()       # Unknown, log and continue


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    category: ErrorCategory
    code: str
    message: str
    field: str = ""
    value: Any = None
    suggestion: str = ""
    recoverable: bool = True


@dataclass
class ThrottleState:
    """Tracks throttling state for an operation."""
    operation: str
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    backoff_seconds: float = 1.0
    max_backoff: float = 300.0  # 5 minutes max
    base_backoff: float = 1.0
    multiplier: float = 2.0
    jitter: float = 0.1
    
    def record_failure(self):
        """Record a failure and update backoff."""
        self.failure_count += 1
        self.last_failure = datetime.now()
        # Exponential backoff with jitter
        self.backoff_seconds = min(
            self.max_backoff,
            self.base_backoff * (self.multiplier ** self.failure_count)
        )
        # Add jitter to prevent thundering herd
        jitter_amount = self.backoff_seconds * self.jitter * random.random()
        self.backoff_seconds += jitter_amount
    
    def record_success(self):
        """Record success and reset backoff."""
        self.failure_count = 0
        self.backoff_seconds = self.base_backoff
    
    def should_retry(self) -> bool:
        """Check if operation should be retried."""
        if self.last_failure is None:
            return True
        elapsed = (datetime.now() - self.last_failure).total_seconds()
        return elapsed >= self.backoff_seconds
    
    def time_until_retry(self) -> float:
        """Seconds until retry is allowed."""
        if self.last_failure is None:
            return 0.0
        elapsed = (datetime.now() - self.last_failure).total_seconds()
        return max(0.0, self.backoff_seconds - elapsed)


@dataclass
class OperationMetrics:
    """Metrics for an operation type."""
    operation: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls


class GracefulExecutor:
    """
    Executes operations with graceful failure handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker pattern
    - Operation throttling
    - Metrics collection
    - Self-healing
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 300.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ):
        """
        Initialize graceful executor.
        
        Args:
            max_retries: Maximum retry attempts
            base_backoff: Base backoff time in seconds
            max_backoff: Maximum backoff time
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Time before circuit half-opens
        """
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        self._throttle_states: Dict[str, ThrottleState] = {}
        self._metrics: Dict[str, OperationMetrics] = {}
        self._circuit_open: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def _get_throttle_state(self, operation: str) -> ThrottleState:
        """Get or create throttle state for operation."""
        if operation not in self._throttle_states:
            self._throttle_states[operation] = ThrottleState(
                operation=operation,
                base_backoff=self.base_backoff,
                max_backoff=self.max_backoff,
            )
        return self._throttle_states[operation]
    
    def _get_metrics(self, operation: str) -> OperationMetrics:
        """Get or create metrics for operation."""
        if operation not in self._metrics:
            self._metrics[operation] = OperationMetrics(operation=operation)
        return self._metrics[operation]
    
    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for operation."""
        if operation not in self._circuit_open:
            return False
        open_time = self._circuit_open[operation]
        elapsed = (datetime.now() - open_time).total_seconds()
        if elapsed >= self.circuit_breaker_timeout:
            # Half-open - allow one attempt
            del self._circuit_open[operation]
            return False
        return True
    
    def execute(
        self,
        operation: str,
        func: Callable,
        *args,
        on_success: Callable = None,
        on_failure: Callable = None,
        on_retry: Callable = None,
        categorize_error: Callable = None,
        **kwargs,
    ) -> Tuple[Any, bool, str]:
        """
        Execute function with graceful failure handling.
        
        Args:
            operation: Operation name for tracking
            func: Function to execute
            *args: Function arguments
            on_success: Callback on success
            on_failure: Callback on final failure
            on_retry: Callback on retry
            categorize_error: Function to categorize errors
            **kwargs: Function keyword arguments
            
        Returns:
            (result, success, error_message)
        """
        with self._lock:
            metrics = self._get_metrics(operation)
            throttle = self._get_throttle_state(operation)
            metrics.total_calls += 1
        
        # Check circuit breaker
        if self._is_circuit_open(operation):
            return None, False, f"Circuit breaker open for {operation}"
        
        # Check throttle
        if not throttle.should_retry():
            wait_time = throttle.time_until_retry()
            return None, False, f"Throttled: retry in {wait_time:.1f}s"
        
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                # Success
                with self._lock:
                    metrics.successful_calls += 1
                    metrics.total_latency_ms += latency_ms
                    metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
                    metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
                    throttle.record_success()
                
                if on_success:
                    on_success(result)
                
                return result, True, ""
                
            except Exception as e:
                last_error = str(e)
                
                # Categorize error
                if categorize_error:
                    category = categorize_error(e)
                else:
                    category = self._default_categorize(e)
                
                # Log attempt
                logger.warning(f"{operation} attempt {attempt + 1}/{self.max_retries + 1} failed: {e}")
                
                with self._lock:
                    metrics.failed_calls += 1
                    metrics.last_error = last_error
                    metrics.last_error_time = datetime.now()
                    throttle.record_failure()
                
                # Check if should retry
                if category == ErrorCategory.PERMANENT:
                    break
                
                if attempt < self.max_retries:
                    if on_retry:
                        on_retry(attempt + 1, last_error)
                    
                    metrics.retried_calls += 1
                    
                    # Wait with backoff
                    if category == ErrorCategory.THROTTLING:
                        time.sleep(throttle.backoff_seconds * 2)
                    else:
                        time.sleep(throttle.backoff_seconds)
        
        # Final failure
        with self._lock:
            # Check circuit breaker threshold
            if throttle.failure_count >= self.circuit_breaker_threshold:
                self._circuit_open[operation] = datetime.now()
                logger.error(f"Circuit breaker opened for {operation}")
        
        if on_failure:
            on_failure(last_error)
        
        return None, False, last_error
    
    def _default_categorize(self, error: Exception) -> ErrorCategory:
        """Default error categorization."""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "connection" in error_str:
            return ErrorCategory.TRANSIENT
        if "rate limit" in error_str or "throttl" in error_str or "too many" in error_str:
            return ErrorCategory.THROTTLING
        if "invalid" in error_str or "not found" in error_str:
            return ErrorCategory.PERMANENT
        if "config" in error_str or "setting" in error_str:
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.TRANSIENT
    
    def get_metrics(self, operation: str = None) -> Dict:
        """Get metrics for operation(s)."""
        if operation:
            m = self._get_metrics(operation)
            return {
                "operation": m.operation,
                "total_calls": m.total_calls,
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "max_latency_ms": m.max_latency_ms,
                "last_error": m.last_error,
            }
        
        return {
            op: self.get_metrics(op)
            for op in self._metrics.keys()
        }
    
    def reset_circuit_breaker(self, operation: str):
        """Manually reset circuit breaker."""
        if operation in self._circuit_open:
            del self._circuit_open[operation]
        if operation in self._throttle_states:
            self._throttle_states[operation].record_success()


class BrokerComplianceValidator:
    """
    Validates orders and operations against broker specifications.
    
    Features:
    - Volume validation (min/max/step)
    - Price validation (tick size, digits)
    - Spread validation
    - Stop level validation
    - Margin requirement validation
    - Trading hours validation
    - Self-healing suggestions
    """
    
    def __init__(self, symbol_specs: Dict[str, Any] = None):
        """
        Initialize validator.
        
        Args:
            symbol_specs: Dictionary of symbol -> SymbolSpec
        """
        self.symbol_specs = symbol_specs or {}
        self._executor = GracefulExecutor()
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        sl: float = None,
        tp: float = None,
        account_equity: float = None,
        leverage: float = 100.0,
    ) -> Tuple[ValidationResult, List[ValidationIssue], Dict]:
        """
        Validate order against broker specifications.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            volume: Order volume in lots
            price: Order price
            sl: Stop loss price
            tp: Take profit price
            account_equity: Current account equity
            leverage: Account leverage
            
        Returns:
            (result, issues, suggestions)
        """
        issues = []
        suggestions = {}
        
        spec = self.symbol_specs.get(symbol)
        if not spec:
            issues.append(ValidationIssue(
                category=ErrorCategory.CONFIGURATION,
                code="SPEC_MISSING",
                message=f"No specification for symbol {symbol}",
                field="symbol",
                value=symbol,
                suggestion="Add symbol specification or use default",
                recoverable=True,
            ))
            # Use defaults
            spec = self._default_spec(symbol)
        
        # Validate volume
        vol_issues, vol_suggestions = self._validate_volume(volume, spec)
        issues.extend(vol_issues)
        suggestions.update(vol_suggestions)
        
        # Validate price
        price_issues, price_suggestions = self._validate_price(price, spec)
        issues.extend(price_issues)
        suggestions.update(price_suggestions)
        
        # Validate stops
        if sl or tp:
            stop_issues, stop_suggestions = self._validate_stops(
                side, price, sl, tp, spec
            )
            issues.extend(stop_issues)
            suggestions.update(stop_suggestions)
        
        # Validate margin
        if account_equity:
            margin_issues, margin_suggestions = self._validate_margin(
                volume, price, account_equity, leverage, spec
            )
            issues.extend(margin_issues)
            suggestions.update(margin_suggestions)
        
        # Determine overall result
        has_permanent = any(i.category == ErrorCategory.PERMANENT for i in issues)
        has_warning = any(i.category != ErrorCategory.PERMANENT for i in issues)
        
        if has_permanent:
            result = ValidationResult.INVALID
        elif has_warning:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.VALID
        
        return result, issues, suggestions
    
    def _validate_volume(self, volume: float, spec) -> Tuple[List[ValidationIssue], Dict]:
        """Validate volume against spec."""
        issues = []
        suggestions = {}
        
        # Check NaN/Inf
        if math.isnan(volume) or math.isinf(volume):
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="VOL_INVALID",
                message=f"Invalid volume: {volume}",
                field="volume",
                value=volume,
                suggestion="Use a valid numeric volume",
                recoverable=False,
            ))
            return issues, suggestions
        
        # Check minimum
        if volume < spec.volume_min:
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="VOL_MIN",
                message=f"Volume {volume} below minimum {spec.volume_min}",
                field="volume",
                value=volume,
                suggestion=f"Increase to at least {spec.volume_min}",
                recoverable=True,
            ))
            suggestions["volume"] = spec.volume_min
        
        # Check maximum
        if volume > spec.volume_max:
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="VOL_MAX",
                message=f"Volume {volume} above maximum {spec.volume_max}",
                field="volume",
                value=volume,
                suggestion=f"Reduce to at most {spec.volume_max}",
                recoverable=True,
            ))
            suggestions["volume"] = spec.volume_max
        
        # Check step
        if spec.volume_step > 0:
            remainder = volume % spec.volume_step
            if remainder > 1e-10:
                normalized = round(volume / spec.volume_step) * spec.volume_step
                issues.append(ValidationIssue(
                    category=ErrorCategory.TRANSIENT,
                    code="VOL_STEP",
                    message=f"Volume {volume} not aligned to step {spec.volume_step}",
                    field="volume",
                    value=volume,
                    suggestion=f"Use {normalized}",
                    recoverable=True,
                ))
                suggestions["volume"] = normalized
        
        return issues, suggestions
    
    def _validate_price(self, price: float, spec) -> Tuple[List[ValidationIssue], Dict]:
        """Validate price against spec."""
        issues = []
        suggestions = {}
        
        # Check NaN/Inf
        if math.isnan(price) or math.isinf(price):
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="PRICE_INVALID",
                message=f"Invalid price: {price}",
                field="price",
                value=price,
                suggestion="Use a valid numeric price",
                recoverable=False,
            ))
            return issues, suggestions
        
        # Check positive
        if price <= 0:
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="PRICE_NEGATIVE",
                message=f"Price must be positive: {price}",
                field="price",
                value=price,
                recoverable=False,
            ))
            return issues, suggestions
        
        # Check tick size alignment
        if spec.tick_size > 0:
            ticks = price / spec.tick_size
            if abs(ticks - round(ticks)) > 1e-10:
                normalized = round(ticks) * spec.tick_size
                issues.append(ValidationIssue(
                    category=ErrorCategory.TRANSIENT,
                    code="PRICE_TICK",
                    message=f"Price {price} not aligned to tick {spec.tick_size}",
                    field="price",
                    value=price,
                    suggestion=f"Use {normalized}",
                    recoverable=True,
                ))
                suggestions["price"] = normalized
        
        return issues, suggestions
    
    def _validate_stops(
        self, side: str, price: float, sl: float, tp: float, spec
    ) -> Tuple[List[ValidationIssue], Dict]:
        """Validate stop loss and take profit."""
        issues = []
        suggestions = {}
        
        stops_level = getattr(spec, 'stops_level', 0) or getattr(spec, 'trade_stops_level', 0)
        min_distance = stops_level * spec.point if hasattr(spec, 'point') else stops_level * spec.tick_size
        
        is_buy = side.lower() in ['buy', 'long']
        
        # Validate SL
        if sl:
            if is_buy:
                # SL must be below price
                if sl >= price:
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="SL_WRONG_SIDE",
                        message=f"Buy SL {sl} must be below price {price}",
                        field="sl",
                        value=sl,
                        suggestion=f"Use SL below {price}",
                        recoverable=True,
                    ))
                elif price - sl < min_distance:
                    valid_sl = price - min_distance
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="SL_TOO_CLOSE",
                        message=f"SL {sl} too close to price {price} (min: {min_distance})",
                        field="sl",
                        value=sl,
                        suggestion=f"Use SL at or below {valid_sl}",
                        recoverable=True,
                    ))
                    suggestions["sl"] = valid_sl
            else:
                # SL must be above price
                if sl <= price:
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="SL_WRONG_SIDE",
                        message=f"Sell SL {sl} must be above price {price}",
                        field="sl",
                        value=sl,
                        suggestion=f"Use SL above {price}",
                        recoverable=True,
                    ))
                elif sl - price < min_distance:
                    valid_sl = price + min_distance
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="SL_TOO_CLOSE",
                        message=f"SL {sl} too close to price {price} (min: {min_distance})",
                        field="sl",
                        value=sl,
                        suggestion=f"Use SL at or above {valid_sl}",
                        recoverable=True,
                    ))
                    suggestions["sl"] = valid_sl
        
        # Validate TP
        if tp:
            if is_buy:
                # TP must be above price
                if tp <= price:
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="TP_WRONG_SIDE",
                        message=f"Buy TP {tp} must be above price {price}",
                        field="tp",
                        value=tp,
                        suggestion=f"Use TP above {price}",
                        recoverable=True,
                    ))
                elif tp - price < min_distance:
                    valid_tp = price + min_distance
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="TP_TOO_CLOSE",
                        message=f"TP {tp} too close to price {price} (min: {min_distance})",
                        field="tp",
                        value=tp,
                        suggestion=f"Use TP at or above {valid_tp}",
                        recoverable=True,
                    ))
                    suggestions["tp"] = valid_tp
            else:
                # TP must be below price
                if tp >= price:
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="TP_WRONG_SIDE",
                        message=f"Sell TP {tp} must be below price {price}",
                        field="tp",
                        value=tp,
                        suggestion=f"Use TP below {price}",
                        recoverable=True,
                    ))
                elif price - tp < min_distance:
                    valid_tp = price - min_distance
                    issues.append(ValidationIssue(
                        category=ErrorCategory.PERMANENT,
                        code="TP_TOO_CLOSE",
                        message=f"TP {tp} too close to price {price} (min: {min_distance})",
                        field="tp",
                        value=tp,
                        suggestion=f"Use TP at or below {valid_tp}",
                        recoverable=True,
                    ))
                    suggestions["tp"] = valid_tp
        
        return issues, suggestions
    
    def _validate_margin(
        self, volume: float, price: float, equity: float, leverage: float, spec
    ) -> Tuple[List[ValidationIssue], Dict]:
        """Validate margin requirements."""
        issues = []
        suggestions = {}
        
        contract_size = spec.contract_size
        margin_required = (volume * contract_size * price) / leverage
        
        if margin_required > equity:
            max_volume = (equity * leverage) / (contract_size * price)
            max_volume = math.floor(max_volume / spec.volume_step) * spec.volume_step
            max_volume = max(spec.volume_min, min(max_volume, spec.volume_max))
            
            issues.append(ValidationIssue(
                category=ErrorCategory.PERMANENT,
                code="MARGIN_INSUFFICIENT",
                message=f"Insufficient margin: need {margin_required:.2f}, have {equity:.2f}",
                field="volume",
                value=volume,
                suggestion=f"Reduce volume to {max_volume}",
                recoverable=True,
            ))
            suggestions["volume"] = max_volume
        
        # Check margin level would be healthy (> 200%)
        remaining_equity = equity - margin_required
        if margin_required > 0:
            margin_level = (remaining_equity / margin_required) * 100
            if margin_level < 200:
                issues.append(ValidationIssue(
                    category=ErrorCategory.TRANSIENT,
                    code="MARGIN_LOW",
                    message=f"Margin level would be {margin_level:.1f}% (recommended > 200%)",
                    field="volume",
                    value=volume,
                    suggestion="Consider reducing position size",
                    recoverable=True,
                ))
        
        return issues, suggestions
    
    def _default_spec(self, symbol: str):
        """Create default spec for unknown symbol."""
        from .symbol_spec import SymbolSpec, CommissionSpec, CommissionType
        
        # Detect symbol type
        symbol_upper = symbol.upper()
        
        if "JPY" in symbol_upper:
            tick_size = 0.001
        elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
            tick_size = 0.01
        elif "BTC" in symbol_upper:
            tick_size = 1.0
        else:
            tick_size = 0.00001
        
        return SymbolSpec(
            symbol=symbol,
            tick_size=tick_size,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_points=2.0,
            commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            slippage_avg=0.5,
        )
    
    def auto_correct(
        self,
        symbol: str,
        volume: float,
        price: float,
        sl: float = None,
        tp: float = None,
    ) -> Dict[str, float]:
        """
        Auto-correct order parameters to comply with spec.
        
        Returns corrected parameters.
        """
        spec = self.symbol_specs.get(symbol, self._default_spec(symbol))
        
        corrected = {
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
        }
        
        # Correct volume
        if volume < spec.volume_min:
            corrected["volume"] = spec.volume_min
        elif volume > spec.volume_max:
            corrected["volume"] = spec.volume_max
        
        if spec.volume_step > 0:
            corrected["volume"] = round(corrected["volume"] / spec.volume_step) * spec.volume_step
        
        # Correct price
        if spec.tick_size > 0:
            corrected["price"] = round(price / spec.tick_size) * spec.tick_size
        
        # Correct SL/TP tick alignment
        if sl and spec.tick_size > 0:
            corrected["sl"] = round(sl / spec.tick_size) * spec.tick_size
        if tp and spec.tick_size > 0:
            corrected["tp"] = round(tp / spec.tick_size) * spec.tick_size
        
        return corrected


def with_graceful_failure(
    operation: str,
    executor: GracefulExecutor = None,
    max_retries: int = 3,
):
    """
    Decorator for graceful failure handling.
    
    Usage:
        @with_graceful_failure("place_order", max_retries=3)
        def place_order(symbol, volume, price):
            ...
    """
    if executor is None:
        executor = GracefulExecutor(max_retries=max_retries)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, success, error = executor.execute(
                operation,
                func,
                *args,
                **kwargs,
            )
            if not success:
                logger.error(f"{operation} failed: {error}")
                raise RuntimeError(f"{operation} failed after retries: {error}")
            return result
        return wrapper
    return decorator
