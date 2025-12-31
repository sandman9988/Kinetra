"""
Financial Audit Module - Defense in Depth
==========================================

Implements financial industry standards for numerical accuracy:
- IEEE 754 floating point validation
- Division by zero protection
- NaN/Inf detection and handling
- Overflow/underflow prevention
- Digit normalization (proper rounding)
- Audit trail logging
- Reconciliation checks

Reference standards:
- MiFID II (EU financial reporting)
- SEC Rule 17a-4 (audit trails)
- Basel III (risk calculations)
- IFRS 9 (financial instruments)
"""

import decimal
import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure audit logger
audit_logger = logging.getLogger("financial_audit")
audit_logger.setLevel(logging.INFO)


class AuditSeverity(Enum):
    """Audit issue severity levels."""
    CRITICAL = "critical"  # Trading must stop
    HIGH = "high"          # Manual review required
    MEDIUM = "medium"      # Logged for review
    LOW = "low"            # Informational


@dataclass
class AuditIssue:
    """Represents an audit finding."""
    timestamp: datetime
    severity: AuditSeverity
    code: str
    message: str
    value: Any = None
    expected: Any = None
    context: Dict = field(default_factory=dict)


class SafeMath:
    """
    Safe mathematical operations with financial precision.
    
    Implements defense-in-depth for numerical calculations:
    - Division by zero protection
    - NaN/Inf detection
    - Overflow/underflow prevention
    - Configurable precision
    """
    
    # Numerical limits
    MAX_PRICE = 1e12      # Maximum reasonable price (1 trillion)
    MIN_PRICE = 1e-12     # Minimum non-zero price
    MAX_VOLUME = 1e9      # Maximum reasonable volume
    MAX_PNL = 1e15        # Maximum reasonable P&L
    EPSILON = 1e-15       # Machine epsilon for comparisons
    
    @staticmethod
    def safe_divide(
        numerator: float, 
        denominator: float, 
        default: float = 0.0,
        allow_inf: bool = False
    ) -> float:
        """
        Safe division with zero and special value handling.
        
        Args:
            numerator: Dividend
            denominator: Divisor
            default: Value to return on division by zero
            allow_inf: Whether to allow infinite results
            
        Returns:
            Result of division or default value
        """
        # Handle NaN inputs
        if math.isnan(numerator) or math.isnan(denominator):
            warnings.warn(f"NaN in division: {numerator}/{denominator}")
            return default
        
        # Handle zero denominator
        if abs(denominator) < SafeMath.EPSILON:
            if abs(numerator) < SafeMath.EPSILON:
                return default  # 0/0 = default
            elif allow_inf:
                return math.copysign(math.inf, numerator * denominator)
            else:
                warnings.warn(f"Division by zero: {numerator}/{denominator}")
                return default
        
        result = numerator / denominator
        
        # Handle infinite results
        if math.isinf(result) and not allow_inf:
            warnings.warn(f"Infinite result: {numerator}/{denominator}")
            return math.copysign(SafeMath.MAX_PNL, result)
        
        return result
    
    @staticmethod
    def safe_multiply(a: float, b: float, max_result: float = None) -> float:
        """
        Safe multiplication with overflow protection.
        
        Args:
            a: First operand
            b: Second operand
            max_result: Maximum allowed result (default: MAX_PNL)
            
        Returns:
            Clamped result
        """
        if max_result is None:
            max_result = SafeMath.MAX_PNL
        
        # Handle NaN
        if math.isnan(a) or math.isnan(b):
            warnings.warn(f"NaN in multiplication: {a}*{b}")
            return 0.0
        
        # Handle infinity
        if math.isinf(a) or math.isinf(b):
            warnings.warn(f"Infinity in multiplication: {a}*{b}")
            return math.copysign(max_result, a * b) if a != 0 and b != 0 else 0.0
        
        result = a * b
        
        # Clamp result
        if abs(result) > max_result:
            warnings.warn(f"Multiplication overflow: {a}*{b}={result}, clamping to {max_result}")
            return math.copysign(max_result, result)
        
        return result
    
    @staticmethod
    def safe_sqrt(x: float, default: float = 0.0) -> float:
        """
        Safe square root with negative number handling.
        
        Args:
            x: Input value
            default: Value to return for negative inputs
            
        Returns:
            Square root or default
        """
        if math.isnan(x):
            return default
        if x < 0:
            warnings.warn(f"Square root of negative: {x}")
            return default
        return math.sqrt(x)
    
    @staticmethod
    def safe_log(x: float, default: float = float('-inf')) -> float:
        """
        Safe logarithm with non-positive handling.
        
        Args:
            x: Input value
            default: Value to return for non-positive inputs
            
        Returns:
            Natural log or default
        """
        if math.isnan(x) or x <= 0:
            return default
        return math.log(x)
    
    @staticmethod
    def validate_price(price: float, symbol: str = "") -> Tuple[bool, str]:
        """
        Validate a price value for financial use.
        
        Args:
            price: Price to validate
            symbol: Symbol name for error messages
            
        Returns:
            (is_valid, error_message)
        """
        if math.isnan(price):
            return False, f"Price is NaN for {symbol}"
        if math.isinf(price):
            return False, f"Price is infinite for {symbol}"
        if price < 0:
            return False, f"Negative price {price} for {symbol}"
        if price > SafeMath.MAX_PRICE:
            return False, f"Price {price} exceeds maximum for {symbol}"
        if 0 < price < SafeMath.MIN_PRICE:
            return False, f"Price {price} below minimum for {symbol}"
        return True, ""
    
    @staticmethod
    def validate_volume(volume: float, min_vol: float = 0.0, max_vol: float = None) -> Tuple[bool, str]:
        """
        Validate a volume/lot size value.
        
        Args:
            volume: Volume to validate
            min_vol: Minimum allowed volume
            max_vol: Maximum allowed volume
            
        Returns:
            (is_valid, error_message)
        """
        if max_vol is None:
            max_vol = SafeMath.MAX_VOLUME
        
        if math.isnan(volume):
            return False, "Volume is NaN"
        if math.isinf(volume):
            return False, "Volume is infinite"
        if volume < 0:
            return False, f"Negative volume: {volume}"
        if volume < min_vol:
            return False, f"Volume {volume} below minimum {min_vol}"
        if volume > max_vol:
            return False, f"Volume {volume} above maximum {max_vol}"
        return True, ""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range with NaN handling."""
        if math.isnan(value):
            return min_val
        return max(min_val, min(max_val, value))


class DigitNormalizer:
    """
    Price/volume normalization to broker specifications.
    
    Ensures all prices and volumes are properly rounded
    according to instrument specifications (tick size, volume step).
    """
    
    @staticmethod
    def normalize_price(price: float, tick_size: float, digits: int = None) -> float:
        """
        Normalize price to tick size.
        
        Uses Decimal for precise rounding (avoids floating point errors).
        
        Args:
            price: Raw price
            tick_size: Minimum price increment
            digits: Number of decimal places (auto-detected if None)
            
        Returns:
            Normalized price
        """
        if tick_size <= 0:
            warnings.warn(f"Invalid tick_size {tick_size}, returning price as-is")
            return price
        
        if math.isnan(price) or math.isinf(price):
            warnings.warn(f"Invalid price {price} for normalization")
            return price
        
        try:
            # Use Decimal for precise calculation
            d_price = Decimal(str(price))
            d_tick = Decimal(str(tick_size))
            
            # Round to nearest tick
            normalized = (d_price / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick
            
            # Apply digits if specified
            if digits is not None and digits >= 0:
                normalized = normalized.quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP)
            
            return float(normalized)
        except (InvalidOperation, ValueError) as e:
            warnings.warn(f"Price normalization error: {e}")
            return price
    
    @staticmethod
    def normalize_volume(volume: float, volume_step: float, volume_min: float = 0.0, volume_max: float = float('inf')) -> float:
        """
        Normalize volume to volume step.
        
        Args:
            volume: Raw volume
            volume_step: Minimum volume increment
            volume_min: Minimum allowed volume
            volume_max: Maximum allowed volume
            
        Returns:
            Normalized and clamped volume
        """
        if volume_step <= 0:
            warnings.warn(f"Invalid volume_step {volume_step}, using volume as-is")
            return max(volume_min, min(volume_max, volume))
        
        if math.isnan(volume) or math.isinf(volume):
            warnings.warn(f"Invalid volume {volume} for normalization")
            return volume_min
        
        try:
            # Use Decimal for precise calculation
            d_volume = Decimal(str(volume))
            d_step = Decimal(str(volume_step))
            
            # Round down to nearest step (never round up lots)
            normalized = (d_volume / d_step).to_integral_value(rounding=decimal.ROUND_DOWN) * d_step
            
            # Clamp to range
            result = float(normalized)
            result = max(volume_min, min(volume_max, result))
            
            return result
        except (InvalidOperation, ValueError) as e:
            warnings.warn(f"Volume normalization error: {e}")
            return max(volume_min, min(volume_max, volume))
    
    @staticmethod
    def calculate_pip_value(
        price: float,
        point: float,
        contract_size: float,
        volume: float
    ) -> float:
        """
        Calculate pip value with safe math.
        
        Pip value = (point / price) * contract_size * volume
        For JPY pairs where point=0.01, pip = 10 * point = 0.1
        
        Args:
            price: Current price
            point: Minimum price unit (e.g., 0.00001 for 5-digit)
            contract_size: Contract size (e.g., 100000 for forex)
            volume: Position size in lots
            
        Returns:
            Value of one pip movement in account currency
        """
        if price <= 0 or point <= 0:
            return 0.0
        
        pip_size = point * 10 if point < 0.01 else point  # Standard vs JPY pairs
        return SafeMath.safe_multiply(
            SafeMath.safe_divide(pip_size, price) * contract_size,
            volume
        )


class PnLCalculator:
    """
    Precise P&L calculation with full audit trail.
    
    Implements proper financial P&L calculation:
    - Gross P&L (before costs)
    - Net P&L (after costs)
    - Unrealized P&L (mark-to-market)
    - Cost breakdown (spread, commission, swap, slippage)
    """
    
    @staticmethod
    def calculate_gross_pnl(
        direction: int,  # 1=long, -1=short
        entry_price: float,
        exit_price: float,
        volume: float,
        contract_size: float,
        tick_size: float,
        tick_value: float
    ) -> Tuple[float, Dict]:
        """
        Calculate gross P&L (before costs).
        
        Args:
            direction: Trade direction (1=long, -1=short)
            entry_price: Entry price
            exit_price: Exit price
            volume: Position size in lots
            contract_size: Contract size per lot
            tick_size: Minimum price increment
            tick_value: Value per tick per lot
            
        Returns:
            (gross_pnl, calculation_details)
        """
        # Validate inputs
        for name, val in [("entry_price", entry_price), ("exit_price", exit_price)]:
            is_valid, msg = SafeMath.validate_price(val)
            if not is_valid:
                warnings.warn(f"Invalid {name}: {msg}")
                return 0.0, {"error": msg}
        
        is_valid, msg = SafeMath.validate_volume(volume)
        if not is_valid:
            warnings.warn(f"Invalid volume: {msg}")
            return 0.0, {"error": msg}
        
        # Calculate price difference
        price_diff = (exit_price - entry_price) * direction
        
        # Calculate ticks moved
        if tick_size > 0:
            ticks_moved = price_diff / tick_size
        else:
            ticks_moved = price_diff  # Fallback
            warnings.warn(f"Invalid tick_size {tick_size}, using price diff directly")
        
        # Calculate P&L
        gross_pnl = SafeMath.safe_multiply(
            ticks_moved * tick_value,
            volume
        )
        
        details = {
            "direction": "long" if direction > 0 else "short",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price_diff": price_diff,
            "ticks_moved": ticks_moved,
            "tick_value": tick_value,
            "volume": volume,
            "gross_pnl": gross_pnl,
        }
        
        return gross_pnl, details
    
    @staticmethod
    def calculate_net_pnl(
        gross_pnl: float,
        spread_cost: float,
        commission: float,
        swap_cost: float,
        slippage: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Calculate net P&L (after costs).
        
        Net P&L = Gross P&L - Spread - Commission - Swap - Slippage
        
        Args:
            gross_pnl: Gross P&L before costs
            spread_cost: Total spread cost (entry + exit)
            commission: Total commission (entry + exit)
            swap_cost: Total swap/rollover cost
            slippage: Total slippage cost
            
        Returns:
            (net_pnl, cost_breakdown)
        """
        # Validate all costs are non-negative (except swap which can be positive)
        if spread_cost < 0:
            warnings.warn(f"Negative spread_cost {spread_cost}, using 0")
            spread_cost = 0.0
        if commission < 0:
            warnings.warn(f"Negative commission {commission}, using 0")
            commission = 0.0
        if slippage < 0:
            warnings.warn(f"Negative slippage {slippage}, using 0")
            slippage = 0.0
        
        total_costs = spread_cost + commission + abs(swap_cost) + slippage
        net_pnl = gross_pnl - total_costs
        
        breakdown = {
            "gross_pnl": gross_pnl,
            "spread_cost": spread_cost,
            "commission": commission,
            "swap_cost": swap_cost,
            "slippage": slippage,
            "total_costs": total_costs,
            "net_pnl": net_pnl,
            "cost_percentage": SafeMath.safe_divide(total_costs, abs(gross_pnl)) * 100 if gross_pnl != 0 else 0,
        }
        
        return net_pnl, breakdown
    
    @staticmethod
    def calculate_mtm(
        direction: int,
        entry_price: float,
        current_price: float,
        volume: float,
        tick_size: float,
        tick_value: float
    ) -> float:
        """
        Calculate mark-to-market (unrealized P&L).
        
        Args:
            direction: Trade direction (1=long, -1=short)
            entry_price: Entry price
            current_price: Current market price
            volume: Position size in lots
            tick_size: Minimum price increment
            tick_value: Value per tick per lot
            
        Returns:
            Unrealized P&L
        """
        gross_pnl, _ = PnLCalculator.calculate_gross_pnl(
            direction, entry_price, current_price, volume,
            0, tick_size, tick_value  # contract_size unused in this calculation
        )
        return gross_pnl


class RiskMetricsCalculator:
    """
    Calculate risk metrics with proper statistical methods.
    
    Implements standard risk metrics:
    - Sharpe Ratio (risk-adjusted return)
    - Sortino Ratio (downside risk-adjusted)
    - Maximum Drawdown
    - VaR/CVaR (Value at Risk)
    - Calmar Ratio
    """
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = np.sqrt(252)
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Sharpe = (E[R] - Rf) / std(R) * sqrt(periods_per_year)
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Risk-free rate (same period as returns)
            annualization_factor: sqrt(periods_per_year)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Remove NaN/Inf
        clean_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(clean_returns) < 2:
            return 0.0
        
        excess_returns = clean_returns - risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(clean_returns, ddof=1)
        
        if std_returns < SafeMath.EPSILON:
            return 0.0 if mean_excess <= 0 else float('inf')
        
        return SafeMath.safe_divide(mean_excess, std_returns) * annualization_factor
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        target_return: float = 0.0,
        annualization_factor: float = np.sqrt(252)
    ) -> float:
        """
        Calculate annualized Sortino ratio.
        
        Sortino = (E[R] - target) / downside_std * sqrt(periods_per_year)
        
        Args:
            returns: Array of periodic returns
            target_return: Target/minimum acceptable return
            annualization_factor: sqrt(periods_per_year)
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Remove NaN/Inf
        clean_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(clean_returns) < 2:
            return 0.0
        
        excess_returns = clean_returns - target_return
        mean_excess = np.mean(excess_returns)
        
        # Downside deviation (only negative returns)
        downside = clean_returns[clean_returns < target_return] - target_return
        if len(downside) == 0:
            return float('inf') if mean_excess > 0 else 0.0
        
        downside_std = np.std(downside, ddof=1)
        if downside_std < SafeMath.EPSILON:
            return float('inf') if mean_excess > 0 else 0.0
        
        return SafeMath.safe_divide(mean_excess, downside_std) * annualization_factor
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            (max_dd_absolute, max_dd_percentage, peak_index, trough_index)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0, 0
        
        # Remove NaN/Inf
        clean_equity = np.array(equity_curve, dtype=float)
        clean_equity = clean_equity[~np.isnan(clean_equity) & ~np.isinf(clean_equity)]
        if len(clean_equity) < 2:
            return 0.0, 0.0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(clean_equity)
        drawdown = running_max - clean_equity
        drawdown_pct = SafeMath.safe_divide(drawdown, running_max, default=0.0)
        
        # Find max drawdown
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx]
        max_dd_pct = drawdown_pct[max_dd_idx]
        
        # Find peak (before trough)
        peak_idx = np.argmax(clean_equity[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        return float(max_dd), float(max_dd_pct), int(peak_idx), int(max_dd_idx)
    
    @staticmethod
    def calculate_var_cvar(
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Array of periodic returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            (VaR, CVaR) - both as positive values representing potential loss
        """
        if len(returns) < 10:
            return 0.0, 0.0
        
        # Remove NaN/Inf
        clean_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(clean_returns) < 10:
            return 0.0, 0.0
        
        # VaR: percentile of losses
        alpha = 1 - confidence_level
        var = -np.percentile(clean_returns, alpha * 100)
        
        # CVaR: expected loss beyond VaR
        tail_returns = clean_returns[clean_returns <= -var]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        return max(0, var), max(0, cvar)


class AuditTrail:
    """
    Comprehensive audit trail for financial compliance.
    
    Implements:
    - Transaction logging with timestamps
    - Checksum verification
    - Reconciliation checks
    - Export for regulatory reporting
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize audit trail.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.entries: List[Dict] = []
        self.issues: List[AuditIssue] = []
        self.checksums: List[str] = []
    
    def log_entry(self, entry_type: str, data: Dict, metadata: Dict = None) -> str:
        """
        Log an audit entry with checksum.
        
        Args:
            entry_type: Type of entry (trade, balance, calculation, etc.)
            data: Entry data
            metadata: Additional metadata
            
        Returns:
            Entry checksum
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "entry_type": entry_type,
            "sequence": len(self.entries),
            "data": data,
            "metadata": metadata or {},
        }
        
        # Generate checksum
        entry_json = json.dumps(entry, sort_keys=True, default=str)
        checksum = hashlib.sha256(entry_json.encode()).hexdigest()[:16]
        entry["checksum"] = checksum
        
        self.entries.append(entry)
        self.checksums.append(checksum)
        
        # Log to audit logger
        audit_logger.info(f"[{entry_type}] {checksum}: {data}")
        
        return checksum
    
    def log_issue(self, issue: AuditIssue):
        """Log an audit issue."""
        self.issues.append(issue)
        
        # Log based on severity
        if issue.severity == AuditSeverity.CRITICAL:
            audit_logger.critical(f"[{issue.code}] {issue.message}")
        elif issue.severity == AuditSeverity.HIGH:
            audit_logger.error(f"[{issue.code}] {issue.message}")
        elif issue.severity == AuditSeverity.MEDIUM:
            audit_logger.warning(f"[{issue.code}] {issue.message}")
        else:
            audit_logger.info(f"[{issue.code}] {issue.message}")
    
    def verify_chain(self) -> bool:
        """
        Verify audit trail integrity.
        
        Returns:
            True if chain is valid
        """
        for i, entry in enumerate(self.entries):
            # Recalculate checksum
            entry_copy = entry.copy()
            original_checksum = entry_copy.pop("checksum")
            entry_json = json.dumps(entry_copy, sort_keys=True, default=str)
            calculated_checksum = hashlib.sha256(entry_json.encode()).hexdigest()[:16]
            
            if calculated_checksum != original_checksum:
                self.log_issue(AuditIssue(
                    timestamp=datetime.now(),
                    severity=AuditSeverity.CRITICAL,
                    code="AUDIT001",
                    message=f"Checksum mismatch at entry {i}",
                    expected=original_checksum,
                    value=calculated_checksum,
                ))
                return False
        
        return True
    
    def reconcile_equity(
        self,
        initial_capital: float,
        trades: List[Dict],
        final_equity: float,
        tolerance: float = 0.01  # $0.01 tolerance
    ) -> Tuple[bool, float]:
        """
        Reconcile final equity with trade P&L.
        
        Args:
            initial_capital: Starting capital
            trades: List of trade dicts with 'net_pnl' key
            final_equity: Reported final equity
            tolerance: Acceptable difference
            
        Returns:
            (is_reconciled, difference)
        """
        total_pnl = sum(t.get('net_pnl', 0) for t in trades)
        calculated_equity = initial_capital + total_pnl
        difference = abs(final_equity - calculated_equity)
        
        is_reconciled = difference <= tolerance
        
        if not is_reconciled:
            self.log_issue(AuditIssue(
                timestamp=datetime.now(),
                severity=AuditSeverity.HIGH,
                code="RECON001",
                message=f"Equity reconciliation failed: {difference:.4f} difference",
                expected=calculated_equity,
                value=final_equity,
                context={"initial_capital": initial_capital, "total_pnl": total_pnl},
            ))
        
        return is_reconciled, difference
    
    def export_report(self) -> Dict:
        """
        Export audit report for regulatory compliance.
        
        Returns:
            Complete audit report dictionary
        """
        return {
            "session_id": self.session_id,
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "total_issues": len(self.issues),
            "issues_by_severity": {
                "critical": len([i for i in self.issues if i.severity == AuditSeverity.CRITICAL]),
                "high": len([i for i in self.issues if i.severity == AuditSeverity.HIGH]),
                "medium": len([i for i in self.issues if i.severity == AuditSeverity.MEDIUM]),
                "low": len([i for i in self.issues if i.severity == AuditSeverity.LOW]),
            },
            "chain_valid": self.verify_chain(),
            "entries": self.entries,
            "issues": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


# Export all components
__all__ = [
    "SafeMath",
    "DigitNormalizer",
    "PnLCalculator",
    "RiskMetricsCalculator",
    "AuditTrail",
    "AuditIssue",
    "AuditSeverity",
]
