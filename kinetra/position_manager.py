"""
Position Manager - Full Lifecycle Tracking
==========================================

Comprehensive position management with:
- Full position lifecycle tracking (pending → open → closing → closed)
- Multi-position support (hedging, pyramiding)
- Real-time P&L tracking
- Margin requirement monitoring
- Position reconciliation
- Audit trail logging
- Event-driven callbacks
"""

import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .financial_audit import SafeMath, AuditTrail, AuditIssue, AuditSeverity

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """Position lifecycle states."""
    PENDING = auto()      # Order submitted, not yet filled
    OPEN = auto()         # Position is open
    PARTIALLY_CLOSED = auto()  # Part of position closed
    CLOSING = auto()      # Close order submitted
    CLOSED = auto()       # Position fully closed
    CANCELLED = auto()    # Order cancelled before fill
    REJECTED = auto()     # Order rejected by broker
    ERROR = auto()        # Position in error state


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionEvent:
    """Event in position lifecycle."""
    timestamp: datetime
    event_type: str
    state_before: PositionState
    state_after: PositionState
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Position:
    """
    Complete position record with full tracking.
    
    Tracks entire lifecycle from order to close with all associated
    costs, P&L, and audit information.
    """
    # Identity
    position_id: str
    symbol: str
    side: PositionSide
    
    # Size and price
    volume: float  # Current open volume (may decrease with partial closes)
    entry_price: float
    entry_time: datetime
    
    # Current state
    state: PositionState = PositionState.PENDING
    
    # Exit information (set when closed)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    # Cost tracking
    spread_cost: float = 0.0
    commission: float = 0.0
    swap_cost: float = 0.0
    slippage: float = 0.0
    
    # P&L tracking (updated in real-time)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    
    # Risk tracking
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    current_stop: Optional[float] = None  # Dynamic stop (trailing)
    
    # Excursion tracking
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L
    highest_price: float = 0.0            # Highest price during position
    lowest_price: float = 0.0             # Lowest price during position
    
    # Margin
    margin_required: float = 0.0
    margin_level: float = float('inf')
    
    # Metadata
    magic_number: int = 0
    comment: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Strategy attribution
    strategy_id: Optional[str] = None
    signal_energy: float = 0.0
    signal_regime: str = ""
    
    # Lifecycle events
    events: List[PositionEvent] = field(default_factory=list)
    
    # Partial fill tracking
    fill_history: List[Dict] = field(default_factory=list)
    initial_volume: float = 0.0  # Volume at open (before partial closes)
    
    # Timing
    last_update: datetime = field(default_factory=datetime.now)
    bars_held: int = 0
    
    def __post_init__(self):
        """Initialize tracking values."""
        if self.initial_volume == 0:
            self.initial_volume = self.volume
        if self.highest_price == 0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0:
            self.lowest_price = self.entry_price
    
    @property
    def is_open(self) -> bool:
        """Check if position is currently open."""
        return self.state in [PositionState.OPEN, PositionState.PARTIALLY_CLOSED]
    
    @property
    def is_pending(self) -> bool:
        """Check if position is pending fill."""
        return self.state == PositionState.PENDING
    
    @property
    def is_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.state == PositionState.CLOSED
    
    @property
    def total_cost(self) -> float:
        """Total transaction costs."""
        return self.spread_cost + self.commission + abs(self.swap_cost) + self.slippage
    
    @property
    def holding_time(self) -> Optional[timedelta]:
        """Time position has been/was held."""
        if self.exit_time:
            return self.exit_time - self.entry_time
        elif self.is_open:
            return datetime.now() - self.entry_time
        return None
    
    @property
    def mfe_efficiency(self) -> float:
        """How much of MFE was captured (0-1)."""
        if self.max_favorable_excursion > 0:
            return max(0, min(1, self.realized_pnl / self.max_favorable_excursion))
        return 0.0
    
    def record_event(self, event_type: str, state_after: PositionState, 
                     details: Dict = None, error: str = None):
        """Record a lifecycle event."""
        event = PositionEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            state_before=self.state,
            state_after=state_after,
            details=details or {},
            error=error,
        )
        self.events.append(event)
        self.state = state_after
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "volume": self.volume,
            "initial_volume": self.initial_volume,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "state": self.state.name,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "net_pnl": self.net_pnl,
            "total_cost": self.total_cost,
            "mfe": self.max_favorable_excursion,
            "mae": self.max_adverse_excursion,
            "bars_held": self.bars_held,
            "events_count": len(self.events),
        }


class PositionManager:
    """
    Centralized position management with full lifecycle tracking.
    
    Features:
    - Thread-safe position operations
    - Real-time P&L updates
    - Margin monitoring
    - Event callbacks
    - Audit trail
    - Position reconciliation
    - Graceful error handling
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        leverage: float = 100.0,
        max_positions: int = 10,
        max_exposure_pct: float = 0.5,  # Max 50% of capital exposed
        enable_hedging: bool = False,
        enable_audit: bool = True,
    ):
        """
        Initialize position manager.
        
        Args:
            initial_capital: Starting capital
            leverage: Account leverage
            max_positions: Maximum concurrent positions
            max_exposure_pct: Maximum exposure as % of capital
            enable_hedging: Allow opposite positions on same symbol
            enable_audit: Enable audit trail logging
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.max_positions = max_positions
        self.max_exposure_pct = max_exposure_pct
        self.enable_hedging = enable_hedging
        
        # Position storage
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._position_lock = threading.RLock()
        
        # Tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Margin tracking
        self.margin_used = 0.0
        self.free_margin = initial_capital
        self.margin_level = float('inf')
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_open': [],
            'on_close': [],
            'on_update': [],
            'on_margin_call': [],
            'on_stop_hit': [],
            'on_error': [],
        }
        
        # Audit
        self.audit = AuditTrail() if enable_audit else None
        
        # Statistics
        self._equity_history: List[float] = [initial_capital]
        self._margin_history: List[float] = []
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for position events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _fire_callbacks(self, event: str, *args, **kwargs):
        """Fire all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        volume: float,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float = None,
        take_profit: float = None,
        magic_number: int = 0,
        comment: str = "",
        strategy_id: str = None,
        signal_energy: float = 0.0,
        signal_regime: str = "",
        spec: Any = None,  # SymbolSpec
    ) -> Tuple[Optional[Position], str]:
        """
        Open a new position with full validation.
        
        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            volume: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            stop_loss: Stop loss price
            take_profit: Take profit price
            magic_number: EA identifier
            comment: Position comment
            strategy_id: Strategy identifier
            signal_energy: Physics signal energy at entry
            signal_regime: Physics regime at entry
            spec: SymbolSpec for validation
            
        Returns:
            (Position or None, error_message)
        """
        with self._position_lock:
            # Validate capacity
            if len(self._positions) >= self.max_positions:
                return None, f"Max positions ({self.max_positions}) reached"
            
            # Check hedging
            if not self.enable_hedging:
                existing = self.get_position_by_symbol(symbol)
                if existing and existing.side != side:
                    return None, f"Hedging disabled - existing {existing.side.value} position on {symbol}"
            
            # Validate volume against spec
            if spec:
                if volume < spec.volume_min:
                    return None, f"Volume {volume} below minimum {spec.volume_min}"
                if volume > spec.volume_max:
                    return None, f"Volume {volume} above maximum {spec.volume_max}"
            
            # Calculate margin requirement
            contract_size = spec.contract_size if spec else 100000
            margin_required = (volume * contract_size * entry_price) / self.leverage
            
            # Check margin
            if margin_required > self.free_margin:
                return None, f"Insufficient margin: required {margin_required:.2f}, available {self.free_margin:.2f}"
            
            # Check exposure limit
            total_exposure = self.margin_used + margin_required
            if total_exposure / self.capital > self.max_exposure_pct:
                return None, f"Exposure limit exceeded: {total_exposure / self.capital:.1%} > {self.max_exposure_pct:.1%}"
            
            # Create position
            position_id = str(uuid.uuid4())[:8]
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=entry_price,
                entry_time=entry_time,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_stop=stop_loss,
                magic_number=magic_number,
                comment=comment,
                strategy_id=strategy_id,
                signal_energy=signal_energy,
                signal_regime=signal_regime,
                margin_required=margin_required,
                state=PositionState.OPEN,
            )
            
            # Calculate costs
            if spec:
                position.spread_cost = spec.spread_cost(volume, entry_price)
                position.commission = spec.commission.calculate_commission(
                    volume, volume * contract_size * entry_price
                ) / 2  # Half on entry
            
            # Record event
            position.record_event(
                "OPEN",
                PositionState.OPEN,
                {
                    "price": entry_price,
                    "volume": volume,
                    "margin": margin_required,
                    "sl": stop_loss,
                    "tp": take_profit,
                }
            )
            
            # Update manager state
            self._positions[position_id] = position
            self.margin_used += margin_required
            self.free_margin = self.capital - self.margin_used
            self._update_margin_level()
            self.total_trades += 1
            
            # Audit
            if self.audit:
                self.audit.log_entry("position_open", position.to_dict())
            
            # Callbacks
            self._fire_callbacks('on_open', position)
            
            logger.info(f"Opened {side.value} position {position_id} on {symbol}: "
                       f"{volume} lots @ {entry_price}")
            
            return position, ""
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        volume: float = None,  # None = close all
        reason: str = "manual",
        spec: Any = None,
    ) -> Tuple[bool, str]:
        """
        Close a position (full or partial).
        
        Args:
            position_id: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            volume: Volume to close (None = all)
            reason: Close reason
            spec: SymbolSpec for cost calculation
            
        Returns:
            (success, error_message)
        """
        with self._position_lock:
            position = self._positions.get(position_id)
            if not position:
                return False, f"Position {position_id} not found"
            
            if not position.is_open:
                return False, f"Position {position_id} is not open (state: {position.state.name})"
            
            # Determine close volume
            close_volume = volume if volume else position.volume
            if close_volume > position.volume:
                return False, f"Close volume {close_volume} exceeds position volume {position.volume}"
            
            is_partial = close_volume < position.volume
            
            # Calculate P&L
            if position.side == PositionSide.LONG:
                price_diff = exit_price - position.entry_price
            else:
                price_diff = position.entry_price - exit_price
            
            contract_size = spec.contract_size if spec else 100000
            tick_size = spec.tick_size if spec else 0.00001
            tick_value = spec.tick_value if spec else 1.0
            
            # Gross P&L for closed portion
            ticks_moved = SafeMath.safe_divide(price_diff, tick_size, 0.0)
            gross_pnl = ticks_moved * tick_value * close_volume
            
            # Exit costs
            if spec:
                exit_spread = spec.spread_cost(close_volume, exit_price)
                exit_commission = spec.commission.calculate_commission(
                    close_volume, close_volume * contract_size * exit_price
                ) / 2
                position.spread_cost += exit_spread
                position.commission += exit_commission
            
            # Update position
            if is_partial:
                # Partial close
                position.volume -= close_volume
                position.realized_pnl += gross_pnl - position.total_cost * (close_volume / position.initial_volume)
                position.record_event(
                    "PARTIAL_CLOSE",
                    PositionState.PARTIALLY_CLOSED,
                    {
                        "close_volume": close_volume,
                        "remaining_volume": position.volume,
                        "exit_price": exit_price,
                        "partial_pnl": gross_pnl,
                        "reason": reason,
                    }
                )
                
                # Record fill
                position.fill_history.append({
                    "time": exit_time,
                    "type": "partial_close",
                    "volume": close_volume,
                    "price": exit_price,
                    "pnl": gross_pnl,
                })
            else:
                # Full close
                position.exit_price = exit_price
                position.exit_time = exit_time
                position.gross_pnl = gross_pnl
                position.net_pnl = gross_pnl - position.total_cost
                position.realized_pnl = position.net_pnl
                position.unrealized_pnl = 0.0
                
                position.record_event(
                    "CLOSE",
                    PositionState.CLOSED,
                    {
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": position.net_pnl,
                        "total_cost": position.total_cost,
                        "reason": reason,
                    }
                )
                
                # Update statistics
                if position.net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.realized_pnl += position.net_pnl
                self.capital += position.net_pnl
                
                # Release margin
                self.margin_used -= position.margin_required
                self.free_margin = self.capital - self.margin_used
                
                # Move to closed
                del self._positions[position_id]
                self._closed_positions.append(position)
            
            self._update_margin_level()
            self._equity_history.append(self.capital + self.unrealized_pnl)
            
            # Audit
            if self.audit:
                self.audit.log_entry("position_close", {
                    "position_id": position_id,
                    "close_volume": close_volume,
                    "exit_price": exit_price,
                    "pnl": gross_pnl,
                    "reason": reason,
                    "is_partial": is_partial,
                })
            
            # Callbacks
            self._fire_callbacks('on_close', position, reason)
            
            logger.info(f"{'Partially closed' if is_partial else 'Closed'} position {position_id}: "
                       f"{close_volume} lots @ {exit_price}, P&L: {gross_pnl:.2f}")
            
            return True, ""
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        current_time: datetime,
        spec: Any = None,
    ) -> Tuple[bool, str]:
        """
        Update position with current market price.
        
        Args:
            position_id: Position to update
            current_price: Current market price
            current_time: Current timestamp
            spec: SymbolSpec
            
        Returns:
            (success, error_message)
        """
        with self._position_lock:
            position = self._positions.get(position_id)
            if not position:
                return False, f"Position {position_id} not found"
            
            if not position.is_open:
                return False, f"Position {position_id} is not open"
            
            # Update price extremes
            position.highest_price = max(position.highest_price, current_price)
            position.lowest_price = min(position.lowest_price, current_price)
            
            # Calculate unrealized P&L
            if position.side == PositionSide.LONG:
                price_diff = current_price - position.entry_price
            else:
                price_diff = position.entry_price - current_price
            
            tick_size = spec.tick_size if spec else 0.00001
            tick_value = spec.tick_value if spec else 1.0
            
            ticks_moved = SafeMath.safe_divide(price_diff, tick_size, 0.0)
            position.unrealized_pnl = ticks_moved * tick_value * position.volume
            
            # Update MFE/MAE
            if position.unrealized_pnl > position.max_favorable_excursion:
                position.max_favorable_excursion = position.unrealized_pnl
            if position.unrealized_pnl < -position.max_adverse_excursion:
                position.max_adverse_excursion = abs(position.unrealized_pnl)
            
            # Update trailing stop
            if position.trailing_stop_distance:
                if position.side == PositionSide.LONG:
                    new_stop = current_price - position.trailing_stop_distance
                    if position.current_stop is None or new_stop > position.current_stop:
                        position.current_stop = new_stop
                else:
                    new_stop = current_price + position.trailing_stop_distance
                    if position.current_stop is None or new_stop < position.current_stop:
                        position.current_stop = new_stop
            
            # Check stop loss
            if position.current_stop:
                if position.side == PositionSide.LONG and current_price <= position.current_stop:
                    self._fire_callbacks('on_stop_hit', position, 'stop_loss', current_price)
                    return self.close_position(position_id, current_price, current_time, 
                                               reason="stop_loss", spec=spec)
                elif position.side == PositionSide.SHORT and current_price >= position.current_stop:
                    self._fire_callbacks('on_stop_hit', position, 'stop_loss', current_price)
                    return self.close_position(position_id, current_price, current_time,
                                               reason="stop_loss", spec=spec)
            
            # Check take profit
            if position.take_profit:
                if position.side == PositionSide.LONG and current_price >= position.take_profit:
                    self._fire_callbacks('on_stop_hit', position, 'take_profit', current_price)
                    return self.close_position(position_id, current_price, current_time,
                                               reason="take_profit", spec=spec)
                elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                    self._fire_callbacks('on_stop_hit', position, 'take_profit', current_price)
                    return self.close_position(position_id, current_price, current_time,
                                               reason="take_profit", spec=spec)
            
            position.last_update = current_time
            position.bars_held += 1
            
            # Update margin level
            position.margin_required = (position.volume * (spec.contract_size if spec else 100000) * 
                                       current_price) / self.leverage
            self._update_margin_level()
            
            # Check margin call (< 100%)
            if self.margin_level < 100:
                self._fire_callbacks('on_margin_call', position, self.margin_level)
            
            # Callbacks
            self._fire_callbacks('on_update', position)
            
            return True, ""
    
    def update_all_positions(self, prices: Dict[str, float], current_time: datetime, 
                            specs: Dict[str, Any] = None):
        """Update all open positions with current prices."""
        specs = specs or {}
        self.unrealized_pnl = 0.0
        
        for position_id, position in list(self._positions.items()):
            if position.symbol in prices:
                self.update_position(
                    position_id, 
                    prices[position.symbol], 
                    current_time,
                    specs.get(position.symbol)
                )
                self.unrealized_pnl += position.unrealized_pnl
        
        self._equity_history.append(self.capital + self.unrealized_pnl)
        self._margin_history.append(self.margin_level)
    
    def _update_margin_level(self):
        """Update margin level calculation."""
        equity = self.capital + self.unrealized_pnl
        if self.margin_used > 0:
            self.margin_level = (equity / self.margin_used) * 100
        else:
            self.margin_level = float('inf')
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self._positions.get(position_id)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position by symbol (first match)."""
        for position in self._positions.values():
            if position.symbol == symbol and position.is_open:
                return position
        return None
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.is_open]
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get positions by strategy ID."""
        return [p for p in self._positions.values() if p.strategy_id == strategy_id]
    
    @property
    def equity(self) -> float:
        """Current equity (capital + unrealized P&L)."""
        return self.capital + self.unrealized_pnl
    
    @property
    def win_rate(self) -> float:
        """Win rate of closed trades."""
        total = self.winning_trades + self.losing_trades
        return SafeMath.safe_divide(self.winning_trades, total, 0.0)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "margin_level": self.margin_level,
            "open_positions": len(self._positions),
            "closed_positions": len(self._closed_positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
        }
    
    def reconcile(self) -> Tuple[bool, List[str]]:
        """
        Reconcile positions and verify integrity.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check margin calculation
        calculated_margin = sum(p.margin_required for p in self._positions.values())
        if abs(calculated_margin - self.margin_used) > 0.01:
            issues.append(f"Margin mismatch: tracked {self.margin_used:.2f}, calculated {calculated_margin:.2f}")
        
        # Check P&L reconciliation
        calculated_realized = sum(p.net_pnl for p in self._closed_positions)
        if abs(calculated_realized - self.realized_pnl) > 0.01:
            issues.append(f"Realized P&L mismatch: tracked {self.realized_pnl:.2f}, calculated {calculated_realized:.2f}")
        
        # Check position states
        for pos_id, position in self._positions.items():
            if not position.is_open:
                issues.append(f"Position {pos_id} in positions dict but not open (state: {position.state.name})")
            if position.volume <= 0:
                issues.append(f"Position {pos_id} has non-positive volume: {position.volume}")
        
        if issues and self.audit:
            for issue in issues:
                self.audit.log_issue(AuditIssue(
                    timestamp=datetime.now(),
                    severity=AuditSeverity.HIGH,
                    code="RECON",
                    message=issue,
                ))
        
        return len(issues) == 0, issues
    
    def reset(self):
        """Reset manager to initial state."""
        with self._position_lock:
            self._positions.clear()
            self._closed_positions.clear()
            self.capital = self.initial_capital
            self.margin_used = 0.0
            self.free_margin = self.initial_capital
            self.margin_level = float('inf')
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self._equity_history = [self.initial_capital]
            self._margin_history = []
