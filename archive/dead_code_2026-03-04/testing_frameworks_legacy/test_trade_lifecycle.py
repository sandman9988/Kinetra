#!/usr/bin/env python3
"""
Trade Lifecycle - Comprehensive & Correct Sequence of Events
=============================================================

This demonstrates the complete trade lifecycle with all events in correct order:

1. PRE-TRADE VALIDATION
   - Symbol validation
   - Volume normalization & limits check
   - Margin requirement check
   - Free margin check
   - Stops level validation

2. ORDER SUBMISSION
   - Create order request
   - Validate order parameters
   - Queue for execution

3. ORDER EXECUTION
   - Apply spread (buy at ask, sell at bid)
   - Apply slippage (if market order)
   - Deduct spread cost
   - Deduct entry commission
   - Reserve margin

4. POSITION OPEN
   - Create position record
   - Set SL/TP if specified
   - Start position monitoring

5. POSITION MONITORING (per tick/bar)
   - Update unrealized P&L
   - Update equity
   - Check margin level
   - Check SL/TP hit
   - Track MFE/MAE

6. OVERNIGHT PROCESSING (if held overnight)
   - Apply swap charge/credit
   - Triple swap on rollover day

7. POSITION CLOSE
   - Execute at current price
   - Apply exit spread (if applicable)
   - Apply exit commission
   - Calculate final P&L
   - Release margin

8. POST-TRADE
   - Update balance
   - Record trade history
   - Update statistics
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import IntEnum
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.symbol_info import get_symbol_info, SymbolInfo
from kinetra.mql5_trade_classes import (
    CAccountInfo, CSymbolInfo, CPositionInfo, CTrade,
    ENUM_ORDER_TYPE, ENUM_POSITION_TYPE, ENUM_TRADE_REQUEST_ACTIONS,
    ENUM_SYMBOL_SWAP_MODE, ENUM_DAY_OF_WEEK
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TradeEvent(IntEnum):
    """Trade lifecycle events."""
    PRE_VALIDATION = 1
    ORDER_SUBMITTED = 2
    ORDER_EXECUTED = 3
    POSITION_OPENED = 4
    TICK_UPDATE = 5
    SWAP_CHARGED = 6
    SL_HIT = 7
    TP_HIT = 8
    MANUAL_CLOSE = 9
    POSITION_CLOSED = 10
    TRADE_COMPLETE = 11


class CloseReason(IntEnum):
    """Position close reason."""
    MANUAL = 0
    STOP_LOSS = 1
    TAKE_PROFIT = 2
    MARGIN_CALL = 3
    END_OF_TEST = 4


# =============================================================================
# TRADE LIFECYCLE MANAGER
# =============================================================================

@dataclass
class TradeRequest:
    """Trade request (mirrors MQL5 MqlTradeRequest)."""
    action: ENUM_TRADE_REQUEST_ACTIONS
    symbol: str
    volume: float
    type: ENUM_ORDER_TYPE
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 10  # Max slippage in points
    magic: int = 0
    comment: str = ""


@dataclass
class TradeResult:
    """Trade result (mirrors MQL5 MqlTradeResult)."""
    retcode: int = 0
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    comment: str = ""


@dataclass
class Position:
    """Open position with full tracking."""
    ticket: int
    symbol: str
    type: ENUM_POSITION_TYPE
    volume: float
    open_price: float
    open_time: datetime
    sl: float = 0.0
    tp: float = 0.0
    
    # Costs at entry
    spread_cost: float = 0.0
    entry_commission: float = 0.0
    slippage_cost: float = 0.0
    
    # Running totals
    swap_total: float = 0.0
    swap_days: int = 0
    
    # Unrealized
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    # MFE/MAE tracking
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion
    
    # Close data
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    close_reason: Optional[CloseReason] = None
    exit_commission: float = 0.0
    
    # Final P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    
    # Event log
    events: List[Dict] = field(default_factory=list)
    
    def log_event(self, event: TradeEvent, details: str = "", **kwargs):
        """Log a trade lifecycle event."""
        self.events.append({
            'time': datetime.now(),
            'event': event.name,
            'details': details,
            **kwargs
        })


class TradeLifecycleManager:
    """
    Complete trade lifecycle management.
    
    Handles all events from order submission to trade completion.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: int = 100,
        commission_per_lot: float = 7.0,
    ):
        # Account
        self.account = CAccountInfo()
        self.account.SetBalance(initial_balance)
        self.account.SetLeverage(leverage)
        
        self.commission_per_lot = commission_per_lot
        
        # State
        self.positions: Dict[int, Position] = {}
        self.closed_positions: List[Position] = []
        self.ticket_counter = 0
        self.current_time = datetime.now()
        
        # Statistics
        self.total_spread_cost = 0.0
        self.total_commission = 0.0
        self.total_swap = 0.0
        self.total_slippage = 0.0
    
    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol specifications."""
        return get_symbol_info(symbol)
    
    def _log(self, msg: str):
        """Log message with timestamp."""
        logger.info(f"[{self.current_time}] {msg}")
    
    # =========================================================================
    # PHASE 1: PRE-TRADE VALIDATION
    # =========================================================================
    
    def validate_trade(self, request: TradeRequest) -> tuple[bool, str]:
        """
        Validate trade request before submission.
        
        Checks:
        1. Symbol exists and is tradeable
        2. Volume is within limits and properly normalized
        3. Sufficient margin available
        4. SL/TP distances are valid
        """
        self._log(f"━━━ PHASE 1: PRE-TRADE VALIDATION ━━━")
        
        # 1. Symbol validation
        try:
            info = self._get_symbol_info(request.symbol)
            self._log(f"  ✓ Symbol {request.symbol} validated")
            self._log(f"    Contract: {info.contract_size}, Point: {info.point}")
        except Exception as e:
            return False, f"Invalid symbol: {e}"
        
        # 2. Volume validation
        if request.volume < info.volume_min:
            return False, f"Volume {request.volume} below minimum {info.volume_min}"
        if request.volume > info.volume_max:
            return False, f"Volume {request.volume} above maximum {info.volume_max}"
        
        # Normalize volume to step
        normalized = round(request.volume / info.volume_step) * info.volume_step
        if abs(normalized - request.volume) > 0.0001:
            self._log(f"  ⚠ Volume normalized: {request.volume} → {normalized}")
            request.volume = normalized
        else:
            self._log(f"  ✓ Volume {request.volume} validated")
        
        # 3. Margin check
        is_buy = request.type in [ENUM_ORDER_TYPE.ORDER_TYPE_BUY, ENUM_ORDER_TYPE.ORDER_TYPE_BUY_LIMIT]
        margin_required = info.calculate_margin(request.volume, request.price, self.account.Leverage())
        
        self._log(f"  Margin required: ${margin_required:.2f}")
        self._log(f"  Free margin: ${self.account.FreeMargin():.2f}")
        
        if margin_required > self.account.FreeMargin():
            return False, f"Insufficient margin: need ${margin_required:.2f}, have ${self.account.FreeMargin():.2f}"
        self._log(f"  ✓ Margin check passed")
        
        # 4. Stops level check
        if info.stops_level > 0:
            min_distance = info.stops_level * info.point
            if request.sl > 0:
                sl_distance = abs(request.price - request.sl)
                if sl_distance < min_distance:
                    return False, f"SL too close: {sl_distance:.5f} < {min_distance:.5f}"
            if request.tp > 0:
                tp_distance = abs(request.price - request.tp)
                if tp_distance < min_distance:
                    return False, f"TP too close: {tp_distance:.5f} < {min_distance:.5f}"
        self._log(f"  ✓ Stops level check passed")
        
        return True, "Validation passed"
    
    # =========================================================================
    # PHASE 2 & 3: ORDER SUBMISSION & EXECUTION
    # =========================================================================
    
    def execute_order(
        self,
        request: TradeRequest,
        bid: float,
        ask: float,
    ) -> tuple[TradeResult, Optional[Position]]:
        """
        Execute order with proper spread and slippage handling.
        
        Steps:
        1. Determine execution price (bid for sell, ask for buy)
        2. Apply slippage
        3. Calculate and deduct costs
        4. Create position
        """
        self._log(f"━━━ PHASE 2: ORDER SUBMISSION ━━━")
        self._log(f"  Order: {request.type.name} {request.volume} {request.symbol}")
        self._log(f"  Bid: {bid:.5f}, Ask: {ask:.5f}, Spread: {(ask-bid)/self._get_symbol_info(request.symbol).point:.0f} pts")
        
        result = TradeResult()
        info = self._get_symbol_info(request.symbol)
        
        # Determine if buy or sell
        is_buy = request.type in [ENUM_ORDER_TYPE.ORDER_TYPE_BUY, ENUM_ORDER_TYPE.ORDER_TYPE_BUY_LIMIT]
        
        self._log(f"━━━ PHASE 3: ORDER EXECUTION ━━━")
        
        # 1. Execution price (buy at ask, sell at bid)
        if is_buy:
            base_price = ask
            self._log(f"  BUY executes at ASK: {ask:.5f}")
        else:
            base_price = bid
            self._log(f"  SELL executes at BID: {bid:.5f}")
        
        # 2. Apply slippage (random within deviation)
        import random
        slippage_points = random.randint(0, request.deviation)
        slippage_price = slippage_points * info.point
        
        if is_buy:
            exec_price = base_price + slippage_price  # Slippage hurts buyer
        else:
            exec_price = base_price - slippage_price  # Slippage hurts seller
        
        slippage_cost = slippage_price * info.contract_size * request.volume
        self._log(f"  Slippage: {slippage_points} points (${slippage_cost:.2f})")
        self._log(f"  Final execution price: {exec_price:.5f}")
        
        # 3. Calculate costs
        spread_points = (ask - bid) / info.point
        spread_cost = (ask - bid) * info.contract_size * request.volume
        entry_commission = self.commission_per_lot * request.volume
        
        self._log(f"  Spread cost: ${spread_cost:.2f} ({spread_points:.0f} points)")
        self._log(f"  Entry commission: ${entry_commission:.2f}")
        
        # 4. Reserve margin
        margin_used = info.calculate_margin(request.volume, exec_price, self.account.Leverage())
        self.account.Update(margin=self.account.Margin() + margin_used)
        self._log(f"  Margin reserved: ${margin_used:.2f}")
        
        # Update statistics
        self.total_spread_cost += spread_cost
        self.total_commission += entry_commission
        self.total_slippage += slippage_cost
        
        # 5. Create position
        self.ticket_counter += 1
        position = Position(
            ticket=self.ticket_counter,
            symbol=request.symbol,
            type=ENUM_POSITION_TYPE.POSITION_TYPE_BUY if is_buy else ENUM_POSITION_TYPE.POSITION_TYPE_SELL,
            volume=request.volume,
            open_price=exec_price,
            open_time=self.current_time,
            sl=request.sl,
            tp=request.tp,
            spread_cost=spread_cost,
            entry_commission=entry_commission,
            slippage_cost=slippage_cost,
            current_price=exec_price,
        )
        
        position.log_event(TradeEvent.PRE_VALIDATION, "Trade validated")
        position.log_event(TradeEvent.ORDER_SUBMITTED, f"Order submitted: {request.type.name}")
        position.log_event(TradeEvent.ORDER_EXECUTED, f"Executed at {exec_price:.5f}")
        position.log_event(TradeEvent.POSITION_OPENED, f"Position opened, margin ${margin_used:.2f}")
        
        self.positions[position.ticket] = position
        
        # Populate result
        result.retcode = 10009  # TRADE_RETCODE_DONE
        result.deal = self.ticket_counter
        result.order = self.ticket_counter
        result.volume = request.volume
        result.price = exec_price
        result.comment = "Order executed"
        
        self._log(f"  ✓ Position #{position.ticket} opened")
        
        return result, position
    
    # =========================================================================
    # PHASE 4 & 5: POSITION MONITORING
    # =========================================================================
    
    def update_position(
        self,
        ticket: int,
        bid: float,
        ask: float,
        timestamp: datetime = None,
    ) -> Optional[CloseReason]:
        """
        Update position with current market price.
        
        Checks:
        1. Update unrealized P&L
        2. Update equity
        3. Check SL/TP
        4. Track MFE/MAE
        """
        if ticket not in self.positions:
            return None
        
        pos = self.positions[ticket]
        info = self._get_symbol_info(pos.symbol)
        
        if timestamp:
            self.current_time = timestamp
        
        # Current close price depends on position type
        is_long = pos.type == ENUM_POSITION_TYPE.POSITION_TYPE_BUY
        pos.current_price = bid if is_long else ask  # Close long at bid, short at ask
        
        # Calculate unrealized P&L
        direction = 1 if is_long else -1
        price_diff = (pos.current_price - pos.open_price) * direction
        pos.unrealized_pnl = price_diff * info.contract_size * pos.volume
        
        # Track MFE/MAE
        if pos.unrealized_pnl > pos.mfe:
            pos.mfe = pos.unrealized_pnl
        if pos.unrealized_pnl < pos.mae:
            pos.mae = pos.unrealized_pnl
        
        # Update equity
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.account.Update(equity=self.account.Balance() + total_unrealized)
        
        pos.log_event(TradeEvent.TICK_UPDATE, f"Price: {pos.current_price:.5f}, P&L: ${pos.unrealized_pnl:.2f}")
        
        # Check SL
        if pos.sl > 0:
            if is_long and bid <= pos.sl:
                self._log(f"  ⚠ STOP LOSS HIT at {bid:.5f}")
                pos.log_event(TradeEvent.SL_HIT, f"SL hit at {bid:.5f}")
                return CloseReason.STOP_LOSS
            elif not is_long and ask >= pos.sl:
                self._log(f"  ⚠ STOP LOSS HIT at {ask:.5f}")
                pos.log_event(TradeEvent.SL_HIT, f"SL hit at {ask:.5f}")
                return CloseReason.STOP_LOSS
        
        # Check TP
        if pos.tp > 0:
            if is_long and bid >= pos.tp:
                self._log(f"  ✓ TAKE PROFIT HIT at {bid:.5f}")
                pos.log_event(TradeEvent.TP_HIT, f"TP hit at {bid:.5f}")
                return CloseReason.TAKE_PROFIT
            elif not is_long and ask <= pos.tp:
                self._log(f"  ✓ TAKE PROFIT HIT at {ask:.5f}")
                pos.log_event(TradeEvent.TP_HIT, f"TP hit at {ask:.5f}")
                return CloseReason.TAKE_PROFIT
        
        # Check margin level
        if self.account.Margin() > 0:
            margin_level = self.account.MarginLevel()
            if margin_level < 50:  # 50% margin call
                self._log(f"  ⚠ MARGIN CALL at {margin_level:.1f}%")
                return CloseReason.MARGIN_CALL
        
        return None
    
    # =========================================================================
    # PHASE 6: OVERNIGHT PROCESSING
    # =========================================================================
    
    def process_overnight(self, ticket: int, is_triple_swap: bool = False):
        """
        Process overnight swap charge.
        
        Called once per day position is held overnight.
        """
        if ticket not in self.positions:
            return
        
        pos = self.positions[ticket]
        info = self._get_symbol_info(pos.symbol)
        
        self._log(f"━━━ PHASE 6: OVERNIGHT PROCESSING ━━━")
        
        is_long = pos.type == ENUM_POSITION_TYPE.POSITION_TYPE_BUY
        swap_rate = info.swap_long if is_long else info.swap_short
        
        # Triple swap on rollover day (usually Wednesday for Sat+Sun)
        multiplier = 3 if is_triple_swap else 1
        
        # Swap calculation: rate × tick_value × lots × multiplier
        daily_swap = swap_rate * info.tick_value * pos.volume * multiplier
        
        pos.swap_total += daily_swap
        pos.swap_days += multiplier
        self.total_swap += daily_swap
        
        swap_type = "TRIPLE SWAP" if is_triple_swap else "SWAP"
        self._log(f"  {swap_type}: {swap_rate} × ${info.tick_value} × {pos.volume} × {multiplier} = ${daily_swap:.2f}")
        self._log(f"  Total swap: ${pos.swap_total:.2f} ({pos.swap_days} days)")
        
        pos.log_event(TradeEvent.SWAP_CHARGED, f"Swap ${daily_swap:.2f} (day {pos.swap_days})")
    
    # =========================================================================
    # PHASE 7: POSITION CLOSE
    # =========================================================================
    
    def close_position(
        self,
        ticket: int,
        bid: float,
        ask: float,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> Optional[Position]:
        """
        Close position with full P&L calculation.
        
        Steps:
        1. Determine close price
        2. Calculate gross P&L
        3. Apply exit commission
        4. Calculate net P&L (gross - all costs)
        5. Update balance
        6. Release margin
        """
        if ticket not in self.positions:
            return None
        
        pos = self.positions[ticket]
        info = self._get_symbol_info(pos.symbol)
        
        self._log(f"━━━ PHASE 7: POSITION CLOSE ━━━")
        self._log(f"  Closing position #{ticket} - Reason: {reason.name}")
        
        is_long = pos.type == ENUM_POSITION_TYPE.POSITION_TYPE_BUY
        
        # 1. Close price (long closes at bid, short closes at ask)
        if reason == CloseReason.STOP_LOSS:
            pos.close_price = pos.sl
        elif reason == CloseReason.TAKE_PROFIT:
            pos.close_price = pos.tp
        else:
            pos.close_price = bid if is_long else ask
        
        pos.close_time = self.current_time
        pos.close_reason = reason
        
        self._log(f"  Open price:  {pos.open_price:.5f}")
        self._log(f"  Close price: {pos.close_price:.5f}")
        
        # 2. Calculate gross P&L
        direction = 1 if is_long else -1
        price_diff = (pos.close_price - pos.open_price) * direction
        pos.gross_pnl = price_diff * info.contract_size * pos.volume
        
        self._log(f"  Price diff: {price_diff:.5f} ({price_diff/info.point:.0f} points)")
        self._log(f"  Gross P&L: ${pos.gross_pnl:.2f}")
        
        # 3. Exit commission
        pos.exit_commission = self.commission_per_lot * pos.volume
        self.total_commission += pos.exit_commission
        
        self._log(f"  Exit commission: ${pos.exit_commission:.2f}")
        
        # 4. Calculate net P&L
        total_costs = (
            pos.spread_cost +
            pos.entry_commission +
            pos.exit_commission +
            pos.slippage_cost +
            (abs(pos.swap_total) if pos.swap_total < 0 else 0)  # Negative swap is cost
        )
        swap_credit = pos.swap_total if pos.swap_total > 0 else 0
        
        pos.net_pnl = pos.gross_pnl - pos.spread_cost - pos.entry_commission - pos.exit_commission - pos.slippage_cost + pos.swap_total
        
        self._log(f"\n  ═══ P&L BREAKDOWN ═══")
        self._log(f"  Gross P&L:        ${pos.gross_pnl:>10.2f}")
        self._log(f"  - Spread:         ${pos.spread_cost:>10.2f}")
        self._log(f"  - Entry comm:     ${pos.entry_commission:>10.2f}")
        self._log(f"  - Exit comm:      ${pos.exit_commission:>10.2f}")
        self._log(f"  - Slippage:       ${pos.slippage_cost:>10.2f}")
        self._log(f"  + Swap:           ${pos.swap_total:>10.2f}")
        self._log(f"  ─────────────────────────────")
        self._log(f"  NET P&L:          ${pos.net_pnl:>10.2f}")
        
        # 5. Update balance
        old_balance = self.account.Balance()
        new_balance = old_balance + pos.net_pnl
        
        # 6. Release margin
        margin_released = info.calculate_margin(pos.volume, pos.open_price, self.account.Leverage())
        new_margin = max(0, self.account.Margin() - margin_released)
        
        self.account.Update(balance=new_balance, margin=new_margin)
        
        self._log(f"\n  Balance: ${old_balance:.2f} → ${new_balance:.2f}")
        self._log(f"  Margin released: ${margin_released:.2f}")
        
        pos.log_event(TradeEvent.POSITION_CLOSED, f"Net P&L: ${pos.net_pnl:.2f}")
        pos.log_event(TradeEvent.TRADE_COMPLETE, f"Balance: ${new_balance:.2f}")
        
        # Move to closed
        del self.positions[ticket]
        self.closed_positions.append(pos)
        
        return pos
    
    # =========================================================================
    # PHASE 8: POST-TRADE SUMMARY
    # =========================================================================
    
    def get_trade_summary(self, pos: Position) -> Dict:
        """Get complete trade summary."""
        return {
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': pos.type.name,
            'volume': pos.volume,
            'open_price': pos.open_price,
            'close_price': pos.close_price,
            'open_time': pos.open_time,
            'close_time': pos.close_time,
            'close_reason': pos.close_reason.name if pos.close_reason else None,
            'gross_pnl': pos.gross_pnl,
            'spread_cost': pos.spread_cost,
            'commission_total': pos.entry_commission + pos.exit_commission,
            'slippage_cost': pos.slippage_cost,
            'swap_total': pos.swap_total,
            'net_pnl': pos.net_pnl,
            'mfe': pos.mfe,
            'mae': pos.mae,
            'events': pos.events,
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("TRADE LIFECYCLE - COMPREHENSIVE SEQUENCE OF EVENTS")
    print("=" * 70)
    
    # Create manager
    manager = TradeLifecycleManager(
        initial_balance=10000.0,
        leverage=100,
        commission_per_lot=7.0,
    )
    
    print(f"\nInitial Account State:")
    print(f"  Balance: ${manager.account.Balance():,.2f}")
    print(f"  Leverage: {manager.account.Leverage()}:1")
    
    # =========================================================================
    # TRADE 1: EURUSD Long - Full lifecycle with overnight
    # =========================================================================
    print("\n" + "═" * 70)
    print("TRADE 1: EURUSD LONG - Full Lifecycle with Overnight Hold")
    print("═" * 70)
    
    # Market prices
    bid = 1.08500
    ask = 1.08510  # 10 point spread
    
    # Create trade request
    request = TradeRequest(
        action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
        symbol="EURUSD",
        volume=0.5,
        type=ENUM_ORDER_TYPE.ORDER_TYPE_BUY,
        price=ask,
        sl=1.08300,  # 21 pips SL
        tp=1.08800,  # 29 pips TP
        deviation=5,
        comment="Test trade"
    )
    
    # Phase 1: Validate
    valid, msg = manager.validate_trade(request)
    if not valid:
        print(f"Validation failed: {msg}")
        return
    
    # Phase 2-4: Execute and open
    result, pos = manager.execute_order(request, bid, ask)
    
    if result.retcode != 10009:
        print(f"Execution failed: {result.comment}")
        return
    
    # Phase 5: Simulate price updates
    print(f"\n━━━ PHASE 5: POSITION MONITORING ━━━")
    
    # Tick 1: Price moves up
    manager.current_time += timedelta(hours=1)
    close_reason = manager.update_position(pos.ticket, 1.08550, 1.08560)
    print(f"  Tick 1: Bid 1.08550 - Unrealized: ${pos.unrealized_pnl:.2f}")
    
    # Tick 2: Price moves more
    manager.current_time += timedelta(hours=2)
    close_reason = manager.update_position(pos.ticket, 1.08600, 1.08610)
    print(f"  Tick 2: Bid 1.08600 - Unrealized: ${pos.unrealized_pnl:.2f}")
    
    # Phase 6: Overnight swap (simulate holding overnight)
    manager.current_time += timedelta(hours=20)
    manager.process_overnight(pos.ticket, is_triple_swap=False)
    
    # Continue monitoring next day
    manager.current_time += timedelta(hours=5)
    close_reason = manager.update_position(pos.ticket, 1.08650, 1.08660)
    print(f"  Tick 3 (next day): Bid 1.08650 - Unrealized: ${pos.unrealized_pnl:.2f}")
    
    # Phase 7: Close position manually
    closed = manager.close_position(pos.ticket, 1.08650, 1.08660, CloseReason.MANUAL)
    
    # Phase 8: Summary
    print(f"\n━━━ PHASE 8: TRADE SUMMARY ━━━")
    summary = manager.get_trade_summary(closed)
    print(f"  Ticket: #{summary['ticket']}")
    print(f"  Symbol: {summary['symbol']}")
    print(f"  Type: {summary['type']}")
    print(f"  Volume: {summary['volume']} lots")
    print(f"  Open → Close: {summary['open_price']:.5f} → {summary['close_price']:.5f}")
    print(f"  Duration: {summary['close_time'] - summary['open_time']}")
    print(f"  MFE: ${summary['mfe']:.2f}, MAE: ${summary['mae']:.2f}")
    
    # =========================================================================
    # TRADE 2: XAUUSD Short - SL Hit
    # =========================================================================
    print("\n" + "═" * 70)
    print("TRADE 2: XAUUSD SHORT - Stop Loss Hit")
    print("═" * 70)
    
    gold_bid = 2650.00
    gold_ask = 2650.30
    
    request2 = TradeRequest(
        action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
        symbol="XAUUSD",
        volume=0.1,
        type=ENUM_ORDER_TYPE.ORDER_TYPE_SELL,
        price=gold_bid,
        sl=2660.00,  # $10 SL
        tp=2630.00,  # $20 TP
        deviation=10,
    )
    
    valid, msg = manager.validate_trade(request2)
    result2, pos2 = manager.execute_order(request2, gold_bid, gold_ask)
    
    # Price moves against us
    print(f"\n━━━ PHASE 5: POSITION MONITORING ━━━")
    manager.current_time += timedelta(hours=1)
    close_reason = manager.update_position(pos2.ticket, 2655.00, 2655.30)
    print(f"  Tick 1: Ask 2655.30 - Unrealized: ${pos2.unrealized_pnl:.2f}")

    # Price hits SL
    manager.current_time += timedelta(hours=1)
    close_reason = manager.update_position(pos2.ticket, 2660.00, 2660.30)

    closed2 = None  # Initialize before conditional block
    if close_reason == CloseReason.STOP_LOSS:
        closed2 = manager.close_position(pos2.ticket, 2660.00, 2660.30, close_reason)
        print(f"\n  Trade closed by STOP LOSS")
        print(f"  Net P&L: ${closed2.net_pnl:.2f}")

    # =========================================================================
    # FINAL ACCOUNT STATE
    # =========================================================================
    print("\n" + "═" * 70)
    print("FINAL ACCOUNT STATE")
    print("═" * 70)

    print(f"\n  Starting Balance:  ${10000.00:>10,.2f}")
    print(f"  Trade 1 P&L:       ${summary['net_pnl']:>10,.2f}")
    if closed2:
        print(f"  Trade 2 P&L:       ${closed2.net_pnl:>10,.2f}")
    else:
        print(f"  Trade 2 P&L:       $0.00")
    print(f"  ─────────────────────────────────")
    print(f"  Final Balance:     ${manager.account.Balance():>10,.2f}")
    
    print(f"\n  Total Costs:")
    print(f"    Spread:     ${manager.total_spread_cost:>8,.2f}")
    print(f"    Commission: ${manager.total_commission:>8,.2f}")
    print(f"    Slippage:   ${manager.total_slippage:>8,.2f}")
    print(f"    Swap:       ${manager.total_swap:>8,.2f}")
    print(f"    ─────────────────────────")
    total_costs = manager.total_spread_cost + manager.total_commission + manager.total_slippage + abs(manager.total_swap)
    print(f"    TOTAL:      ${total_costs:>8,.2f}")
    
    # Event log for first trade
    print(f"\n" + "═" * 70)
    print("TRADE 1 EVENT LOG")
    print("═" * 70)
    for event in summary['events']:
        print(f"  [{event['event']}] {event['details']}")


if __name__ == "__main__":
    main()
