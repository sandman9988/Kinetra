"""
MT5-Style Trade Logger for Kinetra

Provides detailed real-time logging during backtests with extensive measurements:
- Order execution (entry/exit)
- Deal confirmation with prices
- Position tracking
- Friction costs breakdown
- MFE/MAE tracking
- Regime information
- Health metrics
- Performance statistics

Usage:
    logger = MT5Logger(symbol="EURUSD", enable_verbose=True)
    logger.log_order_send(...)
    logger.log_deal(...)
    logger.log_final_summary(...)
"""

from datetime import datetime
from typing import Optional, Dict, Any
import sys


class MT5Logger:
    """
    MT5-style transaction logger with enhanced metrics.

    Mimics MT5's verbose backtest output but adds comprehensive
    measurements for friction costs, execution quality, and regime analysis.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M15",
        initial_balance: float = 10000.0,
        enable_verbose: bool = True,
        log_file: Optional[str] = None,
    ):
        """
        Initialize logger.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            initial_balance: Starting balance
            enable_verbose: Enable detailed logging
            log_file: Optional file path for log output
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.enable_verbose = enable_verbose
        self.log_file = log_file

        # Counters
        self.order_id = 0
        self.deal_id = 0
        self.position_id = 0

        # Open positions
        self.open_positions: Dict[int, Dict] = {}

        # Statistics
        self.total_orders = 0
        self.total_deals = 0
        self.failed_orders = 0
        self.current_balance = initial_balance

        # Start time
        self.start_time = datetime.now()

    def _log(self, message: str, level: str = "INFO"):
        """Write log message to console and/or file."""
        if not self.enable_verbose:
            return

        timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S.%f")[:-3]
        formatted = f"{timestamp}\tCore 01\t{message}"

        print(formatted)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")

    def log_order_send(
        self,
        time: datetime,
        action: str,
        volume: float,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        spread_points: float = 0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        close_position_id: Optional[int] = None,
    ) -> int:
        """
        Log order send (like CTrade::OrderSend).

        Args:
            time: Order time
            action: 'buy' or 'sell'
            volume: Lot size
            price: Execution price
            sl: Stop loss
            tp: Take profit
            spread_points: Current spread in points
            bid: Bid price
            ask: Ask price
            close_position_id: Position to close (if exit order)

        Returns:
            Order ID
        """
        self.order_id += 1
        self.total_orders += 1

        # Format prices
        if bid is None:
            bid = price
        if ask is None:
            ask = price

        # Determine if this is entry or exit
        if close_position_id:
            # Exit order
            position_info = f"position #{close_position_id}"
            self._log(
                f"{time}   market {action} {volume:.2f} {self.symbol}, close #{close_position_id} "
                f"({bid:.{self._get_digits()}f} / {ask:.{self._get_digits()}f} / {price:.{self._get_digits()}f})"
            )
        else:
            # Entry order
            position_info = self.symbol
            self._log(
                f"{time}   market {action} {volume:.2f} {self.symbol} "
                f"({bid:.{self._get_digits()}f} / {ask:.{self._get_digits()}f} / {price:.{self._get_digits()}f})"
            )

        # Log the CTrade call
        sl_str = f" sl: {sl:.{self._get_digits()}f}" if sl else ""
        tp_str = f" tp: {tp:.{self._get_digits()}f}" if tp else ""
        spread_str = f" spread: {spread_points:.0f}pts" if spread_points > 0 else ""

        self._log(
            f"{time}   CTrade::OrderSend: market {action} {volume:.2f} {position_info}{sl_str}{tp_str}{spread_str} "
            f"[done at {price:.{self._get_digits()}f}]"
        )

        return self.order_id

    def log_order_failed(
        self,
        time: datetime,
        action: str,
        volume: float,
        reason: str,
        close_position_id: Optional[int] = None,
    ):
        """Log failed order."""
        self.failed_orders += 1

        if close_position_id:
            self._log(
                f"{time}   failed market {action} {volume:.2f} {self.symbol}, "
                f"close #{close_position_id} [{reason}]"
            )
        else:
            self._log(f"{time}   failed market {action} {volume:.2f} {self.symbol} [{reason}]")

        self._log(f"{time}   CTrade::OrderSend: market {action} {volume:.2f} {self.symbol} [{reason.lower()}]")

    def log_deal(
        self,
        time: datetime,
        action: str,
        volume: float,
        price: float,
        order_id: int,
        position_id: Optional[int] = None,
    ) -> int:
        """
        Log deal execution.

        Args:
            time: Deal time
            action: 'buy' or 'sell'
            volume: Lot size
            price: Execution price
            order_id: Related order ID
            position_id: Position ID (for exits)

        Returns:
            Deal ID
        """
        self.deal_id += 1
        self.total_deals += 1

        self._log(
            f"{time}   deal #{self.deal_id} {action} {volume:.2f} {self.symbol} "
            f"at {price:.{self._get_digits()}f} done (based on order #{order_id})"
        )

        self._log(f"{time}   deal performed [#{self.deal_id} {action} {volume:.2f} {self.symbol} at {price:.{self._get_digits()}f}]")

        self._log(
            f"{time}   order performed {action} {volume:.2f} at {price:.{self._get_digits()}f} "
            f"[#{self.deal_id} {action} {volume:.2f} {self.symbol} at {price:.{self._get_digits()}f}]"
        )

        return self.deal_id

    def log_position_open(
        self,
        time: datetime,
        direction: int,
        volume: float,
        entry_price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        spread: float = 0,
        commission: float = 0,
        regime: Optional[str] = None,
    ) -> int:
        """
        Log position opening with enhanced metrics.

        Args:
            time: Entry time
            direction: 1 for long, -1 for short
            volume: Lot size
            entry_price: Entry price
            sl: Stop loss
            tp: Take profit
            spread: Entry spread cost
            commission: Entry commission
            regime: Market regime at entry

        Returns:
            Position ID
        """
        self.position_id += 1

        action = "buy" if direction == 1 else "sell"

        # Store position
        self.open_positions[self.position_id] = {
            'entry_time': time,
            'direction': direction,
            'volume': volume,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_spread': spread,
            'entry_commission': commission,
            'regime': regime,
        }

        # Log enhanced position details
        self._log(f"{time}   ═══════════════════════════════════════════════════════════")
        self._log(f"{time}   ✅ POSITION OPENED #{self.position_id}")
        self._log(f"{time}   Direction:        {'LONG' if direction == 1 else 'SHORT'} {volume:.2f} lots")
        self._log(f"{time}   Entry price:      {entry_price:.{self._get_digits()}f}")
        if sl:
            pips_to_sl = abs(entry_price - sl) * self._get_pip_multiplier()
            self._log(f"{time}   Stop loss:        {sl:.{self._get_digits()}f} ({pips_to_sl:.1f} pips)")
        if tp:
            pips_to_tp = abs(tp - entry_price) * self._get_pip_multiplier()
            self._log(f"{time}   Take profit:      {tp:.{self._get_digits()}f} ({pips_to_tp:.1f} pips)")
        if spread > 0:
            self._log(f"{time}   Entry spread:     ${spread:.2f}")
        if commission > 0:
            self._log(f"{time}   Entry commission: ${commission:.2f}")
        if regime:
            self._log(f"{time}   Market regime:    {regime}")
        self._log(f"{time}   ═══════════════════════════════════════════════════════════")

        return self.position_id

    def log_position_close(
        self,
        time: datetime,
        position_id: int,
        exit_price: float,
        pnl: float,
        spread: float = 0,
        commission: float = 0,
        swap: float = 0,
        slippage: float = 0,
        mfe: float = 0,
        mae: float = 0,
        mfe_efficiency: float = 0,
        holding_hours: float = 0,
        exit_reason: str = "signal",
    ):
        """
        Log position close with comprehensive metrics.

        Args:
            time: Exit time
            position_id: Position ID
            exit_price: Exit price
            pnl: Net P&L
            spread: Exit spread cost
            commission: Total commission (entry + exit)
            swap: Swap/rollover cost
            slippage: Slippage cost
            mfe: Maximum favorable excursion
            mae: Maximum adverse excursion
            mfe_efficiency: MFE capture percentage
            holding_hours: Hours held
            exit_reason: Why position closed (signal, SL, TP, etc.)
        """
        if position_id not in self.open_positions:
            self._log(f"{time}   ⚠️  WARNING: Position #{position_id} not found")
            return

        position = self.open_positions.pop(position_id)

        # Update balance
        self.current_balance += pnl

        # Calculate metrics
        direction_str = 'LONG' if position['direction'] == 1 else 'SHORT'
        entry_price = position['entry_price']
        price_change = (exit_price - entry_price) * position['direction']
        price_change_pips = price_change * self._get_pip_multiplier()

        gross_pnl = pnl + spread + commission + swap + slippage
        total_costs = spread + commission + swap + slippage

        # Log detailed close information
        self._log(f"{time}   ═══════════════════════════════════════════════════════════")
        self._log(f"{time}   ❌ POSITION CLOSED #{position_id} ({exit_reason})")
        self._log(f"{time}   ───────────────────────────────────────────────────────────")
        self._log(f"{time}   [POSITION INFO]")
        self._log(f"{time}     Direction:      {direction_str} {position['volume']:.2f} lots")
        self._log(f"{time}     Entry:          {entry_price:.{self._get_digits()}f} @ {position['entry_time']}")
        self._log(f"{time}     Exit:           {exit_price:.{self._get_digits()}f} @ {time}")
        self._log(f"{time}     Holding time:   {holding_hours:.1f} hours ({holding_hours/24:.1f} days)")
        self._log(f"{time}     Price captured: {price_change:.{self._get_digits()}f} ({price_change_pips:.1f} pips)")

        self._log(f"{time}   ───────────────────────────────────────────────────────────")
        self._log(f"{time}   [EXECUTION QUALITY]")
        if mfe > 0:
            mfe_pips = mfe * self._get_pip_multiplier()
            self._log(f"{time}     MFE (best):     {mfe:.{self._get_digits()}f} ({mfe_pips:.1f} pips)")
        if mae > 0:
            mae_pips = mae * self._get_pip_multiplier()
            self._log(f"{time}     MAE (worst):    {mae:.{self._get_digits()}f} ({mae_pips:.1f} pips)")
        if mfe > 0 and mae > 0:
            self._log(f"{time}     MFE efficiency: {mfe_efficiency:.1%}")
            self._log(f"{time}     MFE/MAE ratio:  {mfe/mae:.2f}x")

        self._log(f"{time}   ───────────────────────────────────────────────────────────")
        self._log(f"{time}   [TRANSACTION COSTS]")
        if spread > 0:
            self._log(f"{time}     Spread:         ${spread:.2f}")
        if commission > 0:
            self._log(f"{time}     Commission:     ${commission:.2f}")
        if swap != 0:
            self._log(f"{time}     Swap:           ${swap:.2f}")
        if slippage != 0:
            self._log(f"{time}     Slippage:       ${slippage:.2f}")
        self._log(f"{time}     ─────────────────────────")
        self._log(f"{time}     Total costs:    ${total_costs:.2f}")

        self._log(f"{time}   ───────────────────────────────────────────────────────────")
        self._log(f"{time}   [PROFIT & LOSS]")
        self._log(f"{time}     Gross P&L:      ${gross_pnl:,.2f}")
        self._log(f"{time}     Costs:          $({total_costs:.2f})")
        self._log(f"{time}     ─────────────────────────")

        # Color-coded net P&L
        pnl_symbol = "💰" if pnl > 0 else "💸"
        pnl_status = "PROFIT" if pnl > 0 else "LOSS"
        self._log(f"{time}     Net P&L:        ${pnl:,.2f} {pnl_symbol} {pnl_status}")
        self._log(f"{time}     New balance:    ${self.current_balance:,.2f}")
        self._log(f"{time}     Return:         {pnl/self.initial_balance:.2%}")

        if gross_pnl > 0:
            cost_pct = (total_costs / gross_pnl) * 100
            self._log(f"{time}     Cost impact:    {cost_pct:.1f}% of gross")

        self._log(f"{time}   ═══════════════════════════════════════════════════════════")

    def log_regime_change(self, time: datetime, old_regime: str, new_regime: str, reason: str):
        """Log market regime transition."""
        self._log(f"{time}   📊 REGIME CHANGE: {old_regime} → {new_regime} ({reason})")

    def log_health_update(
        self,
        time: datetime,
        score: float,
        state: str,
        action: str,
        risk_multiplier: float,
    ):
        """Log portfolio health update."""
        emoji = {"HEALTHY": "✅", "WARNING": "⚠️", "DEGRADED": "🔴", "CRITICAL": "💀"}.get(state, "❓")
        self._log(f"{time}   {emoji} HEALTH: {score:.1f}/100 ({state}) | Risk: {risk_multiplier:.0%} | Action: {action}")

    def log_agent_event(self, time: datetime, event: str, details: str):
        """Log agent-related events (drift, promotion, etc.)."""
        self._log(f"{time}   🤖 AGENT: {event} | {details}")

    def log_constraint_violation(
        self,
        time: datetime,
        violation_type: str,
        details: str,
    ):
        """Log MT5 constraint violations."""
        self._log(f"{time}   ⚠️  CONSTRAINT VIOLATION: {violation_type} | {details}")

    def log_final_summary(
        self,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        total_return_pct: float,
        max_drawdown: float,
        max_drawdown_pct: float,
        sharpe_ratio: float,
        total_spread_cost: float,
        total_commission: float,
        total_swap: float,
        total_slippage: float,
        freeze_violations: int,
        stop_violations: int,
        bars_processed: int,
        memory_mb: float = 0,
    ):
        """
        Log comprehensive final summary.

        Args:
            total_trades: Total closed trades
            winning_trades: Winning trades
            losing_trades: Losing trades
            total_pnl: Total net P&L
            total_return_pct: Total return percentage
            max_drawdown: Max drawdown (absolute)
            max_drawdown_pct: Max drawdown (percentage)
            sharpe_ratio: Sharpe ratio
            total_spread_cost: Total spread costs
            total_commission: Total commission
            total_swap: Total swap
            total_slippage: Total slippage
            freeze_violations: Freeze zone violations
            stop_violations: Stop level violations
            bars_processed: Bars processed
            memory_mb: Memory used
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        total_costs = total_spread_cost + total_commission + total_swap + total_slippage

        self._log("")
        self._log("═══════════════════════════════════════════════════════════════════════")
        self._log("                        BACKTEST COMPLETED")
        self._log("═══════════════════════════════════════════════════════════════════════")
        self._log("")
        self._log(f"final balance {self.current_balance:.2f} USD")
        self._log("")
        self._log("[TRADING STATISTICS]")
        self._log(f"  Total trades:        {total_trades}")
        self._log(f"  Winning trades:      {winning_trades} ({win_rate:.1%})")
        self._log(f"  Losing trades:       {losing_trades} ({1-win_rate:.1%})")
        self._log(f"  Average P&L:         ${avg_pnl:,.2f}")
        self._log("")
        self._log("[PERFORMANCE]")
        self._log(f"  Total P&L:           ${total_pnl:,.2f}")
        self._log(f"  Total return:        {total_return_pct:.2%}")
        self._log(f"  Max drawdown:        ${max_drawdown:,.2f} ({max_drawdown_pct:.1%})")
        self._log(f"  Sharpe ratio:        {sharpe_ratio:.2f}")
        self._log("")
        self._log("[FRICTION COSTS]")
        self._log(f"  Total spread:        ${total_spread_cost:,.2f}")
        self._log(f"  Total commission:    ${total_commission:,.2f}")
        self._log(f"  Total swap:          ${total_swap:,.2f}")
        if total_slippage > 0:
            self._log(f"  Total slippage:      ${total_slippage:,.2f}")
        self._log(f"  ─────────────────────────────")
        self._log(f"  Total costs:         ${total_costs:,.2f}")
        if total_pnl + total_costs > 0:
            cost_pct = (total_costs / (total_pnl + total_costs)) * 100
            self._log(f"  Cost impact:         {cost_pct:.1f}% of gross P&L")
        self._log("")
        self._log("[MT5 CONSTRAINTS]")
        self._log(f"  Freeze violations:   {freeze_violations}")
        self._log(f"  Stop violations:     {stop_violations}")
        self._log(f"  Failed orders:       {self.failed_orders}")
        self._log("")
        self._log("[EXECUTION STATS]")
        self._log(f"  Bars processed:      {bars_processed:,}")
        self._log(f"  Total orders:        {self.total_orders}")
        self._log(f"  Total deals:         {self.total_deals}")
        self._log(f"  Duration:            {duration:.2f} seconds")
        if memory_mb > 0:
            self._log(f"  Memory used:         {memory_mb:.1f} MB")
        self._log("")
        self._log(f"{self.symbol},{self.timeframe}: Test completed in {duration:.2f}s")
        self._log("═══════════════════════════════════════════════════════════════════════")

        # Color-coded final result
        if total_return_pct > 0:
            self._log("✅ BACKTEST PROFITABLE")
        else:
            self._log("❌ BACKTEST UNPROFITABLE")
        self._log("═══════════════════════════════════════════════════════════════════════")

    def _get_digits(self) -> int:
        """Get number of decimal places for symbol."""
        # Simple heuristic - can be made configurable
        if "JPY" in self.symbol:
            return 3
        return 5

    def _get_pip_multiplier(self) -> float:
        """Get pip multiplier for symbol."""
        if "JPY" in self.symbol:
            return 1000  # 1 pip = 0.01 for JPY pairs
        return 10000  # 1 pip = 0.0001 for other pairs
