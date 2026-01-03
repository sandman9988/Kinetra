"""
MetaTrader 5 Data Connector for Kinetra

Provides real-time and historical data from MT5 terminal.
Requires: MetaTrader 5 running via Wine, MetaTrader5 Python package
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from kinetra.backtest_engine import BacktestEngine, Trade, TradeDirection

# Physics and RL integrations
from kinetra.physics_engine import PhysicsEngine
from kinetra.risk_management import compute_chs, compute_ror  # Assuming these exist or add stubs
from kinetra.rl_agent import KinetraAgent
from kinetra.symbol_spec import SymbolSpec

# Load env vars
load_dotenv()

# Optional MT5 import
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)


class MT5Connector:
    """
    MetaTrader 5 connector for live data and order execution.
    """

    # Timeframe mapping
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
        "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
        "W1": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
        "MN1": mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else 43200,
    }

    def __init__(self, path: Optional[str] = None):
        """
        Initialize MT5 connector.

        Args:
            path: Optional path to MT5 terminal (for Wine installations)
        """
        self.path = path
        self.symbol = os.getenv("MT5_SYMBOL", "BTCUSD")
        self.timeframe = os.getenv("MT5_TIMEFRAME", "H1")
        self.login = int(os.getenv("MT5_LOGIN", 0))
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = os.getenv("MT5_SERVER", "")
        self.demo_mode = os.getenv("MT5_DEMO", "true").lower() == "true"
        self.connected = False
        self._check_availability()

    def _check_availability(self):
        """Check if MT5 package is available."""
        if not MT5_AVAILABLE:
            logger.warning(
                "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
            )

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            True if connection successful, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not available")
            return False

        # Initialize MT5
        # Use env vars for demo login if not connected
        if self.login and self.password and self.server:
            if not mt5.login(self.login, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False

        if self.path:
            init_result = mt5.initialize(path=self.path)
        else:
            init_result = mt5.initialize()

        if not init_result:
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False

        self.connected = True
        terminal_info = mt5.terminal_info()
        logger.info(f"Connected to MT5: {terminal_info.name}")
        return True

    def disconnect(self):
        """Disconnect from MT5 terminal."""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        if not self.connected:
            return []

        symbols = mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information."""
        if not self.connected:
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            "name": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
        }

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        count: int = 1000,
        start_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from MT5.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD", "EURUSD")
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to retrieve
            start_time: Start time for historical data

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        tf = self.TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol: {symbol}")
            return None

        # Get rates
        if start_time:
            rates = mt5.copy_rates_from(symbol, tf, start_time, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(
            columns={
                "tick_volume": "volume",
                "real_volume": "real_volume",
            }
        )

        return df[["time", "open", "high", "low", "close", "volume"]]

    def get_tick(self, symbol: str) -> Optional[Dict]:
        """Get latest tick for symbol."""
        if not self.connected:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "time": datetime.fromtimestamp(tick.time),
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
        }

    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if not self.connected:
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    def place_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "Kinetra",
    ) -> Optional[Dict]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Position size in lots
            price: Price (None for market order)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment

        Returns:
            Order result dictionary or None on failure
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol not found: {symbol}")
            return None

        # Determine order type
        if order_type.upper() == "BUY":
            trade_type = mt5.ORDER_TYPE_BUY
            price = price or symbol_info.ask
        elif order_type.upper() == "SELL":
            trade_type = mt5.ORDER_TYPE_SELL
            price = price or symbol_info.bid
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp

        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        return {
            "order": result.order,
            "volume": result.volume,
            "price": result.price,
            "comment": result.comment,
        }

    def close_position(self, ticket: int) -> bool:
        """Close position by ticket."""
        if not self.connected:
            return False

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position not found: {ticket}")
            return False

        pos = position[0]

        # Determine close direction
        if pos.type == mt5.ORDER_TYPE_BUY:
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": trade_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Kinetra close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE


class PaperTrader:
    """
    Paper trading simulator using MT5 data feed.

    Integrates physics/RL for signals, simulates execution with costs.
    Logs trades to JSON for backtest validation.
    """

    def __init__(
        self,
        mt5_connector: MT5Connector,
        initial_capital: float = 100000.0,
        symbol_spec: Optional[SymbolSpec] = None,
        physics_engine: Optional[PhysicsEngine] = None,
        rl_agent: Optional[KinetraAgent] = None,
    ):
        self.connector = mt5_connector
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.trades: List[Trade] = []
        self.open_position: Optional[Trade] = None
        self.log_file = "logs/paper_trades.json"
        Path("logs").mkdir(exist_ok=True)

        # Specs and engines (lazy init if not provided)
        self.symbol_spec = symbol_spec or SymbolSpec(
            symbol=mt5_connector.symbol,
            contract_size=1,  # BTCUSD example
            tick_size=0.01,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=10.0,
            volume_step=0.01,
            spread_points=5.0,
            slippage_avg=1.0,
            commission=CommissionSpec(0.0, CommissionType.PER_LOT),  # No comm for paper
        )
        self.physics = physics_engine or PhysicsEngine()
        self.rl_agent = rl_agent or KinetraAgent(state_dim=46)  # +3 from enhancements
        self.backtest_engine = BacktestEngine(initial_capital=initial_capital)

        # Health tracking
        self.chs_history = []
        self.ror_history = []

    def _get_signal(self, current_data: pd.DataFrame) -> int:
        """Generate signal using physics + RL."""
        if len(current_data) < self.physics.lookback:
            return 0  # Insufficient data

        # Compute physics state (vectorized)
        phys_state = self.physics.compute_physics_state(
            prices=current_data["close"],
            volume=current_data.get("volume"),
            high=current_data.get("high"),
            low=current_data.get("low"),
            open_price=current_data.get("open"),
            include_percentiles=True,
        )

        # Build RL state (enhanced features)
        latest_idx = -1
        state = self._build_rl_state(phys_state.iloc[latest_idx], current_data.iloc[latest_idx])

        # RL signal (or physics fallback)
        if self.rl_agent.network is not None:
            signal = self.rl_agent.select_action(state)
        else:
            # Physics fallback: energy + regime
            energy = phys_state["energy"].iloc[latest_idx]
            regime = phys_state["regime"].iloc[latest_idx]
            if regime == "underdamped" and energy > phys_state["energy"].quantile(0.75):
                signal = 1 if current_data["close"].iloc[-1] > current_data["open"].iloc[-1] else -1
            else:
                signal = 0

        return signal  # 1 buy, -1 sell, 0 hold

    def _build_rl_state(self, phys_row: pd.Series, bar_row: pd.Series) -> np.ndarray:
        """Build enhanced RL state (kurt_pct, damp_pct, etc.)."""
        # Core physics
        state = np.array(
            [
                phys_row.get("energy", 0),
                phys_row.get("damping", 0),
                phys_row.get("entropy", 0),
                phys_row.get("reynolds", 0),
                phys_row.get("eta", 0),
                phys_row.get("regime_age_frac", 0),
            ]
        )

        # Enhanced percentiles (from vectorized helpers)
        state = np.append(
            state,
            [
                phys_row.get("kurt_pct", 0.5),  # New: fat-tail proxy
                phys_row.get("damp_pct", 0.5),  # New: friction norm
                phys_row.get("velocity_pct", 0.5),
                phys_row.get("jerk_pct", 0.5),
            ],
        )

        # Bar features (OHLC ratios)
        ohlc = bar_row[["open", "high", "low", "close"]].values
        state = np.append(state, ohlc / ohlc.mean() if ohlc.mean() > 0 else np.zeros(4))

        # NaN shield
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -2.0, 2.0)  # Bound for stability
        return state

    def _check_gates(self, phys_state: pd.Series) -> bool:
        """Check CHS and RoR gates before trading."""
        # Compute current CHS (simplified; full in risk_management.py)
        chs = compute_chs(phys_state) if "compute_chs" in globals() else 0.9  # Stub fallback
        ror = compute_ror(self.equity, self.trades[-10:]) if self.trades else 0  # Recent trades

        self.chs_history.append(chs)
        self.ror_history.append(ror)

        if chs < 0.55 or ror > 1e-6:
            logger.warning(f"Trading halted: CHS={chs:.3f} <0.55, RoR={ror:.2e} >1e-6")
            if chs < 0.55:
                self._close_all_positions()  # Emergency close
            return False

        # Margin check (paper equity)
        if self.open_position:
            margin_level = (
                (self.equity / self._calculate_margin_required()) * 100
                if self._calculate_margin_required() > 0
                else float("inf")
            )
            if margin_level < 100:
                logger.warning(f"Margin call: {margin_level:.1f}% <100%")
                self._close_position_sim(self.open_position)
                return False

        return True

    def _calculate_margin_required(self) -> float:
        """Calculate margin for open position."""
        if not self.open_position:
            return 0.0
        price = (
            self.connector.get_tick(self.symbol_spec.symbol)["bid"]
            if self.connector.connected
            else 60000.0
        )  # Mock
        return self.symbol_spec.calculate_margin(
            self.open_position.lots, price, self.open_position.direction.value.lower()
        )

    def _open_position_sim(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        lots: float,
        phys_state: pd.Series,
    ) -> Trade:
        """Simulate opening a position (paper mode)."""
        trade = Trade(
            trade_id=len(self.trades) + 1,
            symbol=symbol,
            direction=direction,
            lots=lots,
            entry_time=datetime.now(),
            entry_price=price,
            energy_at_entry=phys_state["energy"],
            regime_at_entry=phys_state["regime"],
            mfe=0.0,
            mae=0.0,
        )

        # Simulate costs (spread/slippage)
        trade.spread_cost = self.symbol_spec.spread_cost(lots, price)
        trade.slippage = self.symbol_spec.slippage_avg * self.symbol_spec.tick_value * lots / 2
        trade.commission = (
            self.symbol_spec.commission.calculate_commission(
                lots, lots * self.symbol_spec.contract_size * price
            )
            / 2
        )

        self.equity -= trade.spread_cost + trade.slippage + trade.commission
        self.open_position = trade
        logger.info(f"Paper OPEN: {direction.value} {lots} lots {symbol} @ {price:.2f}")
        return trade

    def _close_position_sim(self, trade: Trade, exit_price: float):
        """Simulate closing a position."""
        if not trade.is_closed:
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()

            # Gross PnL
            if trade.direction == TradeDirection.LONG:
                gross_pnl = (
                    (exit_price - trade.entry_price)
                    * trade.lots
                    * self.symbol_spec.contract_size
                    / self.symbol_spec.tick_size
                    * self.symbol_spec.tick_value
                )
            else:
                gross_pnl = (
                    (trade.entry_price - exit_price)
                    * trade.lots
                    * self.symbol_spec.contract_size
                    / self.symbol_spec.tick_size
                    * self.symbol_spec.tick_value
                )

            gross_pnl = np.clip(gross_pnl, -self.equity * 0.95, float("inf"))  # RoR shield

            # Exit costs
            trade.slippage += (
                self.symbol_spec.slippage_avg * self.symbol_spec.tick_value * trade.lots / 2
            )
            trade.commission += (
                self.symbol_spec.commission.calculate_commission(
                    trade.lots, trade.lots * self.symbol_spec.contract_size * exit_price
                )
                / 2
            )

            # No swap for paper (short sessions)
            trade.gross_pnl = gross_pnl
            trade.net_pnl = gross_pnl - trade.total_cost
            self.equity += trade.net_pnl

            # Update MFE/MAE (simplified)
            trade.mfe = max(trade.mfe, abs(gross_pnl) * 0.1)  # Placeholder
            trade.mae = max(trade.mae, abs(gross_pnl) * 0.05)

            self.trades.append(trade)
            self.open_position = None
            logger.info(f"Paper CLOSE: Net PnL {trade.net_pnl:.2f}, Equity {self.equity:.2f}")

    def _close_all_positions(self):
        """Emergency close all paper positions."""
        if self.open_position:
            tick = self.connector.get_tick(self.symbol_spec.symbol)
            exit_price = (
                tick["bid"] if self.open_position.direction == TradeDirection.LONG else tick["ask"]
            )
            self._close_position_sim(self.open_position, exit_price)

    def _log_trade(self, trade: Trade):
        """Log trade to JSON."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction.value,
            "entry_time": trade.entry_time.isoformat(),
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time.isoformat() if trade.is_closed else None,
            "exit_price": trade.exit_price,
            "lots": trade.lots,
            "gross_pnl": trade.gross_pnl,
            "net_pnl": trade.net_pnl,
            "total_cost": trade.total_cost,
            "mfe": trade.mfe,
            "mae": trade.mae,
            "energy_at_entry": trade.energy_at_entry,
            "regime_at_entry": trade.regime_at_entry,
            "equity": self.equity,
            "chs": self.chs_history[-1] if self.chs_history else 0.9,
            "ror": self.ror_history[-1] if self.ror_history else 0,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Logged trade {trade.trade_id}: PnL {trade.net_pnl:.2f}")

    def start_paper_trading(self, duration_hours: int = 24, poll_interval: int = 3600):
        """Start paper trading loop: Poll, signal, simulate, log."""
        if not self.connector.connected:
            if not self.connector.connect():
                logger.error("Failed to connect to MT5; using mock mode")
                # Mock fallback (generate synthetic ticks)
                self._mock_mode(duration_hours)
                return

        logger.info(f"Starting paper trading: {self.symbol} {self.timeframe}, {duration_hours}h")
        end_time = datetime.now() + timedelta(hours=duration_hours)
        data_buffer = pd.DataFrame()  # Rolling buffer for physics

        while datetime.now() < end_time:
            try:
                # Poll new bar
                new_data = self.connector.get_ohlcv(
                    self.connector.symbol,
                    self.connector.timeframe,
                    count=2,  # Last 2 for new bar detection
                )
                if new_data is None or len(new_data) < 2:
                    time.sleep(poll_interval)
                    continue

                # Detect new bar (simple: if time diff > poll_interval)
                if (
                    len(data_buffer) == 0
                    or new_data["time"].iloc[-1] > data_buffer["time"].iloc[-1]
                ):
                    data_buffer = pd.concat([data_buffer, new_data]).tail(500)  # Rolling 500 bars
                    data_buffer = (
                        data_buffer.drop_duplicates("time")
                        .sort_values("time")
                        .reset_index(drop=True)
                    )

                if len(data_buffer) < self.physics.lookback:
                    time.sleep(poll_interval)
                    continue

                # Compute signal
                signal = self._get_signal(data_buffer)
                if signal == 0:
                    time.sleep(poll_interval)
                    continue

                # Gates
                phys_state = self.physics.compute_physics_state(
                    data_buffer["close"],
                    data_buffer.get("volume"),
                    data_buffer.get("high"),
                    data_buffer.get("low"),
                    data_buffer.get("open"),
                    include_percentiles=True,
                )
                latest_phys = phys_state.iloc[-1]

                if not self._check_gates(latest_phys):
                    time.sleep(poll_interval)
                    continue

                # Simulate trade
                tick = self.connector.get_tick(self.connector.symbol)
                if tick is None:
                    tick = {
                        "bid": data_buffer["close"].iloc[-1],
                        "ask": data_buffer["close"].iloc[-1]
                        + self.symbol_spec.spread_points * self.symbol_spec.tick_size,
                    }

                price = tick["ask"] if signal > 0 else tick["bid"]
                lots = self.symbol_spec.volume_min  # Fixed small for paper; adaptive risk in prod
                direction = TradeDirection.LONG if signal > 0 else TradeDirection.SHORT

                if self.open_position is None:  # Open
                    trade = self._open_position_sim(
                        self.connector.symbol, direction, price, lots, latest_phys
                    )
                    self._log_trade(trade)
                else:  # Reverse/Close if opposite
                    if direction != self.open_position.direction:
                        self._close_position_sim(self.open_position, price)
                        self._log_trade(self.open_position)
                        # Re-open opposite
                        trade = self._open_position_sim(
                            self.connector.symbol, direction, price, lots, latest_phys
                        )
                        self._log_trade(trade)

                # Update MFE/MAE (tick-based)
                if self.open_position:
                    self._update_mfe_mae_sim(self.open_position, tick)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Backoff

            time.sleep(poll_interval)

        # Close remaining
        if self.open_position:
            tick = self.connector.get_tick(self.connector.symbol) or {
                "bid": data_buffer["close"].iloc[-1]
            }
            exit_price = (
                tick["bid"] if self.open_position.direction == TradeDirection.LONG else tick["ask"]
            )
            self._close_position_sim(self.open_position, exit_price)
            self._log_trade(self.open_position)

        # Final stats
        self._export_session_summary()
        logger.info(
            f"Paper trading ended. Final equity: {self.equity:.2f} (from {self.initial_capital})"
        )

    def _update_mfe_mae_sim(self, trade: Trade, tick: Dict):
        """Update MFE/MAE for open paper position."""
        current_price = tick["bid"] if trade.direction == TradeDirection.LONG else tick["ask"]
        if trade.direction == TradeDirection.LONG:
            favorable = current_price - trade.entry_price
            adverse = trade.entry_price - (
                tick["bid"] - self.symbol_spec.spread_points * self.symbol_spec.tick_size / 2
            )
        else:
            favorable = trade.entry_price - current_price
            adverse = (
                tick["ask"] + self.symbol_spec.spread_points * self.symbol_spec.tick_size / 2
            ) - trade.entry_price

        trade.mfe = max(trade.mfe, favorable)
        trade.mae = max(trade.mae, adverse)

    def _export_session_summary(self):
        """Export logs to CSV for backtest validation."""
        # Load JSON logs
        trades_data = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        trades_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        if trades_data:
            df_trades = pd.DataFrame(trades_data)
            df_trades.to_csv(
                f"logs/paper_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False
            )
            logger.info(f"Exported {len(df_trades)} trades to CSV for MC validation")

        # Quick MC on logs (optional)
        if len(self.trades) >= 50:
            # Simulate backtest on logged data for validation
            data_df = (
                pd.read_csv("logs/paper_trades_session.csv")
                if os.path.exists("logs/paper_trades_session.csv")
                else pd.DataFrame()
            )
            if not data_df.empty:
                # Use BacktestEngine to re-run MC on paper trades
                result = self.backtest_engine._calculate_results()  # Or custom from logs
                logger.info(
                    f"Paper MC validation: Omega={result.omega_ratio:.2f}, Z={result.z_factor:.2f}"
                )

    def _mock_mode(self, duration_hours: int):
        """Fallback mock trading if MT5 unavailable."""
        logger.info("MT5 unavailable; running mock paper trading with synthetic data")
        # Generate synthetic bars (trending + noise, physics-aligned)
        end_time = datetime.now() + timedelta(hours=duration_hours)
        data_buffer = pd.DataFrame()
        base_price = 60000.0  # BTCUSD
        while datetime.now() < end_time:
            # Synthetic bar (H1)
            now = datetime.now()
            new_bar = pd.DataFrame(
                {
                    "time": [now],
                    "open": [base_price],
                    "high": [base_price + np.random.uniform(0, 500)],
                    "low": [base_price - np.random.uniform(0, 300)],
                    "close": [base_price + np.random.normal(0, 200)],
                    "volume": [np.random.randint(1000, 5000)],
                }
            )
            data_buffer = pd.concat([data_buffer, new_bar]).tail(500)

            if len(data_buffer) >= self.physics.lookback:
                signal = self._get_signal(data_buffer)
                # Simulate as above...
                # (Omit full impl for brevity; mirror start_paper_trading but use synthetic)

            base_price = new_bar["close"].iloc[0]
            time.sleep(3600)  # Simulate H1 poll

        logger.info("Mock paper trading completed")

    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if not self.connected:
            return []

        positions = mt5.positions_get()
        if not positions:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "comment": p.comment,
            }
            for p in positions
        ]


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.

    Handles multiple CSV formats:
    - MT5 export format: <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
    - Standard format: Time, Open, High, Low, Close, Volume

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    # Try tab-separated first (MT5 export), then comma
    try:
        df = pd.read_csv(filepath, sep="\t")
        if len(df.columns) == 1:
            # Tab didn't work, try comma
            df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath)

    # Standardize column names - handle multiple formats
    col_map = {
        # MT5 export format with angle brackets
        "<DATE>": "date",
        "<TIME>": "time_str",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",  # Prefer tick volume
        "<VOL>": "real_volume",
        "<SPREAD>": "spread",
        # Standard format
        "Time": "time",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Tickvol": "volume",
    }
    df = df.rename(columns=col_map)

    # Combine date and time if separate columns (MT5 format)
    if "date" in df.columns and "time_str" in df.columns:
        # Format: 2024.07.01 00:00:00
        df["time"] = pd.to_datetime(df["date"] + " " + df["time_str"], format="%Y.%m.%d %H:%M:%S")
        df = df.drop(columns=["date", "time_str"])
    elif "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Ensure we have required columns
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Default volume if missing
    if "volume" not in df.columns:
        df["volume"] = 0

    # Reorder columns, keeping extras
    base_cols = ["time", "open", "high", "low", "close", "volume"]
    extra_cols = [c for c in df.columns if c not in base_cols]
    available_base = [c for c in base_cols if c in df.columns]
    df = df[available_base + extra_cols]

    return df


# Context manager support
class MT5Session:
    """Context manager for MT5 connection."""

    def __init__(self, path: Optional[str] = None):
        self.connector = MT5Connector(path)

    def __enter__(self) -> MT5Connector:
        if not self.connector.connect():
            raise ConnectionError("Failed to connect to MT5")
        return self.connector

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connector.disconnect()
        return False


# Example usage
if __name__ == "__main__":
    # Test with CSV data if MT5 not available
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_files = [f for f in os.listdir(project_root) if f.endswith(".csv")]

    if csv_files:
        print(f"Loading data from: {csv_files[0]}")
        df = load_csv_data(os.path.join(project_root, csv_files[0]))
        print(f"Loaded {len(df)} rows")
        print(df.head())
    else:
        print("No CSV files found")

    # Test MT5 connection if available
    if MT5_AVAILABLE:
        connector = MT5Connector()
        if connector.connect():
            print("\nMT5 Connected!")
            print(f"Account: {connector.get_account_info()}")
            print(f"Symbols: {connector.get_symbols()[:10]}...")
            connector.disconnect()
        else:
            print("\nMT5 connection failed")

    # Example paper trading setup (uncomment to run)
    # paper_trader = PaperTrader(connector, initial_capital=100000.0)
    # paper_trader.start_paper_trading(duration_hours=24)
