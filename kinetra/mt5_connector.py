"""
MetaTrader 5 Data Connector for Kinetra

Provides real-time and historical data from MT5 terminal.
Requires: MetaTrader 5 running via Wine, MetaTrader5 Python package
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging

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
        self.connected = False
        self._check_availability()

    def _check_availability(self):
        """Check if MT5 package is available."""
        if not MT5_AVAILABLE:
            logger.warning(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
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

    Fallback when MT5 is not available.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with OHLCV data
    """
    df = pd.read_csv(filepath)

    # Standardize column names
    col_map = {
        "Time": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Tickvol": "volume",
    }
    df = df.rename(columns=col_map)

    # Parse datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

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
