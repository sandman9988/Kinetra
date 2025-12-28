"""
MetaTrader 5 Python Bridge

Interfaces with MT5 terminal to get live symbol specifications,
account info, and market data for accurate friction calculation.

Three modes:
1. Direct mode (Windows): Import MetaTrader5 directly
2. MetaAPI mode (Cloud): Use MetaAPI cloud service (works from anywhere)
3. Bridge mode (WSL2): Connect via socket to Windows MT5
4. Offline mode: Use cached/config specs

Usage:
    # Direct (Windows with MT5 installed)
    bridge = MT5Bridge(mode="direct")
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

    # MetaAPI (Cloud - works from WSL2/Linux)
    bridge = MT5Bridge(mode="metaapi", token="your-token", account_id="your-account")
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

    # Bridge (WSL2 -> Windows socket server)
    bridge = MT5Bridge(mode="bridge", host="localhost", port=5555)
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

Install dependencies:
    Windows: pip install MetaTrader5
    Cloud:   pip install metaapi-cloud-sdk
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, time
import numpy as np

from .market_microstructure import SymbolSpec, AssetClass, SYMBOL_SPECS


class MT5Bridge:
    """
    Bridge to MetaTrader 5 terminal for live market data.

    Supports:
    - Direct mode: MT5 Python package on Windows
    - MetaAPI mode: Cloud API (works from anywhere, including WSL2)
    - Bridge mode: Socket connection to Windows from WSL2
    - Offline mode: Cached/config file specs
    """

    def __init__(
        self,
        mode: str = "auto",
        config_path: Optional[str] = None,
        host: str = "localhost",
        port: int = 5555,
        token: Optional[str] = None,
        account_id: Optional[str] = None
    ):
        """
        Initialize MT5 bridge.

        Args:
            mode: "direct", "metaapi", "bridge", "offline", or "auto"
            config_path: Path to symbol specs JSON (for offline mode)
            host: Bridge server host (for bridge mode)
            port: Bridge server port (for bridge mode)
            token: MetaAPI token (for metaapi mode)
            account_id: MetaAPI account ID (for metaapi mode)
        """
        self.mode = mode
        self.config_path = config_path or str(Path(__file__).parent.parent / "config" / "symbols.json")
        self.host = host
        self.port = port
        self.token = token or os.environ.get("METAAPI_TOKEN")
        self.account_id = account_id or os.environ.get("METAAPI_ACCOUNT_ID")

        self.mt5 = None
        self.metaapi = None
        self.metaapi_connection = None
        self.connected = False
        self.cached_specs: Dict[str, SymbolSpec] = {}

        if mode == "auto":
            self._auto_detect_mode()

    def _auto_detect_mode(self):
        """Auto-detect best available mode."""
        # Try direct MT5 first (Windows only)
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mode = "direct"
            print("Auto-detected: Direct MT5 mode (Windows)")
            return
        except ImportError:
            pass

        # Try MetaAPI if token is available
        if self.token and self.account_id:
            try:
                from metaapi_cloud_sdk import MetaApi
                self.mode = "metaapi"
                print("Auto-detected: MetaAPI cloud mode")
                return
            except ImportError:
                print("MetaAPI token found but SDK not installed. Run: pip install metaapi-cloud-sdk")

        # Try bridge mode (socket to Windows)
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            if result == 0:
                self.mode = "bridge"
                print(f"Auto-detected: Bridge mode ({self.host}:{self.port})")
                return
        except:
            pass

        # Fall back to offline
        self.mode = "offline"
        print("Auto-detected: Offline mode (using cached/built-in specs)")

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            True if connected successfully
        """
        if self.mode == "direct":
            return self._connect_direct()
        elif self.mode == "metaapi":
            return self._connect_metaapi()
        elif self.mode == "bridge":
            return self._connect_bridge()
        else:
            return self._load_offline_config()

    async def _connect_metaapi_async(self) -> bool:
        """Async connection to MetaAPI (internal)."""
        try:
            from metaapi_cloud_sdk import MetaApi

            self.metaapi = MetaApi(token=self.token)
            account = await self.metaapi.metatrader_account_api.get_account(self.account_id)

            # Wait for deployment if needed
            if account.state != 'DEPLOYED':
                print(f"Deploying account {self.account_id}...")
                await account.deploy()
                await account.wait_deployed()

            # Connect to RPC
            self.metaapi_connection = account.get_rpc_connection()
            await self.metaapi_connection.connect()
            await self.metaapi_connection.wait_synchronized()

            # Get account info
            info = await self.metaapi_connection.get_account_information()
            print(f"Connected to MetaAPI: {info['broker']} - {info['server']}")
            print(f"Account: {info['login']} ({info['name']})")
            print(f"Balance: {info['balance']} {info['currency']}")

            self.connected = True
            return True

        except Exception as e:
            print(f"MetaAPI connection failed: {e}")
            return False

    def _connect_metaapi(self) -> bool:
        """Connect to MetaAPI cloud service."""
        import asyncio

        try:
            # Run async connection
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._connect_metaapi_async())
                    return future.result()
            else:
                return loop.run_until_complete(self._connect_metaapi_async())
        except Exception as e:
            print(f"MetaAPI connection error: {e}")
            print("Falling back to offline mode...")
            self.mode = "offline"
            return self._load_offline_config()

    def _connect_direct(self) -> bool:
        """Connect directly to MT5 (Windows only)."""
        if self.mt5 is None:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
            except ImportError:
                print("MetaTrader5 package not installed. Run: pip install MetaTrader5")
                return False

        if not self.mt5.initialize():
            print(f"MT5 initialize failed: {self.mt5.last_error()}")
            return False

        self.connected = True
        account_info = self.mt5.account_info()
        if account_info:
            print(f"Connected to MT5: {account_info.server}")
            print(f"Account: {account_info.login} ({account_info.name})")
            print(f"Balance: {account_info.balance} {account_info.currency}")

        # Preload all symbol specs
        self._load_all_symbols_direct()

        return True

    def _load_all_symbols_direct(self, group: str = None):
        """Load all symbol specs from MT5 using symbols_get."""
        if not self.mt5 or not self.connected:
            return

        # Get all symbols (or filter by group)
        if group:
            symbols = self.mt5.symbols_get(group=group)
        else:
            symbols = self.mt5.symbols_get()

        if symbols is None:
            print(f"Failed to get symbols: {self.mt5.last_error()}")
            return

        print(f"Loading {len(symbols)} symbol specifications...")

        for info in symbols:
            try:
                spec = self._mt5_info_to_spec(info)
                self.cached_specs[info.name] = spec
            except Exception as e:
                # Skip symbols we can't parse
                pass

        print(f"Loaded {len(self.cached_specs)} symbol specs")

    def _mt5_info_to_spec(self, info) -> SymbolSpec:
        """
        Convert MT5 SymbolInfo object to SymbolSpec.

        MT5 symbol_info returns all these fields:
        - digits, point, spread, spread_float
        - trade_contract_size, trade_tick_value, trade_tick_size
        - volume_min, volume_max, volume_step
        - swap_long, swap_short, swap_mode, swap_rollover3days
        - margin_initial, margin_maintenance, margin_hedged
        - bid, ask, trade_mode, path
        """
        symbol = info.name

        # Detect asset class from path or calc mode
        path = getattr(info, 'path', '').lower()
        if 'crypto' in path or 'btc' in symbol.lower() or 'eth' in symbol.lower():
            asset_class = AssetClass.CRYPTO
        elif 'index' in path or 'indices' in path:
            asset_class = AssetClass.INDEX
        elif 'metal' in path or 'xau' in symbol.lower() or 'xag' in symbol.lower():
            asset_class = AssetClass.COMMODITY
        elif 'stock' in path or 'shares' in path:
            asset_class = AssetClass.STOCK
        else:
            asset_class = AssetClass.FOREX

        # Current spread (in points)
        current_spread = getattr(info, 'spread', 0)

        # Real-time spread from bid/ask
        bid = getattr(info, 'bid', 0)
        ask = getattr(info, 'ask', 0)
        point = getattr(info, 'point', 0.00001)
        if bid > 0 and ask > 0 and point > 0:
            live_spread = (ask - bid) / point
            current_spread = max(current_spread, live_spread)

        # Margin rate from tick value
        tick_value = getattr(info, 'trade_tick_value', 0)
        contract_size = getattr(info, 'trade_contract_size', 100000)
        if tick_value > 0:
            margin_rate = (tick_value * 10) / contract_size  # Approximate
        else:
            margin_rate = 0.01  # Default 1% margin (100:1 leverage)

        return SymbolSpec(
            symbol=symbol,
            asset_class=asset_class,
            digits=getattr(info, 'digits', 5),
            point=point,
            contract_size=contract_size,
            volume_min=getattr(info, 'volume_min', 0.01),
            volume_max=getattr(info, 'volume_max', 100.0),
            volume_step=getattr(info, 'volume_step', 0.01),
            margin_initial=margin_rate,
            spread_typical=current_spread,
            spread_min=max(1, current_spread * 0.5),  # Min spread estimate
            spread_max=current_spread * 10,  # Max spread (during news/rollover)
            commission_per_lot=0.0,  # Commission not in symbol_info, set per account
            swap_long=getattr(info, 'swap_long', 0.0),
            swap_short=getattr(info, 'swap_short', 0.0),
        )

    def get_live_spread(self, symbol: str) -> Optional[float]:
        """
        Get real-time spread in points.

        Args:
            symbol: Symbol name

        Returns:
            Current spread in points or None
        """
        if self.mode == "direct" and self.mt5:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick:
                spec = self.get_symbol_spec(symbol)
                point = spec.point if spec else 0.00001
                return (tick.ask - tick.bid) / point
        return None

    def get_swap_cost(self, symbol: str, lots: float, is_long: bool, days: int = 1) -> float:
        """
        Calculate swap cost for holding a position.

        Args:
            symbol: Symbol name
            lots: Position size in lots
            is_long: Long or short position
            days: Number of days held

        Returns:
            Swap cost in account currency
        """
        spec = self.get_symbol_spec(symbol)
        if not spec:
            return 0.0

        swap_points = spec.swap_long if is_long else spec.swap_short
        swap_price = swap_points * spec.point

        # Swap in account currency
        tick_value = spec.contract_size * spec.point  # Approximate tick value
        swap_cost = swap_price * lots * tick_value * days / spec.point

        return swap_cost

    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """
        Get order book (Market Depth) for liquidity estimation.

        Requires: mt5.market_book_add(symbol) first to subscribe.

        Args:
            symbol: Symbol name

        Returns:
            Dict with bids, asks, and liquidity metrics
        """
        if self.mode != "direct" or not self.mt5:
            return None

        # Subscribe to market depth
        if not self.mt5.market_book_add(symbol):
            return None

        book = self.mt5.market_book_get(symbol)
        if not book:
            return None

        bids = []
        asks = []

        for item in book:
            entry = {
                'price': item.price,
                'volume': item.volume,
                'volume_real': item.volume_real if hasattr(item, 'volume_real') else item.volume
            }
            if item.type == 1:  # BOOK_TYPE_SELL
                asks.append(entry)
            else:  # BOOK_TYPE_BUY
                bids.append(entry)

        # Calculate liquidity metrics
        total_bid_volume = sum(b['volume_real'] for b in bids) if bids else 0
        total_ask_volume = sum(a['volume_real'] for a in asks) if asks else 0

        # Imbalance ratio (> 1 = more buying pressure)
        imbalance = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0

        # Depth (how many levels)
        depth = len(bids) + len(asks)

        return {
            'bids': bids,
            'asks': asks,
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'imbalance_ratio': imbalance,
            'depth_levels': depth,
        }

    def calculate_margin(self, symbol: str, lots: float, is_buy: bool) -> Optional[float]:
        """
        Calculate margin required for a trade.

        Args:
            symbol: Symbol name
            lots: Position size
            is_buy: True for buy, False for sell

        Returns:
            Required margin in account currency
        """
        if self.mode != "direct" or not self.mt5:
            return None

        action = self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL

        tick = self.mt5.symbol_info_tick(symbol)
        if not tick:
            return None

        price = tick.ask if is_buy else tick.bid
        margin = self.mt5.order_calc_margin(action, symbol, lots, price)

        return margin

    def calculate_profit(
        self,
        symbol: str,
        lots: float,
        is_buy: bool,
        open_price: float,
        close_price: float
    ) -> Optional[float]:
        """
        Calculate profit for a trade.

        Args:
            symbol: Symbol name
            lots: Position size
            is_buy: True for buy, False for sell
            open_price: Entry price
            close_price: Exit price

        Returns:
            Profit/loss in account currency
        """
        if self.mode != "direct" or not self.mt5:
            return None

        action = self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL
        profit = self.mt5.order_calc_profit(action, symbol, lots, open_price, close_price)

        return profit

    def _connect_bridge(self) -> bool:
        """Connect via socket bridge to Windows MT5."""
        # This would connect to a bridge server running on Windows
        # For now, return False and use offline mode
        print(f"Bridge mode: Attempting connection to {self.host}:{self.port}")
        print("Bridge server not implemented yet - falling back to offline mode")
        self.mode = "offline"
        return self._load_offline_config()

    def _load_offline_config(self) -> bool:
        """Load symbol specs from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    for symbol, spec_dict in data.items():
                        self.cached_specs[symbol] = self._dict_to_spec(spec_dict)
                print(f"Loaded {len(self.cached_specs)} symbol specs from {self.config_path}")
                self.connected = True
                return True
            except Exception as e:
                print(f"Error loading config: {e}")

        # Fall back to built-in specs
        self.cached_specs = SYMBOL_SPECS.copy()
        print(f"Using built-in specs for {len(self.cached_specs)} symbols")
        self.connected = True
        return True

    def disconnect(self):
        """Disconnect from MT5."""
        if self.mode == "direct" and self.mt5:
            self.mt5.shutdown()
        self.connected = False

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """
        Get symbol specification.

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            SymbolSpec or None if not found
        """
        symbol = symbol.upper()

        # Check cache first
        if symbol in self.cached_specs:
            return self.cached_specs[symbol]

        # Try to get from MT5
        if self.mode == "direct" and self.mt5 and self.connected:
            spec = self._get_mt5_symbol_spec(symbol)
            if spec:
                self.cached_specs[symbol] = spec
                return spec

        # Fall back to built-in or generate generic
        if symbol in SYMBOL_SPECS:
            return SYMBOL_SPECS[symbol]

        return None

    def _get_mt5_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """Get symbol spec directly from MT5 terminal."""
        if not self.mt5:
            return None

        info = self.mt5.symbol_info(symbol)
        if info is None:
            return None

        # Determine asset class from symbol properties
        if info.trade_calc_mode == 0:  # Forex mode
            asset_class = AssetClass.FOREX
        elif "BTC" in symbol or "ETH" in symbol or "CRYPTO" in info.path:
            asset_class = AssetClass.CRYPTO
        elif info.trade_calc_mode == 4:  # CFD indices
            asset_class = AssetClass.INDEX
        else:
            asset_class = AssetClass.COMMODITY

        # Get current tick for spread
        tick = self.mt5.symbol_info_tick(symbol)
        current_spread = 0
        if tick:
            current_spread = (tick.ask - tick.bid) / info.point

        spec = SymbolSpec(
            symbol=symbol,
            asset_class=asset_class,
            digits=info.digits,
            point=info.point,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            margin_initial=1.0 / info.trade_tick_value if info.trade_tick_value > 0 else 0.01,
            spread_typical=current_spread,
            spread_min=info.spread if hasattr(info, 'spread') else current_spread * 0.5,
            spread_max=current_spread * 5,  # Estimate max as 5x current
            commission_per_lot=0.0,  # Would need to get from account
            swap_long=info.swap_long,
            swap_short=info.swap_short,
        )

        return spec

    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick data (bid, ask, spread, time).

        Args:
            symbol: Symbol name

        Returns:
            Dict with tick data or None
        """
        if self.mode != "direct" or not self.mt5:
            return None

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        spec = self.get_symbol_spec(symbol)
        point = spec.point if spec else 0.00001

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': (tick.ask - tick.bid) / point,
            'spread_price': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time),
            'volume': tick.volume,
            'last': tick.last,
        }

    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if self.mode != "direct" or not self.mt5:
            return None

        info = self.mt5.account_info()
        if info is None:
            return None

        return {
            'login': info.login,
            'server': info.server,
            'name': info.name,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'margin_free': info.margin_free,
            'margin_level': info.margin_level,
            'currency': info.currency,
            'leverage': info.leverage,
            'profit': info.profit,
        }

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000
    ) -> Optional[np.ndarray]:
        """
        Get historical OHLCV rates from MT5.

        Args:
            symbol: Symbol name
            timeframe: Timeframe string (M1, M5, H1, D1, etc.)
            count: Number of bars to get

        Returns:
            Numpy structured array with OHLCV data
        """
        if self.mode != "direct" or not self.mt5:
            return None

        # Map timeframe string to MT5 constant
        tf_map = {
            'M1': self.mt5.TIMEFRAME_M1,
            'M5': self.mt5.TIMEFRAME_M5,
            'M15': self.mt5.TIMEFRAME_M15,
            'M30': self.mt5.TIMEFRAME_M30,
            'H1': self.mt5.TIMEFRAME_H1,
            'H4': self.mt5.TIMEFRAME_H4,
            'D1': self.mt5.TIMEFRAME_D1,
            'W1': self.mt5.TIMEFRAME_W1,
            'MN1': self.mt5.TIMEFRAME_MN1,
        }

        tf = tf_map.get(timeframe.upper(), self.mt5.TIMEFRAME_H1)
        rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, count)

        return rates

    def save_specs_to_config(self, symbols: List[str] = None):
        """
        Save current symbol specs to config file.

        Args:
            symbols: List of symbols to save (None = all cached)
        """
        if symbols is None:
            symbols = list(self.cached_specs.keys())

        specs_dict = {}
        for symbol in symbols:
            spec = self.get_symbol_spec(symbol)
            if spec:
                specs_dict[symbol] = self._spec_to_dict(spec)

        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(specs_dict, f, indent=2, default=str)

        print(f"Saved {len(specs_dict)} symbol specs to {self.config_path}")

    def _spec_to_dict(self, spec: SymbolSpec) -> Dict:
        """Convert SymbolSpec to JSON-serializable dict."""
        d = {
            'symbol': spec.symbol,
            'asset_class': spec.asset_class.value,
            'digits': spec.digits,
            'point': spec.point,
            'contract_size': spec.contract_size,
            'volume_min': spec.volume_min,
            'volume_max': spec.volume_max,
            'volume_step': spec.volume_step,
            'margin_initial': spec.margin_initial,
            'spread_typical': spec.spread_typical,
            'spread_min': spec.spread_min,
            'spread_max': spec.spread_max,
            'commission_per_lot': spec.commission_per_lot,
            'swap_long': spec.swap_long,
            'swap_short': spec.swap_short,
        }
        return d

    def _dict_to_spec(self, d: Dict) -> SymbolSpec:
        """Convert dict to SymbolSpec."""
        asset_class = AssetClass(d.get('asset_class', 'forex'))
        return SymbolSpec(
            symbol=d['symbol'],
            asset_class=asset_class,
            digits=d.get('digits', 5),
            point=d.get('point'),
            contract_size=d.get('contract_size', 100000),
            volume_min=d.get('volume_min', 0.01),
            volume_max=d.get('volume_max', 100),
            volume_step=d.get('volume_step', 0.01),
            margin_initial=d.get('margin_initial', 0.01),
            spread_typical=d.get('spread_typical', 2.0),
            spread_min=d.get('spread_min', 1.0),
            spread_max=d.get('spread_max', 20.0),
            commission_per_lot=d.get('commission_per_lot', 0.0),
            swap_long=d.get('swap_long', 0.0),
            swap_short=d.get('swap_short', 0.0),
        )


def create_bridge_server_script() -> str:
    """
    Generate a Python script to run on Windows as MT5 bridge server.

    This script runs on Windows, connects to MT5, and serves data
    over a socket to WSL2.

    Returns:
        Python script as string
    """
    script = '''#!/usr/bin/env python3
"""
MT5 Bridge Server for WSL2

Run this script on Windows with MT5 terminal open.
It serves symbol specs and tick data to WSL2 clients.

Usage:
    python mt5_bridge_server.py [--port 5555]
"""

import socket
import json
import threading
from datetime import datetime
import MetaTrader5 as mt5


class MT5BridgeServer:
    def __init__(self, port: int = 5555):
        self.port = port
        self.running = False
        self.mt5_connected = False

    def connect_mt5(self) -> bool:
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        self.mt5_connected = True
        info = mt5.account_info()
        print(f"Connected: {info.server} - {info.login}")
        return True

    def get_symbol_spec(self, symbol: str) -> dict:
        info = mt5.symbol_info(symbol)
        if not info:
            return {"error": f"Symbol {symbol} not found"}

        tick = mt5.symbol_info_tick(symbol)
        spread = (tick.ask - tick.bid) / info.point if tick else 0

        return {
            "symbol": symbol,
            "digits": info.digits,
            "point": info.point,
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "spread_current": spread,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
            "bid": tick.bid if tick else 0,
            "ask": tick.ask if tick else 0,
        }

    def handle_client(self, conn, addr):
        print(f"Client connected: {addr}")
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break

                try:
                    request = json.loads(data.decode())
                    cmd = request.get("cmd", "")

                    if cmd == "symbol_info":
                        response = self.get_symbol_spec(request.get("symbol", ""))
                    elif cmd == "tick":
                        tick = mt5.symbol_info_tick(request.get("symbol", ""))
                        response = {"bid": tick.bid, "ask": tick.ask} if tick else {"error": "No tick"}
                    elif cmd == "account":
                        info = mt5.account_info()
                        response = {"balance": info.balance, "equity": info.equity} if info else {"error": "No account"}
                    else:
                        response = {"error": f"Unknown command: {cmd}"}

                    conn.sendall(json.dumps(response).encode())
                except json.JSONDecodeError:
                    conn.sendall(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            conn.close()
            print(f"Client disconnected: {addr}")

    def run(self):
        if not self.connect_mt5():
            return

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.port))
        server.listen(5)

        self.running = True
        print(f"MT5 Bridge Server running on port {self.port}")
        print("Press Ctrl+C to stop")

        try:
            while self.running:
                conn, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.running = False
            server.close()
            mt5.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    server = MT5BridgeServer(port=args.port)
    server.run()
'''
    return script


def save_bridge_server_script(path: str = "mt5_bridge_server.py"):
    """Save the bridge server script to a file."""
    script = create_bridge_server_script()
    with open(path, 'w') as f:
        f.write(script)
    print(f"Bridge server script saved to: {path}")
    print("Copy this file to Windows and run with: python mt5_bridge_server.py")
