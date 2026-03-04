"""
MetaTrader 5 Python Bridge

Interfaces with MT5 terminal to get live symbol specifications,
account info, and market data for accurate friction calculation.

Connection Priority (auto mode):
1. MetaAPI mode (DEFAULT): Cloud API - works from anywhere (Linux/Mac/Windows/WSL2)
2. Direct mode (FALLBACK): Import MetaTrader5 directly (Windows only)
3. Bridge mode (FALLBACK): Connect via socket to Windows MT5 from WSL2
4. Offline mode (FINAL FALLBACK): Use cached/config specs

Usage:
    # Auto mode (MetaAPI default, MT5 direct fallback)
    bridge = MT5Bridge()  # Uses METAAPI_TOKEN/METAAPI_ACCOUNT_ID env vars
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

    # Explicit MetaAPI (Cloud - works from anywhere)
    bridge = MT5Bridge(mode="metaapi", token="your-token", account_id="your-account")
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

    # Explicit Direct (Windows with MT5 installed)
    bridge = MT5Bridge(mode="direct")
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

    # Explicit Bridge (WSL2 -> Windows socket server)
    bridge = MT5Bridge(mode="bridge", host="localhost", port=5555)
    bridge.connect()
    spec = bridge.get_symbol_spec("EURUSD")

Install dependencies:
    Cloud (recommended): pip install metaapi-cloud-sdk
    Windows direct:      pip install MetaTrader5

Environment variables:
    METAAPI_TOKEN=your-metaapi-token
    METAAPI_ACCOUNT_ID=your-account-uuid
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .market_microstructure import SYMBOL_SPECS, AssetClass, SymbolSpec


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
        """
        Auto-detect best available mode.

        Priority order:
        1. MetaAPI (DEFAULT) - Cloud-based, works everywhere
        2. Direct MT5 (FALLBACK) - Windows only, requires MT5 terminal
        3. Bridge (FALLBACK) - Socket to Windows from WSL2
        4. Offline (FINAL FALLBACK) - Cached/built-in specs
        """
        # PRIORITY 1: Try MetaAPI first (DEFAULT - works from anywhere)
        if self.token and self.account_id:
            try:
                import importlib.util
                if importlib.util.find_spec("metaapi_cloud_sdk") is not None:
                    self.mode = "metaapi"
                    print("✓ Auto-detected: MetaAPI cloud mode (DEFAULT)")
                    print(f"  Account ID: {self.account_id[:8]}...{self.account_id[-4:]}")
                    return
                else:
                    print("⚠ MetaAPI credentials found but SDK not installed.")
                    print("  Run: pip install metaapi-cloud-sdk")
            except Exception as e:
                print(f"⚠ MetaAPI detection error: {e}")
        elif self.token or self.account_id:
            missing = "METAAPI_ACCOUNT_ID" if self.token else "METAAPI_TOKEN"
            print(f"⚠ Partial MetaAPI config: missing {missing}")

        # PRIORITY 2: Try direct MT5 (FALLBACK - Windows only)
        try:
            import importlib.util
            if importlib.util.find_spec("MetaTrader5") is not None:
                import MetaTrader5 as mt5
                self.mt5 = mt5
                self.mode = "direct"
                print("✓ Auto-detected: Direct MT5 mode (Windows fallback)")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠ MT5 direct detection error: {e}")

        # PRIORITY 3: Try bridge mode (FALLBACK - socket to Windows from WSL2)
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            if result == 0:
                self.mode = "bridge"
                print(f"✓ Auto-detected: Bridge mode ({self.host}:{self.port})")
                return
        except Exception:
            pass

        # PRIORITY 4: Fall back to offline (FINAL FALLBACK)
        self.mode = "offline"
        print("✓ Auto-detected: Offline mode (using cached/built-in specs)")
        print("  To enable live data, set METAAPI_TOKEN and METAAPI_ACCOUNT_ID")

    def connect(self) -> bool:
        """
        Connect to MT5 terminal with automatic fallback.

        Connection priority:
        1. MetaAPI (default) -> 2. Direct MT5 (fallback) -> 3. Bridge -> 4. Offline

        Returns:
            True if connected successfully
        """
        original_mode = self.mode

        if self.mode == "metaapi":
            if self._connect_metaapi():
                return True
            print("⚠ MetaAPI connection failed, trying fallback...")
            # Fallback to direct MT5
            self.mode = "direct"

        if self.mode == "direct":
            if self._connect_direct():
                return True
            if original_mode == "direct":
                print("⚠ Direct MT5 connection failed, trying fallback...")
            # Fallback to bridge
            self.mode = "bridge"

        if self.mode == "bridge":
            if self._connect_bridge():
                return True
            if original_mode == "bridge":
                print("⚠ Bridge connection failed, trying fallback...")
            # Fallback to offline
            self.mode = "offline"

        # Final fallback: offline mode
        return self._load_offline_config()

    async def _connect_metaapi_async(self) -> bool:
        """Async connection to MetaAPI (internal)."""
        try:
            from metaapi_cloud_sdk import MetaApi

            print("Connecting to MetaAPI cloud service...")
            self.metaapi = MetaApi(token=self.token)

            # Store account reference for later use
            self.metaapi_account = await self.metaapi.metatrader_account_api.get_account(self.account_id)

            # Wait for deployment if needed
            if self.metaapi_account.state != 'DEPLOYED':
                print(f"  Deploying account {self.account_id[:8]}...")
                await self.metaapi_account.deploy()
                await self.metaapi_account.wait_deployed()

            # Connect to RPC for real-time data
            self.metaapi_connection = self.metaapi_account.get_rpc_connection()
            await self.metaapi_connection.connect()
            await self.metaapi_connection.wait_synchronized()

            # Get account info
            info = await self.metaapi_connection.get_account_information()
            print(f"✓ Connected to MetaAPI: {info['broker']} - {info['server']}")
            print(f"  Account: {info['login']} ({info['name']})")
            print(f"  Balance: {info['balance']} {info['currency']}")
            print(f"  Leverage: 1:{info.get('leverage', 'N/A')}")

            # Cache account info
            self._metaapi_account_info = info
            self.connected = True
            return True

        except Exception as e:
            print(f"✗ MetaAPI connection failed: {e}")
            return False

    def _connect_metaapi(self) -> bool:
        """Connect to MetaAPI cloud service."""
        import asyncio

        try:
            # Check if SDK is available
            import importlib.util
            if importlib.util.find_spec("metaapi_cloud_sdk") is None:
                print("✗ MetaAPI SDK not installed. Run: pip install metaapi-cloud-sdk")
                return False

            # Run async connection with proper event loop handling
            try:
                asyncio.get_running_loop()
                # We're in an async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._connect_metaapi_async())
                    return future.result(timeout=60)
            except RuntimeError:
                # No running loop - create one
                return asyncio.run(self._connect_metaapi_async())

        except asyncio.TimeoutError:
            print("✗ MetaAPI connection timed out")
            return False
        except Exception as e:
            print(f"✗ MetaAPI connection error: {e}")
            return False

    async def _get_metaapi_symbol_spec_async(self, symbol: str) -> Optional[Dict]:
        """Get symbol specification via MetaAPI (async)."""
        if not self.metaapi_connection:
            return None
        try:
            spec = await self.metaapi_connection.get_symbol_specification(symbol)
            price = await self.metaapi_connection.get_symbol_price(symbol)
            return {**spec, **price} if spec else None
        except Exception as e:
            print(f"Error getting MetaAPI spec for {symbol}: {e}")
            return None

    async def _get_metaapi_candles_async(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """Get historical candles via MetaAPI (async)."""
        if not self.metaapi_account:
            return None
        try:
            # MetaAPI timeframe format: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
            tf_map = {
                'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
                'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w'
            }
            mt_tf = tf_map.get(timeframe.upper(), timeframe.lower())

            candles = await self.metaapi_account.get_historical_candles(
                symbol=symbol,
                timeframe=mt_tf,
                start_time=start_time,
                limit=limit
            )
            return candles
        except Exception as e:
            print(f"Error getting MetaAPI candles for {symbol}: {e}")
            return None

    def get_metaapi_symbol_spec(self, symbol: str) -> Optional[Dict]:
        """Get symbol specification via MetaAPI (sync wrapper)."""
        if self.mode != "metaapi" or not self.metaapi_connection:
            return None

        import asyncio
        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._get_metaapi_symbol_spec_async(symbol)
                    )
                    return future.result(timeout=30)
            except RuntimeError:
                return asyncio.run(self._get_metaapi_symbol_spec_async(symbol))
        except Exception as e:
            print(f"Error in get_metaapi_symbol_spec: {e}")
            return None

    def get_metaapi_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """Get historical candles via MetaAPI (sync wrapper)."""
        if self.mode != "metaapi" or not self.metaapi_account:
            return None

        import asyncio
        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._get_metaapi_candles_async(symbol, timeframe, start_time, limit)
                    )
                    return future.result(timeout=60)
            except RuntimeError:
                return asyncio.run(
                    self._get_metaapi_candles_async(symbol, timeframe, start_time, limit)
                )
        except Exception as e:
            print(f"Error in get_metaapi_candles: {e}")
            return None

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
            except Exception:
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
        elif 'index' in path or 'indices' in path or any(x in symbol.upper() for x in ['US500', 'NAS100', 'DJ30', 'SPX', 'NDX']):
            asset_class = AssetClass.INDICES
        elif 'metal' in path or any(x in symbol.upper() for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
            asset_class = AssetClass.METALS
        elif 'energy' in path or any(x in symbol.upper() for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
            asset_class = AssetClass.ENERGY
        elif 'etf' in path or 'etf' in symbol.lower():
            asset_class = AssetClass.ETFS
        elif 'stock' in path or 'shares' in path or 'equity' in path:
            asset_class = AssetClass.SHARES
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
        # MetaAPI mode (default)
        if self.mode == "metaapi" and self.connected:
            data = self.get_metaapi_symbol_spec(symbol)
            if data and "bid" in data and "ask" in data:
                spec = self.get_symbol_spec(symbol)
                point = data.get('point', spec.point if spec else 0.00001)
                if point > 0:
                    return (data["ask"] - data["bid"]) / point

        # Direct MT5 mode
        if self.mode == "direct" and self.mt5:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick:
                spec = self.get_symbol_spec(symbol)
                point = spec.point if spec else 0.00001
                return (tick.ask - tick.bid) / point

        # Bridge mode
        elif self.mode == "bridge":
            data = self._bridge_request("tick", symbol=symbol)
            if data and "bid" in data and "ask" in data:
                spec = self.get_symbol_spec(symbol)
                point = spec.point if spec else 0.00001
                return (data["ask"] - data["bid"]) / point

        return None

    def get_live_friction(
        self,
        symbol: str,
        lots: float = 0.1,
        is_long: bool = True,
        holding_hours: float = 1.0
    ) -> Optional[Dict]:
        """
        Get REAL-TIME friction data from MT5.

        This is the key function for physics-based cost modeling.
        Returns actual spread, estimated slippage, swap, commission.

        Args:
            symbol: Symbol name
            lots: Position size in lots
            is_long: Long or short position
            holding_hours: Expected holding period in hours

        Returns:
            Dict with real friction components
        """
        spec = self.get_symbol_spec(symbol)
        if not spec:
            return None

        # Get live spread
        live_spread_points = self.get_live_spread(symbol)
        if live_spread_points is None:
            live_spread_points = spec.spread_typical

        # Convert to price
        spread_price = live_spread_points * spec.point

        # Get current price for percentage calculation
        tick_data = self.get_current_tick(symbol)
        if tick_data:
            current_price = (tick_data['bid'] + tick_data['ask']) / 2
        else:
            current_price = 1.0  # Fallback

        # Calculate costs as percentages
        spread_pct = (spread_price / current_price) * 100 * 2  # Entry + exit

        # Commission
        position_value = current_price * spec.contract_size * lots
        commission_total = spec.commission_per_lot * lots * 2  # Both sides
        commission_pct = (commission_total / position_value) * 100 if position_value > 0 else 0

        # Swap (if holding)
        holding_days = holding_hours / 24
        swap_points = spec.swap_long if is_long else spec.swap_short
        swap_price = swap_points * spec.point
        swap_pct = (swap_price / current_price) * 100 * holding_days

        # Slippage estimate (based on spread width)
        # Wide spread = likely higher slippage
        spread_stress = live_spread_points / spec.spread_typical if spec.spread_typical > 0 else 1.0
        slippage_base = 0.01  # 0.01% base slippage
        slippage_pct = slippage_base * spread_stress * 2  # Entry + exit

        # Total
        total_friction = spread_pct + commission_pct + slippage_pct + abs(swap_pct)

        return {
            'symbol': symbol,
            'spread_points': live_spread_points,
            'spread_price': spread_price,
            'spread_pct': spread_pct,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_pct,
            'swap_pct': swap_pct,
            'total_friction_pct': total_friction,
            'spread_stress': spread_stress,
            'current_price': current_price,
            'is_live': self.mode in ["direct", "bridge", "metaapi"],
            'data_source': self.mode,
        }

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
        import socket

        print(f"Bridge mode: Connecting to {self.host}:{self.port}...")

        try:
            # Test connection
            self.bridge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.bridge_socket.settimeout(5)
            self.bridge_socket.connect((self.host, self.port))

            # Test with account request
            self.bridge_socket.sendall(json.dumps({"cmd": "account"}).encode())
            response = self.bridge_socket.recv(4096)
            data = json.loads(response.decode())

            if "error" in data:
                print(f"Bridge error: {data['error']}")
                self.bridge_socket.close()
                self.mode = "offline"
                return self._load_offline_config()

            print(f"Connected via bridge: Balance={data.get('balance')}, Equity={data.get('equity')}")
            self.connected = True
            return True

        except socket.timeout:
            print("Bridge connection timed out - is the server running on Windows?")
            print("Run: python mt5_bridge_server.py on Windows")
            self.mode = "offline"
            return self._load_offline_config()
        except ConnectionRefusedError:
            print(f"Bridge connection refused - server not running on {self.host}:{self.port}")
            print("Run: python mt5_bridge_server.py on Windows")
            self.mode = "offline"
            return self._load_offline_config()
        except Exception as e:
            print(f"Bridge connection error: {e}")
            self.mode = "offline"
            return self._load_offline_config()

    def _bridge_request(self, cmd: str, **kwargs) -> Optional[Dict]:
        """Send request to bridge server."""
        if not hasattr(self, 'bridge_socket') or self.bridge_socket is None:
            return None

        try:
            request = {"cmd": cmd, **kwargs}
            self.bridge_socket.sendall(json.dumps(request).encode())
            response = self.bridge_socket.recv(8192)
            return json.loads(response.decode())
        except Exception as e:
            print(f"Bridge request error: {e}")
            return None

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
        """Disconnect from MT5/MetaAPI."""
        import asyncio

        # MetaAPI disconnect
        if self.mode == "metaapi":
            if self.metaapi_connection:
                try:
                    try:
                        asyncio.get_running_loop()
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            executor.submit(asyncio.run, self.metaapi_connection.close())
                    except RuntimeError:
                        asyncio.run(self.metaapi_connection.close())
                except Exception:
                    pass
            if self.metaapi:
                try:
                    try:
                        asyncio.get_running_loop()
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            executor.submit(asyncio.run, self.metaapi.close())
                    except RuntimeError:
                        asyncio.run(self.metaapi.close())
                except Exception:
                    pass
            self.metaapi = None
            self.metaapi_connection = None
            self.metaapi_account = None

        # Direct MT5 disconnect
        if self.mode == "direct" and self.mt5:
            try:
                self.mt5.shutdown()
            except Exception:
                pass

        # Bridge disconnect
        if self.mode == "bridge" and hasattr(self, 'bridge_socket'):
            try:
                self.bridge_socket.close()
            except Exception:
                pass

        self.connected = False
        print(f"Disconnected from {self.mode} mode")

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

        # Try MetaAPI (priority for cloud mode)
        if self.mode == "metaapi" and self.connected:
            data = self.get_metaapi_symbol_spec(symbol)
            if data:
                spec = self._metaapi_data_to_spec(data)
                if spec:
                    self.cached_specs[symbol] = spec
                    return spec

        # Try to get from MT5 (direct mode)
        if self.mode == "direct" and self.mt5 and self.connected:
            spec = self._get_mt5_symbol_spec(symbol)
            if spec:
                self.cached_specs[symbol] = spec
                return spec

        # Try bridge mode
        if self.mode == "bridge" and self.connected:
            data = self._bridge_request("symbol_info", symbol=symbol)
            if data and "error" not in data:
                spec = self._bridge_data_to_spec(data)
                if spec:
                    self.cached_specs[symbol] = spec
                    return spec

        # Fall back to built-in or generate generic
        if symbol in SYMBOL_SPECS:
            return SYMBOL_SPECS[symbol]

        return None

    def _metaapi_data_to_spec(self, data: Dict) -> Optional[SymbolSpec]:
        """Convert MetaAPI response to SymbolSpec."""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            description = data.get('description', '')

            # Detect asset class from symbol/description
            if 'btc' in symbol.lower() or 'eth' in symbol.lower() or 'crypto' in description.lower():
                asset_class = AssetClass.CRYPTO
            elif any(x in symbol.upper() for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
                asset_class = AssetClass.METALS
            elif any(x in symbol.upper() for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
                asset_class = AssetClass.ENERGY
            elif any(x in symbol.upper() for x in ['US500', 'US30', 'NAS100', 'SPX', 'NDX', 'DAX', 'FTSE']):
                asset_class = AssetClass.INDICES
            elif 'etf' in symbol.lower() or 'etf' in description.lower():
                asset_class = AssetClass.ETFS
            else:
                asset_class = AssetClass.FOREX

            # Get prices for spread calculation
            bid = data.get('bid', 0)
            ask = data.get('ask', 0)
            point = data.get('point', 0.00001)

            if bid > 0 and ask > 0 and point > 0:
                current_spread = (ask - bid) / point
            else:
                current_spread = data.get('spread', 2.0)

            return SymbolSpec(
                symbol=symbol,
                asset_class=asset_class,
                digits=data.get('digits', 5),
                point=point,
                contract_size=data.get('contractSize', 100000),
                volume_min=data.get('minVolume', 0.01),
                volume_max=data.get('maxVolume', 100),
                volume_step=data.get('volumeStep', 0.01),
                margin_initial=data.get('marginInitial', 0.01) or 0.01,
                spread_typical=current_spread,
                spread_min=max(1, current_spread * 0.5),
                spread_max=current_spread * 10,
                commission_per_lot=0.0,
                swap_long=data.get('swapLong', 0.0),
                swap_short=data.get('swapShort', 0.0),
            )
        except Exception as e:
            print(f"Error converting MetaAPI data to spec: {e}")
            return None

    def _bridge_data_to_spec(self, data: Dict) -> Optional[SymbolSpec]:
        """Convert bridge response to SymbolSpec."""
        try:
            symbol = data.get('symbol', 'UNKNOWN')

            # Detect asset class
            if 'btc' in symbol.lower() or 'eth' in symbol.lower():
                asset_class = AssetClass.CRYPTO
            elif any(x in symbol.upper() for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
                asset_class = AssetClass.METALS
            elif any(x in symbol.upper() for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
                asset_class = AssetClass.ENERGY
            elif symbol.endswith('500') or symbol.endswith('30') or 'index' in symbol.lower():
                asset_class = AssetClass.INDICES
            elif 'etf' in symbol.lower():
                asset_class = AssetClass.ETFS
            else:
                asset_class = AssetClass.FOREX

            return SymbolSpec(
                symbol=symbol,
                asset_class=asset_class,
                digits=data.get('digits', 5),
                point=data.get('point', 0.00001),
                contract_size=data.get('contract_size', 100000),
                volume_min=data.get('volume_min', 0.01),
                volume_max=data.get('volume_max', 100),
                volume_step=data.get('volume_step', 0.01),
                spread_typical=data.get('spread_current', 2.0),
                spread_min=data.get('spread_current', 2.0) * 0.5,
                spread_max=data.get('spread_current', 2.0) * 10,
                swap_long=data.get('swap_long', 0),
                swap_short=data.get('swap_short', 0),
            )
        except Exception as e:
            print(f"Error converting bridge data to spec: {e}")
            return None

    def _get_mt5_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """Get symbol spec directly from MT5 terminal."""
        if not self.mt5:
            return None

        info = self.mt5.symbol_info(symbol)
        if info is None:
            return None

        # Determine asset class from symbol properties
        if "BTC" in symbol or "ETH" in symbol or "CRYPTO" in info.path:
            asset_class = AssetClass.CRYPTO
        elif info.trade_calc_mode == 4:  # CFD indices
            asset_class = AssetClass.INDICES
        elif any(x in symbol.upper() for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
            asset_class = AssetClass.METALS
        elif any(x in symbol.upper() for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
            asset_class = AssetClass.ENERGY
        elif 'ETF' in symbol.upper() or 'ETF' in info.path:
            asset_class = AssetClass.ETFS
        elif info.trade_calc_mode == 0:  # Forex mode
            asset_class = AssetClass.FOREX
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
        spec = self.get_symbol_spec(symbol)
        point = spec.point if spec else 0.00001

        # MetaAPI mode (default)
        if self.mode == "metaapi" and self.connected:
            data = self.get_metaapi_symbol_spec(symbol)
            if data and "bid" in data and "ask" in data:
                bid = data['bid']
                ask = data['ask']
                return {
                    'bid': bid,
                    'ask': ask,
                    'spread': (ask - bid) / point if point > 0 else 0,
                    'spread_price': ask - bid,
                    'time': datetime.now(),  # MetaAPI returns server time
                    'volume': data.get('volume', 0),
                    'last': data.get('last', 0),
                    'source': 'metaapi'
                }

        # Direct MT5 mode
        if self.mode == "direct" and self.mt5:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': (tick.ask - tick.bid) / point,
                'spread_price': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume,
                'last': tick.last,
                'source': 'mt5_direct'
            }

        # Bridge mode
        elif self.mode == "bridge":
            data = self._bridge_request("tick", symbol=symbol)
            if data and "bid" in data and "ask" in data:
                return {
                    'bid': data['bid'],
                    'ask': data['ask'],
                    'spread': (data['ask'] - data['bid']) / point,
                    'spread_price': data['ask'] - data['bid'],
                    'time': datetime.now(),  # Bridge doesn't send time
                    'volume': data.get('volume', 0),
                    'last': data.get('last', 0),
                    'source': 'bridge'
                }

        return None

    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        # MetaAPI mode (default)
        if self.mode == "metaapi" and self.connected:
            if hasattr(self, '_metaapi_account_info') and self._metaapi_account_info:
                info = self._metaapi_account_info
                return {
                    'login': info.get('login', ''),
                    'server': info.get('server', ''),
                    'name': info.get('name', ''),
                    'balance': info.get('balance', 0),
                    'equity': info.get('equity', 0),
                    'margin': info.get('margin', 0),
                    'margin_free': info.get('freeMargin', 0),
                    'margin_level': info.get('marginLevel', 0),
                    'currency': info.get('currency', 'USD'),
                    'leverage': info.get('leverage', 100),
                    'profit': info.get('profit', 0),
                    'broker': info.get('broker', ''),
                    'source': 'metaapi'
                }

        # Direct MT5 mode (fallback)
        if self.mode == "direct" and self.mt5:
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
                'source': 'mt5_direct'
            }

        return None

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        start_time: datetime = None
    ) -> Optional[np.ndarray]:
        """
        Get historical OHLCV rates.

        Works with both MetaAPI (default) and direct MT5 (fallback).

        Args:
            symbol: Symbol name
            timeframe: Timeframe string (M1, M5, H1, D1, etc.)
            count: Number of bars to get
            start_time: Start time for historical data (default: count bars back from now)

        Returns:
            Numpy structured array with OHLCV data
        """
        from datetime import timedelta

        # Calculate start time if not provided
        if start_time is None:
            # Estimate minutes per bar
            tf_minutes = {
                'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080
            }
            minutes = tf_minutes.get(timeframe.upper(), 60)
            start_time = datetime.now() - timedelta(minutes=minutes * count)

        # MetaAPI mode (default)
        if self.mode == "metaapi" and self.connected:
            candles = self.get_metaapi_candles(symbol, timeframe, start_time, count)
            if candles:
                # Convert to numpy structured array (MT5 format)
                dtype = np.dtype([
                    ('time', 'datetime64[s]'),
                    ('open', 'f8'),
                    ('high', 'f8'),
                    ('low', 'f8'),
                    ('close', 'f8'),
                    ('tick_volume', 'i8'),
                    ('spread', 'i4'),
                    ('real_volume', 'i8')
                ])

                rates = np.zeros(len(candles), dtype=dtype)
                for i, c in enumerate(candles):
                    rates[i] = (
                        np.datetime64(c.get('time', datetime.now()), 's'),
                        c.get('open', 0),
                        c.get('high', 0),
                        c.get('low', 0),
                        c.get('close', 0),
                        c.get('tickVolume', 0),
                        c.get('spread', 0),
                        c.get('realVolume', 0)
                    )
                return rates

        # Direct MT5 mode (fallback)
        if self.mode == "direct" and self.mt5:
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

        return None

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
