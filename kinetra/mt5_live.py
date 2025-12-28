"""
MT5 Live Data Stream with Symbol Info for Friction/Viscosity

Uses MT5 symbol_info for real market friction metrics:
- Spread (bid-ask spread as friction)
- Volume constraints (min/max/step)
- Trade mode (liquidity indicator)
- Tick size (price granularity)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
import time
import threading

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .mt5_connector import MT5Connector


@dataclass
class SymbolFriction:
    """Real market friction metrics from MT5 symbol_info."""
    symbol: str
    spread: float              # Current spread in points
    spread_pct: float          # Spread as % of price
    volume_min: float          # Minimum lot size
    volume_max: float          # Maximum lot size
    volume_step: float         # Lot step
    tick_size: float           # Minimum price change
    tick_value: float          # Value per tick
    digits: int                # Price precision
    trade_mode: int            # 0=disabled, 1=long only, 2=short only, 3=full
    liquidity_score: float     # Derived liquidity (0-1)
    friction_score: float      # Derived friction (0-1)

    def to_dict(self) -> Dict:
        return {
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'volume_min': self.volume_min,
            'volume_max': self.volume_max,
            'tick_size': self.tick_size,
            'liquidity_score': self.liquidity_score,
            'friction_score': self.friction_score,
        }


class MT5LiveStream:
    """Live data stream from MT5 with physics friction metrics."""

    def __init__(self, symbols: List[str], path: Optional[str] = None):
        self.symbols = symbols
        self.connector = MT5Connector(path)
        self.running = False
        self._thread = None
        self._latest_data: Dict[str, Dict] = {}
        self._friction: Dict[str, SymbolFriction] = {}

    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        return self.connector.connect()

    def disconnect(self):
        """Disconnect from MT5."""
        self.stop()
        self.connector.disconnect()

    def get_symbol_friction(self, symbol: str) -> Optional[SymbolFriction]:
        """Get friction metrics from MT5 symbol_info."""
        if not self.connector.connected:
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        # Calculate spread
        spread_points = info.spread
        mid_price = (tick.bid + tick.ask) / 2
        spread_pct = (tick.ask - tick.bid) / mid_price * 100 if mid_price > 0 else 0

        # Calculate liquidity score (based on volume constraints)
        # Higher max volume and smaller step = more liquid
        vol_ratio = info.volume_max / info.volume_min if info.volume_min > 0 else 1
        liquidity_score = min(1.0, np.log10(vol_ratio + 1) / 4)  # Normalize to 0-1

        # Calculate friction score (based on spread and tick size)
        # Higher spread = higher friction
        friction_score = min(1.0, spread_pct / 0.1)  # 0.1% spread = max friction

        return SymbolFriction(
            symbol=symbol,
            spread=spread_points,
            spread_pct=spread_pct,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            tick_size=info.point,
            tick_value=info.trade_tick_value,
            digits=info.digits,
            trade_mode=info.trade_mode,
            liquidity_score=liquidity_score,
            friction_score=friction_score,
        )

    def get_live_physics_state(self, symbol: str) -> Optional[Dict]:
        """Get live physics state including symbol friction."""
        if not self.connector.connected:
            return None

        # Get recent bars for physics calculation
        df = self.connector.get_ohlcv(symbol, "M1", count=100)
        if df is None or len(df) < 20:
            return None

        # Get friction
        friction = self.get_symbol_friction(symbol)
        if friction is None:
            return None

        # Calculate physics metrics
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Velocity
        velocity = close.pct_change()

        # Energy
        energy = 0.5 * (velocity ** 2)

        # Damping (volatility-based)
        volatility = velocity.rolling(20).std()
        mean_abs = velocity.abs().rolling(20).mean()
        damping = volatility / (mean_abs + 1e-10)

        # Bar range
        bar_range = (high - low) / close
        avg_volume = volume.rolling(20).mean()
        volume_norm = volume / (avg_volume + 1e-10)

        # Viscosity (using spread as additional friction)
        base_viscosity = bar_range / (volume_norm + 1e-10)
        # Incorporate spread friction
        viscosity = base_viscosity.iloc[-1] * (1 + friction.friction_score)

        # Reynolds number
        reynolds = (velocity.abs() * bar_range * volume_norm) / (volatility + 1e-10)

        # Buying pressure
        bp = (close - low) / (high - low + 1e-10)

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': close.iloc[-1],
            'energy': energy.iloc[-1],
            'damping': damping.iloc[-1],
            'viscosity': viscosity,
            'reynolds': reynolds.iloc[-1],
            'buying_pressure': bp.rolling(5).mean().iloc[-1],
            'friction': friction.to_dict(),
            'spread_pct': friction.spread_pct,
            'liquidity_score': friction.liquidity_score,
        }

    def start_streaming(self, callback=None, interval: float = 1.0):
        """Start streaming live data."""
        if self.running:
            return

        self.running = True

        def stream_loop():
            while self.running:
                for symbol in self.symbols:
                    state = self.get_live_physics_state(symbol)
                    if state:
                        self._latest_data[symbol] = state
                        self._friction[symbol] = self.get_symbol_friction(symbol)
                        if callback:
                            callback(symbol, state)
                time.sleep(interval)

        self._thread = threading.Thread(target=stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop streaming."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_latest(self, symbol: str) -> Optional[Dict]:
        """Get latest data for symbol."""
        return self._latest_data.get(symbol)

    def get_all_latest(self) -> Dict[str, Dict]:
        """Get latest data for all symbols."""
        return self._latest_data.copy()


def calculate_market_friction(
    spread: float,
    tick_size: float,
    volume_step: float,
    avg_volume: float,
) -> float:
    """
    Calculate market friction from MT5 symbol info.

    Market friction = f(spread, tick_size, volume_constraints)

    Higher friction = harder to execute, more slippage, less liquid
    """
    # Spread component (normalized)
    spread_friction = min(1.0, spread / 100)  # 100 points = max

    # Tick size component (larger tick = more friction)
    tick_friction = min(1.0, tick_size * 10000)

    # Volume component (larger step = more friction)
    vol_friction = min(1.0, volume_step / 0.1)  # 0.1 lot step = moderate

    # Combined friction score
    friction = (spread_friction * 0.5 + tick_friction * 0.2 + vol_friction * 0.3)

    return friction


if __name__ == "__main__":
    if not MT5_AVAILABLE:
        print("MetaTrader5 package not installed")
        print("Install with: pip install MetaTrader5")
        exit(1)

    # Test live stream
    symbols = ["BTCUSD", "EURUSD", "XAUUSD"]
    stream = MT5LiveStream(symbols)

    if not stream.connect():
        print("Failed to connect to MT5")
        exit(1)

    print("Connected to MT5!")
    print("\nSymbol Friction Metrics:")
    print("-" * 60)

    for symbol in symbols:
        friction = stream.get_symbol_friction(symbol)
        if friction:
            print(f"\n{symbol}:")
            print(f"  Spread: {friction.spread} pts ({friction.spread_pct:.4f}%)")
            print(f"  Volume: {friction.volume_min} - {friction.volume_max} (step: {friction.volume_step})")
            print(f"  Tick size: {friction.tick_size}")
            print(f"  Liquidity score: {friction.liquidity_score:.2f}")
            print(f"  Friction score: {friction.friction_score:.2f}")

    print("\n\nLive Physics State:")
    print("-" * 60)

    for symbol in symbols:
        state = stream.get_live_physics_state(symbol)
        if state:
            print(f"\n{symbol} @ {state['price']:.2f}:")
            print(f"  Energy: {state['energy']:.6f}")
            print(f"  Viscosity: {state['viscosity']:.4f}")
            print(f"  Reynolds: {state['reynolds']:.4f}")
            print(f"  Buying pressure: {state['buying_pressure']:.2f}")
            print(f"  Spread: {state['spread_pct']:.4f}%")

    stream.disconnect()
