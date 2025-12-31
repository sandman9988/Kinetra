"""
High-Performance Trading Engine
===============================

Integrates all performance components for optimized trading operations:
- Ring buffers for tick/bar data
- LRU caches for indicator calculations
- Async operations for I/O-bound tasks
- Parallel processing for CPU-bound tasks
- Timer-based bar aggregation

Architecture:
    TickDataFeed → TickBuffer → BarAggregator → BarBuffer → IndicatorEngine → SignalGenerator
         ↓             ↓              ↓              ↓             ↓              ↓
    AsyncStream    RingBuffer    TimerBased    RingBuffer    CachedCompute    ParallelProcessing
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .performance import (
    RingBuffer,
    TickBuffer,
    BarBuffer,
    LRUCache,
    ComputeCache,
    AsyncExecutor,
    AsyncDataStream,
    Timer,
    OnTickHandler,
    OnTimerHandler,
    ParallelProcessor,
    PerformanceMonitor,
)

logger = logging.getLogger(__name__)


class DataFeedState(Enum):
    """State of data feed."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class TickData:
    """Tick data structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    spread: float = 0.0
    
    def __post_init__(self):
        self.spread = self.ask - self.bid if self.bid > 0 else 0.0
        if self.last <= 0:
            self.last = (self.bid + self.ask) / 2


@dataclass
class BarData:
    """OHLCV bar data structure."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_count: int = 0
    spread: float = 0.0


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    timestamp: datetime
    direction: int  # 1 = long, -1 = short, 0 = flat
    strength: float  # 0-1
    reason: str = ""
    indicators: Dict[str, float] = field(default_factory=dict)


class HighPerformanceDataFeed:
    """
    High-performance data feed with buffering and async processing.
    
    Features:
    - Multiple symbol support
    - Tick buffering with ring buffers
    - Automatic bar aggregation
    - Async data streaming
    - Backpressure handling
    """
    
    def __init__(
        self,
        tick_buffer_size: int = 10000,
        bar_buffer_size: int = 1000,
        batch_size: int = 10,
    ):
        """
        Initialize data feed.
        
        Args:
            tick_buffer_size: Ticks per symbol buffer
            bar_buffer_size: Bars per symbol/timeframe buffer
            batch_size: Tick batch processing size
        """
        self.tick_buffer_size = tick_buffer_size
        self.bar_buffer_size = bar_buffer_size
        self.batch_size = batch_size
        
        # Per-symbol tick buffers
        self._tick_buffers: Dict[str, TickBuffer] = {}
        self._tick_handlers: Dict[str, OnTickHandler] = {}
        
        # Per-symbol/timeframe bar buffers
        self._bar_buffers: Dict[str, Dict[str, BarBuffer]] = defaultdict(dict)
        self._timer_handlers: Dict[str, OnTimerHandler] = {}
        
        # Async components
        self._async_executor = AsyncExecutor(max_concurrent=10)
        self._tick_streams: Dict[str, AsyncDataStream] = {}
        
        # State
        self._state = DataFeedState.DISCONNECTED
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_tick_callbacks: List[Callable[[TickData], None]] = []
        self._on_bar_callbacks: List[Callable[[BarData], None]] = []
        
        # Statistics
        self._stats = {
            'ticks_received': 0,
            'bars_created': 0,
            'errors': 0,
            'start_time': None,
        }
    
    def add_symbol(self, symbol: str, timeframes: List[str] = None) -> None:
        """
        Add symbol to data feed.
        
        Args:
            symbol: Symbol name
            timeframes: List of timeframes (M1, M5, H1, etc.)
        """
        timeframes = timeframes or ['M1', 'M5', 'M15', 'H1']
        
        with self._lock:
            # Create tick buffer
            self._tick_buffers[symbol] = TickBuffer(capacity=self.tick_buffer_size)
            
            # Create tick handler
            self._tick_handlers[symbol] = OnTickHandler(
                self._tick_buffers[symbol],
                batch_size=self.batch_size
            )
            
            # Create bar buffers for each timeframe
            for tf in timeframes:
                self._bar_buffers[symbol][tf] = BarBuffer(capacity=self.bar_buffer_size)
            
            # Create timer handler for bar aggregation
            timer_handler = OnTimerHandler(self._bar_buffers[symbol].get('M1', BarBuffer(100)))
            for tf in timeframes:
                timer_handler.register_timeframe(tf, lambda bar, s=symbol, t=tf: self._on_bar_complete(s, t, bar))
            self._timer_handlers[symbol] = timer_handler
            
            # Create async stream
            self._tick_streams[symbol] = AsyncDataStream(buffer_size=1000)
            
            logger.info(f"Added symbol {symbol} with timeframes {timeframes}")
    
    def on_tick(self, tick: TickData) -> None:
        """
        Process incoming tick.
        
        Args:
            tick: Tick data
        """
        if tick.symbol not in self._tick_buffers:
            self.add_symbol(tick.symbol)
        
        with self._lock:
            self._stats['ticks_received'] += 1
            
            # Add to tick handler (buffered processing)
            self._tick_handlers[tick.symbol].on_tick(
                tick.timestamp, tick.bid, tick.ask, tick.volume
            )
            
            # Update bar aggregator
            if tick.symbol in self._timer_handlers:
                self._timer_handlers[tick.symbol].update_tick(
                    tick.timestamp, tick.last, tick.volume
                )
            
            # Call callbacks
            for callback in self._on_tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")
                    self._stats['errors'] += 1
    
    def _on_bar_complete(self, symbol: str, timeframe: str, bar: Dict) -> None:
        """Handle completed bar."""
        with self._lock:
            self._stats['bars_created'] += 1
            
            bar_data = BarData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar.get('time', datetime.now()),
                open=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar.get('volume', 0),
            )
            
            # Call callbacks
            for callback in self._on_bar_callbacks:
                try:
                    callback(bar_data)
                except Exception as e:
                    logger.error(f"Bar callback error: {e}")
                    self._stats['errors'] += 1
    
    def register_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """Register tick callback."""
        self._on_tick_callbacks.append(callback)
    
    def register_bar_callback(self, callback: Callable[[BarData], None]) -> None:
        """Register bar callback."""
        self._on_bar_callbacks.append(callback)
    
    def start(self) -> None:
        """Start data feed."""
        self._state = DataFeedState.CONNECTED
        self._stats['start_time'] = time.time()
        
        # Start all timer handlers
        for handler in self._timer_handlers.values():
            handler.start()
        
        logger.info("Data feed started")
    
    def stop(self) -> None:
        """Stop data feed."""
        self._state = DataFeedState.DISCONNECTED
        
        # Stop all timer handlers
        for handler in self._timer_handlers.values():
            handler.stop()
        
        logger.info("Data feed stopped")
    
    def get_tick_buffer(self, symbol: str) -> Optional[TickBuffer]:
        """Get tick buffer for symbol."""
        return self._tick_buffers.get(symbol)
    
    def get_bar_buffer(self, symbol: str, timeframe: str) -> Optional[BarBuffer]:
        """Get bar buffer for symbol/timeframe."""
        return self._bar_buffers.get(symbol, {}).get(timeframe)
    
    def get_stats(self) -> Dict:
        """Get data feed statistics."""
        elapsed = time.time() - self._stats['start_time'] if self._stats['start_time'] else 0
        return {
            **self._stats,
            'state': self._state.name,
            'tick_rate': self._stats['ticks_received'] / max(1, elapsed),
            'bar_rate': self._stats['bars_created'] / max(1, elapsed),
            'symbols': list(self._tick_buffers.keys()),
        }


class CachedIndicatorEngine:
    """
    Indicator calculation engine with caching.
    
    Features:
    - LRU cache for indicator values
    - Automatic cache invalidation on new data
    - Parallel calculation support
    - Vectorized operations
    """
    
    def __init__(self, cache_size: int = 1000, ttl_seconds: float = 60.0):
        """
        Initialize indicator engine.
        
        Args:
            cache_size: Maximum cached calculations
            ttl_seconds: Cache entry TTL
        """
        self._cache = LRUCache(max_size=cache_size, ttl_seconds=ttl_seconds)
        self._compute_cache = ComputeCache(max_size=cache_size, ttl_seconds=ttl_seconds)
        self._parallel = ParallelProcessor(use_processes=False)  # Threads for indicators
        
        # Register cached indicator functions
        self.sma = self._compute_cache.cached(self._sma)
        self.ema = self._compute_cache.cached(self._ema)
        self.rsi = self._compute_cache.cached(self._rsi)
        self.macd = self._compute_cache.cached(self._macd)
        self.bollinger = self._compute_cache.cached(self._bollinger)
        self.atr = self._compute_cache.cached(self._atr)
    
    def _sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA."""
        if len(prices) < period:
            return np.array([])
        
        cumsum = np.cumsum(prices)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        return cumsum[period - 1:] / period
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices) - period + 1)
        ema[0] = prices[:period].mean()
        
        for i in range(1, len(ema)):
            ema[i] = alpha * prices[period - 1 + i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return np.array([])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))
        
        avg_gain[period - 1] = gains[:period].mean()
        avg_loss[period - 1] = losses[:period].mean()
        
        for i in range(period, len(deltas)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
        
        rs = np.divide(avg_gain[period - 1:], avg_loss[period - 1:], 
                       out=np.zeros_like(avg_gain[period - 1:]),
                       where=avg_loss[period - 1:] != 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(
        self, 
        prices: np.ndarray, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD."""
        if len(prices) < slow + signal:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        # Align arrays
        diff = len(ema_fast) - len(ema_slow)
        macd_line = ema_fast[diff:] - ema_slow
        
        signal_line = self._ema(macd_line, signal)
        diff2 = len(macd_line) - len(signal_line)
        histogram = macd_line[diff2:] - signal_line
        
        return macd_line[diff2:], signal_line, histogram
    
    def _bollinger(
        self, 
        prices: np.ndarray, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return np.array([]), np.array([]), np.array([])
        
        sma = self._sma(prices, period)
        
        # Rolling std
        std = np.zeros(len(sma))
        for i in range(len(sma)):
            std[i] = prices[i:i + period].std()
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower
    
    def _atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR."""
        if len(close) < period + 1:
            return np.array([])
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        if len(tr) < period:
            return np.array([])
        
        atr = np.zeros(len(tr) - period + 1)
        atr[0] = tr[:period].mean()
        
        alpha = 1 / period
        for i in range(1, len(atr)):
            atr[i] = alpha * tr[period - 1 + i] + (1 - alpha) * atr[i - 1]
        
        return atr
    
    def calculate_all(
        self, 
        prices: np.ndarray,
        high: np.ndarray = None,
        low: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all indicators in parallel.
        
        Args:
            prices: Close prices
            high: High prices (optional)
            low: Low prices (optional)
            
        Returns:
            Dict of indicator values
        """
        high = high if high is not None else prices
        low = low if low is not None else prices
        
        results = {}
        
        # Calculate in parallel using threads
        futures = []
        indicators = [
            ('sma_20', lambda: self.sma(prices, 20)),
            ('ema_20', lambda: self.ema(prices, 20)),
            ('rsi_14', lambda: self.rsi(prices, 14)),
            ('atr_14', lambda: self._atr(high, low, prices, 14)),
        ]
        
        for name, func in indicators:
            try:
                results[name] = func()
            except Exception as e:
                logger.warning(f"Indicator {name} calculation failed: {e}")
                results[name] = np.array([])
        
        # MACD and Bollinger return tuples
        try:
            macd_line, signal_line, histogram = self.macd(prices)
            results['macd_line'] = macd_line
            results['macd_signal'] = signal_line
            results['macd_histogram'] = histogram
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
        
        try:
            upper, middle, lower = self.bollinger(prices)
            results['bb_upper'] = upper
            results['bb_middle'] = middle
            results['bb_lower'] = lower
        except Exception as e:
            logger.warning(f"Bollinger calculation failed: {e}")
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self._cache.get_stats()


class SignalGenerator:
    """
    Signal generation with parallel processing.
    
    Features:
    - Multiple strategy support
    - Parallel signal calculation
    - Signal aggregation
    - Performance monitoring
    """
    
    def __init__(self, n_workers: int = 4):
        """
        Initialize signal generator.
        
        Args:
            n_workers: Number of parallel workers
        """
        self._strategies: Dict[str, Callable] = {}
        self._parallel = ParallelProcessor(n_workers=n_workers, use_processes=False)
        self._cache = LRUCache(max_size=100, ttl_seconds=5.0)
        
        # Statistics
        self._signal_count = 0
        self._last_signals: Dict[str, Signal] = {}
    
    def register_strategy(self, name: str, strategy: Callable) -> None:
        """
        Register signal generation strategy.
        
        Args:
            name: Strategy name
            strategy: Callable(indicators) -> Signal
        """
        self._strategies[name] = strategy
    
    def generate_signals(
        self, 
        symbol: str,
        indicators: Dict[str, np.ndarray],
        timestamp: datetime = None,
    ) -> List[Signal]:
        """
        Generate signals from all strategies.
        
        Args:
            symbol: Symbol name
            indicators: Indicator values
            timestamp: Signal timestamp
            
        Returns:
            List of signals from all strategies
        """
        timestamp = timestamp or datetime.now()
        signals = []
        
        for name, strategy in self._strategies.items():
            try:
                signal = strategy(symbol, indicators, timestamp)
                if signal:
                    signals.append(signal)
                    self._last_signals[f"{symbol}_{name}"] = signal
                    self._signal_count += 1
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
        
        return signals
    
    def aggregate_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Aggregate multiple signals into consensus.
        
        Args:
            signals: List of signals
            
        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None
        
        # Weighted average of directions by strength
        total_weight = sum(s.strength for s in signals)
        if total_weight == 0:
            return None
        
        weighted_direction = sum(s.direction * s.strength for s in signals) / total_weight
        avg_strength = total_weight / len(signals)
        
        # Determine consensus direction
        if weighted_direction > 0.3:
            direction = 1
        elif weighted_direction < -0.3:
            direction = -1
        else:
            direction = 0
        
        return Signal(
            symbol=signals[0].symbol,
            timestamp=signals[0].timestamp,
            direction=direction,
            strength=avg_strength * abs(weighted_direction),
            reason=f"Consensus from {len(signals)} strategies",
            indicators={s.reason: s.direction for s in signals},
        )
    
    def get_last_signal(self, symbol: str, strategy: str = None) -> Optional[Signal]:
        """Get last signal for symbol/strategy."""
        if strategy:
            return self._last_signals.get(f"{symbol}_{strategy}")
        
        # Get any signal for symbol
        for key, signal in self._last_signals.items():
            if key.startswith(symbol):
                return signal
        return None


class HighPerformanceEngine:
    """
    Complete high-performance trading engine.
    
    Integrates:
    - Data feed with tick/bar buffering
    - Cached indicator calculations
    - Parallel signal generation
    - Performance monitoring
    """
    
    def __init__(
        self,
        tick_buffer_size: int = 10000,
        bar_buffer_size: int = 1000,
        cache_size: int = 1000,
        n_workers: int = 4,
    ):
        """
        Initialize engine.
        
        Args:
            tick_buffer_size: Ticks per symbol buffer
            bar_buffer_size: Bars per symbol/timeframe buffer
            cache_size: Indicator cache size
            n_workers: Signal generation workers
        """
        # Components
        self._data_feed = HighPerformanceDataFeed(
            tick_buffer_size=tick_buffer_size,
            bar_buffer_size=bar_buffer_size,
        )
        self._indicator_engine = CachedIndicatorEngine(cache_size=cache_size)
        self._signal_generator = SignalGenerator(n_workers=n_workers)
        self._monitor = PerformanceMonitor()
        
        # Register callbacks
        self._data_feed.register_bar_callback(self._on_bar)
        
        # State
        self._running = False
        self._signals: List[Signal] = []
        self._on_signal_callbacks: List[Callable[[Signal], None]] = []
    
    def add_symbol(self, symbol: str, timeframes: List[str] = None) -> None:
        """Add symbol to engine."""
        self._data_feed.add_symbol(symbol, timeframes)
    
    def register_strategy(self, name: str, strategy: Callable) -> None:
        """Register trading strategy."""
        self._signal_generator.register_strategy(name, strategy)
    
    def register_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Register signal callback."""
        self._on_signal_callbacks.append(callback)
    
    def on_tick(self, tick: TickData) -> None:
        """Process incoming tick."""
        self._data_feed.on_tick(tick)
    
    def _on_bar(self, bar: BarData) -> None:
        """Process completed bar."""
        # Get bar buffer
        buffer = self._data_feed.get_bar_buffer(bar.symbol, bar.timeframe)
        if not buffer or len(buffer) < 50:
            return  # Need minimum data
        
        # Get OHLCV data
        bars = buffer.get_bars()
        closes = bars['close']
        highs = bars['high']
        lows = bars['low']
        
        # Calculate indicators
        indicators = self._indicator_engine.calculate_all(closes, highs, lows)
        
        # Generate signals
        signals = self._signal_generator.generate_signals(
            bar.symbol, indicators, bar.timestamp
        )
        
        # Aggregate signals
        if signals:
            consensus = self._signal_generator.aggregate_signals(signals)
            if consensus and consensus.direction != 0:
                self._signals.append(consensus)
                
                # Call callbacks
                for callback in self._on_signal_callbacks:
                    try:
                        callback(consensus)
                    except Exception as e:
                        logger.error(f"Signal callback error: {e}")
    
    def start(self) -> None:
        """Start engine."""
        self._running = True
        self._data_feed.start()
        logger.info("High-performance engine started")
    
    def stop(self) -> None:
        """Stop engine."""
        self._running = False
        self._data_feed.stop()
        logger.info("High-performance engine stopped")
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'data_feed': self._data_feed.get_stats(),
            'cache': self._indicator_engine.get_cache_stats(),
            'signals_generated': len(self._signals),
        }


# Default strategies for signal generation
def momentum_strategy(symbol: str, indicators: Dict, timestamp: datetime) -> Optional[Signal]:
    """Simple momentum strategy."""
    if 'rsi_14' not in indicators or len(indicators['rsi_14']) == 0:
        return None
    
    rsi = indicators['rsi_14'][-1]
    
    if rsi > 70:
        return Signal(symbol, timestamp, -1, 0.7, "RSI overbought")
    elif rsi < 30:
        return Signal(symbol, timestamp, 1, 0.7, "RSI oversold")
    
    return None


def trend_strategy(symbol: str, indicators: Dict, timestamp: datetime) -> Optional[Signal]:
    """Simple trend following strategy."""
    if 'sma_20' not in indicators or 'ema_20' not in indicators:
        return None
    
    sma = indicators['sma_20']
    ema = indicators['ema_20']
    
    if len(sma) == 0 or len(ema) == 0:
        return None
    
    # EMA above SMA = bullish
    if ema[-1] > sma[-1]:
        strength = min((ema[-1] - sma[-1]) / sma[-1] * 100, 1.0)
        return Signal(symbol, timestamp, 1, strength, "EMA > SMA")
    else:
        strength = min((sma[-1] - ema[-1]) / sma[-1] * 100, 1.0)
        return Signal(symbol, timestamp, -1, strength, "EMA < SMA")


def volatility_strategy(symbol: str, indicators: Dict, timestamp: datetime) -> Optional[Signal]:
    """Volatility breakout strategy."""
    if 'bb_upper' not in indicators or 'bb_lower' not in indicators:
        return None
    
    upper = indicators.get('bb_upper', np.array([]))
    lower = indicators.get('bb_lower', np.array([]))
    
    if len(upper) == 0 or len(lower) == 0:
        return None
    
    # Check for band expansion
    band_width = (upper[-1] - lower[-1]) / ((upper[-1] + lower[-1]) / 2)
    
    if band_width > 0.05:  # High volatility
        return Signal(symbol, timestamp, 0, 0.5, "High volatility - wait")
    
    return None


# Export all components
__all__ = [
    'DataFeedState',
    'TickData',
    'BarData',
    'Signal',
    'HighPerformanceDataFeed',
    'CachedIndicatorEngine',
    'SignalGenerator',
    'HighPerformanceEngine',
    'momentum_strategy',
    'trend_strategy',
    'volatility_strategy',
]
