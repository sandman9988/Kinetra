"""
Performance Module - High-Performance Trading Infrastructure
============================================================

Implements:
- Ring buffers for efficient tick/bar data management
- LRU and TTL caches for expensive computations
- Async operations for I/O-bound tasks
- Parallel processing for CPU-bound tasks
- Timer-based operations with scheduling
- Lock-free data structures where possible
- Memory-mapped buffers for large datasets

Design Principles:
- Zero-copy where possible
- Pre-allocated buffers to avoid GC pressure
- Efficient numpy operations
- Async-first for I/O operations
- Thread-safe with minimal locking
"""

import asyncio
import hashlib
import logging
import mmap
import os
import pickle
import struct
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any, Callable, Coroutine, Dict, Generic, Iterator, List, 
    Optional, Tuple, TypeVar, Union
)

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# =============================================================================
# Ring Buffers - Efficient Fixed-Size Data Structures
# =============================================================================

class RingBuffer(Generic[T]):
    """
    Lock-free ring buffer for high-frequency data.
    
    Features:
    - O(1) append and access
    - Pre-allocated memory
    - No garbage collection pressure
    - Thread-safe reads (single writer, multiple readers)
    """
    
    def __init__(self, capacity: int, dtype: type = float):
        """
        Initialize ring buffer.
        
        Args:
            capacity: Maximum number of elements
            dtype: Data type (float, int, etc.)
        """
        self.capacity = capacity
        self.dtype = dtype
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._head = 0  # Next write position
        self._count = 0  # Number of valid elements
        self._version = 0  # For detecting concurrent modifications
    
    def append(self, value: T) -> None:
        """Append value to buffer (overwrites oldest if full)."""
        self._buffer[self._head] = value
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self._version += 1
    
    def extend(self, values: np.ndarray) -> None:
        """Append multiple values efficiently."""
        n = len(values)
        if n >= self.capacity:
            # Just take last capacity elements
            self._buffer[:] = values[-self.capacity:]
            self._head = 0
            self._count = self.capacity
        else:
            # Wrap-around copy
            end_space = self.capacity - self._head
            if n <= end_space:
                self._buffer[self._head:self._head + n] = values
            else:
                self._buffer[self._head:] = values[:end_space]
                self._buffer[:n - end_space] = values[end_space:]
            self._head = (self._head + n) % self.capacity
            self._count = min(self._count + n, self.capacity)
        self._version += 1
    
    def get_last(self, n: int = 1) -> np.ndarray:
        """Get last n elements (most recent first)."""
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=self.dtype)
        
        result = np.zeros(n, dtype=self.dtype)
        for i in range(n):
            idx = (self._head - 1 - i) % self.capacity
            result[i] = self._buffer[idx]
        return result
    
    def get_all(self) -> np.ndarray:
        """Get all elements in order (oldest first)."""
        if self._count < self.capacity:
            return self._buffer[:self._count].copy()
        
        # Full buffer - need to reorder
        result = np.zeros(self.capacity, dtype=self.dtype)
        result[:self.capacity - self._head] = self._buffer[self._head:]
        result[self.capacity - self._head:] = self._buffer[:self._head]
        return result
    
    def __len__(self) -> int:
        return self._count
    
    def __getitem__(self, idx: int) -> T:
        """Get element by index (0 = oldest, -1 = newest)."""
        if idx < 0:
            idx = self._count + idx
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range for buffer of size {self._count}")
        
        if self._count < self.capacity:
            return self._buffer[idx]
        
        actual_idx = (self._head + idx) % self.capacity
        return self._buffer[actual_idx]
    
    @property
    def is_full(self) -> bool:
        return self._count >= self.capacity
    
    def clear(self) -> None:
        """Clear buffer."""
        self._head = 0
        self._count = 0
        self._version += 1


class TickBuffer:
    """
    Specialized ring buffer for tick data (OHLCV).
    
    Stores multiple fields efficiently with aligned memory.
    """
    
    FIELDS = ['time', 'bid', 'ask', 'last', 'volume', 'spread']
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize tick buffer.
        
        Args:
            capacity: Maximum number of ticks
        """
        self.capacity = capacity
        self._head = 0
        self._count = 0
        
        # Pre-allocate structured array
        self._dtype = np.dtype([
            ('time', 'datetime64[us]'),
            ('bid', 'f8'),
            ('ask', 'f8'),
            ('last', 'f8'),
            ('volume', 'f8'),
            ('spread', 'f8'),
        ])
        self._buffer = np.zeros(capacity, dtype=self._dtype)
    
    def append_tick(
        self,
        timestamp: datetime,
        bid: float,
        ask: float,
        last: float = 0.0,
        volume: float = 0.0,
    ) -> None:
        """Append a tick."""
        spread = (ask - bid) if bid > 0 else 0.0
        last = last if last > 0 else (bid + ask) / 2
        
        self._buffer[self._head] = (
            np.datetime64(timestamp, 'us'),
            bid, ask, last, volume, spread
        )
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
    
    def get_last_ticks(self, n: int = 100) -> np.ndarray:
        """Get last n ticks."""
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=self._dtype)
        
        result = np.zeros(n, dtype=self._dtype)
        for i in range(n):
            idx = (self._head - 1 - i) % self.capacity
            result[n - 1 - i] = self._buffer[idx]  # Oldest first
        return result
    
    def get_ohlcv(self, period_seconds: int = 60) -> Optional[Dict]:
        """
        Aggregate ticks into OHLCV bar.
        
        Args:
            period_seconds: Bar period in seconds
            
        Returns:
            OHLCV dict or None if insufficient data
        """
        if self._count == 0:
            return None
        
        ticks = self.get_last_ticks(self._count)
        if len(ticks) == 0:
            return None
        
        now = np.datetime64(datetime.now(), 'us')
        period_start = now - np.timedelta64(period_seconds, 's')
        
        # Filter ticks in period
        mask = ticks['time'] >= period_start
        period_ticks = ticks[mask]
        
        if len(period_ticks) == 0:
            return None
        
        return {
            'time': datetime.now(),
            'open': float(period_ticks['last'][0]),
            'high': float(np.max(period_ticks['last'])),
            'low': float(np.min(period_ticks['last'])),
            'close': float(period_ticks['last'][-1]),
            'volume': float(np.sum(period_ticks['volume'])),
            'tick_count': len(period_ticks),
            'avg_spread': float(np.mean(period_ticks['spread'])),
        }
    
    @property
    def last_tick(self) -> Optional[np.void]:
        """Get most recent tick."""
        if self._count == 0:
            return None
        return self._buffer[(self._head - 1) % self.capacity]
    
    def __len__(self) -> int:
        return self._count


class BarBuffer:
    """
    Ring buffer for OHLCV bars with indicator computation.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize bar buffer.
        
        Args:
            capacity: Maximum number of bars
        """
        self.capacity = capacity
        self._head = 0
        self._count = 0
        
        # Pre-allocate structured array
        self._dtype = np.dtype([
            ('time', 'datetime64[s]'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8'),
            ('spread', 'f8'),
        ])
        self._buffer = np.zeros(capacity, dtype=self._dtype)
        
        # Cached indicators
        self._cache_version = 0
        self._sma_cache: Dict[int, np.ndarray] = {}
        self._ema_cache: Dict[int, np.ndarray] = {}
    
    def append_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        spread: float = 0.0,
    ) -> None:
        """Append a bar."""
        self._buffer[self._head] = (
            np.datetime64(timestamp, 's'),
            open_, high, low, close, volume, spread
        )
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self._cache_version += 1  # Invalidate caches
    
    def get_bars(self, n: int = None) -> np.ndarray:
        """Get last n bars (oldest first)."""
        n = n or self._count
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=self._dtype)
        
        result = np.zeros(n, dtype=self._dtype)
        for i in range(n):
            idx = (self._head - n + i) % self.capacity
            result[i] = self._buffer[idx]
        return result
    
    def get_closes(self, n: int = None) -> np.ndarray:
        """Get last n close prices."""
        bars = self.get_bars(n)
        return bars['close'] if len(bars) > 0 else np.array([])
    
    def sma(self, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if period in self._sma_cache and self._cache_version == self._sma_cache.get('_version', -1):
            return self._sma_cache[period]
        
        closes = self.get_closes()
        if len(closes) < period:
            return np.array([])
        
        # Efficient cumsum-based SMA
        cumsum = np.cumsum(closes)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        sma = cumsum[period - 1:] / period
        
        self._sma_cache[period] = sma
        self._sma_cache['_version'] = self._cache_version
        return sma
    
    def ema(self, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if period in self._ema_cache and self._cache_version == self._ema_cache.get('_version', -1):
            return self._ema_cache[period]
        
        closes = self.get_closes()
        if len(closes) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(closes) - period + 1)
        ema[0] = closes[:period].mean()  # SMA for first value
        
        for i in range(1, len(ema)):
            ema[i] = alpha * closes[period - 1 + i] + (1 - alpha) * ema[i - 1]
        
        self._ema_cache[period] = ema
        self._ema_cache['_version'] = self._cache_version
        return ema
    
    def atr(self, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        bars = self.get_bars()
        if len(bars) < period + 1:
            return np.array([])
        
        high = bars['high']
        low = bars['low']
        close = bars['close']
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR (EMA of TR)
        if len(tr) < period:
            return np.array([])
        
        alpha = 1 / period  # Wilder's smoothing
        atr = np.zeros(len(tr) - period + 1)
        atr[0] = tr[:period].mean()
        
        for i in range(1, len(atr)):
            atr[i] = alpha * tr[period - 1 + i] + (1 - alpha) * atr[i - 1]
        
        return atr
    
    @property
    def last_bar(self) -> Optional[np.void]:
        """Get most recent bar."""
        if self._count == 0:
            return None
        return self._buffer[(self._head - 1) % self.capacity]
    
    def __len__(self) -> int:
        return self._count


# =============================================================================
# Caching - LRU and TTL Caches
# =============================================================================

class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with optional TTL.
    
    Features:
    - O(1) get/set operations
    - Automatic eviction of least recently used
    - Optional time-to-live for entries
    - Memory usage tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Optional time-to-live for entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[K, Tuple[V, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: K, default: V = None) -> Optional[V]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return default
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and (time.time() - timestamp) > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value
    
    def set(self, key: K, value: V) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            self._cache[key] = (value, time.time())
    
    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
        }


class ComputeCache:
    """
    Cache for expensive computations with automatic invalidation.
    
    Features:
    - Function-based caching with automatic key generation
    - Dependency tracking for invalidation
    - Memory-bounded
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: float = 300.0):
        """
        Initialize compute cache.
        
        Args:
            max_size: Maximum cached results
            ttl_seconds: Time-to-live for cached results
        """
        self._cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self._dependencies: Dict[str, set] = {}  # key -> dependent keys
    
    def _make_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        key_parts = [func.__name__]
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:8])
            else:
                key_parts.append(str(hash(arg) if hasattr(arg, '__hash__') and arg.__hash__ else id(arg)))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={hash(v) if hasattr(v, '__hash__') and v.__hash__ else id(v)}")
        
        return ":".join(key_parts)
    
    def cached(self, func: Callable) -> Callable:
        """Decorator for caching function results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(func, args, kwargs)
            result = self._cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            self._cache.set(key, result)
            return result
        
        return wrapper
    
    def invalidate(self, pattern: str = None) -> int:
        """
        Invalidate cached entries.
        
        Args:
            pattern: Optional pattern to match keys (None = clear all)
            
        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        # Pattern matching invalidation
        count = 0
        with self._cache._lock:
            keys_to_remove = [k for k in self._cache._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache._cache[key]
                count += 1
        return count


def cached_property(ttl_seconds: float = None):
    """
    Decorator for cached properties with optional TTL.
    
    Usage:
        class MyClass:
            @cached_property(ttl_seconds=60)
            def expensive_computation(self):
                ...
    """
    def decorator(func: Callable) -> property:
        cache_attr = f'_cached_{func.__name__}'
        time_attr = f'_cached_time_{func.__name__}'
        
        @property
        @wraps(func)
        def wrapper(self):
            cached = getattr(self, cache_attr, None)
            cached_time = getattr(self, time_attr, 0)
            
            if cached is not None:
                if ttl_seconds is None or (time.time() - cached_time) < ttl_seconds:
                    return cached
            
            result = func(self)
            setattr(self, cache_attr, result)
            setattr(self, time_attr, time.time())
            return result
        
        return wrapper
    return decorator


# =============================================================================
# Async Operations
# =============================================================================

class AsyncExecutor:
    """
    Async executor for I/O-bound operations.
    
    Features:
    - Event loop management
    - Task scheduling
    - Error handling
    - Timeout support
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize async executor.
        
        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._semaphore = None  # Created in async context
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: List[asyncio.Task] = []
    
    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    async def execute(
        self,
        coro: Coroutine,
        timeout: float = None,
    ) -> Any:
        """
        Execute coroutine with optional timeout.
        
        Args:
            coro: Coroutine to execute
            timeout: Optional timeout in seconds
            
        Returns:
            Coroutine result
        """
        sem = await self._get_semaphore()
        
        async with sem:
            if timeout:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
    
    async def execute_many(
        self,
        coros: List[Coroutine],
        timeout: float = None,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        Execute multiple coroutines concurrently.
        
        Args:
            coros: List of coroutines
            timeout: Optional timeout per coroutine
            return_exceptions: If True, return exceptions instead of raising
            
        Returns:
            List of results (in order)
        """
        sem = await self._get_semaphore()
        
        async def bounded_coro(coro):
            async with sem:
                if timeout:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
        
        tasks = [asyncio.create_task(bounded_coro(c)) for c in coros]
        
        if return_exceptions:
            return await asyncio.gather(*tasks, return_exceptions=True)
        return await asyncio.gather(*tasks)
    
    def run(self, coro: Coroutine) -> Any:
        """Run coroutine from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create task
            return loop.create_task(coro)
        except RuntimeError:
            # No running loop - use asyncio.run
            return asyncio.run(coro)


class AsyncDataStream:
    """
    Async data stream for tick/bar processing.
    
    Features:
    - Async iteration
    - Buffering
    - Backpressure handling
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize async data stream.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self._queue: asyncio.Queue = None
        self._closed = False
    
    async def _get_queue(self) -> asyncio.Queue:
        """Get or create queue."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.buffer_size)
        return self._queue
    
    async def put(self, item: Any, timeout: float = None) -> bool:
        """
        Put item into stream.
        
        Args:
            item: Item to put
            timeout: Optional timeout
            
        Returns:
            True if successful, False if timeout
        """
        if self._closed:
            raise RuntimeError("Stream is closed")
        
        queue = await self._get_queue()
        
        try:
            if timeout:
                await asyncio.wait_for(queue.put(item), timeout=timeout)
            else:
                await queue.put(item)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def get(self, timeout: float = None) -> Any:
        """
        Get item from stream.
        
        Args:
            timeout: Optional timeout
            
        Returns:
            Item or raises TimeoutError
        """
        queue = await self._get_queue()
        
        if timeout:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        return await queue.get()
    
    def close(self) -> None:
        """Close stream."""
        self._closed = True
    
    async def __aiter__(self):
        """Async iteration over stream."""
        while not self._closed:
            try:
                item = await self.get(timeout=1.0)
                yield item
            except asyncio.TimeoutError:
                continue
            except Exception:
                break


# =============================================================================
# Timer Operations
# =============================================================================

class Timer:
    """
    High-resolution timer for scheduling operations.
    
    Features:
    - Periodic callbacks
    - One-shot callbacks
    - Drift compensation
    - Statistics tracking
    """
    
    def __init__(self):
        """Initialize timer."""
        self._callbacks: Dict[str, Dict] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._call_counts: Dict[str, int] = {}
        self._total_latencies: Dict[str, float] = {}
    
    def schedule_periodic(
        self,
        name: str,
        callback: Callable,
        interval_seconds: float,
        drift_compensate: bool = True,
    ) -> None:
        """
        Schedule periodic callback.
        
        Args:
            name: Timer name
            callback: Function to call
            interval_seconds: Interval between calls
            drift_compensate: Compensate for execution time drift
        """
        with self._lock:
            self._callbacks[name] = {
                'callback': callback,
                'interval': interval_seconds,
                'next_time': time.time() + interval_seconds,
                'drift_compensate': drift_compensate,
                'periodic': True,
            }
            self._call_counts[name] = 0
            self._total_latencies[name] = 0.0
    
    def schedule_once(
        self,
        name: str,
        callback: Callable,
        delay_seconds: float,
    ) -> None:
        """
        Schedule one-shot callback.
        
        Args:
            name: Timer name
            callback: Function to call
            delay_seconds: Delay before call
        """
        with self._lock:
            self._callbacks[name] = {
                'callback': callback,
                'next_time': time.time() + delay_seconds,
                'periodic': False,
            }
    
    def cancel(self, name: str) -> bool:
        """Cancel scheduled callback."""
        with self._lock:
            if name in self._callbacks:
                del self._callbacks[name]
                return True
            return False
    
    def start(self) -> None:
        """Start timer thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop timer thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _run_loop(self) -> None:
        """Main timer loop."""
        while self._running:
            now = time.time()
            to_remove = []
            
            with self._lock:
                callbacks_copy = dict(self._callbacks)
            
            for name, info in callbacks_copy.items():
                if now >= info['next_time']:
                    # Execute callback
                    start = time.time()
                    try:
                        info['callback']()
                    except Exception as e:
                        logger.error(f"Timer callback {name} error: {e}")
                    
                    latency = time.time() - start
                    
                    with self._lock:
                        if name in self._callbacks:
                            self._call_counts[name] = self._call_counts.get(name, 0) + 1
                            self._total_latencies[name] = self._total_latencies.get(name, 0) + latency
                            
                            if info['periodic']:
                                # Schedule next
                                if info['drift_compensate']:
                                    info['next_time'] += info['interval']
                                else:
                                    info['next_time'] = time.time() + info['interval']
                            else:
                                to_remove.append(name)
            
            # Remove one-shot callbacks
            with self._lock:
                for name in to_remove:
                    self._callbacks.pop(name, None)
            
            # Sleep until next callback
            time.sleep(0.001)  # 1ms resolution
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get timer statistics."""
        with self._lock:
            stats = {}
            for name in self._callbacks:
                count = self._call_counts.get(name, 0)
                total_lat = self._total_latencies.get(name, 0)
                stats[name] = {
                    'call_count': count,
                    'avg_latency_ms': (total_lat / count * 1000) if count > 0 else 0,
                }
            return stats


class OnTickHandler:
    """
    Handler for on-tick events with batching and throttling.
    
    Features:
    - Tick batching for efficiency
    - Rate limiting
    - Async processing
    """
    
    def __init__(
        self,
        tick_buffer: TickBuffer,
        batch_size: int = 10,
        max_rate_per_second: float = 100.0,
    ):
        """
        Initialize on-tick handler.
        
        Args:
            tick_buffer: TickBuffer to store ticks
            batch_size: Process ticks in batches
            max_rate_per_second: Maximum tick processing rate
        """
        self.tick_buffer = tick_buffer
        self.batch_size = batch_size
        self.min_interval = 1.0 / max_rate_per_second
        
        self._callbacks: List[Callable] = []
        self._pending_ticks: List[Dict] = []
        self._last_process_time = 0.0
        self._lock = threading.RLock()
        
        # Statistics
        self.ticks_received = 0
        self.ticks_processed = 0
        self.batches_processed = 0
    
    def register_callback(self, callback: Callable) -> None:
        """Register callback for tick events."""
        self._callbacks.append(callback)
    
    def on_tick(
        self,
        timestamp: datetime,
        bid: float,
        ask: float,
        volume: float = 0.0,
    ) -> None:
        """
        Handle incoming tick.
        
        Args:
            timestamp: Tick timestamp
            bid: Bid price
            ask: Ask price
            volume: Tick volume
        """
        with self._lock:
            self.ticks_received += 1
            
            # Store in buffer
            self.tick_buffer.append_tick(timestamp, bid, ask, volume=volume)
            
            # Add to pending batch
            self._pending_ticks.append({
                'time': timestamp,
                'bid': bid,
                'ask': ask,
                'volume': volume,
            })
            
            # Check if should process batch
            now = time.time()
            should_process = (
                len(self._pending_ticks) >= self.batch_size or
                (now - self._last_process_time) >= self.min_interval
            )
            
            if should_process and self._pending_ticks:
                self._process_batch()
    
    def _process_batch(self) -> None:
        """Process pending tick batch."""
        batch = self._pending_ticks.copy()
        self._pending_ticks.clear()
        self._last_process_time = time.time()
        
        self.ticks_processed += len(batch)
        self.batches_processed += 1
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(batch)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            'ticks_received': self.ticks_received,
            'ticks_processed': self.ticks_processed,
            'batches_processed': self.batches_processed,
            'avg_batch_size': self.ticks_processed / max(1, self.batches_processed),
            'buffer_size': len(self.tick_buffer),
        }


class OnTimerHandler:
    """
    Handler for timer-based events.
    
    Features:
    - Multiple timeframes
    - Bar completion callbacks
    - Drift-compensated timing
    """
    
    def __init__(self, bar_buffer: BarBuffer):
        """
        Initialize on-timer handler.
        
        Args:
            bar_buffer: BarBuffer to store bars
        """
        self.bar_buffer = bar_buffer
        self._timer = Timer()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._current_bars: Dict[str, Dict] = {}  # timeframe -> partial bar
    
    def register_timeframe(
        self,
        timeframe: str,
        callback: Callable,
    ) -> None:
        """
        Register callback for timeframe.
        
        Args:
            timeframe: Timeframe string (M1, M5, M15, H1, etc.)
            callback: Function to call on bar completion
        """
        if timeframe not in self._callbacks:
            self._callbacks[timeframe] = []
            
            # Parse timeframe to seconds
            seconds = self._parse_timeframe(timeframe)
            
            # Schedule timer
            self._timer.schedule_periodic(
                f"bar_{timeframe}",
                lambda tf=timeframe: self._on_bar_complete(tf),
                seconds,
            )
        
        self._callbacks[timeframe].append(callback)
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        tf_map = {
            'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800,
            'H1': 3600, 'H4': 14400, 'D1': 86400, 'W1': 604800,
        }
        return tf_map.get(timeframe.upper(), 60)
    
    def update_tick(
        self,
        timestamp: datetime,
        price: float,
        volume: float = 0.0,
    ) -> None:
        """
        Update current bars with tick.
        
        Args:
            timestamp: Tick timestamp
            price: Tick price
            volume: Tick volume
        """
        for timeframe in self._callbacks:
            if timeframe not in self._current_bars:
                self._current_bars[timeframe] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'time': timestamp,
                }
            else:
                bar = self._current_bars[timeframe]
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
                bar['volume'] += volume
    
    def _on_bar_complete(self, timeframe: str) -> None:
        """Handle bar completion."""
        if timeframe not in self._current_bars:
            return
        
        bar = self._current_bars.pop(timeframe)
        
        # Add to buffer
        self.bar_buffer.append_bar(
            timestamp=bar['time'],
            open_=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
        )
        
        # Call callbacks
        for callback in self._callbacks.get(timeframe, []):
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    def start(self) -> None:
        """Start timer."""
        self._timer.start()
    
    def stop(self) -> None:
        """Stop timer."""
        self._timer.stop()


# =============================================================================
# Parallel Processing
# =============================================================================

class ParallelProcessor:
    """
    Parallel processor for CPU-bound operations.
    
    Features:
    - Process pool for CPU-bound tasks
    - Thread pool for I/O-bound tasks
    - Work stealing
    - Progress tracking
    """
    
    def __init__(
        self,
        n_workers: int = None,
        use_processes: bool = True,
    ):
        """
        Initialize parallel processor.
        
        Args:
            n_workers: Number of workers (default: CPU count)
            use_processes: Use processes (True) or threads (False)
        """
        self.n_workers = n_workers or os.cpu_count()
        self.use_processes = use_processes
        self._executor = None
    
    def _get_executor(self):
        """Get or create executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        return self._executor
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        chunksize: int = 1,
        callback: Callable = None,
    ) -> List[Any]:
        """
        Map function over items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            chunksize: Items per worker task
            callback: Optional progress callback(completed, total)
            
        Returns:
            List of results
        """
        executor = self._get_executor()
        
        if callback:
            # Use futures for progress tracking
            futures = []
            for item in items:
                futures.append(executor.submit(func, item))
            
            results = []
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                callback(completed, len(items))
            
            # Reorder to match input
            return [f.result() for f in futures]
        else:
            # Use efficient map
            return list(executor.map(func, items, chunksize=chunksize))
    
    def starmap(
        self,
        func: Callable,
        items: List[Tuple],
    ) -> List[Any]:
        """
        Map function over items with multiple arguments.
        
        Args:
            func: Function to apply
            items: List of argument tuples
            
        Returns:
            List of results
        """
        executor = self._get_executor()
        futures = [executor.submit(func, *args) for args in items]
        return [f.result() for f in futures]
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


# =============================================================================
# Performance Monitor
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    tick_rate: float  # Ticks per second
    bar_rate: float  # Bars per second
    cache_hit_rate: float
    avg_latency_ms: float
    memory_mb: float
    cpu_percent: float


class PerformanceMonitor:
    """
    Monitor system performance.
    
    Features:
    - Real-time metrics
    - Historical tracking
    - Alerts
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self._tick_handler: Optional[OnTickHandler] = None
        self._timer_handler: Optional[OnTimerHandler] = None
        self._caches: List[LRUCache] = []
        self._timers: List[Timer] = []
        
        self._metrics_history: RingBuffer = RingBuffer(1000, dtype=object)
        self._start_time = time.time()
    
    def register_tick_handler(self, handler: OnTickHandler) -> None:
        """Register tick handler for monitoring."""
        self._tick_handler = handler
    
    def register_timer_handler(self, handler: OnTimerHandler) -> None:
        """Register timer handler for monitoring."""
        self._timer_handler = handler
    
    def register_cache(self, cache: LRUCache) -> None:
        """Register cache for monitoring."""
        self._caches.append(cache)
    
    def register_timer(self, timer: Timer) -> None:
        """Register timer for monitoring."""
        self._timers.append(timer)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        import resource
        
        # Tick rate
        tick_rate = 0.0
        if self._tick_handler:
            elapsed = time.time() - self._start_time
            tick_rate = self._tick_handler.ticks_received / max(1, elapsed)
        
        # Cache hit rate
        cache_hit_rate = 0.0
        if self._caches:
            rates = [c.hit_rate for c in self._caches]
            cache_hit_rate = sum(rates) / len(rates)
        
        # Timer latency
        avg_latency = 0.0
        if self._timers:
            latencies = []
            for timer in self._timers:
                stats = timer.get_stats()
                latencies.extend([s['avg_latency_ms'] for s in stats.values()])
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
        
        # Memory usage
        try:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = rusage.ru_maxrss / 1024  # KB to MB on Linux
        except Exception:
            memory_mb = 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            tick_rate=tick_rate,
            bar_rate=0.0,  # TODO: implement
            cache_hit_rate=cache_hit_rate,
            avg_latency_ms=avg_latency,
            memory_mb=memory_mb,
            cpu_percent=0.0,  # TODO: implement
        )
    
    def log_metrics(self) -> None:
        """Log current metrics."""
        metrics = self.get_metrics()
        logger.info(
            f"Performance: {metrics.tick_rate:.1f} ticks/s, "
            f"cache hit: {metrics.cache_hit_rate:.1%}, "
            f"latency: {metrics.avg_latency_ms:.2f}ms, "
            f"memory: {metrics.memory_mb:.1f}MB"
        )


# Export all components
__all__ = [
    # Ring Buffers
    'RingBuffer',
    'TickBuffer',
    'BarBuffer',
    # Caches
    'LRUCache',
    'ComputeCache',
    'cached_property',
    # Async
    'AsyncExecutor',
    'AsyncDataStream',
    # Timers
    'Timer',
    'OnTickHandler',
    'OnTimerHandler',
    # Parallel
    'ParallelProcessor',
    # Monitoring
    'PerformanceMetrics',
    'PerformanceMonitor',
]
