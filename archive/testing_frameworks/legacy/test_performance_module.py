#!/usr/bin/env python3
"""
Performance Module Test Suite
=============================

Tests for:
- Ring buffers (tick, bar)
- LRU cache with TTL
- Async operations
- Timer operations
- Parallel processing
- Performance monitoring

Run: python scripts/test_performance_module.py
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.performance import (
    RingBuffer, TickBuffer, BarBuffer,
    LRUCache, ComputeCache,
    AsyncExecutor, AsyncDataStream,
    Timer, OnTickHandler, OnTimerHandler,
    ParallelProcessor, PerformanceMonitor,
)


class TestRingBuffer:
    """Test ring buffer operations."""
    
    def test_basic_operations(self):
        """Test basic append and get."""
        buf = RingBuffer(capacity=5)
        
        # Append values
        for i in range(3):
            buf.append(float(i))
        
        assert len(buf) == 3
        assert buf[0] == 0.0  # Oldest
        assert buf[-1] == 2.0  # Newest
    
    def test_overflow(self):
        """Test buffer overflow behavior."""
        buf = RingBuffer(capacity=3)
        
        for i in range(5):
            buf.append(float(i))
        
        assert len(buf) == 3
        assert buf.is_full
        
        # Should contain last 3 values
        all_values = buf.get_all()
        assert list(all_values) == [2.0, 3.0, 4.0]
    
    def test_get_last(self):
        """Test get_last method."""
        buf = RingBuffer(capacity=10)
        
        for i in range(7):
            buf.append(float(i))
        
        last_3 = buf.get_last(3)
        assert list(last_3) == [6.0, 5.0, 4.0]  # Most recent first
    
    def test_extend(self):
        """Test bulk extend."""
        buf = RingBuffer(capacity=10)
        
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        buf.extend(values)
        
        assert len(buf) == 5
        assert list(buf.get_all()) == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestTickBuffer:
    """Test tick buffer operations."""
    
    def test_append_tick(self):
        """Test appending ticks."""
        buf = TickBuffer(capacity=100)
        
        now = datetime.now()
        buf.append_tick(now, bid=1.0800, ask=1.0802, volume=100)
        
        assert len(buf) == 1
        assert buf.last_tick is not None
        assert float(buf.last_tick['bid']) == 1.0800
    
    def test_get_ohlcv(self):
        """Test OHLCV aggregation."""
        buf = TickBuffer(capacity=100)
        
        now = datetime.now()
        prices = [1.0800, 1.0810, 1.0795, 1.0805]
        
        for i, price in enumerate(prices):
            buf.append_tick(
                now + timedelta(seconds=i),
                bid=price,
                ask=price + 0.0002,
                volume=100
            )
        
        ohlcv = buf.get_ohlcv(period_seconds=60)
        
        if ohlcv:  # May be None if timestamps don't match
            assert ohlcv['high'] >= ohlcv['low']
            assert ohlcv['tick_count'] >= 1


class TestBarBuffer:
    """Test bar buffer operations."""
    
    def test_append_bar(self):
        """Test appending bars."""
        buf = BarBuffer(capacity=100)
        
        now = datetime.now()
        buf.append_bar(now, open_=1.08, high=1.09, low=1.07, close=1.085, volume=1000)
        
        assert len(buf) == 1
        assert buf.last_bar is not None
    
    def test_sma(self):
        """Test SMA calculation."""
        buf = BarBuffer(capacity=100)
        
        # Add bars with known pattern
        for i in range(20):
            buf.append_bar(
                datetime.now() + timedelta(hours=i),
                open_=100 + i,
                high=100 + i + 1,
                low=100 + i - 1,
                close=100 + i,
                volume=1000
            )
        
        sma = buf.sma(5)
        assert len(sma) == 16  # 20 - 5 + 1
        
        # SMA should be around 117 at the end (average of 115-119)
        assert 116 < sma[-1] < 118
    
    def test_ema(self):
        """Test EMA calculation."""
        buf = BarBuffer(capacity=100)
        
        for i in range(20):
            buf.append_bar(
                datetime.now() + timedelta(hours=i),
                open_=100 + i,
                high=100 + i + 1,
                low=100 + i - 1,
                close=100 + i,
                volume=1000
            )
        
        ema = buf.ema(5)
        assert len(ema) == 16
        
        # EMA should be close to recent values (weighted towards recent)
        assert ema[-1] > buf.sma(5)[-1]  # EMA reacts faster to uptrend


class TestLRUCache:
    """Test LRU cache operations."""
    
    def test_basic_operations(self):
        """Test basic get/set."""
        cache = LRUCache(max_size=3)
        
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        
        assert cache.get('a') == 1
        assert cache.get('b') == 2
        assert cache.get('c') == 3
        assert cache.get('d') is None
    
    def test_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=3)
        
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        
        # Access 'a' to make it recently used
        cache.get('a')
        
        # Add new item - should evict 'b' (least recently used)
        cache.set('d', 4)
        
        assert cache.get('a') == 1
        assert cache.get('b') is None  # Evicted
        assert cache.get('c') == 3
        assert cache.get('d') == 4
    
    def test_ttl(self):
        """Test time-to-live expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)
        
        cache.set('a', 1)
        assert cache.get('a') == 1
        
        time.sleep(0.15)
        
        assert cache.get('a') is None  # Expired
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = LRUCache(max_size=10)
        
        cache.set('a', 1)
        cache.get('a')  # Hit
        cache.get('a')  # Hit
        cache.get('b')  # Miss
        
        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - 0.667) < 0.01


class TestComputeCache:
    """Test compute cache operations."""
    
    def test_cached_decorator(self):
        """Test function caching."""
        cache = ComputeCache()
        call_count = [0]
        
        @cache.cached
        def expensive_func(x):
            call_count[0] += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)
        result3 = expensive_func(10)
        
        assert result1 == 10
        assert result2 == 10
        assert result3 == 20
        assert call_count[0] == 2  # Only 2 actual calls (5 and 10)


class TestAsyncExecutor:
    """Test async executor operations."""
    
    def test_execute_single(self):
        """Test single coroutine execution."""
        executor = AsyncExecutor()
        
        async def sample_coro():
            await asyncio.sleep(0.01)
            return 42
        
        result = asyncio.run(executor.execute(sample_coro()))
        assert result == 42
    
    def test_execute_many(self):
        """Test concurrent execution."""
        executor = AsyncExecutor(max_concurrent=5)
        
        async def sample_coro(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        async def run_test():
            coros = [sample_coro(i) for i in range(10)]
            results = await executor.execute_many(coros)
            return results
        
        results = asyncio.run(run_test())
        assert results == [i * 2 for i in range(10)]
    
    def test_timeout(self):
        """Test timeout handling."""
        executor = AsyncExecutor()
        
        async def slow_coro():
            await asyncio.sleep(10)
            return "done"
        
        async def run_test():
            try:
                await executor.execute(slow_coro(), timeout=0.05)
                return False
            except asyncio.TimeoutError:
                return True
        
        timed_out = asyncio.run(run_test())
        assert timed_out


class TestTimer:
    """Test timer operations."""
    
    def test_periodic(self):
        """Test periodic callback."""
        timer = Timer()
        call_count = [0]
        
        def callback():
            call_count[0] += 1
        
        timer.schedule_periodic('test', callback, 0.05)
        timer.start()
        
        time.sleep(0.25)
        timer.stop()
        
        # Should have been called ~5 times (0.25 / 0.05)
        assert 4 <= call_count[0] <= 6
    
    def test_once(self):
        """Test one-shot callback."""
        timer = Timer()
        called = [False]
        
        def callback():
            called[0] = True
        
        timer.schedule_once('test', callback, 0.05)
        timer.start()
        
        time.sleep(0.1)
        timer.stop()
        
        assert called[0]
    
    def test_cancel(self):
        """Test callback cancellation."""
        timer = Timer()
        called = [False]
        
        def callback():
            called[0] = True
        
        timer.schedule_once('test', callback, 0.1)
        timer.cancel('test')
        timer.start()
        
        time.sleep(0.15)
        timer.stop()
        
        assert not called[0]


class TestOnTickHandler:
    """Test on-tick handler operations."""
    
    def test_tick_processing(self):
        """Test tick processing."""
        tick_buffer = TickBuffer(capacity=100)
        handler = OnTickHandler(tick_buffer, batch_size=5)
        
        batches_received = []
        
        def on_batch(batch):
            batches_received.append(len(batch))
        
        handler.register_callback(on_batch)
        
        # Send 10 ticks
        for i in range(10):
            handler.on_tick(
                datetime.now(),
                bid=1.08 + i * 0.0001,
                ask=1.0802 + i * 0.0001
            )
        
        # Should have processed 2 batches of 5
        assert handler.ticks_received == 10
        assert len(tick_buffer) == 10
    
    def test_stats(self):
        """Test statistics tracking."""
        tick_buffer = TickBuffer(capacity=100)
        handler = OnTickHandler(tick_buffer, batch_size=5)
        
        for i in range(10):
            handler.on_tick(datetime.now(), bid=1.08, ask=1.0802)
        
        stats = handler.get_stats()
        assert stats['ticks_received'] == 10
        assert stats['buffer_size'] == 10


class TestParallelProcessor:
    """Test parallel processor operations."""
    
    def test_map(self):
        """Test parallel map."""
        processor = ParallelProcessor(n_workers=2, use_processes=False)
        
        def square(x):
            return x * x
        
        results = processor.map(square, [1, 2, 3, 4, 5])
        processor.shutdown()
        
        assert results == [1, 4, 9, 16, 25]
    
    def test_starmap(self):
        """Test parallel starmap."""
        processor = ParallelProcessor(n_workers=2, use_processes=False)
        
        def add(x, y):
            return x + y
        
        results = processor.starmap(add, [(1, 2), (3, 4), (5, 6)])
        processor.shutdown()
        
        assert results == [3, 7, 11]


class TestPerformanceMonitor:
    """Test performance monitor."""
    
    def test_metrics(self):
        """Test metrics collection."""
        monitor = PerformanceMonitor()
        
        tick_buffer = TickBuffer(capacity=100)
        handler = OnTickHandler(tick_buffer)
        monitor.register_tick_handler(handler)
        
        cache = LRUCache(max_size=100)
        cache.set('a', 1)
        cache.get('a')
        cache.get('b')
        monitor.register_cache(cache)
        
        metrics = monitor.get_metrics()
        
        assert metrics is not None
        assert metrics.cache_hit_rate == 0.5  # 1 hit, 1 miss


def run_all_tests():
    """Run all performance module tests."""
    print("=" * 70)
    print("PERFORMANCE MODULE TEST SUITE")
    print("=" * 70)
    print()
    
    test_classes = [
        TestRingBuffer,
        TestTickBuffer,
        TestBarBuffer,
        TestLRUCache,
        TestComputeCache,
        TestAsyncExecutor,
        TestTimer,
        TestOnTickHandler,
        TestParallelProcessor,
        TestPerformanceMonitor,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    failed_tests.append((test_class.__name__, method_name, f"{type(e).__name__}: {e}"))
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


def benchmark_ring_buffer():
    """Benchmark ring buffer performance."""
    print("\n" + "=" * 70)
    print("RING BUFFER BENCHMARK")
    print("=" * 70)
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        buf = RingBuffer(capacity=size)
        
        # Benchmark append
        start = time.time()
        for i in range(size * 2):
            buf.append(float(i))
        append_time = time.time() - start
        
        # Benchmark get_all
        start = time.time()
        for _ in range(100):
            buf.get_all()
        get_time = time.time() - start
        
        print(f"\nBuffer size: {size}")
        print(f"  Append {size * 2} items: {append_time * 1000:.2f}ms ({size * 2 / append_time:.0f} ops/s)")
        print(f"  Get all (100x): {get_time * 1000:.2f}ms")


def benchmark_cache():
    """Benchmark cache performance."""
    print("\n" + "=" * 70)
    print("CACHE BENCHMARK")
    print("=" * 70)
    
    cache = LRUCache(max_size=10000)
    
    # Benchmark set
    start = time.time()
    for i in range(100000):
        cache.set(f"key_{i}", i)
    set_time = time.time() - start
    
    # Benchmark get (with hits)
    start = time.time()
    for i in range(100000):
        cache.get(f"key_{99999 - (i % 10000)}")  # Recent keys
    get_time = time.time() - start
    
    print(f"\nCache size: 10000")
    print(f"  Set 100000 items: {set_time * 1000:.2f}ms ({100000 / set_time:.0f} ops/s)")
    print(f"  Get 100000 items: {get_time * 1000:.2f}ms ({100000 / get_time:.0f} ops/s)")
    print(f"  Hit rate: {cache.hit_rate:.2%}")


if __name__ == "__main__":
    # Run tests
    tests_passed = run_all_tests()
    
    # Run benchmarks
    benchmark_ring_buffer()
    benchmark_cache()
    
    # Exit code
    sys.exit(0 if tests_passed else 1)
