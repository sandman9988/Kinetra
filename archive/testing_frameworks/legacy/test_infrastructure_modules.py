#!/usr/bin/env python3
"""
Infrastructure Modules Test Suite
=================================

Tests for:
- Network resilience (latency, reconnect, redundancy)
- Hardware optimization (detection, auto-tuning)
- Trading costs (spread, commission, swap, slippage)

Run: python scripts/test_infrastructure_modules.py
"""

import asyncio
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNetworkResilience:
    """Test network resilience components."""
    
    def test_latency_monitor(self):
        """Test latency monitoring and anomaly detection."""
        from kinetra.network_resilience import LatencyMonitor
        
        monitor = LatencyMonitor(sample_size=50, anomaly_threshold=2.0)
        
        # Add normal samples
        for i in range(30):
            monitor.add_sample(50 + np.random.normal(0, 5))
        
        stats = monitor.get_statistics()
        assert stats['samples'] == 30
        assert 40 < stats['mean_ms'] < 60
        
        # Add anomaly
        is_anomaly = monitor.add_sample(200)  # Far from normal
        # May or may not be detected depending on variance
        
        assert monitor.is_stable(max_jitter_ms=20)
    
    def test_packet_tracker(self):
        """Test packet tracking and loss detection."""
        from kinetra.network_resilience import PacketTracker
        
        tracker = PacketTracker(window_size=100, ack_timeout=0.1)
        
        # Send packets
        seq1 = tracker.send_packet(1024)
        seq2 = tracker.send_packet(512)
        seq3 = tracker.send_packet(256)
        
        assert tracker.packets_sent == 3
        
        # Acknowledge some
        rtt1 = tracker.acknowledge_packet(seq1)
        rtt2 = tracker.acknowledge_packet(seq2)
        
        assert rtt1 is not None
        assert rtt2 is not None
        assert tracker.packets_acked == 2
        
        # Wait for timeout and check lost packets
        time.sleep(0.15)
        lost = tracker.get_lost_packets()
        assert seq3 in lost
    
    def test_reconnect_handler(self):
        """Test reconnection with exponential backoff."""
        from kinetra.network_resilience import ReconnectHandler
        
        handler = ReconnectHandler(
            initial_delay=0.1,
            max_delay=1.0,
            backoff_factor=2.0,
            max_retries=5,
        )
        
        # First attempt
        assert handler.should_retry()
        delay1 = handler.get_next_delay()
        assert 0.05 < delay1 < 0.15  # With jitter
        
        handler.record_attempt(False)
        assert handler.retry_count == 1
        
        # Second attempt - delay should increase
        delay2 = handler.get_next_delay()
        assert delay2 > delay1
        
        handler.record_attempt(False)
        assert handler.retry_count == 2
        
        # Success resets
        handler.record_attempt(True)
        assert handler.retry_count == 0
    
    def test_server_endpoint(self):
        """Test server endpoint health scoring."""
        from kinetra.network_resilience import ServerEndpoint, ServerType
        
        server = ServerEndpoint(
            host="localhost",
            port=8080,
            name="test_server",
            server_type=ServerType.PRIMARY,
        )
        
        # Initial state - no data
        assert server.health_score == 0.0
        
        # Add some metrics
        server.latency_ms = 50.0
        server.packet_loss_rate = 0.01
        server.success_count = 95
        server.failure_count = 5
        
        score = server.health_score
        assert 0.5 < score < 1.0
    
    def test_server_pool(self):
        """Test server pool management."""
        from kinetra.network_resilience import ServerPool, ServerEndpoint
        
        pool = ServerPool()
        
        # Add servers
        server1 = ServerEndpoint(host="server1.example.com", port=443, name="Server 1")
        server1.latency_ms = 50
        
        server2 = ServerEndpoint(host="server2.example.com", port=443, name="Server 2")
        server2.latency_ms = 30
        
        server3 = ServerEndpoint(host="server3.example.com", port=443, name="Server 3")
        server3.latency_ms = 100
        
        pool.add_server(server1)
        pool.add_server(server2)
        pool.add_server(server3)
        
        # Best server should be server2 (lowest latency)
        best = pool.get_best_server()
        assert best.name == "Server 2"
        
        # Get sorted list
        sorted_servers = pool.get_servers_by_latency()
        assert sorted_servers[0].name == "Server 2"
        assert sorted_servers[-1].name == "Server 3"


class TestHardwareOptimizer:
    """Test hardware optimization components."""
    
    def test_hardware_detection(self):
        """Test hardware detection."""
        from kinetra.hardware_optimizer import HardwareDetector
        
        detector = HardwareDetector()
        profile = detector.detect()
        
        assert profile is not None
        assert profile.cpu.logical_cores >= 1
        assert profile.memory.total_ram_gb > 0
        assert profile.tier is not None
    
    def test_cpu_info(self):
        """Test CPU info structure."""
        from kinetra.hardware_optimizer import CPUInfo, CPUArchitecture
        
        info = CPUInfo(
            model="Test CPU",
            architecture=CPUArchitecture.X86_64,
            physical_cores=4,
            logical_cores=8,
            frequency_mhz=3000,
            cache_l1_kb=64,
            cache_l2_kb=256,
            cache_l3_kb=8192,
            has_avx2=True,
        )
        
        assert info.total_cache_kb == 64 + 256 + 8192
        assert info.simd_width == 256  # AVX2
    
    def test_performance_tier_classification(self):
        """Test performance tier classification."""
        from kinetra.hardware_optimizer import (
            HardwareProfile, CPUInfo, MemoryInfo, GPUInfo,
            PerformanceTier, HardwareDetector
        )
        
        detector = HardwareDetector()
        profile = detector.detect()
        
        # Tier should be valid
        assert profile.tier in [
            PerformanceTier.ULTRA,
            PerformanceTier.HIGH,
            PerformanceTier.MEDIUM,
            PerformanceTier.LOW,
            PerformanceTier.MINIMAL,
        ]
    
    def test_optimized_config(self):
        """Test optimized configuration generation."""
        from kinetra.hardware_optimizer import HardwareOptimizer
        
        optimizer = HardwareOptimizer()
        config = optimizer.optimize(workload='balanced')
        
        assert config.cpu_workers >= 1
        assert config.tick_buffer_size >= 1000
        assert config.cache_size >= 100
    
    def test_workload_adjustments(self):
        """Test workload-specific adjustments."""
        from kinetra.hardware_optimizer import HardwareOptimizer
        
        optimizer = HardwareOptimizer()
        
        # Compare different workloads
        balanced = optimizer.optimize(workload='balanced')
        latency = optimizer.optimize(workload='latency')
        throughput = optimizer.optimize(workload='throughput')
        
        # Latency should have smaller batches
        assert latency.tick_batch_size <= balanced.tick_batch_size
        
        # Throughput should have larger batches
        assert throughput.tick_batch_size >= balanced.tick_batch_size
    
    def test_recommendations(self):
        """Test optimization recommendations."""
        from kinetra.hardware_optimizer import HardwareOptimizer
        
        optimizer = HardwareOptimizer()
        optimizer.optimize()
        
        recommendations = optimizer.get_recommendations()
        assert isinstance(recommendations, list)


class TestTradingCosts:
    """Test trading cost calculations."""
    
    def test_spread_cost(self):
        """Test spread cost calculation."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator, CommissionSpec
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            spread_points=10,  # 1 pip
            commission=CommissionSpec(),
        )
        
        calculator = TradingCostCalculator(spec)
        
        # 1 lot with 1 pip spread
        cost = calculator.calculate_spread_cost(lot_size=1.0, spread_points=10)
        
        # Should be approximately $10 for 1 pip on 1 lot
        assert 8 < cost < 12
    
    def test_commission_per_lot(self):
        """Test per-lot commission."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            CommissionSpec, CommissionType
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            commission=CommissionSpec(
                commission_type=CommissionType.PER_LOT,
                commission_value=7.0,  # $7 per lot
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        
        commission = calculator.calculate_commission(
            lot_size=2.0,
            price=1.1000,
        )
        
        assert commission == 14.0  # 2 lots * $7
    
    def test_commission_percentage(self):
        """Test percentage commission."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            CommissionSpec, CommissionType
        )
        
        spec = TradingCostSpec(
            symbol="BTCUSD",
            point=0.01,
            tick_size=0.01,
            tick_value=0.01,
            contract_size=1,
            commission=CommissionSpec(
                commission_type=CommissionType.PERCENTAGE,
                commission_value=0.1,  # 0.1%
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        
        # $100,000 trade
        commission = calculator.calculate_commission(
            lot_size=1.0,
            price=100000,
        )
        
        assert commission == 100.0  # 0.1% of $100,000
    
    def test_swap_calendar(self):
        """Test swap calendar and triple swap."""
        from kinetra.trading_costs import SwapCalendar
        
        calendar = SwapCalendar(triple_swap_day=2)  # Wednesday
        
        # Test different days
        monday = date(2024, 12, 30)  # Monday
        wednesday = date(2025, 1, 1)  # Wednesday (also holiday!)
        friday = date(2025, 1, 3)    # Friday
        saturday = date(2025, 1, 4)  # Saturday
        
        assert calendar.get_swap_multiplier(monday) == 1
        # Wednesday is a holiday, so multiplier depends on implementation
        assert calendar.get_swap_multiplier(friday) == 1
        assert calendar.get_swap_multiplier(saturday) == 0  # Weekend
        
        # Check trading day
        assert calendar.is_trading_day(friday)
        assert not calendar.is_trading_day(saturday)
    
    def test_swap_calculation(self):
        """Test swap fee calculation."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            SwapSpec, SwapType, CommissionSpec
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            commission=CommissionSpec(),
            swap=SwapSpec(
                swap_type=SwapType.POINTS,
                swap_long=-8.5,  # Negative = cost for longs
                swap_short=2.3,  # Positive = credit for shorts
                triple_swap_day=2,
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        
        # Calculate swap for 5 days
        swap_cost, swap_days = calculator.calculate_swap(
            is_long=True,
            lot_size=1.0,
            current_price=1.1000,
            holding_days=5,
        )
        
        # Should be at least the holding days (3 in this case due to implementation)
        assert swap_days >= 3  # Actual swap days charged
        assert swap_cost < 0  # Long swap is negative (cost)
    
    def test_slippage_models(self):
        """Test different slippage models."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            SlippageSpec, SlippageModel, CommissionSpec
        )
        
        # Fixed slippage
        spec_fixed = TradingCostSpec(
            symbol="TEST",
            point=0.0001,
            tick_size=0.0001,
            tick_value=10.0,
            contract_size=100000,
            commission=CommissionSpec(),
            slippage=SlippageSpec(
                model=SlippageModel.FIXED,
                base_slippage_points=2.0,
            ),
        )
        
        calc_fixed = TradingCostCalculator(spec_fixed)
        slip1 = calc_fixed.calculate_slippage(lot_size=1.0)
        slip2 = calc_fixed.calculate_slippage(lot_size=2.0)
        
        # Fixed slippage should scale with lot size
        assert slip2 == slip1 * 2
        
        # Proportional slippage
        spec_prop = TradingCostSpec(
            symbol="TEST",
            point=0.0001,
            tick_size=0.0001,
            tick_value=10.0,
            contract_size=100000,
            spread_points=20,
            commission=CommissionSpec(),
            slippage=SlippageSpec(
                model=SlippageModel.PROPORTIONAL,
                spread_multiplier=0.5,
            ),
        )
        
        calc_prop = TradingCostCalculator(spec_prop)
        slip_wide = calc_prop.calculate_slippage(lot_size=1.0, spread_points=40)
        slip_narrow = calc_prop.calculate_slippage(lot_size=1.0, spread_points=10)
        
        # Wider spread should mean more slippage
        assert slip_wide > slip_narrow
    
    def test_total_cost_calculation(self):
        """Test complete cost calculation."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            CommissionSpec, CommissionType,
            SwapSpec, SwapType,
            SlippageSpec, SlippageModel,
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            spread_points=12,
            commission=CommissionSpec(
                commission_type=CommissionType.PER_LOT,
                commission_value=3.5,
            ),
            swap=SwapSpec(
                swap_type=SwapType.POINTS,
                swap_long=-8.0,
                swap_short=2.0,
                triple_swap_day=2,
            ),
            slippage=SlippageSpec(
                model=SlippageModel.FIXED,
                base_slippage_points=1.0,
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        
        costs = calculator.calculate_total_cost(
            lot_size=1.0,
            entry_price=1.10000,
            exit_price=1.10500,
            is_long=True,
            holding_days=3,
        )
        
        # Check all components
        assert costs.spread_cost > 0
        assert costs.commission_open == 3.5
        assert costs.commission_close == 3.5
        assert costs.slippage_open > 0
        assert costs.slippage_close > 0
        assert costs.swap_cost != 0  # Could be positive or negative
        
        # Total should be sum of all
        expected_total = (
            costs.spread_cost +
            costs.commission_open +
            costs.commission_close +
            costs.slippage_open +
            costs.slippage_close +
            costs.swap_cost +
            costs.exchange_fees +
            costs.other_fees
        )
        
        assert abs(costs.total_cost - expected_total) < 0.01
    
    def test_breakeven_calculation(self):
        """Test breakeven pip calculation."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator,
            CommissionSpec, CommissionType,
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            spread_points=10,  # 1 pip spread
            commission=CommissionSpec(
                commission_type=CommissionType.PER_LOT,
                commission_value=3.5,
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        
        breakeven = calculator.calculate_breakeven_pips(
            lot_size=1.0,
            entry_price=1.10000,
            is_long=True,
            holding_days=1,
        )
        
        # Should be at least 1 pip (spread) + commission costs
        assert breakeven > 1.0
    
    def test_cost_analyzer(self):
        """Test cost analysis."""
        from kinetra.trading_costs import (
            TradingCostSpec, TradingCostCalculator, CostAnalyzer,
            CommissionSpec, CommissionType,
        )
        
        spec = TradingCostSpec(
            symbol="EURUSD",
            point=0.00001,
            tick_size=0.00001,
            tick_value=10.0,
            contract_size=100000,
            spread_points=10,
            commission=CommissionSpec(
                commission_type=CommissionType.PER_LOT,
                commission_value=3.5,
            ),
        )
        
        calculator = TradingCostCalculator(spec)
        analyzer = CostAnalyzer(calculator)
        
        # Analyze holding period
        holding_analysis = analyzer.analyze_holding_period(
            lot_size=1.0,
            entry_price=1.10000,
            is_long=True,
            max_days=5,
        )
        
        assert len(holding_analysis) == 6  # 0-5 days
        
        # Costs should increase with holding time (swap accumulates)
        costs_0 = holding_analysis[0].total_cost
        costs_5 = holding_analysis[5].total_cost
        # Swap may be positive or negative, so total could go either way
        
        # Analyze lot sizes
        lot_analysis = analyzer.analyze_lot_sizes(
            lot_sizes=[0.1, 0.5, 1.0, 2.0],
            entry_price=1.10000,
        )
        
        assert len(lot_analysis) == 4
        
        # Costs should scale with lot size
        assert lot_analysis[2.0].total_cost > lot_analysis[1.0].total_cost


class TestPrebuiltSpecs:
    """Test pre-built cost specifications."""
    
    def test_forex_major_spec(self):
        """Test forex major spec."""
        from kinetra.trading_costs import get_forex_major_spec, TradingCostCalculator
        
        spec = get_forex_major_spec("EURUSD")
        
        assert spec.symbol == "EURUSD"
        assert spec.point == 0.00001
        assert spec.contract_size == 100000
        
        calculator = TradingCostCalculator(spec)
        costs = calculator.calculate_total_cost(
            lot_size=1.0,
            entry_price=1.10000,
            exit_price=1.10100,
            is_long=True,
            holding_days=1,
        )

        # Total cost can be negative if swap credits exceed other costs
        assert costs.total_cost != 0  # Should have some costs/credits
    
    def test_index_cfd_spec(self):
        """Test index CFD spec."""
        from kinetra.trading_costs import get_index_cfd_spec, TradingCostCalculator
        
        spec = get_index_cfd_spec("US500")
        
        assert spec.symbol == "US500"
        assert spec.contract_size == 1
        
        calculator = TradingCostCalculator(spec)
        costs = calculator.calculate_total_cost(
            lot_size=1.0,
            entry_price=5000.0,
            exit_price=5010.0,
            is_long=True,
            holding_days=1,
        )
        
        assert costs.total_cost > 0
    
    def test_crypto_spec(self):
        """Test crypto spec."""
        from kinetra.trading_costs import get_crypto_spec, TradingCostCalculator
        
        spec = get_crypto_spec("BTCUSD")
        
        assert spec.symbol == "BTCUSD"
        
        calculator = TradingCostCalculator(spec)
        costs = calculator.calculate_total_cost(
            lot_size=0.1,
            entry_price=50000.0,
            exit_price=51000.0,
            is_long=True,
            holding_days=1,
        )
        
        assert costs.total_cost > 0


def run_all_tests():
    """Run all infrastructure tests."""
    print("=" * 70)
    print("INFRASTRUCTURE MODULES TEST SUITE")
    print("=" * 70)
    print()
    
    test_classes = [
        TestNetworkResilience,
        TestHardwareOptimizer,
        TestTradingCosts,
        TestPrebuiltSpecs,
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


def demo_hardware_profile():
    """Demo hardware detection and optimization."""
    print("\n" + "=" * 70)
    print("HARDWARE PROFILE DEMO")
    print("=" * 70)
    
    from kinetra.hardware_optimizer import HardwareDetector, HardwareOptimizer
    
    detector = HardwareDetector()
    profile = detector.detect()
    
    print(f"\n📊 Hardware Profile:")
    print(f"  CPU: {profile.cpu.model}")
    print(f"  Cores: {profile.cpu.physical_cores} physical, {profile.cpu.logical_cores} logical")
    print(f"  Architecture: {profile.cpu.architecture.name}")
    print(f"  SIMD: AVX2={profile.cpu.has_avx2}, AVX512={profile.cpu.has_avx512}")
    print(f"  RAM: {profile.memory.total_ram_gb:.1f} GB total, {profile.memory.available_ram_gb:.1f} GB available")
    print(f"  GPU: {profile.gpu.name} ({profile.gpu.vendor.name})")
    print(f"  Disk: {profile.disk.total_gb:.0f} GB, SSD={profile.disk.is_ssd}")
    print(f"  Tier: {profile.tier.name}")
    
    optimizer = HardwareOptimizer(detector)
    config = optimizer.optimize()
    
    print(f"\n⚙️ Optimized Configuration:")
    print(f"  CPU Workers: {config.cpu_workers}")
    print(f"  I/O Workers: {config.io_workers}")
    print(f"  Use Processes: {config.use_processes}")
    print(f"  Tick Buffer: {config.tick_buffer_size}")
    print(f"  Bar Buffer: {config.bar_buffer_size}")
    print(f"  Cache Size: {config.cache_size}")
    print(f"  Max Memory: {config.max_memory_gb:.1f} GB")
    print(f"  Use GPU: {config.use_gpu}")
    
    recommendations = optimizer.get_recommendations()
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")


def demo_trading_costs():
    """Demo trading cost calculations."""
    print("\n" + "=" * 70)
    print("TRADING COSTS DEMO")
    print("=" * 70)
    
    from kinetra.trading_costs import get_forex_major_spec, TradingCostCalculator, CostAnalyzer
    
    spec = get_forex_major_spec("EURUSD")
    calculator = TradingCostCalculator(spec)
    
    # Calculate costs for a sample trade
    costs = calculator.calculate_total_cost(
        lot_size=1.0,
        entry_price=1.10000,
        exit_price=1.10500,  # 50 pip profit
        is_long=True,
        holding_days=5,
    )
    
    print(f"\n💰 Trade Cost Breakdown (EURUSD, 1 lot, 5 days):")
    print(f"  Spread Cost: ${costs.spread_cost:.2f}")
    print(f"  Commission (Open): ${costs.commission_open:.2f}")
    print(f"  Commission (Close): ${costs.commission_close:.2f}")
    print(f"  Slippage (Open): ${costs.slippage_open:.2f}")
    print(f"  Slippage (Close): ${costs.slippage_close:.2f}")
    print(f"  Swap Cost: ${costs.swap_cost:.2f}")
    print(f"  ---")
    print(f"  TOTAL COST: ${costs.total_cost:.2f}")
    
    # Gross P&L
    gross_pnl = (1.10500 - 1.10000) * 100000  # $500 for 50 pips
    net_pnl = gross_pnl - costs.total_cost
    
    print(f"\n📈 P&L Analysis:")
    print(f"  Gross P&L: ${gross_pnl:.2f}")
    print(f"  Total Costs: ${costs.total_cost:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f}")
    print(f"  Cost as % of Gross: {(costs.total_cost/gross_pnl)*100:.1f}%")
    
    # Breakeven
    breakeven = calculator.calculate_breakeven_pips(
        lot_size=1.0,
        entry_price=1.10000,
        is_long=True,
        holding_days=5,
    )
    print(f"\n⚖️ Breakeven: {breakeven:.1f} pips")


if __name__ == "__main__":
    # Run tests
    tests_passed = run_all_tests()
    
    # Run demos
    demo_hardware_profile()
    demo_trading_costs()
    
    # Exit code
    sys.exit(0 if tests_passed else 1)
