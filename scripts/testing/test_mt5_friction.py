#!/usr/bin/env python3
"""
Test MT5 Bridge and Friction Model

Demonstrates:
1. Connecting to MT5 (auto-detects mode)
2. Getting live friction data
3. Physics-based friction estimation from bar data

Usage:
    # From WSL2 (needs bridge server on Windows)
    python scripts/test_mt5_friction.py

    # On Windows with MT5
    python scripts/test_mt5_friction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinetra import (
    MT5Bridge,
    FrictionModel,
    get_symbol_spec,
    compute_friction_series,
)
import pandas as pd
import numpy as np


def test_mt5_connection():
    """Test MT5 bridge connection."""
    print("=" * 60)
    print("Testing MT5 Bridge Connection")
    print("=" * 60)

    # Auto-detect mode
    bridge = MT5Bridge(mode="auto")
    connected = bridge.connect()

    print(f"\nMode: {bridge.mode}")
    print(f"Connected: {connected}")

    if connected:
        # Test getting symbol spec
        print("\n--- Symbol Specifications ---")
        for symbol in ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]:
            spec = bridge.get_symbol_spec(symbol)
            if spec:
                print(f"\n{symbol}:")
                print(f"  Digits: {spec.digits}")
                print(f"  Contract: {spec.contract_size}")
                print(f"  Spread: {spec.spread_typical:.1f} pts (min: {spec.spread_min:.1f}, max: {spec.spread_max:.1f})")
                print(f"  Swap: Long={spec.swap_long:.2f}, Short={spec.swap_short:.2f}")
            else:
                print(f"\n{symbol}: Not found")

        # Test live friction (only works with direct/bridge mode)
        if bridge.mode in ["direct", "bridge"]:
            print("\n--- Live Friction Data ---")
            friction = bridge.get_live_friction("EURUSD", lots=0.1)
            if friction:
                print(f"\nEURUSD (0.1 lot, 1 hour hold):")
                print(f"  Live Spread: {friction['spread_points']:.1f} pts")
                print(f"  Spread Cost: {friction['spread_pct']:.4f}%")
                print(f"  Commission:  {friction['commission_pct']:.4f}%")
                print(f"  Slippage:    {friction['slippage_pct']:.4f}%")
                print(f"  Swap:        {friction['swap_pct']:.4f}%")
                print(f"  ---")
                print(f"  TOTAL:       {friction['total_friction_pct']:.4f}%")
                print(f"  Spread Stress: {friction['spread_stress']:.2f}x")

        bridge.disconnect()

    return connected


def test_physics_friction():
    """Test physics-based friction estimation."""
    print("\n" + "=" * 60)
    print("Testing Physics-Based Friction Model")
    print("=" * 60)

    spec = get_symbol_spec("EURUSD")
    model = FrictionModel(spec)

    print(f"\nSymbol: {spec.symbol}")
    print(f"Base spread: {spec.spread_typical} points")

    # Test different market conditions
    conditions = [
        {"name": "Normal market", "vol": 0.01, "vol_base": 0.01, "volume": 1000, "vol_base_v": 1000},
        {"name": "Low liquidity", "vol": 0.01, "vol_base": 0.01, "volume": 200, "vol_base_v": 1000},
        {"name": "High volatility", "vol": 0.03, "vol_base": 0.01, "volume": 1000, "vol_base_v": 1000},
        {"name": "Stress (rollover-like)", "vol": 0.025, "vol_base": 0.01, "volume": 100, "vol_base_v": 1000},
    ]

    print("\n--- Friction Under Different Conditions ---")
    for cond in conditions:
        friction = model.calculate_friction(
            price=1.1000,
            position_size_lots=0.1,
            volatility=cond["vol"],
            volatility_baseline=cond["vol_base"],
            volume=cond["volume"],
            volume_baseline=cond["vol_base_v"],
            high=1.1010,
            low=1.0990,
            holding_days=1/24,  # 1 hour
        )

        print(f"\n{cond['name']}:")
        print(f"  Total Friction: {friction['total_friction_pct']:.4f}%")
        print(f"  Vol Stress:    {friction['volatility_stress']:.2f}x")
        print(f"  Liq Stress:    {friction['liquidity_stress']:.2f}x")
        print(f"  Spread Stress: {friction['spread_stress']:.2f}x")

        # Check tradeability
        is_ok, reason = model.is_tradeable(
            expected_return_pct=0.1,  # 0.1% expected return
            friction=friction,
        )
        status = "TRADEABLE" if is_ok else "BLOCKED"
        print(f"  Status: {status} - {reason}")


def test_friction_series():
    """Test friction calculation over a price series."""
    print("\n" + "=" * 60)
    print("Testing Friction Series Calculation")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n = 100

    # Simulate normal market with a stress period
    volume_normal = 1000
    volume_data = np.ones(n) * volume_normal
    volume_data[40:50] = 100  # Low liquidity period

    volatility_normal = 0.01
    close = 1.1000 + np.cumsum(np.random.randn(n) * volatility_normal * 0.01)

    # Create DataFrame
    dates = pd.date_range("2024-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "Open": close - np.random.rand(n) * 0.001,
        "High": close + np.random.rand(n) * 0.002,
        "Low": close - np.random.rand(n) * 0.002,
        "Close": close,
        "Volume": volume_data,
    }, index=dates)

    # Compute friction series
    friction_df = compute_friction_series(df, "EURUSD", position_size_lots=0.1)

    print(f"\nFriction series computed for {len(friction_df)} bars")
    print(f"\nNormal period (bars 0-39):")
    print(f"  Avg Friction: {friction_df['total_friction_pct'][:40].mean():.4f}%")
    print(f"  Avg Liquidity Stress: {friction_df['liquidity_stress'][:40].mean():.2f}x")

    print(f"\nStress period (bars 40-49):")
    print(f"  Avg Friction: {friction_df['total_friction_pct'][40:50].mean():.4f}%")
    print(f"  Avg Liquidity Stress: {friction_df['liquidity_stress'][40:50].mean():.2f}x")

    print(f"\nRecovery period (bars 50+):")
    print(f"  Avg Friction: {friction_df['total_friction_pct'][50:].mean():.4f}%")
    print(f"  Avg Liquidity Stress: {friction_df['liquidity_stress'][50:].mean():.2f}x")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KINETRA FRICTION MODEL TEST")
    print("Physics-based cost estimation")
    print("=" * 60)

    # Test MT5 connection
    connected = test_mt5_connection()

    # Test physics-based friction (works offline)
    test_physics_friction()

    # Test friction series
    test_friction_series()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    if not connected or connected and MT5Bridge(mode="auto").mode == "offline":
        print("\nTo get LIVE friction data:")
        print("  1. On Windows, run: python mt5_bridge_server.py")
        print("  2. Make sure MT5 terminal is open and connected")
        print("  3. Run this script again")
    print("=" * 60)


if __name__ == "__main__":
    main()
