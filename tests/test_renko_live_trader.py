"""
Renko Live Trader Tests
========================

Tests for live trading infrastructure.

Usage:
    python -m pytest tests/test_renko_live_trader.py -v
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.renko.live_trader import (
    LiveTrade,
    OrderResult,
    PERGate,
    TradeDirection,
)


class TestTradeDirection(unittest.TestCase):
    """Test TradeDirection enum."""

    def test_long_value(self):
        """Test LONG direction value."""
        self.assertEqual(TradeDirection.LONG.value, 1)

    def test_short_value(self):
        """Test SHORT direction value."""
        self.assertEqual(TradeDirection.SHORT.value, -1)


class TestOrderResult(unittest.TestCase):
    """Test OrderResult dataclass."""

    def test_success_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="POS-12345",
            filled_price=2000.5,
            filled_lots=0.01,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "POS-12345")
        self.assertEqual(result.filled_price, 2000.5)
        self.assertEqual(result.filled_lots, 0.01)
        self.assertIsNone(result.error)

    def test_failure_result(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            error="Fill timeout",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Fill timeout")
        self.assertIsNone(result.order_id)

    def test_result_with_raw(self):
        """Test result with raw data."""
        result = OrderResult(
            success=True,
            order_id="POS-12345",
            raw={"client_order_id": "COID-123", "extra": "data"},
        )

        self.assertEqual(result.raw["client_order_id"], "COID-123")


class TestLiveTrade(unittest.TestCase):
    """Test LiveTrade dataclass."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )

        self.assertEqual(trade.trade_id, "POS-12345")
        self.assertEqual(trade.symbol, "XAUUSD")
        self.assertEqual(trade.direction, TradeDirection.LONG)
        self.assertEqual(trade.entry_price, 2000.0)
        self.assertIsNone(trade.exit_price)  # Not closed yet

    def test_trade_close(self):
        """Test trade close."""
        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )

        trade.close(
            exit_price=2010.0,
            exit_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            exit_reason="colour_change",
            friction_usd=0.50,
            usd_per_point=1.0,
        )

        self.assertEqual(trade.exit_price, 2010.0)
        self.assertEqual(trade.exit_reason, "colour_change")
        self.assertEqual(trade.pnl, 10.0)  # (2010 - 2000) * 0.01 * 100
        self.assertEqual(trade.net_pnl, 9.50)  # PnL - friction

    def test_short_trade_pnl(self):
        """Test short trade PnL calculation."""
        trade = LiveTrade(
            trade_id="POS-12346",
            symbol="XAUUSD",
            direction=TradeDirection.SHORT,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )

        trade.close(
            exit_price=1990.0,  # Price went down
            exit_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            exit_reason="colour_change",
            friction_usd=0.50,
            usd_per_point=1.0,
        )

        self.assertGreater(trade.pnl, 0)  # Profit when price drops for short

    def test_trade_duration(self):
        """Test trade duration calculation."""
        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )

        trade.close(
            exit_price=2010.0,
            exit_time=pd.Timestamp("2024-01-01 13:30:00", tz="UTC"),
            exit_reason="colour_change",
            friction_usd=0.50,
            usd_per_point=1.0,
        )

        # Duration should be 1.5 hours
        expected_duration = 1.5
        self.assertAlmostEqual(trade.duration_hours, expected_duration, places=2)


class TestPERGate(unittest.TestCase):
    """Test PER (Performance Evaluation and Readiness) gates."""

    def test_simulated_gate(self):
        """Test SIMULATED gate."""
        gate = PERGate.SIMULATED

        self.assertEqual(gate.max_lot_ceiling, float("inf"))
        self.assertEqual(gate.max_dd_pct, float("inf"))

    def test_micro_gate(self):
        """Test MICRO gate."""
        gate = PERGate.MICRO

        self.assertEqual(gate.max_lot_ceiling, 0.01)
        self.assertEqual(gate.max_dd_pct, 0.03)

    def test_small_gate(self):
        """Test SMALL gate."""
        gate = PERGate.SMALL

        self.assertEqual(gate.max_lot_ceiling, 0.10)
        self.assertEqual(gate.max_dd_pct, 0.05)

    def test_full_gate(self):
        """Test FULL gate."""
        gate = PERGate.FULL

        self.assertEqual(gate.max_lot_ceiling, 50.0)
        self.assertEqual(gate.max_dd_pct, 0.10)


class TestLiveTradeEdgeCases(unittest.TestCase):
    """Test LiveTrade edge cases."""

    def test_trade_with_zero_lots(self):
        """Test trade with zero lots."""
        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.0,  # Zero lots
            target_risk_usd=100.0,
        )

        trade.close(
            exit_price=2010.0,
            exit_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            exit_reason="test",
            friction_usd=0.50,
            usd_per_point=1.0,
        )

        self.assertEqual(trade.pnl, 0.0)  # No PnL with zero lots

    def test_trade_with_negative_pnl(self):
        """Test losing trade."""
        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )

        trade.close(
            exit_price=1990.0,  # Price went down
            exit_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            exit_reason="stop",
            friction_usd=0.50,
            usd_per_point=1.0,
        )

        self.assertLess(trade.pnl, 0)  # Negative PnL
        self.assertLess(trade.net_pnl, 0)  # Negative net PnL


if __name__ == "__main__":
    unittest.main(verbosity=2)
