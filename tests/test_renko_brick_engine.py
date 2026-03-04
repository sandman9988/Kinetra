"""
Renko Brick Engine Tests
=========================

Tests for brick construction and incremental brick building.

Usage:
    python -m pytest tests/test_renko_brick_engine.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.renko.brick_engine import (
    IncrementalRenkoBuilder,
    bricks_per_day,
    build_renko,
)


class TestBuildRenko(unittest.TestCase):
    """Test batch Renko brick construction."""

    def test_single_brick_formation(self):
        """Test basic brick formation."""
        closes = pd.Series(
            [100.0, 101.0, 102.0],  # 2-point move, brick_size=1.0
            index=pd.date_range("2024-01-01", periods=3, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        self.assertIsInstance(bricks, pd.DataFrame)
        self.assertGreater(len(bricks), 0)
        self.assertIn("direction", bricks.columns)
        self.assertIn("brick_close", bricks.columns)

    def test_brick_direction_up(self):
        """Test upward brick formation."""
        closes = pd.Series(
            [100.0, 102.0],  # Up 2 points
            index=pd.date_range("2024-01-01", periods=2, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        if len(bricks) > 0:
            self.assertEqual(bricks.iloc[0]["direction"], 1)

    def test_brick_direction_down(self):
        """Test downward brick formation."""
        closes = pd.Series(
            [100.0, 98.0],  # Down 2 points
            index=pd.date_range("2024-01-01", periods=2, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        if len(bricks) > 0:
            self.assertEqual(bricks.iloc[0]["direction"], -1)

    def test_no_brick_below_threshold(self):
        """Test no brick formed if price move below threshold."""
        closes = pd.Series(
            [100.0, 100.5],  # Only 0.5 move, below brick_size=1.0
            index=pd.date_range("2024-01-01", periods=2, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        self.assertEqual(len(bricks), 0)

    def test_multiple_bricks_same_direction(self):
        """Test multiple bricks in same direction."""
        closes = pd.Series(
            [100.0, 105.0],  # Up 5 points = 5 bricks
            index=pd.date_range("2024-01-01", periods=2, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        self.assertGreaterEqual(len(bricks), 4)  # At least 4-5 bricks

    def test_brick_reversal(self):
        """Test brick reversal on price change direction."""
        closes = pd.Series(
            [100.0, 102.0, 100.0],  # Up then down
            index=pd.date_range("2024-01-01", periods=3, freq="min"),
        )

        bricks = build_renko(closes, brick_size=1.0)

        # Should have both up and down bricks
        directions = bricks["direction"].tolist()
        self.assertIn(1, directions)
        self.assertIn(-1, directions)

    def test_empty_series(self):
        """Test empty price series."""
        closes = pd.Series([], dtype=float)

        bricks = build_renko(closes, brick_size=1.0)

        self.assertEqual(len(bricks), 0)

    def test_single_price(self):
        """Test single price point."""
        closes = pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1, freq="min"))

        bricks = build_renko(closes, brick_size=1.0)

        self.assertEqual(len(bricks), 0)


class TestIncrementalRenkoBuilder(unittest.TestCase):
    """Test incremental Renko brick builder."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        self.assertEqual(builder._brick_size, 1.0)
        self.assertIsNone(builder._last_ref_price)

    def test_first_price_sets_reference(self):
        """Test first price sets reference."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        bricks = builder.update(100.0, pd.Timestamp("2024-01-01"))

        self.assertEqual(len(bricks), 0)  # No brick yet
        self.assertEqual(builder._last_ref_price, 100.0)

    def test_incremental_brick_formation(self):
        """Test brick formation with incremental updates."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        # Initial price
        builder.update(100.0, pd.Timestamp("2024-01-01 00:00"))

        # Move up 2 points
        bricks = builder.update(102.0, pd.Timestamp("2024-01-01 00:01"))

        self.assertGreater(len(bricks), 0)
        self.assertEqual(bricks[0][2], 1)  # Direction up

    def test_no_brick_on_small_move(self):
        """Test no brick on small price move."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        builder.update(100.0, pd.Timestamp("2024-01-01 00:00"))
        bricks = builder.update(100.5, pd.Timestamp("2024-01-01 00:01"))

        self.assertEqual(len(bricks), 0)

    def test_multiple_updates_accumulate(self):
        """Test multiple updates accumulate bricks."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        all_bricks = []
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]

        for i, price in enumerate(prices):
            ts = pd.Timestamp(f"2024-01-01 00:{i:02d}")
            bricks = builder.update(price, ts)
            all_bricks.extend(bricks)

        # Should have multiple bricks
        self.assertGreater(len(all_bricks), 3)

    def test_brick_timestamp(self):
        """Test brick timestamp is from triggering bar."""
        builder = IncrementalRenkoBuilder(brick_size=1.0)

        builder.update(100.0, pd.Timestamp("2024-01-01 00:00"))
        bricks = builder.update(102.0, pd.Timestamp("2024-01-01 00:05"))

        self.assertGreater(len(bricks), 0)
        # Timestamp should be from the update that triggered the brick
        self.assertEqual(bricks[0][3], pd.Timestamp("2024-01-01 00:05"))


class TestBricksPerDay(unittest.TestCase):
    """Test bricks per day calculation."""

    def test_typical_xauusd(self):
        """Test typical XAUUSD bricks per day."""
        closes = pd.Series(
            np.random.randn(1440).cumsum() * 0.5 + 2000,  # 1 day of M1
            index=pd.date_range("2024-01-01", periods=1440, freq="min"),
        )

        bpd = bricks_per_day(closes, brick_size=1.0)

        self.assertGreater(bpd, 0)
        self.assertIsInstance(bpd, float)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        closes = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2, freq="min"))

        bpd = bricks_per_day(closes, brick_size=1.0)

        self.assertEqual(bpd, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
