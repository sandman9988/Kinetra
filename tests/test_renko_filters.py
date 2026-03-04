"""
Renko Filters Tests
===================

Tests for entry evaluation and statistical filters.

Usage:
    python -m pytest tests/test_renko_filters.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.renko.filters import (
    evaluate_entry,
    flip_rate,
    markov_stickiness,
)


class TestFlipRate(unittest.TestCase):
    """Test flip rate calculation."""

    def test_no_flips(self):
        """Test zero flips returns 0."""
        directions = np.array([1, 1, 1, 1, 1])  # All same direction

        fr = flip_rate(directions, window=5)

        self.assertEqual(fr[-1], 0.0)

    def test_all_flips(self):
        """Test all flips returns 1.0."""
        directions = np.array([1, -1, 1, -1, 1])  # Flip every time

        fr = flip_rate(directions, window=5)

        self.assertEqual(fr[-1], 1.0)

    def test_half_flips(self):
        """Test 50% flip rate."""
        directions = np.array([1, 1, -1, -1, 1])  # 2 flips out of 4 transitions

        fr = flip_rate(directions, window=5)

        self.assertEqual(fr[-1], 0.5)

    def test_insufficient_data(self):
        """Test with insufficient data returns nan."""
        directions = np.array([1])

        fr = flip_rate(directions, window=5)

        self.assertTrue(np.isnan(fr[-1]))

    def test_empty_array(self):
        """Test with empty array."""
        directions = np.array([])

        fr = flip_rate(directions, window=5)

        self.assertEqual(len(fr), 0)


class TestMarkovStickiness(unittest.TestCase):
    """Test Markov stickiness calculation."""

    def test_perfect_up_stickiness(self):
        """Test perfect up stickiness (always up after up)."""
        directions = np.array([1, 1, 1, 1, 1])  # Always up

        pUU, pDD = markov_stickiness(directions, window=5)

        self.assertEqual(pUU[-1], 1.0)  # Perfect stickiness

    def test_perfect_down_stickiness(self):
        """Test perfect down stickiness."""
        directions = np.array([-1, -1, -1, -1, -1])  # Always down

        pUU, pDD = markov_stickiness(directions, window=5)

        self.assertEqual(pDD[-1], 1.0)  # Perfect stickiness

    def test_alternating_no_stickiness(self):
        """Test alternating has no stickiness."""
        directions = np.array([1, -1, 1, -1, 1])  # Alternating

        pUU, pDD = markov_stickiness(directions, window=5)

        self.assertEqual(pUU[-1], 0.0)  # No up-after-up
        self.assertEqual(pDD[-1], 0.0)  # No down-after-down

    def test_insufficient_data(self):
        """Test with insufficient data returns nan."""
        directions = np.array([1])

        pUU, pDD = markov_stickiness(directions, window=5)

        self.assertTrue(np.isnan(pUU[-1]))
        self.assertTrue(np.isnan(pDD[-1]))


class TestEvaluateEntry(unittest.TestCase):
    """Test entry evaluation logic."""

    def test_long_entry_passes(self):
        """Test long entry with good metrics passes."""
        result = evaluate_entry(
            direction=1,  # Long
            flip_rate_val=0.2,  # Low flip rate (good)
            pUU=0.8,  # High up stickiness (good)
            pDD=0.3,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        self.assertTrue(result)

    def test_short_entry_passes(self):
        """Test short entry with good metrics passes."""
        result = evaluate_entry(
            direction=-1,  # Short
            flip_rate_val=0.2,
            pUU=0.3,
            pDD=0.8,  # High down stickiness (good for short)
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        self.assertTrue(result)

    def test_high_flip_rate_rejected(self):
        """Test entry rejected on high flip rate."""
        result = evaluate_entry(
            direction=1,
            flip_rate_val=0.5,  # High flip rate (bad)
            pUU=0.8,
            pDD=0.3,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        self.assertFalse(result)

    def test_low_markov_rejected(self):
        """Test entry rejected on low Markov stickiness."""
        result = evaluate_entry(
            direction=1,
            flip_rate_val=0.2,
            pUU=0.4,  # Below threshold
            pDD=0.3,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        self.assertFalse(result)

    def test_nan_metrics_rejected(self):
        """Test entry rejected on NaN metrics."""
        result = evaluate_entry(
            direction=1,
            flip_rate_val=np.nan,
            pUU=0.8,
            pDD=0.3,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        self.assertFalse(result)

    def test_exact_threshold_boundary(self):
        """Test entry at exact threshold."""
        result = evaluate_entry(
            direction=1,
            flip_rate_val=0.35,  # Exactly at threshold
            pUU=0.55,  # Exactly at threshold
            pDD=0.3,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
        )

        # Should pass when values are <= threshold for flip_rate
        # and >= threshold for markov
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
