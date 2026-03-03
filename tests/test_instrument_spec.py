"""
Tests for InstrumentSpec (kinetra.friction_cost)
=================================================

Coverage targets
----------------
- swap_mode field: default, mode 0 (pips), mode 1 (% p.a.)
- swap_long_usd_per_day / swap_short_usd_per_day:
    - mode 0 → swap_points × tick_value_usd
    - mode 1 → nan  (cannot convert without price)
    - USD override present → override wins regardless of mode
- triple_swap_day loading:
    - new key  ``swap_triple_day``
    - legacy key ``triple_swap_day_our_conv``
    - neither key → default 3
- commission_per_lot loading:
    - stored value respected
    - zero + not commission-free + ECN symbol → ECN default inserted
    - zero + is_commission_free flag → stays zero
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from kinetra.friction_cost import InstrumentSpec, ECN_COMMISSION_PER_SIDE_USD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE: Dict[str, Any] = {
    "symbol": "XAUUSD",
    "digits": 2,
    "tick_size": 0.01,
    "tick_value": 1.0,
    "contract_size": 100.0,
    "spread_points": 0.0,
    "commission_per_lot": 3.5,
    "slippage_points": 1.0,
    "swap_long_points": -10.17,
    "swap_short_points": 4.78,
    "swap_mode": 0,
    "swap_triple_day": 3,
    "margin_initial": 0.01,
    "volume_min": 0.01,
    "volume_max": 50.0,
    "volume_step": 0.01,
    "quote_currency": "USD",
    "quote_usd_rate": 1.0,
    "bid": 5165.12,
    "ask": 5166.01,
    "is_commission_free": False,
}


def _spec(**overrides) -> InstrumentSpec:
    raw = {**_BASE, **overrides}
    return InstrumentSpec.from_polled_json(raw)


# ---------------------------------------------------------------------------
# swap_mode default
# ---------------------------------------------------------------------------


def test_swap_mode_default_is_zero() -> None:
    """When swap_mode is absent from the JSON, it defaults to 0 (pips)."""
    raw_without_mode = {k: v for k, v in _BASE.items() if k != "swap_mode"}
    spec = InstrumentSpec.from_polled_json(raw_without_mode)
    assert spec.swap_mode == 0


def test_swap_mode_zero_loaded() -> None:
    spec = _spec(swap_mode=0)
    assert spec.swap_mode == 0


def test_swap_mode_one_loaded() -> None:
    spec = _spec(symbol="NAS100", swap_mode=1, swap_long_points=-6.18, swap_short_points=1.18)
    assert spec.swap_mode == 1


# ---------------------------------------------------------------------------
# swap_long_usd_per_day / swap_short_usd_per_day — mode 0 (pips)
# ---------------------------------------------------------------------------


def test_swap_usd_per_day_mode0_uses_tick_value() -> None:
    """Mode 0: daily USD swap = swap_points × tick_value_usd."""
    # XAUUSD: tick_value_usd = 100 × 0.01 × 1.0 = $1.00
    spec = _spec(swap_mode=0, swap_long_points=-10.17, swap_short_points=4.78)
    assert spec.tick_value_usd == pytest.approx(1.0)
    assert spec.swap_long_usd_per_day == pytest.approx(-10.17)
    assert spec.swap_short_usd_per_day == pytest.approx(4.78)


def test_swap_usd_per_day_mode0_respects_contract_size() -> None:
    """tick_value_usd scales with contract_size — e.g. XAGUSD 5000 oz."""
    # XAGUSD: contract_size=5000, tick_size=0.001 → tick_value_usd = 5.0
    spec = _spec(
        symbol="XAGUSD",
        contract_size=5000.0,
        tick_size=0.001,
        swap_mode=0,
        swap_long_points=-2.42,
        swap_short_points=0.87,
    )
    assert spec.tick_value_usd == pytest.approx(5.0)
    assert spec.swap_long_usd_per_day == pytest.approx(-2.42 * 5.0)
    assert spec.swap_short_usd_per_day == pytest.approx(0.87 * 5.0)


# ---------------------------------------------------------------------------
# swap_long_usd_per_day / swap_short_usd_per_day — mode 1 (% p.a.)
# ---------------------------------------------------------------------------


def test_swap_usd_per_day_mode1_returns_nan() -> None:
    """Mode 1: converting % p.a. to USD requires a price — return nan."""
    spec = _spec(
        symbol="NAS100",
        contract_size=1.0,
        tick_size=0.1,
        swap_mode=1,
        swap_long_points=-6.18,
        swap_short_points=1.18,
    )
    assert math.isnan(spec.swap_long_usd_per_day)
    assert math.isnan(spec.swap_short_usd_per_day)


def test_swap_usd_per_day_mode1_points_preserved() -> None:
    """swap_long_points still holds the raw % rate for display purposes."""
    spec = _spec(symbol="GER40", swap_mode=1, swap_long_points=-4.43, swap_short_points=-0.57)
    assert spec.swap_long_points == pytest.approx(-4.43)
    assert spec.swap_short_points == pytest.approx(-0.57)


# ---------------------------------------------------------------------------
# USD override wins regardless of mode
# ---------------------------------------------------------------------------


def test_usd_override_beats_mode0() -> None:
    """Polled USD value overrides the pips-based fallback."""
    spec = _spec(
        swap_mode=0,
        swap_long_points=-10.17,
        swap_long_usd_per_lot_per_eff_day=-9.50,
        swap_short_usd_per_lot_per_eff_day=4.30,
    )
    assert spec.swap_long_usd_per_day == pytest.approx(-9.50)
    assert spec.swap_short_usd_per_day == pytest.approx(4.30)


def test_usd_override_beats_mode1() -> None:
    """Polled USD value overrides the % p.a. nan path."""
    spec = _spec(
        symbol="NAS100",
        swap_mode=1,
        swap_long_points=-6.18,
        swap_long_usd_per_lot_per_eff_day=-4.23,
        swap_short_usd_per_lot_per_eff_day=1.10,
    )
    assert not math.isnan(spec.swap_long_usd_per_day)
    assert spec.swap_long_usd_per_day == pytest.approx(-4.23)
    assert spec.swap_short_usd_per_day == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# triple_swap_day key resolution
# ---------------------------------------------------------------------------


def test_triple_swap_day_from_swap_triple_day_key() -> None:
    """New JSON key ``swap_triple_day`` is read correctly."""
    spec = _spec(swap_triple_day=3)
    assert spec.triple_swap_day == 3


def test_triple_swap_day_friday() -> None:
    """Indices use Friday (day 5) as triple-swap day."""
    spec = _spec(symbol="NAS100", swap_triple_day=5)
    assert spec.triple_swap_day == 5


def test_triple_swap_day_legacy_key() -> None:
    """Legacy key ``triple_swap_day_our_conv`` still works."""
    raw = {k: v for k, v in _BASE.items() if k != "swap_triple_day"}
    raw["triple_swap_day_our_conv"] = 4
    spec = InstrumentSpec.from_polled_json(raw)
    assert spec.triple_swap_day == 4


def test_triple_swap_day_default_when_absent() -> None:
    """Falls back to 3 (Wednesday) when neither key is present."""
    raw = {k: v for k, v in _BASE.items() if k not in ("swap_triple_day", "triple_swap_day_our_conv")}
    spec = InstrumentSpec.from_polled_json(raw)
    assert spec.triple_swap_day == 3


def test_triple_swap_day_new_key_takes_precedence_over_legacy() -> None:
    """New key wins when both are present."""
    raw = {**_BASE, "symbol": "NAS100", "swap_triple_day": 5, "triple_swap_day_our_conv": 3}
    spec = InstrumentSpec.from_polled_json(raw)
    assert spec.triple_swap_day == 5


# ---------------------------------------------------------------------------
# commission_per_lot loading
# ---------------------------------------------------------------------------


def test_commission_stored_value_respected() -> None:
    """Explicit commission in JSON is loaded as-is."""
    spec = _spec(commission_per_lot=3.5)
    assert spec.commission_per_lot == pytest.approx(3.5)


def test_commission_zero_ecn_symbol_gets_default() -> None:
    """Zero commission on an ECN symbol (XAUUSD) triggers the ECN default."""
    spec = _spec(commission_per_lot=0.0, is_commission_free=False)
    assert spec.commission_per_lot == pytest.approx(ECN_COMMISSION_PER_SIDE_USD)


def test_commission_zero_commission_free_stays_zero() -> None:
    """is_commission_free=True keeps commission at zero even for ECN symbols."""
    spec = _spec(symbol="NAS100", commission_per_lot=0.0, is_commission_free=True)
    assert spec.commission_per_lot == pytest.approx(0.0)


def test_commission_zero_non_ecn_symbol_stays_zero() -> None:
    """Spread-only CFD with no ECN flag keeps commission at zero."""
    # NAS100 doesn't match ECN regex ([A-Z]{6} or XAU/XAG prefix)
    spec = _spec(symbol="NAS100", commission_per_lot=0.0, is_commission_free=False)
    assert spec.commission_per_lot == pytest.approx(0.0)
