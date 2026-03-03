#!/usr/bin/env python3
"""
Friction Cost Calculator — Canonical Source of Truth
=====================================================

Single module consumed by ALL cost-aware subsystems:

    ┌─────────────────────────────────────────────────────────────────┐
    │  data/master_standardized/<cat>/<SYM>/contract_spec.json        │
    │           (swap, contract size, polled from live broker)        │
    │  +  H4 CSV spread column  (time-averaged, most reliable)        │
    └───────────────────────────┬─────────────────────────────────────┘
                                │  load_spec() / get_calculator()
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              kinetra/friction_cost.py   ← YOU ARE HERE          │
    │                                                                 │
    │   InstrumentSpec      — normalised broker spec                  │
    │   FrictionBreakdown   — per-component cost result               │
    │   FrictionCalculator  — ALL cost math lives here                │
    │   BuyHoldCalculator   — accurate B&H baseline                   │
    └──────┬──────────────────┬──────────────────┬────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    backtest_engine    exploration /         B&H benchmark
    (trade costs)      RL reward shaping    (vs random baseline)

Friction model (all components)
--------------------------------
  1. SPREAD — time-averaged from actual H4 CSV spread column (most
     reliable); falls back through typical_spread_points → spread_points
     → bid/ask snapshot in that priority order.

  2. COMMISSION — ECN flat rate: $7.00 round-trip per standard lot
     ($3.50/side).  Canonical constant: ECN_COMMISSION_RT_USD.
     Non-ECN / commission-free instruments set commission_per_lot = 0.

  3. LATENCY SLIPPAGE — 500 ms per side (1 000 ms round-trip).
     Modelled as adverse drift during order transit:
         slippage_per_side = ATR × sqrt(latency_ms / bar_duration_ms)
     Requires the caller to pass the current bar's ATR.  Falls back to
     a fixed slippage_points estimate when ATR is not provided.

  4. SWAP (overnight carry) — USD per lot per effective day from
     contract_spec.json, scaled by the actual hold duration.
     Triple-swap day (typically Wednesday) handled correctly.
     Include swap with ``include_swap=True`` (default True in
     ``friction_pct``; use ``include_swap=False`` for intraday sizing).

Design principles
-----------------
- ONE source of truth.  No other module should hardcode swap rates or
  spreads.  If you need costs, call get_calculator(symbol).
- CSV-measured spread is the highest-reliability source.  Use
  ``measure_spread_from_csv(symbol)`` or ``get_calculator_with_data(symbol)``
  to populate it automatically.
- ECN commission is locked at $7 RT.  Override only for non-ECN accounts.
- Zero heavy dependencies.  Only stdlib + numpy + pandas.
- Fully vectorised swap calendar.  Triple-swap day handled correctly.
- Transparent breakdowns.  Every calculation returns a FrictionBreakdown.

Quick-start
-----------
    from kinetra.friction_cost import get_calculator_with_data

    # Spread auto-measured from CSV, commission=$7RT, latency=500ms baked in
    calc = get_calculator_with_data("GBPUSD")

    # Single trade round-trip (swap included automatically)
    cost = calc.round_trip(
        is_long=True, lots=1.0,
        entry_price=1.2680, exit_price=1.2720,
        entry_dt=datetime(2024, 3, 4, 9, 0),
        exit_dt=datetime(2024, 3, 8, 17, 0),
    )
    print(f"Net P&L: ${cost.gross_pnl_usd - cost.friction.total_usd:.2f}")

    # RL reward shaping — per-bar friction as % of notional
    # Pass atr= for physics-based latency slippage; omit for fixed fallback
    friction_pct = calc.friction_pct(
        price=1.2680, lots=1.0, is_long=True,
        holding_bars=10, atr=0.0035,
        include_swap=True,
    )

    # Full all-in breakdown for tier scoring
    bd = calc.all_in_breakdown(
        price=1.2680, lots=1.0, is_long=True,
        holding_bars=10, atr=0.0035,
    )
    print(bd.summary())
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical ECN commission constant
# ---------------------------------------------------------------------------

#: Round-trip commission per standard lot for an ECN/Raw account.
#: $3.50 entry + $3.50 exit = $7.00 total.  Override per-instrument only for
#: commission-free (spread-only) CFD/index instruments.
ECN_COMMISSION_RT_USD: float = 7.0
ECN_COMMISSION_PER_SIDE_USD: float = ECN_COMMISSION_RT_USD / 2.0  # $3.50


def _is_ecn_commission_symbol(symbol: str) -> bool:
    """Return True for symbols that normally carry ECN per-lot commission."""
    s = (symbol or "").upper()
    if not s:
        return False
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return True
    return bool(re.fullmatch(r"[A-Z]{6}", s))


#: Default one-way execution latency (milliseconds).
#: Models the round-trip order-transit time for an internet-connected algo:
#: strategy → broker gateway → exchange matching engine → fill confirmation.
#: 500 ms per side (1 000 ms RT) is a conservative but realistic estimate
#: for retail algo execution over a VPS with 20–50 ms network latency plus
#: broker processing overhead.
DEFAULT_LATENCY_MS_PER_SIDE: float = 500.0

#: Bar duration in milliseconds for each timeframe.  Used to scale latency
#: slippage relative to bar volatility.
_BAR_DURATION_MS: Dict[str, float] = {
    "M1": 60_000.0,
    "M5": 300_000.0,
    "M15": 900_000.0,
    "M30": 1_800_000.0,
    "H1": 3_600_000.0,
    "H4": 14_400_000.0,
    "D1": 86_400_000.0,
    "W1": 604_800_000.0,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical spec location — per-instrument folder inside master_standardized.
# poll_symbol_specs.py writes broker_friction into contract_spec.json here.
_MASTER_DIR = _PROJECT_ROOT / "data" / "master_standardized"

# Legacy flat-file directory kept for backward-compat; used as a fallback
# when no contract_spec.json exists for a symbol.
_SPECS_DIR = _PROJECT_ROOT / "data" / "symbol_specs"

# Per-symbol spec files older than this trigger a staleness warning (not an
# error — stale data is still far better than hardcoded defaults).
_STALE_DAYS = 7

# ---------------------------------------------------------------------------
# Conservative generic fallback values
# Used ONLY when data/symbol_specs/<SYMBOL>.json has never been written.
# These are deliberately conservative (wider spread, higher commission) so
# callers never accidentally over-estimate edge.
# ---------------------------------------------------------------------------

_GENERIC_DEFAULTS: Dict[str, float] = {
    "digits": 5,
    "contract_size": 100_000.0,
    "tick_size": 0.00001,
    "tick_value": 1.0,
    "spread_points": 20.0,  # 2 pip — conservative for liquid forex
    "swap_long_points": -5.0,  # conservative negative carry
    "swap_short_points": -5.0,
    "triple_swap_day": 3,  # Wednesday (1=Mon…7=Sun)
    "commission_per_lot": ECN_COMMISSION_PER_SIDE_USD,  # $3.50/lot/side ECN
    "slippage_points": 1.0,  # fixed fallback; latency model used when ATR provided
    "margin_initial": 0.01,
    "volume_min": 0.01,
    "volume_max": 500.0,
    "volume_step": 0.01,
}

# Asset-class overrides applied on top of the generic defaults when the
# symbol name pattern matches.  Keyed by prefix patterns.
_ASSET_CLASS_OVERRIDES: Dict[str, Dict[str, float]] = {
    # Crypto — much wider spreads, no real swap (funding rate approximated)
    "BTC": {
        "contract_size": 1.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 500.0,
        "swap_long_points": -10.0,
        "swap_short_points": 2.0,
        "commission_per_lot": 0.0,
    },
    "ETH": {
        "contract_size": 1.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 300.0,
        "swap_long_points": -8.0,
        "swap_short_points": 1.5,
        "commission_per_lot": 0.0,
    },
    # Gold / silver
    "XAU": {
        "contract_size": 100.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 35.0,
        "swap_long_points": -30.0,
        "swap_short_points": 15.0,
        "commission_per_lot": 0.0,
    },
    "XAG": {
        "contract_size": 5000.0,
        "tick_size": 0.001,
        "tick_value": 5.0,
        "spread_points": 30.0,
        "swap_long_points": -8.0,
        "swap_short_points": 3.0,
        "commission_per_lot": 0.0,
    },
    # Indices — no commission, wider spread
    "NAS": {
        "contract_size": 1.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 150.0,
        "swap_long_points": -5.0,
        "swap_short_points": 1.0,
        "commission_per_lot": 0.0,
    },
    "US3": {
        "contract_size": 1.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 250.0,
        "swap_long_points": -4.0,
        "swap_short_points": 0.8,
        "commission_per_lot": 0.0,
    },
    "UK1": {
        "contract_size": 1.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "spread_points": 100.0,
        "swap_long_points": -4.0,
        "swap_short_points": 0.5,
        "commission_per_lot": 0.0,
    },
    "Nik": {
        "contract_size": 1.0,
        "tick_size": 1.0,
        "tick_value": 1.0,
        "spread_points": 80.0,
        "swap_long_points": 2.0,
        "swap_short_points": -3.5,
        "commission_per_lot": 0.0,
    },
    # Oil
    "UKO": {
        "contract_size": 1000.0,
        "tick_size": 0.001,
        "tick_value": 1.0,
        "spread_points": 40.0,
        "swap_long_points": -15.0,
        "swap_short_points": 5.0,
        "commission_per_lot": 0.0,
    },
    "USO": {
        "contract_size": 1000.0,
        "tick_size": 0.001,
        "tick_value": 1.0,
        "spread_points": 40.0,
        "swap_long_points": -14.0,
        "swap_short_points": 4.5,
        "commission_per_lot": 0.0,
    },
}


def _generic_defaults_for(symbol: str) -> Dict[str, float]:
    """Return generic defaults, applying asset-class overrides where known."""
    defaults = dict(_GENERIC_DEFAULTS)
    sym_upper = symbol.upper()
    for prefix, overrides in _ASSET_CLASS_OVERRIDES.items():
        if sym_upper.startswith(prefix.upper()):
            defaults.update(overrides)
            break
    return defaults


# ---------------------------------------------------------------------------
# InstrumentSpec  — normalised broker specification
# ---------------------------------------------------------------------------


@dataclass
class InstrumentSpec:
    """
    Complete instrument specification loaded from
    data/master_standardized/<category>/<SYMBOL>/contract_spec.json
    (broker_friction section).

    All fields match the schema written by poll_symbol_specs.py.
    When no polled file exists, conservative generic defaults are used.

    Account base currency: USD.  All USD-denominated properties
    (spread_usd_per_lot, swap_long_usd_per_day, etc.) already include the
    quote-currency → USD conversion via quote_usd_rate.
    """

    symbol: str

    # Price precision
    digits: int = 5
    tick_size: float = 0.00001  # Minimum price movement (= point)
    tick_value: float = 1.0  # Raw tick value from broker (may be in quote ccy)

    # Contract
    contract_size: float = 100_000.0  # Units per lot

    # Costs — loaded from broker
    spread_points: float = 20.0  # Typical spread in POINTS
    commission_per_lot: float = 3.5  # USD per lot per side (0 if commission-free)
    slippage_points: float = 1.0  # Expected slippage per side in POINTS

    # Swap (overnight carry) — loaded from broker
    swap_long_points: float = -5.0  # Points per day (mode 0) or annual % rate (mode 1)
    swap_short_points: float = -5.0  # Points per day (mode 0) or annual % rate (mode 1)
    triple_swap_day: int = 3  # Day when 3× swap is charged (1=Mon…7=Sun)
    # swap_mode: 0 = pips/points per day, 1 = % per year (annual rate applied to notional).
    # Indices (GER40, NAS100, JPN225) use mode 1; metals/forex/commodities use mode 0.
    swap_mode: int = 0

    # Authoritative USD swap values polled directly from the broker.
    #
    # WHY THIS EXISTS
    # ---------------
    # MT5 stores swap in "points", but the meaning of one "point" varies
    # wildly across instrument classes:
    #   - EURUSD: 1 point = tick_size = 0.00001, tick_value ≈ $1/pip → swap
    #             in points × tick_value_usd gives the correct USD amount.
    #   - BTCUSD: 1 point = $0.01 (tick_size), tick_value_usd = $0.01, yet
    #             MetaAPI's swap_long_points = -20 means -$20/day — NOT
    #             -$0.20/day.  The broker already expressed the swap in USD;
    #             multiplying by tick_value_usd again introduces a 100× error.
    #   - Cross-rates (XAUJPY, BTCXAU, etc.): errors can reach 100 000×.
    #
    # poll_symbol_specs.py stores `swap_long_usd_per_lot_per_eff_day` in
    # contract_spec.json directly from MetaAPI.  When present, this field
    # OVERRIDES the points-based back-calculation in swap_long_usd_per_day.
    #
    # Back-compat: old spec files that pre-date this field will have None here,
    # and the property falls back to the (potentially inaccurate) points path.
    swap_long_usd_per_lot_per_eff_day: Optional[float] = None
    swap_short_usd_per_lot_per_eff_day: Optional[float] = None

    # Margin / sizing
    margin_initial: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 500.0
    volume_step: float = 0.01

    # Metadata
    source: str = "default"  # "metaapi" | "default"
    polled_at: Optional[str] = None
    broker_symbol: Optional[str] = None

    # Point-in-time bid/ask snapshot from the poll moment.
    # Used to compute the real spread directly (bypasses tick_value inaccuracies
    # for CFD instruments where MetaAPI returns TickValue=0 and the poll script
    # defaults to 1.0, causing spread_points * tick_value to be wrong).
    bid: float = 0.0
    ask: float = 0.0

    # Quote currency and FX rate (USD account).
    # quote_currency : 3-char ISO code for the instrument's profit/quote currency
    #                  e.g. "USD" for XAUUSD, "EUR" for XAUEUR, "JPY" for XAUJPY
    # quote_usd_rate : 1 unit of quote_currency expressed in USD.
    #                  e.g. EURUSD = 1.178 → quote_usd_rate = 1.178 for XAUEUR
    #                  e.g. USDJPY = 155   → quote_usd_rate = 1/155 for XAUJPY
    #                  Defaults to 1.0 (USD-quoted) — correct for XAUUSD, EURUSD, etc.
    #
    # These are computed by poll_symbol_specs.py at poll time and stored in
    # contract_spec.json → broker_friction.  All USD-denominated properties
    # below use quote_usd_rate so they are always in true USD for a USD account.
    quote_currency: str = "USD"
    quote_usd_rate: float = 1.0

    # Manually-reviewed field metadata (category C).
    # These timestamps are set by the operator whenever commission_per_lot,
    # slippage_points, or typical_spread_points are verified / updated.
    # They are used by poll_symbol_specs and patch_contract_specs to surface
    # staleness warnings.
    #
    # *_reviewed_at : ISO-8601 string, e.g. "2026-02-21T10:00:00+00:00"
    # None means the operator has never explicitly reviewed the value.
    commission_reviewed_at: Optional[str] = None
    slippage_reviewed_at: Optional[str] = None
    is_commission_free: bool = False

    # typical_spread_points — operator-measured typical spread in points during
    # normal trading hours (Category C: manually reviewed).
    #
    # WHY THIS EXISTS
    # ---------------
    # MetaAPI returns spread_points = 0 for all ECN / variable-spread
    # instruments (the broker does not publish a fixed spread in the symbol
    # specification for these accounts).  The only other source is the
    # bid/ask price snapshot captured at poll time — but that snapshot can
    # be taken during off-hours, a liquidity event, or a wide-spread period,
    # producing wildly unrepresentative values (e.g. XAUEUR polled at a moment
    # where the snapshot showed 2.04 EUR/oz spread = $240/lot, vs the real
    # typical ECN spread of ~$0.25/oz = ~$25/lot).
    #
    # typical_spread_points is the operator's best estimate of the spread
    # that will be seen during actual trading sessions.  For Raw/ECN accounts:
    #   XAUUSD Raw: ~7–10 points  ($7–10/lot)
    #   XAUEUR Raw: ~20–30 points ($23–35/lot)
    #   GBPUSD Raw: ~2–4 points
    #
    # Set to 0.0 (default) to fall back to the automated sources.
    typical_spread_points: float = 0.0
    typical_spread_reviewed_at: Optional[str] = None

    # csv_measured_spread_points — time-averaged spread computed directly from
    # the H4 CSV spread column by measure_spread_from_csv() / build_from_data().
    #
    # This is the most reliable spread source because it reflects actual
    # broker execution over thousands of bars during live trading hours,
    # not a single polled snapshot.  Used as the TOP priority in the
    # spread_usd_per_lot priority chain (above typical_spread_points).
    #
    # Populated automatically by get_calculator_with_data() or explicitly via
    # measure_spread_from_csv(symbol).  Set to 0.0 (default) to disable.
    csv_measured_spread_points: float = 0.0
    csv_spread_measured_at: Optional[str] = None  # ISO-8601 timestamp of measurement
    csv_spread_n_bars: int = 0  # Number of non-zero bars used in measurement

    # latency_ms_per_side — one-way order execution latency in milliseconds.
    # Used in FrictionCalculator.latency_slippage_usd() to compute the
    # expected adverse price drift during order transit.
    # Default: 500 ms (conservative retail algo over VPS).
    latency_ms_per_side: float = DEFAULT_LATENCY_MS_PER_SIDE

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def point(self) -> float:
        """Alias for tick_size (MT5 convention: point = minimum price change)."""
        return self.tick_size

    # -----------------------------------------------------------------------
    # Derived helpers — all USD values assume account base currency = USD
    # -----------------------------------------------------------------------

    @property
    def tick_value_usd(self) -> float:
        """True USD value of 1 tick (tick_size) per 1 lot for a USD account.

        Correct formula:
            tick_value_usd = contract_size × tick_size × quote_usd_rate

        MetaAPI returns TickValue = 0 for most CFD instruments; the poll script
        defaults to 1.0 which is the quote-currency value, not USD.  This
        property always uses the FX-corrected rate so callers get a true USD
        figure regardless of what tick_value contains.
        """
        return self.contract_size * self.tick_size * self.quote_usd_rate

    @property
    def pip_value_usd(self) -> float:
        """USD value of 1 pip (= 10 points for 5-digit FX) per lot.

        For non-FX assets where point = pip, this equals tick_value_usd × 1.
        """
        pip_multiplier = 10 if self.digits == 5 else 1
        return self.tick_value_usd * pip_multiplier

    @property
    def spread_usd_per_lot(self) -> float:
        """Typical spread cost in USD per 1 lot (account = USD).

        Priority chain — uses the first source that produces a positive value:

        1. csv_measured_spread_points  (MOST RELIABLE — time-averaged from
               the actual H4 CSV spread column across thousands of live bars)
               csv_measured_spread_points × tick_value_usd
           Populated by measure_spread_from_csv() / get_calculator_with_data().

        2. typical_spread_points (Category C — operator-measured)
               typical_spread_points × tick_value_usd
           Set manually in broker_friction after observing live spreads.

        3. spread_points (Category A — from MetaAPI symbol specification)
               spread_points × tick_value_usd
           Non-zero only for fixed-spread / standard accounts.  ECN brokers
           always report 0 here by design.

        4. bid/ask snapshot (LAST RESORT — unreliable, point-in-time only)
               (ask − bid) × contract_size × quote_usd_rate
           The snapshot may have been captured during off-hours or a liquidity
           event, producing wildly unrepresentative values.

        All sources use quote_usd_rate so the result is always in true USD
        for a USD account regardless of the instrument's quote currency.
        """
        # 1. CSV time-averaged spread (most reliable)
        if self.csv_measured_spread_points > 0:
            return self.csv_measured_spread_points * self.tick_value_usd

        # 2. Operator-measured typical spread
        if self.typical_spread_points > 0:
            return self.typical_spread_points * self.tick_value_usd

        # 3. Broker-reported fixed spread (non-zero only on standard accounts)
        if self.spread_points > 0:
            return self.spread_points * self.tick_value_usd

        # 4. Bid/ask snapshot — last resort, potentially unrepresentative
        if self.ask > self.bid > 0:
            logger.warning(
                "spread_usd_per_lot for %s is using a raw bid/ask snapshot "
                "(csv_measured_spread_points not set, typical_spread_points not set, "
                "spread_points = 0). The snapshot spread may be from off-hours or a "
                "wide-spread period and could be unreliable. "
                "Call get_calculator_with_data('%s') to measure spread from CSV.",
                self.symbol,
                self.symbol,
            )
            return (self.ask - self.bid) * self.contract_size * self.quote_usd_rate

        return 0.0

    @property
    def spread_pct_of_mid(self) -> float:
        """Spread as % of mid-price (currency-neutral, informational).

        When typical_spread_points is set, computes the percentage using that
        representative spread relative to the bid/ask mid price — this gives
        a stable, comparable figure across instruments.

        Falls back to the raw (ask − bid) / mid ratio when typical_spread_points
        is not configured.  That ratio is still currency-neutral (the quote
        currency cancels) but may reflect an off-hours snapshot spread.

        Returns 0.0 when no price reference (bid/ask) is available.
        """
        if not (self.ask > self.bid > 0):
            return 0.0
        mid = (self.ask + self.bid) / 2.0
        if self.typical_spread_points > 0:
            typical_spread_price = self.typical_spread_points * self.tick_size
            return typical_spread_price / mid * 100.0
        return (self.ask - self.bid) / mid * 100.0

    @property
    def commission_round_trip(self) -> float:
        """Total commission for 1 lot round trip (entry + exit)."""
        return self.commission_per_lot * 2.0

    def is_spread_stale(self, max_days: int = 90) -> bool:
        """Return True when no reliable spread source is configured or is stale.

        Returns False as soon as ANY of these is present and fresh:
          - csv_measured_spread_points  (refreshed each time build_from_data runs)
          - typical_spread_points with a recent review timestamp

        Returns True (stale / unconfigured) when only the broker spread_points
        or bid/ask snapshot is available, since those are unreliable for ECN.
        """
        # CSV-measured is always fresh (computed from current data files)
        if self.csv_measured_spread_points > 0:
            return False
        if self.typical_spread_points <= 0:
            return True
        if self.typical_spread_reviewed_at is None:
            return True
        try:
            ts = datetime.fromisoformat(self.typical_spread_reviewed_at.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - ts).days > max_days
        except (ValueError, AttributeError):
            return True

    def is_commission_stale(self, max_days: int = 90) -> bool:
        """Return True if commission_per_lot has not been reviewed within *max_days*.

        Also returns True when commission_reviewed_at is absent (i.e. the field
        was never explicitly confirmed by the operator).
        """
        if self.commission_reviewed_at is None:
            return True
        try:
            ts = datetime.fromisoformat(self.commission_reviewed_at.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - ts).days > max_days
        except (ValueError, AttributeError):
            return True

    def is_slippage_stale(self, max_days: int = 90) -> bool:
        """Return True if slippage_points has not been reviewed within *max_days*.

        Also returns True when slippage_reviewed_at is absent.
        """
        if self.slippage_reviewed_at is None:
            return True
        try:
            ts = datetime.fromisoformat(self.slippage_reviewed_at.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - ts).days > max_days
        except (ValueError, AttributeError):
            return True

    @property
    def swap_long_usd_per_day(self) -> float:
        """USD swap cost/credit per day for 1 lot long (account = USD).

        Priority:
          1. ``swap_long_usd_per_lot_per_eff_day`` — polled USD value stored
             directly by poll_symbol_specs.py.  This is authoritative because
             the broker already expressed the rate in USD; no conversion needed.
          2. Mode 0 (pips): ``swap_long_points * tick_value_usd`` — fallback for
             legacy spec files that pre-date the USD field.
          3. Mode 1 (% p.a.): cannot convert to USD without a current price;
             returns ``float("nan")`` so callers know the value is unavailable.
        """
        if self.swap_long_usd_per_lot_per_eff_day is not None:
            return self.swap_long_usd_per_lot_per_eff_day
        if self.swap_mode == 1:
            return float("nan")  # % p.a. — needs price to convert; use swap_long_points
        return self.swap_long_points * self.tick_value_usd

    @property
    def swap_short_usd_per_day(self) -> float:
        """USD swap cost/credit per day for 1 lot short (account = USD).

        Priority:
          1. ``swap_short_usd_per_lot_per_eff_day`` — polled USD value (authoritative).
          2. Mode 0 (pips): ``swap_short_points * tick_value_usd`` — legacy fallback.
          3. Mode 1 (% p.a.): returns ``float("nan")``.
        """
        if self.swap_short_usd_per_lot_per_eff_day is not None:
            return self.swap_short_usd_per_lot_per_eff_day
        if self.swap_mode == 1:
            return float("nan")
        return self.swap_short_points * self.tick_value_usd

    def notional_usd(self, price: float, lots: float) -> float:
        """Notional trade value in USD (account = USD).

        price is in the instrument's quote currency; quote_usd_rate converts it.
        """
        return price * self.contract_size * self.quote_usd_rate * lots

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_polled_json(cls, raw: Dict) -> "InstrumentSpec":
        """Build from a polled spec dict.

        Accepts two layouts:
          1. Flat dict  — legacy data/symbol_specs/<SYMBOL>.json
          2. Nested     — data/master_standardized/.../contract_spec.json
             where broker friction lives under the "broker_friction" key.

        bid/ask and quote_usd_rate are read so that spread_usd_per_lot and
        swap_*_usd_per_day always return true USD values for a USD account.

        Category-C metadata (commission_reviewed_at, slippage_reviewed_at) is
        also loaded so callers can surface staleness warnings without re-reading
        the JSON file.

        Note:
            For multi-broker support, prefer :meth:`from_broker_json` which
            wraps this method and adds source tagging.
        """
        # Support nested contract_spec.json layout
        friction = raw.get("broker_friction")
        if isinstance(friction, dict):
            # Merge with top-level defaults preserved (contract_size, digits, etc.)
            # while broker_friction overrides cost fields when provided.
            merged: Dict = {**raw, **friction}
            merged["symbol"] = raw.get("symbol", merged.get("symbol", "UNKNOWN"))
            merged.setdefault("is_commission_free", raw.get("is_commission_free", False))
            raw = merged

        symbol = raw.get("symbol", "UNKNOWN").upper().replace("+", "")

        bid_raw = raw.get("bid")
        ask_raw = raw.get("ask")
        bid = float(bid_raw) if bid_raw is not None else 0.0
        ask = float(ask_raw) if ask_raw is not None else 0.0

        # quote_usd_rate: stored by poll_symbol_specs.py after FX enrichment.
        # Defaults to 1.0 (USD-quoted) for backward-compat with old spec files
        # that pre-date the FX-correction logic.
        quote_usd_rate = float(raw.get("quote_usd_rate") or 1.0) or 1.0
        quote_currency = str(raw.get("quote_currency") or "USD").upper() or "USD"

        # Category-C: manually-reviewed field metadata.  These are never
        # returned by MetaAPI; they are written by the operator and preserved
        # across every re-poll.  Load them so callers can check staleness
        # without opening the JSON file again.
        commission_reviewed_at = raw.get("commission_reviewed_at") or None
        slippage_reviewed_at = raw.get("slippage_reviewed_at") or None

        # typical_spread_points: operator-measured spread during normal trading
        # hours.  MetaAPI returns spread_points = 0 for all ECN accounts so
        # this is the only reliable spread source for those instruments.
        # Default 0.0 signals "not yet configured" → spread_usd_per_lot will
        # fall back through the priority chain and warn.
        typical_spread_points = float(raw.get("typical_spread_points") or 0.0)
        typical_spread_reviewed_at = raw.get("typical_spread_reviewed_at") or None

        # commission_per_lot: MetaAPI does not expose this in the symbol spec.
        # Use ECN_COMMISSION_PER_SIDE_USD ($3.50) as the canonical default so
        # every polled spec gets the correct ECN commission automatically.
        # Zero is used only for explicitly commission-free instruments
        # (spread-only CFDs, indices) where "commission_per_lot" = 0.0 in JSON.
        raw_comm = raw.get("commission_per_lot")
        is_commission_free = bool(raw.get("is_commission_free", False))
        if raw_comm is not None:
            commission_per_lot = float(raw_comm)
            if commission_per_lot <= 0.0 and not is_commission_free:
                if _is_ecn_commission_symbol(symbol):
                    commission_per_lot = ECN_COMMISSION_PER_SIDE_USD
                else:
                    commission_per_lot = 0.0
        else:
            commission_per_lot = 0.0 if is_commission_free else ECN_COMMISSION_PER_SIDE_USD

        csv_measured_spread_points = float(raw.get("csv_measured_spread_points") or 0.0)
        csv_spread_measured_at = raw.get("csv_spread_measured_at") or None
        csv_spread_n_bars = int(raw.get("csv_spread_n_bars") or 0)
        latency_ms_per_side = float(raw.get("latency_ms_per_side") or DEFAULT_LATENCY_MS_PER_SIDE)

        # swap USD overrides — authoritative when present; None means "use fallback".
        raw_sl_usd = raw.get("swap_long_usd_per_lot_per_eff_day")
        raw_ss_usd = raw.get("swap_short_usd_per_lot_per_eff_day")
        swap_long_usd_override = float(raw_sl_usd) if raw_sl_usd is not None else None
        swap_short_usd_override = float(raw_ss_usd) if raw_ss_usd is not None else None

        return cls(
            symbol=symbol,
            digits=int(raw.get("digits", 5)),
            tick_size=float(raw.get("tick_size", 0.00001)),
            tick_value=float(raw.get("tick_value", 1.0)),
            contract_size=float(raw.get("contract_size", 100_000.0)),
            spread_points=float(raw.get("spread_points", 0.0)),
            commission_per_lot=commission_per_lot,
            slippage_points=float(raw.get("slippage_points", 1.0)),
            swap_long_points=float(raw.get("swap_long_points", -5.0)),
            swap_short_points=float(raw.get("swap_short_points", -5.0)),
            swap_long_usd_per_lot_per_eff_day=swap_long_usd_override,
            swap_short_usd_per_lot_per_eff_day=swap_short_usd_override,
            triple_swap_day=int(
                raw.get("swap_triple_day")          # canonical key (new)
                or raw.get("triple_swap_day_our_conv")  # legacy key (fallback)
                or 3
            ),
            swap_mode=int(raw.get("swap_mode", 0)),
            margin_initial=float(raw.get("margin_initial", 0.01)),
            volume_min=float(raw.get("volume_min", 0.01)),
            volume_max=float(raw.get("volume_max", 500.0)),
            volume_step=float(raw.get("volume_step", 0.01)),
            source=raw.get("broker_source") or raw.get("source") or "metaapi",
            polled_at=raw.get("polled_at"),
            broker_symbol=raw.get("broker_symbol"),
            bid=bid,
            ask=ask,
            quote_currency=quote_currency,
            quote_usd_rate=quote_usd_rate,
            commission_reviewed_at=commission_reviewed_at,
            slippage_reviewed_at=slippage_reviewed_at,
            is_commission_free=is_commission_free,
            typical_spread_points=typical_spread_points,
            typical_spread_reviewed_at=typical_spread_reviewed_at,
            csv_measured_spread_points=csv_measured_spread_points,
            csv_spread_measured_at=csv_spread_measured_at,
            csv_spread_n_bars=csv_spread_n_bars,
            latency_ms_per_side=latency_ms_per_side,
        )

    @classmethod
    def from_broker_json(
        cls,
        raw: Dict,
        source: str = "metaapi",
    ) -> "InstrumentSpec":
        """Build from a broker-specific JSON dict with source tagging.

        This is the **canonical multi-broker factory** (§28 Phase 2).
        It delegates to :meth:`from_polled_json` for the actual field
        mapping, then stamps the ``source`` field so downstream code
        can identify which broker produced the spec.

        When cTrader integration is added (Phase 3), a ``source='ctrader'``
        path will apply cTrader-specific field remapping before delegating
        to the same core constructor.

        Args:
            raw: Normalised spec dict matching the ``from_polled_json()``
                input schema.  For MetaAPI this is the output of
                ``poll_symbol_specs._fetch_one()``.  For cTrader this will
                be the output of a future ``poll_ctrader_specs`` normaliser.
            source: Broker identifier — ``'metaapi'``, ``'mt5'``,
                ``'ctrader'``, or ``'default'``.

        Returns:
            An ``InstrumentSpec`` with ``self.source`` set to *source*.

        Examples::

            # MetaAPI (current)
            spec = InstrumentSpec.from_broker_json(polled_data, source='metaapi')

            # cTrader (future — Phase 3)
            spec = InstrumentSpec.from_broker_json(normalized_ctrader, source='ctrader')

        See Also:
            - :meth:`from_polled_json` — the field-level mapping logic
            - ``kinetra/broker.py`` → ``BrokerSpecHandler`` — the ABC
            - ``kinetra/spec_utils.py`` — shared save/merge orchestration
            - ``AGENT_RULES_MASTER.md §28`` — multi-broker architecture
        """
        # Validate source identifier
        _KNOWN_SOURCES = {"metaapi", "mt5", "ctrader", "default", "csv"}
        if source not in _KNOWN_SOURCES:
            import logging

            logging.getLogger(__name__).warning(
                "InstrumentSpec.from_broker_json: unknown source=%r "
                "(expected one of %s). Proceeding anyway.",
                source,
                _KNOWN_SOURCES,
            )

        # cTrader field remapping (Phase 3 — placeholder)
        # When cTrader integration is implemented, add remapping here:
        #   if source == 'ctrader':
        #       raw = _remap_ctrader_fields(raw)
        #
        # cTrader differences to handle:
        #   - pipPosition → digits (inverted: pipPosition=4 → digits=5 for FX)
        #   - stepVolume → volume_step
        #   - lotSize → contract_size
        #   - swapLong/swapShort → swap_long_points/swap_short_points
        #   - timestamps in ms → ISO-8601
        #   See AGENT_RULES_MASTER.md §28.4 for the full field mapping.

        # Delegate to the existing field parser
        spec = cls.from_polled_json(raw)

        # Override the source tag (from_polled_json always sets source="metaapi")
        object.__setattr__(spec, "source", source)

        return spec

    @classmethod
    def generic_default(cls, symbol: str) -> "InstrumentSpec":
        """
        Conservative generic defaults for when no polled data exists.

        Asset-class-specific overrides applied where the symbol prefix matches.
        Commission defaults to ECN_COMMISSION_PER_SIDE_USD ($3.50/side = $7 RT).
        """
        d = _generic_defaults_for(symbol)
        return cls(
            symbol=symbol.upper(),
            digits=int(d["digits"]),
            tick_size=float(d["tick_size"]),
            tick_value=float(d["tick_value"]),
            contract_size=float(d["contract_size"]),
            spread_points=float(d["spread_points"]),
            commission_per_lot=float(d["commission_per_lot"]),
            slippage_points=float(d["slippage_points"]),
            swap_long_points=float(d["swap_long_points"]),
            swap_short_points=float(d["swap_short_points"]),
            triple_swap_day=int(d["triple_swap_day"]),
            margin_initial=float(d["margin_initial"]),
            volume_min=float(d["volume_min"]),
            volume_max=float(d["volume_max"]),
            volume_step=float(d["volume_step"]),
            source="default",
            latency_ms_per_side=DEFAULT_LATENCY_MS_PER_SIDE,
        )

    def to_dict(self) -> Dict:
        """Serialise to plain dict (JSON-safe)."""
        return {
            "symbol": self.symbol,
            "digits": self.digits,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "contract_size": self.contract_size,
            "spread_points": self.spread_points,
            # --- CSV-measured spread (highest reliability) ---
            "csv_measured_spread_points": self.csv_measured_spread_points,
            "csv_spread_measured_at": self.csv_spread_measured_at,
            "csv_spread_n_bars": self.csv_spread_n_bars,
            # --- Category C: manually-reviewed costs ---
            "typical_spread_points": self.typical_spread_points,
            "typical_spread_reviewed_at": self.typical_spread_reviewed_at,
            "commission_per_lot": self.commission_per_lot,
            "commission_reviewed_at": self.commission_reviewed_at,
            "is_commission_free": self.is_commission_free,
            "slippage_points": self.slippage_points,
            "slippage_reviewed_at": self.slippage_reviewed_at,
            "latency_ms_per_side": self.latency_ms_per_side,
            # -------------------------------------------
            "swap_long_points": self.swap_long_points,
            "swap_short_points": self.swap_short_points,
            # USD overrides — round-tripped so they survive spec re-serialisation
            "swap_long_usd_per_lot_per_eff_day": self.swap_long_usd_per_lot_per_eff_day,
            "swap_short_usd_per_lot_per_eff_day": self.swap_short_usd_per_lot_per_eff_day,
            "triple_swap_day": self.triple_swap_day,
            "margin_initial": self.margin_initial,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "source": self.source,
            "polled_at": self.polled_at,
            "broker_symbol": self.broker_symbol,
            # Bid/ask snapshot — present only for polled specs
            "bid": self.bid if self.bid > 0 else None,
            "ask": self.ask if self.ask > 0 else None,
            # Quote-currency FX conversion (USD account)
            "quote_currency": self.quote_currency,
            "quote_usd_rate": self.quote_usd_rate,
            # Broker provenance — top-level alias for contamination detection.
            # Mirrors the value already in broker_friction.source so that
            # save_specs() can check broker_source without unwrapping the
            # nested broker_friction dict (e.g. on a first write before
            # broker_friction exists).
            "broker_source": self.source,
            # Derived convenience fields (read-only, not re-loaded)
            "tick_value_usd": round(self.tick_value_usd, 8),
            "spread_usd_per_lot": round(self.spread_usd_per_lot, 6),
            "spread_pct_of_mid": round(self.spread_pct_of_mid, 6),
            "swap_long_usd_per_day": round(self.swap_long_usd_per_day, 6),
            "swap_short_usd_per_day": round(self.swap_short_usd_per_day, 6),
            # Latency slippage reference (informational, 1-lot at 1× ATR)
            "ecn_commission_rt_usd": ECN_COMMISSION_RT_USD,
            "latency_ms_round_trip": self.latency_ms_per_side * 2,
        }


# ---------------------------------------------------------------------------
# Spec registry / loader
# ---------------------------------------------------------------------------

# Module-level cache so each symbol is loaded from disk only once per process.
_spec_cache: Dict[str, InstrumentSpec] = {}


def _find_contract_spec_path(symbol: str) -> Optional[Path]:
    """Locate the contract_spec.json for *symbol* inside master_standardized.

    ECN IDENTITY RULE
    -----------------
    ``XAUUSD+`` and ``XAUUSD`` are **distinct** instruments with different
    cost structures:
    • ECN  (``+``): raw interbank spread + commission per lot.
    • Standard    : spread includes broker markup, no separate commission.
    • Swap rates may also differ between account types on some brokers.

    Search order (strict → lenient):
    1. Exact match on folder name (XAUUSD+ → XAUUSD+/, XAUUSD → XAUUSD/).
       This is the correct result in almost all cases.
    2. ECN cross-variant fallback — only reached when the exact folder does
       NOT exist.  If searching for ``XAUUSD`` and only ``XAUUSD+/`` exists
       (or vice versa), use it with a module-level cache note.  This avoids
       falling back to generic defaults when the user has polled only one
       variant; the contract size, tick size, and digits are identical — only
       the cost fields differ, and costs from the cross-variant are still
       far better than generic defaults.
    Both searches walk up to four layout depths so the multi-broker folder
    structure is supported:
    - depth 2: {category}/{SYMBOL}/                          (legacy)
    - depth 3: {broker}/{category}/{SYMBOL}/                 (MetaAPI)
    - depth 4: {broker_type}/{account}/{category}/{SYMBOL}/  (cTrader)

    Returns None when nothing is found.
    """
    if not _MASTER_DIR.exists():
        return None

    sym_upper = symbol.upper()
    sym_is_ecn = sym_upper.endswith("+")
    sym_bare = sym_upper.rstrip("+")
    # Cross-variant: if we searched for XAUUSD look for XAUUSD+, and vice versa.
    sym_cross = (sym_bare + "+") if not sym_is_ecn else sym_bare

    def _iter_inst_dirs():
        """Yield every instrument-level directory under _MASTER_DIR (depth 2–4).

        Layout variants supported:
          depth 2: _MASTER_DIR/{category}/{SYMBOL}/
          depth 3: _MASTER_DIR/{broker}/{category}/{SYMBOL}/         (MetaAPI)
          depth 4: _MASTER_DIR/{broker_type}/{account}/{category}/{SYMBOL}/  (cTrader)
        """
        _SKIP = {"denoised", "__pycache__", ".git"}
        for d1 in _MASTER_DIR.iterdir():
            if not d1.is_dir() or d1.name.startswith(".") or d1.name in _SKIP:
                continue
            for d2 in d1.iterdir():
                if not d2.is_dir() or d2.name.startswith("."):
                    continue
                yield d2  # depth-2: d1=category, d2=symbol
                for d3 in d2.iterdir():
                    if not d3.is_dir() or d3.name.startswith("."):
                        continue
                    yield d3  # depth-3: d1=broker, d2=category, d3=symbol
                    for d4 in d3.iterdir():
                        if d4.is_dir() and not d4.name.startswith("."):
                            yield d4  # depth-4: d1=broker_type, d2=account, d3=category, d4=symbol

    exact_match: Optional[Path] = None
    cross_match: Optional[Path] = None

    for inst_dir in _iter_inst_dirs():
        name_upper = inst_dir.name.upper()
        candidate = inst_dir / "contract_spec.json"
        if not candidate.exists():
            continue
        if name_upper == sym_upper:
            exact_match = candidate
            break  # exact match wins immediately
        if name_upper == sym_cross and cross_match is None:
            cross_match = candidate  # remember first cross-variant hit

    if exact_match is not None:
        return exact_match

    if cross_match is not None:
        # Log once per process so callers know a cross-variant is being used.
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "friction_cost: no exact spec for %r — using cross-variant %r "
            "(contract size / tick size identical; cost fields may differ between "
            "ECN and standard account types)",
            symbol,
            cross_match.parent.name,
        )
        return cross_match

    return None


def load_spec(symbol: str, force_reload: bool = False) -> InstrumentSpec:
    """Load InstrumentSpec for a symbol.

    Resolution order
    ----------------
    1. Module-level cache (skips disk read after first load).
    2. data/master_standardized/<category>/<SYMBOL>/contract_spec.json
       ── canonical location written by poll_symbol_specs.py + reorganize_data.py.
       ── broker_friction section is read; contains FX-corrected USD cost fields
          (quote_usd_rate, tick_value_usd, swap_*_usd, annual_carry_*).
    3. data/symbol_specs/<SYMBOL>.json  (legacy flat file, backward-compat).
       ── Also tries <SYMBOL>+.json for ECN variants.
    4. Conservative generic defaults (asset-class pattern overrides applied).

    Args:
        symbol:       Instrument name, e.g. "GBPUSD", "XAUUSD", "BTCUSD".
                      The '+' ECN suffix is stripped for cache keying but the
                      search checks both variants on disk.
        force_reload: Bypass cache and re-read from disk.

    Returns:
        InstrumentSpec — always succeeds; falls back to generic defaults when
        no polled data exists.
    """
    clean = symbol.upper().replace("+", "")

    if not force_reload and clean in _spec_cache:
        return _spec_cache[clean]

    spec: Optional[InstrumentSpec] = None
    spec_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # 1. Canonical: contract_spec.json inside per-instrument folder
    # ------------------------------------------------------------------
    contract_path = _find_contract_spec_path(clean)
    if contract_path and contract_path.exists():
        spec_path = contract_path
        logger.debug("Loading spec for %s from contract_spec.json: %s", clean, contract_path)

    # ------------------------------------------------------------------
    # 2. Legacy fallback: flat data/symbol_specs/<SYMBOL>.json
    # ------------------------------------------------------------------
    if spec_path is None:
        legacy = _SPECS_DIR / f"{clean}.json"
        if not legacy.exists():
            ecn = _SPECS_DIR / f"{clean}+.json"
            if ecn.exists():
                legacy = ecn
                logger.debug("Using ECN variant legacy spec %s for symbol %s", ecn.name, clean)
        if legacy.exists():
            spec_path = legacy
            logger.debug("Loading spec for %s from legacy path: %s", clean, legacy)

    # ------------------------------------------------------------------
    # 3. Parse whichever path we found
    # ------------------------------------------------------------------
    if spec_path is not None:
        try:
            raw = json.loads(spec_path.read_text())
            spec = InstrumentSpec.from_polled_json(raw)

            # ----------------------------------------------------------
            # Symbol mismatch guard
            # ----------------------------------------------------------
            # A spec file might have been written for a different symbol
            # (e.g. stale copy, manual file move, reorganise bug).  When
            # broker_friction.symbol is present and does NOT match what
            # was requested, the spec is silently wrong and any friction
            # cost computed from it will be for the wrong instrument.
            #
            # Rule: if the parsed spec.symbol resolves to a non-UNKNOWN
            # value AND it doesn't match the requested ``clean`` symbol,
            # reject the spec and fall back to generic defaults so the
            # caller at least gets a safe, labelled placeholder instead
            # of silently-incorrect numbers.
            #
            # Cross-variant ECN/standard pairs (XAUUSD vs XAUUSD+) are
            # intentionally allowed — both strip to the same bare symbol
            # after removing the '+' suffix.
            loaded_sym = spec.symbol.upper().replace("+", "")
            if loaded_sym not in ("UNKNOWN", "") and loaded_sym != clean:
                logger.warning(
                    "Symbol mismatch loading spec from %s: "
                    "requested '%s' but the file identifies itself as '%s'. "
                    "Rejecting the spec and using generic defaults to prevent "
                    "wrong friction costs being applied to the wrong instrument. "
                    "Re-run reorganize_data.py or re-poll to fix the spec file.",
                    spec_path,
                    clean,
                    spec.symbol,
                )
                spec = None
            else:
                # Staleness warning (non-fatal)
                if spec.polled_at:
                    try:
                        polled = datetime.fromisoformat(spec.polled_at.replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - polled).total_seconds() / 86400
                        if age_days > _STALE_DAYS:
                            logger.warning(
                                "Spec for %s is %.0f days old (> %d). "
                                "Re-poll via Menu 2.8 or: "
                                "python scripts/data/poll_symbol_specs.py",
                                clean,
                                age_days,
                                _STALE_DAYS,
                            )
                    except Exception:
                        pass

                logger.debug("Loaded spec for %s from %s (source=metaapi)", clean, spec_path)

        except Exception as exc:
            logger.warning("Failed to parse %s: %s — using generic defaults", spec_path, exc)
            spec = None

    # ------------------------------------------------------------------
    # 4. Generic defaults
    # ------------------------------------------------------------------
    if spec is None:
        logger.info(
            "No spec file found for %s — using generic defaults. "
            "Poll specs via Menu 2.8 or: python scripts/data/poll_symbol_specs.py",
            clean,
        )
        spec = InstrumentSpec.generic_default(clean)

    _spec_cache[clean] = spec
    return spec


def clear_cache() -> None:
    """Clear the module-level spec cache (useful in tests or after re-polling)."""
    _spec_cache.clear()


def list_available_specs() -> List[str]:
    """Return sorted list of symbols that have polled spec files on disk."""
    if not _SPECS_DIR.exists():
        return []
    return sorted(
        p.stem.upper() for p in _SPECS_DIR.glob("*.json") if not p.stem.startswith("symbol_specs")
    )


# ---------------------------------------------------------------------------
# CSV-measured spread utility
# ---------------------------------------------------------------------------


def measure_spread_from_csv(
    symbol: str,
    data_dir: Optional[Path] = None,
    timeframe: str = "H4",
    percentile: float = 75.0,
    min_bars: int = 100,
) -> Optional[float]:
    """Measure typical spread in points from the actual OHLCV CSV data.

    Reads the largest matching ``<timeframe>`` CSV for *symbol* from
    ``data/master_standardized/``, extracts the ``spread`` column, and
    returns the *percentile*-th value of all non-zero spread bars.
    If no files exist for the requested timeframe, falls back to M1 and
    then to any timeframe for the same symbol.

    Using the 75th percentile (default) rather than the mean guards against
    occasional off-hours spread spikes while still being conservative enough
    for friction modelling.

    Args:
        symbol:     Instrument name, e.g. ``"EURUSD"`` or ``"XAUUSD"``.
                    Both ECN (``+``) and non-ECN files are searched; the
                    ECN version is preferred when both exist.
        data_dir:   Root of ``master_standardized/``.  Defaults to the
                    canonical project path.
        timeframe:  Timeframe suffix to search for, e.g. ``"H4"``.
        percentile: Which percentile of non-zero spreads to return.
                    75 is the canonical Kinetra default.
        min_bars:   Minimum non-zero spread bars required; returns ``None``
                    when fewer bars are available.

    Returns:
        Spread in **points** (same unit as ``InstrumentSpec.spread_points``),
        or ``None`` if no CSV is found or the spread column is all zeros.
    """
    root = data_dir or _MASTER_DIR
    sym_clean = symbol.upper().replace("+", "")

    def _find_candidates(tf: str) -> List[Path]:
        def _rank(p: Path) -> Tuple[int, int, int]:
            s = p.as_posix().lower()
            name = p.name.lower()
            stem_tokens = p.stem.split("_")
            has_timestamp_range = (
                len(stem_tokens) >= 4
                and stem_tokens[-1].isdigit()
                and stem_tokens[-2].isdigit()
                and len(stem_tokens[-1]) >= 8
                and len(stem_tokens[-2]) >= 8
            )

            score = 0
            if "ctrader/" in s:
                score += 50
            if "_demo_" in s:
                score += 40
            if has_timestamp_range:
                score += 80
            if "generated" in name or "current" in name or "accurate" in name:
                score -= 40
            return (score, int(p.stat().st_mtime), int(p.stat().st_size))

        # Prefer ECN (+) files.
        files: List[Path] = list(root.rglob(f"*{sym_clean}+*_{tf}_*.csv"))
        if files:
            return sorted(files, key=_rank, reverse=True)

        files = list(root.rglob(f"*{sym_clean}*_{tf}_*.csv"))
        return sorted(files, key=_rank, reverse=True)

    tf = timeframe.upper()
    candidates = _find_candidates(tf)
    if not candidates and tf != "M1":
        candidates = _find_candidates("M1")
    if not candidates:
        # Last resort: any timeframe for the symbol.
        candidates = _find_candidates("*")

    if not candidates:
        logger.debug(
            "measure_spread_from_csv: no %s %s file found for %s", timeframe, sym_clean, symbol
        )
        return None

    csv_path = candidates[0]
    spread_vals: List[float] = []
    close_samples: List[str] = []

    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if "spread" not in (reader.fieldnames or []):
                logger.debug("measure_spread_from_csv: no spread column in %s", csv_path.name)
                return None
            for row in reader:
                sp_str = str(row.get("spread", "") or "").strip()
                if not sp_str:
                    continue
                try:
                    sp = float(sp_str)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(sp) and sp > 0:
                    spread_vals.append(sp)
                    if len(close_samples) < 50:
                        close_samples.append(row.get("close", ""))
    except Exception as exc:
        logger.warning("measure_spread_from_csv: failed to read %s — %s", csv_path, exc)
        return None

    if len(spread_vals) < min_bars:
        logger.debug(
            "measure_spread_from_csv: only %d non-zero bars (need %d) for %s",
            len(spread_vals),
            min_bars,
            symbol,
        )
        return None

    result = float(np.percentile(spread_vals, percentile))
    logger.debug(
        "measure_spread_from_csv: %s  p%d=%.1f pts  n=%d  file=%s",
        symbol,
        int(percentile),
        result,
        len(spread_vals),
        csv_path.name,
    )
    return result


def build_spec_with_csv_spread(
    symbol: str,
    data_dir: Optional[Path] = None,
    timeframe: str = "H4",
    percentile: float = 75.0,
    commission_rt_usd: float = ECN_COMMISSION_RT_USD,
    latency_ms_per_side: float = DEFAULT_LATENCY_MS_PER_SIDE,
    force_reload: bool = False,
) -> InstrumentSpec:
    """Load InstrumentSpec and enrich it with CSV-measured spread + ECN defaults.

    This is the canonical factory for production use.  It:

    1. Calls ``load_spec(symbol)`` to get swap rates, contract size, etc.
       from ``contract_spec.json``.
    2. Measures the actual spread from the H4 CSV and stores it in
       ``csv_measured_spread_points`` (highest priority in the spread chain).
    3. Ensures ``commission_per_lot`` is set to the canonical ECN value
       ($3.50/side) unless the spec explicitly carries a zero (commission-free).
    4. Sets ``latency_ms_per_side`` for use in latency slippage calculations.

    The returned spec is NOT cached in ``_spec_cache`` (it carries runtime-
    measured spread data that changes as new CSV bars arrive).

    Args:
        symbol:             Instrument name.
        data_dir:           Override for the master_standardized root.
        timeframe:          CSV timeframe to measure spread from.
        percentile:         Spread percentile (75 = canonical Kinetra default).
        commission_rt_usd:  Round-trip commission in USD.  Halved and stored
                            as commission_per_lot (per-side).
        latency_ms_per_side: One-way latency in milliseconds.
        force_reload:       Force re-read of contract_spec.json from disk.

    Returns:
        Enriched ``InstrumentSpec`` ready for ``FrictionCalculator``.
    """
    spec = load_spec(symbol, force_reload=force_reload)

    # Measure spread from CSV data — most reliable source
    csv_pts = measure_spread_from_csv(
        symbol, data_dir=data_dir, timeframe=timeframe, percentile=percentile
    )
    if csv_pts is not None and csv_pts > 0:
        object.__setattr__(spec, "csv_measured_spread_points", csv_pts)
        object.__setattr__(spec, "csv_spread_measured_at", datetime.now(timezone.utc).isoformat())
    else:
        logger.info(
            "build_spec_with_csv_spread: no CSV spread for %s — "
            "falling back to typical_spread_points / bid-ask chain",
            symbol,
        )

    # Lock in ECN commission unless explicitly marked commission-free.
    comm_per_side = commission_rt_usd / 2.0
    if spec.commission_per_lot <= 0.0:
        if getattr(spec, "is_commission_free", False):
            logger.debug(
                "build_spec_with_csv_spread: %s explicitly commission-free (spread-only)",
                symbol,
            )
        elif _is_ecn_commission_symbol(symbol):
            object.__setattr__(spec, "commission_per_lot", comm_per_side)
            logger.debug(
                "build_spec_with_csv_spread: %s had zero commission in spec; "
                "applied ECN default %.2f USD/lot/side",
                symbol,
                comm_per_side,
            )
        else:
            logger.debug(
                "build_spec_with_csv_spread: %s treated as commission-free (non-ECN symbol)",
                symbol,
            )
    else:
        object.__setattr__(spec, "commission_per_lot", comm_per_side)

    # Set latency
    object.__setattr__(spec, "latency_ms_per_side", latency_ms_per_side)

    return spec


# ---------------------------------------------------------------------------
# FrictionBreakdown  — per-component cost result
# ---------------------------------------------------------------------------


@dataclass
class FrictionBreakdown:
    """
    Full breakdown of trading costs for a position.

    All values in USD (account currency).
    Convention: costs are stored as POSITIVE numbers representing the magnitude
    of the drag so callers can always do: net_pnl = gross_pnl - breakdown.total_usd

    swap_usd sign convention (consistent with all other cost fields):
        swap_usd > 0  →  carry COST   (debit, dragging equity down)
        swap_usd < 0  →  carry CREDIT (reduces total friction, e.g. positive-carry short)

    Note: broker swap_*_points values follow MT5 convention (negative = cost).
    FrictionCalculator negates those raw points before storing them here so
    the POSITIVE = COST invariant is preserved throughout this class.
    """

    # Transaction costs (paid once at entry)
    spread_usd: float = 0.0  # Round-trip spread (entry + exit spread combined)
    commission_usd: float = 0.0  # Both sides: 2 × commission_per_lot × lots
    slippage_usd: float = 0.0  # Both sides: entry + exit execution slippage

    # Holding costs (accumulate daily)
    swap_usd: float = 0.0  # Total swap across all rollover events
    # Positive = cost (debit), negative = carry credit

    # Metadata for diagnostics
    normal_rollovers: int = 0  # Number of normal (1×) overnight rollovers
    triple_rollovers: int = 0  # Number of triple (3×) overnight rollovers
    holding_days: int = 0  # Calendar days held (for display)

    @property
    def transaction_usd(self) -> float:
        """One-way transaction costs (spread + commission + slippage)."""
        return self.spread_usd + self.commission_usd + self.slippage_usd

    @property
    def total_usd(self) -> float:
        """
        Total friction in USD.

        Note: swap_usd can be negative (carry credit) which reduces total cost.
        We add it directly so carry trades show lower net friction.
        """
        return self.transaction_usd + self.swap_usd

    def as_pct_of(self, notional_usd: float) -> float:
        """Total friction as % of notional trade value."""
        if notional_usd <= 0:
            return 0.0
        return (self.total_usd / notional_usd) * 100.0

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"spread=${self.spread_usd:.2f}  "
            f"comm=${self.commission_usd:.2f}  "
            f"slip=${self.slippage_usd:.2f}  "
            f"swap=${self.swap_usd:+.2f}  "
            f"TOTAL=${self.total_usd:.2f}"
        )

    def to_dict(self) -> Dict:
        return {
            "spread_usd": round(self.spread_usd, 4),
            "commission_usd": round(self.commission_usd, 4),
            "slippage_usd": round(self.slippage_usd, 4),
            "swap_usd": round(self.swap_usd, 4),
            "transaction_usd": round(self.transaction_usd, 4),
            "total_usd": round(self.total_usd, 4),
            "normal_rollovers": self.normal_rollovers,
            "triple_rollovers": self.triple_rollovers,
            "holding_days": self.holding_days,
        }


# ---------------------------------------------------------------------------
# Swap calendar helpers  (vectorised, no Python day loops)
# ---------------------------------------------------------------------------

# Mapping from our 1=Mon…7=Sun convention to pandas dayofweek (0=Mon…6=Sun)
_OUR_DOW_TO_PANDAS: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


def _count_rollovers(
    entry_dt: datetime,
    exit_dt: datetime,
    triple_swap_day: int,
) -> Tuple[int, int]:
    """
    Count overnight swap rollovers between entry and exit datetimes.

    Rules (standard MT5 / broker convention):
      - A rollover is charged once per TRADING DAY the position is held
        overnight (i.e. crosses the broker's daily rollover time, ~22:00 GMT).
      - Weekends (Sat, Sun) do NOT generate additional rollovers — the triple
        swap on Wednesday already covers the two weekend days.
      - If entry and exit are on the same calendar day, no rollover is charged.

    Args:
        entry_dt:         Position entry datetime.
        exit_dt:          Position exit datetime.
        triple_swap_day:  Day of week that carries 3× swap (1=Mon…7=Sun).

    Returns:
        (normal_rollovers, triple_rollovers)
        Normal rollovers carry 1× the daily rate.
        Triple rollovers carry 3× the daily rate (covers the weekend).
    """
    entry_date = entry_dt.date() if isinstance(entry_dt, datetime) else entry_dt
    exit_date = exit_dt.date() if isinstance(exit_dt, datetime) else exit_dt

    if exit_date <= entry_date:
        return 0, 0

    # All calendar dates from entry_date up to (but not including) exit_date.
    # We do NOT include exit_date because you don't pay a rollover on the day
    # you close the position (unless you hold past the rollover time, which we
    # conservatively ignore for simplicity).
    date_range = pd.date_range(start=entry_date, end=exit_date - timedelta(days=1), freq="D")

    # Filter to weekdays only (Mon=0…Fri=4)
    trading_days = date_range[date_range.dayofweek < 5]

    if len(trading_days) == 0:
        return 0, 0

    pandas_triple = _OUR_DOW_TO_PANDAS.get(triple_swap_day, 2)  # default Wednesday
    is_triple = trading_days.dayofweek == pandas_triple

    triple_count = int(is_triple.sum())
    normal_count = len(trading_days) - triple_count

    return normal_count, triple_count


def _build_daily_swap_series(
    index: pd.DatetimeIndex,
    spec: InstrumentSpec,
    is_long: bool,
    lots: float,
) -> pd.Series:
    """
    Build a per-bar accumulated swap series aligned to a price DataFrame index.

    For each bar we determine whether a rollover would have been charged since
    the previous bar.  This is used by BuyHoldCalculator to build a mark-to-
    market equity curve that correctly reflects daily swap drag.

    Only trading days (Mon–Fri) that cross the broker rollover generate a swap
    charge.  For intraday data (H1, H4) we fire the swap on the first bar of
    each new trading day (i.e. the bar whose date differs from the previous bar).

    Returns:
        pd.Series of cumulative swap cost (USD) at each bar, starting at 0.
        Values are COSTS (positive = drain on equity).
    """
    # Use the authoritative USD swap value when available (avoids the
    # swap_points × tick_value_usd mis-scaling bug for non-FX instruments).
    rate_usd_per_day = spec.swap_long_usd_per_day if is_long else spec.swap_short_usd_per_day
    # Negate: MT5 convention uses negative rate for a cost (debit).
    # FrictionBreakdown convention requires positive values for costs so that
    # total_usd = transaction + swap_usd correctly inflates friction for carry
    # costs and deflates it for carry credits.
    rate_usd_per_lot = -(rate_usd_per_day * lots)  # positive = cost

    pandas_triple = _OUR_DOW_TO_PANDAS.get(spec.triple_swap_day, 2)

    # Detect bar boundaries that cross into a new trading day
    dates = index.normalize()  # floor to midnight
    new_day = np.zeros(len(index), dtype=float)

    if len(index) > 1:
        day_changed = dates[1:] != dates[:-1]
        day_of_week = dates[1:].dayofweek  # 0=Mon … 6=Sun

        # Only charge on trading day rollovers (Mon–Fri)
        is_trading_day = day_of_week < 5
        is_rollover = day_changed & is_trading_day

        # Triple swap on the designated day
        is_triple = is_rollover & (day_of_week == pandas_triple)
        is_normal = is_rollover & ~is_triple

        daily_cost = np.zeros(len(index) - 1, dtype=float)
        daily_cost[is_normal] = rate_usd_per_lot  # positive = drain on equity
        daily_cost[is_triple] = rate_usd_per_lot * 3.0  # triple-swap day

        new_day[1:] = daily_cost

    cumulative = np.cumsum(new_day)
    return pd.Series(cumulative, index=index, name="swap_cumulative_usd")


# ---------------------------------------------------------------------------
# FrictionCalculator  — all cost arithmetic in one place
# ---------------------------------------------------------------------------


class FrictionCalculator:
    """
    Calculate all trading friction costs for a given instrument.

    Instantiate once per symbol, reuse for all cost calculations.

    Friction components
    -------------------
    1. Spread     — from ``spec.spread_usd_per_lot`` (CSV-measured preferred)
    2. Commission — ECN $7 RT / $3.50 per side from ``spec.commission_per_lot``
    3. Latency    — ``latency_slippage_usd(atr, lots)`` requires current ATR
    4. Swap       — ``swap_cost_usd(...)`` or ``daily_swap_usd(...)``

    Use ``all_in_breakdown()`` for a single-call summary of all four.

    Example:
        calc = get_calculator_with_data("GBPUSD")
        bd   = calc.all_in_breakdown(price=1.268, lots=1.0, is_long=True,
                                      holding_bars=10, atr=0.0035)
        print(bd.summary())
    """

    def __init__(self, spec: InstrumentSpec) -> None:
        self.spec = spec

    # ------------------------------------------------------------------
    # Entry / exit component calculations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Latency slippage
    # ------------------------------------------------------------------

    def latency_slippage_usd(
        self,
        atr: float,
        lots: float,
        timeframe: str = "H4",
    ) -> float:
        """Estimate one-side execution slippage due to order-transit latency (USD).

        Models adverse price drift during the ``spec.latency_ms_per_side``
        window using a random-walk approximation:

            slippage_price = ATR × sqrt(latency_ms / bar_duration_ms)

        This gives the expected price drift during latency as a fraction of
        the bar's volatility.  For H4 at 500 ms:
            sqrt(500 / 14_400_000) ≈ 0.00589
        So slippage ≈ 0.59 % of ATR per side — small but non-trivial for
        high-spread instruments where ATR is already tight.

        The result is converted to USD using the same quote_usd_rate and
        contract_size as all other cost components.

        Args:
            atr:        Current bar ATR in price units (same currency as
                        ``price`` — i.e. quote currency of the instrument).
            lots:       Position size in lots.
            timeframe:  Bar timeframe; used to look up bar_duration_ms.
                        Defaults to ``"H4"``.

        Returns:
            Latency slippage in USD for ONE side of the trade.
            Multiply by 2 for round-trip (entry + exit).
        """
        bar_ms = _BAR_DURATION_MS.get(timeframe.upper(), _BAR_DURATION_MS["H4"])
        latency_ms = self.spec.latency_ms_per_side
        if bar_ms <= 0 or latency_ms <= 0 or atr <= 0:
            return 0.0
        # Adverse drift during latency under a random-walk model
        drift_price = atr * (latency_ms / bar_ms) ** 0.5
        return drift_price * self.spec.contract_size * self.spec.quote_usd_rate * lots

    def latency_slippage_pct(
        self,
        price: float,
        atr: float,
        lots: float = 1.0,
        timeframe: str = "H4",
    ) -> float:
        """One-side latency slippage as % of notional trade value."""
        notional = self.spec.notional_usd(price, lots)
        if notional <= 0:
            return 0.0
        return 100.0 * self.latency_slippage_usd(atr, lots, timeframe) / notional

    # ------------------------------------------------------------------
    # Entry / exit component calculations
    # ------------------------------------------------------------------

    def entry_cost_usd(
        self,
        price: float,
        lots: float,
        atr: Optional[float] = None,
        timeframe: str = "H4",
    ) -> float:
        """
        Cost to OPEN a position (USD).

        Includes:
          - Full spread (market order pays the spread at entry)
          - Commission (entry side only — ECN $3.50)
          - Slippage: latency-based when *atr* provided; fixed fallback otherwise

        Args:
            price:     Entry price (used only for notional; not affecting USD cost
                       directly when tick_value_usd is already FX-corrected).
            lots:      Position size in lots.
            atr:       Current bar ATR in price units.  When provided, latency
                       slippage is computed from the physics-based model.
                       When ``None``, falls back to ``spec.slippage_points``.
            timeframe: Bar timeframe for latency scaling.
        """
        spread = self.spec.spread_usd_per_lot * lots
        commission = self.spec.commission_per_lot * lots
        if atr is not None:
            slippage = self.latency_slippage_usd(atr, lots, timeframe)
        else:
            slippage = self.spec.slippage_points * self.spec.tick_value_usd * lots
        return spread + commission + slippage

    def exit_cost_usd(
        self,
        price: float,
        lots: float,
        atr: Optional[float] = None,
        timeframe: str = "H4",
    ) -> float:
        """
        Cost to CLOSE a position (USD).

        Includes:
          - Commission (exit side only — ECN $3.50)
          - Slippage: latency-based when *atr* provided; fixed fallback otherwise
          Note: spread is NOT charged again on exit (it was paid at entry).
        """
        commission = self.spec.commission_per_lot * lots
        if atr is not None:
            slippage = self.latency_slippage_usd(atr, lots, timeframe)
        else:
            slippage = self.spec.slippage_points * self.spec.tick_value_usd * lots
        return commission + slippage

    def swap_cost_usd(
        self,
        is_long: bool,
        lots: float,
        entry_dt: datetime,
        exit_dt: datetime,
        price: float = 0.0,  # kept for API compatibility; not used in POINTS mode
    ) -> Tuple[float, int, int]:
        """
        Calculate total swap cost for a holding period (USD).

        Handles triple-swap day correctly using vectorised calendar arithmetic.

        Args:
            is_long:   True for long position, False for short.
            lots:      Position size in lots.
            entry_dt:  Position entry datetime.
            exit_dt:   Position exit datetime.
            price:     Current price (unused in POINTS swap mode — kept for
                       API forward-compatibility with INTEREST swap mode).

        Returns:
            (total_swap_usd, normal_rollovers, triple_rollovers)
            total_swap_usd is POSITIVE for costs, NEGATIVE for carry credits.
            This matches the FrictionBreakdown convention so callers can do:
                net_pnl = gross_pnl - friction.total_usd
        """
        normal, triple = _count_rollovers(entry_dt, exit_dt, self.spec.triple_swap_day)

        rate_points = self.spec.swap_long_points if is_long else self.spec.swap_short_points
        # POINTS mode: raw_per_day carries the MT5 sign (negative = cost).
        # Negate to convert to FrictionBreakdown convention (positive = cost).
        raw_per_day = rate_points * self.spec.tick_value_usd * lots
        total_swap = -(raw_per_day * (normal + 3 * triple))
        return total_swap, normal, triple

    # ------------------------------------------------------------------
    # Complete round-trip (the primary method for backtesting / B&H)
    # ------------------------------------------------------------------

    def round_trip(
        self,
        is_long: bool,
        lots: float,
        entry_price: float,
        exit_price: float,
        entry_dt: datetime,
        exit_dt: datetime,
    ) -> "RoundTripResult":
        """
        Calculate the complete economics of one round-trip trade.

        Returns a RoundTripResult containing:
          - gross_pnl_usd:  raw P&L from price movement (before costs)
          - friction:       FrictionBreakdown with each cost component
          - net_pnl_usd:    gross_pnl_usd − friction.total_usd

        Usage:
            result = calc.round_trip(True, 1.0, 1.268, 1.272, t_in, t_out)
            print(f"Net: ${result.net_pnl_usd:.2f}")
        """
        # --- Gross P&L ---
        price_diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
        # Multiply by quote_usd_rate to convert from quote currency → USD.
        # For USD-quoted instruments (XAUUSD, EURUSD) the rate is 1.0 — no change.
        # For non-USD quotes (XAUEUR → EUR, XAUJPY → JPY) this is essential.
        gross_pnl = price_diff * self.spec.contract_size * self.spec.quote_usd_rate * lots

        # --- Friction ---
        swap_usd, normal_r, triple_r = self.swap_cost_usd(
            is_long, lots, entry_dt, exit_dt, entry_price
        )

        # Spread is already included in entry_cost_usd (full round-trip spread
        # is charged at entry for market orders — exit spread = 0).
        friction = FrictionBreakdown(
            spread_usd=self.spec.spread_usd_per_lot * lots,  # uses bid/ask when available
            commission_usd=(self.spec.commission_per_lot * lots * 2),
            slippage_usd=(self.spec.slippage_points * self.spec.tick_value_usd * lots * 2),
            swap_usd=swap_usd,
            normal_rollovers=normal_r,
            triple_rollovers=triple_r,
            holding_days=(exit_dt.date() - entry_dt.date()).days
            if isinstance(entry_dt, datetime)
            else 0,
        )

        return RoundTripResult(
            symbol=self.spec.symbol,
            is_long=is_long,
            lots=lots,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            gross_pnl_usd=gross_pnl,
            friction=friction,
            spec_source=self.spec.source,
        )

    # ------------------------------------------------------------------
    # Per-bar helpers  (for RL reward shaping)
    # ------------------------------------------------------------------

    def daily_swap_usd(
        self,
        is_long: bool,
        lots: float,
        bar_dt: datetime,
    ) -> float:
        """
        Swap charge for a single bar / day (USD).

        Returns the swap that would be charged on a rollover crossing at
        bar_dt.  Returns 0 if bar_dt falls on a weekend.

        Handles triple-swap day automatically.

        Sign convention (FrictionBreakdown):
            Positive return  →  carry COST   (equity decreases)
            Negative return  →  carry CREDIT (equity increases)

        Use this in the RL environment step() to deduct carry from reward:
            reward -= calc.daily_swap_usd(is_long, lots, bar_dt)
        """
        dow = bar_dt.weekday()  # 0=Mon … 6=Sun

        if dow >= 5:  # Weekend — no swap
            return 0.0

        rate_points = self.spec.swap_long_points if is_long else self.spec.swap_short_points
        # Negate: MT5 negative = cost → FrictionBreakdown positive = cost
        cost_per_day = -(rate_points * self.spec.tick_value_usd * lots)

        pandas_triple = _OUR_DOW_TO_PANDAS.get(self.spec.triple_swap_day, 2)
        multiplier = 3 if dow == pandas_triple else 1

        return cost_per_day * multiplier

    def friction_pct(
        self,
        price: float,
        lots: float,
        is_long: bool = True,
        holding_bars: int = 0,
        bars_per_day: float = 6.0,  # default H4 (6 bars per 24h)
        atr: Optional[float] = None,
        timeframe: str = "H4",
        include_swap: bool = True,
    ) -> float:
        """
        Total friction as a percentage of notional value.

        Intended for RL reward shaping — converts absolute USD costs into a
        normalised signal that is comparable across instruments.

        Components included:
          - Spread (CSV-measured preferred)
          - Commission (ECN $7 RT locked in)
          - Latency slippage (500 ms/side physics model when *atr* provided;
            fixed ``slippage_points`` fallback otherwise)
          - Swap (worst-case of long/short rate × hold days) when
            ``include_swap=True``

        Args:
            price:         Current price.
            lots:          Position size in lots.
            is_long:       True for long, False for short.
            holding_bars:  How many bars the position is expected to be held.
            bars_per_day:  Bars per 24-hour day for the active timeframe
                           (6 for H4, 24 for H1, etc.).
            atr:           Current bar ATR for latency slippage model.
                           Pass ``None`` to use fixed slippage_points fallback.
            timeframe:     Bar timeframe string for latency scaling.
            include_swap:  Include swap (overnight carry) cost.  Set to
                           ``False`` for intraday sizing where no overnight
                           is expected.

        Returns:
            Total friction as % of notional  (e.g. 0.035 = 3.5 basis points).
        """
        notional = self.spec.notional_usd(price, lots)
        if notional <= 0:
            return 0.0

        transaction = self.entry_cost_usd(
            price, lots, atr=atr, timeframe=timeframe
        ) + self.exit_cost_usd(price, lots, atr=atr, timeframe=timeframe)

        swap_approx = 0.0
        if include_swap and holding_bars > 0:
            holding_days = holding_bars / max(bars_per_day, 1.0)
            rate_points = self.spec.swap_long_points if is_long else self.spec.swap_short_points
            # abs() treats carry credits conservatively — never understate friction.
            swap_approx = abs(rate_points * self.spec.tick_value_usd * lots * holding_days)

        total = transaction + swap_approx
        return (total / notional) * 100.0

    def all_in_breakdown(
        self,
        price: float,
        lots: float = 1.0,
        is_long: bool = True,
        holding_bars: int = 10,
        bars_per_day: float = 6.0,
        atr: Optional[float] = None,
        timeframe: str = "H4",
        include_swap: bool = True,
    ) -> "FrictionBreakdown":
        """Return a fully populated FrictionBreakdown for display / tier scoring.

        Provides individual USD amounts for every friction component so callers
        can log, report, or use them in tier-scoring without re-implementing
        the math.

        Args:
            price:         Mid-price for notional computation.
            lots:          Position size in lots.
            is_long:       Direction (affects swap rate selection).
            holding_bars:  Expected hold in bars (swap scales with this).
            bars_per_day:  Bars per 24h for the active timeframe (6 for H4).
            atr:           Current bar ATR for latency slippage (None = fixed).
            timeframe:     Bar timeframe string.
            include_swap:  Include swap cost in the breakdown.

        Returns:
            :class:`FrictionBreakdown` with all components populated.
        """
        spread_usd = self.spec.spread_usd_per_lot * lots
        commission_usd = self.spec.commission_per_lot * 2.0 * lots  # RT = both sides

        if atr is not None:
            slip_entry = self.latency_slippage_usd(atr, lots, timeframe)
            slip_exit = self.latency_slippage_usd(atr, lots, timeframe)
        else:
            slip_entry = self.spec.slippage_points * self.spec.tick_value_usd * lots
            slip_exit = slip_entry
        slippage_usd = slip_entry + slip_exit  # both sides

        swap_usd = 0.0
        normal_rollovers = 0
        triple_rollovers = 0
        if include_swap and holding_bars > 0:
            holding_days = holding_bars / max(bars_per_day, 1.0)
            rate_points = self.spec.swap_long_points if is_long else self.spec.swap_short_points
            # Conservative: use abs so carry credits don't mask other costs here
            swap_usd = abs(rate_points * self.spec.tick_value_usd * lots * holding_days)
            # Approximate rollover counts from hold days
            normal_rollovers = max(0, int(holding_days) - int(holding_days // 7))
            triple_rollovers = int(holding_days // 7)

        return FrictionBreakdown(
            spread_usd=round(spread_usd, 6),
            commission_usd=round(commission_usd, 6),
            slippage_usd=round(slippage_usd, 6),
            swap_usd=round(swap_usd, 6),
            normal_rollovers=normal_rollovers,
            triple_rollovers=triple_rollovers,
            holding_days=int(holding_bars / max(bars_per_day, 1.0)),
        )

    def all_in_pct(
        self,
        price: float,
        lots: float = 1.0,
        is_long: bool = True,
        holding_bars: int = 10,
        bars_per_day: float = 6.0,
        atr: Optional[float] = None,
        timeframe: str = "H4",
        include_swap: bool = True,
    ) -> Dict[str, float]:
        """Return each friction component as % of notional plus a TOTAL key.

        Convenience wrapper around :meth:`all_in_breakdown` for quick
        display in exploration scripts and tier-scoring tables.

        Returns dict with keys:
            spread_pct, commission_pct, latency_pct, swap_pct, total_pct
        """
        notional = self.spec.notional_usd(price, lots)
        bd = self.all_in_breakdown(
            price=price,
            lots=lots,
            is_long=is_long,
            holding_bars=holding_bars,
            bars_per_day=bars_per_day,
            atr=atr,
            timeframe=timeframe,
            include_swap=include_swap,
        )
        if notional <= 0:
            return {
                "spread_pct": 0.0,
                "commission_pct": 0.0,
                "latency_pct": 0.0,
                "swap_pct": 0.0,
                "total_pct": 0.0,
            }

        def _pct(usd: float) -> float:
            return round(100.0 * usd / notional, 6)

        return {
            "spread_pct": _pct(bd.spread_usd),
            "commission_pct": _pct(bd.commission_usd),
            "latency_pct": _pct(bd.slippage_usd),
            "swap_pct": _pct(bd.swap_usd),
            "total_pct": _pct(bd.total_usd),
        }

    def breakeven_points(
        self,
        lots: float = 1.0,
        atr: Optional[float] = None,
        timeframe: str = "H4",
    ) -> float:
        """
        Minimum price move (in points) needed to cover round-trip transaction costs.

        Ignores swap — gives the minimum edge threshold per trade.

        Args:
            lots:      Position size in lots.
            atr:       Current ATR for latency slippage (None = fixed fallback).
            timeframe: Bar timeframe string for latency scaling.
        """
        if atr is not None:
            slip_rt = self.latency_slippage_usd(atr, lots, timeframe) * 2
        else:
            slip_rt = self.spec.slippage_points * self.spec.tick_value_usd * lots * 2

        transaction = (
            self.spec.spread_usd_per_lot * lots + self.spec.commission_per_lot * lots * 2 + slip_rt
        )
        tv_x_lots = self.spec.tick_value_usd * lots
        if tv_x_lots <= 0:
            return 0.0
        return transaction / tv_x_lots


# ---------------------------------------------------------------------------
# RoundTripResult  — trade economics summary
# ---------------------------------------------------------------------------


@dataclass
class RoundTripResult:
    """Complete economics of a single round-trip trade."""

    symbol: str
    is_long: bool
    lots: float
    entry_price: float
    exit_price: float
    entry_dt: datetime
    exit_dt: datetime
    gross_pnl_usd: float
    friction: FrictionBreakdown
    spec_source: str = "default"

    @property
    def net_pnl_usd(self) -> float:
        """P&L after all costs."""
        return self.gross_pnl_usd - self.friction.total_usd

    @property
    def gross_return_pct(self) -> float:
        """Gross return as % of entry notional (1 lot basis)."""
        notional = self.entry_price * 1.0  # per unit, not per lot
        if notional <= 0:
            return 0.0
        price_diff = (
            (self.exit_price - self.entry_price)
            if self.is_long
            else (self.entry_price - self.exit_price)
        )
        return (price_diff / notional) * 100.0

    @property
    def direction(self) -> str:
        return "LONG" if self.is_long else "SHORT"

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.direction}  {self.lots}lot  "
            f"entry={self.entry_price:.5f}  exit={self.exit_price:.5f}  "
            f"gross=${self.gross_pnl_usd:+.2f}  "
            f"{self.friction.summary()}  "
            f"net=${self.net_pnl_usd:+.2f}"
        )

    def to_dict(self) -> Dict:
        d = {
            "symbol": self.symbol,
            "direction": self.direction,
            "lots": self.lots,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_dt": self.entry_dt.isoformat()
            if isinstance(self.entry_dt, datetime)
            else str(self.entry_dt),
            "exit_dt": self.exit_dt.isoformat()
            if isinstance(self.exit_dt, datetime)
            else str(self.exit_dt),
            "gross_pnl_usd": round(self.gross_pnl_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "gross_return_pct": round(self.gross_return_pct, 4),
            "spec_source": self.spec_source,
        }
        d.update(self.friction.to_dict())
        return d


# ---------------------------------------------------------------------------
# BuyHoldResult
# ---------------------------------------------------------------------------


@dataclass
class BuyHoldResult:
    """
    Complete B&H benchmark result for one instrument + direction.

    Used as the comparison baseline in exploration and backtesting instead
    of a random agent or shuffled-returns null hypothesis.

    Why B&H instead of random:
      A random agent has zero expectation by definition.  B&H has a real,
      instrument-specific expectation that accounts for trend (equity indices
      trend up, commodity currencies trend with carry).  Beating a properly-
      costed B&H baseline is a much more meaningful scientific threshold.
    """

    symbol: str
    is_long: bool
    lots: float
    initial_capital: float
    entry_price: float
    exit_price: float
    entry_dt: datetime
    exit_dt: datetime

    # P&L
    gross_pnl_usd: float
    friction: FrictionBreakdown
    net_pnl_usd: float

    # Returns
    gross_return_pct: float
    net_return_pct: float
    cagr_pct: float  # Compounded annual growth rate (net)

    # Risk metrics (from equity curve)
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_annualised: float

    # Equity curve (for plotting)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # Metadata
    spec_source: str = "default"
    n_bars: int = 0
    timeframe: str = ""

    @property
    def direction(self) -> str:
        return "LONG" if self.is_long else "SHORT"

    def summary(self) -> str:
        return (
            f"{self.symbol} B&H {self.direction}  "
            f"gross={self.gross_return_pct:+.2f}%  "
            f"net={self.net_return_pct:+.2f}%  "
            f"CAGR={self.cagr_pct:+.2f}%  "
            f"Sharpe={self.sharpe_ratio:.2f}  "
            f"MaxDD={self.max_drawdown_pct:.1f}%  "
            f"friction=${self.friction.total_usd:,.0f}  "
            f"[{self.spec_source}]"
        )

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "lots": self.lots,
            "initial_capital": self.initial_capital,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_dt": self.entry_dt.isoformat()
            if isinstance(self.entry_dt, datetime)
            else str(self.entry_dt),
            "exit_dt": self.exit_dt.isoformat()
            if isinstance(self.exit_dt, datetime)
            else str(self.exit_dt),
            "gross_pnl_usd": round(self.gross_pnl_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "gross_return_pct": round(self.gross_return_pct, 4),
            "net_return_pct": round(self.net_return_pct, 4),
            "cagr_pct": round(self.cagr_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "volatility_annualised": round(self.volatility_annualised, 4),
            "n_bars": self.n_bars,
            "timeframe": self.timeframe,
            "spec_source": self.spec_source,
            **{
                k: round(v, 4)
                for k, v in self.friction.to_dict().items()
                if k not in ("normal_rollovers", "triple_rollovers", "holding_days")
            },
            "normal_rollovers": self.friction.normal_rollovers,
            "triple_rollovers": self.friction.triple_rollovers,
        }


# ---------------------------------------------------------------------------
# BuyHoldCalculator
# ---------------------------------------------------------------------------


class BuyHoldCalculator:
    """
    Compute a properly-costed Buy & Hold baseline for any instrument.

    Replaces the RandomAgent baseline in exploration and backtesting with a
    scientifically meaningful comparison:

        "Did the RL agent beat simply holding the instrument for the same period,
         after paying all real broker costs?"

    Usage:
        calc = BuyHoldCalculator("GBPUSD")
        result = calc.calculate(df, is_long=True, lots=1.0)
        print(result.summary())

        # Or compare long and short:
        long_r, short_r = calc.both_directions(df)

    What it models:
        1. Entry at first bar close (market order — pays spread + commission + slippage)
        2. Bar-by-bar mark-to-market equity curve
        3. Daily swap accumulation with correct triple-swap handling
        4. Exit at last bar close (pays commission + slippage)
        5. Sharpe ratio, max drawdown, CAGR from the equity curve

    What it does NOT model:
        - Margin calls (assumes sufficient capital)
        - Partial fills / liquidity constraints
        - Gap risk (overnight gaps between sessions)
    """

    def __init__(self, symbol: str) -> None:
        self.spec = load_spec(symbol)
        self.calc = FrictionCalculator(self.spec)

    def calculate(
        self,
        df: pd.DataFrame,
        is_long: bool = True,
        lots: float = 1.0,
        initial_capital: float = 100_000.0,
        bars_per_year: Optional[float] = None,
    ) -> BuyHoldResult:
        """
        Calculate B&H result for a DataFrame of OHLCV data.

        Args:
            df:               OHLCV DataFrame with DatetimeIndex.
                              Must have at least a 'close' column.
            is_long:          True = buy and hold, False = sell and hold short.
            lots:             Position size in lots.
            initial_capital:  Starting account equity (USD).
            bars_per_year:    Override bars-per-year for CAGR/Sharpe scaling.
                              Auto-detected from the index frequency if None.

        Returns:
            BuyHoldResult with full equity curve and all metrics.
        """
        if df is None or len(df) < 2:
            raise ValueError("DataFrame must have at least 2 bars")

        close = df["close"].values.astype(float)
        index = df.index

        entry_price = float(close[0])
        exit_price = float(close[-1])

        entry_dt = index[0].to_pydatetime() if hasattr(index[0], "to_pydatetime") else index[0]
        exit_dt = index[-1].to_pydatetime() if hasattr(index[-1], "to_pydatetime") else index[-1]

        # --- Transaction costs (paid once) ---
        entry_c = self.calc.entry_cost_usd(entry_price, lots)
        exit_c = self.calc.exit_cost_usd(exit_price, lots)

        # --- Bar-by-bar swap accumulation (vectorised) ---
        swap_series = _build_daily_swap_series(index, self.spec, is_long, lots)

        # --- Gross P&L at each bar (mark to market) ---
        # Multiply by quote_usd_rate to convert from quote currency → USD.
        # For USD-quoted instruments the rate is 1.0 — no change.
        # For non-USD quotes (XAUEUR → EUR, XAUJPY → JPY) this is essential.
        qr = self.spec.quote_usd_rate
        if is_long:
            price_pnl = (close - entry_price) * self.spec.contract_size * qr * lots
        else:
            price_pnl = (entry_price - close) * self.spec.contract_size * qr * lots

        # --- Net equity curve ---
        # equity = initial_capital + price_pnl - entry_cost - cumulative_swap
        # swap_series values are POSITIVE for costs (drain equity) so subtracting
        # them correctly reduces equity for carry-cost instruments and increases
        # it for carry-credit instruments.
        equity_curve = pd.Series(
            initial_capital + price_pnl - entry_c - swap_series.values,
            index=index,
        )
        # Deduct exit cost at the last bar
        equity_curve.iloc[-1] -= exit_c

        # --- Final P&L ---
        gross_pnl = price_pnl[-1]
        total_swap_usd = float(swap_series.iloc[-1])
        normal_r, triple_r = _count_rollovers(entry_dt, exit_dt, self.spec.triple_swap_day)

        friction = FrictionBreakdown(
            spread_usd=self.spec.spread_usd_per_lot * lots,  # uses bid/ask when available
            commission_usd=self.spec.commission_per_lot * lots * 2,
            slippage_usd=self.spec.slippage_points * self.spec.tick_value_usd * lots * 2,
            swap_usd=total_swap_usd,
            normal_rollovers=normal_r,
            triple_rollovers=triple_r,
            holding_days=(exit_dt.date() - entry_dt.date()).days
            if isinstance(entry_dt, datetime)
            else int(len(df) / 24),
        )

        net_pnl = gross_pnl - friction.total_usd

        gross_return_pct = (gross_pnl / initial_capital) * 100.0
        net_return_pct = (net_pnl / initial_capital) * 100.0

        # --- CAGR ---
        years = _years_held(entry_dt, exit_dt, len(df), bars_per_year)
        if years > 0 and initial_capital > 0:
            end_equity = float(equity_curve.iloc[-1])
            if end_equity > 0:
                # Standard CAGR: only valid when end_equity is positive.
                # A negative end_equity raised to a fractional power produces a
                # complex number in Python (RuntimeWarning + nan in NumPy, but
                # TypeError in pure-Python arithmetic).
                cagr_pct = ((end_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
            else:
                # Total loss or negative equity — CAGR is meaningfully -100%
                cagr_pct = -100.0
        else:
            cagr_pct = 0.0

        # --- Risk metrics from equity curve ---
        sharpe, max_dd, vol = _equity_curve_metrics(equity_curve, bars_per_year)

        # --- Timeframe detection ---
        timeframe = _detect_timeframe(index)

        return BuyHoldResult(
            symbol=self.spec.symbol,
            is_long=is_long,
            lots=lots,
            initial_capital=initial_capital,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            gross_pnl_usd=round(gross_pnl, 4),
            friction=friction,
            net_pnl_usd=round(net_pnl, 4),
            gross_return_pct=round(gross_return_pct, 4),
            net_return_pct=round(net_return_pct, 4),
            cagr_pct=round(cagr_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            volatility_annualised=round(vol, 4),
            equity_curve=equity_curve,
            spec_source=self.spec.source,
            n_bars=len(df),
            timeframe=timeframe,
        )

    def both_directions(
        self,
        df: pd.DataFrame,
        lots: float = 1.0,
        initial_capital: float = 100_000.0,
    ) -> Tuple[BuyHoldResult, BuyHoldResult]:
        """
        Calculate both long and short B&H.

        Returns (long_result, short_result).

        Useful for instruments where the natural directional bias is unclear
        (e.g. commodities, crypto) — the worse of the two sets the bar that
        the RL agent must beat.
        """
        long_r = self.calculate(df, is_long=True, lots=lots, initial_capital=initial_capital)
        short_r = self.calculate(df, is_long=False, lots=lots, initial_capital=initial_capital)
        return long_r, short_r

    def harder_baseline(
        self,
        df: pd.DataFrame,
        lots: float = 1.0,
        initial_capital: float = 100_000.0,
    ) -> BuyHoldResult:
        """
        Return whichever direction (long or short) produces the HIGHER net return.

        Use this when you want the most demanding comparison baseline — forces
        the RL agent to beat the best passive strategy, not just the weaker one.
        """
        long_r, short_r = self.both_directions(df, lots, initial_capital)
        return long_r if long_r.net_return_pct >= short_r.net_return_pct else short_r


# ---------------------------------------------------------------------------
# Equity curve metric helpers  (private)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NaiveOUCalculator  — mean-reversion null-hypothesis baseline
# ---------------------------------------------------------------------------


@dataclass
class NaiveOUResult:
    """
    Result of the Naïve OU (Ornstein-Uhlenbeck) reversion baseline.

    The correct null hypothesis for mean-reverting instruments.  Instead of
    passively holding (B&H), the naïve OU strategy trades the restoring force
    directly — enter against the deviation, exit when price returns to the mean.

    No feature engineering, no learning — just the OU spring law applied with
    the instrument's own measured spring constant (θ) and adaptive thresholds
    derived from rolling percentiles of the instrument's own deviation
    distribution (no magic σ multiples).

    Why this is the correct null for mean-reverting instruments
    -----------------------------------------------------------
    A mean-reverting market has a restoring force (like a spring).  The naïve
    strategy is: "trade in the direction of the restoring force, using the
    instrument's own measured spring constant."  An RL agent that cannot beat
    the spring law has learned nothing useful about the market.

    Fields
    ------
    symbol              : Instrument name.
    lots                : Position size in lots.
    initial_capital     : Starting account equity (USD).
    ou_speed            : Scalar OU reversion speed θ (estimated over full series).
    half_life_bars      : ln(2) / θ — natural mean-reversion window in bars.
    entry_quantile      : Rolling quantile used for entry threshold (e.g. 0.65).
    n_trades            : Total number of trades (position entries) executed.
    gross_pnl_usd       : Total P&L before friction.
    friction            : FrictionBreakdown aggregated across all trades.
    net_pnl_usd         : gross_pnl_usd − friction.total_usd.
    gross_return_pct    : gross_pnl / initial_capital × 100.
    net_return_pct      : net_pnl / initial_capital × 100.
    cagr_pct            : Compounded annual growth rate (net).
    sharpe_ratio        : Annualised Sharpe (no risk-free rate).
    max_drawdown_pct    : Maximum peak-to-trough drawdown (%).
    volatility_annualised: Annualised volatility of equity curve returns (%).
    equity_curve        : Bar-by-bar equity curve (pd.Series).
    n_bars              : Total bars in the dataset.
    timeframe           : Detected timeframe string (e.g. "H1").
    spec_source         : Source of the InstrumentSpec ("polled" / "default").
    """

    symbol: str
    lots: float
    initial_capital: float

    # OU physics parameters
    ou_speed: float
    half_life_bars: float
    entry_quantile: float
    n_trades: int

    # P&L
    gross_pnl_usd: float
    friction: "FrictionBreakdown"
    net_pnl_usd: float

    # Returns
    gross_return_pct: float
    net_return_pct: float
    cagr_pct: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    volatility_annualised: float

    # Equity curve (for plotting / downstream metrics)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # Metadata
    spec_source: str = "default"
    n_bars: int = 0
    timeframe: str = ""

    @property
    def direction(self) -> str:
        return "MEAN-REV"

    def summary(self) -> str:
        return (
            f"{self.symbol} NaïveOU  "
            f"θ={self.ou_speed:.4f}  hl={self.half_life_bars:.1f}bars  "
            f"trades={self.n_trades}  "
            f"gross={self.gross_return_pct:+.2f}%  "
            f"net={self.net_return_pct:+.2f}%  "
            f"CAGR={self.cagr_pct:+.2f}%  "
            f"Sharpe={self.sharpe_ratio:.2f}  "
            f"MaxDD={self.max_drawdown_pct:.1f}%  "
            f"friction=${self.friction.total_usd:,.0f}  "
            f"[{self.spec_source}]"
        )

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "baseline_type": "NaiveOU",
            "lots": self.lots,
            "initial_capital": self.initial_capital,
            "ou_speed": round(self.ou_speed, 6),
            "half_life_bars": round(self.half_life_bars, 2),
            "entry_quantile": round(self.entry_quantile, 4),
            "n_trades": self.n_trades,
            "gross_pnl_usd": round(self.gross_pnl_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "gross_return_pct": round(self.gross_return_pct, 4),
            "net_return_pct": round(self.net_return_pct, 4),
            "cagr_pct": round(self.cagr_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "volatility_annualised": round(self.volatility_annualised, 4),
            "n_bars": self.n_bars,
            "timeframe": self.timeframe,
            "spec_source": self.spec_source,
            **{
                k: round(v, 4)
                for k, v in self.friction.to_dict().items()
                if k not in ("normal_rollovers", "triple_rollovers", "holding_days")
            },
            "normal_rollovers": self.friction.normal_rollovers,
            "triple_rollovers": self.friction.triple_rollovers,
        }


def _estimate_ou_speed(close: np.ndarray, window: int = 200) -> float:
    """
    Estimate scalar OU mean-reversion speed θ from a price series.

    Uses OLS: ΔP_t = α + β·P_{t-1} + ε_t,  θ ≈ −β

    A pure-numpy duplicate of RegimeStructureMeasures.ou_mean_reversion_speed
    kept here so friction_cost.py remains import-free from kinetra internals.

    Returns:
        θ in [0, 1] — larger means faster reversion.
        Returns 0.0 for insufficient or non-reverting data.
    """
    log_p = np.log(np.clip(close, 1e-10, None))
    dp = np.diff(log_p)
    x = log_p[:-1]

    n = min(len(dp), window)
    if n < 20:
        return 0.0

    y = dp[-n:]
    x = x[-n:]

    std_x = float(np.std(x))
    if std_x < 1e-12:
        return 0.0

    # OLS slope via covariance
    cov_matrix = np.cov(y, x)
    beta = float(cov_matrix[0, 1]) / (float(np.var(x)) + 1e-20)
    # θ = −β; clamp to [0, 1] — negative θ means explosive, not mean-reverting
    return float(np.clip(-beta, 0.0, 1.0))


class NaiveOUCalculator:
    """
    Compute the naïve OU (Ornstein-Uhlenbeck) reversion baseline for
    mean-reverting instruments.

    This is the physically correct null hypothesis when DFA α < 0.48:
    the market has a measurable restoring force (spring constant θ).
    The naïve strategy simply trades in the direction of that restoring
    force using the instrument's own adaptive thresholds — no learning,
    no feature engineering.

    An RL agent operating on a mean-reverting instrument must beat this
    baseline to demonstrate a learnable edge beyond the raw spring law.

    Physics
    -------
    OU process:  dP = θ(μ − P)dt + σdW

        θ  : reversion speed (estimated via OLS regression of ΔP on P_{t-1})
        μ  : rolling mean (EWM with half-life = ln(2)/θ — no fixed period)
        σ  : rolling std of deviation (not used directly — thresholds come
             from rolling percentiles of |deviation| instead)

    Entry/exit rules (no magic numbers)
    ------------------------------------
    - Entry threshold: rolling ``entry_quantile``-th percentile of |P − μ|
      over a window of ``round(half_life × window_multiplier)`` bars.
      Adapts automatically to the instrument's volatility regime.
    - Go **long**  when deviation < −threshold  (price below mean by threshold)
    - Go **short** when deviation > +threshold  (price above mean by threshold)
    - Exit to **flat** when |deviation| < exit_fraction × threshold
      (price has reverted sufficiently toward the mean)
    - No re-entry on the same bar as an exit.

    Friction
    --------
    Every position change charges entry and/or exit cost via FrictionCalculator.
    The total across all trades is accumulated into a FrictionBreakdown and
    deducted from the equity curve bar by bar.

    Usage
    -----
        calc = NaiveOUCalculator("GBPUSD")
        result = calc.calculate(df)
        print(result.summary())

    Parameters
    ----------
    symbol : str
        Instrument name — used to load the InstrumentSpec.
    """

    def __init__(self, symbol: str) -> None:
        self.spec = load_spec(symbol)
        self.calc = FrictionCalculator(self.spec)

    def calculate(
        self,
        df: pd.DataFrame,
        ou_speed_series: Optional[np.ndarray] = None,
        lots: float = 1.0,
        initial_capital: float = 100_000.0,
        entry_quantile: float = 0.65,
        exit_fraction: float = 0.15,
        window_multiplier: float = 2.0,
        bars_per_year: Optional[float] = None,
    ) -> NaiveOUResult:
        """
        Compute the naïve OU reversion baseline for a DataFrame of OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.  Must have a ``close`` column.
        ou_speed_series : np.ndarray, optional
            Pre-computed rolling OU speed array from MeasurementEngine (the
            ``ou_speed`` column of the physics_state DataFrame).  If None,
            the scalar θ is estimated internally from the full price series.
            Passing the pre-computed series avoids redundant computation when
            the features are already available.
        lots : float
            Position size in lots.
        initial_capital : float
            Starting account equity (USD).
        entry_quantile : float
            Rolling quantile of |deviation| used as the entry threshold.
            0.65 means: enter only when deviation is in the top 35% of its
            recent distribution — no fixed σ multiple.
        exit_fraction : float
            Exit when |deviation| < exit_fraction × entry_threshold.
            0.15 means: exit when price has reverted to within 15% of the
            entry threshold from the mean.
        window_multiplier : float
            Threshold-window = round(half_life × window_multiplier).
            2.0 gives roughly two full mean-reversion cycles of history.
        bars_per_year : float, optional
            Override for CAGR/Sharpe annualisation.  Auto-detected if None.

        Returns
        -------
        NaiveOUResult
        """
        if df is None or len(df) < 20:
            raise ValueError("DataFrame must have at least 20 bars for NaiveOU baseline")

        close = df["close"].values.astype(float)
        index = df.index
        n = len(close)

        # ------------------------------------------------------------------
        # 1. OU speed θ and half-life
        # ------------------------------------------------------------------
        if ou_speed_series is not None and len(ou_speed_series) == n:
            # Use the tail of the pre-computed rolling series (skip warm-up zeros)
            valid = ou_speed_series[ou_speed_series > 1e-6]
            ou_speed = float(np.median(valid)) if len(valid) > 0 else _estimate_ou_speed(close)
        else:
            ou_speed = _estimate_ou_speed(close, window=min(200, n // 4))

        # Guard against zero/near-zero θ (random walk) — caller should not
        # have dispatched here, but be safe.
        ou_speed = max(ou_speed, 1e-6)
        half_life = np.log(2) / ou_speed  # bars

        # Adaptive window: at least 10 bars, at most 20% of the series
        window = int(np.clip(round(half_life * window_multiplier), 10, max(10, n // 5)))

        # ------------------------------------------------------------------
        # 2. Rolling mean (EWM parameterised by OU half-life — no fixed period)
        # ------------------------------------------------------------------
        close_s = pd.Series(close, index=index)
        rolling_mean = close_s.ewm(halflife=half_life, adjust=False).mean().values
        deviation = close - rolling_mean

        # ------------------------------------------------------------------
        # 3. Entry threshold: rolling entry_quantile-th percentile of |dev|
        #    — derives entirely from the instrument's own distribution,
        #      never a fixed σ multiple.
        # ------------------------------------------------------------------
        abs_dev = pd.Series(np.abs(deviation), index=index)
        thresh_series = abs_dev.rolling(window, min_periods=max(5, window // 4)).quantile(
            entry_quantile
        )
        # Fill warm-up NaNs with expanding quantile so we never have a zero threshold
        expanding_q = abs_dev.expanding(min_periods=5).quantile(entry_quantile)
        entry_thresh = thresh_series.fillna(expanding_q).values
        # Safety: ensure positive threshold to avoid division by zero downstream
        entry_thresh = np.where(
            entry_thresh < 1e-12, np.abs(rolling_mean) * 1e-4 + 1e-12, entry_thresh
        )
        exit_thresh = entry_thresh * exit_fraction

        # ------------------------------------------------------------------
        # 4. Desired position signal (vectorised)
        #    +1 = long (price below mean — expect reversion upward)
        #    −1 = short (price above mean — expect reversion downward)
        #     0 = flat
        # ------------------------------------------------------------------
        desired = np.where(
            deviation < -entry_thresh, 1.0, np.where(deviation > entry_thresh, -1.0, 0.0)
        )

        # Burn-in: no trading until the warm-up window is filled
        burn_in = max(window, 10)
        desired[:burn_in] = 0.0

        # ------------------------------------------------------------------
        # 5. Actual position with exit rule
        #    Hold the position until price reverts past exit_thresh toward mean.
        #    This prevents thrashing if desired briefly returns to 0 mid-trade.
        # ------------------------------------------------------------------
        position = np.zeros(n)
        pos = 0.0
        for i in range(1, n):
            d = desired[i]
            if d != 0.0:
                # New entry signal — take it (may be same direction or flip)
                pos = d
            elif pos != 0.0:
                # We are in a position — exit only when price reverts past exit_thresh
                if np.abs(deviation[i]) < exit_thresh[i]:
                    pos = 0.0
                # else: stay in position (price has not fully reverted yet)
            position[i] = pos

        # ------------------------------------------------------------------
        # 6. Vectorised P&L and friction accounting
        # ------------------------------------------------------------------
        # prev_pos[i] = position held ENTERING bar i
        prev_pos = np.empty(n)
        prev_pos[0] = 0.0
        prev_pos[1:] = position[:-1]

        # Bar P&L = prev_pos × Δclose × contract_size × lots
        delta_close = np.empty(n)
        delta_close[0] = 0.0
        delta_close[1:] = np.diff(close)
        # Multiply by quote_usd_rate to convert from quote currency → USD.
        bar_pnl = prev_pos * delta_close * self.spec.contract_size * self.spec.quote_usd_rate * lots
        cumulative_pnl = np.cumsum(bar_pnl)

        # Friction events: bars where position changes
        pos_diff = position - prev_pos
        changed = pos_diff != 0.0

        # Entry cost (spread + commission + slippage) charged when a new position opens
        # Exit cost (commission + slippage, no spread) charged when a position closes
        # A flip (e.g. long→short in one bar) pays BOTH exit cost AND entry cost.
        entry_c = self.calc.entry_cost_usd(float(close[0]), lots)  # price-invariant
        exit_c = self.calc.exit_cost_usd(float(close[0]), lots)

        # Masks: entering a position (including flips)
        entering = changed & (position != 0.0)
        # Masks: exiting a position (including flips)
        exiting = changed & (prev_pos != 0.0)

        friction_per_bar = np.zeros(n)
        friction_per_bar[entering] += entry_c
        friction_per_bar[exiting] += exit_c
        cumulative_friction = np.cumsum(friction_per_bar)

        # ------------------------------------------------------------------
        # 7. Equity curve
        # ------------------------------------------------------------------
        equity_arr = initial_capital + cumulative_pnl - cumulative_friction
        equity_curve = pd.Series(equity_arr, index=index)

        # ------------------------------------------------------------------
        # 8. Aggregate friction breakdown for reporting
        # ------------------------------------------------------------------
        n_entries = int(np.sum(entering))
        n_exits = int(np.sum(exiting))
        n_total_sides = n_entries + n_exits
        total_friction_usd = float(cumulative_friction[-1])

        friction = FrictionBreakdown(
            # Spread is charged once per entry (full round-trip spread at entry,
            # none at exit — same convention as FrictionCalculator.entry_cost_usd).
            # Uses bid/ask directly when available (bypasses tick_value inaccuracies).
            spread_usd=float(n_entries * self.spec.spread_usd_per_lot * lots),
            commission_usd=float(n_total_sides * self.spec.commission_per_lot * lots),
            slippage_usd=float(
                n_total_sides * self.spec.slippage_points * self.spec.tick_value_usd * lots
            ),
            swap_usd=0.0,  # OU trades are typically short-hold; swap ignored
            normal_rollovers=0,
            triple_rollovers=0,
            holding_days=0,  # N/A for multi-trade strategy
        )
        # Sanity: aggregated FrictionBreakdown must match the per-bar accumulation.
        # Both should equal: n_entries * (spread + comm + slip) + n_exits * (comm + slip)
        assert abs(friction.total_usd - total_friction_usd) < 1e-2, (
            f"Friction mismatch: aggregate={friction.total_usd:.4f} "
            f"vs per-bar={total_friction_usd:.4f}"
        )

        # ------------------------------------------------------------------
        # 9. Final P&L, returns, risk metrics
        # ------------------------------------------------------------------
        gross_pnl = float(cumulative_pnl[-1])
        net_pnl = gross_pnl - total_friction_usd

        gross_return_pct = (gross_pnl / initial_capital) * 100.0
        net_return_pct = (net_pnl / initial_capital) * 100.0

        entry_dt = index[0].to_pydatetime() if hasattr(index[0], "to_pydatetime") else index[0]
        exit_dt = index[-1].to_pydatetime() if hasattr(index[-1], "to_pydatetime") else index[-1]

        years = _years_held(entry_dt, exit_dt, n, bars_per_year)
        if years > 0 and initial_capital > 0:
            end_equity = float(equity_curve.iloc[-1])
            cagr_pct = ((max(end_equity, 1e-6) / initial_capital) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr_pct = 0.0

        sharpe, max_dd, vol = _equity_curve_metrics(equity_curve, bars_per_year)

        timeframe = _detect_timeframe(index)

        return NaiveOUResult(
            symbol=self.spec.symbol,
            lots=lots,
            initial_capital=initial_capital,
            ou_speed=round(ou_speed, 6),
            half_life_bars=round(half_life, 2),
            entry_quantile=entry_quantile,
            n_trades=n_entries,
            gross_pnl_usd=round(gross_pnl, 4),
            friction=friction,
            net_pnl_usd=round(net_pnl, 4),
            gross_return_pct=round(gross_return_pct, 4),
            net_return_pct=round(net_return_pct, 4),
            cagr_pct=round(cagr_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            volatility_annualised=round(vol, 4),
            equity_curve=equity_curve,
            spec_source=self.spec.source,
            n_bars=n,
            timeframe=timeframe,
        )


def _years_held(
    entry_dt: datetime,
    exit_dt: datetime,
    n_bars: int,
    bars_per_year: Optional[float],
) -> float:
    """Estimate holding period in years."""
    if bars_per_year is not None and bars_per_year > 0:
        return n_bars / bars_per_year

    try:
        delta_days = (exit_dt.date() - entry_dt.date()).days
        return delta_days / 365.25
    except Exception:
        return n_bars / (252 * 24)  # fallback: assume H1


def _detect_timeframe(index: pd.DatetimeIndex) -> str:
    """Infer timeframe string from bar spacing."""
    if len(index) < 2:
        return "unknown"
    try:
        median_mins = int(pd.Series(index).diff().median().total_seconds() / 60)
        mapping = {15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1", 10080: "W1"}
        return mapping.get(median_mins, f"{median_mins}m")
    except Exception:
        return "unknown"


def _equity_curve_metrics(
    equity: pd.Series,
    bars_per_year: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute Sharpe ratio, max drawdown %, and annualised volatility from
    an equity curve Series.

    Returns (sharpe, max_drawdown_pct, vol_annualised).
    """
    if len(equity) < 2:
        return 0.0, 0.0, 0.0

    returns = equity.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        return 0.0, 0.0, 0.0

    # Annualisation factor
    if bars_per_year is None:
        try:
            # Infer from index spacing
            idx = equity.index
            median_mins = pd.Series(idx).diff().median().total_seconds() / 60
            bars_per_year = (365.25 * 24 * 60) / max(median_mins, 1)
        except Exception:
            bars_per_year = 252 * 24  # H1 fallback

    ann_factor = float(bars_per_year) ** 0.5

    # Sharpe (no risk-free rate — excess return vs zero threshold)
    sharpe = float(returns.mean() / returns.std() * ann_factor)
    sharpe = np.clip(sharpe, -10.0, 10.0)  # Clamp against numerical outliers

    # Max drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd_pct = float(drawdown.min() * 100.0)

    # Annualised volatility
    vol = float(returns.std() * ann_factor * 100.0)

    return sharpe, max_dd_pct, vol


# ---------------------------------------------------------------------------
# Convenience API  (the two functions most callers will use)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regime-aware baseline selection
# ---------------------------------------------------------------------------

#: DFA α above this → trending → use B&H harder baseline
_DFA_TRENDING_THRESHOLD: float = 0.52
#: DFA α below this → mean-reverting → use NaïveOU baseline
_DFA_REVERTING_THRESHOLD: float = 0.48


def load_dfa_alpha_from_discovery(
    instrument_key: str,
    results_dir: Optional[Path] = None,
) -> Optional[float]:
    """
    Load the DFA α for a specific instrument from the most-recent
    scientific-discovery JSON written by ``run_scientific_discovery.py``.

    The discovery JSON structure is::

        {
          "per_instrument": [
            {"key": "GBPUSD_H1", "dfa": {"alpha": 0.512, ...}},
            ...
          ]
        }

    Parameters
    ----------
    instrument_key : str
        Instrument key as it appears in the discovery output, e.g.
        ``"GBPUSD_H1"`` or ``"BTCUSD_H4"``.
    results_dir : Path, optional
        Directory to search for ``scientific_discovery_*.json`` files.
        Defaults to ``<project_root>/results/``.

    Returns
    -------
    float or None
        DFA α value, or ``None`` if no discovery file exists or the
        instrument is not found in the results.
    """
    if results_dir is None:
        results_dir = _PROJECT_ROOT / "results"

    candidates = sorted(
        Path(results_dir).glob("scientific_discovery_*.json"),
        reverse=True,  # most-recent first (lexicographic timestamp)
    )
    if not candidates:
        return None

    try:
        with open(candidates[0]) as fh:
            data = json.load(fh)
        for inst_r in data.get("per_instrument", []):
            if inst_r.get("key") == instrument_key:
                dfa = inst_r.get("dfa", {})
                if isinstance(dfa, dict) and "alpha" in dfa:
                    return float(dfa["alpha"])
    except Exception as exc:
        logger.warning(
            "load_dfa_alpha_from_discovery: could not read %s — %s",
            candidates[0],
            exc,
        )

    return None


def regime_aware_baseline(
    symbol: str,
    df: pd.DataFrame,
    dfa_alpha: float,
    ou_speed_series: Optional[np.ndarray] = None,
    lots: float = 1.0,
    initial_capital: float = 100_000.0,
    trending_threshold: float = _DFA_TRENDING_THRESHOLD,
    reverting_threshold: float = _DFA_REVERTING_THRESHOLD,
    entry_quantile: float = 0.65,
) -> "BuyHoldResult | NaiveOUResult":
    """
    Select and compute the physically appropriate passive baseline for an
    instrument given its DFA scaling exponent.

    Regime dispatch table
    ---------------------
    ============================  =========================================
    DFA α                         Baseline
    ============================  =========================================
    > ``trending_threshold``      B&H harder direction (long or short,
                                  whichever is more profitable) with full
                                  friction.  Drift is real — passive
                                  participation captures it.
    < ``reverting_threshold``     Naïve OU reversion with full friction.
                                  The reversion IS the signal; a naïve
                                  spring-law trade sets the correct null.
    Otherwise (random walk)       B&H harder direction with friction.
                                  Near-zero drift + friction → natural
                                  hard-to-beat null hypothesis.
    ============================  =========================================

    Parameters
    ----------
    symbol : str
        Instrument name for loading the InstrumentSpec.
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex and ``close`` column.
    dfa_alpha : float
        DFA scaling exponent α from ``run_scientific_discovery.py``.
    ou_speed_series : np.ndarray, optional
        Pre-computed rolling OU speed (from MeasurementEngine ``ou_speed``
        column).  Only used when dispatching to NaïveOU.  If None the speed
        is estimated internally.
    lots : float
        Position size in lots.
    initial_capital : float
        Starting account equity (USD).
    trending_threshold : float
        α above this → trending regime → use B&H.
    reverting_threshold : float
        α below this → mean-reverting regime → use NaïveOU.
    entry_quantile : float
        Rolling quantile for the NaïveOU entry threshold (ignored for B&H).

    Returns
    -------
    BuyHoldResult or NaiveOUResult
    """
    if dfa_alpha < reverting_threshold:
        # Mean-reverting: NaïveOU is the correct null hypothesis
        ou_calc = NaiveOUCalculator(symbol)
        return ou_calc.calculate(
            df,
            ou_speed_series=ou_speed_series,
            lots=lots,
            initial_capital=initial_capital,
            entry_quantile=entry_quantile,
        )

    # Trending (α > 0.52) OR random-walk (0.48 ≤ α ≤ 0.52):
    # B&H harder direction is the correct null in both cases.
    return get_buy_hold_baseline(
        symbol, df, use_harder=True, lots=lots, initial_capital=initial_capital
    )


def get_calculator(symbol: str) -> FrictionCalculator:
    """Load spec for symbol and return a ready-to-use FrictionCalculator.

    Uses ``load_spec()`` which reads ``contract_spec.json`` (swap, contract
    size, etc.) but does **not** measure spread from CSV.  Spread will fall
    back through typical_spread_points → spread_points → bid/ask snapshot.

    For production use, prefer :func:`get_calculator_with_data` which also
    measures the spread from the actual H4 CSV data.

    Example:
        from kinetra.friction_cost import get_calculator
        calc = get_calculator("GBPUSD")
        pct = calc.friction_pct(price=1.268, lots=1.0, holding_bars=4)
    """
    return FrictionCalculator(load_spec(symbol))


def get_calculator_with_data(
    symbol: str,
    data_dir: Optional[Path] = None,
    timeframe: str = "H4",
    percentile: float = 75.0,
    commission_rt_usd: float = ECN_COMMISSION_RT_USD,
    latency_ms_per_side: float = DEFAULT_LATENCY_MS_PER_SIDE,
    force_reload: bool = False,
) -> FrictionCalculator:
    """Build a FrictionCalculator with all friction components populated.

    This is the **canonical entry point** for production and backtesting.

    Combines:
      - Swap rates and contract size from ``contract_spec.json``
      - Spread measured from the actual H4 CSV (most reliable source)
      - ECN commission locked at $7 RT ($3.50/side)
      - Latency slippage at 500 ms/side (overrideable)

    Example:
        from kinetra.friction_cost import get_calculator_with_data

        calc = get_calculator_with_data("XAUUSD")

        # Per-bar RL reward shaping with ATR-based latency slippage
        pct = calc.friction_pct(
            price=2650.0, lots=1.0, is_long=True,
            holding_bars=10, atr=12.5,
        )

        # Full breakdown for tier-scoring
        bd = calc.all_in_breakdown(
            price=2650.0, lots=1.0, is_long=True,
            holding_bars=10, atr=12.5,
        )
        print(bd.summary())

    Args:
        symbol:              Instrument name.
        data_dir:            Override master_standardized root.
        timeframe:           CSV timeframe for spread measurement.
        percentile:          Spread percentile (75 = Kinetra canonical).
        commission_rt_usd:   Round-trip commission in USD (default $7).
        latency_ms_per_side: One-way latency in ms (default 500).
        force_reload:        Bypass spec cache.

    Returns:
        :class:`FrictionCalculator` ready for all cost calculations.
    """
    spec = build_spec_with_csv_spread(
        symbol=symbol,
        data_dir=data_dir,
        timeframe=timeframe,
        percentile=percentile,
        commission_rt_usd=commission_rt_usd,
        latency_ms_per_side=latency_ms_per_side,
        force_reload=force_reload,
    )
    return FrictionCalculator(spec)


def get_buy_hold_baseline(
    symbol: str,
    df: pd.DataFrame,
    is_long: bool = True,
    lots: float = 1.0,
    initial_capital: float = 100_000.0,
    use_harder: bool = False,
) -> BuyHoldResult:
    """
    One-liner: compute the B&H baseline for a symbol and DataFrame.

    Args:
        symbol:         Instrument name.
        df:             OHLCV DataFrame with DatetimeIndex.
        is_long:        True = long B&H baseline, False = short.
        lots:           Position size in lots.
        initial_capital: Starting account equity.
        use_harder:     If True, return whichever direction (long/short) is
                        more profitable — sets the toughest comparison bar.

    Example:
        from kinetra.friction_cost import get_buy_hold_baseline
        bh = get_buy_hold_baseline("GBPUSD", df, use_harder=True)
        print(bh.summary())
    """
    bh_calc = BuyHoldCalculator(symbol)
    if use_harder:
        return bh_calc.harder_baseline(df, lots, initial_capital)
    return bh_calc.calculate(df, is_long=is_long, lots=lots, initial_capital=initial_capital)


def env_friction_pct(
    symbol: str,
    price: float,
    is_long: bool = True,
    holding_bars: int = 0,
    bars_per_day: float = 6.0,
    atr: Optional[float] = None,
    timeframe: str = "H4",
) -> float:
    """Convenience helper for RL environments.

    Returns total round-trip friction as a **fraction** (0 to 1, not percent)
    — ready to subtract directly from a gross P&L fraction.

    Combines spread + commission + latency slippage + swap (when
    ``holding_bars > 0``) via the canonical :class:`FrictionCalculator`.
    Gracefully returns ``0.0`` if no spec is available for *symbol* so
    environments can always call this without a try/except guard.

    Args:
        symbol:       Instrument name (e.g. ``"EURUSD"``, ``"XAUUSD"``).
        price:        Current mid price in the instrument's quote currency.
        is_long:      ``True`` for long positions, ``False`` for short.
        holding_bars: Expected (or actual) number of bars the position is
                      held.  Used to prorate the swap component.
        bars_per_day: Bars per 24-hour day for the active timeframe
                      (6 for H4, 24 for H1, 1 for D1, etc.).
        atr:          Current bar ATR in price units.  Passed to the
                      latency-slippage model; ``None`` uses the fixed
                      ``slippage_points`` fallback.
        timeframe:    Bar timeframe string for latency scaling.

    Returns:
        Total friction as a fraction of notional (e.g. ``0.00035`` = 3.5 bp).
        Returns ``0.0`` on any error (missing spec, import failure, etc.).

    Example::

        # Inside an RL environment step():
        gross_pnl_frac = position * bar_return
        if position_changed:
            cost_frac = env_friction_pct(
                symbol, price, is_long=position > 0,
                holding_bars=bars_held, atr=current_atr,
            )
            net_pnl_frac = gross_pnl_frac - cost_frac
    """
    try:
        calc = get_calculator(symbol)
        return (
            calc.friction_pct(
                price=price,
                lots=1.0,
                is_long=is_long,
                holding_bars=holding_bars,
                bars_per_day=bars_per_day,
                atr=atr,
                timeframe=timeframe,
            )
            / 100.0
        )
    except Exception:
        return 0.0


def get_naive_ou_baseline(
    symbol: str,
    df: pd.DataFrame,
    ou_speed_series: Optional[np.ndarray] = None,
    lots: float = 1.0,
    initial_capital: float = 100_000.0,
    entry_quantile: float = 0.65,
) -> NaiveOUResult:
    """
    One-liner: compute the naïve OU reversion baseline for a symbol and DataFrame.

    Use this for mean-reverting instruments (DFA α < 0.48) as the
    correct null-hypothesis baseline — harder to beat than B&H for
    assets with a measurable restoring force.

    Args:
        symbol:           Instrument name.
        df:               OHLCV DataFrame with DatetimeIndex.
        ou_speed_series:  Pre-computed rolling OU speed array (optional).
                          Pass ``inst_data.physics_state["ou_speed"].values``
                          to avoid redundant computation.
        lots:             Position size in lots.
        initial_capital:  Starting account equity (USD).
        entry_quantile:   Rolling quantile for the entry threshold.
                          0.65 = enter when deviation is in the top 35% of its
                          recent distribution (no fixed σ multiple).

    Example:
        from kinetra.friction_cost import get_naive_ou_baseline
        ou = get_naive_ou_baseline("EURUSD", df)
        print(ou.summary())
    """
    ou_calc = NaiveOUCalculator(symbol)
    return ou_calc.calculate(
        df,
        ou_speed_series=ou_speed_series,
        lots=lots,
        initial_capital=initial_capital,
        entry_quantile=entry_quantile,
    )


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  friction_cost.py — self-test")
    print("=" * 70)

    # List available specs
    available = list_available_specs()
    print(f"\nAvailable polled specs: {available or ['(none — run poll_symbol_specs.py first)']}")

    # Test with first available or fallback to EURUSD
    test_symbol = available[0] if available else "EURUSD"
    print(f"\nTesting with: {test_symbol}")

    spec = load_spec(test_symbol)
    print("\nInstrumentSpec:")
    print(f"  Source          : {spec.source}")
    print(f"  Contract size   : {spec.contract_size:,.0f}")
    print(f"  Tick value      : ${spec.tick_value:.4f}")
    print(
        f"  Spread          : {spec.spread_points} pts / "
        f"bid={spec.bid} ask={spec.ask} → "
        f"${spec.spread_usd_per_lot:.4f}/lot  ({spec.spread_pct_of_mid:.4f}% of mid)"
    )
    print(f"  Commission      : ${spec.commission_per_lot:.2f}/lot/side")
    print(
        f"  Swap long       : {spec.swap_long_points} pts/day = ${spec.swap_long_usd_per_day:+.4f}/lot/day"
    )
    print(
        f"  Swap short      : {spec.swap_short_points} pts/day = ${spec.swap_short_usd_per_day:+.4f}/lot/day"
    )
    print(f"  Triple swap day : {spec.triple_swap_day} (1=Mon…7=Sun)")
    print(f"  Breakeven       : {FrictionCalculator(spec).breakeven_points():.1f} points")

    # Round-trip test
    calc = get_calculator(test_symbol)
    t_in = datetime(2024, 3, 4, 9, 0)
    t_out = datetime(2024, 3, 15, 17, 0)  # ~2 weeks including a Wednesday
    price_in = spec.contract_size  # dummy: use contract size as price for scale
    price_out = price_in * 1.002  # 0.2% move

    result = calc.round_trip(
        is_long=True,
        lots=0.1,
        entry_price=1.2680,
        exit_price=1.2720,
        entry_dt=t_in,
        exit_dt=t_out,
    )
    print(f"\nRound-trip test (1.2680 → 1.2720, 0.1 lot, {(t_out - t_in).days} days):")
    print(f"  {result.summary()}")

    # Rollover counting
    normal, triple = _count_rollovers(t_in, t_out, spec.triple_swap_day)
    print(
        f"\nRollover count: {normal} normal + {triple} triple = "
        f"{normal + triple * 3} effective swap days"
    )

    # Friction pct
    pct = calc.friction_pct(price=1.268, lots=0.1, is_long=True, holding_bars=24)
    print(f"\nFriction % (1-day hold, 0.1 lot): {pct:.4f}%")

    # B&H test using synthetic data
    print("\n--- B&H Baseline test (synthetic 30-day H1 data) ---")
    n = 30 * 24  # 30 days of H1
    t_idx = pd.date_range("2024-01-02", periods=n, freq="h")
    np.random.seed(42)
    prices = 1.2600 * np.exp(np.cumsum(np.random.normal(0, 0.0002, n)))
    df_test = pd.DataFrame(
        {
            "open": prices * 0.9999,
            "high": prices * 1.0002,
            "low": prices * 0.9998,
            "close": prices,
            "volume": np.random.randint(100, 1000, n).astype(float),
        },
        index=t_idx,
    )

    bh_long = get_buy_hold_baseline(test_symbol, df_test, is_long=True)
    bh_short = get_buy_hold_baseline(test_symbol, df_test, is_long=False)

    print(f"  Long  B&H: {bh_long.summary()}")
    print(f"  Short B&H: {bh_short.summary()}")
    print("\n  Friction breakdown (long):")
    print(f"    {bh_long.friction.summary()}")

    print("\n✅ Self-test complete")
    sys.exit(0)
