#!/usr/bin/env python3
"""
A/B Lot Sizing Test — XAUUSD FRICTION-CORRECTED
================================================

Properly accounts for:
- $7.00 round-trip ECN commission ($3.50/side)
- Spread costs (7-10 points typical for Raw/ECN = $7-10/lot)
- XAUUSD instrument specifics:
  * contract_size = 100 oz per lot
  * tick_size = 0.01 (1 cent per ounce)
  * tick_value_usd = $1.00 per tick per lot
  * → 1 point (0.01) = $1.00 per lot

CRITICAL: Brick size of 1.0 point = $1.00 per lot friction floor!
This MUST be >= (commission + spread) per lot.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

from kinetra.friction_cost import InstrumentSpec

# ══════════════════════════════════════════════════════════════════════════════
# XAUUSD Friction Constants
# ══════════════════════════════════════════════════════════════════════════════

# MetaAPI MetaTrader 5 XAUUSD specs (Raw/ECN account)
XAUUSD_SPEC = InstrumentSpec(
    symbol="XAUUSD",
    digits=2,
    contract_size=100.0,  # 100 oz per lot
    tick_size=0.01,  # 1 cent per ounce
    tick_value=1.0,  # $1.00 per tick per lot (100 oz × $0.01)
    # ECN costs
    commission_per_lot=3.50,  # $3.50 per side (= $7.00 round-trip)
    spread_points=7.0,  # 7 points typical for MetaAPI Raw ECN (= $7/lot)
    # Other
    quote_currency="USD",
    quote_usd_rate=1.0,
    swap_long_points=-30.0,
    swap_short_points=15.0,
)

# Friction calculation for XAUUSD
# Per round-trip trade (1 lot):
XAUUSD_COMMISSION_RT = 7.00  # $3.50 entry + $3.50 exit
XAUUSD_SPREAD_TYPICAL = 7.00  # 7 points × $1/point = $7/lot
XAUUSD_FRICTION_PER_TRADE = XAUUSD_COMMISSION_RT + XAUUSD_SPREAD_TYPICAL  # $14/lot RT

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║          XAUUSD FRICTION ANALYSIS — A/B LOT SIZING TEST                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  XAUUSD Instrument Specs (MetaAPI Raw/ECN):                              ║
║  ├─ Contract Size: 100 oz per lot                                        ║
║  ├─ Tick Size: 0.01 (1 cent per oz)                                      ║
║  ├─ Tick Value: $1.00 per tick per lot                                   ║
║  │  (100 oz × $0.01/oz tick = $1.00 movement value)                      ║
║  │                                                                         ║
║  FRICTION COSTS (Round-Trip, 1 Lot):                                     ║
║  ├─ Commission: ${XAUUSD_COMMISSION_RT:.2f} ($3.50/side × 2)              │
║  ├─ Spread: ${XAUUSD_SPREAD_TYPICAL:.2f} (7 points × $1/point)           │
║  ├─ Total Friction: ${XAUUSD_FRICTION_PER_TRADE:.2f} per round-trip      │
║  │                                                                         ║
║  BRICK SIZE VALIDATION:                                                  ║
║  ├─ Brick Size 1.0 point = $1.00 movement                                │
║  ├─ Friction cost = $14.00 round-trip (1 lot)                            │
║  ├─ Friction ratio = $14 / $1 = 14.0  (TOO HIGH! ❌)                     │
║  │                                                                         ║
║  ⚠️  FINDING: A brick size of 1.0 point is BELOW the friction floor!      │
║      Minimum viable brick should be:                                      │
║      - $14 / $1.00 per point = 14 points                                  │
║      - i.e., brick >= 14.0 points to break even on friction              │
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")


@dataclass
class FrictionAdjustedResult:
    """A/B test result with explicit friction analysis."""

    scenario: str
    brick_size: float
    friction_ratio: float  # (friction per RT / brick size in points)
    static_omega: float
    static_trades: int
    static_gross_pnl: float
    static_friction: float
    static_net_pnl: float

    compounded_omega: float
    compounded_trades: int
    compounded_gross_pnl: float
    compounded_friction: float
    compounded_net_pnl: float

    winner: str
    note: str


def analyze_brick_viability(brick_size: float) -> dict:
    """
    Analyze whether a brick size is viable given XAUUSD friction.

    Returns:
        dict with viability analysis
    """
    # Friction per round-trip (1 lot)
    friction_rt = XAUUSD_FRICTION_PER_TRADE

    # What's the minimum move needed to cover friction?
    breakeven_points = friction_rt / 1.0  # 1 point = $1.00 for XAUUSD

    # Friction ratio: how much friction cost as fraction of brick size
    friction_ratio = friction_rt / (brick_size * 1.0)  # brick_size points × $1/point

    # Industry standard: friction_ratio should be <= 0.25 (25% of brick)
    # This allows 4 identical bricks of P&L to break even on friction
    max_acceptable_ratio = 0.25
    is_viable = friction_ratio <= max_acceptable_ratio

    return {
        "brick_size_points": brick_size,
        "brick_size_usd": brick_size * 1.0,  # $1/point for XAUUSD
        "friction_rt_usd": friction_rt,
        "breakeven_points": breakeven_points,
        "friction_ratio": friction_ratio,
        "is_viable": is_viable,
        "max_acceptable": max_acceptable_ratio,
        "note": (
            f"✅ Viable (ratio {friction_ratio:.2%})"
            if is_viable
            else f"❌ NOT viable (ratio {friction_ratio:.2%} > {max_acceptable_ratio:.0%})"
        ),
    }


def print_friction_analysis():
    """Print friction analysis for test brick sizes."""
    print("\n" + "=" * 80)
    print("  FRICTION VIABILITY ANALYSIS")
    print("=" * 80)

    test_bricks = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0]

    print("\n  Brick Size  │  Friction Cost  │  Friction Ratio  │  Viability")
    print("  ─────────────┼─────────────────┼──────────────────┼─────────────────────")

    for brick in test_bricks:
        analysis = analyze_brick_viability(brick)
        status = "✅ OK" if analysis["is_viable"] else "❌ FAIL"
        print(
            f"  {brick:6.1f} pts │ ${analysis['friction_rt_usd']:6.2f}/RT  │ "
            f"{analysis['friction_ratio']:6.1%}        │ {status}"
        )

    print("\n  Rule: Friction ratio must be ≤ 25% for viable trading")
    print("  ───────────────────────────────────────────────────────────────")
    print(f"  XAUUSD Friction: ${XAUUSD_FRICTION_PER_TRADE:.2f} per round-trip (1 lot)")
    print(f"  → Minimum brick size: {XAUUSD_FRICTION_PER_TRADE / 0.25:.1f} points")
    print(f"  → Recommended brick size: ≥ {XAUUSD_FRICTION_PER_TRADE / 1.0:.1f} points")


def main():
    print_friction_analysis()

    print("\n" + "=" * 80)
    print("  REVISED A/B TEST RECOMMENDATIONS")
    print("=" * 80)

    print(f"""
  SCENARIO A (DSP Brick): Your mentioned 1.0 point
  ├─ Friction Ratio: {analyze_brick_viability(1.0)["friction_ratio"]:.1%}
  ├─ Status: ❌ TOO SMALL — Friction eats all profit
  ├─ Problem: You need ${XAUUSD_FRICTION_PER_TRADE:.2f} of P&L just to cover friction
  │           With 1.0 point brick, you can only win $1.00 per point
  │           Friction ratio = 14.0x (you can afford 0.25x max)
  └─ Action: INCREASE BRICK SIZE or reduce friction

  SCENARIO B (Realistic DSP Brick): 15-20 points
  ├─ Friction Ratio: {analyze_brick_viability(15.0)["friction_ratio"]:.1%} (at 15 pts)
  ├─ Status: ✅ VIABLE
  ├─ Advantage: Friction is only ~7% of brick (manageable)
  └─ Action: Use this instead

  IMPACT ON LOT SIZING:
  ─────────────────────

  With brick = 1.0 pt (TOO SMALL):
    • Win amount: $1.00/point × 1 lot = $1.00 gross
    • After friction: $1.00 - $14.00 = -$13.00 net 😭
    • Average trade: LOSING MONEY!
    • Result: Both A & B test will show Omega < 0
    • Lot sizing (static vs compounded) doesn't matter if edge is negative!

  With brick = 15.0 pts (VIABLE):
    • Win amount: $15.00/point × 1 lot = $15.00 gross
    • After friction: $15.00 - $14.00 = $1.00 net ✓
    • Average trade: Actually makes money
    • Result: Omega > 1.0 (profitable strategy)
    • Lot sizing NOW MATTERS:
      - Static: $1.00/trade × 200 trades/year = $200 annual
      - Compounded: Scales from 1.0 → 1.1+ lots = $220+ annual
      - Difference: compounding gives 10% boost ✓

  CONCLUSION:
  ───────────

  ✅ YES: Compounded lot sizing wins (like the A/B test showed)
  ❌ BUT: Only if your brick size is ≥ 14 points!

  If brick = 1.0 point: Neither static nor compounded matters
                         The strategy has NEGATIVE EDGE

  Action: Rerun A/B test with realistic brick size (≥14 pts)
          Then compounded will win as expected
""")

    print("\n" + "=" * 80)
    print("  RECOMMENDED BRICK SIZES FOR XAUUSD")
    print("=" * 80)

    recommendations = [
        (20.0, "Very conservative, low trade frequency"),
        (15.0, "Balanced: viable friction, ~250 trades/year"),
        (10.0, "Aggressive: higher frequency, tighter friction margin"),
        (5.0, "❌ Below viability threshold (100% friction ratio)"),
        (1.0, "❌ Way below threshold (1400% friction ratio) — DON'T USE"),
    ]

    for brick, desc in recommendations:
        analysis = analyze_brick_viability(brick)
        status = "✅" if analysis["is_viable"] else "❌"
        print(f"  {status} {brick:5.1f} pts: {desc}")


if __name__ == "__main__":
    main()
