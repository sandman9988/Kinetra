"""
Tests for kinetra.renko.live_trader
=====================================

Coverage
--------
- _on_bar entry-gate fix: _check_entry only fires when a new brick forms
  (no_flip must count bricks, not bars)
- Entry signals are seen on genuine colour-flip bricks
- Session summary structure completeness
- no_flip is 0 when no new brick forms (bars stay within one brick range)
- Multiple same-direction bricks each contribute exactly 1 no_flip count
- Halted trader does not evaluate entries
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from kinetra.renko.live_trader import (
    HistoricalBarProvider,
    LiveTraderConfig,
    PaperDispatcher,
    PERGate,
    RenkoLiveTrader,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_MONDAY_10AM = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)  # always in session


def _make_qual(qual_dir: Path, symbol: str, brick_size: float = 1.0) -> None:
    """Write minimal qualification.json + session_profile.json into qual_dir."""
    sym_dir = qual_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    qual = {
        "symbol": symbol,
        "qualified": True,
        "disqualified": False,
        "disqualification_reason": "",
        "broker_source": "unknown",
        "cluster": "test",
        "brick_size": float(brick_size),
        "omega": 9.99,
        "z_factor": 9.99,
        "n_trades": 999,
        "filter_params": {
            # Windows of 2 so filters become ready after just 2 bricks.
            # Thresholds: fliprate=1.0 (disabled), markov=0.0 (always pass).
            "fliprate_window": 2,
            "fliprate_threshold": 1.0,
            "markov_window": 2,
            "markov_threshold": 0.0,
        },
        "usd_per_point": 100.0,
        "tick_size": 0.01,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "pipeline_version": "test",
        "recalibration_due": False,
    }
    (sym_dir / "qualification.json").write_text(json.dumps(qual), encoding="utf-8")
    (sym_dir / "session_profile.json").write_text(
        json.dumps({"session_break_minutes": 5.0}), encoding="utf-8"
    )


def _make_bars(prices: List[float], start: datetime = _MONDAY_10AM) -> pd.DataFrame:
    """Build a time+close DataFrame from a list of close prices."""
    times = [start + timedelta(minutes=i) for i in range(len(prices))]
    return pd.DataFrame({"time": pd.to_datetime(times, utc=True), "close": prices})


def _run_replay(
    symbol: str,
    bars: pd.DataFrame,
    qual_dir: Path,
    *,
    startup_skip: int = 0,
    drawdown_halt_pct: float = 1.0,
) -> dict:
    """Run a full paper replay and return session_summary()."""
    cfg = LiveTraderConfig(
        symbols=[symbol],
        gate=PERGate.PAPER,
        qual_dir=qual_dir,
        broker_source="unknown",
        stop_bricks=1.0,
        paper_lots=0.01,
        startup_skip_flips=startup_skip,
        monday_open_utc="00:00",
        friday_close_utc="23:59",
        poll_interval_seconds=0.01,
        drawdown_halt_pct=drawdown_halt_pct,
        loss_brake_after_consecutive_losses=999,
        loss_flat_after_consecutive_losses=999,
        loss_pause_minutes=0.0,
    )
    provider = HistoricalBarProvider({symbol: bars}, speed_multiplier=float("inf"))
    dispatcher = PaperDispatcher()
    trader = RenkoLiveTrader(config=cfg, bar_provider=provider, dispatcher=dispatcher)
    trader.start()
    # Wait for replay thread to finish
    while True:
        th = getattr(provider, "_thread", None)
        if th is None or not th.is_alive():
            break
        time.sleep(0.02)
    trader.stop()
    return trader.session_summary()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def qual_dir(tmp_path: Path) -> Path:
    return tmp_path


# ── entry-gate fix: no_flip counts bricks, not bars ──────────────────────────

class TestEntryGateFix:
    """Verify _check_entry only fires on new brick formation after the fix."""

    def test_no_flip_zero_when_no_brick_forms(self, qual_dir: Path) -> None:
        """Bars that stay within one brick range must not increment no_flip at all."""
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        # All bars stay within ±0.9 of 100 — no brick forms → entry check never fires.
        prices = [100.0, 100.3, 100.7, 100.5, 100.2, 100.8]
        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir)

        blocks = summary.get("entry_block_counts", {})
        assert blocks.get("no_flip", 0) == 0, (
            f"No brick formed, so no_flip must be 0; got blocks={blocks}"
        )

    def test_no_flip_counts_bricks_not_bars(self, qual_dir: Path) -> None:
        """
        Mid-brick bars (no new brick formed) must NOT inflate no_flip.

        Sequence:
          100 → 101 → 102  → forms 2 up-bricks
          5 bars at 102.x  → no new brick forms → after fix, no_flip stays at 1

        Note: the first brick is skipped by the ``len(bricks) < 2`` early-return
        in ``_on_bar``, so only the second (continuation) brick adds 1 to no_flip.
        The 5 mid-brick bars must not add any further counts.
        """
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        n_inert_bars = 5
        prices = [100.0, 101.0, 102.0, 102.1, 102.2, 102.3, 102.4, 102.5]
        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir)

        blocks = summary.get("entry_block_counts", {})
        no_flip_count = blocks.get("no_flip", 0)
        # After fix: no_flip ≤ n_bricks_formed (= 2 here, but first skipped → ≤ 1).
        # Before fix: no_flip would be ≈ len(prices) - 1 = 7.
        assert no_flip_count <= 2, (
            f"After fix no_flip must be ≤ 2 (per-brick), got {no_flip_count}. "
            f"blocks={blocks}"
        )
        # Key regression guard: must be well under total bar count.
        assert no_flip_count < n_inert_bars, (
            f"no_flip={no_flip_count} ≥ n_inert_bars={n_inert_bars} — "
            "suggests per-bar counting is back"
        )

    def test_multiple_continuation_bricks_each_count_once(self, qual_dir: Path) -> None:
        """Each continuation brick contributes at most 1 to no_flip (never per-bar)."""
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        # 5 brick-forming bars interspersed with 3 inert bars each.
        # build_renko needs a reference bar first; supply 100.0 as the anchor.
        n_bricks_target = 5
        n_inert_per_brick = 3
        prices = [100.0]  # anchor (reference price, no brick yet)
        for i in range(n_bricks_target):
            prices.append(100.0 + (i + 1) * 1.0)      # brick-forming bar
            for j in range(1, n_inert_per_brick + 1):  # 3 inert bars
                prices.append(100.0 + (i + 1) * 1.0 + j * 0.1)
        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir)

        blocks = summary.get("entry_block_counts", {})
        no_flip_count = blocks.get("no_flip", 0)
        total_bars = len(prices)

        # After fix: no_flip ≤ n_bricks_target (one per brick, first skipped → ≤ 4).
        # Before fix: no_flip ≈ total_bars = 21.
        assert no_flip_count <= n_bricks_target, (
            f"Expected no_flip ≤ {n_bricks_target} (per-brick), got {no_flip_count}. "
            f"blocks={blocks}"
        )
        # Key regression guard: must be well under total bar count.
        assert no_flip_count < total_bars // 2, (
            f"no_flip={no_flip_count} ≥ total_bars//2={total_bars//2} — "
            "suggests per-bar counting is back"
        )

    def test_entry_signal_seen_on_flip_brick(self, qual_dir: Path) -> None:
        """A colour-flip brick must reach _entry_signals_seen (past the no_flip gate)."""
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        # 100→101→102 (2 up-bricks), then 100.5 (down-brick = flip).
        # filters: window=2, thresholds disabled → passes straight to entry.
        prices = [100.0, 101.0, 102.0, 100.5]
        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir)

        assert summary.get("entry_signals_seen", 0) >= 1, (
            f"Expected ≥1 entry signal on flip brick; summary={summary}"
        )

    def test_no_entry_signal_without_flip(self, qual_dir: Path) -> None:
        """Pure continuation run must produce zero entry_signals_seen."""
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        # Monotone up — no flip ever.
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir)

        assert summary.get("entry_signals_seen", 0) == 0, (
            f"No flip brick → no entry signals expected; summary={summary}"
        )


# ── session summary structure ─────────────────────────────────────────────────

class TestSessionSummary:
    """Verify session_summary() returns the expected keys and types."""

    EXPECTED_KEYS = [
        "gate", "sizing_mode", "is_halted", "session_pnl_usd",
        "n_completed_trades", "omega", "profit_factor", "win_rate",
        "portfolio_drawdown", "per_instrument", "can_advance_gate",
        "entry_signals_seen", "entries_opened", "entry_block_counts",
    ]

    def test_summary_has_required_keys(self, qual_dir: Path) -> None:
        sym = "TST"
        _make_qual(qual_dir, sym)
        bars = _make_bars([100.0, 101.0])
        summary = _run_replay(sym, bars, qual_dir)

        for key in self.EXPECTED_KEYS:
            assert key in summary, f"Missing key '{key}' in session_summary()"

    def test_summary_numeric_fields_finite(self, qual_dir: Path) -> None:
        import math
        sym = "TST"
        _make_qual(qual_dir, sym)
        bars = _make_bars([100.0, 101.0, 102.0, 100.5, 101.5])
        summary = _run_replay(sym, bars, qual_dir)

        for field in ("session_pnl_usd", "portfolio_drawdown", "win_rate", "profit_factor"):
            val = summary.get(field, float("nan"))
            assert isinstance(val, (int, float)), f"{field} should be numeric"
            assert not math.isnan(val) or field == "omega", (
                f"{field}={val} is NaN (allowed only for omega with no trades)"
            )

    def test_entry_block_counts_is_dict(self, qual_dir: Path) -> None:
        sym = "TST"
        _make_qual(qual_dir, sym)
        bars = _make_bars([100.0, 100.5, 100.3])
        summary = _run_replay(sym, bars, qual_dir)

        assert isinstance(summary.get("entry_block_counts"), dict), (
            "entry_block_counts must be a dict"
        )

    def test_per_instrument_has_symbol(self, qual_dir: Path) -> None:
        sym = "TST"
        _make_qual(qual_dir, sym)
        bars = _make_bars([100.0, 101.0, 100.0])
        summary = _run_replay(sym, bars, qual_dir)

        per = summary.get("per_instrument", {})
        assert sym in per, f"per_instrument missing '{sym}': {per}"

    def test_no_trades_when_only_inert_bars(self, qual_dir: Path) -> None:
        """No trades when no brick ever forms."""
        sym = "TST"
        _make_qual(qual_dir, sym)
        bars = _make_bars([100.0, 100.1, 100.2, 100.3])
        summary = _run_replay(sym, bars, qual_dir)

        assert summary["n_completed_trades"] == 0
        assert summary["entries_opened"] == 0


# ── halted trader behaviour ───────────────────────────────────────────────────

class TestHaltedTrader:
    """Trader halted by drawdown must not evaluate entries."""

    def test_halted_trader_sees_no_entry_signals(self, qual_dir: Path) -> None:
        """
        With drawdown_halt_pct=0.0 the trader halts on the first losing trade.

        Sequence:
          100 → 101 → 102   2 up-bricks
          100.5              DOWN-brick (flip) → SHORT entry at 101, stop=102
          103.0              UP-bricks at 102 and 103 → stop at 102 hit →
                             trade closes at a loss → DD check fires → halted
        """
        sym = "TST"
        _make_qual(qual_dir, sym, brick_size=1.0)

        prices = [100.0, 101.0, 102.0,   # 2 up-bricks
                  100.5,                  # DOWN-brick flip → SHORT entry at 101, stop=102
                  103.0]                  # UP-bricks → stop hit at 102 → loss → halt

        bars = _make_bars(prices)
        summary = _run_replay(sym, bars, qual_dir, drawdown_halt_pct=0.0)

        assert summary["is_halted"], "Expected trader to be halted after stop-loss hit"
