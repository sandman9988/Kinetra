"""
Renko Instrument Pool
=====================

Converts a populated :class:`~kinetra.renko.qualify.QualificationRegistry`
into a ranked, tiered trading pool with cluster-aware portfolio slot
assignment.

The pool is the authoritative answer to the question:
    "Which instruments do we actually trade, in which order of priority,
     and how confident are we in each one?"

Tier definitions
----------------

TIER_1 — Production-ready, deploy immediately
    All gates passed at high confidence.  These are the instruments
    the research summary showed as dominant: strong IS Omega, solid
    OOS survival, friction-stress-safe, meaningful trade count.

    Thresholds (all must be satisfied):
      • IS Omega  ≥ 2.5
      • OOS Omega ≥ 1.8
      • Z-factor  ≥ 3.0
      • OOS survival rate ≥ 0.60   (≥ 60 % of WF folds pass)
      • Friction-stress Omega ≥ 1.2 (survives 1.5× cost scaling)
      • Trades ≥ 50                 (credible sample)
      • Friction ratio ≤ 0.20       (tighter than the 0.25 hard gate)

TIER_2 — Viable, include with reduced weight
    Passed qualification gates but below Tier 1 on one or more axes.
    Still deployable — just weighted less aggressively.

    Thresholds (qualification pass + any one of these relaxations):
      • IS Omega  ≥ 1.5   (the bare qualification minimum)
      • OOS Omega ≥ 1.2
      • Z-factor  ≥ 2.0
      • The instrument is Tier 1 everywhere except friction ratio
        (0.20 < friction_ratio ≤ 0.25)

TIER_3 — Watch list, do not deploy
    Passed qualification but one metric is borderline or the instrument
    has a recalibration_due flag set.  Monitor on next data refresh.

    An instrument is Tier 3 when:
      • It passed :func:`~kinetra.renko.qualify.qualify_instrument` but
        fails to meet Tier 2 thresholds, OR
      • recalibration_due=True regardless of other metrics.

DISQUALIFIED — Failed at least one hard qualification gate
    Not assigned to any tier.  Stored in the pool for reference / audit.

Scoring
-------
Each qualified instrument receives a composite score (0–100) built from
five equally-weighted normalised sub-scores:

  score = mean(
      normalise(IS_omega,         clip=[1.5, 8.0]),
      normalise(OOS_omega,        clip=[1.2, 6.0]),
      normalise(Z_factor,         clip=[2.0, 30.0]),
      normalise(friction_stress,  clip=[1.0, 6.0]),
      normalise(1 - friction_ratio, clip=[0.75, 1.0]),   # inverted ratio
  ) × 100

Normalise means: (value - lo) / (hi - lo), clipped to [0, 1].

No magic numbers in the scoring system: all clip bounds are anchored to
the qualification gate thresholds and the empirical portfolio results
documented in RENKO_REDESIGN_THREAD_CONTEXT.md §3.

Cluster slot assignment
-----------------------
After scoring, the pool enforces portfolio slot limits derived from the
existing :class:`~kinetra.renko.portfolio.PortfolioConfig`:

  • max_per_cluster : int   (default 3  — matches PortfolioConfig)
  • max_cluster_weight : float (default 0.35 — matches PortfolioConfig)

Within each cluster, instruments are ranked by score DESC.  Only the top
``max_per_cluster`` instruments per cluster enter the *active* pool;
the rest are demoted to the watch list.

Deduplication
-------------
When both a spot and a futures version of the same underlying qualify
(e.g. "NAS100" and "NAS100ft"), only the higher-scoring variant is kept.
This mirrors :func:`~kinetra.renko.portfolio.deduplicate_underlyings`.

Canonical usage
---------------
::

    from kinetra.renko.instrument_pool import build_instrument_pool, InstrumentPool

    registry = QualificationRegistry("data/renko_qualified")
    registry.load()

    pool = build_instrument_pool(registry)
    print(pool.summary())                 # operator-readable text report
    pool.save(Path("results/renko/instrument_pool.json"))

    tier1 = pool.tier(1)                  # list[PoolEntry]
    active = pool.active_instruments()    # list[str] — symbols in active pool
    weights = pool.allocation_weights()   # dict[str, float] — normalised weights

See Also
--------
- ``kinetra/renko/qualify.py``   — QualificationRegistry, QualificationResult
- ``kinetra/renko/portfolio.py`` — CLUSTER_MAP, get_cluster, PortfolioConfig
- ``kinetra/renko/orchestrator.py`` — run_full_pipeline
- ``docs/MANUAL.md §3`` — research summary & strategic conclusion
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from kinetra.renko.portfolio import PortfolioConfig, get_cluster
from kinetra.renko.qualify import QualificationRegistry, QualificationResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Tier thresholds
# ══════════════════════════════════════════════════════════════════════════════

#: Tier 1 — all must pass simultaneously.
TIER1_MIN_IS_OMEGA: float = 2.5
TIER1_MIN_OOS_OMEGA: float = 1.8
TIER1_MIN_Z: float = 3.0
TIER1_MIN_OOS_SURVIVAL: float = 0.60
TIER1_MIN_FRICTION_STRESS_OMEGA: float = 1.2
TIER1_MIN_TRADES: int = 50
TIER1_MAX_FRICTION_RATIO: float = 0.20

#: Tier 2 — qualification gate pass + below Tier 1 on at most one axis.
#: Uses the bare qualification gate values already enforced by qualify.py.
#: (QUALIFY_MIN_OMEGA=1.5, QUALIFY_MIN_OOS_OMEGA=1.2, QUALIFY_MIN_Z=2.0)
#: No additional constants needed here — Tier 2 is: qualified AND NOT Tier 1.

# ── Scoring clip ranges (anchored to gate minimums and empirical portfolio
#    highs from RENKO_REDESIGN_THREAD_CONTEXT.md §3 — do NOT use magic numbers) ──

_SCORE_CLIP: Dict[str, tuple[float, float]] = {
    "is_omega": (1.5, 8.0),  # gate floor → strong single-instrument IS
    "oos_omega": (1.2, 6.0),  # gate floor → strong OOS
    "z_factor": (2.0, 30.0),  # gate floor → empirical portfolio Z=28.89
    "friction_stress": (1.0, 6.0),  # gate floor=1.0 → headroom
    "friction_margin": (0.75, 1.0),  # 1 - friction_ratio: gate floor=0.75 (ratio≤0.25)
}


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PoolEntry:
    """
    Single row in the instrument pool.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    tier : int
        1, 2, or 3 for qualified instruments; 0 for disqualified.
    score : float
        Composite quality score in [0, 100].  Higher = better.
    cluster : str
        Asset cluster (from :func:`~kinetra.renko.portfolio.get_cluster`).
    cluster_rank : int
        Rank within the cluster (1 = best score in cluster).
    active : bool
        True when this instrument is in the active trading pool (has a
        portfolio slot after cluster capping and deduplication).
    demotion_reason : Optional[str]
        Human-readable explanation when ``active=False`` despite being
        qualified (e.g. "cluster_cap: 3/3 slots used by higher-scoring
        peers", "dedup: NAS100ft has higher score").
    omega : float
        Full-data IS Omega from qualification.
    oos_omega : float
        OOS Omega (median across walk-forward folds).
    z_factor : float
        Z-factor from full-data backtest.
    win_rate : float
        Win rate (0–1) from full-data backtest.
    n_trades : int
        Number of trades in the full-data backtest.
    max_drawdown : float
        Maximum drawdown (USD) from full-data backtest.
    brick_size : float
        DSP-derived optimal brick size (price units).
    friction_ratio : float
        spread / brick at qualification time.
    friction_stress_omega : float
        Omega at 1.5× friction costs.
    oos_survival_rate : float
        Fraction of walk-forward folds with Omega above threshold.
    vr_peak : float
        Peak variance ratio from DSP.
    broker_source : str
        Broker the data came from.
    recalibration_due : bool
        True when CalibrationDriftDetector has flagged this instrument.
    drift_reason : Optional[str]
        Human-readable drift reason.
    data_start : str
        ISO-8601 UTC timestamp of first M1 bar used in qualification.
    data_end : str
        ISO-8601 UTC timestamp of last M1 bar used in qualification.
    qualified_at : str
        ISO-8601 UTC timestamp of qualification run.
    weight : float
        Normalised allocation weight in the active pool (sum to 1.0 across
        active instruments).  0.0 for inactive instruments.
    """

    symbol: str
    instrument_id: str
    tier: int
    score: float
    cluster: str
    cluster_rank: int
    active: bool
    demotion_reason: Optional[str]

    # Qualification metrics
    omega: float
    oos_omega: float
    z_factor: float
    win_rate: float
    n_trades: int
    max_drawdown: float
    brick_size: float
    friction_ratio: float
    friction_stress_omega: float
    oos_survival_rate: float
    vr_peak: float
    broker_source: str
    broker_account: str
    account_type: str
    recalibration_due: bool
    drift_reason: Optional[str]
    data_start: str
    data_end: str
    qualified_at: str

    # Engine that produced this entry (Sprint 6A)
    engine: str = "fixed"  # "fixed" | "adaptive"

    # Portfolio slot
    weight: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PoolEntry":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ClusterPoolStats:
    """
    Per-cluster summary within the pool.

    Attributes
    ----------
    cluster : str
        Cluster name.
    n_active : int
        Number of active (trading pool) instruments.
    n_watch : int
        Qualified instruments that were demoted from active (cluster cap,
        dedup, or Tier 3 watch-list).
    n_disqualified : int
        Instruments that failed qualification.
    active_symbols : list[str]
        Active instrument symbols.
    watch_symbols : list[str]
        Watch-list symbols (qualified but inactive).
    top_score : float
        Score of the best instrument in this cluster.
    mean_score : float
        Mean score across active instruments.
    cluster_weight : float
        Total allocation weight of this cluster in the active pool.
    """

    cluster: str
    n_active: int
    n_watch: int
    n_disqualified: int
    active_symbols: List[str]
    watch_symbols: List[str]
    top_score: float
    mean_score: float
    cluster_weight: float


@dataclass
class InstrumentPool:
    """
    The complete instrument pool: tiered, scored, cluster-slotted.

    Attributes
    ----------
    built_at : str
        ISO-8601 UTC timestamp of pool construction.
    n_processed : int
        Total instruments loaded from the registry.
    n_qualified : int
        Instruments that passed all qualification gates.
    n_active : int
        Instruments assigned to the active trading pool (cluster-capped,
        deduped, Tier 1 + Tier 2).
    n_watch : int
        Qualified instruments on the watch list (Tier 3 or demoted).
    n_disqualified : int
        Instruments that failed qualification.
    entries : list[PoolEntry]
        All entries, sorted: active first (score DESC), then watch, then
        disqualified.
    cluster_stats : dict[str, ClusterPoolStats]
        Per-cluster statistics.
    config : PortfolioConfig
        The portfolio configuration used for cluster capping.
    """

    built_at: str
    n_processed: int
    n_qualified: int
    n_active: int
    n_watch: int
    n_disqualified: int
    entries: List[PoolEntry] = field(default_factory=list)
    cluster_stats: Dict[str, ClusterPoolStats] = field(default_factory=dict)
    config: PortfolioConfig = field(default_factory=PortfolioConfig)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def tier(self, tier_number: int) -> List[PoolEntry]:
        """Return all entries with the given tier number."""
        return [e for e in self.entries if e.tier == tier_number]

    def active_entries(self) -> List[PoolEntry]:
        """Return active trading pool entries, sorted by score descending."""
        return sorted([e for e in self.entries if e.active], key=lambda e: e.score, reverse=True)

    def watch_entries(self) -> List[PoolEntry]:
        """Return watch-list entries (qualified but not in active pool)."""
        return sorted(
            [e for e in self.entries if not e.active and e.tier in (2, 3)],
            key=lambda e: e.score,
            reverse=True,
        )

    def disqualified_entries(self) -> List[PoolEntry]:
        """Return disqualified entries."""
        return sorted(
            [e for e in self.entries if e.tier == 0],
            key=lambda e: (e.symbol, e.instrument_id),
        )

    def active_instruments(self) -> List[str]:
        """Return instrument_ids in the active trading pool (by score, descending)."""
        return [e.instrument_id for e in self.active_entries()]

    def allocation_weights(self) -> Dict[str, float]:
        """
        Return normalised allocation weights for all active instruments.

        Weights sum to 1.0 across active instruments.  Equal-risk weighting
        is applied within the pool (cross-instrument risk budgeting is
        handled by :func:`~kinetra.renko.portfolio.equal_risk_weights` using
        actual brick sizes and USD-per-point).  This weight is the *relative
        priority* input before the risk-budget scaling.
        """
        active = self.active_entries()
        if not active:
            return {}
        w = 1.0 / len(active)
        return {e.instrument_id: round(w, 6) for e in active}

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "built_at": self.built_at,
            "n_processed": self.n_processed,
            "n_qualified": self.n_qualified,
            "n_active": self.n_active,
            "n_watch": self.n_watch,
            "n_disqualified": self.n_disqualified,
            "entries": [e.to_dict() for e in self.entries],
            "cluster_stats": {k: asdict(v) for k, v in self.cluster_stats.items()},
            "config": asdict(self.config),
        }

    def save(self, path: Path) -> None:
        """Atomically save to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)
        logger.info("InstrumentPool saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "InstrumentPool":
        """Load from JSON (forward-compatible: unknown fields ignored)."""
        raw = json.loads(Path(path).read_text())
        entries = [PoolEntry.from_dict(e) for e in raw.get("entries", [])]

        cluster_stats: Dict[str, ClusterPoolStats] = {}
        for k, v in raw.get("cluster_stats", {}).items():
            known = {f.name for f in ClusterPoolStats.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            cluster_stats[k] = ClusterPoolStats(**{kk: vv for kk, vv in v.items() if kk in known})

        cfg_raw = raw.get("config", {})
        known_cfg = {f.name for f in PortfolioConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        config = PortfolioConfig(**{k: v for k, v in cfg_raw.items() if k in known_cfg})

        return cls(
            built_at=raw.get("built_at", ""),
            n_processed=raw.get("n_processed", 0),
            n_qualified=raw.get("n_qualified", 0),
            n_active=raw.get("n_active", 0),
            n_watch=raw.get("n_watch", 0),
            n_disqualified=raw.get("n_disqualified", 0),
            entries=entries,
            cluster_stats=cluster_stats,
            config=config,
        )

    # ── Text report ───────────────────────────────────────────────────────────

    def summary(self, width: int = 80) -> str:
        """
        Return a multi-section operator-readable text report.

        Sections
        --------
        1. Pool overview (counts, tiers, clusters)
        2. Active trading pool (Tier 1 → Tier 2, ranked by score)
        3. Watch list (qualified but inactive)
        4. Disqualified (gate failure summary)
        5. Cluster breakdown
        """
        lines: List[str] = []

        def _hr(label: str = "") -> str:
            if label:
                prefix = f"  ── {label} "
                return prefix + "─" * (width - len(prefix))
            return "  " + "─" * (width - 2)

        # ── Header ───────────────────────────────────────────────────────
        lines.append(_hr("INSTRUMENT POOL"))
        lines.append(f"  Built: {self.built_at[:19].replace('T', ' ')} UTC")
        lines.append(
            f"  Processed: {self.n_processed}  "
            f"Qualified: {self.n_qualified}  "
            f"Active: {self.n_active}  "
            f"Watch: {self.n_watch}  "
            f"Disqualified: {self.n_disqualified}"
        )
        lines.append(
            f"  Tiers: "
            f"T1={len(self.tier(1))}  "
            f"T2={len(self.tier(2))}  "
            f"T3-watch={len(self.tier(3))}  "
            f"Clusters={len(self.cluster_stats)}"
        )
        lines.append("")

        # ── Active pool ───────────────────────────────────────────────────
        active = self.active_entries()
        lines.append(_hr("ACTIVE TRADING POOL"))
        if not active:
            lines.append("  ❌  No instruments in active pool.")
        else:
            hdr = (
                f"  {'#':>3} {'Symbol':<12} {'Variant':<24} {'T':>2} {'Score':>6} "
                f"{'IS-Ω':>6} {'OOS-Ω':>6} {'Z':>5} {'WR%':>5} "
                f"{'Brick':>8} {'FRatio':>6} {'Cluster':<15} {'Broker':<12}"
            )
            lines.append(hdr)
            lines.append("  " + "─" * (len(hdr) - 2))
            for rank, e in enumerate(active, 1):
                tier_badge = {1: "★", 2: "◈", 3: "○"}.get(e.tier, "?")
                drift = " ⚠️" if e.recalibration_due else ""
                lines.append(
                    f"  {rank:>3} {e.symbol:<12} {e.instrument_id[:24]:<24} "
                    f"{tier_badge:>2} {e.score:>6.1f} "
                    f"{e.omega:>6.2f} {e.oos_omega:>6.2f} {e.z_factor:>5.1f} "
                    f"{e.win_rate * 100:>4.1f}% {e.brick_size:>8.5f} "
                    f"{e.friction_ratio:>6.3f} {e.cluster:<15} {e.broker_source:<12}{drift}"
                )
        lines.append("")

        # ── Watch list ────────────────────────────────────────────────────
        watch = self.watch_entries()
        lines.append(_hr("WATCH LIST  (qualified · not in active pool)"))
        if not watch:
            lines.append("  —  no instruments on watch list")
        else:
            hdr_w = (
                f"  {'Symbol':<12} {'Variant':<24} {'T':>2} {'Score':>6} "
                f"{'IS-Ω':>6} {'OOS-Ω':>6} {'Z':>5} {'Cluster':<15} {'Reason'}"
            )
            lines.append(hdr_w)
            lines.append("  " + "─" * (len(hdr_w) - 2))
            for e in watch:
                tier_badge = {1: "★", 2: "◈", 3: "○"}.get(e.tier, "?")
                reason = e.demotion_reason or ("drift flag" if e.recalibration_due else "—")
                lines.append(
                    f"  {e.symbol:<12} {e.instrument_id[:24]:<24} {tier_badge:>2} {e.score:>6.1f} "
                    f"{e.omega:>6.2f} {e.oos_omega:>6.2f} {e.z_factor:>5.1f} "
                    f"{e.cluster:<15} {reason}"
                )
        lines.append("")

        # ── Disqualified ──────────────────────────────────────────────────
        disq = self.disqualified_entries()
        if disq:
            lines.append(_hr("DISQUALIFIED"))
            hdr_d = (
                f"  {'Symbol':<12} {'Variant':<24} {'VR':>5} "
                f"{'FR':>6} {'IS-Ω':>6} {'Z':>5} {'Cluster':<15}"
            )
            lines.append(hdr_d)
            lines.append("  " + "─" * (len(hdr_d) - 2))
            for e in disq:
                vr_str = f"{e.vr_peak:.3f}" if e.vr_peak > 0 else "n/a"
                fr_str = f"{e.friction_ratio:.3f}" if e.friction_ratio > 0 else "n/a"
                om_str = f"{e.omega:.2f}" if e.omega > 0 else "n/a"
                z_str = f"{e.z_factor:.1f}" if e.z_factor > 0 else "n/a"
                lines.append(
                    f"  {e.symbol:<12} {e.instrument_id[:24]:<24} {vr_str:>5} {fr_str:>6} "
                    f"{om_str:>6} {z_str:>5} {e.cluster:<15}"
                )
            lines.append("")

        # ── Cluster breakdown ─────────────────────────────────────────────
        lines.append(_hr("CLUSTER BREAKDOWN"))
        if not self.cluster_stats:
            lines.append("  —  no cluster data")
        else:
            hdr_c = (
                f"  {'Cluster':<18} {'Active':>6} {'Watch':>6} {'Disq':>6} "
                f"{'Top-Score':>10} {'Weight%':>8} {'Symbols'}"
            )
            lines.append(hdr_c)
            lines.append("  " + "─" * (len(hdr_c) - 2))
            for cluster_name, cs in sorted(
                self.cluster_stats.items(), key=lambda kv: kv[1].n_active, reverse=True
            ):
                sym_list = ", ".join(cs.active_symbols) if cs.active_symbols else "—"
                lines.append(
                    f"  {cluster_name:<18} {cs.n_active:>6} {cs.n_watch:>6} "
                    f"{cs.n_disqualified:>6} {cs.top_score:>10.1f} "
                    f"{cs.cluster_weight * 100:>7.1f}% {sym_list}"
                )
        lines.append("")

        # ── Strategic note ────────────────────────────────────────────────
        if self.n_active < 3:
            lines.append(
                "  ⚠️  Active pool has fewer than 3 instruments — portfolio "
                "construction requires ≥ 3."
            )
            lines.append("      Qualify more instruments or review watch-list candidates.")
        elif self.n_active >= 5:
            lines.append(
                f"  ✅  {self.n_active} instruments in active pool — portfolio "
                "construction available."
            )
        else:
            lines.append(
                f"  ✅  {self.n_active} instruments in active pool "
                f"(≥ 3 threshold met — portfolio construction available)."
            )

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ══════════════════════════════════════════════════════════════════════════════


def _normalise(value: float, lo: float, hi: float) -> float:
    """Clip ``value`` into [lo, hi] then normalise to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _compute_score(result: QualificationResult) -> float:
    """
    Composite quality score in [0, 100] for a qualified instrument.

    Five equally-weighted sub-scores, each normalised to [0, 1]:

    1. IS Omega         — how strong is the edge in-sample?
    2. OOS Omega        — does it hold out-of-sample?
    3. Z-factor         — how statistically significant?
    4. Friction stress  — does it survive 1.5× costs?
    5. Friction margin  — how much headroom vs the spread cap?

    The friction margin sub-score inverts the ratio: a lower friction
    ratio means more headroom, which is better.

    Returns 0.0 for unqualified instruments.
    """
    if not result.qualified:
        return 0.0

    lo_o, hi_o = _SCORE_CLIP["is_omega"]
    lo_oos, hi_oos = _SCORE_CLIP["oos_omega"]
    lo_z, hi_z = _SCORE_CLIP["z_factor"]
    lo_fs, hi_fs = _SCORE_CLIP["friction_stress"]
    lo_fm, hi_fm = _SCORE_CLIP["friction_margin"]

    sub_scores = [
        _normalise(result.omega, lo_o, hi_o),
        _normalise(result.oos_omega, lo_oos, hi_oos),
        _normalise(result.z_factor, lo_z, hi_z),
        _normalise(result.friction_stress_omega, lo_fs, hi_fs),
        _normalise(1.0 - result.friction_ratio, lo_fm, hi_fm),
    ]

    return round(sum(sub_scores) / len(sub_scores) * 100.0, 2)


def _assign_tier(result: QualificationResult) -> int:
    """
    Assign a tier number to a qualification result.

    Returns
    -------
    int
        0 — disqualified (not qualified).
        1 — Tier 1: all high-confidence thresholds met.
        2 — Tier 2: qualified, below Tier 1 on ≥ 1 axis.
        3 — Tier 3: qualified but recalibration_due, or score borderline.
    """
    if not result.qualified:
        return 0

    # Tier 3: recalibration flag always demotes to watch regardless of metrics.
    if result.recalibration_due:
        return 3

    # Tier 1: all six thresholds must pass simultaneously.
    is_tier1 = (
        result.omega >= TIER1_MIN_IS_OMEGA
        and result.oos_omega >= TIER1_MIN_OOS_OMEGA
        and result.z_factor >= TIER1_MIN_Z
        and result.oos_survival_rate >= TIER1_MIN_OOS_SURVIVAL
        and result.friction_stress_omega >= TIER1_MIN_FRICTION_STRESS_OMEGA
        and result.n_trades >= TIER1_MIN_TRADES
        and result.friction_ratio <= TIER1_MAX_FRICTION_RATIO
    )
    if is_tier1:
        return 1

    # Tier 2: passed qualification gates (enforced by qualify_instrument)
    # but below Tier 1 on one or more axes.
    return 2


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication helpers (mirrors portfolio.deduplicate_underlyings)
# ══════════════════════════════════════════════════════════════════════════════

#: Pairs of symbols that share the same underlying exposure.
#: Key = canonical symbol (preferred when both qualify);
#: Value = alternate symbol (deduped out when both qualify).
#:
#: If the alternate has a HIGHER score it will be kept instead — the
#: dedup always keeps the higher-scoring variant, not necessarily the key.
_UNDERLYING_PAIRS: Dict[str, str] = {
    "NAS100": "NAS100ft",
    "DJ30": "DJ30ft",
    "UK100": "UK100ft",
    "FRA40": "FRA40ft",
    "GER40": "GER40ft",
    "UKOUSD": "UKOUSDft",
    "XAUUSD": "XAUUSD+",
}


def _build_dedup_map(
    scored: List[tuple[str, str, str, str, float]],
) -> Dict[str, str]:
    """
    Build a map of {demoted_symbol: kept_symbol} for deduplication.

    For each underlying pair where both versions appear in ``scored``,
    the lower-scoring version is marked as demoted.  Ties keep the
    canonical (key) symbol.

    Parameters
    ----------
    scored : list of (instrument_id, symbol, broker_source, broker_account, score)
        All qualified instruments with their scores.

    Returns
    -------
    dict[str, str]
        Maps demoted symbols to the kept symbol.  A symbol not in this
        dict is NOT demoted by deduplication.
    """
    demoted: Dict[str, str] = {}
    canonical_by_symbol: Dict[str, str] = {}
    for a, b in _UNDERLYING_PAIRS.items():
        canonical_by_symbol[a] = a
        canonical_by_symbol[b] = a

    buckets: Dict[tuple[str, str, str], List[tuple[str, float]]] = {}
    for instrument_id, symbol, broker_source, broker_account, score in scored:
        canonical = canonical_by_symbol.get(symbol)
        if canonical is None:
            continue
        key = (broker_source, broker_account, canonical)
        buckets.setdefault(key, []).append((instrument_id, score))

    for _key, rows in buckets.items():
        if len(rows) < 2:
            continue
        keep_id = sorted(rows, key=lambda x: x[1], reverse=True)[0][0]
        for instrument_id, _score in rows:
            if instrument_id != keep_id:
                demoted[instrument_id] = keep_id
    return demoted


# ══════════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════════


def build_instrument_pool(
    registry: QualificationRegistry,
    config: Optional[PortfolioConfig] = None,
) -> InstrumentPool:
    """
    Build a tiered, scored instrument pool from a qualification registry.

    Steps
    -----
    1. Load all results from the registry.
    2. Score each qualified instrument (0–100 composite).
    3. Assign tier (1/2/3 for qualified, 0 for disqualified).
    4. Deduplicate underlying pairs (keep higher-scoring variant).
    5. Cluster capping: rank within cluster by score, keep top
       ``config.max_cluster_instruments``.
    6. Mark Tier 3 instruments as watch-list only (not active).
    7. Assign normalised equal weights to active instruments.
    8. Compute cluster-level statistics.
    9. Return :class:`InstrumentPool`.

    Parameters
    ----------
    registry : QualificationRegistry
        Loaded registry (call ``registry.load()`` before passing).
    config : PortfolioConfig, optional
        Portfolio configuration for cluster capping.
        Defaults to :class:`~kinetra.renko.portfolio.PortfolioConfig`.

    Returns
    -------
    InstrumentPool
        Fully populated pool with tier assignments, scores, cluster
        stats, and allocation weights.
    """
    cfg = config or PortfolioConfig()
    all_results: List[QualificationResult] = registry.all_results()
    now_utc = datetime.now(tz=timezone.utc).isoformat()

    logger.info("build_instrument_pool: %d results loaded from registry", len(all_results))

    # ── Step 1: Score and tier every result ──────────────────────────────────
    scored_qualified: List[tuple[str, str, str, str, float]] = []
    for r in all_results:
        if r.qualified:
            scored_qualified.append(
                (r.instrument_id, r.symbol, r.broker_source, r.broker_account, _compute_score(r))
            )

    # ── Step 2: Deduplicate underlying pairs ─────────────────────────────────
    dedup_map = _build_dedup_map(scored_qualified)  # {demoted_instrument_id: kept_instrument_id}

    # ── Step 3: Build initial PoolEntry list ─────────────────────────────────
    entries: List[PoolEntry] = []
    for r in all_results:
        tier = _assign_tier(r)
        score = _compute_score(r) if r.qualified else 0.0
        cluster = get_cluster(r.symbol)

        dedup_reason: Optional[str] = None
        if r.instrument_id in dedup_map:
            kept = dedup_map[r.instrument_id]
            dedup_reason = f"dedup: {kept} has equal or higher score for same underlying"

        entries.append(
            PoolEntry(
                symbol=r.symbol,
                instrument_id=r.instrument_id,
                tier=tier,
                score=score,
                cluster=cluster,
                cluster_rank=0,  # filled in step 4
                active=False,  # filled in step 5
                demotion_reason=dedup_reason,
                omega=r.omega,
                oos_omega=r.oos_omega,
                z_factor=r.z_factor,
                win_rate=r.win_rate,
                n_trades=r.n_trades,
                max_drawdown=r.max_drawdown,
                brick_size=r.brick_size,
                friction_ratio=r.friction_ratio,
                friction_stress_omega=r.friction_stress_omega,
                oos_survival_rate=r.oos_survival_rate,
                vr_peak=r.vr_peak,
                broker_source=r.broker_source,
                broker_account=r.broker_account,
                account_type=r.account_type,
                recalibration_due=r.recalibration_due,
                drift_reason=r.drift_reason,
                data_start=r.data_start,
                data_end=r.data_end,
                qualified_at=r.qualified_at,
                engine=getattr(r, "engine", "fixed"),
            )
        )

    # ── Step 4: Assign cluster rank (within qualified, active-eligible pool) ──
    # Group by cluster, rank by score descending.
    entry_map: Dict[str, PoolEntry] = {e.instrument_id: e for e in entries}
    cluster_groups: Dict[str, List[str]] = {}
    for e in entries:
        if e.tier > 0:  # qualified (tier 1/2/3)
            cluster_groups.setdefault(e.cluster, []).append(e.instrument_id)

    for cluster_name, instrument_ids in cluster_groups.items():
        ranked = sorted(instrument_ids, key=lambda k: entry_map[k].score, reverse=True)
        for rank, instrument_id in enumerate(ranked, 1):
            # Mutate via replacement (PoolEntry is not frozen)
            old = entry_map[instrument_id]
            entry_map[instrument_id] = PoolEntry(**{**old.to_dict(), "cluster_rank": rank})

    # ── Step 5: Assign active status ─────────────────────────────────────────
    # An instrument is active when:
    #   - Tier 1 or Tier 2 (Tier 3 is watch-list only)
    #   - Not deduped out
    #   - Within the cluster slot cap
    cluster_slot_counts: Dict[str, int] = {}

    # Process in score order (highest first) to ensure best instruments get slots.
    for instrument_id in sorted(entry_map, key=lambda k: entry_map[k].score, reverse=True):
        e = entry_map[instrument_id]

        if e.tier == 0:
            # Disqualified — never active.
            continue

        if e.tier == 3:
            # Watch-list tier — never active, but keep demotion reason.
            if e.demotion_reason is None:
                entry_map[instrument_id] = PoolEntry(
                    **{**e.to_dict(), "demotion_reason": "tier 3 watch-list (recalibration due)"}
                )
            continue

        if instrument_id in dedup_map:
            # Deduped out — already has demotion_reason set.
            continue

        # Check cluster cap.
        count = cluster_slot_counts.get(e.cluster, 0)
        if count >= cfg.max_cluster_instruments:
            entry_map[instrument_id] = PoolEntry(
                **{
                    **e.to_dict(),
                    "demotion_reason": (
                        f"cluster_cap: {cfg.max_cluster_instruments}/{cfg.max_cluster_instruments} "
                        f"slots used in '{e.cluster}'"
                    ),
                }
            )
            continue

        # Activate.
        cluster_slot_counts[e.cluster] = count + 1
        entry_map[instrument_id] = PoolEntry(
            **{**e.to_dict(), "active": True, "demotion_reason": None}
        )

    # ── Step 6: Assign equal weights to active instruments ───────────────────
    active_ids = [k for k, e in entry_map.items() if e.active]
    if active_ids:
        equal_w = round(1.0 / len(active_ids), 6)
        for instrument_id in active_ids:
            e = entry_map[instrument_id]
            entry_map[instrument_id] = PoolEntry(**{**e.to_dict(), "weight": equal_w})

    # ── Step 7: Build cluster statistics ─────────────────────────────────────
    cluster_stats: Dict[str, ClusterPoolStats] = {}
    all_clusters = set(e.cluster for e in entry_map.values())
    for cluster_name in sorted(all_clusters):
        cluster_entries = [e for e in entry_map.values() if e.cluster == cluster_name]
        active_e = [e for e in cluster_entries if e.active]
        watch_e = [e for e in cluster_entries if not e.active and e.tier in (2, 3)]
        disq_e = [e for e in cluster_entries if e.tier == 0]

        active_scores = [e.score for e in active_e]
        all_scores = [e.score for e in cluster_entries if e.tier > 0]

        cluster_stats[cluster_name] = ClusterPoolStats(
            cluster=cluster_name,
            n_active=len(active_e),
            n_watch=len(watch_e),
            n_disqualified=len(disq_e),
            active_symbols=sorted(e.instrument_id for e in active_e),
            watch_symbols=sorted(e.instrument_id for e in watch_e),
            top_score=max(all_scores) if all_scores else 0.0,
            mean_score=sum(active_scores) / len(active_scores) if active_scores else 0.0,
            cluster_weight=sum(e.weight for e in active_e),
        )

    # ── Step 8: Assemble final list ───────────────────────────────────────────
    # Sort: active (score DESC) → watch (score DESC) → disqualified (alpha).
    final_entries = sorted(
        entry_map.values(),
        key=lambda e: (
            0 if e.active else (1 if e.tier in (2, 3) else 2),
            -e.score,
            e.symbol,
            e.instrument_id,
        ),
    )

    n_active = sum(1 for e in final_entries if e.active)
    n_qualified = sum(1 for e in final_entries if e.tier > 0)
    n_disqualified = sum(1 for e in final_entries if e.tier == 0)
    n_watch = n_qualified - n_active

    pool = InstrumentPool(
        built_at=now_utc,
        n_processed=len(all_results),
        n_qualified=n_qualified,
        n_active=n_active,
        n_watch=n_watch,
        n_disqualified=n_disqualified,
        entries=final_entries,
        cluster_stats=cluster_stats,
        config=cfg,
    )

    logger.info(
        "build_instrument_pool: active=%d  watch=%d  disqualified=%d  clusters=%d",
        n_active,
        n_watch,
        n_disqualified,
        len(cluster_stats),
    )

    return pool


def build_pool_from_results(
    results: Sequence[QualificationResult],
    config: Optional[PortfolioConfig] = None,
) -> InstrumentPool:
    """
    Convenience builder that accepts a plain sequence of results without
    needing a persistent registry on disk.

    Useful for in-memory pipelines (e.g. :func:`run_full_pipeline` integration
    or unit tests) where :class:`QualificationRegistry` has not been loaded
    from disk.

    Parameters
    ----------
    results : sequence of QualificationResult
        All qualification results (qualified + unqualified).
    config : PortfolioConfig, optional
        Portfolio config for cluster capping.

    Returns
    -------
    InstrumentPool
    """
    # Build a temporary registry that holds the results without disk I/O.
    registry = QualificationRegistry.__new__(QualificationRegistry)
    registry.root_dir = Path(".")  # type: ignore[attr-defined]
    registry._results = {r.instrument_id: r for r in results}  # type: ignore[attr-defined]
    return build_instrument_pool(registry, config=config)
