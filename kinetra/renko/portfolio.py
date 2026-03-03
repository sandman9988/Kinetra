"""
Renko Portfolio Construction
=============================

Cluster management, equal-risk sizing, portfolio equity curve construction,
and diversification controls for the Renko Kinetra pipeline.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §7):
  - Instruments are grouped into macro-factor clusters for diversification.
  - Equal-risk sizing: 1R = 0.5 × brick_size × usd_per_point.
  - Cluster caps prevent concentration (max instruments per cluster, max weight).
  - Portfolio equity curve: time-ordered merge of all trade P&L streams.
  - Walk-forward: IS (70%) vs OOS (30%) with OOS pass criteria.
  - Stress tests: friction sensitivity, correlation stress, worst-period.

This module is the canonical home for Renko portfolio logic.  All scripts
should import from here rather than maintaining inline portfolio code.

Usage::

    from kinetra.renko.portfolio import (
        get_cluster,
        equal_risk_weights,
        apply_cluster_caps,
        build_portfolio_equity,
        PortfolioConfig,
        PortfolioConstruction,
        ClusterStats,
        SizingInfo,
    )

See Also:
    - ``kinetra/renko/backtest.py`` — Renko backtester (instrument + portfolio)
    - ``kinetra/renko/dsp.py`` — DSP analysis (VR, brick sizing, friction)
    - ``docs/MANUAL.md §7`` — portfolio specification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Cluster Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

#: Canonical cluster mapping — symbol → cluster name.
#: This is the single source of truth.  Never duplicate this table.
CLUSTER_MAP: Dict[str, str] = {
    # ── Precious metals ───────────────────────────────────────────────────
    "XAUUSD": "precious_metals",
    "XAUUSD+": "precious_metals",
    "XAGUSD": "precious_metals",
    "XAGAUD": "precious_metals",
    "XAUJPY+": "precious_metals",
    "XAUAUD+": "precious_metals",
    "XAUEUR+": "precious_metals",
    "XPDUSD": "precious_metals",
    "XPTUSD": "precious_metals",
    # ── Energy ────────────────────────────────────────────────────────────
    "GASOIL-C": "energy",
    "NG-C": "energy",
    "GAS-C": "energy",
    "UKOUSD": "energy",
    "UKOUSDft": "energy",
    "USOUSD": "energy",
    "CL": "energy",
    "CL-OIL": "energy",
    "CLX": "energy",
    "CLNX": "energy",
    "USO": "energy",
    # ── Commodities ───────────────────────────────────────────────────────
    "COPPER-C": "commodities",
    # ── US indices ────────────────────────────────────────────────────────
    "NAS100": "us_indices",
    "NAS100ft": "us_indices",
    "DJ30": "us_indices",
    "DJ30ft": "us_indices",
    "US2000": "us_indices",
    "USDX": "us_indices",
    "BTCO": "us_indices",
    # ── EU indices ────────────────────────────────────────────────────────
    "UK100": "eu_indices",
    "UK100ft": "eu_indices",
    "FRA40": "eu_indices",
    "FRA40ft": "eu_indices",
    "FRA": "eu_indices",
    "FRAS": "eu_indices",
    "GER40": "eu_indices",
    "GER40ft": "eu_indices",
    "EU50": "eu_indices",
    "NETH25": "eu_indices",
    "NGRID": "eu_indices",
    # ── Asia indices ──────────────────────────────────────────────────────
    "Nikkei225": "asia_indices",
    "CHINA50": "asia_indices",
    "CHINA50ft": "asia_indices",
    "CHINAH": "asia_indices",
    # ── EM indices ────────────────────────────────────────────────────────
    "SA40": "em_indices",
    "SGDJ": "em_indices",
    # ── Crypto ────────────────────────────────────────────────────────────
    "BTCUSD": "crypto",
    "BTCEUR": "crypto",
    "BTCJPY": "crypto",
    "ETHUSD": "crypto",
    "ETHEUR": "crypto",
    "ETHJPY": "crypto",
    "LTCUSD": "crypto",
    "LTCJPY": "crypto",
    "ADAUSD": "crypto",
    "ADAJPY": "crypto",
    "BCHUSD": "crypto",
    "BCHJPY": "crypto",
    "DOTUSD": "crypto",
    "SOLUSD": "crypto",
    "SOLJPY": "crypto",
    "XRPUSD": "crypto",
    "XRPJPY": "crypto",
    "AVAUSD": "crypto",
    "BERAUSD": "crypto",
    "BNBUSD": "crypto",
    "XLMJPY": "crypto",
    "USDTJPY": "crypto",
    # ── USD majors ────────────────────────────────────────────────────────
    "EURUSD": "usd_major",
    "EURUSD+": "usd_major",
    "GBPUSD": "usd_major",
    "GBPUSD+": "usd_major",
    "USDCAD": "usd_major",
    "USDCAD+": "usd_major",
    "USDCHF+": "usd_major",
    "USDJPY": "usd_major",
    "USDJPY+": "usd_major",
    "AUDUSD": "usd_major",
    "AUDUSD+": "usd_major",
    "NZDUSD": "usd_major",
    "NZDUSD+": "usd_major",
    # ── EUR crosses ───────────────────────────────────────────────────────
    "EURAUD+": "eur_cross",
    "EURCAD+": "eur_cross",
    "EURCHF+": "eur_cross",
    "EURGBP+": "eur_cross",
    "EURJPY+": "eur_cross",
    "EURNZD+": "eur_cross",
    "EURCZK+": "eur_cross",
    "EURDKK+": "eur_cross",
    "EURHUF+": "eur_cross",
    "EURNOK+": "eur_cross",
    "EURPLN+": "eur_cross",
    "EURSEK+": "eur_cross",
    "EURSGD+": "eur_cross",
    # ── GBP crosses ───────────────────────────────────────────────────────
    "GBPAUD+": "gbp_cross",
    "GBPCAD+": "gbp_cross",
    "GBPCHF+": "gbp_cross",
    "GBPJPY+": "gbp_cross",
    "GBPNZD+": "gbp_cross",
    "GBPSGD+": "gbp_cross",
    # ── AUD crosses ───────────────────────────────────────────────────────
    "AUDCAD+": "aud_cross",
    "AUDCHF+": "aud_cross",
    "AUDCNH+": "aud_cross",
    "AUDJPY": "aud_cross",
    "AUDJPY+": "aud_cross",
    "AUDNZD+": "aud_cross",
    "AUDSGD+": "aud_cross",
    # ── NZD crosses ───────────────────────────────────────────────────────
    "NZDCAD+": "nzd_cross",
    "NZDCHF+": "nzd_cross",
    "NZDJPY+": "nzd_cross",
    "NZDSGD+": "nzd_cross",
    # ── CHF/CAD crosses ───────────────────────────────────────────────────
    "CADCHF+": "chf_cross",
    "CADJPY+": "chf_cross",
    "CHFJPY+": "chf_cross",
    "CHFSGD+": "chf_cross",
    "SGDJPY+": "chf_cross",
    # ── Interest rates ────────────────────────────────────────────────────
    "EURIBOR3M": "rates",
    # ── Single stocks ─────────────────────────────────────────────────────
    "ABNB": "stocks",
}

#: All known cluster names (sorted for deterministic iteration).
ALL_CLUSTERS: Tuple[str, ...] = tuple(sorted(set(CLUSTER_MAP.values())))


def get_cluster(symbol: str) -> str:
    """
    Return the cluster name for *symbol*.

    Falls back to a heuristic based on symbol prefix if the symbol is not
    in the canonical ``CLUSTER_MAP``.

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g. ``"XAUUSD+"``).

    Returns
    -------
    str
        Cluster name (e.g. ``"precious_metals"``).
    """
    if symbol in CLUSTER_MAP:
        return CLUSTER_MAP[symbol]

    # ── Heuristic fallback ──────────────────────────────────────────────
    sym = symbol.upper().replace("+", "")

    if any(sym.startswith(p) for p in ("XAU", "XAG", "XPD", "XPT")):
        return "precious_metals"
    if any(sym.startswith(p) for p in ("GASOIL", "NG-", "GAS-", "UKOU", "USOU", "CL")):
        return "energy"
    if any(sym.endswith(s) for s in ("USD", "EUR", "JPY", "GBP")) and len(sym) == 6:
        # 6-char FX pair — try to classify by base or quote
        base = sym[:3]
        if base == "EUR":
            return "eur_cross"
        if base == "GBP":
            return "gbp_cross"
        if base == "AUD":
            return "aud_cross"
        if base == "NZD":
            return "nzd_cross"
        if base in ("CHF", "CAD"):
            return "chf_cross"
        if base == "USD" or sym[3:] == "USD":
            return "usd_major"
    if any(sym.startswith(p) for p in ("BTC", "ETH", "LTC", "ADA", "SOL", "DOT", "XRP", "BCH")):
        return "crypto"
    if any(
        sym.startswith(p)
        for p in ("NAS", "DJ", "US2", "UK1", "FRA", "GER", "EU5", "NIK", "CHI", "SA4")
    ):
        return "indices"

    return "other"


def get_cluster_members(cluster: str) -> List[str]:
    """
    Return all symbols in ``CLUSTER_MAP`` belonging to *cluster*.

    Parameters
    ----------
    cluster : str
        Cluster name.

    Returns
    -------
    list[str]
        Sorted list of symbols.
    """
    return sorted(sym for sym, c in CLUSTER_MAP.items() if c == cluster)


def cluster_summary() -> Dict[str, int]:
    """
    Return a dict mapping each cluster name to its member count.

    Useful for quick diagnostics.
    """
    out: Dict[str, int] = {}
    for c in CLUSTER_MAP.values():
        out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items()))


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """
    Configuration for portfolio construction.

    Attributes
    ----------
    target_risk_usd : float
        Target risk per trade in USD (1R).
    max_cluster_instruments : int
        Maximum instruments allowed from a single cluster.
    max_cluster_weight : float
        Maximum weight fraction a single cluster can have (0–1).
    deduplicate_underlyings : bool
        If True, when both spot and futures versions of an instrument
        qualify (e.g. NAS100 and NAS100ft), select only the one with
        lower friction ratio.  Prevents double-counting the same
        underlying exposure.
    """

    target_risk_usd: float = 100.0
    max_cluster_instruments: int = 3
    max_cluster_weight: float = 0.35
    deduplicate_underlyings: bool = True


@dataclass(slots=True)
class SizingInfo:
    """
    Equal-risk sizing calculation for one instrument.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    brick_size : float
        Brick size in price units.
    usd_per_point : float
        Empirical USD per price-point (from trade outcomes or spec).
    one_r_usd : float
        Risk per trade: 0.5 × brick_size × usd_per_point.
    lot_scale : float
        Lot-scale factor: target_risk_usd / one_r_usd.
    cluster : str
        Cluster name for this instrument.
    """

    symbol: str
    brick_size: float
    usd_per_point: float
    one_r_usd: float
    lot_scale: float
    cluster: str


@dataclass(frozen=True, slots=True)
class ClusterStats:
    """
    Per-cluster portfolio statistics.

    Attributes
    ----------
    name : str
        Cluster name.
    n_instruments : int
        Number of instruments selected from this cluster.
    symbols : tuple[str, ...]
        Symbols in this cluster.
    weight_fraction : float
        Fraction of total portfolio weight this cluster holds (0–1).
    total_pnl : float
        Sum of net P&L from instruments in this cluster.
    """

    name: str
    n_instruments: int
    symbols: Tuple[str, ...]
    weight_fraction: float
    total_pnl: float = 0.0


@dataclass(slots=True)
class PortfolioConstruction:
    """
    Complete portfolio construction result.

    This is the output of :func:`build_portfolio` — it contains the
    selected instruments, sizing, cluster breakdown, and portfolio-level
    metrics.

    Attributes
    ----------
    instruments : list[str]
        Final selected symbols (after cluster capping and dedup).
    sizing : dict[str, SizingInfo]
        Per-instrument sizing info.
    allocation_weights : dict[str, float]
        Final allocation weight per instrument (lot_scale, post-cap).
    cluster_stats : dict[str, ClusterStats]
        Per-cluster statistics.
    n_clusters : int
        Number of distinct clusters represented.
    herfindahl : float
        Herfindahl index for cluster concentration (lower = more diverse).
    max_cluster_weight : float
        Actual maximum cluster weight fraction.
    config : PortfolioConfig
        Configuration used.
    """

    instruments: List[str]
    sizing: Dict[str, SizingInfo]
    allocation_weights: Dict[str, float]
    cluster_stats: Dict[str, ClusterStats]
    n_clusters: int
    herfindahl: float
    max_cluster_weight: float
    config: PortfolioConfig


# ══════════════════════════════════════════════════════════════════════════════
# USD-per-point estimation
# ══════════════════════════════════════════════════════════════════════════════


def estimate_usd_per_point(
    trades: Sequence,
    *,
    fallback: float = 1.0,
) -> float:
    """
    Empirically estimate USD-per-price-point from completed trades.

    Uses the median of ``|gross_usd / gross_pts|`` across trades where
    ``|gross_pts| > 0``.  This gives a robust per-point dollar value that
    accounts for contract size, lot size, and quote-currency conversion.

    Parameters
    ----------
    trades : sequence of trade objects
        Each trade must have ``.gross_usd`` and ``.gross_pts`` attributes.
    fallback : float
        Value to return if no usable trades exist.

    Returns
    -------
    float
        Estimated USD per price point.
    """
    ratios = []
    for t in trades:
        pts = getattr(t, "gross_pts", 0.0)
        usd = getattr(t, "gross_usd", 0.0)
        if abs(pts) > 1e-12:
            ratios.append(abs(usd / pts))

    if not ratios:
        return fallback

    return float(np.median(ratios))


# ══════════════════════════════════════════════════════════════════════════════
# Equal-risk sizing
# ══════════════════════════════════════════════════════════════════════════════


def equal_risk_weights(
    instruments: Dict[str, float],
    usd_per_points: Dict[str, float],
    target_risk_usd: float = 100.0,
) -> Dict[str, SizingInfo]:
    """
    Compute equal-risk lot-scale factors for a set of instruments.

    Parameters
    ----------
    instruments : dict[str, float]
        Mapping of symbol → brick_size.
    usd_per_points : dict[str, float]
        Mapping of symbol → USD per price point.
    target_risk_usd : float
        Target risk per trade in USD.

    Returns
    -------
    dict[str, SizingInfo]
        Per-instrument sizing info keyed by symbol.
    """
    result: Dict[str, SizingInfo] = {}
    for sym, brick in instruments.items():
        upp = usd_per_points.get(sym, 1.0)
        one_r = 0.5 * brick * upp
        scale = target_risk_usd / one_r if one_r > 0 else 1.0
        cluster = get_cluster(sym)
        result[sym] = SizingInfo(
            symbol=sym,
            brick_size=brick,
            usd_per_point=upp,
            one_r_usd=one_r,
            lot_scale=scale,
            cluster=cluster,
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Cluster capping
# ══════════════════════════════════════════════════════════════════════════════


def apply_cluster_caps(
    sizing: Dict[str, SizingInfo],
    max_instruments_per_cluster: int = 3,
    max_cluster_weight: float = 0.35,
    rank_key: Optional[str] = None,
    rank_values: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, SizingInfo], Dict[str, float]]:
    """
    Apply cluster caps to a set of sized instruments.

    Limits the number of instruments per cluster and scales down weights
    when any cluster exceeds ``max_cluster_weight`` of the total.

    Parameters
    ----------
    sizing : dict[str, SizingInfo]
        Per-instrument sizing (from :func:`equal_risk_weights`).
    max_instruments_per_cluster : int
        Max instruments allowed from a single cluster.
    max_cluster_weight : float
        Maximum weight fraction (0–1) for a single cluster.
    rank_key : str or None
        Attribute name to rank instruments within a cluster (e.g.
        ``"lot_scale"``).  Higher is better.  If None, rank by lot_scale.
    rank_values : dict[str, float] or None
        External ranking values (e.g. Omega) per symbol.  If provided,
        this takes precedence over ``rank_key``.

    Returns
    -------
    tuple[dict[str, SizingInfo], dict[str, float]]
        (capped_sizing, allocation_weights) — the capped sizing dict
        and the final allocation weight per instrument.
    """
    # Group by cluster
    clusters: Dict[str, List[SizingInfo]] = {}
    for info in sizing.values():
        c = info.cluster
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(info)

    # Select top-N per cluster
    selected: List[SizingInfo] = []
    for cluster_name, members in clusters.items():
        if rank_values is not None:
            members_sorted = sorted(
                members,
                key=lambda m: rank_values.get(m.symbol, 0.0),
                reverse=True,
            )
        elif rank_key is not None:
            members_sorted = sorted(
                members,
                key=lambda m: getattr(m, rank_key, 0.0),
                reverse=True,
            )
        else:
            members_sorted = sorted(members, key=lambda m: m.lot_scale, reverse=True)

        selected.extend(members_sorted[:max_instruments_per_cluster])

    if not selected:
        return {}, {}

    # Build allocation weights (lot_scale)
    capped_sizing: Dict[str, SizingInfo] = {s.symbol: s for s in selected}
    weights: Dict[str, float] = {s.symbol: s.lot_scale for s in selected}

    # Apply cluster weight cap
    total_instruments = len(selected)
    cluster_counts: Dict[str, List[str]] = {}
    for s in selected:
        if s.cluster not in cluster_counts:
            cluster_counts[s.cluster] = []
        cluster_counts[s.cluster].append(s.symbol)

    for cluster_name, syms in cluster_counts.items():
        cluster_frac = len(syms) / total_instruments
        if cluster_frac > max_cluster_weight and len(syms) > 1:
            allowed = max(1, int(max_cluster_weight * total_instruments))
            scale_down = allowed / len(syms)
            for sym in syms:
                weights[sym] *= scale_down

    return capped_sizing, weights


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════════════════════════

#: Known spot/futures pairs (spot, futures) — only keep the better one.
_SPOT_FUTURES_PAIRS: List[Tuple[str, str]] = [
    ("NAS100", "NAS100ft"),
    ("DJ30", "DJ30ft"),
    ("UK100", "UK100ft"),
    ("FRA40", "FRA40ft"),
    ("GER40", "GER40ft"),
    ("CHINA50", "CHINA50ft"),
    ("UKOUSD", "UKOUSDft"),
]


def deduplicate_underlyings(
    symbols: Sequence[str],
    friction_ratios: Dict[str, float],
) -> List[str]:
    """
    Remove duplicate underlying exposures (spot vs futures).

    When both members of a spot/futures pair are present, keeps the one
    with the lower friction ratio.

    Parameters
    ----------
    symbols : sequence of str
        Candidate symbols.
    friction_ratios : dict[str, float]
        Friction ratio per symbol.

    Returns
    -------
    list[str]
        Deduplicated symbol list (sorted).
    """
    sym_set: Set[str] = set(symbols)
    remove: Set[str] = set()

    for spot, futures in _SPOT_FUTURES_PAIRS:
        if spot in sym_set and futures in sym_set:
            fr_spot = friction_ratios.get(spot, 999.0)
            fr_futures = friction_ratios.get(futures, 999.0)
            if fr_spot <= fr_futures:
                remove.add(futures)
            else:
                remove.add(spot)

    return sorted(sym_set - remove)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio equity curve
# ══════════════════════════════════════════════════════════════════════════════


def build_portfolio_equity(
    instrument_trades: Dict[str, Sequence],
    allocation_weights: Dict[str, float],
) -> Tuple[List[float], List[float], List[Tuple[datetime, float, str]]]:
    """
    Merge all trades across instruments, sorted by exit time, with sizing.

    Parameters
    ----------
    instrument_trades : dict[str, sequence]
        Mapping of symbol → list of trade objects.  Each trade must have
        ``.exit_time`` (datetime) and ``.net_usd`` (float) attributes.
    allocation_weights : dict[str, float]
        Allocation weight per instrument.

    Returns
    -------
    tuple[list[float], list[float], list[tuple]]
        ``(equity_curve, net_returns, trade_log)`` where:
        - ``equity_curve`` starts at 0.0 and accumulates scaled net P&L;
        - ``net_returns`` is the per-trade scaled net P&L;
        - ``trade_log`` is ``(exit_time, scaled_net, symbol)`` per trade.
    """
    all_trades: List[Tuple[datetime, float, str]] = []

    for sym, trades in instrument_trades.items():
        weight = allocation_weights.get(sym, 1.0)
        for t in trades:
            exit_time = getattr(t, "exit_time", None)
            net = getattr(t, "net_usd", 0.0) * weight
            if exit_time is not None:
                all_trades.append((exit_time, net, sym))

    all_trades.sort(key=lambda x: x[0])

    equity: List[float] = [0.0]
    returns: List[float] = []
    cumulative = 0.0

    for _, net, _ in all_trades:
        cumulative += net
        equity.append(cumulative)
        returns.append(net)

    return equity, returns, all_trades


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-level metrics
# ══════════════════════════════════════════════════════════════════════════════


def herfindahl_index(weights: Dict[str, float]) -> float:
    """
    Compute the Herfindahl–Hirschman Index for concentration.

    A value of 1.0 means perfectly concentrated (one instrument).
    Lower values mean more diversified.

    Parameters
    ----------
    weights : dict[str, float]
        Weights (counts, allocation weights, or P&L) per group.

    Returns
    -------
    float
        HHI ∈ [1/N, 1].  Returns 1.0 if no weights.
    """
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        return 1.0
    shares = [abs(v) / total for v in weights.values()]
    return float(sum(s * s for s in shares))


def max_drawdown(equity: Sequence[float]) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Parameters
    ----------
    equity : sequence of float
        Cumulative equity curve.

    Returns
    -------
    float
        Maximum drawdown (negative value).  Zero if monotonically rising.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def max_drawdown_duration_days(
    equity: Sequence[float],
    timestamps: Sequence[datetime],
) -> float:
    """
    Compute the longest drawdown duration in calendar days.

    Parameters
    ----------
    equity : sequence of float
        Cumulative equity curve.
    timestamps : sequence of datetime
        Timestamp corresponding to each equity point (must be same length).

    Returns
    -------
    float
        Longest drawdown duration in days.  Zero if never in drawdown.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2 or len(timestamps) < 2:
        return 0.0

    peak = np.maximum.accumulate(eq)
    in_dd = eq < peak

    longest = 0.0
    dd_start_idx: Optional[int] = None

    for i in range(len(eq)):
        if in_dd[i]:
            if dd_start_idx is None:
                dd_start_idx = i
        else:
            if dd_start_idx is not None:
                duration = (timestamps[i] - timestamps[dd_start_idx]).total_seconds() / 86400.0
                longest = max(longest, duration)
                dd_start_idx = None

    # Handle case where we're still in drawdown at the end
    if dd_start_idx is not None:
        duration = (timestamps[-1] - timestamps[dd_start_idx]).total_seconds() / 86400.0
        longest = max(longest, duration)

    return longest


def calmar_ratio(
    net_pnl: float,
    max_dd: float,
    years: float,
) -> float:
    """
    Compute the Calmar ratio: annualised return / |max drawdown|.

    Parameters
    ----------
    net_pnl : float
        Total net P&L.
    max_dd : float
        Maximum drawdown (should be ≤ 0).
    years : float
        Duration of the period in years.

    Returns
    -------
    float
        Calmar ratio.  Zero if max_dd is zero or years is zero.
    """
    if years <= 0 or max_dd >= 0:
        return 0.0
    annual_pnl = net_pnl / years
    return annual_pnl / abs(max_dd)


def cvar(returns: Sequence[float], alpha: float = 0.05) -> float:
    """
    Compute Conditional Value at Risk (Expected Shortfall) at level *alpha*.

    Parameters
    ----------
    returns : sequence of float
        Trade-level or period-level returns.
    alpha : float
        Confidence level (default 5%).

    Returns
    -------
    float
        Mean of returns below the alpha-quantile.  Zero if insufficient data.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    threshold = np.percentile(arr, alpha * 100)
    tail = arr[arr <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())


# ══════════════════════════════════════════════════════════════════════════════
# Full portfolio builder
# ══════════════════════════════════════════════════════════════════════════════


def build_portfolio(
    instrument_data: Dict[str, dict],
    config: Optional[PortfolioConfig] = None,
) -> PortfolioConstruction:
    """
    Build a complete portfolio from qualified instrument data.

    This is the high-level orchestrator that chains sizing, cluster capping,
    deduplication, and diversity metrics.

    Parameters
    ----------
    instrument_data : dict[str, dict]
        Per-instrument data.  Each value dict must contain:
        - ``"brick_size"`` (float): selected brick size
        - ``"usd_per_point"`` (float): USD per price point
        - ``"omega"`` (float): Omega ratio (for ranking within clusters)
        - ``"friction_ratio"`` (float): spread / brick (for dedup)
    config : PortfolioConfig or None
        Portfolio configuration.  Defaults to ``PortfolioConfig()``.

    Returns
    -------
    PortfolioConstruction
        Complete portfolio construction result.
    """
    if config is None:
        config = PortfolioConfig()

    if not instrument_data:
        return _empty_construction(config)

    # ── Deduplication ───────────────────────────────────────────────────
    symbols = list(instrument_data.keys())
    if config.deduplicate_underlyings:
        fr_map = {sym: d.get("friction_ratio", 999.0) for sym, d in instrument_data.items()}
        symbols = deduplicate_underlyings(symbols, fr_map)

    # ── Equal-risk sizing ───────────────────────────────────────────────
    bricks = {sym: instrument_data[sym]["brick_size"] for sym in symbols}
    upps = {sym: instrument_data[sym].get("usd_per_point", 1.0) for sym in symbols}
    sizing = equal_risk_weights(bricks, upps, target_risk_usd=config.target_risk_usd)

    # ── Cluster capping ─────────────────────────────────────────────────
    omegas = {sym: instrument_data[sym].get("omega", 0.0) for sym in symbols}
    capped_sizing, weights = apply_cluster_caps(
        sizing,
        max_instruments_per_cluster=config.max_cluster_instruments,
        max_cluster_weight=config.max_cluster_weight,
        rank_values=omegas,
    )

    if not capped_sizing:
        return _empty_construction(config)

    # ── Cluster statistics ──────────────────────────────────────────────
    cluster_groups: Dict[str, List[str]] = {}
    for sym, info in capped_sizing.items():
        c = info.cluster
        if c not in cluster_groups:
            cluster_groups[c] = []
        cluster_groups[c].append(sym)

    total_instr = len(capped_sizing)
    cluster_stats: Dict[str, ClusterStats] = {}
    for c_name, syms in cluster_groups.items():
        cluster_stats[c_name] = ClusterStats(
            name=c_name,
            n_instruments=len(syms),
            symbols=tuple(sorted(syms)),
            weight_fraction=len(syms) / total_instr,
        )

    # ── Diversity metrics ───────────────────────────────────────────────
    cluster_weight_map = {c: len(syms) for c, syms in cluster_groups.items()}
    herf = herfindahl_index(cluster_weight_map)
    max_cw = max(
        (len(syms) / total_instr for syms in cluster_groups.values()),
        default=0.0,
    )

    return PortfolioConstruction(
        instruments=sorted(capped_sizing.keys()),
        sizing=capped_sizing,
        allocation_weights=weights,
        cluster_stats=cluster_stats,
        n_clusters=len(cluster_groups),
        herfindahl=herf,
        max_cluster_weight=max_cw,
        config=config,
    )


def _empty_construction(config: PortfolioConfig) -> PortfolioConstruction:
    """Return an empty portfolio construction."""
    return PortfolioConstruction(
        instruments=[],
        sizing={},
        allocation_weights={},
        cluster_stats={},
        n_clusters=0,
        herfindahl=1.0,
        max_cluster_weight=0.0,
        config=config,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stress testing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CorrelationStressResult:
    """
    Result of a correlation = 1 stress scenario.

    Simulates all instruments entering simultaneously and measures
    the worst-case portfolio-level drawdown.
    """

    n_instruments: int
    total_simultaneous_risk_usd: float
    worst_case_dd_usd: float
    risk_to_dd_ratio: float


def stress_correlation_one(
    sizing: Dict[str, SizingInfo],
    allocation_weights: Dict[str, float],
) -> CorrelationStressResult:
    """
    Simulate a correlation=1 stress scenario.

    Assumes all instruments enter simultaneously and all hit their stop
    at the same time.  Reports the total simultaneous 1R risk.

    Parameters
    ----------
    sizing : dict[str, SizingInfo]
        Per-instrument sizing info.
    allocation_weights : dict[str, float]
        Allocation weight per instrument.

    Returns
    -------
    CorrelationStressResult
        Worst-case simultaneous drawdown estimate.
    """
    total_risk = 0.0
    for sym, info in sizing.items():
        weight = allocation_weights.get(sym, 1.0)
        total_risk += info.one_r_usd * weight

    return CorrelationStressResult(
        n_instruments=len(sizing),
        total_simultaneous_risk_usd=total_risk,
        worst_case_dd_usd=-total_risk,
        risk_to_dd_ratio=1.0,
    )


@dataclass(frozen=True, slots=True)
class WorstPeriodResult:
    """
    Worst N-day period result for a trade series.

    Attributes
    ----------
    symbol : str
        Instrument symbol (or "portfolio" for portfolio-level).
    period_days : int
        Window size in calendar days.
    worst_pnl : float
        Net P&L during the worst period.
    worst_start : datetime or None
        Start of the worst period.
    worst_end : datetime or None
        End of the worst period.
    n_trades_in_worst : int
        Number of trades during the worst period.
    """

    symbol: str
    period_days: int
    worst_pnl: float
    worst_start: Optional[datetime]
    worst_end: Optional[datetime]
    n_trades_in_worst: int


def find_worst_period(
    trades: Sequence,
    symbol: str = "portfolio",
    period_days: int = 180,
) -> WorstPeriodResult:
    """
    Find the worst N-day window in a trade series.

    Uses a sliding window over trades sorted by exit time.

    Parameters
    ----------
    trades : sequence
        Trade objects with ``.exit_time`` and ``.net_usd`` attributes.
    symbol : str
        Label for the result.
    period_days : int
        Window size in calendar days.

    Returns
    -------
    WorstPeriodResult
        The worst period found.
    """
    if not trades:
        return WorstPeriodResult(
            symbol=symbol,
            period_days=period_days,
            worst_pnl=0.0,
            worst_start=None,
            worst_end=None,
            n_trades_in_worst=0,
        )

    # Sort by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    n = len(sorted_trades)
    from datetime import timedelta

    window = timedelta(days=period_days)

    worst_pnl = float("inf")
    worst_start: Optional[datetime] = None
    worst_end: Optional[datetime] = None
    worst_count = 0

    # Sliding window
    left = 0
    running_pnl = 0.0

    for right in range(n):
        running_pnl += sorted_trades[right].net_usd

        # Shrink window from the left
        while (
            left < right
            and (sorted_trades[right].exit_time - sorted_trades[left].exit_time) > window
        ):
            running_pnl -= sorted_trades[left].net_usd
            left += 1

        if running_pnl < worst_pnl:
            worst_pnl = running_pnl
            worst_start = sorted_trades[left].exit_time
            worst_end = sorted_trades[right].exit_time
            worst_count = right - left + 1

    return WorstPeriodResult(
        symbol=symbol,
        period_days=period_days,
        worst_pnl=worst_pnl if worst_pnl != float("inf") else 0.0,
        worst_start=worst_start,
        worst_end=worst_end,
        n_trades_in_worst=worst_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tail-risk analysis
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class TailRiskReport:
    """
    Tail-risk analysis for a set of trade returns.

    Attributes
    ----------
    n_trades : int
        Total number of trades.
    worst_trade_usd : float
        Worst single-trade loss.
    max_consecutive_losses : int
        Longest streak of consecutive losing trades.
    cvar_5pct : float
        Conditional Value at Risk at 5%.
    worst_month_usd : float
        Worst calendar-month P&L.
    worst_week_usd : float
        Worst calendar-week P&L.
    """

    n_trades: int
    worst_trade_usd: float
    max_consecutive_losses: int
    cvar_5pct: float
    worst_month_usd: float
    worst_week_usd: float


def tail_risk_analysis(
    trades: Sequence,
) -> TailRiskReport:
    """
    Compute tail-risk metrics from completed trades.

    Parameters
    ----------
    trades : sequence
        Trade objects with ``.exit_time`` and ``.net_usd`` attributes.

    Returns
    -------
    TailRiskReport
        Tail-risk summary.
    """
    if not trades:
        return TailRiskReport(
            n_trades=0,
            worst_trade_usd=0.0,
            max_consecutive_losses=0,
            cvar_5pct=0.0,
            worst_month_usd=0.0,
            worst_week_usd=0.0,
        )

    nets = np.array([t.net_usd for t in trades], dtype=np.float64)
    n = len(nets)

    # Worst single trade
    worst_trade = float(nets.min())

    # Max consecutive losses
    max_consec = 0
    current_streak = 0
    for net in nets:
        if net <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    # CVaR at 5%
    cvar_val = cvar(nets, alpha=0.05)

    # Worst month and worst week
    import pandas as pd

    exit_times = [t.exit_time for t in trades]
    ts = pd.Series(nets, index=pd.DatetimeIndex(exit_times))

    # Monthly P&L
    monthly = ts.resample("ME").sum()
    worst_month = float(monthly.min()) if len(monthly) > 0 else 0.0

    # Weekly P&L
    weekly = ts.resample("W").sum()
    worst_week = float(weekly.min()) if len(weekly) > 0 else 0.0

    return TailRiskReport(
        n_trades=n,
        worst_trade_usd=worst_trade,
        max_consecutive_losses=max_consec,
        cvar_5pct=cvar_val,
        worst_month_usd=worst_month,
        worst_week_usd=worst_week,
    )
