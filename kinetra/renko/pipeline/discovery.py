"""
Canonical instrument discovery for the unified Renko pipeline.

:func:`discover_candidates` replaces the two functionally-identical
``discover_instruments()`` functions that existed independently in
``qualify_instruments.py`` and ``build_instrument_pool.py``.

Supported directory layouts
---------------------------
Canonical (broker-isolated)::

    data_root/<broker>/<account>/<category>/<SYMBOL>/*_M1_*.csv

Legacy::

    data_root/<category>/<SYMBOL>/*_M1_*.csv

The ``_M1_`` timeframe token is matched case-insensitively so files
named ``_m1_`` on case-sensitive filesystems are found correctly
(this fixes the bug present in all legacy CLI scripts).

De-duplication
--------------
When multiple M1 files resolve to the same ``instrument_id``, the file
with the highest ``(n_rows, date_span_seconds)`` score is kept.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .identity import InstrumentKey
from .specs import ContractSpec, load_contract_spec

logger = logging.getLogger(__name__)

_KNOWN_CATEGORIES: Set[str] = {
    "forex",
    "crypto",
    "metals",
    "indices",
    "energy",
    "commodities",
    "other",
}


@dataclass
class Candidate:
    """
    One discovered instrument ready for the qualification pipeline.

    Attributes
    ----------
    key : InstrumentKey
        Full identity (engine field is set by the caller / CLI flag).
    m1_path : Path
        Absolute path to the best M1 CSV file for this variant.
    spec : ContractSpec
        Contract specification loaded from ``contract_spec.json``.
    n_rows : int
        Approximate bar count (line count minus header).
    date_span_days : float
        Calendar days spanned by the filename timestamps (0 if not parseable).
    """

    key: InstrumentKey
    m1_path: Path
    spec: ContractSpec
    n_rows: int = 0
    date_span_days: float = 0.0

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def symbol(self) -> str:
        return self.key.symbol

    @property
    def instrument_id(self) -> str:
        return self.key.instrument_id

    def with_engine(self, engine: str) -> "Candidate":
        """Return a copy of this candidate with a different engine."""
        return Candidate(
            key=self.key.with_engine(engine),
            m1_path=self.m1_path,
            spec=self.spec,
            n_rows=self.n_rows,
            date_span_days=self.date_span_days,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ══════════════════════════════════════════════════════════════════════════════


def _parse_identity_from_path(
    data_root: Path,
    m1_file: Path,
) -> Tuple[str, str, str]:
    """
    Infer ``(broker_source, broker_account, category)`` from path structure.

    Returns
    -------
    (broker_source, broker_account, category) — all strings.
    Falls back to ``("legacy", "legacy", "unknown")`` when layout is ambiguous.
    """
    rel = m1_file.relative_to(data_root)
    parts = rel.parts
    lower = [p.lower() for p in parts]

    # Canonical: data_root/{broker}/{account}/{category}/{symbol}/{file}
    if len(parts) >= 5 and lower[2] in _KNOWN_CATEGORIES:
        return parts[0], parts[1], parts[2]

    # Legacy: data_root/{category}/{symbol}/{file}
    if len(parts) >= 3 and lower[0] in _KNOWN_CATEGORIES:
        return "legacy", "legacy", parts[0]

    # Fallback: scan for a category token and infer broker/account
    for i, token in enumerate(lower):
        if token not in _KNOWN_CATEGORIES:
            continue
        category = parts[i]
        if i >= 2:
            return parts[i - 2], parts[i - 1], category
        return "legacy", "legacy", category

    return "unknown", "unknown", "unknown"


def _candidate_score(m1_file: Path) -> Tuple[int, float]:
    """
    Score an M1 CSV for de-duplication.  Higher = prefer this file.

    Returns ``(n_rows, date_span_seconds)``.
    """
    try:
        with m1_file.open("r", encoding="utf-8", errors="ignore") as fh:
            n_rows = max(sum(1 for _ in fh) - 1, 0)
    except Exception:
        n_rows = 0

    span_seconds = 0.0
    m = re.search(r"_(\d{12})_(\d{12})\.csv$", m1_file.name, re.IGNORECASE)
    if m:
        try:
            t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            span_seconds = max((t1 - t0).total_seconds(), 0.0)
        except Exception:
            span_seconds = 0.0

    return n_rows, span_seconds


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def discover_candidates(
    data_root: Path,
    *,
    category: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    broker_source: str = "metaapi",
    engine: str = "fixed",
) -> List[Candidate]:
    """
    Scan *data_root* and return one :class:`Candidate` per discovered instrument.

    Parameters
    ----------
    data_root : Path
        Root directory to scan (e.g. ``data/master_standardized``).
    category : str or None
        If given, only instruments in this category folder are returned.
    symbols : list[str] or None
        If given, filter to these symbols only (case-insensitive).
    broker_source : str
        Broker source label written into the :class:`InstrumentKey`.
        Note: the actual broker is inferred from the path; this acts as
        a fallback when path-based inference returns ``"legacy"``.
    engine : str
        Engine label (``"fixed"`` or ``"adaptive"``) for the returned keys.

    Returns
    -------
    list[Candidate]
        One candidate per unique ``instrument_id``, sorted by ``instrument_id``.
    """
    if not data_root.exists():
        logger.error("data_root does not exist: %s", data_root)
        return []

    requested_syms: Optional[Set[str]] = (
        {s.upper() for s in symbols} if symbols else None
    )
    category_lower = category.lower() if category else None

    # best_by_id: instrument_id → (score, Candidate)
    best_by_id: Dict[str, Tuple[Tuple[int, float], Candidate]] = {}

    for m1_file in sorted(data_root.rglob("*.csv")):
        # Case-insensitive M1 detection (fixes the legacy case-sensitivity bug)
        if "_M1_" not in m1_file.name.upper():
            continue

        symbol = m1_file.name.split("_")[0].upper()
        if requested_syms and symbol not in requested_syms:
            continue

        rel_parts_lower = [p.lower() for p in m1_file.relative_to(data_root).parts]
        if category_lower is not None and category_lower not in rel_parts_lower:
            continue

        sym_dir = m1_file.parent
        inferred_broker, inferred_account, _cat = _parse_identity_from_path(
            data_root, m1_file
        )

        # Prefer path-inferred broker over the fallback parameter
        effective_broker = (
            inferred_broker
            if inferred_broker not in ("unknown", "legacy")
            else broker_source
        )
        effective_account = (
            inferred_account
            if inferred_account not in ("unknown", "legacy")
            else "unknown"
        )

        spec = load_contract_spec(sym_dir, symbol=symbol)

        key = InstrumentKey.build(
            symbol=symbol,
            broker_source=effective_broker,
            broker_account=effective_account,
            broker_symbol=spec.broker_symbol or symbol,
            is_ecn=spec.is_ecn,
            account_type=spec.account_type,
            engine=engine,
        )

        score = _candidate_score(m1_file)
        n_rows, span_seconds = score
        span_days = span_seconds / 86400.0

        candidate = Candidate(
            key=key,
            m1_path=m1_file,
            spec=spec,
            n_rows=n_rows,
            date_span_days=span_days,
        )

        iid = key.instrument_id
        existing = best_by_id.get(iid)
        if existing is None or score > existing[0]:
            best_by_id[iid] = (score, candidate)

    result = [cand for _, cand in sorted(best_by_id.values(), key=lambda x: x[1].instrument_id)]
    logger.info("discover_candidates: found %d candidates in %s", len(result), data_root)
    return result
