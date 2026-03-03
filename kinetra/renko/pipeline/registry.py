"""
Unified qualification registry and result schema.

:class:`PipelineQualificationResult` is the canonical JSON record written
by both the fixed and adaptive engines.  It is a superset of the Sprint ≤5B
``QualificationResult``; :meth:`PipelineQualificationResult.from_legacy`
upgrades old files on first load.

:class:`PipelineRegistry` loads all ``qualification.json`` files from a
root directory, accepting both old-format (4-part ``instrument_id``) and
new-format (5-part) files.

JSON file location
------------------
``data/renko_qualified/{instrument_id}/qualification.json``

The old 4-part directories coexist with new 5-part directories because the
old ``instrument_id`` (without ``__{engine}`` suffix) is preserved as the
on-disk directory name for existing files.  New files use the 5-part id.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .identity import InstrumentKey

logger = logging.getLogger(__name__)

QUALIFICATION_FILENAME = "qualification.json"
SESSION_PROFILE_FILENAME = "session_profile.json"
RECALIBRATION_LOG_FILENAME = "recalibration_log.json"


# ══════════════════════════════════════════════════════════════════════════════
# Sub-schemas
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataMeta:
    """M1 dataset statistics collected at qualification time."""

    bars_m1: int = 0
    start: str = ""
    end: str = ""
    coverage_ratio: float = 0.0
    spike_count: int = 0
    session_break_minutes: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataMeta":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class FrictionMeta:
    """Cost structure at qualification time."""

    spread_pts: float = 0.0
    commission_per_lot: float = 0.0
    tick_size: float = 0.0
    friction_ratio: float = 0.0
    stress_cost_mult: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrictionMeta":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ParamsMeta:
    """Selected parameters from qualification."""

    brick_size: float = 0.0
    filter_params: Dict[str, Any] = field(default_factory=dict)
    stop_params: Dict[str, Any] = field(default_factory=dict)
    adaptive: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brick_size": self.brick_size,
            "filter_params": self.filter_params,
            "stop_params": self.stop_params,
            "adaptive": self.adaptive,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParamsMeta":
        return cls(
            brick_size=float(d.get("brick_size", 0.0)),
            filter_params=dict(d.get("filter_params") or {}),
            stop_params=dict(d.get("stop_params") or {}),
            adaptive=dict(d.get("adaptive") or {}),
        )


@dataclass
class ISMetrics:
    omega: float = 0.0
    z: float = 0.0
    trades: int = 0
    pnl_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ISMetrics":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OOSMetrics:
    omega: float = 0.0
    z: float = 0.0
    trades: int = 0
    survival: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OOSMetrics":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class StressMetrics:
    omega: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StressMetrics":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GateMetrics:
    """Spread-gate coverage.  Both values are 0.0 for the fixed engine."""

    bar_fraction: float = 0.0
    trade_fraction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateMetrics":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MetricsMeta:
    vr_peak: float = 0.0
    is_: ISMetrics = field(default_factory=ISMetrics)
    oos: OOSMetrics = field(default_factory=OOSMetrics)
    stress: StressMetrics = field(default_factory=StressMetrics)
    gate: GateMetrics = field(default_factory=GateMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vr_peak": self.vr_peak,
            "is": self.is_.to_dict(),
            "oos": self.oos.to_dict(),
            "stress": self.stress.to_dict(),
            "gate": self.gate.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsMeta":
        return cls(
            vr_peak=float(d.get("vr_peak", 0.0)),
            is_=ISMetrics.from_dict(d.get("is") or {}),
            oos=OOSMetrics.from_dict(d.get("oos") or {}),
            stress=StressMetrics.from_dict(d.get("stress") or {}),
            gate=GateMetrics.from_dict(d.get("gate") or {}),
        )


@dataclass
class DriftState:
    """Drift detection state, updated by drift-check and recalibration stages."""

    recalibration_due: bool = False
    drift_reason: str = ""
    last_checked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ══════════════════════════════════════════════════════════════════════════════
# Top-level result
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineQualificationResult:
    """
    Unified qualification record written by both engines (Sprint 6A+).

    JSON layout::

        {
          "instrument_id": "XAUUSD__metaapi__live__ecn__fixed",
          "key": { ...InstrumentKey... },
          "qualified": true,
          "engine": "fixed",
          "qualified_at": "2026-03-01T...",
          "data":    { ...DataMeta... },
          "friction": { ...FrictionMeta... },
          "params":  { ...ParamsMeta... },
          "metrics": { "vr_peak": 2.8, "is": {...}, "oos": {...}, "stress": {...}, "gate": {...} },
          "drift":   { "recalibration_due": false, "drift_reason": "", "last_checked_at": "" },
          "failure_reason": "",
          "pipeline_version": "6A"
        }
    """

    instrument_id: str
    key: InstrumentKey
    qualified: bool
    engine: str
    qualified_at: str
    data: DataMeta
    friction: FrictionMeta
    params: ParamsMeta
    metrics: MetricsMeta
    drift: DriftState
    failure_reason: str = ""
    pipeline_version: str = "6A"

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "key": self.key.to_dict(),
            "qualified": self.qualified,
            "engine": self.engine,
            "qualified_at": self.qualified_at,
            "data": self.data.to_dict(),
            "friction": self.friction.to_dict(),
            "params": self.params.to_dict(),
            "metrics": self.metrics.to_dict(),
            "drift": self.drift.to_dict(),
            "failure_reason": self.failure_reason,
            "pipeline_version": self.pipeline_version,
        }

    def save(self, path: Path) -> None:
        """Atomically write to *path* (``tmp`` → ``replace``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineQualificationResult":
        key_d = d.get("key") or {}
        key = InstrumentKey.from_dict(key_d) if key_d else InstrumentKey.from_legacy_id(
            d.get("instrument_id", ""), engine=d.get("engine", "fixed")
        )
        return cls(
            instrument_id=str(d.get("instrument_id", key.instrument_id)),
            key=key,
            qualified=bool(d.get("qualified", False)),
            engine=str(d.get("engine", "fixed")),
            qualified_at=str(d.get("qualified_at", "")),
            data=DataMeta.from_dict(d.get("data") or {}),
            friction=FrictionMeta.from_dict(d.get("friction") or {}),
            params=ParamsMeta.from_dict(d.get("params") or {}),
            metrics=MetricsMeta.from_dict(d.get("metrics") or {}),
            drift=DriftState.from_dict(d.get("drift") or {}),
            failure_reason=str(d.get("failure_reason", "")),
            pipeline_version=str(d.get("pipeline_version", "legacy")),
        )

    @classmethod
    def load(cls, path: Path) -> "PipelineQualificationResult":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "metrics" in raw and "data" in raw:
            # New format
            return cls.from_dict(raw)
        # Old format — upgrade via from_legacy
        from kinetra.renko.qualify import QualificationResult
        legacy = QualificationResult.from_dict(raw)
        return cls.from_legacy(legacy)

    @classmethod
    def from_legacy(cls, result: Any) -> "PipelineQualificationResult":
        """
        Convert a Sprint ≤5B ``QualificationResult`` to the new schema.

        The engine defaults to ``"fixed"`` (the only engine that existed).
        The ``instrument_id`` is preserved as-is so existing on-disk
        directories are still found.
        """
        legacy_id = str(getattr(result, "instrument_id", "") or "")
        engine = str(getattr(result, "engine", "fixed") or "fixed")

        key = InstrumentKey.from_legacy_id(legacy_id, engine=engine)
        key = InstrumentKey(
            symbol=key.symbol,
            broker_source=str(getattr(result, "broker_source", key.broker_source)),
            broker_account=str(getattr(result, "broker_account", key.broker_account)),
            broker_symbol=str(getattr(result, "broker_symbol", key.symbol) or key.symbol),
            exec_tag=key.exec_tag,
            engine=engine,
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        return cls(
            instrument_id=legacy_id or key.legacy_instrument_id,
            key=key,
            qualified=bool(result.qualified),
            engine=engine,
            qualified_at=str(getattr(result, "qualified_at", now_iso) or now_iso),
            data=DataMeta(
                bars_m1=int(getattr(result, "bars_m1", 0)),
                start=str(getattr(result, "data_start", "")),
                end=str(getattr(result, "data_end", "")),
                coverage_ratio=float(getattr(result, "coverage_ratio", 0.0)),
                spike_count=int(getattr(result, "spike_count", 0)),
                session_break_minutes=float(getattr(result, "session_break_minutes", 30.0)),
            ),
            friction=FrictionMeta(
                spread_pts=float(getattr(result, "spread_pts", 0.0)),
                commission_per_lot=float(getattr(result, "commission_per_lot", 0.0)),
                tick_size=float(getattr(result, "tick_size", 0.0)),
                friction_ratio=float(getattr(result, "friction_ratio", 0.0)),
                stress_cost_mult=1.5,
            ),
            params=ParamsMeta(
                brick_size=float(getattr(result, "brick_size", 0.0)),
                filter_params=dict(getattr(result, "filter_params", {}) or {}),
                stop_params={},
                adaptive={},
            ),
            metrics=MetricsMeta(
                vr_peak=float(getattr(result, "vr_peak", 0.0)),
                is_=ISMetrics(
                    omega=float(getattr(result, "omega", 0.0)),
                    z=float(getattr(result, "z_factor", 0.0)),
                    trades=int(getattr(result, "n_trades", 0)),
                    pnl_usd=0.0,
                ),
                oos=OOSMetrics(
                    omega=float(getattr(result, "oos_omega", 0.0)),
                    z=0.0,
                    trades=0,
                    survival=float(getattr(result, "oos_survival_rate", 0.0)),
                ),
                stress=StressMetrics(
                    omega=float(getattr(result, "friction_stress_omega", 0.0)),
                    z=0.0,
                ),
                gate=GateMetrics(
                    bar_fraction=float(getattr(result, "gate_bar_fraction", 0.0)),
                    trade_fraction=float(getattr(result, "gate_trade_fraction", 0.0)),
                ),
            ),
            drift=DriftState(
                recalibration_due=bool(getattr(result, "recalibration_due", False)),
                drift_reason=str(getattr(result, "drift_reason", "") or ""),
                last_checked_at=str(getattr(result, "drift_last_checked_at", "") or ""),
            ),
            failure_reason=str(getattr(result, "disqualification_reason", "") or ""),
            pipeline_version=str(getattr(result, "pipeline_version", "legacy")),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════


class PipelineRegistry:
    """
    Disk-backed registry of :class:`PipelineQualificationResult` records.

    Accepts both old-format (4-part ``instrument_id``) and new-format
    (5-part) ``qualification.json`` files.  Old files are upgraded to the
    new schema on load; the upgraded version is written back on the next
    :meth:`save_result` call.

    Parameters
    ----------
    root : Path
        Registry root directory (e.g. ``data/renko_qualified``).
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._records: Dict[str, PipelineQualificationResult] = {}
        self._loaded = False

    def load(self) -> None:
        """Scan *root* and load all ``qualification.json`` files."""
        self._records.clear()
        if not self._root.exists():
            logger.debug("PipelineRegistry: root does not exist: %s", self._root)
            return

        for qfile in sorted(self._root.rglob(QUALIFICATION_FILENAME)):
            try:
                result = PipelineQualificationResult.load(qfile)
                self._records[result.instrument_id] = result
            except Exception as exc:
                logger.warning("Could not load %s: %s", qfile, exc)

        self._loaded = True
        logger.info(
            "PipelineRegistry: loaded %d records from %s", len(self._records), self._root
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_results(self) -> List[PipelineQualificationResult]:
        """All records (qualified and disqualified), sorted by instrument_id."""
        self._ensure_loaded()
        return sorted(self._records.values(), key=lambda r: r.instrument_id)

    def get_qualified(
        self, engine: Optional[str] = None
    ) -> List[PipelineQualificationResult]:
        """
        Return qualified records, optionally filtered by engine.

        Parameters
        ----------
        engine : str or None
            ``"fixed"``, ``"adaptive"``, or ``None`` (all engines).
        """
        self._ensure_loaded()
        results = [r for r in self._records.values() if r.qualified]
        if engine is not None:
            results = [r for r in results if r.engine == engine]
        return sorted(results, key=lambda r: r.instrument_id)

    def get_by_id(self, instrument_id: str) -> Optional[PipelineQualificationResult]:
        self._ensure_loaded()
        return self._records.get(instrument_id)

    def get_by_symbol(self, symbol: str) -> List[PipelineQualificationResult]:
        self._ensure_loaded()
        sym_up = symbol.upper()
        return [r for r in self._records.values() if r.key.symbol.upper() == sym_up]

    def save_result(self, result: PipelineQualificationResult) -> Path:
        """
        Persist *result* to disk and update the in-memory cache.

        Returns the path to the written file.
        """
        out_dir = self._root / result.instrument_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / QUALIFICATION_FILENAME
        result.save(path)
        self._records[result.instrument_id] = result
        return path

    def append_recalibration_log(
        self, instrument_id: str, entry: Dict[str, Any]
    ) -> None:
        """Append one entry to the instrument's recalibration_log.json."""
        log_path = self._root / instrument_id / RECALIBRATION_LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)

        existing: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        existing.append(entry)
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp.replace(log_path)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._records)
