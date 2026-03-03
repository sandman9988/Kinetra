"""
Canonical contract-spec loading for the unified Renko pipeline.

:class:`ContractSpec` is a frozen dataclass that captures all broker-friction
and sizing metadata for one instrument variant.  :func:`load_contract_spec`
replaces the byte-for-byte duplicated ``_load_contract_spec_meta()`` helpers
that existed independently in ``qualify_instruments.py`` and
``build_instrument_pool.py``.

Field-priority rules
--------------------
``spread_pts``
    ``spread_typical_pts`` → ``spread_typical`` → ``spread`` →
    ``typical_spread_pts`` → ``broker_friction.spread_points`` →
    ``broker_friction.spread`` → 1.0

``tick_size``
    ``tick_size`` → ``tickSize`` → ``point_value`` → ``tickValue`` →
    ``broker_friction.tick_size`` → ``broker_friction.tickSize`` → 0.00001

All other fields fall back to 0.0 / ``None`` / ``""`` on missing data so
callers never receive a ``KeyError``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONTRACT_SPEC_FILENAME = "contract_spec.json"


@dataclass(frozen=True)
class ContractSpec:
    """
    Broker-neutral instrument specification.

    All fields have safe defaults so partial ``contract_spec.json`` files
    do not raise exceptions.

    Parameters
    ----------
    symbol : str
        Canonical symbol (e.g. ``"XAUUSD"``).
    broker_symbol : str
        Exact broker ticker (may equal ``symbol``).
    broker_source : str
        Data / execution provider.
    spread_pts : float
        Typical bid-ask spread in price points.
    tick_size : float
        Minimum price increment.
    commission_per_lot : float
        Round-trip commission in USD per standard lot.
    contract_size : float
        Units per lot (e.g. 100 for XAU/USD).
    tick_value_usd : float
        USD value of one tick move per lot.
    pip_value_usd : float
        USD value of one pip move per lot.
    volume_min : float
        Minimum lot size.
    volume_step : float
        Lot-size increment.
    volume_max : float
        Maximum lot size.
    swap_long_points : float
        Overnight swap for long positions (price points).
    swap_short_points : float
        Overnight swap for short positions (price points).
    is_ecn : Optional[bool]
        True = ECN execution, False = standard desk, None = unknown.
    account_type : str
        Raw account-type label from the broker spec.
    usd_per_price_unit : float
        Conversion factor: USD gain per 1-point move on 1 lot.
    """

    symbol: str
    broker_symbol: str
    broker_source: str
    spread_pts: float
    tick_size: float
    commission_per_lot: float
    contract_size: float
    tick_value_usd: float
    pip_value_usd: float
    volume_min: float
    volume_step: float
    volume_max: float
    swap_long_points: float
    swap_short_points: float
    is_ecn: Optional[bool]
    account_type: str
    usd_per_price_unit: float

    @classmethod
    def default(cls, symbol: str = "", broker_source: str = "unknown") -> "ContractSpec":
        """Return a spec with all-zero defaults (no contract_spec.json found)."""
        return cls(
            symbol=symbol,
            broker_symbol=symbol,
            broker_source=broker_source,
            spread_pts=1.0,
            tick_size=0.00001,
            commission_per_lot=0.0,
            contract_size=0.0,
            tick_value_usd=0.0,
            pip_value_usd=0.0,
            volume_min=0.0,
            volume_step=0.0,
            volume_max=0.0,
            swap_long_points=0.0,
            swap_short_points=0.0,
            is_ecn=None,
            account_type="",
            usd_per_price_unit=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker_symbol": self.broker_symbol,
            "broker_source": self.broker_source,
            "spread_pts": self.spread_pts,
            "tick_size": self.tick_size,
            "commission_per_lot": self.commission_per_lot,
            "contract_size": self.contract_size,
            "tick_value_usd": self.tick_value_usd,
            "pip_value_usd": self.pip_value_usd,
            "volume_min": self.volume_min,
            "volume_step": self.volume_step,
            "volume_max": self.volume_max,
            "swap_long_points": self.swap_long_points,
            "swap_short_points": self.swap_short_points,
            "is_ecn": self.is_ecn,
            "account_type": self.account_type,
            "usd_per_price_unit": self.usd_per_price_unit,
        }


def load_contract_spec(instrument_dir: Path, *, symbol: str = "") -> ContractSpec:
    """
    Load a :class:`ContractSpec` from ``contract_spec.json`` in *instrument_dir*.

    Returns a spec with safe defaults if the file is missing or malformed.
    Field priority matches the original ``_load_contract_spec_meta()`` helpers
    in the legacy CLI scripts (Sprint ≤5B).

    Parameters
    ----------
    instrument_dir : Path
        Directory containing ``contract_spec.json``.
    symbol : str
        Fallback symbol if not readable from the spec (defaults to dir name).
    """
    symbol = symbol or instrument_dir.name
    spec_path = instrument_dir / _CONTRACT_SPEC_FILENAME

    if not spec_path.exists():
        logger.debug("No contract_spec.json in %s — using defaults", instrument_dir)
        return ContractSpec.default(symbol=symbol)

    try:
        raw: Dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", spec_path, exc)
        return ContractSpec.default(symbol=symbol)

    bf: Dict[str, Any] = raw.get("broker_friction") or {}

    def _f(keys: list, default: float = 0.0) -> float:
        """Return first non-None, non-zero value from *keys* across raw and bf."""
        for k in keys:
            v = raw.get(k) or bf.get(k)
            if v is not None:
                try:
                    fv = float(v)
                    if fv:
                        return fv
                except (TypeError, ValueError):
                    pass
        return default

    spread_pts = _f(
        ["spread_typical_pts", "spread_typical", "spread", "typical_spread_pts",
         "spread_points"],
        default=1.0,
    )
    tick_size = _f(
        ["tick_size", "tickSize", "point_value", "tickValue"],
        default=0.00001,
    )

    broker_sym = str(
        raw.get("broker_symbol")
        or bf.get("broker_symbol")
        or raw.get("symbol")
        or instrument_dir.name
    )
    broker_source = str(raw.get("broker_source") or bf.get("broker_source") or "unknown")

    is_ecn_raw = raw.get("is_ecn")
    is_ecn: Optional[bool] = bool(is_ecn_raw) if isinstance(is_ecn_raw, bool) else None

    usd_per_price_unit = _f(
        ["usd_per_price_unit", "usd_per_point", "pip_value_usd"],
        default=0.0,
    )

    return ContractSpec(
        symbol=str(raw.get("symbol") or symbol),
        broker_symbol=broker_sym,
        broker_source=broker_source,
        spread_pts=spread_pts,
        tick_size=tick_size,
        commission_per_lot=_f(["commission_per_lot", "commission"]),
        contract_size=_f(["contract_size"]),
        tick_value_usd=_f(["tick_value_usd"]),
        pip_value_usd=_f(["pip_value_usd"]),
        volume_min=_f(["volume_min"]),
        volume_step=_f(["volume_step"]),
        volume_max=_f(["volume_max"]),
        swap_long_points=_f(["swap_long_points"]),
        swap_short_points=_f(["swap_short_points"]),
        is_ecn=is_ecn,
        account_type=str(raw.get("account_type") or ""),
        usd_per_price_unit=usd_per_price_unit,
    )
