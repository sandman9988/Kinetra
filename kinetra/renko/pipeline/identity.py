"""
Canonical instrument identity for the unified Renko pipeline.

An :class:`InstrumentKey` uniquely identifies one instrument variant:
same symbol at different brokers, accounts, or execution types are
distinct variants.  Adding the ``engine`` field makes fixed and adaptive
variants independently addressable in the registry.

instrument_id format
--------------------
``{symbol}__{broker_source}__{broker_account}__{exec_tag}__{engine}``

All five parts are sanitised with :func:`_sanitize_id` so they are
safe for use as directory names and JSON keys.

Backward-compatibility
----------------------
Old-format ``instrument_id`` values (Sprint ≤5B) have only four parts
(no engine suffix).  :meth:`InstrumentKey.from_legacy_id` reconstructs
a key from the old format by adding ``engine="fixed"``.  The old-format
``instrument_id`` string is preserved unchanged so existing
``data/renko_qualified/`` directories are still found on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


def _sanitize_id(value: str) -> str:
    """Replace non-alphanumeric chars (except ._-) with underscore."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


@dataclass(frozen=True)
class InstrumentKey:
    """
    Immutable identity of one instrument variant.

    Parameters
    ----------
    symbol : str
        Canonical symbol (e.g. ``"XAUUSD"``, ``"GER40"``).
    broker_source : str
        Data / execution provider (``"metaapi"``, ``"ctrader"``, ``"mt5"``).
    broker_account : str
        Account label at that broker (``"live"``, ``"demo"``, ``"raw"``).
    broker_symbol : str
        Exact ticker on the broker if it differs from ``symbol``
        (e.g. ``"XAUUSD+"`` for an ECN variant).
    exec_tag : str
        Execution type tag: ``"ecn"`` | ``"std"`` | sanitised account_type.
    engine : str
        Brick engine: ``"fixed"`` | ``"adaptive"``.
    """

    symbol: str
    broker_source: str
    broker_account: str
    broker_symbol: str
    exec_tag: str
    engine: str  # "fixed" | "adaptive"

    @property
    def instrument_id(self) -> str:
        """Five-part canonical id string."""
        return "__".join(
            [
                _sanitize_id(self.symbol),
                _sanitize_id(self.broker_source),
                _sanitize_id(self.broker_account),
                _sanitize_id(self.exec_tag),
                _sanitize_id(self.engine),
            ]
        )

    @property
    def legacy_instrument_id(self) -> str:
        """Four-part id as written by Sprint ≤5B qualification pipeline."""
        return "__".join(
            [
                _sanitize_id(self.symbol),
                _sanitize_id(self.broker_source),
                _sanitize_id(self.broker_account),
                _sanitize_id(self.exec_tag),
            ]
        )

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        symbol: str,
        broker_source: str,
        broker_account: str,
        broker_symbol: str = "",
        *,
        is_ecn: Optional[bool] = None,
        account_type: str = "",
        engine: str = "fixed",
    ) -> "InstrumentKey":
        """
        Construct a key from raw broker metadata.

        The ``exec_tag`` is derived from ``is_ecn`` / ``account_type`` using
        the same priority rule as the Sprint 5B CLI scripts:

        * ``is_ecn=True``  → ``exec_tag="ecn"``
        * ``is_ecn=False`` → ``exec_tag="std"``
        * otherwise        → sanitised ``account_type`` (or ``"unknown_exec"``)
        """
        if is_ecn is True:
            exec_tag = "ecn"
        elif is_ecn is False:
            exec_tag = "std"
        else:
            exec_tag = _sanitize_id(account_type.lower()) if account_type else "unknown_exec"

        return cls(
            symbol=symbol,
            broker_source=broker_source,
            broker_account=broker_account,
            broker_symbol=broker_symbol or symbol,
            exec_tag=exec_tag,
            engine=engine,
        )

    @classmethod
    def from_legacy_id(cls, legacy_id: str, *, engine: str = "fixed") -> "InstrumentKey":
        """
        Reconstruct a key from an old-format (4-part) ``instrument_id``.

        The ``broker_symbol`` defaults to ``symbol`` (may be overridden later).
        """
        parts = legacy_id.split("__")
        if len(parts) == 5:
            # Already new format — parse directly
            return cls(
                symbol=parts[0],
                broker_source=parts[1],
                broker_account=parts[2],
                broker_symbol=parts[0],
                exec_tag=parts[3],
                engine=parts[4],
            )
        if len(parts) == 4:
            return cls(
                symbol=parts[0],
                broker_source=parts[1],
                broker_account=parts[2],
                broker_symbol=parts[0],
                exec_tag=parts[3],
                engine=engine,
            )
        # Fallback: treat the whole string as symbol
        return cls(
            symbol=legacy_id,
            broker_source="unknown",
            broker_account="unknown",
            broker_symbol=legacy_id,
            exec_tag="unknown_exec",
            engine=engine,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "broker_source": self.broker_source,
            "broker_account": self.broker_account,
            "broker_symbol": self.broker_symbol,
            "exec_tag": self.exec_tag,
            "engine": self.engine,
            "instrument_id": self.instrument_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InstrumentKey":
        return cls(
            symbol=d.get("symbol", ""),
            broker_source=d.get("broker_source", "unknown"),
            broker_account=d.get("broker_account", "unknown"),
            broker_symbol=d.get("broker_symbol", d.get("symbol", "")),
            exec_tag=d.get("exec_tag", "unknown_exec"),
            engine=d.get("engine", "fixed"),
        )

    def with_engine(self, engine: str) -> "InstrumentKey":
        """Return a new key with a different engine."""
        return InstrumentKey(
            symbol=self.symbol,
            broker_source=self.broker_source,
            broker_account=self.broker_account,
            broker_symbol=self.broker_symbol,
            exec_tag=self.exec_tag,
            engine=engine,
        )
