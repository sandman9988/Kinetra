"""
Central telemetry logger for system health and workflow observability.

Writes structured JSONL events to a single sink tree so backtesting,
paper/live trading, downloads, and infra health can be viewed together.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {str(k): _safe_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_safe_json(v) for v in obj]
        return str(obj)


@dataclass(frozen=True)
class EventRef:
    path: Path
    ts_utc: str
    stream: str
    event_type: str


class CentralTelemetry:
    """Thread-safe JSONL telemetry sink."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        env_root = os.getenv("KINETRA_TELEMETRY_DIR", "").strip()
        base = Path(env_root) if env_root else (Path.cwd() / "outputs" / "telemetry")
        self._root = (root_dir or base).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hostname = socket.gethostname()
        self._pid = os.getpid()

    @property
    def root(self) -> Path:
        return self._root

    def _stream_path(self, stream: str, ts: datetime) -> Path:
        day = ts.strftime("%Y-%m-%d")
        day_dir = self._root / day
        day_dir.mkdir(parents=True, exist_ok=True)
        stream_clean = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stream)
        return day_dir / f"{stream_clean}.jsonl"

    def emit(
        self,
        *,
        stream: str,
        component: str,
        event_type: str,
        status: str = "info",
        payload: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> EventRef:
        ts = _utc_now()
        event = {
            "ts_utc": ts.isoformat(),
            "component": component,
            "event_type": event_type,
            "status": status,
            "host": self._hostname,
            "pid": self._pid,
            "payload": _safe_json(payload or {}),
            "tags": _safe_json(tags or {}),
        }
        path = self._stream_path(stream, ts)
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return EventRef(path=path, ts_utc=event["ts_utc"], stream=stream, event_type=event_type)

    def health(
        self,
        *,
        component: str,
        status: str,
        checks: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> EventRef:
        return self.emit(
            stream="health",
            component=component,
            event_type="health_snapshot",
            status=status,
            payload={
                "checks": checks or {},
                "metrics": metrics or {},
                "details": details or {},
            },
        )


_GLOBAL: Optional[CentralTelemetry] = None
_GLOBAL_LOCK = threading.Lock()


def get_telemetry() -> CentralTelemetry:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = CentralTelemetry()
    return _GLOBAL


def emit_event(
    *,
    stream: str,
    component: str,
    event_type: str,
    status: str = "info",
    payload: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        get_telemetry().emit(
            stream=stream,
            component=component,
            event_type=event_type,
            status=status,
            payload=payload,
            tags=tags,
        )
    except Exception:
        # Telemetry must never break business logic.
        pass


def emit_health(
    *,
    component: str,
    status: str,
    checks: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        get_telemetry().health(
            component=component,
            status=status,
            checks=checks,
            metrics=metrics,
            details=details,
        )
    except Exception:
        pass


@contextmanager
def telemetry_span(
    *,
    stream: str,
    component: str,
    operation: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    start = _utc_now()
    emit_event(
        stream=stream,
        component=component,
        event_type=f"{operation}_start",
        status="info",
        payload=payload or {},
    )
    try:
        yield
    except Exception as exc:
        end = _utc_now()
        emit_event(
            stream=stream,
            component=component,
            event_type=f"{operation}_failed",
            status="error",
            payload={
                "duration_s": (end - start).total_seconds(),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            },
        )
        raise
    else:
        end = _utc_now()
        emit_event(
            stream=stream,
            component=component,
            event_type=f"{operation}_done",
            status="ok",
            payload={"duration_s": (end - start).total_seconds()},
        )
