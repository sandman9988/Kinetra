#!/usr/bin/env python3
"""Summarize central telemetry into a quick system health overview."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_events(root: Path, since: datetime) -> List[Dict]:
    events: List[Dict] = []
    if not root.exists():
        return events
    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir():
            continue
        for path in sorted(day_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        ts = _parse_ts(str(obj.get("ts_utc", "")))
                        if ts is None or ts < since:
                            continue
                        events.append(obj)
            except Exception:
                continue
    events.sort(key=lambda e: str(e.get("ts_utc", "")))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Central telemetry health overview")
    parser.add_argument(
        "--telemetry-root",
        default="outputs/telemetry",
        help="Telemetry root folder (default: outputs/telemetry)",
    )
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    parser.add_argument("--tail", type=int, default=20, help="Show N most recent events")
    parser.add_argument(
        "--max-health-age-minutes",
        type=float,
        default=30.0,
        help="Ignore health snapshots older than this age (minutes)",
    )
    args = parser.parse_args()

    root = Path(args.telemetry_root).resolve()
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=max(args.hours, 1))
    events = _load_events(root, since=since)

    if not events:
        print(f"[INFO] No telemetry events found in {root} for last {args.hours}h")
        return 0

    by_component: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    health_latest: Dict[str, Dict] = {}
    max_age_min = max(float(args.max_health_age_minutes), 0.0)
    for e in events:
        comp = str(e.get("component", "unknown"))
        status = str(e.get("status", "info"))
        by_component[comp][status] += 1
        if e.get("event_type") == "health_snapshot":
            ts = _parse_ts(str(e.get("ts_utc", "")))
            if ts is None:
                continue
            age_min = (now - ts).total_seconds() / 60.0
            if age_min <= max_age_min:
                health_latest[comp] = e

    print(f"[OVERVIEW] telemetry_root={root}")
    print(f"[OVERVIEW] lookback_hours={args.hours} events={len(events)}")
    print("")
    print("[COMPONENT STATUS COUNTS]")
    for comp in sorted(by_component):
        counts = by_component[comp]
        print(
            f"- {comp}: ok={counts.get('ok', 0)} warn={counts.get('warn', 0)} "
            f"error={counts.get('error', 0)} critical={counts.get('critical', 0)} info={counts.get('info', 0)}"
        )

    print("")
    print("[LATEST HEALTH SNAPSHOTS]")
    for comp in sorted(health_latest):
        e = health_latest[comp]
        ts = _parse_ts(str(e.get("ts_utc", "")))
        age_min = ((now - ts).total_seconds() / 60.0) if ts is not None else float("nan")
        payload = e.get("payload", {}) or {}
        checks = payload.get("checks", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        print(
            f"- {comp}: ts={e.get('ts_utc')} status={e.get('status')} "
            f"age_min={age_min:.1f} checks={checks} metrics={metrics}"
        )

    tail_n = max(args.tail, 1)
    print("")
    print(f"[RECENT EVENTS x{tail_n}]")
    for e in reversed(events[-tail_n:]):
        print(
            f"- {e.get('ts_utc')} {e.get('component')} {e.get('event_type')} "
            f"status={e.get('status')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
