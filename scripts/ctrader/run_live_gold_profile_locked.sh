#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"

cd "$ROOT"
source .venv/bin/activate

python scripts/ctrader/run_gold_control_center.py \
  --symbol XAUUSD \
  --brick-size 1.0 \
  --stop-bricks 0.5 \
  --monday-open-utc 03:00 \
  --friday-close-utc 20:55 \
  --loss-brake-after 8 \
  --loss-flat-after 12 \
  --loss-pause-minutes 120 \
  --status-interval-seconds 20 \
  --trailing-mae-enabled \
  --trailing-mae-after-bars 2 \
  --trailing-mae-fraction 0.5 \
  --no-break-even-enabled \
  --live-mode live \
  --ack-live I_UNDERSTAND_LIVE_RISK \
  --no-run-preflight
