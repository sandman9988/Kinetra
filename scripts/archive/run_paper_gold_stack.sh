#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"
DATA_ROOT="${DATA_ROOT:-data/master_standardized/ctrader/pepperstone_demo_45841299}"
SYMBOL="${SYMBOL:-XAUUSD}"
BROKER_SOURCE="${BROKER_SOURCE:-ctrader}"
BRICK_SIZE="${BRICK_SIZE:-1.0}"
STOP_BRICKS="${STOP_BRICKS:-1.0}"
PAPER_LOTS="${PAPER_LOTS:-0.01}"
STARTUP_SKIP_FLIPS="${STARTUP_SKIP_FLIPS:-2}"
DD_HALT="${DD_HALT:-1.0}"
MARKOV_THRESHOLD="${MARKOV_THRESHOLD:-0.60}"
MARKOV_WINDOW="${MARKOV_WINDOW:-50}"
FLIPRATE_THRESHOLD="${FLIPRATE_THRESHOLD:-1.0}"
FLIPRATE_WINDOW="${FLIPRATE_WINDOW:-50}"
MIN_TRADES="${MIN_TRADES:-30}"
MIN_OMEGA="${MIN_OMEGA:-1.5}"
MONDAY_OPEN_UTC="${MONDAY_OPEN_UTC:-00:00}"
FRIDAY_CLOSE_UTC="${FRIDAY_CLOSE_UTC:-23:59}"
LOSS_BRAKE_AFTER="${LOSS_BRAKE_AFTER:-8}"
LOSS_FLAT_AFTER="${LOSS_FLAT_AFTER:-12}"
LOSS_PAUSE_MINUTES="${LOSS_PAUSE_MINUTES:-120}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SESSION_NAME="${SESSION_NAME:-kinetra-paper-gold}"

RUN_CMD="cd \"$ROOT\" && source .venv/bin/activate && python scripts/ctrader/run_paper_gold.py --symbol \"$SYMBOL\" --data-root \"$DATA_ROOT\" --broker-source \"$BROKER_SOURCE\" --brick-size \"$BRICK_SIZE\" --stop-bricks \"$STOP_BRICKS\" --paper-lots \"$PAPER_LOTS\" --startup-skip-flips \"$STARTUP_SKIP_FLIPS\" --drawdown-halt-pct \"$DD_HALT\" --markov-threshold \"$MARKOV_THRESHOLD\" --markov-window \"$MARKOV_WINDOW\" --fliprate-threshold \"$FLIPRATE_THRESHOLD\" --fliprate-window \"$FLIPRATE_WINDOW\" --monday-open-utc \"$MONDAY_OPEN_UTC\" --friday-close-utc \"$FRIDAY_CLOSE_UTC\" --loss-brake-after \"$LOSS_BRAKE_AFTER\" --loss-flat-after \"$LOSS_FLAT_AFTER\" --loss-pause-minutes \"$LOSS_PAUSE_MINUTES\" --log-level \"$LOG_LEVEL\""
HEALTH_CMD="cd \"$ROOT\" && source .venv/bin/activate && watch -n 5 \"python scripts/ctrader/paper_readiness_report.py --min-trades $MIN_TRADES --min-omega $MIN_OMEGA --initial-equity 1000 --summary-compact\""

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
  fi
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '$RUN_CMD'"
  tmux split-window -h -t "$SESSION_NAME" "bash -lc '$HEALTH_CMD'"
  tmux select-layout -t "$SESSION_NAME" even-horizontal >/dev/null 2>&1 || true
  tmux attach-session -t "$SESSION_NAME"
else
  cat <<EOF
[INFO] tmux is not installed.
[INFO] Use Zed split terminal and run:

Pane 1:
$RUN_CMD

Pane 2:
$HEALTH_CMD
EOF
fi
