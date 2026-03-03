#!/usr/bin/env bash
# Default live trading profile for XAUUSD.
# Mirrors run_paper_gold_stack.sh but targets run_live_gold.py.
#
# Safety defaults:
#   DRY_RUN=1   — shadow mode (paper dispatcher, live cTrader bars).
#                 Set DRY_RUN=0 and ACK=I_UNDERSTAND_LIVE_RISK to go live.
#   GATE=micro  — PER gate: micro=0.01 lot / small=0.10 lot / full=unlimited
#   PREFLIGHT=1 — runs check_live_wiring.py before launching the stack.
#
# Usage:
#   ./scripts/ctrader/run_live_gold_stack.sh
#   DRY_RUN=0 ACK=I_UNDERSTAND_LIVE_RISK ./scripts/ctrader/run_live_gold_stack.sh
#   DRY_RUN=0 ACK=I_UNDERSTAND_LIVE_RISK GATE=small ./scripts/ctrader/run_live_gold_stack.sh
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"
DATA_ROOT="${DATA_ROOT:-data/master_standardized/ctrader/pepperstone_demo_45841299}"
SYMBOL="${SYMBOL:-XAUUSD}"
BROKER_SOURCE="${BROKER_SOURCE:-ctrader}"
BRICK_SIZE="${BRICK_SIZE:-1.0}"
STOP_BRICKS="${STOP_BRICKS:-0.5}"
STARTUP_SKIP_FLIPS="${STARTUP_SKIP_FLIPS:-2}"
TARGET_RISK_USD="${TARGET_RISK_USD:-100.0}"
DD_HALT="${DD_HALT:-0.02}"
MONDAY_OPEN_UTC="${MONDAY_OPEN_UTC:-03:00}"
FRIDAY_CLOSE_UTC="${FRIDAY_CLOSE_UTC:-20:55}"
LOSS_BRAKE_AFTER="${LOSS_BRAKE_AFTER:-8}"
LOSS_FLAT_AFTER="${LOSS_FLAT_AFTER:-12}"
LOSS_PAUSE_MINUTES="${LOSS_PAUSE_MINUTES:-120}"
TRAILING_MAE_ENABLED="${TRAILING_MAE_ENABLED:-1}"
TRAILING_MAE_AFTER_BRICKS="${TRAILING_MAE_AFTER_BRICKS:-2}"
TRAILING_MAE_FRACTION="${TRAILING_MAE_FRACTION:-0.5}"
BREAK_EVEN_ENABLED="${BREAK_EVEN_ENABLED:-0}"
BREAK_EVEN_AFTER_BRICKS="${BREAK_EVEN_AFTER_BRICKS:-1}"
BREAK_EVEN_TRIGGER_BRICKS="${BREAK_EVEN_TRIGGER_BRICKS:-1.0}"
BREAK_EVEN_BUFFER_TICKS="${BREAK_EVEN_BUFFER_TICKS:-0}"
STATUS_INTERVAL_SECONDS="${STATUS_INTERVAL_SECONDS:-20}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-30}"
FILL_TIMEOUT="${FILL_TIMEOUT:-10}"
GATE="${GATE:-micro}"
DRY_RUN="${DRY_RUN:-1}"
ACK="${ACK:-}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-20}"
PREFLIGHT_OBSERVE_SECONDS="${PREFLIGHT_OBSERVE_SECONDS:-10}"
MIN_TRADES="${MIN_TRADES:-30}"
MIN_OMEGA="${MIN_OMEGA:-1.5}"
SESSION_NAME="${SESSION_NAME:-kinetra-live-gold}"

# --- Safety guard -----------------------------------------------------------
if [[ "$DRY_RUN" != "1" && "$ACK" != "I_UNDERSTAND_LIVE_RISK" ]]; then
  echo "[ERROR] DRY_RUN=0 requires ACK=I_UNDERSTAND_LIVE_RISK"
  exit 2
fi

# --- Build live command ------------------------------------------------------
LIVE_CMD="cd \"$ROOT\" && source .venv/bin/activate && python scripts/ctrader/run_live_gold.py"
LIVE_CMD="$LIVE_CMD --symbol \"$SYMBOL\""
LIVE_CMD="$LIVE_CMD --data-root \"$DATA_ROOT\""
LIVE_CMD="$LIVE_CMD --broker-source \"$BROKER_SOURCE\""
LIVE_CMD="$LIVE_CMD --brick-size \"$BRICK_SIZE\""
LIVE_CMD="$LIVE_CMD --stop-bricks \"$STOP_BRICKS\""
LIVE_CMD="$LIVE_CMD --startup-skip-flips \"$STARTUP_SKIP_FLIPS\""
LIVE_CMD="$LIVE_CMD --target-risk-usd \"$TARGET_RISK_USD\""
LIVE_CMD="$LIVE_CMD --drawdown-halt-pct \"$DD_HALT\""
LIVE_CMD="$LIVE_CMD --monday-open-utc \"$MONDAY_OPEN_UTC\""
LIVE_CMD="$LIVE_CMD --friday-close-utc \"$FRIDAY_CLOSE_UTC\""
LIVE_CMD="$LIVE_CMD --loss-brake-after \"$LOSS_BRAKE_AFTER\""
LIVE_CMD="$LIVE_CMD --loss-flat-after \"$LOSS_FLAT_AFTER\""
LIVE_CMD="$LIVE_CMD --loss-pause-minutes \"$LOSS_PAUSE_MINUTES\""
LIVE_CMD="$LIVE_CMD --trailing-mae-after-bricks \"$TRAILING_MAE_AFTER_BRICKS\""
LIVE_CMD="$LIVE_CMD --trailing-mae-fraction \"$TRAILING_MAE_FRACTION\""
LIVE_CMD="$LIVE_CMD --break-even-after-bricks \"$BREAK_EVEN_AFTER_BRICKS\""
LIVE_CMD="$LIVE_CMD --break-even-trigger-bricks \"$BREAK_EVEN_TRIGGER_BRICKS\""
LIVE_CMD="$LIVE_CMD --break-even-buffer-ticks \"$BREAK_EVEN_BUFFER_TICKS\""
LIVE_CMD="$LIVE_CMD --status-interval-seconds \"$STATUS_INTERVAL_SECONDS\""
LIVE_CMD="$LIVE_CMD --connect-timeout \"$CONNECT_TIMEOUT\""
LIVE_CMD="$LIVE_CMD --fill-timeout \"$FILL_TIMEOUT\""

if [[ "$TRAILING_MAE_ENABLED" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --trailing-mae-enabled"
fi
if [[ "$BREAK_EVEN_ENABLED" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --break-even-enabled"
fi
if [[ "$DRY_RUN" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --dry-run"
else
  LIVE_CMD="$LIVE_CMD --gate \"$GATE\" --ack-live \"$ACK\""
fi

HEALTH_CMD="cd \"$ROOT\" && source .venv/bin/activate && watch -n 5 \"python scripts/ctrader/paper_readiness_report.py --min-trades $MIN_TRADES --min-omega $MIN_OMEGA --initial-equity 1000 --summary-compact\""

# --- Preflight ---------------------------------------------------------------
if [[ "$PREFLIGHT" == "1" ]]; then
  PREFLIGHT_CMD="cd \"$ROOT\" && source .venv/bin/activate && python scripts/ctrader/check_live_wiring.py --symbols \"$SYMBOL\" --timeout \"$PREFLIGHT_TIMEOUT\" --observe-seconds \"$PREFLIGHT_OBSERVE_SECONDS\" --require-bars --data-root \"$DATA_ROOT\""
  echo "[INFO] Running preflight wiring check..."
  bash -lc "$PREFLIGHT_CMD"
fi

# --- Launch ------------------------------------------------------------------
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
  fi
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '$LIVE_CMD'"
  tmux set-option -t "$SESSION_NAME" -g mouse on >/dev/null 2>&1 || true
  tmux set-option -t "$SESSION_NAME" -g history-limit 200000 >/dev/null 2>&1 || true
  tmux split-window -h -t "$SESSION_NAME" "bash -lc '$HEALTH_CMD'"
  tmux select-layout -t "$SESSION_NAME" even-horizontal >/dev/null 2>&1 || true
  tmux attach-session -t "$SESSION_NAME"
else
  cat <<EOF
[INFO] tmux is not installed.
[INFO] Use Zed split terminal and run:

Pane 1 (live trader):
$LIVE_CMD

Pane 2 (health):
$HEALTH_CMD
EOF
fi
