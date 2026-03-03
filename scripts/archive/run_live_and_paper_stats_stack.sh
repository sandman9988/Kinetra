#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"

# Live trader pane
LIVE_SYMBOL="${LIVE_SYMBOL:-XAUUSD}"
LIVE_DATA_ROOT="${LIVE_DATA_ROOT:-data/master_standardized/ctrader/pepperstone_demo_45841299}"
LIVE_BRICK_SIZE="${LIVE_BRICK_SIZE:-1.0}"
LIVE_STOP_BRICKS="${LIVE_STOP_BRICKS:-0.5}"
LIVE_MONDAY_OPEN_UTC="${LIVE_MONDAY_OPEN_UTC:-03:00}"
LIVE_FRIDAY_CLOSE_UTC="${LIVE_FRIDAY_CLOSE_UTC:-20:55}"
LIVE_LOSS_BRAKE_AFTER="${LIVE_LOSS_BRAKE_AFTER:-8}"
LIVE_LOSS_FLAT_AFTER="${LIVE_LOSS_FLAT_AFTER:-12}"
LIVE_LOSS_PAUSE_MINUTES="${LIVE_LOSS_PAUSE_MINUTES:-120}"
LIVE_TRAILING_MAE_ENABLED="${LIVE_TRAILING_MAE_ENABLED:-0}"
LIVE_TRAILING_MAE_AFTER_BRICKS="${LIVE_TRAILING_MAE_AFTER_BRICKS:-1}"
LIVE_TRAILING_MAE_FRACTION="${LIVE_TRAILING_MAE_FRACTION:-0.5}"
LIVE_BREAK_EVEN_ENABLED="${LIVE_BREAK_EVEN_ENABLED:-0}"
LIVE_BREAK_EVEN_AFTER_BRICKS="${LIVE_BREAK_EVEN_AFTER_BRICKS:-1}"
LIVE_BREAK_EVEN_TRIGGER_BRICKS="${LIVE_BREAK_EVEN_TRIGGER_BRICKS:-1.0}"
LIVE_BREAK_EVEN_BUFFER_TICKS="${LIVE_BREAK_EVEN_BUFFER_TICKS:-0}"
LIVE_STATUS_INTERVAL_SECONDS="${LIVE_STATUS_INTERVAL_SECONDS:-20}"
LIVE_DRY_RUN="${LIVE_DRY_RUN:-1}" # 1=safe shadow mode, 0=real live mode
LIVE_ACK="${LIVE_ACK:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-20}"
PREFLIGHT_OBSERVE_SECONDS="${PREFLIGHT_OBSERVE_SECONDS:-10}"

# Paper stats pane
MIN_TRADES="${MIN_TRADES:-30}"
MIN_OMEGA="${MIN_OMEGA:-1.5}"
INITIAL_EQUITY="${INITIAL_EQUITY:-1000}"
STATS_WATCH_SECONDS="${STATS_WATCH_SECONDS:-5}"
PAPER_LOG_PATH="${PAPER_LOG_PATH:-}"

LIVE_CMD="cd \"$ROOT\" && source .venv/bin/activate && python scripts/ctrader/run_live_gold.py --symbol \"$LIVE_SYMBOL\" --data-root \"$LIVE_DATA_ROOT\" --brick-size \"$LIVE_BRICK_SIZE\" --stop-bricks \"$LIVE_STOP_BRICKS\" --monday-open-utc \"$LIVE_MONDAY_OPEN_UTC\" --friday-close-utc \"$LIVE_FRIDAY_CLOSE_UTC\" --loss-brake-after \"$LIVE_LOSS_BRAKE_AFTER\" --loss-flat-after \"$LIVE_LOSS_FLAT_AFTER\" --loss-pause-minutes \"$LIVE_LOSS_PAUSE_MINUTES\" --status-interval-seconds \"$LIVE_STATUS_INTERVAL_SECONDS\""
LIVE_CMD="$LIVE_CMD --trailing-mae-after-bricks \"$LIVE_TRAILING_MAE_AFTER_BRICKS\" --trailing-mae-fraction \"$LIVE_TRAILING_MAE_FRACTION\" --break-even-after-bricks \"$LIVE_BREAK_EVEN_AFTER_BRICKS\" --break-even-trigger-bricks \"$LIVE_BREAK_EVEN_TRIGGER_BRICKS\" --break-even-buffer-ticks \"$LIVE_BREAK_EVEN_BUFFER_TICKS\""
if [[ "$LIVE_TRAILING_MAE_ENABLED" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --trailing-mae-enabled"
fi
if [[ "$LIVE_BREAK_EVEN_ENABLED" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --break-even-enabled"
fi
if [[ "$LIVE_DRY_RUN" == "1" ]]; then
  LIVE_CMD="$LIVE_CMD --dry-run"
else
  if [[ -z "$LIVE_ACK" ]]; then
    echo "[ERROR] LIVE_DRY_RUN=0 requires LIVE_ACK=I_UNDERSTAND_LIVE_RISK"
    exit 2
  fi
  LIVE_CMD="$LIVE_CMD --ack-live \"$LIVE_ACK\""
fi

STATS_INNER="python scripts/ctrader/paper_readiness_report.py --min-trades $MIN_TRADES --min-omega $MIN_OMEGA --initial-equity $INITIAL_EQUITY"
if [[ -n "$PAPER_LOG_PATH" ]]; then
  STATS_INNER="$STATS_INNER --log-path $PAPER_LOG_PATH"
fi
STATS_CMD="cd \"$ROOT\" && source .venv/bin/activate && watch -n \"$STATS_WATCH_SECONDS\" \"$STATS_INNER\""

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  PREFLIGHT_CMD="cd \"$ROOT\" && source .venv/bin/activate && python scripts/ctrader/check_live_wiring.py --symbols \"$LIVE_SYMBOL\" --timeout \"$PREFLIGHT_TIMEOUT\" --observe-seconds \"$PREFLIGHT_OBSERVE_SECONDS\" --require-bars --data-root \"$LIVE_DATA_ROOT\""
  echo "[INFO] Running preflight wiring check..."
  bash -lc "$PREFLIGHT_CMD"
fi

SESSION_NAME="${SESSION_NAME:-kinetra-live-paper}"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
  fi
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '$LIVE_CMD'"
  tmux set-option -t "$SESSION_NAME" -g mouse on >/dev/null 2>&1 || true
  tmux set-option -t "$SESSION_NAME" -g history-limit 200000 >/dev/null 2>&1 || true
  tmux set-window-option -t "$SESSION_NAME" -g mode-keys vi >/dev/null 2>&1 || true
  tmux bind-key -T root WheelUpPane if-shell -F '#{pane_in_mode}' 'send-keys -M' 'copy-mode -e' >/dev/null 2>&1 || true
  tmux bind-key -T copy-mode-vi WheelUpPane send-keys -X scroll-up >/dev/null 2>&1 || true
  tmux bind-key -T copy-mode-vi WheelDownPane send-keys -X scroll-down >/dev/null 2>&1 || true
  tmux split-window -h -t "$SESSION_NAME" "bash -lc '$STATS_CMD'"
  tmux select-layout -t "$SESSION_NAME" even-horizontal >/dev/null 2>&1 || true
  tmux attach-session -t "$SESSION_NAME"
else
  cat <<EOF
[INFO] tmux is not installed.
[INFO] Use Zed split terminal and run:

Pane 1 (live trading):
$LIVE_CMD

Pane 2 (paper stats):
$STATS_CMD
EOF
fi
