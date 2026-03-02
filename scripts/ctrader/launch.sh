#!/usr/bin/env bash
# Interactive launcher for backtest, paper, and live trading.
# Uses bash `select` menus — type the number and press Enter.
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"
DATA_ROOT="${DATA_ROOT:-data/master_standardized/ctrader/pepperstone_demo_45841299}"

# ── helpers ──────────────────────────────────────────────────────────────────

bold()  { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
cyan()  { printf '\033[36m%s\033[0m' "$*"; }
yellow(){ printf '\033[33m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
dim()   { printf '\033[2m%s\033[0m' "$*"; }

hr() { printf '%0.s─' {1..60}; echo; }

# ── discover symbols ──────────────────────────────────────────────────────────

discover_symbols() {
  local root="$ROOT/$DATA_ROOT"
  if [[ -d "$root" ]]; then
    find "$root" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
      | awk -F/ '{print $NF}' | sort -u
  fi
}

# ── per-symbol defaults ───────────────────────────────────────────────────────
# Keys: BRICK STOP_PAPER STOP_LIVE MONDAY FRIDAY DD_HALT_LIVE TARGET_RISK

declare -A DEFAULTS_BRICK        DEFAULTS_STOP_PAPER  DEFAULTS_STOP_LIVE
declare -A DEFAULTS_MONDAY       DEFAULTS_FRIDAY
declare -A DEFAULTS_DD_HALT_LIVE DEFAULTS_TARGET_RISK

DEFAULTS_BRICK[XAUUSD]="1.0";        DEFAULTS_BRICK[NAS100]="5.0"
DEFAULTS_STOP_PAPER[XAUUSD]="1.0";   DEFAULTS_STOP_PAPER[NAS100]="1.0"
DEFAULTS_STOP_LIVE[XAUUSD]="0.5";    DEFAULTS_STOP_LIVE[NAS100]="1.0"
DEFAULTS_MONDAY[XAUUSD]="03:00";     DEFAULTS_MONDAY[NAS100]="14:30"
DEFAULTS_FRIDAY[XAUUSD]="20:55";     DEFAULTS_FRIDAY[NAS100]="21:00"
DEFAULTS_DD_HALT_LIVE[XAUUSD]="0.02";DEFAULTS_DD_HALT_LIVE[NAS100]="0.02"
DEFAULTS_TARGET_RISK[XAUUSD]="100.0";DEFAULTS_TARGET_RISK[NAS100]="100.0"

get_default() {
  local key="$1" sym="$2"
  local arr_name="DEFAULTS_${key}"
  local val
  # Use indirect reference
  eval "val=\${${arr_name}[$sym]:-}"
  echo "${val:--}"
}

# ── main menu ────────────────────────────────────────────────────────────────

pick_mode() {
  echo
  bold "  Select mode"; echo
  hr
  printf "  %-4s %-10s %s\n" "#" "Mode" "Description"
  printf "  %-4s %-10s %s\n" "──" "────" "───────────────────────────────────────────"
  printf "  %-4s %-10s %s\n" "1" "backtest"  "Quick 3-month historical validation"
  printf "  %-4s %-10s %s\n" "2" "full"      "Comprehensive 2-year backtest"
  printf "  %-4s %-10s %s\n" "3" "paper"     "Paper trading with live broker data"
  printf "  %-4s %-10s %s\n" "4" "dry-run"   "Live bars, paper orders (shadow mode)"
  printf "  %-4s %-10s %s\n" "5" "LIVE"      "REAL orders (requires confirmation)"
  hr
  PS3="  Choice: "
  select MODE_OPT in "backtest  — 3 months quick validation" \
                     "full      — 2 years comprehensive backtest" \
                     "paper     — live broker data, no real orders" \
                     "dry-run   — live bars, paper orders" \
                     "LIVE      — REAL orders (requires ACK)"; do
    case "$REPLY" in
      1) MODE="backtest"; MONTHS=3;  break ;;
      2) MODE="full";     MONTHS=24; break ;;
      3) MODE="paper";    break ;;
      4) MODE="dry_run";  break ;;
      5) MODE="live";     break ;;
      *) echo "  Enter 1-5." ;;
    esac
  done
}

# ── gate menu (live only) ────────────────────────────────────────────────────

pick_gate() {
  echo
  bold "  Select PER order gate"; echo
  hr
  printf "  %-8s %-12s %-10s %s\n" "Gate" "Lot ceiling" "Max DD" "Instruments"
  printf "  %-8s %-12s %-10s %s\n" "────" "───────────" "──────" "───────────"
  printf "  %-8s %-12s %-10s %s\n" "micro" "0.01"  "3%"  "≤2"
  printf "  %-8s %-12s %-10s %s\n" "small" "0.10"  "5%"  "≤5"
  printf "  %-8s %-12s %-10s %s\n" "full"  "broker" "10%" "unlimited"
  hr
  PS3="  Choice: "
  select GATE_OPT in "micro  — 0.01 lot ceiling, 3% max DD" \
                     "small  — 0.10 lot ceiling, 5% max DD" \
                     "full   — broker limits,    10% max DD"; do
    case "$REPLY" in
      1) GATE="micro"; break ;;
      2) GATE="small"; break ;;
      3) GATE="full";  break ;;
      *) echo "  Enter 1, 2, or 3." ;;
    esac
  done
}

# ── symbol menu ──────────────────────────────────────────────────────────────

pick_symbol() {
  local -a syms
  mapfile -t syms < <(discover_symbols)
  if [[ ${#syms[@]} -eq 0 ]]; then
    syms=("XAUUSD")
  fi

  echo
  bold "  Select symbol"; echo
  hr
  PS3="  Choice: "
  select SYM_OPT in "${syms[@]}"; do
    if [[ -n "$SYM_OPT" ]]; then
      SYMBOL="$SYM_OPT"
      break
    fi
    echo "  Invalid selection."
  done
}

# ── settings summary ─────────────────────────────────────────────────────────

show_settings() {
  local mode="$1" sym="$2" gate="${3:-}" months="${4:-3}"

  local brick stop monday friday

  brick=$(get_default BRICK "$sym")
  monday=$(get_default MONDAY "$sym")
  friday=$(get_default FRIDAY "$sym")

  if [[ "$mode" == "paper" ]]; then
    stop=$(get_default STOP_PAPER "$sym")
  else
    stop=$(get_default STOP_LIVE "$sym")
  fi

  echo
  if [[ "$mode" == "backtest" || "$mode" == "full" ]]; then
    bold "  Settings — "; cyan "$sym"; printf "  "; dim "(mode: $mode, $months months)"; echo
  else
    bold "  Settings — "; yellow "$sym"; printf "  "; dim "(mode: $mode)"; echo
  fi
  hr

  if [[ "$mode" == "backtest" || "$mode" == "full" ]]; then
    printf "  %-24s %s\n" "mode"           "$(cyan "BACKTEST")"
    printf "  %-24s %s\n" "duration"       "$(bold "$months months")"
    printf "  %-24s %s\n" "brick_size"     "$(bold "$brick")"
    printf "  %-24s %s\n" "stop_bricks"    "$(bold "1.0")"
    printf "  %-24s %s\n" "initial_equity" "$(bold "\$1,000")"
    printf "  %-24s %s\n" "sizing"         "$(bold "static + compounding")"
    hr
    printf "  %-24s %s\n" "min_omega"      "$(bold "≥ 1.5")"
    printf "  %-24s %s\n" "min_trades"     "$(bold "≥ 30")"
  else
    printf "  %-24s %s\n" "brick_size"      "$(bold "$brick")"
    printf "  %-24s %s\n" "stop_bricks"     "$(bold "$stop")"
    printf "  %-24s %s\n" "session UTC"     "$(bold "$monday → $friday")"
    printf "  %-24s %s\n" "loss_brake_after" "$(bold "8")"
    printf "  %-24s %s\n" "loss_flat_after"  "$(bold "12")"
    printf "  %-24s %s\n" "loss_pause_min"   "$(bold "120")"

    if [[ "$mode" == "paper" ]]; then
      printf "  %-24s %s\n" "paper_lots"      "$(bold "0.01")"
      printf "  %-24s %s\n" "dd_halt_pct"     "$(bold "1.0  (effectively off)")"
      printf "  %-24s %s\n" "trailing_mae"    "$(bold "off")"
    else
      local dd tr
      dd=$(get_default DD_HALT_LIVE "$sym")
      tr=$(get_default TARGET_RISK "$sym")
      printf "  %-24s %s\n" "drawdown_halt_pct" "$(bold "$dd")"
      printf "  %-24s %s\n" "target_risk_usd"   "$(bold "$tr")"
      printf "  %-24s %s\n" "trailing_mae"      "$(bold "on  (after 2 bars, 0.5 frac)")"
      if [[ "$mode" == "live" ]]; then
        printf "  %-24s %s\n" "preflight"   "$(bold "yes")"
        printf "  %-24s %s\n" "order_gate"  "$(red "REAL — $gate gate")"
      else
        printf "  %-24s %s\n" "preflight"   "$(bold "yes")"
        printf "  %-24s %s\n" "order_gate"  "$(yellow "paper dispatcher (dry-run)")"
      fi
    fi
  fi
  hr
}

# ── confirm ──────────────────────────────────────────────────────────────────

confirm() {
  local ans
  printf "  Launch? [Y/n] "
  read -r ans
  ans="${ans:-y}"
  [[ "$ans" =~ ^[Yy]$ ]]
}

# ── launch ───────────────────────────────────────────────────────────────────

do_launch() {
  local mode="$1" sym="$2" gate="${3:-micro}" months="${4:-3}"

  # Resolve absolute path to renko_engine.py regardless of cwd
  local engine
  engine="$(cd "$(dirname "$0")/../.." && pwd)/scripts/renko_engine.py"

  if [[ "$mode" == "backtest" ]]; then
    echo
    echo "  $(cyan '[LAUNCH]') backtest — $sym ($months months)"
    exec python "$engine" "$sym" --stage backtest --months "$months"

  elif [[ "$mode" == "full" ]]; then
    echo
    echo "  $(cyan '[LAUNCH]') full backtest — $sym ($months months)"
    exec python "$engine" "$sym" --stage backtest --months "$months"

  elif [[ "$mode" == "paper" ]]; then
    echo
    echo "  $(green '[LAUNCH]') paper — $sym"
    exec python "$engine" "$sym" --stage paper

  elif [[ "$mode" == "dry_run" ]]; then
    echo
    echo "  $(yellow '[LAUNCH]') dry-run — $sym  (live bars, paper orders)"
    _run_with_reconnect python "$engine" "$sym" --stage live --live-size "$gate" --dry-run

  elif [[ "$mode" == "live" ]]; then
    echo
    printf "  "
    red "You are about to send REAL orders."
    echo
    printf "  Type exactly: I_UNDERSTAND_LIVE_RISK\n  > "
    local ack
    read -r ack
    if [[ "$ack" != "I_UNDERSTAND_LIVE_RISK" ]]; then
      echo "  Aborted."
      exit 1
    fi
    echo
    echo "  $(red '[LAUNCH]') LIVE — $sym  gate=$gate"
    _run_with_reconnect python "$engine" "$sym" --stage live --live-size "$gate"
  fi
}

# ── reconnect loop ────────────────────────────────────────────────────────────
# Retries the command when the engine exits with code 2 (broker disconnected).
# Tries the backup cTrader server on each attempt via CTRADER_ALT_ENDPOINTS.
# Max 5 reconnect attempts; 30-second cooldown between each.

_run_with_reconnect() {
  local max_retries=5
  local cooldown=30
  local attempt=0
  while true; do
    "$@"
    local code=$?
    if [[ $code -eq 0 ]]; then
      break
    elif [[ $code -eq 2 ]]; then
      attempt=$(( attempt + 1 ))
      if [[ $attempt -ge $max_retries ]]; then
        echo
        red "  Max reconnect attempts ($max_retries) reached. Exiting."
        echo
        exit 1
      fi
      echo
      yellow "  [RECONNECT] Attempt $attempt/$max_retries — waiting ${cooldown}s..."
      echo
      sleep "$cooldown"
    else
      # Non-recoverable exit (code 1, etc.)
      exit $code
    fi
  done
}

# ── main ─────────────────────────────────────────────────────────────────────

clear
echo
bold "Kinetra — Trading Launcher"; echo
printf '%0.s─' {1..60}; echo

pick_mode
pick_symbol

GATE="micro"
MONTHS=3
if [[ "$MODE" == "live" ]]; then
  pick_gate
fi

show_settings "$MODE" "$SYMBOL" "$GATE" "$MONTHS"

if confirm; then
  do_launch "$MODE" "$SYMBOL" "$GATE" "$MONTHS"
else
  echo "  Aborted."
  exit 0
fi
