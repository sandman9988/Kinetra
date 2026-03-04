#!/usr/bin/env bash
# Interactive launcher for backtest, paper, and live trading.
# Uses bash `select` menus — type the number and press Enter.
set -euo pipefail

ROOT="${ROOT:-$HOME/Projects/Kinetra}"
DATA_ROOT="${DATA_ROOT:-data/master_standardized/ctrader/pepperstone_demo_45841299}"
PROFILES_DIR="${PROFILES_DIR:-$ROOT/configs/trading_profiles}"

# ── helpers ──────────────────────────────────────────────────────────────────

bold()  { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
cyan()  { printf '\033[36m%s\033[0m' "$*"; }
yellow(){ printf '\033[33m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
dim()   { printf '\033[2m%s\033[0m' "$*"; }

hr() { printf '%0.s─' {1..60}; echo; }

# ── data availability check ──────────────────────────────────────────────────

check_data_availability() {
  # Returns the number of months of M1 data available for a symbol
  # Output: "months|start_date|end_date" or "0||" if no data
  local sym="$1"
  local data_dir="$ROOT/$DATA_ROOT/metals/$sym"

  if [[ ! -d "$data_dir" ]]; then
    # Try alternate paths
    for alt_dir in "$ROOT/data/master_standardized/ctrader/pepperstone/metals/$sym" \
                   "$ROOT/data/master_standardized/ctrader/pepperstone_demo"*/"metals/$sym"; do
      if [[ -d "$alt_dir" ]]; then
        data_dir="$alt_dir"
        break
      fi
    done
  fi

  # Find M1 file
  local m1_file
  m1_file=$(find "$data_dir" -name "*_M1_*.csv" 2>/dev/null | head -1)

  if [[ -z "$m1_file" || ! -f "$m1_file" ]]; then
    echo "0||"
    return
  fi

  # Get date range from CSV (first and last timestamp)
  # Assumes format: time,open,high,low,close,volume,spread
  local first_date last_date
  first_date=$(head -2 "$m1_file" | tail -1 | cut -d',' -f1)
  last_date=$(tail -1 "$m1_file" | cut -d',' -f1)

  # Calculate months (approximate)
  local start_epoch end_epoch days months
  if command -v date &>/dev/null; then
    # Try to parse dates (handle ISO format with or without timezone)
    start_epoch=$(date -d "${first_date%%+*}" +%s 2>/dev/null || date -d "${first_date%%T*}" +%s 2>/dev/null || echo "0")
    end_epoch=$(date -d "${last_date%%+*}" +%s 2>/dev/null || date -d "${last_date%%T*}" +%s 2>/dev/null || echo "0")

    if [[ "$start_epoch" != "0" && "$end_epoch" != "0" ]]; then
      days=$(( (end_epoch - start_epoch) / 86400 ))
      months=$(( days / 30 ))
      echo "${months}|${first_date%%T*}|${last_date%%T*}"
      return
    fi
  fi

  # Fallback: count lines and estimate (rough approximation)
  local lines
  lines=$(wc -l < "$m1_file")
  months=$(( lines / 43200 ))  # ~43200 M1 bars per month (30 days * 1440 min)
  echo "${months}|${first_date%%T*}|${last_date%%T*}"
}

prompt_auto_download() {
  # Ask user if they want to auto-download missing data
  local sym="$1"
  local requested_months="$2"
  local available_months="$3"
  local missing_months=$(( requested_months - available_months ))

  echo
  yellow "  ⚠️  Data insufficient: ${available_months} months available, ${requested_months} requested"
  echo
  printf "  Auto-download %s additional months? [Y/n] " "$missing_months"
  local ans
  read -r ans
  ans="${ans:-y}"
  [[ "$ans" =~ ^[Yy]$ ]]
}

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
DEFAULTS_STOP_LIVE[XAUUSD]="1.0";    DEFAULTS_STOP_LIVE[NAS100]="1.0"
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

prompt_with_default() {
  local label="$1"
  local current="$2"
  local out
  printf "  %-24s [%s]: " "$label" "$current" >&2
  read -r out
  if [[ -z "$out" ]]; then
    echo "$current"
  else
    echo "$out"
  fi
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN {exit !($1 > 0)}"
}

is_non_negative_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN {exit !($1 >= 0)}"
}

is_fraction_0_1() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN {exit !($1 > 0 && $1 < 1)}"
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_bool01() {
  [[ "$1" == "0" || "$1" == "1" ]]
}

is_hhmm_utc() {
  [[ "$1" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]]
}

is_valid_mode() {
  case "$1" in
    backtest|full|dry_run|live) return 0 ;;
    *) return 1 ;;
  esac
}

is_valid_gate() {
  case "$1" in
    micro|small|full) return 0 ;;
    *) return 1 ;;
  esac
}

is_valid_profile() {
  case "$1" in
    balanced|conservative|aggressive|bayesian|saved) return 0 ;;
    *) return 1 ;;
  esac
}

prompt_numeric() {
  local label="$1" current="$2" kind="${3:-pos_num}"
  local val
  while true; do
    val=$(prompt_with_default "$label" "$current")
    case "$kind" in
      pos_num) if is_positive_number "$val"; then echo "$val"; return; fi ;;
      nn_num)  if is_non_negative_number "$val"; then echo "$val"; return; fi ;;
      frac01)  if is_fraction_0_1 "$val"; then echo "$val"; return; fi ;;
      pos_int) if is_positive_int "$val"; then echo "$val"; return; fi ;;
      nn_int)  if is_non_negative_int "$val"; then echo "$val"; return; fi ;;
      bool01)  if is_bool01 "$val"; then echo "$val"; return; fi ;;
      hhmm)    if is_hhmm_utc "$val"; then echo "$val"; return; fi ;;
      *)       echo "$val"; return ;;
    esac
    echo "  Invalid value. Try again." >&2
  done
}

apply_profile_defaults() {
  local profile="$1"
  case "$profile" in
    conservative)
      BRICK_SIZE="1.5"
      STOP_BRICKS="1.5"
      TRAILING_MFE_FRACTION="0.50"
      TRAILING_MFE_AFTER_BRICKS="3"
      PE_ENTRY_THRESHOLD="0.70"
      PE_EXIT_THRESHOLD="0.85"
      STALE_BRICK_FACTOR="2.0"
      MARKOV_STALE_PENALTY="0.15"
      USE_OBI_SLIPPAGE_BUFFER="0"
      OBI_LEVELS="5"
      OBI_MAX_BUFFER_BRICKS="0.25"
      OBI_EMA_ALPHA="0.30"
      TARGET_RISK="50.0"
      FLIPRATE_THRESHOLD="0.25"
      MARKOV_THRESHOLD="0.65"
      MIN_OMEGA="2.0"
      MIN_TRADES="40"
      ;;
    aggressive)
      BRICK_SIZE="0.8"
      STOP_BRICKS="0.8"
      TRAILING_MFE_FRACTION="0.35"
      TRAILING_MFE_AFTER_BRICKS="2"
      PE_ENTRY_THRESHOLD="0.72"
      PE_EXIT_THRESHOLD="0.88"
      STALE_BRICK_FACTOR="1.8"
      MARKOV_STALE_PENALTY="0.12"
      USE_OBI_SLIPPAGE_BUFFER="0"
      OBI_LEVELS="5"
      OBI_MAX_BUFFER_BRICKS="0.20"
      OBI_EMA_ALPHA="0.35"
      TARGET_RISK="150.0"
      FLIPRATE_THRESHOLD="0.45"
      MARKOV_THRESHOLD="0.50"
      MIN_OMEGA="1.2"
      MIN_TRADES="20"
      ;;
    balanced|*)
      BRICK_SIZE=$(get_default BRICK "$SYMBOL")
      STOP_BRICKS=$(get_default STOP_LIVE "$SYMBOL")
      TARGET_RISK=$(get_default TARGET_RISK "$SYMBOL")
      TRAILING_MFE_FRACTION="0.50"
      TRAILING_MFE_AFTER_BRICKS="2"
      PE_ENTRY_THRESHOLD="0.70"
      PE_EXIT_THRESHOLD="0.85"
      STALE_BRICK_FACTOR="2.0"
      MARKOV_STALE_PENALTY="0.15"
      USE_OBI_SLIPPAGE_BUFFER="0"
      OBI_LEVELS="5"
      OBI_MAX_BUFFER_BRICKS="0.25"
      OBI_EMA_ALPHA="0.30"
      FLIPRATE_THRESHOLD="0.35"
      MARKOV_THRESHOLD="0.55"
      MIN_OMEGA="1.5"
      MIN_TRADES="30"
      ;;
  esac
}

run_bayesian_profile() {
  local opt_months
  if [[ "$MODE" == "backtest" || "$MODE" == "full" ]]; then
    opt_months="$MONTHS"
  else
    opt_months="$BAYES_MONTHS"
  fi

  echo
  yellow "  Running Bayesian profile optimization..."
  dim "  symbol=$SYMBOL months=$opt_months trials=$BAYES_TRIALS seed=$BAYES_SEED"
  echo

  local py="${ROOT}/scripts/ctrader/bayesian_profile.py"
  if [[ ! -f "$py" ]]; then
    echo "  Missing optimizer script: $py"
    exit 1
  fi

  local out
  if ! out=$(python "$py" \
    --symbol "$SYMBOL" \
    --months "$opt_months" \
    --trials "$BAYES_TRIALS" \
    --seed "$BAYES_SEED" \
    --base-brick "$BRICK_SIZE" \
    --base-stop "$STOP_BRICKS" \
    --base-risk "$TARGET_RISK" \
    --base-flip "$FLIPRATE_THRESHOLD" \
    --base-markov "$MARKOV_THRESHOLD" \
    --base-trail-frac "$TRAILING_MFE_FRACTION" \
    --base-trail-after "$TRAILING_MFE_AFTER_BRICKS" \
    --base-pe-entry "$PE_ENTRY_THRESHOLD" \
    --base-pe-exit "$PE_EXIT_THRESHOLD" \
    --base-stale-factor "$STALE_BRICK_FACTOR" \
    --base-stale-penalty "$MARKOV_STALE_PENALTY" \
    --format shell); then
    echo "  Bayesian optimization failed."
    exit 1
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      BRICK_SIZE) BRICK_SIZE="$value" ;;
      STOP_BRICKS) STOP_BRICKS="$value" ;;
      TARGET_RISK) TARGET_RISK="$value" ;;
      FLIPRATE_THRESHOLD) FLIPRATE_THRESHOLD="$value" ;;
      MARKOV_THRESHOLD) MARKOV_THRESHOLD="$value" ;;
      TRAILING_MFE_FRACTION) TRAILING_MFE_FRACTION="$value" ;;
      TRAILING_MFE_AFTER_BRICKS) TRAILING_MFE_AFTER_BRICKS="$value" ;;
      PE_ENTRY_THRESHOLD) PE_ENTRY_THRESHOLD="$value" ;;
      PE_EXIT_THRESHOLD) PE_EXIT_THRESHOLD="$value" ;;
      STALE_BRICK_FACTOR) STALE_BRICK_FACTOR="$value" ;;
      MARKOV_STALE_PENALTY) MARKOV_STALE_PENALTY="$value" ;;
      MIN_OMEGA) MIN_OMEGA="$value" ;;
      MIN_TRADES) MIN_TRADES="$value" ;;
      BAYES_BEST_SCORE) BAYES_BEST_SCORE="$value" ;;
      BAYES_BEST_OMEGA) BAYES_BEST_OMEGA="$value" ;;
      BAYES_BEST_TRADES) BAYES_BEST_TRADES="$value" ;;
    esac
  done <<< "$out"

  green "  Bayesian profile ready."
  echo
  dim "  score=$BAYES_BEST_SCORE omega=$BAYES_BEST_OMEGA trades=$BAYES_BEST_TRADES"
  echo
}

list_saved_profiles() {
  local sym="$1"
  local dir="$PROFILES_DIR/$sym"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name "*.env" -printf "%f\n" 2>/dev/null | sed 's/\.env$//' | sort -u
  fi
}

load_saved_profile() {
  local sym="$1"
  local name="$2"
  local fp="$PROFILES_DIR/$sym/$name.env"
  if [[ ! -f "$fp" ]]; then
    echo "  Saved profile not found: $fp"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$fp"
}

pick_saved_profile() {
  local -a names
  mapfile -t names < <(list_saved_profiles "$SYMBOL")
  if [[ ${#names[@]} -eq 0 ]]; then
    echo
    yellow "  No saved profiles for $SYMBOL in $PROFILES_DIR/$SYMBOL"
    echo "  Create one after a good backtest run."
    exit 1
  fi

  if [[ -n "${PRESET_PROFILE_NAME:-}" ]]; then
    SAVED_PROFILE_NAME="$PRESET_PROFILE_NAME"
    load_saved_profile "$SYMBOL" "$SAVED_PROFILE_NAME"
    return
  fi

  echo
  bold "  Select saved profile"; echo
  hr
  PS3="  Choice: "
  select N in "${names[@]}"; do
    if [[ -n "$N" ]]; then
      SAVED_PROFILE_NAME="$N"
      load_saved_profile "$SYMBOL" "$SAVED_PROFILE_NAME"
      break
    fi
    echo "  Invalid selection."
  done
}

save_profile_prompt() {
  # Save only from successful backtest/full runs after validation completed.
  if [[ "$MODE" != "backtest" && "$MODE" != "full" ]]; then
    return 0
  fi

  echo
  printf "  Save these parameters as reusable profile? [y/N] "
  local ans
  read -r ans
  ans="${ans:-n}"
  [[ "$ans" =~ ^[Yy]$ ]] || return 0

  local suggested
  suggested="$(echo "${SYMBOL,,}_${PROFILE}_$(date +%Y%m%d_%H%M%S)" | tr -cd 'a-z0-9_.-')"
  printf "  Profile name [%s]: " "$suggested"
  local name
  read -r name
  name="${name:-$suggested}"
  name="$(echo "$name" | tr -cd 'a-zA-Z0-9_.-')"
  if [[ -z "$name" ]]; then
    echo "  Invalid empty profile name."
    return 1
  fi

  local dir="$PROFILES_DIR/$SYMBOL"
  mkdir -p "$dir"
  local fp="$dir/$name.env"
  cat > "$fp" <<EOF
# Kinetra saved launcher profile
SYMBOL="$SYMBOL"
BRICK_SIZE="$BRICK_SIZE"
STOP_BRICKS="$STOP_BRICKS"
TRAILING_MFE_FRACTION="$TRAILING_MFE_FRACTION"
TRAILING_MFE_AFTER_BRICKS="$TRAILING_MFE_AFTER_BRICKS"
PE_ENTRY_THRESHOLD="$PE_ENTRY_THRESHOLD"
PE_EXIT_THRESHOLD="$PE_EXIT_THRESHOLD"
STALE_BRICK_FACTOR="$STALE_BRICK_FACTOR"
MARKOV_STALE_PENALTY="$MARKOV_STALE_PENALTY"
USE_OBI_SLIPPAGE_BUFFER="$USE_OBI_SLIPPAGE_BUFFER"
OBI_LEVELS="$OBI_LEVELS"
OBI_MAX_BUFFER_BRICKS="$OBI_MAX_BUFFER_BRICKS"
OBI_EMA_ALPHA="$OBI_EMA_ALPHA"
TARGET_RISK="$TARGET_RISK"
LOT_CEILING="$LOT_CEILING"
MAX_LEVERAGE="$MAX_LEVERAGE"
FLIPRATE_THRESHOLD="$FLIPRATE_THRESHOLD"
MARKOV_THRESHOLD="$MARKOV_THRESHOLD"
CONSERVATIVE_FILLS="$CONSERVATIVE_FILLS"
ENTRY_LATENCY_MS="$ENTRY_LATENCY_MS"
EXIT_LATENCY_MS="$EXIT_LATENCY_MS"
LATENCY_JITTER_MS="$LATENCY_JITTER_MS"
LATENCY_SEED="$LATENCY_SEED"
SPREAD_SIDES="$SPREAD_SIDES"
ENTRY_SLIP_BRICKS="$ENTRY_SLIP_BRICKS"
EXIT_SLIP_BRICKS="$EXIT_SLIP_BRICKS"
STOP_WORST_CASE_BRICKS="$STOP_WORST_CASE_BRICKS"
FLIP_TRADE_THROUGH_BRICKS="$FLIP_TRADE_THROUGH_BRICKS"
MIN_OMEGA="$MIN_OMEGA"
MIN_TRADES="$MIN_TRADES"
MONDAY_START="$MONDAY_START"
FRIDAY_END="$FRIDAY_END"
EOF
  green "  Saved profile: $fp"
  echo
}

pick_profile() {
  if [[ -n "${PRESET_PROFILE:-}" ]]; then
    if ! is_valid_profile "$PRESET_PROFILE"; then
      echo "  Invalid preset profile: $PRESET_PROFILE"
      exit 1
    fi
    PROFILE="$PRESET_PROFILE"
    if [[ "$PROFILE" == "bayesian" ]]; then
      apply_profile_defaults "balanced"
      run_bayesian_profile
    elif [[ "$PROFILE" == "saved" ]]; then
      apply_profile_defaults "balanced"
      pick_saved_profile
    else
      apply_profile_defaults "$PROFILE"
    fi
    return
  fi

  echo
  bold "  Select parameter profile"; echo
  hr
  printf "  %-12s %s\n" "balanced"     "Current defaults (recommended baseline)"
  printf "  %-12s %s\n" "conservative" "Wider bricks/stops, stricter filters, lower risk"
  printf "  %-12s %s\n" "aggressive"   "Tighter bricks/stops, looser filters, higher risk"
  printf "  %-12s %s\n" "bayesian"     "Data-driven optimization from sane defaults"
  printf "  %-12s %s\n" "saved"        "Load previously saved backtest profile"
  hr
  PS3="  Choice: "
  select PROFILE_OPT in "balanced" "conservative" "aggressive" "bayesian" "saved"; do
    case "$REPLY" in
      1) PROFILE="balanced"; break ;;
      2) PROFILE="conservative"; break ;;
      3) PROFILE="aggressive"; break ;;
      4) PROFILE="bayesian"; break ;;
      5) PROFILE="saved"; break ;;
      *) echo "  Enter 1, 2, 3, 4, or 5." ;;
    esac
  done
  if [[ "$PROFILE" == "bayesian" ]]; then
    apply_profile_defaults "balanced"
    run_bayesian_profile
  elif [[ "$PROFILE" == "saved" ]]; then
    apply_profile_defaults "balanced"
    pick_saved_profile
  else
    apply_profile_defaults "$PROFILE"
  fi
}

# ── main menu ────────────────────────────────────────────────────────────────

pick_mode() {
  if [[ -n "${PRESET_MODE:-}" ]]; then
    if ! is_valid_mode "$PRESET_MODE"; then
      echo "  Invalid preset mode: $PRESET_MODE"
      exit 1
    fi
    MODE="$PRESET_MODE"
    if [[ -n "${PRESET_MONTHS:-}" ]]; then
      MONTHS="$PRESET_MONTHS"
    elif [[ "$MODE" == "full" ]]; then
      MONTHS=24
    else
      MONTHS=3
    fi
    return
  fi

  echo
  bold "  Select mode"; echo
  hr
  printf "  %-4s %-10s %s\n" "#" "Mode" "Description"
  printf "  %-4s %-10s %s\n" "──" "────" "───────────────────────────────────────────"
  printf "  %-4s %-10s %s\n" "1" "backtest" "Quick 3-month historical validation"
  printf "  %-4s %-10s %s\n" "2" "full"     "Comprehensive 2-year rolling OOS"
  printf "  %-4s %-10s %s\n" "3" "dry-run"  "Live data, paper orders — same as live"
  printf "  %-4s %-10s %s\n" "4" "LIVE"     "REAL orders (requires confirmation)"
  hr
  PS3="  Choice: "
  select MODE_OPT in "backtest  — 3 months quick validation" \
                     "full      — 2 years rolling OOS backtest" \
                     "dry-run   — live data, paper orders" \
                     "LIVE      — REAL orders (requires ACK)"; do
    case "$REPLY" in
      1) MODE="backtest"; MONTHS=3;  break ;;
      2) MODE="full";     MONTHS=24; break ;;
      3) MODE="dry_run";  MONTHS=3;  break ;;
      4) MODE="live";     MONTHS=3;  break ;;
      *) echo "  Enter 1-4." ;;
    esac
  done
}

# ── gate menu (live only) ────────────────────────────────────────────────────

pick_gate() {
  if [[ -n "${PRESET_GATE:-}" ]]; then
    if ! is_valid_gate "$PRESET_GATE"; then
      echo "  Invalid preset gate: $PRESET_GATE"
      exit 1
    fi
    GATE="$PRESET_GATE"
    return
  fi

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

  if [[ -n "${PRESET_SYMBOL:-}" ]]; then
    SYMBOL="$PRESET_SYMBOL"
    return
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

  echo
  if [[ "$mode" == "backtest" || "$mode" == "full" ]]; then
    bold "  Settings — "; cyan "$sym"; printf "  "; dim "(mode: $mode, $months months)"; echo
  else
    bold "  Settings — "; yellow "$sym"; printf "  "; dim "(mode: $mode)"; echo
  fi
  hr

  if [[ "$mode" == "backtest" || "$mode" == "full" ]]; then
    printf "  %-24s %s\n" "mode"           "$(cyan "BACKTEST")"
    printf "  %-24s %s\n" "profile"        "$(bold "$PROFILE")"
    printf "  %-24s %s\n" "duration"       "$(bold "$months months")"
    printf "  %-24s %s\n" "brick_size"     "$(bold "$BRICK_SIZE")"
    printf "  %-24s %s\n" "stop_bricks"    "$(bold "$STOP_BRICKS")"
    printf "  %-24s %s\n" "initial_equity" "$(bold "\$1,000")"
    printf "  %-24s %s\n" "sizing"         "$(bold "static + compounding")"
    printf "  %-24s %s\n" "target_risk_usd" "$(bold "$TARGET_RISK")"
    printf "  %-24s %s\n" "lot_ceiling"    "$(bold "$LOT_CEILING")"
    printf "  %-24s %s\n" "max_leverage"   "$(bold "1:$MAX_LEVERAGE")"
    hr
    printf "  %-24s %s\n" "fliprate_threshold" "$(bold "$FLIPRATE_THRESHOLD")"
    printf "  %-24s %s\n" "markov_threshold"   "$(bold "$MARKOV_THRESHOLD")"
    printf "  %-24s %s\n" "min_omega"          "$(bold "≥ $MIN_OMEGA")"
    printf "  %-24s %s\n" "min_trades"         "$(bold "≥ $MIN_TRADES")"
    printf "  %-24s %s\n" "trail_mfe_fraction" "$(bold "$TRAILING_MFE_FRACTION")"
    printf "  %-24s %s\n" "trail_after_bricks" "$(bold "$TRAILING_MFE_AFTER_BRICKS")"
    printf "  %-24s %s\n" "pe_entry_threshold" "$(bold "$PE_ENTRY_THRESHOLD")"
    printf "  %-24s %s\n" "pe_exit_threshold" "$(bold "$PE_EXIT_THRESHOLD")"
    printf "  %-24s %s\n" "stale_brick_factor" "$(bold "$STALE_BRICK_FACTOR")"
    printf "  %-24s %s\n" "markov_stale_penalty" "$(bold "$MARKOV_STALE_PENALTY")"
    printf "  %-24s %s\n" "use_obi_slippage_buffer" "$(bold "$USE_OBI_SLIPPAGE_BUFFER")"
    printf "  %-24s %s\n" "obi_levels" "$(bold "$OBI_LEVELS")"
    printf "  %-24s %s\n" "obi_max_buffer_bricks" "$(bold "$OBI_MAX_BUFFER_BRICKS")"
    printf "  %-24s %s\n" "obi_ema_alpha" "$(bold "$OBI_EMA_ALPHA")"
    hr
    printf "  %-24s %s\n" "entry_latency_ms"   "$(bold "$ENTRY_LATENCY_MS")"
    printf "  %-24s %s\n" "exit_latency_ms"    "$(bold "$EXIT_LATENCY_MS")"
    printf "  %-24s %s\n" "latency_jitter_ms"  "$(bold "$LATENCY_JITTER_MS")"
    printf "  %-24s %s\n" "latency_seed"       "$(bold "$LATENCY_SEED")"
    printf "  %-24s %s\n" "latency_model"      "$(bold "backtest_only")"
    printf "  %-24s %s\n" "conservative_fills" "$(bold "$CONSERVATIVE_FILLS")"
    printf "  %-24s %s\n" "spread_sides"       "$(bold "$SPREAD_SIDES")"
    printf "  %-24s %s\n" "entry_slip_bricks"  "$(bold "$ENTRY_SLIP_BRICKS")"
    printf "  %-24s %s\n" "exit_slip_bricks"   "$(bold "$EXIT_SLIP_BRICKS")"
    printf "  %-24s %s\n" "stop_worst_case_bricks" "$(bold "$STOP_WORST_CASE_BRICKS")"
    printf "  %-24s %s\n" "flip_trade_through_bricks" "$(bold "$FLIP_TRADE_THROUGH_BRICKS")"
  else
    printf "  %-24s %s\n" "profile"         "$(bold "$PROFILE")"
    printf "  %-24s %s\n" "brick_size"      "$(bold "$BRICK_SIZE")"
    printf "  %-24s %s\n" "stop_bricks"     "$(bold "$STOP_BRICKS")"
    printf "  %-24s %s\n" "session UTC"     "$(bold "$MONDAY_START → $FRIDAY_END")"
    printf "  %-24s %s\n" "loss_brake_after" "$(bold "8")"
    printf "  %-24s %s\n" "loss_flat_after"  "$(bold "12")"
    printf "  %-24s %s\n" "loss_pause_min"   "$(bold "120")"
    printf "  %-24s %s\n" "fliprate_threshold" "$(bold "$FLIPRATE_THRESHOLD")"
    printf "  %-24s %s\n" "markov_threshold"   "$(bold "$MARKOV_THRESHOLD")"
    printf "  %-24s %s\n" "trail_mfe_fraction"  "$(bold "$TRAILING_MFE_FRACTION")"
    printf "  %-24s %s\n" "trail_after_bricks"  "$(bold "$TRAILING_MFE_AFTER_BRICKS")"
    printf "  %-24s %s\n" "pe_entry_threshold"   "$(bold "$PE_ENTRY_THRESHOLD")"
    printf "  %-24s %s\n" "pe_exit_threshold"    "$(bold "$PE_EXIT_THRESHOLD")"
    printf "  %-24s %s\n" "stale_brick_factor"   "$(bold "$STALE_BRICK_FACTOR")"
    printf "  %-24s %s\n" "markov_stale_penalty" "$(bold "$MARKOV_STALE_PENALTY")"
    printf "  %-24s %s\n" "use_obi_slippage_buffer" "$(bold "$USE_OBI_SLIPPAGE_BUFFER")"
    printf "  %-24s %s\n" "obi_levels" "$(bold "$OBI_LEVELS")"
    printf "  %-24s %s\n" "obi_max_buffer_bricks" "$(bold "$OBI_MAX_BUFFER_BRICKS")"
    printf "  %-24s %s\n" "obi_ema_alpha" "$(bold "$OBI_EMA_ALPHA")"
    printf "  %-24s %s\n" "max_leverage"       "$(bold "1:$MAX_LEVERAGE")"
    printf "  %-24s %s\n" "latency_model"       "$(bold "backtest_only (broker/live execution timing in non-backtest modes)")"

    local dd tr
    dd=$(get_default DD_HALT_LIVE "$sym")
    tr=$(get_default TARGET_RISK "$sym")
    printf "  %-24s %s\n" "drawdown_halt_pct" "$(bold "$dd")"
    printf "  %-24s %s\n" "target_risk_usd"   "$(bold "$tr")"
    printf "  %-24s %s\n" "trailing_mae"      "$(bold "on  (after 2 bars, 0.5 frac)")"
    printf "  %-24s %s\n" "preflight"   "$(bold "yes")"
    if [[ "$mode" == "live" ]]; then
      printf "  %-24s %s\n" "order_gate"  "$(red "REAL — $gate gate")"
    else
      printf "  %-24s %s\n" "order_gate"  "$(yellow "paper dispatcher (dry-run, gate=$gate)")"
    fi
  fi
  hr
}

edit_parameters() {
  echo
  printf "  Edit parameters? [y/N] "
  local ans
  read -r ans
  ans="${ans:-n}"
  [[ "$ans" =~ ^[Yy]$ ]] || return 0

  echo
  bold "  Parameter Overrides"; echo
  hr

  if [[ "$MODE" == "backtest" || "$MODE" == "full" ]]; then
    MONTHS=$(prompt_numeric "months" "$MONTHS" pos_int)
  fi

  BRICK_SIZE=$(prompt_numeric "brick_size" "$BRICK_SIZE" pos_num)
  STOP_BRICKS=$(prompt_numeric "stop_bricks" "$STOP_BRICKS" pos_num)
  TARGET_RISK=$(prompt_numeric "target_risk_usd" "$TARGET_RISK" pos_num)
  LOT_CEILING=$(prompt_numeric "lot_ceiling" "$LOT_CEILING" pos_num)
  MAX_LEVERAGE=$(prompt_numeric "max_leverage" "$MAX_LEVERAGE" pos_num)
  FLIPRATE_THRESHOLD=$(prompt_numeric "fliprate_threshold" "$FLIPRATE_THRESHOLD" frac01)
  MARKOV_THRESHOLD=$(prompt_numeric "markov_threshold" "$MARKOV_THRESHOLD" frac01)
  MIN_OMEGA=$(prompt_numeric "min_omega" "$MIN_OMEGA" pos_num)
  MIN_TRADES=$(prompt_numeric "min_trades" "$MIN_TRADES" pos_int)
  TRAILING_MFE_FRACTION=$(prompt_numeric "trail_mfe_fraction" "$TRAILING_MFE_FRACTION" frac01)
  TRAILING_MFE_AFTER_BRICKS=$(prompt_numeric "trail_after_bricks" "$TRAILING_MFE_AFTER_BRICKS" nn_int)
  PE_ENTRY_THRESHOLD=$(prompt_numeric "pe_entry_threshold" "$PE_ENTRY_THRESHOLD" frac01)
  PE_EXIT_THRESHOLD=$(prompt_numeric "pe_exit_threshold" "$PE_EXIT_THRESHOLD" frac01)
  STALE_BRICK_FACTOR=$(prompt_numeric "stale_brick_factor" "$STALE_BRICK_FACTOR" pos_num)
  MARKOV_STALE_PENALTY=$(prompt_numeric "markov_stale_penalty" "$MARKOV_STALE_PENALTY" nn_num)
  USE_OBI_SLIPPAGE_BUFFER=$(prompt_numeric "use_obi_slippage_buffer (1/0)" "$USE_OBI_SLIPPAGE_BUFFER" bool01)
  OBI_LEVELS=$(prompt_numeric "obi_levels" "$OBI_LEVELS" pos_int)
  OBI_MAX_BUFFER_BRICKS=$(prompt_numeric "obi_max_buffer_bricks" "$OBI_MAX_BUFFER_BRICKS" nn_num)
  OBI_EMA_ALPHA=$(prompt_numeric "obi_ema_alpha" "$OBI_EMA_ALPHA" frac01)

  if [[ "$MODE" == "backtest" || "$MODE" == "full" ]]; then
    ENTRY_LATENCY_MS=$(prompt_numeric "entry_latency_ms" "$ENTRY_LATENCY_MS" nn_int)
    EXIT_LATENCY_MS=$(prompt_numeric "exit_latency_ms" "$EXIT_LATENCY_MS" nn_int)
    LATENCY_JITTER_MS=$(prompt_numeric "latency_jitter_ms" "$LATENCY_JITTER_MS" nn_int)
    LATENCY_SEED=$(prompt_numeric "latency_seed" "$LATENCY_SEED" nn_int)
    CONSERVATIVE_FILLS=$(prompt_numeric "conservative_fills (1/0)" "$CONSERVATIVE_FILLS" bool01)
    SPREAD_SIDES=$(prompt_numeric "spread_sides" "$SPREAD_SIDES" pos_num)
    ENTRY_SLIP_BRICKS=$(prompt_numeric "entry_slip_bricks" "$ENTRY_SLIP_BRICKS" nn_num)
    EXIT_SLIP_BRICKS=$(prompt_numeric "exit_slip_bricks" "$EXIT_SLIP_BRICKS" nn_num)
    STOP_WORST_CASE_BRICKS=$(prompt_numeric "stop_worst_case_bricks" "$STOP_WORST_CASE_BRICKS" nn_num)
    FLIP_TRADE_THROUGH_BRICKS=$(prompt_numeric "flip_trade_through_bricks" "$FLIP_TRADE_THROUGH_BRICKS" nn_num)
  fi

  if [[ "$MODE" == "dry_run" || "$MODE" == "live" ]]; then
    MONDAY_START=$(prompt_numeric "session_start_utc" "$MONDAY_START" hhmm)
    FRIDAY_END=$(prompt_numeric "session_end_utc" "$FRIDAY_END" hhmm)
  fi
}

apply_env_overrides() {
  [[ -n "${OVERRIDE_MONTHS:-}" ]] && MONTHS="$OVERRIDE_MONTHS"
  [[ -n "${OVERRIDE_BRICK_SIZE:-}" ]] && BRICK_SIZE="$OVERRIDE_BRICK_SIZE"
  [[ -n "${OVERRIDE_STOP_BRICKS:-}" ]] && STOP_BRICKS="$OVERRIDE_STOP_BRICKS"
  [[ -n "${OVERRIDE_TARGET_RISK:-}" ]] && TARGET_RISK="$OVERRIDE_TARGET_RISK"
  [[ -n "${OVERRIDE_LOT_CEILING:-}" ]] && LOT_CEILING="$OVERRIDE_LOT_CEILING"
  [[ -n "${OVERRIDE_MAX_LEVERAGE:-}" ]] && MAX_LEVERAGE="$OVERRIDE_MAX_LEVERAGE"
  [[ -n "${OVERRIDE_FLIPRATE_THRESHOLD:-}" ]] && FLIPRATE_THRESHOLD="$OVERRIDE_FLIPRATE_THRESHOLD"
  [[ -n "${OVERRIDE_MARKOV_THRESHOLD:-}" ]] && MARKOV_THRESHOLD="$OVERRIDE_MARKOV_THRESHOLD"
  [[ -n "${OVERRIDE_MIN_OMEGA:-}" ]] && MIN_OMEGA="$OVERRIDE_MIN_OMEGA"
  [[ -n "${OVERRIDE_MIN_TRADES:-}" ]] && MIN_TRADES="$OVERRIDE_MIN_TRADES"
  [[ -n "${OVERRIDE_TRAILING_MFE_FRACTION:-}" ]] && TRAILING_MFE_FRACTION="$OVERRIDE_TRAILING_MFE_FRACTION"
  [[ -n "${OVERRIDE_TRAILING_MFE_AFTER_BRICKS:-}" ]] && TRAILING_MFE_AFTER_BRICKS="$OVERRIDE_TRAILING_MFE_AFTER_BRICKS"
  [[ -n "${OVERRIDE_PE_ENTRY_THRESHOLD:-}" ]] && PE_ENTRY_THRESHOLD="$OVERRIDE_PE_ENTRY_THRESHOLD"
  [[ -n "${OVERRIDE_PE_EXIT_THRESHOLD:-}" ]] && PE_EXIT_THRESHOLD="$OVERRIDE_PE_EXIT_THRESHOLD"
  [[ -n "${OVERRIDE_STALE_BRICK_FACTOR:-}" ]] && STALE_BRICK_FACTOR="$OVERRIDE_STALE_BRICK_FACTOR"
  [[ -n "${OVERRIDE_MARKOV_STALE_PENALTY:-}" ]] && MARKOV_STALE_PENALTY="$OVERRIDE_MARKOV_STALE_PENALTY"
  [[ -n "${OVERRIDE_USE_OBI_SLIPPAGE_BUFFER:-}" ]] && USE_OBI_SLIPPAGE_BUFFER="$OVERRIDE_USE_OBI_SLIPPAGE_BUFFER"
  [[ -n "${OVERRIDE_OBI_LEVELS:-}" ]] && OBI_LEVELS="$OVERRIDE_OBI_LEVELS"
  [[ -n "${OVERRIDE_OBI_MAX_BUFFER_BRICKS:-}" ]] && OBI_MAX_BUFFER_BRICKS="$OVERRIDE_OBI_MAX_BUFFER_BRICKS"
  [[ -n "${OVERRIDE_OBI_EMA_ALPHA:-}" ]] && OBI_EMA_ALPHA="$OVERRIDE_OBI_EMA_ALPHA"
  [[ -n "${OVERRIDE_ENTRY_LATENCY_MS:-}" ]] && ENTRY_LATENCY_MS="$OVERRIDE_ENTRY_LATENCY_MS"
  [[ -n "${OVERRIDE_EXIT_LATENCY_MS:-}" ]] && EXIT_LATENCY_MS="$OVERRIDE_EXIT_LATENCY_MS"
  [[ -n "${OVERRIDE_LATENCY_JITTER_MS:-}" ]] && LATENCY_JITTER_MS="$OVERRIDE_LATENCY_JITTER_MS"
  [[ -n "${OVERRIDE_LATENCY_SEED:-}" ]] && LATENCY_SEED="$OVERRIDE_LATENCY_SEED"
  [[ -n "${OVERRIDE_CONSERVATIVE_FILLS:-}" ]] && CONSERVATIVE_FILLS="$OVERRIDE_CONSERVATIVE_FILLS"
  [[ -n "${OVERRIDE_SPREAD_SIDES:-}" ]] && SPREAD_SIDES="$OVERRIDE_SPREAD_SIDES"
  [[ -n "${OVERRIDE_ENTRY_SLIP_BRICKS:-}" ]] && ENTRY_SLIP_BRICKS="$OVERRIDE_ENTRY_SLIP_BRICKS"
  [[ -n "${OVERRIDE_EXIT_SLIP_BRICKS:-}" ]] && EXIT_SLIP_BRICKS="$OVERRIDE_EXIT_SLIP_BRICKS"
  [[ -n "${OVERRIDE_STOP_WORST_CASE_BRICKS:-}" ]] && STOP_WORST_CASE_BRICKS="$OVERRIDE_STOP_WORST_CASE_BRICKS"
  [[ -n "${OVERRIDE_FLIP_TRADE_THROUGH_BRICKS:-}" ]] && FLIP_TRADE_THROUGH_BRICKS="$OVERRIDE_FLIP_TRADE_THROUGH_BRICKS"
  [[ -n "${OVERRIDE_MONDAY_START:-}" ]] && MONDAY_START="$OVERRIDE_MONDAY_START"
  [[ -n "${OVERRIDE_FRIDAY_END:-}" ]] && FRIDAY_END="$OVERRIDE_FRIDAY_END"
  [[ -n "${OVERRIDE_AUTO_DOWNLOAD:-}" ]] && AUTO_DOWNLOAD="$OVERRIDE_AUTO_DOWNLOAD"
  return 0
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
  local mode="$1" sym="$2" gate="${3:-micro}" months="${4:-3}" auto_dl="${5:-}"

  # Resolve absolute path to renko_engine.py regardless of cwd
  local engine
  engine="$(cd "$(dirname "$0")/../.." && pwd)/scripts/renko_engine.py"

  # Build common strategy options
  local common_opts=""
  common_opts="$common_opts --brick-size $BRICK_SIZE"
  common_opts="$common_opts --stop-bricks $STOP_BRICKS"
  common_opts="$common_opts --target-risk $TARGET_RISK"
  common_opts="$common_opts --lot-ceiling $LOT_CEILING"
  common_opts="$common_opts --max-leverage $MAX_LEVERAGE"
  common_opts="$common_opts --fliprate-threshold $FLIPRATE_THRESHOLD"
  common_opts="$common_opts --markov-threshold $MARKOV_THRESHOLD"
  common_opts="$common_opts --min-omega $MIN_OMEGA --min-trades $MIN_TRADES"
  common_opts="$common_opts --trailing-mfe-fraction $TRAILING_MFE_FRACTION"
  common_opts="$common_opts --trailing-mfe-after-bricks $TRAILING_MFE_AFTER_BRICKS"
  common_opts="$common_opts --pe-entry-threshold $PE_ENTRY_THRESHOLD --pe-exit-threshold $PE_EXIT_THRESHOLD"
  common_opts="$common_opts --stale-brick-factor $STALE_BRICK_FACTOR --markov-stale-penalty $MARKOV_STALE_PENALTY"
  if [[ "$USE_OBI_SLIPPAGE_BUFFER" == "1" ]]; then
    common_opts="$common_opts --use-obi-slippage-buffer"
  else
    common_opts="$common_opts --no-use-obi-slippage-buffer"
  fi
  common_opts="$common_opts --obi-levels $OBI_LEVELS --obi-max-buffer-bricks $OBI_MAX_BUFFER_BRICKS --obi-ema-alpha $OBI_EMA_ALPHA"
  if [[ "$CONSERVATIVE_FILLS" == "1" ]]; then
    common_opts="$common_opts --conservative-fills"
  else
    common_opts="$common_opts --no-conservative-fills"
  fi
  common_opts="$common_opts --spread-sides $SPREAD_SIDES"
  common_opts="$common_opts --entry-slip-bricks $ENTRY_SLIP_BRICKS"
  common_opts="$common_opts --exit-slip-bricks $EXIT_SLIP_BRICKS"
  common_opts="$common_opts --stop-worst-case-bricks $STOP_WORST_CASE_BRICKS"
  common_opts="$common_opts --flip-trade-through-bricks $FLIP_TRADE_THROUGH_BRICKS"
  local backtest_latency_opts=""
  backtest_latency_opts="$backtest_latency_opts --entry-latency-ms $ENTRY_LATENCY_MS --exit-latency-ms $EXIT_LATENCY_MS"
  backtest_latency_opts="$backtest_latency_opts --latency-jitter-ms $LATENCY_JITTER_MS --latency-seed $LATENCY_SEED"

  # Build live trading options
  local live_opts=""
  live_opts="--live-size $gate"
  live_opts="$live_opts --monday-start $MONDAY_START --friday-end $FRIDAY_END"

  if [[ "$mode" == "backtest" ]]; then
    echo
    echo "  $(cyan '[LAUNCH]') backtest — $sym ($months months)"
    python "$engine" "$sym" --stage backtest --months "$months" $common_opts $backtest_latency_opts $auto_dl
    local code=$?
    if [[ $code -eq 0 ]]; then
      save_profile_prompt
    fi
    return $code

  elif [[ "$mode" == "full" ]]; then
    echo
    echo "  $(cyan '[LAUNCH]') full backtest — $sym ($months months)"
    python "$engine" "$sym" --stage full --months "$months" $common_opts $backtest_latency_opts $auto_dl
    local code=$?
    if [[ $code -eq 0 ]]; then
      save_profile_prompt
    fi
    return $code

  elif [[ "$mode" == "dry_run" ]]; then
    echo
    echo "  $(yellow '[LAUNCH]') dry-run — $sym  (live bars, paper orders)"
    _run_with_reconnect python "$engine" "$sym" --stage live $common_opts $live_opts --dry-run

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
    echo "  Running preflight checks before trading..."
    _run_with_reconnect python "$engine" "$sym" --stage live $common_opts $live_opts
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

# Optional presets/overrides (passed via `make launch ...` env vars)
PRESET_MODE="${LAUNCH_MODE:-${MODE:-}}"
PRESET_SYMBOL="${LAUNCH_SYMBOL:-${SYMBOL:-}}"
PRESET_PROFILE="${LAUNCH_PROFILE:-${PROFILE:-}}"
PRESET_PROFILE_NAME="${LAUNCH_PROFILE_NAME:-${PROFILE_NAME:-}}"
PRESET_GATE="${LAUNCH_GATE:-${GATE:-}}"
PRESET_MONTHS=""
if [[ -n "${LAUNCH_MONTHS:-}" ]]; then
  # When launched via Makefile, only treat MONTHS as explicit if it did not come
  # from the makefile default (origin=file).
  if [[ -z "${LAUNCH_MONTHS_SET:-}" || "${LAUNCH_MONTHS_SET}" != "file" ]]; then
    PRESET_MONTHS="${LAUNCH_MONTHS}"
  fi
elif [[ -n "${MONTHS:-}" && -z "${LAUNCH_MONTHS_SET:-}" ]]; then
  PRESET_MONTHS="${MONTHS}"
fi

if [[ -n "$PRESET_MODE" ]]; then
  PRESET_MODE="${PRESET_MODE,,}"
  PRESET_MODE="${PRESET_MODE//-/_}"
fi
if [[ -n "$PRESET_PROFILE" ]]; then
  PRESET_PROFILE="${PRESET_PROFILE,,}"
fi
if [[ -n "$PRESET_GATE" ]]; then
  PRESET_GATE="${PRESET_GATE,,}"
fi

OVERRIDE_MONTHS="$PRESET_MONTHS"
OVERRIDE_BRICK_SIZE="${LAUNCH_BRICK_SIZE:-${BRICK_SIZE:-}}"
OVERRIDE_STOP_BRICKS="${LAUNCH_STOP_BRICKS:-${STOP_BRICKS:-}}"
OVERRIDE_TARGET_RISK="${LAUNCH_TARGET_RISK:-${TARGET_RISK:-}}"
OVERRIDE_LOT_CEILING="${LAUNCH_LOT_CEILING:-${LOT_CEILING:-}}"
OVERRIDE_MAX_LEVERAGE="${LAUNCH_MAX_LEVERAGE:-${MAX_LEVERAGE:-}}"
OVERRIDE_FLIPRATE_THRESHOLD="${LAUNCH_FLIPRATE_THRESHOLD:-${FLIPRATE_THRESHOLD:-}}"
OVERRIDE_MARKOV_THRESHOLD="${LAUNCH_MARKOV_THRESHOLD:-${MARKOV_THRESHOLD:-}}"
OVERRIDE_TRAILING_MFE_FRACTION="${LAUNCH_TRAILING_MFE_FRACTION:-${TRAILING_MFE_FRACTION:-}}"
OVERRIDE_TRAILING_MFE_AFTER_BRICKS="${LAUNCH_TRAILING_MFE_AFTER_BRICKS:-${TRAILING_MFE_AFTER_BRICKS:-}}"
OVERRIDE_PE_ENTRY_THRESHOLD="${LAUNCH_PE_ENTRY_THRESHOLD:-${PE_ENTRY_THRESHOLD:-}}"
OVERRIDE_PE_EXIT_THRESHOLD="${LAUNCH_PE_EXIT_THRESHOLD:-${PE_EXIT_THRESHOLD:-}}"
OVERRIDE_STALE_BRICK_FACTOR="${LAUNCH_STALE_BRICK_FACTOR:-${STALE_BRICK_FACTOR:-}}"
OVERRIDE_MARKOV_STALE_PENALTY="${LAUNCH_MARKOV_STALE_PENALTY:-${MARKOV_STALE_PENALTY:-}}"
OVERRIDE_USE_OBI_SLIPPAGE_BUFFER="${LAUNCH_USE_OBI_SLIPPAGE_BUFFER:-${USE_OBI_SLIPPAGE_BUFFER:-}}"
OVERRIDE_OBI_LEVELS="${LAUNCH_OBI_LEVELS:-${OBI_LEVELS:-}}"
OVERRIDE_OBI_MAX_BUFFER_BRICKS="${LAUNCH_OBI_MAX_BUFFER_BRICKS:-${OBI_MAX_BUFFER_BRICKS:-}}"
OVERRIDE_OBI_EMA_ALPHA="${LAUNCH_OBI_EMA_ALPHA:-${OBI_EMA_ALPHA:-}}"
OVERRIDE_MIN_OMEGA="${LAUNCH_MIN_OMEGA:-${MIN_OMEGA:-}}"
OVERRIDE_MIN_TRADES="${LAUNCH_MIN_TRADES:-${MIN_TRADES:-}}"
OVERRIDE_ENTRY_LATENCY_MS="${LAUNCH_ENTRY_LATENCY_MS:-${ENTRY_LATENCY_MS:-}}"
OVERRIDE_EXIT_LATENCY_MS="${LAUNCH_EXIT_LATENCY_MS:-${EXIT_LATENCY_MS:-}}"
OVERRIDE_LATENCY_JITTER_MS="${LAUNCH_LATENCY_JITTER_MS:-${LATENCY_JITTER_MS:-}}"
OVERRIDE_LATENCY_SEED="${LAUNCH_LATENCY_SEED:-${LATENCY_SEED:-}}"
OVERRIDE_CONSERVATIVE_FILLS="${LAUNCH_CONSERVATIVE_FILLS:-${CONSERVATIVE_FILLS:-}}"
OVERRIDE_SPREAD_SIDES="${LAUNCH_SPREAD_SIDES:-${SPREAD_SIDES:-}}"
OVERRIDE_ENTRY_SLIP_BRICKS="${LAUNCH_ENTRY_SLIP_BRICKS:-${ENTRY_SLIP_BRICKS:-}}"
OVERRIDE_EXIT_SLIP_BRICKS="${LAUNCH_EXIT_SLIP_BRICKS:-${EXIT_SLIP_BRICKS:-}}"
OVERRIDE_STOP_WORST_CASE_BRICKS="${LAUNCH_STOP_WORST_CASE_BRICKS:-${STOP_WORST_CASE_BRICKS:-}}"
OVERRIDE_FLIP_TRADE_THROUGH_BRICKS="${LAUNCH_FLIP_TRADE_THROUGH_BRICKS:-${FLIP_TRADE_THROUGH_BRICKS:-}}"
OVERRIDE_MONDAY_START="${LAUNCH_MONDAY_START:-${MONDAY_START:-}}"
OVERRIDE_FRIDAY_END="${LAUNCH_FRIDAY_END:-${FRIDAY_END:-}}"
OVERRIDE_AUTO_DOWNLOAD="${LAUNCH_AUTO_DOWNLOAD:-${AUTO_DOWNLOAD:-}}"
BAYES_TRIALS="${LAUNCH_BAYES_TRIALS:-20}"
BAYES_SEED="${LAUNCH_BAYES_SEED:-42}"
BAYES_MONTHS="${LAUNCH_BAYES_MONTHS:-3}"

if [[ -n "$PRESET_MONTHS" ]] && ! is_positive_int "$PRESET_MONTHS"; then
  echo "  Invalid preset months: $PRESET_MONTHS"
  exit 1
fi
if ! is_positive_int "$BAYES_TRIALS"; then
  echo "  Invalid bayes trials: $BAYES_TRIALS"
  exit 1
fi
if ! is_non_negative_int "$BAYES_SEED"; then
  echo "  Invalid bayes seed: $BAYES_SEED"
  exit 1
fi
if ! is_positive_int "$BAYES_MONTHS"; then
  echo "  Invalid bayes months: $BAYES_MONTHS"
  exit 1
fi
if [[ -n "$OVERRIDE_BRICK_SIZE" ]] && ! is_positive_number "$OVERRIDE_BRICK_SIZE"; then
  echo "  Invalid override brick_size: $OVERRIDE_BRICK_SIZE"
  exit 1
fi
if [[ -n "$OVERRIDE_STOP_BRICKS" ]] && ! is_positive_number "$OVERRIDE_STOP_BRICKS"; then
  echo "  Invalid override stop_bricks: $OVERRIDE_STOP_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_TARGET_RISK" ]] && ! is_positive_number "$OVERRIDE_TARGET_RISK"; then
  echo "  Invalid override target_risk: $OVERRIDE_TARGET_RISK"
  exit 1
fi
if [[ -n "$OVERRIDE_LOT_CEILING" ]] && ! is_positive_number "$OVERRIDE_LOT_CEILING"; then
  echo "  Invalid override lot_ceiling: $OVERRIDE_LOT_CEILING"
  exit 1
fi
if [[ -n "$OVERRIDE_MAX_LEVERAGE" ]] && ! is_positive_number "$OVERRIDE_MAX_LEVERAGE"; then
  echo "  Invalid override max_leverage: $OVERRIDE_MAX_LEVERAGE"
  exit 1
fi
if [[ -n "$OVERRIDE_FLIPRATE_THRESHOLD" ]] && ! is_fraction_0_1 "$OVERRIDE_FLIPRATE_THRESHOLD"; then
  echo "  Invalid override fliprate_threshold: $OVERRIDE_FLIPRATE_THRESHOLD"
  exit 1
fi
if [[ -n "$OVERRIDE_MARKOV_THRESHOLD" ]] && ! is_fraction_0_1 "$OVERRIDE_MARKOV_THRESHOLD"; then
  echo "  Invalid override markov_threshold: $OVERRIDE_MARKOV_THRESHOLD"
  exit 1
fi
if [[ -n "$OVERRIDE_TRAILING_MFE_FRACTION" ]] && ! is_fraction_0_1 "$OVERRIDE_TRAILING_MFE_FRACTION"; then
  echo "  Invalid override trailing_mfe_fraction: $OVERRIDE_TRAILING_MFE_FRACTION"
  exit 1
fi
if [[ -n "$OVERRIDE_TRAILING_MFE_AFTER_BRICKS" ]] && ! is_non_negative_int "$OVERRIDE_TRAILING_MFE_AFTER_BRICKS"; then
  echo "  Invalid override trailing_mfe_after_bricks: $OVERRIDE_TRAILING_MFE_AFTER_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_PE_ENTRY_THRESHOLD" ]] && ! is_fraction_0_1 "$OVERRIDE_PE_ENTRY_THRESHOLD"; then
  echo "  Invalid override pe_entry_threshold: $OVERRIDE_PE_ENTRY_THRESHOLD"
  exit 1
fi
if [[ -n "$OVERRIDE_PE_EXIT_THRESHOLD" ]] && ! is_fraction_0_1 "$OVERRIDE_PE_EXIT_THRESHOLD"; then
  echo "  Invalid override pe_exit_threshold: $OVERRIDE_PE_EXIT_THRESHOLD"
  exit 1
fi
if [[ -n "$OVERRIDE_STALE_BRICK_FACTOR" ]] && ! is_positive_number "$OVERRIDE_STALE_BRICK_FACTOR"; then
  echo "  Invalid override stale_brick_factor: $OVERRIDE_STALE_BRICK_FACTOR"
  exit 1
fi
if [[ -n "$OVERRIDE_MARKOV_STALE_PENALTY" ]] && ! is_non_negative_number "$OVERRIDE_MARKOV_STALE_PENALTY"; then
  echo "  Invalid override markov_stale_penalty: $OVERRIDE_MARKOV_STALE_PENALTY"
  exit 1
fi
if [[ -n "$OVERRIDE_USE_OBI_SLIPPAGE_BUFFER" ]] && ! is_bool01 "$OVERRIDE_USE_OBI_SLIPPAGE_BUFFER"; then
  echo "  Invalid override use_obi_slippage_buffer: $OVERRIDE_USE_OBI_SLIPPAGE_BUFFER"
  exit 1
fi
if [[ -n "$OVERRIDE_OBI_LEVELS" ]] && ! is_positive_int "$OVERRIDE_OBI_LEVELS"; then
  echo "  Invalid override obi_levels: $OVERRIDE_OBI_LEVELS"
  exit 1
fi
if [[ -n "$OVERRIDE_OBI_MAX_BUFFER_BRICKS" ]] && ! is_non_negative_number "$OVERRIDE_OBI_MAX_BUFFER_BRICKS"; then
  echo "  Invalid override obi_max_buffer_bricks: $OVERRIDE_OBI_MAX_BUFFER_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_OBI_EMA_ALPHA" ]] && ! is_fraction_0_1 "$OVERRIDE_OBI_EMA_ALPHA"; then
  echo "  Invalid override obi_ema_alpha: $OVERRIDE_OBI_EMA_ALPHA"
  exit 1
fi
if [[ -n "$OVERRIDE_MIN_OMEGA" ]] && ! is_positive_number "$OVERRIDE_MIN_OMEGA"; then
  echo "  Invalid override min_omega: $OVERRIDE_MIN_OMEGA"
  exit 1
fi
if [[ -n "$OVERRIDE_MIN_TRADES" ]] && ! is_positive_int "$OVERRIDE_MIN_TRADES"; then
  echo "  Invalid override min_trades: $OVERRIDE_MIN_TRADES"
  exit 1
fi
if [[ -n "$OVERRIDE_ENTRY_LATENCY_MS" ]] && ! is_non_negative_int "$OVERRIDE_ENTRY_LATENCY_MS"; then
  echo "  Invalid override entry_latency_ms: $OVERRIDE_ENTRY_LATENCY_MS"
  exit 1
fi
if [[ -n "$OVERRIDE_EXIT_LATENCY_MS" ]] && ! is_non_negative_int "$OVERRIDE_EXIT_LATENCY_MS"; then
  echo "  Invalid override exit_latency_ms: $OVERRIDE_EXIT_LATENCY_MS"
  exit 1
fi
if [[ -n "$OVERRIDE_LATENCY_JITTER_MS" ]] && ! is_non_negative_int "$OVERRIDE_LATENCY_JITTER_MS"; then
  echo "  Invalid override latency_jitter_ms: $OVERRIDE_LATENCY_JITTER_MS"
  exit 1
fi
if [[ -n "$OVERRIDE_LATENCY_SEED" ]] && ! is_non_negative_int "$OVERRIDE_LATENCY_SEED"; then
  echo "  Invalid override latency_seed: $OVERRIDE_LATENCY_SEED"
  exit 1
fi
if [[ -n "$OVERRIDE_CONSERVATIVE_FILLS" ]] && ! is_bool01 "$OVERRIDE_CONSERVATIVE_FILLS"; then
  echo "  Invalid override conservative_fills: $OVERRIDE_CONSERVATIVE_FILLS"
  exit 1
fi
if [[ -n "$OVERRIDE_SPREAD_SIDES" ]] && ! is_positive_number "$OVERRIDE_SPREAD_SIDES"; then
  echo "  Invalid override spread_sides: $OVERRIDE_SPREAD_SIDES"
  exit 1
fi
if [[ -n "$OVERRIDE_ENTRY_SLIP_BRICKS" ]] && ! is_non_negative_number "$OVERRIDE_ENTRY_SLIP_BRICKS"; then
  echo "  Invalid override entry_slip_bricks: $OVERRIDE_ENTRY_SLIP_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_EXIT_SLIP_BRICKS" ]] && ! is_non_negative_number "$OVERRIDE_EXIT_SLIP_BRICKS"; then
  echo "  Invalid override exit_slip_bricks: $OVERRIDE_EXIT_SLIP_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_STOP_WORST_CASE_BRICKS" ]] && ! is_non_negative_number "$OVERRIDE_STOP_WORST_CASE_BRICKS"; then
  echo "  Invalid override stop_worst_case_bricks: $OVERRIDE_STOP_WORST_CASE_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_FLIP_TRADE_THROUGH_BRICKS" ]] && ! is_non_negative_number "$OVERRIDE_FLIP_TRADE_THROUGH_BRICKS"; then
  echo "  Invalid override flip_trade_through_bricks: $OVERRIDE_FLIP_TRADE_THROUGH_BRICKS"
  exit 1
fi
if [[ -n "$OVERRIDE_MONDAY_START" ]] && ! is_hhmm_utc "$OVERRIDE_MONDAY_START"; then
  echo "  Invalid override monday_start: $OVERRIDE_MONDAY_START"
  exit 1
fi
if [[ -n "$OVERRIDE_FRIDAY_END" ]] && ! is_hhmm_utc "$OVERRIDE_FRIDAY_END"; then
  echo "  Invalid override friday_end: $OVERRIDE_FRIDAY_END"
  exit 1
fi
if [[ -n "$OVERRIDE_AUTO_DOWNLOAD" ]] && [[ "$OVERRIDE_AUTO_DOWNLOAD" != "--auto-download" ]]; then
  echo "  Invalid override auto_download: $OVERRIDE_AUTO_DOWNLOAD"
  exit 1
fi

clear
echo
bold "Kinetra — Trading Launcher"; echo
printf '%0.s─' {1..60}; echo

pick_mode
pick_symbol

GATE="micro"
AUTO_DOWNLOAD=""
PROFILE="balanced"
# Editable runtime params (preloaded from defaults)
BRICK_SIZE=$(get_default BRICK "$SYMBOL")
STOP_BRICKS=$(get_default STOP_LIVE "$SYMBOL")
MONDAY_START=$(get_default MONDAY "$SYMBOL")
FRIDAY_END=$(get_default FRIDAY "$SYMBOL")
TARGET_RISK=$(get_default TARGET_RISK "$SYMBOL")
LOT_CEILING="50"
MAX_LEVERAGE="100"
FLIPRATE_THRESHOLD="0.35"
MARKOV_THRESHOLD="0.55"
TRAILING_MFE_FRACTION="0.50"
TRAILING_MFE_AFTER_BRICKS="2"
PE_ENTRY_THRESHOLD="0.70"
PE_EXIT_THRESHOLD="0.85"
STALE_BRICK_FACTOR="2.0"
MARKOV_STALE_PENALTY="0.15"
USE_OBI_SLIPPAGE_BUFFER="0"
OBI_LEVELS="5"
OBI_MAX_BUFFER_BRICKS="0.25"
OBI_EMA_ALPHA="0.30"
MIN_OMEGA="1.5"
MIN_TRADES="30"
ENTRY_LATENCY_MS="250"
EXIT_LATENCY_MS="250"
LATENCY_JITTER_MS="10"
LATENCY_SEED="42"
CONSERVATIVE_FILLS="1"
SPREAD_SIDES="1.0"
ENTRY_SLIP_BRICKS="0.0"
EXIT_SLIP_BRICKS="0.10"
STOP_WORST_CASE_BRICKS="0.25"
FLIP_TRADE_THROUGH_BRICKS="0.10"

pick_profile
apply_env_overrides
# MONTHS is already set by pick_mode (3 for backtest, 24 for full)
if [[ "$MODE" == "live" || "$MODE" == "dry_run" ]]; then
  pick_gate
fi

# Check data availability for backtest modes
if [[ "$MODE" == "backtest" || "$MODE" == "full" ]]; then
  data_info=$(check_data_availability "$SYMBOL")
  available_months=$(echo "$data_info" | cut -d'|' -f1)
  data_start=$(echo "$data_info" | cut -d'|' -f2)
  data_end=$(echo "$data_info" | cut -d'|' -f3)

  if [[ "$available_months" -eq 0 ]]; then
    echo
    red "  ❌ No M1 data found for $SYMBOL"
    echo
    printf "  Download %s months now? [Y/n] " "$MONTHS"
    read -r ans
    ans="${ans:-y}"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      AUTO_DOWNLOAD="--auto-download"
    else
      echo "  Aborted. Download data first:"
      echo "    python scripts/ctrader/download_ctrader_history.py --symbols $SYMBOL --days $(( MONTHS * 31 ))"
      exit 1
    fi
  elif [[ "$available_months" -lt "$MONTHS" ]]; then
    if prompt_auto_download "$SYMBOL" "$MONTHS" "$available_months"; then
      AUTO_DOWNLOAD="--auto-download"
    fi
    echo
    dim "  Data range: $data_start to $data_end (${available_months} months)"
    echo
  fi
fi

edit_parameters
show_settings "$MODE" "$SYMBOL" "$GATE" "$MONTHS"
if confirm; then
  do_launch "$MODE" "$SYMBOL" "$GATE" "$MONTHS" "$AUTO_DOWNLOAD"
  rc=$?
  exit $rc
else
  echo "  Aborted."
  exit 0
fi
