# Renko Live State (Canonical)

Last updated: 2026-03-03

This is the canonical status/reference for `scripts/renko_engine.py` live/paper/backtest behavior.

## What Is Live Now

- Single Renko engine shared across backtest, paper, dry-run, and live stages.
- Live dashboard uses **Renko Bricks** (not bars) as strategy progress.
- cTrader live trendbar feed has per-symbol M1 deduplication.
- Engine tracks and drops out-of-order/duplicate stream bars.
- Live snapshot supports replace mode (single latest panel) and throttled rendering.
- Connection telemetry is displayed in snapshot.

## Snapshot Fields (Current)

### Trade Performance
- `Renko Bricks`: count since current engine run started.
- `Dupes dropped`: stream duplicates/out-of-order bars dropped by engine callback guard.
- `Trades`, `Net P&L`, `Omega`.

### Trade Eval / Decision
- `Eval`: current brick direction + flip + gate pass state.
- `Warmup`: filter warmup readiness and remaining bricks.
- `Eval metrics`: FR, Markov, computed lots.
- `Decision`: last decision reason (`no_flip`, `warmup`, `filters_not_ready`, `gate_reject`, `entry_pass`, `entered`, etc).

### Live Filter Activity
- `Bars seen`: raw stream bars received by engine callback.
- `Flips seen`: renko colour flips observed.
- `Filter-ready`: bricks where FR + Markov values are finite.
- `Last brick UTC`: timestamp of last processed brick.

### Connection
- `Status`: UP/DOWN
- `Heartbeat`: slow blinking dot (green up, red down)
- `Endpoint`: selected cTrader endpoint
- `Req timeouts`: connector request timeout counter (session)
- `Snapshot source`: `fresh` or `cached`

### Account / Broker Info
- `Balance`, `Account`, `Broker`, `Environment`
- `Account type`: human-friendly label + raw code

## Live Safety / Execution

- Preflight checks run before live stage.
- Optional execution preflight test order:
  - `--preflight-test-order`
  - `--preflight-lots`
  - requires `--ack-live I_UNDERSTAND_LIVE_RISK`

## Key CLI Flags (Current)

- Stage/mode:
  - `--stage {dsp,backtest,paper,live,all}`
  - `--dry-run`
  - `--live-size {micro,small,full}`

- Strategy overrides:
  - `--brick-size`
  - `--stop-bricks`
  - `--target-risk`
  - `--fliprate-window`, `--markov-window`
  - `--fliprate-threshold`, `--markov-threshold`

- Drift adaptation:
  - `--drift-adapt` (+ `--adapt-*` controls)

- Dashboard behavior:
  - `KINETRA_LIVE_DASHBOARD_REPLACE` (default `1`)
  - `KINETRA_LIVE_DASHBOARD_MIN_INTERVAL_S` (default `2` in replace mode, else `30`)
  - `KINETRA_LIVE_HEARTBEAT_INTERVAL_S` (default `2.0`)

## Notes

- `Renko Bricks` are per-run counters (reset when engine starts), not account-lifetime values.
- If broker terminal shows fills but snapshot `Trades` stays `0`, that means broker fills occurred outside engine-closed-trade accounting for that run (e.g., separate preflight/order path).
