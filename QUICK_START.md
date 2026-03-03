# Kinetra Quick Start (Current)

Last updated: 2026-03-03

For current live/paper/backtest runtime semantics, read:
- [`archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md`](archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md)

## 1) Launch Interactive Menu

```bash
make launch
```

## 2) Direct CLI

```bash
# Backtest
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

# Paper (live bars, paper fills)
python scripts/renko_engine.py XAUUSD --stage paper

# Live dry-run (alias to paper path)
python scripts/renko_engine.py XAUUSD --stage live --dry-run --live-size micro

# LIVE (real orders)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro
```

## 3) Optional Execution Preflight (Real Tiny Order)

```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK
```

## 4) Useful Strategy Overrides

```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --brick-size 1.0 --stop-bricks 0.5 --target-risk 100 \
  --fliprate-window 431 --markov-window 431 \
  --fliprate-threshold 0.35 --markov-threshold 0.55
```

## 5) Live Dashboard Notes

- `Renko Bricks` are per-run, not account-lifetime.
- `Dupes dropped` shows stream duplicates dropped before strategy processing.
- `Connection` block shows endpoint, timeout count, and heartbeat.

## 6) Troubleshooting

- Auth/connect failures: verify `.env.openapi` credentials.
- If launcher exits on prompt: run in an interactive shell (it needs input).
- For latest live snapshot semantics: use `archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md`.
