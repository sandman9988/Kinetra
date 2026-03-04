# Production Cleanup Archive (2026-03-03)

This archive contains non-production artifacts moved out of active runtime
paths to reduce the repository surface area for live operation.

## What was moved
- `root_docs/`: one-off analysis/summary markdown files.
- `root_misc/`: ad-hoc scripts/data exports not used by launcher runtime.
- `scripts/`: non-runtime script trees (`analysis`, `exploration`, `research`,
  `testing`, `training`, `examples`).
- `repo/tests/`: full test tree moved out of production paths.
- `repo/examples/`: top-level examples moved out of production paths.

## Runtime kept in place
- `kinetra/`
- `scripts/renko_engine.py`
- `scripts/renko/`
- `scripts/ctrader/`
- `scripts/download/`
- core configs and launcher docs (`README.md`, `QUICK_START.md`,
  `LIVE_TRADING_CONFIG.md`, `PREFLIGHT_CHECKS.md`)

## Verification after move
- `python -m py_compile scripts/renko_engine.py kinetra/connectors/ctrader_connector.py kinetra/renko/trading_engine.py kinetra/renko/ctrader_dispatcher.py`
- `ruff check kinetra/renko scripts/renko_engine.py kinetra/connectors/ctrader_connector.py kinetra/renko/ctrader_dispatcher.py`
- `python scripts/renko_engine.py --help`

## Restore
Move directories/files back from this archive if you need test/research assets.
