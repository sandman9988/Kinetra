# Legacy Archive (2026-03-03)

This folder contains files moved out of the project root/scripts as part of a
conservative cleanup pass.

Archive policy used in this pass:
- Keep runtime code and canonical docs in place.
- Move obvious legacy/one-off artifacts and backup copies only.
- Verify Renko lint + tests before and after archiving.

Moved groups:
- `root/`: one-off AB test scripts/docs, transient session/final notes,
  accidental launcher flag files (`--ack-live`, `--symbols`, `--target-gate`).
- `scripts/`: `*.backup` files.

Validation run after archive:
- `ruff check kinetra/renko scripts/renko_engine.py tests/test_renko_modules_core.py`
- `pytest -q tests/test_renko_engine_cli.py tests/test_renko_engine_runtime.py tests/test_renko_modules_core.py`
