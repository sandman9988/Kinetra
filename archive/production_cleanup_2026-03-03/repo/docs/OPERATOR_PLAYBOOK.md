# Kinetra Operator Playbook

> **Quick-reference remediation guide for pipeline operators.**
> Last updated: 2026-02-27

---

## Table of Contents

1. [Stale Model Artefacts](#1-stale-model-artefacts)
2. [Pipeline Recovery Procedures](#2-pipeline-recovery-procedures)
3. [Common Failure Modes](#3-common-failure-modes)
4. [Gate Debugging Cheatsheet](#4-gate-debugging-cheatsheet)
5. [Manifest Management](#5-manifest-management)
6. [Data Integrity Issues](#6-data-integrity-issues)
7. [Reward Orchestrator Diagnostics](#7-reward-orchestrator-diagnostics)

Canonical production promotion workflow:
- `docs/PRODUCTION_7_OUTCOME_DECISION_PROCESS.md`
- `python scripts/ops/run_production_readiness.py --target-stage stress_validation`

---

## 1. Stale Model Artefacts

### Symptom

The main menu shows `⚠️ (stale models — re-run pipeline)` next to Exploration,
or backtesting gates remain locked despite model files existing on disk.

### Root Cause

Historical `.joblib` / `.pt` / `.pkl` files exist in `models/`, `checkpoints/`,
or `data/runs/` but were NOT produced by the current pipeline run.  The legacy
`models_trained` flag sees any model file and returns `True`, but the provenance
manifest (`results/training_manifest.json`) is either missing, expired, or
records `agent_comparison_done = false`.

### Remediation

#### Option A: Re-run the pipeline (recommended)

```bash
# 1. Verify pipeline state
python kinetra_menu.py   # check suggest_next_step()

# 2. If additive testing is complete, run agent comparison
#    Menu 3 → option 7 (Agent Comparison)

# 3. Train production agent
#    Menu 3 → option 8 (Train Production Agent)

# 4. The training script writes a fresh manifest automatically
```

#### Option B: Clean up stale files manually

```bash
# 1. Inventory stale artefacts
python -m kinetra.model_manifest   # shows manifest status + files on disk

# 2. Archive old models (NEVER delete without backup)
mkdir -p archive/models_backup_$(date +%Y%m%d)
mv models/trigger/v1/*.joblib archive/models_backup_$(date +%Y%m%d)/

# 3. Verify cleanup
python -m kinetra.model_manifest   # should show "No training manifest found"
```

#### Option C: Write a manifest for existing legitimate models

Only do this if you are **certain** the model files on disk are from a valid,
recent pipeline run that simply didn't write a manifest (e.g., trained before
the manifest feature was added).

```python
from pathlib import Path
from kinetra.model_manifest import (
    build_manifest_from_status,
    discover_model_files,
    write_manifest,
)

root = Path(".")  # from Kinetra project root
manifest = build_manifest_from_status(
    wavelet_additive_step=2,          # match your current step
    agent_comparison_done=True,       # must be True to unlock gates
    agent_best_name="TQC",            # from agent comparison results
    agent_best_omega=4.40,            # from agent comparison results
    training_config={"episodes": 200},
    model_files=discover_model_files(root),
)
write_manifest(manifest, root)
print("Manifest written. Restart the menu to see updated status.")
```

### Prevention

- Always run training through the menu or scripts that call `write_manifest()`.
- The manifest expires after 30 days by default — retrain periodically.
- The CI gate smoke tests (`tests/test_model_manifest.py::TestCIGateSmoke`)
  catch regressions in this logic.

---

## 2. Pipeline Recovery Procedures

### 2.1 Recovery from "No Structure Found"

**Symptom:** `suggest_next_step()` returns
`⚠️ Scientific Discovery found no statistically significant structure`.

**What happened:** The null-hypothesis tests (Hurst, DFA, PE) failed to reject
randomness for any instrument at p < 0.01.

**Recovery steps:**

1. **Check data quality first:**
   ```bash
   # Re-run integrity check
   python scripts/data/check_data_integrity.py --json-output
   # Menu 2 → option 5
   ```

2. **Ensure sufficient data:**
   - Each instrument needs ≥ 5,000 H4 bars (~2.8 years).
   - Run `python scripts/data/curate_instruments.py --report` to check bar counts.

3. **Re-run with broader universe:**
   - Download additional instruments/timeframes.
   - Re-run Scientific Discovery: Menu 3 → option 2.

4. **If structure genuinely absent:** The pipeline is working correctly —
   there is no alpha to find.  Do NOT force past Gate 2.

### 2.2 Recovery from Audit Failure

**Symptom:** `🚨 Prepared data audit FAILED — N NaN, M degenerate features.`

**Recovery steps:**

1. **Inspect the audit report:**
   ```bash
   cat data/.prepared_audit.json | python -m json.tool
   ```

2. **Identify affected instruments:**
   ```bash
   python scripts/data/audit_prepared_data.py --verbose
   ```

3. **Re-prepare affected instruments:**
   ```bash
   python scripts/download/parallel_data_prep.py \
       --instruments XAUUSD DJ30 \
       --tier 1
   ```

4. **Re-run audit:** Menu 2 → option 13.

### 2.3 Recovery from Agent Comparison Failure

**Symptom:** `⚠️ No agent beats the regime-aware baseline.`

**Recovery steps:**

1. **Check exploration health:**
   ```bash
   cat results/wavelet/additive_results.json | python -c "
   import json, sys
   d = json.load(sys.stdin)
   for k, v in d.items():
       print(f'{k}: omega={v.get(\"aggregate_omega\", \"?\")}')"
   ```

2. **If Omega < 1.0 across all steps:** The wavelet features lack learnable
   edge.  Consider:
   - Re-running wavelet analysis with different parameters.
   - Adding more data / instruments.
   - Investigating alternative feature sets.

3. **If Omega > 1.0 but agents lose:** The RL agents are not learning the
   signal well enough.  Try:
   - Increasing episodes (≥ 200 per instrument).
   - Using the orchestrator (`--use-orchestrator`) for better reward shaping.
   - Adjusting the TQC entropy coefficient (alpha).

4. **Re-run agent comparison:** Menu 3 → option 7.

### 2.4 Recovery from Wavelet Gate Lock

**Symptom:** Wavelet pipeline (Menu 3 → option 4) shows 🔒.

**Cause:** Gate requires `has_t1_data AND (has_prepared OR has_wavelet)`.

**Recovery:**

```bash
# Check what's missing
python kinetra_menu.py  # read the lock reason text

# If no Tier-1 data:
#   Run Prime Discovery first: Menu 3 → option 3
#   Then download Tier-1 H4 data: Menu 2 → option 2

# If no prepared data AND no wavelet features:
#   Run Prepare Data: Menu 2 → option 12
#   OR if Gate 2 isn't passed yet, complete Phase 2 first
```

---

## 3. Common Failure Modes

### 3.1 Training Collapse (Agent Never Trades)

**Symptom:** All episodes show 0 trades, reward ≈ 0 (minus inaction penalty).

**Diagnosis:**

```bash
# Check training logs (if orchestrator was used)
ls results/wavelet/training_logs/step_*/
cat results/wavelet/training_logs/step_2/XAUUSD_training_log.csv | head -20
```

Look for:
- `ep_trades = 0` for all episodes → agent stuck in flat policy.
- `shaping_weight` declining too fast → decay floor too low.
- `quality_multiplier < 0.5` → quality gate suppressing bonuses.

**Fix:**

```bash
# Re-run with orchestrator (has anti-collapse pulses)
python scripts/features/run_additive_step.py \
    --step 2 --bands D2,D3 \
    --episodes 200 \
    --use-orchestrator \
    --no-commit   # dry-run first
```

If collapse persists, check:
- Is `inaction_penalty` > 0?  (Should be ~0.001 from reward config.)
- Is the TQC alpha too low?  (< 0.1 causes premature entropy collapse.)

### 3.2 Reckless Trading (Too Many Trades, All Losses)

**Symptom:** High trade count but win rate < 20%, Omega < 0.5.

**Diagnosis:** Check the training log for `quality_multiplier` — if it stays
near 1.0 despite poor win rate, the quality gate isn't engaging.

**Fix:**
- The orchestrator's quality gate activates after `quality_min_trades` (default 3).
- If using inline rewards (no orchestrator), switch to `--use-orchestrator`.
- Increase `quality_win_rate_floor` in `OrchestratorConfig` if needed.

### 3.3 MetaAPI Connection Failures

**Symptom:** `❌ MetaAPI not configured` or connection timeouts.

**Recovery:**

```bash
# 1. Check .env file
cat .env | grep METAAPI

# 2. Test connection
python kinetra_menu.py  # Menu 1 → option 2 (Test MetaAPI Connection)

# 3. If token expired, regenerate at https://app.metaapi.cloud/
#    Then update .env:
#    METAAPI_TOKEN=your_new_token
#    METAAPI_ACCOUNT_ID=your_account_id
```

### 3.4 Import Errors (Missing Dependencies)

**Symptom:** `ModuleNotFoundError` when running scripts.

**Fix:**

```bash
# Install all dependencies
pip install -e ".[dev]"

# Or use make
make setup

# For optional heavy deps (torch, etc.)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Gate Debugging Cheatsheet

Use this table to diagnose why a gate is locked:

| Gate | Required Flags | How to Check | How to Unlock |
|------|---------------|--------------|---------------|
| **Prepare Data** (Menu 2 → 12) | `gate2_ok` = SciDisc + Prime | `status.scientific_discovery_has_structure` + `status.prime_instruments_discovered` | Run SciDisc (Menu 3 → 2) then Prime (Menu 3 → 3) |
| **Audit** (Menu 2 → 13) | `data_prepared` | `ls data/prepared_standardized/train/` | Run Prepare Data (Menu 2 → 12) |
| **Denoise** (Menu 2 → 14) | `data_prepared` | Same as Audit | Same as Audit |
| **Wavelet** (Menu 3 → 4) | `t1_data AND (prepared OR wavelet)` | `status.tier1_instruments_ready > 0` | Download T1 H4 data + prepare or have existing wavelet |
| **Additive** (Menu 3 → 5) | `wavelet_features_ready` | `ls results/wavelet/*_wavelet_features.csv` | Run Wavelet Pipeline (Menu 3 → 4) |
| **Agent Comparison** (Menu 3 → 7) | Additive complete | `status.wavelet_additive_step >= total_steps` | Complete all additive steps |
| **Training** (Menu 3 → 8) | `agent_comparison_done AND agent_beats_baseline` | `cat results/exploration/agent_comparison_FINAL.json` | Run Agent Comparison |
| **Backtest** (Menu 4 → 1-4) | `models_trained_current OR (models_trained AND agent_comparison_done)` | `python -m kinetra.model_manifest` | Train agent + ensure manifest exists |
| **Explore badge ✅** | `models_trained_current OR (models_trained AND agent_comparison_done)` | Main menu badge | Complete full pipeline |

### Quick Status Dump

```bash
# Full status from the menu's perspective
python -c "
import sys; sys.path.insert(0, '.')
from kinetra_menu import check_system_status
s = check_system_status(force=True)
print(s.get_status_line())
print()
print('suggest_next_step:', s.suggest_next_step())
print()
print('Key flags:')
for attr in ['data_ready', 'data_prepared', 'instruments_curated',
             'scientific_discovery_ready', 'scientific_discovery_has_structure',
             'prime_instruments_discovered', 'prime_instruments_tier1',
             'wavelet_features_ready', 'wavelet_additive_step',
             'agent_comparison_done', 'agent_beats_baseline',
             'models_trained', 'models_trained_current',
             'manifest_age_days', 'manifest_run_id']:
    print(f'  {attr} = {getattr(s, attr)}')
"
```

---

## 5. Manifest Management

### Viewing the Current Manifest

```bash
python -m kinetra.model_manifest
```

This prints:
- Manifest version, run ID, creation time, git commit.
- Agent comparison status and best agent info.
- Number of model files tracked.
- Validation result (✅ Current or ❌ Not current + reasons).

### Manifest Lifecycle

```
Training script completes
        │
        ▼
write_manifest() called
        │
        ▼
results/training_manifest.json written
        │
        ▼
check_system_status() reads manifest
        │
        ▼
validate_manifest() checks:
  ├── agent_comparison_done?
  ├── age < expires_days?
  └── ≥50% model files present?
        │
        ▼
models_trained_current = True/False
```

### Manual Manifest Inspection

```bash
cat results/training_manifest.json | python -m json.tool
```

Key fields to check:
- `agent_comparison_done`: Must be `true` for gates to unlock.
- `created_at`: Must be within `expires_days` (default 30).
- `model_files`: Listed paths must exist on disk (≥50%).
- `pipeline_run_id`: Unique per training run — compare against logs.

### Extending Manifest Expiry

If models are still valid but the manifest is about to expire:

```python
from pathlib import Path
from kinetra.model_manifest import read_manifest, write_manifest

root = Path(".")
m = read_manifest(root)
if m:
    m.expires_days = 60  # extend to 60 days
    write_manifest(m, root, discover_files=False)
    print(f"Manifest expiry extended to {m.expires_days} days")
```

---

## 6. Data Integrity Issues

### Checking Data Quality

```bash
# Full integrity scan
python scripts/data/check_data_integrity.py --json-output

# Quick bar count check
python scripts/data/curate_instruments.py --report

# Prepared data audit
python scripts/data/audit_prepared_data.py --verbose
```

### Common Data Problems

| Problem | Symptom | Fix |
|---------|---------|-----|
| Missing H4 bars | Integrity warnings, gap counts | Re-download via Menu 2 → 2 |
| Misclassified instruments | Wrong category folders | Run curation: Menu 2 → 7 |
| Stale symbol specs | `⚠️ Specs (Nd old)` in status | Re-poll: Menu 2 → 8 |
| NaN in prepared data | Audit failure | Re-prepare affected instruments |
| Degenerate features | Zero-variance columns | Check PhysicsEngine input data |

### Safe Data Operations

**NEVER** write data files directly.  Always use:

```python
from kinetra.persistence_manager import get_persistence_manager

pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)
pm.atomic_save(
    filepath="data/master_standardized/metaapi/forex/BTCUSD/BTCUSD_M1_202401010000_202412312359.csv",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False),
)
```

---

## 7. Reward Orchestrator Diagnostics

### Checking Orchestrator Status

When training with `--use-orchestrator`, per-episode CSV logs are written to
`results/wavelet/training_logs/step_N/`.

```bash
# List available logs
ls results/wavelet/training_logs/

# Quick summary of a training run
python -c "
import pandas as pd
df = pd.read_csv('results/wavelet/training_logs/step_2/XAUUSD_training_log.csv')
print(df[['episode', 'ep_trades', 'ep_win_rate', 'shaping_weight',
           'quality_multiplier', 'ep_reward']].describe())
"
```

### Key Diagnostics

| Column | Healthy Range | Problem If |
|--------|--------------|------------|
| `ep_trades` | 5–50 per episode | 0 = collapse, >100 = reckless |
| `ep_win_rate` | 0.35–0.65 | <0.20 = quality gate should activate |
| `shaping_weight` | 3.0→1.0→0.05 | Stuck at 3.0 = warmup never ends |
| `quality_multiplier` | 0.5–1.0 | Always 1.0 despite losses = gate broken |
| `vol_scale` | 0.001–0.1 | 0 = vol floor hit, NaN = crash |
| `flat_streak` | 0–30 | >30 = anti-collapse should fire |

### Orchestrator Configuration Override

If the default orchestrator config doesn't suit an instrument:

```python
from kinetra.reward_orchestrator import OrchestratorConfig, RewardOrchestrator

cfg = OrchestratorConfig(
    training_horizon_steps=200 * 500,  # episodes × episode_len
    flat_streak_threshold=20,          # more aggressive anti-collapse
    anti_collapse_magnitude=0.60,      # stronger pulse
    warmup_multiplier=4.0,             # more exploration early on
    decay_floor=0.10,                  # keep some shaping longer
    quality_win_rate_floor=0.25,       # more lenient quality gate
)
orch = RewardOrchestrator(config=cfg, instrument="XAUUSD")
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    KINETRA OPERATOR QUICK REF                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Check status:   python kinetra_menu.py                         │
│  Check manifest: python -m kinetra.model_manifest               │
│  Run tests:      pytest tests/ -v                               │
│  Lint:           ruff check .                                   │
│  Format:         black .                                        │
│                                                                 │
│  Pipeline order:                                                │
│    ACQUIRE → VALIDATE → CURATE → SPECS → ORGANISE               │
│    → [Gate 2: SciDisc + Prime] → PREPARE → AUDIT                │
│    → WAVELET → ADDITIVE → COMPARE → TRAIN → VALIDATE            │
│                                                                 │
│  Key files:                                                     │
│    results/training_manifest.json    — model provenance          │
│    results/wavelet/additive_step.txt — pipeline progress         │
│    results/exploration/agent_comparison_FINAL.json               │
│    data/.curation_complete.json      — curation marker           │
│    data/.integrity_result.json       — integrity check           │
│    data/.prepared_audit.json         — prepared data audit       │
│                                                                 │
│  Emergency contacts:                                            │
│    See AGENT_RULES_MASTER.md for full rule set                  │
│    See MENU_RESTRUCTURE_PLAN.md for gate design rationale       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
