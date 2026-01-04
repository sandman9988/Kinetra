# KINETRA WORKFLOW & DATA PATHS DOCUMENTATION
===============================================

**Complete mapping of workflows, data paths, and script integrations**

Last Updated: 2026-01-04
Status: Production Ready

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Directory Structure](#directory-structure)
4. [Complete Workflow Map](#complete-workflow-map)
5. [Script Integration Matrix](#script-integration-matrix)
6. [Menu System Navigation](#menu-system-navigation)
7. [Quick Start Guide](#quick-start-guide)

---

## SYSTEM OVERVIEW

Kinetra is a physics-first adaptive trading system with the following pipeline:

```
Authentication → Download → Discovery → Preparation → Training → Backtesting → Analysis
      ↓              ↓          ↓            ↓            ↓           ↓           ↓
    .env          Raw CSV    JSON Index   Features    Models     Results     Reports
```

### Core Principles

- **NO ASSUMPTIONS**: Every decision backed by data
- **PHYSICS-FIRST**: No traditional TA without physics justification
- **ATOMIC OPERATIONS**: All data operations are crash-safe
- **STATISTICAL VALIDATION**: p < 0.01 threshold for all claims
- **VECTORIZATION**: NumPy/Pandas operations preferred over loops

---

## DATA FLOW ARCHITECTURE

### Data Sources

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   MetaAPI       │         │  MT5 Terminal   │         │  Mock/Synthetic │
│  (Cloud Broker) │         │    (Local)      │         │   (Testing)     │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         └───────────────────────────┴───────────────────────────┘
                                     ↓
                              ┌──────────────┐
                              │   Raw Data   │
                              │ (Immutable)  │
                              └──────┬───────┘
                                     ↓
                              ┌──────────────┐
                              │  Discovery   │
                              │   Scanner    │
                              └──────┬───────┘
                                     ↓
                         ┌───────────┴────────────┐
                         ↓                        ↓
                  ┌──────────────┐        ┌──────────────┐
                  │ Standardized │        │   Feature    │
                  │   Training   │   →    │  Engineering │
                  │     Data     │        │  (Physics)   │
                  └──────┬───────┘        └──────┬───────┘
                         │                       │
                         └───────────┬───────────┘
                                     ↓
                              ┌──────────────┐
                              │  RL Training │
                              │    (PPO/     │
                              │  DQN/Linear) │
                              └──────┬───────┘
                                     ↓
                         ┌───────────┴────────────┐
                         ↓                        ↓
                  ┌──────────────┐        ┌──────────────┐
                  │  Backtesting │        │    Risk      │
                  │ (Monte Carlo)│   →    │  Assessment  │
                  └──────┬───────┘        └──────┬───────┘
                         │                       │
                         └───────────┬───────────┘
                                     ↓
                              ┌──────────────┐
                              │   Results    │
                              │  & Analysis  │
                              └──────────────┘
```

---

## DIRECTORY STRUCTURE

### Current State

```
Kinetra/
├── data/
│   ├── master_standardized/          # PRIMARY: Current location of CSV data
│   │   ├── BTCUSD_H1.csv
│   │   ├── EURUSD_H1.csv
│   │   └── ...
│   │
│   ├── available_data.json           # Discovery output (generated)
│   │
│   ├── raw/                          # PLANNED: Immutable source data
│   │   └── {broker}/{account}/{asset_class}/
│   │
│   ├── training/                     # PLANNED: Standardized training data
│   │   └── {asset_class}/
│   │
│   ├── cache/                        # Feature cache
│   └── test_runs/                    # Isolated test runs
│
├── models/                           # Trained RL models
│   ├── BTCUSD_H1_PPO.pkl
│   └── ...
│
├── results/                          # Backtest results & metrics
│   ├── batch_backtest_results.csv
│   ├── comprehensive_exploration_*.json
│   └── backtest_*.json
│
├── plots/                           # Visualizations
│   └── {SYMBOL}_{TF}/
│
├── logs/                            # System logs
│   ├── batch_backtest.log
│   └── train_rl.log
│
├── backups/                         # Data backups (auto-created)
│
├── .env                             # Credentials (NEVER commit)
│
└── scripts/                         # All executable scripts
    ├── download/                    # Data download scripts
    ├── training/                    # RL training scripts
    ├── exploration/                 # Exploration framework
    ├── analysis/                    # Analysis & visualization
    ├── testing/                     # Test scripts
    ├── servers/                     # MT5 bridge server
    └── utils/                       # Utilities
```

### Data Path Transitions

**Current → Planned Migration:**

```
data/master_standardized/*.csv  →  data/raw/{broker}/{account}/{asset_class}/*.csv
                                →  data/training/{asset_class}/*_standardized.parquet
```

**Why?**
- Raw data immutable (append-only)
- Training data regenerated fresh
- Better organization by broker/account
- Atomic operations at all levels

---

## COMPLETE WORKFLOW MAP

### 1. AUTHENTICATION WORKFLOW

**Goal**: Store credentials securely in `.env` file

**Scripts**:
- `scripts/master_workflow.py` (auth functions)
- `scripts/download/setup_metaapi_credentials.py`

**Flow**:
```
User Input (Token/Account ID)
    ↓
Validation (API call test)
    ↓
Encryption (optional)
    ↓
Save to .env
    ↓
Verify credentials
```

**Outputs**:
- `.env` file with:
  - `METAAPI_TOKEN`
  - `METAAPI_ACCOUNT_ID`

**Menu Path**: Main → 1 → 1

---

### 2. DATA DOWNLOAD WORKFLOW

**Goal**: Download historical OHLCV data from brokers

#### Option A: MetaAPI (Cloud)

**Scripts**:
- `scripts/download/download_metaapi.py` (interactive)
- `scripts/download/metaapi_bulk_download.py` (bulk)
- `scripts/download/metaapi_sync.py` (sync operations)

**Flow**:
```
Load credentials from .env
    ↓
Connect to MetaAPI
    ↓
Query available symbols
    ↓
Select symbols + timeframes + date range
    ↓
Fetch OHLCV data (paginated, 1000 bars/request)
    ↓
Save to data/master_standardized/SYMBOL_TF.csv
```

**Outputs**:
- `data/master_standardized/SYMBOL_TF.csv` (tab-separated)
- Columns: `time, open, high, low, close, tick_volume`

**Menu Path**: Main → 2 → 2

#### Option B: MT5 Local Terminal

**Scripts**:
- `scripts/download/download_mt5_data.py`
- `scripts/servers/mt5_bridge_server.py` (for remote access)

**Requirements**:
- MetaTrader 5 terminal running
- `pip install MetaTrader5`
- Windows or Wine (Linux)

**Flow**:
```
Connect to MT5 terminal
    ↓
Query available symbols
    ↓
Request historical data (copy_rates_range)
    ↓
Convert to pandas DataFrame
    ↓
Save to data/master_standardized/SYMBOL_TF.csv
```

**Outputs**:
- Same format as MetaAPI

**Menu Path**: Main → 2 → 3

#### Option C: Mock/Synthetic Data

**Scripts**:
- `scripts/batch_backtest.py` (has `generate_mock_data()`)

**Use Case**: Testing when no broker access

**Flow**:
```
Define symbol characteristics (volatility, drift)
    ↓
Generate random walk with realistic properties
    ↓
Add noise and microstructure
    ↓
Return as DataFrame
```

---

### 3. DATA DISCOVERY WORKFLOW

**Goal**: Scan filesystem and create inventory of available data

**Script**: `scripts/discover_available_data.py`

**Flow**:
```
Scan data/master_standardized/
    ↓
Parse filenames (SYMBOL_TF.csv)
    ↓
Read each file (get row count, date range)
    ↓
Classify as usable (>= 1000 bars) or insufficient
    ↓
Export to JSON
```

**Outputs**:
- `data/available_data.json`:
```json
[
  {
    "symbol": "BTCUSD",
    "timeframe": "H1",
    "bars": 8760,
    "file": "BTCUSD_H1.csv",
    "size_mb": 0.5,
    "date_range": ["2023-01-01", "2023-12-31"]
  }
]
```

**Console Output**:
```
Found 15 unique symbols:
  BTCUSD: 3 timeframes (H1, H4, D1) - 26,280 total bars
  EURUSD: 2 timeframes (H1, H4) - 17,520 total bars
  ...

✅ USABLE (12/15 combinations with >=1000 bars)
```

**Menu Path**: Main → 2 → 1

---

### 4. DATA PREPARATION WORKFLOW

**Goal**: Standardize, validate, and engineer features

**Scripts**:
- `scripts/download/prepare_data.py`
- `scripts/download/check_data_integrity.py`
- `scripts/consolidate_data.py`

**Flow**:
```
Load raw CSV
    ↓
Validate integrity (gaps, duplicates, NaNs)
    ↓
Apply non-linear transformations:
  - Log-returns (log price changes)
  - Median filtering (gap detection)
  - Asymmetric quantile clipping
    ↓
Engineer physics features:
  - Kinetic energy (0.5 * velocity^2)
  - Potential energy (price * volume)
  - Entropy (Shannon, Tsallis)
  - Flow (cumulative volume delta)
  - Damping (energy decay rate)
    ↓
Save standardized data
```

**Physics Features** (NO traditional TA):
- `log_E_t`: Log kinetic energy
- `median_Damping`: Energy decay (friction)
- `clipped_Delta`: Asymmetric volume flow
- `CVD`: Cumulative volume delta
- `RSI`: ONLY if physics-justified
- `ATR`: Average true range (volatility)

**Outputs**:
- Validated CSV with additional feature columns
- Integrity report

**Menu Path**: Main → 2 → 4, 2 → 6

---

### 5. TRAINING WORKFLOW

**Goal**: Train RL agents on physics features

#### Quick RL Training

**Script**: `scripts/training/train_rl.py`

**Flow**:
```
Load standardized data
    ↓
Create SimpleTradingEnv (Gymnasium)
    ↓
Initialize PPO agent (Stable-Baselines3)
    ↓
Train for N episodes
    ↓
Checkpoint periodically (atomic writes)
    ↓
Log metrics (Omega, win rate, entropy)
    ↓
Save final model
```

**Configuration**:
```python
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.01,  # Exploration
}
```

**Outputs**:
- `models/SYMBOL_TF_PPO.pkl`
- Training logs in `logs/train_rl.log`

**Menu Path**: Main → 3 → 1

#### Agent Comparison

**Script**: `scripts/training/explore_compare_agents.py`

**Agents Compared**:
1. PPO (Proximal Policy Optimization)
2. DQN (Deep Q-Network)
3. Linear (Baseline)
4. Triad (Ensemble)

**Flow**:
```
For each agent type:
    Train on same data
    Evaluate on same test set
    Compute metrics (Omega, Sharpe, Win%)
    ↓
Compare results statistically
    ↓
Generate comparison report
```

**Outputs**:
- `results/agent_comparison_*.json`
- Comparative plots

**Menu Path**: Main → 3 → 2

#### Specialist Agents

**Scripts**:
- `scripts/training/train_berserker.py` (trend following)
- `scripts/training/train_sniper.py` (mean reversion)
- `scripts/training/train_triad.py` (ensemble)

**Menu Path**: Main → 3 → 4

#### Comprehensive Exploration

**Script**: `scripts/exploration/run_comprehensive_exploration.py`

**What It Does**:
- All asset classes
- All timeframes
- Agent comparison
- Measurement impact analysis
- Statistical validation
- PCA/ICA/Chaos analysis

**Outputs**:
- `results/comprehensive_exploration_TIMESTAMP.json`

**Menu Path**: Main → 3 → 3

---

### 6. BACKTESTING WORKFLOW

**Goal**: Validate RL agents on out-of-sample data

#### Quick Backtest

**Script**: `scripts/batch_backtest.py`

**Flow**:
```
Load data (split: 70% train, 30% test)
    ↓
Prepare non-linear features
    ↓
Train SuperPot PPO agent
    ↓
Evaluate on training set:
  - Omega ratio
  - Win rate
    ↓
Evaluate on out-of-sample set:
  - Omega ratio
  - Win rate
  - OOS degradation
  - KS test (distribution similarity)
    ↓
Check survival criteria:
  - Omega > 2.7
  - Win rate > 55%
  - OOS drop < 5%
  - KS p-value > 0.05
    ↓
If survives:
  - Compute triggers (asymmetric features)
  - Compute harvesters (log-trail MFE)
  - Compute risk (CHS, RoR)
    ↓
Save results
```

**Survival Criteria**:
```python
SURVIVAL_OME = 2.7      # Omega ratio threshold
SURVIVAL_WIN = 55.0     # Win rate threshold
MAX_OOS_DROP = 5.0      # Max out-of-sample degradation
MIN_KS_P = 0.05         # Distribution similarity
```

**Outputs**:
- `data/batch_backtest_results.csv`:
```csv
symbol,year,omega_train,win_train,trigger,mfe,chs,entropy
BTCUSD,2023,3.2,62.5,log_E_t,0.68,0.85,0.12
```

**Menu Path**: Main → 4 → 1

#### Monte Carlo Validation

**Script**: `scripts/batch_backtest.py --mc-runs 100`

**Flow**:
```
For each run (different random seed):
    Train agent
    Backtest
    Record metrics
    ↓
Aggregate results:
    Mean, Std, CI (95%)
    ↓
Statistical significance test (p < 0.01)
```

**Outputs**:
- `results/monte_carlo_TIMESTAMP.json`
- Statistical distribution plots

**Menu Path**: Main → 4 → 3

---

### 7. ANALYSIS & VISUALIZATION WORKFLOW

**Scripts**:
- `scripts/research/analyze_results.py`
- `scripts/research/fat_candle_forensics.py`

**Flow**:
```
Load backtest results
    ↓
Generate plots:
  - Equity curves
  - Drawdown analysis
  - Win/loss distribution
  - Omega ratio over time
    ↓
Statistical tests:
  - Sharpe ratio significance
  - Distribution normality
  - Autocorrelation
    ↓
Export HTML report
```

**Outputs**:
- `plots/SYMBOL_TF/equity_curve.png`
- `plots/SYMBOL_TF/drawdown.png`
- `results/analysis_report.html`

**Menu Path**: Main → 4 → 6

---

## SCRIPT INTEGRATION MATRIX

### By Functionality

| Functionality | Primary Script | Dependencies | Output |
|--------------|----------------|--------------|--------|
| **Authentication** | `scripts/master_workflow.py` | dotenv | `.env` |
| | `scripts/download/setup_metaapi_credentials.py` | metaapi_cloud_sdk | `.env` |
| **Data Download** | `scripts/download/download_metaapi.py` | metaapi_cloud_sdk | `data/master_standardized/*.csv` |
| | `scripts/download/download_mt5_data.py` | MetaTrader5 | `data/master_standardized/*.csv` |
| | `scripts/download/metaapi_bulk_download.py` | metaapi_cloud_sdk | Multiple CSVs |
| **Discovery** | `scripts/discover_available_data.py` | pandas | `data/available_data.json` |
| **Preparation** | `scripts/download/prepare_data.py` | pandas, numpy | Standardized CSV |
| | `scripts/download/check_data_integrity.py` | pandas | Integrity report |
| | `scripts/consolidate_data.py` | pandas | Merged data |
| **Training** | `scripts/training/train_rl.py` | stable-baselines3 | `models/*.pkl` |
| | `scripts/training/train_rl_physics.py` | stable-baselines3 | `models/*.pkl` |
| | `scripts/training/explore_compare_agents.py` | stable-baselines3 | `results/*.json` |
| **Backtesting** | `scripts/batch_backtest.py` | stable-baselines3 | `data/batch_backtest_results.csv` |
| | `kinetra/backtest_engine.py` | None (core) | Backtest metrics |
| **Analysis** | `scripts/research/analyze_results.py` | matplotlib | Plots, reports |
| **System** | `scripts/cache_manager.py` | None | Cache stats |
| | `scripts/backup_data.py` | shutil | `backups/` |
| | `scripts/devops_manager.py` | None | System maintenance |

---

## MENU SYSTEM NAVIGATION

### Main Menu Structure

```
KINETRA PRODUCTION MENU
========================

1. 🔐 Setup & Authentication
   1.1 Configure MetaAPI Credentials
   1.2 Test MetaAPI Connection
   1.3 Configure MT5 (Local Terminal)
   1.4 Test MT5 Connection
   1.5 View Current Configuration

2. 📊 Data Management
   2.1 Discover Available Data
   2.2 Download Data (MetaAPI)
   2.3 Download Data (MT5 Local)
   2.4 Check Data Integrity
   2.5 View Data Coverage
   2.6 Consolidate & Clean Data

3. 🔬 Exploration & Training
   3.1 Quick RL Training (Physics-Only)
   3.2 Agent Comparison (PPO vs DQN vs Linear)
   3.3 Comprehensive Exploration Suite
   3.4 Train Specialist Agents
   3.5 Scientific Discovery (PCA/ICA/Chaos)
   3.6 View Training Results

4. 📈 Backtesting & Validation
   4.1 Quick Backtest
   4.2 Batch Backtest
   4.3 Monte Carlo Validation (100 runs)
   4.4 Walk-Forward Analysis
   4.5 View Backtest Results
   4.6 Generate Performance Report

5. 🛠️ System Tools & Monitoring
   5.1 System Status & Diagnostics
   5.2 Cache Management
   5.3 Backup Data
   5.4 Clean Temporary Files
   5.5 Run Tests
   5.6 View Logs

0. Exit
```

### Context-Aware Features

The menu system checks:
- ✅ Data availability
- ✅ Credentials configured
- ✅ Dependencies installed (MT5, GPU)
- ✅ Previous workflow state
- ✅ Trained models existence

**Smart Suggestions**:
```
Status: ❌ No data | ⚠️ No creds | ✅ MetaAPI | ✅ GPU
💡 Next Step: Setup credentials (Menu 1)
```

---

## QUICK START GUIDE

### First-Time Setup (Morning Testing)

```bash
# 1. Start the menu
python kinetra_production_menu.py

# 2. Setup credentials (if not done)
Main → 1 → 1
# Enter MetaAPI token and account ID

# 3. Discover existing data
Main → 2 → 1
# Scans filesystem, creates available_data.json

# 4. Download more data if needed
Main → 2 → 2
# Interactive MetaAPI download

# 5. Quick backtest to verify system
Main → 4 → 1
# Select BTCUSD H1 (or any available symbol)

# 6. View results
Main → 4 → 5
```

### Typical Daily Workflow

```bash
# Check system status
Main → 5 → 1

# Download latest data
Main → 2 → 2

# Train on fresh data
Main → 3 → 1

# Backtest trained model
Main → 4 → 2

# Analyze results
Main → 4 → 6
```

### For Real-World Trading

```bash
# 1. Comprehensive exploration
Main → 3 → 3
# All asset classes, all timeframes

# 2. Monte Carlo validation
Main → 4 → 3
# 100 runs, statistical validation

# 3. Only trade if:
#    - Omega > 2.7
#    - Win rate > 55%
#    - p-value < 0.01
#    - Out-of-sample validated
```

---

## KEY FILES & THEIR ROLES

### Entry Points

| File | Purpose | When to Use |
|------|---------|-------------|
| `kinetra_production_menu.py` | **MAIN MENU** | All user interactions |
| `scripts/batch_backtest.py` | Full backtest pipeline | Standalone backtesting |
| `scripts/discover_available_data.py` | Data inventory | Check what data exists |
| `scripts/training/train_rl.py` | Quick RL training | Standalone training |

### Core Modules

| Module | Purpose |
|--------|---------|
| `kinetra/data/manager.py` | Data management API |
| `kinetra/backtest_engine.py` | Backtesting engine |
| `kinetra/physics_engine.py` | Physics calculations |
| `kinetra/rl_agent.py` | RL agent interface |
| `kinetra/context_aware_menu.py` | Menu context checking |

### Configuration

| File | Purpose |
|------|---------|
| `.env` | Credentials (NEVER commit) |
| `pyproject.toml` | Python dependencies |
| `configs/*.yaml` | Agent configurations |
| `data/available_data.json` | Data inventory (generated) |

---

## DATA VALIDATION CHECKLIST

Before any operation:

✅ **Raw Data**:
- [ ] No NaN values in OHLCV
- [ ] No duplicate timestamps
- [ ] Chronological order
- [ ] Realistic price ranges
- [ ] Volume > 0

✅ **Prepared Data**:
- [ ] Features are finite (no inf/-inf)
- [ ] Log-space calculations stable
- [ ] Quantile clipping applied
- [ ] Gap detection done

✅ **Backtest**:
- [ ] Train/test split > 70/30
- [ ] No lookahead bias
- [ ] Realistic slippage/costs
- [ ] Statistical significance (p < 0.01)

---

## TROUBLESHOOTING

### Common Issues

**"No data found"**:
```bash
# Run discovery first
Main → 2 → 1

# If still empty, download data
Main → 2 → 2
```

**"MetaAPI connection failed"**:
```bash
# Check credentials
Main → 1 → 5

# Re-configure if needed
Main → 1 → 1
```

**"MT5 not available"**:
```bash
# Install package
pip install MetaTrader5

# Start MT5 terminal
# Then test connection
Main → 1 → 4
```

**"Training fails with NaN"**:
- Check data integrity (Main → 2 → 4)
- Ensure log-space calculations have epsilon
- Validate feature engineering

---

## PERFORMANCE TARGETS

### Backtest Thresholds

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Win Rate** | > 55% | Consistency |
| **Z-Factor** | > 2.5 | Statistical edge |
| **% Energy Captured** | > 65% | Physics alignment |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |
| **OOS Degradation** | < 5% | Generalization |
| **KS Test p-value** | > 0.05 | Distribution match |

### Statistical Validation

All claims must meet:
- **p-value < 0.01** (99% confidence)
- **Monte Carlo runs ≥ 100** (for production)
- **Out-of-sample validation** (mandatory)

---

## NEXT STEPS FOR PRODUCTION

1. **Complete Data Migration**:
   - Move `data/master_standardized/` → `data/raw/{broker}/{account}/{asset_class}/`
   - Implement training data generation pipeline
   - Add Parquet support for performance

2. **Enhanced Monitoring**:
   - Real-time health dashboard
   - Alerting system for anomalies
   - Performance tracking over time

3. **Walk-Forward Analysis**:
   - Implement rolling window training
   - Automated out-of-sample validation
   - Prevent overfitting

4. **Live Trading Integration**:
   - Virtual trading (paper)
   - Demo account testing
   - Production broker integration (with safety gates)

---

## CONTACT & SUPPORT

For questions about workflows:
1. Check this document first
2. Review `AGENT_RULES_MASTER.md` for design principles
3. Consult specific script's docstring
4. Check `docs/` directory for detailed guides

---

**Last Updated**: 2026-01-04  
**Version**: 1.0  
**Status**: Production Ready ✅