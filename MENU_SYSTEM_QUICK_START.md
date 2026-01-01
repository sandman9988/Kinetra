# Kinetra Menu System - Quick Start

## First-Time Setup

### 1. Installation

```bash
# Clone repository
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Configure API Token

Set up your MetaAPI credentials (encrypted storage):

```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your MetaAPI token
# METAAPI_TOKEN=your_token_here
```

**Security Note:** Credentials are encrypted using Fernet encryption and stored securely.

## Launch Menu

```bash
python kinetra_menu.py
```

## Main Menu Options

```
┌─────────────────────────────────────────────────────────────────┐
│                     KINETRA MAIN MENU                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Login & Authentication                                       │
│  2. Exploration Testing (Hypothesis & Theorem Generation)        │
│  3. Backtesting (ML/RL EA Validation)                           │
│  4. Data Management                                              │
│  5. System Status & Health                                       │
│  0. Exit                                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Menu System Workflows

### System Status & Health Menu

```
┌─────────────────────────────────────────────────────────────────┐
│                   System Status & Health                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Current System Health
             │   ├─→ Composite Health Score (CHS)
             │   ├─→ Active agents status
             │   ├─→ Circuit breaker status
             │   └─→ Risk management gates
             │
             ├─→ [2] Recent Test Results
             │   ├─→ Last 10 test runs
             │   ├─→ Success/failure summary
             │   └─→ Performance metrics
             │
             ├─→ [3] Data Summary
             │   ├─→ Master data inventory
             │   ├─→ Prepared data status
             │   └─→ Data integrity scores
             │
             ├─→ [4] Performance Metrics
             │   ├─→ Current Omega Ratio
             │   ├─→ Z-Factor trend
             │   ├─→ Energy capture efficiency
             │   └─→ MFE capture rate
             │
             └─→ [0] Back to Main Menu
```

### Quick Workflows

#### Authentication Setup (5 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 1 → 1 (Login → Select MetaAPI Account)
3. Choose account from list
4. Done! Credentials encrypted and stored
```

#### Quick Exploration (30-60 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 2 → 1 (Exploration → Quick Exploration)
3. System auto-downloads data and runs exploration
4. Review results in data/results/
```

**Features:**
- Automatic data orchestration (download, verify, prepare)
- Resumable on interruption (checkpointed state)
- Statistical validation (p < 0.01 required)

#### Quick Backtest (15-30 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 3 → 1 (Backtesting → Quick Backtest)
3. Loads best agents from exploration
4. Review performance metrics
```

**Features:**
- MT5-accurate friction modeling (spread, slippage, commission)
- Monte Carlo validation (100 runs)
- Realistic cost analysis

#### Check Data Status (1 minute)

```
1. Launch: python kinetra_menu.py
2. Select: 4 → 3 (Data Management → Check & Fill Missing Data)
3. System scans and reports missing data
```

## End-to-End (E2E) Testing

**Note:** E2E testing is a command-line framework (not in the interactive menu) for comprehensive system validation across all combinations.

### Quick Validation (15 minutes)

```bash
python e2e_testing_framework.py --quick
```

**Test Matrix:** 12 combinations
- Asset Classes: crypto, forex
- Instruments: Top 3 per class
- Timeframes: H1, H4
- Agent: PPO

### Crypto Testing (3 hours)

```bash
python e2e_testing_framework.py --asset-class crypto
```

**Test Matrix:** 150 combinations
- 10 crypto instruments
- 5 timeframes
- 3 agents (PPO, DQN, Linear)

### PPO Agent Testing (4.6 hours)

```bash
python e2e_testing_framework.py --agent-type ppo
```

**Test Matrix:** 220 combinations
- All asset classes
- All instruments
- All timeframes
- PPO agent only

### Full System Test (24-48 hours)

```bash
python e2e_testing_framework.py --full
```

**Test Matrix:** ~2200 combinations
- All asset classes
- All instruments
- All timeframes
- All agents

### Dry Run (Preview Test Matrix)

```bash
python e2e_testing_framework.py --quick --dry-run
```

Shows test configuration and estimated duration without running.

## Command Line Examples

```bash
# Interactive menu
python kinetra_menu.py

# Quick E2E validation
python e2e_testing_framework.py --quick

# Dry run (no execution - preview test matrix)
python e2e_testing_framework.py --quick --dry-run

# Asset class test
python e2e_testing_framework.py --asset-class crypto

# Agent type test
python e2e_testing_framework.py --agent-type ppo

# Timeframe test
python e2e_testing_framework.py --timeframe H1

# Full system test (24-48 hours)
python e2e_testing_framework.py --full

# Custom configuration
python e2e_testing_framework.py --config my_config.json
```

### Advanced Usage

**Parallel Execution** (automatic in E2E framework):
```bash
# Parallel execution enabled by default in config
# Set "parallel_execution": true in JSON config
python e2e_testing_framework.py --config parallel_config.json
```

**Resume Interrupted Runs:**
```bash
# Runs are automatically checkpointed
# Re-run the same command to resume from last checkpoint
python e2e_testing_framework.py --full
# System detects previous state and asks to resume
```

## Configuration Files

### Custom E2E Configuration

Create `config/my_test_config.json` (or any location):

```json
{
  "name": "my_test",
  "description": "My custom E2E test",
  "asset_classes": ["crypto", "forex"],
  "instruments": ["top_5"],
  "timeframes": ["H1", "H4"],
  "agent_types": ["ppo", "dqn"],
  "episodes": 100,
  "parallel_execution": true,
  "auto_data_management": true,
  "statistical_validation": true,
  "monte_carlo_runs": 100
}
```

**Default values:**
- `episodes`: 100
- `parallel_execution`: true
- `auto_data_management`: true
- `statistical_validation`: true
- `monte_carlo_runs`: 100

Run with:

```bash
python e2e_testing_framework.py --config config/my_test_config.json
```

## Key Concepts

### Asset Classes
- **crypto**: Cryptocurrency (BTC, ETH, etc.)
- **forex**: Foreign Exchange (EUR/USD, GBP/USD, etc.)
- **indices**: Stock Indices (US30, SPX500, etc.)
- **metals**: Precious Metals (XAU/USD, XAG/USD)
- **commodities**: Commodities (Oil, Gas, etc.)

### Timeframes
- **M15**: 15 Minutes (intraday scalping)
- **M30**: 30 Minutes (intraday trading)
- **H1**: 1 Hour (short-term swing)
- **H4**: 4 Hours (medium-term swing)
- **D1**: 1 Day (position trading)

### Agent Types
- **ppo**: Proximal Policy Optimization
- **dqn**: Deep Q-Network
- **linear**: Linear Q-Learning
- **berserker**: Berserker Strategy
- **triad**: Triad System (Incumbent/Competitor/Researcher)

## Key Metrics Glossary

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Omega Ratio** | > 2.7 | Asymmetric returns measure (upside vs downside probability ratio) |
| **Z-Factor** | > 2.5 | Statistical robustness score (edge significance, similar to Sharpe but stricter) |
| **% Energy Captured** | > 65% | Physics-based efficiency: % of kinetic energy (from price momentum) extracted |
| **Composite Health Score (CHS)** | > 0.90 | Weighted system health: combines Omega, Z-Factor, Energy, Risk metrics |
| **% MFE Captured** | > 60% | Execution quality: % of Maximum Favorable Excursion captured before exit |

### Metric Details

**Composite Health Score (CHS):**

CHS is a proprietary holistic health metric that aggregates multiple performance, risk, and efficiency factors into a single normalized score (0-1 scale). It serves as a real-time "health check" to prevent overfitting, ensure risk management, and guide decisions like halting tests or flagging underperforming agents.

**Calculation Process:**

1. **Data Collection**: Gather raw metrics from BacktestEngine/HealthMonitor
   - Omega Ratio (Ω): Gains/Losses ratio (threshold-adjusted)
   - Z-Factor (Z): Statistical robustness score
   - Energy Captured (E): % of market kinetic energy extracted
   - MFE Captured (M): % of Maximum Favorable Excursion realized
   - Risk-of-Ruin (RoR): Probability of catastrophic loss
   - Statistical Validity: p-values, Monte Carlo confidence intervals

2. **Normalization**: Scale each metric to [0,1] range
   ```
   Normalized_X = (X - Min_Target) / (Max_Target - Min_Target)
   ```
   Example: If Omega = 3.0, Normalized_Ω ≈ 0.60 (scaled to 0-5 range)
   
   For inverse metrics (RoR): `Normalized_RoR = 1 - (RoR / Max_Acceptable)`

3. **Weighting**: Assign importance weights (sum to 1.0)
   - Performance (Omega, Z-Factor): 40% (0.20 each)
   - Efficiency (Energy, MFE): 30% (0.15 each)
   - Risk/Stability (RoR, circuit factors): 20%
   - Statistical Validity (p-values, MC CI): 10%

4. **Aggregation**: Compute weighted sum
   ```
   CHS = Σ (Normalized_Metric_i × Weight_i)
   ```
   
   Example:
   ```
   CHS = (0.80×0.25) + (0.75×0.25) + (0.90×0.25) + (0.85×0.25) = 0.825
   ```

5. **Validation & Thresholds**:
   - CHS > 0.90: Strong health (viable for production)
   - CHS 0.55-0.90: Marginal (needs tuning)
   - CHS < 0.55: Critical (circuit breaker triggers halt)

**Circuit Breaker Logic:**
```python
if CHS < 0.55:
    halt_trading()
    log_alert("CHS below threshold - system halted")
```

**Usage in Kinetra:**
- Real-time monitoring during exploration/backtesting
- Triggers auto-actions via risk gates
- Reported in performance matrices
- Checkpointed for reproducibility

**Z-Factor:**
Custom robustness metric incorporating:
- Statistical significance (p-value)
- Win rate asymmetry
- Profit factor
- Drawdown recovery

**% Energy Captured:**
Based on physics-first approach:
```
Energy = 0.5 * m * (ΔP / Δt)²
Efficiency = PnL_extracted / Total_Market_Energy
```

**Pythagorean Distance:**
MFE/MAE efficiency vector:
```
Efficiency = sqrt(MFE² + MAE²) / Ideal_Distance
```

## File Locations

- **Menu System**: `kinetra_menu.py`
- **E2E Framework**: `e2e_testing_framework.py`
- **Results**: `data/results/`
- **Logs**: `logs/`
- **Master Data**: `data/master/`
- **Prepared Data**: `data/prepared/`

## Documentation

- **[Flowchart](docs/MENU_SYSTEM_FLOWCHART.md)**: Complete workflow diagrams
- **[User Guide](docs/MENU_SYSTEM_USER_GUIDE.md)**: Detailed documentation
- **[Implementation Summary](MENU_SYSTEM_IMPLEMENTATION_SUMMARY.md)**: Technical details
- **[Main README](README.md)**: Project overview

## Troubleshooting

### "Data not prepared"

```bash
python scripts/download/download_interactive.py
python scripts/download/prepare_data.py
```

### "No credentials found"

```bash
python kinetra_menu.py
# Select: 1 → 1 (Login → Select MetaAPI Account)
```

### "Module not found"

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Support

- Check logs: `logs/`
- Review results: `data/results/`
- Full documentation: `docs/`

---

**Quick Tip:** Always start with `--dry-run` to preview test matrix before running long tests!

```bash
python e2e_testing_framework.py --full --dry-run
```
