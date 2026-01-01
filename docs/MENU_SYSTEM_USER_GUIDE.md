# Kinetra Menu System - User Guide

## Overview

The Kinetra Menu System provides a comprehensive, user-friendly interface for all trading system workflows, from data management to exploration testing to backtesting. Built on first principles with automated data management and comprehensive E2E testing support.

## Quick Start

### Launch the Menu

```bash
python kinetra_menu.py
```

This will display the main menu with options for:
1. Login & Authentication
2. Exploration Testing (Hypothesis & Theorem Generation)
3. Backtesting (ML/RL EA Validation)
4. Data Management
5. System Status & Health

## Workflows

### 1. Login & Authentication

Before running any tests, authenticate with MetaAPI:

```
Main Menu → 1. Login & Authentication → 1. Select MetaAPI Account
```

This will:
- List available MetaAPI accounts
- Display account details (broker, server, type)
- Save credentials securely (encrypted)
- Verify connection

### 2. Exploration Testing

Exploration discovers what works where by **measuring, not assuming**.

#### Quick Exploration (Recommended for First-Time Users)

```
Main Menu → 2. Exploration Testing → 1. Quick Exploration
```

Preset configuration:
- Asset Classes: Crypto + Forex
- Instruments: Top 3 per class
- Timeframes: H1, H4
- Agent: PPO
- Episodes: 100

Duration: ~30-60 minutes

#### Custom Exploration

```
Main Menu → 2. Exploration Testing → 2. Custom Exploration
```

Full control over:
- Asset classes (crypto, forex, indices, metals, commodities)
- Instruments (all, top N, or custom selection)
- Timeframes (M15, M30, H1, H4, D1)
- Agent types (PPO, DQN, Linear, Berserker, Triad)
- Number of episodes

The system will:
1. **Auto-manage data** (download, verify, prepare)
2. **Run exploration** (discovery methods, statistical validation)
3. **Generate hypotheses** (what works where)
4. **Validate theorems** (p < 0.01 required)
5. **Save results** (comprehensive reports)

#### Scientific Discovery Suite

```
Main Menu → 2. Exploration Testing → 3. Scientific Discovery Suite
```

Advanced discovery methods:
- Hidden dimension discovery (PCA/ICA)
- Chaos theory analysis (Lyapunov, Hurst exponent)
- Adversarial filtering (GAN-style)
- Meta-learning feature discovery

#### Agent Comparison

```
Main Menu → 2. Exploration Testing → 4. Agent Comparison
```

Compare agent performance:
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- Linear Q-Learning
- Berserker Strategy
- Triad System

### 3. Backtesting

Validate discovered strategies with realistic cost modeling.

#### Quick Backtest

```
Main Menu → 3. Backtesting → 1. Quick Backtest
```

Automatically:
- Loads best performing agents from exploration
- Runs on test set
- Generates performance report
- Displays key metrics (Omega, Z-factor, CHS)

#### Custom Backtesting

```
Main Menu → 3. Backtesting → 2. Custom Backtesting
```

Full configuration:
- Testing mode (Virtual, Demo Account, Historical)
- Agent/strategy selection
- Risk parameters (max drawdown, CHS threshold)
- Instruments and timeframes

Features:
- MT5-accurate friction (spread, commission, slippage)
- Walk-forward validation
- Monte Carlo simulation (100 runs)
- Efficiency metrics (MFE/MAE, Pythagorean distance)

#### Monte Carlo Validation

```
Main Menu → 3. Backtesting → 3. Monte Carlo Validation
```

Statistical robustness testing:
- 100+ simulation runs
- Confidence intervals
- Statistical significance (p < 0.01)

### 4. Data Management

Automated data handling with integrity checks.

#### Auto-Download for Configuration

```
Main Menu → 4. Data Management → 1. Auto-Download for Configuration
```

The system will:
1. Analyze your test configuration
2. Identify required instruments and timeframes
3. Check for existing data
4. Download missing data
5. Verify integrity
6. Prepare data (train/test split)

#### Check & Fill Missing Data

```
Main Menu → 4. Data Management → 3. Check & Fill Missing Data
```

Scans for:
- Missing timeframes (e.g., has M15/H1/H4 but missing M30)
- Data gaps (non-weekend gaps)
- Corrupted files

Offers to:
- Download missing timeframes
- Re-download files with gaps
- Fill all gaps

#### Data Integrity Check

```
Main Menu → 4. Data Management → 4. Data Integrity Check
```

Validates:
- CSV format and required columns
- Data completeness (minimum bars)
- Data quality (no negative prices, valid OHLC)
- No duplicate timestamps or NaN values

### 5. System Status & Health

Monitor system health and performance.

```
Main Menu → 5. System Status & Health
```

View:
- Current system health (CHS)
- Recent test results
- Data summary
- Performance metrics

## End-to-End Testing

For comprehensive system testing across all combinations.

### Quick Validation

```bash
python e2e_testing_framework.py --quick --dry-run
```

Test matrix: 12 combinations
Duration: ~15 minutes

### Asset Class Testing

```bash
python e2e_testing_framework.py --asset-class crypto
```

Test all instruments, timeframes, and agents for specific asset class.

Example for crypto:
- Test matrix: 150 combinations
- Duration: ~3.1 hours

### Agent Type Testing

```bash
python e2e_testing_framework.py --agent-type ppo
```

Test specific agent across all asset classes, instruments, and timeframes.

Example for PPO:
- Test matrix: 220 combinations
- Duration: ~4.6 hours

### Full System Test

```bash
python e2e_testing_framework.py --full
```

Complete E2E test across all combinations:
- All asset classes (crypto, forex, indices, metals, commodities)
- All instruments (50+ symbols)
- All timeframes (M15, M30, H1, H4, D1)
- All agents (PPO, DQN, Linear)

Estimated duration: 24-48 hours

### Dry Run Mode

Test matrix generation without running:

```bash
python e2e_testing_framework.py --quick --dry-run
python e2e_testing_framework.py --asset-class crypto --dry-run
python e2e_testing_framework.py --full --dry-run
```

Shows:
- Test configuration
- Number of combinations
- Estimated duration
- First 10 test cases

### Custom Configuration

Create a JSON configuration file:

```json
{
  "name": "custom_test",
  "description": "Custom E2E test",
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

Run with:

```bash
python e2e_testing_framework.py --config custom_test.json
```

## Performance Targets

All tests validate against these targets:

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |

## Philosophy

### First Principles
- No magic numbers or fixed thresholds
- Physics-based features (energy, entropy, damping)
- Rolling, adaptive distributions
- Let data guide decisions

### Statistical Rigor
- All theorems validated (p < 0.01)
- Monte Carlo simulation (100+ runs)
- Confidence intervals
- PBO (Probability of Backtest Overfitting)

### Automated Workflows
- Auto-detect required data
- Download missing data
- Verify integrity
- Prepare for testing
- No manual intervention needed

### Security & Safety
- Encrypted credential storage
- Atomic file operations
- Circuit breakers (CHS < 0.55 → halt)
- Risk-of-Ruin gates
- Non-linear risk management

## Troubleshooting

### "Data not prepared"

Run the data preparation workflow:

```bash
python scripts/download/download_interactive.py    # Download data
python scripts/download/check_data_integrity.py    # Check integrity
python scripts/download/prepare_data.py            # Prepare data
```

Or use the menu:
```
Main Menu → 4. Data Management → 2. Manual Download
Main Menu → 4. Data Management → 4. Data Integrity Check
Main Menu → 4. Data Management → 5. Prepare Data
```

### "No credentials found"

Authenticate first:

```
Main Menu → 1. Login & Authentication → 1. Select MetaAPI Account
```

### "Testing framework not available"

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Examples

### Example 1: First-Time Setup

```bash
# 1. Launch menu
python kinetra_menu.py

# 2. Select: 1. Login & Authentication
#    Select: 1. Select MetaAPI Account
#    Choose your account from the list

# 3. Select: 2. Exploration Testing
#    Select: 1. Quick Exploration
#    Confirm to run

# System will:
# - Auto-download data
# - Run exploration
# - Generate report

# 4. Review results in data/results/
```

### Example 2: Custom Exploration

```bash
# 1. Launch menu
python kinetra_menu.py

# 2. Select: 2. Exploration Testing
#    Select: 2. Custom Exploration

# 3. Asset Classes: Select "a. All asset classes"
# 4. Instruments: Select "a. All instruments"
# 5. Timeframes: Select "a. All timeframes"
# 6. Agents: Select "a. All agents"
# 7. Episodes: Enter "100"

# 8. Confirm to run

# System will:
# - Check for data (auto-download if missing)
# - Run comprehensive exploration
# - Generate hypotheses and validate theorems
# - Save detailed reports
```

### Example 3: E2E Testing

```bash
# Quick validation (15 minutes)
python e2e_testing_framework.py --quick

# Asset class test (3 hours)
python e2e_testing_framework.py --asset-class crypto

# Full system test (24-48 hours)
python e2e_testing_framework.py --full
```

## Advanced Usage

### Parallel Execution

For faster testing, E2E framework supports parallel execution:

```python
config = E2ETestConfig(
    # ... other config ...
    parallel_execution=True  # Enable parallel workers
)
```

### Custom Instrument Selection

Instead of "all" or "top_N", specify exact instruments:

```json
{
  "instruments": ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD"]
}
```

### Statistical Validation

All tests include statistical validation by default:

```python
config = E2ETestConfig(
    # ... other config ...
    statistical_validation=True,  # Enabled by default
    monte_carlo_runs=100  # Number of MC simulations
)
```

## Support

For issues or questions:
1. Check the flowchart: `docs/MENU_SYSTEM_FLOWCHART.md`
2. Review the main README: `README.md`
3. Check test results: `data/results/`
4. Review logs: `logs/`

## Files

- `kinetra_menu.py` - Main menu interface
- `e2e_testing_framework.py` - End-to-end testing framework
- `docs/MENU_SYSTEM_FLOWCHART.md` - Comprehensive flowchart
- `kinetra/workflow_manager.py` - Workflow management
- `kinetra/testing_framework.py` - Scientific testing framework

---

**Remember**: Kinetra is built on first principles with rigorous validation. Every decision is justified mathematically, tested statistically, and validated continuously.
