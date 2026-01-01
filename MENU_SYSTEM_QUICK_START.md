# Kinetra Menu System - Quick Start

## Installation

```bash
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra
pip install -r requirements.txt
pip install -e ".[dev]"
```

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

## Quick Workflows

### First-Time Setup (5 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 1 → 1 (Login → Select MetaAPI Account)
3. Choose account from list
4. Done!
```

### Quick Exploration (30-60 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 2 → 1 (Exploration → Quick Exploration)
3. System auto-downloads data and runs exploration
4. Review results in data/results/
```

### Quick Backtest (15-30 minutes)

```
1. Launch: python kinetra_menu.py
2. Select: 3 → 1 (Backtesting → Quick Backtest)
3. Loads best agents from exploration
4. Review performance metrics
```

### Check Data Status (1 minute)

```
1. Launch: python kinetra_menu.py
2. Select: 4 → 3 (Data Management → Check & Fill Missing Data)
3. System scans and reports missing data
```

## E2E Testing

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

# Dry run (no execution)
python e2e_testing_framework.py --quick --dry-run

# Asset class test
python e2e_testing_framework.py --asset-class crypto

# Agent type test
python e2e_testing_framework.py --agent-type ppo

# Timeframe test
python e2e_testing_framework.py --timeframe H1

# Full system test
python e2e_testing_framework.py --full

# Custom configuration
python e2e_testing_framework.py --config my_config.json
```

## Configuration Files

### Custom E2E Configuration

Create `my_config.json`:

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

Run with:

```bash
python e2e_testing_framework.py --config my_config.json
```

## Key Concepts

### Asset Classes
- **crypto**: Cryptocurrency (BTC, ETH, etc.)
- **forex**: Foreign Exchange (EUR/USD, GBP/USD, etc.)
- **indices**: Stock Indices (US30, SPX500, etc.)
- **metals**: Precious Metals (XAU/USD, XAG/USD)
- **commodities**: Commodities (Oil, Gas, etc.)

### Timeframes
- **M15**: 15 Minutes
- **M30**: 30 Minutes
- **H1**: 1 Hour
- **H4**: 4 Hours
- **D1**: 1 Day

### Agent Types
- **ppo**: Proximal Policy Optimization
- **dqn**: Deep Q-Network
- **linear**: Linear Q-Learning
- **berserker**: Berserker Strategy
- **triad**: Triad System (Incumbent/Competitor/Researcher)

## Performance Targets

| Metric | Target |
|--------|--------|
| Omega Ratio | > 2.7 |
| Z-Factor | > 2.5 |
| % Energy Captured | > 65% |
| Composite Health Score | > 0.90 |
| % MFE Captured | > 60% |

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
