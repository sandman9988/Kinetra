# Kinetra Menu System Flowchart

## Overview

This document describes the comprehensive menu system architecture for Kinetra, designed to support flexible testing workflows for exploration (hypothesis & theorem generation) and backtesting (ML/RL EA validation).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     KINETRA MAIN MENU                            │
│                                                                  │
│  1. Login & Authentication                                       │
│  2. Exploration Testing (Hypothesis & Theorem Generation)        │
│  3. Backtesting (ML/RL EA Validation)                           │
│  4. Data Management                                              │
│  5. System Status & Health                                       │
│  6. Exit                                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Flowcharts

### 1. Login & Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   Login & Authentication                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Select MetaAPI Account
             │   ├─→ List available accounts
             │   ├─→ Show account details (broker, server, type)
             │   ├─→ Save credentials (encrypted)
             │   └─→ Verify connection
             │
             ├─→ [2] Test Connection
             │   ├─→ Ping MetaAPI
             │   ├─→ Check data access
             │   └─→ Display status
             │
             └─→ [0] Back to Main Menu
```

### 2. Exploration Testing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│            Exploration Testing (Hypothesis Generation)           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Quick Exploration (Preset Configuration)
             │   ├─→ Select asset class: [Crypto|Forex|All]
             │   ├─→ Auto-select top instruments
             │   ├─→ Auto-select timeframes: [H1, H4]
             │   ├─→ Run universal agent baseline
             │   └─→ Generate performance report
             │
             ├─→ [2] Custom Exploration
             │   ├─→ Select Asset Classes
             │   │   ├─→ [a] All Classes
             │   │   ├─→ [b] Crypto (BTC, ETH, etc.)
             │   │   ├─→ [c] Forex (Major pairs)
             │   │   ├─→ [d] Indices (US30, SPX500, etc.)
             │   │   ├─→ [e] Metals (XAUUSD, XAGUSD)
             │   │   └─→ [f] Commodities (XTIUSD, etc.)
             │   │
             │   ├─→ Select Instruments
             │   │   ├─→ [a] All instruments in selected classes
             │   │   ├─→ [b] Top N per class
             │   │   └─→ [c] Custom selection
             │   │
             │   ├─→ Select Timeframes
             │   │   ├─→ [a] All timeframes (M15, M30, H1, H4, D1)
             │   │   ├─→ [b] Intraday (M15, M30, H1)
             │   │   ├─→ [c] Daily+ (H4, D1)
             │   │   └─→ [d] Custom selection
             │   │
             │   ├─→ Select Agent Types
             │   │   ├─→ [a] All agents
             │   │   ├─→ [b] Universal baseline (PPO)
             │   │   ├─→ [c] Compare agents (PPO vs DQN vs Linear)
             │   │   └─→ [d] Specialized agents (Berserker, Triad)
             │   │
             │   ├─→ Auto-trigger Data Management
             │   │   ├─→ Check for missing data
             │   │   ├─→ Download required data
             │   │   ├─→ Verify data integrity
             │   │   └─→ Prepare data (train/test split)
             │   │
             │   └─→ Run Exploration
             │       ├─→ Discovery methods (PCA, ICA, Chaos)
             │       ├─→ Statistical validation (PBO, CPCV)
             │       ├─→ Generate hypotheses
             │       ├─→ Validate theorems (p < 0.01)
             │       └─→ Save results & reports
             │
             ├─→ [3] Scientific Discovery Suite
             │   ├─→ Hidden dimension discovery (PCA/ICA)
             │   ├─→ Chaos theory analysis (Lyapunov, Hurst)
             │   ├─→ Adversarial filtering
             │   ├─→ Meta-learning feature discovery
             │   └─→ Generate empirical theorems
             │
             ├─→ [4] Agent Comparison
             │   ├─→ Universal baseline vs specialists
             │   ├─→ Cross-agent performance matrix
             │   ├─→ Per-regime analysis
             │   └─→ Statistical significance testing
             │
             ├─→ [5] Measurement Impact Analysis
             │   ├─→ Physics features (energy, entropy, damping)
             │   ├─→ Correlation exploration
             │   ├─→ Feature importance per asset class
             │   └─→ Stacking combinations
             │
             └─→ [0] Back to Main Menu
```

### 3. Backtesting Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                Backtesting (ML/RL EA Validation)                 │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Quick Backtest (Preset Configuration)
             │   ├─→ Load best performing agents from exploration
             │   ├─→ Run on test set
             │   ├─→ Generate performance report
             │   └─→ Display key metrics (Omega, Z-factor, CHS)
             │
             ├─→ [2] Custom Backtesting
             │   ├─→ Select Testing Mode
             │   │   ├─→ [a] Virtual Testing (simulated)
             │   │   ├─→ [b] Demo Account Testing (MT5 demo)
             │   │   └─→ [c] Historical Backtest (test data)
             │   │
             │   ├─→ Select Agent/Strategy
             │   │   ├─→ [a] Load from exploration results
             │   │   ├─→ [b] Select specific agent type
             │   │   └─→ [c] Compare multiple agents
             │   │
             │   ├─→ Select Instruments
             │   │   ├─→ [a] Use exploration configuration
             │   │   └─→ [b] Custom selection
             │   │
             │   ├─→ Select Timeframes
             │   │   ├─→ [a] Use exploration configuration
             │   │   └─→ [b] Custom selection
             │   │
             │   ├─→ Configure Risk Parameters
             │   │   ├─→ Max drawdown threshold
             │   │   ├─→ Position sizing method
             │   │   ├─→ CHS circuit breaker level
             │   │   └─→ Risk-of-Ruin gates
             │   │
             │   └─→ Run Backtest
             │       ├─→ Realistic cost modeling (spread, slippage)
             │       ├─→ MT5-accurate friction
             │       ├─→ Walk-forward validation
             │       ├─→ Monte Carlo simulation (100 runs)
             │       ├─→ Efficiency metrics (MFE/MAE, Pythagorean)
             │       └─→ Generate comprehensive report
             │
             ├─→ [3] Monte Carlo Validation
             │   ├─→ Select agent/strategy
             │   ├─→ Configure runs (default 100)
             │   ├─→ Run parallel simulations
             │   ├─→ Statistical analysis (p < 0.01)
             │   └─→ Generate confidence intervals
             │
             ├─→ [4] Walk-Forward Testing
             │   ├─→ Configure window size
             │   ├─→ Configure step size
             │   ├─→ Run sequential validation
             │   └─→ Plot performance over time
             │
             ├─→ [5] Comparative Analysis
             │   ├─→ Load multiple agents/strategies
             │   ├─→ Same instruments, same period
             │   ├─→ Side-by-side metrics
             │   └─→ Statistical significance tests
             │
             └─→ [0] Back to Main Menu
```

### 4. Data Management Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Management                             │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Auto-Download for Configuration
             │   ├─→ Parse testing configuration
             │   ├─→ Identify required instruments
             │   ├─→ Identify required timeframes
             │   ├─→ Check for existing data
             │   ├─→ Download missing data
             │   ├─→ Verify integrity
             │   └─→ Prepare data (train/test split)
             │
             ├─→ [2] Manual Download
             │   ├─→ Select asset classes
             │   ├─→ Select instruments
             │   ├─→ Select timeframes
             │   ├─→ Download from MetaAPI
             │   └─→ Save to master data directory
             │
             ├─→ [3] Check & Fill Missing Data
             │   ├─→ Scan existing data
             │   ├─→ Detect gaps (non-weekend)
             │   ├─→ Detect missing timeframes
             │   ├─→ Offer to download missing pieces
             │   └─→ Re-download corrupted files
             │
             ├─→ [4] Data Integrity Check
             │   ├─→ Validate CSV format
             │   ├─→ Check required columns
             │   ├─→ Verify data completeness
             │   ├─→ Check for anomalies (negative prices, etc.)
             │   └─→ Generate integrity report
             │
             ├─→ [5] Prepare Data
             │   ├─→ Chronological train/test split (80/20)
             │   ├─→ Filter trading hours by market type
             │   ├─→ Handle missing data (forward fill)
             │   ├─→ Save prepared data
             │   └─→ Generate manifest
             │
             ├─→ [6] Backup & Restore
             │   ├─→ Backup master data
             │   ├─→ Backup prepared data
             │   ├─→ List available backups
             │   └─→ Restore from backup
             │
             └─→ [0] Back to Main Menu
```

### 5. End-to-End Testing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              E2E Testing (All Combinations)                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ [1] Full System Test
             │   ├─→ All asset classes
             │   ├─→ All instruments
             │   ├─→ All timeframes
             │   ├─→ All agent types
             │   ├─→ Data management → Exploration → Backtesting
             │   ├─→ Generate comprehensive report
             │   └─→ Estimated time: 24-48 hours
             │
             ├─→ [2] Asset Class Test
             │   ├─→ Select asset class
             │   ├─→ All instruments in class
             │   ├─→ All timeframes
             │   ├─→ All agent types
             │   └─→ Generate class-specific report
             │
             ├─→ [3] Agent Type Test
             │   ├─→ Select agent type
             │   ├─→ All asset classes
             │   ├─→ All instruments
             │   ├─→ All timeframes
             │   └─→ Generate agent-specific report
             │
             ├─→ [4] Timeframe Test
             │   ├─→ Select timeframe
             │   ├─→ All asset classes
             │   ├─→ All instruments
             │   ├─→ All agent types
             │   └─→ Generate timeframe-specific report
             │
             ├─→ [5] Custom Combination Matrix
             │   ├─→ Select asset classes (multi-select)
             │   ├─→ Select instruments (multi-select)
             │   ├─→ Select timeframes (multi-select)
             │   ├─→ Select agent types (multi-select)
             │   ├─→ Generate test matrix
             │   ├─→ Estimate execution time
             │   ├─→ Run all combinations
             │   └─→ Generate matrix report
             │
             └─→ [0] Back to Main Menu
```

## Key Design Principles

### 1. First Principles Alignment
- No magic numbers or fixed thresholds
- Physics-based features (energy, entropy, damping)
- Statistical validation (p < 0.01)
- Let data guide decisions

### 2. Data Management Automation
- Auto-detect required data from test configuration
- Download missing data automatically
- Verify integrity before testing
- Atomic operations with checksums

### 3. Security & Performance
- Encrypted credential storage
- Atomic file operations
- Circuit breakers (CHS < 0.55 → halt)
- Risk-of-Ruin gates
- Non-linear risk management

### 4. Comprehensive Testing
- Support all combinations of:
  - Asset classes: Crypto, Forex, Indices, Metals, Commodities
  - Instruments: 50+ symbols
  - Timeframes: M15, M30, H1, H4, D1
  - Agents: PPO, DQN, Linear, Berserker, Triad
- Statistical significance required (p < 0.01)
- Monte Carlo validation (100+ runs)

### 5. Reproducibility
- All tests fully reproducible
- State checkpointing
- Resume capability
- Immutable master data
- Versioned results

## Integration Points

### Existing Components Used
1. **WorkflowManager** - Step tracking, atomic operations, backups
2. **TestingFramework** - Scientific testing, statistical validation
3. **DataManagement** - Atomic file ops, integrity checks
4. **AgentFactory** - Agent creation and configuration
5. **BacktestEngine** - Monte Carlo validation, realistic costs
6. **HealthMonitor** - CHS tracking, circuit breakers
7. **RiskManagement** - Risk-of-Ruin, position sizing

### New Components Created
1. **KinetraMenu** - Main menu interface
2. **ExplorationOrchestrator** - Manages exploration workflows
3. **BacktestOrchestrator** - Manages backtesting workflows
4. **DataOrchestrator** - Automated data management
5. **E2ETestRunner** - End-to-end testing across all combinations

## Configuration Files

Menu system uses JSON configuration files for flexibility:

```json
{
  "quick_exploration": {
    "asset_classes": ["crypto", "forex"],
    "instruments_per_class": 3,
    "timeframes": ["H1", "H4"],
    "agent_type": "ppo",
    "episodes": 100
  },
  "quick_backtest": {
    "use_exploration_results": true,
    "monte_carlo_runs": 100,
    "risk_threshold": 0.55
  },
  "e2e_full": {
    "asset_classes": ["crypto", "forex", "indices", "metals", "commodities"],
    "all_instruments": true,
    "all_timeframes": true,
    "all_agents": true,
    "parallel_execution": true
  }
}
```

## Performance Metrics Tracked

All workflows track and report:

1. **Performance Metrics**
   - Omega Ratio (target > 2.7)
   - Z-Factor (target > 2.5)
   - % Energy Captured (target > 65%)
   - Composite Health Score (target > 0.90)
   - % MFE Captured (target > 60%)

2. **Statistical Metrics**
   - P-values (must be < 0.01)
   - Confidence intervals (95%)
   - PBO (Probability of Backtest Overfitting)
   - CPCV results

3. **Efficiency Metrics**
   - MFE/MAE ratios
   - Pythagorean distance
   - Execution quality
   - Slippage capture

4. **System Health**
   - CHS per agent/risk/market
   - Circuit breaker triggers
   - Error rates
   - Retry counts

## Usage Examples

### Example 1: Quick Exploration
```bash
python kinetra_menu.py
# Select: 1. Login & Authentication
# Select: 1. Select MetaAPI Account → Choose account
# Select: 0. Back
# Select: 2. Exploration Testing
# Select: 1. Quick Exploration
# System auto-downloads data, runs exploration, generates report
```

### Example 2: Custom Backtesting
```bash
python kinetra_menu.py
# Select: 3. Backtesting
# Select: 2. Custom Backtesting
# Select: [a] Virtual Testing
# Select: [a] Load from exploration results
# Configure risk parameters
# Run backtest
```

### Example 3: Full E2E Test
```bash
python kinetra_menu.py
# Select: 2. Exploration Testing
# Select: 2. Custom Exploration
# Select all asset classes, all instruments, all timeframes, all agents
# System auto-manages data, runs complete exploration
# Then: 3. Backtesting → Validate all discovered strategies
```

## Error Handling & Recovery

- Auto-retry with exponential backoff
- Checkpoint state every N steps
- Resume from last checkpoint on failure
- Automatic error fixing where possible
- Comprehensive logging
- User notification on critical errors

## Future Enhancements

1. **Web UI** - Browser-based interface
2. **Real-time Monitoring** - Live dashboard during tests
3. **Distributed Execution** - Multi-node parallel testing
4. **Auto-tuning** - Hyperparameter optimization
5. **Cloud Integration** - AWS/GCP execution
6. **Results Database** - SQL storage for historical results
7. **API Interface** - REST API for automation
