# Kinetra Trading System - Complete Workflow

First principles approach to trading system development.

## Philosophy

**THE MARKET TELLS US, WE DON'T ASSUME!**

- Start with ONE universal agent on ALL data
- MEASURE performance across all dimensions
- Let the data reveal where specialization helps
- Never assume asset classes need different treatment upfront

---

## Complete Workflow

### Step 1-4: Download Data

```bash
python scripts/download_interactive.py
```

**Interactive workflow:**
1. Select MetaAPI account (lists available accounts)
2. Select asset classes (forex, crypto, indices, metals, commodities)
3. Select symbols (all, specific, or top N per class)
4. Select timeframes (M15, M30, H1, H4, D1)
5. Download efficiently (skips existing files, shows progress)

**Output:** `data/master/*.csv`

---

### Step 4.5: Check and Fill Missing Data

```bash
python scripts/check_and_fill_data.py
```

**Continuously improve database:**
1. Analyzes existing data (what symbols, what timeframes)
2. Finds missing timeframes (e.g., has M15/H1/H4 but missing M30)
3. Detects data gaps in existing files (non-weekend gaps)
4. Offers to download missing pieces:
   - Download missing timeframes only
   - Re-download files with gaps
   - Do both (recommended)

**Example issues detected:**
- `GBPUSD+: has M15, H1, H4 | missing M30` → offers to download M30
- `XAUUSD_M15_*.csv: 3 gaps (largest: 48.5h)` → offers to re-download

**Output:** Fills gaps in `data/master/*.csv`

---

### Step 5: Data Integrity Check

```bash
python scripts/check_data_integrity.py
```

**Validates data quality:**
1. File exists and readable
2. CSV format valid with required columns (time, OHLC)
3. Data completeness (minimum bars for timeframe)
4. Data gaps (accounting for weekends/holidays)
5. Data quality:
   - No negative/zero prices
   - High >= Low
   - Open/Close within High/Low
   - No duplicate timestamps
   - No NaN values

**Output:** Detailed integrity report

---

### Step 6: Data Preparation

```bash
python scripts/prepare_data.py
```

**Prepares data for training:**
1. **NO PEEKING** - Chronological train/test split (80/20 default)
2. Filter trading hours by market type:
   - Forex: 24/5 (Mon-Fri)
   - Crypto: 24/7
   - Indices: Market hours (simplified 8:00-20:00 UTC)
   - Metals: 23/5
3. Handle missing data (forward fill small gaps)
4. Save manifest with metadata

**Output:**
- `data/prepared/train/*.csv`
- `data/prepared/test/*.csv`
- `data/prepared/manifest.json`

---

## Testing Framework

```bash
python scripts/test_menu.py
```

### 1. EXPLORATION - First Principles Discovery

**Goal:** MEASURE what works where, don't ASSUME

#### 1.1 Universal Agent Baseline

```bash
python scripts/explore_universal.py
```

- Train ONE LinearQ agent on ALL instruments
- Track performance by asset_class × regime × timeframe × volatility
- Establish baseline before exploring specialization

#### 1.2 Compare Agents

```bash
python scripts/explore_compare_agents.py
```

- Test 4 agents: **LinearQ vs PPO vs SAC vs TD3**
- Measure performance across:
  - Asset class (forex, crypto, indices, metals, commodities)
  - Regime (overdamped, underdamped, laminar, breakout)
  - Timeframe (M15, M30, H1, H4)
  - Volatility (low, medium, high)
- Identify best agent per category
- **Outcomes:**
  - One agent dominates all → use universally
  - Different agents excel → specialize accordingly
  - All perform poorly → measurement/feature issue

#### 1.3 Measurement Impact *(TODO)*

Discover which physics measurements (energy, entropy, damping) impact which markets.

#### 1.4 Stacking Analysis *(TODO)*

Test feature combinations per asset class.

#### 1.5 Policy Discovery *(TODO)*

Determine optimal agent policy per discovered specialization.

#### 1.6 Risk Management *(TODO)*

Optimize risk parameters per class/regime.

#### 1.7 Full Exploration *(TODO)*

Run all exploration tests in sequence.

---

### 2. OPTIMIZATION - Replay Learning *(TODO)*

Based on exploration results:
- Replay successful episodes
- Optimize hyperparameters
- Fine-tune risk management per specialization

---

### 3. BACKTESTING - Validate Strategies *(TODO)*

Test discovered strategies on held-out test data:
- MT5-accurate friction modeling
- Realistic slippage and spread
- Proper margin requirements
- Currency conversion
- Swap/rollover costs

---

## Data Flow

```
MetaAPI Cloud
     ↓
download_interactive.py → data/master/*.csv
     ↓
check_and_fill_data.py → fills missing timeframes & gaps
     ↓
check_data_integrity.py → validates quality
     ↓
prepare_data.py → data/prepared/train/*.csv
                → data/prepared/test/*.csv
     ↓
explore_universal.py → results/exploration/universal_baseline_*.json
     ↓
explore_compare_agents.py → results/exploration/agent_comparison_*.json
     ↓
(optimization) → optimized agents
     ↓
(backtesting) → final validation
```

---

## Key Agents

### 1. LinearQAgent (Baseline)
- Simple linear function approximation
- Q(s,a) = w_a · s + b_a
- Interpretable - can see which features drive decisions
- Fast training, low compute

### 2. PPOAgent (On-policy)
- Proximal Policy Optimization
- Clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- Good default choice, stable learning

### 3. SACAgent (Off-policy)
- Soft Actor-Critic
- Maximum entropy RL (explores effectively)
- Twin Q-networks (reduces overestimation)
- Sample efficient, exploration-heavy

### 4. TD3Agent (Deterministic)
- Twin Delayed DDPG
- Deterministic policy
- Delayed policy updates
- Precise control, low variance

---

## Philosophy in Practice

### DON'T:
- ❌ Assume forex needs different treatment than crypto
- ❌ Train separate agents per asset class upfront
- ❌ Hard-code rules for different markets
- ❌ Skip universal baseline

### DO:
- ✅ Start with ONE universal agent
- ✅ MEASURE performance across all dimensions
- ✅ Track breakdown by asset_class × regime × timeframe × volatility
- ✅ Only specialize IF data shows it helps
- ✅ Let the market tell us what works where

---

## Quick Start

```bash
# 1. Download data
python scripts/download_interactive.py

# 2. Fill missing data
python scripts/check_and_fill_data.py

# 3. Check integrity
python scripts/check_data_integrity.py

# 4. Prepare data
python scripts/prepare_data.py

# 5. Run exploration
python scripts/test_menu.py
  → 1. Exploration
    → 2. Compare Agents

# 6. Review results
cat results/exploration/agent_comparison_*.json
```

---

## Results Interpretation

After running agent comparison:

**Scenario 1: One agent dominates**
- PPO performs best across ALL categories
- → Use PPO universally, no specialization needed

**Scenario 2: Different agents excel**
- LinearQ best for forex low-volatility
- SAC best for crypto high-volatility
- → Specialize agents per discovered pattern

**Scenario 3: All agents perform poorly**
- Similar poor performance across all agents
- → Measurement/feature issue, revisit physics features
- → Or reward shaping needs adjustment

**THE MARKET HAS TOLD US!**

---

## Requirements

### Python Packages
```bash
pip install metaapi-cloud-sdk
pip install pandas numpy
pip install torch  # For PPO, SAC, TD3 agents
```

### Environment Variables
```bash
export METAAPI_TOKEN="your-token-here"
export METAAPI_ACCOUNT_ID="your-account-id-here"
```

Get credentials from: https://app.metaapi.cloud/

---

## File Structure

```
Kinetra/
├── scripts/
│   ├── download_interactive.py          # Steps 1-4
│   ├── check_and_fill_data.py          # Step 4.5
│   ├── check_data_integrity.py         # Step 5
│   ├── prepare_data.py                 # Step 6
│   ├── test_menu.py                    # Main testing menu
│   ├── explore_universal.py            # Universal baseline
│   └── explore_compare_agents.py       # 4-agent comparison
├── data/
│   ├── master/                         # Raw downloaded data
│   └── prepared/
│       ├── train/                      # Training data (80%)
│       ├── test/                       # Test data (20%)
│       └── manifest.json               # Metadata
├── results/
│   └── exploration/                    # Exploration results
├── rl_exploration_framework.py         # Core RL framework
├── rl_exploration_framework_agents.py  # PPO, SAC, TD3 agents
└── WORKFLOW.md                         # This file
```

---

**Built with first principles. Validated by markets. No assumptions.**
