# Kinetra Interactive CLI & E2E Testing - Quick Reference

## 🚀 Getting Started

### Installation
```bash
cd Kinetra
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Launch Interactive Menu
```bash
python kinetra_menu.py
```

## 📋 Menu Navigation

### Main Menu Options
```
1. Login & Authentication       - MetaAPI account setup
2. Exploration Testing          - Hypothesis & theorem generation
3. Backtesting                  - ML/RL EA validation
4. Data Management              - Download, prepare, validate data
5. System Status & Health       - Monitor system health
0. Exit
```

### Quick Workflows

#### 1. First-Time Setup
```
Main Menu → Option 1 (Login & Authentication)
  → Option 1 (Select MetaAPI Account)
  → Follow prompts to configure account
```

#### 2. Run Quick Exploration
```
Main Menu → Option 2 (Exploration Testing)
  → Option 1 (Quick Exploration)
  → Review configuration
  → Confirm (Y)
```

#### 3. Download Data
```
Main Menu → Option 4 (Data Management)
  → Option 1 (Auto-Download for Configuration)
  → Select asset classes
  → Select timeframes
```

#### 4. Check System Health
```
Main Menu → Option 5 (System Status & Health)
  → Option 1 (Current System Health)
```

## 🧪 E2E Testing Framework

### Command-Line Interface

#### Quick Tests
```bash
# Quick validation (15 min)
python e2e_testing_framework.py --quick --dry-run

# Preview without running
python e2e_testing_framework.py --quick --dry-run
```

#### Preset Tests
```bash
# Test specific asset class
python e2e_testing_framework.py --asset-class crypto --dry-run

# Test specific agent type
python e2e_testing_framework.py --agent-type ppo --dry-run

# Test specific timeframe
python e2e_testing_framework.py --timeframe H1 --dry-run

# Full system test (preview)
python e2e_testing_framework.py --full --dry-run
```

#### Custom Configuration Tests
```bash
# Crypto + Forex focused (60 tests)
python e2e_testing_framework.py \
  --config configs/e2e_examples/crypto_forex_focused.json \
  --dry-run

# Single instrument deep dive (15 tests)
python e2e_testing_framework.py \
  --config configs/e2e_examples/single_instrument_test.json \
  --dry-run

# Agent comparison study (90 tests)
python e2e_testing_framework.py \
  --config configs/e2e_examples/agent_comparison.json \
  --dry-run
```

### Actually Run Tests (Remove --dry-run)
```bash
# Run quick validation
python e2e_testing_framework.py --quick

# Run custom configuration
python e2e_testing_framework.py --config configs/e2e_examples/crypto_forex_focused.json
```

## 🔧 Creating Custom E2E Configurations

### Basic Template
```json
{
  "name": "my_test",
  "description": "Description of test",
  "asset_classes": ["crypto", "forex"],
  "instruments": ["top_3"],
  "timeframes": ["H1", "H4"],
  "agent_types": ["ppo", "dqn"],
  "episodes": 100,
  "parallel_execution": true,
  "auto_data_management": true,
  "statistical_validation": true,
  "monte_carlo_runs": 100
}
```

### Instruments Options
```json
"instruments": ["all"]                    // All instruments
"instruments": ["top_5"]                  // Top 5 per asset class
"instruments": ["BTCUSD", "EURUSD"]      // Specific symbols
```

### Save and Run
```bash
# Save as my_config.json
python e2e_testing_framework.py --config my_config.json --dry-run
```

## ✅ Testing the System

### Run All Tests
```bash
# Basic menu tests
python tests/test_menu_system.py

# Comprehensive workflow tests
python tests/test_menu_workflow.py
```

### Expected Output
```
Basic menu tests: 6/6 PASS ✅
Workflow tests: 10/10 PASS ✅
```

## 📊 Test Matrix Size Examples

| Configuration | Combinations | Duration (Parallel) | Use Case |
|--------------|--------------|---------------------|----------|
| Quick validation | 12 | 15 min | Smoke test |
| Crypto focused | 150 | 3 hours | Asset class validation |
| Agent comparison | 90 | 2 hours | Agent evaluation |
| Single instrument | 15 | 20 min | Deep dive |
| Full system | 2250+ | 12+ hours | Complete validation |

## 🎯 Common Tasks

### Task 1: Validate a New Trading Strategy
```bash
# 1. Create custom config
cat > configs/my_strategy.json << EOF
{
  "name": "my_strategy_test",
  "description": "Test my new strategy",
  "asset_classes": ["crypto"],
  "instruments": ["BTCUSD"],
  "timeframes": ["H1", "H4"],
  "agent_types": ["ppo"],
  "episodes": 200,
  "parallel_execution": true,
  "auto_data_management": true,
  "statistical_validation": true,
  "monte_carlo_runs": 100
}
EOF

# 2. Preview test matrix
python e2e_testing_framework.py --config configs/my_strategy.json --dry-run

# 3. Run the test
python e2e_testing_framework.py --config configs/my_strategy.json
```

### Task 2: Quick System Health Check
```bash
# Launch menu
python kinetra_menu.py

# Navigate: 5 → 1 → 0 → 0
# (System Status → Current Health → Back → Exit)
```

### Task 3: Download Missing Data
```bash
# Launch menu
python kinetra_menu.py

# Navigate: 4 → 3 → 0 → 0
# (Data Management → Check & Fill Missing Data → Back → Exit)
```

## 📚 Documentation References

- **Menu User Guide**: `MENU_SYSTEM_QUICK_START.md`
- **E2E Examples**: `configs/e2e_examples/README.md`
- **Implementation Verification**: `MENU_IMPLEMENTATION_VERIFICATION.md`
- **Main README**: `README.md`

## 🐛 Troubleshooting

### Menu Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify imports
python -c "import kinetra_menu; print('OK')"
```

### E2E Framework Errors
```bash
# Check framework
python -c "import e2e_testing_framework; print('OK')"

# Verify configuration syntax
python -m json.tool configs/e2e_examples/crypto_forex_focused.json
```

### Tests Failing
```bash
# Re-run tests with verbose output
python tests/test_menu_system.py
python tests/test_menu_workflow.py
```

## 💡 Tips

1. **Always use --dry-run first** to preview test matrices
2. **Start with quick validation** before running full tests
3. **Enable parallel execution** for 4x speedup
4. **Use auto data management** to avoid missing data errors
5. **Check system status** regularly during long test runs

## 🔗 Quick Links

```bash
# Launch menu
python kinetra_menu.py

# Quick E2E test
python e2e_testing_framework.py --quick --dry-run

# Run all tests
python tests/test_menu_system.py && python tests/test_menu_workflow.py

# View example configs
ls -l configs/e2e_examples/
```

---

**Need Help?** See the full documentation in the docs/ directory or run tests to verify your setup.
