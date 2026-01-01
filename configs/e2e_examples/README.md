# E2E Testing Framework - Example Configurations

This directory contains example configuration files for the Kinetra E2E Testing Framework.

## Usage

Run any configuration with:

```bash
python e2e_testing_framework.py --config configs/e2e_examples/<config_file>.json
```

Add `--dry-run` to preview the test matrix without executing:

```bash
python e2e_testing_framework.py --config configs/e2e_examples/<config_file>.json --dry-run
```

## Available Examples

### 1. crypto_forex_focused.json
**Purpose**: Focused test on crypto and forex markets with intraday timeframes

**Configuration**:
- Asset Classes: Crypto, Forex
- Instruments: Top 5 per class
- Timeframes: M15, M30, H1
- Agents: PPO, DQN
- Test Matrix: ~60 combinations
- Estimated Duration: ~1 hour (parallel)

**Use Case**: Quick validation of intraday trading strategies on liquid markets

```bash
python e2e_testing_framework.py --config configs/e2e_examples/crypto_forex_focused.json --dry-run
```

### 2. single_instrument_test.json
**Purpose**: Comprehensive test for a single instrument (BTCUSD) across all parameters

**Configuration**:
- Asset Classes: Crypto
- Instruments: BTCUSD only
- Timeframes: M15, M30, H1, H4, D1 (all)
- Agents: PPO, DQN, Linear
- Test Matrix: 15 combinations
- Estimated Duration: ~20 minutes (parallel)

**Use Case**: Deep dive into BTCUSD behavior across different timeframes and agents

```bash
python e2e_testing_framework.py --config configs/e2e_examples/single_instrument_test.json --dry-run
```

### 3. agent_comparison.json
**Purpose**: Systematic comparison of all agent types across diverse market conditions

**Configuration**:
- Asset Classes: Crypto, Forex, Indices
- Instruments: Top 3 per class
- Timeframes: H1, H4
- Agents: PPO, DQN, Linear, Berserker, Triad (all)
- Test Matrix: ~90 combinations
- Estimated Duration: ~2 hours (parallel)

**Use Case**: Scientific study to determine which agent performs best in different market regimes

```bash
python e2e_testing_framework.py --config configs/e2e_examples/agent_comparison.json --dry-run
```

## Creating Custom Configurations

Create a custom JSON configuration file with the following structure:

```json
{
  "name": "my_custom_test",
  "description": "Description of what this test validates",
  "asset_classes": ["crypto", "forex", "indices", "metals", "commodities"],
  "instruments": ["all"] or ["top_3"] or ["BTCUSD", "EURUSD"],
  "timeframes": ["M15", "M30", "H1", "H4", "D1"],
  "agent_types": ["ppo", "dqn", "linear", "berserker", "triad"],
  "episodes": 100,
  "parallel_execution": true,
  "auto_data_management": true,
  "statistical_validation": true,
  "monte_carlo_runs": 100
}
```

### Configuration Parameters

| Parameter | Type | Description | Examples |
|-----------|------|-------------|----------|
| `name` | string | Unique identifier for the test | `"quick_validation"` |
| `description` | string | Human-readable description | `"Test crypto markets"` |
| `asset_classes` | list | Market categories to test | `["crypto", "forex"]` |
| `instruments` | list | Specific symbols or patterns | `["all"]`, `["top_5"]`, `["BTCUSD"]` |
| `timeframes` | list | Chart timeframes | `["M15", "H1", "D1"]` |
| `agent_types` | list | RL agents to test | `["ppo", "dqn"]` |
| `episodes` | int | Training episodes per test | `100` (default) |
| `parallel_execution` | bool | Enable parallel testing | `true` (recommended) |
| `auto_data_management` | bool | Auto-download missing data | `true` (recommended) |
| `statistical_validation` | bool | Apply statistical tests (p < 0.01) | `true` |
| `monte_carlo_runs` | int | Number of MC simulations | `100` (default) |

### Instruments Specification

The `instruments` field supports three patterns:

1. **All instruments**: `["all"]` - Uses all instruments in selected asset classes
2. **Top N**: `["top_N"]` - Uses top N instruments per asset class (e.g., `["top_3"]`, `["top_5"]`)
3. **Specific list**: `["BTCUSD", "EURUSD", "US30"]` - Uses exactly these symbols

## Test Matrix Size Estimation

The test matrix size is calculated as:
```
combinations = sum(instruments_per_class) × timeframes × agent_types
```

For example:
- 2 asset classes with top 3 each = 6 instruments
- 3 timeframes (H1, H4, D1)
- 2 agent types (PPO, DQN)
- **Total: 6 × 3 × 2 = 36 combinations**

## Performance Guidelines

| Test Size | Combinations | Estimated Duration (Parallel) | Use Case |
|-----------|-------------|-------------------------------|----------|
| Quick | < 20 | < 30 min | Smoke testing, CI/CD |
| Medium | 20-100 | 1-3 hours | Feature validation |
| Large | 100-500 | 3-12 hours | Comprehensive testing |
| Full | 500+ | 12+ hours | Complete system validation |

## Tips

1. **Start small**: Use `--dry-run` to preview test matrix before running
2. **Parallel execution**: Enable for 4x speedup on multi-core systems
3. **Auto data management**: Ensures all required data is downloaded automatically
4. **Statistical validation**: Always enabled for production tests (p < 0.01 threshold)
5. **Custom configs**: Create domain-specific tests for your trading strategies

## See Also

- [E2E Testing Framework](../../e2e_testing_framework.py) - Main implementation
- [Menu System Quick Start](../../MENU_SYSTEM_QUICK_START.md) - Interactive menu usage
- [Scientific Testing Guide](../../docs/SCIENTIFIC_TESTING_GUIDE.md) - Statistical validation methods
