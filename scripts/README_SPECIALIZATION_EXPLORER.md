# Specialization Strategy Explorer

## Overview

The `explore_specialization.py` script systematically compares different agent specialization strategies to discover which approach yields the best edge robustness and generalization.

## Specialization Strategies

### 1. **Universal**
- **Agents**: 1
- **Strategy**: Single agent trained on all instruments and timeframes
- **Pros**: Maximum data for training, universal patterns
- **Cons**: May dilute edge signals, no specialization

### 2. **Asset Class**
- **Agents**: One per market type (forex, crypto, metals, energy, etc.)
- **Strategy**: Separate agents for different asset classes
- **Pros**: Market-specific dynamics (e.g., forex swaps vs crypto volatility)
- **Cons**: Less data per agent

### 3. **Timeframe**
- **Agents**: One per timeframe (H1, H4, D1, etc.)
- **Strategy**: Separate agents for different timeframes
- **Pros**: Time-scale-specific patterns (HFT vs swing trading)
- **Cons**: Limited to instruments with matching timeframes

### 4. **Regime**
- **Agents**: 3 (laminar, underdamped, overdamped)
- **Strategy**: Regime-specialized via reward shaping
- **How**: Each agent gets bonus rewards when trading in its target regime:
  - **Laminar specialist**: High reward for laminar (trend-following)
  - **Underdamped specialist**: High reward for underdamped (mean-reversion)
  - **Overdamped specialist**: Trained to avoid overdamped (choppy conditions)
- **Pros**: Physics-aligned specialization, regime-switching strategies
- **Cons**: Requires regime detection at inference

### 5. **Hybrid (Asset Class + Timeframe)**
- **Agents**: N × M (all combinations)
- **Strategy**: Maximum specialization (e.g., crypto_H1, forex_D1)
- **Pros**: Highly specialized agents
- **Cons**: Many agents, limited data per agent

## Metrics

For each strategy, the explorer computes:

- **Avg Reward**: Mean shaped reward across all episodes
- **Sharpe Ratio**: Reward mean / std (risk-adjusted performance)
- **Consistency** (Edge Robustness): Std/mean of per-instrument rewards (lower = more robust)

## Usage

### Basic Usage

```bash
python3 scripts/explore_specialization.py
```

This will:
1. Auto-discover all instruments in `data/master/`
2. Train all 5 specialization strategies
3. Compare strategies and recommend best approach
4. Save results to `results/specialization_exploration/`

### Custom Configuration

```bash
python3 scripts/explore_specialization.py \
  --data-dir data/master \
  --episodes 50 \
  --lr 0.0001 \
  --gamma 0.99 \
  --output-dir results/my_exploration
```

**Parameters:**
- `--data-dir`: Path to data directory containing CSV files (default: `data/master`)
- `--episodes`: Episodes per agent (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--gamma`: Discount factor (default: 0.99)
- `--output-dir`: Output directory for results (default: `results/specialization_exploration`)

## Output

### Console Output

```
======================================================================
SPECIALIZATION STRATEGY EXPLORATION
======================================================================

[1/5] Running UNIVERSAL strategy...
  Training universal agent on all instruments...

[2/5] Running ASSET CLASS strategy...
  Training asset-class-specific agents...
    Training forex agent (8 instruments)...
    Training crypto agent (3 instruments)...
    ...

[3/5] Running TIMEFRAME strategy...
  Training timeframe-specific agents...
    Training H1 agent (5 instruments)...
    Training H4 agent (4 instruments)...
    ...

[4/5] Running REGIME strategy...
  Training regime-specific agents...
    Training laminar specialist (Laminar flow - low entropy, stable trends)...
    Training underdamped specialist (Underdamped - mean-reverting oscillations)...
    Training overdamped specialist (Overdamped - choppy, high friction)...

[5/5] Running HYBRID strategy...
  Training hybrid (asset class + timeframe) agents...
    Training forex_H1 agent (3 instruments)...
    Training crypto_H4 agent (2 instruments)...
    ...

======================================================================
STRATEGY COMPARISON
======================================================================

Strategy                Agents   Avg Reward       Sharpe  Consistency
----------------------------------------------------------------------
Universal                    1        12.34        0.543        0.234
Asset Class                  4        14.56        0.612        0.198
Timeframe                    3        13.89        0.578        0.215
Regime                       3        15.23        0.641        0.187
Hybrid (Asset+TF)           12        16.78        0.689        0.156

======================================================================
RECOMMENDATIONS
======================================================================
Best Sharpe Ratio: Hybrid (Asset+TF)
  Sharpe: 0.689

Best Consistency (Edge Robustness): Hybrid (Asset+TF)
  Consistency: 0.156 (lower is better)
```

### JSON Results

Results are saved to `results/specialization_exploration/specialization_exploration_YYYYMMDD_HHMMSS.json`:

```json
{
  "universal": {
    "strategy": "universal",
    "n_agents": 1,
    "metrics": {
      "avg_reward": 12.34,
      "sharpe_ratio": 0.543,
      "reward_consistency": 0.234,
      ...
    }
  },
  "asset_class": {
    "strategy": "asset_class",
    "n_agents": 4,
    "agents": {
      "forex": {
        "instruments": ["EURUSD_H1", ...],
        "n_instruments": 8,
        "metrics": {...}
      },
      ...
    }
  },
  ...
  "comparison": [...]
}
```

## Interpretation

### Sharpe Ratio
- **Higher is better**
- Measures risk-adjusted returns
- Sharpe > 0.5 = good, > 1.0 = excellent

### Consistency (Edge Robustness)
- **Lower is better**
- Measures performance variation across instruments
- Consistency < 0.2 = robust edge, > 0.5 = instrument-specific edge

### Recommendations

**If Sharpe and Consistency favor same strategy**: Use that strategy.

**If different strategies win**:
- **Production deployment**: Prioritize consistency (robust edge)
- **Research/exploration**: Prioritize Sharpe (maximum potential)

**Ensemble approach**:
- Deploy top 2-3 strategies in parallel
- Use regime detector to switch between agents
- Allocate capital based on recent performance

## Example Workflow

1. **Run exploration**:
   ```bash
   python3 scripts/explore_specialization.py --episodes 100
   ```

2. **Review results**:
   - Check console output for winning strategy
   - Examine JSON for per-agent details

3. **Deploy winning strategy**:
   - If `hybrid` wins → Use asset-class + timeframe specific agents
   - If `regime` wins → Implement regime detector + regime-specific agents
   - If `universal` wins → Single agent for all instruments

4. **Monitor in production**:
   - Track per-instrument Sharpe ratio
   - Detect when edges degrade (drift)
   - Re-run exploration to adapt

## Integration with Existing Framework

The specialization explorer uses the existing infrastructure:

- **MultiInstrumentLoader**: Auto-discovers and loads all instruments from data directory
- **MultiInstrumentEnv**: Gym-compatible environment for multi-instrument training
- **LinearQAgent**: Linear Q-learning agent (interpretable feature weights)
- **RewardShaper**: Physics-based reward shaping
- **RegimeSpecializedRewardShaper**: Custom reward shaper for regime specialization

## Next Steps After Exploration

1. **Implement winning strategy** in production deployment
2. **Feature importance analysis**: Examine learned weights from LinearQAgent
3. **Cross-validation**: Split instruments into train/test sets, validate generalization
4. **Ensemble deployment**: Combine top strategies for maximum robustness
5. **Continuous monitoring**: Re-run exploration monthly to detect market regime changes

## Technical Details

### Regime Specialization Implementation

Regime-based specialization is challenging because regimes change per-bar, not per-instrument. The explorer solves this via **reward shaping**:

```python
class RegimeSpecializedRewardShaper(RewardShaper):
    """Rewards agent for trading in target regime."""

    def __init__(self, regime_bonus_map: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.regime_bonus_map = regime_bonus_map

    def shape_reward(self, ...):
        # Base reward from parent
        reward = super().shape_reward(...)

        # Regime-specific bonus
        regime = physics_state.iloc[bar_index].get("regime")
        regime_bonus = self.regime_bonus_map.get(regime, 0.0)
        reward += self.regime_bonus_weight * regime_bonus

        return reward
```

Example regime bonus maps:
- **Laminar specialist**: `{'LAMINAR': +0.5, 'UNDERDAMPED': +0.1, 'OVERDAMPED': -0.2}`
- **Underdamped specialist**: `{'UNDERDAMPED': +0.5, 'LAMINAR': +0.1, 'OVERDAMPED': -0.1}`
- **Overdamped specialist**: `{'OVERDAMPED': +0.3, 'UNDERDAMPED': -0.1, 'LAMINAR': -0.2}`

This incentivizes each agent to specialize in its target regime without filtering data.

## References

- **Universal agents**: "One model to rule them all" (simplicity, maximum data)
- **Specialization**: Domain adaptation, transfer learning
- **Regime detection**: Hidden Markov Models, physics-based classification
- **Ensemble methods**: Model stacking, weighted voting
