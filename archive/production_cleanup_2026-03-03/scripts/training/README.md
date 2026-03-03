# Training Scripts

Scripts for training RL agents and exploring strategy spaces.

## Agent Training

- **`train_rl.py`** - Base RL agent training
- **`train_rl_gpu.py`** - GPU-accelerated RL training
- **`train_rl_physics.py`** - Physics-based RL training
- **`train_fast_multi.py`** - Fast multi-agent training
- **`train_with_metrics.py`** - Training with metric tracking

## Specialized Agents

- **`train_berserker.py`** - Train Berserker strategy
- **`train_sniper.py`** - Train Sniper strategy
- **`train_triad.py`** - Train Tripleganger triad system

## Exploration Framework

- **`explore_universal.py`** - Universal exploration framework
- **`explore_specialization.py`** - Specialized agent exploration
- **`explore_compare_agents.py`** - Compare different agents
- **`explore_interactive.py`** - Interactive exploration
- **`explorer_standalone.py`** - Standalone explorer

## Batch Processing

- **`run_exploration_batch.py`** - Run exploration in batch mode
- **`pathfinder_explore.py`** - Pathfinder exploration
- **`quick_rl_test.py`** - Quick RL testing

## Advanced Training

- **`demo_continual_learning.py`** - Continual learning demo
- **`monitor_training.py`** - Training progress monitor

## Quick Start

```bash
# Train base RL agent
python scripts/training/train_rl.py --data data/master/forex

# Train on GPU
python scripts/training/train_rl_gpu.py --symbol EURUSD

# Quick exploration test
python scripts/training/quick_rl_test.py

# Interactive exploration
python scripts/training/explore_interactive.py
```
