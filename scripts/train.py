#!/usr/bin/env python3
"""
Unified Training Interface
==========================

Single CLI entry point for all RL training tasks.

Replaces scattered training scripts with one unified interface.

Usage:
    # Train universal PPO agent
    python scripts/train.py --agent ppo --strategy universal --episodes 100

    # Train asset-class DQN specialists  
    python scripts/train.py --agent dqn --strategy asset_class --episodes 200

    # Train on specific instruments
    python scripts/train.py --agent ppo --instruments BTCUSD ETHUSD --episodes 50

Philosophy: Make empirical comparison EASY.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.agent_factory import AgentFactory
from kinetra.unified_trading_env import UnifiedTradingEnv, TradingMode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kinetra Unified Training Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train universal PPO agent
  python scripts/train.py --agent ppo --strategy universal --episodes 100

  # Train DQN on specific instruments
  python scripts/train.py --agent dqn --instruments BTCUSD ETHUSD --episodes 50

  # Train with physics and regime filtering
  python scripts/train.py --agent ppo --use-physics --regime-filter
        """
    )
    
    # Agent configuration
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        choices=['ppo', 'dqn'],
        help='Agent algorithm to train'
    )
    
    # Specialization strategy
    parser.add_argument(
        '--strategy',
        type=str,
        default='universal',
        choices=['universal', 'asset_class', 'timeframe', 'regime'],
        help='Specialization strategy'
    )
    
    # Data selection
    parser.add_argument(
        '--instruments',
        nargs='+',
        default=None,
        help='Specific instruments to train on'
    )
    parser.add_argument(
        '--asset-classes',
        nargs='+',
        default=None,
        help='Filter by asset classes (crypto, forex, metals, etc.)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to data file (CSV with OHLCV)'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    
    # Environment configuration
    parser.add_argument(
        '--use-physics',
        action='store_true',
        default=True,
        help='Use physics engine (default: True)'
    )
    parser.add_argument(
        '--no-physics',
        action='store_true',
        help='Disable physics engine'
    )
    parser.add_argument(
        '--regime-filter',
        action='store_true',
        help='Filter trades by regime'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='exploration',
        choices=['exploration', 'validation', 'production'],
        help='Trading mode'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/training',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-agent',
        action='store_true',
        help='Save trained agent'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def load_data(args) -> pd.DataFrame:
    """
    Load training data.
    
    Args:
        args: Command-line arguments
        
    Returns:
        DataFrame with OHLCV data
    """
    if args.data_file:
        logger.info(f"Loading data from {args.data_file}")
        data = pd.read_csv(args.data_file)
        return data
    
    # Generate dummy data for testing
    logger.warning("No data file specified, generating dummy data")
    np.random.seed(args.seed if args.seed else 42)
    n = 5000
    
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
        'open': prices + np.random.randn(n) * 0.2,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n),
    })
    
    return data


def train_agent(args):
    """
    Train RL agent.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=" * 80)
    logger.info("KINETRA UNIFIED TRAINING")
    logger.info("=" * 80)
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Use Physics: {args.use_physics and not args.no_physics}")
    logger.info("=" * 80)
    
    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
        logger.info(f"Random seed: {args.seed}")
    
    # Load data
    data = load_data(args)
    logger.info(f"Data loaded: {len(data)} bars")
    
    # Create environment
    use_physics = args.use_physics and not args.no_physics
    mode = TradingMode[args.mode.upper()]
    
    env = UnifiedTradingEnv(
        data=data,
        mode=mode,
        use_physics=use_physics,
        regime_filter=args.regime_filter
    )
    logger.info(f"Environment created: obs_dim={env.observation_dim}")
    
    # Create agent
    agent_config = {
        'lr': args.learning_rate,
        'gamma': args.gamma,
    }
    
    agent = AgentFactory.create(
        agent_type=args.agent,
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=agent_config
    )
    logger.info(f"Agent created: {type(agent).__name__}")
    
    # Training loop
    logger.info("\nStarting training...")
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Agent selects action
            if hasattr(agent, 'select_action_with_prob'):
                # PPO-style agent with log probs
                action, log_prob, value = agent.select_action_with_prob(state)
            elif hasattr(agent, 'select_action'):
                action = agent.select_action(state)
                log_prob = None
                value = None
            elif hasattr(agent, 'act'):
                action = agent.act(state)
                log_prob = None
                value = None
            else:
                # Fallback: random action
                action = np.random.randint(env.action_dim)
                log_prob = None
                value = None
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition (PPO agents)
            if hasattr(agent, 'store_transition') and log_prob is not None:
                agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps >= 1000:  # Max steps per episode
                break
        
        # Update agent after episode (PPO-style)
        if hasattr(agent, 'update') and hasattr(agent, 'buffer'):
            if len(agent.buffer.states) > 0:
                agent.update()
        
        episode_rewards.append(total_reward)
        
        # Get environment metrics
        metrics = env.get_metrics()
        episode_metrics.append(metrics)
        
        # Log progress
        if (episode + 1) % 10 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(
                f"Episode {episode+1}/{args.episodes}: "
                f"Reward={total_reward:.2f}, "
                f"Avg10={avg_reward:.2f}, "
                f"Trades={metrics['num_trades']}, "
                f"WinRate={metrics['win_rate']:.2%}"
            )
    
    # Calculate final metrics
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    final_metrics = {
        'agent': args.agent,
        'strategy': args.strategy,
        'episodes': args.episodes,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'total_trades': sum(m['num_trades'] for m in episode_metrics),
        'avg_win_rate': float(np.mean([m['win_rate'] for m in episode_metrics])),
        'final_balance': episode_metrics[-1]['final_balance'] if episode_metrics else 0,
    }
    
    logger.info(f"Average Reward: {final_metrics['avg_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    logger.info(f"Total Trades: {final_metrics['total_trades']}")
    logger.info(f"Average Win Rate: {final_metrics['avg_win_rate']:.2%}")
    logger.info(f"Final Balance: ${final_metrics['final_balance']:.2f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"train_{args.agent}_{args.strategy}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'metrics': final_metrics,
            'episode_rewards': episode_rewards,
            'episode_metrics': episode_metrics
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    
    # Save agent (if requested)
    if args.save_agent:
        agent_file = output_dir / f"agent_{args.agent}_{timestamp}.pkl"
        if hasattr(agent, 'save'):
            agent.save(str(agent_file))
            logger.info(f"✅ Agent saved to: {agent_file}")
        else:
            logger.warning("Agent does not support saving")
    
    logger.info("=" * 80)
    return final_metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        metrics = train_agent(args)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
