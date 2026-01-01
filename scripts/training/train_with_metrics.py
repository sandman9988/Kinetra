#!/usr/bin/env python3
"""
RL Training with Prometheus Metrics Export

Run this script and monitor at: http://localhost:8000/metrics

For Grafana:
1. Run Prometheus+Grafana: cd monitoring && docker-compose up -d
2. Access Grafana at: http://localhost:3000 (admin/kinetra)
3. Dashboard is auto-provisioned

Or view raw metrics at: http://localhost:8000/metrics
"""

import sys
from pathlib import Path
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.metrics_server import start_metrics_server, RLMetrics
from kinetra.rl_gpu_trainer import DQN as DQNetwork, ReplayBuffer, TrainingConfig, TradingEnv, PhysicsFeatureComputer

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    print("  (For GPU acceleration with AMD, install PyTorch with ROCm)")


def compute_feature_importance(q_network: DQNetwork, state_dim: int) -> dict:
    """Estimate feature importance from Q-network weights."""
    # Get first layer weights
    first_layer = q_network.network[0]
    if hasattr(first_layer, 'weight'):
        weights = first_layer.weight.abs().mean(dim=0).detach().cpu().numpy()

        # Feature names matching rl_physics_env.py
        feature_names = [
            'energy', 'damping', 'entropy', 'acceleration', 'jerk',
            'impulse', 'liquidity', 'buying_pressure', 'range_position',
            'flow_consistency', 'roc', 'inertia', 'volume_pct', 'reynolds',
            'viscosity', 'angular_momentum', 'potential_energy'
        ]

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total

        importance = {}
        for i, name in enumerate(feature_names[:len(weights)]):
            importance[name] = float(weights[i])

        return importance
    return {}


def train_with_metrics(data_path: str, n_episodes: int = 1000, metrics_port: int = 8000):
    """Train RL with live metrics export."""

    # Start metrics server
    print(f"\n{'='*70}")
    print("KINETRA RL TRAINING WITH PROMETHEUS METRICS")
    print(f"{'='*70}")
    metrics = start_metrics_server(metrics_port)

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = load_csv_data(data_path)
    print(f"Loaded {len(data)} bars")

    # Compute physics features
    print("Computing physics features...")
    feature_computer = PhysicsFeatureComputer()
    feature_df = feature_computer.compute(data)
    print(f"Computed {len(feature_df.columns)} features")

    # Create environment
    env = TradingEnv(data, feature_df)
    state_dim = env.state_dim
    action_dim = env.action_dim

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize network and training components
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        target_update_freq=100,
        n_episodes=n_episodes,
    )

    q_network = DQNetwork(state_dim, action_dim).to(device)
    target_network = DQNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size, device)

    epsilon = config.epsilon_start
    total_steps = 0

    # Set training info
    metrics.set_training_info(
        instrument=Path(data_path).stem,
        device=str(device),
        episodes=str(n_episodes),
        state_dim=str(state_dim),
    )

    print(f"\nStarting training for {n_episodes} episodes...")
    print(f"Monitor at: http://localhost:{metrics_port}/metrics")
    print("-" * 70)

    best_pnl = float('-inf')
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_trades = []
        losses = []

        done = False
        while not done:
            # Epsilon-greedy action selection
            action = 0  # Initialize before conditional block
            if np.random.random() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            buffer.push(state, action, reward, next_state, done)

            # Track trades
            if info.get('trade_pnl') is not None:
                pnl = info['trade_pnl']
                episode_trades.append({
                    'pnl': pnl,
                    'mfe': info.get('mfe', 0),
                    'mae': info.get('mae', 0),
                })
                metrics.record_trade(pnl)

            # Update physics metrics periodically
            if total_steps % 100 == 0 and 'physics_state' in info:
                ps = info['physics_state']
                metrics.update_physics_state(
                    energy_pct=ps.get('energy_pct', 0.5),
                    damping_pct=ps.get('damping_pct', 0.5),
                    entropy_pct=ps.get('entropy_pct', 0.5),
                    reynolds_pct=ps.get('reynolds_pct', 0.5),
                    viscosity_pct=ps.get('viscosity_pct', 0.5),
                    buying_pressure=ps.get('buying_pressure', 0.5),
                )

            # Training step
            if len(buffer) >= config.batch_size:
                batch = buffer.sample(config.batch_size)
                states, actions, rewards, next_states, dones = batch

                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute Q targets
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0]
                    targets = rewards + config.gamma * next_q * (1 - dones)

                # Compute current Q
                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Loss and update
                loss = torch.nn.functional.mse_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Update target network
            if total_steps % config.target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

            episode_reward += reward
            state = next_state
            total_steps += 1

        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        # Episode stats
        n_trades = len(episode_trades)
        if n_trades > 0:
            pnls = [t['pnl'] for t in episode_trades]
            mfes = [t['mfe'] for t in episode_trades]
            maes = [t['mae'] for t in episode_trades]

            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / n_trades * 100
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            avg_mfe = np.mean(mfes) if mfes else 0
            avg_mae = np.mean(maes) if maes else 0
            mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

            # MFE captured (how much of max gain we kept)
            mfe_captured = 0
            for t in episode_trades:
                if t['mfe'] > 0:
                    mfe_captured += (t['pnl'] / t['mfe'] * 100) if t['pnl'] > 0 else 0
            mfe_captured = mfe_captured / n_trades if n_trades > 0 else 0
        else:
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            avg_mfe = 0
            avg_mae = 0
            mfe_mae_ratio = 0
            mfe_captured = 0

        avg_loss = np.mean(losses) if losses else 0

        # Update metrics
        rl_metrics = RLMetrics(
            episode=episode + 1,
            total_trades=n_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            mfe_mae_ratio=mfe_mae_ratio,
            mfe_captured=mfe_captured,
            epsilon=epsilon,
            loss=avg_loss,
            reward=episode_reward,
        )
        metrics.update_rl_metrics(rl_metrics)
        metrics.complete_episode()

        # Update feature importance every 10 episodes
        if episode % 10 == 0:
            importance = compute_feature_importance(q_network, state_dim)
            if importance:
                metrics.update_feature_importance(importance)

        # Track best
        if total_pnl > best_pnl:
            best_pnl = total_pnl

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Ep {episode+1:4d} | Trades: {n_trades:3d} | "
                  f"WR: {win_rate:5.1f}% | PnL: {total_pnl:+7.2f}% | "
                  f"ε: {epsilon:.3f} | Loss: {avg_loss:.4f} | "
                  f"Reward: {avg_reward:.1f}")

    print("-" * 70)
    print(f"Training complete! Best P&L: {best_pnl:.2f}%")

    # Final feature importance
    importance = compute_feature_importance(q_network, state_dim)
    if importance:
        print("\nFEATURE IMPORTANCE (RL learned):")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, value in sorted_imp[:10]:
            print(f"  {name:<20}: {value*100:.1f}%")

    return q_network, episode_rewards


def main():
    # Find data files
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))

    if not csv_files:
        print("No BTCUSD CSV file found in project root")
        return

    print(f"Found data file: {csv_files[0]}")

    # Train with metrics
    train_with_metrics(
        data_path=str(csv_files[0]),
        n_episodes=500,  # Start with 500 episodes
        metrics_port=8000,
    )


if __name__ == "__main__":
    main()
