#!/usr/bin/env python3
"""
Fast Multi-Instrument RL Training with Prometheus Metrics

Trains across multiple instruments/timeframes simultaneously.
Physics features fed to neural network - RL discovers the patterns.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.metrics_server import start_metrics_server, RLMetrics
from kinetra.rl_gpu_trainer import DQN, ReplayBuffer, TrainingConfig, TradingEnv, PhysicsFeatureComputer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def load_instrument(path: str) -> tuple:
    """Load and compute features for one instrument."""
    name = Path(path).stem
    print(f"  Loading {name}...")
    data = load_csv_data(path)
    fc = PhysicsFeatureComputer()
    features = fc.compute(data)
    return name, data, features


def train_fast(data_paths: list, n_episodes: int = 100, metrics_port: int = 8000):
    """Fast training across multiple instruments."""

    print("=" * 70)
    print("KINETRA FAST MULTI-INSTRUMENT RL TRAINING")
    print("=" * 70)

    metrics = start_metrics_server(metrics_port)

    # Load all instruments in parallel
    print(f"\nLoading {len(data_paths)} instruments...")
    instruments = []
    for path in data_paths:
        name, data, features = load_instrument(path)
        instruments.append((name, data, features))

    # Create environments
    envs = []
    for name, data, features in instruments:
        env = TradingEnv(data, features)
        envs.append((name, env))
        print(f"  {name}: {len(data)} bars, {env.state_dim} state dims")

    # Use first env for network dimensions
    state_dim = envs[0][1].state_dim
    action_dim = envs[0][1].action_dim

    # Fast training config
    config = TrainingConfig(
        learning_rate=3e-4,      # Higher LR for faster learning
        batch_size=128,          # Larger batches
        gamma=0.95,              # Shorter horizon
        epsilon_start=0.8,       # Start with less exploration
        epsilon_end=0.05,
        epsilon_decay=0.99,      # Faster decay
        buffer_size=20000,
        target_update_freq=50,
        n_episodes=n_episodes,
    )

    # Single network learns from all instruments
    q_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size, device)

    epsilon = config.epsilon_start
    total_steps = 0

    print(f"\nTraining for {n_episodes} episodes across {len(envs)} instruments...")
    print(f"Monitor: http://localhost:{metrics_port}/metrics")
    print("-" * 70)

    best_total_pnl = float('-inf')

    for episode in range(n_episodes):
        episode_stats = {'trades': 0, 'wins': 0, 'pnl': 0, 'mfe': [], 'mae': []}
        losses = []

        # Train on each instrument this episode
        for inst_name, env in envs:
            state = env.reset()
            done = False

            while not done:
                # Action selection
                action = 0  # Initialize before conditional block
                if np.random.random() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    with torch.no_grad():
                        q = q_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                        action = q.argmax().item()

                next_state, reward, done, info = env.step(action)
                buffer.push(state, action, reward, next_state, done)

                # Track trades
                if info.get('trade_pnl') is not None:
                    pnl = info['trade_pnl']
                    episode_stats['trades'] += 1
                    episode_stats['pnl'] += pnl
                    if pnl > 0:
                        episode_stats['wins'] += 1
                    episode_stats['mfe'].append(info.get('mfe', 0))
                    episode_stats['mae'].append(info.get('mae', 0))
                    metrics.record_trade(pnl)

                # Training step
                if len(buffer) >= config.batch_size:
                    batch = buffer.sample(config.batch_size)
                    states, actions, rewards, next_states, dones = batch

                    # Convert to tensors with NaN handling
                    states_np = np.array(states)
                    next_states_np = np.array(next_states)

                    # Replace NaN/inf with 0.5 (neutral value for percentiles)
                    states_np = np.nan_to_num(states_np, nan=0.5, posinf=1.0, neginf=0.0)
                    next_states_np = np.nan_to_num(next_states_np, nan=0.5, posinf=1.0, neginf=0.0)

                    # Clip to reasonable range (percentiles should be 0-1, but position info may vary)
                    states_np = np.clip(states_np, -10, 10)
                    next_states_np = np.clip(next_states_np, -10, 10)

                    states = torch.FloatTensor(states_np).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(np.clip(rewards, -10, 10)).to(device)  # Clip rewards too
                    next_states = torch.FloatTensor(next_states_np).to(device)
                    dones = torch.FloatTensor(dones).to(device)

                    with torch.no_grad():
                        next_q = target_net(next_states).max(1)[0]
                        targets = rewards + config.gamma * next_q * (1 - dones)
                        # Clip targets to prevent explosion
                        targets = torch.clamp(targets, -100, 100)

                    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                    loss = torch.nn.functional.huber_loss(current_q, targets)  # Huber loss more stable

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    losses.append(loss.item())

                if total_steps % config.target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

                state = next_state
                total_steps += 1

        # Epsilon decay
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        # Episode metrics
        n_trades = episode_stats['trades']
        win_rate = (episode_stats['wins'] / n_trades * 100) if n_trades > 0 else 0
        total_pnl = episode_stats['pnl']
        avg_loss = np.mean(losses) if losses else 0
        avg_mfe = np.mean(episode_stats['mfe']) if episode_stats['mfe'] else 0
        avg_mae = np.mean(episode_stats['mae']) if episode_stats['mae'] else 0
        mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

        # Update Prometheus metrics
        rl_metrics = RLMetrics(
            episode=episode + 1,
            total_trades=n_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=total_pnl / n_trades if n_trades > 0 else 0,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            mfe_mae_ratio=mfe_mae_ratio,
            mfe_captured=0,
            epsilon=epsilon,
            loss=avg_loss,
            reward=0,
        )
        metrics.update_rl_metrics(rl_metrics)
        metrics.complete_episode()

        # Feature importance
        if episode % 5 == 0:
            w = q_net.network[0].weight.abs().mean(dim=0).detach().cpu().numpy()
            features = ['energy', 'damping', 'entropy', 'acceleration', 'jerk', 'impulse',
                       'liquidity', 'buying_pressure', 'reynolds', 'viscosity',
                       'angular_momentum', 'potential_energy', 'torque', 'market_reynolds',
                       'range_position', 'flow_consistency', 'roc']
            total = w.sum()
            if total > 0:
                imp = {f: float(w[i] / total) for i, f in enumerate(features[:len(w)])}
                metrics.update_feature_importance(imp)

        if total_pnl > best_total_pnl:
            best_total_pnl = total_pnl

        # Print every episode (fast training)
        print(f"Ep {episode+1:3d} | Trades: {n_trades:3d} | "
              f"WR: {win_rate:5.1f}% | PnL: {total_pnl:+8.2f}% | "
              f"ε: {epsilon:.3f} | Loss: {avg_loss:.4f}")

    print("-" * 70)
    print(f"Training complete! Best P&L: {best_total_pnl:.2f}%")

    # Final feature importance
    w = q_net.network[0].weight.abs().mean(dim=0).detach().cpu().numpy()
    features = ['energy', 'damping', 'entropy', 'acceleration', 'jerk', 'impulse',
               'liquidity', 'buying_pressure', 'reynolds', 'viscosity',
               'angular_momentum', 'potential_energy', 'torque', 'market_reynolds',
               'range_position', 'flow_consistency', 'roc']
    print("\nFEATURE IMPORTANCE (RL discovered):")
    total = w.sum()
    sorted_imp = sorted([(f, w[i]/total) for i, f in enumerate(features[:len(w)])],
                        key=lambda x: x[1], reverse=True)
    for name, val in sorted_imp[:10]:
        bar = "█" * int(val * 50)
        print(f"  {name:<18}: {val*100:5.1f}% {bar}")

    return q_net


def main():
    project_root = Path(__file__).parent.parent

    # Find all trading data files
    data_files = [
        str(f) for f in project_root.glob("*.csv")
        if 'BTCUSD' in f.name or 'COPPER' in f.name
    ]

    if not data_files:
        print("No trading data found!")
        return

    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  - {Path(f).name}")

    train_fast(
        data_paths=data_files,
        n_episodes=50,  # Fast: 50 episodes
        metrics_port=8001,  # Different port to avoid conflicts
    )


if __name__ == "__main__":
    main()
