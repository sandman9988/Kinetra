"""
Neural Network RL Agent for Physics Trading

GPU-ready architecture (numpy now, PyTorch/ROCm when available)

For AMD Radeon 7700 XT:
- Install ROCm: https://rocm.docs.amd.com/
- Install PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

Network: 2-layer MLP with ReLU
Input: All physics features as percentiles
Output: Q-values for each action
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

from numpy import floating


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, exp: Experience):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class NeuralAgent:
    """
    DQN-style neural network agent.

    Uses numpy for computation (GPU-ready for PyTorch port).

    Architecture:
    - Input: state_dim (physics features)
    - Hidden: 64 -> ReLU -> 32 -> ReLU
    - Output: action_dim (Q-values)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int] = (64, 32),
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        buffer_size: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Network weights (Xavier initialization)
        self.W1 = np.random.randn(state_dim, hidden_sizes[0]) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_sizes[0])
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros(hidden_sizes[1])
        self.W3 = np.random.randn(hidden_sizes[1], action_dim) * np.sqrt(2.0 / hidden_sizes[1])
        self.b3 = np.zeros(action_dim)

        # Target network (for stable learning)
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        self.W3_target = self.W3.copy()
        self.b3_target = self.b3.copy()

        # Experience replay
        self.buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.train_steps = 0
        self.target_update_freq = 100

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(float)

    def forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through network."""
        if use_target:
            W1, b1, W2, b2, W3, b3 = (
                self.W1_target, self.b1_target,
                self.W2_target, self.b2_target,
                self.W3_target, self.b3_target
            )
        else:
            W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

        # Layer 1
        z1 = state @ W1 + b1
        a1 = self.relu(z1)

        # Layer 2
        z2 = a1 @ W2 + b2
        a2 = self.relu(z2)

        # Output
        q_values = a2 @ W3 + b3

        return q_values

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.forward(state)
        return int(np.argmax(q_values))

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.buffer.push(Experience(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        Train on batch from replay buffer.

        Returns loss if training occurred, None otherwise.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Current Q-values
        current_q = self.forward(states)

        # Target Q-values (using target network for stability)
        next_q = self.forward(next_states, use_target=True)
        max_next_q = np.max(next_q, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss and gradients
        # Only update Q-value for taken action
        target_full = current_q.copy()
        for i, action in enumerate(actions):
            target_full[i, action] = target_q[i]

        # Backward pass
        loss = self._backward(states, target_full)

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self._update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def _backward(self, states: np.ndarray, targets: np.ndarray) -> floating[Any]:
        """Backward pass with gradient descent."""
        batch_size = states.shape[0]

        # Forward pass (save activations)
        z1 = states @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        output = a2 @ self.W3 + self.b3

        # Loss (MSE)
        loss = np.mean((output - targets) ** 2)

        # Output layer gradients
        d_output = 2 * (output - targets) / batch_size
        dW3 = a2.T @ d_output
        db3 = np.sum(d_output, axis=0)

        # Layer 2 gradients
        d_a2 = d_output @ self.W3.T
        d_z2 = d_a2 * self.relu_grad(z2)
        dW2 = a1.T @ d_z2
        db2 = np.sum(d_z2, axis=0)

        # Layer 1 gradients
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * self.relu_grad(z1)
        dW1 = states.T @ d_z1
        db1 = np.sum(d_z1, axis=0)

        # Gradient clipping
        max_grad = 1.0
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(grad, -max_grad, max_grad, out=grad)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        return loss

    def _update_target_network(self):
        """Copy weights to target network."""
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        self.W3_target = self.W3.copy()
        self.b3_target = self.b3.copy()

    def save(self, path: str):
        """Save model weights."""
        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'epsilon': self.epsilon,
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path: str):
        """Load model weights."""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.W1, self.b1 = weights['W1'], weights['b1']
        self.W2, self.b2 = weights['W2'], weights['b2']
        self.W3, self.b3 = weights['W3'], weights['b3']
        self.epsilon = weights.get('epsilon', self.epsilon_min)
        self._update_target_network()

    def get_feature_importance(self, feature_names: List[str]) -> dict:
        """
        Estimate feature importance from first layer weights.

        Higher absolute weight = more important feature.
        """
        # Sum absolute weights for each input feature
        importance = np.abs(self.W1).sum(axis=1)
        importance = importance / importance.sum()

        return dict(zip(feature_names, importance))


def train_neural_agent(env, n_episodes: int = 200, verbose: bool = True):
    """
    Train neural DQN agent on physics trading environment.
    """
    agent = NeuralAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_sizes=(64, 32),
        learning_rate=0.001,
        batch_size=32,
    )

    episode_stats = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        n_updates = 0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store and train
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                total_loss += loss
                n_updates += 1

            state = next_state
            total_reward += reward

        # Episode stats
        stats = env.get_episode_stats()
        stats['episode'] = episode
        stats['total_reward'] = total_reward
        stats['avg_loss'] = total_loss / n_updates if n_updates > 0 else 0
        stats['epsilon'] = agent.epsilon
        episode_stats.append(stats)

        if verbose and (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Trades={stats['n_trades']}, "
                  f"WR={stats['win_rate']:.1%}, "
                  f"PnL={stats['total_pnl']:.2f}%, "
                  f"Loss={stats['avg_loss']:.4f}, "
                  f"ε={agent.epsilon:.3f}")

    return agent, episode_stats


# GPU acceleration stub (for when PyTorch/ROCm is available)
def create_gpu_agent(state_dim: int, action_dim: int, device: str = 'auto'):
    """
    Create GPU-accelerated agent.

    For AMD Radeon 7700 XT with ROCm:
    1. Install ROCm: sudo apt install rocm-hip-libraries
    2. Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
    3. Set device='cuda' (ROCm uses CUDA API)

    Returns numpy agent if PyTorch not available.
    """
    try:
        import torch
        import torch.nn as nn

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Using device: {device}")

        class DQN(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_dim)
                )

            def forward(self, x):
                return self.net(x)

        model = DQN(state_dim, action_dim).to(device)
        return model, device

    except ImportError:
        print("PyTorch not available. Using numpy agent.")
        return NeuralAgent(state_dim, action_dim), 'cpu'
