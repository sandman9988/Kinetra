"""
Reinforcement Learning Agent

PPO-based agent that learns adaptive trigger conditions for berserker mode.
No fixed thresholds - learns from physics state percentiles.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Categorical = None


class PPOBuffer:
    """Experience buffer for PPO."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            torch.stack(self.log_probs),
            np.array(self.rewards),
            torch.stack(self.values).squeeze(),
            np.array(self.dones),
        )


class KinetraAgent:
    """
    PPO-based RL agent for berserker trigger learning.

    Learns to identify high-energy release opportunities from
    adaptive physics state percentiles.
    """

    def __init__(
        self,
        state_dim: int = 43,
        action_dim: int = 4,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, using random agent")
            self.network = None
            self.optimizer = None
            self.buffer = PPOBuffer()
            self.device = None
            self.episode_rewards = deque(maxlen=100)
            self.training_step = 0
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Actor-Critic network
        self.network = self._build_network(state_dim, action_dim, hidden_dim)
        self.network = self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Experience buffer
        self.buffer = PPOBuffer()

        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.training_step = 0

    def _build_network(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Build Actor-Critic network."""

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                # Shared feature extractor
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )
                # Actor head (policy)
                self.actor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim),
                    nn.Softmax(dim=-1),
                )
                # Critic head (value)
                self.critic = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )

            def forward(self, state):
                features = self.shared(state)
                action_probs = self.actor(features)
                value = self.critic(features)
                return action_probs, value

            def get_action(self, state):
                action_probs, value = self.forward(state)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob, value

        return ActorCritic()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action based on current state."""
        if self.network is None:
            return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)

            if deterministic:
                action = action_probs.argmax().item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()

        return action

    def select_action_with_prob(self, state: np.ndarray) -> Tuple[int, Any, Any]:
        """Select action and return log prob and value for training."""
        if self.network is None:
            return np.random.randint(0, self.action_dim), None, None

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor)
        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob,
        reward: float,
        value,
        done: bool,
    ):
        """Store transition in buffer."""
        if log_prob is not None:
            self.buffer.store(state, action, log_prob, reward, value, done)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: Any,
        dones: np.ndarray,
        next_value: float = 0.0,
    ) -> Tuple[Any, Any]:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t].item()

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if self.network is None or len(self.buffer.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()

        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                action_probs, values_pred = self.network(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values_pred.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Clear buffer
        self.buffer.clear()
        self.training_step += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        if self.network is not None:
            torch.save(
                {
                    "network_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "training_step": self.training_step,
                },
                path,
            )

    def load(self, path: str):
        """Load model checkpoint."""
        if self.network is not None:
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_step = checkpoint["training_step"]
            self.training_step = checkpoint["training_step"]

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for analysis."""
        if self.network is None:
            return np.ones(self.action_dim) / self.action_dim

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.network(state_tensor)
            return action_probs.cpu().numpy().squeeze()
