"""
Deep RL Agents for Exploration Framework
==========================================

Implements PPO, SAC, and TD3 agents following the BaseAgent interface.
Lightweight implementations optimized for trading exploration.
"""

import numpy as np
from typing import Dict, Tuple
from collections import deque
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    F = None
    Normal = None
    print("⚠️  PyTorch not available. Deep RL agents (PPO, SAC, TD3) will not work.")

from rl_exploration_framework import BaseAgent


# =============================================================================
# Neural Network Components
# =============================================================================

if TORCH_AVAILABLE:
    class MLPNetwork(nn.Module):
        """Simple MLP network for value/policy functions."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dims=(64, 64)):
            super().__init__()
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class GaussianPolicyNetwork(nn.Module):
        """Gaussian policy for continuous action space (adapted for discrete)."""

        def __init__(self, state_dim: int, n_actions: int, hidden_dims=(64, 64)):
            super().__init__()
            self.n_actions = n_actions

            # Shared feature extractor
            self.feature_net = MLPNetwork(state_dim, hidden_dims[-1], hidden_dims[:-1])

            # Action mean and log_std heads
            self.mean_head = nn.Linear(hidden_dims[-1], n_actions)
            self.log_std_head = nn.Linear(hidden_dims[-1], n_actions)

        def forward(self, state):
            features = self.feature_net(state)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            return mean, log_std

        def sample(self, state):
            """Sample action from policy distribution."""
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action_logits = normal.rsample()  # Reparameterization trick
            action = torch.softmax(action_logits, dim=-1)
            return action, action_logits, normal

        def get_action_probs(self, state):
            """Get action probabilities (for discrete action space)."""
            mean, _ = self.forward(state)
            return torch.softmax(mean, dim=-1)


    class ReplayBuffer:
        """Simple replay buffer for off-policy algorithms."""

        def __init__(self, capacity: int = 10000):
            self.buffer = deque(maxlen=capacity)

        def add(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size: int):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]

            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
            dones = np.array([x[4] for x in batch])

            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.

    On-policy actor-critic with clipped objective for stability.
    Good default choice for most environments.

    Key features:
    - Clipped surrogate objective prevents large policy updates
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs of minibatch updates
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_dims: Tuple = (64, 64),
    ):
        super().__init__(state_dim, n_actions)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPO. Install: pip install torch")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Policy and value networks
        self.policy = GaussianPolicyNetwork(state_dim, n_actions, hidden_dims)
        self.value_net = MLPNetwork(state_dim, 1, hidden_dims)

        # Optimizer
        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Episode buffer (for on-policy updates)
        self.episode_buffer = []
        self.update_count = 0

    def _to_tensor(self, x):
        """Convert numpy array to tensor."""
        return torch.FloatTensor(x)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using current policy."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.policy.get_action_probs(state_tensor)

        # Epsilon-greedy exploration (optional)
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        # Sample from policy distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        return action

    def update(self, state, action, reward, next_state, done):
        """Store transition in episode buffer."""
        self.episode_buffer.append((state, action, reward, next_state, done))

        # Update at end of episode
        if done:
            loss = self._update_policy()
            self.episode_buffer.clear()
            return loss

        return 0.0

    def _update_policy(self):
        """Update policy using PPO algorithm."""
        if len(self.episode_buffer) < 4:  # Need minimum trajectory length
            return 0.0

        # Unpack episode
        states = np.array([x[0] for x in self.episode_buffer])
        actions = np.array([x[1] for x in self.episode_buffer])
        rewards = np.array([x[2] for x in self.episode_buffer])
        next_states = np.array([x[3] for x in self.episode_buffer])
        dones = np.array([x[4] for x in self.episode_buffer])

        # Convert to tensors
        states_t = self._to_tensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        # Compute advantages using GAE
        with torch.no_grad():
            values = self.value_net(states_t).squeeze()
            next_values = self.value_net(next_states_t).squeeze()

            # TD errors
            td_target = rewards_t + self.gamma * next_values * (1 - dones_t)
            td_error = td_target - values

            # GAE
            advantages = torch.zeros_like(rewards_t)
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = td_error[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[t]) * gae
                advantages[t] = gae

            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old policy log probs
        with torch.no_grad():
            action_probs_old = self.policy.get_action_probs(states_t)
            old_log_probs = torch.log(action_probs_old.gather(1, actions_t.unsqueeze(1)).squeeze() + 1e-8)

        # PPO update
        self.optimizer.zero_grad()

        # Current policy
        action_probs = self.policy.get_action_probs(states_t)
        log_probs = torch.log(action_probs.gather(1, actions_t.unsqueeze(1)).squeeze() + 1e-8)

        # Ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_pred = self.value_net(states_t).squeeze()
        value_loss = F.mse_loss(value_pred, returns)

        # Entropy bonus
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.optimizer.step()

        self.update_count += 1

        return loss.item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Approximate Q-values using policy and value function."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            value = self.value_net(state_tensor).item()
            action_probs = self.policy.get_action_probs(state_tensor).squeeze().numpy()

        # Approximate Q(s,a) = V(s) + A(s,a), where A(s,a) weighted by policy
        q_values = np.ones(self.n_actions) * value
        q_values = q_values * action_probs  # Weight by policy preference

        return q_values


# =============================================================================
# SAC Agent
# =============================================================================

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) Agent.

    Off-policy actor-critic with entropy regularization.
    Excellent for exploration and sample efficiency.

    Key features:
    - Maximum entropy RL (explores more effectively)
    - Twin Q-networks to reduce overestimation
    - Automatic temperature tuning
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 10000,
        batch_size: int = 64,
        hidden_dims: Tuple = (64, 64),
    ):
        super().__init__(state_dim, n_actions)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SAC. Install: pip install torch")

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Temperature parameter
        self.batch_size = batch_size

        # Networks
        self.policy = GaussianPolicyNetwork(state_dim, n_actions, hidden_dims)

        # Twin Q-networks
        self.q1 = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.q2 = MLPNetwork(state_dim, n_actions, hidden_dims)

        # Target Q-networks
        self.q1_target = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.q2_target = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.update_count = 0

    def _to_tensor(self, x):
        """Convert numpy array to tensor."""
        return torch.FloatTensor(x)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using current policy."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.policy.get_action_probs(state_tensor).squeeze()

        # Sample from policy
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        return action

    def update(self, state, action, reward, next_state, done):
        """Update networks using SAC."""
        # Store in buffer
        self.buffer.add(state, action, reward, next_state, done)

        # Update only if enough samples
        if len(self.buffer) < self.batch_size:
            return 0.0

        return self._update_networks()

    def _update_networks(self):
        """Update Q-networks and policy using SAC algorithm."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = self._to_tensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        # Update Q-functions
        with torch.no_grad():
            # Next action probabilities from current policy
            next_action_probs = self.policy.get_action_probs(next_states_t)

            # Target Q-values (minimum of twin targets)
            next_q1 = self.q1_target(next_states_t)
            next_q2 = self.q2_target(next_states_t)
            next_q = torch.min(next_q1, next_q2)

            # Entropy term
            next_value = (next_action_probs * (next_q - self.alpha * torch.log(next_action_probs + 1e-8))).sum(dim=1)

            # TD target
            q_target = rewards_t + self.gamma * (1 - dones_t) * next_value

        # Current Q-values
        q1_pred = self.q1(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        q2_pred = self.q2(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Q-losses
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        action_probs = self.policy.get_action_probs(states_t)
        q_values = torch.min(self.q1(states_t), self.q2(states_t))

        # Policy loss (maximize Q + entropy)
        policy_loss = (action_probs * (self.alpha * torch.log(action_probs + 1e-8) - q_values)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_count += 1

        return (q1_loss.item() + q2_loss.item() + policy_loss.item()) / 3

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values from twin Q-networks."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            q1 = self.q1(state_tensor).squeeze().numpy()
            q2 = self.q2(state_tensor).squeeze().numpy()

        # Return minimum of twin Q-values
        return np.minimum(q1, q2)


# =============================================================================
# TD3 Agent
# =============================================================================

class TD3Agent(BaseAgent):
    """
    Twin Delayed DDPG (TD3) Agent.

    Deterministic policy with twin Q-networks and target smoothing.
    Best for precise control and reduced overestimation.

    Key features:
    - Twin Q-networks to reduce overestimation bias
    - Delayed policy updates
    - Target policy smoothing
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        buffer_size: int = 10000,
        batch_size: int = 64,
        hidden_dims: Tuple = (64, 64),
    ):
        super().__init__(state_dim, n_actions)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TD3. Install: pip install torch")

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        # Actor (deterministic policy)
        self.actor = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.actor_target = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics
        self.critic1 = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.critic2 = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.critic1_target = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.critic2_target = MLPNetwork(state_dim, n_actions, hidden_dims)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.update_count = 0

    def _to_tensor(self, x):
        """Convert numpy array to tensor."""
        return torch.FloatTensor(x)

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using deterministic policy."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.actor(state_tensor).squeeze()

        # Convert to action probabilities
        action_probs = torch.softmax(action_logits, dim=0).numpy()

        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        # Select best action
        return np.argmax(action_probs)

    def update(self, state, action, reward, next_state, done):
        """Update networks using TD3."""
        # Store in buffer
        self.buffer.add(state, action, reward, next_state, done)

        # Update only if enough samples
        if len(self.buffer) < self.batch_size:
            return 0.0

        return self._update_networks()

    def _update_networks(self):
        """Update critics and actor using TD3 algorithm."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = self._to_tensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        # Update critics
        with torch.no_grad():
            # Target action from target actor
            next_action_logits = self.actor_target(next_states_t)
            next_action_probs = torch.softmax(next_action_logits, dim=1)

            # Target Q-values
            next_q1 = self.critic1_target(next_states_t)
            next_q2 = self.critic2_target(next_states_t)

            # Expected next Q (weighted by action probabilities)
            next_q = torch.min(
                (next_action_probs * next_q1).sum(dim=1),
                (next_action_probs * next_q2).sum(dim=1)
            )

            # TD target
            q_target = rewards_t + self.gamma * (1 - dones_t) * next_q

        # Current Q-values
        q1_pred = self.critic1(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        q2_pred = self.critic2(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Critic losses
        critic1_loss = F.mse_loss(q1_pred, q_target)
        critic2_loss = F.mse_loss(q2_pred, q_target)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        total_loss = critic1_loss.item() + critic2_loss.item()

        # Delayed policy update
        if self.update_count % self.policy_delay == 0:
            # Actor loss (maximize Q)
            action_logits = self.actor(states_t)
            action_probs = torch.softmax(action_logits, dim=1)
            q_values = self.critic1(states_t)

            actor_loss = -(action_probs * q_values).sum(dim=1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            total_loss += actor_loss.item()

            # Soft update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_count += 1

        return total_loss / (2 if self.update_count % self.policy_delay == 0 else 1)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values from twin critics."""
        state_tensor = self._to_tensor(state).unsqueeze(0)

        with torch.no_grad():
            q1 = self.critic1(state_tensor).squeeze().numpy()
            q2 = self.critic2(state_tensor).squeeze().numpy()

        # Return minimum of twin Q-values
        return np.minimum(q1, q2)


__all__ = ['PPOAgent', 'SACAgent', 'TD3Agent']
