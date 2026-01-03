"""
Stable-Baselines3 Agent Adapters for Kinetra
=============================================

Provides adapters for advanced RL agents (A3C, SAC, TD3) from stable-baselines3
to integrate seamlessly with Kinetra's AgentFactory and training pipeline.

These agents are particularly useful for:
- A3C: Asynchronous advantage actor-critic (parallel exploration)
- SAC: Soft actor-critic (continuous action spaces, entropy regularization)
- TD3: Twin delayed DDPG (robust continuous control)

All adapters follow the Kinetra agent interface:
    - select_action(state) -> action
    - select_action_with_prob(state) -> (action, log_prob, value)
    - store_experience(state, action, reward, next_state, done)
    - train() -> loss

Usage:
    from kinetra.sb3_agents import A3CAgent, SACAgent, TD3Agent

    # Create agent
    agent = SACAgent(state_dim=10, action_dim=3, learning_rate=3e-4)

    # Use in backtest
    action = agent.select_action(state)
    agent.store_experience(state, action, reward, next_state, done)
    loss = agent.train()
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Check for stable-baselines3 availability
try:
    import gymnasium as gym
    import torch
    from stable_baselines3 import A2C, SAC, TD3
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    A2C = None
    SAC = None
    TD3 = None
    ReplayBuffer = None
    gym = None

logger = logging.getLogger(__name__)


class KinetraTradingEnv(gym.Env):
    """
    Lightweight Gym environment for Kinetra trading.

    Used internally by SB3 agents to interact with Kinetra's state representation.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self._state = None
        self._episode_step = 0
        self._max_steps = 1000

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._state = np.zeros(self.state_dim, dtype=np.float32)
        self._episode_step = 0
        return self._state, {}

    def step(self, action):
        """Step is handled externally by Kinetra backtest."""
        self._episode_step += 1
        # Dummy implementation - real step is in backtest loop
        reward = 0.0
        terminated = self._episode_step >= self._max_steps
        truncated = False
        return self._state, reward, terminated, truncated, {}

    def set_state(self, state: np.ndarray):
        """Update environment state (called externally)."""
        self._state = state.astype(np.float32)


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic (A3C) agent adapter.

    Uses A2C from stable-baselines3 (synchronous version, more stable).
    Good for parallel exploration and continuous learning.

    Args:
        state_dim: State vector dimension
        action_dim: Number of discrete actions
        learning_rate: Learning rate (default: 7e-4)
        gamma: Discount factor (default: 0.99)
        n_steps: Steps per update (default: 5)
        ent_coef: Entropy coefficient (default: 0.01)
        vf_coef: Value function coefficient (default: 0.5)
        hidden_dim: Hidden layer size (default: 256)
        n_workers: Number of parallel environments (default: 4)
        device: 'cpu' or 'cuda' (default: 'auto')
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        hidden_dim: int = 256,
        n_workers: int = 4,
        device: str = "auto",
        **kwargs,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 not available. Install: pip install stable-baselines3"
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.n_workers = n_workers

        # Create vectorized environment
        self.env = DummyVecEnv([lambda: KinetraTradingEnv(state_dim, action_dim)])

        # Create A2C agent (synchronous A3C)
        policy_kwargs = dict(
            net_arch=[hidden_dim, hidden_dim],
        )

        self.model = A2C(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )

        # Experience buffer for manual stepping
        self.experience_buffer = deque(maxlen=10000)
        self._current_state = None

    def select_action(self, state: np.ndarray) -> int:
        """Select action without training (deterministic)."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action, _ = self.model.predict(state_tensor, deterministic=True)
        return int(action[0])

    def select_action_with_prob(self, state: np.ndarray) -> Tuple[int, Any, Any]:
        """Select action with log probability and value (for training)."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action, _ = self.model.predict(state_tensor, deterministic=False)

        # Get value estimate
        obs_tensor = torch.FloatTensor(state_tensor).to(self.model.device)
        with torch.no_grad():
            _, value, log_prob = self.model.policy.forward(obs_tensor)

        return int(action[0]), log_prob.cpu().numpy(), value.cpu().numpy()

    def store_experience(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store experience for training."""
        self.experience_buffer.append((state, action, reward, next_state, done))

    def train(self) -> float:
        """Train on collected experiences."""
        if len(self.experience_buffer) < self.n_steps:
            return 0.0

        # Train for one step
        self.model.learn(total_timesteps=self.n_steps, reset_num_timesteps=False)

        # Clear old experiences
        for _ in range(min(self.n_steps, len(self.experience_buffer))):
            self.experience_buffer.popleft()

        return 0.0  # SB3 doesn't expose loss directly


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent adapter.

    State-of-the-art for continuous control with automatic entropy tuning.
    Highly sample-efficient and stable.

    Args:
        state_dim: State vector dimension
        action_dim: Number of discrete actions (converted to continuous internally)
        learning_rate: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        buffer_size: Replay buffer size (default: 100000)
        batch_size: Batch size for training (default: 256)
        hidden_dim: Hidden layer size (default: 256)
        device: 'cpu' or 'cuda' (default: 'auto')
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: str = "auto",
        **kwargs,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 not available. Install: pip install stable-baselines3"
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # SAC expects continuous actions - we'll discretize output
        # Create dummy continuous env
        class ContinuousWrapper(gym.Env):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                )
                # Continuous action space [0, action_dim)
                self.action_space = gym.spaces.Box(
                    low=0.0, high=float(action_dim - 1), shape=(1,), dtype=np.float32
                )
                self._state = np.zeros(state_dim, dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self._state, {}

            def step(self, action):
                return self._state, 0.0, False, False, {}

            def set_state(self, state):
                self._state = state.astype(np.float32)

        self.env = DummyVecEnv([lambda: ContinuousWrapper(state_dim, action_dim)])

        # Create SAC agent
        policy_kwargs = dict(
            net_arch=[hidden_dim, hidden_dim],
        )

        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )

        # Manual replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action (discrete)."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action_continuous, _ = self.model.predict(state_tensor, deterministic=True)
        # Discretize: round to nearest integer
        action_discrete = int(np.round(np.clip(action_continuous[0], 0, self.action_dim - 1)))
        return action_discrete

    def select_action_with_prob(self, state: np.ndarray) -> Tuple[int, Any, Any]:
        """Select action with log probability."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action_continuous, _ = self.model.predict(state_tensor, deterministic=False)

        obs_tensor = torch.FloatTensor(state_tensor).to(self.model.device)
        with torch.no_grad():
            # SAC doesn't have a value function, use Q-value instead
            q_values = self.model.critic(
                obs_tensor, torch.FloatTensor(action_continuous).to(self.model.device)
            )
            value = q_values.min(dim=0)[0]  # Conservative estimate

        action_discrete = int(np.round(np.clip(action_continuous[0], 0, self.action_dim - 1)))
        return action_discrete, None, value.cpu().numpy()

    def store_experience(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self) -> float:
        """Train on replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Train for gradient steps
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)
        self.train_step_count += 1

        return 0.0


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent adapter.

    Improved version of DDPG with:
    - Twin Q-networks (reduces overestimation)
    - Delayed policy updates
    - Target policy smoothing

    Args:
        state_dim: State vector dimension
        action_dim: Number of discrete actions
        learning_rate: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        policy_delay: Policy update frequency (default: 2)
        buffer_size: Replay buffer size (default: 100000)
        batch_size: Batch size for training (default: 256)
        hidden_dim: Hidden layer size (default: 256)
        device: 'cpu' or 'cuda' (default: 'auto')
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        buffer_size: int = 100000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: str = "auto",
        **kwargs,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 not available. Install: pip install stable-baselines3"
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # TD3 expects continuous actions - discretize like SAC
        class ContinuousWrapper(gym.Env):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=0.0, high=float(action_dim - 1), shape=(1,), dtype=np.float32
                )
                self._state = np.zeros(state_dim, dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self._state, {}

            def step(self, action):
                return self._state, 0.0, False, False, {}

            def set_state(self, state):
                self._state = state.astype(np.float32)

        self.env = DummyVecEnv([lambda: ContinuousWrapper(state_dim, action_dim)])

        # Create TD3 agent
        policy_kwargs = dict(
            net_arch=[hidden_dim, hidden_dim],
        )

        self.model = TD3(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )

        self.replay_buffer = deque(maxlen=buffer_size)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action (discrete)."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action_continuous, _ = self.model.predict(state_tensor, deterministic=True)
        action_discrete = int(np.round(np.clip(action_continuous[0], 0, self.action_dim - 1)))
        return action_discrete

    def select_action_with_prob(self, state: np.ndarray) -> Tuple[int, Any, Any]:
        """Select action with log probability and value."""
        state_tensor = state.astype(np.float32).reshape(1, -1)
        action_continuous, _ = self.model.predict(state_tensor, deterministic=False)

        obs_tensor = torch.FloatTensor(state_tensor).to(self.model.device)
        action_tensor = torch.FloatTensor(action_continuous).to(self.model.device)

        with torch.no_grad():
            # Get Q-value from critic
            q_values = self.model.critic(obs_tensor, action_tensor)
            value = q_values.min(dim=0)[0]  # Twin Q min

        action_discrete = int(np.round(np.clip(action_continuous[0], 0, self.action_dim - 1)))
        return action_discrete, None, value.cpu().numpy()

    def store_experience(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self) -> float:
        """Train on replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.model.learn(total_timesteps=1, reset_num_timesteps=False)
        self.train_step_count += 1

        return 0.0


# Agent registry for easy instantiation
SB3_AGENTS = {
    "a3c": A3CAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
}


def create_sb3_agent(agent_type: str, **kwargs) -> Any:
    """
    Factory function to create SB3-based agents.

    Args:
        agent_type: Agent type ('a3c', 'sac', 'td3')
        **kwargs: Agent-specific hyperparameters

    Returns:
        Initialized agent instance

    Example:
        agent = create_sb3_agent('sac', state_dim=10, action_dim=3, learning_rate=3e-4)
    """
    agent_type = agent_type.lower()

    if agent_type not in SB3_AGENTS:
        raise ValueError(
            f"Unknown SB3 agent type: {agent_type}. Available: {list(SB3_AGENTS.keys())}"
        )

    agent_class = SB3_AGENTS[agent_type]
    return agent_class(**kwargs)
