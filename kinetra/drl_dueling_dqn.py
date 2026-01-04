"""
Dueling Deep Q-Network (DQN) for Trading
=========================================

Physics-first, non-prescriptive DRL framework.
Discovers optimal trigger points and directions via shaped rewards.

Architecture:
- Dueling network (separate value and advantage streams)
- Experience replay (decorrelates temporal dependencies)
- Target network (stabilizes training)
- Double DQN (reduces overestimation bias)

Reward Shaping (Non-Prescriptive):
- Base: Net PnL (gross - friction costs)
- Efficiency: MAE/MFE ratio (Pythagorean path efficiency)
- Risk: Drawdown penalty

NO hand-crafted indicators, NO static rules.
Agent discovers patterns from denoised price features.

__version__ = "1.0.0"
__author__ = "Kinetra Project"
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ============================================================================
# DUELING DQN NETWORK
# ============================================================================


class DuelingDQN(nn.Module):
    """
    Dueling architecture separates value and advantage estimation.

    Better credit assignment for trading:
    - Value stream: overall state quality
    - Advantage stream: relative action benefits
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: Optional[List[int]] = None):
        super(DuelingDQN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        # Shared feature extraction
        layers = []
        in_size = state_size
        for h_size in hidden_sizes:
            layers.extend([nn.Linear(in_size, h_size), nn.ReLU()])
            in_size = h_size

        self.feature = nn.Sequential(*layers)

        # Value stream V(s)
        self.value_stream = nn.Linear(in_size, 1)

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Linear(in_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        # Centering advantages improves identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


# ============================================================================
# REPLAY BUFFER
# ============================================================================


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# TRADING ENVIRONMENT
# ============================================================================


@dataclass
class TradeMetrics:
    """Per-trade efficiency metrics."""

    entry_price: float
    exit_price: float
    mfe: float  # Maximum Favorable Excursion
    mae: float  # Maximum Adverse Excursion
    pnl: float
    direction: int  # 1 long, -1 short
    bars_held: int


class TradingEnvironment:
    """
    Vectorized trading environment for DRL.

    State: Normalized returns + position + technical features
    Actions: 0=flat/sell, 1=hold, 2=buy/long
    Reward: PnL - friction + efficiency bonus - risk penalty
    """

    def __init__(
        self,
        prices: np.ndarray,
        prices_denoised: Optional[np.ndarray] = None,
        window_size: int = 50,
        initial_cash: float = 100000,
        commission: float = 0.0005,  # 0.05%
        slippage: float = 0.0001,  # 0.01%
    ):
        self.prices = prices  # Original for PnL
        self.prices_denoised = prices_denoised if prices_denoised is not None else prices
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.cash = self.initial_cash
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.entry_idx = 0
        self.step_idx = self.window_size
        self.net_worth_history = [self.initial_cash]
        self.max_net_worth = self.initial_cash
        self.current_mfe = 0.0
        self.current_mae = 0.0
        self.trades: List[TradeMetrics] = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.

        Features (vectorized, NO loops):
        - Normalized returns from denoised prices
        - Current position
        - Unrealized PnL (if in position)
        """
        start = max(0, self.step_idx - self.window_size)
        end = self.step_idx

        # Returns from denoised prices (cleaner signal)
        window_prices = self.prices_denoised[start:end]
        returns = np.diff(window_prices) / window_prices[:-1]

        # Normalize returns
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Add position and unrealized PnL
        unrealized_pnl = 0.0
        if self.position != 0:
            current_price = self.prices[self.step_idx - 1]
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position

        state = np.concatenate([returns_norm, [self.position, unrealized_pnl]])

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).

        Reward Shaping (Non-Prescriptive):
        - Base: Realized PnL - friction
        - Efficiency: Bonus for high MFE/MAE ratio
        - Risk: Penalty for drawdown
        """
        if self.step_idx >= len(self.prices):
            return self._get_state(), 0.0, True, {}

        current_price = self.prices[self.step_idx]
        prev_position = self.position
        reward = 0.0

        # Execute action: 0=flat, 1=hold, 2=long
        if action == 0:  # Go flat / close
            if prev_position != 0:
                # Close position - calculate realized PnL
                raw_pnl = (current_price - self.entry_price) * prev_position
                friction = abs(raw_pnl) * (self.commission + self.slippage)
                net_pnl = raw_pnl - friction

                # Efficiency bonus (Pythagorean path efficiency)
                mfe = self.current_mfe
                mae = self.current_mae
                excursion = np.sqrt(mfe**2 + mae**2) if (mfe > 0 or mae > 0) else 1.0
                efficiency = abs(net_pnl) / excursion if excursion > 0 else 0.0

                # Reward = PnL + efficiency bonus
                reward = net_pnl + efficiency * 100

                # Record trade
                self.trades.append(
                    TradeMetrics(
                        entry_price=self.entry_price,
                        exit_price=current_price,
                        mfe=mfe,
                        mae=mae,
                        pnl=net_pnl,
                        direction=prev_position,
                        bars_held=self.step_idx - self.entry_idx,
                    )
                )

                # Reset trade tracking
                self.current_mfe = 0.0
                self.current_mae = 0.0

            self.position = 0

        elif action == 2:  # Go long / open
            if prev_position != 1:
                # Close any existing position first
                if prev_position != 0:
                    # Close logic (simplified, same as action 0)
                    pass

                # Open new long position
                self.position = 1
                self.entry_price = current_price
                self.entry_idx = self.step_idx
                self.current_mfe = 0.0
                self.current_mae = 0.0

        elif action == 1:  # Hold
            pass

        # Update MAE/MFE for open positions
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            signed_change = price_change * self.position

            self.current_mfe = max(self.current_mfe, signed_change)
            self.current_mae = min(self.current_mae, signed_change)

            # Small holding reward (unrealized)
            reward += signed_change * 10  # Scale factor

        # Risk penalty: drawdown
        net_worth = self.cash + (current_price * self.position * 1000)  # Scale
        self.net_worth_history.append(net_worth)
        self.max_net_worth = max(self.max_net_worth, net_worth)

        if self.max_net_worth > 0:
            drawdown = (self.max_net_worth - net_worth) / self.max_net_worth
            reward -= drawdown * 1000  # Penalty

        self.step_idx += 1
        done = self.step_idx >= len(self.prices) - 1

        return self._get_state(), reward, done, {}


# ============================================================================
# DQN AGENT
# ============================================================================


class DQNAgent:
    """
    Dueling Double DQN agent with experience replay.

    GPU-accelerated if available.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: Optional[torch.device] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 5000,
        buffer_size: int = 10000,
        batch_size: int = 128,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps_done = 0

        # Networks
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        logger.info(f"DQN Agent initialized on device: {self.device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )

        if training and random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def train_step(self):
        """Single training step with experience replay."""
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN: action selection from policy net, evaluation from target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss and backprop
        loss = F.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

    def update_target_network(self):
        """Copy weights from policy to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: Path):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        logger.info(f"Model loaded from {filepath}")
