"""
Single Symbol RL Environment

Offline RL environment for a single symbol / single timeframe.
Uses PhysicsEngine for state representation and provides dense rewards.

Compatible with Gym-like API for use with standard RL libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .physics_engine import PhysicsEngine


@dataclass
class RewardConfig:
    """
    Reward shaping parameters.
    All terms are applied per-step.

    reward_t = pnl_t
               - lambda_dd * max(0, drawdown_frac_t)
               - lambda_turnover * |pos_t - pos_{t-1}|
    """

    lambda_dd: float = 2.0  # penalty for drawdown (fractional)
    lambda_turnover: float = 0.1  # penalty per unit of position change
    max_dd_clip: float = 0.50  # clip drawdown at 50% for penalty

    # Energy harvesting bonus
    lambda_energy: float = 0.5  # bonus for harvesting in high-energy regimes

    # Path efficiency bonus
    lambda_path_eff: float = 0.2  # bonus for monotone equity path


class SingleSymbolRLEnv:
    """
    Offline RL environment for a single symbol / single timeframe.

    - Data: OHLC (+ optional volume) with DatetimeIndex.
    - State: physics sensors + regime + position info + running risk stats.
    - Actions (discrete):
        0: flat
        1: long 1x
        2: short 1x

    PnL model:
        - Mark-to-market on close-to-close log-returns.
        - Position is applied continuously over the bar [t, t+1].

    This is meant as a fast RL sandbox, not a full execution simulator.

    Observation Space (13 dimensions):
        [0-5]   Physics sensors: KE_pct, Re_m_pct, zeta_pct, Hs_pct, PE_pct, eta_pct
        [6-9]   Regime one-hot: UNDERDAMPED, OVERDAMPED, LAMINAR, BREAKOUT
        [10]    Current position (-1, 0, +1)
        [11]    Realised volatility (recent)
        [12]    Realised CVaR 95 (recent)

    Action Space (discrete, 3 actions):
        0: flat
        1: long 1x
        2: short 1x
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_equity: float = 100_000.0,
        physics_engine: Optional[PhysicsEngine] = None,
        reward_config: Optional[RewardConfig] = None,
        start_offset: int = 300,  # skip first N bars for physics warmup
        max_steps: Optional[int] = None,
        spread_pct: float = 0.0005,  # 5 bps spread for realistic costs
        slippage_pct: float = 0.0002,  # 2 bps slippage
    ) -> None:
        # Basic checks
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        self.data = data.copy()
        self.initial_equity = float(initial_equity)
        self.physics_engine = physics_engine or PhysicsEngine()
        self.reward_config = reward_config or RewardConfig()
        self.start_offset = max(start_offset, self.physics_engine.lookback)
        self.max_steps = max_steps
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct

        # Precompute physics state
        self.physics_state = self.physics_engine.compute_physics_state_from_ohlcv(self.data)
        # Align / trim to common index
        self.data, self.physics_state = self._align_data(self.data, self.physics_state)

        # Internal env state
        self.current_step: int = 0
        self.equity: float = self.initial_equity
        self.peak_equity: float = self.initial_equity
        self.position: float = 0.0  # -1, 0, +1
        self.position_history: List[float] = []
        self.equity_history: List[float] = []
        self.return_history: List[float] = []
        self.trade_count: int = 0

        # Cached lengths
        self.n_bars = len(self.data)
        if self.n_bars <= self.start_offset + 2:
            raise ValueError("Not enough data after warmup offset.")

        # Observation and action space info (Gym compatibility)
        self.observation_dim = 13
        self.action_dim = 3

    # -------------- public API (Gym-style) --------------

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Shape of observation space."""
        return (self.observation_dim,)

    @property
    def action_space_n(self) -> int:
        """Number of discrete actions."""
        return self.action_dim

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to starting state.

        Returns:
            Initial observation (np.ndarray).
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.start_offset
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.position = 0.0
        self.position_history = [self.position]
        self.equity_history = [self.equity]
        self.return_history = []
        self.trade_count = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advance one bar with the given action.

        Args:
            action: 0=flat, 1=long, 2=short

        Returns:
            obs, reward, done, info
        """
        if self.current_step >= self.n_bars - 1:
            # Already at end
            return self._get_observation(), 0.0, True, {"reason": "end_of_data"}

        # Map action -> desired position
        if action == 0:
            new_position = 0.0
        elif action == 1:
            new_position = 1.0
        elif action == 2:
            new_position = -1.0
        else:
            raise ValueError(f"Invalid action {action}")

        prev_position = self.position

        # Apply transaction costs on position change
        position_change = abs(new_position - prev_position)
        transaction_cost = 0.0
        if position_change > 0:
            transaction_cost = self.equity * (self.spread_pct + self.slippage_pct) * position_change
            self.trade_count += 1 if position_change >= 1.0 else 0

        self.position = new_position

        # Compute bar return (log-return close_t -> close_{t+1})
        t = self.current_step
        close_t = float(self.data["close"].iloc[t])
        close_tp1 = float(self.data["close"].iloc[t + 1])

        if close_t <= 0.0:
            bar_ret = 0.0
        else:
            bar_ret = math.log(close_tp1 / close_t)

        # PnL for this bar
        pnl_frac = self.position * bar_ret
        pnl = self.equity * pnl_frac - transaction_cost
        self.equity += pnl

        # Update running stats
        self.return_history.append(pnl_frac)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_history.append(self.equity)
        self.position_history.append(self.position)

        # Drawdown fraction
        if self.peak_equity > 0:
            dd_frac = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        else:
            dd_frac = 0.0

        # Reward shaping
        cfg = self.reward_config
        base_reward = pnl_frac
        dd_penalty = cfg.lambda_dd * min(dd_frac, cfg.max_dd_clip)
        turnover_penalty = cfg.lambda_turnover * position_change

        # Energy harvesting bonus: reward aligned positions in high-energy regimes
        ps_row = self.physics_state.iloc[t]
        ke_pct = float(ps_row.get("KE_pct", 0.5))
        regime = str(ps_row.get("regime", "critical"))

        energy_bonus = 0.0
        if regime in ("underdamped", "laminar", "breakout") and ke_pct > 0.7:
            if self.position != 0 and bar_ret * self.position > 0:
                # Position aligned with move in high-energy regime
                energy_bonus = cfg.lambda_energy * ke_pct * abs(bar_ret)

        reward = base_reward - dd_penalty - turnover_penalty + energy_bonus

        # Advance time
        self.current_step += 1

        done = False
        info: Dict[str, Any] = {
            "equity": self.equity,
            "pnl": pnl,
            "pnl_frac": pnl_frac,
            "drawdown_frac": dd_frac,
            "position": self.position,
            "trade_count": self.trade_count,
            "regime": regime,
            "ke_pct": ke_pct,
        }

        # Episode termination conditions
        if self.current_step >= self.n_bars - 1:
            done = True
            info["reason"] = "end_of_data"

        if self.equity <= 0:
            done = True
            info["reason"] = "bankrupt"
            reward = -10.0  # Large penalty for bankruptcy

        if self.max_steps is not None and (self.current_step - self.start_offset) >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        return self._get_observation(), float(reward), done, info

    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.

        Returns:
            Dict with performance metrics
        """
        if len(self.equity_history) < 2:
            return {}

        equity_arr = np.array(self.equity_history)
        returns_arr = np.array(self.return_history) if self.return_history else np.array([0.0])

        # Sharpe ratio (annualized for H1: 8760 bars/year)
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = float(drawdown.max())

        # CVaR 95
        if len(returns_arr) > 10:
            q05 = np.percentile(returns_arr, 5)
            tail = returns_arr[returns_arr <= q05]
            cvar_95 = float(tail.mean()) if len(tail) > 0 else 0.0
        else:
            cvar_95 = 0.0

        return {
            "final_equity": self.equity,
            "total_return_pct": (self.equity - self.initial_equity) / self.initial_equity * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "cvar_95": cvar_95,
            "trade_count": self.trade_count,
            "bars_traded": len(self.equity_history) - 1,
        }

    # -------------- observation construction --------------

    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from physics + position state.
        """
        t = self.current_step

        ps_row = self.physics_state.iloc[t]

        # Physics sensors (Layer 1)
        sensors = np.array(
            [
                float(ps_row.get("KE_pct", 0.5)),
                float(ps_row.get("Re_m_pct", 0.5)),
                float(ps_row.get("zeta_pct", 0.5)),
                float(ps_row.get("Hs_pct", 0.5)),
                float(ps_row.get("PE_pct", 0.5)),
                float(ps_row.get("eta_pct", 0.5)),
            ],
            dtype=np.float32,
        )

        # Regime one-hot
        regime = str(ps_row.get("regime", "critical"))
        regime_vec = np.zeros(4, dtype=np.float32)
        regime_map = {
            "underdamped": 0,
            "overdamped": 1,
            "laminar": 2,
            "breakout": 3,
        }
        if regime in regime_map:
            regime_vec[regime_map[regime]] = 1.0

        # Position + risk stats
        pos = np.array(
            [
                self.position,  # current position (-1..1)
                self._recent_vol(self.return_history, 64),  # realised vol
                self._recent_cvar(self.return_history, 0.95),  # realised CVaR 95
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([sensors, regime_vec, pos], axis=0)

        # Replace any NaN with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    # -------------- helpers --------------

    @staticmethod
    def _align_data(
        data: pd.DataFrame,
        physics_state: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align data and physics_state on common index and drop missing.
        """
        # inner join on index
        joined = data.join(physics_state, how="inner", rsuffix="_phys")
        # Split back
        data_cols = [c for c in data.columns]
        phys_cols = [c for c in physics_state.columns]
        data_aligned = joined[data_cols]
        phys_aligned = joined[phys_cols]
        return data_aligned, phys_aligned

    @staticmethod
    def _recent_vol(returns: List[float], window: int) -> float:
        if not returns:
            return 0.0
        arr = np.array(returns[-window:], dtype=float)
        if arr.size < 2:
            return 0.0
        return float(np.std(arr))

    @staticmethod
    def _recent_cvar(returns: List[float], alpha: float = 0.95) -> float:
        """
        Empirical CVaR of the last window of returns (negative tail),
        reported as a positive number (loss magnitude).
        """
        if not returns:
            return 0.0
        arr = np.array(returns, dtype=float)
        if arr.size < 10:
            return 0.0
        losses = arr[arr < 0.0]
        if losses.size == 0:
            return 0.0
        q = np.quantile(losses, 1.0 - alpha)  # e.g. 5% quantile of losses
        tail = losses[losses <= q]
        if tail.size == 0:
            return 0.0
        return float(-tail.mean())  # positive loss magnitude


class DummyPhysicsAgent:
    """
    Simple deterministic agent that mimics physics-based rules.

    Use this for baseline testing before implementing RL.
    """

    def select_action(self, obs: np.ndarray) -> int:
        """
        Select action based on physics state.

        Args:
            obs: Observation from SingleSymbolRLEnv

        Returns:
            action: 0=flat, 1=long, 2=short
        """
        # Extract features from observation
        ke_pct = obs[0]
        re_pct = obs[1]
        zeta_pct = obs[2]
        hs_pct = obs[3]
        pe_pct = obs[4]
        eta_pct = obs[5]

        # Regime one-hot (indices 6-9)
        regime_underdamped = obs[6]
        regime_overdamped = obs[7]
        regime_laminar = obs[8]
        regime_breakout = obs[9]

        # Current position
        current_pos = obs[10]

        # Trading rules
        # Only trade in favorable regimes
        in_favorable_regime = (
            regime_underdamped > 0.5 or regime_laminar > 0.5 or regime_breakout > 0.5
        )

        if not in_favorable_regime:
            return 0  # flat in overdamped/critical

        # Energy and friction conditions
        high_energy = ke_pct > 0.6
        low_friction = zeta_pct < 0.5

        if not (high_energy and low_friction):
            return 0  # flat when conditions not met

        # Direction: simple momentum from eta
        if eta_pct > 0.6:
            return 1  # long when efficiency high
        elif eta_pct < 0.4:
            return 2  # short when efficiency low

        return 0  # flat otherwise


def run_episode(
    env: SingleSymbolRLEnv,
    agent: Optional[DummyPhysicsAgent] = None,
    random_policy: bool = False,
) -> Dict[str, Any]:
    """
    Run a single episode with the given agent.

    Args:
        env: SingleSymbolRLEnv instance
        agent: Agent with select_action method (default: DummyPhysicsAgent)
        random_policy: If True, use random actions

    Returns:
        Episode statistics
    """
    if agent is None and not random_policy:
        agent = DummyPhysicsAgent()

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if random_policy:
            action = np.random.randint(0, 3)
        else:
            action = agent.select_action(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward

    stats = env.get_episode_stats()
    stats["total_reward"] = total_reward

    return stats
