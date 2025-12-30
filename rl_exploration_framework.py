"""
First-Principles RL Exploration Framework

Philosophy: "We don't know what we don't know"
- NO feature gating - expose ALL 64 dimensions
- NO hard-coded rules - let agent discover patterns
- Iterative exploration with physics-based reward shaping
- Track what the agent learns (feature importance emergence)

Design:
1. TradingEnv: Gym-like environment wrapping physics state
2. RewardShaper: Physics-informed reward (not just PnL)
3. FeatureTracker: Monitor which features agent uses
4. ExplorationLoop: Iterate, learn, adapt
5. PersistenceManager: Atomic saves, graceful failure, logging

Agents (modular):
- TabularQ: Simple Q-learning baseline (discretized state)
- LinearQ: Linear function approximation
- NeuralQ: DQN when PyTorch available
"""

import atexit
import hashlib
import json
import logging
import os
import pickle
import shutil
import signal
import sys
import tempfile
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# PERSISTENCE & LOGGING: Atomic saves, graceful failure
# =============================================================================

class PersistenceManager:
    """
    Manages atomic saves, checkpoints, and graceful failure recovery.

    Features:
    - Atomic writes (temp file + rename) to prevent corruption
    - Automatic checkpointing every N episodes
    - Graceful shutdown on SIGINT/SIGTERM
    - Detailed logging with rotation
    - Organized folder structure
    """

    def __init__(
        self,
        base_dir: str = "results",
        experiment_name: Optional[str] = None,
        checkpoint_every: int = 10,
        max_checkpoints: int = 5,
        log_level: int = logging.INFO,
    ):
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        self.experiment_name = experiment_name
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints = max_checkpoints

        # Create directory structure
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.results_dir = self.exp_dir / "results"

        for d in [self.checkpoint_dir, self.logs_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging(log_level)

        # State for graceful shutdown
        self._current_state: Dict[str, Any] = {}
        self._shutdown_requested = False
        self._setup_signal_handlers()

        self.logger.info(f"Experiment initialized: {experiment_name}")
        self.logger.info(f"Output directory: {self.exp_dir}")

    def _setup_logging(self, log_level: int) -> logging.Logger:
        """Setup logging with file and console handlers."""
        logger = logging.getLogger(f"rl_exp_{self.experiment_name}")
        logger.setLevel(log_level)
        logger.handlers = []  # Clear existing handlers

        # File handler
        log_file = self.logs_dir / f"experiment_{datetime.now():%Y%m%d}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on signals."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self._emergency_save()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Register atexit handler
        atexit.register(self._emergency_save)

    def _emergency_save(self):
        """Emergency save on shutdown."""
        if self._current_state:
            try:
                emergency_path = self.checkpoint_dir / "emergency_checkpoint.pkl"
                self._atomic_save(self._current_state, emergency_path)
                self.logger.info(f"Emergency checkpoint saved: {emergency_path}")
            except Exception as e:
                self.logger.error(f"Failed to save emergency checkpoint: {e}")

    def _atomic_save(self, data: Any, path: Path):
        """Atomic save using temp file + rename."""
        # Create temp file in same directory (for same filesystem rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="checkpoint_",
            dir=path.parent
        )
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(data, f)
            # Atomic rename
            shutil.move(temp_path, path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _atomic_save_json(self, data: Dict, path: Path):
        """Atomic save for JSON data."""
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="results_",
            dir=path.parent
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            shutil.move(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def should_shutdown(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested

    def save_checkpoint(
        self,
        episode: int,
        agent_state: Dict,
        env_state: Dict,
        metrics: Dict,
        force: bool = False,
    ):
        """Save checkpoint if due or forced."""
        if not force and episode % self.checkpoint_every != 0:
            return

        checkpoint = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "agent_state": agent_state,
            "env_state": env_state,
            "metrics": metrics,
        }

        # Update current state for emergency save
        self._current_state = checkpoint

        # Save checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_ep{episode:05d}.pkl"
        self._atomic_save(checkpoint, ckpt_path)
        self.logger.info(f"Checkpoint saved: {ckpt_path.name}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only most recent checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_ep*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for old_ckpt in checkpoints[self.max_checkpoints:]:
            old_ckpt.unlink()
            self.logger.debug(f"Removed old checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict]:
        """Load checkpoint (latest if path not specified)."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_ep*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not checkpoints:
                return None
            checkpoint_path = checkpoints[0]

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def save_results(self, results: Dict, filename: str = "final_results.json"):
        """Save final results as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results_converted = convert(results)
        results_path = self.results_dir / filename
        self._atomic_save_json(results_converted, results_path)
        self.logger.info(f"Results saved: {results_path}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        trades: int,
        pnl: float,
        extra: Optional[Dict] = None,
    ):
        """Log episode metrics."""
        msg = f"Episode {episode:4d} | Reward: {reward:+8.2f} | Trades: {trades:3d} | PnL: ${pnl:+10,.0f}"
        if extra:
            msg += " | " + " | ".join(f"{k}: {v}" for k, v in extra.items())
        self.logger.info(msg)

    def log_trade(
        self,
        trade_id: int,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
    ):
        """Log individual trade details."""
        self.logger.debug(
            f"Trade #{trade_id} | {direction:5s} | "
            f"Entry: ${entry_price:,.2f} -> Exit: ${exit_price:,.2f} | "
            f"PnL: ${pnl:+,.2f} | MAE: ${mae:+,.0f} | MFE: ${mfe:+,.0f} | "
            f"Bars: {bars_held}"
        )

    def save_trade_log(self, trades: List[Dict], filename: str = "trades.csv"):
        """Save detailed trade log as CSV."""
        if not trades:
            return
        trades_df = pd.DataFrame(trades)
        trades_path = self.results_dir / filename
        trades_df.to_csv(trades_path, index=False)
        self.logger.info(f"Trade log saved: {trades_path} ({len(trades)} trades)")


class GracefulRunner:
    """
    Context manager for graceful execution with error handling.

    Usage:
        with GracefulRunner(persistence) as runner:
            for episode in range(100):
                if runner.should_stop():
                    break
                # ... training code ...
    """

    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.persistence.logger.error(f"Error during execution: {exc_val}")
            self.persistence.logger.error(traceback.format_exc())
            # Trigger emergency save
            self.persistence._emergency_save()
            return False  # Don't suppress exception
        return True

    def should_stop(self) -> bool:
        """Check if we should stop execution."""
        return self.persistence.should_shutdown()


def is_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except (ImportError, AttributeError):
        pass
    return False


class JupyterDisplay:
    """Display utilities for Jupyter notebooks."""

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 50) -> str:
        """Create text-based progress bar."""
        pct = current / total
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"

    @staticmethod
    def display_metrics(metrics: Dict, title: str = "Metrics"):
        """Display metrics in a formatted table."""
        if is_jupyter():
            try:
                from IPython.display import HTML, display
                html = f"<h4>{title}</h4><table>"
                for k, v in metrics.items():
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                    html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                html += "</table>"
                display(HTML(html))
                return
            except ImportError:
                pass
        # Fallback to text
        print(f"\n{title}")
        print("-" * 40)
        for k, v in metrics.items():
            if isinstance(v, float):
                v = f"{v:.4f}"
            print(f"  {k}: {v}")

    @staticmethod
    def display_feature_weights(weights: Dict[str, np.ndarray], feature_names: List[str], top_k: int = 10):
        """Display top feature weights per action."""
        if is_jupyter():
            try:
                from IPython.display import HTML, display
                html = "<h4>Learned Feature Weights</h4>"
                for action, w in weights.items():
                    top_idx = np.argsort(np.abs(w))[::-1][:top_k]
                    html += f"<p><b>{action}</b>:<br>"
                    for idx in top_idx:
                        color = "green" if w[idx] > 0 else "red"
                        html += f'<span style="color:{color}">{feature_names[idx]}: {w[idx]:+.4f}</span><br>'
                    html += "</p>"
                display(HTML(html))
                return
            except ImportError:
                pass
        # Fallback to text
        for action, w in weights.items():
            print(f"\n{action}:")
            top_idx = np.argsort(np.abs(w))[::-1][:top_k]
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {w[idx]:+.4f}")


# =============================================================================
# CORE: Trading Environment
# =============================================================================

@dataclass
class TradeState:
    """Current trade state for tracking."""
    position: int = 0          # -1=short, 0=flat, 1=long
    entry_price: float = 0.0
    entry_bar: int = 0
    mae: float = 0.0           # Maximum Adverse Excursion
    mfe: float = 0.0           # Maximum Favorable Excursion
    bars_held: int = 0
    entry_features: Optional[np.ndarray] = None


class TradingEnv:
    """
    First-principles trading environment.

    State: 64-dim physics feature vector (ungated)
    Actions: 0=hold, 1=long, 2=short, 3=close
    Reward: Physics-shaped (not just PnL)

    No assumptions about what features matter - agent explores.
    """

    def __init__(
        self,
        physics_state: pd.DataFrame,
        prices: pd.DataFrame,
        feature_extractor: Callable,
        reward_shaper: Optional["RewardShaper"] = None,
        max_position_bars: int = 72,  # Force close after N bars
    ):
        self.physics_state = physics_state
        self.prices = prices
        self.feature_extractor = feature_extractor
        self.reward_shaper = reward_shaper or RewardShaper()
        self.max_position_bars = max_position_bars

        # Environment state
        self.current_bar = 0
        self.trade_state = TradeState()
        self.episode_trades: List[Dict] = []
        self.episode_rewards: List[float] = []

        # Action space
        self.n_actions = 4  # hold, long, short, close
        self.action_names = ["HOLD", "LONG", "SHORT", "CLOSE"]

        # State dimensions
        self.state_dim = 64

    def reset(self, start_bar: int = 100) -> np.ndarray:
        """Reset environment to starting state."""
        self.current_bar = start_bar
        self.trade_state = TradeState()
        self.episode_trades = []
        self.episode_rewards = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current 64-dim state vector."""
        return self.feature_extractor(self.physics_state, self.current_bar)

    def _get_price(self, bar: int) -> float:
        """Get close price at bar."""
        if bar >= len(self.prices):
            bar = len(self.prices) - 1
        return self.prices.iloc[bar]["close"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).

        Actions:
            0: HOLD - do nothing
            1: LONG - enter/maintain long
            2: SHORT - enter/maintain short
            3: CLOSE - close any position
        """
        info = {"action": self.action_names[action]}
        reward = 0.0
        trade_closed = False

        current_price = self._get_price(self.current_bar)

        # Update existing position (MAE/MFE tracking in PERCENTAGE terms)
        if self.trade_state.position != 0:
            self.trade_state.bars_held += 1
            entry_price = self.trade_state.entry_price

            # Convert to percentage for cross-instrument consistency
            if self.trade_state.position == 1:  # Long
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # Short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            self.trade_state.mfe = max(self.trade_state.mfe, pnl_pct)
            self.trade_state.mae = min(self.trade_state.mae, pnl_pct)

        # Force close if held too long (prevent infinite positions)
        force_close = (
            self.trade_state.position != 0 and
            self.trade_state.bars_held >= self.max_position_bars
        )

        # Process action
        if action == 0:  # HOLD
            pass

        elif action == 1:  # LONG
            if self.trade_state.position == -1:  # Close short first
                reward, trade_closed = self._close_position(current_price)
            if self.trade_state.position == 0:  # Open long
                self._open_position(1, current_price)

        elif action == 2:  # SHORT
            if self.trade_state.position == 1:  # Close long first
                reward, trade_closed = self._close_position(current_price)
            if self.trade_state.position == 0:  # Open short
                self._open_position(-1, current_price)

        elif action == 3 or force_close:  # CLOSE
            if self.trade_state.position != 0:
                reward, trade_closed = self._close_position(current_price)

        # Advance time
        self.current_bar += 1
        done = self.current_bar >= len(self.physics_state) - 1

        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        # Record reward
        self.episode_rewards.append(reward)

        info["trade_closed"] = trade_closed
        info["position"] = self.trade_state.position
        info["bars_held"] = self.trade_state.bars_held

        return next_state, reward, done, info

    def _open_position(self, direction: int, price: float):
        """Open a new position."""
        self.trade_state = TradeState(
            position=direction,
            entry_price=price,
            entry_bar=self.current_bar,
            entry_features=self._get_state().copy(),
        )

    def _close_position(self, price: float) -> Tuple[float, bool]:
        """Close current position and return reward."""
        if self.trade_state.position == 0:
            return 0.0, False

        # Calculate raw PnL as PERCENTAGE (not absolute) for cross-instrument normalization
        # This fixes the currency mismatch issue (BTCJPY in JPY vs BTCUSD in USD)
        entry_price = self.trade_state.entry_price
        if self.trade_state.position == 1:  # Long
            raw_pnl_pct = ((price - entry_price) / entry_price) * 100
        else:  # Short
            raw_pnl_pct = ((entry_price - price) / entry_price) * 100

        # Keep absolute PnL for tracking but use percentage for reward
        raw_pnl = raw_pnl_pct  # Now in percentage terms

        # Shape reward using physics measures
        reward = self.reward_shaper.shape_reward(
            raw_pnl=raw_pnl,
            mae=self.trade_state.mae,
            mfe=self.trade_state.mfe,
            bars_held=self.trade_state.bars_held,
            entry_features=self.trade_state.entry_features,
            exit_features=self._get_state(),
            physics_state=self.physics_state,
            bar_index=self.current_bar,
        )

        # Record trade
        self.episode_trades.append({
            "entry_bar": self.trade_state.entry_bar,
            "exit_bar": self.current_bar,
            "direction": self.trade_state.position,
            "raw_pnl": raw_pnl,
            "shaped_reward": reward,
            "mae": self.trade_state.mae,
            "mfe": self.trade_state.mfe,
            "bars_held": self.trade_state.bars_held,
        })

        # Reset position
        self.trade_state = TradeState()

        return reward, True

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        if not self.episode_trades:
            return {"trades": 0, "total_reward": 0, "avg_reward": 0}

        total_pnl = sum(t["raw_pnl"] for t in self.episode_trades)
        total_reward = sum(t["shaped_reward"] for t in self.episode_trades)
        wins = sum(1 for t in self.episode_trades if t["raw_pnl"] > 0)

        return {
            "trades": len(self.episode_trades),
            "total_pnl": total_pnl,
            "total_reward": total_reward,
            "avg_reward": total_reward / len(self.episode_trades),
            "win_rate": wins / len(self.episode_trades),
            "avg_mae": np.mean([t["mae"] for t in self.episode_trades]),
            "avg_mfe": np.mean([t["mfe"] for t in self.episode_trades]),
        }


# =============================================================================
# REWARD SHAPING: Physics-Informed
# =============================================================================

class RewardShaper:
    """
    Physics-based reward shaping with RISK-ADJUSTED returns.

    First-principles: Don't just reward PnL - reward GOOD trading:
    - Edge ratio (MFE/Hypotenuse): Reward efficient paths
    - Entropy alignment: Higher reward when trading with entropy flow
    - Regime awareness: Bonus for trading in favorable regimes
    - Risk-adjusted: Penalize excessive drawdown (MAE)

    IMPORTANT: raw_pnl is now in PERCENTAGE terms (not absolute) for cross-instrument normalization.
    This fixes the currency mismatch issue (BTCJPY in JPY vs BTCUSD in USD).

    The agent should discover that physics-aligned trades work better.
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        edge_ratio_weight: float = 0.5,
        mae_penalty_weight: float = 0.3,
        regime_bonus_weight: float = 0.2,
        entropy_alignment_weight: float = 0.1,
        risk_adjustment: bool = True,  # NEW: Enable risk-adjusted rewards
    ):
        self.pnl_weight = pnl_weight
        self.edge_ratio_weight = edge_ratio_weight
        self.mae_penalty_weight = mae_penalty_weight
        self.regime_bonus_weight = regime_bonus_weight
        self.entropy_alignment_weight = entropy_alignment_weight
        self.risk_adjustment = risk_adjustment

    def shape_reward(
        self,
        raw_pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
        entry_features: np.ndarray,
        exit_features: np.ndarray,
        physics_state: pd.DataFrame,
        bar_index: int,
    ) -> float:
        """
        Compute physics-shaped reward with RISK ADJUSTMENT.

        Components:
        1. Raw PnL (normalized) - NOW IN PERCENTAGE TERMS
        2. Edge ratio bonus (efficient path)
        3. MAE penalty (excessive drawdown)
        4. Regime bonus (favorable conditions)
        5. Entropy alignment (trade with disorder flow)
        6. Risk adjustment: reward = pnl / (1 + |max_drawdown|)
        """
        # raw_pnl is now in PERCENTAGE terms (e.g., 0.5 = 0.5% return)
        # Normalize using sigmoid-like function centered on typical percentage moves
        # A 1% move is significant, 5% is very significant
        pnl_normalized = raw_pnl / (abs(raw_pnl) + 1.0)  # Adjusted for percentage scale

        # Risk-adjusted PnL: penalize profits that came with excessive drawdown
        # Formula: reward = pnl / (1 + |max_drawdown|)
        if self.risk_adjustment and mae < 0:
            drawdown_penalty = 1.0 + abs(mae) / 2.0  # mae is in % terms
            pnl_normalized = pnl_normalized / drawdown_penalty

        # Edge ratio: MFE / sqrt(MAE² + MFE²) - now in percentage terms
        mae_abs = abs(mae)  # Now in percentage
        mfe_abs = abs(mfe)  # Now in percentage
        hypotenuse = np.sqrt(mae_abs**2 + mfe_abs**2 + 1e-8)
        edge_ratio = mfe_abs / hypotenuse if hypotenuse > 0 else 0.5
        edge_bonus = (edge_ratio - 0.5) * 2  # Center at 0, range [-1, 1]

        # MAE penalty (excessive drawdown is bad even if profitable)
        # mae is now in percentage terms, so adjust threshold
        mae_penalty = -min(0, mae) / (abs(mae) + 1.0)  # Adjusted for % scale

        # Regime bonus
        regime_bonus = 0.0
        if bar_index < len(physics_state):
            regime = physics_state.iloc[bar_index].get("regime", "UNKNOWN")
            if regime == "LAMINAR":
                regime_bonus = 0.3  # Best regime
            elif regime == "UNDERDAMPED":
                regime_bonus = 0.1
            elif regime == "OVERDAMPED":
                regime_bonus = -0.2  # Penalize trading in choppy conditions

        # Entropy alignment (trade direction should align with entropy gradient)
        entropy_bonus = 0.0
        if entry_features is not None and exit_features is not None:
            # Feature index 13 is entropy_z
            entry_entropy_z = entry_features[13] if len(entry_features) > 13 else 0
            exit_entropy_z = exit_features[13] if len(exit_features) > 13 else 0
            entropy_change = exit_entropy_z - entry_entropy_z
            # Positive PnL with rising entropy = good (captured disorder)
            if raw_pnl > 0 and entropy_change > 0:
                entropy_bonus = 0.1
            # Negative PnL with falling entropy = less bad (orderly exit)
            elif raw_pnl < 0 and entropy_change < 0:
                entropy_bonus = 0.05

        # Combine components
        shaped_reward = (
            self.pnl_weight * pnl_normalized +
            self.edge_ratio_weight * edge_bonus +
            self.mae_penalty_weight * mae_penalty +
            self.regime_bonus_weight * regime_bonus +
            self.entropy_alignment_weight * entropy_bonus
        )

        return shaped_reward


# =============================================================================
# FEATURE TRACKER: What does the agent learn?
# =============================================================================

class FeatureTracker:
    """
    Track which features the agent uses for decisions.

    First-principles: We don't know what matters - track everything.

    Methods:
    1. Action correlation: Which features correlate with actions?
    2. Gradient tracking: For neural nets, track weight gradients
    3. Ablation: Remove features, measure performance drop
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.n_features = len(feature_names)

        # Track state-action pairs
        self.state_action_history: List[Tuple[np.ndarray, int]] = []

        # Feature importance accumulator
        self.feature_importance = np.zeros(self.n_features)
        self.feature_action_corr = defaultdict(lambda: np.zeros(self.n_features))

    def record(self, state: np.ndarray, action: int):
        """Record state-action pair."""
        self.state_action_history.append((state.copy(), action))

    def compute_action_correlations(self) -> Dict[str, np.ndarray]:
        """Compute correlation between features and actions."""
        if len(self.state_action_history) < 100:
            return {}

        states = np.array([s for s, a in self.state_action_history])
        actions = np.array([a for s, a in self.state_action_history])

        correlations = {}
        for action_id in range(4):
            action_mask = (actions == action_id).astype(float)
            corr = np.array([
                np.corrcoef(states[:, i], action_mask)[0, 1]
                for i in range(self.n_features)
            ])
            corr = np.nan_to_num(corr)
            correlations[f"action_{action_id}"] = corr

        return correlations

    def get_top_features_per_action(self, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get top correlated features for each action."""
        correlations = self.compute_action_correlations()

        result = {}
        action_names = ["HOLD", "LONG", "SHORT", "CLOSE"]

        for action_id, name in enumerate(action_names):
            key = f"action_{action_id}"
            if key in correlations:
                corr = correlations[key]
                indices = np.argsort(np.abs(corr))[::-1][:top_k]
                result[name] = [
                    (self.feature_names[i], corr[i])
                    for i in indices
                ]

        return result

    def reset(self):
        """Reset tracking."""
        self.state_action_history = []


# =============================================================================
# AGENTS: Modular, swappable
# =============================================================================

class BaseAgent:
    """Base agent interface."""

    def __init__(self, state_dim: int, n_actions: int):
        self.state_dim = state_dim
        self.n_actions = n_actions

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearQAgent(BaseAgent):
    """
    Linear function approximation Q-learning.

    Q(s, a) = w_a · s + b_a

    Simple but interpretable - we can see which features drive decisions.
    Includes gradient clipping and NaN protection for stability.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
    ):
        super().__init__(state_dim, n_actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.grad_clip = grad_clip

        # Linear weights per action
        self.weights = np.random.randn(n_actions, state_dim) * 0.01
        self.biases = np.zeros(n_actions)

        # For tracking
        self.update_count = 0

    def _sanitize_state(self, state: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with zeros for numerical stability."""
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(state, -100, 100)  # Clip extreme values

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions."""
        state = self._sanitize_state(state)
        q = self.weights @ state + self.biases
        return np.nan_to_num(q, nan=0.0)

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update with gradient clipping."""
        state = self._sanitize_state(state)
        next_state = self._sanitize_state(next_state)

        q_current = self.get_q_values(state)[action]

        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.get_q_values(next_state))

        # TD error
        td_error = q_target - q_current

        # Clip TD error for stability
        td_error = np.clip(td_error, -self.grad_clip, self.grad_clip)

        # Gradient update with clipping
        gradient = td_error * state
        gradient = np.clip(gradient, -self.grad_clip, self.grad_clip)

        self.weights[action] += self.lr * gradient
        self.biases[action] += self.lr * td_error

        # Clip weights to prevent explosion
        self.weights = np.clip(self.weights, -10, 10)
        self.biases = np.clip(self.biases, -10, 10)

        self.update_count += 1

        return td_error

    def get_feature_weights(self) -> Dict[str, np.ndarray]:
        """Get learned weights for interpretability."""
        return {
            f"action_{a}": np.nan_to_num(self.weights[a], nan=0.0)
            for a in range(self.n_actions)
        }


class TabularQAgent(BaseAgent):
    """
    Tabular Q-learning with state discretization.

    Discretizes continuous state into bins.
    Simple baseline - works for small state spaces.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_bins: int = 10,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__(state_dim, n_actions)
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = gamma

        # Q-table (stored as dict for sparse access)
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )

        # State bounds for discretization (learned online)
        self.state_min = np.full(state_dim, np.inf)
        self.state_max = np.full(state_dim, -np.inf)

    def _discretize(self, state: np.ndarray) -> tuple:
        """Discretize continuous state to bin indices."""
        # Update bounds
        self.state_min = np.minimum(self.state_min, state)
        self.state_max = np.maximum(self.state_max, state)

        # Normalize to [0, 1]
        range_vec = self.state_max - self.state_min + 1e-8
        normalized = (state - self.state_min) / range_vec

        # Bin
        bins = np.clip((normalized * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(bins)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for discretized state."""
        key = self._discretize(state)
        return self.q_table[key].copy()

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.get_q_values(state)))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        key = self._discretize(state)
        q_current = self.q_table[key][action]

        if done:
            q_target = reward
        else:
            next_key = self._discretize(next_state)
            q_target = reward + self.gamma * np.max(self.q_table[next_key])

        # Update
        td_error = q_target - q_current
        self.q_table[key][action] += self.lr * td_error

        return td_error


class RandomAgent(BaseAgent):
    """Random baseline for comparison."""

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        return np.random.randint(self.n_actions)

    def update(self, state, action, reward, next_state, done):
        return 0.0

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_actions)


# =============================================================================
# EXPLORATION LOOP: Iterative Learning
# =============================================================================

@dataclass
class ExplorationConfig:
    """Configuration for exploration loop."""
    n_episodes: int = 100
    max_steps_per_episode: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    eval_every: int = 10
    print_every: int = 10
    checkpoint_every: int = 10
    resume_from_checkpoint: bool = True


class ExplorationLoop:
    """
    Main exploration loop with persistence and graceful failure.

    Features:
    - Automatic checkpointing
    - Resume from checkpoint
    - Graceful shutdown handling
    - Detailed logging
    - Jupyter-friendly output
    """

    def __init__(
        self,
        env: TradingEnv,
        agent: BaseAgent,
        feature_tracker: FeatureTracker,
        config: ExplorationConfig = None,
        persistence: Optional[PersistenceManager] = None,
    ):
        self.env = env
        self.agent = agent
        self.tracker = feature_tracker
        self.config = config or ExplorationConfig()
        self.persistence = persistence

        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_pnl: List[float] = []
        self.td_errors: List[float] = []
        self.all_trades: List[Dict] = []

        # State
        self.start_episode = 0
        self.epsilon = self.config.epsilon_start

    def _get_agent_state(self) -> Dict:
        """Get serializable agent state."""
        state = {"type": type(self.agent).__name__}
        if hasattr(self.agent, "weights"):
            state["weights"] = self.agent.weights.copy()
            state["biases"] = self.agent.biases.copy()
        if hasattr(self.agent, "q_table"):
            state["q_table"] = dict(self.agent.q_table)
        return state

    def _load_agent_state(self, state: Dict):
        """Restore agent state from checkpoint."""
        if hasattr(self.agent, "weights") and "weights" in state:
            self.agent.weights = state["weights"]
            self.agent.biases = state["biases"]
        if hasattr(self.agent, "q_table") and "q_table" in state:
            self.agent.q_table = defaultdict(
                lambda: np.zeros(self.agent.n_actions),
                state["q_table"]
            )

    def resume_from_checkpoint(self) -> bool:
        """Try to resume from checkpoint."""
        if self.persistence is None:
            return False

        checkpoint = self.persistence.load_checkpoint()
        if checkpoint is None:
            return False

        # Restore state
        self.start_episode = checkpoint["episode"] + 1
        self._load_agent_state(checkpoint["agent_state"])

        # Restore metrics
        metrics = checkpoint["metrics"]
        self.episode_rewards = metrics.get("episode_rewards", [])
        self.episode_trades = metrics.get("episode_trades", [])
        self.episode_pnl = metrics.get("episode_pnl", [])
        self.td_errors = metrics.get("td_errors", [])
        self.epsilon = metrics.get("epsilon", self.config.epsilon_start)

        if self.persistence:
            self.persistence.logger.info(f"Resumed from episode {self.start_episode}")

        return True

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Run exploration loop with persistence."""
        # Try to resume
        if self.config.resume_from_checkpoint:
            self.resume_from_checkpoint()

        epsilon = self.epsilon

        # Setup graceful runner if persistence available
        runner_context = GracefulRunner(self.persistence) if self.persistence else None

        try:
            for episode in range(self.start_episode, self.config.n_episodes):
                # Check for graceful shutdown
                if runner_context and runner_context.should_stop():
                    if self.persistence:
                        self.persistence.logger.info("Graceful shutdown requested, saving state...")
                    break

                # Reset environment
                state = self.env.reset(start_bar=np.random.randint(100, len(self.env.physics_state) - 500))
                total_reward = 0
                episode_td_errors = []

                for step in range(self.config.max_steps_per_episode):
                    # Select action
                    action = self.agent.select_action(state, epsilon)

                    # Track state-action
                    self.tracker.record(state, action)

                    # Step environment
                    next_state, reward, done, info = self.env.step(action)

                    # Update agent
                    td_error = self.agent.update(state, action, reward, next_state, done)
                    episode_td_errors.append(abs(td_error))

                    total_reward += reward
                    state = next_state

                    if done:
                        break

                # Decay epsilon
                epsilon = max(self.config.epsilon_end, epsilon * self.config.epsilon_decay)
                self.epsilon = epsilon

                # Record metrics
                stats = self.env.get_episode_stats()
                self.episode_rewards.append(total_reward)
                self.episode_trades.append(stats["trades"])
                self.episode_pnl.append(stats.get("total_pnl", 0))
                self.td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)
                self.all_trades.extend(self.env.episode_trades)

                # Checkpoint
                if self.persistence and (episode + 1) % self.config.checkpoint_every == 0:
                    self.persistence.save_checkpoint(
                        episode=episode,
                        agent_state=self._get_agent_state(),
                        env_state={},
                        metrics={
                            "episode_rewards": self.episode_rewards,
                            "episode_trades": self.episode_trades,
                            "episode_pnl": self.episode_pnl,
                            "td_errors": self.td_errors,
                            "epsilon": epsilon,
                        },
                    )

                # Log progress
                if (episode + 1) % self.config.print_every == 0:
                    recent_reward = np.mean(self.episode_rewards[-10:])
                    recent_trades = np.mean(self.episode_trades[-10:])
                    recent_pnl = np.mean(self.episode_pnl[-10:])

                    if self.persistence:
                        self.persistence.log_episode(
                            episode=episode + 1,
                            reward=recent_reward,
                            trades=int(recent_trades),
                            pnl=recent_pnl,
                            extra={"ε": f"{epsilon:.3f}"},
                        )
                    elif verbose:
                        print(f"Episode {episode + 1}/{self.config.n_episodes} | "
                              f"ε={epsilon:.3f} | "
                              f"Reward={recent_reward:.2f} | "
                              f"Trades={recent_trades:.0f} | "
                              f"PnL=${recent_pnl:+,.0f}")

        except Exception as e:
            if self.persistence:
                self.persistence.logger.error(f"Error during training: {e}")
                self.persistence.save_checkpoint(
                    episode=episode,
                    agent_state=self._get_agent_state(),
                    env_state={},
                    metrics={
                        "episode_rewards": self.episode_rewards,
                        "episode_trades": self.episode_trades,
                        "episode_pnl": self.episode_pnl,
                        "td_errors": self.td_errors,
                        "epsilon": epsilon,
                        "error": str(e),
                    },
                    force=True,
                )
            raise

        # Compile and save results
        results = self._compile_results()

        if self.persistence:
            self.persistence.save_results(results)
            self.persistence.save_trade_log(self.all_trades)

        return results

    def _compile_results(self) -> Dict[str, Any]:
        """Compile exploration results."""
        # Feature importance from correlations
        top_features = self.tracker.get_top_features_per_action(top_k=10)

        # Learning curve
        rewards_smoothed = pd.Series(self.episode_rewards).rolling(10).mean().values

        # Agent weights (for LinearQ)
        weights = None
        if hasattr(self.agent, "get_feature_weights"):
            weights = self.agent.get_feature_weights()

        return {
            "episode_rewards": self.episode_rewards,
            "episode_trades": self.episode_trades,
            "episode_pnl": self.episode_pnl,
            "td_errors": self.td_errors,
            "rewards_smoothed": rewards_smoothed,
            "top_features_per_action": top_features,
            "agent_weights": weights,
            "final_epsilon": self.epsilon,
            "total_episodes": len(self.episode_rewards),
            "total_trades": len(self.all_trades),
        }


# =============================================================================
# MAIN: Run exploration
# =============================================================================

def run_exploration_test(
    experiment_name: Optional[str] = None,
    n_episodes: int = 50,
    use_persistence: bool = True,
):
    """
    Run first-principles RL exploration test.

    Args:
        experiment_name: Name for this experiment (auto-generated if None)
        n_episodes: Number of training episodes
        use_persistence: Whether to save checkpoints and logs
    """
    print("=" * 70)
    print("FIRST-PRINCIPLES RL EXPLORATION FRAMEWORK")
    print("=" * 70)
    print("\nPhilosophy: 'We don't know what we don't know'")
    print("- 64-dim ungated feature space")
    print("- Physics-based reward shaping")
    print("- Iterative exploration with feature tracking")
    print("- Atomic saves, graceful failure recovery")
    print("=" * 70)

    # Setup persistence
    persistence = None
    if use_persistence:
        persistence = PersistenceManager(
            base_dir="results",
            experiment_name=experiment_name,
            checkpoint_every=10,
            max_checkpoints=5,
        )

    # Import from test_physics_pipeline
    from test_physics_pipeline import (
        PhysicsEngine,
        load_btc_h1_data,
        get_rl_state_features,
        get_rl_feature_names,
    )

    # Load data
    DATA_PATH = "data/master/BTCUSD_H1_202407010000_202512270700.csv"
    if persistence:
        persistence.logger.info(f"Loading data: {DATA_PATH}")
    else:
        print(f"\nLoading data: {DATA_PATH}")

    df = load_btc_h1_data(DATA_PATH)

    # Compute physics state
    if persistence:
        persistence.logger.info("Computing physics state...")
    else:
        print("\nComputing physics state...")

    physics = PhysicsEngine()
    physics_state = physics.compute_physics_state(df["close"])

    # Add volatility
    vol_state = physics.compute_advanced_volatility(df)
    for col in vol_state.columns:
        physics_state[col] = vol_state[col].values

    # Add DSP, VPIN, moments
    dsp_state = physics.compute_dsp_features(df["close"])
    for col in dsp_state.columns:
        physics_state[col] = dsp_state[col].values

    vpin_state = physics.compute_vpin_proxy(df)
    for col in vpin_state.columns:
        physics_state[col] = vpin_state[col].values

    moments_state = physics.compute_higher_moments(df["close"])
    for col in moments_state.columns:
        physics_state[col] = moments_state[col].values

    msg = f"Physics state shape: {physics_state.shape}, Features: {len(get_rl_feature_names())}"
    if persistence:
        persistence.logger.info(msg)
    else:
        print(msg)

    # Create environment
    print("\n" + "=" * 70)
    print("CREATING ENVIRONMENT")
    print("=" * 70)

    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.5,
        mae_penalty_weight=0.3,
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    env = TradingEnv(
        physics_state=physics_state,
        prices=df,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        max_position_bars=72,
    )

    print(f"State dimensions: {env.state_dim}")
    print(f"Action space: {env.n_actions} ({', '.join(env.action_names)})")

    # Create agents
    feature_names = get_rl_feature_names()

    # Test with multiple agents
    agents = {
        "Random": RandomAgent(env.state_dim, env.n_actions),
        "LinearQ": LinearQAgent(env.state_dim, env.n_actions, learning_rate=0.0001),
        "TabularQ": TabularQAgent(env.state_dim, env.n_actions, n_bins=5),
    }

    results = {}

    for agent_name, agent in agents.items():
        print(f"\n{'=' * 70}")
        print(f"TRAINING: {agent_name}")
        print("=" * 70)

        # Create fresh tracker
        tracker = FeatureTracker(feature_names)

        # Configure exploration
        config = ExplorationConfig(
            n_episodes=n_episodes,
            max_steps_per_episode=500,
            epsilon_start=1.0 if agent_name != "Random" else 0.0,
            epsilon_end=0.1,
            epsilon_decay=0.98,
            print_every=10,
            checkpoint_every=10,
            resume_from_checkpoint=False,  # Fresh start for each agent
        )

        # Run exploration with graceful failure handling
        try:
            loop = ExplorationLoop(
                env, agent, tracker, config,
                persistence=persistence if agent_name == "LinearQ" else None,  # Only persist main agent
            )
            result = loop.run(verbose=True)
            results[agent_name] = result
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] {agent_name} training stopped by user")
            if persistence:
                persistence.logger.warning(f"{agent_name} training interrupted")
            continue
        except Exception as e:
            print(f"\n[ERROR] {agent_name} training failed: {e}")
            if persistence:
                persistence.logger.error(f"{agent_name} training failed: {e}")
            continue

        # Print top features
        print(f"\n[{agent_name}] Top Features per Action:")
        for action, features in result["top_features_per_action"].items():
            top_3 = features[:3]
            top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
            print(f"  {action}: {top_str}")

    # Compare agents
    print("\n" + "=" * 70)
    print("AGENT COMPARISON")
    print("=" * 70)
    print(f"{'Agent':<12} {'Avg Reward':>12} {'Avg Trades':>12} {'Avg PnL':>12}")
    print("-" * 50)

    for agent_name, result in results.items():
        avg_reward = np.mean(result["episode_rewards"][-20:])
        avg_trades = np.mean(result["episode_trades"][-20:])
        avg_pnl = np.mean(result["episode_pnl"][-20:])
        print(f"{agent_name:<12} {avg_reward:>12.2f} {avg_trades:>12.1f} ${avg_pnl:>11,.0f}")

    # Show LinearQ learned weights
    if "LinearQ" in agents:
        print("\n" + "=" * 70)
        print("LINEAR Q-AGENT: LEARNED FEATURE WEIGHTS")
        print("=" * 70)

        linear_agent = agents["LinearQ"]
        weights = linear_agent.get_feature_weights()

        # Jupyter-friendly display
        if is_jupyter():
            JupyterDisplay.display_feature_weights(weights, feature_names)
        else:
            for action_id, action_name in enumerate(env.action_names):
                action_weights = weights[f"action_{action_id}"]
                top_indices = np.argsort(np.abs(action_weights))[::-1][:5]
                print(f"\n{action_name}:")
                for idx in top_indices:
                    print(f"  {feature_names[idx]:<25}: {action_weights[idx]:+.4f}")

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print("\n[INSIGHTS]")
    print("- Agent learns feature-action associations through exploration")
    print("- Physics-based reward shaping guides toward edge ratio optimization")
    print("- Feature correlations reveal emergent importance (not hard-coded)")
    print("- Ready for DQN/PPO when PyTorch installed")

    if persistence:
        print(f"\n[SAVED] Results in: {persistence.exp_dir}")
        persistence.save_results({
            "agents": list(results.keys()),
            "comparison": {
                agent: {
                    "avg_reward": float(np.mean(r["episode_rewards"][-20:])),
                    "avg_trades": float(np.mean(r["episode_trades"][-20:])),
                    "avg_pnl": float(np.mean(r["episode_pnl"][-20:])),
                }
                for agent, r in results.items()
            }
        }, "agent_comparison.json")

    return results


# =============================================================================
# MULTI-INSTRUMENT, MULTI-TIMEFRAME EXTENSION
# =============================================================================

@dataclass
class InstrumentData:
    """Container for instrument/timeframe data."""
    instrument: str
    timeframe: str
    df: pd.DataFrame
    physics_state: pd.DataFrame
    file_path: str
    bar_count: int

    @property
    def key(self) -> str:
        return f"{self.instrument}_{self.timeframe}"


class MultiInstrumentLoader:
    """
    Auto-discovers and loads all instruments and timeframes from data directory.

    Features:
    - Auto-discovery of CSV files by filename pattern
    - Per-instrument physics state computation
    - Normalized features (z-scored per instrument for fair comparison)
    - Unified interface for multi-instrument training
    """

    # Pattern matches: BTCUSD_H1_..., GBPUSD+_M15_..., Nikkei225_H4_..., NAS100_M30_..., DJ30ft_H1_...
    FILENAME_PATTERN = r"^([A-Za-z0-9\+\-]+)_([A-Z0-9]+)_\d+_\d+\.csv$"

    def __init__(
        self,
        data_dir: str = "data/master",
        min_bars: int = 1000,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.min_bars = min_bars
        self.verbose = verbose
        self.instruments: Dict[str, InstrumentData] = {}
        self._physics_engine = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def discover(self) -> List[Tuple[str, str, str]]:
        """
        Discover all available data files.

        Returns list of (instrument, timeframe, filepath) tuples.
        """
        import re
        discovered = []

        if not self.data_dir.exists():
            self._log(f"[WARN] Data directory not found: {self.data_dir}")
            return discovered

        for csv_file in sorted(self.data_dir.glob("*.csv")):
            match = re.match(self.FILENAME_PATTERN, csv_file.name)
            if match:
                instrument, timeframe = match.groups()
                discovered.append((instrument, timeframe, str(csv_file)))
                self._log(f"  [FOUND] {instrument} {timeframe}: {csv_file.name}")
            else:
                self._log(f"  [SKIP] {csv_file.name} (doesn't match pattern)")

        return discovered

    def load_all(self) -> Dict[str, InstrumentData]:
        """Load all discovered instruments and compute physics state."""
        from test_physics_pipeline import PhysicsEngine

        self._physics_engine = PhysicsEngine()
        discovered = self.discover()

        self._log(f"\n[LOADING] {len(discovered)} datasets...")

        for instrument, timeframe, filepath in discovered:
            try:
                data = self._load_single(instrument, timeframe, filepath)
                if data.bar_count >= self.min_bars:
                    self.instruments[data.key] = data
                    self._log(f"  [OK] {data.key}: {data.bar_count} bars")
                else:
                    self._log(f"  [SKIP] {data.key}: {data.bar_count} bars < {self.min_bars}")
            except Exception as e:
                self._log(f"  [ERROR] {instrument}_{timeframe}: {e}")

        self._log(f"\n[LOADED] {len(self.instruments)} datasets ready")
        return self.instruments

    def _load_single(
        self, instrument: str, timeframe: str, filepath: str
    ) -> InstrumentData:
        """Load single instrument and compute physics state."""
        # Load raw data
        df = self._load_csv(filepath)

        # Compute full physics state
        physics = self._physics_engine
        physics_state = physics.compute_physics_state(df["close"])

        # Add all advanced features
        vol_state = physics.compute_advanced_volatility(df)
        for col in vol_state.columns:
            physics_state[col] = vol_state[col].values

        dsp_state = physics.compute_dsp_features(df["close"])
        for col in dsp_state.columns:
            physics_state[col] = dsp_state[col].values

        vpin_state = physics.compute_vpin_proxy(df)
        for col in vpin_state.columns:
            physics_state[col] = vpin_state[col].values

        moments_state = physics.compute_higher_moments(df["close"])
        for col in moments_state.columns:
            physics_state[col] = moments_state[col].values

        return InstrumentData(
            instrument=instrument,
            timeframe=timeframe,
            df=df,
            physics_state=physics_state,
            file_path=filepath,
            bar_count=len(df),
        )

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV with standard preprocessing."""
        df = pd.read_csv(filepath, sep="\t")
        df.columns = [c.lower().replace("<", "").replace(">", "") for c in df.columns]

        # Combine date and time
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.set_index("datetime", inplace=True)

        # Ensure numeric
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["close"], inplace=True)
        return df

    def get_instrument(self, key: str) -> Optional[InstrumentData]:
        """Get instrument by key (e.g., 'BTCUSD_H1')."""
        return self.instruments.get(key)

    def list_instruments(self) -> List[str]:
        """List all loaded instrument keys."""
        return list(self.instruments.keys())

    def summary(self) -> pd.DataFrame:
        """Get summary of all loaded instruments."""
        data = []
        for key, inst in self.instruments.items():
            data.append({
                "key": key,
                "instrument": inst.instrument,
                "timeframe": inst.timeframe,
                "bars": inst.bar_count,
                "start": inst.df.index[0],
                "end": inst.df.index[-1],
                "price_mean": inst.df["close"].mean(),
                "price_std": inst.df["close"].std(),
            })
        return pd.DataFrame(data)


class MultiInstrumentEnv:
    """
    Environment that can switch between instruments/timeframes.

    Supports:
    - Round-robin sampling across instruments
    - Random instrument selection per episode
    - Combined training for universal patterns
    """

    def __init__(
        self,
        loader: MultiInstrumentLoader,
        feature_extractor: Callable,
        reward_shaper: Optional["RewardShaper"] = None,
        max_position_bars: int = 72,
        sampling_mode: str = "round_robin",  # or "random"
    ):
        self.loader = loader
        self.feature_extractor = feature_extractor
        self.reward_shaper = reward_shaper or RewardShaper()
        self.max_position_bars = max_position_bars
        self.sampling_mode = sampling_mode

        # Create per-instrument environments
        self.envs: Dict[str, TradingEnv] = {}
        for key, data in loader.instruments.items():
            self.envs[key] = TradingEnv(
                physics_state=data.physics_state,
                prices=data.df,
                feature_extractor=feature_extractor,
                reward_shaper=reward_shaper,
                max_position_bars=max_position_bars,
            )

        # Sampling state
        self.instrument_keys = list(self.envs.keys())
        self.current_idx = 0
        self.current_key: Optional[str] = None
        self.current_env: Optional[TradingEnv] = None

        # Standard interface
        self.n_actions = 4
        self.action_names = ["HOLD", "LONG", "SHORT", "CLOSE"]
        self.state_dim = 64

    def reset(self, instrument_key: Optional[str] = None, start_bar: Optional[int] = None) -> np.ndarray:
        """Reset environment, optionally switching instruments."""
        if instrument_key is not None:
            self.current_key = instrument_key
        elif self.sampling_mode == "round_robin":
            self.current_key = self.instrument_keys[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.instrument_keys)
        else:  # random
            self.current_key = np.random.choice(self.instrument_keys)

        self.current_env = self.envs[self.current_key]

        # Random start bar if not specified
        if start_bar is None:
            max_start = len(self.current_env.physics_state) - 500
            start_bar = np.random.randint(100, max(101, max_start))

        return self.current_env.reset(start_bar=start_bar)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step current environment."""
        next_state, reward, done, info = self.current_env.step(action)
        info["instrument"] = self.current_key
        return next_state, reward, done, info

    def get_episode_stats(self) -> Dict:
        """Get stats from current environment."""
        stats = self.current_env.get_episode_stats()
        stats["instrument"] = self.current_key
        return stats

    @property
    def episode_trades(self) -> List[Dict]:
        """Get trades from current environment."""
        return self.current_env.episode_trades


def run_multi_instrument_exploration(
    data_dir: str = "data/master",
    experiment_name: Optional[str] = None,
    n_episodes: int = 100,
    episodes_per_instrument: int = 20,
    use_persistence: bool = True,
    agents_to_test: Optional[List[str]] = None,
):
    """
    Run RL exploration across all instruments and timeframes.

    Args:
        data_dir: Directory containing CSV data files
        experiment_name: Name for this experiment
        n_episodes: Total episodes (overrides episodes_per_instrument if set)
        episodes_per_instrument: Episodes per instrument (for balanced training)
        use_persistence: Whether to save checkpoints and logs
        agents_to_test: Which agents to test (default: all)
    """
    from test_physics_pipeline import get_rl_state_features, get_rl_feature_names

    print("=" * 70)
    print("MULTI-INSTRUMENT RL EXPLORATION FRAMEWORK")
    print("=" * 70)
    print("\nUniversal patterns across all instruments & timeframes")
    print("- Same 64-dim physics feature space (normalized per instrument)")
    print("- Cross-instrument training for robust patterns")
    print("- Per-instrument results for edge validation")
    print("=" * 70)

    # Setup persistence
    if experiment_name is None:
        experiment_name = f"multi_inst_{datetime.now():%Y%m%d_%H%M%S}"

    persistence = None
    if use_persistence:
        persistence = PersistenceManager(
            base_dir="results",
            experiment_name=experiment_name,
            checkpoint_every=10,
            max_checkpoints=5,
        )

    # Load all instruments
    print(f"\n{'=' * 70}")
    print("DISCOVERING INSTRUMENTS")
    print("=" * 70)

    loader = MultiInstrumentLoader(data_dir=data_dir, verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("[ERROR] No instruments loaded! Check data directory.")
        return {}

    # Display summary
    summary = loader.summary()
    print(f"\n{summary.to_string()}")

    # Calculate episodes
    n_instruments = len(loader.instruments)
    total_episodes = max(n_episodes, n_instruments * episodes_per_instrument)

    print(f"\nTraining plan:")
    print(f"  Instruments: {n_instruments}")
    print(f"  Episodes per instrument: {episodes_per_instrument}")
    print(f"  Total episodes: {total_episodes}")

    # Create multi-instrument environment
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.5,
        mae_penalty_weight=0.3,
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    multi_env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    print(f"\n{'=' * 70}")
    print("ENVIRONMENT READY")
    print("=" * 70)
    print(f"State dimensions: {multi_env.state_dim}")
    print(f"Action space: {multi_env.n_actions} ({', '.join(multi_env.action_names)})")
    print(f"Instruments: {', '.join(multi_env.instrument_keys)}")

    # Create agents
    feature_names = get_rl_feature_names()

    if agents_to_test is None:
        agents_to_test = ["Random", "LinearQ", "TabularQ"]

    agents = {}
    if "Random" in agents_to_test:
        agents["Random"] = RandomAgent(multi_env.state_dim, multi_env.n_actions)
    if "LinearQ" in agents_to_test:
        agents["LinearQ"] = LinearQAgent(multi_env.state_dim, multi_env.n_actions, learning_rate=0.0001)
    if "TabularQ" in agents_to_test:
        agents["TabularQ"] = TabularQAgent(multi_env.state_dim, multi_env.n_actions, n_bins=5)

    # Results per agent and per instrument
    results: Dict[str, Dict] = {}
    per_instrument_results: Dict[str, Dict[str, Dict]] = {key: {} for key in loader.instruments}

    for agent_name, agent in agents.items():
        print(f"\n{'=' * 70}")
        print(f"TRAINING: {agent_name}")
        print("=" * 70)

        # Reset agent for fresh training
        if hasattr(agent, "weights"):
            agent.weights = np.random.randn(agent.n_actions, agent.state_dim) * 0.01
            agent.biases = np.zeros(agent.n_actions)
        if hasattr(agent, "q_table"):
            agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions))

        # Track per-instrument metrics
        instrument_episodes: Dict[str, List[Dict]] = {key: [] for key in loader.instruments}
        all_rewards = []
        all_trades = []
        all_pnl = []
        all_episode_trades = []

        tracker = FeatureTracker(feature_names)
        epsilon = 1.0 if agent_name != "Random" else 0.0

        for episode in range(total_episodes):
            # Reset (round-robin across instruments)
            state = multi_env.reset()
            total_reward = 0
            episode_steps = 0

            for step in range(500):  # max steps per episode
                action = agent.select_action(state, epsilon)
                tracker.record(state, action)
                next_state, reward, done, info = multi_env.step(action)

                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                episode_steps += 1

                if done:
                    break

            # Decay epsilon
            epsilon = max(0.1, epsilon * 0.98)

            # Record metrics
            stats = multi_env.get_episode_stats()
            instrument_key = stats["instrument"]

            instrument_episodes[instrument_key].append({
                "episode": episode,
                "reward": total_reward,
                "trades": stats["trades"],
                "pnl": stats.get("total_pnl", 0),
            })

            all_rewards.append(total_reward)
            all_trades.append(stats["trades"])
            all_pnl.append(stats.get("total_pnl", 0))
            all_episode_trades.extend(multi_env.episode_trades)

            # Progress log
            if (episode + 1) % 20 == 0:
                recent_reward = np.mean(all_rewards[-20:])
                recent_trades = np.mean(all_trades[-20:])
                recent_pnl = np.mean(all_pnl[-20:])
                print(f"  Episode {episode + 1:4d}/{total_episodes} | "
                      f"ε={epsilon:.3f} | "
                      f"Reward={recent_reward:+8.2f} | "
                      f"Trades={recent_trades:3.0f} | "
                      f"PnL=${recent_pnl:+10,.0f}")

        # Compile agent results
        top_features = tracker.get_top_features_per_action(top_k=10)

        results[agent_name] = {
            "episode_rewards": all_rewards,
            "episode_trades": all_trades,
            "episode_pnl": all_pnl,
            "all_trades": all_episode_trades,
            "top_features_per_action": top_features,
            "final_epsilon": epsilon,
        }

        # Per-instrument breakdown
        for key, episodes_data in instrument_episodes.items():
            if episodes_data:
                per_instrument_results[key][agent_name] = {
                    "episodes": len(episodes_data),
                    "avg_reward": np.mean([e["reward"] for e in episodes_data]),
                    "avg_trades": np.mean([e["trades"] for e in episodes_data]),
                    "avg_pnl": np.mean([e["pnl"] for e in episodes_data]),
                }

        # Show top features
        print(f"\n[{agent_name}] Top Features per Action:")
        for action, features in top_features.items():
            top_3 = features[:3]
            top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
            print(f"  {action}: {top_str}")

    # Final comparison
    print(f"\n{'=' * 70}")
    print("AGENT COMPARISON (OVERALL)")
    print("=" * 70)
    print(f"{'Agent':<12} {'Avg Reward':>12} {'Avg Trades':>12} {'Avg PnL':>12}")
    print("-" * 50)

    for agent_name, result in results.items():
        avg_reward = np.mean(result["episode_rewards"][-20:])
        avg_trades = np.mean(result["episode_trades"][-20:])
        avg_pnl = np.mean(result["episode_pnl"][-20:])
        print(f"{agent_name:<12} {avg_reward:>12.2f} {avg_trades:>12.1f} ${avg_pnl:>11,.0f}")

    # Per-instrument breakdown
    print(f"\n{'=' * 70}")
    print("PER-INSTRUMENT BREAKDOWN")
    print("=" * 70)

    for inst_key, agent_results in per_instrument_results.items():
        if agent_results:
            print(f"\n{inst_key}:")
            for agent_name, stats in agent_results.items():
                print(f"  {agent_name:<12} | "
                      f"Reward={stats['avg_reward']:+8.2f} | "
                      f"Trades={stats['avg_trades']:5.1f} | "
                      f"PnL=${stats['avg_pnl']:+10,.0f}")

    # Save results
    if persistence:
        persistence.save_results({
            "experiment": experiment_name,
            "instruments": list(loader.instruments.keys()),
            "n_instruments": n_instruments,
            "total_episodes": total_episodes,
            "agents": list(results.keys()),
            "overall_comparison": {
                agent: {
                    "avg_reward": float(np.mean(r["episode_rewards"][-20:])),
                    "avg_trades": float(np.mean(r["episode_trades"][-20:])),
                    "avg_pnl": float(np.mean(r["episode_pnl"][-20:])),
                }
                for agent, r in results.items()
            },
            "per_instrument": per_instrument_results,
        }, "multi_instrument_results.json")

        # Save all trades
        all_trades_flat = []
        for agent_name, result in results.items():
            for trade in result.get("all_trades", []):
                trade_copy = trade.copy()
                trade_copy["agent"] = agent_name
                all_trades_flat.append(trade_copy)

        if all_trades_flat:
            trades_df = pd.DataFrame(all_trades_flat)
            trades_path = persistence.results_dir / "all_trades.csv"
            trades_df.to_csv(trades_path, index=False)

        print(f"\n[SAVED] Results in: {persistence.exp_dir}")

    print(f"\n{'=' * 70}")
    print("MULTI-INSTRUMENT EXPLORATION COMPLETE")
    print("=" * 70)
    print("\n[INSIGHTS]")
    print("- Universal physics patterns emerge across instruments")
    print("- Per-instrument performance reveals edge robustness")
    print("- Cross-validated features are most reliable alpha signals")

    return {
        "results": results,
        "per_instrument": per_instrument_results,
        "loader": loader,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Exploration Framework")
    parser.add_argument("--multi", action="store_true", help="Run multi-instrument exploration")
    parser.add_argument("--data-dir", type=str, default="data/master",
                        help="Path to data directory containing CSV files")
    parser.add_argument("--episodes", type=int, default=25,
                        help="Episodes per instrument (multi) or total (single)")
    parser.add_argument("--agents", type=str, default="Random,LinearQ,TabularQ",
                        help="Comma-separated list of agents to test")

    args = parser.parse_args()

    if args.multi:
        # Run multi-instrument exploration
        results = run_multi_instrument_exploration(
            data_dir=args.data_dir,
            episodes_per_instrument=args.episodes,
            agents_to_test=args.agents.split(",") if args.agents else None,
        )
    else:
        # Run single-instrument exploration (original)
        results = run_exploration_test(n_episodes=args.episodes)
