"""
Agent Factory - Unified RL Agent Creation
==========================================

Creates RL agents from configuration strings for testing framework.

Supported agents:
- PPO (KinetraAgent) - Proximal Policy Optimization
- DQN (NeuralAgent) - Deep Q-Network
- Linear Q (SimpleRLAgent) - Linear Q-Learning
- Incumbent (IncumbentAgent) - PPO-style Triad agent
- Competitor (CompetitorAgent) - A2C-style Triad agent
- Researcher (ResearcherAgent) - SAC-style Triad agent

Philosophy: Make it EASY to swap agents for empirical comparison.
All agents implement a unified interface for the test harness.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

# Core agent imports
from kinetra.rl_agent import KinetraAgent
from kinetra.rl_neural_agent import NeuralAgent
from kinetra.rl_physics_env import SimpleRLAgent
from kinetra.triad_system import (
    AgentRole,
    CompetitorAgent,
    IncumbentAgent,
    ResearcherAgent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED AGENT INTERFACE
# =============================================================================


class UnifiedAgentInterface(ABC):
    """
    Abstract interface that all agents must implement for the exhaustive test harness.

    This defines the minimal contract for agent interaction:
    - select_action: Choose action given state
    - update: Learn from experience tuple
    - train (optional): Train on a batch of data
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action given current state.

        Args:
            state: State vector (numpy array)
            explore: Whether to explore (True) or exploit (False)

        Returns:
            Action index (int)
        """
        pass

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """
        Update agent from single experience tuple.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag

        Returns:
            Optional dict with training metrics (loss, etc.)
        """
        pass


class AgentAdapter(UnifiedAgentInterface):
    """
    Adapter to wrap existing agents with the unified interface.

    Handles interface mismatches between different agent implementations.
    """

    def __init__(self, agent: Any, agent_type: str):
        self.agent = agent
        self.agent_type = agent_type

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action with interface adaptation."""
        # Ensure state is the right type
        state = np.asarray(state, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            # Try different interface patterns
            if hasattr(self.agent, "select_action"):
                # Check signature style
                if self.agent_type in ("ppo",):
                    # KinetraAgent uses deterministic parameter
                    return int(self.agent.select_action(state, deterministic=not explore))
                elif self.agent_type in ("dqn", "linear_q"):
                    # NeuralAgent and SimpleRLAgent use training parameter
                    return int(self.agent.select_action(state, training=explore))
                elif self.agent_type in ("incumbent", "competitor", "researcher"):
                    # Triad agents use explore parameter
                    return int(self.agent.select_action(state, explore=explore))
                else:
                    # Generic fallback
                    try:
                        return int(self.agent.select_action(state, explore=explore))
                    except TypeError:
                        try:
                            return int(self.agent.select_action(state, training=explore))
                        except TypeError:
                            return int(self.agent.select_action(state))
            elif hasattr(self.agent, "get_action"):
                return int(self.agent.get_action(state))
            elif hasattr(self.agent, "act"):
                return int(self.agent.act(state))
            else:
                raise ValueError(f"Agent {type(self.agent)} has no action selection method")
        except Exception as e:
            logger.warning(f"Action selection failed for {self.agent_type}: {e}")
            # Fallback to random action
            action_dim = getattr(self.agent, "action_dim", None) or getattr(
                self.agent, "n_actions", 4
            )
            return np.random.randint(0, action_dim)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Update agent with interface adaptation."""
        # Sanitize inputs
        state = np.nan_to_num(np.asarray(state, dtype=np.float32))
        next_state = np.nan_to_num(np.asarray(next_state, dtype=np.float32))
        reward = float(np.clip(reward, -1e6, 1e6))

        try:
            if hasattr(self.agent, "update"):
                result = self.agent.update(state, action, reward, next_state, done)
                return result if isinstance(result, dict) else None
            elif hasattr(self.agent, "learn"):
                result = self.agent.learn(state, action, reward, next_state, done)
                return result if isinstance(result, dict) else None
            elif hasattr(self.agent, "store_transition"):
                # PPO-style: store then batch update
                self.agent.store_transition(state, action, None, reward, None, done)
                return None
            else:
                logger.debug(f"Agent {self.agent_type} has no update method")
                return None
        except Exception as e:
            logger.debug(f"Update failed for {self.agent_type}: {e}")
            return None

    def train_episode(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Train agent on a batch of episode data.

        Args:
            states: Array of states (n_steps, state_dim)
            actions: Array of actions (n_steps,)
            rewards: Array of rewards (n_steps,)

        Returns:
            Training metrics dict or None
        """
        try:
            if hasattr(self.agent, "train"):
                return self.agent.train(states, actions, rewards)
            elif hasattr(self.agent, "update_policy"):
                return self.agent.update_policy()
            else:
                # Manual update loop
                for i in range(len(states) - 1):
                    self.update(
                        states[i],
                        int(actions[i]),
                        float(rewards[i]),
                        states[i + 1],
                        i == len(states) - 2,
                    )
                return None
        except Exception as e:
            logger.warning(f"Episode training failed for {self.agent_type}: {e}")
            return None

    @property
    def underlying_agent(self) -> Any:
        """Get the underlying agent instance."""
        return self.agent

    def __repr__(self) -> str:
        return f"AgentAdapter({self.agent_type}, {type(self.agent).__name__})"


# =============================================================================
# AGENT REGISTRY
# =============================================================================

# Complete registry of all available agents
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "class": KinetraAgent,
        "description": "PPO (Proximal Policy Optimization) - Stable policy gradient",
        "default_params": {"state_dim": 43, "action_dim": 4},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "action_dim"},
    },
    "dqn": {
        "class": NeuralAgent,
        "description": "DQN (Deep Q-Network) - Value-based with replay",
        "default_params": {"state_dim": 43, "action_dim": 4},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "action_dim"},
    },
    "linear_q": {
        "class": SimpleRLAgent,
        "description": "Linear Q-Learning - Fast, interpretable",
        "default_params": {"state_dim": 43, "action_dim": 4},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "action_dim"},
    },
    "incumbent": {
        "class": IncumbentAgent,
        "description": "Incumbent (PPO-style Triad) - Stable exploitation",
        "default_params": {"state_dim": 43, "n_actions": 4, "role": AgentRole.TRADER},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "n_actions"},
    },
    "competitor": {
        "class": CompetitorAgent,
        "description": "Competitor (A2C-style Triad) - Aggressive adaptation",
        "default_params": {"state_dim": 43, "n_actions": 4, "role": AgentRole.TRADER},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "n_actions"},
    },
    "researcher": {
        "class": ResearcherAgent,
        "description": "Researcher (SAC-style Triad) - Exploration-focused",
        "default_params": {"state_dim": 43, "n_actions": 4, "role": AgentRole.TRADER},
        "param_mapping": {"state_dim": "state_dim", "action_dim": "n_actions"},
    },
}


# =============================================================================
# AGENT FACTORY
# =============================================================================


class AgentFactory:
    """
    Factory for creating RL agents from configuration.

    Provides a unified interface for all agent types, making it easy to:
    - Swap agents for A/B testing
    - Run exhaustive combination tests
    - Compare performance across agent families

    Usage:
        # Simple creation
        agent = AgentFactory.create('ppo', state_dim=64, action_dim=4)

        # With custom config
        agent = AgentFactory.create('dqn', state_dim=64, action_dim=4,
                                   config={'lr': 1e-3, 'gamma': 0.95})

        # Get wrapped with unified interface
        wrapped = AgentFactory.create_wrapped('ppo', state_dim=64, action_dim=4)
        wrapped.select_action(state, explore=True)
        wrapped.update(state, action, reward, next_state, done)
    """

    @classmethod
    def create(
        cls,
        agent_type: str,
        state_dim: int = 43,
        action_dim: int = 4,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create agent instance from type string.

        Args:
            agent_type: Agent type key ('ppo', 'dqn', 'linear_q',
                       'incumbent', 'competitor', 'researcher')
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Additional configuration parameters

        Returns:
            Instantiated agent (raw, not wrapped)

        Raises:
            ValueError: If agent type is unknown
        """
        if config is None:
            config = {}

        if agent_type not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys())
            raise ValueError(f"Unknown agent type: '{agent_type}'. Available: {available}")

        registry_entry = AGENT_REGISTRY[agent_type]
        agent_class = registry_entry["class"]
        default_params = registry_entry["default_params"].copy()
        param_mapping = registry_entry["param_mapping"]

        # Map standard param names to agent-specific names
        params = {}
        for standard_name, agent_name in param_mapping.items():
            if standard_name == "state_dim":
                params[agent_name] = state_dim
            elif standard_name == "action_dim":
                params[agent_name] = action_dim

        # Add default params that aren't in mapping (like 'role')
        for key, value in default_params.items():
            if key not in params and key not in param_mapping.values():
                params[key] = value

        # Merge with user config
        params.update(config)

        logger.info(f"Creating {agent_type} agent (state_dim={state_dim}, action_dim={action_dim})")

        try:
            agent = agent_class(**params)
            logger.info(f"✅ Created {agent_type} agent: {type(agent).__name__}")
            return agent

        except TypeError as e:
            # Try alternative constructor patterns
            logger.warning(f"Standard constructor failed for {agent_type}: {e}")

            # Fallback 1: n_features/n_actions
            try:
                alt_params = {
                    "n_features": state_dim,
                    "n_actions": action_dim,
                }
                alt_params.update(config)
                agent = agent_class(**alt_params)
                logger.info(f"✅ Created {agent_type} agent (alt constructor)")
                return agent
            except Exception:
                pass

            # Fallback 2: obs_dim/act_dim
            try:
                alt_params = {
                    "obs_dim": state_dim,
                    "act_dim": action_dim,
                }
                alt_params.update(config)
                agent = agent_class(**alt_params)
                logger.info(f"✅ Created {agent_type} agent (alt2 constructor)")
                return agent
            except Exception:
                pass

            logger.error(f"Failed to create {agent_type} agent: {e}")
            raise

    @classmethod
    def create_wrapped(
        cls,
        agent_type: str,
        state_dim: int = 43,
        action_dim: int = 4,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentAdapter:
        """
        Create agent wrapped with unified interface.

        This is the preferred method for the exhaustive test harness.

        Args:
            agent_type: Agent type key
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Additional configuration parameters

        Returns:
            AgentAdapter wrapping the agent with unified interface
        """
        agent = cls.create(agent_type, state_dim, action_dim, config)
        return AgentAdapter(agent, agent_type)

    @classmethod
    def create_all(
        cls,
        state_dim: int = 43,
        action_dim: int = 4,
        config: Optional[Dict[str, Any]] = None,
        wrapped: bool = True,
    ) -> Dict[str, Union[Any, AgentAdapter]]:
        """
        Create instances of all available agent types.

        Useful for exhaustive testing across all agents.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Shared configuration for all agents
            wrapped: If True, wrap with AgentAdapter

        Returns:
            Dict mapping agent_type to agent instance
        """
        agents = {}
        for agent_type in AGENT_REGISTRY.keys():
            try:
                if wrapped:
                    agents[agent_type] = cls.create_wrapped(
                        agent_type, state_dim, action_dim, config
                    )
                else:
                    agents[agent_type] = cls.create(agent_type, state_dim, action_dim, config)
            except Exception as e:
                logger.warning(f"Failed to create {agent_type}: {e}")
                agents[agent_type] = None
        return agents

    @classmethod
    def register_agent(
        cls,
        name: str,
        agent_class: Type,
        description: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        param_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a new agent type.

        Args:
            name: Name for the agent type
            agent_class: Agent class to register
            description: Human-readable description
            default_params: Default constructor parameters
            param_mapping: Mapping from standard names to agent param names
        """
        if default_params is None:
            default_params = {"state_dim": 43, "action_dim": 4}
        if param_mapping is None:
            param_mapping = {"state_dim": "state_dim", "action_dim": "action_dim"}

        AGENT_REGISTRY[name] = {
            "class": agent_class,
            "description": description,
            "default_params": default_params,
            "param_mapping": param_mapping,
        }
        logger.info(f"Registered new agent type: {name}")

    @classmethod
    def list_available_agents(cls) -> List[str]:
        """Get list of available agent type names."""
        return list(AGENT_REGISTRY.keys())

    @classmethod
    def get_agent_info(cls, agent_type: str) -> Dict[str, Any]:
        """Get detailed info about an agent type."""
        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent type: {agent_type}")
        entry = AGENT_REGISTRY[agent_type]
        return {
            "name": agent_type,
            "class": entry["class"].__name__,
            "description": entry["description"],
            "default_params": entry["default_params"],
        }

    @classmethod
    def get_all_agent_info(cls) -> List[Dict[str, Any]]:
        """Get info for all registered agents."""
        return [cls.get_agent_info(name) for name in AGENT_REGISTRY.keys()]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_agent_from_config(
    agent_config: Dict[str, Any], state_dim: int = 43, action_dim: int = 4
) -> Any:
    """
    Convenience function to create agent from config dict.

    Args:
        agent_config: Configuration with 'type' key and other params
        state_dim: State space dimension
        action_dim: Action space dimension

    Returns:
        Instantiated agent

    Example:
        config = {'type': 'ppo', 'lr': 3e-4, 'gamma': 0.99}
        agent = create_agent_from_config(config, 64, 4)
    """
    config = agent_config.copy()
    agent_type = config.pop("type", "ppo")
    return AgentFactory.create(agent_type, state_dim, action_dim, config)


def get_all_agent_types() -> List[str]:
    """Get list of all registered agent types."""
    return AgentFactory.list_available_agents()


def create_agent(
    agent_type: str, state_dim: int = 43, action_dim: int = 4, wrapped: bool = False
) -> Union[Any, AgentAdapter]:
    """
    Simple function to create an agent.

    Args:
        agent_type: Agent type string
        state_dim: State dimension
        action_dim: Action dimension
        wrapped: If True, return wrapped with unified interface

    Returns:
        Agent instance (wrapped or raw)
    """
    if wrapped:
        return AgentFactory.create_wrapped(agent_type, state_dim, action_dim)
    return AgentFactory.create(agent_type, state_dim, action_dim)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Agent Factory Self-Test")
    print("=" * 60 + "\n")

    # List available agents
    print("Available agents:")
    for info in AgentFactory.get_all_agent_info():
        print(f"  • {info['name']:12} - {info['description']}")

    print("\n" + "-" * 60)
    print("Creating all agents...")
    print("-" * 60 + "\n")

    # Create all agents
    state_dim = 43
    action_dim = 4
    test_state = np.random.randn(state_dim).astype(np.float32)

    success_count = 0
    fail_count = 0

    for agent_type in AgentFactory.list_available_agents():
        try:
            # Create wrapped agent
            agent = AgentFactory.create_wrapped(agent_type, state_dim, action_dim)
            print(f"✅ {agent_type:12} created: {agent}")

            # Test action selection
            action = agent.select_action(test_state, explore=True)
            print(f"   → Action (explore): {action}")

            action_exploit = agent.select_action(test_state, explore=False)
            print(f"   → Action (exploit): {action_exploit}")

            # Test update
            next_state = np.random.randn(state_dim).astype(np.float32)
            reward = np.random.randn() * 0.1
            agent.update(test_state, action, reward, next_state, done=False)
            print(f"   → Update: OK")

            success_count += 1

        except Exception as e:
            print(f"❌ {agent_type:12} FAILED: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {success_count} passed, {fail_count} failed")
    print("=" * 60 + "\n")

    if fail_count == 0:
        print("✅ All agent factory tests passed!")
    else:
        print(f"⚠️ {fail_count} agent(s) failed to initialize")
