"""
Agent Factory - Unified RL Agent Creation
==========================================

Creates RL agents from configuration strings for testing framework.

Supported agents:
- PPO (KinetraAgent)
- DQN (NeuralAgent)
- Triad agents (Incumbent, Competitor, Researcher)
- Linear Q (from exploration framework)

Philosophy: Make it EASY to swap agents for empirical comparison.
"""

import logging
from typing import Any, Dict, Optional, Type

from kinetra.rl_agent import KinetraAgent
from kinetra.rl_neural_agent import NeuralAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating RL agents from configuration.
    
    Usage:
        agent = AgentFactory.create(
            agent_type='ppo',
            state_dim=64,
            action_dim=4,
            config={'lr': 3e-4}
        )
    """
    
    # Registry of available agents
    AGENT_REGISTRY = {
        'ppo': KinetraAgent,
        'dqn': NeuralAgent,
    }
    
    @classmethod
    def create(
        cls,
        agent_type: str,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create agent instance from type string.
        
        Args:
            agent_type: Agent type ('ppo', 'dqn', etc.)
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Additional configuration parameters
            
        Returns:
            Instantiated agent
            
        Raises:
            ValueError: If agent type is unknown
        """
        if config is None:
            config = {}
        
        if agent_type not in cls.AGENT_REGISTRY:
            available = ', '.join(cls.AGENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. "
                f"Available: {available}"
            )
        
        agent_class = cls.AGENT_REGISTRY[agent_type]
        
        logger.info(f"Creating {agent_type} agent (state_dim={state_dim}, action_dim={action_dim})")
        
        # Create agent with standard interface
        try:
            # Most agents use state_dim/action_dim interface
            agent = agent_class(
                state_dim=state_dim,
                action_dim=action_dim,
                **config
            )
            logger.info(f"✅ Successfully created {agent_type} agent")
            return agent
            
        except TypeError as e:
            # Try alternative constructor signatures
            logger.warning(f"Standard constructor failed, trying alternatives: {e}")
            
            # Some agents might use different parameter names
            try:
                agent = agent_class(
                    n_features=state_dim,
                    n_actions=action_dim,
                    **config
                )
                logger.info(f"✅ Successfully created {agent_type} agent (alternative constructor)")
                return agent
            except Exception as e2:
                logger.error(f"Failed to create {agent_type} agent: {e2}")
                raise
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type) -> None:
        """
        Register a new agent type.
        
        Args:
            name: Name for the agent type
            agent_class: Agent class to register
        """
        cls.AGENT_REGISTRY[name] = agent_class
        logger.info(f"Registered new agent type: {name}")
    
    @classmethod
    def list_available_agents(cls) -> list:
        """Get list of available agent types."""
        return list(cls.AGENT_REGISTRY.keys())


def create_agent_from_config(
    agent_config: Dict[str, Any],
    state_dim: int,
    action_dim: int
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
    agent_type = agent_config.pop('type', 'ppo')
    return AgentFactory.create(agent_type, state_dim, action_dim, agent_config)


# Quick test
if __name__ == "__main__":
    print("\n=== Agent Factory Test ===\n")
    
    # List available agents
    print("Available agents:")
    for agent_type in AgentFactory.list_available_agents():
        print(f"  - {agent_type}")
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    ppo_agent = AgentFactory.create('ppo', state_dim=64, action_dim=4)
    print(f"✅ PPO agent created: {type(ppo_agent).__name__}")
    
    # Create DQN agent
    print("\nCreating DQN agent...")
    dqn_agent = AgentFactory.create('dqn', state_dim=64, action_dim=4)
    print(f"✅ DQN agent created: {type(dqn_agent).__name__}")
    
    # Create with config
    print("\nCreating agent from config...")
    config = {'type': 'ppo', 'lr': 1e-3, 'gamma': 0.95}
    agent = create_agent_from_config(config, 64, 4)
    print(f"✅ Agent created from config: {type(agent).__name__}")
    
    print("\n✅ All tests passed!")
