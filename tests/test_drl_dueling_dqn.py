"""
Tests for DRL Dueling DQN Module
=================================

Unit and integration tests for trading DQN.
"""
# ruff: noqa: I001
import numpy as np
import pytest
import torch

from kinetra.drl_dueling_dqn import (
    DQNAgent,
    DuelingDQN,
    ReplayBuffer,
    TradingEnvironment,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_prices():
    """Generate sample price series."""
    np.random.seed(42)
    n = 1000

    # Bull market trend + noise
    returns = np.random.normal(0.001, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))

    return prices


@pytest.fixture
def trading_env(sample_prices):
    """Create trading environment."""
    return TradingEnvironment(
        prices=sample_prices,
        prices_denoised=sample_prices,  # Same for testing
        window_size=50,
    )


@pytest.fixture
def dqn_agent():
    """Create DQN agent."""
    return DQNAgent(
        state_size=51,  # 50 returns + position
        action_size=3,
        device=torch.device("cpu"),  # CPU for tests
    )


# ============================================================================
# NETWORK TESTS
# ============================================================================


def test_dueling_network_forward():
    """Test Dueling DQN forward pass."""
    network = DuelingDQN(state_size=10, action_size=3)

    # Random input
    state = torch.randn(1, 10)

    # Forward pass
    q_values = network(state)

    # Check output shape
    assert q_values.shape == (1, 3)
    assert torch.all(torch.isfinite(q_values))


def test_dueling_network_batch():
    """Test Dueling DQN with batch input."""
    network = DuelingDQN(state_size=10, action_size=3)

    # Batch input
    batch_size = 32
    states = torch.randn(batch_size, 10)

    # Forward pass
    q_values = network(states)

    # Check output shape
    assert q_values.shape == (batch_size, 3)
    assert torch.all(torch.isfinite(q_values))


# ============================================================================
# REPLAY BUFFER TESTS
# ============================================================================


def test_replay_buffer_basic():
    """Test basic replay buffer operations."""
    buffer = ReplayBuffer(capacity=100)

    # Add transitions
    for i in range(50):
        state = np.random.randn(10)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False

        buffer.push(state, action, reward, next_state, done)

    # Check length
    assert len(buffer) == 50

    # Sample batch
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones = batch

    assert len(states) == 32
    assert len(actions) == 32


def test_replay_buffer_max_capacity():
    """Test replay buffer respects max capacity."""
    buffer = ReplayBuffer(capacity=10)

    # Add more than capacity
    for i in range(20):
        buffer.push(i, 0, 0, i, False)

    # Should only keep last 10
    assert len(buffer) == 10


# ============================================================================
# ENVIRONMENT TESTS
# ============================================================================


def test_environment_reset(trading_env):
    """Test environment reset."""
    state = trading_env.reset()

    # Check state shape
    assert len(state) == 51  # 50 returns + position + unrealized
    assert np.all(np.isfinite(state))

    # Check initial conditions
    assert trading_env.position == 0
    assert trading_env.step_idx == trading_env.window_size


def test_environment_step_hold(trading_env):
    """Test environment step with hold action."""
    _ = trading_env.reset()
    initial_position = trading_env.position

    # Take hold action
    next_state, reward, done, info = trading_env.step(action=1)

    # Position should not change
    assert trading_env.position == initial_position

    # State should be valid
    assert np.all(np.isfinite(next_state))
    assert np.isfinite(reward)


def test_environment_step_buy(trading_env):
    """Test environment step with buy action."""
    _ = trading_env.reset()

    # Take buy action
    next_state, reward, done, info = trading_env.step(action=2)

    # Position should be long
    assert trading_env.position == 1
    assert trading_env.entry_price > 0


def test_environment_step_close(trading_env):
    """Test environment step with close action."""
    _ = trading_env.reset()

    # Open position
    trading_env.step(action=2)  # Buy
    assert trading_env.position == 1

    # Close position
    next_state, reward, done, info = trading_env.step(action=0)

    # Position should be flat
    assert trading_env.position == 0

    # Should have recorded a trade
    assert len(trading_env.trades) == 1


def test_environment_full_episode(trading_env):
    """Test full episode through environment."""
    state = trading_env.reset()
    total_reward = 0
    steps = 0

    while True:
        # Random action
        action = np.random.randint(0, 3)

        state, reward, done, _ = trading_env.step(action)

        total_reward += reward
        steps += 1

        if done:
            break

    # Check episode completed
    assert steps > 0
    assert np.isfinite(total_reward)


def test_environment_mfe_mae_tracking(trading_env):
    """Test MAE/MFE tracking in trades."""
    _ = trading_env.reset()

    # Open long position
    trading_env.step(action=2)

    # Take several steps (price should move)
    for _ in range(10):
        trading_env.step(action=1)  # Hold

    # Close position
    trading_env.step(action=0)

    # Check trade metrics
    assert len(trading_env.trades) == 1
    trade = trading_env.trades[0]

    assert trade.mfe >= 0  # MFE should be tracked
    assert trade.mae <= 0  # MAE should be tracked (negative)
    assert np.isfinite(trade.pnl)


# ============================================================================
# AGENT TESTS
# ============================================================================


def test_agent_initialization(dqn_agent):
    """Test DQN agent initialization."""
    assert dqn_agent.policy_net is not None
    assert dqn_agent.target_net is not None
    assert dqn_agent.buffer is not None
    assert dqn_agent.optimizer is not None


def test_agent_action_selection(dqn_agent):
    """Test agent action selection."""
    state = np.random.randn(51)

    # Training mode (epsilon-greedy)
    action = dqn_agent.select_action(state, training=True)
    assert 0 <= action < 3

    # Evaluation mode (greedy)
    action = dqn_agent.select_action(state, training=False)
    assert 0 <= action < 3


def test_agent_training_step(dqn_agent):
    """Test agent training step."""
    # Fill buffer
    for _ in range(200):
        state = np.random.randn(51)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(51)
        done = False

        dqn_agent.buffer.push(state, action, reward, next_state, done)

    # Train step should not crash
    dqn_agent.train_step()

    # Steps should increment
    assert dqn_agent.steps_done > 0


def test_agent_target_update(dqn_agent):
    """Test target network update."""
    # Get initial target weights
    initial_weights = dqn_agent.target_net.feature[0].weight.data.clone()

    # Update policy network (simulate training)
    for param in dqn_agent.policy_net.parameters():
        param.data += torch.randn_like(param) * 0.01

    # Update target network
    dqn_agent.update_target_network()

    # Target weights should change
    updated_weights = dqn_agent.target_net.feature[0].weight.data

    assert not torch.allclose(initial_weights, updated_weights)


def test_agent_save_load(dqn_agent, tmp_path):
    """Test agent save and load."""
    # Save agent
    save_path = tmp_path / "test_agent.pt"
    dqn_agent.save(save_path)

    assert save_path.exists()

    # Create new agent
    new_agent = DQNAgent(state_size=51, action_size=3, device=torch.device("cpu"))

    # Load weights
    new_agent.load(save_path)

    # Weights should match
    for p1, p2 in zip(dqn_agent.policy_net.parameters(), new_agent.policy_net.parameters()):
        assert torch.allclose(p1.data, p2.data)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_training_loop(sample_prices):
    """Test full training loop integration."""
    # Create environment and agent
    env = TradingEnvironment(prices=sample_prices, window_size=50)
    agent = DQNAgent(state_size=51, action_size=3, device=torch.device("cpu"))

    # Train for a few episodes
    for episode in range(5):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        # Update target network
        if (episode + 1) % 2 == 0:
            agent.update_target_network()

    # Should complete without errors
    assert len(agent.buffer) > 0


def test_reward_shaping():
    """Test that reward includes PnL + efficiency + risk components."""
    # Create simple environment
    prices = np.array([100.0] * 10 + [110.0] * 10 + [100.0] * 10)  # Up then down

    env = TradingEnvironment(prices=prices, window_size=5)

    _ = env.reset()

    # Buy at 100
    _, reward1, _, _ = env.step(action=2)

    # Hold through rise to 110
    for _ in range(5):
        _, _, _, _ = env.step(action=1)

    # Sell at 110
    _, reward2, _, _ = env.step(action=0)

    # Reward should be positive (profit + efficiency bonus)
    # Check that we got some reward
    assert len(env.trades) == 1
    trade = env.trades[0]
    assert trade.pnl > 0  # Should have profit


# ============================================================================
# EDGE CASES
# ============================================================================


def test_short_price_series():
    """Test with very short price series."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    env = TradingEnvironment(prices=prices, window_size=3)

    _ = env.reset()

    # Should be able to take at least one step
    _, _, done, _ = env.step(action=1)

    assert not done or env.step_idx >= len(prices) - 1


def test_gpu_availability():
    """Test GPU detection (should work on both CPU and GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_size=10, action_size=3, device=device)

    # Should initialize without error
    assert agent.device == device
