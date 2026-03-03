#!/usr/bin/env python3
"""
Complete Denoise + DRL Workflow Example
========================================

Demonstrates end-to-end workflow:
1. Load data
2. Denoise using Savitzky-Golay
3. Train Dueling DQN
4. Backtest and analyze

Usage:
    python scripts/examples/denoise_drl_example.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.denoise_filters import DenoiseMethod, denoise_ohlc  # noqa: E402
from kinetra.drl_dueling_dqn import DQNAgent, TradingEnvironment  # noqa: E402


def main():
    print("=" * 80)
    print("DENOISE + DRL WORKFLOW EXAMPLE")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Generate/Load Sample Data
    # ========================================================================
    print("\n📊 Step 1: Generate Sample Data")
    print("-" * 80)

    # Generate synthetic BTCUSD-like data (2024 bull market)
    np.random.seed(42)
    n_bars = 2000

    # Bull trend + volatility + noise
    t = np.arange(n_bars)
    trend = 40000 + 3000 * (t / n_bars)  # Rising from 40k to 43k
    volatility = 2000 * np.sin(2 * np.pi * t / 200)  # Cycles
    noise = np.random.normal(0, 500, n_bars)  # High-frequency noise

    close = trend + volatility + noise

    # Create OHLC
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n_bars, freq="30min"),
            "open": close + np.random.normal(0, 100, n_bars),
            "high": close + np.abs(np.random.normal(200, 100, n_bars)),
            "low": close - np.abs(np.random.normal(200, 100, n_bars)),
            "close": close,
            "volume": np.random.randint(100, 1000, n_bars),
        }
    )

    print(f"✅ Generated {len(df)} bars")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # ========================================================================
    # STEP 2: Denoise Data
    # ========================================================================
    print("\n🔬 Step 2: Denoise Data (Savitzky-Golay)")
    print("-" * 80)

    # Denoise all OHLC columns
    df_denoised = denoise_ohlc(df, method=DenoiseMethod.SAVGOL)

    # Calculate metrics
    original_vol = df["close"].pct_change().std()
    denoised_vol = df_denoised["close_denoised"].pct_change().std()
    reduction = (1 - denoised_vol / original_vol) * 100

    print("✅ Denoising complete")
    print(f"   Original volatility: {original_vol*100:.3f}%")
    print(f"   Denoised volatility: {denoised_vol*100:.3f}%")
    print(f"   Noise reduction: {reduction:.1f}%")

    # ========================================================================
    # STEP 3: Train DQN Agent
    # ========================================================================
    print("\n🤖 Step 3: Train Dueling DQN Agent")
    print("-" * 80)

    prices = df["close"].values
    prices_denoised = df_denoised["close_denoised"].values

    # Create environment
    env = TradingEnvironment(
        prices=prices,
        prices_denoised=prices_denoised,
        window_size=50,
        commission=0.0005,
    )

    # Create agent
    agent = DQNAgent(state_size=51, action_size=3, lr=1e-4)

    print("✅ Environment and agent created")
    print(f"   Device: {agent.device}")
    print("   State size: 51 (50 returns + position)")
    print("   Action size: 3 (flat, hold, long)")

    # Training loop
    num_episodes = 50
    rewards_history = []

    print(f"\n🚀 Training for {num_episodes} episodes...")

    for episode in range(num_episodes):
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
        if (episode + 1) % 10 == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"   Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f}")

    print("\n✅ Training complete")

    # ========================================================================
    # STEP 4: Backtest Trained Agent
    # ========================================================================
    print("\n📈 Step 4: Backtest Trained Agent")
    print("-" * 80)

    # Reset environment
    state = env.reset()
    equity_curve = [env.initial_cash]
    positions = [0]

    while True:
        action = agent.select_action(state, training=False)  # Greedy
        state, _, done, _ = env.step(action)

        # Track equity
        current_price = env.prices[env.step_idx - 1]
        equity = env.cash + current_price * env.position * 10  # 10 BTC equivalent
        equity_curve.append(equity)
        positions.append(env.position)

        if done:
            break

    # Calculate metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    max_equity = max(equity_curve)
    max_drawdown = (max_equity - min(equity_curve)) / max_equity

    print("✅ Backtest Results:")
    print(f"   Total Return: {total_return*100:.2f}%")
    print(f"   Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"   Total Trades: {len(env.trades)}")

    # MAE/MFE analysis
    if env.trades:
        avg_mfe = np.mean([t.mfe for t in env.trades])
        avg_mae = np.mean([abs(t.mae) for t in env.trades])
        avg_pnl = np.mean([t.pnl for t in env.trades])

        print(f"   Avg MFE: {avg_mfe*100:.2f}%")
        print(f"   Avg MAE: {avg_mae*100:.2f}%")
        print(f"   Avg PnL per trade: ${avg_pnl:.2f}")
        print(f"   Efficiency (MFE/MAE): {avg_mfe/avg_mae:.2f}x")

    # ========================================================================
    # STEP 5: Visualize Results
    # ========================================================================
    print("\n📊 Step 5: Visualize Results")
    print("-" * 80)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Original vs Denoised Prices
    ax1 = axes[0]
    ax1.plot(df["close"].values[:500], alpha=0.6, label="Original", linewidth=1)
    ax1.plot(
        df_denoised["close_denoised"].values[:500],
        label="Denoised",
        linewidth=2,
        color="red",
    )
    ax1.set_title("Original vs Denoised Prices (First 500 bars)")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Rewards
    ax2 = axes[1]
    ax2.plot(rewards_history, alpha=0.6, label="Episode Reward")
    ax2.plot(
        pd.Series(rewards_history).rolling(5).mean(),
        linewidth=2,
        label="MA(5)",
        color="red",
    )
    ax2.set_title("Training Rewards per Episode")
    ax2.set_ylabel("Total Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Equity Curve
    ax3 = axes[2]
    ax3.plot(equity_curve, linewidth=2)
    ax3.set_title(f"Equity Curve (Return: {total_return*100:.2f}%, Drawdown: {max_drawdown*100:.2f}%)")
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Positions Over Time
    ax4 = axes[3]
    ax4.plot(positions, linewidth=1, alpha=0.7)
    ax4.set_title("Position Over Time (1=Long, 0=Flat, -1=Short)")
    ax4.set_ylabel("Position")
    ax4.set_xlabel("Time Step")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.5, 1.5)

    plt.tight_layout()

    # Save figure
    output_dir = PROJECT_ROOT / "results" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "denoise_drl_example.png"
    plt.savefig(output_file, dpi=150)

    print(f"✅ Visualization saved to: {output_file}")

    # Show plot
    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE ✅")
    print("=" * 80)
    print("\nKey Takeaways:")
    print(f"  • Denoising reduced volatility by {reduction:.1f}%")
    print(f"  • DQN agent trained in {num_episodes} episodes")
    print(f"  • Achieved {total_return*100:.2f}% return with {max_drawdown*100:.2f}% max drawdown")
    print(f"  • Agent executed {len(env.trades)} trades")
    if env.trades:
        print(f"  • Average efficiency (MFE/MAE): {avg_mfe/avg_mae:.2f}x")
    print("\nNext Steps:")
    print("  • Try different denoising methods (median, lowess, wavelet)")
    print("  • Increase training episodes for better convergence")
    print("  • Experiment with reward shaping weights")
    print("  • Test on real market data (BTCUSD, EURUSD, etc.)")
    print("=" * 80)


if __name__ == "__main__":
    main()
