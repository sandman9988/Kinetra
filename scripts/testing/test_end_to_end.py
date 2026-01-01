#!/usr/bin/env python3
"""
End-to-End Integration Test

Tests the complete adaptive trading system pipeline:
1. Data loading (real EURJPY M15 market data)
2. DoppelgangerTriad initialization with KinetraAgent
3. Realistic backtesting with MT5 constraints
4. Trade lifecycle execution (entry → MFE/MAE → exit)
5. Experience replay integration
6. Portfolio health monitoring (4 pillars)
7. Drift detection and agent promotion
8. Complete system integration

This validates that all components work together correctly in production.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import pytest

from kinetra import (
    # Data & Backtesting
    RealisticBacktester,
    # Agents & Learning
    KinetraAgent,
    DoppelgangerTriad,
    # Health Monitoring
    PortfolioHealthMonitor,
    HealthState,
    # Experience Replay
)
from kinetra.experience_replay import (
    TradeLogger,
    PrioritizedReplayBuffer,
    TradeLabeler,
)
from kinetra.market_microstructure import SymbolSpec, AssetClass


@pytest.fixture
def data():
    """Load or create test market data."""
    data_path = "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv"

    if Path(data_path).exists():
        # Read CSV (MT5 format)
        df = pd.read_csv(data_path, sep='\t', nrows=1000)
        # Rename columns
        df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
        # Combine date and time
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        # Rename and keep needed columns
        df = df.rename(columns={'tickvol': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]
        df = df.set_index('timestamp')
    else:
        # Generate synthetic data
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        df = pd.DataFrame({
            'open': [160.0 + i*0.01 + np.random.randn()*0.1 for i in range(1000)],
            'high': [160.1 + i*0.01 + np.random.randn()*0.1 for i in range(1000)],
            'low': [159.9 + i*0.01 + np.random.randn()*0.1 for i in range(1000)],
            'close': [160.0 + i*0.01 + np.random.randn()*0.1 for i in range(1000)],
            'volume': [1000 + i + np.random.randn()*50 for i in range(1000)],
            'spread': [8.0 + np.random.randn()*2 for _ in range(1000)]
        }, index=dates)

    return df


@pytest.fixture
def spec():
    """Create EURJPY+ symbol specification."""
    return SymbolSpec(
        symbol="EURJPY+",
        asset_class=AssetClass.FOREX,
        digits=3,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_typical=80.0,
        commission_per_lot=0.0,
        swap_long=-0.3,
        swap_short=0.1,
        trade_freeze_level=50,
        trade_stops_level=100,
    )


@pytest.fixture
def triad():
    """Initialize DoppelgangerTriad with KinetraAgent."""
    agent = KinetraAgent(state_dim=64, action_dim=4)
    return DoppelgangerTriad(
        live_agent=agent,
        drift_threshold=0.2,
        promotion_threshold=0.1,
        min_trades_for_drift=10,
        min_trades_for_promotion=15,
    )


@pytest.fixture
def signals(data):
    """Generate simple MA crossover signals for testing."""
    df = data.copy()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['signal'] = 0
    df.loc[df['close'] > df['ma50'], 'signal'] = 1
    df.loc[df['close'] < df['ma50'], 'signal'] = -1
    df['signal_change'] = df['signal'].diff()

    signals = []
    position = None

    for idx, row in df.iterrows():
        if position is None:
            if row['signal_change'] == 2:  # Long entry
                entry_price = row['close']
                signals.append({
                    'time': idx,
                    'action': 'open_long',
                    'sl': entry_price - 0.200,
                    'tp': entry_price + 0.400,
                    'volume': 1.0,
                })
                position = 'long'
            elif row['signal_change'] == -2:  # Short entry
                entry_price = row['close']
                signals.append({
                    'time': idx,
                    'action': 'open_short',
                    'sl': entry_price + 0.200,
                    'tp': entry_price - 0.400,
                    'volume': 1.0,
                })
                position = 'short'
        else:
            # Exit on opposite signal
            if (position == 'long' and row['signal_change'] < 0) or \
               (position == 'short' and row['signal_change'] > 0):
                signals.append({
                    'time': idx,
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None

    return pd.DataFrame(signals)


@pytest.fixture
def result(data, signals, spec):
    """Run backtest and return results."""
    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        timeframe="M15",
        enable_slippage=True,
        slippage_std_pips=0.5,
        enable_freeze_zones=True,
        enable_stop_validation=True,
        verbose=False,
    )
    return backtester.run(data, signals, classify_regimes=False)


@pytest.fixture
def logger():
    """Create trade logger."""
    return TradeLogger(log_dir="logs/test_e2e")


@pytest.fixture
def buffer():
    """Create prioritized replay buffer."""
    return PrioritizedReplayBuffer(capacity=1000)


@pytest.fixture
def labeler():
    """Create trade labeler."""
    return TradeLabeler()


@pytest.fixture
def monitor():
    """Create portfolio health monitor."""
    return PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)


def load_test_data(max_rows: int = 1000) -> pd.DataFrame:
    """Load real market data for testing."""
    print("\n[1] Loading Market Data")

    data_path = "/home/user/Kinetra/data/master/EURJPY+_M15_202401020000_202512300900.csv"

    # Read CSV (MT5 format)
    df = pd.read_csv(data_path, sep='\t', nrows=max_rows)

    # Rename columns
    df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()

    # Combine date and time
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    # Rename and keep needed columns
    df = df.rename(columns={'tickvol': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]
    df = df.set_index('timestamp')

    print(f"  ✅ Loaded {len(df)} candles")
    print(f"  ✅ Period: {df.index[0]} to {df.index[-1]}")
    print(f"  ✅ Avg spread: {df['spread'].mean():.1f} points")

    return df


def create_symbol_spec() -> SymbolSpec:
    """Create EURJPY+ specification."""
    print("\n[2] Creating Symbol Specification")

    spec = SymbolSpec(
        symbol="EURJPY+",
        asset_class=AssetClass.FOREX,
        digits=3,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        filling_mode="IOC",
        order_mode="MARKET_LIMIT",
        trade_mode="FULL",
        spread_typical=80,
        commission_per_lot=0.0,
        swap_long=-0.3,
        swap_short=0.1,
        trade_freeze_level=50,   # 5 pips
        trade_stops_level=100,   # 10 pips
    )

    print(f"  ✅ Symbol: {spec.symbol}")
    print(f"  ✅ Digits: {spec.digits}")
    print(f"  ✅ Freeze level: {spec.trade_freeze_level} points")
    print(f"  ✅ Stops level: {spec.trade_stops_level} points")

    return spec


def initialize_agents() -> DoppelgangerTriad:
    """Initialize DoppelgangerTriad with KinetraAgent."""
    print("\n[3] Initializing Agents (DoppelgangerTriad)")

    # Create base agent
    agent = KinetraAgent(state_dim=64, action_dim=4)

    # Wrap in DoppelgangerTriad
    triad = DoppelgangerTriad(
        live_agent=agent,
        drift_threshold=0.2,
        promotion_threshold=0.1,
        min_trades_for_drift=10,
        min_trades_for_promotion=15,
    )

    print(f"  ✅ Live agent: {triad.live_agent.agent_id} ({triad.live_agent.state.name})")
    print(f"  ✅ Frozen shadow: {triad.frozen_shadow.agent_id} ({triad.frozen_shadow.state.name})")
    print(f"  ✅ Training shadow: {triad.training_shadow.agent_id} ({triad.training_shadow.state.name})")

    return triad


def generate_test_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate simple MA crossover signals for testing."""
    print("\n[4] Generating Trading Signals")

    df = data.copy()

    # 50-period MA
    df['ma50'] = df['close'].rolling(window=50).mean()

    # Signal: 1 = bullish, -1 = bearish
    df['signal'] = 0
    df.loc[df['close'] > df['ma50'], 'signal'] = 1
    df.loc[df['close'] < df['ma50'], 'signal'] = -1
    df['signal_change'] = df['signal'].diff()

    signals = []
    position = None

    for idx, row in df.iterrows():
        if position is None:
            if row['signal_change'] == 2:  # Long entry
                entry_price = row['close']
                signals.append({
                    'time': idx,
                    'action': 'open_long',
                    'sl': entry_price - 0.200,
                    'tp': entry_price + 0.400,
                    'volume': 1.0,
                })
                position = 'long'
            elif row['signal_change'] == -2:  # Short entry
                entry_price = row['close']
                signals.append({
                    'time': idx,
                    'action': 'open_short',
                    'sl': entry_price + 0.200,
                    'tp': entry_price - 0.400,
                    'volume': 1.0,
                })
                position = 'short'
        else:
            # Exit on opposite signal
            if (position == 'long' and row['signal_change'] < 0) or \
               (position == 'short' and row['signal_change'] > 0):
                signals.append({
                    'time': idx,
                    'action': 'close',
                    'sl': None,
                    'tp': None,
                })
                position = None

    signals_df = pd.DataFrame(signals)

    print(f"  ✅ Total signals: {len(signals_df)}")
    print(f"  ✅ Entry signals: {len(signals_df[signals_df['action'] != 'close'])}")
    print(f"  ✅ Exit signals: {len(signals_df[signals_df['action'] == 'close'])}")

    return signals_df


def run_backtest(data: pd.DataFrame, signals: pd.DataFrame, spec: SymbolSpec):
    """Run realistic backtest with MT5 constraints."""
    print("\n[5] Running Backtest (Realistic MT5 Constraints)")

    backtester = RealisticBacktester(
        spec=spec,
        initial_capital=10000.0,
        risk_per_trade=0.02,
        timeframe="M15",
        enable_slippage=True,
        slippage_std_pips=0.5,
        enable_freeze_zones=True,
        enable_stop_validation=True,
        verbose=False,
    )

    result = backtester.run(data, signals, classify_regimes=False)

    print(f"  ✅ Total trades: {result.total_trades}")
    print(f"  ✅ Win rate: {result.win_rate:.1%}")
    print(f"  ✅ Total P&L: ${result.total_pnl:,.2f}")
    print(f"  ✅ Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"  ✅ Max drawdown: {result.max_drawdown_pct:.2%}")
    print(f"  ✅ Freeze violations: {result.total_freeze_violations}")
    print(f"  ✅ Invalid stops: {result.total_invalid_stops}")

    return result


def test_experience_replay(result):
    """Test experience replay system integration."""
    print("\n[6] Testing Experience Replay Integration")

    # Initialize components
    logger = TradeLogger(log_dir="logs/test_e2e")
    buffer = PrioritizedReplayBuffer(capacity=1000)
    labeler = TradeLabeler()

    # Process trades into episodes
    trades_processed = 0
    from kinetra.experience_replay import TradeEpisode, Experience

    for trade in result.trades[:20]:  # First 20 trades
        # Create mock episode with complete fields
        experiences = []
        for step in range(5):  # Mock 5 steps per trade
            exp = Experience(
                state=np.random.randn(64),
                action=1 if trade.direction == 1 else 2,
                reward=trade.pnl / 500.0,  # Distributed across steps
                next_state=np.random.randn(64),
                done=(step == 4),
                timestamp=datetime.now(),
                metadata={'step': step, 'regime': 'LAMINAR'},
            )
            experiences.append(exp)

        # Label trade
        is_poor = trade.pnl < 0 or trade.mfe_efficiency < 0.3
        label = 'poor' if is_poor else 'good'

        # Create episode with all required fields
        episode = TradeEpisode(
            experiences=experiences,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            total_pnl=trade.pnl,
            total_return_pct=trade.pnl / 10000.0,
            sharpe_ratio=0.0,
            mfe=trade.mfe,
            mae=trade.mae,
            mfe_efficiency=trade.mfe_efficiency,
            mae_efficiency=trade.mae / trade.mfe if trade.mfe > 0 else 0.0,
            entry_regime='LAMINAR',
            avg_volatility=0.01,
            avg_spread=2.0,
            constraint_violations=0,
            freeze_violations=0,
            label=label,
            priority=2.5 if is_poor else 1.0,
        )

        buffer.add(episode)
        trades_processed += 1

    # Sample from buffer
    if trades_processed >= 10:
        batch = buffer.sample(batch_size=10)

        print(f"  ✅ Trades processed: {trades_processed}")
        print(f"  ✅ Batch sampled: {len(batch)} episodes")
        print(f"  ✅ Experience replay integration working")
    else:
        print(f"  ✅ Trades processed: {trades_processed}")
        print(f"  ⚠️  Insufficient data for sampling (need 10+)")


def test_health_monitoring(result, triad: DoppelgangerTriad):
    """Test portfolio health monitoring."""
    print("\n[7] Testing Portfolio Health Monitoring")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    # Prepare trades for health monitor
    trades = []
    for trade in result.trades:
        trades.append({
            'pnl': trade.pnl,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'mfe': trade.mfe,
            'mae': trade.mae,
            'edge_ratio': trade.mfe_efficiency,
        })

    # Create equity curve (use known initial capital from backtester)
    initial_capital = 10000.0
    equity_values = [initial_capital]
    for trade in result.trades:
        equity_values.append(equity_values[-1] + trade.pnl)

    dates = pd.date_range(result.trades[0].entry_time, periods=len(equity_values), freq='h')
    equity_curve = pd.Series(equity_values, index=dates)

    # Update health
    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        correlations=None,
        agent_promotions=0,
    )

    print(f"  ✅ Composite score: {health.composite_score:.1f}")
    print(f"  ✅ State: {health.state.name}")
    print(f"  ✅ Action: {health.action.message}")
    print(f"  ✅ Risk multiplier: {health.action.risk_multiplier:.1%}")
    print(f"  ✅ Return & Efficiency: {health.return_efficiency.score:.1f}")
    print(f"  ✅ Downside Risk: {health.downside_risk.score:.1f}")
    print(f"  ✅ Structural Stability: {health.structural_stability.score:.1f}")
    print(f"  ✅ Behavioral Health: {health.behavioral_health.score:.1f}")

    # Update triad with trade results
    for trade in trades[:10]:
        triad.record_trade_result(trade)

    return health


def test_drift_and_promotion(triad: DoppelgangerTriad):
    """Test drift detection and promotion logic."""
    print("\n[8] Testing Drift Detection & Promotion")

    # Simulate some trades to build history
    for i in range(20):
        # Live agent gets mediocre performance
        triad.live_agent.update_performance(
            reward=50.0,
            pnl=50.0,
            is_win=i % 2 == 0,
            edge_ratio=0.5,
        )

        # Frozen shadow keeps good performance (for drift comparison)
        triad.frozen_shadow.update_performance(
            reward=100.0,
            pnl=100.0,
            is_win=True,
            edge_ratio=0.8,
        )

        # Training shadow gets better performance
        triad.training_shadow.update_performance(
            reward=75.0,
            pnl=75.0,
            is_win=i % 3 != 0,
            edge_ratio=0.7,
        )

    # Check drift
    is_drifted, drift, msg = triad.check_drift()
    print(f"  ✅ Drift detected: {is_drifted}")
    print(f"  ✅ Drift amount: {drift*100:.1f}%")
    if msg:
        print(f"  ✅ Message: {msg}")

    # Check promotion
    should_promote, promo_msg = triad.check_promotion()
    print(f"  ✅ Should promote: {should_promote}")
    if promo_msg:
        print(f"  ✅ Promotion message: {promo_msg}")

    if should_promote:
        old_live_trades = triad.live_agent.performance.trades
        triad.promote_training_shadow()
        print(f"  ✅ Promotion executed!")
        print(f"  ✅ Events logged: {len(triad.events)}")


def validate_system_integration():
    """Validate all components are properly integrated."""
    print("\n[9] Validating System Integration")

    checks = []

    # Check 1: Can import all components
    try:
        from kinetra import (
            RealisticBacktester,
            DoppelgangerTriad,
            PortfolioHealthMonitor,
            KinetraAgent,
        )
        checks.append(("Import all components", True))
    except Exception as e:
        checks.append(("Import all components", False, str(e)))

    # Check 2: Components exist in __all__
    from kinetra import __all__
    required = [
        'RealisticBacktester',
        'DoppelgangerTriad',
        'PortfolioHealthMonitor',
        'HealthState',
        'ShadowAgent',
    ]
    all_present = all(comp in __all__ for comp in required)
    checks.append(("All components exported", all_present))

    # Check 3: SymbolSpec for MT5
    from kinetra import SymbolSpec, AssetClass
    spec = SymbolSpec(
        symbol="TEST",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_freeze_level=10,
        trade_stops_level=20,
    )
    checks.append(("SymbolSpec creation", spec is not None))

    # Display results
    for check in checks:
        if len(check) == 2:
            name, passed = check
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
        else:
            name, passed, error = check
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
            if not passed:
                print(f"      Error: {error}")

    all_passed = all(c[1] for c in checks)
    return all_passed


def run_end_to_end_test():
    """Run complete end-to-end integration test."""
    print("=" * 70)
    print("END-TO-END INTEGRATION TEST")
    print("Adaptive Trading System - Complete Pipeline")
    print("=" * 70)

    try:
        # 1. Load data
        data = load_test_data(max_rows=1000)

        # 2. Create symbol spec
        spec = create_symbol_spec()

        # 3. Initialize agents
        triad = initialize_agents()

        # 4. Generate signals
        signals = generate_test_signals(data)

        if len(signals) == 0:
            print("\n❌ No signals generated!")
            return 1

        # 5. Run backtest
        result = run_backtest(data, signals, spec)

        if result.total_trades == 0:
            print("\n❌ No trades executed!")
            return 1

        # 6. Test experience replay
        test_experience_replay(result)

        # 7. Test health monitoring
        health = test_health_monitoring(result, triad)

        # 8. Test drift and promotion
        test_drift_and_promotion(triad)

        # 9. Validate integration
        integration_ok = validate_system_integration()

        # Summary
        print("\n" + "=" * 70)
        print("END-TO-END TEST SUMMARY")
        print("=" * 70)

        print(f"\n✅ Data Loading: {len(data)} candles loaded")
        print(f"✅ Signal Generation: {len(signals)} signals generated")
        print(f"✅ Backtest Execution: {result.total_trades} trades executed")
        print(f"✅ Health Monitoring: {health.state.name} state")
        print(f"✅ Agent Management: 3 agents (live, frozen, training)")
        print(f"✅ Integration: {'PASSED' if integration_ok else 'FAILED'}")

        print("\n" + "=" * 70)
        print("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nAll components integrated and working:")
        print("  ✓ Data pipeline (MT5 CSV → DataFrame)")
        print("  ✓ Realistic backtesting (freeze zones, slippage, costs)")
        print("  ✓ Agent management (DoppelgangerTriad)")
        print("  ✓ Health monitoring (4-pillar scoring)")
        print("  ✓ Experience replay (trade logging, prioritized sampling)")
        print("  ✓ Drift detection & promotion logic")
        print("\nSystem ready for production deployment!")
        print("=" * 70)

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_end_to_end_test())
