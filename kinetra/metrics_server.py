"""
Prometheus Metrics Server for Kinetra RL Training

Exposes real-time metrics for Grafana visualization:
- Episode stats (trades, win rate, PnL)
- Feature importance
- Physics state (energy, damping, Reynolds)
- Training progress (loss, epsilon)
"""

from prometheus_client import start_http_server, Gauge, Counter, Summary, Info
import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RLMetrics:
    """Container for RL training metrics."""
    episode: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_mae_ratio: float = 0.0
    mfe_captured: float = 0.0
    epsilon: float = 1.0
    loss: float = 0.0
    reward: float = 0.0


class MetricsServer:
    """Prometheus metrics server for RL training."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.started = False

        # === RL TRAINING METRICS ===
        self.episode = Gauge('kinetra_rl_episode', 'Current episode number')
        self.total_trades = Gauge('kinetra_rl_total_trades', 'Total trades in current episode')
        self.win_rate = Gauge('kinetra_rl_win_rate', 'Win rate percentage')
        self.total_pnl = Gauge('kinetra_rl_total_pnl', 'Total P&L percentage')
        self.avg_pnl = Gauge('kinetra_rl_avg_pnl', 'Average P&L per trade')
        self.epsilon = Gauge('kinetra_rl_epsilon', 'Exploration rate')
        self.loss = Gauge('kinetra_rl_loss', 'Training loss')
        self.reward = Gauge('kinetra_rl_reward', 'Episode total reward')

        # === MFE/MAE METRICS ===
        self.avg_mfe = Gauge('kinetra_rl_avg_mfe', 'Average MFE')
        self.avg_mae = Gauge('kinetra_rl_avg_mae', 'Average MAE')
        self.mfe_mae_ratio = Gauge('kinetra_rl_mfe_mae_ratio', 'MFE/MAE efficiency ratio')
        self.mfe_captured = Gauge('kinetra_rl_mfe_captured', 'MFE capture percentage')

        # === PHYSICS STATE METRICS ===
        self.energy_pct = Gauge('kinetra_physics_energy_pct', 'Current energy percentile')
        self.damping_pct = Gauge('kinetra_physics_damping_pct', 'Current damping percentile')
        self.entropy_pct = Gauge('kinetra_physics_entropy_pct', 'Current entropy percentile')
        self.reynolds_pct = Gauge('kinetra_physics_reynolds_pct', 'Current Reynolds percentile')
        self.viscosity_pct = Gauge('kinetra_physics_viscosity_pct', 'Current viscosity percentile')
        self.buying_pressure = Gauge('kinetra_physics_buying_pressure', 'Current buying pressure')

        # === FEATURE IMPORTANCE ===
        self.feature_importance = Gauge(
            'kinetra_rl_feature_importance',
            'Feature importance from RL',
            ['feature']
        )

        # === SIGNAL METRICS ===
        self.is_berserker = Gauge('kinetra_signal_is_berserker', 'Berserker signal active')
        self.flow_regime = Gauge('kinetra_signal_flow_regime', 'Flow regime (0=laminar, 1=trans, 2=turb)')
        self.direction_confidence = Gauge('kinetra_signal_direction_confidence', 'Direction confidence')
        self.magnitude_probability = Gauge('kinetra_signal_magnitude_prob', 'Fat candle probability')

        # === COUNTERS ===
        self.episodes_completed = Counter('kinetra_rl_episodes_completed_total', 'Total episodes completed')
        self.trades_executed = Counter('kinetra_rl_trades_executed_total', 'Total trades executed')
        self.wins = Counter('kinetra_rl_wins_total', 'Total winning trades')
        self.losses = Counter('kinetra_rl_losses_total', 'Total losing trades')

        # === INFO ===
        self.training_info = Info('kinetra_training', 'Training session info')

    def start(self):
        """Start the Prometheus HTTP server."""
        if not self.started:
            start_http_server(self.port)
            self.started = True
            print(f"Prometheus metrics server started on port {self.port}")
            print(f"Access metrics at: http://localhost:{self.port}/metrics")

    def update_rl_metrics(self, metrics: RLMetrics):
        """Update RL training metrics."""
        self.episode.set(metrics.episode)
        self.total_trades.set(metrics.total_trades)
        self.win_rate.set(metrics.win_rate)
        self.total_pnl.set(metrics.total_pnl)
        self.avg_pnl.set(metrics.avg_pnl)
        self.epsilon.set(metrics.epsilon)
        self.loss.set(metrics.loss)
        self.reward.set(metrics.reward)
        self.avg_mfe.set(metrics.avg_mfe)
        self.avg_mae.set(metrics.avg_mae)
        self.mfe_mae_ratio.set(metrics.mfe_mae_ratio)
        self.mfe_captured.set(metrics.mfe_captured)

    def update_physics_state(
        self,
        energy_pct: float,
        damping_pct: float,
        entropy_pct: float,
        reynolds_pct: float,
        viscosity_pct: float,
        buying_pressure: float,
    ):
        """Update physics state metrics."""
        self.energy_pct.set(energy_pct)
        self.damping_pct.set(damping_pct)
        self.entropy_pct.set(entropy_pct)
        self.reynolds_pct.set(reynolds_pct)
        self.viscosity_pct.set(viscosity_pct)
        self.buying_pressure.set(buying_pressure)

    def update_feature_importance(self, importance: Dict[str, float]):
        """Update feature importance from RL."""
        for feature, value in importance.items():
            self.feature_importance.labels(feature=feature).set(value)

    def update_signal(
        self,
        is_berserker: bool,
        flow_regime: int,  # 0=laminar, 1=transitional, 2=turbulent
        direction_confidence: float,
        magnitude_probability: float,
    ):
        """Update signal metrics."""
        self.is_berserker.set(1 if is_berserker else 0)
        self.flow_regime.set(flow_regime)
        self.direction_confidence.set(direction_confidence)
        self.magnitude_probability.set(magnitude_probability)

    def record_trade(self, pnl: float):
        """Record a trade execution."""
        self.trades_executed.inc()
        if pnl > 0:
            self.wins.inc()
        else:
            self.losses.inc()

    def complete_episode(self):
        """Mark episode as completed."""
        self.episodes_completed.inc()

    def set_training_info(self, **kwargs):
        """Set training session info."""
        self.training_info.info(kwargs)


# Global metrics server instance
_metrics_server: Optional[MetricsServer] = None


def get_metrics_server(port: int = 8000) -> MetricsServer:
    """Get or create the global metrics server."""
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer(port)
    return _metrics_server


def start_metrics_server(port: int = 8000):
    """Start the global metrics server."""
    server = get_metrics_server(port)
    server.start()
    return server


if __name__ == "__main__":
    # Test the metrics server
    server = start_metrics_server(8000)

    print("\nSimulating RL training metrics...")
    print("Press Ctrl+C to stop")

    try:
        episode = 0
        while True:
            # Simulate training updates
            episode += 1
            metrics = RLMetrics(
                episode=episode,
                total_trades=episode * 10,
                win_rate=50 + (episode % 20),
                total_pnl=episode * 0.1,
                avg_pnl=0.05,
                avg_mfe=0.8,
                avg_mae=0.5,
                mfe_mae_ratio=1.6,
                mfe_captured=45.0,
                epsilon=max(0.01, 1.0 - episode * 0.01),
                loss=0.1 / (episode + 1),
                reward=episode * 5,
            )
            server.update_rl_metrics(metrics)

            # Simulate physics state
            import random
            server.update_physics_state(
                energy_pct=random.random(),
                damping_pct=random.random(),
                entropy_pct=random.random(),
                reynolds_pct=random.random(),
                viscosity_pct=random.random(),
                buying_pressure=random.random(),
            )

            server.complete_episode()

            print(f"Episode {episode}: WR={metrics.win_rate:.1f}%, PnL={metrics.total_pnl:.2f}%")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped")
