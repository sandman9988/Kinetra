"""
Prometheus Metrics Server for Kinetra RL Training

Exposes real-time metrics for Grafana visualization:
- Episode stats (trades, win rate, PnL)
- Per-instrument stats with timeframe breakdown
- Instrument class aggregates (Forex, Crypto, Indices, Commodities)
- Portfolio-level metrics
- Journey efficiency (MFE capture, MFE-first rate)
- Feature importance
- Physics state (energy, damping, Reynolds)
- Training progress (loss, epsilon)
"""

from prometheus_client import start_http_server, Gauge, Counter, Summary, Info
import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field


# Instrument classification
INSTRUMENT_CLASSES = {
    'BTCUSD': 'crypto',
    'GBPUSD+': 'forex',
    'DJ30ft': 'indices',
    'NAS100': 'indices',
    'Nikkei225': 'indices',
    'UKOUSD': 'commodities',  # Brent Oil
    'XAUUSD+': 'commodities',  # Gold
}

TIMEFRAMES = ['M15', 'M30', 'H1', 'H4']


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


@dataclass
class InstrumentMetrics:
    """Container for per-instrument metrics."""
    instrument: str = ""
    timeframe: str = ""
    episode: int = 0
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_captured: float = 0.0
    mfe_first_rate: float = 0.0  # % of trades where MFE came before MAE
    move_capture: float = 0.0    # % of available move captured
    journey_efficiency: float = 0.0  # Overall trade path quality


class MetricsServer:
    """Prometheus metrics server for RL training."""

    def __init__(self, port: int = 8000):
        self.port = port
        self.started = False
        self._lock = threading.Lock()

        # Track instrument data for aggregations
        self._instrument_data: Dict[str, Dict[str, InstrumentMetrics]] = {}

        # === RL TRAINING METRICS (global) ===
        self.episode = Gauge('kinetra_rl_episode', 'Current episode number')
        self.total_trades = Gauge('kinetra_rl_total_trades', 'Total trades in current episode')
        self.win_rate = Gauge('kinetra_rl_win_rate', 'Win rate percentage')
        self.total_pnl = Gauge('kinetra_rl_total_pnl', 'Total P&L percentage')
        self.avg_pnl = Gauge('kinetra_rl_avg_pnl', 'Average P&L per trade')
        self.epsilon = Gauge('kinetra_rl_epsilon', 'Exploration rate')
        self.loss = Gauge('kinetra_rl_loss', 'Training loss')
        self.reward = Gauge('kinetra_rl_reward', 'Episode total reward')

        # === MFE/MAE METRICS (global) ===
        self.avg_mfe = Gauge('kinetra_rl_avg_mfe', 'Average MFE')
        self.avg_mae = Gauge('kinetra_rl_avg_mae', 'Average MAE')
        self.mfe_mae_ratio = Gauge('kinetra_rl_mfe_mae_ratio', 'MFE/MAE efficiency ratio')
        self.mfe_captured = Gauge('kinetra_rl_mfe_captured', 'MFE capture percentage')

        # === PER-INSTRUMENT METRICS ===
        self.instrument_episode = Gauge(
            'kinetra_instrument_episode', 'Current episode for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_trades = Gauge(
            'kinetra_instrument_trades', 'Total trades for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_wins = Gauge(
            'kinetra_instrument_wins', 'Winning trades for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_win_rate = Gauge(
            'kinetra_instrument_win_rate', 'Win rate for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_pnl = Gauge(
            'kinetra_instrument_pnl', 'Total P&L for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_avg_pnl = Gauge(
            'kinetra_instrument_avg_pnl', 'Average P&L per trade for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_mfe = Gauge(
            'kinetra_instrument_mfe', 'Average MFE for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_mae = Gauge(
            'kinetra_instrument_mae', 'Average MAE for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_mfe_captured = Gauge(
            'kinetra_instrument_mfe_captured', 'MFE capture % for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_mfe_first = Gauge(
            'kinetra_instrument_mfe_first', 'MFE-first rate (trade quality)',
            ['instrument', 'timeframe']
        )
        self.instrument_move_capture = Gauge(
            'kinetra_instrument_move_capture', 'Move capture % for instrument',
            ['instrument', 'timeframe']
        )
        self.instrument_journey_efficiency = Gauge(
            'kinetra_instrument_journey_efficiency', 'Journey efficiency score',
            ['instrument', 'timeframe']
        )

        # === PER-INSTRUMENT CLASS METRICS ===
        self.class_trades = Gauge(
            'kinetra_class_trades', 'Total trades for instrument class',
            ['class']
        )
        self.class_wins = Gauge(
            'kinetra_class_wins', 'Winning trades for instrument class',
            ['class']
        )
        self.class_win_rate = Gauge(
            'kinetra_class_win_rate', 'Win rate for instrument class',
            ['class']
        )
        self.class_pnl = Gauge(
            'kinetra_class_pnl', 'Total P&L for instrument class',
            ['class']
        )
        self.class_avg_pnl = Gauge(
            'kinetra_class_avg_pnl', 'Average P&L for instrument class',
            ['class']
        )
        self.class_journey_efficiency = Gauge(
            'kinetra_class_journey_efficiency', 'Journey efficiency for class',
            ['class']
        )

        # === PORTFOLIO METRICS ===
        self.portfolio_trades = Gauge('kinetra_portfolio_trades', 'Total portfolio trades')
        self.portfolio_wins = Gauge('kinetra_portfolio_wins', 'Total portfolio wins')
        self.portfolio_win_rate = Gauge('kinetra_portfolio_win_rate', 'Portfolio win rate')
        self.portfolio_pnl = Gauge('kinetra_portfolio_pnl', 'Total portfolio P&L')
        self.portfolio_avg_pnl = Gauge('kinetra_portfolio_avg_pnl', 'Portfolio average P&L')
        self.portfolio_journey_efficiency = Gauge(
            'kinetra_portfolio_journey_efficiency', 'Portfolio journey efficiency'
        )
        self.portfolio_instruments_active = Gauge(
            'kinetra_portfolio_instruments_active', 'Number of active instruments'
        )

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

    def start(self, max_retries: int = 5):
        """Start the Prometheus HTTP server with automatic port fallback.

        If the port is in use, tries subsequent ports up to max_retries times.
        If all ports fail, continues without metrics server.
        """
        if self.started:
            return True

        original_port = self.port

        for attempt in range(max_retries):
            try:
                start_http_server(self.port)
                self.started = True
                if self.port != original_port:
                    print(f"Prometheus metrics server started on port {self.port} (original {original_port} was in use)")
                else:
                    print(f"Prometheus metrics server started on port {self.port}")
                print(f"Access metrics at: http://localhost:{self.port}/metrics")
                return True
            except OSError as e:
                if "Address already in use" in str(e):
                    self.port += 1
                else:
                    print(f"Failed to start metrics server: {e}")
                    break

        # All ports failed - continue without metrics
        print(f"WARNING: Could not start metrics server (ports {original_port}-{self.port} in use)")
        print("Training will continue without Prometheus metrics.")
        self.started = False
        return False

    def update_rl_metrics(self, metrics: RLMetrics):
        """Update RL training metrics."""
        if not self.started:
            return
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
        if not self.started:
            return
        self.energy_pct.set(energy_pct)
        self.damping_pct.set(damping_pct)
        self.entropy_pct.set(entropy_pct)
        self.reynolds_pct.set(reynolds_pct)
        self.viscosity_pct.set(viscosity_pct)
        self.buying_pressure.set(buying_pressure)

    def update_feature_importance(self, importance: Dict[str, float]):
        """Update feature importance from RL."""
        if not self.started:
            return
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
        if not self.started:
            return
        self.is_berserker.set(1 if is_berserker else 0)
        self.flow_regime.set(flow_regime)
        self.direction_confidence.set(direction_confidence)
        self.magnitude_probability.set(magnitude_probability)

    def record_trade(self, pnl: float):
        """Record a trade execution."""
        if not self.started:
            return
        self.trades_executed.inc()
        if pnl > 0:
            self.wins.inc()
        else:
            self.losses.inc()

    def update_instrument_metrics(self, metrics: InstrumentMetrics):
        """Update per-instrument metrics and recalculate aggregates."""
        if not self.started:
            return

        instrument = metrics.instrument
        timeframe = metrics.timeframe

        # Store for aggregation
        with self._lock:
            if instrument not in self._instrument_data:
                self._instrument_data[instrument] = {}
            self._instrument_data[instrument][timeframe] = metrics

        # Update per-instrument gauges
        labels = {'instrument': instrument, 'timeframe': timeframe}
        self.instrument_episode.labels(**labels).set(metrics.episode)
        self.instrument_trades.labels(**labels).set(metrics.trades)
        self.instrument_wins.labels(**labels).set(metrics.wins)
        self.instrument_win_rate.labels(**labels).set(metrics.win_rate)
        self.instrument_pnl.labels(**labels).set(metrics.total_pnl)
        self.instrument_avg_pnl.labels(**labels).set(metrics.avg_pnl)
        self.instrument_mfe.labels(**labels).set(metrics.avg_mfe)
        self.instrument_mae.labels(**labels).set(metrics.avg_mae)
        self.instrument_mfe_captured.labels(**labels).set(metrics.mfe_captured)
        self.instrument_mfe_first.labels(**labels).set(metrics.mfe_first_rate)
        self.instrument_move_capture.labels(**labels).set(metrics.move_capture)
        self.instrument_journey_efficiency.labels(**labels).set(metrics.journey_efficiency)

        # Recalculate aggregates
        self._update_aggregates()

    def _update_aggregates(self):
        """Recalculate class and portfolio aggregates from instrument data."""
        if not self.started:
            return

        with self._lock:
            # Aggregate by class
            class_stats: Dict[str, Dict] = {
                'crypto': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'journey_eff': []},
                'forex': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'journey_eff': []},
                'indices': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'journey_eff': []},
                'commodities': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'journey_eff': []},
            }

            # Portfolio totals
            portfolio_trades = 0
            portfolio_wins = 0
            portfolio_pnl = 0.0
            portfolio_journey_eff = []

            for instrument, timeframes in self._instrument_data.items():
                inst_class = INSTRUMENT_CLASSES.get(instrument, 'other')
                if inst_class not in class_stats:
                    class_stats[inst_class] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'journey_eff': []}

                for tf, m in timeframes.items():
                    # Class aggregation
                    class_stats[inst_class]['trades'] += m.trades
                    class_stats[inst_class]['wins'] += m.wins
                    class_stats[inst_class]['pnl'] += m.total_pnl
                    if m.journey_efficiency > 0:
                        class_stats[inst_class]['journey_eff'].append(m.journey_efficiency)

                    # Portfolio aggregation
                    portfolio_trades += m.trades
                    portfolio_wins += m.wins
                    portfolio_pnl += m.total_pnl
                    if m.journey_efficiency > 0:
                        portfolio_journey_eff.append(m.journey_efficiency)

            # Update class metrics
            for cls, stats in class_stats.items():
                self.class_trades.labels(**{'class': cls}).set(stats['trades'])
                self.class_wins.labels(**{'class': cls}).set(stats['wins'])
                win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                self.class_win_rate.labels(**{'class': cls}).set(win_rate)
                self.class_pnl.labels(**{'class': cls}).set(stats['pnl'])
                avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                self.class_avg_pnl.labels(**{'class': cls}).set(avg_pnl)
                avg_journey = sum(stats['journey_eff']) / len(stats['journey_eff']) if stats['journey_eff'] else 0
                self.class_journey_efficiency.labels(**{'class': cls}).set(avg_journey)

            # Update portfolio metrics
            self.portfolio_trades.set(portfolio_trades)
            self.portfolio_wins.set(portfolio_wins)
            self.portfolio_win_rate.set(
                (portfolio_wins / portfolio_trades * 100) if portfolio_trades > 0 else 0
            )
            self.portfolio_pnl.set(portfolio_pnl)
            self.portfolio_avg_pnl.set(
                portfolio_pnl / portfolio_trades if portfolio_trades > 0 else 0
            )
            self.portfolio_journey_efficiency.set(
                sum(portfolio_journey_eff) / len(portfolio_journey_eff) if portfolio_journey_eff else 0
            )
            self.portfolio_instruments_active.set(len(self._instrument_data))

    def complete_episode(self):
        """Mark episode as completed."""
        if not self.started:
            return
        self.episodes_completed.inc()

    def set_training_info(self, **kwargs):
        """Set training session info."""
        if not self.started:
            return
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
