#!/usr/bin/env python3
"""
Terminal Dashboard for Kinetra RL Training

Shows live metrics in the terminal - no browser needed.
"""

import sys
import time
import urllib.request
import re
from datetime import datetime

METRICS_URLS = [
    "http://localhost:8000/metrics",
    "http://localhost:8001/metrics",
]


def parse_metrics(text: str) -> dict:
    """Parse Prometheus metrics text format."""
    metrics = {}
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        # Handle labeled metrics
        match = re.match(r'^(\w+)\{([^}]+)\}\s+([\d.eE+-]+)', line)
        if match:
            name, labels, value = match.groups()
            if name not in metrics:
                metrics[name] = {}
            label_match = re.search(r'feature="([^"]+)"', labels)
            if label_match:
                metrics[name][label_match.group(1)] = float(value)
        else:
            match = re.match(r'^(\w+)\s+([\d.eE+-]+)', line)
            if match:
                metrics[match.group(1)] = float(match.group(2))
    return metrics


def fetch_metrics(url: str) -> dict:
    """Fetch metrics from Prometheus endpoint."""
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return parse_metrics(response.read().decode())
    except Exception:
        return {}


def clear_screen():
    print("\033[2J\033[H", end="")


def format_pnl(val: float) -> str:
    if val >= 0:
        return f"\033[32m+{val:.2f}%\033[0m"  # Green
    return f"\033[31m{val:.2f}%\033[0m"  # Red


def format_bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def display_dashboard(metrics: dict, port: int):
    """Display metrics in terminal."""
    clear_screen()

    print("=" * 60)
    print(f"  🔥 KINETRA RL TRAINING DASHBOARD (port {port})")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Training stats
    episode = metrics.get('kinetra_rl_episode', 0)
    trades = metrics.get('kinetra_rl_total_trades', 0)
    win_rate = metrics.get('kinetra_rl_win_rate', 0)
    pnl = metrics.get('kinetra_rl_total_pnl', 0)
    epsilon = metrics.get('kinetra_rl_epsilon', 1.0)
    loss = metrics.get('kinetra_rl_loss', 0)

    print(f"\n  📊 TRAINING PROGRESS")
    print(f"  ─────────────────────────────────────────")
    print(f"  Episode:    {int(episode):>6}")
    print(f"  Trades:     {int(trades):>6}")
    print(f"  Win Rate:   {win_rate:>5.1f}%  {format_bar(win_rate/100)}")
    print(f"  P&L:        {format_pnl(pnl)}")
    print(f"  Epsilon:    {epsilon:>5.3f}  {format_bar(epsilon)}")
    print(f"  Loss:       {loss:.4f}")

    # MFE/MAE
    mfe = metrics.get('kinetra_rl_avg_mfe', 0)
    mae = metrics.get('kinetra_rl_avg_mae', 0)
    ratio = metrics.get('kinetra_rl_mfe_mae_ratio', 0)

    print(f"\n  📈 TRADE QUALITY (MFE/MAE)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg MFE:      {mfe:.3f}%")
    print(f"  Avg MAE:      {mae:.3f}%")
    print(f"  MFE/MAE:      {ratio:.2f}")

    # Physics state
    energy = metrics.get('kinetra_physics_energy_pct', 0.5)
    damping = metrics.get('kinetra_physics_damping_pct', 0.5)
    reynolds = metrics.get('kinetra_physics_reynolds_pct', 0.5)
    viscosity = metrics.get('kinetra_physics_viscosity_pct', 0.5)
    bp = metrics.get('kinetra_physics_buying_pressure', 0.5)

    print(f"\n  ⚡ PHYSICS STATE")
    print(f"  ─────────────────────────────────────────")
    print(f"  Energy:     {energy*100:>5.0f}%  {format_bar(energy)}")
    print(f"  Damping:    {damping*100:>5.0f}%  {format_bar(damping)}")
    print(f"  Reynolds:   {reynolds*100:>5.0f}%  {format_bar(reynolds)}")
    print(f"  Viscosity:  {viscosity*100:>5.0f}%  {format_bar(viscosity)}")
    print(f"  Buy Press:  {bp:.2f}")

    # Feature importance
    fi = metrics.get('kinetra_rl_feature_importance', {})
    if fi:
        print(f"\n  🧠 FEATURE IMPORTANCE (RL Learned)")
        print(f"  ─────────────────────────────────────────")
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
        max_val = max(v for _, v in sorted_fi) if sorted_fi else 1
        for name, val in sorted_fi:
            bar = format_bar(val / max_val, width=15)
            print(f"  {name:<16} {val*100:>5.1f}%  {bar}")

    print(f"\n  Press Ctrl+C to exit")


def main():
    print("Connecting to metrics server...")

    active_url = None
    active_port = None

    # Find active metrics server
    for url in METRICS_URLS:
        port = int(url.split(':')[2].split('/')[0])
        metrics = fetch_metrics(url)
        if metrics:
            active_url = url
            active_port = port
            print(f"Connected to {url}")
            break

    if not active_url:
        print("No metrics server found!")
        print("Start training first: python scripts/train_fast_multi.py")
        return

    try:
        while True:
            metrics = fetch_metrics(active_url)
            if metrics:
                display_dashboard(metrics, active_port)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
