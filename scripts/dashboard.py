#!/usr/bin/env python3
"""
Kinetra Training Dashboard Server

A proper web dashboard for monitoring training progress.
Opens in browser automatically.

Usage:
    python scripts/dashboard.py
"""

import sys
import json
import urllib.request
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# Configuration
METRICS_PORT = 8001
DASHBOARD_PORT = 8080


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for dashboard."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_html()
        elif self.path == '/api/metrics':
            self.send_metrics()
        else:
            self.send_error(404)

    def send_html(self):
        """Send the dashboard HTML."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Kinetra Training Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #888; font-size: 1.1em; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .card-title {
            font-size: 0.9em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label { color: #666; }
        .positive { color: #00ff88; }
        .negative { color: #ff4466; }
        .neutral { color: #ffaa00; }
        .bar-container {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 12px;
            margin: 10px 0;
            overflow: hidden;
        }
        .bar {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .bar-energy { background: linear-gradient(90deg, #ff6b6b, #ffd93d); }
        .bar-epsilon { background: linear-gradient(90deg, #6bcb77, #4d96ff); }
        .bar-winrate { background: linear-gradient(90deg, #00d4ff, #00ff88); }
        .feature-list { margin-top: 15px; }
        .feature-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .feature-bar {
            width: 100px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        .feature-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #ff00ff);
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .status-running {
            background: rgba(0,255,136,0.2);
            color: #00ff88;
        }
        .status-stopped {
            background: rgba(255,68,102,0.2);
            color: #ff4466;
        }
        .updated {
            color: #666;
            font-size: 0.8em;
            margin-top: 10px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .physics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .physics-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
        }
        .physics-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .physics-label {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KINETRA</h1>
            <p class="subtitle">Physics-Based RL Trading Dashboard</p>
        </header>

        <div class="grid">
            <!-- Training Status -->
            <div class="card">
                <div class="card-title">Training Status</div>
                <div id="status">
                    <span class="status status-stopped">Connecting...</span>
                </div>
                <div class="metric-value" id="episode">0</div>
                <div class="metric-label">Episode</div>
                <div class="bar-container">
                    <div class="bar bar-epsilon" id="epsilon-bar" style="width: 80%"></div>
                </div>
                <div class="metric-label">Epsilon: <span id="epsilon">0.80</span></div>
            </div>

            <!-- Performance -->
            <div class="card">
                <div class="card-title">Performance</div>
                <div class="metric-value" id="pnl">+0.00%</div>
                <div class="metric-label">Total P&L</div>
                <div style="margin-top: 20px;">
                    <div class="bar-container">
                        <div class="bar bar-winrate" id="winrate-bar" style="width: 50%"></div>
                    </div>
                    <div class="metric-label">Win Rate: <span id="winrate">50.0%</span></div>
                </div>
            </div>

            <!-- Trade Quality -->
            <div class="card">
                <div class="card-title">Trade Quality (MFE/MAE)</div>
                <div class="physics-grid">
                    <div class="physics-item">
                        <div class="physics-value positive" id="mfe">0.00%</div>
                        <div class="physics-label">Avg MFE</div>
                    </div>
                    <div class="physics-item">
                        <div class="physics-value negative" id="mae">0.00%</div>
                        <div class="physics-label">Avg MAE</div>
                    </div>
                    <div class="physics-item">
                        <div class="physics-value neutral" id="ratio">0.00</div>
                        <div class="physics-label">MFE/MAE Ratio</div>
                    </div>
                    <div class="physics-item">
                        <div class="physics-value" id="trades">0</div>
                        <div class="physics-label">Trades</div>
                    </div>
                </div>
            </div>

            <!-- Training Metrics -->
            <div class="card">
                <div class="card-title">Training Metrics</div>
                <div class="physics-grid">
                    <div class="physics-item">
                        <div class="physics-value" id="loss">0.0000</div>
                        <div class="physics-label">Loss</div>
                    </div>
                    <div class="physics-item">
                        <div class="physics-value" id="total-eps">0</div>
                        <div class="physics-label">Total Episodes</div>
                    </div>
                </div>
            </div>

            <!-- Feature Importance -->
            <div class="card" style="grid-column: span 2;">
                <div class="card-title">Feature Importance (RL Learned)</div>
                <div class="feature-list" id="features">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>

        <div class="updated">
            <span class="live-indicator"></span>
            Last updated: <span id="updated">-</span>
        </div>
    </div>

    <script>
        async function fetchMetrics() {
            try {
                const resp = await fetch('/api/metrics');
                const data = await resp.json();
                updateDashboard(data);
            } catch (e) {
                document.getElementById('status').innerHTML =
                    '<span class="status status-stopped">Disconnected</span>';
            }
        }

        function updateDashboard(m) {
            // Status
            document.getElementById('status').innerHTML =
                '<span class="status status-running"><span class="live-indicator"></span>Training</span>';

            // Main metrics
            document.getElementById('episode').textContent = m.episode || 0;
            document.getElementById('epsilon').textContent = (m.epsilon || 0).toFixed(3);
            document.getElementById('epsilon-bar').style.width = ((m.epsilon || 0) * 100) + '%';

            // PnL
            const pnl = m.total_pnl || 0;
            const pnlEl = document.getElementById('pnl');
            pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
            pnlEl.className = 'metric-value ' + (pnl >= 0 ? 'positive' : 'negative');

            // Win rate
            const wr = m.win_rate || 0;
            document.getElementById('winrate').textContent = wr.toFixed(1) + '%';
            document.getElementById('winrate-bar').style.width = wr + '%';

            // Trade quality
            document.getElementById('mfe').textContent = (m.avg_mfe || 0).toFixed(3) + '%';
            document.getElementById('mae').textContent = (m.avg_mae || 0).toFixed(3) + '%';
            document.getElementById('ratio').textContent = (m.mfe_mae_ratio || 0).toFixed(2);
            document.getElementById('trades').textContent = m.total_trades || 0;

            // Training
            document.getElementById('loss').textContent = (m.loss || 0).toFixed(4);
            document.getElementById('total-eps').textContent = m.total_episodes || m.episode || 0;

            // Features
            if (m.features) {
                const featuresEl = document.getElementById('features');
                const sorted = Object.entries(m.features).sort((a, b) => b[1] - a[1]).slice(0, 10);
                const maxVal = sorted.length ? sorted[0][1] : 1;
                featuresEl.innerHTML = sorted.map(([name, val]) => `
                    <div class="feature-item">
                        <span>${name}</span>
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: ${(val/maxVal)*100}%"></div>
                        </div>
                        <span>${(val * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            }

            // Updated time
            document.getElementById('updated').textContent = new Date().toLocaleTimeString();
        }

        // Poll every 2 seconds
        fetchMetrics();
        setInterval(fetchMetrics, 2000);
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def send_metrics(self):
        """Fetch and return metrics as JSON."""
        try:
            url = f'http://localhost:{METRICS_PORT}/metrics'
            with urllib.request.urlopen(url, timeout=2) as resp:
                text = resp.read().decode()

            metrics = {}
            features = {}

            for line in text.split('\\n'):
                if line.startswith('#') or not line.strip():
                    continue

                # Feature importance
                if 'feature_importance' in line:
                    import re
                    match = re.search(r'feature="([^"]+)".*\\s+([\\d.]+)', line)
                    if match:
                        features[match.group(1)] = float(match.group(2))
                # Regular metrics
                elif line.startswith('kinetra_'):
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0].replace('kinetra_rl_', '').replace('kinetra_physics_', '')
                        try:
                            metrics[name] = float(parts[1])
                        except:
                            pass

            metrics['features'] = features

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())


def run_dashboard():
    """Run the dashboard server."""
    print("=" * 60)
    print("KINETRA TRAINING DASHBOARD")
    print("=" * 60)
    print(f"Dashboard: http://localhost:{DASHBOARD_PORT}")
    print(f"Metrics:   http://localhost:{METRICS_PORT}/metrics")
    print("-" * 60)

    server = HTTPServer(('', DASHBOARD_PORT), DashboardHandler)

    # Open browser
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f'http://localhost:{DASHBOARD_PORT}')

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nDashboard stopped.")


if __name__ == '__main__':
    run_dashboard()
