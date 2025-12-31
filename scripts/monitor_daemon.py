#!/usr/bin/env python3
"""
Kinetra Monitoring Daemon
=========================

Background daemon for continuous monitoring.
Can be run as a systemd service.

Usage:
    python scripts/monitor_daemon.py
    
Or install as systemd service:
    sudo cp scripts/kinetra-monitor.service /etc/systemd/system/
    sudo systemctl enable kinetra-monitor
    sudo systemctl start kinetra-monitor
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add workspace to path
workspace = Path(__file__).parent.parent
sys.path.insert(0, str(workspace))

from kinetra.devops.monitor import FolderMonitor, MonitorEvent, EventType

# Configuration
CONFIG = {
    "watch_paths": [
        str(workspace / "kinetra"),
        str(workspace / "scripts"),
        str(workspace / "data"),
    ],
    "patterns": ["*.py", "*.csv", "*.log"],
    "ignore_patterns": ["__pycache__", ".git", "*.pyc", ".venv"],
    "interval": 5.0,
    "log_file": str(workspace / "logs" / "monitor.log"),
    "event_log": str(workspace / "logs" / "events.jsonl"),
    "alert_thresholds": {
        "cpu": 90,
        "memory": 85,
        "disk": 90
    }
}


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path(CONFIG["log_file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_file"]),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("monitor_daemon")


def log_event_to_file(event: MonitorEvent, filepath: str):
    """Log event to JSONL file."""
    event_data = {
        "timestamp": event.timestamp.isoformat(),
        "type": event.event_type.value,
        "severity": event.severity,
        "path": event.path,
        "message": event.message,
        "metadata": event.metadata
    }
    
    with open(filepath, 'a') as f:
        f.write(json.dumps(event_data) + '\n')


class MonitorDaemon:
    """Monitoring daemon class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.monitor = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.monitor:
            self.monitor.stop()
    
    def _on_event(self, event: MonitorEvent):
        """Handle monitoring event."""
        # Log to file
        self.logger.info(f"[{event.event_type.value}] {event.message}")
        
        # Log to event file
        log_event_to_file(event, self.config["event_log"])
        
        # Handle critical events
        if event.severity in ("error", "critical"):
            self._handle_critical_event(event)
    
    def _handle_critical_event(self, event: MonitorEvent):
        """Handle critical events (could send alerts, etc.)."""
        self.logger.error(f"CRITICAL: {event.message}")
        
        # Here you could add:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts
        # - Auto-restart services
    
    def run(self):
        """Run the monitoring daemon."""
        self.logger.info("Starting Kinetra Monitoring Daemon")
        self.logger.info(f"Watching: {self.config['watch_paths']}")
        
        self.monitor = FolderMonitor(
            watch_paths=self.config["watch_paths"],
            patterns=self.config["patterns"],
            ignore_patterns=self.config["ignore_patterns"]
        )
        
        # Set thresholds
        self.monitor.performance.cpu_threshold = self.config["alert_thresholds"]["cpu"]
        self.monitor.performance.memory_threshold = self.config["alert_thresholds"]["memory"]
        self.monitor.performance.disk_threshold = self.config["alert_thresholds"]["disk"]
        
        # Set event handler
        self.monitor.on_event = self._on_event
        
        # Start monitoring
        self.monitor.start(interval=self.config["interval"])
        self.running = True
        
        # Main loop
        status_interval = 60  # Log status every minute
        last_status = time.time()
        
        while self.running:
            try:
                time.sleep(1)
                
                # Periodic status log
                if time.time() - last_status > status_interval:
                    status = self.monitor.get_status()
                    perf = status.get("performance", {})
                    cpu = perf.get("cpu", {}).get("current", 0)
                    mem = perf.get("memory", {}).get("current", 0)
                    
                    self.logger.info(
                        f"Status: Files={status['files_tracked']}, "
                        f"Events={status['events_recorded']}, "
                        f"CPU={cpu:.1f}%, Memory={mem:.1f}%"
                    )
                    last_status = time.time()
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
        
        self.logger.info("Monitoring daemon stopped")


def main():
    """Main entry point."""
    # Load config from file if exists
    config_file = workspace / "monitor_config.json"
    if config_file.exists():
        with open(config_file) as f:
            user_config = json.load(f)
            CONFIG.update(user_config)
    
    # Create and run daemon
    daemon = MonitorDaemon(CONFIG)
    daemon.run()


if __name__ == "__main__":
    main()
