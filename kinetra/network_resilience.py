"""
Network Resilience Module
=========================

Implements robust network connectivity for trading systems:
- Auto-reconnect with exponential backoff
- Lowest latency server selection
- Dual/redundant connections
- Packet loss detection and handling
- Throughput optimization
- Connection health monitoring

Architecture:
    ConnectionManager
        ├── ServerPool (multiple endpoints)
        ├── LatencyMonitor (continuous testing)
        ├── PacketTracker (loss detection)
        ├── ReconnectHandler (auto-recovery)
        └── RedundantConnection (failover)
"""

import asyncio
import heapq
import logging
import random
import socket
import statistics
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state machine."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    DEGRADED = auto()  # Connected but with issues
    FAILED = auto()


class ServerType(Enum):
    """Server type classification."""
    PRIMARY = auto()
    SECONDARY = auto()
    BACKUP = auto()


@dataclass
class ServerEndpoint:
    """Server endpoint configuration."""
    host: str
    port: int
    name: str = ""
    server_type: ServerType = ServerType.PRIMARY
    region: str = ""
    weight: float = 1.0  # Load balancing weight
    max_connections: int = 10
    timeout_seconds: float = 30.0
    
    # Runtime metrics
    latency_ms: float = float('inf')
    packet_loss_rate: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    
    @property
    def health_score(self) -> float:
        """Calculate server health score (0-1)."""
        if self.latency_ms == float('inf'):
            return 0.0
        
        # Latency component (0-0.4)
        latency_score = max(0, 0.4 - (self.latency_ms / 1000))
        
        # Packet loss component (0-0.3)
        loss_score = 0.3 * (1 - self.packet_loss_rate)
        
        # Success rate component (0-0.3)
        total = self.success_count + self.failure_count
        if total > 0:
            success_score = 0.3 * (self.success_count / total)
        else:
            success_score = 0.15
        
        return latency_score + loss_score + success_score
    
    def __lt__(self, other):
        """Compare by latency for heap operations."""
        return self.latency_ms < other.latency_ms


@dataclass
class PacketInfo:
    """Packet tracking information."""
    sequence: int
    timestamp: float
    size: int
    acknowledged: bool = False
    ack_time: Optional[float] = None
    retries: int = 0


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    latency_samples: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    reconnect_count: int = 0
    last_latency_ms: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return statistics.mean(self.latency_samples[-100:])
    
    @property
    def latency_std_ms(self) -> float:
        if len(self.latency_samples) < 2:
            return 0.0
        return statistics.stdev(self.latency_samples[-100:])
    
    @property
    def packet_loss_rate(self) -> float:
        total = self.packets_sent
        if total == 0:
            return 0.0
        return self.packets_lost / total
    
    @property
    def avg_throughput_bps(self) -> float:
        if not self.throughput_samples:
            return 0.0
        return statistics.mean(self.throughput_samples[-100:])


class LatencyMonitor:
    """
    Continuous latency monitoring with statistical analysis.
    
    Features:
    - Periodic ping measurements
    - Latency distribution analysis
    - Jitter detection
    - Anomaly detection
    """
    
    def __init__(
        self,
        ping_interval: float = 1.0,
        sample_size: int = 100,
        anomaly_threshold: float = 3.0,  # Standard deviations
    ):
        """
        Initialize latency monitor.
        
        Args:
            ping_interval: Seconds between pings
            sample_size: Rolling window size
            anomaly_threshold: Std devs for anomaly detection
        """
        self.ping_interval = ping_interval
        self.sample_size = sample_size
        self.anomaly_threshold = anomaly_threshold
        
        self._samples: deque = deque(maxlen=sample_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._anomaly_count = 0
        self._last_anomaly: Optional[datetime] = None
    
    def add_sample(self, latency_ms: float) -> bool:
        """
        Add latency sample and check for anomaly.
        
        Returns:
            True if anomaly detected
        """
        with self._lock:
            is_anomaly = False
            
            if len(self._samples) >= 10:
                mean = statistics.mean(self._samples)
                std = statistics.stdev(self._samples)
                
                if std > 0 and abs(latency_ms - mean) > self.anomaly_threshold * std:
                    is_anomaly = True
                    self._anomaly_count += 1
                    self._last_anomaly = datetime.now()
                    logger.warning(f"Latency anomaly detected: {latency_ms:.1f}ms (mean: {mean:.1f}ms)")
            
            self._samples.append(latency_ms)
            return is_anomaly
    
    def get_statistics(self) -> Dict:
        """Get latency statistics."""
        with self._lock:
            if not self._samples:
                return {'samples': 0}
            
            samples = list(self._samples)
            return {
                'samples': len(samples),
                'min_ms': min(samples),
                'max_ms': max(samples),
                'mean_ms': statistics.mean(samples),
                'median_ms': statistics.median(samples),
                'std_ms': statistics.stdev(samples) if len(samples) > 1 else 0,
                'p95_ms': np.percentile(samples, 95),
                'p99_ms': np.percentile(samples, 99),
                'jitter_ms': self._calculate_jitter(samples),
                'anomaly_count': self._anomaly_count,
            }
    
    def _calculate_jitter(self, samples: List[float]) -> float:
        """Calculate jitter (variation in latency)."""
        if len(samples) < 2:
            return 0.0
        
        diffs = [abs(samples[i] - samples[i-1]) for i in range(1, len(samples))]
        return statistics.mean(diffs)
    
    def is_stable(self, max_jitter_ms: float = 50.0, max_loss_rate: float = 0.01) -> bool:
        """Check if connection is stable."""
        stats = self.get_statistics()
        if stats['samples'] < 10:
            return True  # Not enough data
        
        return stats['jitter_ms'] <= max_jitter_ms


class PacketTracker:
    """
    Track packets for loss detection and retransmission.
    
    Features:
    - Sequence number tracking
    - Acknowledgment handling
    - Loss detection
    - Retransmission triggering
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        ack_timeout: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize packet tracker.
        
        Args:
            window_size: Tracking window size
            ack_timeout: Acknowledgment timeout in seconds
            max_retries: Maximum retransmission attempts
        """
        self.window_size = window_size
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries
        
        self._packets: Dict[int, PacketInfo] = {}
        self._sequence = 0
        self._lock = threading.RLock()
        
        # Statistics
        self.packets_sent = 0
        self.packets_acked = 0
        self.packets_lost = 0
        self.packets_retried = 0
    
    def send_packet(self, size: int) -> int:
        """
        Register sent packet.
        
        Args:
            size: Packet size in bytes
            
        Returns:
            Sequence number
        """
        with self._lock:
            seq = self._sequence
            self._sequence += 1
            
            self._packets[seq] = PacketInfo(
                sequence=seq,
                timestamp=time.time(),
                size=size,
            )
            self.packets_sent += 1
            
            # Clean old packets
            self._cleanup()
            
            return seq
    
    def acknowledge_packet(self, sequence: int) -> Optional[float]:
        """
        Acknowledge received packet.
        
        Args:
            sequence: Packet sequence number
            
        Returns:
            Round-trip time in ms, or None if not found
        """
        with self._lock:
            if sequence not in self._packets:
                return None
            
            packet = self._packets[sequence]
            if packet.acknowledged:
                return None
            
            packet.acknowledged = True
            packet.ack_time = time.time()
            self.packets_acked += 1
            
            rtt = (packet.ack_time - packet.timestamp) * 1000
            return rtt
    
    def get_lost_packets(self) -> List[int]:
        """Get list of lost packet sequences."""
        with self._lock:
            now = time.time()
            lost = []
            
            for seq, packet in self._packets.items():
                if not packet.acknowledged:
                    if now - packet.timestamp > self.ack_timeout:
                        if packet.retries < self.max_retries:
                            lost.append(seq)
                        else:
                            self.packets_lost += 1
            
            return lost
    
    def mark_retry(self, sequence: int) -> bool:
        """Mark packet for retry."""
        with self._lock:
            if sequence not in self._packets:
                return False
            
            packet = self._packets[sequence]
            if packet.retries >= self.max_retries:
                return False
            
            packet.retries += 1
            packet.timestamp = time.time()  # Reset timeout
            self.packets_retried += 1
            return True
    
    def _cleanup(self) -> None:
        """Remove old acknowledged packets."""
        if len(self._packets) > self.window_size:
            # Remove oldest acknowledged packets
            to_remove = []
            for seq, packet in self._packets.items():
                if packet.acknowledged and len(self._packets) - len(to_remove) > self.window_size // 2:
                    to_remove.append(seq)
            
            for seq in to_remove:
                del self._packets[seq]
    
    @property
    def loss_rate(self) -> float:
        """Current packet loss rate."""
        if self.packets_sent == 0:
            return 0.0
        return self.packets_lost / self.packets_sent


class ReconnectHandler:
    """
    Automatic reconnection with exponential backoff.
    
    Features:
    - Exponential backoff with jitter
    - Maximum retry limits
    - Cooldown periods
    - Event callbacks
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        max_retries: int = 10,
    ):
        """
        Initialize reconnect handler.
        
        Args:
            initial_delay: Initial retry delay in seconds
            max_delay: Maximum retry delay
            backoff_factor: Delay multiplier per retry
            jitter: Random jitter factor (0-1)
            max_retries: Maximum retry attempts (0 = infinite)
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_retries = max_retries
        
        self._current_delay = initial_delay
        self._retry_count = 0
        self._last_attempt: Optional[datetime] = None
        self._callbacks: List[Callable[[int, float], None]] = []
    
    def on_reconnect(self, callback: Callable[[int, float], None]) -> None:
        """Register reconnect callback(retry_count, delay)."""
        self._callbacks.append(callback)
    
    def get_next_delay(self) -> float:
        """Get next retry delay with jitter."""
        delay = min(self._current_delay, self.max_delay)
        
        # Add jitter
        jitter_amount = delay * self.jitter * random.uniform(-1, 1)
        delay = max(0.1, delay + jitter_amount)
        
        return delay
    
    def should_retry(self) -> bool:
        """Check if should attempt retry."""
        if self.max_retries > 0 and self._retry_count >= self.max_retries:
            return False
        return True
    
    def record_attempt(self, success: bool) -> None:
        """Record connection attempt result."""
        self._last_attempt = datetime.now()
        
        if success:
            self.reset()
        else:
            self._retry_count += 1
            self._current_delay = min(
                self._current_delay * self.backoff_factor,
                self.max_delay
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self._retry_count, self._current_delay)
                except Exception as e:
                    logger.error(f"Reconnect callback error: {e}")
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._current_delay = self.initial_delay
        self._retry_count = 0
    
    @property
    def retry_count(self) -> int:
        return self._retry_count


class RedundantConnection:
    """
    Manage redundant connections for failover.
    
    Features:
    - Primary/secondary connection management
    - Automatic failover
    - Connection health comparison
    - Seamless switchover
    """
    
    def __init__(
        self,
        failover_threshold: float = 0.5,  # Health score threshold
        switchback_delay: float = 30.0,  # Seconds before switching back
    ):
        """
        Initialize redundant connection manager.
        
        Args:
            failover_threshold: Health score below which to failover
            switchback_delay: Delay before switching back to primary
        """
        self.failover_threshold = failover_threshold
        self.switchback_delay = switchback_delay
        
        self._connections: Dict[str, 'Connection'] = {}
        self._active_id: Optional[str] = None
        self._primary_id: Optional[str] = None
        self._failover_time: Optional[datetime] = None
        self._lock = threading.RLock()
    
    def add_connection(self, conn_id: str, connection: 'Connection', is_primary: bool = False) -> None:
        """Add connection to pool."""
        with self._lock:
            self._connections[conn_id] = connection
            
            if is_primary or self._primary_id is None:
                self._primary_id = conn_id
            
            if self._active_id is None:
                self._active_id = conn_id
    
    def get_active(self) -> Optional['Connection']:
        """Get active connection."""
        with self._lock:
            if self._active_id:
                return self._connections.get(self._active_id)
            return None
    
    def check_failover(self) -> bool:
        """
        Check if failover is needed.
        
        Returns:
            True if failover occurred
        """
        with self._lock:
            if not self._active_id or len(self._connections) < 2:
                return False
            
            active = self._connections.get(self._active_id)
            if not active:
                return False
            
            # Check if active connection is degraded
            if active.health_score < self.failover_threshold:
                # Find best alternative
                best_id = None
                best_score = 0.0
                
                for conn_id, conn in self._connections.items():
                    if conn_id != self._active_id and conn.health_score > best_score:
                        best_id = conn_id
                        best_score = conn.health_score
                
                if best_id and best_score > self.failover_threshold:
                    logger.warning(f"Failover from {self._active_id} to {best_id}")
                    self._active_id = best_id
                    self._failover_time = datetime.now()
                    return True
            
            # Check if should switch back to primary
            if (self._active_id != self._primary_id and 
                self._failover_time and
                (datetime.now() - self._failover_time).total_seconds() > self.switchback_delay):
                
                primary = self._connections.get(self._primary_id)
                if primary and primary.health_score > self.failover_threshold:
                    logger.info(f"Switching back to primary {self._primary_id}")
                    self._active_id = self._primary_id
                    self._failover_time = None
                    return True
            
            return False


class Connection(ABC):
    """Abstract base connection class."""
    
    @property
    @abstractmethod
    def health_score(self) -> float:
        """Get connection health score (0-1)."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from server."""
        pass
    
    @abstractmethod
    async def send(self, data: bytes) -> bool:
        """Send data."""
        pass
    
    @abstractmethod
    async def receive(self, timeout: float = None) -> Optional[bytes]:
        """Receive data."""
        pass


class ServerPool:
    """
    Pool of server endpoints with automatic selection.
    
    Features:
    - Latency-based server selection
    - Health monitoring
    - Load balancing
    - Automatic failover
    """
    
    def __init__(
        self,
        latency_test_interval: float = 10.0,
        health_check_interval: float = 5.0,
    ):
        """
        Initialize server pool.
        
        Args:
            latency_test_interval: Seconds between latency tests
            health_check_interval: Seconds between health checks
        """
        self.latency_test_interval = latency_test_interval
        self.health_check_interval = health_check_interval
        
        self._servers: List[ServerEndpoint] = []
        self._lock = threading.RLock()
        self._running = False
        self._test_thread: Optional[threading.Thread] = None
    
    def add_server(self, server: ServerEndpoint) -> None:
        """Add server to pool."""
        with self._lock:
            self._servers.append(server)
            logger.info(f"Added server {server.name or server.host}:{server.port}")
    
    def get_best_server(self) -> Optional[ServerEndpoint]:
        """Get server with lowest latency."""
        with self._lock:
            if not self._servers:
                return None
            
            # Filter healthy servers
            healthy = [s for s in self._servers if s.health_score > 0.3]
            
            if not healthy:
                healthy = self._servers  # Fallback to all servers
            
            # Return lowest latency
            return min(healthy, key=lambda s: s.latency_ms)
    
    def get_servers_by_latency(self, limit: int = None) -> List[ServerEndpoint]:
        """Get servers sorted by latency."""
        with self._lock:
            sorted_servers = sorted(self._servers, key=lambda s: s.latency_ms)
            if limit:
                return sorted_servers[:limit]
            return sorted_servers
    
    async def test_latency(self, server: ServerEndpoint) -> float:
        """
        Test latency to server.
        
        Returns:
            Latency in milliseconds
        """
        try:
            start = time.time()
            
            # TCP connection test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(server.host, server.port),
                timeout=server.timeout_seconds
            )
            
            latency = (time.time() - start) * 1000
            
            writer.close()
            await writer.wait_closed()
            
            # Update server metrics
            server.latency_ms = latency
            server.success_count += 1
            server.last_success = datetime.now()
            
            return latency
            
        except Exception as e:
            logger.warning(f"Latency test failed for {server.host}: {e}")
            server.failure_count += 1
            server.last_failure = datetime.now()
            return float('inf')
    
    async def test_all_servers(self) -> Dict[str, float]:
        """Test latency to all servers concurrently."""
        tasks = []
        for server in self._servers:
            tasks.append(self.test_latency(server))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            server.host: result if isinstance(result, float) else float('inf')
            for server, result in zip(self._servers, results)
        }
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._test_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._test_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._test_thread:
            self._test_thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self._running:
            try:
                loop.run_until_complete(self.test_all_servers())
            except Exception as e:
                logger.error(f"Server monitoring error: {e}")
            
            time.sleep(self.latency_test_interval)
        
        loop.close()
    
    def get_statistics(self) -> Dict:
        """Get pool statistics."""
        with self._lock:
            return {
                'total_servers': len(self._servers),
                'healthy_servers': sum(1 for s in self._servers if s.health_score > 0.5),
                'best_latency_ms': min((s.latency_ms for s in self._servers), default=float('inf')),
                'avg_latency_ms': statistics.mean([s.latency_ms for s in self._servers if s.latency_ms < float('inf')]) if self._servers else 0,
                'servers': [
                    {
                        'name': s.name or f"{s.host}:{s.port}",
                        'latency_ms': s.latency_ms,
                        'health_score': s.health_score,
                        'packet_loss': s.packet_loss_rate,
                    }
                    for s in self._servers
                ]
            }


class ConnectionManager:
    """
    Complete connection management system.
    
    Integrates:
    - Server pool with latency-based selection
    - Automatic reconnection
    - Redundant connections
    - Packet tracking
    - Performance monitoring
    """
    
    def __init__(
        self,
        enable_redundancy: bool = True,
        enable_latency_routing: bool = True,
    ):
        """
        Initialize connection manager.
        
        Args:
            enable_redundancy: Enable redundant connections
            enable_latency_routing: Enable latency-based routing
        """
        self.enable_redundancy = enable_redundancy
        self.enable_latency_routing = enable_latency_routing
        
        # Components
        self._server_pool = ServerPool()
        self._latency_monitor = LatencyMonitor()
        self._packet_tracker = PacketTracker()
        self._reconnect_handler = ReconnectHandler()
        self._redundant = RedundantConnection() if enable_redundancy else None
        
        # State
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_state_change: List[Callable[[ConnectionState], None]] = []
        self._on_data: List[Callable[[bytes], None]] = []
    
    def add_server(
        self,
        host: str,
        port: int,
        name: str = "",
        server_type: ServerType = ServerType.PRIMARY,
        region: str = "",
    ) -> None:
        """Add server endpoint."""
        server = ServerEndpoint(
            host=host,
            port=port,
            name=name or f"{host}:{port}",
            server_type=server_type,
            region=region,
        )
        self._server_pool.add_server(server)
    
    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)
    
    def on_data(self, callback: Callable[[bytes], None]) -> None:
        """Register data callback."""
        self._on_data.append(callback)
    
    async def connect(self) -> bool:
        """Connect to best available server."""
        self._set_state(ConnectionState.CONNECTING)
        
        # Test all servers
        await self._server_pool.test_all_servers()
        
        # Get best server
        server = self._server_pool.get_best_server()
        if not server:
            logger.error("No servers available")
            self._set_state(ConnectionState.FAILED)
            return False
        
        try:
            logger.info(f"Connecting to {server.name} (latency: {server.latency_ms:.1f}ms)")
            
            # TODO: Implement actual connection logic based on protocol
            # This is a placeholder for the connection implementation
            
            self._set_state(ConnectionState.CONNECTED)
            self._reconnect_handler.reset()
            
            # Start monitoring
            self._server_pool.start_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._reconnect_handler.record_attempt(False)
            self._set_state(ConnectionState.FAILED)
            return False
    
    async def reconnect(self) -> bool:
        """Attempt reconnection with backoff."""
        if not self._reconnect_handler.should_retry():
            logger.error("Max reconnection attempts reached")
            return False
        
        delay = self._reconnect_handler.get_next_delay()
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_handler.retry_count + 1})")
        
        self._set_state(ConnectionState.RECONNECTING)
        await asyncio.sleep(delay)
        
        success = await self.connect()
        self._reconnect_handler.record_attempt(success)
        
        return success
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._server_pool.stop_monitoring()
        self._set_state(ConnectionState.DISCONNECTED)
    
    async def send(self, data: bytes) -> bool:
        """Send data with tracking."""
        if self._state != ConnectionState.CONNECTED:
            return False
        
        seq = self._packet_tracker.send_packet(len(data))
        self._metrics.packets_sent += 1
        self._metrics.bytes_sent += len(data)
        
        # TODO: Implement actual send logic
        
        return True
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency sample."""
        self._latency_monitor.add_sample(latency_ms)
        self._metrics.latency_samples.append(latency_ms)
        self._metrics.last_latency_ms = latency_ms
    
    def _set_state(self, state: ConnectionState) -> None:
        """Update connection state."""
        with self._lock:
            if state != self._state:
                old_state = self._state
                self._state = state
                logger.info(f"Connection state: {old_state.name} -> {state.name}")
                
                for callback in self._on_state_change:
                    try:
                        callback(state)
                    except Exception as e:
                        logger.error(f"State change callback error: {e}")
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    def get_statistics(self) -> Dict:
        """Get connection statistics."""
        return {
            'state': self._state.name,
            'latency': self._latency_monitor.get_statistics(),
            'packet_loss_rate': self._packet_tracker.loss_rate,
            'metrics': {
                'avg_latency_ms': self._metrics.avg_latency_ms,
                'latency_std_ms': self._metrics.latency_std_ms,
                'packets_sent': self._metrics.packets_sent,
                'packets_received': self._metrics.packets_received,
                'bytes_sent': self._metrics.bytes_sent,
                'bytes_received': self._metrics.bytes_received,
                'reconnect_count': self._metrics.reconnect_count,
            },
            'servers': self._server_pool.get_statistics(),
        }


class StressTest:
    """
    Network stress testing utility.
    
    Features:
    - Throughput testing
    - Latency under load
    - Packet loss simulation
    - Connection stability
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize stress test.
        
        Args:
            connection_manager: Connection manager to test
        """
        self._conn_manager = connection_manager
        self._results: List[Dict] = []
    
    async def run_throughput_test(
        self,
        duration_seconds: float = 10.0,
        message_size: int = 1024,
        messages_per_second: int = 100,
    ) -> Dict:
        """
        Run throughput test.
        
        Args:
            duration_seconds: Test duration
            message_size: Message size in bytes
            messages_per_second: Target message rate
            
        Returns:
            Test results
        """
        logger.info(f"Starting throughput test: {duration_seconds}s, {message_size}B, {messages_per_second}/s")
        
        start = time.time()
        messages_sent = 0
        bytes_sent = 0
        errors = 0
        latencies = []
        
        interval = 1.0 / messages_per_second
        
        while time.time() - start < duration_seconds:
            msg_start = time.time()
            
            data = bytes([random.randint(0, 255) for _ in range(message_size)])
            success = await self._conn_manager.send(data)
            
            if success:
                messages_sent += 1
                bytes_sent += message_size
                latencies.append((time.time() - msg_start) * 1000)
            else:
                errors += 1
            
            # Rate limiting
            elapsed = time.time() - msg_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
        
        actual_duration = time.time() - start
        
        result = {
            'duration_seconds': actual_duration,
            'messages_sent': messages_sent,
            'bytes_sent': bytes_sent,
            'errors': errors,
            'throughput_msgs_per_sec': messages_sent / actual_duration,
            'throughput_bytes_per_sec': bytes_sent / actual_duration,
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'error_rate': errors / (messages_sent + errors) if (messages_sent + errors) > 0 else 0,
        }
        
        self._results.append(result)
        logger.info(f"Throughput test complete: {result['throughput_msgs_per_sec']:.1f} msg/s")
        
        return result
    
    async def run_latency_test(
        self,
        iterations: int = 100,
        message_size: int = 64,
    ) -> Dict:
        """
        Run latency test.
        
        Args:
            iterations: Number of test iterations
            message_size: Message size in bytes
            
        Returns:
            Test results
        """
        logger.info(f"Starting latency test: {iterations} iterations")
        
        latencies = []
        errors = 0
        
        for _ in range(iterations):
            start = time.time()
            
            data = bytes([random.randint(0, 255) for _ in range(message_size)])
            success = await self._conn_manager.send(data)
            
            if success:
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            else:
                errors += 1
            
            await asyncio.sleep(0.01)  # Small delay between tests
        
        if not latencies:
            return {'error': 'No successful measurements'}
        
        result = {
            'iterations': iterations,
            'successful': len(latencies),
            'errors': errors,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }
        
        self._results.append(result)
        logger.info(f"Latency test complete: avg={result['avg_latency_ms']:.2f}ms, p99={result['p99_latency_ms']:.2f}ms")
        
        return result
    
    def get_all_results(self) -> List[Dict]:
        """Get all test results."""
        return self._results


# Export all components
__all__ = [
    'ConnectionState',
    'ServerType',
    'ServerEndpoint',
    'PacketInfo',
    'ConnectionMetrics',
    'LatencyMonitor',
    'PacketTracker',
    'ReconnectHandler',
    'RedundantConnection',
    'Connection',
    'ServerPool',
    'ConnectionManager',
    'StressTest',
]
