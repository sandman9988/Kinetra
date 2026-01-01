#!/usr/bin/env python3
"""
Live Testing Script for Kinetra
================================

Implements live testing workflow with safety gates:
- Virtual (paper) trading mode
- Demo account testing
- Live connection testing
- Real-time monitoring with CHS tracking

Safety Philosophy:
- NEVER deploy to live without demo validation
- Circuit breakers halt on CHS < 0.55
- All trades validated by OrderValidator
- Real-time risk monitoring

Usage:
    # Virtual/paper trading (no real connection)
    python scripts/testing/run_live_test.py --mode virtual --agent ppo
    
    # Demo account testing
    python scripts/testing/run_live_test.py --mode demo --agent ppo
    
    # Connection test only
    python scripts/testing/run_live_test.py --test-connection
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from kinetra.order_executor import create_executor, OrderExecutor, OrderResult
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.mt5_connector import MT5Connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTestRunner:
    """
    Manages live testing with safety gates and monitoring.
    """
    
    def __init__(
        self,
        mode: str,
        symbol: str = "EURUSD",
        agent_type: str = "ppo",
        max_trades: int = 10,
        chs_threshold: float = 0.55
    ):
        """
        Initialize live test runner.
        
        Args:
            mode: 'virtual', 'demo', or 'live'
            symbol: Trading symbol
            agent_type: Agent type (ppo, dqn, linear, berserker, triad)
            max_trades: Maximum number of trades before auto-stop
            chs_threshold: CHS circuit breaker threshold
        """
        self.mode = mode
        self.symbol = symbol
        self.agent_type = agent_type
        self.max_trades = max_trades
        self.chs_threshold = chs_threshold
        
        self.executor: Optional[OrderExecutor] = None
        self.spec: Optional[SymbolSpec] = None
        self.trades_executed = 0
        self.chs_history = []
        self.is_running = False
        
        # Safety gates
        self.circuit_breaker_active = False
        
    def setup_connection(self) -> bool:
        """
        Set up connection based on mode.
        
        Returns:
            True if connection successful
        """
        logger.info(f"Setting up {self.mode} mode connection...")
        
        if self.mode == 'virtual':
            # Virtual mode: Generate synthetic data
            logger.info("Virtual mode: Using synthetic data stream")
            return self._setup_virtual_mode()
        
        elif self.mode in ['demo', 'live']:
            # Real MT5 connection required
            logger.info(f"{self.mode.upper()} mode: Connecting to MT5...")
            return self._setup_mt5_connection()
        
        else:
            logger.error(f"Invalid mode: {self.mode}")
            return False
    
    def _setup_virtual_mode(self) -> bool:
        """Set up virtual (paper) trading mode."""
        # Create synthetic data stream
        # In production, this would be replaced with historical replay
        
        self.spec = SymbolSpec(
            symbol=self.symbol,
            asset_class=AssetClass.FOREX,
            digits=5,
            trade_stops_level=15,
            trade_freeze_level=10,
        )
        
        # Generate synthetic data for testing
        data = self._generate_synthetic_data()
        
        self.executor = create_executor(
            spec=self.spec,
            mode='backtest',  # Use backtest executor for virtual mode
            data=data
        )
        
        logger.info("✅ Virtual mode initialized")
        return True
    
    def _setup_mt5_connection(self) -> bool:
        """Set up real MT5 connection."""
        try:
            # Check if MetaTrader5 is available
            try:
                import MetaTrader5 as mt5
            except ImportError:
                logger.error("MetaTrader5 not installed. Install with: pip install MetaTrader5")
                return False
            
            # Initialize MT5 connector
            connector = MT5Connector()
            
            if not connector.connect():
                logger.error("Failed to connect to MT5 terminal")
                logger.info("Make sure MT5 is running and credentials are configured in .env")
                return False
            
            # Get symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found in MT5")
                connector.disconnect()
                return False
            
            # Create spec from MT5 symbol info
            self.spec = SymbolSpec(
                symbol=self.symbol,
                asset_class=self._get_asset_class(self.symbol),
                digits=symbol_info.digits,
                trade_stops_level=symbol_info.trade_stops_level,
                trade_freeze_level=symbol_info.trade_freeze_level,
            )
            
            # Create live executor
            self.executor = create_executor(
                spec=self.spec,
                mode='live',
                mt5_connection=connector
            )
            
            logger.info(f"✅ Connected to MT5 - {self.symbol}")
            logger.info(f"   Mode: {self.mode.upper()}")
            logger.info(f"   Stops Level: {self.spec.trade_stops_level} points")
            logger.info(f"   Freeze Level: {self.spec.trade_freeze_level} points")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up MT5 connection: {e}")
            return False
    
    def _generate_synthetic_data(self, n_bars: int = 1000) -> pd.DataFrame:
        """Generate synthetic market data for virtual mode."""
        # Simple random walk with realistic spreads
        np.random.seed(42)
        
        base_price = 1.08500
        prices = base_price + np.random.randn(n_bars).cumsum() * 0.0001
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(n_bars) * 0.00005,
            'high': prices + np.abs(np.random.randn(n_bars)) * 0.0001,
            'low': prices - np.abs(np.random.randn(n_bars)) * 0.0001,
            'close': prices,
            'spread': np.random.randint(10, 20, n_bars),  # 1-2 pip spread
        }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1min'))
        
        return data
    
    def _get_asset_class(self, symbol: str) -> AssetClass:
        """Determine asset class from symbol."""
        if any(x in symbol for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD']):
            return AssetClass.FOREX
        elif any(x in symbol for x in ['BTC', 'ETH', 'XRP', 'LTC']):
            return AssetClass.CRYPTO
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return AssetClass.METALS
        elif any(x in symbol for x in ['US30', 'SPX', 'NAS', 'DAX']):
            return AssetClass.INDICES
        else:
            return AssetClass.COMMODITIES
    
    def calculate_chs(self) -> float:
        """
        Calculate Composite Health Score.
        
        Simplified version for live testing.
        In production, this would use full HealthMonitor.
        """
        # Placeholder implementation
        # Real CHS would aggregate: Omega, Z-Factor, Energy, MFE, RoR
        
        if len(self.chs_history) < 5:
            # Not enough data, return neutral
            return 0.75
        
        # Simple moving average of recent performance
        recent_chs = np.mean(self.chs_history[-10:])
        
        # Add random variation for demo
        noise = np.random.randn() * 0.05
        chs = np.clip(recent_chs + noise, 0.0, 1.0)
        
        return chs
    
    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should activate.
        
        Returns:
            True if trading should halt
        """
        chs = self.calculate_chs()
        self.chs_history.append(chs)
        
        if chs < self.chs_threshold:
            if not self.circuit_breaker_active:
                logger.warning(f"⚠️ CIRCUIT BREAKER ACTIVATED - CHS {chs:.3f} < {self.chs_threshold}")
                self.circuit_breaker_active = True
            return True
        
        if self.circuit_breaker_active and chs > (self.chs_threshold + 0.1):
            logger.info(f"✅ Circuit breaker reset - CHS {chs:.3f}")
            self.circuit_breaker_active = False
        
        return self.circuit_breaker_active
    
    def run_test(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """
        Run live test for specified duration.
        
        Args:
            duration_minutes: Test duration in minutes
            
        Returns:
            Test results summary
        """
        logger.info("="*70)
        logger.info(f"STARTING LIVE TEST - {self.mode.upper()} MODE")
        logger.info("="*70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Agent: {self.agent_type.upper()}")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Max Trades: {self.max_trades}")
        logger.info(f"CHS Threshold: {self.chs_threshold}")
        logger.info("")
        
        # Setup connection
        if not self.setup_connection():
            logger.error("❌ Connection setup failed")
            return {'success': False, 'error': 'Connection failed'}
        
        # Initialize tracking
        start_time = datetime.now()
        self.is_running = True
        results = {
            'mode': self.mode,
            'symbol': self.symbol,
            'agent_type': self.agent_type,
            'start_time': start_time.isoformat(),
            'trades': [],
            'chs_history': [],
            'circuit_breaker_events': 0,
        }
        
        logger.info("🚀 Test started - monitoring for circuit breakers...")
        logger.info("")
        
        # Main test loop
        iteration = 0
        while self.is_running:
            iteration += 1
            
            # Check circuit breaker
            if self.check_circuit_breaker():
                results['circuit_breaker_events'] += 1
                logger.warning(f"Circuit breaker active - skipping trade {iteration}")
                
                # In real implementation, would pause and wait for recovery
                if iteration % 10 == 0:
                    logger.info("Monitoring CHS... waiting for recovery")
            else:
                # Execute trade (simplified demo logic)
                if np.random.rand() > 0.7:  # Random entry
                    trade_result = self._execute_demo_trade()
                    results['trades'].append(trade_result)
                    
                    self.trades_executed += 1
                    
                    if self.trades_executed >= self.max_trades:
                        logger.info(f"✅ Max trades ({self.max_trades}) reached - stopping")
                        break
            
            # Record CHS
            chs = self.chs_history[-1] if self.chs_history else 0.75
            results['chs_history'].append(chs)
            
            # Check duration
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            if elapsed >= duration_minutes:
                logger.info(f"✅ Test duration ({duration_minutes} min) reached - stopping")
                break
            
            # In virtual mode, step forward
            if self.mode == 'virtual':
                if hasattr(self.executor, 'step_forward'):
                    self.executor.step_forward()
            
            # Wait before next iteration (real-time simulation)
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration} - Trades: {self.trades_executed}, CHS: {chs:.3f}")
        
        # Finalize results
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        results['trades_executed'] = self.trades_executed
        results['success'] = True
        
        # Summary
        logger.info("")
        logger.info("="*70)
        logger.info("TEST COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {results['duration_seconds']:.1f}s")
        logger.info(f"Trades Executed: {self.trades_executed}")
        logger.info(f"Circuit Breaker Events: {results['circuit_breaker_events']}")
        logger.info(f"Final CHS: {results['chs_history'][-1]:.3f}")
        logger.info("")
        
        return results
    
    # In kinetra/order_executor.py

    def get_current_price(self) -> float:
        """Get current price from MT5."""
        symbol_info = self.mt5_connection.mt5.symbol_info_tick(self.spec.symbol)
        if symbol_info is None:
            raise ConnectionError(f"Could not get tick for {self.spec.symbol}")
    
        # Use a representative price, e.g., the average of bid and ask
        return (symbol_info.bid + symbol_info.ask) / 2.0
        action = 'open_long' if direction == 1 else 'open_short'
        
        # Calculate safe SL/TP
        sl, tp = self.executor.validator.get_safe_sl_tp(
            price=current_price,
            direction=direction,
            sl_distance_pips=20,
            tp_distance_pips=40,
        )
        
        result = self.executor.execute_order(
            action=action,
            volume=0.1,  # Small lot size for demo
            sl=sl,
            tp=tp,
            comment=f"LiveTest_{self.mode}"
        )
        
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'price': current_price,
            'sl': sl,
            'tp': tp,
            'success': result.success,
            'fill_price': result.fill_price,
            'error': result.error_message if not result.success else None,
        }
        
        if result.success:
            logger.info(f"✅ Trade {self.trades_executed + 1}: {action} @ {result.fill_price:.5f}")
        else:
            logger.warning(f"❌ Trade rejected: {result.error_message}")
        
        return trade_log


def test_connection() -> bool:
    """
    Test MT5 connection only.
    
    Returns:
        True if connection successful
    """
    logger.info("="*70)
    logger.info("TESTING MT5 CONNECTION")
    logger.info("="*70)
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("❌ MetaTrader5 not installed")
        logger.info("   Install with: pip install MetaTrader5")
        return False
    
    connector = MT5Connector()
    
    if not connector.connect():
        logger.error("❌ Failed to connect to MT5")
        logger.info("   Make sure:")
        logger.info("   1. MT5 terminal is running")
        logger.info("   2. Credentials configured in .env")
        logger.info("   3. MT5 allows automated trading")
        return False
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        logger.info("✅ Connected to MT5 terminal")
        logger.info(f"   Company: {terminal_info.company}")
        logger.info(f"   Name: {terminal_info.name}")
        logger.info(f"   Path: {terminal_info.path}")
        logger.info(f"   Connected: {terminal_info.connected}")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        logger.info("")
        logger.info("Account Information:")
        logger.info(f"   Login: {account_info.login}")
        logger.info(f"   Server: {account_info.server}")
        logger.info(f"   Balance: ${account_info.balance:.2f}")
        logger.info(f"   Equity: ${account_info.equity:.2f}")
        logger.info(f"   Margin Free: ${account_info.margin_free:.2f}")
    
    connector.disconnect()
    logger.info("")
    logger.info("✅ Connection test passed")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kinetra Live Testing")
    
    parser.add_argument(
        '--mode',
        choices=['virtual', 'demo', 'live'],
        default='virtual',
        help='Testing mode'
    )
    parser.add_argument(
        '--symbol',
        default='EURUSD',
        help='Trading symbol'
    )
    parser.add_argument(
        '--agent',
        default='ppo',
        choices=['ppo', 'dqn', 'linear', 'berserker', 'triad'],
        help='Agent type'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in minutes'
    )
    parser.add_argument(
        '--max-trades',
        type=int,
        default=10,
        help='Maximum trades before auto-stop'
    )
    parser.add_argument(
        '--chs-threshold',
        type=float,
        default=0.55,
        help='CHS circuit breaker threshold'
    )
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test MT5 connection only'
    )
    
    args = parser.parse_args()
    
    # Connection test only
    if args.test_connection:
        success = test_connection()
        sys.exit(0 if success else 1)
    
    # Run live test
    runner = LiveTestRunner(
        mode=args.mode,
        symbol=args.symbol,
        agent_type=args.agent,
        max_trades=args.max_trades,
        chs_threshold=args.chs_threshold
    )
    
    results = runner.run_test(duration_minutes=args.duration)
    
    if results['success']:
        logger.info("✅ Test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Test failed")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
