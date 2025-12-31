"""
Multi-Agent Trading Architecture Design Document
================================================

This module defines the architecture for specialized trading agents
organized by asset class, with independent risk management per class.

DESIGN RATIONALE
----------------
1. Different asset classes have fundamentally different market dynamics:
   - Forex: Mean-reverting intraday, 24/5, session-driven
   - Indices: Exchange-hour limited, gap risk, VIX-driven
   - Crypto: 24/7, volatility clustering, low MR, high trend
   - Precious Metals: Trending, safe-haven flows, USD inverse, NOT mean-reverting
   - Energy/Industrial: Inventory cycles, seasonal, macro-sensitive

2. A single model trained on all treats fundamentally different market
   microstructures as equivalent, which is statistically unsound.

3. Leverage ratios from broker don't reflect economic risk:
   - Broker may offer 500:1 on Gold, but volatility demands 30-50:1 effective

ASSET CLASS TAXONOMY
--------------------
Class               | Instruments                      | Leverage Cap | Profile
--------------------|----------------------------------|--------------|------------------
Forex               | EURUSD, GBPUSD, AUDJPY, etc.    | 100:1        | Mean-reverting
Equity Indices      | NAS100, DJ30ft, Nikkei225, etc. | 30:1         | Exchange hours only
Crypto              | BTCUSD, ETHUSD, XRPJPY          | 20:1         | 24/7, high vol
Precious Metals     | XAUUSD, XAGUSD, XPTUSD          | 40:1         | Trending, USD inverse
Energy/Commodities  | UKOUSD, COPPER-C                | 30:1         | Inventory cycles
Stocks (future)     | AAPL, TSLA, etc.                | 20:1         | Idiosyncratic risk

AGENT ARCHITECTURE
------------------
1. Router Layer:
   - AssetClassRouter: Maps instrument -> specialist agent

2. Specialist Agents (one per asset class):
   - Alpha_Forex: Mean-reversion focus, session awareness
   - Alpha_Index: Exchange-hour trading, VIX regime
   - Alpha_Crypto: Momentum + volatility regime
   - Alpha_Metals: Trend-following, macro regime
   - Alpha_Energy: Inventory cycles, seasonal patterns

3. Doppelganger System (per agent):
   - Live Agent: Executes real trades
   - Shadow A (Frozen): Copy from N days ago, detects drift
   - Shadow B (Online): Candidate for promotion

4. Risk Management Layer:
   - Per-instrument volatility sizing (Yang-Zhang / Rogers-Satchell)
   - Class-specific leverage caps
   - Market calendar filtering (no trades during closures)
   - TradeValidator: spread, volume, MAE checks

POSITION SIZING FORMULA
-----------------------
    position_size = min(
        margin_available * leverage_cap,
        portfolio_risk_budget / (instrument_vol * point_value)
    )

This normalizes RISK across instruments, not lot size.

REWARD SHAPING PER CLASS
------------------------
- Forex: Balanced (PnL + edge_ratio + MAE penalty)
- Index: High regime penalty (avoid non-trading hours)
- Crypto: Risk-averse (high MAE penalty for whipsaw)
- Metals: Trend bonus (reward holding winners)
- Energy: Inventory-aware (contango/backwardation signals)

IMPLEMENTATION PHASES
---------------------
Phase 0: Data pipeline fixes (datetime, calendar awareness)
Phase 1: Instrument metadata + AssetClassRouter
Phase 2: Specialist agents with class-specific reward
Phase 3: Volatility-adjusted sizing + TradeValidator
Phase 4: Doppelganger system + rotation protocol
Phase 5: Portfolio-level risk (correlation filter, VaR guardrail)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class AssetClass(Enum):
    """Asset class enumeration with distinct trading characteristics."""
    FOREX = "Forex"
    EQUITY_INDEX = "EquityIndex"
    CRYPTO = "Crypto"
    PRECIOUS_METALS = "PreciousMetals"
    ENERGY_COMMODITIES = "EnergyCommodities"
    STOCKS = "Stocks"  # Future expansion


@dataclass
class InstrumentProfile:
    """
    Complete profile for an instrument including trading characteristics.

    This replaces the simple ASSET_CLASS dict with rich metadata that
    informs agent selection, risk sizing, and trade validation.
    """
    symbol: str
    asset_class: AssetClass
    leverage_cap: float  # Maximum effective leverage (NOT broker leverage)
    volatility_model: str = "yang_zhang"  # or "rogers_satchell", "parkinson"

    # Trading hours (24h format, UTC)
    trading_hours: Tuple[int, int] = (0, 24)  # Default 24/7
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Exchange calendar (for holiday filtering)
    exchange: str = "24/7"  # or "NYSE", "TSE", "LSE", etc.

    # Mean reversion profile (higher = more MR, lower = more trending)
    mr_coefficient: float = 0.5  # 0-1 scale

    # Typical spread in points (for trade validation)
    typical_spread: float = 1.0

    # Point value for PnL calculation
    point_value: float = 1.0


# Master instrument database with correct classification
INSTRUMENT_PROFILES: Dict[str, InstrumentProfile] = {
    # === FOREX (Mean-Reverting, 24/5, Session-Driven) ===
    "AUDJPY+": InstrumentProfile(
        symbol="AUDJPY+", asset_class=AssetClass.FOREX, leverage_cap=100,
        trading_hours=(0, 24), mr_coefficient=0.7, typical_spread=2.0
    ),
    "AUDUSD+": InstrumentProfile(
        symbol="AUDUSD+", asset_class=AssetClass.FOREX, leverage_cap=100,
        trading_hours=(0, 24), mr_coefficient=0.7, typical_spread=1.5
    ),
    "EURJPY+": InstrumentProfile(
        symbol="EURJPY+", asset_class=AssetClass.FOREX, leverage_cap=100,
        trading_hours=(0, 24), mr_coefficient=0.6, typical_spread=2.0
    ),
    "GBPJPY+": InstrumentProfile(
        symbol="GBPJPY+", asset_class=AssetClass.FOREX, leverage_cap=100,
        trading_hours=(0, 24), mr_coefficient=0.5, typical_spread=3.0
    ),
    "GBPUSD+": InstrumentProfile(
        symbol="GBPUSD+", asset_class=AssetClass.FOREX, leverage_cap=100,
        trading_hours=(0, 24), mr_coefficient=0.6, typical_spread=1.5
    ),

    # === EQUITY INDICES (Exchange Hours, Gap Risk, VIX-Driven) ===
    "NAS100": InstrumentProfile(
        symbol="NAS100", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(14, 21), exchange="NYSE", mr_coefficient=0.4, typical_spread=1.0
    ),
    "DJ30ft": InstrumentProfile(
        symbol="DJ30ft", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(14, 21), exchange="NYSE", mr_coefficient=0.4, typical_spread=2.0
    ),
    "Nikkei225": InstrumentProfile(
        symbol="Nikkei225", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(0, 6), exchange="TSE", mr_coefficient=0.5, typical_spread=5.0
    ),
    "GER40": InstrumentProfile(
        symbol="GER40", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(8, 16), exchange="XETRA", mr_coefficient=0.5, typical_spread=1.0
    ),
    "EU50": InstrumentProfile(
        symbol="EU50", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(8, 16), exchange="EUREX", mr_coefficient=0.5, typical_spread=1.0
    ),
    "SA40": InstrumentProfile(
        symbol="SA40", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(7, 15), exchange="JSE", mr_coefficient=0.5, typical_spread=5.0
    ),
    "US2000": InstrumentProfile(
        symbol="US2000", asset_class=AssetClass.EQUITY_INDEX, leverage_cap=30,
        trading_hours=(14, 21), exchange="NYSE", mr_coefficient=0.4, typical_spread=0.5
    ),

    # === CRYPTO (24/7, High Volatility, Trending) ===
    "BTCUSD": InstrumentProfile(
        symbol="BTCUSD", asset_class=AssetClass.CRYPTO, leverage_cap=20,
        trading_hours=(0, 24), trading_days=[0, 1, 2, 3, 4, 5, 6],
        mr_coefficient=0.2, typical_spread=50.0
    ),
    "BTCJPY": InstrumentProfile(
        symbol="BTCJPY", asset_class=AssetClass.CRYPTO, leverage_cap=20,
        trading_hours=(0, 24), trading_days=[0, 1, 2, 3, 4, 5, 6],
        mr_coefficient=0.2, typical_spread=5000.0
    ),
    "ETHEUR": InstrumentProfile(
        symbol="ETHEUR", asset_class=AssetClass.CRYPTO, leverage_cap=20,
        trading_hours=(0, 24), trading_days=[0, 1, 2, 3, 4, 5, 6],
        mr_coefficient=0.3, typical_spread=3.0
    ),
    "XRPJPY": InstrumentProfile(
        symbol="XRPJPY", asset_class=AssetClass.CRYPTO, leverage_cap=20,
        trading_hours=(0, 24), trading_days=[0, 1, 2, 3, 4, 5, 6],
        mr_coefficient=0.3, typical_spread=0.5
    ),

    # === PRECIOUS METALS (Trending, Safe-Haven, NOT Mean-Reverting) ===
    "XAUUSD+": InstrumentProfile(
        symbol="XAUUSD+", asset_class=AssetClass.PRECIOUS_METALS, leverage_cap=40,
        trading_hours=(0, 24), mr_coefficient=0.3, typical_spread=30.0
    ),
    "XAUAUD+": InstrumentProfile(
        symbol="XAUAUD+", asset_class=AssetClass.PRECIOUS_METALS, leverage_cap=40,
        trading_hours=(0, 24), mr_coefficient=0.3, typical_spread=50.0
    ),
    "XAGUSD": InstrumentProfile(
        symbol="XAGUSD", asset_class=AssetClass.PRECIOUS_METALS, leverage_cap=40,
        trading_hours=(0, 24), mr_coefficient=0.25, typical_spread=3.0
    ),
    "XPTUSD": InstrumentProfile(
        symbol="XPTUSD", asset_class=AssetClass.PRECIOUS_METALS, leverage_cap=40,
        trading_hours=(0, 24), mr_coefficient=0.35, typical_spread=100.0
    ),

    # === ENERGY & INDUSTRIAL COMMODITIES (Inventory Cycles, Seasonal) ===
    "UKOUSD": InstrumentProfile(
        symbol="UKOUSD", asset_class=AssetClass.ENERGY_COMMODITIES, leverage_cap=30,
        trading_hours=(1, 22), mr_coefficient=0.4, typical_spread=5.0
    ),
    "COPPER-C": InstrumentProfile(
        symbol="COPPER-C", asset_class=AssetClass.ENERGY_COMMODITIES, leverage_cap=30,
        trading_hours=(1, 22), mr_coefficient=0.45, typical_spread=10.0
    ),
}


def get_instrument_profile(symbol: str) -> Optional[InstrumentProfile]:
    """Get instrument profile, matching by prefix if exact match not found."""
    # Exact match
    if symbol in INSTRUMENT_PROFILES:
        return INSTRUMENT_PROFILES[symbol]

    # Prefix match (for symbols like "AUDJPY+_H1")
    for key, profile in INSTRUMENT_PROFILES.items():
        if symbol.startswith(key.rstrip('+')):
            return profile

    return None


def get_asset_class(symbol: str) -> AssetClass:
    """Get asset class for a symbol."""
    profile = get_instrument_profile(symbol)
    if profile:
        return profile.asset_class
    return AssetClass.FOREX  # Default fallback


@dataclass
class RewardProfile:
    """
    Class-specific reward shaping parameters.

    Different asset classes need different reward emphasis:
    - Forex: Balanced, penalize whipsaw
    - Index: Heavy regime penalty (avoid non-trading)
    - Crypto: High MAE penalty (control tail risk)
    - Metals: Trend bonus (reward holding winners)
    - Energy: Inventory-aware signals
    """
    pnl_weight: float = 1.0
    edge_ratio_weight: float = 0.3
    mae_penalty_weight: float = 2.5
    regime_bonus_weight: float = 0.2
    trend_bonus_weight: float = 0.0  # For trending assets
    holding_bonus_weight: float = 0.0  # Reward holding winners


# Class-specific reward profiles
REWARD_PROFILES: Dict[AssetClass, RewardProfile] = {
    AssetClass.FOREX: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.3, mae_penalty_weight=2.5,
        regime_bonus_weight=0.2, trend_bonus_weight=0.0, holding_bonus_weight=0.0
    ),
    AssetClass.EQUITY_INDEX: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.4, mae_penalty_weight=3.0,
        regime_bonus_weight=0.5,  # High - avoid non-trading hours
        trend_bonus_weight=0.0, holding_bonus_weight=0.0
    ),
    AssetClass.CRYPTO: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.2, mae_penalty_weight=4.0,  # Very high
        regime_bonus_weight=0.3, trend_bonus_weight=0.2, holding_bonus_weight=0.1
    ),
    AssetClass.PRECIOUS_METALS: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.2, mae_penalty_weight=2.0,
        regime_bonus_weight=0.2, trend_bonus_weight=0.4,  # High - reward trends
        holding_bonus_weight=0.3  # Reward holding through pullbacks
    ),
    AssetClass.ENERGY_COMMODITIES: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.3, mae_penalty_weight=2.5,
        regime_bonus_weight=0.3, trend_bonus_weight=0.2, holding_bonus_weight=0.1
    ),
    AssetClass.STOCKS: RewardProfile(
        pnl_weight=1.0, edge_ratio_weight=0.4, mae_penalty_weight=3.0,
        regime_bonus_weight=0.4, trend_bonus_weight=0.1, holding_bonus_weight=0.1
    ),
}


class AssetClassRouter:
    """
    Routes instruments to their specialist agents.

    Each asset class has its own agent with:
    - Class-specific reward shaping
    - Appropriate leverage caps
    - Trading hour awareness
    """

    def __init__(self):
        self.agents: Dict[AssetClass, object] = {}  # Populated during training

    def get_agent_class(self, symbol: str) -> AssetClass:
        """Determine which agent class should handle this symbol."""
        return get_asset_class(symbol)

    def get_reward_profile(self, symbol: str) -> RewardProfile:
        """Get class-specific reward profile for symbol."""
        asset_class = self.get_agent_class(symbol)
        return REWARD_PROFILES.get(asset_class, REWARD_PROFILES[AssetClass.FOREX])

    def is_trading_allowed(self, symbol: str, hour_utc: int, day_of_week: int) -> bool:
        """Check if trading is allowed for this symbol at this time."""
        profile = get_instrument_profile(symbol)
        if not profile:
            return True  # Default allow

        # Check day of week
        if day_of_week not in profile.trading_days:
            return False

        # Check trading hours
        start_hour, end_hour = profile.trading_hours
        if start_hour <= end_hour:
            return start_hour <= hour_utc < end_hour
        else:
            # Wraps around midnight
            return hour_utc >= start_hour or hour_utc < end_hour

    def get_leverage_cap(self, symbol: str) -> float:
        """Get maximum effective leverage for this symbol."""
        profile = get_instrument_profile(symbol)
        return profile.leverage_cap if profile else 50.0


class VolatilityEstimator:
    """
    Volatility estimation using advanced estimators.

    Yang-Zhang and Rogers-Satchell are more efficient than simple ATR
    for position sizing and risk management.
    """

    @staticmethod
    def yang_zhang(high: np.ndarray, low: np.ndarray,
                   open_: np.ndarray, close: np.ndarray,
                   window: int = 20) -> np.ndarray:
        """
        Yang-Zhang volatility estimator.

        Combines overnight and intraday components for more accurate
        volatility estimation than simple ATR or Parkinson.
        """
        n = len(close)
        vol = np.zeros(n)

        for i in range(window, n):
            h = high[i-window:i]
            l = low[i-window:i]
            o = open_[i-window:i]
            c = close[i-window:i]
            c_prev = close[i-window-1:i-1] if i > window else c

            # Overnight variance
            log_oc = np.log(o / c_prev)
            overnight_var = np.var(log_oc)

            # Open-to-close variance
            log_co = np.log(c / o)
            open_close_var = np.var(log_co)

            # Rogers-Satchell component
            log_hc = np.log(h / c)
            log_lc = np.log(l / c)
            rs_var = np.mean(log_hc * (log_hc - log_co) + log_lc * (log_lc - log_co))

            # Yang-Zhang combination
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            vol[i] = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)

        return vol * np.sqrt(252)  # Annualized

    @staticmethod
    def rogers_satchell(high: np.ndarray, low: np.ndarray,
                        open_: np.ndarray, close: np.ndarray,
                        window: int = 20) -> np.ndarray:
        """
        Rogers-Satchell volatility estimator.

        Drift-independent, good for trending markets (metals, crypto).
        """
        n = len(close)
        vol = np.zeros(n)

        for i in range(window, n):
            h = high[i-window:i]
            l = low[i-window:i]
            o = open_[i-window:i]
            c = close[i-window:i]

            log_ho = np.log(h / o)
            log_hc = np.log(h / c)
            log_lo = np.log(l / o)
            log_lc = np.log(l / c)

            rs = log_ho * log_hc + log_lo * log_lc
            vol[i] = np.sqrt(np.mean(rs))

        return vol * np.sqrt(252)  # Annualized


@dataclass
class TradeValidation:
    """Result of trade validation check."""
    allowed: bool
    reason: str = ""
    adjusted_size: float = 1.0  # Multiplier for position sizing


class TradeValidator:
    """
    Validates trades before execution based on:
    - Spread conditions (from SpreadGate)
    - Volume/liquidity
    - Market hours
    - Volatility regime
    - MAE threshold
    """

    def __init__(self,
                 max_spread_ratio: float = 3.0,
                 min_volume_ratio: float = 0.3,
                 max_mae_threshold: float = 2.0):
        self.max_spread_ratio = max_spread_ratio
        self.min_volume_ratio = min_volume_ratio
        self.max_mae_threshold = max_mae_threshold

        # Rolling statistics per instrument
        self.mae_history: Dict[str, List[float]] = {}

    def validate(self,
                 symbol: str,
                 spread_ratio: float,
                 volume_ratio: float,
                 hour_utc: int,
                 day_of_week: int,
                 recent_mae: Optional[float] = None) -> TradeValidation:
        """
        Comprehensive trade validation.

        Returns TradeValidation with allowed=True/False and reason.
        """
        profile = get_instrument_profile(symbol)

        # Check trading hours
        if profile:
            if day_of_week not in profile.trading_days:
                return TradeValidation(False, f"Market closed (day {day_of_week})")

            start_h, end_h = profile.trading_hours
            if start_h <= end_h:
                if not (start_h <= hour_utc < end_h):
                    return TradeValidation(False, f"Outside trading hours ({hour_utc}h)")
            else:
                if not (hour_utc >= start_h or hour_utc < end_h):
                    return TradeValidation(False, f"Outside trading hours ({hour_utc}h)")

        # Check spread
        if spread_ratio > self.max_spread_ratio:
            return TradeValidation(False, f"Spread too wide ({spread_ratio:.1f}x)")

        # Check volume
        if volume_ratio < self.min_volume_ratio:
            return TradeValidation(False, f"Volume too low ({volume_ratio:.2f}x)")

        # Check MAE history
        if recent_mae is not None and recent_mae > self.max_mae_threshold:
            return TradeValidation(
                False,
                f"Recent MAE too high ({recent_mae:.2f} > {self.max_mae_threshold})"
            )

        # Adjust size based on conditions
        size_mult = 1.0
        if spread_ratio > 2.0:
            size_mult *= 0.5  # Reduce size in wide spread
        if volume_ratio < 0.5:
            size_mult *= 0.7  # Reduce size in low volume

        return TradeValidation(True, "OK", size_mult)


# Export for use in framework
__all__ = [
    'AssetClass',
    'InstrumentProfile',
    'INSTRUMENT_PROFILES',
    'get_instrument_profile',
    'get_asset_class',
    'RewardProfile',
    'REWARD_PROFILES',
    'AssetClassRouter',
    'VolatilityEstimator',
    'TradeValidator',
    'TradeValidation',
]
