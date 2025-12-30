"""
Market Microstructure and Friction Model

Real trading costs and liquidity dynamics for physics-accurate friction.

Friction = f(spread, commission, swap, slippage, liquidity)

This becomes a NATURAL GATE - when friction > expected_alpha, don't trade.
No arbitrary thresholds - the physics decides.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime, time, timedelta
from enum import Enum


class AssetClass(Enum):
    # Currency markets
    FOREX = "forex"

    # Digital assets
    CRYPTO = "crypto"

    # Equity markets
    SHARES = "shares"
    STOCK = "stock"  # Alias for shares
    INDICES = "indices"
    INDEX = "index"  # Alias for indices (backward compatibility)

    # Commodity markets - Metals
    METALS = "metals"
    METAL = "metal"  # Alias for metals

    # Commodity markets - Energy
    ENERGY = "energy"

    # General commodities (backward compatibility)
    COMMODITY = "commodity"

    # Structured products
    ETFS = "etfs"
    ETF = "etf"  # Alias for etfs


class Session(Enum):
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_TOKYO_LONDON = "overlap_tokyo_london"
    OFF_HOURS = "off_hours"


@dataclass
class SymbolSpec:
    """
    Symbol contract specification from MT5/broker.

    All values that affect friction and position sizing.
    """
    symbol: str
    asset_class: AssetClass

    # Price precision
    digits: int                     # Price decimal places (5 for EURUSD = 0.00001)
    point: float = None             # Minimum price change (auto-calculated if None)

    # Contract size
    contract_size: float = 100000   # Standard lot size (100k for forex)
    volume_min: float = 0.01        # Minimum lot
    volume_max: float = 100.0       # Maximum lot
    volume_step: float = 0.01       # Lot increment

    # Margin
    margin_initial: float = 0.01    # Initial margin rate (1% = 100:1 leverage)
    margin_maintenance: float = 0.005
    margin_currency: str = "USD"

    # Costs
    spread_typical: float = 0.0     # Typical spread in points
    spread_min: float = 0.0         # Minimum spread
    spread_max: float = 0.0         # Maximum spread (during news/rollover)
    commission_per_lot: float = 0.0 # Commission per lot per side
    swap_long: float = 0.0          # Daily swap for long positions (points)
    swap_short: float = 0.0         # Daily swap for short positions (points)

    # Session info
    session_start: time = time(0, 0)   # Market open (UTC)
    session_end: time = time(23, 59)   # Market close (UTC)
    rollover_time: time = time(21, 0)  # Daily rollover (UTC) - typically 21:00

    # Liquidity profile (normalized 0-1 by hour)
    liquidity_profile: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.point is None:
            self.point = 10 ** (-self.digits)

        # Default liquidity profile if not provided
        if not self.liquidity_profile:
            self.liquidity_profile = self._default_liquidity_profile()

    def _default_liquidity_profile(self) -> Dict[int, float]:
        """Default hourly liquidity profile based on asset class."""
        if self.asset_class == AssetClass.FOREX:
            # Forex: peaks during London/NY overlap
            return {
                0: 0.3, 1: 0.25, 2: 0.2, 3: 0.2, 4: 0.25, 5: 0.3,  # Sydney/Tokyo
                6: 0.4, 7: 0.5, 8: 0.7, 9: 0.85, 10: 0.9, 11: 0.9,  # London open
                12: 0.95, 13: 1.0, 14: 1.0, 15: 0.95, 16: 0.9,      # London/NY overlap
                17: 0.7, 18: 0.5, 19: 0.4, 20: 0.35, 21: 0.3,       # NY afternoon
                22: 0.25, 23: 0.25                                   # Sydney pre-open
            }
        elif self.asset_class == AssetClass.CRYPTO:
            # Crypto: 24/7 but peaks during US hours
            return {h: 0.7 + 0.3 * np.sin((h - 14) * np.pi / 12) for h in range(24)}
        elif self.asset_class in (AssetClass.SHARES, AssetClass.STOCK):
            # Shares: US market hours (9:30-16:00 ET = roughly 14:30-21:00 UTC)
            return {
                0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,      # Pre-market closed
                6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0,    # Pre-market closed
                12: 0.0, 13: 0.2, 14: 0.6, 15: 0.9, 16: 1.0, 17: 0.95,  # Market open
                18: 0.9, 19: 0.85, 20: 0.7, 21: 0.3,                  # Market close
                22: 0.0, 23: 0.0                                      # After hours
            }
        elif self.asset_class in (AssetClass.INDICES, AssetClass.INDEX):
            # Indices: Extended hours trading, peaks during regular session
            return {
                0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2,      # Asian session
                6: 0.3, 7: 0.4, 8: 0.5, 9: 0.6, 10: 0.7, 11: 0.75,   # European session
                12: 0.8, 13: 0.85, 14: 0.9, 15: 0.95, 16: 1.0, 17: 0.95,  # US session peak
                18: 0.9, 19: 0.85, 20: 0.7, 21: 0.5,                  # US session close
                22: 0.3, 23: 0.25                                     # After hours
            }
        elif self.asset_class in (AssetClass.METALS, AssetClass.METAL):
            # Metals (Gold, Silver): 23-hour trading, 1 hour maintenance
            return {
                0: 0.7, 1: 0.65, 2: 0.6, 3: 0.6, 4: 0.65, 5: 0.7,    # Asian session
                6: 0.75, 7: 0.8, 8: 0.85, 9: 0.9, 10: 0.95, 11: 0.95,  # London session
                12: 0.95, 13: 1.0, 14: 1.0, 15: 0.95, 16: 0.9,       # London/NY overlap
                17: 0.85, 18: 0.8, 19: 0.75, 20: 0.7, 21: 0.65,      # NY session
                22: 0.0, 23: 0.6                                      # Maintenance hour (22:00)
            }
        elif self.asset_class == AssetClass.ENERGY:
            # Energy (Oil, Gas): Peaks during London/NY hours
            return {
                0: 0.5, 1: 0.45, 2: 0.4, 3: 0.4, 4: 0.45, 5: 0.5,    # Asian session
                6: 0.6, 7: 0.7, 8: 0.8, 9: 0.85, 10: 0.9, 11: 0.9,   # London open
                12: 0.95, 13: 1.0, 14: 1.0, 15: 0.95, 16: 0.9,       # London/NY overlap
                17: 0.85, 18: 0.8, 19: 0.7, 20: 0.6, 21: 0.5,        # NY afternoon
                22: 0.45, 23: 0.5                                     # After hours
            }
        elif self.asset_class in (AssetClass.ETFS, AssetClass.ETF):
            # ETFs: Follow equity market hours
            return {
                0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,      # Pre-market closed
                6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0,    # Pre-market closed
                12: 0.0, 13: 0.2, 14: 0.6, 15: 0.9, 16: 1.0, 17: 0.95,  # Market open
                18: 0.9, 19: 0.85, 20: 0.7, 21: 0.3,                  # Market close
                22: 0.0, 23: 0.0                                      # After hours
            }
        else:
            # Default flat profile (COMMODITY and others)
            return {h: 0.8 for h in range(24)}

    def spread_in_price(self) -> float:
        """Convert typical spread from points to price units."""
        return self.spread_typical * self.point

    def pip_value(self, lot_size: float = 1.0) -> float:
        """Value of 1 pip for given lot size."""
        # For most pairs, 1 pip = 0.0001 (or 0.01 for JPY pairs)
        pip_size = 10 * self.point if self.digits == 5 or self.digits == 3 else self.point
        return pip_size * self.contract_size * lot_size


# Pre-defined symbol specifications
SYMBOL_SPECS = {
    # Major Forex Pairs
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        contract_size=100000,
        spread_typical=1.0,      # 1 pip typical
        spread_min=0.5,
        spread_max=10.0,         # During news/rollover
        commission_per_lot=3.5,  # $3.50 per lot per side
        swap_long=-6.5,
        swap_short=1.2,
    ),
    "GBPUSD": SymbolSpec(
        symbol="GBPUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        contract_size=100000,
        spread_typical=1.2,
        spread_min=0.8,
        spread_max=15.0,
        commission_per_lot=3.5,
        swap_long=-4.8,
        swap_short=0.5,
    ),
    "USDJPY": SymbolSpec(
        symbol="USDJPY",
        asset_class=AssetClass.FOREX,
        digits=3,
        contract_size=100000,
        spread_typical=1.0,
        spread_min=0.5,
        spread_max=8.0,
        commission_per_lot=3.5,
        swap_long=8.5,
        swap_short=-15.2,
    ),
    # Crypto
    "BTCUSD": SymbolSpec(
        symbol="BTCUSD",
        asset_class=AssetClass.CRYPTO,
        digits=2,
        contract_size=1,
        spread_typical=50.0,     # $50 typical spread
        spread_min=20.0,
        spread_max=500.0,        # During volatility
        commission_per_lot=0.0,  # Usually in spread
        swap_long=-0.05,         # Daily financing
        swap_short=-0.05,
    ),
    "ETHUSD": SymbolSpec(
        symbol="ETHUSD",
        asset_class=AssetClass.CRYPTO,
        digits=2,
        contract_size=1,
        spread_typical=2.0,
        spread_min=1.0,
        spread_max=50.0,
        commission_per_lot=0.0,
        swap_long=-0.05,
        swap_short=-0.05,
    ),
    # Indices
    "US500": SymbolSpec(
        symbol="US500",
        asset_class=AssetClass.INDICES,
        digits=2,
        contract_size=1,
        spread_typical=0.5,
        spread_min=0.3,
        spread_max=5.0,
        commission_per_lot=0.0,
        swap_long=-2.5,
        swap_short=-1.0,
    ),
    "US30": SymbolSpec(
        symbol="US30",
        asset_class=AssetClass.INDICES,
        digits=0,
        contract_size=1,
        spread_typical=2.0,
        spread_min=1.0,
        spread_max=20.0,
        commission_per_lot=0.0,
        swap_long=-3.5,
        swap_short=-1.5,
    ),
    "NAS100": SymbolSpec(
        symbol="NAS100",
        asset_class=AssetClass.INDICES,
        digits=2,
        contract_size=1,
        spread_typical=1.0,
        spread_min=0.5,
        spread_max=10.0,
        commission_per_lot=0.0,
        swap_long=-2.0,
        swap_short=-1.0,
    ),
    # Metals
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        asset_class=AssetClass.METALS,
        digits=2,
        contract_size=100,       # 100 oz per lot
        spread_typical=0.30,     # $0.30 typical
        spread_min=0.15,
        spread_max=3.0,
        commission_per_lot=0.0,
        swap_long=-5.0,
        swap_short=1.0,
    ),
    "XAGUSD": SymbolSpec(
        symbol="XAGUSD",
        asset_class=AssetClass.METALS,
        digits=3,
        contract_size=5000,      # 5000 oz per lot
        spread_typical=0.020,    # $0.020 typical
        spread_min=0.010,
        spread_max=0.200,
        commission_per_lot=0.0,
        swap_long=-4.0,
        swap_short=1.0,
    ),
    # Energy
    "XTIUSD": SymbolSpec(  # WTI Crude Oil
        symbol="XTIUSD",
        asset_class=AssetClass.ENERGY,
        digits=2,
        contract_size=1000,      # 1000 barrels per lot
        spread_typical=0.05,     # $0.05 typical
        spread_min=0.03,
        spread_max=0.50,
        commission_per_lot=0.0,
        swap_long=-3.0,
        swap_short=-1.0,
    ),
    "XBRUSD": SymbolSpec(  # Brent Crude Oil
        symbol="XBRUSD",
        asset_class=AssetClass.ENERGY,
        digits=2,
        contract_size=1000,      # 1000 barrels per lot
        spread_typical=0.05,     # $0.05 typical
        spread_min=0.03,
        spread_max=0.50,
        commission_per_lot=0.0,
        swap_long=-3.0,
        swap_short=-1.0,
    ),
    # Shares (Example: Major US stocks)
    "AAPL": SymbolSpec(
        symbol="AAPL",
        asset_class=AssetClass.SHARES,
        digits=2,
        contract_size=1,         # 1 share per lot
        spread_typical=0.02,     # $0.02 typical
        spread_min=0.01,
        spread_max=0.20,
        commission_per_lot=0.0,  # Usually commission-free or in spread
        swap_long=-0.02,
        swap_short=-0.02,
    ),
    # ETFs (Example)
    "SPY": SymbolSpec(
        symbol="SPY",
        asset_class=AssetClass.ETFS,
        digits=2,
        contract_size=1,         # 1 share per lot
        spread_typical=0.01,     # $0.01 typical
        spread_min=0.01,
        spread_max=0.10,
        commission_per_lot=0.0,
        swap_long=-0.01,
        swap_short=-0.01,
    ),
}


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """Get symbol specification, with fallback to generic."""
    # Normalize symbol name
    symbol = symbol.upper().replace("_", "").replace("/", "")

    if symbol in SYMBOL_SPECS:
        return SYMBOL_SPECS[symbol]

    # Try partial match
    for key, spec in SYMBOL_SPECS.items():
        if key in symbol or symbol in key:
            return spec

    # Return generic forex spec
    return SymbolSpec(
        symbol=symbol,
        asset_class=AssetClass.FOREX,
        digits=5,
        spread_typical=2.0,
        spread_min=1.0,
        spread_max=20.0,
        commission_per_lot=5.0,
    )


class TradingMode(Enum):
    """Trading mode determines whether gates are enforced."""
    VIRTUAL = "virtual"    # Exploration - NO gates, learn freely
    LIVE = "live"          # Execution - Dynamic conditional locks


class FrictionModel:
    """
    Dynamic friction model based on market microstructure.

    Total Friction = Spread Cost + Commission + Slippage + Swap (if holding)

    All normalized to price percentage for comparability across assets.

    IMPORTANT: Friction is always CALCULATED for reward shaping.
    Gates are only ENFORCED in LIVE mode, not during VIRTUAL exploration.
    """

    def __init__(self, symbol_spec: SymbolSpec, mode: TradingMode = TradingMode.VIRTUAL):
        self.spec = symbol_spec
        self.mode = mode  # Virtual = explore freely, Live = enforce gates

        # Adaptive spread model parameters (learned from data)
        self.spread_volatility_sensitivity = 2.0  # How much spread widens with vol
        self.spread_liquidity_sensitivity = 1.5   # How much spread widens with low liquidity

        # Slippage model
        self.slippage_base_pct = 0.0001  # Base slippage (0.01%)
        self.slippage_size_factor = 0.5  # How slippage scales with position size

    def get_current_session(self, hour_utc: int) -> Session:
        """Determine current trading session from UTC hour."""
        if 22 <= hour_utc or hour_utc < 6:
            if 0 <= hour_utc < 6:
                return Session.TOKYO
            return Session.SYDNEY
        elif 6 <= hour_utc < 8:
            return Session.OVERLAP_TOKYO_LONDON
        elif 8 <= hour_utc < 12:
            return Session.LONDON
        elif 12 <= hour_utc < 17:
            return Session.OVERLAP_LONDON_NY
        elif 17 <= hour_utc < 22:
            return Session.NEW_YORK
        else:
            return Session.OFF_HOURS

    def get_liquidity_factor(self, hour_utc: int) -> float:
        """
        Get liquidity factor for given hour (0-1).

        Lower liquidity = higher friction.
        """
        return self.spec.liquidity_profile.get(hour_utc, 0.5)

    def estimate_spread_from_bar(
        self,
        high: float,
        low: float,
        close: float,
        volume: float,
        volume_baseline: float,
        volatility: float,
        volatility_baseline: float = 0.01,
    ) -> float:
        """
        Estimate spread from OBSERVED bar data - no clock assumptions.

        Physics-based: spread is a function of volatility and liquidity,
        both of which we can OBSERVE from the bar data.

        Wide bars + low volume = wide spread (low liquidity, high volatility)
        Narrow bars + high volume = tight spread (high liquidity, low volatility)

        Works regardless of DST, timezone, or rollover schedule.

        Args:
            high: Bar high price
            low: Bar low price
            close: Bar close price
            volume: Bar volume (tick count or actual volume)
            volume_baseline: Rolling average volume
            volatility: Current volatility (ATR/price or similar)
            volatility_baseline: Rolling average volatility

        Returns:
            Estimated spread in price units
        """
        base_spread = self.spec.spread_typical * self.spec.point

        # PHYSICS-BASED LIQUIDITY from observed volume
        # Low volume relative to baseline = low liquidity = wide spread
        if volume_baseline > 0 and volume > 0:
            liquidity_ratio = volume / volume_baseline
            # liquidity_ratio < 1 means low liquidity, spread widens
            # Inverse relationship: 0.5x volume = 2x spread multiplier
            liquidity_multiplier = 1.0 / max(0.1, min(2.0, liquidity_ratio))
        else:
            liquidity_multiplier = 1.0

        # PHYSICS-BASED VOLATILITY from observed bar range
        # High volatility = market makers widen spreads
        vol_ratio = volatility / volatility_baseline if volatility_baseline > 0 else 1.0
        vol_multiplier = 1.0 + self.spread_volatility_sensitivity * max(0, vol_ratio - 1.0)

        # Bar range as additional spread indicator
        # Wide bar relative to normal = stressed market = wider spread
        bar_range = (high - low) / close if close > 0 else 0
        range_baseline = volatility_baseline * 2  # Approximate normal bar range
        if range_baseline > 0:
            range_ratio = bar_range / range_baseline
            range_multiplier = 1.0 + 0.5 * max(0, range_ratio - 1.5)  # Only penalize extreme ranges
        else:
            range_multiplier = 1.0

        estimated_spread = base_spread * liquidity_multiplier * vol_multiplier * range_multiplier

        # Clamp to min/max
        min_spread = self.spec.spread_min * self.spec.point
        max_spread = self.spec.spread_max * self.spec.point

        return np.clip(estimated_spread, min_spread, max_spread)

    def estimate_spread(
        self,
        hour_utc: int,
        volatility: float,
        volatility_baseline: float = 0.01,
        volume: Optional[float] = None,
        volume_baseline: Optional[float] = None,
    ) -> float:
        """
        Estimate spread - prefers physics-based if volume available.

        Falls back to hour-based liquidity profile only if no volume data.
        This handles DST gracefully because physics don't change with clocks.

        Args:
            hour_utc: Current hour in UTC (fallback only)
            volatility: Current volatility (e.g., ATR/price)
            volatility_baseline: Normal volatility level
            volume: Current bar volume (optional, for physics-based)
            volume_baseline: Rolling average volume (optional)

        Returns:
            Estimated spread in price units
        """
        base_spread = self.spec.spread_typical * self.spec.point

        # PREFER physics-based liquidity from volume if available
        if volume is not None and volume_baseline is not None and volume_baseline > 0:
            liquidity_ratio = volume / volume_baseline
            liquidity_multiplier = 1.0 / max(0.1, min(2.0, liquidity_ratio))
        else:
            # Fallback to hour-based profile (less accurate due to DST)
            liquidity = self.get_liquidity_factor(hour_utc)
            liquidity_multiplier = 1.0 + self.spread_liquidity_sensitivity * (1.0 - liquidity)

        # Volatility adjustment
        vol_ratio = volatility / volatility_baseline if volatility_baseline > 0 else 1.0
        vol_multiplier = 1.0 + self.spread_volatility_sensitivity * max(0, vol_ratio - 1.0)

        estimated_spread = base_spread * liquidity_multiplier * vol_multiplier

        # Clamp to min/max
        min_spread = self.spec.spread_min * self.spec.point
        max_spread = self.spec.spread_max * self.spec.point

        return np.clip(estimated_spread, min_spread, max_spread)

    def estimate_slippage(
        self,
        position_size_lots: float,
        liquidity_factor: float,
        is_market_order: bool = True
    ) -> float:
        """
        Estimate slippage for a given order.

        Args:
            position_size_lots: Order size in lots
            liquidity_factor: Current liquidity (0-1)
            is_market_order: True for market orders, False for limits

        Returns:
            Expected slippage as price percentage
        """
        if not is_market_order:
            return 0.0  # Limit orders have no slippage (but may not fill)

        base_slippage = self.slippage_base_pct

        # Size impact: larger orders have more slippage
        size_impact = self.slippage_size_factor * np.log1p(position_size_lots)

        # Liquidity impact: low liquidity = more slippage
        liquidity_impact = (1.0 - liquidity_factor) * 2.0

        total_slippage = base_slippage * (1.0 + size_impact) * (1.0 + liquidity_impact)

        return total_slippage

    def calculate_friction(
        self,
        price: float,
        position_size_lots: float,
        volatility: float,
        volatility_baseline: float = 0.01,
        volume: Optional[float] = None,
        volume_baseline: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        holding_days: float = 0.0,
        is_long: bool = True,
        hour_utc: int = 12,  # Only used as fallback if no volume
    ) -> Dict[str, float]:
        """
        Calculate total friction for a trade - PHYSICS BASED.

        No clock-based assumptions. Friction is derived from:
        - Observed volatility vs baseline (spread explosion)
        - Observed volume vs baseline (liquidity drying up)
        - Observed bar range (market stress)

        Returns breakdown of all costs as price percentages.

        Args:
            price: Current price
            position_size_lots: Position size in lots
            volatility: Current volatility (observed)
            volatility_baseline: Rolling average volatility
            volume: Current bar volume (observed)
            volume_baseline: Rolling average volume
            high: Current bar high (for range-based spread)
            low: Current bar low
            holding_days: Expected holding period in days
            is_long: Long or short position
            hour_utc: Fallback hour if no volume data

        Returns:
            Dict with friction components and total
        """
        # 1. SPREAD COST - physics-based
        if high is not None and low is not None and volume is not None and volume_baseline is not None:
            # Full physics-based spread estimation
            spread = self.estimate_spread_from_bar(
                high=high,
                low=low,
                close=price,
                volume=volume,
                volume_baseline=volume_baseline,
                volatility=volatility,
                volatility_baseline=volatility_baseline,
            )
        else:
            # Fallback with optional volume
            spread = self.estimate_spread(
                hour_utc=hour_utc,
                volatility=volatility,
                volatility_baseline=volatility_baseline,
                volume=volume,
                volume_baseline=volume_baseline,
            )

        spread_pct = (spread / price) * 100  # As percentage
        spread_cost = spread_pct * 2  # Entry and exit

        # 2. COMMISSION
        commission_total = self.spec.commission_per_lot * position_size_lots * 2  # Both sides
        position_value = price * self.spec.contract_size * position_size_lots
        commission_pct = (commission_total / position_value) * 100 if position_value > 0 else 0

        # 3. SLIPPAGE - physics-based from volume
        if volume is not None and volume_baseline is not None and volume_baseline > 0:
            # Liquidity from observed volume
            liquidity_factor = min(1.0, volume / volume_baseline)
        else:
            # Fallback to hour profile
            liquidity_factor = self.get_liquidity_factor(hour_utc)

        slippage_pct = self.estimate_slippage(position_size_lots, liquidity_factor) * 100
        slippage_cost = slippage_pct * 2  # Entry and exit

        # 4. SWAP COST (if holding overnight)
        swap_points = self.spec.swap_long if is_long else self.spec.swap_short
        swap_price = swap_points * self.spec.point
        swap_pct = (swap_price / price) * 100 * holding_days

        # 5. PHYSICS STRESS INDICATORS
        # Volatility stress: how much higher than normal?
        vol_stress = volatility / volatility_baseline if volatility_baseline > 0 else 1.0

        # Liquidity stress: how much lower than normal?
        if volume is not None and volume_baseline is not None and volume_baseline > 0:
            liq_stress = volume_baseline / max(volume, 1e-10)  # Inverse: low vol = high stress
        else:
            liq_stress = 1.0

        # Spread stress: how much wider than typical?
        typical_spread = self.spec.spread_typical * self.spec.point
        spread_stress = spread / typical_spread if typical_spread > 0 else 1.0

        # Total friction
        total_friction = spread_cost + commission_pct + slippage_cost + abs(swap_pct)

        return {
            'spread_pct': spread_cost,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_cost,
            'swap_pct': swap_pct,
            'total_friction_pct': total_friction,
            'spread_price': spread,
            'liquidity_factor': liquidity_factor,
            # Physics stress indicators (for diagnostics/learning)
            'volatility_stress': vol_stress,
            'liquidity_stress': liq_stress,
            'spread_stress': spread_stress,
        }

    def is_tradeable(
        self,
        expected_return_pct: float,
        friction: Dict[str, float],
        min_edge_ratio: float = 1.5,
        force_mode: Optional[TradingMode] = None
    ) -> Tuple[bool, str]:
        """
        Check if trade makes sense given friction - PHYSICS BASED.

        VIRTUAL MODE: Always returns True - let it explore and learn!
        LIVE MODE: Enforces dynamic conditional locks based on physics.

        Args:
            expected_return_pct: Expected return as percentage
            friction: Friction dict from calculate_friction()
            min_edge_ratio: Minimum ratio of return to friction (default 1.5x)
            force_mode: Override the model's mode for this check

        Returns:
            (is_tradeable, reason)
        """
        mode = force_mode or self.mode

        # VIRTUAL MODE: No gates! Explore freely to find alpha.
        # The friction is still calculated for reward shaping, but we don't block.
        if mode == TradingMode.VIRTUAL:
            return True, "Virtual mode - exploring freely"

        # LIVE MODE: Dynamic conditional locks based on physics
        total_friction = friction['total_friction_pct']

        # PRIMARY GATE: expected return must exceed friction with margin
        required_return = total_friction * min_edge_ratio

        if expected_return_pct < required_return:
            return False, f"Expected {expected_return_pct:.3f}% < required {required_return:.3f}% (friction={total_friction:.3f}%)"

        # PHYSICS-BASED STRESS CHECKS
        # These detect rollover, news, or any other stress event from the DATA

        # Spread explosion: spread > 3x normal = something is wrong
        spread_stress = friction.get('spread_stress', 1.0)
        if spread_stress > 3.0:
            return False, f"Spread explosion ({spread_stress:.1f}x normal)"

        # Liquidity crisis: volume < 20% of normal
        liquidity_stress = friction.get('liquidity_stress', 1.0)
        if liquidity_stress > 5.0:  # 5x stress = 20% of normal volume
            return False, f"Liquidity dried up ({1/liquidity_stress:.0%} of normal)"

        # Extreme volatility: 5x normal vol = unstable for entry
        vol_stress = friction.get('volatility_stress', 1.0)
        if vol_stress > 5.0:
            return False, f"Volatility explosion ({vol_stress:.1f}x normal)"

        # Combined stress: if multiple stresses compound, be cautious
        combined_stress = spread_stress * liquidity_stress * vol_stress
        if combined_stress > 10.0:
            return False, f"Combined market stress too high ({combined_stress:.1f})"

        # Spread eating too much of expected return
        if friction['spread_pct'] > expected_return_pct * 0.5:
            return False, f"Spread too wide ({friction['spread_pct']:.3f}% > 50% of expected return)"

        return True, "Trade viable"


class AdaptiveFrictionTracker:
    """
    Online tracker that learns friction dynamics from actual fills.

    Updates spread/slippage models based on observed vs expected.
    PHYSICS-BASED: learns from observed conditions, not clock time.
    """

    def __init__(self, symbol_spec: SymbolSpec, window: int = 100):
        self.spec = symbol_spec
        self.model = FrictionModel(symbol_spec)
        self.window = window

        # Observed data - physics-based
        # (volatility, volume_ratio, spread_observed)
        self.observed_spreads: List[Tuple[float, float, float]] = []
        # (size, liquidity_factor, slippage)
        self.observed_slippage: List[Tuple[float, float, float]] = []

        # Adaptive parameters
        self.spread_adjustment = 1.0
        self.slippage_adjustment = 1.0

    def record_fill(
        self,
        volatility: float,
        volatility_baseline: float,
        volume: float,
        volume_baseline: float,
        position_size_lots: float,
        expected_price: float,
        fill_price: float,
        spread_observed: float
    ):
        """
        Record an actual trade fill to update models - PHYSICS BASED.

        Uses observed volatility and volume, not clock time.
        """
        # Calculate observed slippage
        slippage = abs(fill_price - expected_price) / expected_price

        # Volume-based liquidity
        vol_ratio = volatility / volatility_baseline if volatility_baseline > 0 else 1.0
        liquidity_factor = volume / volume_baseline if volume_baseline > 0 else 1.0

        self.observed_spreads.append((vol_ratio, liquidity_factor, spread_observed))
        self.observed_slippage.append((position_size_lots, liquidity_factor, slippage))

        # Keep window size
        if len(self.observed_spreads) > self.window:
            self.observed_spreads.pop(0)
        if len(self.observed_slippage) > self.window:
            self.observed_slippage.pop(0)

        # Update adjustments
        self._update_adjustments()

    def _update_adjustments(self):
        """Update model adjustments based on observations."""
        if len(self.observed_spreads) < 10:
            return

        # Compare predicted vs observed spreads
        # For each observation, estimate what spread we would have predicted
        # given the vol_ratio and liquidity_factor
        predicted_spreads = []
        actual_spreads = []
        base_spread = self.spec.spread_typical * self.spec.point

        for vol_ratio, liq_factor, actual in self.observed_spreads:
            # Replicate physics-based estimation
            liq_mult = 1.0 / max(0.1, min(2.0, liq_factor))
            vol_mult = 1.0 + self.model.spread_volatility_sensitivity * max(0, vol_ratio - 1.0)
            predicted = base_spread * liq_mult * vol_mult
            predicted_spreads.append(predicted)
            actual_spreads.append(actual)

        if predicted_spreads:
            avg_predicted = np.mean(predicted_spreads)
            avg_actual = np.mean(actual_spreads)
            if avg_predicted > 0:
                self.spread_adjustment = avg_actual / avg_predicted

        # Similar for slippage
        if len(self.observed_slippage) >= 10:
            predicted_slip = []
            actual_slip = []

            for size, liq, actual in self.observed_slippage:
                predicted = self.model.estimate_slippage(size, liq)
                predicted_slip.append(predicted)
                actual_slip.append(actual)

            if predicted_slip:
                avg_pred = np.mean(predicted_slip)
                avg_act = np.mean(actual_slip)
                if avg_pred > 0:
                    self.slippage_adjustment = avg_act / avg_pred

    def get_adjusted_friction(self, **kwargs) -> Dict[str, float]:
        """Get friction with learned adjustments applied."""
        friction = self.model.calculate_friction(**kwargs)

        # Apply learned adjustments
        friction['spread_pct'] *= self.spread_adjustment
        friction['slippage_pct'] *= self.slippage_adjustment
        friction['total_friction_pct'] = (
            friction['spread_pct'] +
            friction['commission_pct'] +
            friction['slippage_pct'] +
            abs(friction['swap_pct'])
        )

        return friction


def compute_friction_series(
    df: pd.DataFrame,
    symbol: str,
    position_size_lots: float = 0.1,
    holding_bars: int = 1
) -> pd.DataFrame:
    """
    Compute friction for each bar in a dataset - PHYSICS BASED.

    Uses observed volume and volatility to estimate friction.
    No clock-based assumptions - works for any timezone/DST.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        symbol: Symbol name for spec lookup
        position_size_lots: Assumed position size
        holding_bars: Assumed holding period in bars

    Returns:
        DataFrame with friction columns added
    """
    spec = get_symbol_spec(symbol)
    model = FrictionModel(spec)

    # Extract OHLCV
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    # Volume (if available)
    has_volume = 'Volume' in df.columns
    if has_volume:
        volume = df['Volume'].values
        # Rolling volume baseline (20-bar average)
        volume_baseline_series = pd.Series(volume).rolling(20, min_periods=1).mean().values
    else:
        volume = None
        volume_baseline_series = None

    # Calculate volatility (ATR-based)
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(20, min_periods=1).mean().values
    volatility = atr / close
    volatility_baseline_series = pd.Series(volatility).rolling(100, min_periods=20).mean().values
    volatility_baseline_fallback = np.nanmean(volatility)

    # Fallback hours (only used if no volume data)
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour.values
    else:
        hours = np.full(len(df), 12, dtype=int)  # Noon as neutral fallback

    # Compute friction for each bar
    frictions = []
    for i in range(len(df)):
        vol = volatility[i] if not np.isnan(volatility[i]) else volatility_baseline_fallback
        vol_baseline = volatility_baseline_series[i] if not np.isnan(volatility_baseline_series[i]) else volatility_baseline_fallback

        kwargs = {
            'price': close[i],
            'position_size_lots': position_size_lots,
            'volatility': vol,
            'volatility_baseline': vol_baseline,
            'high': high[i],
            'low': low[i],
            'holding_days': holding_bars / 24,  # Assuming hourly bars
            'is_long': True,
            'hour_utc': int(hours[i]),  # Fallback only
        }

        # Add volume if available (enables full physics-based estimation)
        if has_volume and volume_baseline_series is not None:
            kwargs['volume'] = volume[i]
            kwargs['volume_baseline'] = volume_baseline_series[i]

        friction = model.calculate_friction(**kwargs)
        frictions.append(friction)

    friction_df = pd.DataFrame(frictions, index=df.index)

    return friction_df
