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
    FOREX = "forex"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY = "commodity"
    STOCK = "stock"


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
        else:
            # Default flat profile
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
        asset_class=AssetClass.INDEX,
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
        asset_class=AssetClass.INDEX,
        digits=0,
        contract_size=1,
        spread_typical=2.0,
        spread_min=1.0,
        spread_max=20.0,
        commission_per_lot=0.0,
        swap_long=-3.5,
        swap_short=-1.5,
    ),
    # Gold
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        asset_class=AssetClass.COMMODITY,
        digits=2,
        contract_size=100,       # 100 oz per lot
        spread_typical=0.30,     # $0.30 typical
        spread_min=0.15,
        spread_max=3.0,
        commission_per_lot=0.0,
        swap_long=-5.0,
        swap_short=1.0,
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


class FrictionModel:
    """
    Dynamic friction model based on market microstructure.

    Total Friction = Spread Cost + Commission + Slippage + Swap (if holding)

    All normalized to price percentage for comparability across assets.
    """

    def __init__(self, symbol_spec: SymbolSpec):
        self.spec = symbol_spec

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

    def estimate_spread(
        self,
        hour_utc: int,
        volatility: float,
        volatility_baseline: float = 0.01,
        minutes_to_rollover: Optional[int] = None
    ) -> float:
        """
        Estimate current spread based on conditions.

        Args:
            hour_utc: Current hour in UTC
            volatility: Current volatility (e.g., ATR/price)
            volatility_baseline: Normal volatility level
            minutes_to_rollover: Minutes until daily rollover

        Returns:
            Estimated spread in price units
        """
        base_spread = self.spec.spread_typical * self.spec.point

        # Liquidity adjustment
        liquidity = self.get_liquidity_factor(hour_utc)
        liquidity_multiplier = 1.0 + self.spread_liquidity_sensitivity * (1.0 - liquidity)

        # Volatility adjustment
        vol_ratio = volatility / volatility_baseline if volatility_baseline > 0 else 1.0
        vol_multiplier = 1.0 + self.spread_volatility_sensitivity * max(0, vol_ratio - 1.0)

        # Rollover penalty - spreads widen significantly near rollover
        rollover_multiplier = 1.0
        if minutes_to_rollover is not None:
            if minutes_to_rollover <= 10:
                # Within 10 minutes of rollover: spreads can be 5-10x normal
                rollover_multiplier = 5.0 + 5.0 * (1.0 - minutes_to_rollover / 10.0)
            elif minutes_to_rollover <= 30:
                # 10-30 minutes: spreads 2-5x
                rollover_multiplier = 2.0 + 3.0 * (1.0 - (minutes_to_rollover - 10) / 20.0)
            elif minutes_to_rollover <= 60:
                # 30-60 minutes: spreads 1-2x
                rollover_multiplier = 1.0 + (1.0 - (minutes_to_rollover - 30) / 30.0)

        estimated_spread = base_spread * liquidity_multiplier * vol_multiplier * rollover_multiplier

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
        hour_utc: int,
        volatility: float,
        volatility_baseline: float = 0.01,
        holding_days: float = 0.0,
        is_long: bool = True,
        minutes_to_rollover: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate total friction for a trade.

        Returns breakdown of all costs as price percentages.

        Args:
            price: Current price
            position_size_lots: Position size in lots
            hour_utc: Current hour UTC
            volatility: Current volatility
            volatility_baseline: Normal volatility
            holding_days: Expected holding period in days
            is_long: Long or short position
            minutes_to_rollover: Minutes until rollover

        Returns:
            Dict with friction components and total
        """
        # 1. Spread cost (entry + exit)
        spread = self.estimate_spread(hour_utc, volatility, volatility_baseline, minutes_to_rollover)
        spread_pct = (spread / price) * 100  # As percentage
        spread_cost = spread_pct * 2  # Entry and exit

        # 2. Commission
        commission_total = self.spec.commission_per_lot * position_size_lots * 2  # Both sides
        position_value = price * self.spec.contract_size * position_size_lots
        commission_pct = (commission_total / position_value) * 100 if position_value > 0 else 0

        # 3. Slippage
        liquidity = self.get_liquidity_factor(hour_utc)
        slippage_pct = self.estimate_slippage(position_size_lots, liquidity) * 100
        slippage_cost = slippage_pct * 2  # Entry and exit

        # 4. Swap cost (if holding overnight)
        swap_points = self.spec.swap_long if is_long else self.spec.swap_short
        swap_price = swap_points * self.spec.point
        swap_pct = (swap_price / price) * 100 * holding_days

        # Total friction
        total_friction = spread_cost + commission_pct + slippage_cost + abs(swap_pct)

        return {
            'spread_pct': spread_cost,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_cost,
            'swap_pct': swap_pct,
            'total_friction_pct': total_friction,
            'spread_price': spread,
            'liquidity_factor': liquidity,
            'minutes_to_rollover': minutes_to_rollover,
        }

    def is_tradeable(
        self,
        expected_return_pct: float,
        friction: Dict[str, float],
        min_edge_ratio: float = 1.5
    ) -> Tuple[bool, str]:
        """
        Check if trade makes sense given friction.

        Natural gate: Only trade if expected_return > friction * min_edge_ratio

        Args:
            expected_return_pct: Expected return as percentage
            friction: Friction dict from calculate_friction()
            min_edge_ratio: Minimum ratio of return to friction (default 1.5x)

        Returns:
            (is_tradeable, reason)
        """
        total_friction = friction['total_friction_pct']

        # Required edge
        required_return = total_friction * min_edge_ratio

        if expected_return_pct < required_return:
            return False, f"Expected {expected_return_pct:.3f}% < required {required_return:.3f}% (friction={total_friction:.3f}%)"

        # Check specific conditions
        if friction.get('minutes_to_rollover', 60) < 15:
            return False, f"Too close to rollover ({friction['minutes_to_rollover']} min)"

        if friction['liquidity_factor'] < 0.3:
            return False, f"Low liquidity ({friction['liquidity_factor']:.2f})"

        if friction['spread_pct'] > expected_return_pct * 0.5:
            return False, f"Spread too wide ({friction['spread_pct']:.3f}% > 50% of expected return)"

        return True, "Trade viable"


class AdaptiveFrictionTracker:
    """
    Online tracker that learns friction dynamics from actual fills.

    Updates spread/slippage models based on observed vs expected.
    """

    def __init__(self, symbol_spec: SymbolSpec, window: int = 100):
        self.spec = symbol_spec
        self.model = FrictionModel(symbol_spec)
        self.window = window

        # Observed data
        self.observed_spreads: List[Tuple[int, float, float]] = []  # (hour, volatility, spread)
        self.observed_slippage: List[Tuple[float, float, float]] = []  # (size, liquidity, slippage)

        # Adaptive parameters
        self.spread_adjustment = 1.0
        self.slippage_adjustment = 1.0

    def record_fill(
        self,
        hour_utc: int,
        volatility: float,
        position_size_lots: float,
        expected_price: float,
        fill_price: float,
        spread_observed: float
    ):
        """Record an actual trade fill to update models."""
        # Calculate observed slippage
        slippage = abs(fill_price - expected_price) / expected_price
        liquidity = self.model.get_liquidity_factor(hour_utc)

        self.observed_spreads.append((hour_utc, volatility, spread_observed))
        self.observed_slippage.append((position_size_lots, liquidity, slippage))

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
        predicted_spreads = []
        actual_spreads = []

        for hour, vol, actual in self.observed_spreads:
            predicted = self.model.estimate_spread(hour, vol)
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
    Compute friction for each bar in a dataset.

    Useful for backtesting with realistic costs.

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

    # Calculate volatility (ATR-based)
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    atr = pd.Series(tr).rolling(20).mean().values
    volatility = atr / close
    volatility_baseline = np.nanmean(volatility)

    # Determine hours (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour.values
    else:
        hours = np.zeros(len(df), dtype=int)

    # Compute friction for each bar
    frictions = []
    for i in range(len(df)):
        friction = model.calculate_friction(
            price=close[i],
            position_size_lots=position_size_lots,
            hour_utc=int(hours[i]),
            volatility=volatility[i] if not np.isnan(volatility[i]) else volatility_baseline,
            volatility_baseline=volatility_baseline,
            holding_days=holding_bars / 24,  # Assuming hourly bars
            is_long=True
        )
        frictions.append(friction)

    friction_df = pd.DataFrame(frictions, index=df.index)

    return friction_df
