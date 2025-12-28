"""
Symbol Specifications and Friction Costs

Uses MT5 symbol_info and contract specs to calculate realistic trading friction:
- Spread costs (bid-ask friction)
- Swap costs (overnight carry costs)
- Margin requirements (capital efficiency)
- Contract specifications (lot sizing)

This feeds into the physics engine as market viscosity/friction.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class SymbolSpec:
    """Complete symbol specification from MT5."""

    # Identity
    symbol: str
    description: str = ""
    sector: str = "Currency"

    # Price precision
    digits: int = 5
    point: float = 0.00001  # Minimum price change

    # Contract details
    contract_size: float = 100000  # Standard forex lot
    margin_currency: str = "USD"
    profit_currency: str = "USD"
    calculation_mode: str = "Forex"

    # Margin requirements
    initial_margin: float = 0  # 0 = calculated from leverage
    maintenance_margin: float = 0
    margin_rate: float = 1.0  # 1.0 = 100% (1:1 leverage on margin)

    # Spread
    spread_type: str = "floating"  # "floating" or "fixed"
    spread_typical: float = 0  # Typical spread in points
    spread_min: float = 0
    spread_max: float = 0

    # Swap/Carry costs
    swap_type: str = "points"  # "points" or "percentage"
    swap_long: float = 0  # Per lot per day
    swap_short: float = 0
    swap_3day: str = "Wednesday"  # Triple swap day

    # Volume constraints
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

    # Trading sessions (simplified)
    trade_enabled: bool = True

    def spread_cost_pct(self, price: float) -> float:
        """Calculate spread cost as percentage of price."""
        spread_price = self.spread_typical * self.point
        return (spread_price / price) * 100 if price > 0 else 0

    def swap_cost_daily(self, position_type: str, lots: float, price: float) -> float:
        """Calculate daily swap cost in account currency.

        Args:
            position_type: 'long' or 'short'
            lots: Position size in lots
            price: Current price

        Returns:
            Daily swap cost (positive = credit, negative = cost)
        """
        swap_points = self.swap_long if position_type == 'long' else self.swap_short

        if self.swap_type == 'points':
            # Swap in points - convert to currency
            swap_value = swap_points * self.point * self.contract_size * lots
        else:
            # Swap as percentage
            swap_value = (swap_points / 100) * price * self.contract_size * lots

        return swap_value

    def margin_required(self, lots: float, price: float, leverage: float = 100) -> float:
        """Calculate margin required for position.

        Args:
            lots: Position size
            price: Current price
            leverage: Account leverage (e.g., 100 for 1:100)

        Returns:
            Margin required in margin currency
        """
        if self.initial_margin > 0:
            return self.initial_margin * lots

        notional = lots * self.contract_size * price
        return notional / leverage * self.margin_rate

    def friction_score(self, price: float, avg_daily_range: float = 0.01) -> float:
        """Calculate overall friction score (0-1).

        Combines spread, swap costs, and margin requirements into
        a single friction metric for the physics engine.

        Args:
            price: Current price
            avg_daily_range: Average daily range as percentage

        Returns:
            Friction score 0-1 (higher = more friction)
        """
        # Spread friction (as % of typical daily range)
        spread_pct = self.spread_cost_pct(price)
        spread_friction = min(1.0, spread_pct / avg_daily_range) if avg_daily_range > 0 else 0.5

        # Swap friction (magnitude of overnight cost)
        max_swap = max(abs(self.swap_long), abs(self.swap_short))
        swap_friction = min(1.0, max_swap / 20)  # 20 points = high swap

        # Margin friction (higher margin = less capital efficiency)
        margin_friction = min(1.0, self.margin_rate)

        # Weighted combination
        friction = (
            0.5 * spread_friction +    # Spread is biggest friction
            0.3 * swap_friction +      # Swap matters for swing trades
            0.2 * margin_friction      # Margin affects sizing
        )

        return friction

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'description': self.description,
            'sector': self.sector,
            'digits': self.digits,
            'contract_size': self.contract_size,
            'spread_typical': self.spread_typical,
            'swap_long': self.swap_long,
            'swap_short': self.swap_short,
            'volume_min': self.volume_min,
            'volume_max': self.volume_max,
            'margin_rate': self.margin_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolSpec':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Default specifications for common instruments
DEFAULT_SPECS = {
    # Forex Majors
    'EURUSD': SymbolSpec(
        symbol='EURUSD', description='Euro vs US Dollar', sector='Currency',
        digits=5, point=0.00001, contract_size=100000,
        spread_typical=10, swap_long=-6.5, swap_short=1.2,
    ),
    'USDJPY': SymbolSpec(
        symbol='USDJPY', description='US Dollar vs Japanese Yen', sector='Currency',
        digits=3, point=0.001, contract_size=100000,
        spread_typical=12, swap_long=7.26, swap_short=-16.66,
    ),
    'GBPUSD': SymbolSpec(
        symbol='GBPUSD', description='British Pound vs US Dollar', sector='Currency',
        digits=5, point=0.00001, contract_size=100000,
        spread_typical=15, swap_long=-4.2, swap_short=-1.8,
    ),
    'AUDUSD': SymbolSpec(
        symbol='AUDUSD', description='Australian Dollar vs US Dollar', sector='Currency',
        digits=5, point=0.00001, contract_size=100000,
        spread_typical=12, swap_long=-2.5, swap_short=-1.2,
    ),

    # Crypto
    'BTCUSD': SymbolSpec(
        symbol='BTCUSD', description='Bitcoin vs US Dollar', sector='Crypto',
        digits=2, point=0.01, contract_size=1,
        spread_typical=5000, swap_long=-25, swap_short=-25,  # High carry cost
        volume_min=0.01, volume_max=10,
    ),
    'ETHUSD': SymbolSpec(
        symbol='ETHUSD', description='Ethereum vs US Dollar', sector='Crypto',
        digits=2, point=0.01, contract_size=1,
        spread_typical=300, swap_long=-20, swap_short=-20,
        volume_min=0.01, volume_max=50,
    ),

    # Commodities
    'XAUUSD': SymbolSpec(
        symbol='XAUUSD', description='Gold vs US Dollar', sector='Commodities',
        digits=2, point=0.01, contract_size=100,  # 100 oz per lot
        spread_typical=30, swap_long=-8.5, swap_short=2.1,
    ),
    'COPPER-C': SymbolSpec(
        symbol='COPPER-C', description='Copper Futures', sector='Commodities',
        digits=4, point=0.0001, contract_size=25000,  # 25,000 lbs
        spread_typical=15, swap_long=-5, swap_short=-5,
        volume_min=0.1, volume_max=50,
    ),
    'USOIL': SymbolSpec(
        symbol='USOIL', description='US Crude Oil', sector='Commodities',
        digits=2, point=0.01, contract_size=1000,  # 1000 barrels
        spread_typical=5, swap_long=-10, swap_short=-10,
    ),

    # Indices
    'US500': SymbolSpec(
        symbol='US500', description='S&P 500 Index', sector='Indices',
        digits=1, point=0.1, contract_size=1,
        spread_typical=5, swap_long=-8, swap_short=-2,
    ),
    'US30': SymbolSpec(
        symbol='US30', description='Dow Jones 30', sector='Indices',
        digits=1, point=0.1, contract_size=1,
        spread_typical=20, swap_long=-10, swap_short=-3,
    ),
}


class SymbolSpecManager:
    """Manages symbol specifications."""

    def __init__(self, specs_file: Optional[Path] = None):
        self.specs: Dict[str, SymbolSpec] = DEFAULT_SPECS.copy()
        self.specs_file = specs_file

        if specs_file and Path(specs_file).exists():
            self.load(specs_file)

    def get(self, symbol: str) -> SymbolSpec:
        """Get spec for symbol, or create default."""
        # Try exact match
        if symbol in self.specs:
            return self.specs[symbol]

        # Try without suffix (e.g., BTCUSD from BTCUSD.r)
        base = symbol.split('.')[0].split('_')[0]
        if base in self.specs:
            return self.specs[base]

        # Return generic spec
        return SymbolSpec(symbol=symbol)

    def add(self, spec: SymbolSpec):
        """Add or update symbol spec."""
        self.specs[spec.symbol] = spec

    def save(self, path: Optional[Path] = None):
        """Save specs to JSON file."""
        path = path or self.specs_file
        if path:
            data = {k: v.to_dict() for k, v in self.specs.items()}
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load specs from JSON file."""
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.specs[k] = SymbolSpec.from_dict(v)

    def friction_by_symbol(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get friction scores for multiple symbols.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            Dict of symbol -> friction score
        """
        return {
            symbol: self.get(symbol).friction_score(price)
            for symbol, price in prices.items()
        }


def calculate_trading_costs(
    symbol: str,
    entry_price: float,
    exit_price: float,
    lots: float,
    hold_days: int,
    position_type: str,
    specs: Optional[SymbolSpecManager] = None,
) -> Dict[str, float]:
    """Calculate complete trading costs for a trade.

    Args:
        symbol: Trading symbol
        entry_price: Entry price
        exit_price: Exit price
        lots: Position size
        hold_days: Days position held
        position_type: 'long' or 'short'
        specs: Symbol spec manager

    Returns:
        Dictionary with cost breakdown
    """
    specs = specs or SymbolSpecManager()
    spec = specs.get(symbol)

    # Spread cost (paid on entry and exit)
    spread_value = spec.spread_typical * spec.point * spec.contract_size * lots
    spread_cost = spread_value * 2  # Entry + exit

    # Swap cost (per day held)
    daily_swap = spec.swap_cost_daily(position_type, lots, entry_price)
    total_swap = daily_swap * hold_days

    # Gross P&L
    price_diff = exit_price - entry_price
    if position_type == 'short':
        price_diff = -price_diff
    gross_pnl = price_diff * spec.contract_size * lots

    # Net P&L
    net_pnl = gross_pnl - spread_cost + total_swap  # Swap can be + or -

    return {
        'gross_pnl': gross_pnl,
        'spread_cost': spread_cost,
        'swap_cost': -total_swap,  # Show as cost (negate if positive)
        'total_costs': spread_cost - total_swap,
        'net_pnl': net_pnl,
        'cost_pct': ((spread_cost - total_swap) / abs(gross_pnl) * 100) if gross_pnl != 0 else 0,
    }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("KINETRA SYMBOL SPECIFICATIONS")
    print("=" * 60)

    manager = SymbolSpecManager()

    # Show some specs
    for symbol in ['USDJPY', 'BTCUSD', 'XAUUSD', 'COPPER-C']:
        spec = manager.get(symbol)
        print(f"\n{symbol} ({spec.description})")
        print(f"  Contract: {spec.contract_size:,}")
        print(f"  Spread: {spec.spread_typical} pts ({spec.spread_cost_pct(150):.4f}%)")
        print(f"  Swap Long: {spec.swap_long}")
        print(f"  Swap Short: {spec.swap_short}")
        print(f"  Friction Score: {spec.friction_score(150):.3f}")

    # Example trade costs
    print("\n" + "=" * 60)
    print("EXAMPLE TRADE COST CALCULATION")
    print("=" * 60)

    costs = calculate_trading_costs(
        symbol='USDJPY',
        entry_price=150.000,
        exit_price=151.500,
        lots=0.1,
        hold_days=5,
        position_type='long',
    )

    print("\nUSDJPY Long 0.1 lot, 150 pips profit, 5 days:")
    for k, v in costs.items():
        print(f"  {k}: ${v:.2f}" if 'pct' not in k else f"  {k}: {v:.2f}%")
