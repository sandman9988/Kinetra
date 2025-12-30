"""
MT5 Specification Extractor

Extracts complete symbol specifications from MT5 terminal.
Captures all available contract details for accurate backtesting.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from .market_microstructure import AssetClass, SymbolSpec


class MT5SpecExtractor:
    """
    Extract complete symbol specifications from MT5 terminal.

    Usage:
        extractor = MT5SpecExtractor()
        spec = extractor.extract_spec("BTCUSD")
        specs_dict = extractor.extract_all(["EURUSD", "BTCUSD", "XAUUSD"])
        extractor.save_to_json(specs_dict, "data/master/instrument_specs.json")
    """

    def __init__(self, mt5_module=None):
        """
        Initialize extractor.

        Args:
            mt5_module: MetaTrader5 module (imported separately to avoid dependency)
        """
        self.mt5 = mt5_module

    def extract_spec(
        self,
        symbol: str,
        asset_class: Optional[AssetClass] = None
    ) -> SymbolSpec:
        """
        Extract complete specification for a single symbol.

        Args:
            symbol: Trading symbol (e.g., BTCUSD, EURUSD)
            asset_class: Market type (auto-detected if None)

        Returns:
            SymbolSpec with all MT5 data populated

        Raises:
            ValueError: If symbol not found in MT5
            RuntimeError: If MT5 not initialized
        """
        if self.mt5 is None:
            raise RuntimeError("MT5 module not provided. Pass mt5 module to constructor.")

        # Get symbol info from MT5
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} not found in MT5")

        # Auto-detect asset class if not provided
        if asset_class is None:
            asset_class = self._detect_asset_class(symbol, info)

        # Extract all fields from MT5
        spec = SymbolSpec(
            symbol=symbol,
            asset_class=asset_class,

            # Price precision
            digits=info.digits,
            point=info.point,

            # Contract size
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,

            # Margin (complete MT5 specification)
            margin_initial_rate_buy=getattr(info, 'margin_initial', 0.0),
            margin_initial_rate_sell=getattr(info, 'margin_initial', 0.0),
            margin_maintenance_rate_buy=getattr(info, 'margin_maintenance', 0.0),
            margin_maintenance_rate_sell=getattr(info, 'margin_maintenance', 0.0),
            margin_hedge=getattr(info, 'margin_hedged', 0.0),
            margin_currency=info.currency_margin,
            margin_mode=self._get_margin_mode(info.trade_calc_mode),

            # Costs
            spread_typical=info.spread,
            spread_min=info.spread,  # MT5 doesn't provide separate min/max
            spread_max=info.spread * 3,  # Estimate: 3x typical for news events
            commission_per_lot=0.0,  # Not available in symbol_info (from account)

            # Swap (complete specification)
            swap_long=info.swap_long,
            swap_short=info.swap_short,
            swap_type=self._get_swap_type(info.swap_type),
            swap_triple_day=self._get_swap_triple_day(info.swap_rollover3days),

            # Trading calculation
            profit_calc_mode=self._get_profit_calc_mode(info.trade_calc_mode),

            # Trading hours (extract from MT5 if available)
            trading_hours=self._extract_trading_hours(symbol),

            # Metadata
            last_updated=datetime.now(),
            source="mt5"
        )

        return spec

    def extract_all(
        self,
        symbols: List[str],
        auto_detect_asset_class: bool = True
    ) -> Dict[str, SymbolSpec]:
        """
        Extract specifications for multiple symbols.

        Args:
            symbols: List of symbols to extract
            auto_detect_asset_class: Auto-detect market type for each symbol

        Returns:
            Dictionary mapping symbol -> SymbolSpec
        """
        specs = {}

        for symbol in symbols:
            try:
                spec = self.extract_spec(symbol)
                specs[symbol] = spec
            except Exception as e:
                print(f"Warning: Failed to extract {symbol}: {e}")
                continue

        return specs

    def save_to_json(
        self,
        specs: Dict[str, SymbolSpec],
        filepath: str,
        pretty: bool = True
    ):
        """
        Save specifications to JSON file.

        Args:
            specs: Dictionary of symbol specs
            filepath: Output file path
            pretty: Pretty-print JSON (default True)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert SymbolSpec objects to dictionaries
        specs_dict = {}
        for symbol, spec in specs.items():
            specs_dict[symbol] = self._spec_to_dict(spec)

        # Write to file
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(specs_dict, f, indent=2, default=str)
            else:
                json.dump(specs_dict, f, default=str)

        print(f"Saved {len(specs)} symbol specs to {filepath}")

    def _spec_to_dict(self, spec: SymbolSpec) -> Dict[str, Any]:
        """Convert SymbolSpec to JSON-serializable dictionary."""
        return {
            'symbol': spec.symbol,
            'asset_class': spec.asset_class.value,

            # Price precision
            'digits': spec.digits,
            'point': spec.point,

            # Contract
            'contract_size': spec.contract_size,
            'volume_min': spec.volume_min,
            'volume_max': spec.volume_max,
            'volume_step': spec.volume_step,

            # Margin
            'margin_initial_rate_buy': spec.margin_initial_rate_buy,
            'margin_initial_rate_sell': spec.margin_initial_rate_sell,
            'margin_maintenance_rate_buy': spec.margin_maintenance_rate_buy,
            'margin_maintenance_rate_sell': spec.margin_maintenance_rate_sell,
            'margin_hedge': spec.margin_hedge,
            'margin_currency': spec.margin_currency,
            'margin_mode': spec.margin_mode,

            # Costs
            'spread_typical': spec.spread_typical,
            'commission_per_lot': spec.commission_per_lot,

            # Swap
            'swap_long': spec.swap_long,
            'swap_short': spec.swap_short,
            'swap_type': spec.swap_type,
            'swap_triple_day': spec.swap_triple_day,

            # Trading
            'profit_calc_mode': spec.profit_calc_mode,
            'trading_hours': spec.trading_hours,

            # Metadata
            'last_updated': spec.last_updated.isoformat() if spec.last_updated else None,
            'source': spec.source,
        }

    def _detect_asset_class(self, symbol: str, info) -> AssetClass:
        """Auto-detect asset class from symbol name and MT5 info."""
        symbol_upper = symbol.upper()
        path = getattr(info, 'path', '').lower()

        # Crypto
        if 'crypto' in path or any(x in symbol_upper for x in ['BTC', 'ETH']):
            return AssetClass.CRYPTO

        # Indices
        if 'index' in path or 'indices' in path or any(x in symbol_upper for x in ['US500', 'NAS100', 'DJ30', 'SPX', 'NDX']):
            return AssetClass.INDICES

        # Metals
        if 'metal' in path or any(x in symbol_upper for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
            return AssetClass.METALS

        # Energy
        if 'energy' in path or any(x in symbol_upper for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
            return AssetClass.ENERGY

        # ETFs
        if 'etf' in path or 'etf' in symbol.lower():
            return AssetClass.ETFS

        # Shares
        if 'stock' in path or 'shares' in path or 'equity' in path:
            return AssetClass.SHARES

        # Default: Forex
        return AssetClass.FOREX

    def _get_margin_mode(self, calc_mode: int) -> str:
        """Convert MT5 calc_mode to string."""
        modes = {
            0: "FOREX",
            1: "FUTURES",
            2: "CFD",
            3: "CFDINDEX",
            4: "CFDLEVERAGE",
            5: "FOREX_NO_LEVERAGE",
            6: "EXCH_STOCKS",
            7: "EXCH_FUTURES",
            8: "EXCH_FUTURES_FORTS",
        }
        return modes.get(calc_mode, "UNKNOWN")

    def _get_profit_calc_mode(self, calc_mode: int) -> str:
        """Convert MT5 calc_mode to profit calculation string."""
        return self._get_margin_mode(calc_mode)

    def _get_swap_type(self, swap_type: int) -> str:
        """Convert MT5 swap_type to string."""
        types = {
            0: "points",
            1: "base_currency",
            2: "interest",
            3: "margin_currency",
            4: "deposit_currency",
            5: "percent_open",
            6: "percent_annual",
        }
        return types.get(swap_type, "points")

    def _get_swap_triple_day(self, rollover_day: int) -> str:
        """Convert MT5 rollover day to string."""
        days = {
            0: "sunday",
            1: "monday",
            2: "tuesday",
            3: "wednesday",
            4: "thursday",
            5: "friday",
            6: "saturday",
        }
        return days.get(rollover_day, "wednesday")

    def _extract_trading_hours(self, symbol: str) -> Optional[Dict[str, str]]:
        """
        Extract detailed trading hours for the symbol.

        Returns:
            Dictionary mapping day names to trading hours (e.g., {"monday": "00:01-23:58"})
            Returns None if not available
        """
        # TODO: Implement trading hours extraction from MT5
        # This requires parsing the trading sessions structure from MT5
        # For now, return None (will use default market hours)
        return None


def extract_specs_from_mt5(
    symbols: List[str],
    output_file: str = "data/master/instrument_specs.json"
) -> Dict[str, SymbolSpec]:
    """
    Convenience function to extract specs and save to JSON.

    Args:
        symbols: List of symbols to extract
        output_file: Output JSON file path

    Returns:
        Dictionary of extracted specs

    Example:
        >>> specs = extract_specs_from_mt5(["EURUSD", "BTCUSD", "XAUUSD"])
        >>> # Specs automatically saved to data/master/instrument_specs.json
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 module not installed. Install with: pip install MetaTrader5")

    # Initialize MT5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        # Extract specs
        extractor = MT5SpecExtractor(mt5_module=mt5)
        specs = extractor.extract_all(symbols)

        # Save to JSON
        extractor.save_to_json(specs, output_file)

        return specs

    finally:
        mt5.shutdown()
