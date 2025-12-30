"""
UnifiedDataLoader: One loader for all market data.

Automatically handles:
- CSV loading (any format)
- Symbol spec lookup
- Market type detection
- Market-specific preprocessing
- Physics state computation
- Quality validation

Returns standardized DataPackage ready for any backtest engine.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import re

import pandas as pd
import numpy as np

from .data_package import DataPackage, DataFormat
from .data_utils import load_mt5_csv
from .market_microstructure import AssetClass, SymbolSpec, SYMBOL_SPECS, get_symbol_spec

import json


class UnifiedDataLoader:
    """
    Universal data loader for all market types.

    Usage:
        loader = UnifiedDataLoader()
        pkg = loader.load("data/master/BTCUSD_H1_20240101_20241231.csv")

        # Now use with any engine
        backtest_data = pkg.to_backtest_engine_format()
        physics_state, prices = pkg.to_rl_environment_format()
    """

    def __init__(
        self,
        validate: bool = True,
        compute_physics: bool = False,  # Disabled by default (requires numpy/heavy deps)
        verbose: bool = False,
        specs_file: Optional[str] = None
    ):
        """
        Initialize data loader.

        Args:
            validate: Run data quality validation
            compute_physics: Compute physics state (requires PhysicsEngine)
            verbose: Print loading progress
            specs_file: Path to instrument_specs.json (auto-detected if None)
        """
        self.validate = validate
        self.compute_physics = compute_physics
        self.verbose = verbose

        # Load instrument specs from JSON if available
        self.specs_cache = self._load_specs_from_json(specs_file)

    def load(
        self,
        filepath: Union[str, Path],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        market_type: Optional[AssetClass] = None,
        symbol_spec: Optional[SymbolSpec] = None
    ) -> DataPackage:
        """
        Load data from CSV file and return standardized DataPackage.

        Args:
            filepath: Path to CSV file
            symbol: Trading symbol (auto-detected from filename if None)
            timeframe: Data timeframe (auto-detected from filename if None)
            market_type: Market classification (auto-detected if None)
            symbol_spec: Symbol specification (auto-loaded if None)

        Returns:
            DataPackage ready for backtesting or exploration

        Example:
            >>> loader = UnifiedDataLoader()
            >>> pkg = loader.load("data/master/EURUSD_H1_20240101_20241231.csv")
            >>> pkg.validate()
            >>> backtest_data = pkg.to_backtest_engine_format()
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect metadata from filename
        if symbol is None or timeframe is None:
            detected_symbol, detected_timeframe = self._parse_filename(filepath.name)
            symbol = symbol or detected_symbol
            timeframe = timeframe or detected_timeframe

        if self.verbose:
            print(f"Loading {symbol} {timeframe} from {filepath.name}")

        # Load CSV data
        prices = load_mt5_csv(str(filepath))

        if self.verbose:
            print(f"  Loaded {len(prices):,} bars")

        # Auto-detect market type if not provided
        if market_type is None:
            market_type = self._detect_market_type(symbol)

        # Auto-load symbol spec if not provided
        if symbol_spec is None:
            symbol_spec = self._load_symbol_spec(symbol, market_type)

        if self.verbose:
            print(f"  Market type: {market_type.value}")

        # Apply market-type-specific preprocessing
        prices, preprocessing_steps = self._preprocess(
            prices, symbol, market_type, symbol_spec
        )

        if self.verbose:
            print(f"  After preprocessing: {len(prices):,} bars")

        # Create DataPackage
        pkg = DataPackage(
            prices=prices,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type,
            symbol_spec=symbol_spec,
            source_file=str(filepath),
            preprocessing_applied=preprocessing_steps
        )

        # Validate if requested
        if self.validate:
            is_valid = pkg.validate()
            if self.verbose:
                status = "✓ PASSED" if is_valid else "✗ FAILED"
                print(f"  Validation: {status}")
                if pkg.validation_warnings:
                    print(f"    Warnings: {len(pkg.validation_warnings)}")
                if pkg.validation_errors:
                    print(f"    Errors: {len(pkg.validation_errors)}")

        # Compute physics state if requested
        if self.compute_physics:
            pkg.physics_state = self._compute_physics_state(prices, symbol_spec)
            if self.verbose:
                print(f"  Physics state: {pkg.physics_state.shape}")

        return pkg

    def _parse_filename(self, filename: str) -> tuple[str, str]:
        """
        Extract symbol and timeframe from filename.

        Supports formats:
        - BTCUSD_H1_20240101_20241231.csv
        - EURUSD_M30.csv
        - XAUUSD_D1_data.csv

        Returns:
            (symbol, timeframe) tuple
        """
        # Remove extension
        name = filename.replace('.csv', '').replace('.txt', '')

        # Pattern: SYMBOL_TIMEFRAME_...
        # Timeframe: H1, M5, M15, M30, D1, W1, etc.
        pattern = r'^([A-Z0-9\-]+)_([MHDW]\d+)'
        match = re.match(pattern, name)

        if match:
            return match.group(1), match.group(2)

        # Fallback: try to split on underscore
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[0], parts[1]

        # Last resort: return filename as symbol, unknown timeframe
        return name, "UNKNOWN"

    def _detect_market_type(self, symbol: str) -> AssetClass:
        """
        Auto-detect market type from symbol name.

        Args:
            symbol: Trading symbol (e.g., BTCUSD, EURUSD, AAPL)

        Returns:
            AssetClass enum value
        """
        symbol_upper = symbol.upper()

        # Crypto detection
        if any(x in symbol_upper for x in ['BTC', 'ETH', 'CRYPTO']):
            return AssetClass.CRYPTO

        # Indices detection
        if any(x in symbol_upper for x in ['US500', 'NAS100', 'DJ30', 'SPX', 'NDX']):
            return AssetClass.INDICES

        # Metals detection
        if any(x in symbol_upper for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER']):
            return AssetClass.METALS

        # Energy detection
        if any(x in symbol_upper for x in ['WTI', 'BRENT', 'OIL', 'NGAS', 'XBRUSD', 'XTIUSD']):
            return AssetClass.ENERGY

        # ETF detection
        if any(x in symbol_upper for x in ['ETF', 'SPY', 'QQQ', 'IWM']):
            return AssetClass.ETFS

        # Shares detection (single stock symbols)
        # Typically 1-4 letter symbols
        if len(symbol) <= 4 and symbol.isalpha():
            return AssetClass.SHARES

        # Default to forex
        return AssetClass.FOREX

    def _load_specs_from_json(self, specs_file: Optional[str] = None) -> Dict[str, SymbolSpec]:
        """
        Load symbol specifications from JSON file.

        Args:
            specs_file: Path to JSON file (auto-detected if None)

        Returns:
            Dictionary mapping symbol -> SymbolSpec
        """
        # Auto-detect specs file if not provided
        if specs_file is None:
            # Try common locations
            possible_paths = [
                Path("data/master/instrument_specs.json"),
                Path("../data/master/instrument_specs.json"),
                Path("../../data/master/instrument_specs.json"),
            ]
            for path in possible_paths:
                if path.exists():
                    specs_file = str(path)
                    break

        if specs_file is None or not Path(specs_file).exists():
            if self.verbose:
                print("  No instrument_specs.json found, using hardcoded specs")
            return {}

        # Load JSON
        try:
            with open(specs_file, 'r') as f:
                specs_data = json.load(f)

            # Convert to SymbolSpec objects
            specs = {}
            for symbol, data in specs_data.items():
                specs[symbol] = self._dict_to_spec(data)

            if self.verbose:
                print(f"  Loaded {len(specs)} specs from {specs_file}")

            return specs

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to load specs from {specs_file}: {e}")
            return {}

    def _dict_to_spec(self, data: Dict[str, Any]) -> SymbolSpec:
        """Convert JSON dictionary to SymbolSpec object."""
        from datetime import datetime

        return SymbolSpec(
            symbol=data['symbol'],
            asset_class=AssetClass(data['asset_class']),
            digits=data['digits'],
            point=data.get('point'),
            contract_size=data.get('contract_size', 100000),
            volume_min=data.get('volume_min', 0.01),
            volume_max=data.get('volume_max', 100.0),
            volume_step=data.get('volume_step', 0.01),
            margin_initial_rate_buy=data.get('margin_initial_rate_buy', 0.01),
            margin_initial_rate_sell=data.get('margin_initial_rate_sell', 0.01),
            margin_maintenance_rate_buy=data.get('margin_maintenance_rate_buy', 0.005),
            margin_maintenance_rate_sell=data.get('margin_maintenance_rate_sell', 0.005),
            margin_hedge=data.get('margin_hedge', 0.0),
            margin_currency=data.get('margin_currency', 'USD'),
            margin_mode=data.get('margin_mode', 'FOREX'),
            spread_typical=data.get('spread_typical', 0.0),
            spread_min=data.get('spread_min', 0.0),
            spread_max=data.get('spread_max', 0.0),
            commission_per_lot=data.get('commission_per_lot', 0.0),
            swap_long=data.get('swap_long', 0.0),
            swap_short=data.get('swap_short', 0.0),
            swap_type=data.get('swap_type', 'points'),
            swap_triple_day=data.get('swap_triple_day', 'wednesday'),
            profit_calc_mode=data.get('profit_calc_mode', 'FOREX'),
            trading_hours=data.get('trading_hours'),
            # Stop placement & freeze zones
            trade_stops_level=data.get('trade_stops_level', 0),
            trade_freeze_level=data.get('trade_freeze_level', 0),
            # Order execution modes
            trade_mode=data.get('trade_mode', 'FULL'),
            filling_mode=data.get('filling_mode', 'IOC'),
            order_mode=data.get('order_mode', 'MARKET_LIMIT'),
            order_gtc_mode=data.get('order_gtc_mode', 'GTC'),
            # Metadata
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None,
            source=data.get('source', 'json')
        )

    def _load_symbol_spec(self, symbol: str, market_type: AssetClass) -> SymbolSpec:
        """
        Load or generate symbol specification.

        Priority:
        1. JSON specs file (real MT5 data)
        2. Hardcoded SYMBOL_SPECS
        3. Generic fallback

        Args:
            symbol: Trading symbol
            market_type: Market classification

        Returns:
            SymbolSpec instance
        """
        # First: Try JSON cache (real MT5 data)
        if symbol in self.specs_cache:
            return self.specs_cache[symbol]

        # Second: Try hardcoded specs
        return get_symbol_spec(symbol)

    def _preprocess(
        self,
        df: pd.DataFrame,
        symbol: str,
        market_type: AssetClass,
        symbol_spec: Optional[SymbolSpec]
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply market-type-specific preprocessing.

        Args:
            df: Raw OHLCV data
            symbol: Trading symbol
            market_type: Market classification
            symbol_spec: Symbol specification

        Returns:
            (preprocessed_df, steps_applied) tuple
        """
        steps = []

        # Normalize column names to lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Market-specific preprocessing
        if market_type == AssetClass.FOREX:
            df, forex_steps = self._preprocess_forex(df)
            steps.extend(forex_steps)

        elif market_type == AssetClass.CRYPTO:
            df, crypto_steps = self._preprocess_crypto(df)
            steps.extend(crypto_steps)

        elif market_type in (AssetClass.SHARES, AssetClass.STOCK):
            df, shares_steps = self._preprocess_shares(df)
            steps.extend(shares_steps)

        elif market_type in (AssetClass.INDICES, AssetClass.INDEX):
            df, indices_steps = self._preprocess_indices(df)
            steps.extend(indices_steps)

        elif market_type in (AssetClass.METALS, AssetClass.METAL):
            df, metals_steps = self._preprocess_metals(df)
            steps.extend(metals_steps)

        elif market_type == AssetClass.ENERGY:
            df, energy_steps = self._preprocess_energy(df)
            steps.extend(energy_steps)

        elif market_type in (AssetClass.ETFS, AssetClass.ETF):
            df, etf_steps = self._preprocess_etfs(df)
            steps.extend(etf_steps)

        return df, steps

    def _preprocess_forex(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Forex-specific preprocessing."""
        steps = []

        # Remove weekend data (Sat 00:00 - Sun 21:59 UTC)
        initial_len = len(df)
        df = df[~((df.index.dayofweek == 5) | (df.index.dayofweek == 6))]

        if len(df) < initial_len:
            removed = initial_len - len(df)
            steps.append(f"Removed {removed} weekend bars")

        return df, steps

    def _preprocess_crypto(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Crypto-specific preprocessing (24/7 market)."""
        steps = []

        # Crypto trades 24/7, no weekend removal
        steps.append("24/7 market - no weekend removal")

        return df, steps

    def _preprocess_shares(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Shares/stocks-specific preprocessing."""
        steps = []

        # Remove weekend data
        initial_len = len(df)
        df = df[~((df.index.dayofweek == 5) | (df.index.dayofweek == 6))]

        if len(df) < initial_len:
            removed = initial_len - len(df)
            steps.append(f"Removed {removed} weekend bars")

        # TODO: Apply corporate actions (splits, dividends)
        # TODO: Filter to exchange hours (9:30-16:00 ET)

        return df, steps

    def _preprocess_indices(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Indices-specific preprocessing."""
        steps = []

        # Indices trade extended hours, keep weekday data only
        initial_len = len(df)
        df = df[~((df.index.dayofweek == 5) | (df.index.dayofweek == 6))]

        if len(df) < initial_len:
            removed = initial_len - len(df)
            steps.append(f"Removed {removed} weekend bars")

        return df, steps

    def _preprocess_metals(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Metals-specific preprocessing (23-hour trading)."""
        steps = []

        # Metals have 1-hour maintenance window (22:00-23:00 UTC)
        # Keep all data, but note maintenance hour
        steps.append("23-hour trading (1-hour maintenance window)")

        return df, steps

    def _preprocess_energy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Energy-specific preprocessing."""
        steps = []

        # Energy commodities trade weekdays
        initial_len = len(df)
        df = df[~((df.index.dayofweek == 5) | (df.index.dayofweek == 6))]

        if len(df) < initial_len:
            removed = initial_len - len(df)
            steps.append(f"Removed {removed} weekend bars")

        return df, steps

    def _preprocess_etfs(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """ETFs-specific preprocessing."""
        steps = []

        # ETFs follow equity market hours
        initial_len = len(df)
        df = df[~((df.index.dayofweek == 5) | (df.index.dayofweek == 6))]

        if len(df) < initial_len:
            removed = initial_len - len(df)
            steps.append(f"Removed {removed} weekend bars")

        return df, steps

    def _compute_physics_state(
        self,
        prices: pd.DataFrame,
        symbol_spec: Optional[SymbolSpec]
    ) -> Optional[pd.DataFrame]:
        """
        Compute physics state features.

        Args:
            prices: OHLCV data
            symbol_spec: Symbol specification

        Returns:
            Physics state DataFrame (64-dim features) or None if computation fails
        """
        try:
            from .physics_engine import PhysicsEngine

            if symbol_spec is None:
                # Cannot compute without symbol spec
                return None

            engine = PhysicsEngine(symbol_spec)
            physics_state = engine.compute_physics_state(prices)

            return physics_state

        except ImportError:
            # PhysicsEngine not available (missing dependencies)
            if self.verbose:
                print("  Warning: PhysicsEngine not available")
            return None

        except Exception as e:
            # Computation failed
            if self.verbose:
                print(f"  Warning: Physics state computation failed: {e}")
            return None
