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
from typing import Optional, Union
import re

import pandas as pd
import numpy as np

from .data_package import DataPackage, DataFormat
from .data_utils import load_mt5_csv
from .market_microstructure import AssetClass, SymbolSpec, SYMBOL_SPECS, get_symbol_spec


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
        verbose: bool = False
    ):
        """
        Initialize data loader.

        Args:
            validate: Run data quality validation
            compute_physics: Compute physics state (requires PhysicsEngine)
            verbose: Print loading progress
        """
        self.validate = validate
        self.compute_physics = compute_physics
        self.verbose = verbose

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

    def _load_symbol_spec(self, symbol: str, market_type: AssetClass) -> SymbolSpec:
        """
        Load or generate symbol specification.

        Args:
            symbol: Trading symbol
            market_type: Market classification

        Returns:
            SymbolSpec instance
        """
        # Try to get from predefined specs
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
