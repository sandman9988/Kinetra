"""
Canonical Asset Classification for Kinetra
===========================================

Consistent asset classification for testing, regardless of broker categorization.

Problem: 
- Broker A classifies XAUUSD as "forex"
- Broker B classifies XAUUSD as "metal"
- We need XAUUSD in "metals" category EVERY TIME for consistent testing

Solution:
- Define canonical symbol→asset_class mapping
- Use broker classification for download/specs
- Use canonical classification for testing/analysis
"""

from enum import Enum
from typing import Dict, Optional
from kinetra.market_microstructure import AssetClass


# Canonical symbol mappings - THESE NEVER CHANGE regardless of broker
CANONICAL_SYMBOLS = {
    # Forex Major Pairs (G7 currencies)
    "EURUSD": AssetClass.FOREX,
    "GBPUSD": AssetClass.FOREX,
    "USDJPY": AssetClass.FOREX,
    "USDCHF": AssetClass.FOREX,
    "AUDUSD": AssetClass.FOREX,
    "USDCAD": AssetClass.FOREX,
    "NZDUSD": AssetClass.FOREX,
    
    # Forex Cross Pairs
    "EURJPY": AssetClass.FOREX,
    "EURGBP": AssetClass.FOREX,
    "EURCHF": AssetClass.FOREX,
    "GBPJPY": AssetClass.FOREX,
    "AUDJPY": AssetClass.FOREX,
    "EURAUD": AssetClass.FOREX,
    "GBPAUD": AssetClass.FOREX,
    
    # Crypto (major coins only - broker determines what's available)
    "BTCUSD": AssetClass.CRYPTO,
    "ETHUSD": AssetClass.CRYPTO,
    "BTCJPY": AssetClass.CRYPTO,
    "BTCEUR": AssetClass.CRYPTO,
    "ETHEUR": AssetClass.CRYPTO,
    "ETHJPY": AssetClass.CRYPTO,
    
    # Metals (ALWAYS metals, never forex)
    "XAUUSD": AssetClass.METAL,
    "XAGUSD": AssetClass.METAL,
    "GOLD": AssetClass.METAL,
    "SILVER": AssetClass.METAL,
    "XPTUSD": AssetClass.METAL,
    "XPDUSD": AssetClass.METAL,
    
    # Energy
    "USOIL": AssetClass.ENERGY,
    "UKOIL": AssetClass.ENERGY,
    "BRENT": AssetClass.ENERGY,
    "WTI": AssetClass.ENERGY,
    "NGAS": AssetClass.ENERGY,
    
    # Indices (US)
    "US30": AssetClass.INDEX,
    "NAS100": AssetClass.INDEX,
    "SPX500": AssetClass.INDEX,
    "US500": AssetClass.INDEX,
    "DJ30": AssetClass.INDEX,
    
    # Indices (Europe)
    "GER40": AssetClass.INDEX,
    "DAX40": AssetClass.INDEX,
    "UK100": AssetClass.INDEX,
    "FRA40": AssetClass.INDEX,
    "EU50": AssetClass.INDEX,
    
    # Indices (Asia)
    "JP225": AssetClass.INDEX,
    "HK50": AssetClass.INDEX,
    
    # Indices (Other)
    "AUS200": AssetClass.INDEX,
    "SA40": AssetClass.INDEX,
}


def get_canonical_asset_class(symbol: str) -> Optional[AssetClass]:
    """
    Get canonical asset class for a symbol.
    
    Args:
        symbol: Symbol name (e.g., "XAUUSD", "EURUSD+")
        
    Returns:
        AssetClass or None if unknown
        
    Example:
        >>> get_canonical_asset_class("XAUUSD")
        AssetClass.METAL  # ALWAYS metal, even if broker says "forex"
        
        >>> get_canonical_asset_class("EURUSD+")
        AssetClass.FOREX  # Strips suffix
    """
    # Normalize: uppercase, remove common suffixes
    normalized = symbol.upper().replace("+", "").replace("-", "").replace(".", "")
    
    # Direct lookup
    if normalized in CANONICAL_SYMBOLS:
        return CANONICAL_SYMBOLS[normalized]
    
    # Pattern matching for variants (e.g., "XAUUSD.m", "GOLD-C")
    for canonical_symbol, asset_class in CANONICAL_SYMBOLS.items():
        if canonical_symbol in normalized:
            return asset_class
    
    return None


def classify_by_pattern(symbol: str) -> AssetClass:
    """
    Fallback classification using pattern matching.
    Used when symbol not in canonical list.
    
    Args:
        symbol: Symbol name
        
    Returns:
        AssetClass (defaults to FOREX if unknown)
    """
    symbol_upper = symbol.upper()
    
    # Metals (prioritize - check before forex)
    if any(metal in symbol_upper for metal in ["XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER", "PLATINUM"]):
        return AssetClass.METAL
    
    # Crypto
    if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "DOGE"]):
        return AssetClass.CRYPTO
    
    # Energy
    if any(energy in symbol_upper for energy in ["OIL", "WTI", "BRENT", "GAS", "NGAS"]):
        return AssetClass.ENERGY
    
    # Indices
    if any(idx in symbol_upper for idx in ["NAS", "SPX", "DOW", "DAX", "FTSE", "NIKKEI", "JP225"]):
        return AssetClass.INDEX
    if any(idx in symbol_upper for idx in ["US30", "US500", "GER40", "UK100", "EU50"]):
        return AssetClass.INDEX
    
    # Forex (default for currency pairs)
    forex_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    if any(curr in symbol_upper for curr in forex_currencies):
        return AssetClass.FOREX
    
    # Default to forex
    return AssetClass.FOREX


def get_asset_class_with_fallback(symbol: str, broker_classification: Optional[AssetClass] = None) -> AssetClass:
    """
    Get asset class with canonical→pattern→broker fallback.
    
    Priority:
    1. Canonical mapping (highest priority - for testing consistency)
    2. Pattern matching (if not in canonical list)
    3. Broker classification (if provided, lowest priority)
    4. FOREX (default)
    
    Args:
        symbol: Symbol name
        broker_classification: Optional broker's classification
        
    Returns:
        AssetClass
        
    Example:
        # Gold is ALWAYS metal, even if broker says forex
        >>> get_asset_class_with_fallback("XAUUSD", AssetClass.FOREX)
        AssetClass.METAL
    """
    # 1. Try canonical
    canonical = get_canonical_asset_class(symbol)
    if canonical:
        return canonical
    
    # 2. Try pattern
    pattern_class = classify_by_pattern(symbol)
    
    # 3. Use broker if provided and pattern was default
    if broker_classification and pattern_class == AssetClass.FOREX:
        return broker_classification
    
    return pattern_class


def group_symbols_by_asset_class(symbols: list) -> Dict[AssetClass, list]:
    """
    Group symbols by canonical asset class.
    
    Args:
        symbols: List of symbol names
        
    Returns:
        Dict mapping AssetClass → list of symbols
        
    Example:
        >>> group_symbols_by_asset_class(["EURUSD", "XAUUSD", "BTCUSD"])
        {
            AssetClass.FOREX: ["EURUSD"],
            AssetClass.METAL: ["XAUUSD"],
            AssetClass.CRYPTO: ["BTCUSD"]
        }
    """
    grouped = {}
    
    for symbol in symbols:
        asset_class = get_asset_class_with_fallback(symbol)
        if asset_class not in grouped:
            grouped[asset_class] = []
        grouped[asset_class].append(symbol)
    
    return grouped


if __name__ == "__main__":
    # Test canonical mappings
    print("="*80)
    print("CANONICAL ASSET CLASSIFICATION TESTS")
    print("="*80)
    
    test_symbols = [
        "EURUSD", "GBPUSD", "USDJPY",  # Forex
        "XAUUSD", "XAGUSD", "GOLD",    # Metals (NOT forex!)
        "BTCUSD", "ETHUSD",            # Crypto
        "US30", "NAS100", "SPX500",    # Indices
        "USOIL", "UKOIL",              # Energy
    ]
    
    print("\nTesting canonical mappings:")
    for symbol in test_symbols:
        canonical = get_canonical_asset_class(symbol)
        print(f"  {symbol:10} → {canonical.value if canonical else 'unknown'}")
    
    print("\nTesting broker override (XAUUSD):")
    print(f"  Broker says: AssetClass.FOREX")
    print(f"  We classify: {get_asset_class_with_fallback('XAUUSD', AssetClass.FOREX).value}")
    print(f"  ✅ Canonical wins! Always METAL for testing consistency")
    
    print("\nGrouping test symbols:")
    grouped = group_symbols_by_asset_class(test_symbols)
    for asset_class, symbols in sorted(grouped.items(), key=lambda x: x[0].value):
        print(f"  {asset_class.value:10} ({len(symbols)}): {', '.join(symbols)}")
