"""
Fetch Broker Symbol Specification from MetaAPI
================================================

Gets real-time symbol specs directly from broker via MetaAPI.
No hardcoded values - everything comes from the API.

This eliminates broker-specific hardcoding and ensures accuracy.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.market_microstructure import AssetClass, SymbolSpec


async def fetch_symbol_spec_from_metaapi(
    api_token: str, account_id: str, symbol: str
) -> SymbolSpec:
    """
    Fetch symbol specification from MetaAPI.

    Args:
        api_token: MetaAPI authentication token
        account_id: MT5 account ID
        symbol: Symbol to fetch (e.g., "EURJPY+", "XAUUSD+")

    Returns:
        SymbolSpec with real broker specifications

    Example:
        spec = await fetch_symbol_spec_from_metaapi(
            api_token="your_token",
            account_id="account_id",
            symbol="EURJPY+"
        )
    """
    from metaapi_cloud_sdk import MetaApi

    print(f"\n{'=' * 80}")
    print(f"FETCHING SYMBOL SPEC FROM METAAPI")
    print(f"{'=' * 80}")
    print(f"\n  Symbol: {symbol}")
    print(f"  Account: {account_id}")

    # Initialize MetaAPI
    api = MetaApi(api_token)

    try:
        # Get account
        account = await api.metatrader_account_api.get_account(account_id)

        if account.state != "DEPLOYED":
            print(f"\n⚠️  Account not deployed. Deploying...")
            await account.deploy()
            await account.wait_deployed()

        print(f"\n✅ Account deployed: {account.name}")

        # Connect
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()

        print(f"✅ Connected to {account.server}")

        # Get symbol specification
        print(f"\n[Fetching symbol specification...]")
        symbol_spec_raw = await connection.get_symbol_specification(symbol)

        if not symbol_spec_raw:
            raise ValueError(f"Symbol {symbol} not found on {account.server}")

        print(f"✅ Symbol specification retrieved")

        # Get account info for currency
        account_info = await connection.get_account_information()
        account_currency = account_info.get("currency", "USD")

        # Parse asset class from symbol type
        symbol_type = symbol_spec_raw.get("type", "").lower()
        if "forex" in symbol_type or "currency" in symbol_type:
            asset_class = AssetClass.FOREX
        elif "metal" in symbol_type or "gold" in symbol.lower() or "silver" in symbol.lower():
            asset_class = AssetClass.METAL
        elif "crypto" in symbol_type:
            asset_class = AssetClass.CRYPTO
        elif "stock" in symbol_type or "equity" in symbol_type:
            asset_class = AssetClass.STOCK
        elif "index" in symbol_type or "indices" in symbol_type:
            asset_class = AssetClass.INDEX
        elif "energy" in symbol_type or "oil" in symbol.lower():
            asset_class = AssetClass.ENERGY
        else:
            asset_class = AssetClass.FOREX  # Default

        # Get current market data for spread
        tick = await connection.get_symbol_price(symbol)
        current_spread = 0
        if tick:
            bid = tick.get("bid", 0)
            ask = tick.get("ask", 0)
            point = symbol_spec_raw.get("point", 0.00001)
            if point > 0:
                current_spread = int((ask - bid) / point)

        # Build SymbolSpec
        spec = SymbolSpec(
            symbol=symbol,
            asset_class=asset_class,
            digits=symbol_spec_raw.get("digits", 5),
            point=symbol_spec_raw.get("point", 0.00001),
            contract_size=symbol_spec_raw.get("contractSize", 100000),
            volume_min=symbol_spec_raw.get("volumeMin", 0.01),
            volume_max=symbol_spec_raw.get("volumeMax", 100.0),
            volume_step=symbol_spec_raw.get("volumeStep", 0.01),
            # Trading costs from API
            spread_typical=current_spread,  # Current spread as typical
            # Commission: Not available in MetaAPI symbol spec
            # Must be configured per broker or queried from trade history
            commission_per_lot=0.0,  # Set externally or from broker config
            # Swap from API
            swap_long=symbol_spec_raw.get("swapLong", 0.0),
            swap_short=symbol_spec_raw.get("swapShort", 0.0),
            swap_triple_day="wednesday",  # Standard, could parse from broker
            # MT5 constraints from API
            trade_freeze_level=symbol_spec_raw.get("freezeLevel", 0),
            trade_stops_level=symbol_spec_raw.get("stopsLevel", 0),
        )

        print(f"\n✅ SymbolSpec created:")
        print(f"  Asset Class:     {spec.asset_class.value}")
        print(f"  Digits:          {spec.digits}")
        print(f"  Point:           {spec.point}")
        print(f"  Contract Size:   {spec.contract_size:,.0f}")
        print(f"  Volume Min/Max:  {spec.volume_min} - {spec.volume_max}")
        print(
            f"  Spread:          {spec.spread_typical} points (~{spec.spread_typical * spec.point} price)"
        )
        print(f"  Swap Long/Short: {spec.swap_long} / {spec.swap_short} points/day")
        print(f"  Freeze Level:    {spec.trade_freeze_level}")
        print(f"  Stops Level:     {spec.trade_stops_level}")
        print(
            f"\n⚠️  Commission:      {spec.commission_per_lot} (not available from API - set manually)"
        )

        # Close connection
        connection.close()

        return spec

    except Exception as e:
        print(f"\n❌ Failed to fetch symbol spec: {e}")
        import traceback

        traceback.print_exc()
        raise


async def test_fetch_vantage_spec():
    """Test fetching Vantage EURJPY spec from MetaAPI."""

    # Get MetaAPI token from environment
    API_TOKEN = os.getenv("METAAPI_TOKEN")
    if not API_TOKEN:
        print("\n❌ ERROR: METAAPI_TOKEN environment variable not set")
        print("   Set it with: export METAAPI_TOKEN='your_token_here'")
        print("   Or add to .env file: METAAPI_TOKEN=your_token_here")
        print("   Get your token from: https://app.metaapi.cloud/token")
        return

    # Get account ID from environment
    ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")
    if not ACCOUNT_ID:
        print("\n❌ ERROR: METAAPI_ACCOUNT_ID environment variable not set")
        print("   Set it with: export METAAPI_ACCOUNT_ID='your_account_id'")
        print("   Or add to .env file: METAAPI_ACCOUNT_ID=your_account_id")
        print("   Run scripts/download/select_metaapi_account.py to find your account ID")
        return

    # Fetch spec
    spec = await fetch_symbol_spec_from_metaapi(
        api_token=API_TOKEN, account_id=ACCOUNT_ID, symbol="EURJPY+"
    )

    # Manually set commission (not available from API)
    spec.commission_per_lot = 3.0  # Vantage: $3/side = $6 round trip

    print(f"\n{'=' * 80}")
    print(f"✅ FINAL SPEC (with manual commission)")
    print(f"{'=' * 80}")
    print(f"\n{spec}")


if __name__ == "__main__":
    print(__doc__)
    asyncio.run(test_fetch_vantage_spec())
