"""
Fetch Broker Symbol Specification from MetaAPI
================================================

Gets real-time symbol specs directly from broker via MetaAPI.
No hardcoded values - everything comes from the API.

This eliminates broker-specific hardcoding and ensures accuracy.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.market_microstructure import SymbolSpec, AssetClass


async def fetch_symbol_spec_from_metaapi(
    api_token: str,
    account_id: str,
    symbol: str
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

    print(f"\n{'='*80}")
    print(f"FETCHING SYMBOL SPEC FROM METAAPI")
    print(f"{'='*80}")
    print(f"\n  Symbol: {symbol}")
    print(f"  Account: {account_id}")

    # Initialize MetaAPI
    api = MetaApi(api_token)

    try:
        # Get account
        account = await api.metatrader_account_api.get_account(account_id)

        if account.state != 'DEPLOYED':
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
        account_currency = account_info.get('currency', 'USD')

        # Parse asset class from symbol type
        symbol_type = symbol_spec_raw.get('type', '').lower()
        if 'forex' in symbol_type or 'currency' in symbol_type:
            asset_class = AssetClass.FOREX
        elif 'metal' in symbol_type or 'gold' in symbol.lower() or 'silver' in symbol.lower():
            asset_class = AssetClass.METAL
        elif 'crypto' in symbol_type:
            asset_class = AssetClass.CRYPTO
        elif 'stock' in symbol_type or 'equity' in symbol_type:
            asset_class = AssetClass.STOCK
        elif 'index' in symbol_type or 'indices' in symbol_type:
            asset_class = AssetClass.INDEX
        elif 'energy' in symbol_type or 'oil' in symbol.lower():
            asset_class = AssetClass.ENERGY
        else:
            asset_class = AssetClass.FOREX  # Default

        # Get current market data for spread
        tick = await connection.get_symbol_price(symbol)
        current_spread = 0
        if tick:
            bid = tick.get('bid', 0)
            ask = tick.get('ask', 0)
            point = symbol_spec_raw.get('point', 0.00001)
            if point > 0:
                current_spread = int((ask - bid) / point)

        # Build SymbolSpec
        spec = SymbolSpec(
            symbol=symbol,
            asset_class=asset_class,
            digits=symbol_spec_raw.get('digits', 5),
            point=symbol_spec_raw.get('point', 0.00001),
            contract_size=symbol_spec_raw.get('contractSize', 100000),
            volume_min=symbol_spec_raw.get('volumeMin', 0.01),
            volume_max=symbol_spec_raw.get('volumeMax', 100.0),
            volume_step=symbol_spec_raw.get('volumeStep', 0.01),

            # Trading costs from API
            spread_typical=current_spread,  # Current spread as typical

            # Commission: Not available in MetaAPI symbol spec
            # Must be configured per broker or queried from trade history
            commission_per_lot=0.0,  # Set externally or from broker config

            # Swap from API
            swap_long=symbol_spec_raw.get('swapLong', 0.0),
            swap_short=symbol_spec_raw.get('swapShort', 0.0),
            swap_triple_day="wednesday",  # Standard, could parse from broker

            # MT5 constraints from API
            trade_freeze_level=symbol_spec_raw.get('freezeLevel', 0),
            trade_stops_level=symbol_spec_raw.get('stopsLevel', 0),
        )

        print(f"\n✅ SymbolSpec created:")
        print(f"  Asset Class:     {spec.asset_class.value}")
        print(f"  Digits:          {spec.digits}")
        print(f"  Point:           {spec.point}")
        print(f"  Contract Size:   {spec.contract_size:,.0f}")
        print(f"  Volume Min/Max:  {spec.volume_min} - {spec.volume_max}")
        print(f"  Spread:          {spec.spread_typical} points (~{spec.spread_typical * spec.point} price)")
        print(f"  Swap Long/Short: {spec.swap_long} / {spec.swap_short} points/day")
        print(f"  Freeze Level:    {spec.trade_freeze_level}")
        print(f"  Stops Level:     {spec.trade_stops_level}")
        print(f"\n⚠️  Commission:      {spec.commission_per_lot} (not available from API - set manually)")

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

    # Your MetaAPI token
    API_TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJjMTdhODAwNThhOWE3OWE0NDNkZjBlOGM1NDZjZjlmMSIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiYzE3YTgwMDU4YTlhNzlhNDQzZGYwZThjNTQ2Y2Y5ZjEiLCJpYXQiOjE3NjcxMjE0MDYsImV4cCI6MTc3NDg5NzQwNn0.oHB_5iSe2nm_lWbIRTvFKDy1sXkq1xMXaROHvoJjgXAa2n8OkjJuj6bYbqAO4F_xHrEEKykjEgWN6Vfm7tMG9AU1o-XoHf3ayUMro90NLq-kUTcMBoE6GkAPugQenj3-1oySJ7nlnZdH-luqSRhpcnHob91O4kO670gzKUbWO2jCTpr_9d8ZzRHHqTQbukW-JdNQ53C0KSR7RGg50MBGr55IlyDmMsstqznsmCms7vDkbtoxRfUWssMOZ-4eKA-wtJJz47jQUAnJEDwGFWwoKweIhK_WnjJgFfJoOP7S_7rLBr6elkhQbzd5xENGJqmNj1I0CdiiQpNuDX5sLCt2PLvQ-Owll3LdBDpRGlb-rWJR4gaAPY3nZzCaMakfjRmZtQsCN9FGvOphG0b0IQAD3sQKZ2FzO08IenPWiZS90s4mP88vmafnC-lybMWWXKT8CnQu4YSgFsY-v74lJ_xGi6Ye-4nwzECrkGam9WceD5cGnk8bDchH-4WN68LAjPnKg0XxABd1AYnops89qcmzupoiM34BfaigMLYin5Ea81YgvGcSEwF8UQ070SDdGL2NptuznhMA2iCJoGwF0FN-uKA-jBQvPcyUEDUTjl3cbV9JECry7uAk_HeQKPzF2l0KQBOqENAytnNyWYwaq9lY3XsH7d5ZG35jFzeFCCdrokA"

    # Your account ID (get from MetaAPI dashboard or test_metaapi_auth.py)
    ACCOUNT_ID = None  # Set this to your account ID

    if ACCOUNT_ID is None:
        print("\n⚠️  Please set ACCOUNT_ID in this script")
        print("   Run test_metaapi_auth.py to find your account ID")
        return

    # Fetch spec
    spec = await fetch_symbol_spec_from_metaapi(
        api_token=API_TOKEN,
        account_id=ACCOUNT_ID,
        symbol="EURJPY+"
    )

    # Manually set commission (not available from API)
    spec.commission_per_lot = 3.0  # Vantage: $3/side = $6 round trip

    print(f"\n{'='*80}")
    print(f"✅ FINAL SPEC (with manual commission)")
    print(f"{'='*80}")
    print(f"\n{spec}")


if __name__ == "__main__":
    print(__doc__)
    asyncio.run(test_fetch_vantage_spec())
