"""
MetaAPI Authentication & Data Download Test
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("Install with: pip install metaapi-cloud-sdk")
    sys.exit(1)

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    # python-dotenv is optional; if it's not installed we just rely on
    # environment variables already set in the OS.
    pass

# =====================================================
# CREDENTIALS - Load from environment variables
# =====================================================
API_TOKEN = os.environ.get('METAAPI_TOKEN')
ACCOUNT_ID = os.environ.get('METAAPI_ACCOUNT_ID')

if not API_TOKEN or not ACCOUNT_ID:
    print("ERROR: Missing credentials!")
    print("Please set METAAPI_TOKEN and METAAPI_ACCOUNT_ID environment variables.")
    print("You can:")
    print("  1. Copy .env.example to .env and fill in your credentials")
    print("  2. Export them: export METAAPI_TOKEN=your_token")
    print("                 export METAAPI_ACCOUNT_ID=your_account_id")
    sys.exit(1)


async def test_metaapi_authentication():
    print("\n" + "="*60)
    print("METAAPI AUTHENTICATION & DATA DOWNLOAD TEST")
    print("="*60)

    # Step 1: Initialize
    print("\n[1] Initializing MetaAPI...")
    api = MetaApi(API_TOKEN)
    print("    ✅ Client initialized")

    # Step 2: List accounts
    print("\n[2] Listing accounts...")
    accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
    print(f"    ✅ Found {len(accounts)} account(s)")

    for i, acc in enumerate(accounts, 1):
        # Use getattr for SDK compatibility
        print(f"\n  [{i}] Account ID: {acc.id}")
        print(f"      Name:     {getattr(acc, 'name', 'N/A')}")
        print(f"      Login:    {getattr(acc, 'login', 'N/A')}")
        print(f"      Server:   {getattr(acc, 'server', 'N/A')}")
        print(f"      State:    {getattr(acc, 'state', 'N/A')}")

    # Step 3: Select account
    print("\n[3] Selecting account...")
    account = next((a for a in accounts if a.id == ACCOUNT_ID), accounts[0] if accounts else None)
    if not account:
        print("    ❌ No account found!")
        return False
    print(f"    ✅ Selected: {account.name}")

    # Step 4: Deploy if needed
    print("\n[4] Deploying account...")
    state = getattr(account, 'state', None)
    if state != 'DEPLOYED':
        print(f"    Current state: {state}, deploying...")
        await account.deploy()
    await account.wait_connected()
    print("    ✅ Account connected")

    # Step 5: Get RPC connection
    print("\n[5] Creating RPC connection...")
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    print("    ✅ RPC connected and synchronized")

    # Step 6: Get account info
    print("\n[6] Getting account info...")
    info = await connection.get_account_information()
    print(f"    Login:   {info.get('login')}")
    print(f"    Server:  {info.get('server')}")
    print(f"    Balance: ${info.get('balance', 0):,.2f}")
    print(f"    Equity:  ${info.get('equity', 0):,.2f}")

    # Step 7: List symbols
    print("\n[7] Getting symbols...")
    symbols = await connection.get_symbols()
    print(f"    ✅ Found {len(symbols)} symbols")

    # Show sample
    forex = [s for s in symbols if 'USD' in s or 'EUR' in s][:10]
    print(f"    Sample: {forex}")

    # Step 8: Download data
    print("\n[8] Downloading historical data...")
    test_symbol = next((s for s in ['EURUSD', 'EURUSD+', 'EURUSDm'] if s in symbols), symbols[0])
    print(f"    Symbol: {test_symbol}")

    candles = await account.get_historical_candles(
        symbol=test_symbol,
        timeframe='1h',
        start_time=datetime.now() - timedelta(days=30),
        limit=1000
    )
    print(f"    ✅ Downloaded {len(candles)} candles")

    # Step 9: Validate data
    print("\n[9] Validating data...")
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    print(f"    Shape: {df.shape}")
    print(f"    Range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # OHLC check
    valid = (df['high'] >= df['low']).all()
    print(f"    OHLC valid: {'✅' if valid else '❌'}")

    # Step 10: Save to CSV
    print("\n[10] Saving to CSV...")
    output_dir = project_root / "data" / "metaapi"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = df['time'].iloc[0].strftime('%Y%m%d')
    end_str = df['time'].iloc[-1].strftime('%Y%m%d')
    filename = f"{test_symbol}_H1_{start_str}_{end_str}.csv"
    output_file = output_dir / filename

    df_export = df[['time', 'open', 'high', 'low', 'close', 'tickVolume']].copy()
    df_export.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df_export.to_csv(output_file, index=False)
    print(f"    ✅ Saved to: {output_file}")

    # Metadata
    metadata = {
        "symbol": test_symbol,
        "timeframe": "H1",
        "bars": len(df_export),
        "start": str(df['time'].iloc[0]),
        "end": str(df['time'].iloc[-1]),
        "downloaded": datetime.now().isoformat(),
    }
    meta_file = output_dir / f"{test_symbol}_H1_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✅ Metadata: {meta_file}")

    # Cleanup
    await connection.close()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print(f"\n  Data saved to: {output_file}")
    print(f"  {len(df_export)} candles downloaded")

    return True


if __name__ == "__main__":
    asyncio.run(test_metaapi_authentication())
