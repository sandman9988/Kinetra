"""
MetaAPI Authentication & Data Download Test
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("❌ metaapi-cloud-sdk not installed. Run: pip install metaapi-cloud-sdk")
    sys.exit(1)

# =====================================================
# YOUR CREDENTIALS - THESE ARE THE ACTUAL VALUES!
# =====================================================
API_TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJjMTdhODAwNThhOWE3OWE0NDNkZjBlOGM1NDZjZjlmMSIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiYzE3YTgwMDU4YTlhNzlhNDQzZGYwZThjNTQ2Y2Y5ZjEiLCJpYXQiOjE3NjcxMjY5NzQsImV4cCI6MTc3NDkwMjk3NH0.MNG5qH4ufgoKivTCTuvfVywtTYgYhkIEWCLoff9F1tP3MvGLNRhHNwe2dyMppSTr5mzEFlkF1VRlpFthpq2KnOUvCATFNUM04cUYJcpcv6Arp_Pf653Lrtm1DK2Br4NZYQr9eh_ZndXIN2qm2QYSAi2W5wXovAaMkLPjs1x2J1G4ZxFM48u7xrqCci0Sri2dhCLNI6eVX9-VlfLJb4iYJqbKcS7GacodmtUHQqzKKusazLPoEe0cJmVPVj0h5OwXiWnZRH07VY9e9s3i-5BzHp9syGVDh7rU3D7IU8jCaB8oBWl6S49MW-wpY41_cdxf3eo53CN0MY3GikfZbusgO_2xAxxBfbsmMIC9l0g2TiUIuATEfMILPzcAhCjKE35AAc0JEbXw0XxBWyIZoCAcdqI2FuyyMOyddKfSQ7y7kkW_0tu5d9P8p-HUdE5FEI_rEHbfxfEy4CLI9LY_5ZycuhZwrnOyKLS_CPX4iFtdTT40eHynaeNv8ok8_h_wirm5YQuFv_YL0u0HqTqiy5Q_f-vDJVLob7et779DsBj9myILCFGg7RlwzEcxsZNGCkbNRvsCjZE7HwqQy2IjqGNo5vI8AiEfHD0c3PGfPhdKqKS5mBa7w4md-90T_Um9VnUXHZ8EnQlrVCnP8NpsfCWGQRDPMepd_D1lvL6XjxaVMfM"

ACCOUNT_ID = "e8f8c21a-32b5-40b0-9bf7-672e8ffab91f"


async def test_metaapi():
    print("\n" + "="*60)
    print("METAAPI CONNECTION TEST")
    print("="*60)

    # Step 1: Initialize
    print("\n[1] Initializing MetaAPI...")
    api = MetaApi(API_TOKEN)
    print("    ✅ Client initialized")

    # Step 2: List accounts
    print("\n[2] Listing accounts...")
    accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
    print(f"    ✅ Found {len(accounts)} account(s)")

    for acc in accounts:
        server = getattr(acc, 'server', 'N/A')
        state = getattr(acc, 'state', 'N/A')
        print(f"       - {acc.id}: {acc.name} ({server}) - {state}")

    # Step 3: Select account
    print("\n[3] Selecting account...")
    account = next((a for a in accounts if a.id == ACCOUNT_ID), accounts[0] if accounts else None)
    if not account:
        print("    ❌ No account found!")
        return
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
    print(f"    Sample: {symbols[:5]}")

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

    # Step 9: Save to CSV
    print("\n[9] Saving to CSV...")
    df = pd.DataFrame(candles)
    output_dir = project_root / "data" / "metaapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{test_symbol}_H1_test.csv"
    df.to_csv(output_file, index=False)
    print(f"    ✅ Saved to: {output_file}")

    # Cleanup
    await connection.close()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_metaapi())
