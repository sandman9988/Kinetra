"""
MetaAPI Authentication & Data Download Test
===========================================

Complete test of MetaAPI cloud connection including:
1. Authentication with MetaAPI token
2. Account connection
3. Account info retrieval
4. Historical data download
5. Data validation
6. Save to CSV

MetaAPI allows you to connect to MT5 without having it installed locally.
Get your API token from: https://app.metaapi.cloud/
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import metaapi
try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("❌ metaapi-cloud-sdk package not installed")
    print("   Install with: pip install metaapi-cloud-sdk")
    sys.exit(1)


async def test_metaapi_authentication(
    api_token: str,
    account_id: str = None,
):
    """
    Test complete MetaAPI authentication and data download pipeline.

    Args:
        api_token: Your MetaAPI API token (from https://app.metaapi.cloud/)
        account_id: MT5 account ID from MetaAPI (optional, will list all if not provided)
    """
    print("\n" + "="*80)
    print(" "*20 + "METAAPI AUTHENTICATION & DATA DOWNLOAD TEST")
    print("="*80)

    # ===========================
    # STEP 1: INITIALIZE METAAPI
    # ===========================
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING METAAPI CLIENT")
    print("="*80)

    try:
        api = MetaApi(api_token)
        print(f"\n✅ MetaAPI client initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize MetaAPI")
        print(f"   Error: {e}")
        print(f"\n   Make sure your API token is valid")
        print(f"   Get token from: https://app.metaapi.cloud/")
        return False

    # ===========================
    # STEP 2: LIST ACCOUNTS
    # ===========================
    print("\n" + "="*80)
    print("STEP 2: LISTING METATRADER ACCOUNTS")
    print("="*80)

    try:
        accounts = await api.metatrader_account_api.get_accounts()

        if not accounts:
            print(f"\n❌ No accounts found")
            print(f"   Add an MT5 account at: https://app.metaapi.cloud/accounts")
            return False

        print(f"\n✅ Found {len(accounts)} account(s):")

        for i, acc in enumerate(accounts, 1):
            print(f"\n  [{i}] Account ID: {acc.id}")
            print(f"      Name:        {acc.name}")
            print(f"      Login:       {acc.login}")
            print(f"      Server:      {acc.server}")
            print(f"      Platform:    {acc.platform}")
            print(f"      Type:        {acc.type}")
            print(f"      State:       {acc.state}")
            print(f"      Deployed:    {acc.connection_status if hasattr(acc, 'connection_status') else 'N/A'}")

        # Select account
        if account_id is None:
            if len(accounts) == 1:
                account = accounts[0]
                print(f"\n✅ Auto-selected account: {account.name} ({account.id})")
            else:
                print(f"\n⚠️  Multiple accounts available. Using first account.")
                account = accounts[0]
        else:
            account = next((a for a in accounts if a.id == account_id), None)
            if account is None:
                print(f"\n❌ Account ID {account_id} not found")
                return False
            print(f"\n✅ Selected account: {account.name} ({account.id})")

    except Exception as e:
        print(f"\n❌ Failed to list accounts")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 3: DEPLOY ACCOUNT
    # ===========================
    print("\n" + "="*80)
    print("STEP 3: DEPLOYING METATRADER ACCOUNT")
    print("="*80)

    try:
        print(f"\n[Deploying account: {account.name}]")

        # Deploy account if not deployed
        if account.state != 'DEPLOYED':
            print(f"  Account state: {account.state}")
            print(f"  Deploying...")
            await account.deploy()
            print(f"✅ Account deployed")
        else:
            print(f"  Account already deployed")

        # Wait until account is connected
        print(f"\n[Waiting for connection...]")
        await account.wait_connected()
        print(f"✅ Account connected")

    except Exception as e:
        print(f"\n❌ Failed to deploy account")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 4: GET CONNECTION
    # ===========================
    print("\n" + "="*80)
    print("STEP 4: ESTABLISHING RPC CONNECTION")
    print("="*80)

    try:
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()
        print(f"\n✅ RPC connection established and synchronized")

    except Exception as e:
        print(f"\n❌ Failed to establish connection")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 5: GET ACCOUNT INFO
    # ===========================
    print("\n" + "="*80)
    print("STEP 5: RETRIEVING ACCOUNT INFORMATION")
    print("="*80)

    try:
        account_info = await connection.get_account_information()

        print(f"\n✅ Account Information:")
        print(f"  Login:          {account_info.get('login', 'N/A')}")
        print(f"  Server:         {account_info.get('server', 'N/A')}")
        print(f"  Name:           {account_info.get('name', 'N/A')}")
        print(f"  Platform:       {account_info.get('platform', 'N/A')}")
        print(f"  Currency:       {account_info.get('currency', 'N/A')}")
        print(f"  Leverage:       1:{account_info.get('leverage', 'N/A')}")
        print(f"  Balance:        ${account_info.get('balance', 0):,.2f}")
        print(f"  Equity:         ${account_info.get('equity', 0):,.2f}")
        print(f"  Margin:         ${account_info.get('margin', 0):,.2f}")
        print(f"  Free Margin:    ${account_info.get('freeMargin', 0):,.2f}")
        print(f"  Margin Level:   {account_info.get('marginLevel', 'N/A')}")
        print(f"  Trade Allowed:  {account_info.get('tradeAllowed', 'N/A')}")

    except Exception as e:
        print(f"\n❌ Failed to get account info")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 6: LIST SYMBOLS
    # ===========================
    print("\n" + "="*80)
    print("STEP 6: LISTING AVAILABLE SYMBOLS")
    print("="*80)

    try:
        symbols = await connection.get_symbols()

        print(f"\n✅ Found {len(symbols)} symbols")

        # Show some common forex pairs
        forex_symbols = [s for s in symbols if any(pair in s for pair in ['EUR', 'GBP', 'USD', 'JPY', 'AUD'])]
        print(f"\n  Sample Forex symbols ({len(forex_symbols)} total):")
        for sym in sorted(forex_symbols)[:10]:
            print(f"    - {sym}")

        # Show metals
        metal_symbols = [s for s in symbols if any(metal in s for metal in ['XAU', 'XAG', 'GOLD', 'SILVER'])]
        if metal_symbols:
            print(f"\n  Metals ({len(metal_symbols)} total):")
            for sym in sorted(metal_symbols)[:5]:
                print(f"    - {sym}")

    except Exception as e:
        print(f"\n❌ Failed to list symbols")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 7: DOWNLOAD HISTORICAL DATA
    # ===========================
    print("\n" + "="*80)
    print("STEP 7: DOWNLOADING HISTORICAL DATA")
    print("="*80)

    # Select test symbol
    test_symbols = ['EURJPY+', 'EURJPY', 'EURJPYm', 'EURUSD+', 'EURUSD']
    symbol = None

    for test_sym in test_symbols:
        if test_sym in symbols:
            symbol = test_sym
            break

    if symbol is None:
        print(f"\n⚠️  None of the preferred symbols found: {test_symbols}")
        symbol = symbols[0]
        print(f"   Using first available symbol: {symbol}")
    else:
        print(f"\n✅ Selected symbol: {symbol}")

    # Get symbol info
    try:
        symbol_spec = await connection.get_symbol_specification(symbol)
        print(f"\n  Symbol Specification:")
        print(f"    Digits:       {symbol_spec.get('digits', 'N/A')}")
        print(f"    Point:        {symbol_spec.get('point', 'N/A')}")
        print(f"    Min Volume:   {symbol_spec.get('volumeMin', 'N/A')}")
        print(f"    Max Volume:   {symbol_spec.get('volumeMax', 'N/A')}")
        print(f"    Volume Step:  {symbol_spec.get('volumeStep', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️  Could not get symbol spec: {e}")

    # Download historical data
    print(f"\n[Downloading historical data]")
    print(f"  Symbol:    {symbol}")
    print(f"  Timeframe: M15")
    print(f"  Bars:      5000")

    try:
        # Get data from last 30 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)

        candles = await account.get_historical_candles(
            symbol=symbol,
            timeframe='15m',  # M15
            start_time=start_time,
            limit=5000
        )

        if not candles:
            print(f"\n❌ No data returned")
            return False

        print(f"\n✅ Downloaded {len(candles)} candles")

    except Exception as e:
        print(f"\n❌ Failed to download data")
        print(f"   Error: {e}")
        return False

    # ===========================
    # STEP 8: VALIDATE DATA
    # ===========================
    print("\n" + "="*80)
    print("STEP 8: VALIDATING DOWNLOADED DATA")
    print("="*80)

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Rename columns to standard format
    df = df.rename(columns={
        'time': 'time',
        'brokerTime': 'broker_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tickVolume': 'volume',
        'spread': 'spread',
        'realVolume': 'real_volume',
    })

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])

    print(f"\n  Data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Validation checks
    checks_passed = 0
    checks_total = 5

    # Check 1: No missing values
    missing = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
    print(f"\n  [1/5] No missing values: {'✅ PASS' if missing == 0 else f'❌ FAIL ({missing} missing)'}")
    if missing == 0:
        checks_passed += 1

    # Check 2: OHLC validity
    valid_ohlc = (
        (df['high'] >= df['low']).all() and
        (df['high'] >= df['open']).all() and
        (df['high'] >= df['close']).all() and
        (df['low'] <= df['open']).all() and
        (df['low'] <= df['close']).all()
    )
    print(f"  [2/5] OHLC validity: {'✅ PASS' if valid_ohlc else '❌ FAIL'}")
    if valid_ohlc:
        checks_passed += 1

    # Check 3: Volume >= 0
    has_volume = (df['volume'] >= 0).all()
    print(f"  [3/5] Volume >= 0: {'✅ PASS' if has_volume else '❌ FAIL'}")
    if has_volume:
        checks_passed += 1

    # Check 4: No duplicates
    duplicates = df['time'].duplicated().sum()
    no_dupes = duplicates == 0
    print(f"  [4/5] No duplicate timestamps: {'✅ PASS' if no_dupes else f'❌ FAIL ({duplicates} duplicates)'}")
    if no_dupes:
        checks_passed += 1

    # Check 5: Chronological order
    chronological = (df['time'].diff().dropna() > pd.Timedelta(0)).all()
    print(f"  [5/5] Chronological order: {'✅ PASS' if chronological else '❌ FAIL'}")
    if chronological:
        checks_passed += 1

    print(f"\n  Validation result: {checks_passed}/{checks_total} checks passed")

    # ===========================
    # STEP 9: SAVE TO CSV
    # ===========================
    print("\n" + "="*80)
    print("STEP 9: SAVING DATA TO CSV")
    print("="*80)

    # Prepare DataFrame
    df_export = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Create filename
    start_date = df['time'].iloc[0].strftime('%Y%m%d%H%M')
    end_date = df['time'].iloc[-1].strftime('%Y%m%d%H%M')
    filename = f"{symbol}_M15_{start_date}_{end_date}_metaapi.csv"

    output_dir = project_root / "data" / "metaapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    df_export.to_csv(output_file, index=False)

    print(f"\n✅ Data saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"  Rows: {len(df_export)}")

    # ===========================
    # STEP 10: DISCONNECT
    # ===========================
    print("\n" + "="*80)
    print("STEP 10: DISCONNECTING FROM METAAPI")
    print("="*80)

    try:
        await connection.close()
        print(f"\n✅ Disconnected from MetaAPI")
    except Exception as e:
        print(f"\n⚠️  Disconnect warning: {e}")

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print("\n" + "="*80)
    print(" "*25 + "TEST SUMMARY")
    print("="*80)

    print(f"\n✅ All steps completed successfully!")
    print(f"\n  📊 Data Statistics:")
    print(f"    Symbol:     {symbol}")
    print(f"    Timeframe:  M15")
    print(f"    Bars:       {len(df_export)}")
    print(f"    Period:     {df_export['time'].iloc[0]} to {df_export['time'].iloc[-1]}")
    print(f"    File:       {output_file}")
    print(f"\n  ✅ Authentication:  SUCCESS")
    print(f"  ✅ Data Download:   SUCCESS")
    print(f"  ✅ Data Validation: {checks_passed}/{checks_total} PASS")
    print(f"  ✅ Data Export:     SUCCESS")

    print("\n" + "="*80)

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("METAAPI AUTHENTICATION & DATA DOWNLOAD TEST")
    print("="*80)
    print("\nThis script will:")
    print("  1. Connect to MetaAPI cloud service")
    print("  2. List your MT5 accounts")
    print("  3. Deploy and connect to account")
    print("  4. Retrieve account information")
    print("  5. Download historical data")
    print("  6. Validate data quality")
    print("  7. Save to CSV file")
    print("\n" + "="*80)
    print("\n⚠️  You need to provide your MetaAPI API token")
    print("   Get it from: https://app.metaapi.cloud/")
    print("\n" + "="*80)

    # IMPORTANT: Set your MetaAPI token here
    API_TOKEN = None  # Get from https://app.metaapi.cloud/
    ACCOUNT_ID = None  # Optional: specific account ID

    if API_TOKEN is None:
        print("\n❌ ERROR: API_TOKEN not set")
        print("\n   Please edit this script and set your MetaAPI token:")
        print("   API_TOKEN = 'your_token_here'")
        print("\n   Get token from: https://app.metaapi.cloud/")
        sys.exit(1)

    # Run test
    asyncio.run(test_metaapi_authentication(
        api_token=API_TOKEN,
        account_id=ACCOUNT_ID,
    ))
