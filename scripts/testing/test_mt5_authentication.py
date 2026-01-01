"""
MT5 Terminal Authentication & Data Download Test
================================================

Complete test of MT5 terminal connection including:
1. Authentication/login to trading account
2. Terminal info verification
3. Account info retrieval
4. Symbol data download
5. Data validation
6. Save to CSV

This demonstrates the FULL pipeline from authentication to data download.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("❌ MetaTrader5 package not installed")
    print("   Install with: pip install MetaTrader5")
    sys.exit(1)


def test_mt5_authentication(
    login: int = None,
    password: str = None,
    server: str = None,
    terminal_path: str = None,
):
    """
    Test complete MT5 authentication and data download pipeline.

    Args:
        login: MT5 account login number
        password: MT5 account password
        server: MT5 server name (e.g., "VantageInternational-Demo")
        terminal_path: Path to MT5 terminal executable (for Wine)
    """
    print("\n" + "="*80)
    print(" "*20 + "MT5 AUTHENTICATION & DATA DOWNLOAD TEST")
    print("="*80)

    # ===========================
    # STEP 1: INITIALIZE MT5
    # ===========================
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING MT5 TERMINAL")
    print("="*80)

    if terminal_path:
        print(f"\n[Attempting to initialize with custom path]")
        print(f"  Path: {terminal_path}")
        initialized = mt5.initialize(path=terminal_path)
    else:
        print(f"\n[Attempting to initialize with default path]")
        initialized = mt5.initialize()

    if not initialized:
        error = mt5.last_error()
        print(f"\n❌ MT5 initialization failed")
        print(f"   Error code: {error[0]}")
        print(f"   Error message: {error[1]}")
        print(f"\n   Possible causes:")
        print(f"   - MT5 terminal is not running")
        print(f"   - MT5 is not installed")
        print(f"   - Incorrect terminal path (if using Wine)")
        print(f"   - Permission issues")
        return False

    print(f"✅ MT5 terminal initialized successfully")

    # ===========================
    # STEP 2: GET TERMINAL INFO
    # ===========================
    print("\n" + "="*80)
    print("STEP 2: GETTING TERMINAL INFORMATION")
    print("="*80)

    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("❌ Failed to get terminal info")
        mt5.shutdown()
        return False

    print(f"\n✅ Terminal Information:")
    print(f"  Company:        {terminal_info.company}")
    print(f"  Name:           {terminal_info.name}")
    print(f"  Path:           {terminal_info.path}")
    print(f"  Data Path:      {terminal_info.data_path}")
    print(f"  Build:          {terminal_info.build}")
    print(f"  Connected:      {terminal_info.connected}")
    print(f"  Trade Allowed:  {terminal_info.trade_allowed}")

    # ===========================
    # STEP 3: LOGIN (if credentials provided)
    # ===========================
    if login and password and server:
        print("\n" + "="*80)
        print("STEP 3: LOGGING IN TO TRADING ACCOUNT")
        print("="*80)

        print(f"\n[Attempting login]")
        print(f"  Account: {login}")
        print(f"  Server:  {server}")

        authorized = mt5.login(login=login, password=password, server=server)

        if not authorized:
            error = mt5.last_error()
            print(f"\n❌ Login failed")
            print(f"   Error code: {error[0]}")
            print(f"   Error message: {error[1]}")
            print(f"\n   Possible causes:")
            print(f"   - Incorrect login credentials")
            print(f"   - Incorrect server name")
            print(f"   - Account is locked/disabled")
            print(f"   - Network connectivity issues")
            print(f"\n⚠️  Continuing with current account...")
        else:
            print(f"✅ Login successful")
    else:
        print("\n" + "="*80)
        print("STEP 3: USING CURRENT TERMINAL ACCOUNT")
        print("="*80)
        print("\n⚠️  No login credentials provided")
        print("   Using currently logged-in account from terminal")

    # ===========================
    # STEP 4: GET ACCOUNT INFO
    # ===========================
    print("\n" + "="*80)
    print("STEP 4: RETRIEVING ACCOUNT INFORMATION")
    print("="*80)

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info")
        print("   Make sure you're logged in to a trading account")
        mt5.shutdown()
        return False

    print(f"\n✅ Account Information:")
    print(f"  Login:          {account_info.login}")
    print(f"  Server:         {account_info.server}")
    print(f"  Name:           {account_info.name}")
    print(f"  Company:        {account_info.company}")
    print(f"  Currency:       {account_info.currency}")
    print(f"  Leverage:       1:{account_info.leverage}")
    print(f"  Balance:        ${account_info.balance:,.2f}")
    print(f"  Equity:         ${account_info.equity:,.2f}")
    print(f"  Margin:         ${account_info.margin:,.2f}")
    print(f"  Free Margin:    ${account_info.margin_free:,.2f}")
    print(f"  Margin Level:   {account_info.margin_level:.2f}%" if account_info.margin_level else "  Margin Level:   N/A")
    print(f"  Trade Allowed:  {account_info.trade_allowed}")
    print(f"  Trade Expert:   {account_info.trade_expert}")

    # ===========================
    # STEP 5: LIST AVAILABLE SYMBOLS
    # ===========================
    print("\n" + "="*80)
    print("STEP 5: LISTING AVAILABLE SYMBOLS")
    print("="*80)

    symbols = mt5.symbols_get()
    if symbols is None or len(symbols) == 0:
        print("❌ No symbols available")
        mt5.shutdown()
        return False

    print(f"\n✅ Found {len(symbols)} symbols")

    # Show some common forex pairs
    forex_symbols = [s.name for s in symbols if any(pair in s.name for pair in ['EUR', 'GBP', 'USD', 'JPY', 'AUD'])]
    print(f"\n  Sample Forex symbols ({len(forex_symbols)} total):")
    for sym in sorted(forex_symbols)[:10]:
        print(f"    - {sym}")

    # Show metals
    metal_symbols = [s.name for s in symbols if any(metal in s.name for metal in ['XAU', 'XAG', 'GOLD', 'SILVER'])]
    if metal_symbols:
        print(f"\n  Metals ({len(metal_symbols)} total):")
        for sym in sorted(metal_symbols)[:5]:
            print(f"    - {sym}")

    # ===========================
    # STEP 6: DOWNLOAD DATA FOR SPECIFIC SYMBOL
    # ===========================
    print("\n" + "="*80)
    print("STEP 6: DOWNLOADING HISTORICAL DATA")
    print("="*80)

    # Test symbol - try common variations
    test_symbols = ['EURJPY+', 'EURJPY', 'EURJPYm', 'EURUSD+', 'EURUSD']
    symbol = None

    for test_sym in test_symbols:
        if mt5.symbol_select(test_sym, True):
            symbol = test_sym
            print(f"\n✅ Selected symbol: {symbol}")
            break

    if symbol is None:
        print(f"\n❌ None of the test symbols available: {test_symbols}")
        print(f"   Using first available symbol...")
        symbol = symbols[0].name
        mt5.symbol_select(symbol, True)
        print(f"✅ Selected symbol: {symbol}")

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"\n  Symbol Information:")
        print(f"    Bid:          {symbol_info.bid}")
        print(f"    Ask:          {symbol_info.ask}")
        print(f"    Spread:       {symbol_info.spread} points")
        print(f"    Digits:       {symbol_info.digits}")
        print(f"    Point:        {symbol_info.point}")
        print(f"    Volume Min:   {symbol_info.volume_min}")
        print(f"    Volume Max:   {symbol_info.volume_max}")
        print(f"    Volume Step:  {symbol_info.volume_step}")

    # Download data
    print(f"\n[Downloading data]")
    timeframe = mt5.TIMEFRAME_M15
    bars_to_download = 5000

    print(f"  Symbol:    {symbol}")
    print(f"  Timeframe: M15")
    print(f"  Bars:      {bars_to_download}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars_to_download)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"\n❌ Failed to download data")
        print(f"   Error code: {error[0]}")
        print(f"   Error message: {error[1]}")
        mt5.shutdown()
        return False

    print(f"\n✅ Downloaded {len(rates)} bars")

    # ===========================
    # STEP 7: VALIDATE DATA
    # ===========================
    print("\n" + "="*80)
    print("STEP 7: VALIDATING DOWNLOADED DATA")
    print("="*80)

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print(f"\n  Data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Validation checks
    checks_passed = 0
    checks_total = 5

    # Check 1: No missing values
    missing = df[['open', 'high', 'low', 'close', 'tick_volume']].isnull().sum().sum()
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

    # Check 3: Volume > 0
    has_volume = (df['tick_volume'] >= 0).all()
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
    # STEP 8: SAVE TO CSV
    # ===========================
    print("\n" + "="*80)
    print("STEP 8: SAVING DATA TO CSV")
    print("="*80)

    # Prepare DataFrame
    df_export = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
    df_export.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    # Create filename
    start_date = df['time'].iloc[0].strftime('%Y%m%d%H%M')
    end_date = df['time'].iloc[-1].strftime('%Y%m%d%H%M')
    filename = f"{symbol}_M15_{start_date}_{end_date}.csv"

    output_dir = project_root / "data" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    df_export.to_csv(output_file, index=False)

    print(f"\n✅ Data saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"  Rows: {len(df_export)}")

    # ===========================
    # STEP 9: DISCONNECT
    # ===========================
    print("\n" + "="*80)
    print("STEP 9: DISCONNECTING FROM MT5")
    print("="*80)

    mt5.shutdown()
    print(f"\n✅ Disconnected from MT5")

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
    print(f"\n  ✅ Authentication: {'SUCCESS' if account_info else 'N/A'}")
    print(f"  ✅ Data Download:  SUCCESS")
    print(f"  ✅ Data Validation: {checks_passed}/{checks_total} PASS")
    print(f"  ✅ Data Export:     SUCCESS")

    print("\n" + "="*80)

    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MT5 AUTHENTICATION & DATA DOWNLOAD TEST")
    print("="*80)
    print("\nThis script will:")
    print("  1. Connect to MT5 terminal")
    print("  2. Login to trading account (if credentials provided)")
    print("  3. Retrieve account information")
    print("  4. Download historical data")
    print("  5. Validate data quality")
    print("  6. Save to CSV file")
    print("\n" + "="*80)

    # Option 1: Use terminal's current login
    print("\n[Using current terminal account]")
    test_mt5_authentication()

    # Option 2: Login with credentials (uncomment and fill in)
    # print("\n[Login with specific credentials]")
    # test_mt5_authentication(
    #     login=12345678,  # Your MT5 account number
    #     password="your_password",
    #     server="VantageInternational-Demo",  # Or VantageInternational-Live
    #     terminal_path=None,  # Or path to MT5 terminal for Wine
    # )
