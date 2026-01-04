#!/usr/bin/env python3
"""
Smart Download Menu - Interactive Data Pipeline
================================================

Features:
- Download top N symbols per asset class
- Start data prep as soon as symbols complete
- Parallel download + prep pipeline
- Progress monitoring
- Resume capability

Usage:
    python scripts/download/smart_download_menu.py
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kinetra.cpu_utils import get_cpu_info, get_optimal_concurrency, get_optimal_workers

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("❌ Install: pip install metaapi-cloud-sdk")
    sys.exit(1)

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except ImportError:
    pass

# Credentials
API_TOKEN = os.environ.get("METAAPI_TOKEN")
ACCOUNT_ID = os.environ.get("METAAPI_ACCOUNT_ID")

if not API_TOKEN or not ACCOUNT_ID:
    print("❌ Missing METAAPI_TOKEN or METAAPI_ACCOUNT_ID in .env")
    sys.exit(1)


@dataclass
class DownloadConfig:
    """Download configuration."""

    mode: str  # 'quick', 'top_n', 'all', 'custom'
    asset_classes: List[str]
    symbols_per_class: int
    timeframes: List[str]
    history_days: int
    concurrency: int
    auto_prep: bool  # Start data prep as soon as symbol completes


# Predefined asset classes with priority symbols
ASSET_CLASSES = {
    "forex": {
        "top_10": [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCHF",
            "USDCAD",
            "NZDUSD",
            "EURJPY",
            "GBPJPY",
            "EURGBP",
        ],
        "patterns": [r"^(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD){2}\+?$"],
    },
    "crypto": {
        "top_10": [
            "BTCUSD",
            "ETHUSD",
            "LTCUSD",
            "XRPUSD",
            "BCHUSD",
            "ADAUSD",
            "DOTUSD",
            "SOLUSD",
            "BNBUSD",
            "LINKUSD",
        ],
        "patterns": [r"^(BTC|ETH|LTC|XRP|BCH|ADA|DOT|SOL|BNB|LINK)(USD|EUR|JPY)\+?$"],
    },
    "indices": {
        "top_10": [
            "US30",
            "US500",
            "NAS100",
            "GER40",
            "UK100",
            "JP225",
            "AUS200",
            "EU50",
            "HK50",
            "CHINA50",
        ],
        "patterns": [r"^(US30|US500|NAS100|GER40|UK100|JP225|DJ30|SP500)"],
    },
    "metals": {
        "top_10": [
            "XAUUSD",
            "XAGUSD",
            "XPTUSD",
            "XPDUSD",
            "XAUEUR",
            "XAUAUD",
            "XAUJPY",
            "XAGEUR",
            "COPPER",
        ],
        "patterns": [r"^(XAU|XAG|XPT|XPD|GOLD|SILVER|COPPER)"],
    },
    "energy": {
        "top_10": ["USOIL", "UKOIL", "NGAS", "BRENT", "WTI", "CL", "NG"],
        "patterns": [r"^(USOIL|UKOIL|WTI|BRENT|NGAS|CL|NG)"],
    },
}


def print_header(text: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_menu():
    """Display interactive menu."""
    print_header("KINETRA SMART DOWNLOAD MENU")

    print("📊 QUICK START OPTIONS:")
    print("  1. Quick Start (Top 5 per class, 2 timeframes) - ~50 files")
    print("  2. Standard (Top 10 per class, 2 timeframes) - ~100 files")
    print("  3. Extended (Top 20 per class, 2 timeframes) - ~200 files")
    print("  4. Full Download (All available symbols) - ~900+ files")
    print()

    print("🎯 SELECTIVE OPTIONS:")
    print("  5. Forex Only (Select number of symbols)")
    print("  6. Crypto Only (Select number of symbols)")
    print("  7. Indices Only (Select number of symbols)")
    print("  8. Metals Only (Select number of symbols)")
    print("  9. Energy Only (Select number of symbols)")
    print()

    print("⚙️  ADVANCED:")
    print("  10. Custom Selection (Pick asset classes + counts)")
    print("  11. Resume Previous Download")
    print("  12. View Download Status")
    print()

    print("  0. Exit")
    print()


def get_config_from_choice(choice: str, available_symbols: List[str]) -> Optional[DownloadConfig]:
    """Generate download config based on menu choice."""

    if choice == "1":  # Quick Start
        return DownloadConfig(
            mode="quick",
            asset_classes=["forex", "crypto", "indices", "metals", "energy"],
            symbols_per_class=5,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "2":  # Standard
        return DownloadConfig(
            mode="top_n",
            asset_classes=["forex", "crypto", "indices", "metals", "energy"],
            symbols_per_class=10,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "3":  # Extended
        return DownloadConfig(
            mode="top_n",
            asset_classes=["forex", "crypto", "indices", "metals", "energy"],
            symbols_per_class=20,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "4":  # Full
        return DownloadConfig(
            mode="all",
            asset_classes=["forex", "crypto", "indices", "metals", "energy"],
            symbols_per_class=999,  # All
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "5":  # Forex only
        n = int(input("  How many forex symbols? (1-50): ").strip() or "10")
        return DownloadConfig(
            mode="top_n",
            asset_classes=["forex"],
            symbols_per_class=n,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "6":  # Crypto only
        n = int(input("  How many crypto symbols? (1-30): ").strip() or "10")
        return DownloadConfig(
            mode="top_n",
            asset_classes=["crypto"],
            symbols_per_class=n,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "7":  # Indices only
        n = int(input("  How many index symbols? (1-20): ").strip() or "10")
        return DownloadConfig(
            mode="top_n",
            asset_classes=["indices"],
            symbols_per_class=n,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "8":  # Metals only
        n = int(input("  How many metal symbols? (1-15): ").strip() or "10")
        return DownloadConfig(
            mode="top_n",
            asset_classes=["metals"],
            symbols_per_class=n,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "9":  # Energy only
        n = int(input("  How many energy symbols? (1-15): ").strip() or "10")
        return DownloadConfig(
            mode="top_n",
            asset_classes=["energy"],
            symbols_per_class=n,
            timeframes=["1h", "4h"],
            history_days=730,
            concurrency=get_optimal_concurrency(),
            auto_prep=True,
        )

    elif choice == "10":  # Custom
        return build_custom_config()

    elif choice == "11":  # Resume
        return load_resume_config()

    elif choice == "12":  # Status
        show_download_status()
        return None

    else:
        return None


def build_custom_config() -> DownloadConfig:
    """Build custom download configuration interactively."""
    print("\n📝 CUSTOM CONFIGURATION")
    print("-" * 70)

    # Select asset classes
    print("\nSelect asset classes (comma-separated):")
    print("  Available: forex, crypto, indices, metals, energy")
    print("  Example: forex,crypto,metals")
    classes_input = input("  Asset classes: ").strip().lower()
    asset_classes = [c.strip() for c in classes_input.split(",")]

    # Symbols per class
    symbols_per_class = int(input("\nSymbols per class (1-50): ").strip() or "10")

    # Timeframes
    print("\nTimeframes (comma-separated):")
    print("  Available: 1h, 4h, 1d")
    print("  Default: 1h,4h")
    tf_input = input("  Timeframes: ").strip().lower() or "1h,4h"
    timeframes = [t.strip() for t in tf_input.split(",")]

    # History
    history_days = int(input("\nHistory (days, default 730): ").strip() or "730")

    # Concurrency
    optimal = get_optimal_concurrency()
    concurrency_input = input(f"Concurrency (8-64, default auto={optimal}): ").strip()
    concurrency = int(concurrency_input) if concurrency_input else optimal

    # Auto prep
    auto_prep_input = input("Start data prep as symbols complete? (Y/n): ").strip().lower()
    auto_prep = auto_prep_input != "n"

    return DownloadConfig(
        mode="custom",
        asset_classes=asset_classes,
        symbols_per_class=symbols_per_class,
        timeframes=timeframes,
        history_days=history_days,
        concurrency=concurrency,
        auto_prep=auto_prep,
    )


def load_resume_config() -> Optional[DownloadConfig]:
    """Load configuration from previous incomplete download."""
    manifest_file = project_root / "data" / "master" / "download_manifest.json"

    if not manifest_file.exists():
        print("❌ No previous download found")
        return None

    with open(manifest_file) as f:
        manifest = json.load(f)

    print(f"\n📄 Previous download:")
    print(f"  Date: {manifest.get('downloaded_at', 'unknown')}")
    print(f"  Downloaded: {manifest.get('total_files', 0)} files")
    print(f"  Skipped: {manifest.get('skipped', 0)} files")
    print(f"  Failed: {manifest.get('failed', 0)} files")

    resume = input("\nRetry failed/skipped symbols? (Y/n): ").strip().lower()
    if resume == "n":
        return None

    # Build config from manifest
    # (Simplified - would need to reconstruct from manifest data)
    print("⚠️  Resume feature coming soon - use option 2 (Standard) for now")
    return None


def show_download_status():
    """Show current download status."""
    manifest_file = project_root / "data" / "master" / "download_manifest.json"

    if not manifest_file.exists():
        print("\n❌ No downloads found")
        return

    with open(manifest_file) as f:
        manifest = json.load(f)

    print_header("DOWNLOAD STATUS")

    print(f"Last download: {manifest.get('downloaded_at', 'unknown')}")
    print(f"Broker: {manifest.get('broker', 'unknown')}")
    print()

    print(f"📊 Results:")
    print(f"  ✅ Downloaded: {manifest.get('total_files', 0)} files")
    print(f"  ⏭️  Up to date: {manifest.get('up_to_date', 0)} files")
    print(f"  ⚠️  Skipped: {manifest.get('skipped', 0)} files")
    print(f"  ❌ Failed: {manifest.get('failed', 0)} files")
    print(f"  📈 Total bars: {manifest.get('total_bars', 0):,}")
    print()

    # Show by asset class
    symbols = manifest.get("symbols", [])
    if symbols:
        by_class = {}
        for sym in symbols:
            ac = sym.get("asset_class", "unknown")
            by_class.setdefault(ac, []).append(sym)

        print("By asset class:")
        for ac in sorted(by_class.keys()):
            syms = by_class[ac]
            bars = sum(s.get("bars", 0) for s in syms)
            print(f"  {ac}: {len(syms)} files, {bars:,} bars")

    input("\nPress Enter to continue...")


async def discover_symbols(account) -> List[str]:
    """Discover available symbols from broker."""
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    symbols = await connection.get_symbols()
    symbol_names = [s["symbol"] for s in symbols if s.get("symbol")]

    return symbol_names


def select_symbols(
    available: List[str], asset_classes: List[str], symbols_per_class: int
) -> Dict[str, List[str]]:
    """Select top N symbols per asset class."""
    import re

    selected = {}

    for asset_class in asset_classes:
        if asset_class not in ASSET_CLASSES:
            continue

        class_info = ASSET_CLASSES[asset_class]
        top_symbols = class_info.get("top_10", [])
        patterns = class_info.get("patterns", [])

        # Find matches
        matches = []

        # First, try exact matches from top list (prefer ECN +)
        for sym in top_symbols[:symbols_per_class]:
            candidates = [s for s in available if sym in s.upper()]
            if candidates:
                # Sort: ECN first (+), then shortest
                candidates.sort(key=lambda x: ("+" not in x, len(x)))
                matches.append(candidates[0])

        # If not enough, search by pattern
        if len(matches) < symbols_per_class:
            for pattern in patterns:
                for sym in available:
                    if len(matches) >= symbols_per_class:
                        break
                    if re.match(pattern, sym.upper(), re.IGNORECASE):
                        if sym not in matches:
                            matches.append(sym)

        selected[asset_class] = matches[:symbols_per_class]

    return selected


async def download_symbol(
    account,
    symbol: str,
    timeframe: str,
    asset_class: str,
    start_time,
    end_time,
    semaphore: asyncio.Semaphore,
    prep_queue: asyncio.Queue,
) -> dict:
    """Download single symbol and queue for prep."""
    async with semaphore:
        try:
            tf_label = "H1" if timeframe == "1h" else "H4" if timeframe == "4h" else "D1"
            output_dir = project_root / "data" / "master" / asset_class
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download chunks
            all_candles = []
            chunk_start = start_time

            while chunk_start < end_time:
                try:
                    candles = await account.get_historical_candles(
                        symbol=symbol, timeframe=timeframe, start_time=chunk_start, limit=1000
                    )
                except Exception as e:
                    if "not found" in str(e).lower():
                        return {
                            "status": "skipped",
                            "symbol": symbol,
                            "tf": tf_label,
                            "reason": "Symbol not found",
                        }
                    raise

                if not candles:
                    break

                all_candles.extend(candles)

                last_time = pd.to_datetime(candles[-1]["time"])
                if last_time.tzinfo is None:
                    from datetime import timezone

                    last_time = last_time.replace(tzinfo=timezone.utc)

                chunk_start = last_time + timedelta(hours=1 if timeframe == "1h" else 4)
                await asyncio.sleep(0.05)

            if len(all_candles) < 100:
                return {
                    "status": "skipped",
                    "symbol": symbol,
                    "tf": tf_label,
                    "reason": f"Only {len(all_candles)} bars",
                }

            # Save file
            df = pd.DataFrame(all_candles)
            df["time"] = pd.to_datetime(df["time"])
            df = df.drop_duplicates(subset=["time"]).sort_values("time")

            start_str = df["time"].iloc[0].strftime("%Y%m%d%H%M")
            end_str = df["time"].iloc[-1].strftime("%Y%m%d%H%M")
            filename = f"{symbol}_{tf_label}_{start_str}_{end_str}.csv"
            output_file = output_dir / filename

            df_export = pd.DataFrame(
                {
                    "<DATE>": df["time"].dt.strftime("%Y.%m.%d"),
                    "<TIME>": df["time"].dt.strftime("%H:%M:%S"),
                    "<OPEN>": df["open"],
                    "<HIGH>": df["high"],
                    "<LOW>": df["low"],
                    "<CLOSE>": df["close"],
                    "<TICKVOL>": df["tickVolume"],
                }
            )

            df_export.to_csv(output_file, sep="\t", index=False)

            # Queue for prep
            await prep_queue.put(
                {
                    "file": output_file,
                    "symbol": symbol,
                    "timeframe": tf_label,
                    "asset_class": asset_class,
                }
            )

            return {
                "status": "success",
                "symbol": symbol,
                "tf": tf_label,
                "bars": len(df),
                "file": str(output_file),
                "asset_class": asset_class,
            }

        except Exception as e:
            return {
                "status": "failed",
                "symbol": symbol,
                "tf": timeframe,
                "reason": str(e),
            }


async def data_prep_worker(prep_queue: asyncio.Queue, completed: Set[str]):
    """Background worker to prep data as files complete."""
    while True:
        item = await prep_queue.get()

        if item is None:  # Sentinel to stop
            break

        try:
            # Basic data prep (can be expanded)
            file_path = item["file"]
            symbol = item["symbol"]

            # Read file
            df = pd.read_csv(file_path, sep="\t")

            # Basic validation
            required_cols = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]
            if all(col in df.columns for col in required_cols):
                # Mark as ready for training
                completed.add(f"{symbol}_{item['timeframe']}")
                tqdm.write(f"  ✅ Prepped: {symbol} {item['timeframe']}")
            else:
                tqdm.write(f"  ⚠️ Invalid data: {symbol} {item['timeframe']}")

        except Exception as e:
            tqdm.write(f"  ❌ Prep failed: {item['symbol']} - {e}")

        finally:
            prep_queue.task_done()


async def run_download(config: DownloadConfig):
    """Execute download with configuration."""
    from datetime import timezone

    print_header("STARTING DOWNLOAD")

    print(f"Mode: {config.mode}")
    print(f"Asset classes: {', '.join(config.asset_classes)}")
    print(f"Symbols per class: {config.symbols_per_class}")
    print(f"Timeframes: {', '.join(config.timeframes)}")
    print(f"History: {config.history_days} days")

    cpu_info = get_cpu_info()
    print(f"CPU: {cpu_info.brand}")
    print(f"Concurrency: {config.concurrency} ({cpu_info.logical_cores} threads available)")
    print(
        f"Auto prep: {'✅' if config.auto_prep else '❌'} (workers: {get_optimal_workers()} adaptive)"
    )
    print()

    # Connect
    print("[1] Connecting to MetaAPI...")
    api = MetaApi(token=API_TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    print(f"    ✅ Connected: {account.login}")

    # Discover symbols
    print("\n[2] Discovering symbols...")
    available_symbols = await discover_symbols(account)
    print(f"    ✅ {len(available_symbols)} total symbols available")

    # Select symbols
    selected = select_symbols(available_symbols, config.asset_classes, config.symbols_per_class)

    print("\n    Selected symbols:")
    for asset_class, symbols in selected.items():
        print(
            f"    {asset_class}: {', '.join(symbols[:5])}"
            + (f" ... +{len(symbols) - 5}" if len(symbols) > 5 else "")
        )

    total_tasks = sum(len(syms) for syms in selected.values()) * len(config.timeframes)
    print(f"\n    Total: {total_tasks} downloads")

    # Setup
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=config.history_days)

    semaphore = asyncio.Semaphore(config.concurrency)
    prep_queue = asyncio.Queue()
    completed_prep = set()

    # Start prep worker if enabled
    prep_worker = None
    if config.auto_prep:
        prep_worker = asyncio.create_task(data_prep_worker(prep_queue, completed_prep))

    # Progress bars
    print(f"\n[3] Downloading ({start_time.date()} to {end_time.date()})...\n")

    pbar = tqdm(
        total=total_tasks,
        desc="📊 Downloads",
        unit="file",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    # Build tasks
    tasks = []
    for asset_class, symbols in selected.items():
        for symbol in symbols:
            for tf in config.timeframes:
                tasks.append(
                    download_symbol(
                        account,
                        symbol,
                        tf,
                        asset_class,
                        start_time,
                        end_time,
                        semaphore,
                        prep_queue,
                    )
                )

    # Execute
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        pbar.update(1)

    pbar.close()

    # Wait for prep to complete
    if config.auto_prep:
        print("\n[4] Waiting for data prep to complete...")
        await prep_queue.join()
        await prep_queue.put(None)  # Stop worker
        await prep_worker
        print(f"    ✅ Prepped {len(completed_prep)} files")

    # Summary
    success = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") == "failed"]

    print_header("DOWNLOAD COMPLETE")

    print(f"✅ Downloaded: {len(success)} files")
    print(f"⚠️  Skipped: {len(skipped)} files")
    print(f"❌ Failed: {len(failed)} files")

    if config.auto_prep:
        print(f"✅ Prepped: {len(completed_prep)} files ready for training")

    if success:
        total_bars = sum(s.get("bars", 0) for s in success)
        print(f"\n📊 Total bars: {total_bars:,}")

    # Save manifest
    manifest_file = project_root / "data" / "master" / "download_manifest.json"
    manifest = {
        "downloaded_at": datetime.now().isoformat(),
        "config": {
            "mode": config.mode,
            "asset_classes": config.asset_classes,
            "symbols_per_class": config.symbols_per_class,
        },
        "total_files": len(success),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_bars": sum(s.get("bars", 0) for s in success),
        "symbols": success,
        "skipped_symbols": skipped,
        "failed_symbols": failed,
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📄 Manifest: {manifest_file}")
    print("\n✅ Ready for training!")


async def main_async():
    """Main async entry point."""
    while True:
        print_menu()
        choice = input("Select option [0-12]: ").strip()

        if choice == "0":
            print("\n👋 Goodbye!\n")
            break

        # For status view, handle synchronously
        if choice == "12":
            show_download_status()
            continue

        # Get config
        config = get_config_from_choice(choice, [])

        if config is None:
            continue

        # Confirm
        print(f"\n📋 Configuration:")
        print(f"  Asset classes: {', '.join(config.asset_classes)}")
        print(f"  Symbols per class: {config.symbols_per_class}")
        print(
            f"  Total downloads: ~{len(config.asset_classes) * config.symbols_per_class * len(config.timeframes)} files"
        )
        print(f"  Auto prep: {'✅' if config.auto_prep else '❌'}")

        confirm = input("\nProceed? (Y/n): ").strip().lower()
        if confirm == "n":
            continue

        # Execute
        try:
            await run_download(config)
        except KeyboardInterrupt:
            print("\n\n⚠️  Download cancelled by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()

        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
