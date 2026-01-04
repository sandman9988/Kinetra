#!/usr/bin/env python3
"""
MetaAPI Matrix Runner
=====================

Coordinates MetaAPI historical data downloads across broker, asset class, symbol,
and timeframe combinations. The runner reads a structured configuration, resolves
each requested matrix entry, and downloads the associated OHLCV data into the
standard Kinetra data layout using atomic persistence safeguards.

Configuration format (JSON or YAML):

{
  "history_days": 730,
  "brokers": {
    "vantage": {
      "account_id": "YOUR_ACCOUNT_ID",
      "token": "optional-broker-specific-token",
      "output_subdir": "vantage"
    }
  },
  "matrix": [
    {
      "broker": "vantage",
      "asset_class": "forex",
      "symbols": ["EURUSD", "GBPUSD"],
      "timeframes": ["1h", "4h"]
    }
  ]
}

If a broker token is omitted, METAAPI_TOKEN from the environment is used.
When no configuration file is supplied, the runner attempts to build a minimal
default matrix by reading METAAPI_ACCOUNT_ID and METAAPI_DEFAULT_SYMBOLS
(environment variable containing a comma-separated list such as "EURUSD,BTCUSD").

Example usage:
    python scripts/download/metaapi_matrix_runner.py \
        --config configs/metaapi_matrix.json \
        --output-root data/master \
        --max-concurrency 6

"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

import pandas as pd
from metaapi_cloud_sdk import MetaApi  # type: ignore

from kinetra.persistence_manager import get_persistence_manager

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixTarget:
    broker: str
    account_id: str
    token: str
    asset_class: str
    symbol: str
    timeframe: str
    output_subdir: Optional[str] = None


@dataclass
class MatrixConfig:
    targets: List[MatrixTarget]
    history_days: int
    output_root: Path
    max_concurrency: int
    dry_run: bool


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


SYMBOL_ALIASES: Dict[str, Sequence[str]] = {
    "US30": ("DJ30", "DJI30", "DOW30", "US30Cash"),
    "US500": ("SP500", "SPX500", "S&P500", "US500Cash"),
    "NAS100": ("NDX100", "USTEC", "NAS100Cash"),
    "GER40": ("DE40", "GER30", "DAX40", "DAX"),
    "UK100": ("FTSE100", "FTSE", "UKX"),
    "JP225": ("NI225", "Nikkei225", "JP225Cash"),
    "USOIL": ("WTI", "CRUDEOIL", "OIL", "WTIUSD", "USOUSD"),
    "UKOIL": ("BRENT", "BRN", "BRENTOIL", "UKOUSD"),
    "NGAS": ("NATGAS", "NATURALGAS", "NGASUSD"),
    "BTCUSD": ("BTC", "BITCOIN", "XBTUSD"),
    "ETHUSD": ("ETH", "ETHEREUM"),
    "XAUUSD": ("GOLD", "XAUXAG", "XAU"),
    "XAGUSD": ("SILVER", "XAG"),
}


def load_env() -> None:
    if load_dotenv:
        load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MetaAPI downloads across matrix combinations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to matrix configuration (JSON or YAML).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "master",
        help="Destination root directory for downloaded data.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="Override history window length (days). Overrides config value if provided.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent download tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without executing any API calls.",
    )
    return parser.parse_args()


def read_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    content = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configuration files.")
        return yaml.safe_load(content)
    return json.loads(content)


def build_default_config(output_root: Path) -> MatrixConfig:
    token = os.environ.get("METAAPI_TOKEN")
    account_id = os.environ.get("METAAPI_ACCOUNT_ID")
    if not token or not account_id:
        raise RuntimeError(
            "METAAPI_TOKEN and METAAPI_ACCOUNT_ID must be set when no configuration file is supplied."
        )
    default_symbols = os.environ.get("METAAPI_DEFAULT_SYMBOLS", "EURUSD,BTCUSD")
    symbols = [s.strip().upper() for s in default_symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("METAAPI_DEFAULT_SYMBOLS resulted in an empty symbol list.")
    targets: List[MatrixTarget] = []
    for symbol in symbols:
        asset_class = (
            "crypto" if symbol.endswith("USD") and symbol.startswith(("BTC", "ETH")) else "forex"
        )
        for timeframe in ("1h", "4h"):
            targets.append(
                MatrixTarget(
                    broker="default",
                    account_id=account_id,
                    token=token,
                    asset_class=asset_class,
                    symbol=symbol,
                    timeframe=timeframe,
                    output_subdir=None,
                )
            )
    return MatrixConfig(
        targets=targets,
        history_days=730,
        output_root=output_root,
        max_concurrency=4,
        dry_run=False,
    )


def load_matrix_config(
    args: argparse.Namespace,
) -> MatrixConfig:
    if args.config is None:
        config = build_default_config(args.output_root)
        if args.history_days:
            config.history_days = args.history_days
        if args.max_concurrency:
            config.max_concurrency = max(1, args.max_concurrency)
        config.dry_run = args.dry_run
        return config

    raw = read_config_file(args.config)
    history_days = int(raw.get("history_days", 730))
    if args.history_days:
        history_days = args.history_days
    brokers_cfg = raw.get("brokers", {})
    matrix_entries = raw.get("matrix", [])
    if not brokers_cfg or not matrix_entries:
        raise ValueError("Configuration must define both 'brokers' and 'matrix' sections.")

    targets: List[MatrixTarget] = []
    for entry in matrix_entries:
        broker_name = entry["broker"]
        broker_cfg = brokers_cfg.get(broker_name)
        if not broker_cfg:
            raise ValueError(
                f"Broker '{broker_name}' is referenced but not defined under 'brokers'."
            )
        account_id = broker_cfg.get("account_id")
        token = broker_cfg.get("token") or os.environ.get("METAAPI_TOKEN")
        if not account_id or not token:
            raise ValueError(
                f"Broker '{broker_name}' requires both 'account_id' and a token "
                "(either specified in config or via METAAPI_TOKEN)."
            )
        output_subdir = broker_cfg.get("output_subdir")
        asset_class = entry["asset_class"]
        timeframes = entry.get("timeframes", [])
        symbols = entry.get("symbols", [])
        if not timeframes or not symbols:
            raise ValueError(
                f"Matrix entry for broker '{broker_name}' must include symbols and timeframes."
            )
        for symbol in symbols:
            for timeframe in timeframes:
                targets.append(
                    MatrixTarget(
                        broker=broker_name,
                        account_id=account_id,
                        token=token,
                        asset_class=asset_class,
                        symbol=symbol.upper(),
                        timeframe=timeframe,
                        output_subdir=output_subdir,
                    )
                )

    max_concurrency = int(raw.get("max_concurrency", 4))
    if args.max_concurrency:
        max_concurrency = max(1, args.max_concurrency)

    return MatrixConfig(
        targets=targets,
        history_days=history_days,
        output_root=args.output_root,
        max_concurrency=max_concurrency,
        dry_run=args.dry_run,
    )


def timeframe_to_label(timeframe: str) -> str:
    mapping = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D1",
        "1w": "W1",
        "1mn": "MN1",
    }
    tf = timeframe.lower()
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    return mapping[tf]


def sanitize_symbol(symbol: str) -> str:
    return symbol.upper().replace(" ", "_")


def resolve_symbol(requested: str, available: Iterable[str]) -> Optional[str]:
    available_upper = {sym.upper(): sym for sym in available}
    if requested.upper() in available_upper:
        return available_upper[requested.upper()]
    aliases = SYMBOL_ALIASES.get(requested.upper(), ())
    for alias in aliases:
        if alias.upper() in available_upper:
            return available_upper[alias.upper()]
    for symbol in available_upper:
        if requested.upper() in symbol:
            return available_upper[symbol]
    return None


# ---------------------------------------------------------------------------
# MetaAPI session management
# ---------------------------------------------------------------------------


class MetaApiSession:
    def __init__(self, token: str):
        self.api = MetaApi(token)
        self._accounts: Dict[str, Any] = {}
        self._connections: Dict[str, Any] = {}
        self._symbols_cache: Dict[str, List[str]] = {}

    async def get_connection(self, account_id: str):
        if account_id not in self._accounts:
            account = await self.api.metatrader_account_api.get_account(account_id)
            if getattr(account, "state", None) != "DEPLOYED":
                await account.deploy()
            await account.wait_connected()
            self._accounts[account_id] = account
        account = self._accounts[account_id]
        if account_id not in self._connections:
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()
            self._connections[account_id] = connection
        return self._connections[account_id]

    async def list_symbols(self, account_id: str) -> List[str]:
        if account_id not in self._symbols_cache:
            connection = await self.get_connection(account_id)
            symbols = await connection.get_symbols()
            self._symbols_cache[account_id] = symbols
        return self._symbols_cache[account_id]

    async def close(self) -> None:
        for connection in self._connections.values():
            try:
                await connection.close()
            except Exception:
                pass
        self._connections.clear()
        self._accounts.clear()
        self._symbols_cache.clear()


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


@dataclass
class ComboResult:
    status: str
    broker: str
    asset_class: str
    symbol: str
    timeframe: str
    bars: int
    filepath: Optional[Path]
    reason: Optional[str]
    duration_seconds: float


class MatrixRunner:
    def __init__(self, config: MatrixConfig):
        self.config = config
        self.sessions: Dict[str, MetaApiSession] = {}
        self.pm = get_persistence_manager(
            backup_dir=str(config.output_root.parent / "backups"),
            max_backups=10,
        )

    def _get_session(self, token: str) -> MetaApiSession:
        if token not in self.sessions:
            self.sessions[token] = MetaApiSession(token)
        return self.sessions[token]

    async def close(self) -> None:
        await asyncio.gather(*(session.close() for session in self.sessions.values()))
        self.sessions.clear()

    async def run(self) -> List[ComboResult]:
        if self.config.dry_run:
            for target in self.config.targets:
                print(
                    f"[DRY-RUN] broker={target.broker} symbol={target.symbol} "
                    f"timeframe={target.timeframe} asset_class={target.asset_class}"
                )
            return []

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._run_single(target, semaphore) for target in self.config.targets]
        results = await asyncio.gather(*tasks)
        return results

    async def _run_single(self, target: MatrixTarget, semaphore: asyncio.Semaphore) -> ComboResult:
        start_wall = datetime.now(timezone.utc)
        async with semaphore:
            try:
                session = self._get_session(target.token)
                connection = await session.get_connection(target.account_id)
                available_symbols = await session.list_symbols(target.account_id)
                resolved_symbol = resolve_symbol(target.symbol, available_symbols)
                if not resolved_symbol:
                    reason = "symbol not available on broker"
                    print(
                        f"[WARN] {target.symbol} unavailable for broker {target.broker}: {reason}"
                    )
                    return ComboResult(
                        status="skipped",
                        broker=target.broker,
                        asset_class=target.asset_class,
                        symbol=target.symbol,
                        timeframe=target.timeframe,
                        bars=0,
                        filepath=None,
                        reason=reason,
                        duration_seconds=self._elapsed_seconds(start_wall),
                    )

                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=self.config.history_days)
                candles = await self._collect_candles(
                    connection, resolved_symbol, target.timeframe, start_time, end_time
                )
                if not candles:
                    reason = "no historical data returned"
                    print(f"[WARN] {target.symbol} {target.timeframe} -> {reason}")
                    return ComboResult(
                        status="skipped",
                        broker=target.broker,
                        asset_class=target.asset_class,
                        symbol=target.symbol,
                        timeframe=target.timeframe,
                        bars=0,
                        filepath=None,
                        reason=reason,
                        duration_seconds=self._elapsed_seconds(start_wall),
                    )

                df_export = self._prepare_dataframe(candles)
                timeframe_label = timeframe_to_label(target.timeframe)
                output_dir = self._destination_dir(target)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = self._build_filename(resolved_symbol, timeframe_label, df_export)
                filepath = output_dir / filename
                self.pm.atomic_save(
                    filepath=str(filepath),
                    content=df_export,
                    writer=lambda path, frame: frame.to_csv(
                        path, index=False, sep="\t", header=True
                    ),
                )
                bars = len(df_export)
                print(
                    f"[OK] {target.broker}: {resolved_symbol} {timeframe_label} -> {bars} bars saved at {filepath}"
                )
                return ComboResult(
                    status="success",
                    broker=target.broker,
                    asset_class=target.asset_class,
                    symbol=resolved_symbol,
                    timeframe=target.timeframe,
                    bars=bars,
                    filepath=filepath,
                    reason=None,
                    duration_seconds=self._elapsed_seconds(start_wall),
                )
            except Exception as exc:
                reason = str(exc)
                print(f"[ERROR] {target.broker} {target.symbol} {target.timeframe}: {reason}")
                return ComboResult(
                    status="failed",
                    broker=target.broker,
                    asset_class=target.asset_class,
                    symbol=target.symbol,
                    timeframe=target.timeframe,
                    bars=0,
                    filepath=None,
                    reason=reason,
                    duration_seconds=self._elapsed_seconds(start_wall),
                )

    @staticmethod
    def _elapsed_seconds(start_wall: datetime) -> float:
        return (datetime.now(timezone.utc) - start_wall).total_seconds()

    async def _collect_candles(
        self,
        connection: Any,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        candles: List[Dict[str, Any]] = []
        cursor = (
            start_time.astimezone(timezone.utc)
            if start_time.tzinfo
            else start_time.replace(tzinfo=timezone.utc)
        )
        final_time = (
            end_time.astimezone(timezone.utc)
            if end_time.tzinfo
            else end_time.replace(tzinfo=timezone.utc)
        )
        max_gap = 1000  # MetaAPI limit per request
        while cursor < final_time:
            batch = await connection.get_historical_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_time=cursor,
                limit=max_gap,
            )
            if not batch:
                break
            candles.extend(batch)
            last_time = pd.to_datetime(batch[-1]["time"], utc=True)
            if last_time <= cursor:
                break
            cursor = last_time + timedelta(milliseconds=1)
        return candles

    @staticmethod
    def _prepare_dataframe(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
        df_export = pd.DataFrame(
            {
                "<DATE>": df["time"].dt.strftime("%Y.%m.%d"),
                "<TIME>": df["time"].dt.strftime("%H:%M:%S"),
                "<OPEN>": df["open"],
                "<HIGH>": df["high"],
                "<LOW>": df["low"],
                "<CLOSE>": df["close"],
                "<TICKVOL>": df.get("tickVolume", df.get("volume")),
            }
        )
        return df_export

    def _destination_dir(self, target: MatrixTarget) -> Path:
        parts = [self.config.output_root, target.asset_class]
        if target.output_subdir:
            parts.append(target.output_subdir)
        parts.append(target.broker)
        return Path(*parts)

    @staticmethod
    def _build_filename(symbol: str, timeframe_label: str, df: pd.DataFrame) -> str:
        start_str = df["<DATE>"].iloc[0].replace(".", "") + df["<TIME>"].iloc[0].replace(":", "")
        end_str = df["<DATE>"].iloc[-1].replace(".", "") + df["<TIME>"].iloc[-1].replace(":", "")
        return f"{sanitize_symbol(symbol)}_{timeframe_label}_{start_str}_{end_str}.csv"


# ---------------------------------------------------------------------------
# Manifest & reporting
# ---------------------------------------------------------------------------


def write_manifest(results: List[ComboResult], output_root: Path) -> None:
    if not results:
        return
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "status": result.status,
                "broker": result.broker,
                "asset_class": result.asset_class,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "bars": result.bars,
                "filepath": str(result.filepath) if result.filepath else None,
                "reason": result.reason,
                "duration_seconds": result.duration_seconds,
            }
            for result in results
        ],
    }
    manifest_path = output_root / "metaapi_matrix_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def async_main() -> None:
    load_env()
    args = parse_args()
    config = load_matrix_config(args)
    runner = MatrixRunner(config)
    try:
        results = await runner.run()
        write_manifest(results, config.output_root)
        successes = sum(1 for r in results if r.status == "success")
        failures = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")
        print(
            f"[SUMMARY] success={successes} skipped={skipped} failed={failures} "
            f"total={len(results)}"
        )
    finally:
        await runner.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
