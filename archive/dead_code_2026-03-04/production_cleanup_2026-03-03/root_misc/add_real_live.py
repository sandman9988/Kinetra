#!/usr/bin/env python3
"""Add REAL live trading stage."""

import sys
from pathlib import Path

root = Path(__file__).parent

with open(root / "scripts/renko_engine.py", "r") as f:
    content = f.read()

# 1. Update live stage handler to call real trading
old_live_else = """        else:
            # REAL live trading - not implemented yet
            click.secho("\\n❌ LIVE trading (REAL orders) not yet implemented\\n", fg="red", bold=True)
            click.echo("  Use --dry-run for paper trading with live data:")
            click.echo(f"    python scripts/renko_engine.py {symbol} --stage live --dry-run")
            click.echo()
            click.secho("  ⚠️  REAL order execution requires additional safety systems", fg="yellow")
            raise SystemExit(1)"""

new_live_else = """        else:
            # REAL live trading with REAL orders
            click.secho("\\n⚠️  WARNING: REAL MONEY AT RISK ⚠️", fg="red", bold=True)
            click.echo("  REAL orders WILL be submitted to the broker")
            click.echo()

            if not stage_live_real(
                symbol,
                live_size=live_size,
                sizing_mode="static",
                stop_bricks=1.0,
            ):
                click.secho("❌ Live trading failed", fg="red")
                raise SystemExit(1)"""

if old_live_else in content:
    content = content.replace(old_live_else, new_live_else)
    print("✓ Updated live stage handler")
else:
    print("✗ Could not find live stage handler")
    sys.exit(1)

# 2. Add stage_live_real function before stage_live_dryrun
insert_marker = "def stage_live_dryrun("

real_live_func = '''

def stage_live_real(
    symbol: str,
    live_size: str = "micro",
    sizing_mode: str = "static",
    stop_bricks: float = 1.0,
) -> bool:
    """REAL live trading with REAL orders via cTrader.

    ⚠️  WARNING: This submits REAL orders to the broker.
    """
    import threading
    import time
    from kinetra.connectors.ctrader_connector import build_connector
    from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, CTraderOrderDispatcher

    click.echo(f"\\n{'=' * 60}")
    click.secho("STAGE 5: LIVE TRADING (REAL ORDERS)", fg="red", bold=True)
    click.echo(f"{'=' * 60}")

    # Lot size mapping
    LOT_CEILINGS = {"micro": 0.01, "small": 0.10, "full": 50.0}
    lot_ceiling = LOT_CEILINGS.get(live_size, 0.01)

    # Load DSP profile
    dsp_dir = get_data_path(symbol)
    dsp_file = dsp_dir / "dsp_profile.json"
    if not dsp_file.exists():
        click.secho(f"❌ No DSP profile found", fg="red")
        return False

    with open(dsp_file) as f:
        dsp = json.load(f)

    # Load broker spec
    from kinetra.friction_cost import load_spec
    spec = load_spec(symbol)

    # Build config
    cfg, _ = _build_engine_config(symbol, dsp, sizing_mode=sizing_mode, lot_ceiling=lot_ceiling)
    cfg.stop_bricks = stop_bricks

    # Print config
    click.echo()
    _print_system_spec(cfg, spec, dsp)

    click.echo()
    click.secho("=" * 60, fg="red")
    click.secho("  LIVE TRADING CONFIGURATION", fg="red", bold=True)
    click.secho("=" * 60, fg="red")
    click.echo(f"  Symbol:        {symbol}")
    click.echo(f"  Lot Ceiling:   {lot_ceiling:.2f} lots ({live_size})")
    click.echo(f"  Stop Loss:     {cfg.stop_bricks:.1f} bricks")
    click.echo()

    # Connect
    click.secho("Connecting to cTrader...", fg="yellow")
    try:
        connector = build_connector(timeout_s=30.0)
        click.secho("✅ Connected", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed: {e}", fg="red")
        return False

    # Preflight
    click.echo()
    click.secho("Running preflight checks...", fg="yellow", bold=True)
    if not _run_preflight_checks(connector, symbol):
        click.secho("❌ Preflight failed - aborting", fg="red")
        try:
            connector.stop()
        except Exception:
            pass
        return False
    click.secho("✅ Preflight passed", fg="green")
    click.echo()

    # Suppress verbose logs
    import logging
    logging.getLogger('kinetra.connectors').setLevel(logging.WARNING)
    logging.getLogger('kinetra.dns_hardening').setLevel(logging.WARNING)
    logging.getLogger('kinetra.renko.ctrader_dispatcher').setLevel(logging.WARNING)

    # Create dispatcher with REAL orders
    bar_provider = CTraderBarProvider(connector)
    live_dispatcher = CTraderOrderDispatcher(connector, bar_provider)

    # Create engine
    engine = RenkoEngine(cfg, spec=spec, quiet_mode=True)

    # Dashboard
    stats_lock = threading.Lock()
    bar_count = [0]
    stop_event = threading.Event()

    def update_dashboard():
        last_bar = 0
        while not stop_event.is_set():
            time.sleep(2)
            with stats_lock:
                current_bars = len(getattr(engine, '_dir_deque', []))
                if current_bars > last_bar:
                    bar_count[0] = current_bars
                    last_bar = current_bars
                    try:
                        snapshot = connector.get_account_snapshot()
                        acct_info = {
                            'broker': snapshot.get('broker_name', 'Pepperstone'),
                            'account_id': snapshot.get('account_id', connector.credentials.account_id),
                            'account_type': snapshot.get('account_type', 'ECN'),
                            'environment': connector.credentials.environment,
                            'balance': snapshot.get('balance', engine._live_equity),
                        }
                    except Exception:
                        acct_info = {
                            'broker': 'Pepperstone',
                            'account_id': connector.credentials.account_id,
                            'account_type': 'ECN',
                            'environment': connector.credentials.environment,
                            'balance': engine._live_equity,
                        }
                    _print_dashboard(engine, symbol, "LIVE", current_bars, acct_info)

    dashboard_thread = threading.Thread(target=update_dashboard, daemon=True)
    dashboard_thread.start()

    click.secho("Starting LIVE trading...", fg="red", bold=True)
    click.echo("REAL orders will be submitted")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    try:
        results = engine.run(
            bar_provider=bar_provider,
            dispatcher=live_dispatcher,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        click.echo()
        click.secho("Stopping live trading...", fg="yellow")
        stop_event.set()
        results = engine._make_results()
    finally:
        stop_event.set()
        click.echo()
        click.secho("Closing connections...", fg="yellow")
        try:
            bar_provider.stop()
            connector.stop()
        except Exception as e:
            click.secho(f"  Warning: {e}", fg="yellow")

        # Final stats
        summary = results.get("summary", {})
        trades = results.get("trades", [])

        click.echo()
        click.secho("=" * 60, fg="cyan")
        click.secho("  LIVE TRADING RESULTS", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
        _print_stats(summary, symbol, "LIVE", trades=trades)

        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        net_pnl = summary.get("net_usd", 0.0)

        click.echo()
        if n_trades > 0:
            click.secho(f"  Trades:  {n_trades}", fg="cyan")
            click.secho(f"  Net P&L: ${net_pnl:,.2f}", fg="green" if net_pnl > 0 else "red")
            click.secho(f"  Omega:   {omega:.3f}", fg="green" if omega >= 1.5 else "yellow")
        else:
            click.secho("  No trades executed", fg="yellow")

        click.echo()
        click.secho("⚠️  Verify positions in cTrader terminal", fg="yellow")

    return True


'''

if insert_marker in content:
    content = content.replace(insert_marker, real_live_func + "\n" + insert_marker)
    print("✓ Added stage_live_real function")
else:
    print("✗ Could not find insertion point")
    sys.exit(1)

with open(root / "scripts/renko_engine.py", "w") as f:
    f.write(content)

print("✓ REAL live trading implemented")
