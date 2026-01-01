# Kinetra MT5 Bridge Setup Script for Windows
# Run this in PowerShell as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Kinetra MT5 Bridge Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find Python installations
Write-Host "[1/5] Scanning for Python installations..." -ForegroundColor Yellow

$pythonPaths = @()

# Check common locations
$searchPaths = @(
    "$env:USERPROFILE\Anaconda3",
    "$env:USERPROFILE\miniconda3",
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:LOCALAPPDATA\Continuum\anaconda3",
    "C:\Anaconda3",
    "C:\miniconda3",
    "C:\Python*",
    "C:\ProgramData\Anaconda3",
    "C:\ProgramData\miniconda3"
)

foreach ($path in $searchPaths) {
    $resolved = Resolve-Path $path -ErrorAction SilentlyContinue
    if ($resolved) {
        foreach ($p in $resolved) {
            $pythonExe = Join-Path $p.Path "python.exe"
            if (Test-Path $pythonExe) {
                $pythonPaths += $pythonExe
                Write-Host "  Found: $pythonExe" -ForegroundColor Green
            }
        }
    }
}

# Also check PATH
$whereResult = where.exe python 2>$null
if ($whereResult) {
    foreach ($p in $whereResult) {
        if ($p -notin $pythonPaths) {
            $pythonPaths += $p
            Write-Host "  Found in PATH: $p" -ForegroundColor Green
        }
    }
}

# Check conda environments
$condaResult = where.exe conda 2>$null
if ($condaResult) {
    Write-Host "  Conda found: $condaResult" -ForegroundColor Green
    $condaInfo = conda info --base 2>$null
    if ($condaInfo) {
        Write-Host "  Conda base: $condaInfo" -ForegroundColor Green
    }
}

if ($pythonPaths.Count -eq 0) {
    Write-Host "  No Python installations found!" -ForegroundColor Red
    Write-Host "  Please install Python or Anaconda first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/5] Select Python installation:" -ForegroundColor Yellow
for ($i = 0; $i -lt $pythonPaths.Count; $i++) {
    Write-Host "  [$i] $($pythonPaths[$i])"
}

if ($pythonPaths.Count -eq 1) {
    $selection = 0
    Write-Host "  Auto-selected: $($pythonPaths[0])" -ForegroundColor Green
} else {
    $selection = Read-Host "Enter number (0-$($pythonPaths.Count - 1))"
}

$selectedPython = $pythonPaths[$selection]
Write-Host "  Using: $selectedPython" -ForegroundColor Green

# Get pip path
$pipPath = Join-Path (Split-Path $selectedPython) "Scripts\pip.exe"
if (-not (Test-Path $pipPath)) {
    $pipPath = "pip"
}

Write-Host ""
Write-Host "[3/5] Installing MetaTrader5 package..." -ForegroundColor Yellow
& $selectedPython -m pip install --upgrade MetaTrader5

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed to install MetaTrader5!" -ForegroundColor Red
    Write-Host "  Try: pip install MetaTrader5" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[4/5] Testing MT5 connection..." -ForegroundColor Yellow
$testScript = @"
import MetaTrader5 as mt5
if mt5.initialize():
    info = mt5.terminal_info()
    print(f'MT5 Connected: {info.name}')
    print(f'Build: {info.build}')
    print(f'Path: {info.path}')
    account = mt5.account_info()
    if account:
        print(f'Account: {account.login}')
        print(f'Server: {account.server}')
        print(f'Balance: {account.balance} {account.currency}')
    mt5.shutdown()
else:
    print(f'MT5 init failed: {mt5.last_error()}')
    print('Make sure MT5 terminal is running and logged in!')
"@

& $selectedPython -c $testScript

Write-Host ""
Write-Host "[5/5] Creating bridge server script..." -ForegroundColor Yellow

# Get WSL IP
$wslIP = (wsl hostname -I 2>$null).Trim().Split()[0]
if (-not $wslIP) {
    $wslIP = "localhost"
}

$bridgeScript = @"
#!/usr/bin/env python3
"""
Kinetra MT5 Bridge Server
Runs on Windows, serves data to WSL2/Linux clients
"""

import socket
import json
import threading
import time
from datetime import datetime
import MetaTrader5 as mt5

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5555

def get_symbol_info(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return {'error': f'Symbol {symbol} not found'}
    return {
        'symbol': symbol,
        'bid': info.bid,
        'ask': info.ask,
        'spread': info.spread,
        'point': info.point,
        'digits': info.digits,
        'trade_contract_size': info.trade_contract_size,
        'volume_min': info.volume_min,
        'volume_max': info.volume_max,
        'volume_step': info.volume_step,
        'swap_long': info.swap_long,
        'swap_short': info.swap_short,
        'margin_initial': info.margin_initial,
    }

def get_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {'error': f'No tick for {symbol}'}
    return {
        'symbol': symbol,
        'time': tick.time,
        'bid': tick.bid,
        'ask': tick.ask,
        'last': tick.last,
        'volume': tick.volume,
        'flags': tick.flags,
    }

def get_account_info():
    info = mt5.account_info()
    if info is None:
        return {'error': 'No account info'}
    return {
        'login': info.login,
        'server': info.server,
        'balance': info.balance,
        'equity': info.equity,
        'margin': info.margin,
        'margin_free': info.margin_free,
        'margin_level': info.margin_level,
        'currency': info.currency,
        'leverage': info.leverage,
    }

def get_rates(symbol, timeframe, count):
    tf_map = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1,
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None:
        return {'error': f'No rates for {symbol}'}
    return [{
        'time': int(r[0]),
        'open': float(r[1]),
        'high': float(r[2]),
        'low': float(r[3]),
        'close': float(r[4]),
        'tick_volume': int(r[5]),
        'spread': int(r[6]),
        'real_volume': int(r[7]),
    } for r in rates]

def handle_client(conn, addr):
    print(f'Client connected: {addr}')
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            try:
                request = json.loads(data.decode())
                cmd = request.get('cmd', '')

                if cmd == 'ping':
                    response = {'status': 'ok', 'time': time.time()}
                elif cmd == 'symbol_info':
                    response = get_symbol_info(request.get('symbol', 'EURUSD'))
                elif cmd == 'tick':
                    response = get_tick(request.get('symbol', 'EURUSD'))
                elif cmd == 'account':
                    response = get_account_info()
                elif cmd == 'rates':
                    response = get_rates(
                        request.get('symbol', 'EURUSD'),
                        request.get('timeframe', 'H1'),
                        request.get('count', 100)
                    )
                elif cmd == 'symbols':
                    symbols = mt5.symbols_get()
                    response = [s.name for s in symbols] if symbols else []
                else:
                    response = {'error': f'Unknown command: {cmd}'}

                conn.sendall(json.dumps(response).encode() + b'\\n')

            except json.JSONDecodeError:
                conn.sendall(json.dumps({'error': 'Invalid JSON'}).encode() + b'\\n')

    except Exception as e:
        print(f'Error with {addr}: {e}')
    finally:
        conn.close()
        print(f'Client disconnected: {addr}')

def main():
    if not mt5.initialize():
        print(f'MT5 initialization failed: {mt5.last_error()}')
        print('Make sure MetaTrader 5 is running and logged in!')
        return

    print(f'MT5 initialized: {mt5.terminal_info().name}')
    account = mt5.account_info()
    if account:
        print(f'Account: {account.login} @ {account.server}')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    print(f'Bridge server listening on {HOST}:{PORT}')
    print(f'From WSL2, connect to: {wslIP}:{PORT}' if '{wslIP}' != 'localhost' else '')
    print('Press Ctrl+C to stop')

    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print('\\nShutting down...')
    finally:
        server.close()
        mt5.shutdown()

if __name__ == '__main__':
    main()
"@

$bridgePath = "$env:USERPROFILE\kinetra_mt5_bridge.py"
$bridgeScript | Out-File -FilePath $bridgePath -Encoding UTF8
Write-Host "  Created: $bridgePath" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start the bridge server:" -ForegroundColor Yellow
Write-Host "  1. Make sure MT5 terminal is running and logged in"
Write-Host "  2. Run: $selectedPython $bridgePath"
Write-Host ""
Write-Host "From WSL2/Linux, connect with:" -ForegroundColor Yellow
Write-Host "  from kinetra import MT5Bridge"
Write-Host "  bridge = MT5Bridge(mode='bridge', bridge_host='$(hostname).local', bridge_port=5555)"
Write-Host ""
Write-Host "Windows Firewall: Allow Python through firewall when prompted" -ForegroundColor Yellow
