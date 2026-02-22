#!/bin/bash
# Launch MetaTrader 5 via Wine
export WINEPREFIX="$HOME/.wine-mt5"
export WINEARCH=win64

MT5_EXE="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"

if [ -f "$MT5_EXE" ]; then
    wine "$MT5_EXE" &
else
    echo "MetaTrader 5 not found at: $MT5_EXE"
    echo "Run 'make setup-mt5' to install MetaTrader 5 first."
    exit 1
fi
