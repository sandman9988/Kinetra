#!/bin/bash
set -e
echo "Installing Wine and MT5..."

# Enable 32-bit
sudo dpkg --add-architecture i386
sudo apt update

# Install Wine
sudo apt install -y wine64 wine32 winetricks

# Create Wine prefix for MT5
export WINEPREFIX="$HOME/.wine-mt5"
export WINEARCH=win64
wineboot --init

# Install components
winetricks -q vcrun2019 corefonts

# Download MT5
wget -O ~/Downloads/mt5setup.exe "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"

echo "Run: wine ~/Downloads/mt5setup.exe"
