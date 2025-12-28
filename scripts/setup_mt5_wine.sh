#!/bin/bash
# MetaTrader 5 Installation via Wine for Pop!_OS / Ubuntu
# Run with: chmod +x setup_mt5_wine.sh && ./setup_mt5_wine.sh

set -e

echo "=========================================="
echo "  MetaTrader 5 Installation via Wine"
echo "  Pop!_OS / Ubuntu 22.04+"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

# ==========================================
# STEP 1: Enable 32-bit Architecture
# ==========================================
echo ""
echo "Step 1: Enabling 32-bit architecture..."

sudo dpkg --add-architecture i386
print_status "32-bit architecture enabled"

# ==========================================
# STEP 2: Add WineHQ Repository
# ==========================================
echo ""
echo "Step 2: Adding WineHQ repository..."

# Download and add the WineHQ repository key
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key

# Detect Ubuntu/Pop!_OS version
. /etc/os-release
if [[ "$ID" == "pop" ]]; then
    # Pop!_OS is based on Ubuntu
    UBUNTU_CODENAME=$(grep UBUNTU_CODENAME /etc/os-release | cut -d= -f2)
else
    UBUNTU_CODENAME=$VERSION_CODENAME
fi

print_info "Detected: $ID $VERSION_ID (Ubuntu codename: $UBUNTU_CODENAME)"

# Add sources list
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/${UBUNTU_CODENAME}/winehq-${UBUNTU_CODENAME}.sources

sudo apt update
print_status "WineHQ repository added"

# ==========================================
# STEP 3: Install Wine
# ==========================================
echo ""
echo "Step 3: Installing Wine Staging..."

# Install Wine Staging (best compatibility with MT5)
sudo apt install -y --install-recommends winehq-staging

# Install additional Wine tools
sudo apt install -y \
    winetricks \
    cabextract \
    winbind

print_status "Wine installed: $(wine --version)"

# ==========================================
# STEP 4: Configure Wine Prefix for MT5
# ==========================================
echo ""
echo "Step 4: Configuring Wine prefix for MT5..."

export WINEPREFIX="$HOME/.wine-mt5"
export WINEARCH=win64

# Create fresh Wine prefix
if [ -d "$WINEPREFIX" ]; then
    print_warning "Wine prefix exists. Backing up to ${WINEPREFIX}.bak"
    mv "$WINEPREFIX" "${WINEPREFIX}.bak.$(date +%Y%m%d%H%M%S)"
fi

# Initialize Wine prefix
wineboot --init

print_status "Wine prefix created at $WINEPREFIX"

# ==========================================
# STEP 5: Install Required Windows Components
# ==========================================
echo ""
echo "Step 5: Installing Windows components via winetricks..."

# These components are required for MT5 to run properly
winetricks -q vcrun2019        # Visual C++ Runtime 2019
winetricks -q corefonts        # Core Windows fonts
winetricks -q dotnet48         # .NET Framework 4.8

print_status "Windows components installed"

# ==========================================
# STEP 6: Download MetaTrader 5
# ==========================================
echo ""
echo "Step 6: Downloading MetaTrader 5..."

MT5_INSTALLER="$HOME/Downloads/mt5setup.exe"

# Download MT5 installer
if [ ! -f "$MT5_INSTALLER" ]; then
    wget -O "$MT5_INSTALLER" "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
    print_status "MT5 installer downloaded"
else
    print_warning "MT5 installer already exists, using existing file"
fi

# ==========================================
# STEP 7: Install MetaTrader 5
# ==========================================
echo ""
echo "Step 7: Installing MetaTrader 5..."

print_info "Running MT5 installer. Please follow the installation wizard."
print_info "Recommended: Install to the default location"
echo ""

wine "$MT5_INSTALLER"

print_status "MT5 installation complete"

# ==========================================
# STEP 8: Create Launch Script
# ==========================================
echo ""
echo "Step 8: Creating MT5 launch script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cat > "$PROJECT_DIR/scripts/run_mt5.sh" << 'LAUNCH_SCRIPT'
#!/bin/bash
# Launch MetaTrader 5

export WINEPREFIX="$HOME/.wine-mt5"
export WINEARCH=win64

# Find MT5 terminal
MT5_PATHS=(
    "$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
    "$WINEPREFIX/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe"
)

for path in "${MT5_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "Launching MetaTrader 5..."
        wine "$path" &
        exit 0
    fi
done

echo "Error: MetaTrader 5 not found. Please reinstall."
exit 1
LAUNCH_SCRIPT

chmod +x "$PROJECT_DIR/scripts/run_mt5.sh"
print_status "Launch script created: scripts/run_mt5.sh"

# ==========================================
# STEP 9: Create Desktop Entry
# ==========================================
echo ""
echo "Step 9: Creating desktop entry..."

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/metatrader5.desktop" << EOF
[Desktop Entry]
Name=MetaTrader 5
Comment=Trading Platform
Exec=env WINEPREFIX=$HOME/.wine-mt5 wine "$HOME/.wine-mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
Type=Application
Categories=Finance;Office;
Icon=wine
Terminal=false
EOF

print_status "Desktop entry created"

# ==========================================
# STEP 10: Setup MT5 Python Integration
# ==========================================
echo ""
echo "Step 10: Setting up MT5 Python integration..."

# Activate project venv if it exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"

    # Install MetaTrader5 Python package
    pip install MetaTrader5

    print_status "MetaTrader5 Python package installed"
else
    print_warning "Virtual environment not found. Run setup_dev_env.sh first."
    print_info "Then manually install: pip install MetaTrader5"
fi

# ==========================================
# DONE
# ==========================================
echo ""
echo "=========================================="
echo -e "${GREEN}  MetaTrader 5 Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "To launch MT5:"
echo "  ./scripts/run_mt5.sh"
echo ""
echo "Or find 'MetaTrader 5' in your applications menu"
echo ""
echo "Wine prefix location: $WINEPREFIX"
echo ""
print_warning "Note: First launch may take a while as Wine configures itself"
echo ""
echo "For MT5 Python integration in your code:"
echo "  import MetaTrader5 as mt5"
echo "  mt5.initialize()"
echo ""
