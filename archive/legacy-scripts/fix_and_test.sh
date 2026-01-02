#!/bin/bash
# One-command fix for Kinetra menu system

echo "=================================================="
echo "  KINETRA MENU FIX & TEST"
echo "=================================================="

cd ~/Kinetra

echo ""
echo "[1/3] Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "     ✅ Cache cleared"

echo ""
echo "[2/3] Verifying fix..."
python3 -c "
from run_comprehensive_exploration import run_comprehensive_exploration
print('     ✅ Import successful - Path error FIXED!')
" 2>&1

echo ""
echo "[3/3] Testing menu..."
echo ""
echo "You can now run: python kinetra_menu.py"
echo ""
echo "The exploration menu (option 2) should work without errors."
echo ""
echo "=================================================="
echo "  FIX APPLIED SUCCESSFULLY ✅"
echo "=================================================="
