#!/bin/bash
# Quick backtest execution script

cd /home/renierdejager/Projects/Kinetra

# Create output directories
mkdir -p outputs/results outputs/logs

# Verify imports work
echo "Checking imports..."
python -c "
import sys
sys.path.insert(0, '.')
from kinetra.renko.trading_engine import RenkoEngine
from kinetra.renko.dsp import run_dsp
from kinetra.renko.brick_engine import build_renko, bricks_per_day
print('✓ All imports OK')
" || exit 1

# Run backtest
echo "Running 3-month backtest..."
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

echo ""
echo "Done! Results saved to outputs/results/ and logs to outputs/logs/"
