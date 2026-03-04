#!/bin/bash
# Kinetra Live Trading - Conservative Settings for XAUUSD
#
# Problem: Getting chopped up by spread on frequent flips
# Solution: Longer warmup, skip more initial flips, tighter filters

SYMBOL="${1:-XAUUSD}"

echo "=============================================="
echo "Kinetra Live Trading - Conservative Settings"
echo "=============================================="
echo ""
echo "Changes from defaults:"
echo "  --min-warmup-bricks 50    # Was: 2 (need full filter window)"
echo "  --startup-skip-flips 10   # Was: 2 (skip noisy initial flips)"
echo "  --fliprate-threshold 0.45 # Was: 0.35 (tighter filter)"
echo "  --markov-threshold 0.60   # Was: 0.55 (tighter filter)"
echo ""
echo "This should reduce overtrading and spread churn."
echo ""

python scripts/renko_engine.py "$SYMBOL" \
    --stage live \
    --size micro \
    --min-warmup-bricks 50 \
    --startup-skip-flips 10 \
    --fliprate-threshold 0.45 \
    --markov-threshold 0.60 \
    --fliprate-window 50 \
    --markov-window 50 \
    "$@"
