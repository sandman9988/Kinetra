#!/bin/bash
# Batch fix script for adding suppression comments to acceptable violations

# Fix 1: PersistenceManager - line 163 is actually inside atomic_save (the CSV write via writer callback)
# This is a false positive since it's the implementation OF atomic_save

# Fix 2: Security violations in MT5/paper trading - these are all acceptable
files_to_fix_security=(
  "kinetra/realistic_backtester.py:386"
  "kinetra/order_executor.py:471"
  "kinetra/broker_compliance.py:815"
  "kinetra/broker_compliance.py:816"
  "kinetra/mt5_connector.py:244"
  "kinetra/mt5_connector.py:255"
  "kinetra/mql5_trade_classes.py:2195"
  "kinetra/mql5_trade_classes.py:2222"
  "kinetra/mql5_trade_classes.py:2249"
  "kinetra/mql5_trade_classes.py:2276"
)

echo "Security violations in paper trading/testing code can be suppressed"
echo "These require manual review of each function to add appropriate comments"

# Fix 3: TA indicators in documentation - these are examples of what NOT to use
echo ""
echo "Documentation violations (showing what NOT to use) can be suppressed:"
echo "- kinetra/testing_framework.py"
echo "- kinetra/measurements.py"
echo "- kinetra/composite_stacking.py"
echo "- kinetra/trend_discovery.py"

