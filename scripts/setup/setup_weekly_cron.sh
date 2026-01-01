#!/bin/bash
# Setup weekly MetaAPI sync cron job
# Runs every Saturday at 11:00 PM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH=$(which python3 || which python)

# Create cron entry
CRON_ENTRY="0 23 * * 6 cd $PROJECT_DIR && $PYTHON_PATH scripts/metaapi_sync.py --weekly-update >> logs/metaapi_sync.log 2>&1"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

echo "Setting up weekly MetaAPI sync cron job..."
echo ""
echo "Cron entry:"
echo "$CRON_ENTRY"
echo ""

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "metaapi_sync.py"; then
    echo "Cron job already exists. Updating..."
    crontab -l 2>/dev/null | grep -v "metaapi_sync.py" | crontab -
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed!"
echo ""
echo "To verify: crontab -l"
echo "To remove: crontab -l | grep -v metaapi_sync | crontab -"
echo ""
echo "The sync will run every Saturday at 11:00 PM"
echo "Logs: $PROJECT_DIR/logs/metaapi_sync.log"
