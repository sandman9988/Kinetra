#!/bin/bash
# Quick non-interactive cleanup

echo "================================================================================"
echo "  🧹 Removing MetaAPI Placeholder Environment Variables"
echo "================================================================================"
echo ""

# Unset from current environment
echo "Unsetting environment variables..."
unset METAAPI_TOKEN
unset METAAPI_ACCOUNT_ID

echo "✅ Environment variables unset"
echo ""

# Verify
echo "Verification:"
if env | grep -i metaapi > /dev/null 2>&1; then
    echo "⚠️  Still found:"
    env | grep -i metaapi
else
    echo "✅ Environment is clean - no METAAPI variables found"
fi

echo ""
echo "================================================================================"
echo "  ✅ Done!"
echo "================================================================================"
echo ""
echo "Note: This only affects the current shell. To permanently fix:"
echo "  1. Close this terminal and open a new one, OR"
echo "  2. Run: source ~/.bashrc (for bash) or source ~/.zshrc (for zsh)"
echo ""
echo "Test connection:"
echo "  python scripts/download/test_metaapi_connection.py"
echo ""
