#!/bin/bash
# Unset METAAPI Environment Variables
# =====================================
#
# This script unsets METAAPI_TOKEN and METAAPI_ACCOUNT_ID environment
# variables so that Kinetra reads credentials from .env file instead.
#
# Usage:
#   source scripts/unset_metaapi_env.sh
#
# Or add to your shell session:
#   . scripts/unset_metaapi_env.sh
#
# IMPORTANT: Must be sourced (not executed) to affect current shell!

echo "🧹 Unsetting METAAPI environment variables..."

# Check if variables are currently set
if [ -n "$METAAPI_TOKEN" ] || [ -n "$METAAPI_ACCOUNT_ID" ]; then
    echo "   Found:"
    [ -n "$METAAPI_TOKEN" ] && echo "   - METAAPI_TOKEN (${#METAAPI_TOKEN} chars)"
    [ -n "$METAAPI_ACCOUNT_ID" ] && echo "   - METAAPI_ACCOUNT_ID: $METAAPI_ACCOUNT_ID"

    # Unset the variables
    unset METAAPI_TOKEN
    unset METAAPI_ACCOUNT_ID

    echo ""
    echo "✅ Environment variables unset"
    echo "   Kinetra will now read credentials from .env file"
else
    echo "   No METAAPI environment variables found"
    echo "   ✅ Already clean!"
fi

echo ""
echo "💡 To make this permanent, the cleanup script has already removed"
echo "   these variables from ~/.bashrc. Just open a new terminal."
