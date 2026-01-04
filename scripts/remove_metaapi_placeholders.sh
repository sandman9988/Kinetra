#!/bin/bash
################################################################################
# Remove MetaAPI Placeholder Environment Variables
################################################################################
#
# This script removes placeholder METAAPI environment variables from:
# - Current shell session
# - ~/.bashrc
# - ~/.bash_profile
# - ~/.profile
# - ~/.zshrc
# - ~/.zprofile
#
# It creates backups before making any changes and reports what was done.
#
# Usage:
#   bash scripts/remove_metaapi_placeholders.sh
#
# Or make executable and run:
#   chmod +x scripts/remove_metaapi_placeholders.sh
#   ./scripts/remove_metaapi_placeholders.sh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
REMOVED_FROM_ENV=0
REMOVED_FROM_FILES=0
BACKED_UP_FILES=0

echo ""
echo "================================================================================"
echo "  🧹 Remove MetaAPI Placeholder Environment Variables"
echo "================================================================================"
echo ""

################################################################################
# Step 1: Check current environment
################################################################################

echo -e "${BLUE}Step 1: Checking current shell environment...${NC}"
echo ""

if [[ -n "$METAAPI_TOKEN" ]] || [[ -n "$METAAPI_ACCOUNT_ID" ]]; then
    echo -e "${YELLOW}⚠️  Found METAAPI environment variables in current shell:${NC}"

    if [[ -n "$METAAPI_TOKEN" ]]; then
        echo "   METAAPI_TOKEN=${METAAPI_TOKEN:0:20}..."
        REMOVED_FROM_ENV=$((REMOVED_FROM_ENV + 1))
    fi

    if [[ -n "$METAAPI_ACCOUNT_ID" ]]; then
        echo "   METAAPI_ACCOUNT_ID=$METAAPI_ACCOUNT_ID"
        REMOVED_FROM_ENV=$((REMOVED_FROM_ENV + 1))
    fi

    echo ""
    echo -e "${GREEN}✅ These will be unset from current shell${NC}"
else
    echo -e "${GREEN}✅ No METAAPI environment variables found in current shell${NC}"
fi

echo ""

################################################################################
# Step 2: Check shell configuration files
################################################################################

echo -e "${BLUE}Step 2: Checking shell configuration files...${NC}"
echo ""

# List of config files to check
CONFIG_FILES=(
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.profile"
    "$HOME/.zshrc"
    "$HOME/.zprofile"
)

FOUND_IN_FILES=()

for config_file in "${CONFIG_FILES[@]}"; do
    if [[ -f "$config_file" ]]; then
        # Check if file contains METAAPI exports
        if grep -q "export.*METAAPI_TOKEN\|export.*METAAPI_ACCOUNT_ID" "$config_file" 2>/dev/null; then
            FOUND_IN_FILES+=("$config_file")
            echo -e "${YELLOW}⚠️  Found METAAPI exports in: $config_file${NC}"

            # Show the lines
            grep "export.*METAAPI_TOKEN\|export.*METAAPI_ACCOUNT_ID" "$config_file" | while read -r line; do
                echo "   $line"
            done
            echo ""
        fi
    fi
done

if [[ ${#FOUND_IN_FILES[@]} -eq 0 ]]; then
    echo -e "${GREEN}✅ No METAAPI exports found in shell config files${NC}"
else
    echo -e "${YELLOW}Found in ${#FOUND_IN_FILES[@]} file(s)${NC}"
fi

echo ""

################################################################################
# Step 3: Confirm action
################################################################################

if [[ $REMOVED_FROM_ENV -eq 0 ]] && [[ ${#FOUND_IN_FILES[@]} -eq 0 ]]; then
    echo "================================================================================"
    echo -e "${GREEN}✅ Nothing to clean up - system is already clean!${NC}"
    echo "================================================================================"
    echo ""
    exit 0
fi

echo "================================================================================"
echo "  📋 Summary of Changes"
echo "================================================================================"
echo ""

if [[ $REMOVED_FROM_ENV -gt 0 ]]; then
    echo "  • Will unset $REMOVED_FROM_ENV environment variable(s) from current shell"
fi

if [[ ${#FOUND_IN_FILES[@]} -gt 0 ]]; then
    echo "  • Will remove METAAPI exports from ${#FOUND_IN_FILES[@]} config file(s)"
    echo "  • Backups will be created with .backup.TIMESTAMP extension"
fi

echo ""
echo -e "${YELLOW}Do you want to proceed? (y/N)${NC} "
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${RED}❌ Cancelled - no changes made${NC}"
    echo ""
    exit 0
fi

echo ""

################################################################################
# Step 4: Unset from current shell
################################################################################

if [[ $REMOVED_FROM_ENV -gt 0 ]]; then
    echo -e "${BLUE}Step 3: Unsetting from current shell...${NC}"
    echo ""

    unset METAAPI_TOKEN
    unset METAAPI_ACCOUNT_ID

    echo -e "${GREEN}✅ Environment variables unset${NC}"
    echo ""
fi

################################################################################
# Step 5: Remove from config files
################################################################################

if [[ ${#FOUND_IN_FILES[@]} -gt 0 ]]; then
    echo -e "${BLUE}Step 4: Removing from config files...${NC}"
    echo ""

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    for config_file in "${FOUND_IN_FILES[@]}"; do
        # Create backup
        backup_file="${config_file}.backup.${TIMESTAMP}"
        cp "$config_file" "$backup_file"
        BACKED_UP_FILES=$((BACKED_UP_FILES + 1))

        echo -e "${GREEN}📦 Created backup: $backup_file${NC}"

        # Remove METAAPI lines
        grep -v "export.*METAAPI_TOKEN\|export.*METAAPI_ACCOUNT_ID" "$config_file" > "${config_file}.tmp"
        mv "${config_file}.tmp" "$config_file"

        REMOVED_FROM_FILES=$((REMOVED_FROM_FILES + 1))
        echo -e "${GREEN}✅ Cleaned: $config_file${NC}"
        echo ""
    done
fi

################################################################################
# Step 6: Verify .env file
################################################################################

echo -e "${BLUE}Step 5: Verifying .env file...${NC}"
echo ""

ENV_FILE=".env"

if [[ -f "$ENV_FILE" ]]; then
    if grep -q "METAAPI_TOKEN\|METAAPI_ACCOUNT_ID" "$ENV_FILE" 2>/dev/null; then
        echo -e "${GREEN}✅ .env file contains METAAPI credentials${NC}"

        # Show redacted values
        if grep -q "METAAPI_TOKEN=" "$ENV_FILE"; then
            TOKEN_VALUE=$(grep "METAAPI_TOKEN=" "$ENV_FILE" | cut -d'=' -f2 | sed 's/"//g' | sed "s/'//g")
            if [[ ${#TOKEN_VALUE} -gt 20 ]]; then
                echo "   METAAPI_TOKEN=${TOKEN_VALUE:0:10}...${TOKEN_VALUE: -10}"
            else
                echo "   METAAPI_TOKEN=***REDACTED***"
            fi
        fi

        if grep -q "METAAPI_ACCOUNT_ID=" "$ENV_FILE"; then
            ACCOUNT_ID=$(grep "METAAPI_ACCOUNT_ID=" "$ENV_FILE" | cut -d'=' -f2 | sed 's/"//g' | sed "s/'//g")
            echo "   METAAPI_ACCOUNT_ID=$ACCOUNT_ID"
        fi
    else
        echo -e "${YELLOW}⚠️  .env file exists but has no METAAPI credentials${NC}"
        echo ""
        echo "To configure credentials, run:"
        echo "   python scripts/download/setup_metaapi_credentials.py"
    fi
else
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    echo ""
    echo "To configure credentials, run:"
    echo "   python scripts/download/setup_metaapi_credentials.py"
fi

echo ""

################################################################################
# Step 7: Final summary
################################################################################

echo "================================================================================"
echo "  ✅ Cleanup Complete!"
echo "================================================================================"
echo ""

if [[ $REMOVED_FROM_ENV -gt 0 ]]; then
    echo "✅ Unset $REMOVED_FROM_ENV environment variable(s) from current shell"
fi

if [[ $REMOVED_FROM_FILES -gt 0 ]]; then
    echo "✅ Removed METAAPI exports from $REMOVED_FROM_FILES config file(s)"
fi

if [[ $BACKED_UP_FILES -gt 0 ]]; then
    echo "✅ Created $BACKED_UP_FILES backup file(s)"
fi

echo ""
echo "================================================================================"
echo "  📋 Next Steps"
echo "================================================================================"
echo ""

echo "1. Reload your shell configuration (or just close and reopen terminal):"
echo "   source ~/.bashrc    # For bash"
echo "   source ~/.zshrc     # For zsh"
echo ""

echo "2. Verify environment is clean:"
echo "   env | grep -i metaapi"
echo "   (Should show nothing or only real values from .env)"
echo ""

echo "3. Test MetaAPI connection:"
echo "   python scripts/download/test_metaapi_connection.py"
echo ""

echo "4. If you need to restore backups:"
for config_file in "${FOUND_IN_FILES[@]}"; do
    backup_file="${config_file}.backup.${TIMESTAMP}"
    if [[ -f "$backup_file" ]]; then
        echo "   cp $backup_file $config_file"
    fi
done

echo ""
echo "================================================================================"
echo "  🎉 All Done!"
echo "================================================================================"
echo ""
echo "Your shell environment is now clean. METAAPI credentials will be read"
echo "from the .env file only."
echo ""
