# MetaAPI Credentials Setup Guide

> **Single Source of Truth for MetaAPI Authentication**

This guide covers the centralized credential management system for Kinetra's MetaAPI integration.

---

## Quick Start (New Users)

```bash
# 1. Run the setup script
python scripts/setup_metaapi_credentials.py

# 2. Follow the interactive prompts to:
#    - Enter your MetaAPI token (once)
#    - Select broker accounts to configure
#    - Assign friendly labels to each account

# 3. Done! Credentials are saved to .env
```

---

## Overview

Kinetra uses a **centralized credential system** with these key principles:

1. ✅ **One setup script** - `setup_metaapi_credentials.py` handles all credential input
2. ✅ **Multi-account support** - Configure multiple broker accounts with friendly labels
3. ✅ **No hardcoded secrets** - All credentials stored in `.env` (gitignored)
4. ✅ **Read-only access** - All other scripts just read from environment
5. ✅ **No prompting in scripts** - If credentials missing, scripts point to setup

---

## Architecture

### Files

```
scripts/
  ├── setup_metaapi_credentials.py    # Main setup script (run this)
  └── utils/
      └── secure_token_helper.py      # Read-only credential access

.env                                   # Credentials storage (NOT committed)
```

### Environment Variables

```bash
# API Token (required)
METAAPI_TOKEN=eyJhbGciOiJSUzUxMi...

# Default Account
METAAPI_ACCOUNT_ID=a1b2c3d4-e5f6-7890...

# Named Accounts (optional, for multiple brokers)
METAAPI_ACCOUNT_VANTAGE=a1b2c3d4-e5f6-7890...
METAAPI_ACCOUNT_DEMO=b2c3d4e5-f6a7-8901...
METAAPI_ACCOUNT_LIVE=c3d4e5f6-a7b8-9012...
```

---

## Setup Workflow

### First-Time Setup

```bash
$ python scripts/setup_metaapi_credentials.py

================================================================================
 MetaAPI Credential Setup
================================================================================

📋 Get your MetaAPI token from: https://app.metaapi.cloud/token
   (The token is a long JWT string starting with 'eyJ...')

Enter your MetaAPI token (input hidden): ●●●●●●●●●●

🔄 Validating token...
✅ Token is valid! Found 2 account(s)

🔄 Fetching your MetaAPI accounts...

✅ Found 2 account(s):

  [1] Vantage MT5 Demo
      Login: 12345678 | Server: VantageInternational-Demo
      ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
      State: DEPLOYED

  [2] IC Markets Live
      Login: 87654321 | Server: ICMarketsSC-Live
      ID: b2c3d4e5-f6a7-8901-bcde-f23456789abc
      State: DEPLOYED

📝 Select accounts to configure:
   Enter account numbers separated by commas (e.g., 1,3)
   Or press Enter to configure all accounts

Account numbers: 1,2

📋 Configure account: Vantage MT5 Demo (ID: a1b2c3d4...)
   Enter a short label for this account (e.g., 'VANTAGE', 'DEMO', 'LIVE')
   Or press Enter to use the account name

   Label: VANTAGE_DEMO

📋 Configure account: IC Markets Live (ID: b2c3d4e5...)
   Enter a short label for this account (e.g., 'VANTAGE', 'DEMO', 'LIVE')
   Or press Enter to use the account name

   Label: IC_LIVE

================================================================================
 Configuration Summary
================================================================================
✅ Default account: Vantage MT5 Demo
   ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
✅ METAAPI_ACCOUNT_VANTAGE_DEMO = a1b2c3d4-e5f6-7890-abcd-ef1234567890
   Vantage MT5 Demo (Login: 12345678)
✅ METAAPI_ACCOUNT_IC_LIVE = b2c3d4e5-f6a7-8901-bcde-f23456789abc
   IC Markets Live (Login: 87654321)

✅ Configuration saved to /path/to/Kinetra/.env

================================================================================
 Setup Complete!
================================================================================
✅ Credentials configured successfully

Environment variables set:
  • METAAPI_TOKEN (hidden)
  • METAAPI_ACCOUNT_ID = a1b2c3d4-e5f6-7890-abcd-ef1234567890
  • METAAPI_ACCOUNT_VANTAGE_DEMO = a1b2c3d4-e5f6-7890-abcd-ef1234567890
  • METAAPI_ACCOUNT_IC_LIVE = b2c3d4e5-f6a7-8901-bcde-f23456789abc

Next steps:
  1. Download data: python scripts/download/download_interactive.py
  2. Run tests: python scripts/run_exhaustive_tests.py --generate-dashboard

To add more accounts later, run:
  python scripts/setup_metaapi_credentials.py --add-account
```

### Adding More Accounts

```bash
# Re-run with --add-account flag
python scripts/setup_metaapi_credentials.py --add-account

# This will:
# - Use existing token
# - Show all available accounts
# - Let you add new labeled accounts
```

---

## Using Credentials in Your Scripts

### Option 1: Require Token/Account (Auto-Exit)

```python
from scripts.utils.secure_token_helper import require_token, require_account_id

# Get token (exits with helpful message if not found)
token = require_token()

# Get default account (exits if not found)
account_id = require_account_id()

# Get specific labeled account (e.g., "VANTAGE_DEMO")
vantage_account = require_account_id("VANTAGE_DEMO")
```

### Option 2: Check if Available

```python
from scripts.utils.secure_token_helper import get_metaapi_token, get_metaapi_account_id

# Get token (returns None if not found)
token = get_metaapi_token()
if not token:
    print("Run: python scripts/setup_metaapi_credentials.py")
    sys.exit(1)

# Get default account
account_id = get_metaapi_account_id()

# Get labeled account
vantage_account = get_metaapi_account_id("VANTAGE")
```

### Option 3: List All Configured Accounts

```python
from scripts.utils.secure_token_helper import list_configured_accounts

accounts = list_configured_accounts()
# Returns: {'DEFAULT': 'a1b2...', 'VANTAGE_DEMO': 'a1b2...', 'IC_LIVE': 'b2c3...'}

for label, account_id in accounts.items():
    print(f"{label}: {account_id}")
```

---

## Manual Setup (Advanced)

If you prefer to manually edit `.env`:

```bash
# Create .env file in project root
touch .env

# Add credentials (make sure .env is in .gitignore!)
cat >> .env << 'EOF'
# MetaAPI Configuration
METAAPI_TOKEN=eyJhbGciOiJSUzUxMi...your_token_here

# Default account
METAAPI_ACCOUNT_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890

# Named accounts
METAAPI_ACCOUNT_VANTAGE=a1b2c3d4-e5f6-7890-abcd-ef1234567890
METAAPI_ACCOUNT_DEMO=b2c3d4e5-f6a7-8901-bcde-f23456789abc
EOF

# Verify it's gitignored
grep "^\.env$" .gitignore  # Should output: .env
```

---

## Security Best Practices

### ✅ DO

- Run `setup_metaapi_credentials.py` to configure credentials
- Keep `.env` file in `.gitignore` (already configured)
- Use labeled accounts for different brokers/environments
- Use `require_token()` / `require_account_id()` for clean error handling

### ❌ DON'T

- Hardcode tokens or account IDs in scripts
- Commit `.env` file to git
- Share your MetaAPI token publicly
- Use production tokens in test/demo environments

---

## Troubleshooting

### "No token found" Error

```bash
❌ ERROR: No MetaAPI token found
   Run the setup script to configure credentials:
   python scripts/setup_metaapi_credentials.py
```

**Solution:** Run the setup script to configure credentials.

### "No account ID found" Error

```bash
❌ ERROR: No MetaAPI account ID found
   Environment variable METAAPI_ACCOUNT_ID not set

   Run the setup script to configure credentials:
   python scripts/setup_metaapi_credentials.py
```

**Solution:** Run the setup script to select/configure accounts.

### "Account 'VANTAGE' not found" Error

```bash
❌ ERROR: No account ID found for 'VANTAGE'
   Environment variable METAAPI_ACCOUNT_VANTAGE not set
```

**Solution:** Either:
1. Re-run setup script with `--add-account` flag
2. Manually add `METAAPI_ACCOUNT_VANTAGE=...` to `.env`

### Check Current Configuration

```bash
# Run the helper directly to see what's configured
python scripts/utils/secure_token_helper.py

# Output:
# ================================================================================
# Secure Token Helper - Status Check
# ================================================================================
# ✅ Token found: eyJhbGciOiJSUzUxMiIsInR...aROHvoJjgX
#
# ✅ Found 3 configured account(s):
#    • DEFAULT: a1b2c3d4-e5f6-7890-abcd-ef1234567890
#    • IC_LIVE: b2c3d4e5-f6a7-8901-bcde-f23456789abc
#    • VANTAGE_DEMO: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Token Expired

MetaAPI tokens expire after a certain period. If you get authentication errors:

1. Get a new token from https://app.metaapi.cloud/token
2. Re-run setup script: `python scripts/setup_metaapi_credentials.py`
3. Choose to update existing token when prompted

---

## Migration from Old Scripts

If you have old scripts with hardcoded tokens:

```python
# OLD (INSECURE):
API_TOKEN = "eyJhbGciOiJSUzUxMi..."
ACCOUNT_ID = "a1b2c3d4-e5f6-7890..."

# NEW (SECURE):
from scripts.utils.secure_token_helper import require_token, require_account_id

API_TOKEN = require_token()
ACCOUNT_ID = require_account_id()
```

---

## CI/CD Integration

For GitHub Actions or other CI/CD:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    steps:
      - name: Run tests
        env:
          METAAPI_TOKEN: ${{ secrets.METAAPI_TOKEN }}
          METAAPI_ACCOUNT_ID: ${{ secrets.METAAPI_ACCOUNT_ID }}
        run: |
          python scripts/run_exhaustive_tests.py
```

Add secrets in GitHub:
1. Go to repo Settings → Secrets and variables → Actions
2. Add `METAAPI_TOKEN` secret
3. Add `METAAPI_ACCOUNT_ID` secret

---

## Reference

### Get Your MetaAPI Token

1. Log in to https://app.metaapi.cloud
2. Navigate to: **Settings → API Tokens** or https://app.metaapi.cloud/token
3. Copy your token (starts with `eyJ...`)

### Find Account IDs

Run the setup script and it will list all your accounts with their IDs.

Or use the account selector:
```bash
python scripts/download/select_metaapi_account.py
```

---

## Related Documentation

- [METAAPI_SETUP.md](./METAAPI_SETUP.md) - MetaAPI integration overview
- [METAAPI_QUICKSTART.md](./METAAPI_QUICKSTART.md) - Quick start guide
- [METAAPI_QUICK_LOGIN.md](./METAAPI_QUICK_LOGIN.md) - Login workflow

---

**Questions?** Check existing documentation or run `python scripts/setup_metaapi_credentials.py --help`
