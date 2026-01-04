# Setup & Authentication Workflow - Fix Summary

**Date:** 2026-01-04  
**Status:** ✅ FIXED  
**Severity:** High (Blocking user onboarding)

---

## Problem Summary

The account setup & accreditation flow was broken in multiple ways:

1. **API Method Error**: `'MetatraderAccountApi' object has no attribute 'get_accounts'`
2. **Token Type Confusion**: Unclear guidance between API Access Token vs Account Access Token
3. **Workflow Logic Issue**: "Test Connection" was trying to select/list accounts instead of testing configured account
4. **Environment Variable Override**: `.bashrc` had placeholder values overriding `.env` file

---

## Root Causes

### 1. Incorrect MetaAPI SDK Method Name

**File:** `scripts/download/setup_metaapi_credentials.py` (line 108)

**Problem:**
```python
accounts = await api.metatrader_account_api.get_accounts()  # ❌ Method doesn't exist
```

**Fix:**
```python
accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()  # ✅ Correct
```

**Why:** The MetaAPI SDK uses `get_accounts_with_infinite_scroll_pagination()` for paginated account listing, not `get_accounts()`.

---

### 2. Token Type Confusion

**Problem:**
- MetaAPI has TWO types of tokens with different capabilities
- Scripts didn't distinguish or guide users appropriately
- Error messages were cryptic

**Token Types:**

| Type | Purpose | Capabilities | Where to Get |
|------|---------|--------------|--------------|
| 🔑 **API Access Token** | Account management & data download | List all accounts, manage settings, download data | https://app.metaapi.cloud/api-access/generate-token |
| 🔐 **Account Access Token** | Single account trading | Access ONE specific account only | Individual account settings |

**Fix:**
- Added clear documentation in credential setup script
- Added error detection for token type mismatches
- Updated all prompts to specify "API Access Token" where needed
- Added helpful troubleshooting messages

---

### 3. Workflow Logic Issue

**Problem:**
Menu option "Test MetaAPI Connection" (Menu 1 → 2) was calling `select_metaapi_account.py`, which:
- Required API Access Token (not Account token)
- Tried to LIST all accounts (wrong for a "test")
- Confused users about what the option did

**Fix:**

Created new `test_metaapi_connection.py` that:
- ✅ Tests the CONFIGURED account only
- ✅ Works with BOTH token types
- ✅ Shows account status, deployment state, health
- ✅ Provides clear troubleshooting steps

Reorganized menu:
```
OLD Menu:
1. Configure MetaAPI Credentials
2. Test MetaAPI Connection          ← Called select_metaapi_account.py (wrong!)
3. Configure MT5 (Local Terminal)
4. Test MT5 Connection
5. View Current Configuration

NEW Menu:
1. Configure MetaAPI Credentials
2. Test MetaAPI Connection          ← Now calls test_metaapi_connection.py ✅
3. Select/Change MetaAPI Account    ← Moved select_metaapi_account.py here ✅
4. Configure MT5 (Local Terminal)
5. Test MT5 Connection
6. View Current Configuration
```

**Separation of Concerns:**
- **Option 2 (Test)**: Verify configured credentials work (any token type)
- **Option 3 (Select)**: List and choose from available accounts (API token required)

---

### 4. Environment Variable Override

**Problem:**
User's `~/.bashrc` contained:
```bash
export METAAPI_TOKEN="your-token-here"
export METAAPI_ACCOUNT_ID="your-account-id-here"
```

These placeholder values were:
- Loaded into every shell session
- Overriding correct values in `.env` file
- Causing authentication failures

**How It Was Discovered:**
```bash
$ python scripts/download/test_metaapi_connection.py
# Shows: Token: your-token... (15 chars) ← Placeholder!
# But .env has: eyJhbGciOi... (2612 chars) ← Real token!

$ env | grep METAAPI
METAAPI_TOKEN=your-token-here        ← Found the culprit!
METAAPI_ACCOUNT_ID=your-account-id-here
```

**Fix:**

Created `cleanup_bashrc_metaapi.py`:
- ✅ Backs up `.bashrc` before changes
- ✅ Removes all `export METAAPI_*` lines
- ✅ Shows exactly what was removed
- ✅ Provides undo instructions
- ✅ Safe and idempotent

Created `unset_metaapi_env.sh`:
- Quick helper to unset vars in current shell
- Usage: `source scripts/unset_metaapi_env.sh`

---

## Files Changed/Created

### Modified Files

1. **`scripts/download/setup_metaapi_credentials.py`**
   - Fixed API method name
   - Added token type detection and guidance
   - Enhanced error messages
   - Added `print_token_type_info()` helper

2. **`scripts/download/select_metaapi_account.py`**
   - Added token type error handling
   - Updated prompts to specify API Access Token
   - Better error messages

3. **`kinetra_production_menu.py`**
   - Changed "Test Connection" to call new test script
   - Added "Select/Change Account" option
   - Reorganized menu numbering
   - Added confirmation prompt before account selection

### New Files

1. **`scripts/download/test_metaapi_connection.py`** ⭐
   - Tests configured account only (not listing)
   - Works with any token type
   - Shows account details and health status
   - Clear troubleshooting guidance

2. **`scripts/cleanup_bashrc_metaapi.py`**
   - Removes METAAPI env vars from `.bashrc`
   - Creates timestamped backups
   - Safe and reversible

3. **`scripts/unset_metaapi_env.sh`**
   - Quick shell helper to unset env vars
   - Must be sourced (affects current shell)

4. **`docs/SETUP_WORKFLOW_FIX.md`** (this file)
   - Complete documentation of fixes

---

## Testing & Validation

### Before Fix
```bash
$ python kinetra_production_menu.py
# Menu 1 → 1: Configure Credentials
❌ Validation failed: 'MetatraderAccountApi' object has no attribute 'get_accounts'

# Menu 1 → 2: Test Connection
❌ Failed to fetch accounts: You can not invoke get_accounts method...
```

### After Fix
```bash
# Step 1: Clean up environment
$ python scripts/cleanup_bashrc_metaapi.py
✅ Removed 4 line(s) from .bashrc
✅ Backup created: ~/.bashrc.backup.20260104_103508

# Step 2: Unset in current shell
$ source scripts/unset_metaapi_env.sh
✅ Environment variables unset

# Step 3: Configure credentials
$ python kinetra_production_menu.py
# Menu 1 → 1
✅ Token valid! Found 1 account(s)
✅ Account ID verified
✅ Credentials saved

# Step 4: Test connection
# Menu 1 → 2
✅ ACCOUNT FOUND
✅ Account is DEPLOYED and ready to use
✅ CONNECTION TEST PASSED
```

---

## User Action Required

To complete the fix in your current terminal session:

### Option 1: Unset Environment Variables (Quick)
```bash
source scripts/unset_metaapi_env.sh
```

### Option 2: Open New Terminal (Permanent)
The cleanup script already removed the variables from `.bashrc`, so just:
```bash
# Close current terminal
# Open new terminal
# Environment variables will be gone permanently
```

### Option 3: Verify Cleanup
```bash
# Check if variables are still set
env | grep METAAPI

# Should return nothing (or only values from .env)
```

---

## Prevention & Best Practices

### ✅ DO

1. **Store credentials in `.env` file only**
   - Already gitignored
   - Easier to manage
   - Per-project isolation

2. **Use credential setup script**
   - Menu 1 → Configure MetaAPI Credentials
   - Interactive, validated, safe

3. **Understand token types**
   - API Access Token: For development, data download, account management
   - Account Access Token: For production trading with single account

4. **Test credentials regularly**
   - Menu 1 → Test MetaAPI Connection
   - Verifies account is accessible and deployed

### ❌ DON'T

1. **Don't set credentials in shell profiles**
   - Not `.bashrc`, `.bash_profile`, `.zshrc`, etc.
   - They override `.env` and cause confusion
   - Security risk (easier to accidentally expose)

2. **Don't hardcode credentials in scripts**
   - Always use `.env` file
   - Use environment variables only in production deployment

3. **Don't commit `.env` to git**
   - Already in `.gitignore`
   - Use `.env.example` for templates

4. **Don't mix token types**
   - If you have Account token, don't try to list all accounts
   - If you need to list accounts, get API Access Token

---

## Related Documentation

- **Token Generation**: https://app.metaapi.cloud/api-access/generate-token
- **MetaAPI Accounts**: https://app.metaapi.cloud/accounts
- **MetaAPI SDK Docs**: https://github.com/agiliumtrade-ai/metaapi-python-sdk

---

## Architecture Notes

### Credential Loading Priority (After Fix)

```
1. Environment variables (if set) ← Removed from .bashrc
2. .env file                      ← Primary source now
3. Interactive prompt             ← Fallback
```

### Error Handling Flow

```
User runs credential setup
    ↓
Check existing credentials in .env
    ↓
If found → Test with MetaAPI
    ↓
If valid → Offer to keep/update/replace
    ↓
If invalid or missing → Prompt for new
    ↓
Validate with API call
    ↓
If token is Account type → Show warning, offer API token URL
    ↓
If token is API type → List accounts, let user select
    ↓
Save to .env (atomic write)
    ↓
Add .env to .gitignore
```

### Test Connection Flow

```
User runs test connection
    ↓
Load credentials from .env
    ↓
Call get_account(account_id)  ← Single account only
    ↓
Display account info
    ↓
Test streaming connection
    ↓
Show health status
    ↓
Close connection
    ↓
Report success/failure with troubleshooting steps
```

---

## Lessons Learned

1. **Check SDK method names carefully**
   - MetaAPI uses verbose method names
   - `get_accounts_with_infinite_scroll_pagination()` vs `get_accounts()`

2. **Environment variables override everything**
   - Always check `env | grep VAR` when debugging credential issues
   - Shell profiles can cause hard-to-debug issues

3. **Separate test from selection**
   - "Test connection" should test configured account
   - "Select account" should list and choose accounts
   - Different purposes, different token requirements

4. **Token type matters**
   - API tokens and Account tokens have different capabilities
   - Detect and guide users to correct token type

5. **Always backup before modifying user files**
   - `.bashrc` cleanup script creates timestamped backups
   - Provides undo instructions

---

## Future Improvements

### Potential Enhancements

1. **Auto-detect token type on input**
   - Parse token structure
   - Warn immediately if wrong type for operation

2. **Credential validation on startup**
   - Check credentials when menu starts
   - Show warning if misconfigured

3. **Support for multiple accounts**
   - Save multiple account configs
   - Switch between accounts easily

4. **Encrypted credential storage**
   - Optional encryption for `.env` values
   - Decrypt on load

5. **Token refresh automation**
   - Detect expired tokens
   - Guide user to refresh

---

## Verification Checklist

- [x] Fixed MetaAPI SDK method name
- [x] Added token type detection and guidance
- [x] Created proper test connection script
- [x] Separated test from account selection
- [x] Cleaned up `.bashrc` environment variables
- [x] Created helper scripts for cleanup
- [x] Updated menu workflow
- [x] Added comprehensive error messages
- [x] Tested end-to-end workflow
- [x] Documented all changes
- [x] Fixed async close() warning

---

## Status: ✅ RESOLVED

The setup & authentication workflow is now:
- ✅ **Working correctly**
- ✅ **User-friendly** with clear guidance
- ✅ **Error-resilient** with helpful troubleshooting
- ✅ **Secure** (credentials in `.env` only)
- ✅ **Well-documented**

Users can now successfully:
1. Configure MetaAPI credentials
2. Test their connection
3. Select/change accounts when needed
4. Proceed to data download and training

**Ready for morning testing! 🚀**