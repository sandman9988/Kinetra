# E2E Test: 2 Symbols × 2 Timeframes

## Overview

Comprehensive end-to-end test that validates the complete Kinetra pipeline from data acquisition to analysis.

## Features

### ✅ Complete Pipeline Coverage

1. **MetaAPI Authentication** - Connects to MetaAPI cloud service
2. **Data Download/Update** - Downloads historical market data for specified symbols and timeframes
3. **Data Validation** - Validates data quality, completeness, and integrity
4. **Superpot Analysis** - Runs comprehensive feature importance testing
5. **Theorem Validation** - Validates physics-based trading theorems
6. **Report Generation** - Creates detailed JSON and HTML reports

### 🔄 On/Off Ramps

The test implements comprehensive error handling with multiple recovery strategies:

- **Abort**: Exit immediately on critical errors
- **Skip**: Continue to next stage if non-critical stage fails
- **Retry**: Retry failed operations with exponential backoff
- **Prompt**: Ask user for action on error (interactive mode)

### 📊 State Management

- **Checkpointing**: Automatic checkpoint saving after each stage
- **Resume Capability**: Resume from last checkpoint after interruption
- **State Persistence**: Full state saved to JSON for recovery
- **Atomic Operations**: Uses WorkflowManager for atomic file operations

### 📝 Comprehensive Logging

- **Structured Logging**: Timestamped logs with severity levels
- **Real-time Progress**: Live updates on stage progress
- **Error Tracking**: Detailed error messages with stack traces
- **Performance Metrics**: Duration tracking for each stage

## Usage

### Basic Usage

```bash
# Run with default configuration (BTCUSD, EURUSD in H1, H4 timeframes)
python scripts/testing/test_e2e_symbols_timeframes.py
```

### Custom Symbols and Timeframes

```bash
# Test Bitcoin and Gold on M15 and H1 timeframes
python scripts/testing/test_e2e_symbols_timeframes.py \
    --symbols BTCUSD XAUUSD \
    --timeframes M15 H1 \
    --days 180
```

### Quick Test Mode

```bash
# Faster execution with reduced episodes and history
python scripts/testing/test_e2e_symbols_timeframes.py --quick
```

### Resume from Checkpoint

```bash
# Resume from latest checkpoint
python scripts/testing/test_e2e_symbols_timeframes.py --resume

# Resume from specific checkpoint
python scripts/testing/test_e2e_symbols_timeframes.py \
    --checkpoint .e2e_checkpoints/checkpoint_20241201_153045.json
```

### Error Handling Modes

```bash
# Abort on first error
python scripts/testing/test_e2e_symbols_timeframes.py --action-on-error abort

# Skip failed stages and continue
python scripts/testing/test_e2e_symbols_timeframes.py --action-on-error skip

# Retry with exponential backoff (default)
python scripts/testing/test_e2e_symbols_timeframes.py --action-on-error retry

# Prompt user for action (interactive)
python scripts/testing/test_e2e_symbols_timeframes.py --action-on-error prompt
```

## Prerequisites

### Environment Variables

Required environment variables in `.env` file:

```bash
# MetaAPI credentials
METAAPI_TOKEN=your_token_here
METAAPI_ACCOUNT_ID=your_account_id_here
```

Get your MetaAPI credentials from: https://app.metaapi.cloud/

### Dependencies

The test requires the following Python packages:

```bash
# Core dependencies
pip install pandas numpy scipy

# MetaAPI SDK
pip install metaapi-cloud-sdk

# Environment management
pip install python-dotenv

# Kinetra modules (should already be installed)
```

## Configuration

### Default Configuration

```python
# Default symbols
symbols = ['BTCUSD', 'EURUSD']

# Default timeframes
timeframes = ['H1', 'H4']

# History
days_history = 365

# Superpot settings
superpot_episodes = 80
superpot_prune_every = 15
superpot_prune_count = 8

# Theorem settings
theorem_lookback = 20
theorem_top_n = 30

# Error handling
max_retries = 3
retry_delay = 5.0  # seconds
```

### Quick Mode Configuration

When `--quick` flag is used:

```python
days_history = 90
superpot_episodes = 30
superpot_prune_every = 10
superpot_prune_count = 5
```

## Output

### Directory Structure

```
results/e2e/
├── e2e_report_20241201_153045.json    # Detailed JSON report
└── e2e_report_20241201_153045.html    # Human-readable HTML report

.e2e_checkpoints/
└── checkpoint_20241201_153045.json    # Resume checkpoint

logs/e2e/
└── workflow_20241201_153045.log       # Detailed execution log
```

### Report Contents

#### JSON Report

- Test configuration
- Stage execution details
- Downloaded data files
- Superpot analysis results (per symbol/timeframe/role)
- Theorem validation results
- Summary statistics

#### HTML Report

- Executive summary with key metrics
- Configuration details
- Stage execution timeline
- Superpot results visualization
- Theorem validation summary
- Success/failure breakdown

## Stages

### Stage 1: MetaAPI Authentication

**Purpose**: Establish connection to MetaAPI cloud service

**Actions**:
- Load credentials from environment
- Connect to MetaAPI
- Verify account status
- Wait for synchronization

**Critical**: Yes (test aborts if this fails)

**Output**:
```json
{
  "account_id": "uuid",
  "account_name": "My MT5 Account",
  "account_state": "DEPLOYED",
  "platform": "mt5"
}
```

### Stage 2: Data Download/Update

**Purpose**: Download or update market data for all symbol/timeframe combinations

**Actions**:
- For each symbol and timeframe:
  - Download historical OHLCV data
  - Save to `data/master/` directory
  - Validate file creation

**Critical**: Yes (test aborts if this fails)

**Output**:
```json
{
  "downloaded_count": 4,
  "files": {
    "BTCUSD_H1": "data/master/BTCUSD_H1_20240101_20241201.csv",
    "BTCUSD_H4": "data/master/BTCUSD_H4_20240101_20241201.csv",
    ...
  }
}
```

### Stage 3: Data Validation

**Purpose**: Verify data quality and completeness

**Actions**:
- Check for required columns
- Count rows and null values
- Analyze time gaps
- Validate date ranges

**Critical**: Yes (test aborts if data is invalid)

**Output**:
```json
{
  "BTCUSD_H1": {
    "valid": true,
    "rows": 6500,
    "nulls": 0,
    "start": "2024-01-01 00:00:00",
    "end": "2024-12-01 23:00:00",
    "mean_gap_hours": 1.0
  }
}
```

### Stage 4: Superpot Analysis

**Purpose**: Run comprehensive feature importance testing

**Actions**:
- For each symbol/timeframe combination:
  - For each role (trader, risk_manager, portfolio_manager):
    - Train agent with feature pruning
    - Track surviving features
    - Measure PnL and risk metrics

**Critical**: No (test continues even if this fails)

**Output**:
```json
{
  "BTCUSD_H1": {
    "trader": {
      "avg_pnl": 250.0,
      "avg_drawdown": 0.05,
      "win_rate": 55.0,
      "n_surviving": 45,
      "top_features": [["energy_velocity", 0.85], ...]
    },
    ...
  }
}
```

### Stage 5: Theorem Validation

**Purpose**: Validate physics-based trading theorems

**Actions**:
- Compute physics features (energy, damping, entropy)
- Test core theorems (10+ theorems)
- Explore feature combinations
- Analyze regime statistics

**Critical**: No (test continues even if this fails)

**Output**:
```json
{
  "BTCUSD_H1": {
    "theorems": [
      {
        "name": "T1: Underdamped Release",
        "signals": 1500,
        "hit_rate": 25.0,
        "lift": 1.25,
        "status": "CONFIRMED"
      },
      ...
    ],
    "best_combination": {
      "conditions": "energy>p70 + damping_rising",
      "hit_rate": 35.0,
      "lift": 1.75
    },
    "regime_stats": {...}
  }
}
```

### Stage 6: Report Generation

**Purpose**: Create comprehensive test report

**Actions**:
- Compile all results
- Generate summary statistics
- Create JSON report
- Create HTML report

**Critical**: No (nice to have, but not essential)

**Output**:
```json
{
  "json_report": "results/e2e/e2e_report_20241201_153045.json",
  "html_report": "results/e2e/e2e_report_20241201_153045.html"
}
```

## Recovery Mechanisms

### Automatic Retry

Failed operations are automatically retried with exponential backoff:

1. First retry: 5 seconds delay
2. Second retry: 10 seconds delay
3. Third retry: 20 seconds delay

### Checkpoint Resume

If the test is interrupted:

1. Run with `--resume` flag
2. Test loads last checkpoint
3. Continues from last completed stage
4. Skips already-completed stages

### Manual Recovery

1. Check logs in `logs/e2e/` for error details
2. Fix underlying issue (credentials, network, etc.)
3. Resume with `--resume` or rerun from start
4. Use `--action-on-error skip` to bypass problematic stages

## Performance

### Expected Duration

**Default mode** (2 symbols × 2 timeframes):
- MetaAPI Auth: ~10 seconds
- Data Download: ~2-5 minutes
- Data Validation: ~10 seconds
- Superpot Analysis: ~15-30 minutes
- Theorem Validation: ~5-10 minutes
- Report Generation: ~5 seconds

**Total**: ~25-50 minutes

**Quick mode**:
- Total: ~10-15 minutes

### Optimization Tips

1. **Use Quick Mode** for testing: `--quick`
2. **Reduce History**: `--days 90`
3. **Use Fewer Symbols**: `--symbols BTCUSD`
4. **Skip Superpot**: (requires code modification)

## Troubleshooting

### MetaAPI Connection Fails

**Symptoms**:
```
ERROR: METAAPI_TOKEN not set in environment
ERROR: MetaAPI connection timeout
```

**Solutions**:
1. Check `.env` file has correct credentials
2. Verify MetaAPI account is active
3. Check internet connection
4. Try different account with `METAAPI_ACCOUNT_ID`

### Data Download Fails

**Symptoms**:
```
ERROR: Failed to download BTCUSD H1
ERROR: No file found for EURUSD H4
```

**Solutions**:
1. Verify symbol exists on broker
2. Check timeframe is valid (M1, M5, M15, M30, H1, H4, D1)
3. Reduce `--days` if requesting too much history
4. Check data/master/ directory has write permissions

### Superpot Analysis Fails

**Symptoms**:
```
ERROR: Superpot components not available
ERROR: Invalid data format for Superpot
```

**Solutions**:
1. Ensure all Kinetra modules are installed
2. Check data has required columns (close, open, high, low)
3. Verify data has sufficient rows (>200)
4. Use `--action-on-error skip` to bypass Superpot

### Theorem Validation Fails

**Symptoms**:
```
ERROR: Theorem validation components not available
ERROR: Physics features computation failed
```

**Solutions**:
1. Check scipy and numpy are installed
2. Verify PhysicsEngine is available
3. Ensure data quality (no NaN values)
4. Use `--action-on-error skip` to bypass theorems

## Integration with CI/CD

### GitHub Actions

```yaml
name: E2E Test

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM
  workflow_dispatch:

jobs:
  e2e-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run E2E test
        env:
          METAAPI_TOKEN: ${{ secrets.METAAPI_TOKEN }}
          METAAPI_ACCOUNT_ID: ${{ secrets.METAAPI_ACCOUNT_ID }}
        run: |
          python scripts/testing/test_e2e_symbols_timeframes.py --quick
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: e2e-reports
          path: results/e2e/
```

### Cron Job

```bash
# Run weekly E2E test
0 2 * * 0 cd /path/to/Kinetra && \
  source venv/bin/activate && \
  python scripts/testing/test_e2e_symbols_timeframes.py --quick \
  >> logs/e2e_cron.log 2>&1
```

## Best Practices

1. **Run in Quick Mode First**: Always test with `--quick` before full run
2. **Monitor Logs**: Check `logs/e2e/` for detailed execution info
3. **Use Checkpoints**: Keep checkpoints for debugging and recovery
4. **Validate Reports**: Review HTML reports after each run
5. **Clean Old Data**: Periodically clean old checkpoints and reports
6. **Version Control**: Don't commit checkpoints or large reports to git
7. **Secure Credentials**: Never commit `.env` file with real credentials

## Future Enhancements

- [ ] Parallel symbol/timeframe processing
- [ ] Advanced statistical validation (PBO, CPCV)
- [ ] Live trading simulation mode
- [ ] Email/Slack notifications on completion
- [ ] Grafana dashboard integration
- [ ] Multi-account testing
- [ ] Custom theorem definition
- [ ] Strategy backtesting integration
- [ ] Performance benchmarking

## Related Documentation

- [Scientific Testing Framework](../../docs/SCIENTIFIC_TESTING_GUIDE.md)
- [Superpot Analysis](../analysis/superpot_complete.py)
- [Theorem Validation](./validate_theorems.py)
- [Workflow Manager](../../kinetra/workflow_manager.py)
- [MetaAPI Sync](../download/metaapi_sync.py)

## Support

For issues or questions:
1. Check logs in `logs/e2e/`
2. Review checkpoint files in `.e2e_checkpoints/`
3. Consult related documentation
4. Open an issue on GitHub with logs and error details
