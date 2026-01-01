# Data Download Scripts

Scripts for downloading and managing market data from various sources.

## Main Scripts

- **`download_interactive.py`** - Interactive MetaAPI data downloader (recommended)
- **`metaapi_bulk_download.py`** - Bulk download from MetaAPI
- **`check_and_fill_data.py`** - Check for gaps and fill missing data

## Data Sources

- **MetaAPI**: `download_metaapi.py`, `metaapi_bulk_download.py`, `metaapi_sync.py`
- **MT5 Direct**: `download_mt5_data.py`
- **Generic**: `download_market_data.py`

## Data Management

- **`backup_data.py`** - Backup historical data
- **`check_data_integrity.py`** - Validate data quality
- **`prepare_data.py`** - Prepare data for training/testing
- **`prepare_exploration_data.py`** - Prepare data for exploration framework
- **`standardize_data_cutoff.py`** - Standardize data cutoff dates
- **`parallel_data_prep.py`** - Parallel data preparation

## Utilities

- **`convert_mt5_format.py`** - Convert MT5 CSV format
- **`load_all_symbols.py`** - Load all available symbols
- **`select_metaapi_account.py`** - Select MetaAPI account
- **`fetch_broker_spec_from_metaapi.py`** - Fetch broker specifications
- **`extract_mt5_specs.py`** - Extract MT5 specifications

## Quick Start

```bash
# Interactive download (recommended)
python scripts/download/download_interactive.py

# Check for missing data
python scripts/download/check_and_fill_data.py

# Prepare data for exploration
python scripts/download/prepare_exploration_data.py
```
