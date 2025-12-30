import os

import pandas as pd


def parse_filename(filename):
    """Extract symbol and timeframe from filename, e.g., BTCUSD_H1_... -> ('BTCUSD', 'H1')"""
    base = filename.split("_")[0]  # e.g., BTCUSD or GBPUSD+
    if "+" in base:
        base = base.replace("+", "")  # Clean up GBPUSD+
    timeframe = filename.split("_")[1]  # e.g., H1, H4, M15
    return base, timeframe


def load_csv(filepath):
    """Load and parse a single tab-separated CSV, add datetime."""
    df = pd.read_csv(filepath, sep="\t")
    # Rename columns (remove < >)
    rename_dict = {
        "<DATE>": "date",
        "<TIME>": "time",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "tickvol",
        "<VOL>": "vol",
        "<SPREAD>": "spread",
    }
    df = df.rename(columns=rename_dict)
    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df


def main():
    data_dir = "data/master"
    all_dfs = []

    # List all CSV files
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            symbol, timeframe = parse_filename(filename)
            df = load_csv(filepath)
            df["symbol"] = symbol
            df["timeframe"] = timeframe
            all_dfs.append(df)
            print(f"Loaded {filename}: {len(df)} rows, symbol={symbol}, timeframe={timeframe}")

    # Concatenate all
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save to CSV
    output_path = "data/all_symbols_combined.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved combined DF to {output_path} with shape: {combined_df.shape}")

    # Print summary
    unique_symbols = combined_df["symbol"].unique()
    unique_timeframes = combined_df["timeframe"].unique()
    print(f"Unique symbols: {len(unique_symbols)} - {unique_symbols}")
    print(f"Unique timeframes: {len(unique_timeframes)} - {unique_timeframes}")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
