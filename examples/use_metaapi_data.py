#!/usr/bin/env python3
"""
Example: Using MetaAPI Downloaded Data for ML Training

This script demonstrates how to load and use the MT5 historical data
downloaded via mt5_metaapi_sync.py for machine learning and backtesting.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_metaapi_data(symbol: str, timeframe: str, data_dir: Path = Path("data/metaapi")) -> pd.DataFrame:
    """
    Load historical data downloaded via MetaAPI.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe (e.g., 'H1', 'M15')
        data_dir: Directory where MetaAPI data is stored

    Returns:
        DataFrame with OHLCV data and basic features
    """
    filepath = data_dir / f"{symbol}_{timeframe}_history.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Download it first using:\n"
            f"  python3 scripts/mt5_metaapi_sync.py --init --symbol {symbol} --timeframe {timeframe}"
        )

    # Load data
    df = pd.read_csv(filepath, index_col='time', parse_dates=True)

    print(f"✅ Loaded {len(df):,} bars of {symbol} {timeframe}")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print(f"   Columns: {list(df.columns)}")

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators for ML features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()

    # Remove NaN from indicators
    df = df.dropna()

    print(f"✅ Added technical indicators")
    print(f"   Features: {len(df.columns)}")
    print(f"   Valid bars after dropna: {len(df):,}")

    return df


def prepare_ml_features(df: pd.DataFrame, lookback: int = 10) -> tuple:
    """
    Prepare features and labels for ML training.

    Args:
        df: DataFrame with OHLCV and indicators
        lookback: Number of bars to look back for features

    Returns:
        (X, y) where X is features and y is labels (next candle direction)
    """
    df = df.copy()

    # Label: Next candle direction (1 = up, 0 = down)
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Select feature columns
    feature_cols = [
        'returns', 'range_pct', 'body_pct',
        'rsi', 'macd', 'macd_hist', 'bb_width', 'atr', 'volatility',
        'sma_20', 'sma_50'
    ]

    # Normalize features
    for col in feature_cols:
        if col in df.columns:
            df[f'{col}_norm'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)

    # Create feature matrix (normalized features)
    norm_cols = [c for c in df.columns if c.endswith('_norm')]
    X = df[norm_cols].values
    y = df['label'].values

    # Remove last row (no label)
    X = X[:-1]
    y = y[:-1]

    print(f"✅ Prepared ML features")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")

    return X, y


def train_test_split_time_series(X, y, train_ratio: float = 0.8):
    """
    Split time series data into train/test (no shuffle!).

    Args:
        X: Feature matrix
        y: Labels
        train_ratio: Ratio of data to use for training

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"✅ Train/Test split")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test


def example_sklearn_model(X_train, X_test, y_train, y_test):
    """
    Example: Train a simple scikit-learn model.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("⚠️  scikit-learn not installed. Skipping example.")
        return

    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*60}\n")

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))

    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\nTop 5 Important Features:")
    top_idx = feature_importance.argsort()[-5:][::-1]
    for idx in top_idx:
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")


def main():
    """Main example workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Example: Use MetaAPI data for ML")
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Symbol to load')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe to load')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"METAAPI DATA → ML PIPELINE EXAMPLE")
    print(f"{'='*60}\n")

    # 1. Load data
    df = load_metaapi_data(args.symbol, args.timeframe)

    # 2. Add technical indicators
    df = add_technical_indicators(df)

    # 3. Prepare ML features
    X, y = prepare_ml_features(df)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, train_ratio=0.8)

    # 5. Train example model
    example_sklearn_model(X_train, X_test, y_train, y_test)

    print(f"\n{'='*60}")
    print("EXAMPLE COMPLETE")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("  1. Add more sophisticated features")
    print("  2. Try different models (XGBoost, LSTM, etc.)")
    print("  3. Implement walk-forward validation")
    print("  4. Backtest predictions with transaction costs")
    print("  5. Deploy to live trading (paper → real)")


if __name__ == '__main__':
    main()
