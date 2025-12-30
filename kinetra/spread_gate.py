"""
Spread-Aware Execution Filter (Spread Gate)

Only permits trade entry when current spread is near historical minimum,
reducing execution cost and improving reward↔PnL alignment.

Key Principle: Avoid trading when execution cost (spread) is abnormally high,
i.e., when signal-to-noise ratio is poor.

Rule: current_spread <= k × rolling_min_spread(window=W)

Where:
- current_spread = bid-ask spread in points
- rolling_min_spread(W) = minimum spread over last W bars
- k = tolerance multiplier (e.g., 1.2 = allow up to 20% above recent min)

Broker-Specific Calibration:
- Spreads vary significantly between ECN brokers
- This module allows per-broker, per-symbol calibration
- Default thresholds based on Vantage International ECN profiles
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SpreadProfile:
    """Broker/symbol-specific spread characteristics."""
    symbol: str
    broker: str = "generic"

    # Typical spread ranges (in points/pips)
    tight_spread: float = 3.0      # Normal low-volatility spread
    normal_spread: float = 8.0     # Average spread
    wide_spread: float = 20.0      # High volatility / news events
    max_acceptable: float = 50.0   # Never trade above this

    # DSP-aligned window (bars) for rolling calculations
    # H1: 500 bars ≈ 3 weeks of trading
    # M15: 2000 bars ≈ 3 weeks
    rolling_window: int = 500

    # Tolerance multiplier (1.2 = allow 20% above min)
    k_multiplier: float = 1.2

    # Adaptive k based on volatility
    adaptive_k: bool = True
    k_min: float = 1.0
    k_max: float = 2.0


# Default profiles for common symbols (calibrated for Vantage International ECN)
VANTAGE_PROFILES = {
    "XAUUSD": SpreadProfile(
        symbol="XAUUSD",
        broker="Vantage",
        tight_spread=2.0,     # 0.2 USD (20 cents)
        normal_spread=5.0,    # 0.5 USD
        wide_spread=15.0,     # 1.5 USD (news events)
        max_acceptable=50.0,  # 5.0 USD (extreme)
        rolling_window=500,   # H1 bars
        k_multiplier=1.2,
    ),
    "XAUUSD+": SpreadProfile(
        symbol="XAUUSD+",
        broker="Vantage",
        tight_spread=2.0,
        normal_spread=5.0,
        wide_spread=15.0,
        max_acceptable=50.0,
        rolling_window=500,
        k_multiplier=1.2,
    ),
    "XAGUSD": SpreadProfile(
        symbol="XAGUSD",
        broker="Vantage",
        tight_spread=2.0,
        normal_spread=4.0,
        wide_spread=10.0,
        max_acceptable=30.0,
        rolling_window=500,
        k_multiplier=1.2,
    ),
    "EURUSD": SpreadProfile(
        symbol="EURUSD",
        broker="Vantage",
        tight_spread=0.5,
        normal_spread=1.5,
        wide_spread=5.0,
        max_acceptable=15.0,
        rolling_window=500,
        k_multiplier=1.1,  # Tighter for major pairs
    ),
    "GBPUSD": SpreadProfile(
        symbol="GBPUSD",
        broker="Vantage",
        tight_spread=0.8,
        normal_spread=2.0,
        wide_spread=6.0,
        max_acceptable=20.0,
        rolling_window=500,
        k_multiplier=1.15,
    ),
    "BTCUSD": SpreadProfile(
        symbol="BTCUSD",
        broker="Vantage",
        tight_spread=50.0,    # 50 points = $50
        normal_spread=150.0,
        wide_spread=500.0,
        max_acceptable=1000.0,
        rolling_window=168,   # 1 week for 24/7 crypto
        k_multiplier=1.3,     # More tolerance for crypto volatility
    ),
    "BTCJPY": SpreadProfile(
        symbol="BTCJPY",
        broker="Vantage",
        tight_spread=5000.0,  # In JPY
        normal_spread=15000.0,
        wide_spread=50000.0,
        max_acceptable=100000.0,
        rolling_window=168,
        k_multiplier=1.3,
    ),
}

# Generic profile for unknown symbols
DEFAULT_PROFILE = SpreadProfile(
    symbol="GENERIC",
    broker="generic",
    tight_spread=5.0,
    normal_spread=10.0,
    wide_spread=30.0,
    max_acceptable=100.0,
    rolling_window=500,
    k_multiplier=1.2,
)


class SpreadGate:
    """
    Dynamic spread-aware execution filter.

    Only allows trade entry when current spread is acceptable
    relative to recent historical minimum.
    """

    def __init__(
        self,
        profile: Optional[SpreadProfile] = None,
        symbol: Optional[str] = None,
        broker: str = "Vantage",
    ):
        if profile is not None:
            self.profile = profile
        elif symbol is not None:
            # Look up symbol in broker profiles
            symbol_key = symbol.replace("+", "").upper()
            for key in VANTAGE_PROFILES:
                if key.replace("+", "").upper() == symbol_key:
                    self.profile = VANTAGE_PROFILES[key]
                    break
            else:
                # Use default with symbol name
                self.profile = SpreadProfile(
                    symbol=symbol,
                    broker=broker,
                    **{k: v for k, v in DEFAULT_PROFILE.__dict__.items()
                       if k not in ['symbol', 'broker']}
                )
        else:
            self.profile = DEFAULT_PROFILE

        # State tracking
        self.spread_history: list = []
        self.rolling_min: float = float('inf')
        self.current_threshold: float = float('inf')
        self.bars_processed: int = 0

        # Statistics
        self.trades_allowed: int = 0
        self.trades_blocked: int = 0
        self.spread_stats: Dict[str, float] = {}

    def update(
        self,
        current_spread: float,
        atr: Optional[float] = None,
    ) -> Tuple[bool, float, Dict]:
        """
        Update spread gate and check if trading is allowed.

        Args:
            current_spread: Current bid-ask spread in points
            atr: Optional ATR for adaptive k calculation

        Returns:
            (allow_trade, threshold, info_dict)
        """
        self.bars_processed += 1

        # Update rolling history
        self.spread_history.append(current_spread)
        window = self.profile.rolling_window

        if len(self.spread_history) > window:
            self.spread_history = self.spread_history[-window:]

        # Calculate rolling minimum
        if len(self.spread_history) >= min(100, window // 5):
            self.rolling_min = min(self.spread_history)
        else:
            # Not enough history - use profile default
            self.rolling_min = self.profile.tight_spread

        # Calculate dynamic k multiplier
        k = self.profile.k_multiplier

        if self.profile.adaptive_k and atr is not None:
            # Scale k with relative volatility
            avg_spread = np.mean(self.spread_history) if self.spread_history else self.profile.normal_spread
            vol_ratio = current_spread / avg_spread if avg_spread > 0 else 1.0

            # k increases when volatility is high
            k = self.profile.k_min + (self.profile.k_max - self.profile.k_min) * min(vol_ratio, 2.0) / 2.0
            k = np.clip(k, self.profile.k_min, self.profile.k_max)

        # Calculate threshold
        self.current_threshold = k * self.rolling_min

        # Hard cap: never trade above max_acceptable
        hard_cap = self.profile.max_acceptable

        # Decision
        allow_trade = (
            current_spread <= self.current_threshold and
            current_spread <= hard_cap
        )

        if allow_trade:
            self.trades_allowed += 1
        else:
            self.trades_blocked += 1

        # Info dict
        info = {
            "current_spread": current_spread,
            "rolling_min": self.rolling_min,
            "threshold": self.current_threshold,
            "k_multiplier": k,
            "hard_cap": hard_cap,
            "allow_trade": allow_trade,
            "block_reason": None if allow_trade else (
                "above_hard_cap" if current_spread > hard_cap else "above_threshold"
            ),
            "spread_regime": self._classify_regime(current_spread),
        }

        return allow_trade, self.current_threshold, info

    def _classify_regime(self, spread: float) -> str:
        """Classify current spread regime."""
        if spread <= self.profile.tight_spread:
            return "TIGHT"
        elif spread <= self.profile.normal_spread:
            return "NORMAL"
        elif spread <= self.profile.wide_spread:
            return "WIDE"
        else:
            return "EXTREME"

    def get_stats(self) -> Dict:
        """Get spread gate statistics."""
        total = self.trades_allowed + self.trades_blocked
        return {
            "symbol": self.profile.symbol,
            "broker": self.profile.broker,
            "bars_processed": self.bars_processed,
            "trades_allowed": self.trades_allowed,
            "trades_blocked": self.trades_blocked,
            "block_rate": self.trades_blocked / total if total > 0 else 0,
            "avg_spread": np.mean(self.spread_history) if self.spread_history else 0,
            "min_spread": min(self.spread_history) if self.spread_history else 0,
            "max_spread": max(self.spread_history) if self.spread_history else 0,
            "current_threshold": self.current_threshold,
        }

    def reset(self):
        """Reset state for new episode."""
        self.spread_history = []
        self.rolling_min = float('inf')
        self.current_threshold = float('inf')
        self.bars_processed = 0
        self.trades_allowed = 0
        self.trades_blocked = 0


def add_spread_gate_to_dataframe(
    df: pd.DataFrame,
    spread_col: str = "<SPREAD>",
    window: int = 500,
    k: float = 1.2,
    max_acceptable: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add spread gate columns to a dataframe.

    Args:
        df: DataFrame with spread column
        spread_col: Name of spread column
        window: Rolling window for min calculation
        k: Tolerance multiplier
        max_acceptable: Hard cap on spread (optional)

    Returns:
        DataFrame with added columns:
        - rolling_min_spread
        - spread_threshold
        - allow_trade
        - spread_regime
    """
    df = df.copy()

    # Handle column name variations
    if spread_col not in df.columns:
        # Try common variations
        for col in ['spread', 'SPREAD', '<SPREAD>', 'Spread']:
            if col in df.columns:
                spread_col = col
                break
        else:
            raise ValueError(f"Spread column not found. Available: {df.columns.tolist()}")

    # Calculate rolling minimum (backward-looking only)
    df['rolling_min_spread'] = df[spread_col].rolling(
        window=window,
        min_periods=min(100, window // 5)
    ).min()

    # Fill initial NaN with first available min
    first_valid_idx = df['rolling_min_spread'].first_valid_index()
    if first_valid_idx is not None:
        first_min = df.loc[first_valid_idx, 'rolling_min_spread']
        df['rolling_min_spread'] = df['rolling_min_spread'].fillna(first_min)

    # Calculate threshold
    df['spread_threshold'] = k * df['rolling_min_spread']

    # Apply hard cap if specified
    if max_acceptable is not None:
        df['spread_threshold'] = df['spread_threshold'].clip(upper=max_acceptable)

    # Trading allowed flag
    df['allow_trade'] = df[spread_col] <= df['spread_threshold']

    if max_acceptable is not None:
        df['allow_trade'] = df['allow_trade'] & (df[spread_col] <= max_acceptable)

    # Classify spread regime
    def classify(spread, threshold):
        if spread <= threshold * 0.5:
            return "TIGHT"
        elif spread <= threshold:
            return "NORMAL"
        elif spread <= threshold * 2:
            return "WIDE"
        else:
            return "EXTREME"

    df['spread_regime'] = df.apply(
        lambda row: classify(row[spread_col], row['spread_threshold']),
        axis=1
    )

    return df


def analyze_spread_profile(
    data_dir: str,
    symbol: str,
    timeframe: str = "H1",
) -> Dict:
    """
    Analyze spread profile for a symbol from historical data.

    Returns percentile-based thresholds for spread gate calibration.
    """
    from pathlib import Path
    import glob

    # Find matching file
    pattern = f"{data_dir}/{symbol}*_{timeframe}_*.csv"
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")

    # Load and analyze
    df = pd.read_csv(files[0], sep='\t')

    # Handle spread column
    spread_col = None
    for col in ['<SPREAD>', 'spread', 'SPREAD']:
        if col in df.columns:
            spread_col = col
            break

    if spread_col is None:
        raise ValueError("No spread column found")

    spreads = df[spread_col].dropna()

    analysis = {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_points": len(spreads),
        "min": float(spreads.min()),
        "max": float(spreads.max()),
        "mean": float(spreads.mean()),
        "median": float(spreads.median()),
        "std": float(spreads.std()),
        "percentiles": {
            "p5": float(spreads.quantile(0.05)),
            "p10": float(spreads.quantile(0.10)),
            "p25": float(spreads.quantile(0.25)),
            "p50": float(spreads.quantile(0.50)),
            "p75": float(spreads.quantile(0.75)),
            "p90": float(spreads.quantile(0.90)),
            "p95": float(spreads.quantile(0.95)),
            "p99": float(spreads.quantile(0.99)),
        },
        "recommended_profile": {
            "tight_spread": float(spreads.quantile(0.10)),
            "normal_spread": float(spreads.quantile(0.50)),
            "wide_spread": float(spreads.quantile(0.90)),
            "max_acceptable": float(spreads.quantile(0.99)),
        }
    }

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze spread profiles")
    parser.add_argument("--data-dir", default="data/master", help="Data directory")
    parser.add_argument("--symbol", default="XAUUSD+", help="Symbol to analyze")
    parser.add_argument("--timeframe", default="H1", help="Timeframe")

    args = parser.parse_args()

    print(f"\nAnalyzing spread profile for {args.symbol} ({args.timeframe})...")

    try:
        analysis = analyze_spread_profile(
            args.data_dir,
            args.symbol,
            args.timeframe,
        )

        print(f"\n{'='*60}")
        print(f"  SPREAD ANALYSIS: {analysis['symbol']} ({analysis['timeframe']})")
        print(f"{'='*60}")
        print(f"  Data points: {analysis['data_points']:,}")
        print(f"  Min:         {analysis['min']:.1f}")
        print(f"  Max:         {analysis['max']:.1f}")
        print(f"  Mean:        {analysis['mean']:.1f}")
        print(f"  Median:      {analysis['median']:.1f}")
        print(f"  Std Dev:     {analysis['std']:.1f}")

        print(f"\n  Percentiles:")
        for pct, val in analysis['percentiles'].items():
            print(f"    {pct}: {val:.1f}")

        print(f"\n  Recommended Profile:")
        rec = analysis['recommended_profile']
        print(f"    tight_spread:   {rec['tight_spread']:.1f}")
        print(f"    normal_spread:  {rec['normal_spread']:.1f}")
        print(f"    wide_spread:    {rec['wide_spread']:.1f}")
        print(f"    max_acceptable: {rec['max_acceptable']:.1f}")

    except Exception as e:
        print(f"Error: {e}")
