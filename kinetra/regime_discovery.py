"""
Unsupervised Regime Discovery Engine
=====================================

NO pre-defined regimes. Let the data speak.
Uses GMM clustering on DSP + liquidity features to discover
natural market states without assumptions.

Key Principles:
- Number of regimes is data-driven (AIC/BIC selection)
- Regimes emerge from feature clustering, not imposed physics
- Transition precursors are empirically identified
- Works across any asset class/timeframe
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

from .dsp_features import DSPFeatureEngine
from .liquidity_features import LiquidityFeatureEngine


@dataclass
class RegimeProfile:
    """Statistical profile of a discovered regime."""
    regime_id: int
    bar_count: int
    return_mean: float
    return_std: float
    return_skew: float
    range_mean: float
    volume_mean: float
    amihud_mean: float
    cvd_trend: str  # 'positive', 'negative', 'neutral'
    label: str      # Human-readable description


@dataclass
class TransitionPrecursor:
    """Features that precede regime transitions."""
    cvd_delta_mean: float
    amihud_spike: float
    wavelet_energy_mean: float
    entropy_drop: float
    predictability: float  # What fraction of transitions showed this


@dataclass
class RegimeDiscoveryResult:
    """Complete results from regime discovery."""
    n_regimes: int
    regime_labels: np.ndarray
    regime_profiles: List[RegimeProfile]
    transition_precursors: Dict[Tuple[int, int], TransitionPrecursor]
    model: GaussianMixture
    scaler: StandardScaler
    feature_names: List[str]
    aic: float
    bic: float


class RegimeDiscoveryEngine:
    """
    Discovers market regimes through unsupervised clustering.

    Process:
    1. Extract DSP + liquidity features for each bar
    2. Standardize features
    3. Fit GMM with 2-10 components, select by AIC
    4. Analyze regime characteristics
    5. Identify transition precursors
    """

    def __init__(self, min_regimes: int = 2, max_regimes: int = 10):
        self.min_regimes = min_regimes
        self.max_regimes = max_regimes
        self.dsp_engine = DSPFeatureEngine()
        self.liquidity_engine = LiquidityFeatureEngine()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []

    def extract_features_for_all_bars(self, prices: pd.DataFrame,
                                       start_bar: int = 100) -> pd.DataFrame:
        """
        Extract DSP + liquidity features for all bars.

        Args:
            prices: OHLCV DataFrame
            start_bar: Skip initial bars (need history for features)

        Returns:
            DataFrame with features for each bar
        """
        features_list = []

        for i in range(start_bar, len(prices)):
            bar_features = {}

            # DSP features
            try:
                dsp = self.dsp_engine.extract_all(prices, i)
                bar_features.update(dsp)
            except Exception:
                pass

            # Liquidity features
            try:
                liq = self.liquidity_engine.extract_all(prices, i)
                bar_features.update(liq)
            except Exception:
                pass

            # Basic price features (non-linear)
            if i > 0:
                log_ret = np.log(prices['close'].iloc[i] / prices['close'].iloc[i-1])
                bar_features['log_return'] = log_ret
                bar_features['abs_return'] = abs(log_ret)
                bar_features['range'] = prices['high'].iloc[i] - prices['low'].iloc[i]

            bar_features['bar_idx'] = i
            features_list.append(bar_features)

        df = pd.DataFrame(features_list)
        self.feature_names = [c for c in df.columns if c != 'bar_idx']

        return df

    def select_optimal_n_regimes(self, features: np.ndarray) -> int:
        """
        Select optimal number of regimes using AIC.

        Args:
            features: Standardized feature matrix

        Returns:
            Optimal number of components
        """
        best_aic = np.inf
        best_n = self.min_regimes

        for n in range(self.min_regimes, self.max_regimes + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    n_init=3,
                    random_state=42
                )
                gmm.fit(features)
                aic = gmm.aic(features)

                if aic < best_aic:
                    best_aic = aic
                    best_n = n
            except Exception:
                continue

        return best_n

    def fit(self, prices: pd.DataFrame, start_bar: int = 100) -> RegimeDiscoveryResult:
        """
        Discover regimes in the data.

        Args:
            prices: OHLCV DataFrame
            start_bar: Skip initial bars

        Returns:
            Complete discovery results
        """
        # Extract features
        feature_df = self.extract_features_for_all_bars(prices, start_bar)

        # Prepare feature matrix (exclude bar_idx and non-numeric)
        feature_cols = [c for c in feature_df.columns
                       if c != 'bar_idx' and feature_df[c].dtype in [np.float64, np.int64, float, int]]

        X = feature_df[feature_cols].values

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal number of regimes
        n_regimes = self.select_optimal_n_regimes(X_scaled)

        # Fit final model
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            n_init=5,
            random_state=42
        )
        regime_labels = self.model.fit_predict(X_scaled)

        # Analyze regime profiles
        profiles = self._analyze_regimes(prices, regime_labels, start_bar)

        # Find transition precursors
        precursors = self._find_transition_precursors(
            feature_df, regime_labels, feature_cols
        )

        return RegimeDiscoveryResult(
            n_regimes=n_regimes,
            regime_labels=regime_labels,
            regime_profiles=profiles,
            transition_precursors=precursors,
            model=self.model,
            scaler=self.scaler,
            feature_names=feature_cols,
            aic=self.model.aic(X_scaled),
            bic=self.model.bic(X_scaled)
        )

    def _analyze_regimes(self, prices: pd.DataFrame,
                         labels: np.ndarray,
                         start_bar: int) -> List[RegimeProfile]:
        """
        Analyze characteristics of each regime.
        """
        profiles = []
        price_data = prices.iloc[start_bar:start_bar + len(labels)]

        log_returns = np.log(price_data['close'] / price_data['close'].shift(1)).values[1:]
        ranges = (price_data['high'] - price_data['low']).values

        if 'tickvol' in price_data.columns:
            volumes = price_data['tickvol'].values
        else:
            volumes = np.ones(len(price_data))

        for regime_id in range(max(labels) + 1):
            mask = labels == regime_id

            if np.sum(mask) < 5:
                continue

            # Get returns for this regime (offset by 1 for alignment)
            regime_returns = log_returns[mask[:-1]] if len(mask) > 1 else np.array([0])

            # Compute statistics
            ret_mean = np.mean(regime_returns) if len(regime_returns) > 0 else 0
            ret_std = np.std(regime_returns) if len(regime_returns) > 1 else 0
            ret_skew = float(pd.Series(regime_returns).skew()) if len(regime_returns) > 3 else 0

            range_mean = np.mean(ranges[mask])
            vol_mean = np.mean(volumes[mask])

            # CVD trend within regime
            # This is simplified - in production would compute actual CVD
            if ret_mean > 0.0001:
                cvd_trend = 'positive'
            elif ret_mean < -0.0001:
                cvd_trend = 'negative'
            else:
                cvd_trend = 'neutral'

            # Generate label based on characteristics
            if ret_std < np.median([np.std(log_returns[labels[:-1] == r]) for r in range(max(labels) + 1) if np.sum(labels == r) > 5]):
                if abs(ret_mean) < 0.0001:
                    label = 'calm_ranging'
                elif ret_mean > 0:
                    label = 'quiet_uptrend'
                else:
                    label = 'quiet_downtrend'
            else:
                if ret_skew < -0.5:
                    label = 'downside_breakout'
                elif ret_skew > 0.5:
                    label = 'upside_explosive'
                elif ret_mean > 0:
                    label = 'volatile_uptrend'
                elif ret_mean < 0:
                    label = 'volatile_downtrend'
                else:
                    label = 'transitional'

            profiles.append(RegimeProfile(
                regime_id=regime_id,
                bar_count=int(np.sum(mask)),
                return_mean=ret_mean,
                return_std=ret_std,
                return_skew=ret_skew,
                range_mean=range_mean,
                volume_mean=vol_mean,
                amihud_mean=0.0,  # Would need to compute
                cvd_trend=cvd_trend,
                label=label
            ))

        return profiles

    def _find_transition_precursors(self, feature_df: pd.DataFrame,
                                    labels: np.ndarray,
                                    feature_cols: List[str]) -> Dict:
        """
        Identify what features precede regime transitions.
        """
        precursors = {}

        # Find transition points
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append((i, labels[i-1], labels[i]))

        if len(transitions) < 5:
            return precursors

        # Analyze pre-transition features (5 bars before)
        lookback = 5

        for from_regime, to_regime in set((t[1], t[2]) for t in transitions):
            relevant = [(i, fr, to) for i, fr, to in transitions
                       if fr == from_regime and to == to_regime]

            if len(relevant) < 3:
                continue

            pre_features = []
            for idx, _, _ in relevant:
                if idx >= lookback:
                    pre_data = feature_df.iloc[idx-lookback:idx][feature_cols].mean()
                    pre_features.append(pre_data)

            if not pre_features:
                continue

            pre_df = pd.DataFrame(pre_features)

            # Key precursor signals
            cvd_col = [c for c in pre_df.columns if 'cvd_delta' in c.lower()]
            amihud_col = [c for c in pre_df.columns if 'amihud' in c.lower()]
            wavelet_col = [c for c in pre_df.columns if 'wavelet_energy' in c.lower()]
            entropy_col = [c for c in pre_df.columns if 'entropy' in c.lower()]

            precursors[(from_regime, to_regime)] = TransitionPrecursor(
                cvd_delta_mean=pre_df[cvd_col[0]].mean() if cvd_col else 0.0,
                amihud_spike=pre_df[amihud_col[0]].mean() if amihud_col else 0.0,
                wavelet_energy_mean=pre_df[wavelet_col[0]].mean() if wavelet_col else 0.0,
                entropy_drop=pre_df[entropy_col[0]].mean() if entropy_col else 0.0,
                predictability=len(relevant) / len(transitions)
            )

        return precursors

    def predict_regime(self, prices: pd.DataFrame, bar_idx: int = -1) -> Tuple[int, np.ndarray]:
        """
        Predict regime for a given bar.

        Returns:
            (regime_id, probabilities for each regime)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        # Extract features for this bar
        dsp = self.dsp_engine.extract_all(prices, bar_idx)
        liq = self.liquidity_engine.extract_all(prices, bar_idx)

        features = {**dsp, **liq}

        if bar_idx > 0:
            features['log_return'] = np.log(prices['close'].iloc[bar_idx] / prices['close'].iloc[bar_idx-1])
            features['abs_return'] = abs(features['log_return'])
            features['range'] = prices['high'].iloc[bar_idx] - prices['low'].iloc[bar_idx]

        # Build feature vector in same order as training
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        regime = self.model.predict(X_scaled)[0]
        probs = self.model.predict_proba(X_scaled)[0]

        return regime, probs


class CrossAssetRegimeAnalyzer:
    """
    Analyzes regime discovery results across multiple assets/timeframes.
    """

    def __init__(self):
        self.results: Dict[str, RegimeDiscoveryResult] = {}

    def add_result(self, asset_key: str, result: RegimeDiscoveryResult):
        """Add discovery result for an asset."""
        self.results[asset_key] = result

    def compare_regime_counts(self) -> pd.DataFrame:
        """Compare number of regimes discovered across assets."""
        data = []
        for key, result in self.results.items():
            parts = key.split('_')
            data.append({
                'asset': parts[0] if parts else key,
                'timeframe': parts[1] if len(parts) > 1 else 'unknown',
                'n_regimes': result.n_regimes,
                'aic': result.aic,
                'bic': result.bic
            })
        return pd.DataFrame(data)

    def get_universal_patterns(self) -> Dict:
        """
        Find patterns that appear across multiple assets.
        e.g., Do all assets have a "calm_ranging" regime?
        """
        label_counts = {}

        for key, result in self.results.items():
            for profile in result.regime_profiles:
                if profile.label not in label_counts:
                    label_counts[profile.label] = []
                label_counts[profile.label].append({
                    'asset': key,
                    'return_mean': profile.return_mean,
                    'return_std': profile.return_std
                })

        # Which labels appear in >50% of assets?
        universal = {
            label: data for label, data in label_counts.items()
            if len(data) >= len(self.results) * 0.5
        }

        return universal

    def get_class_specific_patterns(self) -> Dict:
        """
        Find patterns unique to certain asset classes.
        """
        # Group by asset class prefix
        class_regimes = {}

        for key, result in self.results.items():
            # Infer class from key or directory
            if 'BTC' in key or 'ETH' in key or 'XRP' in key:
                asset_class = 'crypto'
            elif 'EUR' in key or 'GBP' in key or 'AUD' in key:
                asset_class = 'forex'
            elif 'XAU' in key or 'XAG' in key or 'COPPER' in key:
                asset_class = 'metals'
            elif 'NAS' in key or 'DJ' in key or 'US2000' in key:
                asset_class = 'indices'
            else:
                asset_class = 'other'

            if asset_class not in class_regimes:
                class_regimes[asset_class] = []

            for profile in result.regime_profiles:
                class_regimes[asset_class].append({
                    'asset': key,
                    'label': profile.label,
                    'return_std': profile.return_std,
                    'return_skew': profile.return_skew
                })

        return class_regimes


# Convenience function
def discover_regimes(prices: pd.DataFrame, **kwargs) -> RegimeDiscoveryResult:
    """Quick regime discovery on a single dataset."""
    engine = RegimeDiscoveryEngine(**kwargs)
    return engine.fit(prices)
