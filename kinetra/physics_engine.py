"""
Physics Engine for Market Modeling

Models markets as kinetic energy systems with:
- Energy: Market momentum (kinetic energy from price changes)
- Damping: Market friction (resistance to movement)
- Entropy: Market disorder/uncertainty
- Acceleration: Rate of momentum change (d²P/dt²)
- Jerk: Rate of acceleration change (d³P/dt³) - best fat candle predictor
- Impulse: Momentum change over time window
- Liquidity: Volume per price movement
- Buying Pressure: Directional order flow proxy
- Reynolds Number: Turbulent vs laminar flow indicator
- Viscosity: Internal friction / resistance to flow

Layer-1 Sensor Set:
- Base physics + normalized percentiles
- GMM regime clustering with cluster, regime, regime_age_frac
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# Optional sklearn import for GMM clustering
try:
    from sklearn.mixture import GaussianMixture

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RegimeType(Enum):
    """Market regime classifications based on physics state."""

    UNDERDAMPED = "underdamped"  # High energy, low friction -> trending
    CRITICAL = "critical"  # Balanced -> transitional
    OVERDAMPED = "overdamped"  # Low energy, high friction -> ranging
    LAMINAR = "laminar"  # Smooth, predictable trends
    BREAKOUT = "breakout"  # High energy burst


import numpy as np
from hmmlearn import hmm
from sklearn import svm
from sklearn.preprocessing import StandardScaler


class PhysicsEngine:
    """
    Physics-based feature extractor for price series.

    Given a close price series (one symbol, one timeframe), it computes:

    Base signals (per bar):
        - velocity       : log-return
        - acceleration   : Δ velocity
        - jerk           : Δ acceleration
        - energy         : kinetic energy ~ v^2
        - damping        : rolling friction proxy (ζ)
        - entropy        : rolling return-distribution entropy
        - reynolds       : market Reynolds number (trend / noise)
        - potential      : potential energy proxy (1 / long vol)
        - eta            : KE / PE (local efficiency)

    Normalised Layer-1 sensors (0–1 percentiles):
        - KE_pct, Re_m_pct, zeta_pct, Hs_pct, PE_pct, eta_pct

    Regime clustering:
        - cluster        : integer cluster from GMM
        - regime         : UNDERDAMPED / OVERDAMPED / LAMINAR / BREAKOUT
        - regime_age_frac: 0–1, age within current regime segment

    Attributes:
        - lookback       : min bars before signals are trustworthy
    """

    def __init__(
        self,
        vel_window: int = 1,
        damping_window: int = 64,
        entropy_window: int = 64,
        re_slow: int = 24,
        re_fast: int = 6,
        pe_window: int = 72,
        pct_window: int = 500,
        n_clusters: int = 3,
        random_state: int = 42,
        mass: float = 1.0,
        lookback: int = 20,
    ) -> None:
        # Ensure all window parameters are integers (critical for pandas .rolling())
        self.vel_window = int(vel_window)
        self.damping_window = int(damping_window)
        self.entropy_window = int(entropy_window)
        self.re_slow = int(re_slow)
        self.re_fast = int(re_fast)
        self.pe_window = int(pe_window)
        self.pct_window = int(pct_window)
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.mass = float(mass)

        # BacktestEngine uses this to avoid using early noisy bars
        self.lookback = int(
            max(
                lookback,
                damping_window,
                entropy_window,
                re_slow,
                re_fast,
                pe_window,
                pct_window // 4,
            )
        )

        # Hybrid HMM-SVM components
        self.hmm = hmm.GaussianHMM(
            n_components=self.n_clusters, covariance_type="full", random_state=self.random_state
        )
        self.svm = svm.SVC(kernel="rbf", random_state=self.random_state, probability=True)

    # ---------- public API ----------

    def compute_physics_state(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        open_price: Optional[pd.Series] = None,
        include_percentiles: bool = True,
        include_kinematics: bool = True,
        include_flow: bool = True,
    ) -> pd.DataFrame:
        """
        Main entry point - compute complete Layer-1 physics state.

        Args:
            prices: pd.Series of close prices (indexed by datetime)
            volume: Optional volume data (enables liquidity metrics)
            high: Optional high prices (enables range-based metrics)
            low: Optional low prices (enables range-based metrics)
            open_price: Optional open prices (enables buying pressure)
            include_percentiles: Include rolling percentile ranks
            include_kinematics: Include velocity, acceleration, jerk
            include_flow: Include Reynolds number (requires volume)

        Returns:
            pd.DataFrame aligned to 'prices' index with physics / regime columns.
        """
        prices = prices.astype(float)
        idx = prices.index

        # 1) basic kinematics
        x = np.log(prices.replace(0, np.nan)).ffill()
        v = x.diff(self.vel_window)  # velocity = log-return
        a = v.diff()
        j = a.diff()

        # 2) kinetic energy
        energy = 0.5 * self.mass * (v**2)

        # 3) damping (friction proxy, ζ)
        damping = self._rolling_damping(v, self.damping_window)

        # 4) entropy (return-distribution entropy)
        entropy = self._rolling_entropy(v, self.entropy_window)

        # 5) Reynolds number: trend / noise
        reynolds = self._reynolds(v, self.re_slow, self.re_fast)

        # 6) potential energy proxy: 1 / long-window volatility
        potential = self._potential_energy(v, self.pe_window)

        # 7) efficiency: KE / PE
        eta = self._efficiency(energy, potential)

        # Assemble core DF
        df = pd.DataFrame(
            {
                "close": prices,
                "log_price": x,
                "velocity": v.fillna(0.0),
                "acceleration": a.fillna(0.0),
                "jerk": j.fillna(0.0),
                "energy": energy.fillna(0.0),
                "damping": damping,
                "entropy": entropy,
                "reynolds": reynolds,
                "potential": potential,
                "eta": eta,
            },
            index=idx,
        )

        # Force non-negative values (numerical stability)
        df["energy"] = df["energy"].clip(lower=0.0)
        df["damping"] = df["damping"].clip(lower=0.0)
        df["entropy"] = df["entropy"].clip(lower=0.0)

        # 8) Buying pressure if OHLC available
        if high is not None and low is not None:
            bar_range = (high - low).clip(lower=1e-10)
            bp = (prices - low) / bar_range
            df["BP"] = bp.fillna(0.5)

            # PE from volatility compression
            atr = bar_range.rolling(self.lookback).mean().clip(lower=1e-10)
            pe = 1.0 - (bar_range / atr).clip(upper=2.0) / 2.0
            df["pe"] = pe.clip(lower=0.0, upper=1.0).fillna(0.5)
        else:
            df["BP"] = 0.5
            df["pe"] = 0.5

        # 9) Liquidity if volume available
        if volume is not None and high is not None and low is not None:
            liquidity = self._liquidity(high, low, prices, volume)
            df["liquidity"] = liquidity

            # Viscosity
            viscosity = self._viscosity(high, low, prices, volume)
            df["viscosity"] = viscosity
        else:
            df["liquidity"] = 0.0
            df["viscosity"] = 1.0

        # 10) Layer-1 normalised sensors (0–1 rolling percentiles)
        if include_percentiles:
            df["KE_pct"] = self._rolling_percentile(df["energy"], self.pct_window)
            df["Re_m_pct"] = self._rolling_percentile(df["reynolds"], self.pct_window)
            df["zeta_pct"] = self._rolling_percentile(df["damping"], self.pct_window)
            df["Hs_pct"] = self._rolling_percentile(df["entropy"], self.pct_window)
            df["PE_pct"] = self._rolling_percentile(df["potential"], self.pct_window)
            df["eta_pct"] = self._rolling_percentile(df["eta"], self.pct_window)
            df["velocity_pct"] = self._rolling_percentile(df["velocity"].abs(), self.pct_window)
            df["jerk_pct"] = self._rolling_percentile(df["jerk"].abs(), self.pct_window)

            # Legacy aliases for backward compatibility
            df["energy_pct"] = df["KE_pct"]
            df["damping_pct"] = df["zeta_pct"]
            df["entropy_pct"] = df["Hs_pct"]

        # 11) regime clustering (Hybrid HMM-SVM for temporal smoothing and discriminative power)
        # Prepare features (similar to original GMM)
        if include_percentiles:
            feature_cols = ["KE_pct", "Re_m_pct", "zeta_pct", "Hs_pct", "PE_pct", "eta_pct"]
        else:
            feature_cols = ["energy", "damping", "entropy", "reynolds", "potential", "eta"]
        df_raw = df[feature_cols]
        df_clean = df_raw.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_clean) == 0:
            df["cluster"] = -1
            df["regime"] = "UNKNOWN"
            df["regime_age_frac"] = 0.0
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(df_clean)

            # Generative: HMM for temporal dependencies
            self.hmm.fit(X)
            hmm_states = self.hmm.predict(X)
            hmm_states_full = np.full(len(df_raw), -1, dtype=int)
            positions = df_raw.index.get_indexer(df_clean.index)
            hmm_states_full[positions] = hmm_states
            df["hmm_state"] = pd.Series(hmm_states_full, index=df.index)

            # Discriminative: SVM to refine HMM states
            self.svm.fit(X, hmm_states)
            svm_predictions = self.svm.predict(X)
            svm_predictions_full = np.full(len(df_raw), -1, dtype=int)
            svm_predictions_full[positions] = svm_predictions
            df["cluster"] = pd.Series(svm_predictions_full, index=df.index)

            # Map to regimes using existing method, but with hybrid clusters
            df["regime"] = self._map_clusters_to_regimes(df, include_percentiles)
            df["regime_age_frac"] = self._compute_regime_age(df["cluster"])

        return df

    def compute_physics_state_from_ohlcv(
        self,
        df: pd.DataFrame,
        include_percentiles: bool = True,
        include_kinematics: bool = True,
        include_flow: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method to compute physics state from OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume (volume optional)
            include_percentiles: Include rolling percentile ranks
            include_kinematics: Include velocity, acceleration, jerk
            include_flow: Include Reynolds number, viscosity (requires volume)

        Returns:
            DataFrame with all Layer-1 sensor columns
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        close_series: pd.Series = df["close"]
        high_series: Optional[pd.Series] = df["high"] if "high" in df.columns else None
        low_series: Optional[pd.Series] = df["low"] if "low" in df.columns else None
        open_series: Optional[pd.Series] = df["open"] if "open" in df.columns else None
        volume_series: Optional[pd.Series] = df["volume"] if "volume" in df.columns else None

        return self.compute_physics_state(
            prices=close_series,
            volume=volume_series,
            high=high_series,
            low=low_series,
            open_price=open_series,
            include_percentiles=include_percentiles,
            include_kinematics=include_kinematics,
            include_flow=include_flow,
        )

    # ---------- core computations ----------

    @staticmethod
    def _rolling_damping(v: pd.Series, window: int) -> pd.Series:
        """
        ζ ~ sigma(returns) / mean(|returns|) over rolling window.

        High ζ = high friction, mean-reverting.
        Low ζ = low friction, trend-following.
        """
        window = int(window)  # Ensure integer for pandas .rolling()
        abs_v = v.abs()
        sigma = v.rolling(window, min_periods=2).std()
        mu = abs_v.rolling(window, min_periods=2).mean()
        zeta = sigma / (mu + 1e-12)
        return zeta.bfill().fillna(0.0)

    @staticmethod
    def _rolling_entropy(v: pd.Series, window: int, bins: int = 20) -> pd.Series:
        """
        Shannon entropy of return distribution in a rolling window.

        H = - Σ p_i log(p_i)
        Normalised by log(bins) to map roughly to [0,1].

        PERFORMANCE: Uses vectorized implementation when available,
        otherwise falls back to pandas .apply() (slower).
        """
        window = int(window)  # Ensure integer for pandas .rolling()
        bins = int(bins)  # Ensure integer for np.histogram()

        # Try to use optimized version
        try:
            from .performance import rolling_entropy_vectorized

            result = rolling_entropy_vectorized(v.values, window, bins)
            return pd.Series(result, index=v.index).bfill().fillna(0.0)
        except ImportError:
            pass

        # Fallback to pandas .apply() (slower but always works)
        def ent(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            n = x.size
            if n < 5:
                return np.nan
            hist, _ = np.histogram(x, bins=bins, density=True)
            p = hist / (hist.sum() + 1e-12)
            p = p[p > 0]
            H = -np.sum(p * np.log(p))
            return float(H / np.log(bins))

        return (
            v.rolling(window, min_periods=5)
            .apply(lambda s: ent(s.values), raw=False)
            .bfill()
            .fillna(0.0)
        )

    @staticmethod
    def _reynolds(v: pd.Series, slow: int, fast: int) -> pd.Series:
        """
        Market Reynolds number: trend / noise.

        Re_m = | <v>_slow | / (sigma_v_fast + eps)
        High Re_m → laminar, low Re_m → turbulent/noisy.
        """
        slow = int(slow)  # Ensure integer for pandas .rolling()
        fast = int(fast)  # Ensure integer for pandas .rolling()
        trend = v.rolling(slow, min_periods=2).mean()
        noise = v.rolling(fast, min_periods=2).std()
        Re = trend.abs() / (noise + 1e-12)
        return Re.bfill().fillna(0.0)

    @staticmethod
    def _potential_energy(v: pd.Series, window: int) -> pd.Series:
        """
        Potential energy proxy: inverse of long-window volatility.

        High PE → compressed / squeezed (stored energy).
        """
        window = int(window)  # Ensure integer for pandas .rolling()
        vol = v.rolling(window, min_periods=5).std()
        pe = 1.0 / (vol + 1e-12)
        return pe.bfill().fillna(0.0)

    @staticmethod
    def _efficiency(energy: pd.Series, potential: pd.Series) -> pd.Series:
        """
        Local energy conversion efficiency: KE / PE.

        High when kinetic energy is high relative to stored potential (PE).
        """
        eta = energy / (potential + 1e-12)
        # Avoid insane spikes
        q = eta.quantile(0.9999) if len(eta) > 0 else 1e6
        eta = eta.clip(lower=0.0, upper=q)
        return eta.fillna(0.0)

    def _liquidity(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Calculate liquidity proxy from OHLCV.

        Liquidity = Volume / (Range * Price)

        Higher = more liquid (big volume, small price move)
        Lower = thin market (small volume, big move)
        """
        bar_range = (high - low).clip(lower=1e-10)
        price_range_pct = bar_range / close
        liquidity = volume / (price_range_pct * close + 1e-10)
        return liquidity.fillna(0.0)

    def _viscosity(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Calculate viscosity (internal friction / resistance to flow).

        Higher viscosity = harder to move price (thick, resistant)
        Lower viscosity = easier to move price (thin, fluid)
        """
        bar_range_pct = (high - low) / close.clip(lower=1e-10)
        avg_volume = volume.rolling(self.lookback).mean().clip(lower=1e-10)
        volume_norm = volume / avg_volume
        viscosity = bar_range_pct / volume_norm.clip(lower=1e-10)
        viscosity = viscosity.rolling(self.lookback).mean()
        return viscosity.fillna(1.0)

    @staticmethod
    def _rolling_percentile(x: pd.Series, window: int) -> pd.Series:
        """
        Rolling percentile (0–1) of current value within last 'window' samples.

        PERFORMANCE: Uses vectorized implementation when available,
        providing significant speedup over pandas .apply().
        """
        window = int(window)  # Ensure integer for pandas .rolling()

        # Try to use optimized version
        try:
            from .performance import rolling_percentile_vectorized

            result = rolling_percentile_vectorized(x.values, window)
            return pd.Series(result, index=x.index).bfill().fillna(0.5)
        except ImportError:
            pass

        # Fallback to pandas .apply() (slower but always works)
        def pct_last(w: pd.Series) -> float:
            val = w.iloc[-1]
            w = w.dropna()
            if w.empty:
                return np.nan
            return float((w <= val).mean())

        return x.rolling(window, min_periods=10).apply(pct_last, raw=False).bfill().fillna(0.5)

    # ---------- clustering & regimes ----------

    def _cluster_regimes(self, df: pd.DataFrame, include_percentiles: bool) -> pd.Series:
        """
        Run GMM on Layer-1 sensors to get discrete clusters.

        Returns:
            pd.Series of ints in [0, n_clusters-1] (or -1 for early NaNs).
        """
        if not include_percentiles:
            # Fallback to simple threshold-based clustering
            return self._simple_cluster(df)

        if not SKLEARN_AVAILABLE:
            # Fallback if sklearn not installed
            return self._simple_cluster(df)

        cols = ["KE_pct", "Re_m_pct", "zeta_pct", "Hs_pct", "PE_pct", "eta_pct"]

        # Check all columns exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return self._simple_cluster(df)

        X = df[cols].values

        # Mask out rows with NaNs (early window warmup)
        mask = np.isfinite(X).all(axis=1)
        clusters = np.full(len(df), -1, dtype=int)

        if mask.sum() < self.n_clusters * 20:
            # Not enough data; fallback to simple
            return self._simple_cluster(df)

        X_valid = X[mask]

        try:
            gmm = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="full",
                random_state=self.random_state,
            )
            labels = gmm.fit_predict(X_valid)
            clusters[mask] = labels
        except Exception:
            return self._simple_cluster(df)

        return pd.Series(clusters, index=df.index)

    def _simple_cluster(self, df: pd.DataFrame) -> pd.Series:
        """
        Fallback threshold-based clustering when GMM unavailable.
        """
        clusters = np.ones(len(df), dtype=int)  # Default to CRITICAL (1)

        if "KE_pct" not in df.columns or "zeta_pct" not in df.columns:
            return pd.Series(clusters, index=df.index)

        ke_pct = df["KE_pct"].values
        zeta_pct = df["zeta_pct"].values
        jerk_pct = df.get("jerk_pct", pd.Series(np.zeros(len(df)))).values

        for i in range(len(df)):
            if np.isnan(ke_pct[i]) or np.isnan(zeta_pct[i]):
                clusters[i] = 1  # CRITICAL
            elif ke_pct[i] > 0.75 and zeta_pct[i] < 0.25:
                clusters[i] = 0  # UNDERDAMPED
            elif zeta_pct[i] > 0.75 and ke_pct[i] < 0.25:
                clusters[i] = 2  # OVERDAMPED
            elif ke_pct[i] > 0.8 and jerk_pct[i] > 0.8:
                clusters[i] = 3  # BREAKOUT
            else:
                clusters[i] = 1  # CRITICAL

        return pd.Series(clusters, index=df.index)

    def _map_clusters_to_regimes(self, df: pd.DataFrame, include_percentiles: bool) -> pd.Series:
        """
        Map integer clusters to physical regime labels.

        Heuristic mapping based on cluster means:
            - OVERDAMPED   : high zeta_pct, low Re_m_pct
            - UNDERDAMPED  : low zeta_pct, high Re_m_pct
            - LAMINAR      : high Re_m_pct, low Hs_pct (ordered trend)
            - BREAKOUT     : high KE_pct + high eta_pct + high Hs_pct
        """
        clusters = df["cluster"].values
        regimes = np.array(["critical"] * len(df), dtype=object)

        if not include_percentiles or "KE_pct" not in df.columns:
            # Simple mapping
            regime_map = {
                0: "underdamped",
                1: "critical",
                2: "overdamped",
                3: "breakout",
                -1: "critical",
            }
            for i, c in enumerate(clusters):
                regimes[i] = regime_map.get(int(c), "critical")
            return pd.Series(regimes, index=df.index)

        valid_mask = clusters >= 0
        if not valid_mask.any():
            return pd.Series(regimes, index=df.index)

        tmp = df.loc[
            valid_mask, ["cluster", "KE_pct", "Re_m_pct", "zeta_pct", "Hs_pct", "eta_pct"]
        ].copy()

        if tmp.empty:
            return pd.Series(regimes, index=df.index)

        stats = (
            tmp.groupby("cluster")
            .mean()[["KE_pct", "Re_m_pct", "zeta_pct", "Hs_pct", "eta_pct"]]
            .copy()
        )

        # Pre-compute scores per cluster
        labels_for_cluster: dict[int, str] = {}
        for c, row in stats.iterrows():
            ke = row["KE_pct"]
            Re = row["Re_m_pct"]
            zeta = row["zeta_pct"]
            Hs = row["Hs_pct"]
            eta = row["eta_pct"]

            # Simple rule priority order
            if zeta > 0.66 and Re < 0.5:
                lab = "overdamped"
            elif Re > 0.66 and zeta < 0.33:
                lab = "underdamped"
            elif Re > 0.66 and Hs < 0.33:
                lab = "laminar"
            elif ke > 0.66 and eta > 0.5 and Hs > 0.5:
                lab = "breakout"
            else:
                # Fallback: choose the strongest tendency
                scores = {
                    "overdamped": zeta - Re,
                    "underdamped": Re - zeta,
                    "laminar": Re - Hs,
                    "breakout": ke + eta + Hs,
                }
                lab = max(scores, key=lambda k: scores[k])

            labels_for_cluster[int(c)] = lab

        # Apply mapping
        for i, c in enumerate(clusters):
            if c >= 0:
                regimes[i] = labels_for_cluster.get(int(c), "critical")

        return pd.Series(regimes, index=df.index)

    @staticmethod
    def _compute_regime_age(cluster_series: pd.Series) -> pd.Series:
        """
        Normalised age in current cluster/regime: 0 at switch, →1 as streak length grows.

        Works even when cluster = -1 initially.
        """
        clusters = cluster_series.values
        age = np.zeros(len(clusters), dtype=float)
        current_cluster = None
        run_length = 0

        for i, c in enumerate(clusters):
            if c == current_cluster:
                run_length += 1
            else:
                current_cluster = c
                run_length = 1
            age[i] = run_length

        # Normalise by rolling max age to [0,1]
        age_series = pd.Series(age, index=cluster_series.index)
        max_age = age_series.rolling(500, min_periods=1).max()
        age_frac = age_series / (max_age + 1e-9)
        return age_frac.fillna(0.0)

    # ---------- legacy methods for backward compatibility ----------

    def calculate_energy(self, prices: pd.Series) -> pd.Series:
        """Calculate kinetic energy from price momentum."""
        velocity = prices.pct_change().fillna(0.0)
        energy = 0.5 * self.mass * velocity**2
        return energy.clip(lower=0.0)

    def calculate_damping(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> pd.Series:
        """Calculate damping coefficient (friction)."""
        returns = prices.pct_change()
        volatility = returns.rolling(self.lookback).std()
        mean_abs_return = returns.abs().rolling(self.lookback).mean()
        damping = volatility / (mean_abs_return + 1e-10)
        return damping.clip(lower=0.0).fillna(0.0)

    def calculate_entropy(self, prices: pd.Series, bins: int = 10) -> pd.Series:
        """Calculate Shannon entropy of price distribution."""
        returns = prices.pct_change().dropna()
        return self._rolling_entropy(returns, self.entropy_window, bins).reindex(
            prices.index, fill_value=0.0
        )

    def calculate_acceleration(self, prices: pd.Series) -> pd.Series:
        """Calculate acceleration (second derivative of price)."""
        velocity = prices.pct_change()
        acceleration = velocity.diff()
        return acceleration.fillna(0.0)

    def calculate_jerk(self, prices: pd.Series) -> pd.Series:
        """Calculate jerk (third derivative - rate of acceleration change)."""
        acceleration = self.calculate_acceleration(prices)
        jerk = acceleration.diff()
        return jerk.fillna(0.0)

    def classify_regime(
        self, energy: float, damping: float, history_energy: pd.Series, history_damping: pd.Series
    ) -> RegimeType:
        """Classify market regime using rolling percentiles."""
        if len(history_energy.dropna()) < 10 or len(history_damping.dropna()) < 10:
            return RegimeType.CRITICAL

        energy_75pct = np.percentile(history_energy.dropna(), 75)
        damping_25pct = np.percentile(history_damping.dropna(), 25)
        damping_75pct = np.percentile(history_damping.dropna(), 75)

        if energy > energy_75pct and damping < damping_25pct:
            return RegimeType.UNDERDAMPED
        elif damping_25pct <= damping <= damping_75pct:
            return RegimeType.CRITICAL
        else:
            return RegimeType.OVERDAMPED


# Standalone functions for convenience
def calculate_energy(prices: pd.Series, mass: float = 1.0) -> pd.Series:
    """Calculate kinetic energy from prices."""
    engine = PhysicsEngine(mass=mass)
    return engine.calculate_energy(prices)


def calculate_damping(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate damping coefficient from prices."""
    engine = PhysicsEngine(lookback=lookback)
    return engine.calculate_damping(prices)


def calculate_entropy(prices: pd.Series, lookback: int = 20, bins: int = 10) -> pd.Series:
    """Calculate Shannon entropy from prices."""
    engine = PhysicsEngine(lookback=lookback)
    return engine.calculate_entropy(prices, bins=bins)
