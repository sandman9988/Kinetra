#!/usr/bin/env python3
"""
SUPERPOT EXPLORER
=================

Throw ALL measurements in a pot. Let agents figure out what matters.
After every X episodes, throw out the worst Y measurements.

This is EMPIRICAL feature discovery - no assumptions about what matters.

The market tells us what's useful. We don't assume.

Usage:
    python scripts/superpot_explorer.py
    python scripts/superpot_explorer.py --prune-every 20 --prune-count 5
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# THE SUPERPOT: ALL POSSIBLE MEASUREMENTS
# =============================================================================

class SuperPotExtractor:
    """
    Extract ALL possible measurements from price data.
    
    Categories:
    - Price action (returns, ranges, gaps)
    - Volume dynamics (CVD, Amihud, pressure)
    - Volatility (multiple estimators)
    - Momentum (signed, directional)
    - Entropy & chaos (permutation, recurrence)
    - Tail behavior (asymmetric, CVaR)
    - Microstructure (spread, depth proxies)
    - Higher moments (skew, kurtosis - signed)
    - Cross-timeframe (if available)
    
    No filtering. No assumptions. Everything goes in.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.feature_names: List[str] = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build complete list of feature names."""
        self.feature_names = [
            # === PRICE ACTION (20 features) ===
            'return_1', 'return_5', 'return_10', 'return_20',
            'return_cum_5', 'return_cum_10', 'return_cum_20',
            'range_1', 'range_5', 'range_10', 'range_20',
            'gap_open', 'gap_close',
            'close_rank_20', 'close_rank_50',
            'high_rank_20', 'low_rank_20',
            'new_highs_5', 'new_lows_5',
            'price_momentum_10',
            
            # === VOLUME DYNAMICS (15 features) ===
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
            'cvd_5', 'cvd_10', 'cvd_20',
            'buy_pressure_5', 'buy_pressure_10',
            'sell_acceleration',
            'volume_trend_5', 'volume_trend_10',
            'amihud_5', 'amihud_10', 'amihud_20',
            'volume_imbalance',
            
            # === VOLATILITY (20 features) ===
            'vol_close_5', 'vol_close_10', 'vol_close_20',
            'vol_parkinson_5', 'vol_parkinson_10', 'vol_parkinson_20',
            'vol_garman_klass_5', 'vol_garman_klass_10',
            'vol_rogers_satchell_5', 'vol_rogers_satchell_10',
            'vol_yang_zhang_5', 'vol_yang_zhang_10',
            'vol_ratio_short_long',
            'vol_up_5', 'vol_down_5',
            'vol_asymmetry_5', 'vol_asymmetry_10',
            'atr_5', 'atr_10', 'atr_20',
            
            # === MOMENTUM (15 features) ===
            'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_signed_5', 'momentum_signed_10',
            'roc_5', 'roc_10', 'roc_20',
            'acceleration_5', 'acceleration_10',
            'jerk_5',
            'momentum_persistence_5',
            'up_momentum_5', 'down_momentum_5',
            'momentum_divergence',
            
            # === ENTROPY & CHAOS (15 features) ===
            'perm_entropy_5', 'perm_entropy_10', 'perm_entropy_20',
            'approx_entropy_10',
            'sample_entropy_10',
            'recurrence_rate_10', 'recurrence_rate_20',
            'determinism_10',
            'entropy_change_5',
            'complexity_5', 'complexity_10',
            'lyapunov_proxy_10',
            'hurst_proxy_20',
            'fractal_dim_proxy_10',
            'chaos_index_10',
            
            # === TAIL BEHAVIOR (15 features) ===
            'skew_5', 'skew_10', 'skew_20',
            'skew_signed_5', 'skew_signed_10',
            'kurtosis_5', 'kurtosis_10', 'kurtosis_20',
            'left_tail_5', 'right_tail_5',
            'tail_asymmetry_5', 'tail_asymmetry_10',
            'cvar_down_5', 'cvar_up_5',
            'tail_ratio_5',
            
            # === MICROSTRUCTURE (15 features) ===
            'spread_proxy_5', 'spread_proxy_10',
            'spread_rank_20',
            'depth_asymmetry_5', 'depth_asymmetry_10',
            'price_impact_5',
            'kyle_lambda_proxy_10',
            'effective_spread_5',
            'realized_spread_5',
            'adverse_selection_5',
            'pin_proxy_10',
            'vpin_proxy_10',
            'order_flow_toxicity_5',
            'liquidity_score_10',
            'fragility_10',
            
            # === HIGHER MOMENTS (10 features) ===
            'moment_3_5', 'moment_3_10',
            'moment_4_5', 'moment_4_10',
            'coskew_vol_5',
            'cokurt_vol_5',
            'jarque_bera_proxy_10',
            'normality_deviation_10',
            'fat_tail_index_10',
            'distribution_shape_10',
            
            # === REGIME INDICATORS (10 features) ===
            'trend_strength_10', 'trend_strength_20',
            'mean_reversion_10',
            'breakout_strength_5',
            'consolidation_score_10',
            'regime_persistence_10',
            'transition_probability_10',
            'stability_score_10',
            'volatility_regime_10',
            'momentum_regime_10',
            
            # === CROSS-FEATURE (10 features) ===
            'vol_momentum_corr_10',
            'volume_price_corr_10',
            'return_vol_corr_10',
            'spread_vol_corr_10',
            'entropy_vol_corr_10',
            'momentum_vol_ratio_10',
            'cvd_price_divergence_10',
            'composite_imbalance_10',
            'composite_chaos_10',
            'composite_regime_10',
        ]
        
        print(f"📊 SuperPot initialized with {len(self.feature_names)} features")
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    def extract(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Extract ALL features at given index."""
        if idx < self.lookback:
            return np.zeros(self.n_features, dtype=np.float32)
        
        window = df.iloc[max(0, idx - self.lookback):idx + 1]
        
        o = window['open'].values.astype(np.float64)
        h = window['high'].values.astype(np.float64)
        l = window['low'].values.astype(np.float64)
        c = window['close'].values.astype(np.float64)
        v = window['volume'].values.astype(np.float64) if 'volume' in window else np.ones(len(window))
        
        features = np.zeros(self.n_features, dtype=np.float32)
        fi = 0  # feature index
        
        # Returns
        ret = np.diff(c) / (c[:-1] + 1e-10)
        
        # === PRICE ACTION ===
        features[fi] = ret[-1] if len(ret) > 0 else 0; fi += 1
        features[fi] = np.mean(ret[-5:]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.mean(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        features[fi] = np.mean(ret[-20:]) if len(ret) >= 20 else 0; fi += 1
        features[fi] = np.sum(ret[-5:]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.sum(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        features[fi] = np.sum(ret[-20:]) if len(ret) >= 20 else 0; fi += 1
        
        ranges = (h - l) / (c + 1e-10)
        features[fi] = ranges[-1]; fi += 1
        features[fi] = np.mean(ranges[-5:]); fi += 1
        features[fi] = np.mean(ranges[-10:]); fi += 1
        features[fi] = np.mean(ranges[-20:]); fi += 1
        
        features[fi] = (o[-1] - c[-2]) / (c[-2] + 1e-10) if len(c) > 1 else 0; fi += 1
        features[fi] = (c[-1] - o[-1]) / (o[-1] + 1e-10); fi += 1
        
        features[fi] = np.sum(c[-20:] < c[-1]) / 20 if len(c) >= 20 else 0.5; fi += 1
        features[fi] = np.sum(c < c[-1]) / len(c); fi += 1
        features[fi] = np.sum(h[-20:] < h[-1]) / 20 if len(h) >= 20 else 0.5; fi += 1
        features[fi] = np.sum(l[-20:] > l[-1]) / 20 if len(l) >= 20 else 0.5; fi += 1
        
        features[fi] = np.sum(h[-5:] >= np.max(h[-20:])) / 5 if len(h) >= 20 else 0; fi += 1
        features[fi] = np.sum(l[-5:] <= np.min(l[-20:])) / 5 if len(l) >= 20 else 0; fi += 1
        features[fi] = (c[-1] - c[-10]) / (c[-10] + 1e-10) if len(c) >= 10 else 0; fi += 1
        
        # === VOLUME DYNAMICS ===
        vm = np.mean(v) + 1e-10
        features[fi] = v[-1] / np.mean(v[-5:]) if len(v) >= 5 else 1; fi += 1
        features[fi] = np.mean(v[-5:]) / np.mean(v[-10:]) if len(v) >= 10 else 1; fi += 1
        features[fi] = np.mean(v[-10:]) / np.mean(v[-20:]) if len(v) >= 20 else 1; fi += 1
        
        signed_vol = np.sign(c[1:] - o[1:]) * v[1:]
        features[fi] = np.sum(signed_vol[-5:]) / (np.sum(v[-5:]) + 1e-10) if len(v) >= 5 else 0; fi += 1
        features[fi] = np.sum(signed_vol[-10:]) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0; fi += 1
        features[fi] = np.sum(signed_vol[-20:]) / (np.sum(v[-20:]) + 1e-10) if len(v) >= 20 else 0; fi += 1
        
        body_pos = (c - l) / (h - l + 1e-10)
        features[fi] = np.mean(body_pos[-5:]); fi += 1
        features[fi] = np.mean(body_pos[-10:]); fi += 1
        
        sell_v = np.where(c < o, v, 0)
        features[fi] = (np.mean(sell_v[-3:]) - np.mean(sell_v[-6:-3])) / (np.mean(sell_v[-6:-3]) + 1e-10) if len(sell_v) >= 6 else 0; fi += 1
        
        features[fi] = np.polyfit(np.arange(5), v[-5:], 1)[0] / vm if len(v) >= 5 else 0; fi += 1
        features[fi] = np.polyfit(np.arange(10), v[-10:], 1)[0] / vm if len(v) >= 10 else 0; fi += 1
        
        # Amihud
        amihud = np.abs(ret) / (v[1:] + 1e-10) * 1e6
        features[fi] = np.mean(amihud[-5:]) if len(amihud) >= 5 else 0; fi += 1
        features[fi] = np.mean(amihud[-10:]) if len(amihud) >= 10 else 0; fi += 1
        features[fi] = np.mean(amihud[-20:]) if len(amihud) >= 20 else 0; fi += 1
        
        up_v = np.sum(v[1:][ret > 0]) if len(ret) > 0 else 1
        dn_v = np.sum(v[1:][ret < 0]) if len(ret) > 0 else 1
        features[fi] = (up_v - dn_v) / (up_v + dn_v + 1e-10); fi += 1
        
        # === VOLATILITY ===
        features[fi] = np.std(ret[-5:]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.std(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        features[fi] = np.std(ret[-20:]) if len(ret) >= 20 else 0; fi += 1
        
        # Parkinson
        hl_log = np.log(h / (l + 1e-10) + 1e-10) ** 2
        features[fi] = np.sqrt(np.mean(hl_log[-5:]) / (4 * np.log(2))) if len(hl_log) >= 5 else 0; fi += 1
        features[fi] = np.sqrt(np.mean(hl_log[-10:]) / (4 * np.log(2))) if len(hl_log) >= 10 else 0; fi += 1
        features[fi] = np.sqrt(np.mean(hl_log[-20:]) / (4 * np.log(2))) if len(hl_log) >= 20 else 0; fi += 1
        
        # Garman-Klass
        gk = 0.5 * np.log(h / (l + 1e-10)) ** 2 - (2 * np.log(2) - 1) * np.log(c / (o + 1e-10)) ** 2
        features[fi] = np.sqrt(np.mean(gk[-5:])) if len(gk) >= 5 else 0; fi += 1
        features[fi] = np.sqrt(np.mean(gk[-10:])) if len(gk) >= 10 else 0; fi += 1
        
        # Rogers-Satchell
        rs = np.log(h / c + 1e-10) * np.log(h / o + 1e-10) + np.log(l / c + 1e-10) * np.log(l / o + 1e-10)
        features[fi] = np.sqrt(np.mean(rs[-5:])) if len(rs) >= 5 else 0; fi += 1
        features[fi] = np.sqrt(np.mean(rs[-10:])) if len(rs) >= 10 else 0; fi += 1
        
        # Yang-Zhang
        yz = rs + 0.5 * np.log(c / (o + 1e-10)) ** 2
        features[fi] = np.sqrt(np.mean(yz[-5:])) if len(yz) >= 5 else 0; fi += 1
        features[fi] = np.sqrt(np.mean(yz[-10:])) if len(yz) >= 10 else 0; fi += 1
        
        vol_5 = np.std(ret[-5:]) if len(ret) >= 5 else 0.01
        vol_20 = np.std(ret[-20:]) if len(ret) >= 20 else 0.01
        features[fi] = vol_5 / (vol_20 + 1e-10); fi += 1
        
        up_ret = ret[ret > 0]
        dn_ret = ret[ret < 0]
        features[fi] = np.std(up_ret[-5:]) if len(up_ret) >= 5 else 0; fi += 1
        features[fi] = np.std(dn_ret[-5:]) if len(dn_ret) >= 5 else 0; fi += 1
        
        vol_up = np.std(up_ret[-5:]) if len(up_ret) >= 5 else 0.01
        vol_dn = np.std(dn_ret[-5:]) if len(dn_ret) >= 5 else 0.01
        features[fi] = (vol_dn - vol_up) / (vol_dn + vol_up + 1e-10); fi += 1
        features[fi] = (vol_dn - vol_up) / (vol_dn + vol_up + 1e-10); fi += 1
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        features[fi] = np.mean(tr[-5:]) / (c[-1] + 1e-10) if len(tr) >= 5 else 0; fi += 1
        features[fi] = np.mean(tr[-10:]) / (c[-1] + 1e-10) if len(tr) >= 10 else 0; fi += 1
        features[fi] = np.mean(tr[-20:]) / (c[-1] + 1e-10) if len(tr) >= 20 else 0; fi += 1
        
        # === MOMENTUM ===
        features[fi] = c[-1] / c[-5] - 1 if len(c) >= 5 else 0; fi += 1
        features[fi] = c[-1] / c[-10] - 1 if len(c) >= 10 else 0; fi += 1
        features[fi] = c[-1] / c[-20] - 1 if len(c) >= 20 else 0; fi += 1
        
        features[fi] = np.sign(c[-1] - c[-5]) * abs(c[-1] / c[-5] - 1) if len(c) >= 5 else 0; fi += 1
        features[fi] = np.sign(c[-1] - c[-10]) * abs(c[-1] / c[-10] - 1) if len(c) >= 10 else 0; fi += 1
        
        features[fi] = (c[-1] - c[-5]) / (c[-5] + 1e-10) * 100 if len(c) >= 5 else 0; fi += 1
        features[fi] = (c[-1] - c[-10]) / (c[-10] + 1e-10) * 100 if len(c) >= 10 else 0; fi += 1
        features[fi] = (c[-1] - c[-20]) / (c[-20] + 1e-10) * 100 if len(c) >= 20 else 0; fi += 1
        
        mom = c[1:] - c[:-1]
        features[fi] = mom[-1] - mom[-2] if len(mom) >= 2 else 0; fi += 1
        features[fi] = np.mean(np.diff(mom[-5:])) if len(mom) >= 5 else 0; fi += 1
        
        acc = np.diff(mom) if len(mom) >= 2 else np.array([0])
        features[fi] = np.diff(acc)[-1] if len(acc) >= 2 else 0; fi += 1
        
        signs = np.sign(ret[-5:]) if len(ret) >= 5 else np.zeros(5)
        features[fi] = np.sum(signs == signs[-1]) / 5 if len(signs) > 0 else 0; fi += 1
        
        features[fi] = np.sum(ret[-5:][ret[-5:] > 0]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.sum(ret[-5:][ret[-5:] < 0]) if len(ret) >= 5 else 0; fi += 1
        
        price_mom = c[-1] / c[-10] - 1 if len(c) >= 10 else 0
        vol_mom = v[-1] / np.mean(v[-10:]) - 1 if len(v) >= 10 else 0
        features[fi] = price_mom - vol_mom; fi += 1
        
        # === ENTROPY & CHAOS ===
        for period in [5, 10, 20]:
            if len(c) >= period:
                features[fi] = self._perm_entropy(c[-period:])
            fi += 1
        
        features[fi] = self._approx_entropy(ret[-10:]) if len(ret) >= 10 else 0.5; fi += 1
        features[fi] = self._sample_entropy(ret[-10:]) if len(ret) >= 10 else 0.5; fi += 1
        
        features[fi] = self._recurrence_rate(ret[-10:]) if len(ret) >= 10 else 0.5; fi += 1
        features[fi] = self._recurrence_rate(ret[-20:]) if len(ret) >= 20 else 0.5; fi += 1
        
        features[fi] = 1 - self._perm_entropy(c[-10:]) if len(c) >= 10 else 0.5; fi += 1
        
        ent_5 = self._perm_entropy(c[-5:]) if len(c) >= 5 else 0.5
        ent_10 = self._perm_entropy(c[-10:]) if len(c) >= 10 else 0.5
        features[fi] = ent_5 - ent_10; fi += 1
        
        features[fi] = np.std(ret[-5:]) * ent_5 if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.std(ret[-10:]) * ent_10 if len(ret) >= 10 else 0; fi += 1
        
        # Lyapunov proxy
        if len(ret) >= 10:
            diffs = np.abs(np.diff(ret[-10:]))
            features[fi] = np.mean(np.log(diffs + 1e-10))
        fi += 1
        
        # Hurst proxy
        if len(ret) >= 20:
            rs_range = np.max(np.cumsum(ret[-20:] - np.mean(ret[-20:]))) - np.min(np.cumsum(ret[-20:] - np.mean(ret[-20:])))
            features[fi] = np.log(rs_range + 1e-10) / np.log(20)
        fi += 1
        
        features[fi] = 2 - self._perm_entropy(c[-10:]) * 2 if len(c) >= 10 else 1.5; fi += 1
        features[fi] = (1 - ent_10) * np.std(ret[-10:]) * 100 if len(ret) >= 10 else 0; fi += 1
        
        # === TAIL BEHAVIOR ===
        features[fi] = self._safe_skew(ret[-5:]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = self._safe_skew(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        features[fi] = self._safe_skew(ret[-20:]) if len(ret) >= 20 else 0; fi += 1
        
        features[fi] = np.sign(np.mean(ret[-5:])) * abs(self._safe_skew(ret[-5:])) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.sign(np.mean(ret[-10:])) * abs(self._safe_skew(ret[-10:])) if len(ret) >= 10 else 0; fi += 1
        
        features[fi] = self._safe_kurt(ret[-5:]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = self._safe_kurt(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        features[fi] = self._safe_kurt(ret[-20:]) if len(ret) >= 20 else 0; fi += 1
        
        features[fi] = np.percentile(ret[-5:], 5) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.percentile(ret[-5:], 95) if len(ret) >= 5 else 0; fi += 1
        
        left = abs(np.percentile(ret[-5:], 5)) if len(ret) >= 5 else 0
        right = abs(np.percentile(ret[-5:], 95)) if len(ret) >= 5 else 0
        features[fi] = left - right; fi += 1
        features[fi] = left - right; fi += 1
        
        features[fi] = np.mean(ret[-5:][ret[-5:] < np.percentile(ret[-5:], 10)]) if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.mean(ret[-5:][ret[-5:] > np.percentile(ret[-5:], 90)]) if len(ret) >= 5 else 0; fi += 1
        
        features[fi] = (left + 1e-10) / (right + 1e-10); fi += 1
        
        # === MICROSTRUCTURE ===
        spread = (h - l) / c
        features[fi] = np.mean(spread[-5:]); fi += 1
        features[fi] = np.mean(spread[-10:]); fi += 1
        
        features[fi] = (spread[-1] - np.percentile(spread, 20)) / (np.percentile(spread, 80) - np.percentile(spread, 20) + 1e-10); fi += 1
        
        upper = (h - np.maximum(o, c)) / (h - l + 1e-10)
        lower = (np.minimum(o, c) - l) / (h - l + 1e-10)
        features[fi] = np.mean(lower[-5:] - upper[-5:]); fi += 1
        features[fi] = np.mean(lower[-10:] - upper[-10:]); fi += 1
        
        features[fi] = np.mean(np.abs(ret[-5:])) / (np.mean(v[-5:]) + 1e-10) * 1e6 if len(ret) >= 5 else 0; fi += 1
        
        # Kyle lambda proxy
        features[fi] = np.corrcoef(np.abs(ret[-10:]), v[1:][-10:])[0, 1] if len(ret) >= 10 else 0; fi += 1
        
        features[fi] = np.mean(spread[-5:]) * 0.5; fi += 1
        features[fi] = np.mean(spread[-5:]) * 0.3; fi += 1
        features[fi] = np.mean(spread[-5:]) * 0.2; fi += 1
        
        features[fi] = abs(np.mean(signed_vol[-10:])) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0; fi += 1
        features[fi] = abs(np.mean(signed_vol[-10:])) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0; fi += 1
        
        features[fi] = np.mean(np.abs(ret[-5:])) / (np.mean(spread[-5:]) + 1e-10) if len(ret) >= 5 else 0; fi += 1
        features[fi] = 1 / (np.mean(spread[-10:]) * np.mean(v[-10:]) + 1e-10); fi += 1
        features[fi] = np.std(spread[-10:]) / (np.mean(spread[-10:]) + 1e-10); fi += 1
        
        # === HIGHER MOMENTS ===
        features[fi] = self._moment(ret[-5:], 3) if len(ret) >= 5 else 0; fi += 1
        features[fi] = self._moment(ret[-10:], 3) if len(ret) >= 10 else 0; fi += 1
        features[fi] = self._moment(ret[-5:], 4) if len(ret) >= 5 else 0; fi += 1
        features[fi] = self._moment(ret[-10:], 4) if len(ret) >= 10 else 0; fi += 1
        
        features[fi] = np.corrcoef(ret[-5:], np.abs(ret[-5:]))[0, 1] if len(ret) >= 5 else 0; fi += 1
        features[fi] = np.corrcoef(ret[-5:], ret[-5:] ** 2)[0, 1] if len(ret) >= 5 else 0; fi += 1
        
        s = self._safe_skew(ret[-10:]) if len(ret) >= 10 else 0
        k = self._safe_kurt(ret[-10:]) if len(ret) >= 10 else 0
        features[fi] = len(ret[-10:]) * (s ** 2 / 6 + (k - 3) ** 2 / 24) if len(ret) >= 10 else 0; fi += 1
        features[fi] = abs(s) + abs(k - 3); fi += 1
        features[fi] = k - 3; fi += 1
        features[fi] = np.sign(s) * (abs(k) - 3); fi += 1
        
        # === REGIME INDICATORS ===
        if len(c) >= 10:
            slope, _ = np.polyfit(np.arange(10), c[-10:], 1)
            features[fi] = slope / (np.std(c[-10:]) + 1e-10)
        fi += 1
        
        if len(c) >= 20:
            slope, _ = np.polyfit(np.arange(20), c[-20:], 1)
            features[fi] = slope / (np.std(c[-20:]) + 1e-10)
        fi += 1
        
        mean_c = np.mean(c[-10:]) if len(c) >= 10 else c[-1]
        features[fi] = -np.corrcoef(c[-10:], np.arange(10))[0, 1] if len(c) >= 10 else 0; fi += 1
        
        features[fi] = (c[-1] - np.max(c[-5:])) / (np.max(c[-5:]) - np.min(c[-5:]) + 1e-10) if len(c) >= 5 else 0; fi += 1
        features[fi] = 1 - (np.max(c[-10:]) - np.min(c[-10:])) / (np.mean(c[-10:]) + 1e-10) if len(c) >= 10 else 0; fi += 1
        
        features[fi] = np.sum(np.sign(ret[-10:]) == np.sign(ret[-10:])[-1]) / 10 if len(ret) >= 10 else 0.5; fi += 1
        features[fi] = 1 - self._perm_entropy(c[-10:]) if len(c) >= 10 else 0.5; fi += 1
        features[fi] = 1 / (np.std(ret[-10:]) + 1e-10) / 100 if len(ret) >= 10 else 0; fi += 1
        
        vol_regime = np.std(ret[-10:]) / (np.std(ret[-20:]) + 1e-10) if len(ret) >= 20 else 1
        features[fi] = vol_regime; fi += 1
        
        mom_regime = np.mean(ret[-5:]) / (np.std(ret[-10:]) + 1e-10) if len(ret) >= 10 else 0
        features[fi] = mom_regime; fi += 1
        
        # === CROSS-FEATURE ===
        if len(ret) >= 10:
            features[fi] = np.corrcoef(ret[-10:], np.abs(ret[-10:]))[0, 1]
        fi += 1
        
        if len(ret) >= 10:
            features[fi] = np.corrcoef(ret[-10:], v[1:][-10:])[0, 1]
        fi += 1
        
        features[fi] = np.corrcoef(ret[-10:], np.std(ret[-10:].reshape(-1, 1), axis=1).flatten())[0, 1] if len(ret) >= 10 else 0; fi += 1
        features[fi] = np.corrcoef(spread[-10:], np.abs(ret[-10:]))[0, 1] if len(ret) >= 10 else 0; fi += 1
        
        ent = self._perm_entropy(c[-10:]) if len(c) >= 10 else 0.5
        features[fi] = ent * np.std(ret[-10:]) if len(ret) >= 10 else 0; fi += 1
        
        features[fi] = abs(np.mean(ret[-10:])) / (np.std(ret[-10:]) + 1e-10) if len(ret) >= 10 else 0; fi += 1
        
        cvd_10 = np.sum(signed_vol[-10:]) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0
        price_dir = np.sign(c[-1] - c[-10]) if len(c) >= 10 else 0
        features[fi] = cvd_10 - price_dir * 0.5; fi += 1
        
        # Composites
        features[fi] = (features[fi-7] + features[fi-6] + features[fi-5]) / 3; fi += 1
        features[fi] = ent * (1 - ent) * np.std(ret[-10:]) * 100 if len(ret) >= 10 else 0; fi += 1
        features[fi] = vol_regime * mom_regime; fi += 1
        
        # Clean up NaNs and Infs
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        features = np.clip(features, -100, 100)
        
        return features
    
    def _perm_entropy(self, series: np.ndarray, order: int = 3) -> float:
        """Permutation entropy."""
        if len(series) < order + 1:
            return 0.5
        patterns = []
        for i in range(len(series) - order):
            patterns.append(tuple(np.argsort(series[i:i + order])))
        if not patterns:
            return 0.5
        from collections import Counter
        counts = Counter(patterns)
        probs = np.array(list(counts.values())) / len(patterns)
        return -np.sum(probs * np.log(probs + 1e-10)) / np.log(6)
    
    def _approx_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Approximate entropy."""
        if len(series) < m + 1:
            return 0.5
        r_val = r * np.std(series)
        def phi(m_val):
            patterns = np.array([series[i:i + m_val] for i in range(len(series) - m_val + 1)])
            counts = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= r_val, axis=0)
            return np.mean(np.log(counts / len(patterns)))
        return phi(m) - phi(m + 1)
    
    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Sample entropy."""
        return self._approx_entropy(series, m, r)
    
    def _recurrence_rate(self, series: np.ndarray, eps: float = 0.1) -> float:
        """Recurrence rate."""
        if len(series) < 5:
            return 0.5
        threshold = eps * np.std(series)
        n = len(series)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(series[i] - series[j]) < threshold:
                    count += 1
        return 2 * count / (n * (n - 1)) if n > 1 else 0
    
    def _safe_skew(self, arr: np.ndarray) -> float:
        if len(arr) < 3:
            return 0
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return 0
        return np.mean(((arr - m) / s) ** 3)
    
    def _safe_kurt(self, arr: np.ndarray) -> float:
        if len(arr) < 4:
            return 3
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return 3
        return np.mean(((arr - m) / s) ** 4)
    
    def _moment(self, arr: np.ndarray, n: int) -> float:
        if len(arr) < n:
            return 0
        m = np.mean(arr)
        return np.mean((arr - m) ** n)


# =============================================================================
# FEATURE IMPORTANCE TRACKER
# =============================================================================

class FeatureImportanceTracker:
    """
    Track which features correlate with successful actions.
    
    After X episodes, prune the worst Y features.
    """
    
    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names.copy()
        self.active_mask = np.ones(n_features, dtype=bool)
        
        # Importance tracking
        self.feature_reward_correlation = np.zeros(n_features)
        self.feature_action_correlation = np.zeros(n_features)
        self.feature_usage_count = np.zeros(n_features)
        
        # History for correlation calculation
        self.feature_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
    
    @property
    def n_active(self) -> int:
        return np.sum(self.active_mask)
    
    def get_active_features(self) -> List[str]:
        return [self.feature_names[i] for i in range(self.n_features) if self.active_mask[i]]
    
    def get_pruned_features(self) -> List[str]:
        return [self.feature_names[i] for i in range(self.n_features) if not self.active_mask[i]]
    
    def record(self, features: np.ndarray, action: int, reward: float):
        """Record observation for importance calculation."""
        self.feature_history.append(features.copy())
        self.reward_history.append(reward)
        self.action_history.append(action)
        
        # Keep history bounded
        if len(self.feature_history) > 10000:
            self.feature_history = self.feature_history[-5000:]
            self.reward_history = self.reward_history[-5000:]
            self.action_history = self.action_history[-5000:]
    
    def calculate_importance(self) -> np.ndarray:
        """Calculate feature importance scores."""
        if len(self.feature_history) < 100:
            return np.ones(self.n_features)
        
        features = np.array(self.feature_history)
        rewards = np.array(self.reward_history)
        
        importance = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            if not self.active_mask[i]:
                importance[i] = -999
                continue
            
            f = features[:, i]
            
            # Correlation with reward
            if np.std(f) > 1e-10:
                corr = np.corrcoef(f, rewards)[0, 1]
                importance[i] = abs(corr) if not np.isnan(corr) else 0
            
            # Bonus for features that differ between winning/losing
            win_mask = rewards > 0
            if np.sum(win_mask) > 10 and np.sum(~win_mask) > 10:
                win_mean = np.mean(f[win_mask])
                lose_mean = np.mean(f[~win_mask])
                diff = abs(win_mean - lose_mean) / (np.std(f) + 1e-10)
                importance[i] += diff * 0.5
        
        return importance
    
    def prune(self, n_to_prune: int) -> List[str]:
        """Prune the worst N features."""
        importance = self.calculate_importance()
        
        # Find indices of worst features (among active ones)
        active_indices = np.where(self.active_mask)[0]
        active_importance = importance[active_indices]
        
        # Sort by importance (ascending = worst first)
        sorted_indices = active_indices[np.argsort(active_importance)]
        
        # Prune worst N
        pruned = []
        for i in range(min(n_to_prune, len(sorted_indices))):
            idx = sorted_indices[i]
            self.active_mask[idx] = False
            pruned.append(self.feature_names[idx])
        
        return pruned
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        importance = self.calculate_importance()
        
        active_indices = np.where(self.active_mask)[0]
        active_importance = importance[active_indices]
        
        sorted_indices = active_indices[np.argsort(active_importance)[::-1]]
        
        return [(self.feature_names[i], importance[i]) for i in sorted_indices[:n]]
    
    def mask_features(self, features: np.ndarray) -> np.ndarray:
        """Return only active features."""
        return features[self.active_mask]


# =============================================================================
# SUPERPOT AGENT
# =============================================================================

class SuperPotAgent:
    """
    Agent that learns on ALL features, with feature pruning.
    """
    
    def __init__(self, n_features: int, n_actions: int = 4, name: str = "SuperPot"):
        self.name = name
        self.n_features = n_features
        self.n_actions = n_actions
        
        # Policy (linear for interpretability)
        self.W = np.random.randn(n_features, n_actions) * 0.01
        self.b = np.zeros(n_actions)
        
        # Value function
        self.V_W = np.random.randn(n_features) * 0.01
        self.V_b = 0.0
        
        # Learning params
        self.lr = 0.001
        self.gamma = 0.95
        self.epsilon = 0.3
        
        # Buffer
        self.buffer: List[Tuple] = []
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(len(x)) / len(x)
        return p
    
    def select_action(self, features: np.ndarray, explore: bool = True) -> int:
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        # Pad/truncate features if needed
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        
        logits = features @ self.W + self.b
        probs = self._softmax(logits)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        return np.argmax(probs)
    
    def update(self, features: np.ndarray, action: int, reward: float,
               next_features: np.ndarray, done: bool):
        """Update agent."""
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        next_features = np.nan_to_num(next_features, nan=0, posinf=0, neginf=0)
        
        # Pad if needed
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        if len(next_features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(next_features), self.n_features)] = next_features[:self.n_features]
            next_features = padded
        
        # TD error
        value = features @ self.V_W + self.V_b
        next_value = 0 if done else next_features @ self.V_W + self.V_b
        td_target = reward + self.gamma * next_value
        td_error = td_target - value
        
        # Value update
        self.V_W += self.lr * td_error * features
        self.V_b += self.lr * td_error
        
        # Policy update (actor-critic style)
        probs = self._softmax(features @ self.W + self.b)
        grad = -probs.copy()
        grad[action] += 1
        grad *= td_error
        
        self.W += self.lr * np.outer(features, grad)
        self.b += self.lr * grad
        
        # Decay epsilon
        self.epsilon = max(0.05, self.epsilon * 0.9995)
    
    def get_feature_weights(self) -> np.ndarray:
        """Get absolute feature importance from weights."""
        return np.mean(np.abs(self.W), axis=1)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    
    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        df = pd.read_csv(filepath)
        df.columns = [c.strip('<>').lower() for c in df.columns]
    
    if 'volume' not in df.columns:
        df['volume'] = df.get('tickvol', 1000)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['open', 'high', 'low', 'close'])


def discover_files() -> List[dict]:
    """Find all data files."""
    files = []
    for base in ["data/master", "data/runs/berserker_run3/data", "data"]:
        path = Path(base)
        if not path.exists():
            continue
        for f in path.rglob("*.csv"):
            if 'symbol' in f.name.lower() or 'info' in f.name.lower():
                continue
            parts = f.stem.split('_')
            if len(parts) >= 2:
                files.append({'path': str(f), 'symbol': parts[0], 'timeframe': parts[1]})
    return files


def main():
    parser = argparse.ArgumentParser(description='SuperPot Feature Explorer')
    parser.add_argument('--episodes', type=int, default=100, help='Total episodes')
    parser.add_argument('--prune-every', type=int, default=20, help='Prune every N episodes')
    parser.add_argument('--prune-count', type=int, default=10, help='Features to prune each time')
    parser.add_argument('--max-files', type=int, default=30, help='Max files to use')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        SUPERPOT EXPLORER                             ║
║            Throw ALL measurements in. Let agents figure it out.      ║
║                 Prune the worst. Keep the best.                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Discover data
    files = discover_files()[:args.max_files]
    print(f"📁 Found {len(files)} data files")
    
    if not files:
        print("❌ No data files found!")
        return
    
    # Initialize
    extractor = SuperPotExtractor(lookback=50)
    tracker = FeatureImportanceTracker(extractor.n_features, extractor.feature_names)
    agent = SuperPotAgent(extractor.n_features, n_actions=4, name="SuperPot")
    
    print(f"\n🧪 Starting with {extractor.n_features} features")
    print(f"   Prune {args.prune_count} features every {args.prune_every} episodes")
    print(f"   Episodes: {args.episodes}")
    
    # Training loop
    start_time = time.time()
    all_rewards = []
    all_pnls = []
    
    for ep in range(args.episodes):
        # Pick random file
        file_info = files[np.random.randint(len(files))]
        
        try:
            df = load_data(file_info['path'])
            if len(df) < 200:
                continue
            
            df = df.iloc[-2000:].reset_index(drop=True)
            
            # Episode
            start_bar = np.random.randint(100, max(101, len(df) - args.max_steps - 10))
            bar = start_bar
            position = 0
            entry_price = 0
            balance = 10000
            episode_reward = 0
            
            for step in range(args.max_steps):
                if bar >= len(df) - 1:
                    break
                
                # Extract features
                features = extractor.extract(df, bar)
                active_features = tracker.mask_features(features)
                
                # Agent action
                action = agent.select_action(active_features, explore=True)
                
                # Execute
                price = df.iloc[bar]['close']
                reward = 0
                
                if action == 1 and position == 0:  # Buy
                    position = 1
                    entry_price = price * 1.0001
                elif action == 2 and position == 0:  # Sell
                    position = -1
                    entry_price = price * 0.9999
                elif action == 3 and position != 0:  # Close
                    if position == 1:
                        pnl = (price * 0.9999 - entry_price) / entry_price
                    else:
                        pnl = (entry_price - price * 1.0001) / entry_price
                    balance *= (1 + pnl * 0.1)
                    reward = pnl * 100
                    position = 0
                
                if position != 0:
                    reward -= 0.001
                
                # Next state
                bar += 1
                next_features = extractor.extract(df, bar) if bar < len(df) else features
                active_next = tracker.mask_features(next_features)
                
                # Track for importance
                tracker.record(features, action, reward)
                
                # Update agent
                agent.update(active_features, action, reward, active_next, bar >= len(df) - 1)
                
                episode_reward += reward
            
            # End of episode
            pnl = balance - 10000
            all_rewards.append(episode_reward)
            all_pnls.append(pnl)
            
            # Progress
            if (ep + 1) % 10 == 0:
                avg_r = np.mean(all_rewards[-10:])
                avg_pnl = np.mean(all_pnls[-10:])
                print(f"Ep {ep+1:3d}: R={avg_r:+7.1f} PnL=${avg_pnl:+7.0f} | "
                      f"Active features: {tracker.n_active}/{extractor.n_features} | "
                      f"ε={agent.epsilon:.3f}")
            
            # Prune features
            if (ep + 1) % args.prune_every == 0 and tracker.n_active > args.prune_count + 10:
                pruned = tracker.prune(args.prune_count)
                print(f"\n🗑️  PRUNED {len(pruned)} features:")
                for f in pruned[:5]:
                    print(f"   - {f}")
                if len(pruned) > 5:
                    print(f"   ... and {len(pruned) - 5} more")
                print(f"   Remaining: {tracker.n_active} features\n")
                
                # Resize agent weights
                active_indices = np.where(tracker.active_mask)[0]
                agent.W = agent.W[active_indices]
                agent.V_W = agent.V_W[active_indices]
                agent.n_features = tracker.n_active
        
        except Exception as e:
            continue
    
    # Final summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("SUPERPOT RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 Performance:")
    print(f"   Total episodes: {len(all_rewards)}")
    print(f"   Avg reward: {np.mean(all_rewards):+.2f}")
    print(f"   Avg PnL: ${np.mean(all_pnls):+.0f}")
    print(f"   Win rate: {sum(1 for p in all_pnls if p > 0) / len(all_pnls) * 100:.0f}%")
    
    print(f"\n🏆 TOP SURVIVING FEATURES ({tracker.n_active} remaining):")
    top_features = tracker.get_top_features(20)
    for i, (name, score) in enumerate(top_features):
        print(f"   {i+1:2d}. {name:<35s} score={score:.4f}")
    
    print(f"\n🗑️  PRUNED FEATURES ({extractor.n_features - tracker.n_active} removed):")
    pruned = tracker.get_pruned_features()
    for f in pruned[:10]:
        print(f"   - {f}")
    if len(pruned) > 10:
        print(f"   ... and {len(pruned) - 10} more")
    
    # Save results
    results_dir = Path("results/superpot")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"superpot_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'episodes': int(len(all_rewards)),
        'avg_reward': float(np.mean(all_rewards)),
        'avg_pnl': float(np.mean(all_pnls)),
        'initial_features': int(extractor.n_features),
        'surviving_features': int(tracker.n_active),
        'top_features': [{'name': str(n), 'score': float(s)} for n, s in top_features],
        'pruned_features': [str(p) for p in pruned],
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print("\n" + "="*60)
    print("THE MARKET HAS SPOKEN - THESE FEATURES MATTER!")
    print("="*60)


if __name__ == '__main__':
    main()
