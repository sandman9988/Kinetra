#!/usr/bin/env python3
"""Standalone Pro Renko trading engine with adaptive brick sizing."""

from __future__ import annotations

import csv
import os
from collections import deque
from datetime import datetime, timezone

import numpy as np


class ProRenkoSystem:
    def __init__(
        self,
        symbol: str = "EURUSD",
        brick_pips: float = 5.0,
        alpha: float = 0.2,
        window: int = 20,
        pip_size: float = 0.0001,
    ) -> None:
        # 1. DSP & Volatility Config
        self.symbol = symbol
        self.pip_size = float(pip_size)
        self.alpha = float(alpha)
        self.brick_size = float(brick_pips) * self.pip_size
        self.vol_window: deque[float] = deque(maxlen=int(window))

        # 2. State Tracking
        self.last_price: float | None = None
        self.last_brick_close: float | None = None
        self.direction: int = 0  # 1 Up, -1 Down, 0 Unset

        # 3. Warmup & Markov (2x2 Rule)
        self.bricks_count = 0
        self.flips_count = 0
        self.is_ready = False
        # Row/col order: 0=Up, 1=Down
        self.markov_counts = np.ones((2, 2), dtype=np.float64)

        # 4. Trade Management
        self.pos: str | None = None  # "BUY", "SELL", None
        self.entry = 0.0
        self.sl = 0.0
        self.mfe = 0.0

        self.filename = f"renko_{symbol}_pro.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["ts", "open", "close", "dir", "flip", "prob", "state"])

    @staticmethod
    def _dir_to_idx(direction: int) -> int:
        return 0 if direction == 1 else 1

    def _get_markov_prob(self, prev_dir: int, next_dir: int) -> float:
        """P(next_dir | prev_dir) from current matrix (no look-ahead leakage)."""
        if prev_dir not in (-1, 1) or next_dir not in (-1, 1):
            return 0.5
        row = self._dir_to_idx(prev_dir)
        col = self._dir_to_idx(next_dir)
        row_sum = float(self.markov_counts[row].sum())
        if row_sum <= 0.0:
            return 0.5
        return float(self.markov_counts[row, col] / row_sum)

    def process_tick(self, price: float) -> None:
        price = float(price)
        if self.last_price is None:
            self.last_price = price
            self.last_brick_close = round(price / self.brick_size) * self.brick_size
            return

        if self.last_brick_close is None:
            self.last_brick_close = round(price / self.brick_size) * self.brick_size

        self.vol_window.append(abs(price - self.last_price))
        self.last_price = price

        while True:
            # Thresholds
            u_cont = self.last_brick_close + self.brick_size
            d_cont = self.last_brick_close - self.brick_size
            u_rev = self.last_brick_close + (2.0 * self.brick_size)
            d_rev = self.last_brick_close - (2.0 * self.brick_size)

            brick_formed = False
            if self.direction == 1:
                if price >= u_cont:
                    self._create_brick(u_cont, 1, False)
                    brick_formed = True
                elif price <= d_rev:
                    self._create_brick(self.last_brick_close - self.brick_size, -1, True)
                    brick_formed = True
            elif self.direction == -1:
                if price <= d_cont:
                    self._create_brick(d_cont, -1, False)
                    brick_formed = True
                elif price >= u_rev:
                    self._create_brick(self.last_brick_close + self.brick_size, 1, True)
                    brick_formed = True
            else:  # Initial Direction
                if price >= u_cont:
                    self._create_brick(u_cont, 1, True)
                    brick_formed = True
                elif price <= d_cont:
                    self._create_brick(d_cont, -1, True)
                    brick_formed = True

            if brick_formed:
                self._update_dsp_volatility()
                self._manage_trade(self.last_brick_close)
            else:
                if self.pos:
                    self._update_trailing_sl(price)
                break

    def _create_brick(self, close: float, direction: int, is_flip: bool) -> None:
        prev_dir = self.direction

        # Compute probability before updating counts to avoid self-leakage.
        prob = self._get_markov_prob(prev_dir, direction) if prev_dir in (-1, 1) else 0.5

        # Update Markov counts only when previous direction exists.
        if prev_dir in (-1, 1):
            self.markov_counts[self._dir_to_idx(prev_dir), self._dir_to_idx(direction)] += 1.0

        self.bricks_count += 1
        if is_flip:
            self.flips_count += 1
        if not self.is_ready and self.bricks_count >= 2 and self.flips_count >= 2:
            self.is_ready = True

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(
                f, fieldnames=["ts", "open", "close", "dir", "flip", "prob", "state"]
            ).writerow(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "open": self.last_brick_close,
                    "close": close,
                    "dir": direction,
                    "flip": bool(is_flip),
                    "prob": prob,
                    "state": "READY" if self.is_ready else "WARMUP",
                }
            )

        self.last_brick_close = close
        self.direction = direction

    def _update_dsp_volatility(self) -> None:
        if len(self.vol_window) == self.vol_window.maxlen:
            raw_atr = sum(self.vol_window) / len(self.vol_window)
            self.brick_size = max(
                (self.alpha * raw_atr) + ((1.0 - self.alpha) * self.brick_size),
                2.0 * self.pip_size,
            )

    def _reset_position_state(self) -> None:
        self.pos = None
        self.entry = 0.0
        self.sl = 0.0
        self.mfe = 0.0

    def _manage_trade(self, trade_price: float) -> None:
        # 1. Exit on color flip
        if self.pos and (
            (self.pos == "BUY" and self.direction == -1)
            or (self.pos == "SELL" and self.direction == 1)
        ):
            print(f"TP/Flip Exit at {trade_price}")
            self._reset_position_state()

        # 2. Entry on Markov-validated direction
        # Use the opposite side as "previous direction" to estimate continuation.
        prev_dir = -self.direction if self.direction in (-1, 1) else 0
        prob = self._get_markov_prob(prev_dir, self.direction)
        if not self.pos and self.is_ready and prob > 0.65:
            self.pos = "BUY" if self.direction == 1 else "SELL"
            self.entry = trade_price
            self.mfe = trade_price
            self.sl = self.entry - (self.direction * self.brick_size)  # 1 brick stop
            print(f"ENTRY {self.pos} at {self.entry} | SL: {self.sl} | P: {prob:.2f}")

    def _update_trailing_sl(self, price: float) -> None:
        # MFE & 50% trailing stop after 2 bricks in profit.
        if self.pos == "BUY":
            self.mfe = max(self.mfe, price)
            if (self.mfe - self.entry) >= (2.0 * self.brick_size):
                self.sl = max(self.sl, self.entry + (self.mfe - self.entry) * 0.5)
            if price <= self.sl:
                print("SL Hit")
                self._reset_position_state()
        elif self.pos == "SELL":
            self.mfe = min(self.mfe, price)
            if (self.entry - self.mfe) >= (2.0 * self.brick_size):
                self.sl = min(self.sl, self.entry - (self.entry - self.mfe) * 0.5)
            if price >= self.sl:
                print("SL Hit")
                self._reset_position_state()
