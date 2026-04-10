"""
MarketDataFetcher — Fetches historical OHLCV candle data.

Uses yfinance if installed, otherwise falls back to deterministic synthetic data
suitable for testing and development without network access.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import List

from monry.models.candle import Candle


class MarketDataFetcher:
    """Fetches historical OHLCV data."""

    def fetch(self, symbol: str, start: str, end: str) -> List[Candle]:
        """
        Fetches historical daily candles for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            start: Start date ISO format "YYYY-MM-DD"
            end: End date ISO format "YYYY-MM-DD"

        Returns:
            List of Candle objects, sorted by date ascending.

        Uses yfinance if installed, otherwise falls back to synthetic data.
        """
        try:
            import yfinance as yf  # type: ignore[import]

            return self._fetch_yfinance(yf, symbol, start, end)
        except ImportError:
            return self.generate_synthetic(symbol, start, end)

    # ------------------------------------------------------------------
    # yfinance backend
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_yfinance(yf, symbol: str, start: str, end: str) -> List[Candle]:
        """Fetches data using yfinance and converts to Candle objects."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)

        candles: List[Candle] = []
        for idx, row in df.iterrows():
            date_str = str(idx.date())
            candles.append(
                Candle(
                    date=date_str,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )

        return sorted(candles, key=lambda c: c.date)

    # ------------------------------------------------------------------
    # Synthetic data (deterministic, no network)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic(
        symbol: str, start: str, end: str, seed: int = 42
    ) -> List[Candle]:
        """
        Generates synthetic candle data for testing (deterministic with seed).

        Creates realistic-looking price movement using a random walk with drift.
        OHLCV values satisfy: high >= open, close; low <= open, close; high >= low.

        Args:
            symbol: Used only to vary the seed (different symbols → different data).
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD".
            seed: Random seed for reproducibility.

        Returns:
            List of Candle objects sorted by date ascending.
        """
        rng = random.Random(seed + hash(symbol) % 10_000)

        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)

        candles: List[Candle] = []
        current_price = 100.0 + rng.uniform(-10.0, 10.0)
        current_date = start_date

        while current_date < end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Random walk with small positive drift
            drift = 0.0002
            volatility = 0.015
            change_pct = drift + rng.gauss(0, volatility)
            close = current_price * math.exp(change_pct)

            # Intraday range
            range_pct = abs(rng.gauss(0, volatility * 1.5))
            daily_range = current_price * range_pct

            open_price = current_price * math.exp(rng.gauss(0, volatility * 0.3))
            high = max(open_price, close) + daily_range * rng.uniform(0.1, 0.5)
            low = min(open_price, close) - daily_range * rng.uniform(0.1, 0.5)

            # Clamp: low must be positive, high >= low
            low = max(low, 0.01)
            high = max(high, low)

            volume = int(rng.uniform(500_000, 5_000_000))

            candles.append(
                Candle(
                    date=current_date.isoformat(),
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(close, 4),
                    volume=volume,
                )
            )

            current_price = close
            current_date += timedelta(days=1)

        return candles
