"""Tests for monry/data/fetcher.py — MarketDataFetcher."""

from __future__ import annotations

import re

import pytest

from monry.data.fetcher import MarketDataFetcher
from monry.models.candle import Candle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fetcher() -> MarketDataFetcher:
    return MarketDataFetcher()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


class TestSyntheticData:
    def test_generates_correct_number_of_candles(self, fetcher: MarketDataFetcher) -> None:
        """Synthetic data generates approximately the right number of weekday candles."""
        candles = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-01-31")
        # January 2023 has ~22 weekdays
        assert 18 <= len(candles) <= 24

    def test_is_deterministic_with_same_seed(self, fetcher: MarketDataFetcher) -> None:
        """Same symbol/period/seed produces identical candles every time."""
        candles_a = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-02-01", seed=42)
        candles_b = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-02-01", seed=42)
        assert len(candles_a) == len(candles_b)
        for a, b in zip(candles_a, candles_b):
            assert a.date == b.date
            assert a.close == b.close

    def test_different_seeds_produce_different_data(self, fetcher: MarketDataFetcher) -> None:
        """Different seeds produce different candle data."""
        candles_a = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-02-01", seed=1)
        candles_b = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-02-01", seed=2)
        # At least one candle should differ
        assert any(a.close != b.close for a, b in zip(candles_a, candles_b))

    def test_valid_ohlcv_values(self, fetcher: MarketDataFetcher) -> None:
        """All candles satisfy OHLCV constraints."""
        candles = fetcher.generate_synthetic("MSFT", "2022-06-01", "2023-01-01")
        for c in candles:
            assert c.high >= c.low, f"high < low on {c.date}"
            assert c.high >= c.open, f"high < open on {c.date}"
            assert c.high >= c.close, f"high < close on {c.date}"
            assert c.low <= c.open, f"low > open on {c.date}"
            assert c.low <= c.close, f"low > close on {c.date}"
            assert c.low > 0, f"low <= 0 on {c.date}"
            assert c.volume > 0, f"volume <= 0 on {c.date}"

    def test_candles_have_correct_date_format(self, fetcher: MarketDataFetcher) -> None:
        """All candle dates are in YYYY-MM-DD format."""
        candles = fetcher.generate_synthetic("TSLA", "2023-03-01", "2023-04-01")
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for c in candles:
            assert date_pattern.match(c.date), f"Bad date format: {c.date}"

    def test_candles_are_sorted_ascending(self, fetcher: MarketDataFetcher) -> None:
        """Candles are returned sorted by date ascending."""
        candles = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-06-01")
        dates = [c.date for c in candles]
        assert dates == sorted(dates)

    def test_no_weekend_candles(self, fetcher: MarketDataFetcher) -> None:
        """Synthetic data skips Saturdays and Sundays."""
        from datetime import date
        candles = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-04-01")
        for c in candles:
            d = date.fromisoformat(c.date)
            assert d.weekday() < 5, f"Weekend candle found: {c.date}"

    def test_different_symbols_produce_different_data(self, fetcher: MarketDataFetcher) -> None:
        """Different symbols produce different candle values."""
        candles_aapl = fetcher.generate_synthetic("AAPL", "2023-01-01", "2023-02-01", seed=42)
        candles_msft = fetcher.generate_synthetic("MSFT", "2023-01-01", "2023-02-01", seed=42)
        assert any(a.close != b.close for a, b in zip(candles_aapl, candles_msft))


# ---------------------------------------------------------------------------
# fetch() falls back to synthetic when yfinance is not available
# ---------------------------------------------------------------------------


class TestFetchFallback:
    def test_fetch_returns_candles(self, fetcher: MarketDataFetcher) -> None:
        """fetch() always returns a non-empty list (synthetic fallback)."""
        candles = fetcher.fetch("AAPL", "2023-01-01", "2023-02-01")
        assert len(candles) > 0
        assert all(isinstance(c, Candle) for c in candles)
