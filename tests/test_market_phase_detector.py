"""Tests for monry/analysis/market_phase_detector.py."""

from __future__ import annotations

import pytest

from monry.analysis.market_phase_detector import detect_market_phase
from monry.models.candle import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(
            date=f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            open=closes[i],
            high=closes[i],
            low=closes[i],
            close=closes[i],
            volume=1_000_000,
        )
        for i in range(len(closes))
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMarketPhaseDetector:
    def test_detects_uptrend_in_rising_prices(self) -> None:
        # Steadily rising prices: current price above a rising SMA
        closes = [float(i) for i in range(1, 120)]  # 119 candles, strongly rising
        candles = _make_candles(closes)
        phase = detect_market_phase(candles, lookback=50)
        assert phase == "uptrend"

    def test_detects_downtrend_in_falling_prices(self) -> None:
        # Steadily falling prices: current price below a falling SMA
        closes = [float(120 - i) for i in range(120)]
        candles = _make_candles(closes)
        phase = detect_market_phase(candles, lookback=50)
        assert phase == "downtrend"

    def test_detects_sideways_in_flat_prices(self) -> None:
        # Constant prices: SMA == price, SMA not rising or falling
        closes = [100.0] * 120
        candles = _make_candles(closes)
        phase = detect_market_phase(candles, lookback=50)
        assert phase == "sideways"

    def test_returns_sideways_with_insufficient_data(self) -> None:
        # Fewer candles than lookback + 1 → sideways
        closes = [100.0] * 30
        candles = _make_candles(closes)
        phase = detect_market_phase(candles, lookback=50)
        assert phase == "sideways"

    def test_returns_valid_phase_string(self) -> None:
        closes = [float(i) for i in range(1, 120)]
        candles = _make_candles(closes)
        phase = detect_market_phase(candles, lookback=50)
        assert phase in {"uptrend", "downtrend", "sideways"}
