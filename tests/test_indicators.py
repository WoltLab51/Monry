"""Tests for monry/analysis/indicators.py — pure math indicator calculations."""

from __future__ import annotations

import pytest

from monry.analysis.indicators import (
    calculate_atr,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_roc,
    calculate_rsi,
    calculate_sma,
    calculate_volume_ratio,
)
from monry.models.candle import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(closes: list[float], volumes: list[int] | None = None) -> list[Candle]:
    """Build simple candles where open==close and high==low==close."""
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    return [
        Candle(
            date=f"2023-01-{i + 1:02d}",
            open=closes[i],
            high=closes[i],
            low=closes[i],
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(len(closes))
    ]


def _make_ohlcv_candles(n: int, base: float = 100.0) -> list[Candle]:
    """Build n candles with realistic OHLCV (alternating up/down)."""
    candles = []
    price = base
    for i in range(n):
        direction = 1 if i % 2 == 0 else -1
        close = price + direction * 1.0
        high = max(price, close) + 0.5
        low = min(price, close) - 0.5
        candles.append(
            Candle(
                date=f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                open=round(price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=1_000_000,
            )
        )
        price = close
    return candles


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_returns_value_between_0_and_100(self) -> None:
        candles = _make_ohlcv_candles(50)
        value = calculate_rsi(candles, period=14)
        assert value is not None
        assert 0.0 <= value <= 100.0

    def test_rsi_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 10)
        assert calculate_rsi(candles, period=14) is None

    def test_rsi_all_gains_returns_100(self) -> None:
        # Strictly increasing prices → RSI should be 100
        candles = _make_candles([float(i) for i in range(1, 30)])
        value = calculate_rsi(candles, period=14)
        assert value == pytest.approx(100.0)

    def test_rsi_all_losses_returns_0(self) -> None:
        # Strictly decreasing prices → RSI should be 0
        candles = _make_candles([float(30 - i) for i in range(30)])
        value = calculate_rsi(candles, period=14)
        assert value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_of_constant_prices_equals_that_price(self) -> None:
        candles = _make_candles([50.0] * 60)
        assert calculate_sma(candles, period=50) == pytest.approx(50.0)

    def test_sma_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 10)
        assert calculate_sma(candles, period=50) is None

    def test_sma_correct_average(self) -> None:
        candles = _make_candles([10.0, 20.0, 30.0])
        assert calculate_sma(candles, period=3) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_returns_value(self) -> None:
        candles = _make_candles([float(i) for i in range(1, 60)])
        value = calculate_ema(candles, period=10)
        assert value is not None
        assert value > 0

    def test_ema_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 5)
        assert calculate_ema(candles, period=10) is None

    def test_ema_of_constant_prices_equals_that_price(self) -> None:
        candles = _make_candles([42.0] * 60)
        assert calculate_ema(candles, period=10) == pytest.approx(42.0)

    def test_ema_reacts_faster_than_sma_to_price_jump(self) -> None:
        """EMA should be closer to latest price than SMA shortly after a price jump."""
        # 50 candles at 100, then only 5 at 200.
        # SMA(20) = (15*100 + 5*200)/20 = 125; EMA moves faster toward 200.
        base = [100.0] * 50
        jump = [200.0] * 5
        candles = _make_candles(base + jump)
        sma = calculate_sma(candles, period=20)
        ema = calculate_ema(candles, period=20)
        assert ema is not None and sma is not None
        # EMA should be closer to 200 than SMA
        assert abs(ema - 200.0) < abs(sma - 200.0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_macd_returns_tuple_of_3_floats(self) -> None:
        candles = _make_ohlcv_candles(100)
        result = calculate_macd(candles)
        assert result is not None
        assert len(result) == 3
        macd_line, signal_line, histogram = result
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_macd_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 20)
        assert calculate_macd(candles) is None

    def test_macd_histogram_equals_macd_minus_signal(self) -> None:
        candles = _make_ohlcv_candles(100)
        result = calculate_macd(candles)
        assert result is not None
        macd_line, signal_line, histogram = result
        assert histogram == pytest.approx(macd_line - signal_line)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_bollinger_upper_gt_middle_gt_lower(self) -> None:
        candles = _make_ohlcv_candles(50)
        result = calculate_bollinger_bands(candles, period=20, std_dev=2.0)
        assert result is not None
        upper, middle, lower = result
        assert upper > middle > lower

    def test_bollinger_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 10)
        assert calculate_bollinger_bands(candles, period=20) is None

    def test_bollinger_constant_prices_zero_width(self) -> None:
        """Constant prices → std=0, so upper==middle==lower."""
        candles = _make_candles([50.0] * 30)
        result = calculate_bollinger_bands(candles, period=20, std_dev=2.0)
        assert result is not None
        upper, middle, lower = result
        assert upper == pytest.approx(middle)
        assert lower == pytest.approx(middle)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_is_always_positive(self) -> None:
        candles = _make_ohlcv_candles(50)
        value = calculate_atr(candles, period=14)
        assert value is not None
        assert value > 0

    def test_atr_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_ohlcv_candles(5)
        assert calculate_atr(candles, period=14) is None

    def test_atr_constant_prices_equals_zero(self) -> None:
        """Flat candles → TR = 0 for every bar."""
        candles = _make_candles([100.0] * 20)
        value = calculate_atr(candles, period=14)
        assert value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


class TestADX:
    def test_adx_returns_value_between_0_and_100(self) -> None:
        candles = _make_ohlcv_candles(60)
        value = calculate_adx(candles, period=14)
        assert value is not None
        assert 0.0 <= value <= 100.0

    def test_adx_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_ohlcv_candles(10)
        assert calculate_adx(candles, period=14) is None


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


class TestROC:
    def test_roc_correct_for_known_price_change(self) -> None:
        # Price goes from 100 to 110 over 10 periods → ROC = 10%
        closes = [100.0] * 10 + [110.0]
        candles = _make_candles(closes)
        value = calculate_roc(candles, period=10)
        assert value == pytest.approx(10.0)

    def test_roc_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 5)
        assert calculate_roc(candles, period=10) is None

    def test_roc_positive_for_rising_price(self) -> None:
        closes = [float(i) for i in range(1, 20)]
        candles = _make_candles(closes)
        value = calculate_roc(candles, period=5)
        assert value is not None
        assert value > 0


# ---------------------------------------------------------------------------
# Volume Ratio
# ---------------------------------------------------------------------------


class TestVolumeRatio:
    def test_volume_ratio_of_average_volume_equals_one(self) -> None:
        volumes = [1_000_000] * 25
        candles = [
            Candle(
                date=f"2023-01-{i + 1:02d}",
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=volumes[i],
            )
            for i in range(len(volumes))
        ]
        ratio = calculate_volume_ratio(candles, avg_period=20)
        assert ratio == pytest.approx(1.0)

    def test_volume_ratio_returns_none_when_not_enough_candles(self) -> None:
        candles = _make_candles([100.0] * 5)
        assert calculate_volume_ratio(candles, avg_period=20) is None

    def test_volume_spike_returns_ratio_greater_than_one(self) -> None:
        # Average volume = 1M, current = 3M → ratio = 3
        volumes = [1_000_000] * 20 + [3_000_000]
        candles = [
            Candle(
                date=f"2023-01-{i + 1:02d}",
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=volumes[i],
            )
            for i in range(len(volumes))
        ]
        ratio = calculate_volume_ratio(candles, avg_period=20)
        assert ratio is not None
        assert ratio == pytest.approx(3.0)
