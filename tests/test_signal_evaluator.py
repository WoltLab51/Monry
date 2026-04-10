"""Tests for monry/analysis/signal_evaluator.py."""

from __future__ import annotations

import pytest

from monry.analysis.signal_evaluator import SignalEvaluator
from monry.models.candle import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(closes: list[float], volumes: list[int] | None = None) -> list[Candle]:
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    return [
        Candle(
            date=f"2023-01-{i + 1:02d}",
            open=closes[i],
            high=closes[i] + 1.0,
            low=closes[i] - 1.0,
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(len(closes))
    ]


def _alternating_candles(n: int, start: float = 100.0) -> list[Candle]:
    """Alternating up/down candles — used to have realistic ATR/ADX."""
    candles = []
    price = start
    for i in range(n):
        direction = 1 if i % 2 == 0 else -1
        close = price + direction * 1.0
        candles.append(
            Candle(
                date=f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1_000_000,
            )
        )
        price = close
    return candles


@pytest.fixture
def evaluator() -> SignalEvaluator:
    return SignalEvaluator()


# ---------------------------------------------------------------------------
# Basic true/false evaluation
# ---------------------------------------------------------------------------


class TestBasicEvaluation:
    def test_returns_false_with_empty_rules(self, evaluator: SignalEvaluator) -> None:
        candles = _make_candles([100.0] * 30)
        assert evaluator.evaluate([], candles) is False

    def test_returns_false_with_empty_candles(self, evaluator: SignalEvaluator) -> None:
        rules = [{"name": "rsi", "parameters": {"period": 14}, "condition": "< threshold_oversold"}]
        assert evaluator.evaluate(rules, []) is False

    def test_returns_false_when_not_enough_candles(self, evaluator: SignalEvaluator) -> None:
        """RSI needs period+1 candles — with only 5 it returns False."""
        candles = _make_candles([100.0] * 5)
        rules = [{"name": "rsi", "parameters": {"period": 14, "threshold_oversold": 30}, "condition": "< threshold_oversold"}]
        assert evaluator.evaluate(rules, candles) is False

    def test_all_rules_must_pass(self, evaluator: SignalEvaluator) -> None:
        """If one rule fails, the whole evaluation returns False."""
        # RSI rule that passes (oversold) + ROC rule that fails (positive ROC on declining)
        declining = [float(100 - i) for i in range(30)]
        candles = _make_candles(declining)

        rsi_rule = {"name": "rsi", "parameters": {"period": 14, "threshold_oversold": 99}, "condition": "< threshold_oversold"}
        roc_rule = {"name": "roc", "parameters": {"period": 5}, "condition": "> 0"}

        # RSI should pass (very oversold), ROC should fail (negative)
        assert evaluator.evaluate([rsi_rule], candles) is True
        assert evaluator.evaluate([roc_rule], candles) is False
        assert evaluator.evaluate([rsi_rule, roc_rule], candles) is False


# ---------------------------------------------------------------------------
# RSI conditions
# ---------------------------------------------------------------------------


class TestRSIConditions:
    def test_rsi_below_threshold_oversold(self, evaluator: SignalEvaluator) -> None:
        # Strictly declining → RSI near 0 → below any threshold
        candles = _make_candles([float(100 - i) for i in range(30)])
        rules = [{"name": "rsi", "parameters": {"period": 14, "threshold_oversold": 99}, "condition": "< threshold_oversold"}]
        assert evaluator.evaluate(rules, candles) is True

    def test_rsi_above_threshold_overbought(self, evaluator: SignalEvaluator) -> None:
        # Strictly rising → RSI near 100 → above any threshold
        candles = _make_candles([float(i) for i in range(1, 31)])
        rules = [{"name": "rsi", "parameters": {"period": 14, "threshold_overbought": 1}, "condition": "> threshold_overbought"}]
        assert evaluator.evaluate(rules, candles) is True


# ---------------------------------------------------------------------------
# SMA condition
# ---------------------------------------------------------------------------


class TestSMACondition:
    def test_sma_price_above(self, evaluator: SignalEvaluator) -> None:
        # Last price is 200, SMA(10) of the 10 previous prices is around 100
        closes = [100.0] * 10 + [200.0]
        candles = _make_candles(closes)
        rules = [{"name": "sma", "parameters": {"period": 10}, "condition": "price_above"}]
        assert evaluator.evaluate(rules, candles) is True

    def test_sma_price_below(self, evaluator: SignalEvaluator) -> None:
        # Last price is 50, SMA of previous prices is 100
        closes = [100.0] * 10 + [50.0]
        candles = _make_candles(closes)
        rules = [{"name": "sma", "parameters": {"period": 10}, "condition": "price_below"}]
        assert evaluator.evaluate(rules, candles) is True


# ---------------------------------------------------------------------------
# MACD condition
# ---------------------------------------------------------------------------


class TestMACDCondition:
    def test_macd_histogram_positive(self, evaluator: SignalEvaluator) -> None:
        # Exponentially rising prices (3% per day) → MACD histogram positive
        closes = [100.0 * (1.03 ** i) for i in range(100)]
        candles = _make_candles(closes)
        rules = [{"name": "macd", "parameters": {"fast": 12, "slow": 26, "signal": 9}, "condition": "histogram_positive"}]
        assert evaluator.evaluate(rules, candles) is True

    def test_macd_histogram_negative(self, evaluator: SignalEvaluator) -> None:
        # Stable prices then sudden sharp drop → MACD goes negative faster than signal
        stable = [100.0] * 80
        sudden_drop = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
        closes = stable + sudden_drop
        candles = [
            Candle(date=f"2023-01-01", open=c, high=c + 1.0, low=max(c - 1.0, 0.01), close=c, volume=1_000_000)
            for c in closes
        ]
        rules = [{"name": "macd", "parameters": {"fast": 12, "slow": 26, "signal": 9}, "condition": "histogram_negative"}]
        assert evaluator.evaluate(rules, candles) is True


# ---------------------------------------------------------------------------
# Unknown indicator
# ---------------------------------------------------------------------------


class TestUnknownIndicator:
    def test_unknown_indicator_returns_false(self, evaluator: SignalEvaluator) -> None:
        candles = _make_candles([100.0] * 30)
        rules = [{"name": "nonexistent", "parameters": {}, "condition": "something"}]
        assert evaluator.evaluate(rules, candles) is False


# ---------------------------------------------------------------------------
# Bollinger Band edge cases
# ---------------------------------------------------------------------------


class TestBollingerEdgeCases:
    def test_bollinger_squeeze_with_zero_middle_returns_false(
        self, evaluator: SignalEvaluator
    ) -> None:
        """If Bollinger middle band is 0.0, bandwidth_squeeze must not raise ZeroDivisionError."""
        # All candles at 0.0 → middle = SMA = 0.0
        candles = _make_candles([0.0] * 30)
        rules = [
            {
                "name": "bollinger",
                "parameters": {"period": 20, "std_dev": 2.0},
                "condition": "bandwidth_squeeze",
            }
        ]
        # Must not raise, must return False
        result = evaluator.evaluate(rules, candles)
        assert result is False


# ---------------------------------------------------------------------------
# ATR edge cases
# ---------------------------------------------------------------------------


class TestATREdgeCases:
    def test_atr_with_single_candle_returns_false(
        self, evaluator: SignalEvaluator
    ) -> None:
        """With only 1 candle, long_period would be 0 — must return False without error."""
        candles = _make_candles([100.0])
        rules = [
            {
                "name": "atr",
                "parameters": {"period": 14},
                "condition": "above_average",
            }
        ]
        # Must not crash, must return False
        result = evaluator.evaluate(rules, candles)
        assert result is False
