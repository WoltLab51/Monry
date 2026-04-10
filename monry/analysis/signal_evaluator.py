"""
SignalEvaluator — Evaluates whether a strategy's rules produce a signal.

Takes a Strategy's rules and a window of Candle data, evaluates each rule's
condition against the calculated indicators, and determines if a signal
is triggered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from monry.analysis import indicators as ind
from monry.models.candle import Candle


class SignalEvaluator:
    """Evaluates whether a strategy's rules produce a signal on given candle data."""

    def evaluate(self, rules: List[Dict[str, Any]], candles: List[Candle]) -> bool:
        """
        Returns True if ALL rules in the strategy are satisfied for the current
        candle window.

        Each rule has:
        - name: indicator name (e.g. "rsi")
        - parameters: dict of parameter values (e.g. {"period": 14, "threshold_oversold": 30})
        - condition: string describing the condition (e.g. "< threshold_oversold")

        The evaluator:
        1. Calculates the indicator value using indicators.py
        2. Evaluates the condition against the calculated value
        3. Returns True only if ALL rules pass

        Returns False if any rule fails or if there are not enough candles.
        """
        if not rules or not candles:
            return False

        for rule in rules:
            if not self._evaluate_rule(rule, candles):
                return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_rule(self, rule: Dict[str, Any], candles: List[Candle]) -> bool:
        """Evaluate a single rule against candle data."""
        name = rule.get("name", "")
        params = rule.get("parameters", {})
        condition = rule.get("condition", "")

        if name == "rsi":
            return self._eval_rsi(params, condition, candles)
        elif name == "sma":
            return self._eval_sma(params, condition, candles)
        elif name == "ema":
            return self._eval_ema(params, condition, candles)
        elif name == "macd":
            return self._eval_macd(params, condition, candles)
        elif name == "bollinger_bands":
            return self._eval_bollinger(params, condition, candles)
        elif name == "atr":
            return self._eval_atr(params, condition, candles)
        elif name == "adx":
            return self._eval_adx(params, condition, candles)
        elif name == "roc":
            return self._eval_roc(params, condition, candles)
        elif name == "volume":
            return self._eval_volume(params, condition, candles)

        return False

    def _eval_rsi(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 14))
        value = ind.calculate_rsi(candles, period=period)
        if value is None:
            return False

        if condition == "< threshold_oversold":
            threshold = params.get("threshold_oversold", 30)
            return value < threshold
        elif condition == "> threshold_overbought":
            threshold = params.get("threshold_overbought", 70)
            return value > threshold
        elif condition == "crossover":
            # Not enough context for a true crossover without prior state;
            # approximate: RSI near midpoint (40–60)
            return 40.0 <= value <= 60.0

        return False

    def _eval_sma(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 50))
        value = ind.calculate_sma(candles, period=period)
        if value is None:
            return False

        current_price = candles[-1].close

        if condition == "price_above":
            return current_price > value
        elif condition == "price_below":
            return current_price < value
        elif condition == "crossover":
            # Approximate crossover: price within 0.5% of SMA
            return abs(current_price - value) / value < 0.005

        return False

    def _eval_ema(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 50))
        value = ind.calculate_ema(candles, period=period)
        if value is None:
            return False

        current_price = candles[-1].close

        if condition == "price_above":
            return current_price > value
        elif condition == "price_below":
            return current_price < value
        elif condition == "crossover":
            return abs(current_price - value) / value < 0.005

        return False

    def _eval_macd(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        result = ind.calculate_macd(candles, fast=fast, slow=slow, signal=signal)
        if result is None:
            return False

        macd_line, signal_line, histogram = result

        if condition == "histogram_positive":
            return histogram > 0
        elif condition == "histogram_negative":
            return histogram < 0
        elif condition == "signal_crossover":
            # Approximate: MACD line near signal line
            return abs(macd_line - signal_line) < abs(macd_line) * 0.05

        return False

    def _eval_bollinger(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        result = ind.calculate_bollinger_bands(candles, period=period, std_dev=std_dev)
        if result is None:
            return False

        upper, middle, lower = result
        current_price = candles[-1].close

        if condition == "price_below_lower":
            return current_price < lower
        elif condition == "price_above_upper":
            return current_price > upper
        elif condition == "bandwidth_squeeze":
            # Bandwidth squeeze: band width is narrower than 5% of middle
            bandwidth = upper - lower
            if middle == 0.0:
                return False
            return bandwidth / middle < 0.05

        return False

    def _eval_atr(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 14))
        value = ind.calculate_atr(candles, period=period)
        if value is None:
            return False

        # Compare to a longer-term ATR average (2x period) for reference
        long_period = min(period * 2, len(candles) - 1)
        if long_period < 1:
            return False
        long_atr = ind.calculate_atr(candles, period=long_period)

        if long_atr is None or long_atr == 0.0:
            return False

        if condition == "above_average":
            return value > long_atr
        elif condition == "below_average":
            return value < long_atr

        return False

    def _eval_adx(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 14))
        threshold = float(params.get("threshold", 25))
        value = ind.calculate_adx(candles, period=period)
        if value is None:
            return False

        if condition == "> threshold":
            return value > threshold
        elif condition == "< threshold":
            return value < threshold

        return False

    def _eval_roc(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        period = int(params.get("period", 10))
        value = ind.calculate_roc(candles, period=period)
        if value is None:
            return False

        if condition == "> 0":
            return value > 0.0
        elif condition == "< 0":
            return value < 0.0
        elif condition == "above_threshold":
            # Use a default threshold of 5% if not specified
            threshold = float(params.get("threshold", 5.0))
            return value > threshold

        return False

    def _eval_volume(
        self, params: Dict[str, Any], condition: str, candles: List[Candle]
    ) -> bool:
        avg_period = int(params.get("avg_period", 20))
        multiplier = float(params.get("multiplier", 1.5))
        ratio = ind.calculate_volume_ratio(candles, avg_period=avg_period)
        if ratio is None:
            return False

        if condition == "above_average":
            return ratio > 1.0
        elif condition == "spike":
            return ratio > multiplier

        return False
