"""
Pure-math indicator calculations.

All functions accept a list of Candle objects and return calculated values.
Returns None when not enough candles are provided for the calculation period.
All functions are pure — no side effects, no state.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from monry.models.candle import Candle


def calculate_rsi(candles: List[Candle], period: int = 14) -> Optional[float]:
    """
    Standard RSI using Wilder's smoothing (average gain/loss).

    Requires at least period + 1 candles.
    Returns a value in the range [0, 100], or None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    closes = [c.close for c in candles]
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Use the last `period` changes for the initial average
    relevant = changes[-(period):]
    gains = [max(ch, 0.0) for ch in relevant]
    losses = [abs(min(ch, 0.0)) for ch in relevant]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_sma(candles: List[Candle], period: int = 50) -> Optional[float]:
    """
    Simple Moving Average of close prices.

    Requires at least period candles.
    Returns the mean of the last period close prices, or None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles[-period:]]
    return sum(closes) / period


def calculate_ema(candles: List[Candle], period: int = 50) -> Optional[float]:
    """
    Exponential Moving Average using a standard multiplier of 2/(period+1).

    Seeded with the SMA of the first period candles.
    Requires at least period candles.
    Returns the EMA of close prices, or None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles]
    multiplier = 2.0 / (period + 1)

    # Seed EMA with SMA of first period values
    ema = sum(closes[:period]) / period

    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema

    return ema


def calculate_macd(
    candles: List[Candle],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[Tuple[float, float, float]]:
    """
    MACD indicator.

    Requires at least slow + signal candles.
    Returns (macd_line, signal_line, histogram) or None if insufficient data.
    """
    min_required = slow + signal
    if len(candles) < min_required:
        return None

    closes = [c.close for c in candles]

    def _ema_series(values: List[float], period: int) -> List[float]:
        mult = 2.0 / (period + 1)
        result = [sum(values[:period]) / period]
        for v in values[period:]:
            result.append((v - result[-1]) * mult + result[-1])
        return result

    fast_emas = _ema_series(closes, fast)
    slow_emas = _ema_series(closes, slow)

    # Align: slow_emas is shorter; trim fast_emas to match length
    offset = len(fast_emas) - len(slow_emas)
    fast_aligned = fast_emas[offset:]

    macd_series = [f - s for f, s in zip(fast_aligned, slow_emas)]

    if len(macd_series) < signal:
        return None

    signal_emas = _ema_series(macd_series, signal)

    macd_line = macd_series[-1]
    signal_line = signal_emas[-1]
    histogram = macd_line - signal_line

    return (macd_line, signal_line, histogram)


def calculate_bollinger_bands(
    candles: List[Candle],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Bollinger Bands.

    Requires at least period candles.
    Returns (upper, middle, lower) or None if insufficient data.
    """
    if len(candles) < period:
        return None

    closes = [c.close for c in candles[-period:]]
    middle = sum(closes) / period
    variance = sum((c - middle) ** 2 for c in closes) / period
    std = math.sqrt(variance)

    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return (upper, middle, lower)


def calculate_atr(candles: List[Candle], period: int = 14) -> Optional[float]:
    """
    Average True Range.

    Requires at least period + 1 candles (to compute true ranges using prev close).
    Returns the ATR value or None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Use last `period` true ranges
    return sum(true_ranges[-period:]) / period


def calculate_adx(candles: List[Candle], period: int = 14) -> Optional[float]:
    """
    Average Directional Index (simplified Wilder smoothing).

    Requires at least 2 * period candles.
    Returns a value typically in [0, 100], or None if insufficient data.
    """
    if len(candles) < 2 * period:
        return None

    plus_dm_list: List[float] = []
    minus_dm_list: List[float] = []
    tr_list: List[float] = []

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_high = candles[i - 1].high
        prev_low = candles[i - 1].low
        prev_close = candles[i - 1].close

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        tr_list.append(tr)

    def _wilder_smooth(values: List[float], p: int) -> List[float]:
        smoothed = [sum(values[:p])]
        for v in values[p:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / p + v)
        return smoothed

    smooth_tr = _wilder_smooth(tr_list, period)
    smooth_plus = _wilder_smooth(plus_dm_list, period)
    smooth_minus = _wilder_smooth(minus_dm_list, period)

    dx_list: List[float] = []
    for tr_s, plus_s, minus_s in zip(smooth_tr, smooth_plus, smooth_minus):
        if tr_s == 0.0:
            continue
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        di_sum = plus_di + minus_di
        if di_sum == 0.0:
            dx_list.append(0.0)
        else:
            dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum)

    if not dx_list:
        return None

    adx = sum(dx_list[-period:]) / min(len(dx_list), period)
    return adx


def calculate_roc(candles: List[Candle], period: int = 10) -> Optional[float]:
    """
    Rate of Change as a percentage.

    ROC = (close_now - close_n_periods_ago) / close_n_periods_ago * 100

    Requires at least period + 1 candles.
    Returns the ROC value or None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    current_close = candles[-1].close
    past_close = candles[-(period + 1)].close

    if past_close == 0.0:
        return None

    return (current_close - past_close) / past_close * 100.0


def calculate_volume_ratio(
    candles: List[Candle], avg_period: int = 20
) -> Optional[float]:
    """
    Current volume divided by average volume over avg_period.

    Requires at least avg_period + 1 candles (avg uses avg_period candles before current).
    Returns the ratio or None if insufficient data.
    """
    if len(candles) < avg_period + 1:
        return None

    current_volume = candles[-1].volume
    avg_volume = sum(c.volume for c in candles[-(avg_period + 1):-1]) / avg_period

    if avg_volume == 0.0:
        return None

    return current_volume / avg_volume
