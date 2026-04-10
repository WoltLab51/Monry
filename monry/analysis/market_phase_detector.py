"""
Market phase detector — detects the current market phase from candle data.
"""

from __future__ import annotations

from typing import List

from monry.analysis.indicators import calculate_sma
from monry.models.candle import Candle


def detect_market_phase(candles: List[Candle], lookback: int = 50) -> str:
    """
    Detects the current market phase from recent candles.

    Returns one of: "uptrend", "downtrend", "sideways"

    Logic:
    - Calculate SMA(lookback)
    - If current price > SMA and SMA is rising → "uptrend"
    - If current price < SMA and SMA is falling → "downtrend"
    - Otherwise → "sideways"

    Returns "sideways" when there is insufficient data.
    """
    if len(candles) < lookback + 1:
        return "sideways"

    current_price = candles[-1].close

    sma_now = calculate_sma(candles, period=lookback)
    # SMA one period ago (use all candles except the last one)
    sma_prev = calculate_sma(candles[:-1], period=lookback)

    if sma_now is None or sma_prev is None:
        return "sideways"

    sma_rising = sma_now > sma_prev
    sma_falling = sma_now < sma_prev

    if current_price > sma_now and sma_rising:
        return "uptrend"
    elif current_price < sma_now and sma_falling:
        return "downtrend"
    else:
        return "sideways"
