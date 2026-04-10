"""Candle-Datenmodell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candle:
    """OHLCV-Kerze für historische Marktdaten."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
