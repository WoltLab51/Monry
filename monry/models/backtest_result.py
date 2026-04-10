"""BacktestResult-Datenmodell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestResult:
    """
    Ergebnis eines Backtests einer Strategie.

    Wird von SelectionFilter geprüft um zu entscheiden,
    ob eine Strategie robust genug ist.

    trade_dates: ISO-Datums-Strings (YYYY-MM-DD) jedes Trades
                 — wird für Zeitverteilungs-Check verwendet.
    market_phases: Liste der Marktphasen, in denen Trades stattfanden
                   (z.B. "uptrend", "downtrend", "sideways")
    """

    strategy_id: str
    num_signals: int
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    trade_dates: List[str] = field(default_factory=list)
    market_phases: List[str] = field(default_factory=list)
