"""Goal-Datenmodell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Goal:
    """
    Definiert das Ziel eines Monry-Explorationslaufs.

    success_criteria enthält messbare Mindestanforderungen:
    - min_win_rate: Minimale Gewinnrate (z.B. 0.55 = 55%)
    - min_profit_factor: Mindest-Profit-Faktor (z.B. 1.5)
    - max_drawdown: Maximaler Drawdown (z.B. 0.15 = 15%)
    - min_sharpe: Mindest-Sharpe-Ratio (z.B. 1.0)
    - min_signals: Minimale Anzahl an Handelssignalen (z.B. 30)
    """

    symbol: str
    period_start: str
    period_end: str
    success_criteria: Dict[str, float]
