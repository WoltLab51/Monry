"""
SelectionFilter — Harte Selektion zum Overfitting-Schutz.

Hier passiert die eigentliche Magie — nicht in der Exploration, sondern im
Wegwerfen. Strategien die statistisch nicht robust genug sind werden konsequent
aussortiert.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

from monry.models.backtest_result import BacktestResult


@dataclass
class SelectionFilter:
    """
    Filtert Strategien die nicht robust genug sind.

    Harte Kriterien (alle müssen erfüllt sein):
    1. Genug Trades (min_signals)
    2. Trades zeitlich verteilt (kein Cluster in einem Monat)
    3. Mindestens zwei verschiedene Marktphasen
    4. Profit-Factor > min_profit_factor
    5. Sharpe-Ratio >= min_sharpe
    """

    min_signals: int = 30
    min_profit_factor: float = 1.0
    min_sharpe: float = 0.0
    max_single_month_ratio: float = 0.8
    min_market_phases: int = 2

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def is_valid(self, result: BacktestResult) -> bool:
        """
        Prüft ob eine Strategie alle harten Kriterien erfüllt.

        Args:
            result: Das Backtest-Ergebnis der Strategie.

        Returns:
            True wenn alle Kriterien erfüllt sind, sonst False.
        """
        return (
            self._has_enough_signals(result)
            and self._is_time_distributed(result)
            and self._covers_multiple_phases(result)
            and self._has_positive_profit_factor(result)
            and self._has_acceptable_sharpe(result)
        )

    def rejection_reasons(self, result: BacktestResult) -> List[str]:
        """
        Gibt alle Ablehnungsgründe zurück (für Diagnose und Logging).

        Args:
            result: Das Backtest-Ergebnis der Strategie.

        Returns:
            Liste von Strings mit Ablehnungsgründen. Leer wenn gültig.
        """
        reasons = []
        if not self._has_enough_signals(result):
            reasons.append(
                f"Zu wenige Trades: {result.num_signals} < {self.min_signals}"
            )
        if not self._is_time_distributed(result):
            reasons.append(
                f"Trades zu konzentriert: max. {self.max_single_month_ratio:.0%} "
                f"in einem Monat erlaubt"
            )
        if not self._covers_multiple_phases(result):
            reasons.append(
                f"Nur eine Marktphase: mindestens {self.min_market_phases} erwartet"
            )
        if not self._has_positive_profit_factor(result):
            reasons.append(
                f"Profit-Factor zu niedrig: {result.profit_factor:.2f} < "
                f"{self.min_profit_factor}"
            )
        if not self._has_acceptable_sharpe(result):
            reasons.append(
                f"Sharpe-Ratio zu niedrig: {result.sharpe_ratio:.2f} < {self.min_sharpe}"
            )
        return reasons

    # ------------------------------------------------------------------
    # Interne Prüfmethoden
    # ------------------------------------------------------------------

    def _has_enough_signals(self, result: BacktestResult) -> bool:
        return result.num_signals >= self.min_signals

    def _is_time_distributed(self, result: BacktestResult) -> bool:
        """Prüft ob Trades auf mehrere Monate verteilt sind."""
        if not result.trade_dates:
            # Kein Trade-Datum vorhanden → kann nicht prüfen → ablehnen
            return result.num_signals == 0

        month_counts = Counter(
            date[:7] for date in result.trade_dates  # "YYYY-MM"
        )
        total = len(result.trade_dates)
        if total == 0:
            return True

        max_month_ratio = max(month_counts.values()) / total
        return max_month_ratio <= self.max_single_month_ratio

    def _covers_multiple_phases(self, result: BacktestResult) -> bool:
        """Prüft ob Trades in mehreren verschiedenen Marktphasen aufgetreten sind."""
        unique_phases = set(result.market_phases)
        return len(unique_phases) >= self.min_market_phases

    def _has_positive_profit_factor(self, result: BacktestResult) -> bool:
        return result.profit_factor > self.min_profit_factor

    def _has_acceptable_sharpe(self, result: BacktestResult) -> bool:
        return result.sharpe_ratio >= self.min_sharpe
