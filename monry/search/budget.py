"""
SearchBudget — Steuerung der Exploration.

Kontrolliert wie viele Strategien vorgeschlagen werden dürfen,
erkennt Stagnation und passt die Mutations-Gewichtung dynamisch an.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SearchBudget:
    """
    Verwaltet das Exploration-Budget eines Monry-Runs.

    Dynamische Mutations-Gewichtung:
    - Gute Strategie gefunden → kleine Mutationen (parameter_variation bevorzugt)
    - Schlechte Phase → größere Sprünge (add_filter, remove_filter)
    - Cross-Combine erst ab Score > 50 (braucht zwei gute Strategien als Basis)
    """

    # --- Konfiguration (unveränderlich nach dem Start) ---
    max_proposals_per_round: int = 5
    max_stale_rounds: int = 3
    max_total_strategies: int = 50
    min_direction_score: float = 40.0

    # --- Laufzeitstatus ---
    proposals_this_round: int = field(default=0, compare=False)
    total_strategies: int = field(default=0, compare=False)
    stale_rounds: int = field(default=0, compare=False)
    best_score: float = field(default=0.0, compare=False)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def can_propose(self) -> bool:
        """Gibt True zurück wenn noch eine neue Strategie vorgeschlagen werden darf."""
        if self.total_strategies >= self.max_total_strategies:
            return False
        if self.proposals_this_round >= self.max_proposals_per_round:
            return False
        if self.stale_rounds >= self.max_stale_rounds:
            return False
        return True

    def record_proposal(self) -> None:
        """Registriert eine neue Strategie-Proposal."""
        self.proposals_this_round += 1
        self.total_strategies += 1

    def record_round_result(self, best_score_this_round: float) -> None:
        """
        Schließt eine Evaluationsrunde ab.

        Aktualisiert best_score und stale_rounds. Setzt proposals_this_round zurück.

        Args:
            best_score_this_round: Bester Score in der abgeschlossenen Runde.
        """
        if best_score_this_round > self.best_score:
            self.best_score = best_score_this_round
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1

        self.proposals_this_round = 0

    def is_exhausted(self) -> bool:
        """
        Gibt True zurück wenn das Budget erschöpft ist.

        Erschöpft wenn:
        - Maximale Gesamtstrategien erreicht, oder
        - Zu viele Runden ohne Verbesserung (Stagnation)
        """
        return (
            self.total_strategies >= self.max_total_strategies
            or self.stale_rounds >= self.max_stale_rounds
        )

    def get_mutation_weights(self, current_best_score: float) -> Dict[str, float]:
        """
        Berechnet dynamische Mutations-Gewichtungen basierend auf dem aktuellen Score.

        Logik:
        - Score < 30: Hauptsächlich große Sprünge (add/remove filter), kein cross_combine
        - Score 30-60: Ausgewogen, leicht erhöhte parameter_variation
        - Score > 60: Kleine Schritte dominieren, cross_combine aktiviert

        Args:
            current_best_score: Aktuell bester bekannter Score (0–100).

        Returns:
            Dict mit relativen Gewichten für jeden Mutations-Typ.
            Summe der Gewichte = 1.0.
        """
        if current_best_score < 30.0:
            return {
                "parameter_variation": 0.20,
                "add_filter": 0.45,
                "remove_filter": 0.35,
                "cross_combine": 0.00,
            }
        elif current_best_score < 60.0:
            return {
                "parameter_variation": 0.40,
                "add_filter": 0.30,
                "remove_filter": 0.20,
                "cross_combine": 0.10,
            }
        else:
            return {
                "parameter_variation": 0.55,
                "add_filter": 0.20,
                "remove_filter": 0.10,
                "cross_combine": 0.15,
            }

    def reset_round(self) -> None:
        """Setzt den Runden-Zähler zurück (für neuen Evaluationszyklus)."""
        self.proposals_this_round = 0
