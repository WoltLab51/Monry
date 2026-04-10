"""Tests für SearchBudget."""

from __future__ import annotations

import pytest

from monry.search.budget import SearchBudget


# ---------------------------------------------------------------------------
# can_propose
# ---------------------------------------------------------------------------


class TestCanPropose:
    def test_can_propose_initially(self) -> None:
        """Frisches Budget erlaubt Proposals."""
        budget = SearchBudget()
        assert budget.can_propose() is True

    def test_can_propose_respects_per_round_limit(self) -> None:
        """can_propose() gibt False zurück wenn max_proposals_per_round erreicht."""
        budget = SearchBudget(max_proposals_per_round=3)
        for _ in range(3):
            budget.record_proposal()
        assert budget.can_propose() is False

    def test_can_propose_respects_total_limit(self) -> None:
        """can_propose() gibt False zurück wenn max_total_strategies erreicht."""
        budget = SearchBudget(max_total_strategies=5, max_proposals_per_round=10)
        for _ in range(5):
            budget.record_proposal()
        assert budget.can_propose() is False

    def test_can_propose_respects_stale_rounds(self) -> None:
        """can_propose() gibt False zurück bei zu vielen stagnierenden Runden."""
        budget = SearchBudget(max_stale_rounds=2)
        budget.stale_rounds = 2
        assert budget.can_propose() is False

    def test_can_propose_after_round_reset(self) -> None:
        """Nach record_round_result mit Verbesserung können wieder Proposals gemacht werden."""
        budget = SearchBudget(max_proposals_per_round=3)
        for _ in range(3):
            budget.record_proposal()
        assert budget.can_propose() is False
        budget.record_round_result(best_score_this_round=50.0)
        assert budget.can_propose() is True


# ---------------------------------------------------------------------------
# record_proposal
# ---------------------------------------------------------------------------


class TestRecordProposal:
    def test_record_proposal_increments_counters(self) -> None:
        """record_proposal inkrementiert proposals_this_round und total_strategies."""
        budget = SearchBudget()
        budget.record_proposal()
        assert budget.proposals_this_round == 1
        assert budget.total_strategies == 1

    def test_multiple_proposals_accumulate(self) -> None:
        """Mehrere record_proposal-Aufrufe akkumulieren korrekt."""
        budget = SearchBudget()
        for _ in range(4):
            budget.record_proposal()
        assert budget.proposals_this_round == 4
        assert budget.total_strategies == 4


# ---------------------------------------------------------------------------
# record_round_result
# ---------------------------------------------------------------------------


class TestRecordRoundResult:
    def test_improvement_resets_stale_rounds(self) -> None:
        """Verbesserung setzt stale_rounds auf 0 zurück."""
        budget = SearchBudget()
        budget.stale_rounds = 2
        budget.best_score = 30.0
        budget.record_round_result(best_score_this_round=50.0)
        assert budget.stale_rounds == 0

    def test_improvement_updates_best_score(self) -> None:
        """Verbesserung aktualisiert best_score."""
        budget = SearchBudget()
        budget.record_round_result(best_score_this_round=75.0)
        assert budget.best_score == 75.0

    def test_no_improvement_increments_stale_rounds(self) -> None:
        """Keine Verbesserung inkrementiert stale_rounds."""
        budget = SearchBudget()
        budget.best_score = 60.0
        budget.record_round_result(best_score_this_round=50.0)
        assert budget.stale_rounds == 1

    def test_equal_score_counts_as_stale(self) -> None:
        """Gleicher Score gilt als keine Verbesserung."""
        budget = SearchBudget()
        budget.best_score = 50.0
        budget.record_round_result(best_score_this_round=50.0)
        assert budget.stale_rounds == 1

    def test_round_result_resets_per_round_counter(self) -> None:
        """record_round_result setzt proposals_this_round zurück."""
        budget = SearchBudget()
        budget.record_proposal()
        budget.record_proposal()
        budget.record_round_result(best_score_this_round=50.0)
        assert budget.proposals_this_round == 0


# ---------------------------------------------------------------------------
# is_exhausted
# ---------------------------------------------------------------------------


class TestIsExhausted:
    def test_not_exhausted_initially(self) -> None:
        """Frisches Budget ist nicht erschöpft."""
        assert SearchBudget().is_exhausted() is False

    def test_exhausted_when_total_reached(self) -> None:
        """is_exhausted() = True wenn max_total_strategies erreicht."""
        budget = SearchBudget(max_total_strategies=3, max_proposals_per_round=10)
        for _ in range(3):
            budget.record_proposal()
        assert budget.is_exhausted() is True

    def test_exhausted_when_stale_rounds_reached(self) -> None:
        """is_exhausted() = True wenn max_stale_rounds erreicht."""
        budget = SearchBudget(max_stale_rounds=3)
        budget.stale_rounds = 3
        assert budget.is_exhausted() is True

    def test_exhausted_checks_both_conditions(self) -> None:
        """is_exhausted() prüft total_strategies UND stale_rounds."""
        budget = SearchBudget(max_total_strategies=100, max_stale_rounds=5)
        # Keines der Limits erreicht
        budget.total_strategies = 10
        budget.stale_rounds = 2
        assert budget.is_exhausted() is False


# ---------------------------------------------------------------------------
# get_mutation_weights
# ---------------------------------------------------------------------------


class TestGetMutationWeights:
    def test_weights_sum_to_one(self) -> None:
        """Mutations-Gewichte summieren sich zu 1.0."""
        budget = SearchBudget()
        for score in [0.0, 15.0, 45.0, 75.0, 100.0]:
            weights = budget.get_mutation_weights(score)
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Gewichte für score={score} summieren sich zu {total}, nicht 1.0"
            )

    def test_low_score_prefers_large_mutations(self) -> None:
        """Bei niedrigem Score dominieren add_filter und remove_filter."""
        budget = SearchBudget()
        weights = budget.get_mutation_weights(15.0)
        large_mutation_weight = weights["add_filter"] + weights["remove_filter"]
        assert large_mutation_weight > weights["parameter_variation"]

    def test_high_score_prefers_small_mutations(self) -> None:
        """Bei hohem Score dominiert parameter_variation."""
        budget = SearchBudget()
        weights = budget.get_mutation_weights(80.0)
        assert weights["parameter_variation"] > weights["add_filter"]
        assert weights["parameter_variation"] > weights["remove_filter"]

    def test_cross_combine_disabled_at_low_score(self) -> None:
        """cross_combine hat Gewicht 0 bei niedrigem Score (< 30)."""
        budget = SearchBudget()
        weights = budget.get_mutation_weights(10.0)
        assert weights["cross_combine"] == 0.0

    def test_cross_combine_enabled_at_high_score(self) -> None:
        """cross_combine hat positives Gewicht bei hohem Score (> 60)."""
        budget = SearchBudget()
        weights = budget.get_mutation_weights(70.0)
        assert weights["cross_combine"] > 0.0

    def test_all_mutation_types_are_present(self) -> None:
        """Alle vier Mutations-Typen sind in den Gewichten vertreten."""
        budget = SearchBudget()
        weights = budget.get_mutation_weights(50.0)
        assert set(weights.keys()) == {
            "parameter_variation",
            "add_filter",
            "remove_filter",
            "cross_combine",
        }
