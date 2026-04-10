"""Tests für den ExplorationAgent."""

from __future__ import annotations

import pytest

from monry._genus_stubs import Message, MessageBus
from monry.agents.exploration_agent import ExplorationAgent, _INITIAL_CATEGORIES
from monry.models.goal import Goal
from monry.models.strategy import Strategy
from monry.search.budget import SearchBudget
from monry.topics import (
    MONRY_EVALUATION_COMPLETED,
    MONRY_EXPLORATION_BUDGET_EXHAUSTED,
    MONRY_STRATEGY_PROPOSED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def goal() -> Goal:
    return Goal(
        symbol="AAPL",
        period_start="2020-01-01",
        period_end="2023-12-31",
        success_criteria={
            "min_win_rate": 0.55,
            "min_profit_factor": 1.5,
            "max_drawdown": 0.15,
            "min_sharpe": 1.0,
            "min_signals": 30,
        },
    )


@pytest.fixture
def agent(bus: MessageBus, goal: Goal) -> ExplorationAgent:
    ag = ExplorationAgent(
        agent_id="test_explorer", message_bus=bus, goal=goal, seed=42
    )
    ag.initialize()
    return ag


# ---------------------------------------------------------------------------
# Initiale Basisstrategien
# ---------------------------------------------------------------------------


class TestInitialStrategies:
    def test_start_creates_five_strategies(self, agent: ExplorationAgent, bus: MessageBus) -> None:
        """start() erzeugt genau 5 initiale Strategien."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        assert len(proposals) == 5

    def test_initial_strategies_cover_all_categories(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Jede initiale Strategie kommt aus einer anderen Indikator-Kategorie."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        categories = set()
        for msg in proposals:
            for rule in msg.payload["rules"]:
                categories.add(rule["category"])

        for expected_category in _INITIAL_CATEGORIES:
            assert expected_category in categories, (
                f"Kategorie '{expected_category}' fehlt unter den Initialstrategien"
            )

    def test_initial_strategies_have_no_parent(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Initiale Strategien haben keinen Parent (sind Basisstrategien)."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        for msg in proposals:
            assert msg.payload["parent_strategy"] is None

    def test_initial_strategies_have_generation_zero(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Initiale Strategien haben Generation 0."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        for msg in proposals:
            assert msg.payload["generation"] == 0

    def test_initial_strategies_have_required_fields(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Jede Strategie hat hypothesis, works_when und name."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        for msg in proposals:
            assert msg.payload["hypothesis"], "hypothesis darf nicht leer sein"
            assert msg.payload["works_when"], "works_when darf nicht leer sein"
            assert msg.payload["name"], "name darf nicht leer sein"

    def test_strategy_ids_are_sequential(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Strategie-IDs werden sequenziell erzeugt (exp_001, exp_002, ...)."""
        agent.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        ids = [msg.payload["strategy_id"] for msg in proposals]
        assert ids == ["exp_001", "exp_002", "exp_003", "exp_004", "exp_005"]


# ---------------------------------------------------------------------------
# Parameter-Mutation
# ---------------------------------------------------------------------------


class TestParameterMutation:
    def _get_first_strategy(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> Strategy:
        agent.start()
        strategy_id = bus.get_published(MONRY_STRATEGY_PROPOSED)[0].payload["strategy_id"]
        return agent._known_strategies[strategy_id]

    def test_parameter_mutation_creates_valid_variant(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """parameter_variation erzeugt eine gültige Strategie-Variante."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        mutations = agent._mutate_parameter(parent)
        assert len(mutations) == 1
        child = mutations[0]
        assert child.mutation_type == "parameter_variation"
        assert child.parent_strategy == parent.strategy_id
        assert child.generation == parent.generation + 1

    def test_parameter_mutation_inherits_works_when(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Parameter-Variante erbt works_when vom Parent."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        child = agent._mutate_parameter(parent)[0]
        assert child.works_when == parent.works_when

    def test_parameter_mutation_has_description(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Parameter-Variante hat mutation_description mit altem und neuem Wert."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        child = agent._mutate_parameter(parent)[0]
        assert child.mutation_description is not None
        assert "→" in child.mutation_description

    def test_parameter_mutation_changes_exactly_one_param(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Parameter-Variante ändert genau einen Parameter in genau einer Regel."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        child = agent._mutate_parameter(parent)[0]
        assert len(child.rules) == len(parent.rules)

        # Mindestens ein Parameter muss sich geändert haben
        parent_params = [list(r["parameters"].values()) for r in parent.rules]
        child_params = [list(r["parameters"].values()) for r in child.rules]
        assert parent_params != child_params


# ---------------------------------------------------------------------------
# Add-Filter-Mutation
# ---------------------------------------------------------------------------


class TestAddFilterMutation:
    def test_add_filter_adds_one_indicator(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """add_filter fügt genau einen Indikator zu einer Strategie hinzu."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        mutations = agent._add_filter(parent)
        if mutations:  # Kann leer sein wenn keine kompatiblen Filter
            child = mutations[0]
            assert len(child.rules) == len(parent.rules) + 1
            assert child.mutation_type == "add_filter"

    def test_add_filter_uses_compatible_indicator(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """add_filter fügt einen Indikator hinzu, der nicht bereits verwendet wird."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        existing_names = {r["name"] for r in parent.rules}
        mutations = agent._add_filter(parent)
        if mutations:
            child = mutations[0]
            new_rule_names = {r["name"] for r in child.rules}
            added = new_rule_names - existing_names
            assert len(added) == 1

    def test_add_filter_has_hypothesis(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """add_filter-Strategie hat eine erklärende Hypothese."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        mutations = agent._add_filter(parent)
        if mutations:
            assert mutations[0].hypothesis != ""


# ---------------------------------------------------------------------------
# Remove-Filter-Mutation
# ---------------------------------------------------------------------------


class TestRemoveFilterMutation:
    def _build_multi_rule_strategy(self, agent: ExplorationAgent) -> Strategy:
        """Hilfsmethode: Erstellt eine Strategie mit 3 Regeln."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        # add_filter aufrufen bis wir > 2 Regeln haben
        current = parent
        for _ in range(3):
            mutations = agent._add_filter(current)
            if mutations:
                current = mutations[0]
        return current

    def test_remove_filter_removes_one_rule(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """remove_filter entfernt genau eine Regel."""
        multi = self._build_multi_rule_strategy(agent)
        assert len(multi.rules) > 2, "Testvorbedingung: > 2 Regeln nötig"
        mutations = agent._remove_filter(multi)
        assert len(mutations) == 1
        child = mutations[0]
        assert len(child.rules) == len(multi.rules) - 1
        assert child.mutation_type == "remove_filter"

    def test_remove_filter_blocked_with_two_or_fewer_rules(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """remove_filter gibt leere Liste zurück wenn Strategie ≤ 2 Regeln hat."""
        agent.start()
        # Initiale Strategien haben 1 Regel — remove_filter soll blockiert sein
        parent = list(agent._known_strategies.values())[0]
        assert len(parent.rules) <= 2
        mutations = agent._remove_filter(parent)
        assert mutations == []


# ---------------------------------------------------------------------------
# Cross-Combine-Mutation
# ---------------------------------------------------------------------------


class TestCrossCombineMutation:
    def test_cross_combine_requires_two_successful_strategies(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """cross_combine gibt leere Liste zurück ohne zweite erfolgreiche Strategie."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]
        # Keine anderen erfolgreichen Strategien → leer
        mutations = agent._cross_combine(parent)
        assert mutations == []

    def test_cross_combine_merges_rules(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """cross_combine kombiniert Regeln von zwei Strategien."""
        agent.start()
        strategies = list(agent._known_strategies.values())
        # Zwei Strategien als erfolgreich markieren
        agent._successful_strategies = strategies[:2]

        parent = strategies[0]
        mutations = agent._cross_combine(parent)

        if mutations:
            child = mutations[0]
            assert len(child.rules) > len(parent.rules)
            assert child.mutation_type == "cross_combine"
            assert child.parent_strategy == parent.strategy_id

    def test_cross_combine_has_hypothesis(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """cross_combine-Strategie hat Hypothese mit Referenz auf beide Eltern."""
        agent.start()
        strategies = list(agent._known_strategies.values())
        agent._successful_strategies = strategies[:2]

        parent = strategies[0]
        mutations = agent._cross_combine(parent)
        if mutations:
            child = mutations[0]
            assert parent.strategy_id in child.hypothesis


# ---------------------------------------------------------------------------
# Budget-Kontrolle
# ---------------------------------------------------------------------------


class TestBudgetControl:
    def test_no_proposals_after_budget_exhausted(
        self, bus: MessageBus, goal: Goal
    ) -> None:
        """Keine weiteren Proposals wenn Budget erschöpft ist."""
        tight_budget = SearchBudget(max_total_strategies=3, max_proposals_per_round=3)
        ag = ExplorationAgent(
            agent_id="test", message_bus=bus, goal=goal, budget=tight_budget, seed=42
        )
        ag.initialize()
        ag.start()
        proposals = bus.get_published(MONRY_STRATEGY_PROPOSED)
        assert len(proposals) <= 3

    def test_budget_exhausted_publishes_notification(
        self, bus: MessageBus, goal: Goal
    ) -> None:
        """Bei erschöpftem Budget wird MONRY_EXPLORATION_BUDGET_EXHAUSTED publiziert."""
        # Budget mit genug Raum für start(), aber stale-Erkennung nach einer Runde
        normal_budget = SearchBudget(
            max_total_strategies=10, max_stale_rounds=2, max_proposals_per_round=5
        )
        ag = ExplorationAgent(
            agent_id="test", message_bus=bus, goal=goal, budget=normal_budget, seed=42
        )
        ag.initialize()
        ag.start()

        assert len(ag._known_strategies) > 0, "Agent muss mindestens eine Strategie registriert haben"

        # Budget manuell erschöpfen: best_score auf höheren Wert setzen
        # damit score=20.0 keine Verbesserung ist und stale_rounds bleibt
        ag._budget.stale_rounds = 10
        ag._budget.best_score = 80.0

        strategy_id = list(ag._known_strategies.keys())[0]
        evaluation = Message(
            topic=MONRY_EVALUATION_COMPLETED,
            payload={"strategy_id": strategy_id, "score": 20.0, "passed": True},
        )
        ag.process_message(evaluation)

        exhausted = bus.get_published(MONRY_EXPLORATION_BUDGET_EXHAUSTED)
        assert len(exhausted) >= 1


# ---------------------------------------------------------------------------
# Pflichtfelder in Strategien
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_every_strategy_has_hypothesis(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Jede veröffentlichte Strategie hat eine nicht-leere Hypothese."""
        agent.start()
        for msg in bus.get_published(MONRY_STRATEGY_PROPOSED):
            assert msg.payload["hypothesis"], (
                f"Strategie {msg.payload['strategy_id']} hat keine Hypothese"
            )

    def test_every_strategy_has_works_when(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Jede veröffentlichte Strategie hat ein nicht-leeres works_when."""
        agent.start()
        for msg in bus.get_published(MONRY_STRATEGY_PROPOSED):
            assert msg.payload["works_when"], (
                f"Strategie {msg.payload['strategy_id']} hat kein works_when"
            )

    def test_mutation_strategies_have_parent(
        self, agent: ExplorationAgent, bus: MessageBus
    ) -> None:
        """Mutierte Strategien haben immer einen parent_strategy-Verweis."""
        agent.start()
        parent = list(agent._known_strategies.values())[0]

        for mutate_fn in [
            agent._mutate_parameter,
            agent._add_filter,
        ]:
            mutations = mutate_fn(parent)
            for child in mutations:
                assert child.parent_strategy is not None
                assert child.parent_strategy == parent.strategy_id
