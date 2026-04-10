"""
ExplorationAgent — Das Herzstück von Monry.

Exploriert systematisch den Strategie-Suchraum durch evolutionäre Mutationen.
Kommuniziert über den GENUS MessageBus und respektiert das SearchBudget.

Mutations-Logik (v1.5):
1. parameter_variation — Gleiche Strategie, ein Parameter leicht verändert
2. add_filter — Bestehende Strategie + ein neuer Indikator aus anderer Kategorie
3. remove_filter — Zu restriktive Strategie vereinfachen (nur wenn > 2 Regeln)
4. cross_combine — Zwei erfolgreiche Strategien kombinieren (v1.5 Erweiterung)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

try:
    from genus.core.agent import Agent, AgentState
    from genus.communication.message_bus import MessageBus, Message
except ImportError:
    # Minimale Stubs für Standalone-Entwicklung (ohne installiertes GENUS)
    from monry._genus_stubs import Agent, AgentState, MessageBus, Message  # type: ignore[assignment]

from monry.analysis.indicator_toolbox import IndicatorToolbox
from monry.models.goal import Goal
from monry.models.strategy import Strategy
from monry.search.budget import SearchBudget
from monry.topics import (
    MONRY_EVALUATION_COMPLETED,
    MONRY_EXPLORATION_BUDGET_EXHAUSTED,
    MONRY_STRATEGY_PROPOSED,
)

# Initiale Strategie-Kategorien: je eine pro Hauptkategorie
_INITIAL_CATEGORIES = [
    "mean_reversion",
    "trend",
    "momentum",
    "volatility",
    "volume",
]

# Marktkontext-Defaults je Indikator-Kategorie
_WORKS_WHEN_DEFAULTS: Dict[str, Dict[str, str]] = {
    "mean_reversion": {"market_phase": "range", "volatility": "low_to_medium"},
    "trend": {"market_phase": "uptrend", "volatility": "medium"},
    "momentum": {"market_phase": "any", "volatility": "medium"},
    "volatility": {"market_phase": "any", "volatility": "high"},
    "volume": {"market_phase": "any", "volatility": "any"},
}


class ExplorationAgent(Agent):
    """
    Exploriert den Strategie-Suchraum und publiziert Proposals über den MessageBus.

    Lifecycle:
    - initialize(): Subscribt auf monry.evaluation.completed
    - start(): Erzeugt 5 initiale Basisstrategien (verschiedene Kategorien) und
               publiziert sie als monry.strategy.proposed
    - process_message(): Empfängt Evaluationsergebnisse und mutiert erfolgreiche
                         Strategien zu neuen Kandidaten
    """

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        goal: Optional[Goal] = None,
        budget: Optional[SearchBudget] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(agent_id, message_bus)
        self._goal = goal
        self._budget = budget or SearchBudget()
        self._toolbox = IndicatorToolbox(seed=seed)
        self._rng = random.Random(seed)
        self._strategy_counter = 0
        self._known_strategies: Dict[str, Strategy] = {}
        self._successful_strategies: List[Strategy] = []

    # ------------------------------------------------------------------
    # Agent Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Subscribt auf Evaluations-Ergebnisse."""
        super().initialize()
        self._message_bus.subscribe(MONRY_EVALUATION_COMPLETED, self._agent_id, self.process_message)

    def start(self) -> None:
        """
        Erzeugt 5 initiale Basisstrategien aus verschiedenen Indikator-Kategorien
        und publiziert sie als monry.strategy.proposed.
        """
        super().start()
        for category in _INITIAL_CATEGORIES:
            if not self._budget.can_propose():
                break
            strategy = self._create_base_strategy(category)
            self._register_and_publish(strategy)

    def stop(self) -> None:
        """Beendet die Exploration geordnet."""
        super().stop()

    def process_message(self, message: Message) -> None:
        """
        Verarbeitet ein Evaluations-Ergebnis und erzeugt neue Strategien.

        Erwartet im payload:
        - strategy_id: str — ID der evaluierten Strategie
        - score: float — Gesamtscore (0–100)
        - passed: bool — Hat die Strategie die harten Kriterien bestanden?
        """
        if message.topic != MONRY_EVALUATION_COMPLETED:
            return

        payload = message.payload
        strategy_id: str = payload.get("strategy_id", "")
        score: float = float(payload.get("score", 0.0))
        passed: bool = bool(payload.get("passed", False))

        if passed and strategy_id in self._known_strategies:
            strategy = self._known_strategies[strategy_id]
            strategy.score = score
            if strategy not in self._successful_strategies:
                self._successful_strategies.append(strategy)

        self._budget.record_round_result(score)

        if self._budget.is_exhausted():
            self._publish_budget_exhausted()
            return

        if not passed or strategy_id not in self._known_strategies:
            return

        parent = self._known_strategies[strategy_id]
        weights = self._budget.get_mutation_weights(self._budget.best_score)
        mutation_type = self._weighted_choice(weights)

        mutations = self._apply_mutation(parent, mutation_type)
        for strategy in mutations:
            if not self._budget.can_propose():
                break
            self._register_and_publish(strategy)

    # ------------------------------------------------------------------
    # Strategie-Erzeugung
    # ------------------------------------------------------------------

    def _create_base_strategy(self, category: str) -> Strategy:
        """Erzeugt eine initiale Basisstrategie für eine Indikator-Kategorie."""
        indicator = self._toolbox.get_random_indicator_from_category(category)
        strategy_id = self._next_strategy_id()
        name = self._build_name([indicator])
        works_when = dict(_WORKS_WHEN_DEFAULTS.get(category, {"market_phase": "any"}))

        return Strategy(
            strategy_id=strategy_id,
            name=name,
            rules=[indicator],
            hypothesis=f"Initial exploration of {category} signals via {indicator['name'].upper()}",
            parent_strategy=None,
            mutation_type=None,
            mutation_description=None,
            works_when=works_when,
            generation=0,
        )

    def _apply_mutation(
        self, parent: Strategy, mutation_type: str
    ) -> List[Strategy]:
        """Wendet eine Mutation auf eine Elternstrategie an."""
        if mutation_type == "parameter_variation":
            return self._mutate_parameter(parent)
        elif mutation_type == "add_filter":
            return self._add_filter(parent)
        elif mutation_type == "remove_filter":
            return self._remove_filter(parent)
        elif mutation_type == "cross_combine":
            return self._cross_combine(parent)
        return []

    def _mutate_parameter(self, parent: Strategy) -> List[Strategy]:
        """Erzeugt eine Variante mit einem leicht veränderten Parameter."""
        if not parent.rules:
            return []

        rule_idx = self._rng.randrange(len(parent.rules))
        rule = parent.rules[rule_idx]
        if not rule.get("parameters"):
            return []

        param_name = self._rng.choice(list(rule["parameters"].keys()))
        new_rule = self._toolbox.mutate_parameter(rule, param_name)

        old_val = rule["parameters"][param_name]
        new_val = new_rule["parameters"][param_name]

        new_rules = list(parent.rules)
        new_rules[rule_idx] = new_rule

        strategy_id = self._next_strategy_id()
        description = f"{rule['name'].upper()} {param_name}: {old_val} → {new_val}"

        return [
            Strategy(
                strategy_id=strategy_id,
                name=self._build_name(new_rules),
                rules=new_rules,
                hypothesis=(
                    f"Parameter variation of {parent.strategy_id}: "
                    f"{description}"
                ),
                parent_strategy=parent.strategy_id,
                mutation_type="parameter_variation",
                mutation_description=description,
                works_when=dict(parent.works_when),
                generation=parent.generation + 1,
            )
        ]

    def _add_filter(self, parent: Strategy) -> List[Strategy]:
        """Fügt einen kompatiblen Indikator als neuen Filter hinzu."""
        compatible = self._toolbox.get_compatible_filters(parent.rules)
        if not compatible:
            return []

        new_indicator_def = self._rng.choice(compatible)
        new_indicator = self._toolbox.get_random_indicator_from_category(
            new_indicator_def["category"]
        )
        new_rules = list(parent.rules) + [new_indicator]

        works_when = dict(parent.works_when)
        new_category = new_indicator["category"]
        if new_category == "volatility":
            works_when["volatility"] = "high"
        elif new_category == "trend":
            works_when["market_phase"] = "uptrend"
        elif new_category == "volume":
            works_when["volume_confirmed"] = "true"

        strategy_id = self._next_strategy_id()
        description = (
            f"Added {new_indicator['name'].upper()} ({new_indicator['category']}) filter"
        )

        return [
            Strategy(
                strategy_id=strategy_id,
                name=self._build_name(new_rules),
                rules=new_rules,
                hypothesis=(
                    f"Added {new_indicator['name'].upper()} filter to "
                    f"{parent.strategy_id} because parent may have "
                    f"false signals without {new_indicator['category']} confirmation"
                ),
                parent_strategy=parent.strategy_id,
                mutation_type="add_filter",
                mutation_description=description,
                works_when=works_when,
                generation=parent.generation + 1,
            )
        ]

    def _remove_filter(self, parent: Strategy) -> List[Strategy]:
        """Entfernt einen Filter (nur wenn > 2 Regeln vorhanden)."""
        if len(parent.rules) <= 2:
            return []

        remove_idx = self._rng.randrange(len(parent.rules))
        removed_rule = parent.rules[remove_idx]
        new_rules = [r for i, r in enumerate(parent.rules) if i != remove_idx]

        strategy_id = self._next_strategy_id()
        description = f"Removed {removed_rule['name'].upper()} filter"

        return [
            Strategy(
                strategy_id=strategy_id,
                name=self._build_name(new_rules),
                rules=new_rules,
                hypothesis=(
                    f"Simplified {parent.strategy_id} by removing "
                    f"{removed_rule['name'].upper()} — strategy may have been too restrictive"
                ),
                parent_strategy=parent.strategy_id,
                mutation_type="remove_filter",
                mutation_description=description,
                works_when=dict(parent.works_when),
                generation=parent.generation + 1,
            )
        ]

    def _cross_combine(self, parent: Strategy) -> List[Strategy]:
        """
        Kombiniert zwei erfolgreiche Strategien.

        Nimmt alle Regeln der Elternstrategie und fügt den besten Filter
        einer anderen erfolgreichen Strategie hinzu.
        Benötigt mindestens zwei erfolgreiche Strategien.
        """
        other_candidates = [
            s for s in self._successful_strategies if s.strategy_id != parent.strategy_id
        ]
        if not other_candidates:
            return []

        other = self._rng.choice(other_candidates)
        if not other.rules:
            return []

        # Besten Filter (letzten/einzigen) aus der anderen Strategie nehmen
        donor_rule = self._rng.choice(other.rules)

        # Prüfen ob der Filter kompatibel ist (nicht bereits vorhanden)
        existing_names = {r["name"] for r in parent.rules}
        if donor_rule["name"] in existing_names:
            return []

        new_rules = list(parent.rules) + [donor_rule]

        works_when = dict(parent.works_when)
        works_when.update(
            {k: v for k, v in other.works_when.items() if k not in works_when}
        )

        strategy_id = self._next_strategy_id()
        description = (
            f"Combined {parent.strategy_id} with {donor_rule['name'].upper()} "
            f"from {other.strategy_id}"
        )

        return [
            Strategy(
                strategy_id=strategy_id,
                name=self._build_name(new_rules),
                rules=new_rules,
                hypothesis=(
                    f"Cross-combined {parent.strategy_id} (gen {parent.generation}) "
                    f"with best filter from {other.strategy_id} (gen {other.generation}): "
                    f"{donor_rule['name'].upper()} may provide complementary signal"
                ),
                parent_strategy=parent.strategy_id,
                mutation_type="cross_combine",
                mutation_description=description,
                works_when=works_when,
                generation=max(parent.generation, other.generation) + 1,
            )
        ]

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _next_strategy_id(self) -> str:
        self._strategy_counter += 1
        return f"exp_{self._strategy_counter:03d}"

    def _build_name(self, rules: List[Dict[str, Any]]) -> str:
        """Erzeugt einen lesbaren Strategienamen aus den Regeln."""
        parts = []
        for rule in rules:
            name = rule["name"].upper()
            condition = rule.get("condition", "").replace(" ", "_")
            parts.append(f"{name}_{condition}" if condition else name)
        return "_plus_".join(parts)

    def _register_and_publish(self, strategy: Strategy) -> None:
        """Registriert eine Strategie intern und publiziert sie über den Bus."""
        self._known_strategies[strategy.strategy_id] = strategy
        self._budget.record_proposal()
        self._message_bus.publish(
            Message(
                topic=MONRY_STRATEGY_PROPOSED,
                payload={
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "rules": strategy.rules,
                    "hypothesis": strategy.hypothesis,
                    "parent_strategy": strategy.parent_strategy,
                    "mutation_type": strategy.mutation_type,
                    "mutation_description": strategy.mutation_description,
                    "works_when": strategy.works_when,
                    "generation": strategy.generation,
                },
                sender_id=self._agent_id,
            )
        )

    def _publish_budget_exhausted(self) -> None:
        """Publiziert eine Budget-Exhausted-Nachricht."""
        self._message_bus.publish(
            Message(
                topic=MONRY_EXPLORATION_BUDGET_EXHAUSTED,
                payload={
                    "total_strategies": self._budget.total_strategies,
                    "stale_rounds": self._budget.stale_rounds,
                    "best_score": self._budget.best_score,
                },
                sender_id=self._agent_id,
            )
        )

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Wählt einen Schlüssel basierend auf relativen Gewichten."""
        keys = list(weights.keys())
        values = list(weights.values())
        total = sum(values)
        if total == 0:
            return self._rng.choice(keys)
        r = self._rng.random() * total
        cumulative = 0.0
        for key, weight in zip(keys, values):
            cumulative += weight
            if r <= cumulative:
                return key
        return keys[-1]
