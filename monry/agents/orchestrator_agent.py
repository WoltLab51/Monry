"""
OrchestratorAgent — Central coordinator for the Monry exploration loop.

Coordinates the entire workflow without duplicating any domain logic:
1. Starts the loop by publishing the Goal (triggers MarketDataAgent)
2. Creates the ExplorationAgent once market data is ready
3. Tracks evaluation results (only the best score)
4. Ends the run cleanly when the budget is exhausted

Subscribes to:
- monry.market.data_ready          — Market data ready → start ExplorationAgent
- monry.evaluation.completed       — Strategy evaluated → track best result
- monry.exploration.budget_exhausted — Budget exhausted → end the run

Publishes:
- monry.goal.defined               — on start(), triggers MarketDataAgent
- monry.run.completed              — when budget is exhausted, final result
"""

from __future__ import annotations

from typing import Optional

try:
    from genus.core.agent import Agent, AgentState
    from genus.communication.message_bus import MessageBus, Message
except ImportError:
    from monry._genus_stubs import Agent, AgentState, MessageBus, Message  # type: ignore[assignment]

from monry.agents.exploration_agent import ExplorationAgent
from monry.models.goal import Goal
from monry.search.budget import SearchBudget
from monry.topics import (
    MONRY_EVALUATION_COMPLETED,
    MONRY_EXPLORATION_BUDGET_EXHAUSTED,
    MONRY_GOAL_DEFINED,
    MONRY_MARKET_DATA_READY,
    MONRY_RUN_COMPLETED,
)


class OrchestratorAgent(Agent):
    """
    Koordiniert den kompletten Explorations-Loop — ohne eigene Logik.

    Lifecycle:
    - initialize(): Subscribes to all 3 topics
    - start(): Publishes monry.goal.defined → triggers MarketDataAgent
    - _on_market_data_ready(): Creates and starts ExplorationAgent
    - _on_evaluation_completed(): Tracks best result
    - _on_budget_exhausted(): Publishes monry.run.completed, State → STOPPED
    """

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        goal: Goal,
        budget: SearchBudget,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(agent_id, message_bus)
        self._goal: Goal = goal
        self._budget: SearchBudget = budget
        self._seed: Optional[int] = seed
        self._best_strategy_id: Optional[str] = None
        self._best_score: float = 0.0
        self._evaluations_received: int = 0
        self._run_active: bool = False

    # ------------------------------------------------------------------
    # Agent Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Subscribes to all relevant topics."""
        self._message_bus.subscribe(
            MONRY_MARKET_DATA_READY, self._agent_id, self._on_market_data_ready
        )
        self._message_bus.subscribe(
            MONRY_EVALUATION_COMPLETED, self._agent_id, self._on_evaluation_completed
        )
        self._message_bus.subscribe(
            MONRY_EXPLORATION_BUDGET_EXHAUSTED, self._agent_id, self._on_budget_exhausted
        )
        self._state = AgentState.INITIALIZED

    def start(self) -> None:
        """Starts the loop: publishes the Goal which triggers the MarketDataAgent."""
        self._state = AgentState.RUNNING
        self._run_active = True
        self._message_bus.publish(
            Message(
                topic=MONRY_GOAL_DEFINED,
                payload={
                    "symbol": self._goal.symbol,
                    "period_start": self._goal.period_start,
                    "period_end": self._goal.period_end,
                    "success_criteria": self._goal.success_criteria,
                },
                sender_id=self._agent_id,
            )
        )

    def stop(self) -> None:
        """Stops the run in an orderly manner."""
        self._run_active = False
        self._state = AgentState.STOPPED

    def process_message(self, message: Message) -> None:
        """No-Op: all relevant topics are processed by dedicated handlers."""

    # ------------------------------------------------------------------
    # Interne Handler
    # ------------------------------------------------------------------

    def _on_market_data_ready(self, message: Message) -> None:
        """Creates and starts the ExplorationAgent once candle data is available."""
        exploration_agent = ExplorationAgent(
            agent_id=f"{self._agent_id}_explorer",
            message_bus=self._message_bus,
            goal=self._goal,
            budget=self._budget,
            seed=self._seed,
        )
        exploration_agent.initialize()
        exploration_agent.start()

    def _on_evaluation_completed(self, message: Message) -> None:
        """Tracks the best evaluation result."""
        if not self._run_active:
            return

        payload = message.payload
        strategy_id: str = payload.get("strategy_id", "")
        score: float = float(payload.get("score", 0.0))
        passed: bool = bool(payload.get("passed", False))

        self._evaluations_received += 1

        if passed and score > self._best_score:
            self._best_strategy_id = strategy_id
            self._best_score = score

    def _on_budget_exhausted(self, message: Message) -> None:
        """Ends the run and publishes the final result."""
        self._run_active = False
        self._message_bus.publish(
            Message(
                topic=MONRY_RUN_COMPLETED,
                payload={
                    "best_strategy_id": self._best_strategy_id,
                    "best_score": self._best_score,
                    "total_evaluations": self._evaluations_received,
                    "total_strategies": message.payload.get("total_strategies", 0),
                    "stale_rounds": message.payload.get("stale_rounds", 0),
                },
                sender_id=self._agent_id,
            )
        )
        self._state = AgentState.STOPPED
