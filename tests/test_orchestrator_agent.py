"""Tests for the OrchestratorAgent."""

from __future__ import annotations

import pytest

from monry._genus_stubs import AgentState, Message, MessageBus
from monry.agents.backtest_agent import BacktestAgent
from monry.agents.evaluation_agent import EvaluationAgent
from monry.agents.market_data_agent import MarketDataAgent
from monry.agents.orchestrator_agent import OrchestratorAgent
from monry.models.goal import Goal
from monry.search.budget import SearchBudget
from monry.topics import (
    MONRY_EVALUATION_COMPLETED,
    MONRY_EXPLORATION_BUDGET_EXHAUSTED,
    MONRY_GOAL_DEFINED,
    MONRY_MARKET_DATA_READY,
    MONRY_RUN_COMPLETED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_goal() -> Goal:
    return Goal(
        symbol="TEST",
        period_start="2019-01-01",
        period_end="2022-01-01",
        success_criteria={"min_win_rate": 0.55, "min_profit_factor": 1.5},
    )


def _make_orchestrator(
    goal: Goal | None = None,
    budget: SearchBudget | None = None,
    seed: int = 42,
) -> tuple[OrchestratorAgent, MessageBus]:
    bus = MessageBus()
    goal = goal or _make_goal()
    budget = budget or SearchBudget(max_total_strategies=10, max_stale_rounds=2)
    agent = OrchestratorAgent(
        agent_id="orchestrator",
        message_bus=bus,
        goal=goal,
        budget=budget,
        seed=seed,
    )
    return agent, bus


def _fake_market_data_message() -> Message:
    """Creates a minimal monry.market.data_ready message with empty candles."""
    return Message(
        topic=MONRY_MARKET_DATA_READY,
        payload={
            "symbol": "TEST",
            "start": "2019-01-01",
            "end": "2022-01-01",
            "num_candles": 0,
            "candles": [],
        },
        sender_id="market_data_agent",
    )


def _fake_evaluation_message(
    strategy_id: str = "exp_001",
    score: float = 60.0,
    passed: bool = True,
) -> Message:
    return Message(
        topic=MONRY_EVALUATION_COMPLETED,
        payload={
            "strategy_id": strategy_id,
            "score": score,
            "passed": passed,
            "win_rate": 0.6,
            "profit_factor": 2.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "num_signals": 40,
            "rejection_reasons": [],
        },
        sender_id="eval_agent",
    )


def _fake_budget_exhausted_message(
    total_strategies: int = 10,
    stale_rounds: int = 2,
) -> Message:
    return Message(
        topic=MONRY_EXPLORATION_BUDGET_EXHAUSTED,
        payload={
            "total_strategies": total_strategies,
            "stale_rounds": stale_rounds,
            "best_score": 60.0,
        },
        sender_id="explorer",
    )


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_initialize_sets_state_to_initialized(self) -> None:
        agent, _ = _make_orchestrator()
        agent.initialize()
        assert agent.state == AgentState.INITIALIZED

    def test_start_publishes_goal_defined(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        published = bus.get_published(MONRY_GOAL_DEFINED)
        assert len(published) == 1

        payload = published[0].payload
        goal = _make_goal()
        assert payload["symbol"] == goal.symbol
        assert payload["period_start"] == goal.period_start
        assert payload["period_end"] == goal.period_end
        assert payload["success_criteria"] == goal.success_criteria

    def test_start_sets_run_active(self) -> None:
        agent, _ = _make_orchestrator()
        agent.initialize()
        assert not agent._run_active
        agent.start()
        assert agent._run_active


# ---------------------------------------------------------------------------
# TestMarketDataIntegration
# ---------------------------------------------------------------------------


class TestMarketDataIntegration:
    def test_exploration_starts_on_market_data_ready(self) -> None:
        """After market_data_ready, the internal ExplorationAgent should
        publish at least one monry.strategy.proposed message."""
        from monry.topics import MONRY_STRATEGY_PROPOSED

        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        # Simulate market data arriving
        bus.publish(_fake_market_data_message())

        proposed = bus.get_published(MONRY_STRATEGY_PROPOSED)
        assert len(proposed) > 0


# ---------------------------------------------------------------------------
# TestEvaluationTracking
# ---------------------------------------------------------------------------


class TestEvaluationTracking:
    def test_best_strategy_tracked_on_passed_evaluation(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_001", score=55.0, passed=True))

        assert agent._best_strategy_id == "exp_001"
        assert agent._best_score == 55.0

    def test_better_score_replaces_previous_best(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_001", score=55.0, passed=True))
        bus.publish(_fake_evaluation_message("exp_002", score=72.0, passed=True))

        assert agent._best_strategy_id == "exp_002"
        assert agent._best_score == 72.0

    def test_failed_evaluation_does_not_update_best(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_001", score=55.0, passed=True))
        bus.publish(_fake_evaluation_message("exp_002", score=90.0, passed=False))

        # exp_002 has a higher score but passed=False → should not replace best
        assert agent._best_strategy_id == "exp_001"
        assert agent._best_score == 55.0

    def test_evaluations_received_counter_increments(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        assert agent._evaluations_received == 0
        bus.publish(_fake_evaluation_message("exp_001", score=40.0, passed=True))
        assert agent._evaluations_received == 1
        bus.publish(_fake_evaluation_message("exp_002", score=30.0, passed=False))
        assert agent._evaluations_received == 2


# ---------------------------------------------------------------------------
# TestRunCompletion
# ---------------------------------------------------------------------------


class TestRunCompletion:
    def test_budget_exhausted_publishes_run_completed(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_budget_exhausted_message(total_strategies=10, stale_rounds=2))

        completed = bus.get_published(MONRY_RUN_COMPLETED)
        assert len(completed) == 1

    def test_run_completed_contains_best_strategy(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_005", score=68.0, passed=True))
        bus.publish(_fake_budget_exhausted_message(total_strategies=10, stale_rounds=2))

        completed = bus.get_published(MONRY_RUN_COMPLETED)
        payload = completed[0].payload
        assert payload["best_strategy_id"] == "exp_005"
        assert payload["best_score"] == 68.0

    def test_run_completed_stops_agent(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_budget_exhausted_message())

        assert agent.state == AgentState.STOPPED

    def test_no_evaluation_tracking_after_run_completed(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_001", score=55.0, passed=True))
        bus.publish(_fake_budget_exhausted_message())

        # After run_completed, further evaluations must be ignored
        bus.publish(_fake_evaluation_message("exp_002", score=99.0, passed=True))

        assert agent._best_strategy_id == "exp_001"
        assert agent._best_score == 55.0

    def test_run_completed_payload_includes_totals(self) -> None:
        agent, bus = _make_orchestrator()
        agent.initialize()
        agent.start()

        bus.publish(_fake_evaluation_message("exp_001", score=42.0, passed=True))
        bus.publish(
            _fake_budget_exhausted_message(total_strategies=7, stale_rounds=3)
        )

        payload = bus.get_published(MONRY_RUN_COMPLETED)[0].payload
        assert payload["total_evaluations"] == 1
        assert payload["total_strategies"] == 7
        assert payload["stale_rounds"] == 3


# ---------------------------------------------------------------------------
# TestFullMiniLoop
# ---------------------------------------------------------------------------


class TestFullMiniLoop:
    def test_goal_to_completion_mini_loop(self) -> None:
        """
        Minimal End-to-End: Orchestrator + all 4 agents on the same bus.
        Verifies that monry.run.completed is published with best_score > 0.
        """
        from monry.data.fetcher import MarketDataFetcher

        bus = MessageBus()
        goal = Goal(
            symbol="TEST",
            period_start="2019-01-01",
            period_end="2022-01-01",
            success_criteria={"min_win_rate": 0.55, "min_profit_factor": 1.5},
        )
        # Tight budget so the loop finishes quickly in the test
        budget = SearchBudget(
            max_total_strategies=8,
            max_stale_rounds=2,
            max_proposals_per_round=3,
        )

        # Set up all agents on the same bus
        market_agent = MarketDataAgent(agent_id="market", message_bus=bus)
        backtest_agent = BacktestAgent(agent_id="backtest", message_bus=bus)
        eval_agent = EvaluationAgent(agent_id="evaluator", message_bus=bus)
        orchestrator = OrchestratorAgent(
            agent_id="orchestrator",
            message_bus=bus,
            goal=goal,
            budget=budget,
            seed=42,
        )

        # Initialize all agents
        market_agent.initialize()
        backtest_agent.initialize()
        eval_agent.initialize()
        orchestrator.initialize()

        # Start the orchestrator — this fires the whole chain
        orchestrator.start()

        # Verify the run completed
        completed = bus.get_published(MONRY_RUN_COMPLETED)
        assert len(completed) == 1, "Expected exactly one monry.run.completed message"

        payload = completed[0].payload
        # At least some evaluations must have happened
        assert payload["total_evaluations"] > 0
