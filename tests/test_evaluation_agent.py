"""Tests for monry/agents/evaluation_agent.py — EvaluationAgent."""

from __future__ import annotations

import pytest

from monry._genus_stubs import Message, MessageBus
from monry.agents.evaluation_agent import EvaluationAgent
from monry.models.backtest_result import BacktestResult
from monry.search.selection import SelectionFilter
from monry.topics import MONRY_BACKTEST_COMPLETED, MONRY_EVALUATION_COMPLETED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    selection_filter: SelectionFilter | None = None,
) -> tuple[EvaluationAgent, MessageBus]:
    bus = MessageBus()
    agent = EvaluationAgent(
        agent_id="eval_agent",
        message_bus=bus,
        selection_filter=selection_filter,
    )
    agent.initialize()
    return agent, bus


def _trade_dates(n: int = 50) -> list[str]:
    """50 trade dates spread across 6 months."""
    dates = (
        [f"2023-01-{d:02d}" for d in range(1, 9)]
        + [f"2023-02-{d:02d}" for d in range(1, 9)]
        + [f"2023-03-{d:02d}" for d in range(1, 9)]
        + [f"2023-04-{d:02d}" for d in range(1, 9)]
        + [f"2023-05-{d:02d}" for d in range(1, 9)]
        + [f"2023-06-{d:02d}" for d in range(1, 11)]
    )
    return dates[:n]


def _publish_backtest(
    bus: MessageBus,
    strategy_id: str = "s1",
    num_signals: int = 50,
    profit_factor: float = 1.5,
    sharpe_ratio: float = 1.0,
    win_rate: float = 0.6,
    max_drawdown: float = 0.10,
    trade_dates: list | None = None,
    market_phases: list | None = None,
) -> None:
    if trade_dates is None:
        trade_dates = _trade_dates(num_signals)
    if market_phases is None:
        market_phases = ["uptrend", "downtrend", "sideways"] * (num_signals // 3 + 1)
        market_phases = market_phases[:num_signals]

    bus.publish(
        Message(
            topic=MONRY_BACKTEST_COMPLETED,
            payload={
                "strategy_id": strategy_id,
                "num_signals": num_signals,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "trade_dates": trade_dates,
                "market_phases": market_phases,
            },
            sender_id="backtest_agent",
        )
    )


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------


class TestScoreBounds:
    def test_score_is_between_0_and_100(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus)

        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert 0.0 <= payload["score"] <= 100.0

    def test_score_is_0_for_worst_possible_result(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(
            bus,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
        )
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert payload["score"] == pytest.approx(0.0)

    def test_score_is_100_for_best_possible_result(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(
            bus,
            win_rate=1.0,
            profit_factor=10.0,
            sharpe_ratio=5.0,
            max_drawdown=0.0,
        )
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert payload["score"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Pass / Fail via SelectionFilter
# ---------------------------------------------------------------------------


class TestPassFail:
    def test_passed_true_when_selection_filter_approves(self) -> None:
        agent, bus = _make_agent(
            selection_filter=SelectionFilter(
                min_signals=30,
                min_profit_factor=1.0,
                min_sharpe=0.0,
                max_single_month_ratio=0.8,
                min_market_phases=2,
            )
        )
        _publish_backtest(bus, num_signals=50, profit_factor=1.5, sharpe_ratio=1.0)
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert payload["passed"] is True

    def test_passed_false_when_selection_filter_rejects(self) -> None:
        agent, bus = _make_agent(
            selection_filter=SelectionFilter(min_signals=100)
        )
        _publish_backtest(bus, num_signals=5)
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert payload["passed"] is False

    def test_rejection_reasons_forwarded_from_selection_filter(self) -> None:
        agent, bus = _make_agent(
            selection_filter=SelectionFilter(min_signals=100)
        )
        _publish_backtest(bus, num_signals=5)
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert len(payload["rejection_reasons"]) > 0

    def test_no_rejection_reasons_for_passing_strategy(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus, num_signals=50, profit_factor=1.5, sharpe_ratio=1.0)
        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        if payload["passed"]:
            assert payload["rejection_reasons"] == []


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


class TestPublishing:
    def test_publishes_on_correct_topic(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus)

        published = bus.get_published(MONRY_EVALUATION_COMPLETED)
        assert len(published) == 1

    def test_result_contains_all_required_fields(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus, strategy_id="exp_007")

        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        required = {
            "strategy_id",
            "score",
            "passed",
            "win_rate",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown",
            "num_signals",
            "rejection_reasons",
        }
        assert required.issubset(payload.keys())
        assert payload["strategy_id"] == "exp_007"

    def test_strategy_id_is_preserved(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus, strategy_id="alpha_001")

        payload = bus.get_published(MONRY_EVALUATION_COMPLETED)[0].payload
        assert payload["strategy_id"] == "alpha_001"


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------


class TestScoreComponents:
    def test_high_win_rate_increases_score(self) -> None:
        agent_low, bus_low = _make_agent()
        agent_high, bus_high = _make_agent()

        _publish_backtest(bus_low, win_rate=0.5, profit_factor=1.5, sharpe_ratio=1.0, max_drawdown=0.1)
        _publish_backtest(bus_high, win_rate=0.8, profit_factor=1.5, sharpe_ratio=1.0, max_drawdown=0.1)

        score_low = bus_low.get_published(MONRY_EVALUATION_COMPLETED)[0].payload["score"]
        score_high = bus_high.get_published(MONRY_EVALUATION_COMPLETED)[0].payload["score"]
        assert score_high > score_low

    def test_low_drawdown_increases_score(self) -> None:
        agent_high_dd, bus_high_dd = _make_agent()
        agent_low_dd, bus_low_dd = _make_agent()

        _publish_backtest(bus_high_dd, max_drawdown=0.2)
        _publish_backtest(bus_low_dd, max_drawdown=0.0)

        score_high_dd = bus_high_dd.get_published(MONRY_EVALUATION_COMPLETED)[0].payload["score"]
        score_low_dd = bus_low_dd.get_published(MONRY_EVALUATION_COMPLETED)[0].payload["score"]
        assert score_low_dd > score_high_dd

    def test_multiple_backtest_results_each_get_evaluation(self) -> None:
        agent, bus = _make_agent()
        _publish_backtest(bus, strategy_id="s1")
        _publish_backtest(bus, strategy_id="s2")

        published = bus.get_published(MONRY_EVALUATION_COMPLETED)
        assert len(published) == 2
        ids = {msg.payload["strategy_id"] for msg in published}
        assert ids == {"s1", "s2"}
