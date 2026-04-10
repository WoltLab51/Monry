"""Tests for monry/agents/backtest_agent.py — BacktestAgent."""

from __future__ import annotations

import pytest

from monry._genus_stubs import Message, MessageBus
from monry.agents.backtest_agent import BacktestAgent
from monry.data.fetcher import MarketDataFetcher
from monry.models.candle import Candle
from monry.topics import MONRY_BACKTEST_COMPLETED, MONRY_STRATEGY_PROPOSED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    hold_period: int = 5,
    warmup_period: int = 50,
) -> tuple[BacktestAgent, MessageBus]:
    bus = MessageBus()
    agent = BacktestAgent(
        agent_id="backtest_agent",
        message_bus=bus,
        hold_period=hold_period,
        warmup_period=warmup_period,
    )
    agent.initialize()
    return agent, bus


def _synthetic_candles(n: int = 300) -> list[Candle]:
    return MarketDataFetcher.generate_synthetic("TEST", "2019-01-01", "2022-01-01", seed=99)[
        :n
    ]


def _rsi_rule(threshold: int = 99) -> dict:
    """RSI < threshold — with threshold=99 this nearly always triggers."""
    return {
        "name": "rsi",
        "parameters": {"period": 14, "threshold_oversold": threshold},
        "condition": "< threshold_oversold",
    }


def _roc_positive_rule() -> dict:
    return {"name": "roc", "parameters": {"period": 5}, "condition": "> 0"}


def _publish_strategy(bus: MessageBus, rules: list, strategy_id: str = "s1") -> None:
    bus.publish(
        Message(
            topic=MONRY_STRATEGY_PROPOSED,
            payload={
                "strategy_id": strategy_id,
                "rules": rules,
            },
            sender_id="test",
        )
    )


# ---------------------------------------------------------------------------
# No candles
# ---------------------------------------------------------------------------


class TestNoCandles:
    def test_backtest_with_no_candles_produces_0_signals(self) -> None:
        agent, bus = _make_agent()
        # No candles set → empty list
        _publish_strategy(bus, [_rsi_rule()])

        published = bus.get_published(MONRY_BACKTEST_COMPLETED)
        assert len(published) == 1
        assert published[0].payload["num_signals"] == 0

    def test_backtest_with_too_few_candles_produces_0_signals(self) -> None:
        agent, bus = _make_agent(warmup_period=200)
        agent.set_candles(_synthetic_candles(10))
        _publish_strategy(bus, [_rsi_rule()])

        published = bus.get_published(MONRY_BACKTEST_COMPLETED)
        assert published[0].payload["num_signals"] == 0


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


class TestPublishing:
    def test_backtest_publishes_on_correct_topic(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        published = bus.get_published(MONRY_BACKTEST_COMPLETED)
        assert len(published) == 1

    def test_backtest_result_contains_all_required_fields(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()], strategy_id="exp_001")

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        required = {
            "strategy_id",
            "num_signals",
            "profit_factor",
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
            "trade_dates",
            "market_phases",
        }
        assert required.issubset(payload.keys())
        assert payload["strategy_id"] == "exp_001"

    def test_strategy_id_is_preserved(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()], strategy_id="my_strategy_42")

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        assert payload["strategy_id"] == "my_strategy_42"


# ---------------------------------------------------------------------------
# Metric bounds
# ---------------------------------------------------------------------------


class TestMetricBounds:
    def test_win_rate_is_between_0_and_1(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        if payload["num_signals"] > 0:
            assert 0.0 <= payload["win_rate"] <= 1.0

    def test_profit_factor_is_non_negative(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        assert payload["profit_factor"] >= 0.0

    def test_max_drawdown_is_non_negative(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        assert payload["max_drawdown"] >= 0.0


# ---------------------------------------------------------------------------
# Trade dates and market phases
# ---------------------------------------------------------------------------


class TestTradeDatesAndPhases:
    def test_trade_dates_are_valid_strings(self) -> None:
        import re
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in payload["trade_dates"]:
            assert date_pattern.match(d), f"Bad date format: {d}"

    def test_market_phases_are_valid_strings(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        valid_phases = {"uptrend", "downtrend", "sideways"}
        for phase in payload["market_phases"]:
            assert phase in valid_phases, f"Invalid phase: {phase}"

    def test_trade_dates_count_matches_num_signals(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        assert len(payload["trade_dates"]) == payload["num_signals"]
        assert len(payload["market_phases"]) == payload["num_signals"]


# ---------------------------------------------------------------------------
# Hold period
# ---------------------------------------------------------------------------


class TestHoldPeriod:
    def test_hold_period_is_respected(self) -> None:
        """With hold_period=1, last signal must be at most index len-2."""
        candles = _synthetic_candles(100)
        agent, bus = _make_agent(hold_period=1, warmup_period=20)
        agent.set_candles(candles)
        _publish_strategy(bus, [_rsi_rule()])

        payload = bus.get_published(MONRY_BACKTEST_COMPLETED)[0].payload
        # Trade dates must not exceed the second-to-last candle date
        if payload["trade_dates"]:
            last_candle_date = candles[-2].date
            for d in payload["trade_dates"]:
                assert d <= last_candle_date


# ---------------------------------------------------------------------------
# Multiple strategies
# ---------------------------------------------------------------------------


class TestMultipleStrategies:
    def test_each_strategy_gets_own_result(self) -> None:
        agent, bus = _make_agent()
        agent.set_candles(_synthetic_candles())

        _publish_strategy(bus, [_rsi_rule()], strategy_id="s1")
        _publish_strategy(bus, [_roc_positive_rule()], strategy_id="s2")

        published = bus.get_published(MONRY_BACKTEST_COMPLETED)
        assert len(published) == 2
        ids = {msg.payload["strategy_id"] for msg in published}
        assert ids == {"s1", "s2"}
