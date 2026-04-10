"""
BacktestAgent — Runs a strategy against historical candle data.

Subscribes to: monry.strategy.proposed
Publishes to:  monry.backtest.completed

Requires candle data to be set via set_candles() before processing messages.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

try:
    from genus.core.agent import Agent, AgentState
    from genus.communication.message_bus import MessageBus, Message
except ImportError:
    from monry._genus_stubs import Agent, AgentState, MessageBus, Message  # type: ignore[assignment]

from monry.analysis.market_phase_detector import detect_market_phase
from monry.analysis.signal_evaluator import SignalEvaluator
from monry.models.backtest_result import BacktestResult
from monry.models.candle import Candle
from monry.topics import MONRY_BACKTEST_COMPLETED, MONRY_MARKET_DATA_READY, MONRY_STRATEGY_PROPOSED


class BacktestAgent(Agent):
    """
    Runs a strategy against historical candle data and produces results.

    Backtest logic:
    1. Receive strategy proposal
    2. For each day in the candle data (starting after warmup_period):
       a. Take a window of candles up to that day
       b. Evaluate all strategy rules using SignalEvaluator
       c. If signal triggered → record the trade date
    3. For each recorded trade:
       a. Calculate return after hold_period days
       b. Determine if trade was profitable
    4. Calculate aggregate metrics and publish BacktestResult
    """

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        hold_period: int = 5,
        warmup_period: int = 200,
    ) -> None:
        super().__init__(agent_id=agent_id, message_bus=message_bus)
        self._candles: List[Candle] = []
        self._hold_period = hold_period
        self._warmup_period = warmup_period
        self._evaluator = SignalEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_candles(self, candles: List[Candle]) -> None:
        """Sets the candle data used for backtesting."""
        self._candles = list(candles)

    def initialize(self) -> None:
        """Subscribe to monry.strategy.proposed and monry.market.data_ready."""
        self._message_bus.subscribe(
            MONRY_STRATEGY_PROPOSED, self._agent_id, self.process_message
        )
        self._message_bus.subscribe(
            MONRY_MARKET_DATA_READY, self._agent_id, self._on_market_data_ready
        )
        self._state = AgentState.INITIALIZED

    def _on_market_data_ready(self, message: Message) -> None:
        """Receives candle data from MarketDataAgent and stores it for backtesting."""
        payload = message.payload
        candle_dicts = payload.get("candles", [])
        self._candles = [
            Candle(
                date=c["date"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
            )
            for c in candle_dicts
        ]

    def process_message(self, message: Message) -> None:
        """Processes a strategy proposal: runs backtest, publishes result."""
        payload = message.payload
        strategy_id: str = payload.get("strategy_id", "")
        rules: List[Dict[str, Any]] = payload.get("rules", [])

        result = self._run_backtest(strategy_id, rules)
        self._publish_result(result)

    # ------------------------------------------------------------------
    # Backtest logic
    # ------------------------------------------------------------------

    def _run_backtest(
        self, strategy_id: str, rules: List[Dict[str, Any]]
    ) -> BacktestResult:
        """Core backtest loop."""
        candles = self._candles

        # Not enough candles
        if len(candles) < self._warmup_period + self._hold_period + 1:
            return BacktestResult(
                strategy_id=strategy_id,
                num_signals=0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                trade_dates=[],
                market_phases=[],
            )

        signal_indices: List[int] = []

        # Evaluate strategy over all days after warmup
        for i in range(self._warmup_period, len(candles) - self._hold_period):
            window = candles[: i + 1]
            if self._evaluator.evaluate(rules, window):
                signal_indices.append(i)

        if not signal_indices:
            return BacktestResult(
                strategy_id=strategy_id,
                num_signals=0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                trade_dates=[],
                market_phases=[],
            )

        # Calculate per-trade returns
        returns: List[float] = []
        trade_dates: List[str] = []
        market_phases: List[str] = []

        for idx in signal_indices:
            entry_price = candles[idx].close
            exit_price = candles[idx + self._hold_period].close

            if entry_price <= 0:
                continue

            ret = (exit_price - entry_price) / entry_price
            returns.append(ret)
            trade_dates.append(candles[idx].date)
            market_phases.append(detect_market_phase(candles[: idx + 1]))

        if not returns:
            return BacktestResult(
                strategy_id=strategy_id,
                num_signals=0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                trade_dates=[],
                market_phases=[],
            )

        # Aggregate metrics
        num_signals = len(returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = len(wins) / num_signals

        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0.0 else float('inf')

        mean_ret = sum(returns) / num_signals
        if num_signals > 1:
            variance = sum((r - mean_ret) ** 2 for r in returns) / (num_signals - 1)
            std_ret = math.sqrt(variance)
        else:
            std_ret = 0.0
        sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0.0 else 0.0

        max_drawdown = self._calculate_max_drawdown(returns)

        return BacktestResult(
            strategy_id=strategy_id,
            num_signals=num_signals,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            trade_dates=trade_dates,
            market_phases=market_phases,
        )

    @staticmethod
    def _calculate_max_drawdown(returns: List[float]) -> float:
        """Calculates the worst peak-to-trough drawdown in cumulative returns."""
        if not returns:
            return 0.0

        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0

        for ret in returns:
            cumulative *= 1.0 + ret
            if cumulative > peak:
                peak = cumulative
            if peak > 0:
                dd = (peak - cumulative) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_result(self, result: BacktestResult) -> None:
        """Publishes a BacktestResult on monry.backtest.completed."""
        self._message_bus.publish(
            Message(
                topic=MONRY_BACKTEST_COMPLETED,
                payload={
                    "strategy_id": result.strategy_id,
                    "num_signals": result.num_signals,
                    "profit_factor": result.profit_factor,
                    "sharpe_ratio": result.sharpe_ratio,
                    "win_rate": result.win_rate,
                    "max_drawdown": result.max_drawdown,
                    "trade_dates": result.trade_dates,
                    "market_phases": result.market_phases,
                },
                sender_id=self._agent_id,
            )
        )
