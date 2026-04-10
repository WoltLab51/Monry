"""
EvaluationAgent — Evaluates backtest results and publishes evaluation scores.

Subscribes to: monry.backtest.completed
Publishes to:  monry.evaluation.completed

Score (0–100):
- 30 pts from win_rate       (linear: 0.5 → 0, ≥0.8 → 30)
- 25 pts from profit_factor  (linear: 1.0 → 0, ≥3.0 → 25)
- 25 pts from sharpe_ratio   (linear: 0.0 → 0, ≥2.0 → 25)
- 20 pts from max_drawdown   (inverse linear: 0.0 → 20, ≥0.2 → 0)
"""

from __future__ import annotations

try:
    from genus.core.agent import Agent, AgentState
    from genus.communication.message_bus import MessageBus, Message
except ImportError:
    from monry._genus_stubs import Agent, AgentState, MessageBus, Message  # type: ignore[assignment]

from monry.models.backtest_result import BacktestResult
from monry.search.selection import SelectionFilter
from monry.topics import MONRY_BACKTEST_COMPLETED, MONRY_EVALUATION_COMPLETED


class EvaluationAgent(Agent):
    """
    Evaluates backtest results using SelectionFilter and publishes evaluation.

    Published payload:
    {
        "strategy_id": "exp_007",
        "score": 72.5,
        "passed": true,
        "win_rate": 0.68,
        "profit_factor": 1.85,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "num_signals": 47,
        "rejection_reasons": []
    }
    """

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        selection_filter: SelectionFilter | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, message_bus=message_bus)
        self._filter = selection_filter if selection_filter is not None else SelectionFilter()

    def initialize(self) -> None:
        """Subscribe to monry.backtest.completed."""
        self._message_bus.subscribe(
            MONRY_BACKTEST_COMPLETED, self._agent_id, self.process_message
        )
        self._state = AgentState.INITIALIZED

    def process_message(self, message: Message) -> None:
        """Evaluate backtest result and publish evaluation."""
        payload = message.payload

        result = BacktestResult(
            strategy_id=payload.get("strategy_id", ""),
            num_signals=payload.get("num_signals", 0),
            profit_factor=payload.get("profit_factor", 0.0),
            sharpe_ratio=payload.get("sharpe_ratio", 0.0),
            win_rate=payload.get("win_rate", 0.0),
            max_drawdown=payload.get("max_drawdown", 0.0),
            trade_dates=payload.get("trade_dates", []),
            market_phases=payload.get("market_phases", []),
        )

        passed = self._filter.is_valid(result)
        rejection_reasons = self._filter.rejection_reasons(result)
        score = self._calculate_score(result)

        self._message_bus.publish(
            Message(
                topic=MONRY_EVALUATION_COMPLETED,
                payload={
                    "strategy_id": result.strategy_id,
                    "score": score,
                    "passed": passed,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "num_signals": result.num_signals,
                    "rejection_reasons": rejection_reasons,
                },
                sender_id=self._agent_id,
            )
        )

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_score(result: BacktestResult) -> float:
        """
        Computes a score in [0, 100].

        - 30 pts from win_rate       (linear: 0.5 → 0, ≥0.8 → 30)
        - 25 pts from profit_factor  (linear: 1.0 → 0, ≥3.0 → 25)
        - 25 pts from sharpe_ratio   (linear: 0.0 → 0, ≥2.0 → 25)
        - 20 pts from max_drawdown   (inverse: 0.0 → 20, ≥0.2 → 0)
        """
        # win_rate component: 0.5 baseline, 0.8 ceiling
        wr_min, wr_max = 0.5, 0.8
        wr_pts = 30.0 * max(0.0, min(1.0, (result.win_rate - wr_min) / (wr_max - wr_min)))

        # profit_factor component: 1.0 baseline, 3.0 ceiling
        pf_min, pf_max = 1.0, 3.0
        pf_pts = 25.0 * max(0.0, min(1.0, (result.profit_factor - pf_min) / (pf_max - pf_min)))

        # sharpe component: 0.0 baseline, 2.0 ceiling
        sh_min, sh_max = 0.0, 2.0
        sh_pts = 25.0 * max(0.0, min(1.0, (result.sharpe_ratio - sh_min) / (sh_max - sh_min)))

        # drawdown component (inverse): 0.0 → 20 pts, 0.2+ → 0 pts
        dd_max_threshold = 0.2
        dd_pts = 20.0 * max(0.0, min(1.0, 1.0 - result.max_drawdown / dd_max_threshold))

        return round(wr_pts + pf_pts + sh_pts + dd_pts, 4)
