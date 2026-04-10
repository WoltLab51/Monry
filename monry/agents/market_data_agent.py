"""
MarketDataAgent — Fetches historical market data and publishes it on the bus.

Subscribes to: monry.goal.defined
Publishes to:  monry.market.data_ready
"""

from __future__ import annotations

from typing import List

try:
    from genus.core.agent import Agent, AgentState
    from genus.communication.message_bus import MessageBus, Message
except ImportError:
    from monry._genus_stubs import Agent, AgentState, MessageBus, Message  # type: ignore[assignment]

from monry.data.fetcher import MarketDataFetcher
from monry.models.candle import Candle
from monry.topics import MONRY_GOAL_DEFINED, MONRY_MARKET_DATA_READY


class MarketDataAgent(Agent):
    """
    Fetches historical market data and publishes it on the bus.

    On receiving a monry.goal.defined message, fetches all candles for the
    symbol/period specified in the Goal payload and publishes them as a single
    batch message on monry.market.data_ready.
    """

    def __init__(self, agent_id: str, message_bus: MessageBus) -> None:
        super().__init__(agent_id=agent_id, message_bus=message_bus)
        self._fetcher = MarketDataFetcher()

    def initialize(self) -> None:
        """Subscribe to monry.goal.defined."""
        self._message_bus.subscribe(
            MONRY_GOAL_DEFINED, self._agent_id, self.process_message
        )
        self._state = AgentState.INITIALIZED

    def process_message(self, message: Message) -> None:
        """Fetch candles for the goal's symbol/period and publish the batch."""
        payload = message.payload
        symbol: str = payload.get("symbol", "")
        start: str = payload.get("period_start", "")
        end: str = payload.get("period_end", "")

        if not symbol or not start or not end:
            return

        candles: List[Candle] = self._fetcher.fetch(symbol, start, end)

        candle_dicts = [
            {
                "date": c.date,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]

        self._message_bus.publish(
            Message(
                topic=MONRY_MARKET_DATA_READY,
                payload={
                    "symbol": symbol,
                    "start": start,
                    "end": end,
                    "num_candles": len(candles),
                    "candles": candle_dicts,
                },
                sender_id=self._agent_id,
            )
        )
