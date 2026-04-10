"""
Minimale Stubs für GENUS-Interfaces.

Diese Stubs ermöglichen das Ausführen von Tests ohne installiertes GENUS-Package.
Sie bilden nur das notwendige Interface nach — kein echtes Verhalten.

TEMPORÄR: Sobald GENUS als PyPI-Package verfügbar ist, werden diese Stubs
durch echte Imports ersetzt (via `pip install genus`).
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


class AgentState(enum.Enum):
    """Lifecycle-Zustände eines Agents."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Message:
    """Nachricht im GENUS MessageBus."""

    topic: str
    payload: Dict[str, Any]
    sender_id: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MessageBus:
    """
    Minimaler In-Memory MessageBus Stub.

    Unterstützt subscribe/publish für synchrone Tests.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, Dict[str, Callable[[Message], None]]] = {}
        self._published: List[Message] = []

    def subscribe(self, topic: str, subscriber_id: str, handler: Callable[[Message], None]) -> None:
        """Registriert einen Handler für ein Topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = {}
        self._subscribers[topic][subscriber_id] = handler

    def unsubscribe(self, topic: str, subscriber_id: str) -> None:
        """Entfernt einen Handler für ein Topic."""
        if topic in self._subscribers:
            self._subscribers[topic].pop(subscriber_id, None)

    def publish(self, message: Message) -> None:
        """Publiziert eine Nachricht an alle Subscriber des Topics."""
        self._published.append(message)
        for handler in self._subscribers.get(message.topic, {}).values():
            handler(message)

    def get_published(self, topic: Optional[str] = None) -> List[Message]:
        """Gibt alle publizierten Nachrichten zurück, optional nach Topic gefiltert."""
        if topic is None:
            return list(self._published)
        return [m for m in self._published if m.topic == topic]

    def clear(self) -> None:
        """Setzt den Nachrichten-Puffer zurück (für Tests)."""
        self._published.clear()


class Agent:
    """
    Minimaler Agent Stub.

    Bildet das GENUS Agent-Interface nach: initialize, start, stop, process_message.
    """

    def __init__(self, agent_id: str, message_bus: MessageBus) -> None:
        self._agent_id = agent_id
        self._message_bus = message_bus
        self._state = AgentState.INITIALIZED
        self._created_at = datetime.now(timezone.utc)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def state(self) -> AgentState:
        return self._state

    def initialize(self) -> None:
        """Setup: Subscriptions und Ressourcen bereitstellen."""
        self._state = AgentState.INITIALIZED

    def start(self) -> None:
        """Agent startet die aktive Verarbeitung."""
        self._state = AgentState.RUNNING

    def stop(self) -> None:
        """Agent beendet die Verarbeitung geordnet."""
        self._state = AgentState.STOPPED

    def process_message(self, message: Message) -> None:
        """Verarbeitet eine eingehende Nachricht."""
        raise NotImplementedError(
            f"{self.__class__.__name__} muss process_message implementieren."
        )
