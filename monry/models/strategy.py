"""Strategy-Datenmodell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Strategy:
    """
    Repräsentiert eine Trading-Strategie im Monry-Explorationsprozess.

    Jede Strategie ist vollständig nachvollziehbar: Herkunft (parent_strategy),
    Mutationstyp, Hypothese und in welchem Marktkontext sie funktioniert.
    """

    strategy_id: str
    name: str
    rules: List[Dict[str, Any]]
    hypothesis: str
    parent_strategy: Optional[str]
    mutation_type: Optional[str]
    mutation_description: Optional[str]
    works_when: Dict[str, str]
    generation: int = 0
    score: float = 0.0
