"""
Indicator Toolbox — Definierter Suchraum für den ExplorationAgent.

Die Toolbox definiert, welche Indikatoren der ExplorationAgent verwenden darf
und welche Parameter-Ranges erlaubt sind. Dies ist der begrenzte aber breite
Suchraum für die Strategie-Exploration.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Interne Indikator-Definitionen
# ---------------------------------------------------------------------------

_INDICATORS: List[Dict[str, Any]] = [
    {
        "name": "rsi",
        "category": "mean_reversion",
        "parameters": {
            "period": {"min": 7, "max": 21, "step": 1, "type": "int"},
            "threshold_oversold": {"min": 20, "max": 40, "step": 1, "type": "int"},
            "threshold_overbought": {"min": 60, "max": 80, "step": 1, "type": "int"},
        },
        "conditions": ["< threshold_oversold", "> threshold_overbought", "crossover"],
        "description": "Relative Strength Index — Erkennt überkaufte/überverkaufte Bereiche",
    },
    {
        "name": "sma",
        "category": "trend",
        "parameters": {
            "period": {"min": 10, "max": 200, "step": 5, "type": "int"},
        },
        "conditions": ["price_above", "price_below", "crossover"],
        "description": "Simple Moving Average — Trendrichtung",
    },
    {
        "name": "ema",
        "category": "trend",
        "parameters": {
            "period": {"min": 10, "max": 200, "step": 5, "type": "int"},
        },
        "conditions": ["price_above", "price_below", "crossover"],
        "description": "Exponential Moving Average — Gewichteter Trend",
    },
    {
        "name": "macd",
        "category": "momentum",
        "parameters": {
            "fast": {"min": 8, "max": 16, "step": 1, "type": "int"},
            "slow": {"min": 20, "max": 30, "step": 1, "type": "int"},
            "signal": {"min": 7, "max": 12, "step": 1, "type": "int"},
        },
        "conditions": ["histogram_positive", "histogram_negative", "signal_crossover"],
        "description": "MACD — Momentum und Trendwechsel",
    },
    {
        "name": "bollinger_bands",
        "category": "volatility",
        "parameters": {
            "period": {"min": 15, "max": 25, "step": 1, "type": "int"},
            "std_dev": {"min": 1.5, "max": 2.5, "step": 0.1, "type": "float"},
        },
        "conditions": ["price_below_lower", "price_above_upper", "bandwidth_squeeze"],
        "description": "Bollinger Bands — Volatilitätsbänder und Mean-Reversion",
    },
    {
        "name": "atr",
        "category": "volatility",
        "parameters": {
            "period": {"min": 10, "max": 20, "step": 1, "type": "int"},
        },
        "conditions": ["above_average", "below_average"],
        "description": "Average True Range — Marktvolatilität",
    },
    {
        "name": "adx",
        "category": "trend",
        "parameters": {
            "period": {"min": 10, "max": 20, "step": 1, "type": "int"},
            "threshold": {"min": 20, "max": 30, "step": 1, "type": "int"},
        },
        "conditions": ["> threshold", "< threshold"],
        "description": "Average Directional Index — Trendstärke",
    },
    {
        "name": "roc",
        "category": "momentum",
        "parameters": {
            "period": {"min": 5, "max": 20, "step": 1, "type": "int"},
        },
        "conditions": ["> 0", "< 0", "above_threshold"],
        "description": "Rate of Change — Preis-Momentum",
    },
    {
        "name": "volume",
        "category": "volume",
        "parameters": {
            "avg_period": {"min": 10, "max": 30, "step": 1, "type": "int"},
            "multiplier": {"min": 1.2, "max": 2.5, "step": 0.1, "type": "float"},
        },
        "conditions": ["above_average", "spike"],
        "description": "Volume — Handelsvolumen-Bestätigung",
    },
]

# Lookup-Index nach Name
_INDICATOR_BY_NAME: Dict[str, Dict[str, Any]] = {ind["name"]: ind for ind in _INDICATORS}


class IndicatorToolbox:
    """
    Definiert den erlaubten Suchraum für den ExplorationAgent.

    Stellt Methoden bereit um zufällige Indikatoren zu wählen, Parameter zu
    mutieren und kompatible Filter für bestehende Regeln zu finden.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def get_all_indicators(self) -> List[Dict[str, Any]]:
        """Gibt alle verfügbaren Indikator-Definitionen zurück."""
        return deepcopy(_INDICATORS)

    def get_indicator(self, name: str) -> Dict[str, Any]:
        """Gibt die Definition eines Indikators nach Name zurück."""
        if name not in _INDICATOR_BY_NAME:
            raise ValueError(f"Unbekannter Indikator: '{name}'")
        return deepcopy(_INDICATOR_BY_NAME[name])

    def get_random_indicator(self) -> Dict[str, Any]:
        """
        Wählt einen zufälligen Indikator mit zufälligen (aber gültigen) Parametern.

        Returns:
            Dict mit 'name', 'category', 'parameters' (konkrete Werte),
            'condition' und 'description'.
        """
        definition = deepcopy(self._rng.choice(_INDICATORS))
        return self._instantiate(definition)

    def get_random_indicator_from_category(self, category: str) -> Dict[str, Any]:
        """Wählt einen zufälligen Indikator aus einer bestimmten Kategorie."""
        candidates = [ind for ind in _INDICATORS if ind["category"] == category]
        if not candidates:
            raise ValueError(f"Keine Indikatoren in Kategorie: '{category}'")
        definition = deepcopy(self._rng.choice(candidates))
        return self._instantiate(definition)

    def get_indicators_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Gibt alle Indikatoren einer Kategorie zurück.

        Args:
            category: Eine der Kategorien: trend, momentum, volatility,
                      volume, mean_reversion.
        """
        return deepcopy(
            [ind for ind in _INDICATORS if ind["category"] == category]
        )

    def get_available_categories(self) -> List[str]:
        """Gibt alle verfügbaren Kategorien zurück."""
        return sorted({ind["category"] for ind in _INDICATORS})

    def mutate_parameter(
        self, rule: Dict[str, Any], parameter_name: str
    ) -> Dict[str, Any]:
        """
        Variiert einen Parameter leicht innerhalb der erlaubten Range.

        Args:
            rule: Eine Regel-Definition (wie von get_random_indicator() erzeugt).
                  Muss 'name' und 'parameters' enthalten.
            parameter_name: Name des Parameters der mutiert werden soll.

        Returns:
            Neue Regel mit mutiertem Parameter (Original bleibt unverändert).
        """
        indicator_name = rule["name"]
        if indicator_name not in _INDICATOR_BY_NAME:
            raise ValueError(f"Unbekannter Indikator in Regel: '{indicator_name}'")

        definition = _INDICATOR_BY_NAME[indicator_name]
        if parameter_name not in definition["parameters"]:
            raise ValueError(
                f"Parameter '{parameter_name}' nicht in Indikator '{indicator_name}'"
            )

        param_def = definition["parameters"][parameter_name]
        current_value = rule["parameters"][parameter_name]
        new_rule = deepcopy(rule)
        new_rule["parameters"][parameter_name] = self._mutate_value(
            current_value, param_def
        )
        return new_rule

    def get_compatible_filters(
        self, existing_rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Gibt Indikatoren zurück, die zu den bestehenden Regeln passen.

        Kompatibel bedeutet: aus einer Kategorie, die noch nicht (oder wenig)
        in den bestehenden Regeln vertreten ist.
        Verhindert redundante Filter (z.B. zwei Trend-Indikatoren die dasselbe messen).

        Args:
            existing_rules: Liste bestehender Regel-Definitionen.

        Returns:
            Liste von Indikator-Definitionen (keine konkreten Instanzen).
        """
        used_categories = self._get_used_categories(existing_rules)
        used_indicator_names = {rule["name"] for rule in existing_rules}

        compatible = []
        for indicator in _INDICATORS:
            # Indikator selbst nicht nochmal vorschlagen
            if indicator["name"] in used_indicator_names:
                continue
            # Kategorie die noch nicht vertreten ist bevorzugen
            if indicator["category"] not in used_categories:
                compatible.append(deepcopy(indicator))
            elif used_categories.count(indicator["category"]) < 2:
                # Maximal 2 Indikatoren pro Kategorie
                compatible.append(deepcopy(indicator))

        return compatible

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _instantiate(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        """Füllt eine Indikator-Definition mit zufälligen gültigen Parameterwerten."""
        concrete_params: Dict[str, Any] = {}
        for param_name, param_def in definition["parameters"].items():
            concrete_params[param_name] = self._random_value(param_def)

        condition = self._rng.choice(definition["conditions"])

        return {
            "name": definition["name"],
            "category": definition["category"],
            "parameters": concrete_params,
            "condition": condition,
            "description": definition["description"],
        }

    def _random_value(self, param_def: Dict[str, Any]) -> Any:
        """Erzeugt einen zufälligen Wert innerhalb der definierten Range."""
        p_min = param_def["min"]
        p_max = param_def["max"]
        step = param_def["step"]
        p_type = param_def.get("type", "int")

        if p_type == "int":
            steps = int((p_max - p_min) / step)
            chosen_step = self._rng.randint(0, steps)
            return int(p_min + chosen_step * step)
        else:
            steps = round((p_max - p_min) / step)
            chosen_step = self._rng.randint(0, steps)
            value = round(p_min + chosen_step * step, 10)
            return round(min(max(value, p_min), p_max), 4)

    def _mutate_value(self, current: Any, param_def: Dict[str, Any]) -> Any:
        """
        Variiert einen Wert leicht (±1 Step) innerhalb der Range.

        Garantiert eine tatsächliche Änderung: Falls eine Richtung durch eine
        Bereichsgrenze blockiert ist, wird die andere Richtung gewählt.
        """
        p_min = param_def["min"]
        p_max = param_def["max"]
        step = param_def["step"]
        p_type = param_def.get("type", "int")

        # Bestimme welche Richtungen möglich sind
        can_go_up = (current + step) <= p_max
        can_go_down = (current - step) >= p_min

        if can_go_up and can_go_down:
            direction = self._rng.choice([-1, 1])
        elif can_go_up:
            direction = 1
        elif can_go_down:
            direction = -1
        else:
            # Keine Änderung möglich (Range hat nur einen gültigen Wert)
            return current

        new_value = current + direction * step

        if p_type == "int":
            return int(min(max(new_value, p_min), p_max))
        else:
            return round(min(max(new_value, p_min), p_max), 4)

    def _get_used_categories(self, rules: List[Dict[str, Any]]) -> List[str]:
        """Gibt eine Liste der verwendeten Kategorien zurück (mit Duplikaten)."""
        categories = []
        for rule in rules:
            indicator_name = rule.get("name")
            if indicator_name and indicator_name in _INDICATOR_BY_NAME:
                categories.append(_INDICATOR_BY_NAME[indicator_name]["category"])
        return categories
