"""Tests für die IndicatorToolbox."""

from __future__ import annotations

import pytest

from monry.analysis.indicator_toolbox import IndicatorToolbox, _INDICATORS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toolbox() -> IndicatorToolbox:
    return IndicatorToolbox(seed=42)


# ---------------------------------------------------------------------------
# Indikator-Definitionen
# ---------------------------------------------------------------------------


class TestIndicatorDefinitions:
    def test_all_indicators_have_required_fields(self) -> None:
        """Alle Indikatoren haben die Pflichtfelder name, category, parameters, conditions."""
        for indicator in _INDICATORS:
            assert "name" in indicator, f"Fehlendes 'name' in {indicator}"
            assert "category" in indicator, f"Fehlendes 'category' in {indicator}"
            assert "parameters" in indicator, f"Fehlendes 'parameters' in {indicator}"
            assert "conditions" in indicator, f"Fehlendes 'conditions' in {indicator}"
            assert "description" in indicator, f"Fehlendes 'description' in {indicator}"

    def test_all_indicators_have_valid_categories(self) -> None:
        """Alle Indikatoren haben eine der erlaubten Kategorien."""
        valid_categories = {"trend", "momentum", "volatility", "volume", "mean_reversion"}
        for indicator in _INDICATORS:
            assert indicator["category"] in valid_categories, (
                f"Ungültige Kategorie '{indicator['category']}' in {indicator['name']}"
            )

    def test_all_parameter_ranges_are_valid(self) -> None:
        """Alle Parameter haben gültige min < max Ranges."""
        for indicator in _INDICATORS:
            for param_name, param_def in indicator["parameters"].items():
                assert param_def["min"] < param_def["max"], (
                    f"Ungültige Range in {indicator['name']}.{param_name}: "
                    f"min={param_def['min']} >= max={param_def['max']}"
                )
                assert param_def["step"] > 0, (
                    f"step muss > 0 sein in {indicator['name']}.{param_name}"
                )

    def test_all_indicators_have_at_least_one_condition(self) -> None:
        """Alle Indikatoren haben mindestens eine Bedingung."""
        for indicator in _INDICATORS:
            assert len(indicator["conditions"]) >= 1, (
                f"{indicator['name']} hat keine Bedingungen"
            )

    def test_expected_indicators_are_present(self) -> None:
        """Alle geforderten Indikatoren sind vorhanden."""
        expected = {"rsi", "sma", "ema", "macd", "bollinger_bands", "atr", "adx", "roc", "volume"}
        actual = {ind["name"] for ind in _INDICATORS}
        assert expected == actual

    def test_all_five_categories_are_covered(self, toolbox: IndicatorToolbox) -> None:
        """Alle fünf Kategorien (trend, momentum, volatility, volume, mean_reversion) sind abgedeckt."""
        categories = toolbox.get_available_categories()
        assert set(categories) == {"trend", "momentum", "volatility", "volume", "mean_reversion"}


# ---------------------------------------------------------------------------
# get_random_indicator
# ---------------------------------------------------------------------------


class TestGetRandomIndicator:
    def test_returns_valid_indicator(self, toolbox: IndicatorToolbox) -> None:
        """get_random_indicator gibt eine gültige Indikator-Instanz zurück."""
        result = toolbox.get_random_indicator()
        assert "name" in result
        assert "category" in result
        assert "parameters" in result
        assert "condition" in result

    def test_parameters_are_within_range(self, toolbox: IndicatorToolbox) -> None:
        """Zufällige Parameter liegen innerhalb der definierten Ranges."""
        for _ in range(50):
            result = toolbox.get_random_indicator()
            name = result["name"]
            indicator_def = next(i for i in _INDICATORS if i["name"] == name)
            for param_name, value in result["parameters"].items():
                param_def = indicator_def["parameters"][param_name]
                assert value >= param_def["min"], (
                    f"{name}.{param_name}={value} < min={param_def['min']}"
                )
                assert value <= param_def["max"], (
                    f"{name}.{param_name}={value} > max={param_def['max']}"
                )

    def test_condition_is_valid(self, toolbox: IndicatorToolbox) -> None:
        """Die zufällige Bedingung ist eine der erlaubten Bedingungen für den Indikator."""
        for _ in range(30):
            result = toolbox.get_random_indicator()
            name = result["name"]
            indicator_def = next(i for i in _INDICATORS if i["name"] == name)
            assert result["condition"] in indicator_def["conditions"]


# ---------------------------------------------------------------------------
# get_random_indicator_from_category
# ---------------------------------------------------------------------------


class TestGetRandomIndicatorFromCategory:
    def test_returns_indicator_from_correct_category(
        self, toolbox: IndicatorToolbox
    ) -> None:
        """get_random_indicator_from_category gibt Indikator der richtigen Kategorie zurück."""
        for category in ["trend", "momentum", "volatility", "volume", "mean_reversion"]:
            result = toolbox.get_random_indicator_from_category(category)
            assert result["category"] == category

    def test_raises_for_unknown_category(self, toolbox: IndicatorToolbox) -> None:
        """Unbekannte Kategorie wirft ValueError."""
        with pytest.raises(ValueError, match="Keine Indikatoren"):
            toolbox.get_random_indicator_from_category("nonexistent_category")


# ---------------------------------------------------------------------------
# mutate_parameter
# ---------------------------------------------------------------------------


class TestMutateParameter:
    def test_mutated_value_stays_within_range(self, toolbox: IndicatorToolbox) -> None:
        """Mutierter Parameter bleibt innerhalb der definierten Range."""
        for _ in range(100):
            rule = toolbox.get_random_indicator()
            name = rule["name"]
            indicator_def = next(i for i in _INDICATORS if i["name"] == name)
            param_name = list(rule["parameters"].keys())[0]
            param_def = indicator_def["parameters"][param_name]

            mutated = toolbox.mutate_parameter(rule, param_name)
            value = mutated["parameters"][param_name]

            assert value >= param_def["min"], (
                f"{name}.{param_name}={value} < min={param_def['min']}"
            )
            assert value <= param_def["max"], (
                f"{name}.{param_name}={value} > max={param_def['max']}"
            )

    def test_mutation_only_changes_specified_parameter(
        self, toolbox: IndicatorToolbox
    ) -> None:
        """Mutation ändert nur den angegebenen Parameter, nicht andere."""
        rule = toolbox.get_random_indicator_from_category("trend")
        if len(rule["parameters"]) < 2:
            pytest.skip("Indikator hat nur einen Parameter — skip multi-param test")

        params = list(rule["parameters"].keys())
        mutated = toolbox.mutate_parameter(rule, params[0])

        # Alle anderen Parameter unverändert
        for other_param in params[1:]:
            assert mutated["parameters"][other_param] == rule["parameters"][other_param]

    def test_raises_for_unknown_indicator(self, toolbox: IndicatorToolbox) -> None:
        """ValueError bei unbekanntem Indikator."""
        bad_rule = {"name": "nonexistent", "parameters": {"x": 1}, "condition": ""}
        with pytest.raises(ValueError, match="Unbekannter Indikator"):
            toolbox.mutate_parameter(bad_rule, "x")

    def test_raises_for_unknown_parameter(self, toolbox: IndicatorToolbox) -> None:
        """ValueError bei unbekanntem Parameter-Namen."""
        rule = toolbox.get_random_indicator()
        with pytest.raises(ValueError, match="Parameter"):
            toolbox.mutate_parameter(rule, "nonexistent_param")


# ---------------------------------------------------------------------------
# get_compatible_filters
# ---------------------------------------------------------------------------


class TestGetCompatibleFilters:
    def test_no_duplicates_in_result(self, toolbox: IndicatorToolbox) -> None:
        """get_compatible_filters gibt keine doppelten Indikatoren zurück."""
        rsi_rule = toolbox.get_random_indicator_from_category("mean_reversion")
        compatible = toolbox.get_compatible_filters([rsi_rule])
        names = [ind["name"] for ind in compatible]
        assert len(names) == len(set(names)), "Doppelte Indikatoren in get_compatible_filters"

    def test_existing_indicator_not_in_compatible(
        self, toolbox: IndicatorToolbox
    ) -> None:
        """Bereits verwendeter Indikator erscheint nicht in den kompatiblen Filtern."""
        rule = toolbox.get_random_indicator_from_category("mean_reversion")
        compatible = toolbox.get_compatible_filters([rule])
        compatible_names = {ind["name"] for ind in compatible}
        assert rule["name"] not in compatible_names

    def test_compatible_filters_prefer_different_categories(
        self, toolbox: IndicatorToolbox
    ) -> None:
        """Kompatible Filter bevorzugen Kategorien die noch nicht vertreten sind."""
        rsi_rule = toolbox.get_random_indicator_from_category("mean_reversion")
        compatible = toolbox.get_compatible_filters([rsi_rule])
        # Es sollten Indikatoren aus anderen Kategorien vorhanden sein
        other_categories = {ind["category"] for ind in compatible}
        assert len(other_categories) >= 1

    def test_empty_rules_returns_all_indicators(
        self, toolbox: IndicatorToolbox
    ) -> None:
        """Bei leeren existing_rules werden alle Indikatoren zurückgegeben."""
        compatible = toolbox.get_compatible_filters([])
        assert len(compatible) == len(_INDICATORS)

    def test_get_indicators_by_category(self, toolbox: IndicatorToolbox) -> None:
        """get_indicators_by_category gibt nur Indikatoren der richtigen Kategorie zurück."""
        trend_indicators = toolbox.get_indicators_by_category("trend")
        assert all(ind["category"] == "trend" for ind in trend_indicators)
        assert len(trend_indicators) >= 1
