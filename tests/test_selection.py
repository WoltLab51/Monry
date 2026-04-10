"""Tests für SelectionFilter."""

from __future__ import annotations

import pytest

from monry.models.backtest_result import BacktestResult
from monry.search.selection import SelectionFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    num_signals: int = 50,
    profit_factor: float = 1.5,
    sharpe_ratio: float = 1.0,
    win_rate: float = 0.55,
    max_drawdown: float = 0.10,
    trade_dates: list | None = None,
    market_phases: list | None = None,
) -> BacktestResult:
    """Hilfsfunktion: Erstellt ein gültiges BacktestResult."""
    if trade_dates is None:
        # 50 Trades über 6 Monate verteilt
        trade_dates = (
            [f"2023-01-{d:02d}" for d in range(1, 9)]  # 8 im Jan
            + [f"2023-02-{d:02d}" for d in range(1, 9)]  # 8 im Feb
            + [f"2023-03-{d:02d}" for d in range(1, 9)]  # 8 im Mär
            + [f"2023-04-{d:02d}" for d in range(1, 9)]  # 8 im Apr
            + [f"2023-05-{d:02d}" for d in range(1, 9)]  # 8 im Mai
            + [f"2023-06-{d:02d}" for d in range(1, 11)]  # 10 im Jun
        )
    if market_phases is None:
        market_phases = ["uptrend", "downtrend", "sideways"]
    return BacktestResult(
        strategy_id="test_strategy",
        num_signals=num_signals,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        trade_dates=trade_dates,
        market_phases=market_phases,
    )


@pytest.fixture
def selection_filter() -> SelectionFilter:
    return SelectionFilter(
        min_signals=30,
        min_profit_factor=1.0,
        min_sharpe=0.0,
        max_single_month_ratio=0.8,
        min_market_phases=2,
    )


# ---------------------------------------------------------------------------
# Gültige Strategien
# ---------------------------------------------------------------------------


class TestValidStrategies:
    def test_valid_strategy_passes(self, selection_filter: SelectionFilter) -> None:
        """Eine Strategie die alle Kriterien erfüllt wird akzeptiert."""
        result = _make_result()
        assert selection_filter.is_valid(result) is True

    def test_no_rejection_reasons_for_valid_strategy(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Eine gültige Strategie hat keine Ablehnungsgründe."""
        result = _make_result()
        assert selection_filter.rejection_reasons(result) == []


# ---------------------------------------------------------------------------
# Zu wenige Trades
# ---------------------------------------------------------------------------


class TestTooFewSignals:
    def test_too_few_signals_rejected(self, selection_filter: SelectionFilter) -> None:
        """Strategie mit zu wenigen Trades (< min_signals) wird abgelehnt."""
        result = _make_result(num_signals=10)
        assert selection_filter.is_valid(result) is False

    def test_too_few_signals_in_rejection_reasons(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Ablehnungsgrund enthält Hinweis auf zu wenige Trades."""
        result = _make_result(num_signals=10)
        reasons = selection_filter.rejection_reasons(result)
        assert any("Trades" in r or "Signale" in r or "10" in r for r in reasons)

    def test_exactly_min_signals_passes(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Genau min_signals Trades werden akzeptiert (Grenzwert-Test)."""
        trade_dates = (
            [f"2023-01-{d:02d}" for d in range(1, 9)]
            + [f"2023-02-{d:02d}" for d in range(1, 9)]
            + [f"2023-03-{d:02d}" for d in range(1, 8)]
            + [f"2023-04-{d:02d}" for d in range(1, 8)]
        )
        result = _make_result(num_signals=30, trade_dates=trade_dates)
        assert selection_filter.is_valid(result) is True


# ---------------------------------------------------------------------------
# Zeitliche Konzentration (alle Trades in einem Monat)
# ---------------------------------------------------------------------------


class TestTimeConcentration:
    def test_trades_in_single_month_rejected(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Strategie mit allen Trades in einem Monat wird abgelehnt."""
        concentrated_dates = [f"2023-01-{d:02d}" for d in range(1, 31)]
        result = _make_result(num_signals=30, trade_dates=concentrated_dates)
        assert selection_filter.is_valid(result) is False

    def test_concentration_in_rejection_reasons(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Ablehnungsgrund enthält Hinweis auf Konzentration."""
        concentrated_dates = [f"2023-01-{d:02d}" for d in range(1, 31)]
        result = _make_result(num_signals=30, trade_dates=concentrated_dates)
        reasons = selection_filter.rejection_reasons(result)
        assert any("konzentrier" in r.lower() or "monat" in r.lower() or "80" in r for r in reasons)

    def test_trades_in_two_months_passes_concentration_check(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Trades auf zwei Monate aufgeteilt können den Konzentrations-Check bestehen."""
        distributed_dates = (
            [f"2023-01-{d:02d}" for d in range(1, 16)]  # 15 in Jan (50%)
            + [f"2023-02-{d:02d}" for d in range(1, 16)]  # 15 in Feb (50%)
        )
        result = _make_result(num_signals=30, trade_dates=distributed_dates)
        # 50% in einem Monat <= 80% → should pass time check
        assert selection_filter._is_time_distributed(result) is True


# ---------------------------------------------------------------------------
# Profit-Factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_profit_factor_below_one_rejected(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Profit-Factor < 1 wird abgelehnt."""
        result = _make_result(profit_factor=0.8)
        assert selection_filter.is_valid(result) is False

    def test_profit_factor_exactly_one_rejected(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Profit-Factor = 1.0 wird abgelehnt (muss > min_profit_factor sein)."""
        result = _make_result(profit_factor=1.0)
        assert selection_filter.is_valid(result) is False

    def test_profit_factor_above_one_passes(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Profit-Factor > 1 besteht den Check."""
        result = _make_result(profit_factor=1.01)
        assert selection_filter._has_positive_profit_factor(result) is True

    def test_profit_factor_in_rejection_reasons(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Ablehnungsgrund enthält Hinweis auf Profit-Factor."""
        result = _make_result(profit_factor=0.5)
        reasons = selection_filter.rejection_reasons(result)
        assert any("Profit-Factor" in r or "profit" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Marktphasen-Abdeckung
# ---------------------------------------------------------------------------


class TestMarketPhases:
    def test_only_one_market_phase_rejected(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Strategie mit nur einer Marktphase wird abgelehnt."""
        result = _make_result(market_phases=["uptrend"] * 50)
        assert selection_filter.is_valid(result) is False

    def test_two_market_phases_passes(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Strategie mit zwei verschiedenen Marktphasen besteht den Check."""
        result = _make_result(market_phases=["uptrend", "downtrend"] * 25)
        assert selection_filter._covers_multiple_phases(result) is True


# ---------------------------------------------------------------------------
# Mehrere Ablehnungsgründe
# ---------------------------------------------------------------------------


class TestMultipleRejectionReasons:
    def test_multiple_failures_reported(
        self, selection_filter: SelectionFilter
    ) -> None:
        """Alle Ablehnungsgründe werden gemeldet, nicht nur der erste."""
        result = _make_result(
            num_signals=5,
            profit_factor=0.5,
            market_phases=["uptrend"],
        )
        reasons = selection_filter.rejection_reasons(result)
        assert len(reasons) >= 2
