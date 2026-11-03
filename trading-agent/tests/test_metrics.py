"""Tests de las métricas de evaluación (funciones puras)."""

from __future__ import annotations

import math

import pytest

from trading_agent.metrics import max_drawdown, profit_factor, sharpe_ratio
from trading_agent.portfolio import Trade


class TestMaxDrawdown:
    def test_classic_peak_to_trough(self):
        # Pico 120 -> valle 90 = -25%.
        assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)

    def test_monotonic_growth_has_zero_drawdown(self):
        assert max_drawdown([100.0, 101.0, 105.0]) == 0.0

    def test_short_or_empty_curves(self):
        assert max_drawdown([]) == 0.0
        assert max_drawdown([100.0]) == 0.0

    def test_worst_drawdown_is_tracked_across_recoveries(self):
        # Dos caídas: -10% y luego -40% desde un pico nuevo.
        curve = [100.0, 90.0, 150.0, 90.0, 200.0]
        assert max_drawdown(curve) == pytest.approx(0.40)


class TestSharpe:
    def test_zero_volatility_gives_zero(self):
        assert sharpe_ratio([100.0, 100.0, 100.0, 100.0]) == 0.0

    def test_too_short_gives_zero(self):
        assert sharpe_ratio([100.0, 110.0]) == 0.0

    def test_positive_for_steady_growth(self):
        curve = [100.0 * (1.01 ** i) for i in range(50)]
        # Crecimiento constante -> retornos idénticos -> std ~0 numérico;
        # usar retornos con algo de ruido determinista.
        noisy = [c * (1 + 0.001 * ((i % 3) - 1)) for i, c in enumerate(curve)]
        assert sharpe_ratio(noisy) > 0

    def test_negative_for_steady_decline(self):
        curve = [100.0 * (0.99 ** i) * (1 + 0.001 * ((i % 3) - 1))
                 for i in range(50)]
        assert sharpe_ratio(curve) < 0

    def test_annualization_scales_with_sqrt_periods(self):
        curve = [100.0, 101.0, 100.5, 102.0, 101.0, 103.0]
        daily = sharpe_ratio(curve, periods_per_year=252)
        minute = sharpe_ratio(curve, periods_per_year=252 * 4)
        assert minute == pytest.approx(daily * 2.0)


class TestProfitFactor:
    def test_no_sells_gives_zero(self):
        assert profit_factor([Trade("A", "buy", 1, 10.0, 0.0)]) == 0.0

    def test_only_wins_gives_inf(self):
        assert math.isinf(profit_factor([Trade("A", "sell", 1, 10.0, 0.0, 5.0)]))

    def test_ratio_of_gross_win_to_gross_loss(self):
        trades = [Trade("A", "sell", 1, 10.0, 0.0, 30.0),
                  Trade("A", "sell", 1, 10.0, 0.0, -10.0),
                  Trade("A", "sell", 1, 10.0, 0.0, -5.0)]
        assert profit_factor(trades) == pytest.approx(2.0)
