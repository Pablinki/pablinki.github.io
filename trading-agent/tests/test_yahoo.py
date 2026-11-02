"""Tests del proveedor Yahoo (lógica pura, sin red)."""

from __future__ import annotations

import pytest

from trading_agent.data.yahoo import latest_period_for_interval


class TestLatestPeriod:
    """El periodo pedido debe cubrir el calentamiento de indicadores (~36
    velas) más la ventana del estado, para cada granularidad."""

    def test_daily_needs_a_year(self):
        assert latest_period_for_interval("1d") == "1y"

    def test_hourly_needs_months(self):
        assert latest_period_for_interval("1h") == "3mo"

    @pytest.mark.parametrize("interval", ["1m", "5m", "15m", "30m"])
    def test_minutes_use_five_days(self, interval):
        assert latest_period_for_interval(interval) == "5d"
