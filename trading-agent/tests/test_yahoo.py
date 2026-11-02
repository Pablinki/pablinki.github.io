"""Tests del proveedor Yahoo (lógica pura, sin red)."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.data.provider import completed_bars
from trading_agent.data.yahoo import latest_period_for_interval


class TestCompletedBars:
    def _df(self, *dates: str) -> pd.DataFrame:
        idx = pd.to_datetime(list(dates), utc=True)
        return pd.DataFrame({"close": range(1, len(dates) + 1)}, index=idx)

    def test_drops_in_progress_daily_bar(self):
        df = self._df("2024-01-02", "2024-01-03")
        now = pd.Timestamp("2024-01-03 15:00", tz="UTC")
        assert len(completed_bars(df, "1d", now=now)) == 1

    def test_keeps_closed_daily_bar(self):
        df = self._df("2024-01-02", "2024-01-03")
        now = pd.Timestamp("2024-01-04 00:00", tz="UTC")
        assert len(completed_bars(df, "1d", now=now)) == 2

    def test_intraday_boundary(self):
        df = self._df("2024-01-02 15:30", "2024-01-02 15:35")
        assert len(completed_bars(df, "5m", now=pd.Timestamp(
            "2024-01-02 15:39", tz="UTC"))) == 1
        assert len(completed_bars(df, "5m", now=pd.Timestamp(
            "2024-01-02 15:40", tz="UTC"))) == 2

    def test_unknown_interval_or_empty_is_untouched(self):
        df = self._df("2024-01-02")
        assert len(completed_bars(df, "3w")) == 1
        assert completed_bars(df.iloc[:0], "1d").empty


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
