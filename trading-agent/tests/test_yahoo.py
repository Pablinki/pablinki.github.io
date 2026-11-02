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

    @pytest.mark.parametrize("interval", ["15m", "30m"])
    def test_quarter_and_half_hour_need_a_month(self, interval):
        # 5 días de velas de 30m son ~65: no cubren 36 de calentamiento + 30.
        assert latest_period_for_interval(interval) == "1mo"

    @pytest.mark.parametrize("interval", ["1m", "5m"])
    def test_fine_minutes_use_five_days(self, interval):
        assert latest_period_for_interval(interval) == "5d"


class TestRateLimit:
    def test_rate_limit_is_typed_and_not_retried(self, monkeypatch, tmp_path):
        """YFRateLimitError -> RateLimitError sin reintentos con backoff."""
        import sys
        import types

        from trading_agent.config import DataConfig
        from trading_agent.data.yahoo import YahooFinanceProvider
        from trading_agent.exceptions import RateLimitError

        class YFRateLimitError(Exception):
            pass

        calls = {"n": 0}

        def fake_download(**kwargs):
            calls["n"] += 1
            raise YFRateLimitError("Too Many Requests")

        fake_yf = types.SimpleNamespace(download=fake_download)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        provider = YahooFinanceProvider(DataConfig(cache_dir=None, max_retries=3,
                                                   retry_backoff_s=0.0))
        with pytest.raises(RateLimitError):
            provider.fetch_latest("AAPL")
        assert calls["n"] == 1  # ni un reintento
