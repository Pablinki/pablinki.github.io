"""Tests del diario CSV de operaciones."""

from __future__ import annotations

import csv

from trading_agent.journal import COLUMNS, TradeJournal
from trading_agent.portfolio import Trade


class TestTradeJournal:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "sub" / "trades.csv"
        TradeJournal(path)
        assert path.exists()
        assert path.read_text().splitlines() == [",".join(COLUMNS)]

    def test_appends_rows_without_rewriting_header(self, tmp_path):
        path = tmp_path / "trades.csv"
        j = TradeJournal(path)
        j.record(Trade("AAPL", "buy", 10, 100.0, 0.1), reason="agent",
                 equity=99_000.0, timestamp="t1")
        j.record(Trade("AAPL", "sell", 10, 110.0, 0.1, 99.8), reason="stop_loss",
                 equity=99_099.8, timestamp="t2")
        # Reabrir el diario (reinicio del proceso) no borra lo anterior.
        TradeJournal(path).record(Trade("AAPL", "buy", 1, 50.0, 0.0),
                                  reason="agent", equity=1.0, timestamp="t3")
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["timestamp_utc"] for r in rows] == ["t1", "t2", "t3"]
        assert rows[1]["reason"] == "stop_loss"
        assert float(rows[1]["pnl"]) == 99.8

    def test_reconciliation_columns(self, tmp_path):
        path = tmp_path / "trades.csv"
        TradeJournal(path).record(
            Trade("AAPL", "buy", 10, 100.0, 0.0, order_id="o-1",
                  client_order_id="ta-AAPL-buy-x"),
            reason="agent", equity=1.0, timestamp="t1",
            bar_ts="2026-01-02T15:00:00+00:00", broker="AlpacaBroker")
        with path.open(newline="") as fh:
            row = next(iter(csv.DictReader(fh)))
        assert row["bar_ts"] == "2026-01-02T15:00:00+00:00"
        assert row["broker"] == "AlpacaBroker"
        assert row["order_id"] == "o-1"
        assert row["client_order_id"] == "ta-AAPL-buy-x"

    def test_default_timestamp_is_iso_utc(self, tmp_path):
        path = tmp_path / "trades.csv"
        TradeJournal(path).record(Trade("AAPL", "buy", 1, 1.0, 0.0),
                                  reason="agent", equity=1.0)
        ts = path.read_text().splitlines()[1].split(",")[0]
        assert ts.endswith("+00:00")
