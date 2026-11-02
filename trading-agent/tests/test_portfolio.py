"""Tests de la contabilidad del portafolio (dinero: la parte más testeada)."""

from __future__ import annotations

import pytest

from trading_agent.exceptions import ExecutionError, InsufficientFundsError
from trading_agent.portfolio import Portfolio


def make_portfolio(cash=10_000.0, commission=0.0, slippage=0.0) -> Portfolio:
    return Portfolio(cash=cash, commission_pct=commission, slippage_pct=slippage)


class TestOpenPosition:
    def test_open_reduces_cash_and_creates_position(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        assert p.cash == pytest.approx(9_000.0)
        assert p.has_position("AAPL")
        assert p.positions["AAPL"].quantity == 10

    def test_commission_is_charged(self):
        p = make_portfolio(commission=0.01)  # 1%
        p.open_position("AAPL", price=100.0, quantity=10)
        assert p.cash == pytest.approx(10_000.0 - 1_000.0 - 10.0)

    def test_slippage_makes_buys_more_expensive(self):
        p = make_portfolio(slippage=0.01)  # 1%
        p.open_position("AAPL", price=100.0, quantity=10)
        assert p.positions["AAPL"].entry_price == pytest.approx(101.0)

    def test_insufficient_funds_raises(self):
        p = make_portfolio(cash=100.0)
        with pytest.raises(InsufficientFundsError):
            p.open_position("AAPL", price=100.0, quantity=10)

    def test_double_open_raises(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=1)
        with pytest.raises(ExecutionError, match="Ya existe"):
            p.open_position("AAPL", price=100.0, quantity=1)

    @pytest.mark.parametrize("price,qty", [(0.0, 1), (-5.0, 1), (100.0, 0),
                                           (100.0, -3)])
    def test_invalid_order_raises(self, price, qty):
        with pytest.raises(ExecutionError):
            make_portfolio().open_position("AAPL", price=price, quantity=qty)


class TestClosePosition:
    def test_round_trip_pnl_without_costs(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        trade = p.close_position("AAPL", price=110.0)
        assert trade.pnl == pytest.approx(100.0)
        assert p.cash == pytest.approx(10_100.0)
        assert not p.has_position("AAPL")

    def test_round_trip_with_commission_and_slippage(self):
        """Contabilidad exacta: costes en ambas patas de la operación."""
        p = make_portfolio(commission=0.001, slippage=0.001)
        p.open_position("AAPL", price=100.0, quantity=10)
        buy_price = 100.0 * 1.001                       # slippage de compra
        cash_after_buy = 10_000.0 - buy_price * 10 * 1.001  # + comisión
        assert p.cash == pytest.approx(cash_after_buy)

        p.close_position("AAPL", price=110.0)
        sell_price = 110.0 * 0.999                      # slippage de venta
        expected = cash_after_buy + sell_price * 10 * 0.999  # - comisión
        assert p.cash == pytest.approx(expected)

    def test_realized_pnl_includes_entry_commission(self):
        """El P&L realizado es el del viaje completo: ambas comisiones."""
        p = make_portfolio(commission=0.01)       # 1% por pata
        p.open_position("AAPL", price=100.0, quantity=10)   # fee entrada 10
        trade = p.close_position("AAPL", price=100.0)       # fee salida 10
        assert trade.pnl == pytest.approx(-20.0)
        # Y coincide exactamente con la variación de efectivo.
        assert p.cash - 10_000.0 == pytest.approx(trade.pnl)

    def test_partial_close_prorates_entry_commission(self):
        p = make_portfolio(commission=0.01)
        p.open_position("AAPL", price=100.0, quantity=10)
        t1 = p.reduce_position("AAPL", price=100.0, quantity=5)
        t2 = p.reduce_position("AAPL", price=100.0, quantity=5)
        assert t1.pnl + t2.pnl == pytest.approx(-20.0)

    def test_enforce_funds_false_mirrors_real_fill(self):
        """Un broker real refleja el fill aunque el efectivo local no cuadre."""
        p = make_portfolio(cash=100.0)
        trade = p.open_position("AAPL", price=100.0, quantity=10,
                                enforce_funds=False)
        assert trade.quantity == 10 and p.cash == pytest.approx(-900.0)

    def test_close_without_position_raises(self):
        with pytest.raises(ExecutionError, match="No hay posición"):
            make_portfolio().close_position("AAPL", price=100.0)

    def test_trades_are_recorded_for_audit(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=5)
        p.close_position("AAPL", price=90.0)
        assert [t.side for t in p.trades] == ["buy", "sell"]
        assert p.trades[1].pnl == pytest.approx(-50.0)


class TestReducePosition:
    def test_partial_sale_keeps_remainder(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        trade = p.reduce_position("AAPL", price=110.0, quantity=4)
        assert trade.quantity == 4
        assert trade.pnl == pytest.approx(40.0)
        assert p.positions["AAPL"].quantity == 6
        assert p.cash == pytest.approx(9_000.0 + 440.0)

    def test_full_quantity_closes(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        p.reduce_position("AAPL", price=100.0, quantity=10)
        assert not p.has_position("AAPL")

    @pytest.mark.parametrize("qty", [0, -1, 11])
    def test_invalid_quantity_raises(self, qty):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        with pytest.raises(ExecutionError, match="inválida"):
            p.reduce_position("AAPL", price=100.0, quantity=qty)


class TestEquity:
    def test_equity_marks_positions_to_market(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        assert p.equity({"AAPL": 120.0}) == pytest.approx(10_200.0)

    def test_equity_without_price_raises(self):
        p = make_portfolio()
        p.open_position("AAPL", price=100.0, quantity=10)
        with pytest.raises(ExecutionError, match="Falta precio"):
            p.equity({})

    def test_equity_all_cash(self):
        assert make_portfolio().equity({}) == pytest.approx(10_000.0)
