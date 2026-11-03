"""Tests de los adaptadores de broker (PaperBroker y AlpacaBroker).

AlpacaBroker se testea con un cliente HTTP falso que imita la interfaz de
``requests.Session``: todo el flujo (orden -> sondeo -> fill -> espejo
contable) queda cubierto sin tocar la red ni credenciales reales.
"""

from __future__ import annotations

import pytest

from trading_agent.brokers import make_broker
from trading_agent.brokers.alpaca import PAPER_URL, AlpacaBroker
from trading_agent.brokers.base import PaperBroker
from trading_agent.exceptions import (
    ConfigurationError,
    ExecutionError,
    SecurityError,
)
from trading_agent.portfolio import Portfolio


# --------------------------------------------------------------------- #
# Infraestructura de test                                               #
# --------------------------------------------------------------------- #
class FakeResponse:
    """Respuesta HTTP mínima con la interfaz que usa el adaptador.

    ``payload=None`` modela una respuesta sin cuerpo (p. ej. 204): como
    en ``requests``, ``.content`` queda vacío y ``.json()`` lanza.
    """

    def __init__(self, status_code: int = 200, payload=None,
                 text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else b"{}"
        self.text = text or (str(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("sin JSON")
        return self._payload


class FakeSession:
    """Cliente estilo requests.Session con respuestas programadas.

    Attributes:
        responses: lista FIFO de ``(method, path_suffix, FakeResponse,
            sticky)``; cada petición consume la primera entrada cuyo método
            y sufijo coinciden (las ``sticky`` no se consumen: responden
            indefinidamente, útil para bucles de sondeo).
        calls: registro de ``(method, url, json)`` para aserciones.
    """

    def __init__(self) -> None:
        self.headers: dict = {}
        self.responses: list[tuple[str, str, FakeResponse, bool]] = []
        self.calls: list[tuple[str, str, dict | None]] = []
        # Responder programable (tiene prioridad): fn(method, url, json,
        # calls) -> FakeResponse | None. Permite fakes con estado.
        self.responder = None

    def queue(self, method: str, path_suffix: str, *, sticky: bool = False,
              **kwargs) -> None:
        self.responses.append((method, path_suffix, FakeResponse(**kwargs),
                               sticky))

    def request(self, method: str, url: str, json=None, timeout=None,
                **kwargs):
        # El adaptador debe pedir siempre allow_redirects=False (seguridad).
        assert kwargs.get("allow_redirects") is False
        self.calls.append((method, url, json))
        if self.responder is not None:
            resp = self.responder(method, url, json, self.calls)
            if resp is not None:
                return resp
        # Las respuestas de un solo uso tienen prioridad sobre las sticky.
        for want_sticky in (False, True):
            for i, (m, suffix, resp, sticky) in enumerate(self.responses):
                if sticky == want_sticky and m == method and url.endswith(suffix):
                    if not sticky:
                        self.responses.pop(i)
                    return resp
        raise AssertionError(f"Petición inesperada: {method} {url}")


ACCOUNT_OK = {"status": "ACTIVE", "cash": "50000.00"}


def order_json(status: str, *, order_id: str = "ord-1", qty: str | None = None,
               price: str = "100.50") -> dict:
    """Orden JSON como la devuelve Alpaca: sin fill salvo que esté ``filled``
    (o se indique ``qty`` explícitamente para fills parciales)."""
    if qty is None:
        qty = "10" if status == "filled" else "0"
    return {"id": order_id, "status": status, "filled_qty": qty,
            "filled_avg_price": price}


def cancel_flow_session(*, before=("new", "0"), after=(("canceled", "0"),),
                        price: str = "100.50") -> FakeSession:
    """Sesión con estado para el flujo timeout -> DELETE -> lectura final.

    ``GET /v2/orders/ord-1`` devuelve ``before`` (estado, qty) hasta que se
    envía el DELETE; después recorre ``after`` en orden y repite el último
    (la cancelación de Alpaca es asíncrona: pending_cancel -> terminal).
    """
    session = FakeSession()

    def responder(method, url, json, calls):
        if method == "POST" and url.endswith("/v2/orders"):
            return FakeResponse(payload=order_json("new"))
        if method == "DELETE":
            return FakeResponse(status_code=204)
        if method == "GET" and url.endswith("/v2/orders/ord-1"):
            delete_idx = next((i for i, (m, _, _) in enumerate(calls)
                               if m == "DELETE"), None)
            if delete_idx is None:
                st, q = before
            else:
                n_after = sum(1 for m, _, _ in calls[delete_idx + 1:]
                              if m == "GET") - 1
                st, q = after[min(n_after, len(after) - 1)]
            return FakeResponse(payload=order_json(st, qty=q, price=price))
        return None

    session.responder = responder
    return session


def make_alpaca(monkeypatch, session: FakeSession | None = None,
                cash: float = 100_000.0, **kwargs) -> AlpacaBroker:
    """Crea un AlpacaBroker en paper con credenciales de entorno falsas."""
    monkeypatch.setenv("ALPACA_API_KEY", "clave-test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secreto-test")
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    kwargs.setdefault("order_timeout_s", 1.0)
    kwargs.setdefault("poll_interval_s", 0.0)
    return AlpacaBroker(Portfolio(cash=cash), session=session or FakeSession(),
                        **kwargs)


# --------------------------------------------------------------------- #
# PaperBroker y fábrica                                                 #
# --------------------------------------------------------------------- #
class TestPaperBroker:
    def test_delegates_to_portfolio_with_simulated_costs(self):
        p = Portfolio(cash=10_000.0, commission_pct=0.01)
        broker = PaperBroker(p)
        broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert p.cash == pytest.approx(10_000.0 - 1_000.0 - 10.0)
        trade = broker.close_position("AAPL", price_hint=110.0)
        assert trade.side == "sell"
        assert not p.has_position("AAPL")


class TestMakeBroker:
    def test_paper_by_name(self):
        assert isinstance(make_broker("paper", Portfolio(cash=1.0)),
                          PaperBroker)

    def test_unknown_name_raises(self):
        with pytest.raises(ConfigurationError, match="no soportado"):
            make_broker("robinhood", Portfolio(cash=1.0))

    def test_alpaca_without_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(SecurityError, match="ALPACA_API_KEY"):
            make_broker("alpaca", Portfolio(cash=1.0))


# --------------------------------------------------------------------- #
# AlpacaBroker: seguridad y construcción                                #
# --------------------------------------------------------------------- #
class TestAlpacaSecurity:
    def test_paper_url_is_default_and_needs_no_live_flag(self, monkeypatch):
        monkeypatch.delenv("TRADING_AGENT_LIVE", raising=False)
        broker = make_alpaca(monkeypatch)
        assert broker.is_paper
        assert broker.base_url == PAPER_URL

    def test_paper_is_decided_by_hostname_not_exact_text(self, monkeypatch):
        """Mayúsculas o puerto explícito no convierten paper en real."""
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.delenv("TRADING_AGENT_LIVE", raising=False)
        monkeypatch.setenv("ALPACA_BASE_URL", "https://PAPER-API.alpaca.markets:443")
        broker = AlpacaBroker(Portfolio(cash=1.0), session=FakeSession())
        assert broker.is_paper

    def test_live_url_without_flag_raises(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        monkeypatch.delenv("TRADING_AGENT_LIVE", raising=False)
        with pytest.raises(SecurityError, match="deshabilitado"):
            AlpacaBroker(Portfolio(cash=1.0), session=FakeSession())

    def test_live_url_with_flag_is_allowed(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        monkeypatch.setenv("TRADING_AGENT_LIVE", "YES")
        broker = AlpacaBroker(Portfolio(cash=1.0), session=FakeSession())
        assert not broker.is_paper

    def test_simulated_costs_in_mirror_portfolio_raise(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        with pytest.raises(ExecutionError, match="commission_pct=0"):
            AlpacaBroker(Portfolio(cash=1.0, commission_pct=0.001),
                         session=FakeSession())

    def test_credentials_go_to_headers_only(self, monkeypatch):
        session = FakeSession()
        make_alpaca(monkeypatch, session)
        assert session.headers["APCA-API-KEY-ID"] == "clave-test"
        assert session.headers["APCA-API-SECRET-KEY"] == "secreto-test"

    @pytest.mark.parametrize("url", [
        "https://evil.example.com",                     # host ajeno
        "https://api.alpaca.markets.evil.com",          # sufijo engañoso
        "https://user:pass@paper-api.alpaca.markets",   # credenciales en URL
        "https://paper-api.alpaca.markets/?x=1",        # parámetros
    ])
    def test_disallowed_urls_are_rejected(self, monkeypatch, url):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", url)
        monkeypatch.setenv("TRADING_AGENT_LIVE", "YES")
        with pytest.raises(SecurityError):
            AlpacaBroker(Portfolio(cash=1.0), session=FakeSession())

    def test_redirects_are_refused(self, monkeypatch):
        """Las cabeceras con credenciales no deben seguir a otro host."""
        session = FakeSession()
        session.queue("GET", "/v2/account", status_code=302, payload={})
        broker = make_alpaca(monkeypatch, session)
        with pytest.raises(ExecutionError, match="redirigir"):
            broker.account_cash()

    def test_plain_http_url_is_rejected(self, monkeypatch):
        """Las credenciales viajan en cabeceras: nunca sin TLS."""
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "http://paper-api.alpaca.markets")
        with pytest.raises(SecurityError, match="https"):
            AlpacaBroker(Portfolio(cash=1.0), session=FakeSession())


# --------------------------------------------------------------------- #
# AlpacaBroker: cuenta y órdenes                                        #
# --------------------------------------------------------------------- #
class TestAlpacaAccount:
    def test_account_cash(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/account", payload=ACCOUNT_OK)
        broker = make_alpaca(monkeypatch, session)
        assert broker.account_cash() == pytest.approx(50_000.0)

    def test_sync_cash_updates_mirror(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/account", payload=ACCOUNT_OK)
        broker = make_alpaca(monkeypatch, session, cash=1.0)
        broker.sync_cash()
        assert broker.portfolio.cash == pytest.approx(50_000.0)

    def test_inactive_account_raises(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/account",
                      payload={"status": "ACCOUNT_CLOSED", "cash": "0"})
        broker = make_alpaca(monkeypatch, session)
        with pytest.raises(ExecutionError, match="no activa"):
            broker.account_cash()


class TestAlpacaSync:
    def test_sync_positions_mirrors_open_longs(self, monkeypatch):
        """Tras un reinicio, las posiciones reales deben volver al espejo
        (si no, el RiskManager no las vería: sin stop-loss)."""
        session = FakeSession()
        session.queue("GET", "/v2/positions", payload=[
            {"symbol": "AAPL", "qty": "10", "side": "long",
             "avg_entry_price": "123.45"},
        ])
        broker = make_alpaca(monkeypatch, session)
        broker.sync_positions()
        pos = broker.portfolio.positions["AAPL"]
        assert pos.quantity == 10
        assert pos.entry_price == pytest.approx(123.45)

    def test_sync_positions_ignores_foreign_symbols(self, monkeypatch, caplog):
        """Posiciones de otros símbolos (otro proceso, manuales) no entran
        en el espejo: romperían la valoración del equity de este proceso."""
        session = FakeSession()
        session.queue("GET", "/v2/positions", payload=[
            {"symbol": "AAPL", "qty": "10", "side": "long",
             "avg_entry_price": "100"},
            {"symbol": "MSFT", "qty": "5", "side": "short",
             "avg_entry_price": "300"},   # ni siquiera se valida: se ignora
        ])
        broker = make_alpaca(monkeypatch, session)
        with caplog.at_level("WARNING"):
            broker.sync_positions(symbols=("AAPL",))
        assert set(broker.portfolio.positions) == {"AAPL"}
        assert "ajena" in caplog.text

    def test_trades_carry_broker_order_ids(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="o-77"))
        session.queue("GET", "/v2/orders/o-77",
                      payload=order_json("filled", order_id="o-77"))
        broker = make_alpaca(monkeypatch, session)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0,
                                     tag="2026-01-02T15:00:00")
        assert trade.order_id == "o-77"
        assert trade.client_order_id and trade.client_order_id.startswith(
            f"{broker._run_id}-AAPL-b-")
        assert broker.portfolio.trades[-1].order_id == "o-77"

    def test_sync_positions_rejects_shorts(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/positions", payload=[
            {"symbol": "AAPL", "qty": "-10", "side": "short",
             "avg_entry_price": "1"},
        ])
        broker = make_alpaca(monkeypatch, session)
        with pytest.raises(ExecutionError, match="no soportada"):
            broker.sync_positions()

    def test_sync_does_cash_and_positions(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/account", payload=ACCOUNT_OK)
        session.queue("GET", "/v2/positions", payload=[])
        broker = make_alpaca(monkeypatch, session, cash=1.0)
        broker.sync()
        assert broker.portfolio.cash == pytest.approx(50_000.0)
        assert broker.portfolio.positions == {}

    def test_market_clock(self, monkeypatch):
        session = FakeSession()
        session.queue("GET", "/v2/clock", payload={"is_open": True})
        session.queue("GET", "/v2/clock", payload={"is_open": False})
        broker = make_alpaca(monkeypatch, session)
        assert broker.is_market_open() is True
        assert broker.is_market_open() is False

    def test_market_clock_failure_is_closed(self, monkeypatch):
        """Si no se puede saber, no operar es la opción segura."""
        session = FakeSession()
        session.queue("GET", "/v2/clock", status_code=500, payload={})
        broker = make_alpaca(monkeypatch, session)
        assert broker.is_market_open() is False


class TestAlpacaOrders:
    def test_buy_mirrors_real_fill_price(self, monkeypatch):
        """El espejo contable usa el precio del fill, no el price_hint."""
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1",
                      payload=order_json("filled", price="101.25"))
        broker = make_alpaca(monkeypatch, session)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert trade.price == pytest.approx(101.25)
        assert broker.portfolio.positions["AAPL"].entry_price == pytest.approx(101.25)
        assert broker.portfolio.cash == pytest.approx(100_000.0 - 1_012.50)
        # La orden enviada es market/day por la cantidad entera pedida.
        method, _url, body = session.calls[0]
        assert (method, body["type"], body["time_in_force"],
                body["side"], body["qty"]) == ("POST", "market", "day",
                                               "buy", "10")

    def test_fill_above_local_cash_is_still_mirrored(self, monkeypatch):
        """El fill ya ocurrió: el espejo lo refleja y re-sincroniza el
        efectivo en vez de rechazarlo con el dinero ya fuera de la cuenta."""
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1",
                      payload=order_json("filled", price="120.00"))
        session.queue("GET", "/v2/account", payload=ACCOUNT_OK)
        broker = make_alpaca(monkeypatch, session, cash=1_000.0)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert trade.quantity == 10
        assert broker.portfolio.positions["AAPL"].quantity == 10
        assert broker.portfolio.cash == pytest.approx(50_000.0)  # re-sync

    def test_polls_until_filled(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1",
                      payload=order_json("partially_filled"))
        session.queue("GET", "/v2/orders/ord-1", payload=order_json("filled"))
        broker = make_alpaca(monkeypatch, session)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert trade.quantity == 10

    def test_sell_closes_mirror_position(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1",
                      payload=order_json("filled", price="100.00"))
        session.queue("POST", "/v2/orders",
                      payload=order_json("new", order_id="ord-2"))
        session.queue("GET", "/v2/orders/ord-2",
                      payload=order_json("filled", order_id="ord-2",
                                         price="110.00"))
        broker = make_alpaca(monkeypatch, session)
        broker.open_position("AAPL", quantity=10, price_hint=100.0)
        trade = broker.close_position("AAPL", price_hint=109.0)
        assert trade.pnl == pytest.approx(100.0)
        assert not broker.portfolio.has_position("AAPL")

    def test_client_order_id_is_sent_and_duplicate_is_recovered(self, monkeypatch):
        """Idempotencia: si Alpaca ya tiene la orden (422 por client_order_id
        duplicado), se recupera por id en vez de comprar dos veces."""
        session = FakeSession()
        broker = make_alpaca(monkeypatch, session)
        cid = broker._client_order_id("AAPL", "buy", "2026-01-02T15:00:00")
        assert len(cid) <= 48 and cid.startswith(broker._run_id)
        session.queue("POST", "/v2/orders", status_code=422,
                      payload={"message": "client_order_id must be unique"})
        session.queue("GET", f"client_order_id={cid}",
                      payload=order_json("new", order_id="ord-9"))
        session.queue("GET", "/v2/orders/ord-9",
                      payload=order_json("filled", order_id="ord-9"))
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0,
                                     tag="2026-01-02T15:00:00")
        assert trade.quantity == 10
        posted = [c for c in session.calls if c[0] == "POST"]
        assert posted[0][2]["client_order_id"] == cid
        assert len(posted) == 1  # no se reenvió la compra

    def test_timeout_with_partial_fill_mirrors_filled_shares(self, monkeypatch):
        """Tras cancelar por timeout, las acciones ya ejecutadas son reales:
        deben entrar en el espejo (antes se descartaban). La cancelación es
        asíncrona: solo cuenta el estado TERMINAL leído tras el DELETE."""
        session = cancel_flow_session(
            before=("partially_filled", "3"),
            after=(("pending_cancel", "3"), ("canceled", "4")))
        broker = make_alpaca(monkeypatch, session, order_timeout_s=0.01)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0)
        # Se ejecutó UNA acción más entre el DELETE y la confirmación: el
        # espejo refleja el estado terminal (4), no la lectura previa (3).
        assert trade.quantity == 4
        assert broker.portfolio.positions["AAPL"].quantity == 4

    def test_cancel_that_never_turns_terminal_resyncs_and_raises(self, monkeypatch, caplog):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1", sticky=True,
                      payload=order_json("pending_cancel"))
        session.queue("DELETE", "/v2/orders/ord-1", status_code=204)
        session.queue("GET", "/v2/account", payload=ACCOUNT_OK)
        session.queue("GET", "/v2/positions", payload=[])
        broker = make_alpaca(monkeypatch, session, order_timeout_s=0.01,
                             cancel_wait_s=0.01)
        with caplog.at_level("CRITICAL"), pytest.raises(ExecutionError, match="no terminal"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert "re-sincroniza" in caplog.text
        assert broker.portfolio.cash == pytest.approx(50_000.0)  # sync corrió

    def test_retry_after_partial_fill_sends_a_new_order(self, monkeypatch):
        """Mismo tag (misma vela) tras un fill parcial: el reintento debe
        enviar una orden NUEVA por el resto, no recuperar la antigua y
        contabilizarla dos veces."""
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="buy"))
        session.queue("GET", "/v2/orders/buy", payload=order_json("filled", order_id="buy"))
        # Primera venta: termina cancelada con 4 de 10.
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="s1"))
        session.queue("GET", "/v2/orders/s1",
                      payload=order_json("canceled", order_id="s1", qty="4"))
        # Segunda venta (mismo tag): orden nueva por las 6 restantes.
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="s2"))
        session.queue("GET", "/v2/orders/s2",
                      payload=order_json("filled", order_id="s2", qty="6"))
        broker = make_alpaca(monkeypatch, session)
        broker.open_position("AAPL", quantity=10, price_hint=100.0, tag="T")
        broker.close_position("AAPL", price_hint=100.0, tag="T-stop_loss")
        assert broker.portfolio.positions["AAPL"].quantity == 6
        broker.close_position("AAPL", price_hint=100.0, tag="T-stop_loss")
        assert not broker.portfolio.has_position("AAPL")
        posted = [c[2] for c in session.calls if c[0] == "POST"]
        sell_ids = [b["client_order_id"] for b in posted if b["side"] == "sell"]
        assert len(sell_ids) == 2 and sell_ids[0] != sell_ids[1]
        assert posted[-1]["qty"] == "6"

    def test_retry_after_rejection_sends_a_new_order(self, monkeypatch):
        """Un stop rechazado no puede quedar bloqueado el resto de la vela."""
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="buy"))
        session.queue("GET", "/v2/orders/buy", payload=order_json("filled", order_id="buy"))
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="s1"))
        session.queue("GET", "/v2/orders/s1", payload=order_json("rejected", order_id="s1"))
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="s2"))
        session.queue("GET", "/v2/orders/s2", payload=order_json("filled", order_id="s2"))
        broker = make_alpaca(monkeypatch, session)
        broker.open_position("AAPL", quantity=10, price_hint=100.0, tag="T")
        with pytest.raises(ExecutionError, match="rejected"):
            broker.close_position("AAPL", price_hint=95.0, tag="T-stop_loss")
        broker.close_position("AAPL", price_hint=95.0, tag="T-stop_loss")
        assert not broker.portfolio.has_position("AAPL")
        posted = [c[2] for c in session.calls if c[0] == "POST"]
        assert posted[1]["client_order_id"] != posted[2]["client_order_id"]

    def test_recovered_order_already_accounted_is_not_mirrored_twice(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new", order_id="o1"))
        session.queue("GET", "/v2/orders/o1", payload=order_json("filled", order_id="o1"))
        broker = make_alpaca(monkeypatch, session)
        trade = broker.open_position("AAPL", quantity=10, price_hint=100.0, tag="T")
        cid = trade.client_order_id
        # Simular un reintento con el MISMO id (p. ej. respuesta perdida)
        # cuya orden ya está contabilizada: Alpaca responde 422 duplicado.
        broker._attempts.clear()
        broker._attempt_keys.clear()
        session.queue("POST", "/v2/orders", status_code=422,
                      payload={"message": "client_order_id must be unique"})
        session.queue("GET", f"client_order_id={cid}",
                      payload=order_json("filled", order_id="o1"))
        broker.portfolio.positions.clear()  # como si se hubiera vendido
        with pytest.raises(ExecutionError, match="ya contabilizada"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0, tag="T")
        assert not broker.portfolio.has_position("AAPL")  # sin fantasma

    def test_delete_204_is_not_reported_as_failure(self, monkeypatch, caplog):
        session = cancel_flow_session(before=("new", "0"),
                                      after=(("canceled", "0"),))  # DELETE -> 204
        broker = make_alpaca(monkeypatch, session, order_timeout_s=0.01)
        with caplog.at_level("ERROR"), pytest.raises(ExecutionError, match="sin fill"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert "No se pudo cancelar" not in caplog.text

    def test_partial_sell_reduces_mirror_position(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1", payload=order_json("filled"))
        session.queue("POST", "/v2/orders",
                      payload=order_json("new", order_id="ord-2"))
        session.queue("GET", "/v2/orders/ord-2",
                      payload=order_json("canceled", order_id="ord-2", qty="4",
                                         price="105"))
        broker = make_alpaca(monkeypatch, session)
        broker.open_position("AAPL", quantity=10, price_hint=100.0)
        trade = broker.close_position("AAPL", price_hint=105.0)
        assert trade.quantity == 4
        assert broker.portfolio.positions["AAPL"].quantity == 6  # resto abierto

    def test_rejected_order_raises(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", payload=order_json("new"))
        session.queue("GET", "/v2/orders/ord-1", payload=order_json("rejected"))
        broker = make_alpaca(monkeypatch, session)
        with pytest.raises(ExecutionError, match="rejected"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0)
        assert not broker.portfolio.has_position("AAPL")

    def test_fill_timeout_cancels_and_raises(self, monkeypatch):
        # La orden responde "new" hasta el DELETE; después, cancelada sin fill.
        session = cancel_flow_session(before=("new", "0"),
                                      after=(("canceled", "0"),))
        broker = make_alpaca(monkeypatch, session, order_timeout_s=0.01)
        with pytest.raises(ExecutionError, match="sin fill"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0)
        # Se intentó cancelar la orden colgada.
        assert any(m == "DELETE" for m, _, _ in session.calls)

    def test_http_error_raises_execution_error(self, monkeypatch):
        session = FakeSession()
        session.queue("POST", "/v2/orders", status_code=403,
                      payload={"message": "forbidden"})
        broker = make_alpaca(monkeypatch, session)
        with pytest.raises(ExecutionError, match="403"):
            broker.open_position("AAPL", quantity=10, price_hint=100.0)

    @pytest.mark.parametrize("qty", [0, -5, 2.5])
    def test_invalid_quantity_raises(self, monkeypatch, qty):
        broker = make_alpaca(monkeypatch)
        with pytest.raises(ExecutionError, match="Cantidad inválida"):
            broker.open_position("AAPL", quantity=qty, price_hint=100.0)

    def test_close_without_mirror_position_raises(self, monkeypatch):
        broker = make_alpaca(monkeypatch)
        with pytest.raises(ExecutionError, match="posición espejo"):
            broker.close_position("AAPL", price_hint=100.0)
