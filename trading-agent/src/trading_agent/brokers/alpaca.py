"""Adaptador de broker para Alpaca Markets (acciones de EE. UU.).

Alpaca expone una API REST sin comisiones con dos entornos que comparten
contrato: **paper** (``https://paper-api.alpaca.markets``, dinero ficticio)
y **live** (``https://api.alpaca.markets``, dinero real). Este adaptador
usa paper por defecto; apuntarlo al entorno live exige además exportar
``TRADING_AGENT_LIVE=YES`` (ver ``security.py``) — sin eso, el constructor
aborta con :class:`~trading_agent.exceptions.SecurityError`.

Credenciales (solo variables de entorno, nunca en código)::

    ALPACA_API_KEY      clave de API (dashboard de Alpaca)
    ALPACA_SECRET_KEY   secreto de API
    ALPACA_BASE_URL     opcional; defecto: paper. Live: https://api.alpaca.markets

Flujo de una orden (``open_position`` / ``close_position``):

1. ``POST /v2/orders`` con una orden *market* (time_in_force ``day``).
2. Bucle de sondeo de ``GET /v2/orders/{id}`` hasta estado ``filled``
   (o error/timeout; en timeout se intenta cancelar la orden).
3. El fill real (``filled_avg_price``, ``filled_qty``) se refleja en el
   ``Portfolio`` local, que por eso debe tener comisión y slippage en 0:
   los costes reales ya vienen dentro del precio de ejecución.

El cliente HTTP es inyectable (cualquier objeto con la interfaz de
``requests.Session``), así los tests cubren todo el flujo sin red.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..exceptions import ExecutionError, SecurityError
from ..portfolio import Portfolio, Position, Trade
from ..security import assert_live_trading_allowed, get_secret
from .base import BrokerAdapter

logger = logging.getLogger(__name__)

#: URL del entorno paper (dinero ficticio). Es el defecto deliberadamente.
PAPER_URL = "https://paper-api.alpaca.markets"

#: Estados terminales de una orden que significan "no se ejecutó".
_FAILED_STATES = frozenset({"canceled", "expired", "rejected", "suspended",
                            "done_for_day", "stopped"})


class AlpacaBroker(BrokerAdapter):
    """Ejecución real de órdenes de mercado contra Alpaca.

    Args:
        portfolio: portafolio local espejo. Debe tener ``commission_pct``
            y ``slippage_pct`` en 0 (los costes reales llegan en el fill).
        session: cliente HTTP estilo ``requests.Session``; ``None`` crea
            uno real (inyectable en tests).
        order_timeout_s: segundos máximos esperando el fill de una orden.
        poll_interval_s: segundos entre sondeos del estado de la orden.

    Raises:
        SecurityError: si faltan credenciales, o si la URL apunta al
            entorno live sin ``TRADING_AGENT_LIVE=YES``.
        ExecutionError: si el portafolio espejo tiene costes simulados.
    """

    def __init__(self, portfolio: Portfolio, *, session: Any = None,
                 order_timeout_s: float = 30.0,
                 poll_interval_s: float = 0.5) -> None:
        super().__init__(portfolio)
        if portfolio.commission_pct != 0.0 or portfolio.slippage_pct != 0.0:
            raise ExecutionError(
                "Con un broker real, el Portfolio espejo debe tener "
                "commission_pct=0 y slippage_pct=0: los costes reales ya "
                "vienen en el precio del fill.")

        key = get_secret("ALPACA_API_KEY")
        secret = get_secret("ALPACA_SECRET_KEY")
        self.base_url = (get_secret("ALPACA_BASE_URL", required=False)
                         or PAPER_URL).rstrip("/")
        if not self.base_url.startswith("https://"):
            # Las credenciales viajan en cabeceras: nunca en claro.
            raise SecurityError(
                f"ALPACA_BASE_URL debe usar https://, no {self.base_url!r}")
        self.is_paper = self.base_url == PAPER_URL
        if not self.is_paper:
            # Dinero real: exige la habilitación consciente del operador.
            assert_live_trading_allowed()
            logger.warning("AlpacaBroker en modo LIVE (dinero real): %s",
                           self.base_url)
        else:
            logger.info("AlpacaBroker en modo paper: %s", self.base_url)

        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._session.headers.update({
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        })
        self.order_timeout_s = order_timeout_s
        self.poll_interval_s = poll_interval_s

    # ------------------------------------------------------------------ #
    # HTTP                                                               #
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, *, json: dict | None = None,
                 ok_status: tuple[int, ...] = (200, 201)) -> dict:
        """Llama a la API y devuelve el JSON, con errores tipados.

        Args:
            method: verbo HTTP (``"GET"``, ``"POST"``, ``"DELETE"``).
            path: ruta bajo la base, p. ej. ``"/v2/orders"``.
            json: cuerpo JSON opcional.
            ok_status: códigos aceptados como éxito.

        Returns:
            Respuesta deserializada (dict).

        Raises:
            ExecutionError: ante error HTTP, de red o respuesta no-JSON.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(method, url, json=json, timeout=15)
        except Exception as exc:
            raise ExecutionError(f"Fallo de red hacia Alpaca ({path}): {exc}"
                                 ) from exc
        if resp.status_code not in ok_status:
            raise ExecutionError(
                f"Alpaca respondió {resp.status_code} en {path}: "
                f"{resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ExecutionError(f"Respuesta no-JSON de Alpaca en {path}"
                                 ) from exc

    # ------------------------------------------------------------------ #
    # Cuenta                                                             #
    # ------------------------------------------------------------------ #
    def account_cash(self) -> float:
        """Efectivo disponible en la cuenta de Alpaca.

        Returns:
            Efectivo (USD) según ``GET /v2/account``.

        Raises:
            ExecutionError: si la cuenta no está activa o la API falla.
        """
        account = self._request("GET", "/v2/account")
        status = account.get("status")
        if status != "ACTIVE":
            raise ExecutionError(f"Cuenta Alpaca no activa: {status!r}")
        return float(account["cash"])

    def sync_cash(self) -> None:
        """Alinea el efectivo del portafolio espejo con el de la cuenta."""
        self.portfolio.cash = self.account_cash()
        logger.info("Efectivo sincronizado con Alpaca: %.2f USD",
                    self.portfolio.cash)

    def sync_positions(self) -> None:
        """Refleja en el espejo las posiciones largas abiertas en Alpaca.

        Imprescindible tras un reinicio del proceso: sin esto, una
        posición real abierta sería invisible para el RiskManager (sin
        stop-loss) y el agente podría comprar de nuevo encima.

        Raises:
            ExecutionError: si la API falla o hay una posición corta
                (este sistema solo opera en largo).
        """
        raw = self._request("GET", "/v2/positions")
        if not isinstance(raw, list):
            raise ExecutionError("Respuesta inesperada de /v2/positions")
        self.portfolio.positions.clear()
        for p in raw:
            qty = float(p["qty"])
            if p.get("side", "long") != "long" or qty <= 0:
                raise ExecutionError(
                    f"Posición no soportada en {p.get('symbol')}: "
                    f"side={p.get('side')!r} qty={qty}")
            self.portfolio.positions[p["symbol"]] = Position(
                p["symbol"], qty, float(p["avg_entry_price"]))
            logger.info("Posición sincronizada desde Alpaca: %s %g @ %.4f",
                        p["symbol"], qty, float(p["avg_entry_price"]))

    def sync(self) -> None:
        """Sincroniza efectivo y posiciones (llamar al arrancar)."""
        self.sync_cash()
        self.sync_positions()

    def is_market_open(self) -> bool:
        """Consulta el reloj de mercado de Alpaca (``GET /v2/clock``).

        Returns:
            ``True`` si el mercado está abierto. Ante un fallo de la API
            devuelve ``False`` (no operar es la opción segura) y lo registra.
        """
        try:
            return bool(self._request("GET", "/v2/clock").get("is_open", False))
        except ExecutionError as exc:
            logger.error("No se pudo consultar el reloj de mercado: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Órdenes                                                            #
    # ------------------------------------------------------------------ #
    def _await_fill(self, order_id: str) -> dict:
        """Sondea una orden hasta que se ejecute por completo.

        Bucle de sondeo acotado por ``order_timeout_s``; en timeout se
        intenta cancelar la orden (best-effort) antes de fallar, para no
        dejar órdenes vivas sin contabilidad local.

        Args:
            order_id: id de la orden devuelto por ``POST /v2/orders``.

        Returns:
            El JSON final de la orden con estado ``filled``.

        Raises:
            ExecutionError: si la orden termina en estado fallido o no se
                ejecuta dentro del timeout.
        """
        deadline = time.monotonic() + self.order_timeout_s
        while True:
            order = self._request("GET", f"/v2/orders/{order_id}")
            status = order.get("status")
            if status == "filled":
                return order
            if status in _FAILED_STATES:
                if float(order.get("filled_qty") or 0) > 0:
                    # Cancelada/expirada con fill parcial: las acciones
                    # ejecutadas son reales y deben contabilizarse.
                    logger.warning("Orden %s %s con fill parcial %s",
                                   order_id, status, order.get("filled_qty"))
                    return order
                raise ExecutionError(
                    f"Orden {order_id} terminó en estado {status!r}")
            if time.monotonic() >= deadline:
                return self._cancel_and_reconcile(order_id, status)
            time.sleep(self.poll_interval_s)

    def _cancel_and_reconcile(self, order_id: str, last_status: str | None) -> dict:
        """Cancela una orden colgada y recupera lo que SÍ se ejecutó.

        Tras el timeout: ``DELETE`` (best-effort) y nueva lectura de la
        orden. Si tiene ``filled_qty > 0`` se devuelve para que el espejo
        contable refleje las acciones reales; si nada se ejecutó, se falla.

        Args:
            order_id: id de la orden.
            last_status: último estado visto (solo para el mensaje).

        Returns:
            El JSON final de la orden con fill parcial.

        Raises:
            ExecutionError: si no se ejecutó ninguna acción.
        """
        try:
            self._request("DELETE", f"/v2/orders/{order_id}",
                          ok_status=(200, 204, 404, 422))
        except ExecutionError:
            logger.error("No se pudo cancelar la orden %s tras el timeout. "
                         "REVISAR EN EL DASHBOARD.", order_id)
        try:
            final = self._request("GET", f"/v2/orders/{order_id}")
        except ExecutionError:
            final = {}
        if float(final.get("filled_qty") or 0) > 0:
            logger.warning("Orden %s cancelada tras timeout con fill parcial "
                           "%s @ %s", order_id, final.get("filled_qty"),
                           final.get("filled_avg_price"))
            return final
        raise ExecutionError(
            f"Orden {order_id} sin fill tras {self.order_timeout_s}s "
            f"(último estado: {last_status!r})")

    def _market_order(self, symbol: str, side: str, qty: float,
                      client_order_id: str | None = None) -> tuple[float, float]:
        """Envía una orden market y espera su fill (total o parcial).

        Idempotencia: con ``client_order_id`` Alpaca rechaza un duplicado
        (HTTP 422). Si eso ocurre, la orden ya fue enviada en un intento
        anterior (p. ej. el proceso murió tras el POST): se recupera por
        su id de cliente y se espera su fill en vez de duplicar la compra.

        Args:
            symbol: ticker.
            side: ``"buy"`` o ``"sell"``.
            qty: nº de acciones (entero positivo).
            client_order_id: id determinista por decisión (opcional).

        Returns:
            Tupla ``(filled_qty, filled_avg_price)`` del fill real.
        """
        if qty <= 0 or qty != int(qty):
            raise ExecutionError(
                f"Cantidad inválida para Alpaca: {qty!r} (entero > 0)")
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        try:
            order = self._request("POST", "/v2/orders", json=body)
        except ExecutionError as exc:
            if client_order_id and "422" in str(exc) and "client_order_id" in str(exc):
                logger.warning("Orden %s ya existía en Alpaca: se recupera en "
                               "vez de duplicarla", client_order_id)
                order = self._request(
                    "GET", "/v2/orders:by_client_order_id"
                           f"?client_order_id={client_order_id}")
            else:
                raise
        filled = self._await_fill(order["id"])
        filled_qty = float(filled["filled_qty"])
        if filled_qty <= 0:
            raise ExecutionError(f"Orden {order['id']} sin acciones ejecutadas")
        return filled_qty, float(filled["filled_avg_price"])

    @staticmethod
    def _client_order_id(symbol: str, side: str, tag: str | None) -> str | None:
        """Id de cliente determinista por decisión (≤ 48 caracteres)."""
        if not tag:
            return None
        safe = "".join(c if c.isalnum() else "-" for c in tag)
        return f"ta-{symbol}-{side}-{safe}"[:48]

    # ------------------------------------------------------------------ #
    # Interfaz BrokerAdapter                                             #
    # ------------------------------------------------------------------ #
    def open_position(self, symbol: str, quantity: float,
                      price_hint: float, *, tag: str | None = None) -> Trade:
        """Compra a mercado y refleja el fill real en el portafolio local.

        Ver contrato en :meth:`BrokerAdapter.open_position`. ``price_hint``
        solo se usa para registrar la desviación del fill en el log.
        """
        filled_qty, fill_price = self._market_order(
            symbol, "buy", quantity,
            client_order_id=self._client_order_id(symbol, "buy", tag))
        if filled_qty != quantity:
            logger.warning("Fill parcial en compra %s: %g de %g", symbol,
                           filled_qty, quantity)
        logger.info("Alpaca BUY %s: %g @ %.4f (referencia %.4f, desvío %+.2f%%)",
                    symbol, filled_qty, fill_price, price_hint,
                    (fill_price / price_hint - 1.0) * 100 if price_hint else 0.0)
        # Espejo contable al precio real de ejecución (costes incluidos).
        return self.portfolio.open_position(symbol, fill_price, filled_qty)

    def close_position(self, symbol: str, price_hint: float, *,
                       tag: str | None = None) -> Trade:
        """Vende a mercado toda la posición y refleja el fill real.

        Un fill parcial reduce la posición espejo por lo realmente vendido
        (el resto sigue abierto y protegido por el RiskManager) y se
        registra como error operativo.

        Ver contrato en :meth:`BrokerAdapter.close_position`.
        """
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            raise ExecutionError(f"No hay posición espejo abierta en {symbol}")
        filled_qty, fill_price = self._market_order(
            symbol, "sell", pos.quantity,
            client_order_id=self._client_order_id(symbol, "sell", tag))
        if filled_qty != pos.quantity:
            logger.error("Fill parcial en venta %s: %g de %g; el resto sigue "
                         "abierto. REVISAR EN EL DASHBOARD.", symbol,
                         filled_qty, pos.quantity)
        logger.info("Alpaca SELL %s: %g @ %.4f (referencia %.4f)",
                    symbol, filled_qty, fill_price, price_hint)
        return self.portfolio.reduce_position(symbol, fill_price, filled_qty)
