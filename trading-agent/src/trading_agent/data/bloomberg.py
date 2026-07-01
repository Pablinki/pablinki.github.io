"""Proveedor de datos basado en Bloomberg (Desktop/Server API, ``blpapi``).

Requisitos
----------
- Paquete oficial ``blpapi`` instalado (``pip install blpapi`` con el
  índice de Bloomberg) y una Terminal Bloomberg o B-PIPE accesible.
- Variables de entorno (nunca en código, ver ``security.py``):

  - ``BLOOMBERG_HOST``: host del servicio (típicamente ``localhost``).
  - ``BLOOMBERG_PORT``: puerto (típicamente ``8194``).

Si ``blpapi`` no está disponible, la clase se puede importar igualmente
(el import es perezoso) pero instanciarla lanza ``ConfigurationError`` —
así el resto del sistema y los tests no dependen de Bloomberg.

Mapeo de intervalos
-------------------
Bloomberg intradía usa minutos enteros; diario usa ``HistoricalDataRequest``.
``interval="1d"`` -> histórico diario; ``"1m"/"5m"/"1h"`` -> barras intradía.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from ..exceptions import ConfigurationError, DataProviderError
from ..security import get_secret
from .provider import DataProvider, validate_ohlcv, with_retries

logger = logging.getLogger(__name__)

#: Traducción intervalo canónico -> minutos de barra intradía Bloomberg.
_INTRADAY_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


class BloombergProvider(DataProvider):
    """Implementación de :class:`DataProvider` sobre ``blpapi``.

    El ciclo de vida de la sesión es: ``_open_session()`` por petición,
    con cierre garantizado en ``finally`` (las sesiones colgadas agotan
    conexiones de la Terminal).
    """

    def __init__(self, config) -> None:  # noqa: ANN001 - DataConfig
        super().__init__(config)
        try:
            import blpapi  # noqa: F401
        except ImportError as exc:
            raise ConfigurationError(
                "El proveedor 'bloomberg' requiere el paquete 'blpapi' y una "
                "Terminal Bloomberg. Instálalo o usa provider: 'yahoo'."
            ) from exc
        self._host = get_secret("BLOOMBERG_HOST")
        self._port = int(get_secret("BLOOMBERG_PORT"))

    # ------------------------------------------------------------------ #
    # Sesión                                                             #
    # ------------------------------------------------------------------ #
    def _open_session(self):  # noqa: ANN202 - blpapi.Session
        """Abre y arranca una sesión blpapi contra ``//blp/refdata``.

        Returns:
            ``blpapi.Session`` iniciada con el servicio refdata abierto.

        Raises:
            DataProviderError: si la Terminal no responde.
        """
        import blpapi

        opts = blpapi.SessionOptions()
        opts.setServerHost(self._host)
        opts.setServerPort(self._port)
        session = blpapi.Session(opts)
        if not session.start():
            raise DataProviderError("No se pudo iniciar la sesión Bloomberg")
        if not session.openService("//blp/refdata"):
            session.stop()
            raise DataProviderError("No se pudo abrir //blp/refdata")
        return session

    # ------------------------------------------------------------------ #
    # Peticiones                                                         #
    # ------------------------------------------------------------------ #
    def _request_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """HistoricalDataRequest diario -> DataFrame canónico."""
        import blpapi

        session = self._open_session()
        try:
            service = session.getService("//blp/refdata")
            request = service.createRequest("HistoricalDataRequest")
            # Convención Bloomberg: "AAPL US Equity".
            request.getElement("securities").appendValue(f"{symbol} US Equity")
            for f in ("OPEN", "HIGH", "LOW", "PX_LAST", "VOLUME"):
                request.getElement("fields").appendValue(f)
            request.set("startDate", start.strftime("%Y%m%d"))
            request.set("endDate", end.strftime("%Y%m%d"))
            session.sendRequest(request)

            rows: list[dict] = []
            # Bucle de eventos blpapi: se consumen eventos PARTIAL_RESPONSE
            # hasta recibir el evento RESPONSE final.
            while True:
                event = session.nextEvent(timeout=10_000)
                for msg in event:
                    if not msg.hasElement("securityData"):
                        continue
                    field_data = msg.getElement("securityData").getElement("fieldData")
                    for i in range(field_data.numValues()):
                        bar = field_data.getValueAsElement(i)
                        rows.append({
                            "date": bar.getElementAsDatetime("date"),
                            "open": bar.getElementAsFloat("OPEN"),
                            "high": bar.getElementAsFloat("HIGH"),
                            "low": bar.getElementAsFloat("LOW"),
                            "close": bar.getElementAsFloat("PX_LAST"),
                            "volume": bar.getElementAsFloat("VOLUME"),
                        })
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        finally:
            session.stop()

        if not rows:
            raise DataProviderError(f"Bloomberg no devolvió datos para {symbol}")
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        return validate_ohlcv(df, symbol)

    def _request_intraday(self, symbol: str, start: dt.datetime,
                          end: dt.datetime, minutes: int) -> pd.DataFrame:
        """IntradayBarRequest -> DataFrame canónico."""
        import blpapi

        session = self._open_session()
        try:
            service = session.getService("//blp/refdata")
            request = service.createRequest("IntradayBarRequest")
            request.set("security", f"{symbol} US Equity")
            request.set("eventType", "TRADE")
            request.set("interval", minutes)
            request.set("startDateTime", start)
            request.set("endDateTime", end)
            session.sendRequest(request)

            rows: list[dict] = []
            while True:
                event = session.nextEvent(timeout=10_000)
                for msg in event:
                    if not msg.hasElement("barData"):
                        continue
                    bars = msg.getElement("barData").getElement("barTickData")
                    for i in range(bars.numValues()):
                        bar = bars.getValueAsElement(i)
                        rows.append({
                            "date": bar.getElementAsDatetime("time"),
                            "open": bar.getElementAsFloat("open"),
                            "high": bar.getElementAsFloat("high"),
                            "low": bar.getElementAsFloat("low"),
                            "close": bar.getElementAsFloat("close"),
                            "volume": bar.getElementAsFloat("volume"),
                        })
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        finally:
            session.stop()

        if not rows:
            raise DataProviderError(f"Bloomberg no devolvió barras para {symbol}")
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        return validate_ohlcv(df, symbol)

    # ------------------------------------------------------------------ #
    # Interfaz DataProvider                                              #
    # ------------------------------------------------------------------ #
    def fetch_historical(self, symbol: str) -> pd.DataFrame:
        """Ver contrato en :meth:`DataProvider.fetch_historical`."""
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=self.config.lookback_days)

        def _fetch() -> pd.DataFrame:
            if self.config.interval == "1d":
                return self._request_daily(symbol, start.date(), end.date())
            minutes = _INTRADAY_MINUTES.get(self.config.interval)
            if minutes is None:
                raise ConfigurationError(
                    f"Intervalo {self.config.interval!r} no soportado por "
                    f"Bloomberg; usa uno de {sorted(_INTRADAY_MINUTES)} o '1d'")
            return self._request_intraday(symbol, start, end, minutes)

        return with_retries(_fetch, max_retries=self.config.max_retries,
                            backoff_s=self.config.retry_backoff_s,
                            what=f"histórico Bloomberg {symbol}")

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Últimas barras intradía (o última vela diaria)."""
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=5)

        def _fetch() -> pd.DataFrame:
            if self.config.interval == "1d":
                return self._request_daily(symbol, start.date(), end.date())
            minutes = _INTRADAY_MINUTES.get(self.config.interval, 1)
            return self._request_intraday(symbol, start, end, minutes)

        return with_retries(_fetch, max_retries=self.config.max_retries,
                            backoff_s=self.config.retry_backoff_s,
                            what=f"último precio Bloomberg {symbol}")
