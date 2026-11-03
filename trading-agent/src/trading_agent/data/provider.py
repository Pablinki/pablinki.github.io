"""Capa de abstracción de proveedores de datos de mercado.

Arquitectura (patrón *Strategy* + *Factory*)
--------------------------------------------
El resto del sistema (entorno, agente, loop en vivo) solo conoce la
interfaz :class:`DataProvider`. Cambiar de Yahoo Finance a Bloomberg es
cambiar una línea de configuración (``data.provider: bloomberg``), no el
código — esto es clave para la escalabilidad: se pueden añadir proveedores
(IEX, Polygon, un broker propio) implementando dos métodos.

Formato canónico de datos
-------------------------
Todos los proveedores devuelven un ``pandas.DataFrame`` con:

- Índice: ``DatetimeIndex`` ordenado ascendente, zona horaria UTC.
- Columnas (float64): ``open, high, low, close, volume``.

La validación centralizada en :func:`validate_ohlcv` garantiza que datos
corruptos nunca llegan al agente (fail-fast con
:class:`~trading_agent.exceptions.DataValidationError`).
"""

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Callable
from typing import TypeVar

import pandas as pd

from ..config import DataConfig
from ..exceptions import (
    ConfigurationError,
    DataProviderError,
    DataValidationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

#: Columnas obligatorias del formato canónico OHLCV.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

T = TypeVar("T")


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Valida y normaliza un DataFrame OHLCV al formato canónico.

    Args:
        df: DataFrame crudo devuelto por un proveedor.
        symbol: ticker asociado (solo para mensajes de error).

    Returns:
        DataFrame validado: columnas ``open, high, low, close, volume`` en
        float64, índice datetime UTC ascendente, sin NaN ni duplicados.

    Raises:
        DataValidationError: si faltan columnas, está vacío, contiene
            precios no positivos o el índice no es temporal.

    Ejemplo:
        >>> import pandas as pd
        >>> idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        >>> raw = pd.DataFrame({"open": [1., 2.], "high": [2., 3.],
        ...                     "low": [.5, 1.5], "close": [1.5, 2.5],
        ...                     "volume": [100., 200.]}, index=idx)
        >>> validate_ohlcv(raw, "TEST").shape
        (2, 5)
    """
    if df is None or df.empty:
        raise DataValidationError(f"{symbol}: el proveedor devolvió datos vacíos")

    df = df.copy()
    df.columns = pd.Index([str(c).lower() for c in df.columns])

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"{symbol}: faltan columnas {missing}")

    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(f"{symbol}: el índice debe ser DatetimeIndex")

    df = df.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    df.index = index.tz_localize("UTC") if index.tz is None \
        else index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()

    if df.empty:
        raise DataValidationError(f"{symbol}: todas las filas tenían NaN")

    # Una vela corrupta aislada (precio <= 0, volumen negativo) no debe
    # invalidar toda la ventana y parar el trading: se descarta con aviso.
    # Si la corrupción es generalizada (> 1% de las filas), el origen no
    # es fiable y sí se rechaza.
    bad = ((df[["open", "high", "low", "close"]] <= 0).any(axis=1)
           | (df["volume"] < 0))
    n_bad = int(bad.sum())
    if n_bad:
        if n_bad > max(1, len(df) // 100):
            raise DataValidationError(
                f"{symbol}: {n_bad} velas con precios <= 0 o volumen negativo")
        logger.warning("%s: %d vela(s) corrupta(s) descartada(s): %s", symbol,
                       n_bad, list(df.index[bad].astype(str)))
        df = df[~bad]
    return df


#: Duración de cada intervalo canónico de vela.
INTERVAL_DURATION = {
    "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1), "60m": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}


def completed_bars(df: pd.DataFrame, interval: str,
                   now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Descarta la última vela si todavía está en formación.

    Los proveedores devuelven la vela en curso con su cierre parcial. El
    entrenamiento solo vio velas completas, así que en vivo el estado se
    construye únicamente con velas cerradas (la vela parcial sirve solo
    como último precio conocido para valorar y ejecutar).

    Args:
        df: OHLCV canónico (índice datetime UTC ascendente).
        interval: intervalo canónico de las velas.
        now: instante actual UTC (inyectable en tests); por defecto ahora.

    Returns:
        ``df`` sin la última fila si ``último_índice + duración > now``;
        de lo contrario ``df`` intacto. Intervalos desconocidos devuelven
        ``df`` intacto.

    Ejemplo:
        >>> idx = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
        >>> df = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)
        >>> len(completed_bars(df, "1d", now=pd.Timestamp("2024-01-03 15:00", tz="UTC")))
        1
        >>> len(completed_bars(df, "1d", now=pd.Timestamp("2024-01-05", tz="UTC")))
        2
    """
    duration = INTERVAL_DURATION.get(interval)
    if duration is None or df.empty:
        return df
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    if last + duration > now:
        return df.iloc[:-1]
    return df


def with_retries(fn: Callable[[], T], *, max_retries: int,
                 backoff_s: float, what: str) -> T:
    """Ejecuta ``fn`` reintentando ante fallos transitorios de datos.

    Bucle de reintento con **backoff exponencial**: tras el intento ``i``
    fallido espera ``backoff_s * 2**i`` segundos. Los errores de
    validación (:class:`DataValidationError`) NO se reintentan: los datos
    de origen están mal y reintentar no lo arregla.

    Args:
        fn: función sin argumentos a ejecutar (cerrar sobre los args con
            ``lambda`` o ``functools.partial``).
        max_retries: nº máximo de reintentos adicionales al primer intento.
        backoff_s: espera base en segundos.
        what: descripción para los logs, p. ej. ``"histórico AAPL"``.

    Returns:
        El valor devuelto por ``fn``.

    Raises:
        DataProviderError: si se agotan los reintentos.
        DataValidationError: propagada inmediatamente, sin reintentos.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (DataValidationError, RateLimitError):
            raise  # datos corruptos o tasa limitada: reintentar ya es inútil
        except Exception as exc:  # noqa: BLE001 - la red falla de mil formas
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_s * (2 ** attempt)
                logger.warning("Fallo obteniendo %s (intento %d/%d): %s. "
                               "Reintento en %.1fs", what, attempt + 1,
                               max_retries + 1, exc, wait)
                time.sleep(wait)
    raise DataProviderError(
        f"Agotados {max_retries + 1} intentos obteniendo {what}: {last_exc}"
    ) from last_exc


class DataProvider(abc.ABC):
    """Interfaz que debe implementar todo proveedor de datos de mercado."""

    def __init__(self, config: DataConfig) -> None:
        """
        Args:
            config: sección ``data`` de la configuración global.
        """
        self.config = config

    @abc.abstractmethod
    def fetch_historical(self, symbol: str) -> pd.DataFrame:
        """Descarga histórico OHLCV para entrenamiento/backtest.

        Args:
            symbol: ticker, p. ej. ``"AAPL"``.

        Returns:
            DataFrame canónico (ver :func:`validate_ohlcv`) que cubre
            ``config.lookback_days`` con granularidad ``config.interval``.

        Raises:
            DataProviderError / DataValidationError.
        """

    @abc.abstractmethod
    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Obtiene las velas más recientes para trading en (casi) tiempo real.

        Args:
            symbol: ticker.

        Returns:
            DataFrame canónico con al menos la última vela disponible.

        Raises:
            DataProviderError / DataValidationError.
        """


def make_provider(config: DataConfig) -> DataProvider:
    """Fábrica de proveedores a partir de la configuración.

    Args:
        config: sección ``data``; ``config.provider`` decide la clase.

    Returns:
        Instancia de :class:`DataProvider` lista para usar.

    Raises:
        ConfigurationError: si el proveedor no está soportado o su
            dependencia opcional (p. ej. ``blpapi``) no está instalada.
    """
    if config.provider == "yahoo":
        from .yahoo import YahooFinanceProvider
        return YahooFinanceProvider(config)
    if config.provider == "bloomberg":
        from .bloomberg import BloombergProvider
        return BloombergProvider(config)
    raise ConfigurationError(f"Proveedor no soportado: {config.provider!r}")
