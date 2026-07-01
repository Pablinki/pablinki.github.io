"""Proveedor de datos basado en Yahoo Finance (paquete ``yfinance``).

Yahoo Finance es gratuito y no requiere credenciales, por lo que es el
proveedor por defecto para desarrollo, backtesting y paper trading.
Limitaciones conocidas:

- Los datos intradía (``1m``) solo cubren ~7 días hacia atrás.
- "Tiempo real" es en realidad *casi* tiempo real (retraso de ~1 min),
  suficiente para estrategias de baja frecuencia como este DQN.

Incluye un caché en disco (parquet) para no golpear la API en cada
corrida de entrenamiento.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from ..exceptions import DataProviderError
from .provider import DataProvider, validate_ohlcv, with_retries

logger = logging.getLogger(__name__)


class YahooFinanceProvider(DataProvider):
    """Implementación de :class:`DataProvider` sobre ``yfinance``."""

    def _cache_path(self, symbol: str) -> Path | None:
        """Ruta del archivo de caché para un símbolo, o ``None`` si el
        caché está desactivado en la configuración."""
        if self.config.cache_dir is None:
            return None
        d = Path(self.config.cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        # La fecha en el nombre invalida el caché automáticamente cada día.
        today = dt.date.today().isoformat()
        return d / f"{symbol}_{self.config.interval}_{today}.parquet"

    def _download(self, symbol: str, *, period: str) -> pd.DataFrame:
        """Descarga cruda desde yfinance y normaliza al formato canónico.

        Args:
            symbol: ticker.
            period: periodo yfinance (``"730d"``, ``"1d"``, ``"max"``...).

        Returns:
            DataFrame OHLCV canónico validado.
        """
        import yfinance as yf  # import perezoso: facilita testear sin red

        raw = yf.download(
            tickers=symbol,
            period=period,
            interval=self.config.interval,
            auto_adjust=True,     # precios ajustados por splits/dividendos
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise DataProviderError(f"yfinance devolvió vacío para {symbol!r}")
        # yfinance puede devolver columnas MultiIndex (precio, ticker).
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return validate_ohlcv(raw, symbol)

    def fetch_historical(self, symbol: str) -> pd.DataFrame:
        """Descarga ``config.lookback_days`` de histórico (con caché y
        reintentos). Ver contrato en :meth:`DataProvider.fetch_historical`."""
        cache = self._cache_path(symbol)
        if cache is not None and cache.exists():
            logger.info("Cargando %s desde caché %s", symbol, cache)
            return pd.read_parquet(cache)

        period = f"{self.config.lookback_days}d"
        df = with_retries(
            lambda: self._download(symbol, period=period),
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            what=f"histórico {symbol}",
        )
        if cache is not None:
            df.to_parquet(cache)
        logger.info("Histórico %s: %d velas [%s .. %s]", symbol, len(df),
                    df.index[0], df.index[-1])
        return df

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """Últimas velas del día en curso (sin caché: siempre frescas)."""
        return with_retries(
            lambda: self._download(symbol, period="5d"),
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            what=f"último precio {symbol}",
        )
