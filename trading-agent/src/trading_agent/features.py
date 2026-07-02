"""Ingeniería de características (indicadores técnicos) para el estado del DQN.

El agente no ve precios crudos: ve un vector de *features* normalizadas y
estacionarias (retornos, osciladores acotados) — las redes neuronales
aprenden mal de series no estacionarias como el precio absoluto.

Entrada / salida del módulo
---------------------------
:func:`build_features` recibe el DataFrame OHLCV canónico (ver
``data/provider.py``) y devuelve un DataFrame de features alineado por
índice temporal, sin NaN (las primeras filas con ventanas incompletas se
descartan).

Features generadas (todas float64):

==============  ======================================================
Columna         Descripción
==============  ======================================================
``log_ret``     Retorno logarítmico de 1 periodo: ``ln(P_t / P_{t-1})``
``rsi``         RSI de ``rsi_period`` velas, reescalado a [-1, 1]
``macd``        Diferencia EMA rápida - EMA lenta, relativa al precio
``macd_signal`` EMA de ``signal_period`` del MACD relativo
``bb_pos``      Posición dentro de las bandas de Bollinger, en [-1, 1]
``vol``         Volatilidad realizada (std de log_ret, ``vol_period``)
``volume_z``    Z-score del volumen sobre ``vol_period`` velas
==============  ======================================================

Ejemplo
-------
>>> import numpy as np, pandas as pd
>>> idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
>>> close = pd.Series(100 * np.exp(np.linspace(0, 0.1, 100)), index=idx)
>>> df = pd.DataFrame({"open": close, "high": close * 1.01,
...                    "low": close * 0.99, "close": close,
...                    "volume": 1e6}, index=idx)
>>> feats = build_features(df)
>>> sorted(feats.columns)
['bb_pos', 'log_ret', 'macd', 'macd_signal', 'rsi', 'vol', 'volume_z']
>>> bool(feats.isna().any().any())
False
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

#: Nombres de las columnas de features, en orden estable (el orden define
#: la posición de cada feature dentro del vector de estado del DQN).
FEATURE_COLUMNS = ("log_ret", "rsi", "macd", "macd_signal",
                   "bb_pos", "vol", "volume_z")


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index clásico (media móvil de Wilder).

    Args:
        close: serie de precios de cierre.
        period: nº de velas de la media (típicamente 14).

    Returns:
        Serie RSI en [0, 100] (NaN durante el calentamiento inicial).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # ewm con alpha=1/period reproduce el suavizado de Wilder.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Si no hubo pérdidas (avg_loss == 0) el RSI es 100 por definición.
    return rsi.fillna(100.0).where(avg_gain.notna(), np.nan)


def build_features(df: pd.DataFrame, *, rsi_period: int = 14,
                   macd_fast: int = 12, macd_slow: int = 26,
                   signal_period: int = 9, bb_period: int = 20,
                   vol_period: int = 20) -> pd.DataFrame:
    """Construye la matriz de features a partir de un OHLCV canónico.

    Args:
        df: DataFrame OHLCV canónico (columnas ``open..volume``, índice
            datetime UTC ascendente).
        rsi_period: ventana del RSI.
        macd_fast: periodo de la EMA rápida del MACD.
        macd_slow: periodo de la EMA lenta del MACD (> ``macd_fast``).
        signal_period: periodo de la señal del MACD.
        bb_period: ventana de las bandas de Bollinger.
        vol_period: ventana de volatilidad y z-score de volumen.

    Returns:
        DataFrame con las columnas :data:`FEATURE_COLUMNS`, sin NaN.
        Tiene menos filas que ``df`` (se pierde el calentamiento inicial).

    Raises:
        DataValidationError: si ``df`` es demasiado corto para calcular
            los indicadores, o si los parámetros son incoherentes.
    """
    if macd_slow <= macd_fast:
        raise DataValidationError("macd_slow debe ser > macd_fast")
    warmup = max(rsi_period, macd_slow + signal_period, bb_period, vol_period) + 1
    if len(df) <= warmup:
        raise DataValidationError(
            f"Se necesitan > {warmup} velas para las features; hay {len(df)}")

    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Retorno logarítmico: estacionario y aditivo en el tiempo.
    out["log_ret"] = np.log(close / close.shift(1))

    # RSI reescalado de [0, 100] a [-1, 1] para centrarlo en 0.
    out["rsi"] = _rsi(close, rsi_period) / 50.0 - 1.0

    # MACD relativo al precio (adimensional; comparable entre símbolos).
    ema_fast = close.ewm(span=macd_fast, min_periods=macd_fast).mean()
    ema_slow = close.ewm(span=macd_slow, min_periods=macd_slow).mean()
    macd = (ema_fast - ema_slow) / close
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=signal_period,
                                  min_periods=signal_period).mean()

    # Posición dentro de las bandas de Bollinger: -1 = banda inferior,
    # +1 = banda superior, 0 = media móvil.
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    out["bb_pos"] = ((close - sma) / (2.0 * std)).clip(-3.0, 3.0)

    # Volatilidad realizada y z-score de volumen.
    out["vol"] = out["log_ret"].rolling(vol_period).std()
    vol_mean = df["volume"].rolling(vol_period).mean()
    vol_std = df["volume"].rolling(vol_period).std()
    out["volume_z"] = ((df["volume"] - vol_mean)
                       / vol_std.replace(0.0, np.nan)).clip(-3.0, 3.0).fillna(0.0)

    out = out.loc[:, list(FEATURE_COLUMNS)].dropna()
    if out.empty:
        raise DataValidationError("La matriz de features quedó vacía")
    return out
