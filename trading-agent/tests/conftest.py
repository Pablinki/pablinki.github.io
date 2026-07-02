"""Fixtures compartidas de la suite de tests.

Los tests NUNCA tocan la red: los datos de mercado se generan
sintéticamente (paseo aleatorio geométrico con semilla fija), lo que hace
la suite rápida, determinista y ejecutable en CI sin credenciales.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Hace importable el paquete sin instalarlo (modo desarrollo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def make_ohlcv(n: int = 400, seed: int = 7, drift: float = 0.0005,
               vol: float = 0.01) -> pd.DataFrame:
    """Genera un OHLCV sintético canónico (paseo aleatorio geométrico).

    Args:
        n: nº de velas diarias.
        seed: semilla del generador.
        drift: deriva diaria del log-precio.
        vol: volatilidad diaria del log-precio.

    Returns:
        DataFrame OHLCV válido según ``validate_ohlcv``.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.uniform(1e5, 1e7, n),
    }, index=idx)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """OHLCV sintético de 400 velas diarias."""
    return make_ohlcv()


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """OHLCV corto (120 velas) para tests rápidos de entorno/entrenamiento."""
    return make_ohlcv(n=120)
