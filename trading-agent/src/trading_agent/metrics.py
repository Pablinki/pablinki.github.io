"""Métricas de evaluación de estrategias de trading.

Funciones puras sobre la curva de equity y el historial de trades, usadas
por :meth:`TradingEnvironment.episode_summary` y por el entrenamiento para
comparar contra el benchmark de comprar-y-mantener. Sin dependencias del
entorno, así se testean de forma aislada.

Estructuras de datos
--------------------
- ``equity_curve``: secuencia de valores de equity (USD), una por vela,
  empezando por el capital inicial. Longitud ``n_pasos + 1``.
- ``trades``: lista de :class:`~trading_agent.portfolio.Trade`.

Ejemplo
-------
>>> max_drawdown([100.0, 120.0, 90.0, 110.0])
0.25
>>> sharpe_ratio([100.0, 100.0, 100.0])        # sin volatilidad
0.0
>>> sharpe_ratio([100.0, 101.0, 100.5, 102.0]) > 0
True
>>> profit_factor([])
0.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .portfolio import Trade

#: Velas por año según el intervalo (sesión de EE. UU.: 252 días, 6,5 h).
PERIODS_PER_YEAR = {"1d": 252, "1h": 252 * 7, "30m": 252 * 13,
                    "15m": 252 * 26, "5m": 252 * 78, "1m": 252 * 390}


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Máxima caída desde un pico histórico, como fracción del pico.

    Bucle de una pasada: mantiene el máximo acumulado y la peor caída
    relativa a él.

    Args:
        equity_curve: valores de equity por vela (todos > 0).

    Returns:
        Drawdown máximo en [0, 1]; 0.0 si la curva tiene < 2 puntos o
        nunca cae.
    """
    peak = float("-inf")
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = 1.0 - equity / peak if peak > 0 else 0.0
        worst = max(worst, drawdown)
    return worst


def sharpe_ratio(equity_curve: Sequence[float], *,
                 periods_per_year: int = 252,
                 risk_free_rate: float = 0.0) -> float:
    """Ratio de Sharpe anualizado de los retornos logarítmicos por vela.

    Args:
        equity_curve: valores de equity por vela (todos > 0).
        periods_per_year: velas por año para anualizar (252 para velas
            diarias; p. ej. 252*390 para velas de 1 minuto en EE. UU.).
        risk_free_rate: tasa libre de riesgo anual (fracción).

    Returns:
        Sharpe anualizado; 0.0 si hay < 3 puntos o volatilidad nula.
    """
    if len(equity_curve) < 3:
        return 0.0
    returns = [math.log(equity_curve[i] / equity_curve[i - 1])
               for i in range(1, len(equity_curve))]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    return (mean - rf_per_period) / std * math.sqrt(periods_per_year)


def profit_factor(trades: Sequence[Trade]) -> float:
    """Ganancias brutas / pérdidas brutas de las operaciones cerradas.

    Args:
        trades: historial de trades (solo cuentan las ventas, que llevan
            el P&L realizado).

    Returns:
        Profit factor (> 1 = rentable). 0.0 sin ventas; ``inf`` si hubo
        ganancias y ninguna pérdida.
    """
    gross_win = sum(t.pnl for t in trades if t.side == "sell" and t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.side == "sell" and t.pnl < 0)
    if gross_win == 0.0 and gross_loss == 0.0:
        return 0.0
    if gross_loss == 0.0:
        return float("inf")
    return gross_win / gross_loss
