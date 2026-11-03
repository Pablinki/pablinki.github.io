"""Entorno de trading estilo Gym para entrenar el DQN.

Formulación del MDP (Proceso de Decisión de Markov)
---------------------------------------------------
- **Estado** (``numpy.ndarray`` float32 de dimensión
  ``window_size * n_features + 2``): la ventana de las últimas
  ``window_size`` filas de features aplanada, más dos escalares de
  contexto de cartera:

  1. ``has_position``: 1.0 si hay posición abierta, 0.0 si no.
  2. ``unrealized_pnl_pct``: P&L no realizado de la posición (0 si no hay).

- **Acciones** (enteros discretos, ver :class:`Action`):

  ======  =========  ====================================================
  Valor   Nombre     Efecto
  ======  =========  ====================================================
  0       HOLD       No hacer nada.
  1       BUY        Abrir posición larga (si no hay una y el riesgo lo
                     permite); si ya hay posición equivale a HOLD.
  2       SELL       Cerrar la posición abierta; sin posición es HOLD.
  ======  =========  ====================================================

- **Recompensa** (float): variación logarítmica del equity entre velas,
  multiplicada por ``reward_scaling``::

      r_t = ln(equity_t / equity_{t-1}) * reward_scaling

  Este shaping alinea la recompensa acumulada con el log-retorno total
  del episodio: maximizar la suma de recompensas ES maximizar la ganancia
  compuesta, que es exactamente el objetivo del usuario.

- **Fin de episodio**: se agotan los datos o el RiskManager activa el
  kill-switch de drawdown.

Bucle de interacción típico
---------------------------
El bucle estándar de RL (lo usa ``train.py``)::

    state = env.reset()
    while True:
        action = agente.act(state)
        state, reward, done, info = env.step(action)
        if done:
            break

Ejemplo
-------
>>> import numpy as np, pandas as pd
>>> from trading_agent.config import EnvConfig, RiskConfig
>>> rng = np.random.default_rng(0)
>>> idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
>>> close = pd.Series(100 * np.exp(rng.normal(0, .01, 300).cumsum()), index=idx)
>>> ohlcv = pd.DataFrame({"open": close, "high": close * 1.01,
...                       "low": close * .99, "close": close,
...                       "volume": 1e6}, index=idx)
>>> env = TradingEnvironment(ohlcv, EnvConfig(window_size=10), RiskConfig())
>>> state = env.reset()
>>> state.shape == (env.state_dim,)
True
>>> state2, reward, done, info = env.step(Action.BUY)
>>> info["has_position"]
True
"""

from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import EnvConfig, RiskConfig
from .exceptions import EnvironmentError_
from .features import FEATURE_COLUMNS, build_features
from .metrics import max_drawdown, profit_factor, sharpe_ratio
from .portfolio import Portfolio
from .risk import RiskManager

logger = logging.getLogger(__name__)


class Action(enum.IntEnum):
    """Acciones discretas del agente (ver tabla en el docstring del módulo)."""

    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnvironment:
    """Simulador de mercado de un solo símbolo para entrenamiento y backtest.

    Attributes:
        state_dim: dimensión del vector de estado (para dimensionar la red).
        n_actions: nº de acciones válidas (3).
    """

    #: Nº de escalares de contexto de cartera añadidos al final del estado.
    N_PORTFOLIO_FEATURES = 2

    def __init__(self, ohlcv: pd.DataFrame, env_config: EnvConfig,
                 risk_config: RiskConfig, symbol: str = "ASSET",
                 features: pd.DataFrame | None = None,
                 periods_per_year: int = 252) -> None:
        """
        Args:
            ohlcv: DataFrame OHLCV canónico con histórico suficiente
                (ver ``features.build_features`` para el mínimo).
            env_config: reglas de simulación (ventana, comisiones...).
            risk_config: límites del gestor de riesgo.
            symbol: nombre del activo (solo informativo).
            features: matriz de features precalculada (índice ⊆ índice de
                ``ohlcv``). Permite calcular los indicadores UNA vez sobre
                el histórico completo y luego partir por tiempo, sin que
                cada tramo pierda su calentamiento ni arranque con EMAs
                transitorias. Los indicadores son causales (solo usan el
                pasado), así que no introduce fuga de información.
            periods_per_year: velas por año para anualizar el Sharpe (252
                para velas diarias; ver ``metrics.PERIODS_PER_YEAR``).

        Raises:
            DataValidationError: si el histórico es demasiado corto.
        """
        self.symbol = symbol
        self.env_config = env_config
        self.periods_per_year = periods_per_year
        self.risk = RiskManager(risk_config)

        if features is None:
            self._features = build_features(ohlcv)
        else:
            self._features = features.loc[features.index.isin(ohlcv.index),
                                          list(FEATURE_COLUMNS)]
        # OHLC alineado con las filas de features supervivientes. Los stops
        # se evalúan contra el rango completo de la vela (open/high/low),
        # no solo contra el cierre: en vivo se disparan con cualquier precio.
        aligned = ohlcv.loc[self._features.index]
        self._prices = aligned["close"].to_numpy()
        self._opens = aligned["open"].to_numpy()
        self._highs = aligned["high"].to_numpy()
        self._lows = aligned["low"].to_numpy()
        self._feat_matrix = self._features.to_numpy(dtype=np.float32)
        # Fechas de calendario por vela: arman el límite de pérdida diaria.
        self._dates = pd.DatetimeIndex(self._features.index).date

        n_steps = len(self._features) - env_config.window_size
        if n_steps < 2:
            raise EnvironmentError_(
                f"Histórico insuficiente: {len(self._features)} filas de "
                f"features para window_size={env_config.window_size}")

        self.state_dim = (env_config.window_size * len(FEATURE_COLUMNS)
                          + self.N_PORTFOLIO_FEATURES)
        self.n_actions = len(Action)

        # Estado mutable del episodio (se inicializa en reset()).
        self._t = 0
        self._portfolio: Portfolio | None = None
        self._done = True
        self._equity_curve: list[float] = []  # equity por vela (métricas)

    # ------------------------------------------------------------------ #
    # API estilo Gym                                                     #
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Comienza un episodio nuevo.

        Returns:
            El estado inicial (``numpy.ndarray`` float32, ``state_dim``).
        """
        self._t = self.env_config.window_size
        self._portfolio = Portfolio(
            cash=self.env_config.initial_cash,
            commission_pct=self.env_config.commission_pct,
            slippage_pct=self.env_config.slippage_pct,
        )
        self.risk.reset()
        self.risk.update_equity(self.env_config.initial_cash)
        self.risk.start_new_day(self.env_config.initial_cash)
        self._done = False
        self._equity_curve = [self.env_config.initial_cash]
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Ejecuta una acción y avanza una vela.

        Orden de eventos dentro del paso (importante para el realismo):

        1. Se ejecuta la acción del agente al precio de cierre actual.
        2. Se avanza el tiempo a la vela siguiente (si cambia el día de
           calendario se arma el límite de pérdida diaria).
        3. El RiskManager evalúa stop-loss/take-profit al precio nuevo y
           puede forzar un cierre.
        4. Si el episodio termina, se liquida la posición restante.
        5. Se calcula la recompensa como Δlog(equity), incluyendo los
           costes de la liquidación final si la hubo.

        Args:
            action: entero en ``{0, 1, 2}`` (ver :class:`Action`).

        Returns:
            Tupla ``(state, reward, done, info)``:
                - ``state``: siguiente estado (float32, ``state_dim``).
                - ``reward``: recompensa escalada del paso.
                - ``done``: ``True`` si el episodio terminó.
                - ``info``: dict de diagnóstico con ``equity``, ``cash``,
                  ``has_position``, ``price``, ``forced_close`` y ``t``.

        Raises:
            EnvironmentError_: si el episodio ya terminó o la acción es
                inválida.
        """
        if self._done or self._portfolio is None:
            raise EnvironmentError_("Episodio terminado: llama a reset()")
        try:
            action = Action(action)
        except ValueError as exc:
            raise EnvironmentError_(f"Acción inválida: {action!r}") from exc

        price = float(self._prices[self._t])
        equity_before = self._portfolio.equity({self.symbol: price})

        # (1) Ejecutar la decisión del agente.
        self._apply_action(action, price, equity_before)

        # (2) Avanzar el tiempo. Si cambia el día de calendario, se arma el
        # límite de pérdida diaria con el equity de apertura de la jornada.
        self._t += 1
        new_price = float(self._prices[self._t])
        if self._dates[self._t] != self._dates[self._t - 1]:
            # Equity al cierre anterior: así el hueco de apertura cuenta
            # dentro de la pérdida del día (con velas diarias, cada vela
            # es un día y el límite actúa como tope de pérdida por vela).
            self.risk.start_new_day(equity_before)

        # (3) Cierres forzosos por gestión de riesgo, evaluados INTRABARRA:
        # si el mínimo/máximo de la vela cruza el stop/take-profit, se
        # ejecuta al nivel del stop (o a la apertura si la vela abre con
        # hueco más allá), igual que ocurriría en vivo.
        self.risk.new_bar()
        forced_close = None
        if self._portfolio.has_position(self.symbol):
            reason, fill_price = self.risk.evaluate_bar(
                self._portfolio.positions[self.symbol],
                float(self._opens[self._t]), float(self._highs[self._t]),
                float(self._lows[self._t]), new_price)
            if reason is not None:
                self._portfolio.close_position(self.symbol, fill_price)
                self.risk.on_forced_close()
                forced_close = reason

        # (4) Fin de episodio y liquidación ANTES de calcular la recompensa:
        # así los costes de la liquidación final también quedan reflejados
        # en la señal de aprendizaje.
        self.risk.update_equity(
            self._portfolio.equity({self.symbol: new_price}))
        self._done = (self._t >= len(self._prices) - 1) or self.risk.halted
        if self._done and self._portfolio.has_position(self.symbol):
            # Liquidación final: el episodio se evalúa todo en efectivo.
            self._portfolio.close_position(self.symbol, new_price)

        # (5) Recompensa = Δ log-equity escalado.
        equity_after = self._portfolio.equity({self.symbol: new_price})
        reward = (np.log(equity_after / equity_before)
                  * self.env_config.reward_scaling)
        self._equity_curve.append(equity_after)

        info = {
            "t": self._t,
            "price": new_price,
            "equity": equity_after,
            "cash": self._portfolio.cash,
            "has_position": self._portfolio.has_position(self.symbol),
            "forced_close": forced_close,
            "halted": self.risk.halted,
            # Distinción RL estándar: ``terminated`` = fin real del MDP
            # (kill-switch); el agotamiento de datos es un truncamiento
            # artificial y NO debe anular el valor futuro en el objetivo Q.
            "terminated": self.risk.halted,
        }
        return self._observe(), float(reward), self._done, info

    # ------------------------------------------------------------------ #
    # Internos                                                           #
    # ------------------------------------------------------------------ #
    def _apply_action(self, action: Action, price: float, equity: float) -> None:
        """Traduce la acción del agente en órdenes sobre el portafolio.

        BUY sin fondos/veto de riesgo y SELL sin posición degradan a HOLD
        silenciosamente: son acciones legales que simplemente no tienen
        efecto (el agente aprende su inutilidad vía recompensa).
        """
        assert self._portfolio is not None
        if action == Action.BUY and not self._portfolio.has_position(self.symbol):
            if not self.risk.can_open(equity):
                return
            qty = self.risk.position_size(equity, price)
            # El coste real incluye slippage y comisión; recalcular con
            # margen para no rebotar en InsufficientFundsError.
            max_affordable = self._portfolio.cash / (
                price * (1.0 + self.env_config.slippage_pct)
                * (1.0 + self.env_config.commission_pct))
            qty = min(qty, float(int(max_affordable)))
            if qty > 0:
                self._portfolio.open_position(self.symbol, price, qty)
        elif action == Action.SELL and self._portfolio.has_position(self.symbol):
            self._portfolio.close_position(self.symbol, price)

    def _observe(self) -> np.ndarray:
        """Construye el vector de estado en el instante actual.

        Returns:
            ``numpy.ndarray`` float32 de tamaño ``state_dim``: ventana de
            features aplanada + [has_position, unrealized_pnl_pct].
        """
        assert self._portfolio is not None
        w = self.env_config.window_size
        # La ventana TERMINA en la vela actual t (inclusive): la acción se
        # ejecuta al cierre de t con toda la información de t. El bucle en
        # vivo construye el estado exactamente igual (últimas w velas
        # completadas), así entrenamiento y producción ven lo mismo.
        window = self._feat_matrix[self._t - w + 1:self._t + 1].reshape(-1)

        price = float(self._prices[self._t])
        if self._portfolio.has_position(self.symbol):
            pos = self._portfolio.positions[self.symbol]
            extras = np.array([1.0, pos.unrealized_pnl_pct(price)],
                              dtype=np.float32)
        else:
            extras = np.zeros(self.N_PORTFOLIO_FEATURES, dtype=np.float32)
        return np.concatenate([window, extras])

    # ------------------------------------------------------------------ #
    # Métricas                                                           #
    # ------------------------------------------------------------------ #
    def episode_summary(self) -> dict[str, float]:
        """Métricas del episodio en curso/terminado.

        Returns:
            Dict con:
                - ``final_equity``: equity final en USD.
                - ``total_return_pct``: retorno total (0.05 = +5%).
                - ``buy_and_hold_return_pct``: benchmark comprar-y-mantener
                  sobre el mismo tramo de precios (del primer paso al actual).
                - ``excess_return_pct``: ``total_return_pct`` menos el benchmark.
                - ``max_drawdown_pct``: peor caída desde un pico, en [0, 1].
                - ``sharpe``: ratio de Sharpe anualizado con
                  ``periods_per_year`` velas por año (252 en velas diarias).
                - ``profit_factor``: ganancias brutas / pérdidas brutas.
                - ``n_trades``: nº de operaciones (compras + ventas).
                - ``win_rate``: fracción de ventas con P&L > 0.
        """
        if self._portfolio is None:
            raise EnvironmentError_("No hay episodio: llama a reset()")
        sells = [t for t in self._portfolio.trades if t.side == "sell"]
        wins = sum(1 for t in sells if t.pnl > 0)
        final_price = float(self._prices[self._t])
        first_price = float(self._prices[self.env_config.window_size])
        equity = self._portfolio.equity({self.symbol: final_price})
        total_return = equity / self.env_config.initial_cash - 1.0
        buy_and_hold = final_price / first_price - 1.0
        return {
            "final_equity": equity,
            "total_return_pct": total_return,
            "buy_and_hold_return_pct": buy_and_hold,
            "excess_return_pct": total_return - buy_and_hold,
            "max_drawdown_pct": max_drawdown(self._equity_curve),
            "sharpe": sharpe_ratio(self._equity_curve,
                                   periods_per_year=self.periods_per_year),
            "profit_factor": profit_factor(self._portfolio.trades),
            "n_trades": float(len(self._portfolio.trades)),
            "win_rate": wins / len(sells) if sells else 0.0,
        }
