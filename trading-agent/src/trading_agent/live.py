"""Bucle de trading en (casi) tiempo real con un modelo DQN entrenado.

Modos de operación
------------------
- **Paper trading** (defecto): las órdenes se simulan sobre un
  :class:`~trading_agent.portfolio.Portfolio` local con los precios
  reales del proveedor. Cero riesgo; ideal para validar la estrategia.
- **Live**: requiere ``TRADING_AGENT_LIVE=YES`` en el entorno (ver
  ``security.py``) y un adaptador de broker que implemente la ejecución
  real (fuera del alcance de este módulo; el punto de integración es
  el mismo ``Portfolio``).

Robustez
--------
El bucle NUNCA muere por un fallo transitorio de datos: cada iteración
está envuelta en try/except; los ``DataProviderError`` se registran y se
espera al siguiente ciclo. Solo ``RiskLimitExceededError`` (kill-switch)
o una señal del sistema detienen el bucle — y al detenerse cierra
cualquier posición abierta.

Ejemplo (paper trading)::

    python scripts/run_live.py --config config.yaml \
        --checkpoint checkpoints/dqn_best.pt --poll-seconds 60
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import time
from types import FrameType

import numpy as np

from .agent.dqn import DQNAgent
from .config import AppConfig
from .data.provider import DataProvider, make_provider
from .env import Action
from .exceptions import (DataProviderError, ExecutionError,
                         TradingAgentError)
from .features import FEATURE_COLUMNS, build_features
from .portfolio import Portfolio
from .risk import RiskManager

logger = logging.getLogger(__name__)


class LiveTrader:
    """Orquestador del trading en tiempo real (paper por defecto).

    Estructura de una iteración (:meth:`run_once`):

    1. Pedir al proveedor las últimas velas (con reintentos internos).
    2. Construir features y el vector de estado (idéntico al de
       entrenamiento: misma función, mismas columnas, mismo orden).
    3. El RiskManager evalúa cierres forzosos (stop-loss/take-profit).
    4. El agente decide (política greedy) y se ejecuta la orden si el
       riesgo lo permite.
    """

    def __init__(self, config: AppConfig, agent: DQNAgent,
                 provider: DataProvider | None = None) -> None:
        """
        Args:
            config: configuración de la aplicación.
            agent: agente DQN con checkpoint cargado (``agent.load``).
            provider: proveedor de datos; si es ``None`` se construye
                desde la config (inyectable para tests).
        """
        self.config = config
        self.agent = agent
        self.provider = provider or make_provider(config.data)
        self.symbol = config.data.symbols[0]
        self.risk = RiskManager(config.risk)
        self.portfolio = Portfolio(
            cash=config.env.initial_cash,
            commission_pct=config.env.commission_pct,
            slippage_pct=config.env.slippage_pct,
        )
        self._stop_requested = False

    # ------------------------------------------------------------------ #
    # Señales del sistema (Ctrl+C / SIGTERM del orquestador)             #
    # ------------------------------------------------------------------ #
    def install_signal_handlers(self) -> None:
        """Instala manejadores para parar limpiamente con SIGINT/SIGTERM."""
        def _handler(signum: int, _frame: FrameType | None) -> None:
            logger.info("Señal %s recibida: parada limpia solicitada", signum)
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # ------------------------------------------------------------------ #
    # Núcleo                                                             #
    # ------------------------------------------------------------------ #
    def _build_state(self, feats: np.ndarray, price: float) -> np.ndarray:
        """Replica el vector de estado del entorno de entrenamiento.

        Args:
            feats: matriz de features float32 ``(>=window_size, n_features)``.
            price: último precio para valorar el P&L de la posición.

        Returns:
            Vector de estado float32 de la misma dimensión que en
            entrenamiento.

        Raises:
            TradingAgentError: si no hay filas suficientes para la ventana.
        """
        w = self.config.env.window_size
        if len(feats) < w:
            raise TradingAgentError(
                f"Solo hay {len(feats)} filas de features; se necesitan {w}")
        window = feats[-w:].astype(np.float32).reshape(-1)
        if self.portfolio.has_position(self.symbol):
            pos = self.portfolio.positions[self.symbol]
            extras = np.array([1.0, pos.unrealized_pnl_pct(price)],
                              dtype=np.float32)
        else:
            extras = np.zeros(2, dtype=np.float32)
        return np.concatenate([window, extras])

    def run_once(self) -> dict[str, object]:
        """Ejecuta UNA iteración de decisión/ejecución.

        Returns:
            Dict de diagnóstico: ``price``, ``action`` (nombre), ``equity``,
            ``has_position`` y ``forced_close``.

        Raises:
            DataProviderError / DataValidationError: si los datos fallan
                (el llamador decide si continuar).
        """
        latest = self.provider.fetch_latest(self.symbol)
        feats_df = build_features(latest)
        feats = feats_df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        price = float(latest.loc[feats_df.index[-1], "close"])

        equity = self.portfolio.equity({self.symbol: price})
        self.risk.update_equity(equity)

        # (3) Gestión de riesgo primero: los stops mandan sobre el agente.
        forced_close = None
        if self.portfolio.has_position(self.symbol):
            reason = self.risk.should_force_close(
                self.portfolio.positions[self.symbol], price)
            if reason is not None:
                trade = self.portfolio.close_position(self.symbol, price)
                forced_close = reason
                logger.warning("%s: cierre forzoso por %s, P&L %.2f USD",
                               self.symbol, reason, trade.pnl)

        # (4) Decisión del agente (siempre greedy en producción).
        state = self._build_state(feats, price)
        action = Action(self.agent.act(state, greedy=True))

        if action == Action.BUY and not self.portfolio.has_position(self.symbol):
            if self.risk.can_open(equity):
                qty = self.risk.position_size(equity, price)
                max_affordable = self.portfolio.cash / (
                    price * (1.0 + self.config.env.slippage_pct)
                    * (1.0 + self.config.env.commission_pct))
                qty = min(qty, float(int(max_affordable)))
                if qty > 0:
                    self.portfolio.open_position(self.symbol, price, qty)
                    logger.info("%s: COMPRA %g @ %.2f", self.symbol, qty, price)
        elif action == Action.SELL and self.portfolio.has_position(self.symbol):
            trade = self.portfolio.close_position(self.symbol, price)
            logger.info("%s: VENTA %g @ %.2f, P&L %.2f USD",
                        self.symbol, trade.quantity, price, trade.pnl)

        equity = self.portfolio.equity({self.symbol: price})
        return {
            "price": price,
            "action": action.name,
            "equity": equity,
            "has_position": self.portfolio.has_position(self.symbol),
            "forced_close": forced_close,
        }

    def run_forever(self, poll_seconds: float = 60.0,
                    max_iterations: int | None = None) -> None:
        """Bucle principal de trading: decide cada ``poll_seconds``.

        El bucle es resiliente: los fallos de datos se registran y se
        continúa; el kill-switch de riesgo o una señal del sistema lo
        detienen, liquidando la posición abierta si la hay.

        Args:
            poll_seconds: segundos entre iteraciones (>= 1; no tiene
                sentido consultar más rápido que la granularidad de velas).
            max_iterations: tope de iteraciones (``None`` = infinito;
                útil en tests y sesiones acotadas).
        """
        if poll_seconds < 1:
            raise TradingAgentError("poll_seconds debe ser >= 1")
        self.install_signal_handlers()
        logger.info("Bucle en vivo iniciado (paper trading) para %s cada %.0fs",
                    self.symbol, poll_seconds)

        iteration = 0
        while not self._stop_requested:
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1
            try:
                info = self.run_once()
                logger.info("it %d | %s | precio %.2f | equity %.2f | %s",
                            iteration, info["action"], info["price"],
                            info["equity"],
                            "EN POSICIÓN" if info["has_position"] else "en efectivo")
                if self.risk.halted:
                    logger.critical("Kill-switch de drawdown activo: fin del bucle")
                    break
            except DataProviderError as exc:
                # Transitorio: registrar y esperar al siguiente ciclo.
                logger.error("Fallo de datos (se reintenta en el próximo "
                             "ciclo): %s", exc)
            except ExecutionError as exc:
                logger.error("Fallo de ejecución: %s", exc)
            time.sleep(poll_seconds)

        self._liquidate()
        logger.info("Bucle en vivo terminado tras %d iteraciones", iteration)

    def _liquidate(self) -> None:
        """Cierra la posición abierta (si existe) al detener el bucle."""
        if not self.portfolio.has_position(self.symbol):
            return
        try:
            latest = self.provider.fetch_latest(self.symbol)
            price = float(latest["close"].iloc[-1])
            trade = self.portfolio.close_position(self.symbol, price)
            logger.info("Liquidación final: %s P&L %.2f USD", self.symbol, trade.pnl)
        except TradingAgentError as exc:
            logger.error("No se pudo liquidar %s automáticamente: %s. "
                         "REVISAR MANUALMENTE.", self.symbol, exc)


def build_live_trader(config: AppConfig, checkpoint_path: str) -> LiveTrader:
    """Construye un :class:`LiveTrader` con el checkpoint indicado.

    Args:
        config: configuración de la aplicación.
        checkpoint_path: ruta del modelo entrenado (``.pt``).

    Returns:
        ``LiveTrader`` listo para :meth:`LiveTrader.run_forever`.

    Nota:
        El ``state_dim`` se deriva de la config (ventana × features + 2),
        igual que en entrenamiento, y :meth:`DQNAgent.load` verifica que
        el checkpoint coincida.
    """
    state_dim = (config.env.window_size * len(FEATURE_COLUMNS) + 2)
    agent_cfg = dataclasses.replace(config.agent, state_dim=state_dim)
    agent = DQNAgent(agent_cfg)
    agent.load(checkpoint_path)
    return LiveTrader(config, agent)
