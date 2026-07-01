"""Gestor de riesgo: la capa que protege el capital de las decisiones del DQN.

Un agente de RL optimiza recompensa, pero puede equivocarse (y se
equivoca). El :class:`RiskManager` es determinista y se interpone entre
la política del agente y la ejecución:

1. **Antes de abrir**: :meth:`position_size` limita el tamaño de la orden
   y :meth:`can_open` puede vetarla (drawdown, pérdida diaria).
2. **En cada vela**: :meth:`should_force_close` fuerza el cierre por
   stop-loss / take-profit aunque el agente quiera mantener.

Es la misma clase en backtest y en vivo — los límites que se testean son
los límites que protegen dinero real.

Ejemplo
-------
>>> from trading_agent.config import RiskConfig
>>> rm = RiskManager(RiskConfig(max_position_pct=0.5, stop_loss_pct=0.05))
>>> rm.position_size(equity=10_000.0, price=100.0)
50.0
>>> from trading_agent.portfolio import Position
>>> pos = Position("AAPL", 50.0, 100.0)
>>> rm.should_force_close(pos, price=94.0)   # -6% < stop de -5%
'stop_loss'
>>> rm.should_force_close(pos, price=101.0) is None
True
"""

from __future__ import annotations

import logging
import math

from .config import RiskConfig
from .portfolio import Position

logger = logging.getLogger(__name__)


class RiskManager:
    """Aplica los límites de :class:`~trading_agent.config.RiskConfig`.

    Estado interno (se actualiza vía :meth:`update_equity`):
        - ``peak_equity``: máximo histórico de equity (para drawdown).
        - ``day_start_equity``: equity al inicio del día (pérdida diaria).
        - ``halted``: ``True`` si se superó el drawdown máximo; el trading
          queda detenido hasta intervención humana (:meth:`reset`).
    """

    def __init__(self, config: RiskConfig) -> None:
        """
        Args:
            config: límites de riesgo validados.
        """
        self.config = config
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.halted = False

    # ------------------------------------------------------------------ #
    # Ciclo de vida                                                      #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reinicia el estado (nuevo episodio de backtest o rearme manual)."""
        self.peak_equity = None
        self.day_start_equity = None
        self.halted = False

    def start_new_day(self, equity: float) -> None:
        """Marca el inicio de una jornada para el límite de pérdida diaria.

        Args:
            equity: equity al comenzar el día, en USD.
        """
        self.day_start_equity = equity

    def update_equity(self, equity: float) -> None:
        """Actualiza el pico de equity y evalúa el kill-switch de drawdown.

        Debe llamarse en cada vela con el equity valorado a mercado.

        Args:
            equity: equity actual en USD.
        """
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 1.0 - equity / self.peak_equity
        if drawdown > self.config.max_drawdown_pct and not self.halted:
            self.halted = True
            logger.critical(
                "KILL-SWITCH: drawdown %.1f%% supera el máximo %.1f%%. "
                "Trading detenido.", drawdown * 100,
                self.config.max_drawdown_pct * 100)

    # ------------------------------------------------------------------ #
    # Decisiones                                                         #
    # ------------------------------------------------------------------ #
    def can_open(self, equity: float) -> bool:
        """¿Se permite abrir una posición nueva ahora mismo?

        Args:
            equity: equity actual en USD.

        Returns:
            ``False`` si el kill-switch está activo o si la pérdida del
            día supera ``max_daily_loss_pct``; ``True`` en caso contrario.
        """
        if self.halted:
            return False
        if self.day_start_equity is not None:
            daily_loss = 1.0 - equity / self.day_start_equity
            if daily_loss > self.config.max_daily_loss_pct:
                logger.warning("Pérdida diaria %.2f%% > límite %.2f%%: no se "
                               "abren posiciones nuevas hoy.",
                               daily_loss * 100,
                               self.config.max_daily_loss_pct * 100)
                return False
        return True

    def position_size(self, equity: float, price: float) -> float:
        """Calcula cuántas acciones comprar respetando ``max_position_pct``.

        Dimensionamiento *fixed-fraction*: se invierte como máximo una
        fracción fija del equity en cada posición. Redondea hacia abajo a
        acciones enteras.

        Args:
            equity: equity actual en USD.
            price: precio actual del símbolo (> 0).

        Returns:
            Nº de acciones (float con valor entero, p. ej. ``50.0``).
            Puede ser ``0.0`` si el presupuesto no alcanza ni para una.
        """
        if price <= 0 or equity <= 0:
            return 0.0
        budget = equity * self.config.max_position_pct
        return float(math.floor(budget / price))

    def should_force_close(self, position: Position, price: float) -> str | None:
        """Evalúa stop-loss y take-profit sobre una posición abierta.

        Args:
            position: posición abierta a evaluar.
            price: precio actual del símbolo.

        Returns:
            ``"stop_loss"`` si la pérdida supera ``stop_loss_pct``;
            ``"take_profit"`` si la ganancia supera ``take_profit_pct``;
            ``None`` si no procede cierre forzoso.
        """
        pnl_pct = position.unrealized_pnl_pct(price)
        if pnl_pct <= -self.config.stop_loss_pct:
            return "stop_loss"
        if (self.config.take_profit_pct is not None
                and pnl_pct >= self.config.take_profit_pct):
            return "take_profit"
        return None
