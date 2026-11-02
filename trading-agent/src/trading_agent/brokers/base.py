"""Interfaz de ejecución de órdenes: adaptadores de broker.

Patrón idéntico al de la capa de datos (``data/provider.py``): el
``LiveTrader`` solo conoce la interfaz :class:`BrokerAdapter`; cambiar de
simulación a broker real es cambiar un argumento, no el bucle de trading.

Dos implementaciones incluidas:

- :class:`PaperBroker` (defecto): delega en el ``Portfolio`` local, que
  simula comisiones y slippage. Comportamiento histórico del proyecto.
- :class:`~trading_agent.brokers.alpaca.AlpacaBroker`: ejecuta órdenes
  de mercado reales contra la API de Alpaca y refleja cada fill en el
  ``Portfolio`` local (espejo contable para el RiskManager y el estado).

Contrato común
--------------
Ambos métodos devuelven el :class:`~trading_agent.portfolio.Trade`
registrado en el portafolio local, de modo que métricas, logs y gestión
de riesgo son idénticos en simulado y en real.
"""

from __future__ import annotations

import abc

from ..portfolio import Portfolio, Trade


class BrokerAdapter(abc.ABC):
    """Interfaz que debe implementar todo ejecutor de órdenes.

    Attributes:
        portfolio: contabilidad local (fuente de verdad para el agente).
    """

    def __init__(self, portfolio: Portfolio) -> None:
        """
        Args:
            portfolio: portafolio local donde se refleja cada operación.
        """
        self.portfolio = portfolio

    @abc.abstractmethod
    def open_position(self, symbol: str, quantity: float,
                      price_hint: float, *, tag: str | None = None) -> Trade:
        """Compra ``quantity`` acciones de ``symbol`` a mercado.

        Args:
            symbol: ticker a comprar.
            quantity: nº de acciones (> 0, entero en brokers reales).
            price_hint: último precio conocido (el simulador ejecuta a
                este precio; un broker real lo usa solo como referencia
                de logging/validación — el fill manda).
            tag: identificador de la decisión (p. ej. la marca temporal de
                la vela) que un broker real usa como id de orden
                idempotente; el simulador lo ignora.

        Returns:
            El ``Trade`` de compra reflejado en el portafolio local.

        Raises:
            ExecutionError / InsufficientFundsError / SecurityError.
        """

    def is_market_open(self) -> bool:
        """¿Acepta el mercado órdenes ahora mismo?

        El simulador siempre responde ``True``; un broker real consulta el
        reloj de mercado para no enviar órdenes que no pueden ejecutarse
        (quedarían colgadas hasta el timeout y se cancelarían).

        Returns:
            ``True`` si se pueden enviar órdenes ejecutables.
        """
        return True

    @abc.abstractmethod
    def close_position(self, symbol: str, price_hint: float, *,
                       tag: str | None = None) -> Trade:
        """Cierra por completo la posición abierta en ``symbol`` a mercado.

        Args:
            symbol: ticker a vender.
            price_hint: último precio conocido (ver ``open_position``).
            tag: identificador de la decisión (ver ``open_position``).

        Returns:
            El ``Trade`` de venta, con el P&L realizado.

        Raises:
            ExecutionError si no hay posición o la orden falla.
        """


class PaperBroker(BrokerAdapter):
    """Ejecución simulada: delega en el ``Portfolio`` local.

    El portafolio aplica su propia comisión y slippage configurados, que
    es exactamente el comportamiento de backtest — paper trading y
    backtest comparten contabilidad.

    Ejemplo:
        >>> from trading_agent.portfolio import Portfolio
        >>> broker = PaperBroker(Portfolio(cash=1000.0))
        >>> trade = broker.open_position("AAPL", quantity=2, price_hint=100.0)
        >>> broker.portfolio.cash
        800.0
    """

    def open_position(self, symbol: str, quantity: float,
                      price_hint: float, *, tag: str | None = None) -> Trade:
        """Ver contrato en :meth:`BrokerAdapter.open_position`."""
        return self.portfolio.open_position(symbol, price_hint, quantity)

    def close_position(self, symbol: str, price_hint: float, *,
                       tag: str | None = None) -> Trade:
        """Ver contrato en :meth:`BrokerAdapter.close_position`."""
        return self.portfolio.close_position(symbol, price_hint)
