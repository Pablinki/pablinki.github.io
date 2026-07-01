"""Modelo de portafolio: posiciones, efectivo y contabilidad de operaciones.

Estructuras de datos
--------------------
- :class:`Position`: una posición larga abierta en un símbolo (cantidad,
  precio medio de entrada). Inmutable en su identidad; el P&L se deriva.
- :class:`Trade`: registro contable inmutable de una operación ejecutada
  (para auditoría y métricas).
- :class:`Portfolio`: efectivo + posiciones, con las operaciones
  ``open_position`` / ``close_position`` que aplican comisión y slippage.

El portafolio es *agnóstico del broker*: en backtest lo mueve el entorno
de simulación y en vivo lo movería el adaptador de ejecución, pero la
contabilidad (y por tanto las métricas y los tests) es idéntica.

Ejemplo
-------
>>> p = Portfolio(cash=10_000.0, commission_pct=0.0, slippage_pct=0.0)
>>> trade = p.open_position("AAPL", price=100.0, quantity=10)
>>> p.cash
9000.0
>>> p.equity({"AAPL": 110.0})
10100.0
>>> trade = p.close_position("AAPL", price=110.0)
>>> round(trade.pnl, 2)
100.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import ExecutionError, InsufficientFundsError


@dataclass(frozen=True)
class Trade:
    """Registro inmutable de una operación ejecutada.

    Attributes:
        symbol: ticker operado.
        side: ``"buy"`` o ``"sell"``.
        quantity: nº de acciones (> 0).
        price: precio de ejecución efectivo (ya incluye slippage).
        commission: comisión pagada en USD.
        pnl: ganancia/pérdida realizada en USD (solo en ventas; 0 en compras).
    """

    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0


@dataclass
class Position:
    """Posición larga abierta.

    Attributes:
        symbol: ticker.
        quantity: nº de acciones en cartera (> 0).
        entry_price: precio medio de entrada efectivo (con slippage).
    """

    symbol: str
    quantity: float
    entry_price: float

    def unrealized_pnl(self, price: float) -> float:
        """P&L no realizado a un precio de mercado dado.

        Args:
            price: precio actual del símbolo.

        Returns:
            ``(price - entry_price) * quantity`` en USD.
        """
        return (price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, price: float) -> float:
        """P&L no realizado como fracción del coste de entrada.

        Args:
            price: precio actual.

        Returns:
            P&L relativo, p. ej. ``0.05`` = +5%.
        """
        return (price - self.entry_price) / self.entry_price


@dataclass
class Portfolio:
    """Efectivo + posiciones abiertas + historial de trades.

    Attributes:
        cash: efectivo disponible en USD.
        commission_pct: comisión por operación (fracción del nocional).
        slippage_pct: deslizamiento adverso (fracción del precio).
        positions: mapa símbolo -> :class:`Position` abierta.
        trades: historial de :class:`Trade` (auditoría).
    """

    cash: float
    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Consultas                                                          #
    # ------------------------------------------------------------------ #
    def has_position(self, symbol: str) -> bool:
        """``True`` si hay posición abierta en ``symbol``."""
        return symbol in self.positions

    def equity(self, prices: dict[str, float]) -> float:
        """Valor total del portafolio (efectivo + posiciones a mercado).

        Args:
            prices: mapa símbolo -> precio actual. Debe incluir todos los
                símbolos con posición abierta.

        Returns:
            Equity en USD.

        Raises:
            ExecutionError: si falta el precio de una posición abierta
                (valorar con precios viejos escondería riesgo).
        """
        total = self.cash
        for symbol, pos in self.positions.items():
            if symbol not in prices:
                raise ExecutionError(f"Falta precio de {symbol} para valorar equity")
            total += pos.quantity * prices[symbol]
        return total

    # ------------------------------------------------------------------ #
    # Operaciones                                                        #
    # ------------------------------------------------------------------ #
    def open_position(self, symbol: str, price: float, quantity: float) -> Trade:
        """Abre una posición larga comprando ``quantity`` acciones.

        Aplica slippage adverso (compra a ``price * (1 + slippage_pct)``)
        y comisión sobre el nocional.

        Args:
            symbol: ticker a comprar.
            price: precio de mercado de referencia (> 0).
            quantity: nº de acciones (> 0).

        Returns:
            El :class:`Trade` de compra registrado.

        Raises:
            ExecutionError: si ya hay posición abierta en ``symbol`` (este
                modelo mantiene como máximo una posición por símbolo) o si
                los argumentos son inválidos.
            InsufficientFundsError: si el coste total supera el efectivo.
        """
        if price <= 0 or quantity <= 0:
            raise ExecutionError(
                f"Orden inválida: price={price}, quantity={quantity}")
        if self.has_position(symbol):
            raise ExecutionError(f"Ya existe posición abierta en {symbol}")

        exec_price = price * (1.0 + self.slippage_pct)
        notional = exec_price * quantity
        commission = notional * self.commission_pct
        cost = notional + commission
        if cost > self.cash:
            raise InsufficientFundsError(
                f"Coste {cost:.2f} > efectivo {self.cash:.2f} para {symbol}")

        self.cash -= cost
        self.positions[symbol] = Position(symbol, quantity, exec_price)
        trade = Trade(symbol, "buy", quantity, exec_price, commission)
        self.trades.append(trade)
        return trade

    def close_position(self, symbol: str, price: float) -> Trade:
        """Cierra por completo la posición en ``symbol`` vendiendo a mercado.

        Aplica slippage adverso (vende a ``price * (1 - slippage_pct)``)
        y comisión sobre el nocional.

        Args:
            symbol: ticker a vender.
            price: precio de mercado de referencia (> 0).

        Returns:
            El :class:`Trade` de venta, con ``pnl`` realizado (neto de la
            comisión de salida; la de entrada ya se descontó del efectivo).

        Raises:
            ExecutionError: si no hay posición abierta o el precio es inválido.
        """
        if price <= 0:
            raise ExecutionError(f"Precio inválido: {price}")
        pos = self.positions.pop(symbol, None)
        if pos is None:
            raise ExecutionError(f"No hay posición abierta en {symbol}")

        exec_price = price * (1.0 - self.slippage_pct)
        notional = exec_price * pos.quantity
        commission = notional * self.commission_pct
        self.cash += notional - commission
        pnl = (exec_price - pos.entry_price) * pos.quantity - commission
        trade = Trade(symbol, "sell", pos.quantity, exec_price, commission, pnl)
        self.trades.append(trade)
        return trade
