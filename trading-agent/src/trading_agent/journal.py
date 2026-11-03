"""Diario de operaciones en disco (auditoría del trading en vivo).

Cada operación ejecutada por el :class:`~trading_agent.live.LiveTrader`
se registra como una fila CSV *append-only*. Es la fuente de verdad para
reconciliar contra el extracto del broker y para analizar la estrategia
tras días de operación — los logs de consola se pierden; el diario no.

Formato (cabecera escrita al crear el archivo)::

    timestamp_utc,symbol,side,quantity,price,commission,pnl,reason,equity

``reason`` indica el origen: ``agent`` (decisión del DQN), ``stop_loss``,
``take_profit`` (cierres forzosos) o ``shutdown`` (liquidación al parar).

Ejemplo
-------
>>> import tempfile, os
>>> from trading_agent.portfolio import Trade
>>> path = os.path.join(tempfile.mkdtemp(), "trades.csv")
>>> journal = TradeJournal(path)
>>> journal.record(Trade("AAPL", "buy", 10, 100.0, 0.0), reason="agent",
...                equity=100_000.0, timestamp="2026-01-02T15:30:00+00:00")
>>> open(path).read().splitlines()[1]
'2026-01-02T15:30:00+00:00,AAPL,buy,10,100.0,0.0,0.0,agent,100000.0,,,,'
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path

from .portfolio import Trade

logger = logging.getLogger(__name__)

#: Columnas del CSV, en orden. Las cuatro últimas permiten reconciliar
#: cada fila contra el extracto del broker (vela de decisión, broker,
#: id de orden e id de cliente idempotente).
COLUMNS = ("timestamp_utc", "symbol", "side", "quantity", "price",
           "commission", "pnl", "reason", "equity",
           "bar_ts", "broker", "order_id", "client_order_id")


class TradeJournal:
    """Escritor CSV append-only de operaciones.

    Args:
        path: ruta del archivo CSV; se crean los directorios padre. Si el
            archivo no existe se escribe la cabecera.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(COLUMNS)
        logger.info("Diario de operaciones: %s", self.path)

    def record(self, trade: Trade, *, reason: str, equity: float,
               timestamp: str | None = None, bar_ts: str | None = None,
               broker: str | None = None) -> None:
        """Añade una fila por operación.

        Args:
            trade: operación ejecutada (reflejada en el portafolio local).
            reason: ``"agent"``, ``"stop_loss"``, ``"take_profit"`` o
                ``"shutdown"``.
            equity: equity tras la operación, en USD.
            timestamp: ISO-8601 UTC; por defecto el instante actual
                (inyectable para tests deterministas).
            bar_ts: marca temporal de la vela sobre la que se decidió.
            broker: nombre del ejecutor (``"PaperBroker"``, ``"AlpacaBroker"``).
        """
        ts = timestamp or dt.datetime.now(dt.timezone.utc).isoformat()
        row = (ts, trade.symbol, trade.side, trade.quantity, trade.price,
               trade.commission, trade.pnl, reason, equity,
               bar_ts or "", broker or "", trade.order_id or "",
               trade.client_order_id or "")
        try:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(row)
        except OSError as exc:
            # Un fallo de disco no debe detener el trading; sí debe gritar.
            logger.error("No se pudo escribir en el diario %s: %s",
                         self.path, exc)
