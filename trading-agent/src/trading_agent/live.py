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
import json
import logging
import signal
import threading
from pathlib import Path
from types import FrameType

import numpy as np

from .agent.dqn import DQNAgent
from .brokers import BrokerAdapter, PaperBroker, make_broker
from .config import AppConfig
from .data.provider import DataProvider, completed_bars, make_provider
from .env import Action
from .exceptions import DataProviderError, ExecutionError, TradingAgentError
from .features import FEATURE_COLUMNS, build_features
from .journal import TradeJournal
from .portfolio import Portfolio, Trade
from .risk import RiskManager
from .train import checkpoint_metadata

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
                 provider: DataProvider | None = None,
                 broker: BrokerAdapter | None = None,
                 journal: TradeJournal | None = None,
                 max_consecutive_errors: int = 10,
                 risk_state_path: str | Path | None = None) -> None:
        """
        Args:
            config: configuración de la aplicación.
            agent: agente DQN con checkpoint cargado (``agent.load``).
            provider: proveedor de datos; si es ``None`` se construye
                desde la config (inyectable para tests).
            broker: ejecutor de órdenes; si es ``None`` se usa
                :class:`~trading_agent.brokers.PaperBroker` (simulado)
                sobre un portafolio con los costes de la config. Un broker
                real (p. ej. Alpaca) trae su propio portafolio espejo.
            journal: diario CSV de operaciones; ``None`` desactiva la
                escritura en disco.
            max_consecutive_errors: nº de iteraciones seguidas con error
                inesperado tras el cual el bucle se detiene (cortacircuitos:
                un proceso que falla sin parar no debe seguir operando).
            risk_state_path: archivo JSON donde persiste el estado del
                RiskManager (pico de equity, kill-switch, enfriamiento).
                Un reinicio del proceso NO rearma el kill-switch: si está
                activo en el archivo, el trader se niega a arrancar hasta
                que un humano lo edite o borre. ``None`` = solo memoria.

        Raises:
            TradingAgentError: parámetros inválidos, o kill-switch activo
                en el estado persistido.
        """
        if max_consecutive_errors < 1:
            raise TradingAgentError("max_consecutive_errors debe ser >= 1")
        self.config = config
        self.agent = agent
        self.provider = provider or make_provider(config.data)
        self.symbol = config.data.symbols[0]
        self.risk = RiskManager(config.risk)
        self.risk_state_path = Path(risk_state_path) if risk_state_path else None
        self._load_risk_state()
        if broker is None:
            broker = PaperBroker(Portfolio(
                cash=config.env.initial_cash,
                commission_pct=config.env.commission_pct,
                slippage_pct=config.env.slippage_pct,
            ))
        self.broker = broker
        self.portfolio = broker.portfolio  # contabilidad compartida
        self.journal = journal
        self.max_consecutive_errors = max_consecutive_errors
        # Evento de parada: la señal lo activa y el bucle despierta al
        # instante en vez de agotar la espera entre iteraciones.
        self._stop_event = threading.Event()
        self._last_bar_date = None      # arma el límite de pérdida diaria
        self._last_decided_bar = None   # el agente decide UNA vez por vela
        self._last_price: float | None = None  # para liquidar sin red

    def _record(self, trade: Trade, *, reason: str, price: float) -> None:
        """Escribe una operación en el diario (si está activo)."""
        if self.journal is not None:
            self.journal.record(
                trade, reason=reason,
                equity=self.portfolio.equity({self.symbol: price}))

    # ------------------------------------------------------------------ #
    # Estado de riesgo persistente                                       #
    # ------------------------------------------------------------------ #
    def _load_risk_state(self) -> None:
        """Restaura el RiskManager desde disco (si hay archivo).

        Raises:
            TradingAgentError: si el estado guardado tiene el kill-switch
                activo (rearmarlo exige intervención humana) o el archivo
                está corrupto.
        """
        if self.risk_state_path is None or not self.risk_state_path.exists():
            return
        try:
            state = json.loads(self.risk_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TradingAgentError(
                f"Estado de riesgo ilegible en {self.risk_state_path}: {exc}"
            ) from exc
        self.risk.load_dict(state)
        logger.info("Estado de riesgo restaurado desde %s: %s",
                    self.risk_state_path, state)
        if self.risk.halted:
            raise TradingAgentError(
                "Kill-switch de drawdown ACTIVO en el estado persistido "
                f"({self.risk_state_path}). Revisa la cuenta y borra o edita "
                "ese archivo para rearmar conscientemente.")

    def _save_risk_state(self) -> None:
        """Persiste el RiskManager a disco (escritura atómica)."""
        if self.risk_state_path is None:
            return
        try:
            self.risk_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.risk_state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.risk.to_dict()), encoding="utf-8")
            tmp.replace(self.risk_state_path)
        except OSError as exc:
            logger.error("No se pudo guardar el estado de riesgo en %s: %s",
                         self.risk_state_path, exc)

    def _reconcile_with_broker(self) -> None:
        """Re-sincroniza el espejo con el broker y grita si divergían.

        Solo aplica a brokers con ``sync()`` (reales). Se ejecuta una vez
        por jornada: la contabilidad local debe coincidir con la cuenta;
        si no (orden manual, fill no visto...), manda la versión del broker.
        """
        sync = getattr(self.broker, "sync", None)
        if not callable(sync):
            return
        before_cash = self.portfolio.cash
        before_pos = {s: p.quantity for s, p in self.portfolio.positions.items()}
        try:
            sync()
        except ExecutionError as exc:
            logger.error("Reconciliación con el broker fallida: %s", exc)
            return
        after_pos = {s: p.quantity for s, p in self.portfolio.positions.items()}
        if before_pos != after_pos or abs(before_cash - self.portfolio.cash) > 1.0:
            logger.critical(
                "DIVERGENCIA espejo/broker corregida: posiciones %s -> %s, "
                "efectivo %.2f -> %.2f. Revisar el diario y la cuenta.",
                before_pos, after_pos, before_cash, self.portfolio.cash)

    # ------------------------------------------------------------------ #
    # Señales del sistema (Ctrl+C / SIGTERM del orquestador)             #
    # ------------------------------------------------------------------ #
    def install_signal_handlers(self) -> None:
        """Instala manejadores para parar limpiamente con SIGINT/SIGTERM."""
        def _handler(signum: int, _frame: FrameType | None) -> None:
            if self._stop_event.is_set():
                # Segunda señal: el operador quiere salir YA (la primera ya
                # pidió parada limpia; la liquidación corre en el finally).
                logger.warning("Segunda señal %s: salida forzada", signum)
                raise SystemExit(130)
            logger.info("Señal %s recibida: parada limpia solicitada", signum)
            self._stop_event.set()

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

        Reglas de producción aplicadas en cada iteración:

        - **Una decisión por vela**: el agente solo se consulta cuando la
          marca temporal de la última vela cambia. Entre velas (mercado
          cerrado, o sondeos más frecuentes que el intervalo) el mismo
          estado produciría la misma acción y, con la posición ya
          abierta/cerrada, el agente podría alternar compra/venta sobre
          datos idénticos. Los stops sí se re-evalúan con cada precio.
        - **Mercado abierto**: ninguna orden se envía si el broker informa
          que el mercado está cerrado (quedaría colgada hasta el timeout).

        Returns:
            Dict de diagnóstico: ``price``, ``action`` (nombre), ``equity``,
            ``has_position``, ``forced_close``, ``stale_bar`` (``True`` si no
            se consultó al agente por ser la misma vela) y ``market_closed``.

        Raises:
            DataProviderError / DataValidationError: si los datos fallan
                (el llamador decide si continuar).
        """
        latest = self.provider.fetch_latest(self.symbol)
        # El estado se construye SOLO con velas completadas (como en el
        # entrenamiento); la vela parcial en curso aporta el último precio.
        completed = completed_bars(latest, self.config.data.interval)
        feats_df = build_features(completed)
        feats = feats_df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        bar_ts = feats_df.index[-1]
        price = float(latest["close"].iloc[-1])
        self._last_price = price
        tag = bar_ts.isoformat()  # id idempotente de las órdenes de esta vela

        equity = self.portfolio.equity({self.symbol: price})
        self.risk.update_equity(equity)

        # Nueva jornada (fecha de la última vela): arma el límite de
        # pérdida diaria con el equity de apertura del día.
        bar_date = bar_ts.date()
        if bar_date != self._last_bar_date:
            if self._last_bar_date is not None:
                self._reconcile_with_broker()
                equity = self.portfolio.equity({self.symbol: price})
            self.risk.start_new_day(equity)
            self._last_bar_date = bar_date

        market_open = self.broker.is_market_open()
        if not market_open:
            logger.info("%s: mercado cerrado, no se envían órdenes", self.symbol)

        # (3) Gestión de riesgo primero: los stops mandan sobre el agente.
        forced_close = None
        if self.portfolio.has_position(self.symbol):
            reason = self.risk.should_force_close(
                self.portfolio.positions[self.symbol], price)
            if reason is not None and market_open:
                trade = self.broker.close_position(self.symbol, price,
                                                   tag=f"{tag}-{reason}")
                forced_close = reason
                self.risk.on_forced_close()
                self._record(trade, reason=reason, price=price)
                logger.warning("%s: cierre forzoso por %s, P&L %.2f USD",
                               self.symbol, reason, trade.pnl)
            elif reason is not None:
                logger.warning("%s: %s pendiente hasta apertura de mercado",
                               self.symbol, reason)

        # (4) Decisión del agente (greedy), solo ante una vela nueva.
        stale_bar = bar_ts == self._last_decided_bar
        if stale_bar:
            action = Action.HOLD
        else:
            self.risk.new_bar()  # consume el enfriamiento tras un stop
            state = self._build_state(feats, price)
            action = Action(self.agent.act(state, greedy=True))
            self._last_decided_bar = bar_ts

        if not market_open:
            pass  # sin órdenes: la decisión se re-evaluará con la vela siguiente
        elif action == Action.BUY and not self.portfolio.has_position(self.symbol):
            if self.risk.can_open(equity):
                qty = self.risk.position_size(equity, price)
                max_affordable = self.portfolio.cash / (
                    price * (1.0 + self.config.env.slippage_pct)
                    * (1.0 + self.config.env.commission_pct))
                qty = min(qty, float(int(max_affordable)))
                if qty > 0:
                    trade = self.broker.open_position(self.symbol, qty, price,
                                                      tag=tag)
                    self._record(trade, reason="agent", price=price)
                    logger.info("%s: COMPRA %g @ %.2f", self.symbol, qty, price)
                else:
                    logger.warning(
                        "%s: compra vetada por tamaño: presupuesto %.2f USD "
                        "(%.0f%% de %.2f) no alcanza para 1 acción a %.2f",
                        self.symbol, equity * self.config.risk.max_position_pct,
                        self.config.risk.max_position_pct * 100, equity, price)
            else:
                logger.warning("%s: compra vetada por el gestor de riesgo",
                               self.symbol)
        elif action == Action.SELL and self.portfolio.has_position(self.symbol):
            trade = self.broker.close_position(self.symbol, price, tag=tag)
            self._record(trade, reason="agent", price=price)
            logger.info("%s: VENTA %g @ %.2f, P&L %.2f USD",
                        self.symbol, trade.quantity, price, trade.pnl)

        equity = self.portfolio.equity({self.symbol: price})
        self._save_risk_state()
        return {
            "price": price,
            "action": action.name,
            "equity": equity,
            "has_position": self.portfolio.has_position(self.symbol),
            "forced_close": forced_close,
            "stale_bar": stale_bar,
            "market_closed": not market_open,
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
        logger.info("Bucle en vivo iniciado para %s cada %.0fs | broker: %s | "
                    "endpoint: %s", self.symbol, poll_seconds,
                    type(self.broker).__name__,
                    getattr(self.broker, "base_url", "simulación local"))

        iteration = 0
        consecutive_errors = 0
        try:
            while not self._stop_event.is_set():
                if max_iterations is not None and iteration >= max_iterations:
                    break
                iteration += 1
                try:
                    info = self.run_once()
                    consecutive_errors = 0
                    logger.info("it %d | %s | precio %.2f | equity %.2f | %s%s",
                                iteration, info["action"], info["price"],
                                info["equity"],
                                "EN POSICIÓN" if info["has_position"] else "en efectivo",
                                " | misma vela" if info["stale_bar"] else "")
                    if self.risk.halted:
                        logger.critical("Kill-switch de drawdown activo: fin del bucle")
                        break
                except DataProviderError as exc:
                    # Transitorio: registrar y esperar al siguiente ciclo.
                    logger.error("Fallo de datos (se reintenta en el próximo "
                                 "ciclo): %s", exc)
                except ExecutionError as exc:
                    logger.error("Fallo de ejecución: %s", exc)
                except Exception:
                    consecutive_errors += 1
                    logger.exception(
                        "Error inesperado en la iteración %d (%d/%d seguidos)",
                        iteration, consecutive_errors, self.max_consecutive_errors)
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.critical("Cortacircuitos: %d errores seguidos. "
                                        "Fin del bucle.", consecutive_errors)
                        break
                # Espera interrumpible: una señal despierta el bucle al instante.
                self._stop_event.wait(poll_seconds)
        finally:
            # Pase lo que pase (señal, kill-switch, excepción no prevista),
            # nunca se deja una posición abierta sin intentar liquidarla.
            self._liquidate()
            logger.info("Bucle en vivo terminado tras %d iteraciones", iteration)

    def _liquidate(self) -> None:
        """Cierra la posición abierta (si existe) al detener el bucle."""
        if not self.portfolio.has_position(self.symbol):
            return
        try:
            # Con un broker real la orden market usa el precio de mercado;
            # el último precio conocido basta y evita depender de la red
            # (y de sus reintentos) justo cuando hay prisa por parar.
            if self._last_price is not None:
                price = self._last_price
            else:
                latest = self.provider.fetch_latest(self.symbol)
                price = float(latest["close"].iloc[-1])
            if not self.broker.is_market_open():
                logger.error("Mercado cerrado: no se puede liquidar %s ahora. "
                             "La posición sigue abierta en el broker; "
                             "REVISAR MANUALMENTE.", self.symbol)
                return
            trade = self.broker.close_position(self.symbol, price)
            self._record(trade, reason="shutdown", price=price)
            logger.info("Liquidación final: %s P&L %.2f USD", self.symbol, trade.pnl)
        except Exception as exc:  # noqa: BLE001 - último recurso: avisar, no ocultar
            logger.error("No se pudo liquidar %s automáticamente: %s. "
                         "REVISAR MANUALMENTE.", self.symbol, exc)


def build_live_trader(config: AppConfig, checkpoint_path: str,
                      broker_name: str = "paper",
                      journal_path: str | None = None,
                      risk_state_path: str | None = None) -> LiveTrader:
    """Construye un :class:`LiveTrader` con el checkpoint indicado.

    Args:
        config: configuración de la aplicación.
        checkpoint_path: ruta del modelo entrenado (``.pt``).
        broker_name: ``"paper"`` (simulado, defecto) o ``"alpaca"``.
            Con un broker real, el portafolio espejo se crea sin costes
            simulados (los costes reales vienen en cada fill) y se
            sincronizan efectivo y posiciones abiertas con la cuenta.
        journal_path: ruta del diario CSV de operaciones; ``None`` lo
            desactiva.
        risk_state_path: archivo JSON del estado de riesgo persistente
            (ver :class:`LiveTrader`); ``None`` = solo memoria.

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
    # Los metadatos del checkpoint deben coincidir con esta configuración
    # (ventana, features, intervalo, símbolo): si no, no se opera.
    agent.load(checkpoint_path,
               expected_metadata=checkpoint_metadata(config, config.data.symbols[0]))

    if broker_name == "paper":
        broker = None  # LiveTrader crea el PaperBroker con la config
    else:
        # Broker real: portafolio espejo sin costes simulados.
        broker = make_broker(broker_name, Portfolio(
            cash=config.env.initial_cash, commission_pct=0.0,
            slippage_pct=0.0))
        sync = getattr(broker, "sync", None)
        if callable(sync):
            sync()  # efectivo + posiciones reales (sobrevive a reinicios)
    journal = TradeJournal(journal_path) if journal_path else None
    return LiveTrader(config, agent, broker=broker, journal=journal,
                      risk_state_path=risk_state_path)
