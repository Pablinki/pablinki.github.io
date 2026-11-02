"""Tests del bucle en vivo con un proveedor de datos falso (sin red)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from conftest import make_ohlcv

from trading_agent.agent.dqn import DQNAgent
from trading_agent.brokers.base import PaperBroker
from trading_agent.config import (
    AgentConfig,
    AppConfig,
    DataConfig,
    EnvConfig,
    RiskConfig,
)
from trading_agent.data.provider import DataProvider
from trading_agent.exceptions import (
    DataProviderError,
    SecurityError,
    TradingAgentError,
)
from trading_agent.features import FEATURE_COLUMNS
from trading_agent.live import LiveTrader
from trading_agent.security import (
    LIVE_TRADING_ENV_VAR,
    assert_live_trading_allowed,
    get_secret,
    is_live_trading_enabled,
    mask_secret,
)


class FakeProvider(DataProvider):
    """Proveedor de test: sirve un OHLCV sintético fijo, sin tocar la red."""

    def __init__(self, config: DataConfig, df: pd.DataFrame | None = None,
                 fail: bool = False) -> None:
        super().__init__(config)
        self.df = df if df is not None else make_ohlcv(200)
        self.fail = fail

    def fetch_historical(self, symbol: str) -> pd.DataFrame:
        return self.df

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        if self.fail:
            raise DataProviderError("fallo simulado de red")
        return self.df


def make_trader(fail_data: bool = False) -> LiveTrader:
    config = AppConfig(
        data=DataConfig(symbols=("TEST",), cache_dir=None),
        agent=AgentConfig(hidden_sizes=(16,), batch_size=8,
                          min_buffer_size=8, buffer_capacity=100),
        env=EnvConfig(window_size=10),
        risk=RiskConfig(),
    )
    state_dim = config.env.window_size * len(FEATURE_COLUMNS) + 2
    agent = DQNAgent(dataclasses.replace(config.agent, state_dim=state_dim))
    provider = FakeProvider(config.data, fail=fail_data)
    return LiveTrader(config, agent, provider=provider)


class TestRunOnce:
    def test_produces_diagnostic_info(self):
        info = make_trader().run_once()
        assert info["action"] in ("HOLD", "BUY", "SELL")
        assert info["price"] > 0
        assert info["equity"] > 0

    def test_data_failure_propagates_for_caller_to_handle(self):
        with pytest.raises(DataProviderError):
            make_trader(fail_data=True).run_once()

    def test_injected_broker_shares_portfolio_accounting(self):
        """LiveTrader y el broker inyectado deben compartir el mismo
        Portfolio: la contabilidad que ve el riesgo es la del broker."""
        from trading_agent.portfolio import Portfolio

        broker = PaperBroker(Portfolio(cash=12_345.0))
        trader = make_trader()
        trader_with_broker = LiveTrader(trader.config, trader.agent,
                                        provider=trader.provider,
                                        broker=broker)
        assert trader_with_broker.portfolio is broker.portfolio
        assert trader_with_broker.portfolio.cash == pytest.approx(12_345.0)

    def test_daily_loss_limit_gets_armed(self):
        """run_once debe armar el límite de pérdida diaria con la fecha de
        la última vela (antes era código muerto en vivo)."""
        trader = make_trader()
        assert trader.risk.day_start_equity is None
        trader.run_once()
        assert trader.risk.day_start_equity is not None

    def test_equity_is_conserved_without_trades(self):
        trader = make_trader()
        initial = trader.portfolio.cash
        info = trader.run_once()
        if info["action"] == "HOLD" and not info["has_position"]:
            assert info["equity"] == pytest.approx(initial)


class TestRunForever:
    def test_bounded_iterations_terminate(self):
        trader = make_trader()
        trader.run_forever(poll_seconds=1.0, max_iterations=2)
        assert not trader.portfolio.has_position("TEST")  # liquidación final

    def test_survives_transient_data_failures(self):
        """El bucle NO debe morir por fallos de datos: los registra y sigue."""
        trader = make_trader(fail_data=True)
        trader.run_forever(poll_seconds=1.0, max_iterations=2)  # no lanza


class ForcedAgent:
    """Agente de prueba con acción fija (y conteo de consultas)."""

    def __init__(self, action: int, state_dim: int) -> None:
        self.action = action
        self.calls = 0
        self.config = type("Cfg", (), {"state_dim": state_dim})()

    def act(self, state, *, greedy: bool = False) -> int:
        self.calls += 1
        return self.action


class ClosedMarketBroker(PaperBroker):
    def is_market_open(self) -> bool:
        return False


class ExplodingProvider(DataProvider):
    """Proveedor que lanza un error NO tipado (bug inesperado)."""

    def fetch_historical(self, symbol):
        raise RuntimeError("bug inesperado")

    def fetch_latest(self, symbol):
        raise RuntimeError("bug inesperado")


def make_forced_trader(action: int, broker=None, journal=None,
                       provider=None, **kwargs) -> LiveTrader:
    base = make_trader()
    agent = ForcedAgent(action, base.config.env.window_size
                        * len(FEATURE_COLUMNS) + 2)
    return LiveTrader(base.config, agent, provider=provider or base.provider,
                      broker=broker, journal=journal, **kwargs)


class TestProductionGuards:
    def test_agent_decides_once_per_bar(self):
        """Misma vela -> no se consulta al agente (evita alternar compra/
        venta sobre datos idénticos con el mercado cerrado)."""
        from trading_agent.env import Action

        trader = make_forced_trader(Action.BUY)
        info1 = trader.run_once()
        assert info1["action"] == "BUY" and info1["has_position"]
        assert not info1["stale_bar"]
        # El proveedor falso devuelve siempre las mismas velas.
        info2 = trader.run_once()
        assert info2["stale_bar"]
        assert info2["action"] == "HOLD"
        assert trader.agent.calls == 1

    def test_no_orders_when_market_closed(self):
        from trading_agent.env import Action
        from trading_agent.portfolio import Portfolio

        broker = ClosedMarketBroker(Portfolio(cash=100_000.0))
        trader = make_forced_trader(Action.BUY, broker=broker)
        info = trader.run_once()
        assert info["market_closed"]
        assert not info["has_position"]
        assert broker.portfolio.cash == pytest.approx(100_000.0)

    def test_forced_close_waits_for_market_open(self):
        from trading_agent.env import Action
        from trading_agent.portfolio import Portfolio, Position

        broker = ClosedMarketBroker(Portfolio(cash=0.0))
        # Posición muy perdedora: el stop-loss querría cerrar ya.
        broker.portfolio.positions["TEST"] = Position("TEST", 1, 1e9)
        trader = make_forced_trader(Action.HOLD, broker=broker)
        info = trader.run_once()
        assert info["forced_close"] is None  # pendiente hasta apertura
        assert broker.portfolio.has_position("TEST")

    def test_journal_records_agent_trades(self, tmp_path):
        from trading_agent.env import Action
        from trading_agent.journal import TradeJournal

        journal = TradeJournal(tmp_path / "trades.csv")
        trader = make_forced_trader(Action.BUY, journal=journal)
        trader.run_once()
        rows = (tmp_path / "trades.csv").read_text().splitlines()
        assert len(rows) == 2  # cabecera + compra
        assert ",TEST,buy," in rows[1]
        assert rows[1].split(",")[7] == "agent"

    def test_shutdown_liquidation_is_journaled(self, tmp_path):
        from trading_agent.env import Action
        from trading_agent.journal import TradeJournal

        journal = TradeJournal(tmp_path / "trades.csv")
        trader = make_forced_trader(Action.BUY, journal=journal)
        trader.run_forever(poll_seconds=1.0, max_iterations=1)
        rows = (tmp_path / "trades.csv").read_text().splitlines()
        assert any("shutdown" in r for r in rows)
        assert not trader.portfolio.has_position("TEST")

    def test_unexpected_errors_do_not_kill_the_loop(self):
        """Un bug no tipado se registra y el bucle continúa..."""
        from trading_agent.env import Action

        trader = make_forced_trader(
            Action.HOLD, provider=ExplodingProvider(make_trader().config.data),
            max_consecutive_errors=5)
        trader.run_forever(poll_seconds=1.0, max_iterations=3)  # no lanza

    def test_circuit_breaker_stops_after_consecutive_errors(self, caplog):
        """...pero N errores seguidos activan el cortacircuitos."""
        from trading_agent.env import Action

        trader = make_forced_trader(
            Action.HOLD, provider=ExplodingProvider(make_trader().config.data),
            max_consecutive_errors=2)
        with caplog.at_level("CRITICAL"):
            trader.run_forever(poll_seconds=1.0, max_iterations=10)
        assert "Cortacircuitos" in caplog.text

    def test_partial_last_bar_is_not_used_for_the_state(self):
        """Una vela todavía en formación no entra en el estado (el
        entrenamiento solo vio velas completas), pero sí da el último precio."""
        from trading_agent.env import Action

        df = make_ohlcv(200)
        # Reindexar para que la ÚLTIMA vela sea "hoy": todavía abierta.
        today = pd.Timestamp.now(tz="UTC").normalize()
        df.index = pd.date_range(end=today, periods=len(df), freq="D")
        base = make_trader()
        provider = FakeProvider(base.config.data, df=df)
        trader = make_forced_trader(Action.HOLD, provider=provider)
        info = trader.run_once()
        assert trader._last_decided_bar == df.index[-2]   # última completa
        assert info["price"] == pytest.approx(float(df["close"].iloc[-1]))

    def test_stop_event_wakes_the_loop_immediately(self):
        """Una señal no debe esperar el poll completo para parar."""
        import time

        from trading_agent.env import Action

        trader = make_forced_trader(Action.HOLD)
        trader._stop_event.set()  # equivale a recibir SIGINT antes de dormir
        start = time.monotonic()
        trader.run_forever(poll_seconds=30.0, max_iterations=5)
        assert time.monotonic() - start < 5.0

    def test_invalid_max_consecutive_errors_raises(self):
        from trading_agent.env import Action

        with pytest.raises(TradingAgentError):
            make_forced_trader(Action.HOLD, max_consecutive_errors=0)


class TestSecurity:
    def test_get_secret_reads_env(self, monkeypatch):
        monkeypatch.setenv("X_API_KEY", "valor-secreto")
        assert get_secret("X_API_KEY") == "valor-secreto"

    def test_get_secret_missing_required_raises(self, monkeypatch):
        monkeypatch.delenv("X_API_KEY", raising=False)
        with pytest.raises(SecurityError, match="X_API_KEY"):
            get_secret("X_API_KEY")

    def test_get_secret_optional_returns_none(self, monkeypatch):
        monkeypatch.delenv("X_API_KEY", raising=False)
        assert get_secret("X_API_KEY", required=False) is None

    def test_mask_secret_hides_middle(self):
        masked = mask_secret("clave-super-secreta")
        assert "super" not in masked
        assert masked.startswith("cla") and masked.endswith("eta")

    def test_live_trading_off_by_default(self, monkeypatch):
        monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
        assert not is_live_trading_enabled()
        with pytest.raises(SecurityError, match="deshabilitado"):
            assert_live_trading_allowed()

    def test_live_trading_requires_exact_yes(self, monkeypatch):
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "yes")  # minúsculas: NO vale
        assert not is_live_trading_enabled()
        monkeypatch.setenv(LIVE_TRADING_ENV_VAR, "YES")
        assert is_live_trading_enabled()
        assert_live_trading_allowed()  # ya no lanza
