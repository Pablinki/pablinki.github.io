"""Tests del bucle en vivo con un proveedor de datos falso (sin red)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from trading_agent.agent.dqn import DQNAgent
from trading_agent.config import (AgentConfig, AppConfig, DataConfig,
                                  EnvConfig, RiskConfig)
from trading_agent.data.provider import DataProvider
from trading_agent.exceptions import DataProviderError, SecurityError
from trading_agent.features import FEATURE_COLUMNS
from trading_agent.live import LiveTrader
from trading_agent.security import (LIVE_TRADING_ENV_VAR,
                                    assert_live_trading_allowed, get_secret,
                                    is_live_trading_enabled, mask_secret)

from conftest import make_ohlcv


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
