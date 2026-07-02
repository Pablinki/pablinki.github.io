"""Tests del entorno de simulación (MDP de trading)."""

from __future__ import annotations

import numpy as np
import pytest

from trading_agent.config import EnvConfig, RiskConfig
from trading_agent.env import Action, TradingEnvironment
from trading_agent.exceptions import EnvironmentError_
from trading_agent.features import FEATURE_COLUMNS


def make_env(ohlcv, **env_kwargs) -> TradingEnvironment:
    return TradingEnvironment(ohlcv, EnvConfig(window_size=10, **env_kwargs),
                              RiskConfig(), symbol="TEST")


class TestStateSpace:
    def test_state_dim_formula(self, small_ohlcv):
        env = make_env(small_ohlcv)
        assert env.state_dim == 10 * len(FEATURE_COLUMNS) + 2

    def test_reset_returns_valid_state(self, small_ohlcv):
        env = make_env(small_ohlcv)
        state = env.reset()
        assert state.shape == (env.state_dim,)
        assert state.dtype == np.float32
        assert not np.isnan(state).any()
        # Sin posición: los dos extras de cartera valen 0.
        assert state[-2] == 0.0 and state[-1] == 0.0

    def test_state_reflects_open_position(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        state, _, _, info = env.step(Action.BUY)
        assert info["has_position"]
        assert state[-2] == 1.0  # flag has_position dentro del estado


class TestStepMechanics:
    def test_step_before_reset_raises(self, small_ohlcv):
        env = make_env(small_ohlcv)
        with pytest.raises(EnvironmentError_, match="reset"):
            env.step(Action.HOLD)

    def test_invalid_action_raises(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        with pytest.raises(EnvironmentError_, match="inválida"):
            env.step(99)

    def test_hold_in_cash_gives_zero_reward(self, small_ohlcv):
        """Sin posición ni operaciones, el equity no cambia: recompensa 0."""
        env = make_env(small_ohlcv)
        env.reset()
        _, reward, _, info = env.step(Action.HOLD)
        assert reward == pytest.approx(0.0)
        assert info["equity"] == pytest.approx(100_000.0)

    def test_sell_without_position_is_noop(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        _, reward, _, info = env.step(Action.SELL)
        assert reward == pytest.approx(0.0)
        assert not info["has_position"]

    def test_buy_then_sell_roundtrip(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        _, _, _, info = env.step(Action.BUY)
        assert info["has_position"]
        _, _, _, info = env.step(Action.SELL)
        assert not info["has_position"]

    def test_episode_terminates_at_data_end(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(Action.HOLD)
            steps += 1
            assert steps < 10_000, "el episodio nunca terminó"
        with pytest.raises(EnvironmentError_):
            env.step(Action.HOLD)  # tras done, step exige reset

    def test_final_liquidation_leaves_no_position(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        done = False
        first = True
        while not done:
            _, _, done, info = env.step(Action.BUY if first else Action.HOLD)
            first = False
        assert not info["has_position"]


class TestRewardAlignment:
    def test_reward_is_scaled_log_equity_change(self, small_ohlcv):
        """La suma de recompensas debe reconstruir el log-retorno total."""
        cfg = EnvConfig(window_size=10, commission_pct=0.0, slippage_pct=0.0,
                        reward_scaling=100.0)
        env = TradingEnvironment(small_ohlcv, cfg, RiskConfig(
            stop_loss_pct=0.99, take_profit_pct=None, max_drawdown_pct=1.0))
        env.reset()
        total_reward = 0.0
        done = False
        first = True
        while not done:
            _, r, done, info = env.step(Action.BUY if first else Action.HOLD)
            first = False
            total_reward += r
        expected = np.log(info["equity"] / cfg.initial_cash) * cfg.reward_scaling
        assert total_reward == pytest.approx(expected, rel=1e-6)


class TestRiskIntegration:
    def test_stop_loss_forces_close(self, small_ohlcv):
        """Con un stop minúsculo, cualquier caída fuerza el cierre."""
        env = TradingEnvironment(
            small_ohlcv, EnvConfig(window_size=10),
            RiskConfig(stop_loss_pct=0.0001, take_profit_pct=None),
            symbol="TEST")
        env.reset()
        env.step(Action.BUY)
        closes = []
        done = False
        while not done:
            _, _, done, info = env.step(Action.HOLD)
            if info["forced_close"]:
                closes.append(info["forced_close"])
        assert "stop_loss" in closes

    def test_position_size_respects_limit(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        env.step(Action.BUY)
        pos = env._portfolio.positions["TEST"]
        # El límite del 25% se aplica sobre el equity AL ABRIR (el inicial,
        # pues es la primera acción); + 1 acción de margen por el redondeo.
        entry_cost = pos.quantity * pos.entry_price
        assert entry_cost <= 0.25 * 100_000.0 + pos.entry_price


class TestSummary:
    def test_summary_counts_trades_and_winrate(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        env.step(Action.BUY)
        env.step(Action.SELL)
        s = env.episode_summary()
        assert s["n_trades"] == 2.0
        assert 0.0 <= s["win_rate"] <= 1.0
        assert s["final_equity"] > 0
