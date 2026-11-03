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

    def test_window_ends_at_current_bar(self, small_ohlcv):
        """La ventana del estado debe INCLUIR la vela actual t: es lo que
        construye el bucle en vivo (últimas w velas completadas). Antes el
        entorno la excluía y producción veía un estado desplazado 1 vela."""
        env = make_env(small_ohlcv)
        state = env.reset()
        w = env.env_config.window_size
        n_feat = len(FEATURE_COLUMNS)
        last_row_in_state = state[(w - 1) * n_feat: w * n_feat]
        np.testing.assert_allclose(last_row_in_state, env._feat_matrix[env._t])
        # ...y el precio de ejecución es el de esa misma vela.
        assert float(env._prices[env._t]) > 0

    def test_terminated_only_on_kill_switch(self, small_ohlcv):
        """Agotar los datos trunca el episodio pero NO es un estado
        terminal del MDP (no debe anular el valor futuro en el objetivo)."""
        env = make_env(small_ohlcv)
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(Action.HOLD)
        assert done and info["terminated"] is False

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


    def test_reward_includes_final_liquidation_costs(self, small_ohlcv):
        """Con comisiones y slippage, la suma de recompensas debe seguir
        reconstruyendo el log-retorno final (liquidación incluida)."""
        cfg = EnvConfig(window_size=10, commission_pct=0.002,
                        slippage_pct=0.001, reward_scaling=100.0)
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

    def test_stop_is_evaluated_intrabar_and_fills_at_stop_level(self, small_ohlcv):
        """Si el mínimo de la vela cruza el stop, se cierra al nivel del
        stop aunque el cierre haya recuperado (antes solo miraba el cierre)."""
        df = small_ohlcv.copy()
        env = TradingEnvironment(
            df, EnvConfig(window_size=10, commission_pct=0.0, slippage_pct=0.0),
            RiskConfig(stop_loss_pct=0.05, take_profit_pct=None), symbol="TEST")
        env.reset()
        env.step(Action.BUY)
        entry = env._portfolio.positions["TEST"].entry_price
        # Forzar en la vela siguiente un mínimo por debajo del stop con
        # cierre por encima: debe dispararse y ejecutar al nivel del stop.
        t = env._t + 1
        env._lows[t] = entry * 0.90
        env._opens[t] = entry * 0.99
        env._prices[t] = entry * 0.99
        _, _, _, info = env.step(Action.HOLD)
        assert info["forced_close"] == "stop_loss"
        sell = env._portfolio.trades[-1]
        assert sell.price == pytest.approx(entry * 0.95)

    def test_cooldown_blocks_reentry_after_stop(self, small_ohlcv):
        env = TradingEnvironment(
            small_ohlcv, EnvConfig(window_size=10),
            RiskConfig(stop_loss_pct=0.0001, take_profit_pct=None,
                       cooldown_bars=3), symbol="TEST")
        env.reset()
        env.step(Action.BUY)
        done = False
        while not done:
            _, _, done, info = env.step(Action.HOLD)
            if info["forced_close"]:
                break
        assert env.risk.cooldown_remaining == 3
        _, _, _, info = env.step(Action.BUY)   # vetada por enfriamiento
        assert not info["has_position"]

    def test_daily_loss_limit_is_armed(self, small_ohlcv):
        """El límite de pérdida diaria debe armarse en reset y rearmarse al
        cambiar el día de calendario (antes era código muerto)."""
        env = make_env(small_ohlcv)
        env.reset()
        assert env.risk.day_start_equity == pytest.approx(100_000.0)
        env.step(Action.HOLD)  # velas diarias: cada paso es un día nuevo
        assert env.risk.day_start_equity is not None

    def test_daily_limit_arms_with_previous_close_equity(self, small_ohlcv):
        """El día se arma con el equity al cierre ANTERIOR, así el hueco de
        apertura cuenta dentro de la pérdida diaria."""
        env = make_env(small_ohlcv)
        env.reset()
        _, _, _, info = env.step(Action.BUY)   # posición abierta al cierre t
        equity_prev_close = info["equity"]
        env.step(Action.HOLD)                   # nuevo día: se rearma
        assert env.risk.day_start_equity == pytest.approx(equity_prev_close)

    def test_precomputed_features_match_internal(self, small_ohlcv):
        """Pasar features precalculadas sobre el histórico completo da el
        mismo estado que calcularlas dentro, sin perder el calentamiento."""
        from trading_agent.features import build_features

        feats = build_features(small_ohlcv)
        env_a = make_env(small_ohlcv)
        env_b = TradingEnvironment(small_ohlcv, EnvConfig(window_size=10),
                                   RiskConfig(), symbol="TEST", features=feats)
        np.testing.assert_allclose(env_a.reset(), env_b.reset())
        # Un tramo posterior conserva su calentamiento si recibe las
        # features del histórico completo.
        tail = small_ohlcv.iloc[60:]
        env_tail = TradingEnvironment(tail, EnvConfig(window_size=10),
                                      RiskConfig(), symbol="TEST", features=feats)
        assert len(env_tail._features) == len(tail)

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

    def test_summary_includes_risk_metrics_and_benchmark(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        done = False
        first = True
        while not done:
            _, _, done, _ = env.step(Action.BUY if first else Action.HOLD)
            first = False
        s = env.episode_summary()
        for key in ("buy_and_hold_return_pct", "excess_return_pct",
                    "max_drawdown_pct", "sharpe", "profit_factor"):
            assert key in s
        assert 0.0 <= s["max_drawdown_pct"] <= 1.0
        assert s["excess_return_pct"] == pytest.approx(
            s["total_return_pct"] - s["buy_and_hold_return_pct"])

    def test_buy_and_hold_benchmark_matches_prices(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        env.step(Action.HOLD)
        s = env.episode_summary()
        first = env._prices[env.env_config.window_size]
        last = env._prices[env._t]
        assert s["buy_and_hold_return_pct"] == pytest.approx(last / first - 1.0)

    def test_sharpe_uses_the_configured_periods_per_year(self, small_ohlcv):
        def run(ppy):
            env = TradingEnvironment(small_ohlcv, EnvConfig(window_size=10),
                                     RiskConfig(), symbol="TEST",
                                     periods_per_year=ppy)
            env.reset()
            done, first = False, True
            while not done:
                _, _, done, _ = env.step(Action.BUY if first else Action.HOLD)
                first = False
            return env.episode_summary()["sharpe"]

        assert run(252 * 4) == pytest.approx(run(252) * 2.0)

    def test_equity_curve_tracks_every_step(self, small_ohlcv):
        env = make_env(small_ohlcv)
        env.reset()
        assert env._equity_curve == [100_000.0]
        env.step(Action.HOLD)
        env.step(Action.HOLD)
        assert len(env._equity_curve) == 3
