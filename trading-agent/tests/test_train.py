"""Tests de integración del bucle de entrenamiento (sin red: datos sintéticos)."""

from __future__ import annotations

import pytest

from trading_agent.config import (
    AgentConfig,
    AppConfig,
    DataConfig,
    EnvConfig,
    RiskConfig,
)
from trading_agent.exceptions import TradingAgentError
from trading_agent.train import three_way_split, time_split, train


def small_config() -> AppConfig:
    """Config minúscula para que el test de integración corra en segundos."""
    return AppConfig(
        data=DataConfig(symbols=("TEST",), cache_dir=None),
        agent=AgentConfig(hidden_sizes=(16,), batch_size=8,
                          min_buffer_size=8, buffer_capacity=1_000,
                          target_update_every=20),
        env=EnvConfig(window_size=10),
        risk=RiskConfig(),
    )


class TestTimeSplit:
    def test_split_preserves_temporal_order(self, ohlcv):
        train_df, val_df = time_split(ohlcv, 0.8)
        assert len(train_df) + len(val_df) == len(ohlcv)
        # Anti look-ahead: toda la validación es posterior al entrenamiento.
        assert train_df.index.max() < val_df.index.min()

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.5, 2.0])
    def test_invalid_fraction_raises(self, ohlcv, frac):
        with pytest.raises(TradingAgentError):
            time_split(ohlcv, frac)


class TestThreeWaySplit:
    def test_three_consecutive_time_ordered_parts(self, ohlcv):
        tr, va, te = three_way_split(ohlcv, 0.7, 0.15)
        assert len(tr) + len(va) + len(te) == len(ohlcv)
        assert tr.index.max() < va.index.min() < va.index.max() < te.index.min()
        assert len(tr) == int(len(ohlcv) * 0.7)

    @pytest.mark.parametrize("tr,va", [(0.0, 0.5), (0.5, 0.5), (0.9, 0.2),
                                       (1.0, 0.1)])
    def test_invalid_fractions_raise(self, ohlcv, tr, va):
        with pytest.raises(TradingAgentError):
            three_way_split(ohlcv, tr, va)


class TestTrainIntegration:
    def test_end_to_end_training_saves_checkpoint(self, ohlcv, tmp_path):
        """Humo de integración: entrena 2 episodios reales y verifica que
        produce un checkpoint cargable y métricas coherentes."""
        ckpt = tmp_path / "best.pt"
        metrics = train(small_config(), episodes=2, checkpoint_path=ckpt,
                        ohlcv=ohlcv)
        assert ckpt.exists()
        assert metrics["episodes_run"] == 2.0
        assert metrics["best_val_return"] > -1.0  # no perdió todo el capital
        # Evaluación fuera de muestra con el mejor checkpoint.
        for key in ("test_return", "test_buy_and_hold", "test_sharpe",
                    "test_max_drawdown"):
            assert key in metrics
        assert 0.0 <= metrics["test_max_drawdown"] <= 1.0

    def test_zero_episodes_raises(self, ohlcv, tmp_path):
        with pytest.raises(TradingAgentError, match="episodes"):
            train(small_config(), episodes=0,
                  checkpoint_path=tmp_path / "x.pt", ohlcv=ohlcv)
