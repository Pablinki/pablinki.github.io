"""Tests de la configuración: validación, carga YAML y valores por defecto."""

from __future__ import annotations

import pytest

from trading_agent.config import (AgentConfig, AppConfig, DataConfig,
                                  EnvConfig, RiskConfig, load_config)
from trading_agent.exceptions import ConfigurationError


class TestValidation:
    """Cada parámetro fuera de rango debe fallar en construcción (fail-fast)."""

    def test_default_config_is_valid(self):
        cfg = AppConfig.default()
        assert cfg.agent.gamma == 0.99
        assert cfg.data.provider == "yahoo"

    @pytest.mark.parametrize("kwargs", [
        {"gamma": 0.0}, {"gamma": 1.5}, {"lr": -1e-3},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},   # end > start
        {"epsilon_decay": 0.0},
        {"batch_size": 0},
        {"buffer_capacity": 10, "batch_size": 64},    # capacidad < batch
        {"n_actions": 1},
        {"hidden_sizes": (0,)},
    ])
    def test_invalid_agent_params_raise(self, kwargs):
        with pytest.raises(ConfigurationError):
            AgentConfig(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"provider": "iex"}, {"symbols": ()}, {"lookback_days": 0},
        {"max_retries": -1},
    ])
    def test_invalid_data_params_raise(self, kwargs):
        with pytest.raises(ConfigurationError):
            DataConfig(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"window_size": 1}, {"initial_cash": 0.0},
        {"commission_pct": 1.0}, {"reward_scaling": 0.0},
    ])
    def test_invalid_env_params_raise(self, kwargs):
        with pytest.raises(ConfigurationError):
            EnvConfig(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"max_position_pct": 0.0}, {"max_position_pct": 1.5},
        {"stop_loss_pct": 0.0}, {"take_profit_pct": -0.1},
        {"max_drawdown_pct": 2.0},
    ])
    def test_invalid_risk_params_raise(self, kwargs):
        with pytest.raises(ConfigurationError):
            RiskConfig(**kwargs)


class TestFromDict:
    def test_partial_dict_overrides_only_given_fields(self):
        cfg = AppConfig.from_dict({"agent": {"gamma": 0.9}})
        assert cfg.agent.gamma == 0.9
        assert cfg.agent.lr == AgentConfig().lr  # el resto conserva defecto

    def test_unknown_section_raises(self):
        with pytest.raises(ConfigurationError, match="desconocidas"):
            AppConfig.from_dict({"nave_espacial": {}})

    def test_unknown_field_raises(self):
        with pytest.raises(ConfigurationError, match="desconocidos"):
            AppConfig.from_dict({"agent": {"gama": 0.9}})  # typo intencional

    def test_yaml_lists_become_tuples(self):
        cfg = AppConfig.from_dict({"agent": {"hidden_sizes": [32, 16]},
                                   "data": {"symbols": ["MSFT", "AAPL"]}})
        assert cfg.agent.hidden_sizes == (32, 16)
        assert cfg.data.symbols == ("MSFT", "AAPL")


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("agent:\n  gamma: 0.95\nrisk:\n  stop_loss_pct: 0.02\n")
        cfg = load_config(p)
        assert cfg.agent.gamma == 0.95
        assert cfg.risk.stop_loss_pct == 0.02

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="No existe"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("agent: [esto no: es un mapeo válido")
        with pytest.raises(ConfigurationError):
            load_config(p)
