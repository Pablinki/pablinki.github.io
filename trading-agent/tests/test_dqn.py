"""Tests del agente DQN: política, aprendizaje y persistencia."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trading_agent.agent.dqn import DQNAgent
from trading_agent.agent.networks import QNetwork
from trading_agent.config import AgentConfig
from trading_agent.exceptions import TradingAgentError

STATE_DIM = 8


def make_agent(**overrides) -> DQNAgent:
    cfg = AgentConfig(state_dim=STATE_DIM, hidden_sizes=(16,),
                      batch_size=8, min_buffer_size=8, buffer_capacity=100,
                      target_update_every=5, **overrides)
    return DQNAgent(cfg)


def random_state(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=STATE_DIM).astype(np.float32)


class TestQNetwork:
    def test_output_shape(self):
        net = QNetwork(state_dim=6, n_actions=3, hidden_sizes=(8, 4))
        out = net(torch.zeros(5, 6))
        assert out.shape == (5, 3)

    def test_architecture_is_parametrized(self):
        net = QNetwork(4, 2, hidden_sizes=(10, 20, 30))
        widths = [m.out_features for m in net.model
                  if isinstance(m, torch.nn.Linear)]
        assert widths == [10, 20, 30, 2]


class TestPolicy:
    def test_greedy_action_is_deterministic(self):
        agent = make_agent()
        s = np.ones(STATE_DIM, dtype=np.float32)
        actions = {agent.act(s, greedy=True) for _ in range(10)}
        assert len(actions) == 1

    def test_epsilon_one_explores_uniformly(self):
        agent = make_agent(epsilon_start=1.0)
        rng = np.random.default_rng(0)
        actions = {agent.act(random_state(rng)) for _ in range(100)}
        assert actions == {0, 1, 2}  # con eps=1 salen las 3 acciones

    def test_wrong_state_shape_raises(self):
        agent = make_agent()
        with pytest.raises(TradingAgentError, match="forma"):
            agent.act(np.zeros(3, dtype=np.float32))

    def test_epsilon_decays_to_floor(self):
        agent = make_agent(epsilon_start=1.0, epsilon_end=0.1,
                           epsilon_decay=0.5)
        for _ in range(20):
            agent.end_episode()
        assert agent.epsilon == pytest.approx(0.1)


class TestLearning:
    def test_learn_returns_none_during_warmup(self):
        agent = make_agent()
        assert agent.learn() is None

    def test_learn_reduces_loss_on_fixed_problem(self):
        """En un problema estacionario trivial (acción 1 siempre da +1,
        las demás 0) la pérdida debe bajar y la política aprenderla."""
        agent = make_agent(lr=5e-3, epsilon_start=0.0, epsilon_end=0.0,
                           gamma=0.1)
        rng = np.random.default_rng(1)
        for _ in range(200):
            s, s2 = random_state(rng), random_state(rng)
            a = int(rng.integers(0, 3))
            agent.remember(s, a, 1.0 if a == 1 else 0.0, s2, False)

        losses = [agent.learn() for _ in range(300)]
        losses = [l for l in losses if l is not None]
        assert np.mean(losses[-30:]) < np.mean(losses[:30])

        # La política greedy debe preferir la acción 1 casi siempre.
        hits = sum(agent.act(random_state(rng), greedy=True) == 1
                   for _ in range(50))
        assert hits >= 45

    def test_target_network_syncs_on_schedule(self):
        agent = make_agent()
        rng = np.random.default_rng(2)
        for _ in range(20):
            s = random_state(rng)
            agent.remember(s, 0, 1.0, s, False)
        for _ in range(5):  # target_update_every=5
            agent.learn()
        online = agent.online.state_dict()
        target = agent.target.state_dict()
        for k in online:
            torch.testing.assert_close(online[k], target[k])


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        agent = make_agent()
        rng = np.random.default_rng(3)
        for _ in range(20):
            s = random_state(rng)
            agent.remember(s, 1, 0.5, s, False)
        agent.learn()
        path = tmp_path / "ckpt.pt"
        agent.save(path)

        agent2 = make_agent()
        agent2.load(path)
        assert agent2.train_steps == agent.train_steps
        assert agent2.epsilon == agent.epsilon
        s = np.ones(STATE_DIM, dtype=np.float32)
        assert agent.act(s, greedy=True) == agent2.act(s, greedy=True)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(TradingAgentError, match="No existe"):
            make_agent().load(tmp_path / "nope.pt")

    def test_load_incompatible_dims_raises(self, tmp_path):
        agent = make_agent()
        path = tmp_path / "ckpt.pt"
        agent.save(path)
        other = DQNAgent(AgentConfig(state_dim=STATE_DIM + 1,
                                     hidden_sizes=(16,), batch_size=8,
                                     min_buffer_size=8, buffer_capacity=100))
        with pytest.raises(TradingAgentError, match="incompatible"):
            other.load(path)
