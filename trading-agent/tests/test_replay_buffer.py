"""Tests del replay buffer circular."""

from __future__ import annotations

import numpy as np
import pytest

from trading_agent.agent.replay_buffer import ReplayBuffer
from trading_agent.exceptions import EnvironmentError_


def fill(buf: ReplayBuffer, n: int, dim: int = 4) -> None:
    for i in range(n):
        s = np.full(dim, float(i), dtype=np.float32)
        buf.push(s, i % 3, float(i), s + 1, done=(i % 10 == 9))


class TestPushAndSize:
    def test_grows_until_capacity(self):
        buf = ReplayBuffer(capacity=5, state_dim=4)
        fill(buf, 3)
        assert len(buf) == 3
        fill(buf, 10)
        assert len(buf) == 5  # nunca supera la capacidad

    def test_fifo_overwrites_oldest(self):
        buf = ReplayBuffer(capacity=3, state_dim=1, seed=0)
        for i in range(5):  # tras 5 push quedan las transiciones 2, 3, 4
            buf.push(np.array([float(i)], dtype=np.float32), 0, 0.0,
                     np.zeros(1, dtype=np.float32), False)
        batch = buf.sample(3)
        assert set(batch.states.reshape(-1).tolist()) <= {2.0, 3.0, 4.0}


class TestSample:
    def test_batch_shapes_and_dtypes(self):
        buf = ReplayBuffer(capacity=100, state_dim=4)
        fill(buf, 50)
        b = buf.sample(16)
        assert b.states.shape == (16, 4)
        assert b.next_states.shape == (16, 4)
        assert b.actions.shape == (16,) and b.actions.dtype == np.int64
        assert b.rewards.dtype == np.float32
        assert set(np.unique(b.dones)) <= {0.0, 1.0}

    def test_sample_more_than_stored_raises(self):
        buf = ReplayBuffer(capacity=100, state_dim=4)
        fill(buf, 5)
        with pytest.raises(EnvironmentError_):
            buf.sample(10)

    def test_sampling_is_reproducible_with_seed(self):
        b1, b2 = ReplayBuffer(10, 2, seed=42), ReplayBuffer(10, 2, seed=42)
        fill(b1, 10, dim=2)
        fill(b2, 10, dim=2)
        np.testing.assert_array_equal(b1.sample(4).actions,
                                      b2.sample(4).actions)


class TestConstruction:
    @pytest.mark.parametrize("cap,dim", [(0, 4), (10, 0), (-1, 4)])
    def test_invalid_params_raise(self, cap, dim):
        with pytest.raises(EnvironmentError_):
            ReplayBuffer(capacity=cap, state_dim=dim)
