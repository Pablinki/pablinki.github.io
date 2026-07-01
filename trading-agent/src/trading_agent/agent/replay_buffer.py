"""Replay buffer de experiencias para el DQN.

¿Por qué existe? El DQN aprende de minibatches de transiciones pasadas
muestreadas al azar. Esto rompe la correlación temporal de las
experiencias consecutivas (los mercados son series temporales muy
correlacionadas) y reutiliza cada experiencia muchas veces, lo que
estabiliza y acelera el aprendizaje.

Estructura de datos
-------------------
Almacenamiento en **arrays de numpy preasignados** (no una lista de
tuplas): con capacidad 100k y estados de ~200 floats, los arrays
contiguos ahorran memoria y hacen el muestreo O(batch) con indexado
vectorizado. El buffer es circular (FIFO): al llenarse, las experiencias
nuevas sobreescriben las más antiguas.

Cada transición es la tupla clásica de RL::

    (state, action, reward, next_state, done)

Ejemplo
-------
>>> import numpy as np
>>> buf = ReplayBuffer(capacity=10, state_dim=4, seed=0)
>>> s = np.zeros(4, dtype=np.float32)
>>> for i in range(3):
...     buf.push(s, 1, 0.5, s, False)
>>> len(buf)
3
>>> batch = buf.sample(2)
>>> batch.states.shape
(2, 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import EnvironmentError_


@dataclass(frozen=True)
class Batch:
    """Minibatch de transiciones listo para convertir a tensores.

    Attributes:
        states: array float32 de forma ``(batch, state_dim)``.
        actions: array int64 de forma ``(batch,)``.
        rewards: array float32 de forma ``(batch,)``.
        next_states: array float32 de forma ``(batch, state_dim)``.
        dones: array float32 de forma ``(batch,)`` (1.0 = terminal).
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Buffer circular de transiciones con muestreo uniforme."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        """
        Args:
            capacity: nº máximo de transiciones retenidas (> 0).
            state_dim: dimensión del vector de estado.
            seed: semilla del generador de muestreo (reproducibilidad).
        """
        if capacity <= 0 or state_dim <= 0:
            raise EnvironmentError_("capacity y state_dim deben ser > 0")
        self.capacity = capacity
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0          # posición de escritura (circular)
        self._size = 0         # nº de transiciones válidas almacenadas
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Nº de transiciones almacenadas actualmente."""
        return self._size

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Añade una transición (sobrescribe la más antigua si está lleno).

        Args:
            state: estado observado, forma ``(state_dim,)``.
            action: acción tomada (entero).
            reward: recompensa recibida.
            next_state: estado resultante, forma ``(state_dim,)``.
            done: ``True`` si ``next_state`` es terminal.
        """
        self._states[self._idx] = state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._next_states[self._idx] = next_state
        self._dones[self._idx] = float(done)
        # Aritmética modular: al llegar al final vuelve al principio (FIFO).
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Muestrea uniformemente ``batch_size`` transiciones con reemplazo.

        Args:
            batch_size: tamaño del minibatch (> 0).

        Returns:
            :class:`Batch` con arrays alineados por fila.

        Raises:
            EnvironmentError_: si el buffer tiene menos transiciones que
                ``batch_size`` (entrenar con repetidos degenera).
        """
        if batch_size <= 0 or batch_size > self._size:
            raise EnvironmentError_(
                f"batch_size={batch_size} inválido con {self._size} transiciones")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=self._states[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_states=self._next_states[idx],
            dones=self._dones[idx],
        )
