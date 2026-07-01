"""Red neuronal Q para el DQN.

La red aproxima la función de valor acción-estado ``Q(s, a)``: recibe el
vector de estado y devuelve un valor Q por cada acción discreta. La
acción "greedy" es el argmax de esa salida.

Arquitectura: perceptrón multicapa (MLP) totalmente parametrizado —
``hidden_sizes=(128, 64)`` produce::

    state_dim -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(n_actions)

Para un estado de mercado tabular/ventana (no imágenes) un MLP es la
arquitectura estándar y suficiente.

Ejemplo
-------
>>> import torch
>>> net = QNetwork(state_dim=8, n_actions=3, hidden_sizes=(16,))
>>> q = net(torch.zeros(5, 8))   # batch de 5 estados
>>> q.shape
torch.Size([5, 3])
"""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP que mapea estados a valores Q por acción."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden_sizes: tuple[int, ...]) -> None:
        """
        Args:
            state_dim: dimensión del vector de estado de entrada.
            n_actions: nº de acciones (dimensión de salida).
            hidden_sizes: anchura de cada capa oculta, en orden.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        # Bucle de construcción: una pareja Linear+ReLU por capa oculta.
        for width in hidden_sizes:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, n_actions))  # capa de salida (sin activación)
        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propagación hacia delante.

        Args:
            state: tensor float32 de forma ``(batch, state_dim)``.

        Returns:
            Tensor de valores Q de forma ``(batch, n_actions)``.
        """
        return self.model(state)
