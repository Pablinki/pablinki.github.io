"""Agente DQN (Deep Q-Network) con Double DQN y red objetivo.

Algoritmo
---------
DQN (Mnih et al., 2015) con dos mejoras estándar:

1. **Red objetivo**: una copia congelada de la red Q que se sincroniza
   cada ``target_update_every`` pasos. Sin ella, el "objetivo" del
   aprendizaje se movería con cada gradiente y el entrenamiento diverge.
2. **Double DQN** (van Hasselt et al., 2016): la red *online* elige la
   mejor acción del siguiente estado y la red *objetivo* la valora.
   Reduce la sobreestimación sistemática de valores Q — crítica en
   trading, donde el sobreoptimismo se traduce en sobreoperar.

Objetivo de aprendizaje (por transición)::

    y = r                                     si done
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a))   si no

Pérdida: Smooth L1 (Huber) entre ``Q_online(s, a)`` e ``y`` — más robusta
a outliers de recompensa que el MSE.

Exploración: política epsilon-greedy con decaimiento exponencial por
episodio (``epsilon_start -> epsilon_end`` multiplicando por
``epsilon_decay``).

Ejemplo
-------
>>> import numpy as np
>>> from trading_agent.config import AgentConfig
>>> cfg = AgentConfig(state_dim=8, min_buffer_size=4, batch_size=4,
...                   buffer_capacity=100)
>>> agente = DQNAgent(cfg)
>>> s = np.zeros(8, dtype=np.float32)
>>> accion = agente.act(s)          # entero en {0, 1, 2}
>>> accion in (0, 1, 2)
True
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import AgentConfig
from ..exceptions import TradingAgentError
from .networks import QNetwork
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """Fija todas las semillas (random, numpy, torch) para reproducibilidad.

    Args:
        seed: semilla entera.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DQNAgent:
    """Agente Double-DQN con replay buffer y red objetivo.

    Ciclo de uso:
        1. :meth:`act` para decidir la acción en cada paso.
        2. :meth:`remember` para almacenar la transición observada.
        3. :meth:`learn` para dar un paso de gradiente (si hay datos).
        4. :meth:`end_episode` al terminar cada episodio (decae epsilon).
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Args:
            config: hiperparámetros validados; ``config.state_dim`` debe
                coincidir con el ``state_dim`` del entorno.
        """
        self.config = config
        set_global_seeds(config.seed)
        self.device = torch.device(config.device)

        self.online = QNetwork(config.state_dim, config.n_actions,
                               config.hidden_sizes).to(self.device)
        self.target = QNetwork(config.state_dim, config.n_actions,
                               config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()  # la red objetivo nunca entrena directamente

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(config.buffer_capacity, config.state_dim,
                                   seed=config.seed)
        self.epsilon = config.epsilon_start
        self._rng = random.Random(config.seed)
        self.train_steps = 0  # nº de pasos de gradiente ejecutados

    # ------------------------------------------------------------------ #
    # Política                                                           #
    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
        """Elige una acción con política epsilon-greedy.

        Args:
            state: vector de estado float32 de forma ``(state_dim,)``.
            greedy: si ``True`` ignora epsilon y siempre explota (usar en
                evaluación y en trading real; explorar con dinero real
                sería regalar operaciones aleatorias).

        Returns:
            Índice entero de la acción elegida, en ``[0, n_actions)``.

        Raises:
            TradingAgentError: si la forma del estado no coincide con la red.
        """
        if state.shape != (self.config.state_dim,):
            raise TradingAgentError(
                f"Estado de forma {state.shape}; se esperaba "
                f"({self.config.state_dim},)")
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.config.n_actions)
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q_values = self.online(t)          # forma (1, n_actions)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Aprendizaje                                                        #
    # ------------------------------------------------------------------ #
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Almacena una transición en el replay buffer (ver firma en
        :meth:`ReplayBuffer.push`)."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """Ejecuta un paso de gradiente sobre un minibatch del buffer.

        Returns:
            La pérdida (float) del paso, o ``None`` si el buffer todavía
            no alcanza ``min_buffer_size`` (fase de calentamiento).
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return None

        batch = self.buffer.sample(self.config.batch_size)
        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)

        # Q(s, a) de las acciones realmente tomadas: gather selecciona por
        # columna el valor Q de la acción de cada fila del batch.
        q_sa = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: online ELIGE, target VALORA.
                best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            # (1 - dones) anula el valor futuro en estados terminales.
            targets = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Recorte de gradiente: evita explosiones ante recompensas atípicas.
        nn.utils.clip_grad_norm_(self.online.parameters(),
                                 self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
            logger.debug("Red objetivo sincronizada en el paso %d",
                         self.train_steps)
        return float(loss.item())

    def end_episode(self) -> None:
        """Decae epsilon al cerrar un episodio (menos exploración con el
        tiempo, nunca por debajo de ``epsilon_end``)."""
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistencia                                                       #
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Guarda los pesos y el estado de entrenamiento en disco.

        Args:
            path: ruta destino del checkpoint (``.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "state_dim": self.config.state_dim,
            "n_actions": self.config.n_actions,
        }, path)
        logger.info("Checkpoint guardado en %s", path)

    def load(self, path: str | Path) -> None:
        """Restaura un checkpoint guardado con :meth:`save`.

        Args:
            path: ruta del checkpoint.

        Raises:
            TradingAgentError: si el archivo no existe o las dimensiones
                del checkpoint no coinciden con la configuración actual
                (cargar pesos incompatibles produciría basura silenciosa).
        """
        path = Path(path)
        if not path.is_file():
            raise TradingAgentError(f"No existe el checkpoint {path}")
        # weights_only=True: no deserializa objetos arbitrarios (seguridad).
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if (ckpt.get("state_dim") != self.config.state_dim
                or ckpt.get("n_actions") != self.config.n_actions):
            raise TradingAgentError(
                f"Checkpoint incompatible: state_dim/n_actions del archivo "
                f"({ckpt.get('state_dim')}/{ckpt.get('n_actions')}) != "
                f"config ({self.config.state_dim}/{self.config.n_actions})")
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt["epsilon"])
        self.train_steps = int(ckpt["train_steps"])
        logger.info("Checkpoint cargado desde %s (pasos=%d, eps=%.3f)",
                    path, self.train_steps, self.epsilon)
