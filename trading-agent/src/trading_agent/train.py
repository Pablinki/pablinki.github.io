"""Bucle de entrenamiento del DQN sobre datos históricos.

Flujo completo (:func:`train`):

1. Descargar histórico del proveedor configurado (Yahoo/Bloomberg).
2. Partir en train/validación **por tiempo** (nunca aleatorio: mezclar
   el futuro con el pasado es *look-ahead bias* y produce backtests
   fraudulentamente buenos).
3. Bucle de episodios: en cada episodio el agente recorre el tramo de
   entrenamiento completo, almacenando transiciones y aprendiendo.
4. Al final de cada episodio se evalúa en validación con política
   greedy (sin explorar) y se guarda el mejor checkpoint.

Ejemplo de uso (CLI)::

    python scripts/train.py --config config.yaml --episodes 50
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pandas as pd

from .agent.dqn import DQNAgent
from .config import AppConfig
from .data.provider import make_provider
from .env import TradingEnvironment
from .exceptions import TradingAgentError

logger = logging.getLogger(__name__)


def time_split(df: pd.DataFrame, train_fraction: float = 0.8
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide un histórico en train/validación respetando el orden temporal.

    Args:
        df: DataFrame OHLCV canónico ordenado por tiempo.
        train_fraction: fracción inicial destinada a entrenamiento, en (0, 1).

    Returns:
        Tupla ``(train_df, val_df)``; ``val_df`` es estrictamente
        posterior a ``train_df``.

    Raises:
        TradingAgentError: si la fracción es inválida o algún tramo queda
            vacío.
    """
    if not 0.0 < train_fraction < 1.0:
        raise TradingAgentError("train_fraction debe estar en (0, 1)")
    cut = int(len(df) * train_fraction)
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]
    if train_df.empty or val_df.empty:
        raise TradingAgentError("Split temporal dejó un tramo vacío")
    return train_df, val_df


def run_episode(env: TradingEnvironment, agent: DQNAgent, *,
                training: bool) -> dict[str, float]:
    """Ejecuta un episodio completo del agente en el entorno.

    Este es el bucle canónico de interacción de RL:

    - ``training=True``: política epsilon-greedy, se almacenan
      transiciones y se aprende en cada paso.
    - ``training=False``: política greedy pura, sin efectos sobre el
      agente (evaluación limpia).

    Args:
        env: entorno ya construido.
        agent: agente DQN.
        training: modo entrenamiento vs. evaluación.

    Returns:
        Resumen del episodio (ver :meth:`TradingEnvironment.episode_summary`)
        con la clave extra ``avg_loss`` (media de las pérdidas del episodio,
        0.0 en evaluación o durante el calentamiento del buffer).
    """
    state = env.reset()
    losses: list[float] = []
    done = False
    while not done:
        action = agent.act(state, greedy=not training)
        next_state, reward, done, _info = env.step(action)
        if training:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        state = next_state
    if training:
        agent.end_episode()
    summary = env.episode_summary()
    summary["avg_loss"] = sum(losses) / len(losses) if losses else 0.0
    return summary


def train(config: AppConfig, *, episodes: int = 50,
          checkpoint_path: str | Path = "checkpoints/dqn_best.pt",
          ohlcv: pd.DataFrame | None = None) -> dict[str, float]:
    """Entrena el DQN de principio a fin y guarda el mejor checkpoint.

    Args:
        config: configuración completa de la aplicación.
        episodes: nº de episodios de entrenamiento (> 0).
        checkpoint_path: destino del mejor modelo según retorno de
            validación.
        ohlcv: histórico opcional ya descargado (lo usan los tests para
            no depender de la red); si es ``None`` se descarga del
            proveedor configurado.

    Returns:
        Métricas finales: ``best_val_return`` (retorno de validación del
        mejor checkpoint) y ``episodes_run``.

    Raises:
        TradingAgentError (o subclases): ante datos insuficientes,
            configuración inválida, etc.
    """
    if episodes <= 0:
        raise TradingAgentError("episodes debe ser > 0")

    symbol = config.data.symbols[0]
    if ohlcv is None:
        provider = make_provider(config.data)
        ohlcv = provider.fetch_historical(symbol)

    train_df, val_df = time_split(ohlcv)
    train_env = TradingEnvironment(train_df, config.env, config.risk, symbol)
    val_env = TradingEnvironment(val_df, config.env, config.risk, symbol)

    # El state_dim real depende de window_size y nº de features: se fija
    # aquí, no a mano en la config (una fuente de verdad: el entorno).
    agent_cfg = dataclasses.replace(config.agent, state_dim=train_env.state_dim)
    agent = DQNAgent(agent_cfg)

    best_val_return = float("-inf")
    for episode in range(1, episodes + 1):
        train_summary = run_episode(train_env, agent, training=True)
        val_summary = run_episode(val_env, agent, training=False)

        logger.info(
            "Ep %3d/%d | train ret %+7.2f%% | val ret %+7.2f%% | "
            "trades %3.0f | eps %.3f | loss %.4f",
            episode, episodes,
            train_summary["total_return_pct"] * 100,
            val_summary["total_return_pct"] * 100,
            train_summary["n_trades"], agent.epsilon,
            train_summary["avg_loss"])

        if val_summary["total_return_pct"] > best_val_return:
            best_val_return = val_summary["total_return_pct"]
            agent.save(checkpoint_path)

    logger.info("Entrenamiento terminado. Mejor retorno de validación: %+.2f%%",
                best_val_return * 100)
    return {"best_val_return": best_val_return, "episodes_run": float(episodes)}
