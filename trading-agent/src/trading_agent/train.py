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
from .features import FEATURE_COLUMNS, build_features


def checkpoint_metadata(config: AppConfig, symbol: str) -> dict[str, object]:
    """Metadatos que fijan la compatibilidad de un checkpoint.

    Se guardan junto a los pesos y se verifican al cargar: un modelo
    entrenado con otra ventana, otras features u otro intervalo produciría
    decisiones sin sentido de forma silenciosa.

    Args:
        config: configuración de la aplicación.
        symbol: símbolo con el que se entrena/opera.

    Returns:
        Dict serializable con ``window_size``, ``feature_columns``,
        ``interval``, ``symbol`` y ``hidden_sizes``.
    """
    return {
        "window_size": config.env.window_size,
        "feature_columns": list(FEATURE_COLUMNS),
        "interval": config.data.interval,
        "symbol": symbol,
        "hidden_sizes": list(config.agent.hidden_sizes),
    }

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


def three_way_split(df: pd.DataFrame, train_fraction: float = 0.7,
                    val_fraction: float = 0.15
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el histórico en train / validación / test por tiempo.

    El tramo de **test** no interviene ni en el entrenamiento ni en la
    selección del checkpoint: es la única estimación honesta del
    rendimiento fuera de muestra (la validación, al usarse para elegir el
    mejor modelo, está sesgada al alza).

    Args:
        df: DataFrame OHLCV canónico ordenado por tiempo.
        train_fraction: fracción inicial para entrenamiento, en (0, 1).
        val_fraction: fracción siguiente para validación, en (0, 1); el
            resto (``1 - train - val``) es test y debe ser > 0.

    Returns:
        Tupla ``(train_df, val_df, test_df)``, estrictamente consecutivos
        en el tiempo.

    Raises:
        TradingAgentError: si las fracciones son inválidas o algún tramo
            queda vacío.
    """
    if not (0.0 < train_fraction < 1.0 and 0.0 < val_fraction < 1.0
            and train_fraction + val_fraction < 1.0):
        raise TradingAgentError(
            "Se requiere 0 < train_fraction, val_fraction y "
            "train_fraction + val_fraction < 1")
    n = len(df)
    cut_train = int(n * train_fraction)
    cut_val = int(n * (train_fraction + val_fraction))
    parts = (df.iloc[:cut_train], df.iloc[cut_train:cut_val], df.iloc[cut_val:])
    if any(p.empty for p in parts):
        raise TradingAgentError("Split temporal dejó un tramo vacío")
    return parts


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
        next_state, reward, done, info = env.step(action)
        if training:
            # Solo un fin REAL del MDP (kill-switch) anula el valor futuro;
            # agotar los datos es un truncamiento artificial del episodio.
            agent.remember(state, action, reward, next_state,
                           bool(info["terminated"]))
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
        Métricas finales:
            - ``best_val_return``: retorno de validación del mejor checkpoint.
            - ``episodes_run``: nº de episodios ejecutados.
            - ``test_return``: retorno del mejor checkpoint en el tramo de
              **test** (fuera de muestra, nunca usado para elegir).
            - ``test_buy_and_hold``: benchmark comprar-y-mantener en test.
            - ``test_sharpe`` y ``test_max_drawdown``: métricas de riesgo
              en test (ver :meth:`TradingEnvironment.episode_summary`).

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

    # Features UNA vez sobre todo el histórico (indicadores causales: sin
    # fuga) y luego split temporal: cada tramo conserva su calentamiento.
    features = build_features(ohlcv)
    train_df, val_df, test_df = three_way_split(ohlcv)
    train_env = TradingEnvironment(train_df, config.env, config.risk, symbol,
                                   features=features)
    val_env = TradingEnvironment(val_df, config.env, config.risk, symbol,
                                 features=features)
    test_env = TradingEnvironment(test_df, config.env, config.risk, symbol,
                                  features=features)
    checkpoint_meta = checkpoint_metadata(config, symbol)

    # El state_dim real depende de window_size y nº de features: se fija
    # aquí, no a mano en la config (una fuente de verdad: el entorno).
    agent_cfg = dataclasses.replace(config.agent, state_dim=train_env.state_dim)
    agent = DQNAgent(agent_cfg)

    best_val_return = float("-inf")
    for episode in range(1, episodes + 1):
        train_summary = run_episode(train_env, agent, training=True)
        val_summary = run_episode(val_env, agent, training=False)

        logger.info(
            "Ep %3d/%d | train ret %+7.2f%% | val ret %+7.2f%% "
            "(B&H %+6.2f%%) | trades %3.0f | eps %.3f | loss %.4f",
            episode, episodes,
            train_summary["total_return_pct"] * 100,
            val_summary["total_return_pct"] * 100,
            val_summary["buy_and_hold_return_pct"] * 100,
            train_summary["n_trades"], agent.epsilon,
            train_summary["avg_loss"])

        if val_summary["total_return_pct"] > best_val_return:
            best_val_return = val_summary["total_return_pct"]
            agent.save(checkpoint_path, metadata=checkpoint_meta)

    # Evaluación final fuera de muestra con el MEJOR checkpoint (no con el
    # último): el tramo de test nunca influyó en entrenar ni en elegir.
    agent.load(checkpoint_path, expected_metadata=checkpoint_meta)
    test_summary = run_episode(test_env, agent, training=False)
    logger.info(
        "Entrenamiento terminado. Mejor val %+.2f%% | TEST ret %+.2f%% vs "
        "B&H %+.2f%% | Sharpe %.2f | max DD %.1f%% | trades %.0f",
        best_val_return * 100,
        test_summary["total_return_pct"] * 100,
        test_summary["buy_and_hold_return_pct"] * 100,
        test_summary["sharpe"], test_summary["max_drawdown_pct"] * 100,
        test_summary["n_trades"])
    return {
        "best_val_return": best_val_return,
        "episodes_run": float(episodes),
        "test_return": test_summary["total_return_pct"],
        "test_buy_and_hold": test_summary["buy_and_hold_return_pct"],
        "test_sharpe": test_summary["sharpe"],
        "test_max_drawdown": test_summary["max_drawdown_pct"],
    }
