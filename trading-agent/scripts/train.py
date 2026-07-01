#!/usr/bin/env python3
"""CLI de entrenamiento del trading agent.

Uso::

    python scripts/train.py --config config.yaml --episodes 50 \
        --checkpoint checkpoints/dqn_best.pt

Entradas:
    --config: ruta al YAML de configuración (defecto: config.yaml).
    --episodes: nº de episodios de entrenamiento.
    --checkpoint: destino del mejor modelo.

Salida:
    Código 0 y el checkpoint en disco si todo va bien; código 1 con el
    error registrado si falla algo controlado.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete (modo desarrollo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trading_agent.config import load_config          # noqa: E402
from trading_agent.exceptions import TradingAgentError  # noqa: E402
from trading_agent.train import train                  # noqa: E402


def main() -> int:
    """Punto de entrada del CLI. Returns: código de salida del proceso."""
    parser = argparse.ArgumentParser(description="Entrena el DQN de trading")
    parser.add_argument("--config", default="config.yaml",
                        help="Ruta al YAML de configuración")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Nº de episodios de entrenamiento")
    parser.add_argument("--checkpoint", default="checkpoints/dqn_best.pt",
                        help="Destino del mejor checkpoint")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        metrics = train(config, episodes=args.episodes,
                        checkpoint_path=args.checkpoint)
        logging.info("Métricas finales: %s", metrics)
        return 0
    except TradingAgentError as exc:
        logging.error("El entrenamiento falló de forma controlada: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
