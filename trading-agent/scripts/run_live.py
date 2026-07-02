#!/usr/bin/env python3
"""CLI de trading en (casi) tiempo real — paper trading por defecto.

Uso::

    python scripts/run_live.py --config config.yaml \
        --checkpoint checkpoints/dqn_best.pt --poll-seconds 60

Entradas:
    --config: ruta al YAML de configuración.
    --checkpoint: modelo entrenado a cargar.
    --poll-seconds: segundos entre decisiones.
    --max-iterations: tope de iteraciones (0 = infinito).

Seguridad:
    Este script SIEMPRE opera en modo paper (simulado). Conectar un broker
    real exige además exportar TRADING_AGENT_LIVE=YES y escribir el
    adaptador de ejecución correspondiente.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trading_agent.config import load_config            # noqa: E402
from trading_agent.exceptions import TradingAgentError  # noqa: E402
from trading_agent.live import build_live_trader        # noqa: E402


def main() -> int:
    """Punto de entrada del CLI. Returns: código de salida del proceso."""
    parser = argparse.ArgumentParser(
        description="Trading en tiempo real (paper) con un DQN entrenado")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/dqn_best.pt")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="0 = sin límite")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        trader = build_live_trader(config, args.checkpoint)
        trader.run_forever(
            poll_seconds=args.poll_seconds,
            max_iterations=args.max_iterations or None)
        return 0
    except TradingAgentError as exc:
        logging.error("El trader terminó con error controlado: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
