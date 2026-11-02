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

Seguridad (tres niveles):
    1. ``--broker paper`` (defecto): simulación local, sin broker.
    2. ``--broker alpaca``: órdenes reales contra el entorno PAPER de
       Alpaca (dinero ficticio) salvo que se cambie ALPACA_BASE_URL.
    3. Dinero real: además de ALPACA_BASE_URL=https://api.alpaca.markets
       exige TRADING_AGENT_LIVE=YES; sin ambas, el arranque falla.
    El estado de riesgo (pico de equity, kill-switch) persiste en
    ``--risk-state`` para que un reinicio no lo rearme solo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trading_agent.config import load_config
from trading_agent.exceptions import TradingAgentError
from trading_agent.live import build_live_trader


def main() -> int:
    """Punto de entrada del CLI. Returns: código de salida del proceso."""
    parser = argparse.ArgumentParser(
        description="Trading en tiempo real (paper) con un DQN entrenado")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/dqn_best.pt")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="0 = sin límite")
    parser.add_argument("--broker", default="paper",
                        choices=["paper", "alpaca"],
                        help="Ejecutor de órdenes: 'paper' simula en local; "
                             "'alpaca' usa la API de Alpaca (paper de Alpaca "
                             "por defecto; dinero real solo con "
                             "TRADING_AGENT_LIVE=YES y ALPACA_BASE_URL live)")
    parser.add_argument("--journal", default="logs/trades.csv",
                        help="Diario CSV de operaciones (cadena vacía para "
                             "desactivarlo)")
    parser.add_argument("--risk-state", default="logs/risk_state.json",
                        help="Estado persistente del gestor de riesgo "
                             "(cadena vacía = solo memoria)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    try:
        config = load_config(args.config)
        trader = build_live_trader(config, args.checkpoint,
                                   broker_name=args.broker,
                                   journal_path=args.journal or None,
                                   risk_state_path=args.risk_state or None)
        trader.run_forever(
            poll_seconds=args.poll_seconds,
            max_iterations=args.max_iterations or None)
        return 0
    except TradingAgentError as exc:
        logging.getLogger(__name__).error(
            "El trader terminó con error controlado: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
