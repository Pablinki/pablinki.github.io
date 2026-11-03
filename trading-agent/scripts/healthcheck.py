#!/usr/bin/env python3
"""Comprobación de salud del trader a partir de su heartbeat.

Pensado para ``HEALTHCHECK`` de Docker o un supervisor externo: un proceso
puede seguir "vivo" (PID activo) y haber dejado de operar (bucle colgado,
datos que fallan sin parar). El heartbeat se reescribe en cada iteración;
si envejece más de ``--max-age`` segundos, el trader no está sano.

Uso::

    python scripts/healthcheck.py --heartbeat logs/heartbeat.json --max-age 300

Salida: código 0 si está sano; 1 si el heartbeat falta, es ilegible, es
demasiado viejo, el kill-switch está activo o los errores seguidos superan
``--max-errors``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def check(heartbeat_path: str | Path, *, max_age_s: float,
          max_errors: int, now: dt.datetime | None = None) -> tuple[bool, str]:
    """Evalúa el heartbeat.

    Args:
        heartbeat_path: ruta del JSON escrito por el trader.
        max_age_s: antigüedad máxima tolerada en segundos.
        max_errors: errores inesperados seguidos tolerados.
        now: instante actual UTC (inyectable en tests).

    Returns:
        Tupla ``(sano, motivo)``.
    """
    path = Path(heartbeat_path)
    if not path.exists():
        return False, f"sin heartbeat en {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        written = dt.datetime.fromisoformat(data["utc"])
        if written.tzinfo is None:  # sin zona: se asume UTC
            written = written.replace(tzinfo=dt.timezone.utc)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"heartbeat ilegible: {exc}"
    now = now or dt.datetime.now(dt.timezone.utc)
    age = (now - written).total_seconds()
    if age > max_age_s:
        return False, f"heartbeat de hace {age:.0f}s (> {max_age_s:.0f}s)"
    if data.get("halted"):
        return False, "kill-switch de drawdown activo"
    if int(data.get("consecutive_errors") or 0) > max_errors:
        return False, f"{data['consecutive_errors']} errores seguidos"
    return True, f"ok (iteración {data.get('iteration')}, hace {age:.0f}s)"


def main() -> int:
    """Punto de entrada del CLI. Returns: 0 sano, 1 no sano."""
    parser = argparse.ArgumentParser(description="Salud del trader")
    parser.add_argument("--heartbeat", default="logs/heartbeat.json")
    parser.add_argument("--max-age", type=float, default=300.0,
                        help="Segundos máximos desde el último heartbeat "
                             "(usa >= 3x --poll-seconds del trader)")
    parser.add_argument("--max-errors", type=int, default=3)
    args = parser.parse_args()
    healthy, why = check(args.heartbeat, max_age_s=args.max_age,
                         max_errors=args.max_errors)
    print(why)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
