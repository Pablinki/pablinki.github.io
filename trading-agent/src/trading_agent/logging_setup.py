"""Configuración de logging para un proceso 24/7.

Consola (para ``docker logs`` / systemd) **y** archivo con rotación diaria
(``logs/trader.log``, 14 días retenidos): la consola se pierde; el archivo
permite reconstruir qué hizo el agente en cada iteración.

Ejemplo
-------
>>> import tempfile, os, logging
>>> path = os.path.join(tempfile.mkdtemp(), "trader.log")
>>> configure_logging("INFO", path)
>>> logging.getLogger("demo").info("hola")
>>> "hola" in open(path).read()
True
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

#: Formato común a consola y archivo.
FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO",
                      log_file: str | Path | None = None,
                      backup_days: int = 14) -> None:
    """Configura el logger raíz: consola siempre, archivo rotativo opcional.

    Idempotente: reemplaza los manejadores existentes (útil en tests y
    si se llama dos veces).

    Args:
        level: nivel textual (``"DEBUG"``, ``"INFO"``, ``"WARNING"``...).
        log_file: ruta del archivo de log; ``None`` = solo consola. Se
            crean los directorios padre. Rota a medianoche (UTC del
            proceso) conservando ``backup_days`` archivos.
        backup_days: nº de archivos diarios retenidos.

    Raises:
        ValueError: si ``level`` no es un nivel de logging válido.
    """
    levels = logging.getLevelNamesMapping()
    if level.upper() not in levels:
        raise ValueError(f"Nivel de logging inválido: {level!r}; "
                         f"use uno de {sorted(levels)}")
    numeric = levels[level.upper()]

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(numeric)
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            path, when="midnight", backupCount=backup_days, encoding="utf-8",
            utc=True)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
