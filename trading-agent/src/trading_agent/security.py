"""Utilidades de seguridad del trading agent.

Principios aplicados
--------------------
1. **Nunca credenciales en código ni en el repositorio**: las claves de API
   (Bloomberg, brokers) se leen exclusivamente de variables de entorno.
2. **Fail-fast**: si falta una credencial requerida se lanza
   :class:`~trading_agent.exceptions.SecurityError` al arrancar, no a mitad
   de una sesión de trading.
3. **Sin fugas en logs**: :func:`mask_secret` permite loguear que una
   credencial existe sin exponer su valor.
4. **Modo vivo explícito**: operar con dinero real exige la variable
   ``TRADING_AGENT_LIVE=YES`` — un despliegue accidental queda en paper
   trading por defecto (principio de mínimo privilegio).

Ejemplo
-------
>>> import os
>>> os.environ["MI_CLAVE"] = "super-secreta-123"
>>> get_secret("MI_CLAVE")
'super-secreta-123'
>>> mask_secret("super-secreta-123")
'sup***********123'
"""

from __future__ import annotations

import os

from .exceptions import SecurityError

#: Variable de entorno que habilita el trading con dinero real.
LIVE_TRADING_ENV_VAR = "TRADING_AGENT_LIVE"


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Lee una credencial desde una variable de entorno.

    Args:
        name: nombre de la variable de entorno, p. ej. ``"BLOOMBERG_HOST"``.
        required: si ``True`` (defecto), su ausencia es un error de
            seguridad; si ``False`` devuelve ``None`` cuando no existe.

    Returns:
        El valor de la credencial, o ``None`` si no existe y no es requerida.

    Raises:
        SecurityError: si la credencial es requerida y no está definida
            o está vacía.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise SecurityError(
                f"Falta la variable de entorno {name!r}. "
                "Defínela en el entorno, nunca en el código fuente.")
        return None
    return value


def mask_secret(value: str, *, visible: int = 3) -> str:
    """Enmascara un secreto para poder loguearlo sin exponerlo.

    Args:
        value: el secreto en claro.
        visible: nº de caracteres visibles al inicio y al final.

    Returns:
        Cadena con el centro sustituido por ``*``. Si el secreto es muy
        corto se enmascara por completo.

    Ejemplo:
        >>> mask_secret("abcdefghij", visible=2)
        'ab******ij'
        >>> mask_secret("abc")
        '***'
    """
    if len(value) <= 2 * visible:
        return "*" * len(value)
    hidden = len(value) - 2 * visible
    return value[:visible] + "*" * hidden + value[-visible:]


def is_live_trading_enabled() -> bool:
    """Indica si el operador habilitó explícitamente el trading en vivo.

    Returns:
        ``True`` solo si ``TRADING_AGENT_LIVE`` vale exactamente ``"YES"``.
        Cualquier otro valor (o su ausencia) mantiene el modo paper.
    """
    return os.environ.get(LIVE_TRADING_ENV_VAR, "") == "YES"


def assert_live_trading_allowed() -> None:
    """Aborta si se intenta operar en vivo sin autorización explícita.

    Raises:
        SecurityError: si :func:`is_live_trading_enabled` es ``False``.
    """
    if not is_live_trading_enabled():
        raise SecurityError(
            "Trading en vivo deshabilitado. Exporta "
            f"{LIVE_TRADING_ENV_VAR}=YES para habilitarlo conscientemente; "
            "mientras tanto solo se permite paper trading.")
