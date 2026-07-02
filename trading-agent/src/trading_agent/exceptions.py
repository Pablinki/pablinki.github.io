"""Jerarquía de excepciones del trading agent.

Todas las excepciones del sistema heredan de :class:`TradingAgentError`,
lo que permite capturar cualquier error del agente con un solo bloque::

    try:
        agente.ejecutar()
    except TradingAgentError as exc:
        logger.error("Fallo controlado: %s", exc)

Diseño
------
- Cada capa del sistema (datos, entorno, riesgo, ejecución) tiene su propia
  excepción, de modo que el llamador puede decidir la política de reintento
  por tipo de fallo (p. ej. reintentar ``DataProviderError`` con backoff,
  pero abortar inmediatamente ante ``RiskLimitExceededError``).
"""

from __future__ import annotations


class TradingAgentError(Exception):
    """Excepción base de todo el sistema. Nunca se lanza directamente."""


class ConfigurationError(TradingAgentError):
    """Configuración inválida (parámetro fuera de rango, campo faltante).

    Ejemplo:
        >>> raise ConfigurationError("epsilon_decay debe estar en (0, 1]")
        Traceback (most recent call last):
        ...
        trading_agent.exceptions.ConfigurationError: epsilon_decay debe estar en (0, 1]
    """


class DataProviderError(TradingAgentError):
    """Fallo al obtener datos de mercado (red caída, símbolo inexistente,
    respuesta vacía del proveedor). Normalmente es transitorio y se
    reintenta con backoff exponencial."""


class DataValidationError(DataProviderError):
    """Los datos llegaron pero no pasan validación (NaN, precios <= 0,
    huecos temporales excesivos). No tiene sentido reintentar sin
    intervención: los datos de origen están corruptos."""


class EnvironmentError_(TradingAgentError):
    """Error del entorno de simulación (acción inválida, episodio ya
    terminado). El sufijo ``_`` evita colisión con ``builtins.EnvironmentError``."""


class RiskLimitExceededError(TradingAgentError):
    """Una orden violaría un límite de riesgo (drawdown máximo, tamaño de
    posición, pérdida diaria). El agente debe rechazar la operación."""


class ExecutionError(TradingAgentError):
    """Fallo al ejecutar una orden en el broker/simulador."""


class InsufficientFundsError(ExecutionError):
    """No hay efectivo suficiente para abrir la posición solicitada."""


class SecurityError(TradingAgentError):
    """Problema de seguridad: credencial ausente, permiso de archivo
    demasiado abierto, intento de operar en vivo sin confirmación."""
