"""Adaptadores de broker: ejecución de órdenes simulada o real."""

from ..exceptions import ConfigurationError
from ..portfolio import Portfolio
from .base import BrokerAdapter, PaperBroker

__all__ = ["BrokerAdapter", "PaperBroker", "make_broker"]


def make_broker(name: str, portfolio: Portfolio) -> BrokerAdapter:
    """Fábrica de brokers por nombre (mismo patrón que ``make_provider``).

    Args:
        name: ``"paper"`` (simulado, defecto del sistema) o ``"alpaca"``.
        portfolio: portafolio local donde se refleja cada operación.

    Returns:
        Instancia de :class:`BrokerAdapter` lista para usar.

    Raises:
        ConfigurationError: si el nombre no está soportado.
        SecurityError: (alpaca) si faltan credenciales o se apunta a
            dinero real sin ``TRADING_AGENT_LIVE=YES``.
    """
    if name == "paper":
        return PaperBroker(portfolio)
    if name == "alpaca":
        from .alpaca import AlpacaBroker
        return AlpacaBroker(portfolio)
    raise ConfigurationError(f"Broker no soportado: {name!r}")
