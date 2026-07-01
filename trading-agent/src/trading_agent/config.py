"""Configuración centralizada y 100% parametrizada del trading agent.

Toda constante del sistema (hiperparámetros del DQN, límites de riesgo,
parámetros de datos) vive aquí como *dataclass* tipada y validada.
Nada está "hardcodeado" en el resto del código: cada módulo recibe su
sección de configuración por inyección de dependencias, lo que facilita
tests (se pasa una config pequeña) y escalabilidad (una config por
símbolo/estrategia).

Estructura de datos
-------------------
``AppConfig`` es la raíz y agrega cuatro secciones::

    AppConfig
    ├── DataConfig      -> de dónde y cómo se obtienen los datos
    ├── AgentConfig     -> hiperparámetros del DQN
    ├── EnvConfig       -> reglas del entorno de simulación
    └── RiskConfig      -> límites de riesgo del RiskManager

Entradas / salidas principales
------------------------------
- :func:`load_config` : ruta a YAML (str) -> ``AppConfig`` validada.
- ``AppConfig.default()`` : sin argumentos -> ``AppConfig`` con valores
  razonables para experimentar.

Ejemplo
-------
>>> cfg = AppConfig.default()
>>> cfg.agent.gamma
0.99
>>> cfg = load_config("config.yaml")          # doctest: +SKIP
>>> cfg.risk.max_position_pct                  # doctest: +SKIP
0.25
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


def _require(condition: bool, message: str) -> None:
    """Valida una condición de configuración.

    Args:
        condition: expresión booleana que debe cumplirse.
        message: mensaje de error si no se cumple.

    Raises:
        ConfigurationError: si ``condition`` es ``False``.
    """
    if not condition:
        raise ConfigurationError(message)


@dataclass(frozen=True)
class DataConfig:
    """Parámetros de la capa de datos.

    Attributes:
        provider: nombre del proveedor: ``"yahoo"`` o ``"bloomberg"``.
        symbols: lista de tickers a operar, p. ej. ``["AAPL", "MSFT"]``.
        interval: granularidad de las velas (``"1m"``, ``"5m"``, ``"1h"``,
            ``"1d"``...). Debe ser soportada por el proveedor.
        lookback_days: días de histórico a descargar para entrenamiento.
        max_retries: reintentos ante ``DataProviderError`` transitorio.
        retry_backoff_s: espera base del backoff exponencial en segundos
            (espera real = ``retry_backoff_s * 2**intento``).
        cache_dir: directorio donde se cachean descargas (evita golpear
            la API en cada corrida). ``None`` desactiva el caché.
    """

    provider: str = "yahoo"
    symbols: tuple[str, ...] = ("AAPL",)
    interval: str = "1d"
    lookback_days: int = 365 * 4
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    cache_dir: str | None = ".cache/market_data"

    def __post_init__(self) -> None:
        _require(self.provider in ("yahoo", "bloomberg"),
                 f"provider debe ser 'yahoo' o 'bloomberg', no {self.provider!r}")
        _require(len(self.symbols) > 0, "symbols no puede estar vacío")
        _require(self.lookback_days > 0, "lookback_days debe ser > 0")
        _require(self.max_retries >= 0, "max_retries debe ser >= 0")
        _require(self.retry_backoff_s >= 0, "retry_backoff_s debe ser >= 0")


@dataclass(frozen=True)
class AgentConfig:
    """Hiperparámetros del agente DQN (todos parametrizados).

    Attributes:
        state_dim: dimensión del vector de estado que produce el entorno.
            Se calcula en runtime (nº de features * ventana + extras) y se
            sobreescribe con :func:`dataclasses.replace`.
        n_actions: nº de acciones discretas (3: mantener/comprar/vender).
        hidden_sizes: anchura de cada capa oculta de la red Q,
            p. ej. ``(128, 64)`` crea dos capas ocultas.
        lr: learning rate del optimizador Adam.
        gamma: factor de descuento de recompensas futuras, en (0, 1].
        epsilon_start: probabilidad inicial de exploración (política
            epsilon-greedy).
        epsilon_end: probabilidad mínima de exploración.
        epsilon_decay: factor multiplicativo aplicado a epsilon tras cada
            episodio; en (0, 1].
        batch_size: tamaño del minibatch muestreado del replay buffer.
        buffer_capacity: capacidad máxima del replay buffer (FIFO).
        min_buffer_size: nº mínimo de transiciones antes de empezar a
            entrenar (evita sobreajustar a experiencias iniciales).
        target_update_every: cada cuántos pasos de gradiente se sincroniza
            la red objetivo (Double DQN usa red objetivo separada).
        grad_clip_norm: norma máxima del gradiente (estabilidad numérica).
        double_dqn: si ``True`` usa Double DQN (reduce sobreestimación de Q).
        device: ``"cpu"`` o ``"cuda"``.
        seed: semilla global de reproducibilidad.
    """

    state_dim: int = 1  # se recalcula en runtime según las features
    n_actions: int = 3
    hidden_sizes: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1_000
    target_update_every: int = 500
    grad_clip_norm: float = 10.0
    double_dqn: bool = True
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        _require(self.state_dim >= 1, "state_dim debe ser >= 1")
        _require(self.n_actions >= 2, "n_actions debe ser >= 2")
        _require(all(h > 0 for h in self.hidden_sizes),
                 "hidden_sizes debe contener enteros positivos")
        _require(self.lr > 0, "lr debe ser > 0")
        _require(0 < self.gamma <= 1, "gamma debe estar en (0, 1]")
        _require(0 <= self.epsilon_end <= self.epsilon_start <= 1,
                 "se requiere 0 <= epsilon_end <= epsilon_start <= 1")
        _require(0 < self.epsilon_decay <= 1, "epsilon_decay debe estar en (0, 1]")
        _require(self.batch_size > 0, "batch_size debe ser > 0")
        _require(self.buffer_capacity >= self.batch_size,
                 "buffer_capacity debe ser >= batch_size")
        _require(self.min_buffer_size >= self.batch_size,
                 "min_buffer_size debe ser >= batch_size")
        _require(self.target_update_every > 0, "target_update_every debe ser > 0")


@dataclass(frozen=True)
class EnvConfig:
    """Reglas del entorno de simulación (backtest / entrenamiento).

    Attributes:
        window_size: nº de velas pasadas incluidas en el estado.
        initial_cash: efectivo inicial de cada episodio, en USD.
        commission_pct: comisión por operación como fracción del nocional
            (0.001 = 0.1%). Modela costes reales de broker.
        slippage_pct: deslizamiento adverso aplicado al precio de ejecución
            (compras más caro, vendes más barato).
        reward_scaling: multiplicador de la recompensa (ayuda a estabilizar
            el aprendizaje cuando los retornos por paso son minúsculos).
    """

    window_size: int = 30
    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    reward_scaling: float = 100.0

    def __post_init__(self) -> None:
        _require(self.window_size >= 2, "window_size debe ser >= 2")
        _require(self.initial_cash > 0, "initial_cash debe ser > 0")
        _require(0 <= self.commission_pct < 1, "commission_pct debe estar en [0, 1)")
        _require(0 <= self.slippage_pct < 1, "slippage_pct debe estar en [0, 1)")
        _require(self.reward_scaling > 0, "reward_scaling debe ser > 0")


@dataclass(frozen=True)
class RiskConfig:
    """Límites del gestor de riesgo. Se aplican ANTES de cada orden.

    Attributes:
        max_position_pct: fracción máxima del equity invertible en un solo
            símbolo (0.25 = 25%).
        max_drawdown_pct: drawdown máximo tolerado desde el pico de equity;
            al superarse se liquida todo y se detiene el trading.
        stop_loss_pct: pérdida máxima por posición antes de cierre forzoso.
        take_profit_pct: ganancia objetivo por posición para cierre
            automático (``None`` la desactiva).
        max_daily_loss_pct: pérdida diaria máxima del equity; al superarse
            no se abren posiciones nuevas ese día.
    """

    max_position_pct: float = 0.25
    max_drawdown_pct: float = 0.20
    stop_loss_pct: float = 0.05
    take_profit_pct: float | None = 0.10
    max_daily_loss_pct: float = 0.03

    def __post_init__(self) -> None:
        _require(0 < self.max_position_pct <= 1, "max_position_pct debe estar en (0, 1]")
        _require(0 < self.max_drawdown_pct <= 1, "max_drawdown_pct debe estar en (0, 1]")
        _require(0 < self.stop_loss_pct < 1, "stop_loss_pct debe estar en (0, 1)")
        _require(self.take_profit_pct is None or self.take_profit_pct > 0,
                 "take_profit_pct debe ser > 0 o None")
        _require(0 < self.max_daily_loss_pct <= 1,
                 "max_daily_loss_pct debe estar en (0, 1]")


@dataclass(frozen=True)
class AppConfig:
    """Configuración raíz que agrega todas las secciones."""

    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @staticmethod
    def default() -> "AppConfig":
        """Devuelve una configuración por defecto lista para usar."""
        return AppConfig()

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "AppConfig":
        """Construye la configuración desde un ``dict`` anidado (p. ej. YAML).

        Args:
            raw: diccionario con claves opcionales ``data``, ``agent``,
                ``env`` y ``risk``; cada una es un dict de campos.

        Returns:
            ``AppConfig`` validada.

        Raises:
            ConfigurationError: si hay claves desconocidas o valores fuera
                de rango.

        Ejemplo:
            >>> cfg = AppConfig.from_dict({"agent": {"gamma": 0.9}})
            >>> cfg.agent.gamma
            0.9
        """
        sections = {"data": DataConfig, "agent": AgentConfig,
                    "env": EnvConfig, "risk": RiskConfig}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ConfigurationError(f"Secciones desconocidas en config: {sorted(unknown)}")

        kwargs: dict[str, Any] = {}
        for name, cls in sections.items():
            body = raw.get(name, {})
            if not isinstance(body, dict):
                raise ConfigurationError(f"La sección {name!r} debe ser un mapeo")
            valid_fields = {f.name for f in dataclasses.fields(cls)}
            unknown_fields = set(body) - valid_fields
            if unknown_fields:
                raise ConfigurationError(
                    f"Campos desconocidos en {name!r}: {sorted(unknown_fields)}")
            # YAML entrega listas; las dataclasses congeladas usan tuplas.
            body = {k: tuple(v) if isinstance(v, list) else v for k, v in body.items()}
            try:
                kwargs[name] = cls(**body)
            except TypeError as exc:
                raise ConfigurationError(f"Sección {name!r} inválida: {exc}") from exc
        return AppConfig(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    """Carga y valida la configuración desde un archivo YAML.

    Args:
        path: ruta al archivo YAML.

    Returns:
        ``AppConfig`` validada.

    Raises:
        ConfigurationError: si el archivo no existe, no es YAML válido o
            contiene valores inválidos.
    """
    import yaml  # import perezoso: pyyaml solo se necesita si se usa YAML

    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"No existe el archivo de configuración: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML inválido en {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"La raíz de {path} debe ser un mapeo YAML")
    return AppConfig.from_dict(raw)
