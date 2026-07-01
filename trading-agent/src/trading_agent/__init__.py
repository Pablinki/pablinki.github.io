"""Trading agent basado en Deep Q-Learning (Double DQN).

Módulos principales:
    - ``config``: configuración parametrizada y validada.
    - ``data``: proveedores de datos (Yahoo Finance, Bloomberg).
    - ``features``: indicadores técnicos para el estado del agente.
    - ``env``: entorno de simulación estilo Gym.
    - ``agent``: red Q, replay buffer y agente DQN.
    - ``risk``: gestor de riesgo (stops, límites, kill-switch).
    - ``portfolio``: contabilidad de posiciones y trades.
    - ``train`` / ``live``: bucles de entrenamiento y tiempo real.
"""

__version__ = "1.0.0"
