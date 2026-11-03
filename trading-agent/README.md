# Trading Agent con Deep Q-Learning (Double DQN)

Agente de trading autónomo basado en **aprendizaje por refuerzo** que aprende
a **abrir y cerrar posiciones para maximizar la ganancia compuesta**. Toma
datos de **Yahoo Finance** (gratuito, por defecto) o **Bloomberg** (blpapi),
entrena un **Double DQN** sobre histórico y opera en (casi) tiempo real en
modo *paper trading* con una capa de **gestión de riesgo determinista**.

> ⚠️ **Descargo de responsabilidad**: software educativo/experimental. El
> trading conlleva riesgo de pérdida total. Nada de esto es consejo
> financiero. Por diseño, el modo con dinero real está deshabilitado salvo
> habilitación explícita (ver [Seguridad](#seguridad)).

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Instalación](#instalación)
3. [Uso rápido](#uso-rápido)
4. [Cómo funciona el RL (MDP)](#cómo-funciona-el-rl)
5. [Parámetros (todos configurables)](#parámetros)
6. [Gestión de riesgo](#gestión-de-riesgo)
7. [Manejo de excepciones](#manejo-de-excepciones)
8. [Seguridad](#seguridad)
9. [Escalabilidad](#escalabilidad)
10. [Tests](#tests)
11. [Bloomberg](#bloomberg)

---

## Arquitectura

```
                      config.yaml  ──►  AppConfig (validada, inmutable)
                                              │
        ┌─────────────────────────────────────┼──────────────────────────┐
        ▼                                     ▼                          ▼
  DataProvider (ABC)                  TradingEnvironment            RiskManager
  ├── YahooFinanceProvider            (MDP estilo Gym)              (stops, límites,
  └── BloombergProvider                 │        ▲                   kill-switch)
        │                               ▼        │
        │   OHLCV canónico        estado      acción
        │   (validado)                │        │
        └────────────►  features.py   ▼        │
                        (RSI, MACD,  DQNAgent ─┘
                         Bollinger…)  ├── QNetwork (online)
                                      ├── QNetwork (target)
                                      └── ReplayBuffer

  train.py  = bucle de episodios sobre histórico (split temporal train/val)
  live.py   = bucle en tiempo real (paper trading) con el modelo entrenado
  portfolio = contabilidad exacta de cash/posiciones/comisiones/slippage
```

**Módulos** (`src/trading_agent/`):

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Todos los parámetros como dataclasses validadas (fail-fast) |
| `exceptions.py` | Jerarquía de excepciones tipadas |
| `security.py` | Credenciales por variables de entorno, modo vivo explícito |
| `data/provider.py` | Interfaz `DataProvider`, validación OHLCV, reintentos |
| `data/yahoo.py` | Proveedor Yahoo Finance (con caché en disco) |
| `data/bloomberg.py` | Proveedor Bloomberg (blpapi, opcional) |
| `features.py` | Indicadores técnicos → estado del agente |
| `env.py` | Entorno de simulación (MDP) |
| `agent/replay_buffer.py` | Buffer circular numpy de transiciones |
| `agent/networks.py` | Red Q (MLP parametrizado, PyTorch) |
| `agent/dqn.py` | Agente Double DQN + red objetivo + persistencia |
| `portfolio.py` | Posiciones, trades, equity |
| `risk.py` | Stop-loss, take-profit, drawdown, pérdida diaria |
| `train.py` | Bucle de entrenamiento, split temporal 3 vías y evaluación en test |
| `live.py` | Bucle de trading en tiempo real (paper o broker real) |
| `brokers/` | `BrokerAdapter`, `PaperBroker` (simulado) y `AlpacaBroker` (real) |
| `metrics.py` | Drawdown máximo, Sharpe, profit factor |
| `journal.py` | Diario CSV de operaciones (auditoría) |
| `logging_setup.py` | Consola + archivo con rotación diaria |

---

## Instalación

> 🪟 **Windows 11**: sigue la guía paso a paso en
> [INSTALACION_WINDOWS.md](INSTALACION_WINDOWS.md) (Python, venv,
> PowerShell y solución de problemas).

Requiere **Python 3.10–3.12** (probado con 3.11). Las versiones de
`requirements.txt` están fijadas y verificadas compatibles entre sí:

```bash
cd trading-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy 1.26 + pandas 2.2 + torch 2.6 ...
pytest                                  # verifica la instalación: 279 tests
```

---

## Uso rápido

**1. Entrenar** (descarga ~4 años de AAPL de Yahoo Finance y entrena 50 episodios):

```bash
python scripts/train.py --config config.yaml --episodes 50
# Guarda el mejor modelo (según retorno en validación) en checkpoints/dqn_best.pt
```

**2. Operar en tiempo real** (paper trading, decisión cada 60 s):

```bash
python scripts/run_live.py --config config.yaml \
    --checkpoint checkpoints/dqn_best.pt --poll-seconds 60
```

**3. Usar como librería:**

```python
from trading_agent.config import load_config
from trading_agent.train import train

config = load_config("config.yaml")
metrics = train(config, episodes=50)
print(f"Mejor retorno de validación: {metrics['best_val_return']:+.2%}")
```

---

## Cómo funciona el RL

El problema se formula como un **Proceso de Decisión de Markov**:

- **Estado** (`np.ndarray float32`): ventana de las últimas `window_size`
  velas de 7 features técnicas (retorno log, RSI, MACD, señal MACD,
  posición en bandas de Bollinger, volatilidad, z-score de volumen),
  aplanada, más 2 escalares de cartera: `[¿hay posición?, P&L no realizado]`.
  Dimensión = `window_size × 7 + 2` (con la ventana por defecto de 30: 212).

- **Acciones** (discretas): `0=HOLD` (esperar), `1=BUY` (abrir posición
  larga), `2=SELL` (cerrar posición). Acciones imposibles (comprar ya
  comprado, vender sin posición) degradan a HOLD: el agente aprende su
  inutilidad por la recompensa.

- **Recompensa**: `ln(equity_t / equity_{t-1}) × reward_scaling`. La suma
  de recompensas del episodio **es** el log-retorno total → maximizar
  recompensa acumulada = maximizar ganancia compuesta (el objetivo real).

- **Algoritmo**: DQN con las dos mejoras estándar:
  - **Red objetivo** sincronizada cada `target_update_every` pasos
    (estabilidad).
  - **Double DQN**: la red online *elige* la acción del siguiente estado y
    la red objetivo la *valora* (reduce el sobreoptimismo, que en trading
    se traduce en sobreoperar).
  - Pérdida Huber + recorte de gradiente + replay buffer de 100k
    transiciones + política ε-greedy con decaimiento por episodio.

- **Anti *look-ahead bias***: el histórico se divide **por tiempo** (nunca
  aleatorio) en train (70%) / validación (15%) / **test** (15%). El mejor
  checkpoint se elige por retorno en validación y, al terminar, se evalúa
  una sola vez en el tramo de test, que nunca influyó ni en entrenar ni en
  elegir: es la estimación honesta fuera de muestra.

- **Métricas de evaluación** (`episode_summary`): retorno total,
  benchmark **comprar-y-mantener** del mismo tramo y exceso sobre él,
  **drawdown máximo**, **Sharpe** anualizado, *profit factor*, nº de
  operaciones y *win rate*. El log de entrenamiento muestra el B&H de
  validación en cada episodio y las métricas de test al final.

- **Realismo del backtest**: cada operación paga **comisión** y sufre
  **slippage** adverso configurables; al terminar el episodio la posición
  se liquida para evaluar todo en efectivo.

---

## Parámetros

**Todos** los parámetros viven en `config.yaml` y están documentados campo a
campo en `src/trading_agent/config.py`. Cada valor se **valida en
construcción** (rangos, coherencia entre campos) y las dataclasses son
inmutables (`frozen=True`): una config inválida revienta al arrancar, no a
mitad de una sesión.

Resumen:

| Sección | Parámetros |
|---|---|
| `data` | proveedor, símbolos, intervalo, días de histórico, reintentos, backoff, caché |
| `agent` | capas ocultas, lr, γ, ε (inicial/final/decay), batch, buffer, sincronización target, Double DQN, clip de gradiente, device, semilla |
| `env` | ventana, efectivo inicial, comisión, slippage, escala de recompensa |
| `risk` | % máximo por posición, drawdown máximo, stop-loss, take-profit, pérdida diaria máxima |

---

## Gestión de riesgo

El `RiskManager` es **determinista** y manda sobre el agente (los stops se
evalúan antes de consultar a la red):

1. **Dimensionamiento** *fixed-fraction*: nunca más de `max_position_pct`
   del equity en una posición.
2. **Stop-loss / take-profit** por posición: cierre forzoso a
   `-stop_loss_pct` / `+take_profit_pct`. En backtest se evalúan
   **intrabarra** (contra el mínimo/máximo de la vela, ejecutando al nivel
   del stop o a la apertura si hay hueco), igual que en vivo se evalúan con
   cada precio. Tras un cierre forzoso, `cooldown_bars` velas sin reentrar.
3. **Límite de pérdida diaria**: superada, no se abren posiciones nuevas
   ese día.
4. **Kill-switch de drawdown**: si el equity cae `max_drawdown_pct` desde
   su pico, se liquida y se detiene todo hasta intervención humana.

Es la misma clase en backtest y en vivo: los límites que se testean son los
que protegen el dinero.

---

## Manejo de excepciones

Jerarquía tipada con raíz `TradingAgentError` (ver `exceptions.py`):

```
TradingAgentError
├── ConfigurationError        parámetros inválidos (fail-fast al arrancar)
├── DataProviderError         fallo transitorio de datos → reintento c/ backoff
│   └── DataValidationError   datos corruptos → NO se reintenta
├── EnvironmentError_         mal uso del entorno de simulación
├── RiskLimitExceededError    orden vetada por riesgo
├── ExecutionError            fallo de ejecución de orden
│   └── InsufficientFundsError
└── SecurityError             credencial ausente / modo vivo no autorizado
```

Políticas aplicadas:

- **Reintentos con backoff exponencial** (`with_retries`) solo para fallos
  transitorios de red; los datos corruptos fallan inmediatamente.
- El **bucle en vivo nunca muere** por un fallo de datos: lo registra y
  espera al siguiente ciclo. Solo el kill-switch o una señal (SIGINT/
  SIGTERM, con manejadores instalados) lo detienen — y al detenerse
  **liquida la posición abierta**.
- Validación de datos centralizada (`validate_ohlcv`): NaN, precios ≤ 0,
  duplicados y desorden temporal se detectan antes de llegar al agente.

---

## Seguridad

- **Credenciales solo por variables de entorno** (`security.get_secret`);
  nunca en código ni en el repo (`.gitignore` cubre `.env`).
- **`mask_secret`** permite loguear la existencia de una credencial sin
  exponer su valor.
- **Paper trading por defecto**: operar con dinero real exige exportar
  `TRADING_AGENT_LIVE=YES` (exacto, mayúsculas). Un despliegue accidental
  queda en simulado.
- Los checkpoints se cargan con `torch.load(..., weights_only=True)`: no se
  deserializan objetos arbitrarios (evita ejecución de código vía pickle).
- Carga de YAML con `yaml.safe_load` (nunca `load`).
- El checkpoint valida dimensiones antes de cargarse (pesos incompatibles
  fallan ruidosamente, no producen basura silenciosa).

---

## Broker real (Alpaca)

El bucle en vivo ejecuta las órdenes a través de la interfaz
`BrokerAdapter` (`src/trading_agent/brokers/`). Por defecto usa
`PaperBroker` (simulación local). Para ejecutar órdenes reales de mercado
está incluido el adaptador de **Alpaca Markets**:

```bash
export ALPACA_API_KEY="tu-clave"
export ALPACA_SECRET_KEY="tu-secreto"
# Sin ALPACA_BASE_URL se usa el PAPER de Alpaca (dinero ficticio):
python scripts/run_live.py --broker alpaca --checkpoint checkpoints/dqn_best.pt
```

Flujo de cada orden: orden *market* → sondeo hasta el *fill* (con timeout
y cancelación best-effort) → el precio real de ejecución se refleja en el
portafolio espejo local, que es el que ven el RiskManager y el estado del
agente. El efectivo se sincroniza con la cuenta al arrancar.

Escalera de despliegue recomendada: (1) `--broker paper` semanas,
(2) `--broker alpaca` contra el paper de Alpaca (misma API que la real),
(3) solo entonces, dinero real: exportar `TRADING_AGENT_LIVE=YES` **y**
`ALPACA_BASE_URL=https://api.alpaca.markets`. Sin ambas cosas, el
adaptador rechaza arrancar en live (`SecurityError`). Ten presente la
regla *Pattern Day Trader* de EE. UU. (25.000 USD mínimos para >3
day-trades en 5 días hábiles) si usas intervalos intradía.

## Operación en producción

Reglas que aplica el bucle en vivo (`LiveTrader`) para funcionar 24/7 sin
supervisión:

- **Una decisión por vela**: el agente solo se consulta cuando cambia la
  marca temporal de la última vela. Con el mercado cerrado (o sondeando
  más rápido que el intervalo) el mismo estado produciría la misma acción
  y el agente alternaría compra/venta sobre datos idénticos. Los stops sí
  se re-evalúan con cada precio.
- **Reloj de mercado**: con un broker real no se envía ninguna orden si el
  mercado está cerrado (quedaría colgada hasta el timeout). Un stop-loss
  pendiente se ejecuta en la primera iteración con mercado abierto.
- **Diario de operaciones** (`--journal logs/trades.csv`, activo por
  defecto): cada compra, venta, cierre forzoso y liquidación queda en un
  CSV *append-only* con timestamp UTC, precio, comisión, P&L, motivo y
  equity — la fuente de verdad para reconciliar contra el broker.
- **Reinicios seguros**: con Alpaca, al arrancar se sincronizan efectivo
  y posiciones abiertas reales en el espejo local, así una posición
  abierta antes de un reinicio sigue protegida por el RiskManager.
- **Cortacircuitos**: un error no previsto se registra con traza y el
  bucle continúa; `max_consecutive_errors` (10) errores seguidos lo
  detienen. Pase lo que pase (señal, kill-switch, excepción), la
  liquidación final se intenta siempre (`finally`).
- **Estado de riesgo persistente** (`--risk-state logs/risk_state.json`):
  pico de equity, kill-switch y enfriamiento sobreviven a un reinicio. Si
  el proceso paró por drawdown, **se niega a arrancar** hasta que un humano
  borre o edite ese archivo: un reinicio automático nunca rearma el
  kill-switch solo.
- **Reconciliación diaria**: con un broker real, al cambiar de jornada el
  espejo local se re-sincroniza con la cuenta (efectivo y posiciones); una
  divergencia se registra como `CRITICAL` y manda la versión del broker.
- **Órdenes idempotentes** (Alpaca): cada decisión lleva un
  `client_order_id` determinista; si el proceso muere tras enviar la orden
  y vuelve a intentarlo, Alpaca la rechaza como duplicada y el adaptador la
  recupera por id en lugar de comprar dos veces. Los fills parciales tras
  un timeout se reflejan en el espejo (nunca acciones reales sin contar).
- **Supervisión**: `logs/heartbeat.json` se reescribe en cada iteración;
  `scripts/healthcheck.py` (y el `HEALTHCHECK` de Docker) lo leen y marcan
  el proceso como no sano si envejece, si el kill-switch está activo o si
  acumula errores — un proceso "vivo" que dejó de operar no pasa
  desapercibido. Log con rotación diaria en `logs/trader.log`.
- **Códigos de salida** distinguibles para la política de reinicio:
  `0` parada deliberada, `1` error controlado al arrancar, `2` kill-switch
  (no reiniciar a ciegas: el estado persistido impide rearrancar),
  `3` cortacircuitos (reiniciar tiene sentido).
- **Docker / Compose**: `Dockerfile` (imagen `python:3.11-slim`, torch CPU,
  usuario sin privilegios, `HEALTHCHECK`, `SIGTERM` → parada limpia) y
  `docker-compose.yml` con `restart: on-failure`, `stop_grace_period: 120s`
  (tiempo para liquidar) y credenciales desde `.env` (plantilla en
  `.env.example`):

  ```bash
  cp .env.example .env            # rellena las claves; .env nunca se sube
  docker compose up -d --build
  docker compose logs -f trader
  docker compose stop             # parada limpia con liquidación
  ```

- **Un símbolo por proceso**: cada proceso opera `symbols[0]` con su propia
  config y carpeta `logs/`; las posiciones de otros símbolos en la misma
  cuenta se ignoran (con aviso). Para N símbolos, N servicios.
- **CI**: en cada push/PR, GitHub Actions ejecuta lint (`ruff`), tipado
  (`mypy`), la suite completa y los doctests (los ejemplos de la
  documentación se verifican).

## Escalabilidad

- **Proveedores intercambiables** (patrón Strategy + Factory): añadir IEX,
  Polygon o un broker propio = implementar 2 métodos de `DataProvider`.
- **Multi-símbolo**: `Portfolio` y `RiskManager` ya son multi-posición; se
  escala horizontalmente lanzando un proceso `run_live.py` por símbolo con
  su propia config (aislamiento de fallos entre símbolos).
- **Replay buffer en numpy preasignado**: memoria contigua y muestreo
  vectorizado O(batch) — 100k transiciones × 212 floats ≈ 170 MB, constante.
- **Caché de datos en parquet** con invalidación diaria: entrenar N veces
  descarga 1 vez.
- **`device: cuda`** en la config mueve el entrenamiento a GPU sin tocar
  código.
- Config inmutable + inyección de dependencias: N estrategias en paralelo
  sin estado global compartido.

---

## Tests

279 tests automatizados (`pytest`) que **no tocan la red**: los datos de mercado se
generan sintéticamente con semilla fija (deterministas, corren en CI sin
credenciales).

```bash
cd trading-agent
python3 -m pytest tests/ -v
```

Cobertura por área:

- `test_config.py` — validación de cada parámetro y carga YAML.
- `test_features.py` — indicadores: rangos, determinismo, cálculo manual.
- `test_portfolio.py` — contabilidad exacta con comisiones y slippage.
- `test_risk.py` — stops, kill-switch, límite diario, dimensionamiento.
- `test_env.py` — mecánica del MDP y **alineación de la recompensa** (la
  suma de recompensas reconstruye el log-retorno del episodio).
- `test_replay_buffer.py` — FIFO circular, formas, reproducibilidad.
- `test_dqn.py` — la política greedy es determinista, ε=1 explora, **la
  pérdida decrece y la política aprende** en un problema trivial,
  sincronización de la red objetivo, guardado/carga de checkpoints.
- `test_train.py` — split temporal anti look-ahead + entrenamiento
  end-to-end de humo.
- `test_live.py` — bucle en vivo con proveedor falso: una decisión por
  vela, mercado cerrado, diario, cortacircuitos, estado de riesgo
  persistente, reconciliación, heartbeat; y toda la capa de seguridad.
- `test_brokers.py` — Alpaca con sesión HTTP falsa: fills, sondeo,
  idempotencia (`client_order_id`), fills parciales, timeout + cancelación,
  allowlist de hosts, redirecciones, sincronización de cuenta.
- `test_metrics.py`, `test_journal.py`, `test_cli.py` — métricas de
  evaluación, diario CSV, healthcheck y logging.

---

## Bloomberg

El proveedor `bloomberg` usa la **Desktop/Server API** oficial (`blpapi`) y
requiere una Terminal Bloomberg o B-PIPE:

```bash
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
export BLOOMBERG_HOST=localhost
export BLOOMBERG_PORT=8194
```

y en `config.yaml`:

```yaml
data:
  provider: bloomberg
  interval: 1m        # intradía: 1m/5m/15m/30m/1h; o 1d para diario
```

Si `blpapi` no está instalado, el sistema sigue funcionando con Yahoo; el
proveedor Bloomberg falla de forma controlada (`ConfigurationError`) solo si
se selecciona.
