# Guía de instalación y configuración — Windows 11 (laptop)

Guía paso a paso para dejar el trading agent funcionando en una laptop con
Windows 11, usando un **entorno virtual** (`venv`) para aislar las
dependencias del resto del sistema.

---

## 1. Instalar Python 3.11

El proyecto requiere Python 3.10–3.12; recomendamos **3.11** (es la versión
con la que están verificadas todas las dependencias).

**Opción A — winget (recomendada, viene con Windows 11):**

Abre **PowerShell** (busca "PowerShell" en el menú Inicio) y ejecuta:

```powershell
winget install Python.Python.3.11
```

Cierra y vuelve a abrir PowerShell para que se actualice el PATH.

**Opción B — instalador oficial:**

1. Descarga desde https://www.python.org/downloads/windows/ el instalador
   de Python 3.11.x (64-bit).
2. Al ejecutarlo, **marca la casilla "Add python.exe to PATH"** antes de
   pulsar *Install Now* (si la olvidas, los comandos `python`/`pip` no se
   encontrarán en la terminal).

**Verifica la instalación:**

```powershell
python --version
# Debe mostrar: Python 3.11.x
```

> Si `python` abre la Microsoft Store en lugar de responder, ve a
> **Configuración → Aplicaciones → Configuración avanzada de aplicaciones →
> Alias de ejecución de aplicaciones** y desactiva los alias
> `python.exe` y `python3.exe` de la Store.

## 2. Instalar Git y clonar el repositorio

```powershell
winget install Git.Git
```

Cierra y reabre PowerShell, y clona el proyecto (por ejemplo en tu carpeta
de usuario):

```powershell
cd $HOME
git clone https://github.com/Pablinki/pablinki.github.io.git
cd pablinki.github.io\trading-agent
```

## 3. Crear y activar el entorno virtual

Dentro de la carpeta `trading-agent`:

```powershell
# Crea el entorno virtual en la carpeta .venv
python -m venv .venv

# Actívalo
.\.venv\Scripts\Activate.ps1
```

Sabrás que está activo porque el prompt cambia a `(.venv) PS C:\...>`.

> **Si PowerShell bloquea el script de activación** con el error
> *"running scripts is disabled on this system"*, habilita los scripts
> firmados solo para tu usuario (es seguro y no requiere administrador):
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
>
> y vuelve a ejecutar `.\.venv\Scripts\Activate.ps1`.
>
> Alternativa sin cambiar políticas: usa **CMD** en lugar de PowerShell y
> activa con `.venv\Scripts\activate.bat`.

Comprueba que el `python` activo es el del entorno:

```powershell
Get-Command python | Select-Object Source
# Debe apuntar a ...\trading-agent\.venv\Scripts\python.exe
```

Para **salir** del entorno cuando termines: `deactivate`.

## 4. Instalar las dependencias

Con el entorno activado:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala las versiones fijadas y compatibles entre sí (numpy 1.26.4,
pandas 2.2.3, torch 2.6.0, yfinance 0.2.54, etc.).

> **Notas sobre PyTorch en Windows:**
> - El `torch` de PyPI para Windows es la variante **CPU**: perfecta para
>   este proyecto (la red Q es pequeña) y no requiere nada más.
> - Solo si tu laptop tiene GPU NVIDIA y quieres usarla: instala la
>   variante CUDA con
>   `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
>   y cambia `device: cuda` en `config.yaml`.
> - La descarga de torch pesa varios cientos de MB; con WiFi lento tarda
>   unos minutos.

## 5. Verificar la instalación con los tests

```powershell
pytest
```

Deben pasar **279 tests** (tardan menos de un minuto; no necesitan
internet: usan datos sintéticos). Si todos están en verde, la instalación
es correcta.

## 6. Configurar el agente

Edita `config.yaml` con el Bloc de notas o VS Code:

```powershell
notepad config.yaml
```

Los cambios más habituales:

```yaml
data:
  symbols: [AAPL]     # cambia el ticker que quieras operar
  interval: 1d        # granularidad de las velas
risk:
  stop_loss_pct: 0.05 # ajusta tus límites de riesgo
```

Cada parámetro está documentado en `src/trading_agent/config.py` y en el
`README.md`.

## 7. Entrenar el modelo

```powershell
python scripts\train.py --config config.yaml --episodes 50
```

- Descarga ~4 años de histórico de Yahoo Finance (se cachea en
  `.cache\market_data`, así que solo descarga la primera vez).
- Entrena 50 episodios y guarda el mejor modelo (según el retorno en
  datos de validación que nunca vio) en `checkpoints\dqn_best.pt`.
- En una laptop con CPU moderna tarda del orden de minutos.

## 8. Ejecutar en tiempo real (paper trading)

```powershell
python scripts\run_live.py --config config.yaml `
    --checkpoint checkpoints\dqn_best.pt --poll-seconds 60
```

El agente consulta el precio cada 60 s, decide (HOLD/BUY/SELL) **una vez
por vela nueva** y lo simula sobre un portafolio virtual con comisiones y
slippage. Se detiene limpiamente con **Ctrl+C** (liquida la posición
abierta antes de salir). Todo lo operativo queda en la carpeta `logs\`:

| Archivo | Contenido |
|---|---|
| `trades.csv` | Diario de operaciones (ábrelo con Excel); cada compra, venta, stop y liquidación con precio, P&L, motivo e ids de orden |
| `trader.log` | Log completo con rotación diaria (14 días) |
| `risk_state.json` | Estado del gestor de riesgo; si el kill-switch saltó, el trader **no arranca** hasta que borres/edites este archivo |
| `heartbeat.json` | Latido por iteración (equity, errores, última vela) para supervisión |

Códigos de salida del proceso: `0` parada deliberada, `1` error al arrancar,
`2` kill-switch de drawdown (no reiniciar sin revisar), `3` cortacircuitos
por errores repetidos.

> **Seguridad:** este modo es 100% simulado. El trading con dinero real
> está deshabilitado por diseño salvo que definas explícitamente la
> variable de entorno `TRADING_AGENT_LIVE=YES` **y** apuntes el adaptador
> de Alpaca (paso 9) al endpoint real.

## 9. Conectar Alpaca (órdenes reales — empieza por su modo paper)

El proyecto incluye un adaptador para **Alpaca Markets** que ejecuta
órdenes de mercado reales. La escalera recomendada: paper local (paso 8)
→ paper de Alpaca (esta sección) → dinero real, solo si lo anterior
convence.

**9.1. Crear la cuenta (gratis, solo correo):**

1. Entra en https://app.alpaca.markets/signup y regístrate con tu correo.
2. Confirma el enlace de verificación que te llega por email.
3. Ya tienes acceso al dashboard con **paper trading ilimitado y
   gratuito** (~100.000 USD virtuales). No necesitas verificación de
   identidad ni depósito para esto.

> La cuenta con dinero real es un trámite aparte dentro del mismo
> dashboard (verificación de identidad/KYC). Solo la necesitas en el
> último escalón.

**9.2. Obtener las API keys:**

1. En el dashboard, en la **esquina superior izquierda**, selecciona el
   entorno **"Paper Trading"**. Importante: las claves de paper y las de
   real son pares distintos; mezclarlas da error de autenticación.
2. En la página principal, localiza **"Your API Keys"** (panel derecho;
   según versiones del sitio puede estar bajo *Settings/Account*).
3. Pulsa **View / Generate**: copia el **API Key ID** y el **Secret
   Key** en ese momento (el secreto se muestra una sola vez; si lo
   pierdes o se filtra, regenera el par — invalida el anterior).

**9.3. Conectar el agente (PowerShell, con el venv activo):**

```powershell
$env:ALPACA_API_KEY = "tu-key-id"
$env:ALPACA_SECRET_KEY = "tu-secret"
# NO definas ALPACA_BASE_URL: el defecto ya es el paper de Alpaca.

python scripts\run_live.py --broker alpaca --checkpoint checkpoints\dqn_best.pt
```

Verás cada operación reflejada en el dashboard de Alpaca (entorno paper)
mientras el agente opera. Las claves van **solo** en variables de
entorno, nunca en archivos del repositorio.

**9.4. Dinero real (el último escalón, cuando el paper convenza):**

```powershell
$env:TRADING_AGENT_LIVE = "YES"                        # habilitación consciente
$env:ALPACA_BASE_URL = "https://api.alpaca.markets"    # endpoint real
# y usa las API keys del entorno LIVE (no las de paper)
```

Sin **ambas** variables, el adaptador rechaza arrancar en real
(`SecurityError`). Empieza con capital pequeño y límites de riesgo
estrictos en `config.yaml`. Si usas intervalos intradía, ten presente la
regla *Pattern Day Trader* de EE. UU. (25.000 USD mínimos para más de 3
day-trades en 5 días hábiles).

## 10. Uso diario (resumen)

Cada vez que abras una terminal nueva:

```powershell
cd $HOME\pablinki.github.io\trading-agent
.\.venv\Scripts\Activate.ps1
python scripts\run_live.py --checkpoint checkpoints\dqn_best.pt
```

## 11. Solución de problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `python` no se reconoce | Python no está en el PATH: reinstala marcando *Add to PATH*, o usa `py -3.11` en su lugar. |
| *running scripts is disabled* | Ejecuta `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` (ver paso 3). |
| `pip install` muy lento o falla con torch | Red lenta/cortada: vuelve a ejecutar el comando (pip retoma la descarga). |
| `DataProviderError ... yfinance devolvió vacío` | Sin internet, o firewall/proxy corporativo bloqueando `finance.yahoo.com`. Prueba desde otra red. |
| Los tests fallan tras editar código | Ejecuta `pytest -v` para ver cuál y por qué; la suite es determinista. |
| Antivirus ralentiza el entrenamiento | Añade la carpeta del proyecto a las exclusiones de Microsoft Defender (opcional). |
| Alpaca responde `401/403` | Claves mal copiadas o de entorno equivocado: las keys de paper solo valen con la URL de paper (y las de live con la real). Regenera el par en el dashboard si hace falta. |
| `SecurityError: Trading en vivo deshabilitado` | Intentaste apuntar al endpoint real sin `TRADING_AGENT_LIVE=YES`. Es intencional (paso 9.4). |

## 12. Bloomberg (opcional)

Solo si tienes una **Terminal Bloomberg** instalada en esa laptop:

```powershell
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/

# Variables de entorno de la sesión (nunca en el código):
$env:BLOOMBERG_HOST = "localhost"
$env:BLOOMBERG_PORT = "8194"
```

y en `config.yaml` cambia `provider: bloomberg`. Sin Terminal, deja
`provider: yahoo` (el sistema completo funciona igual).
