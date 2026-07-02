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
cd pablinki.github.io

# Mientras el PR no esté fusionado a main, cambia a la rama del agente:
git checkout claude/trading-agent-dqn-sc1u4p

cd trading-agent
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

Deben pasar **126 tests** (tardan menos de un minuto; no necesitan
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

El agente consulta el precio cada 60 s, decide (HOLD/BUY/SELL) y lo simula
sobre un portafolio virtual con comisiones y slippage. Se detiene
limpiamente con **Ctrl+C** (liquida la posición abierta antes de salir).

> **Seguridad:** este modo es 100% simulado. El trading con dinero real
> está deshabilitado por diseño salvo que definas explícitamente la
> variable de entorno `TRADING_AGENT_LIVE=YES` **y** conectes un adaptador
> de broker (no incluido).

## 9. Uso diario (resumen)

Cada vez que abras una terminal nueva:

```powershell
cd $HOME\pablinki.github.io\trading-agent
.\.venv\Scripts\Activate.ps1
python scripts\run_live.py --checkpoint checkpoints\dqn_best.pt
```

## 10. Solución de problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `python` no se reconoce | Python no está en el PATH: reinstala marcando *Add to PATH*, o usa `py -3.11` en su lugar. |
| *running scripts is disabled* | Ejecuta `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` (ver paso 3). |
| `pip install` muy lento o falla con torch | Red lenta/cortada: vuelve a ejecutar el comando (pip retoma la descarga). |
| `DataProviderError ... yfinance devolvió vacío` | Sin internet, o firewall/proxy corporativo bloqueando `finance.yahoo.com`. Prueba desde otra red. |
| Los tests fallan tras editar código | Ejecuta `pytest -v` para ver cuál y por qué; la suite es determinista. |
| Antivirus ralentiza el entrenamiento | Añade la carpeta del proyecto a las exclusiones de Microsoft Defender (opcional). |

## 11. Bloomberg (opcional)

Solo si tienes una **Terminal Bloomberg** instalada en esa laptop:

```powershell
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/

# Variables de entorno de la sesión (nunca en el código):
$env:BLOOMBERG_HOST = "localhost"
$env:BLOOMBERG_PORT = "8194"
```

y en `config.yaml` cambia `provider: bloomberg`. Sin Terminal, deja
`provider: yahoo` (el sistema completo funciona igual).
