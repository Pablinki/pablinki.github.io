# Imagen de producción del trading agent (paper trading por defecto).
#
# Construir:  docker build -t trading-agent .
# Entrenar:   docker run --rm -v $(pwd)/checkpoints:/app/checkpoints trading-agent \
#                 python scripts/train.py --episodes 50
# Operar:     docker run -d --restart unless-stopped --name trader \
#                 -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/logs:/app/logs \
#                 -e ALPACA_API_KEY -e ALPACA_SECRET_KEY \
#                 trading-agent python scripts/run_live.py --broker alpaca
#
# Las credenciales entran SOLO por variables de entorno (-e), nunca en la
# imagen. El contenedor corre sin privilegios (usuario no-root).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (capa cacheable): torch CPU es la variante ligera.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY config.yaml pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# Usuario sin privilegios; volúmenes para checkpoints, caché de datos y diario.
RUN useradd --create-home --uid 1000 trader \
    && mkdir -p /app/checkpoints /app/.cache /app/logs \
    && chown -R trader:trader /app
USER trader
VOLUME ["/app/checkpoints", "/app/.cache", "/app/logs"]

# SIGTERM (docker stop) dispara la parada limpia con liquidación. Dale
# tiempo: `docker stop --time 120` (o stop_grace_period en compose); el
# cierre de la posición espera el fill de la orden (hasta 30 s).
STOPSIGNAL SIGTERM

# Un proceso "vivo" que ha dejado de operar se marca unhealthy: el
# heartbeat se reescribe en cada iteración (poll de 60 s -> 300 s de margen).
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "scripts/healthcheck.py", "--heartbeat", "logs/heartbeat.json", "--max-age", "300"]

CMD ["python", "scripts/run_live.py", "--config", "config.yaml", \
     "--checkpoint", "checkpoints/dqn_best.pt", "--poll-seconds", "60"]
