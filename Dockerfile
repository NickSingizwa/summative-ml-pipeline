# API + dashboard in one container (runs locally, in docker-compose and on Azure Container Apps).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    PORT=7860 \
    WEB_CONCURRENCY=1

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --retries 10 --timeout 120 -r requirements.txt

COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app ui ./ui
COPY --chown=app:app models ./models
COPY --chown=app:app data ./data

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY}"]
