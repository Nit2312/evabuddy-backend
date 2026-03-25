# Pin the patch version to match runtime.txt / .python-version (avoids surprise breakages).
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_DEBUG=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app

USER app

EXPOSE 8080

ENV PORT=8080 \
    WEB_CONCURRENCY=2

# Render sets PORT at runtime; gunicorn must listen on 0.0.0.0.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" > /dev/null || exit 1

CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --timeout 120 --access-logfile - --error-logfile - --capture-output"]
