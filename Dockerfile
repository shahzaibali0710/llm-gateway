FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gateway ./gateway
COPY config.yaml .

# Numeric non-root user so Kubernetes can verify runAsNonRoot.
RUN useradd --system --uid 10001 --no-create-home gateway
USER 10001

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]

CMD ["uvicorn", "--factory", "gateway.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
