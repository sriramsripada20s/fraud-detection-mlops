# Dockerfile — Fraud Detection API
#
# Build:
#   docker build -t fraud-api .
#
# Run:
#   docker run -p 8000:8000 fraud-api
#
# Test:
#   curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/      ./app/
COPY src/      ./src/
COPY model/    ./model/

# Environment variables
ENV MODEL_PATH=/app/model/model.joblib
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV PYTHONPATH=/app

# Health check — Docker will restart container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]