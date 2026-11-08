FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VENTUREGUARD_HOME=/app
WORKDIR /app
COPY requirements-runtime.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY src/ src/
RUN pip install --no-deps .
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY reports/ventureguard/permutation_importance.csv reports/ventureguard/permutation_importance.csv
COPY --from=frontend /ui/dist frontend/dist/
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "ventureguard.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
