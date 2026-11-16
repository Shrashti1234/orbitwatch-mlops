FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --system --create-home orbitwatch
COPY orbitwatch ./orbitwatch
COPY artifacts/metrics.json artifacts/*-model.joblib artifacts/*-timeline.csv ./artifacts/
USER orbitwatch
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=30s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()"
CMD ["python", "-m", "uvicorn", "orbitwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
