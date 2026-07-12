FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer caches — only re-runs on requirements.txt change,
# not on every code edit. Biggest single Docker build-speed win available.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model_artifacts/ ./model_artifacts/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
