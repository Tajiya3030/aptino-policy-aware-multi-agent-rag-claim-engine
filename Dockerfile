FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build RAG policy index
RUN python -m rag.indexer

# Render provides the PORT environment variable
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]