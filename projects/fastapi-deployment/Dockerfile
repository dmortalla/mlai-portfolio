# Simple Dockerfile for deploying the Iris FastAPI service.
# Build with:
#   docker build -t iris-fastapi .
# Run with:
#   docker run -p 8000:8000 iris-fastapi

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \        build-essential \        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
