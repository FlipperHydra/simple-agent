# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Persist memory and soul files in a named volume
VOLUME ["/app/memory.json", "/app/soul.md"]

# Default command — runs the interactive REPL
CMD ["python", "main.py"]
