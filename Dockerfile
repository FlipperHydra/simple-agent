# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# All persistent state (memory.json, soul.md, conversation.json, session
# snapshots, output.txt) lives in a single data directory. Mounting a named
# volume over an individual FILE (the old approach) made Docker create a
# directory in its place and broke every read/write; mounting a single
# directory volume is correct and robust.
ENV AGENT_DATA_DIR=/app/data
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Default command — runs the interactive REPL
CMD ["python", "main.py"]
