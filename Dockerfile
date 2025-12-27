# Kinetra Production Container

FROM python:3.10-slim

# Security: run as non-root
RUN useradd -r -u 1000 -m kinetra

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kinetra/ /app/kinetra/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Create directories for results
RUN mkdir -p /app/results /app/logs && \
    chown -R kinetra:kinetra /app

# Switch to non-root user
USER kinetra

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from kinetra import PhysicsEngine; print('healthy')" || exit 1

# Default command: run backtest
CMD ["python", "scripts/batch_backtest.py", "--runs", "1"]
