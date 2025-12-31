# Kinetra Production Container
# Supports both AMD ROCm and NVIDIA CUDA

FROM python:3.11-slim

# Security: run as non-root
RUN useradd -r -u 1000 -m kinetra

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: libopenblas-base was renamed to libopenblas0 in newer Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kinetra/ /app/kinetra/
COPY scripts/ /app/scripts/
COPY rl_exploration_framework.py /app/

# Create directories for results (data mounted at runtime)
RUN mkdir -p /app/results /app/logs /app/data && \
    chown -R kinetra:kinetra /app

# Switch to non-root user
USER kinetra

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import kinetra; print('healthy')" || exit 1

# Default command: run backtest
CMD ["python", "scripts/batch_backtest.py", "--runs", "1"]
