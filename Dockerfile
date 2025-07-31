# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (excluding -e . for Docker)
RUN grep -v "^-e" requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . .

# Set PYTHONPATH to include src directory for imports
ENV PYTHONPATH=/app/src

# Set proper permissions
RUN chmod +x scripts/deploy.sh

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/models \
    /app/reports \
    /app/reports/figures

# Set environment variables for the application
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATA_SOURCE=local \
    MODEL_PATH=/app/models/nutrition_grade_model.pkl \
    METADATA_PATH=/app/models/model_metadata.pkl

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command to run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 