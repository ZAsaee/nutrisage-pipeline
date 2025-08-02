# Stage 1: build dependencies
FROM python:3.10-slim AS builder
WORKDIR /app

# Install OS build-tools
RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential \
&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Stage 2: final image
FROM python:3.10-slim
WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/src/ /app/src/

# Ensure the local bin directory is in PATH
ENV PATH="/root/.local/bin:${PATH}"

# Expose the application port
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "80"]