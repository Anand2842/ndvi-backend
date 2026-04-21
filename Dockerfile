# Ultra-minimal Railway deployment - optimized for <4GB
FROM python:3.10-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_minimal.txt .

# Install Python packages with aggressive optimization
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy only essential files
COPY main.py .
COPY model.py .
COPY download_model.py .

# Create models directory (model will be downloaded on startup)
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Start server (Railway uses $PORT environment variable)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
