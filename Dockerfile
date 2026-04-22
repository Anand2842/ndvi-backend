# Railway deployment - NDVI.AI Backend
FROM python:3.10-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements_torch.txt ./

# Install dependencies in two steps to avoid index conflicts
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_torch.txt

# Copy application files
COPY main.py .
COPY model.py .
COPY download_model.py .

# Create models directory (model downloaded from HuggingFace at startup)
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Start server - sh -c expands $PORT from Railway env
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
