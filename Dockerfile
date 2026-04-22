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

# Run via Python directly — reads PORT from os.environ inside main.py
# No shell expansion needed, no $PORT issues possible
CMD ["python", "main.py"]
