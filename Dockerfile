FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required to build some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create non-root user to run the app
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose the application port
EXPOSE 8080

# Healthcheck endpoint for docker
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run with production-friendly uvicorn options
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
