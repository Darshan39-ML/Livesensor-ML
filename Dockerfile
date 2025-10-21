FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for building some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port
EXPOSE 8080

# Use uvicorn to serve the app (imports main:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
