FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for legacy numpy/scikit-learn builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render dynamically assigns the port via $PORT, so bind to 0.0.0.0:$PORT
CMD gunicorn --bind 0.0.0.0:$PORT app:app
