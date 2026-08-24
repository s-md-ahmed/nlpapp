FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel globally first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

# Tell pip to install without build isolation so it uses our global setuptools!
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

CMD gunicorn --bind 0.0.0.0:$PORT app:app
