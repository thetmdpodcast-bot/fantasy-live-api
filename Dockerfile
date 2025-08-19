# ---- Base ----
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create app user and dir
WORKDIR /app

# System deps (minimal; add more if your requirements need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverages Docker layer caching)
COPY requirements.txt ./

# Install deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Render provides PORT; don't hardcode it
# (Expose is optional/helpful for local runs)
EXPOSE 10000

# Start the server. Render sets $PORT automatically.
# If your entry module is server.py with "app" inside, this is correct:
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "$PORT"]
