FROM python:3.11-slim

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + the CSV
COPY server.py rankings.csv ./

# Port for Render
ENV PORT=10000
ENV RANKINGS_CSV_PATH=/app/rankings.csv

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
