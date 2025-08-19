FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
# Render/Railway set $PORT; default to 8000 locally
CMD sh -c 'uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}'
