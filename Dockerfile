FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/

# Default local port; Render will set $PORT at runtime
EXPOSE 10000

# Use shell form so ${PORT} expands; default to 10000 if not set
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}
