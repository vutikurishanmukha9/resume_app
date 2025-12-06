FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# DO NOT set a fixed PORT here — Render provides $PORT at runtime
# ENV PORT=8080   <-- remove this line

# EXPOSE is optional for Render; leaving it is harmless but not required
EXPOSE 8080

# Bind to the runtime $PORT. Use 1 worker while debugging to avoid OOM/worker crashes.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --log-level info"]
