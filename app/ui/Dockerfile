FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/ui/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ui/ /app/

EXPOSE 8501
CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
