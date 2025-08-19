FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m pip install --upgrade pip

COPY app/ui/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ui/ /app/
EXPOSE 8501
CMD ["streamlit", "run", "app/ui/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
