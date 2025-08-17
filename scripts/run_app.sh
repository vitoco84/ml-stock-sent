#!/bin/bash

if ! pgrep -x "ollama" > /dev/null; then
  echo "Starting Ollama..."
  nohup ollama serve > ollama.log 2>&1 &
  sleep 5
else
  echo "Ollama already running."
fi

echo "Starting FastAPI backend..."
uvicorn app.api.main:app --reload --port 8000 &
FASTAPI_PID=$!

sleep 5

echo "Starting Streamlit UI..."
streamlit run app/streamlit/ui.py

# Stop FastAPI when Streamlit stops
echo "Stopping FastAPI backend..."
kill $FASTAPI_PID
