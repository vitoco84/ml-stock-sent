#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Shutting down…"; kill 0 2>/dev/null || true' INT TERM EXIT

API_APP="app.api.main:app"
API_PORT="${API_PORT:-8000}"
UI_APP="app/ui/ui.py"

# Start Ollama if needed
if ! pgrep -x ollama >/dev/null; then
  echo "Starting Ollama…"
  nohup ollama serve >> ollama.log 2>&1 &
else
  echo "Ollama already running."
fi

echo "Starting FastAPI on :$API_PORT…"
uvicorn "$API_APP" --reload --port "$API_PORT" >> uvicorn.log 2>&1 &

# Wait for API to come up
echo "Waiting for API to be ready..."
for i in {1..10}; do
  if curl -s "http://localhost:$API_PORT/healthz" > /dev/null; then
    echo "API is ready."
    break
  fi
  sleep 1
done

echo "Starting Streamlit…"
streamlit run "$UI_APP"
