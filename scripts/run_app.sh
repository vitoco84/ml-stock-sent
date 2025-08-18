#!/usr/bin/env bash
set -euo pipefail

# Clean up everything we started when the script exits (Ctrl+C, error, etc.)
trap 'echo "Shutting down…"; kill 0 2>/dev/null || true' INT TERM EXIT

# Config
API_APP="app.api.main:app"
API_PORT="${API_PORT:-8000}"
UI_APP="app/streamlit/ui.py"

# Start Ollama if not running
if ! pgrep -x ollama >/dev/null; then
  echo "Starting Ollama…"
  nohup ollama serve >> ollama.log 2>&1 &
else
  echo "Ollama already running."
fi

# Start FastAPI (background)
echo "Starting FastAPI on :$API_PORT…"
uvicorn "$API_APP" --reload --port "$API_PORT" >> uvicorn.log 2>&1 &

# (optional) wait a moment for API to come up
sleep 5

# Start Streamlit (foreground)
echo "Starting Streamlit…"
streamlit run "$UI_APP"
