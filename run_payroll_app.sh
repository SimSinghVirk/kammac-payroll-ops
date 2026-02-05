#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8506}"
APP_PATH="/Users/simbizz/Documents/New project/app.py"
VENV_PATH="/Users/simbizz/Documents/New project/.venv"

if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti :"$PORT" || true)
  if [ -n "$PIDS" ]; then
    echo "Stopping existing process(es) on port $PORT: $PIDS"
    kill -9 $PIDS
  fi
fi

echo "Starting Streamlit on port $PORT"
if [ -d "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi
streamlit run "$APP_PATH" --server.port "$PORT"
