#!/bin/zsh
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "Starting Internal Help Bot from: $ROOT_DIR"

if [ ! -f ".env.example" ] || [ ! -f "requirements.txt" ]; then
  echo "This launcher must run from the project folder."
  echo "Expected files .env.example and requirements.txt were not found in: $ROOT_DIR"
  echo "Run this file instead:"
  echo "  /Users/moldovancsaba/Projects/reply-amanoba/start_helpbot.command"
  exit 1
fi

PY_CMD=""
if command -v python3.12 >/dev/null 2>&1; then
  PY_CMD="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_CMD="python3.11"
else
  echo "Python 3.12 or 3.11 is required."
  echo "Install one of them, then run this file again."
  echo "Tip (Homebrew): brew install python@3.12"
  exit 1
fi

echo "Using $($PY_CMD --version)"

if [ ! -f ".env" ]; then
  cp .env.example .env
fi

if [ -d ".venv" ]; then
  VENV_PY_VERSION="$("./.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [ "$VENV_PY_VERSION" != "3.12" ] && [ "$VENV_PY_VERSION" != "3.11" ]; then
    BACKUP_DIR=".venv.backup.$(date +%Y%m%d-%H%M%S)"
    echo "Existing .venv uses Python $VENV_PY_VERSION (unsupported for this project)."
    echo "Moving it to $BACKUP_DIR and creating a fresh one."
    mv .venv "$BACKUP_DIR"
  fi
fi

if [ ! -d ".venv" ]; then
  $PY_CMD -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

MODEL="$(grep -E '^OLLAMA_MODEL=' .env | head -n1 | cut -d= -f2-)"
if [ -z "$MODEL" ]; then
  MODEL="qwen2.5:3b"
fi

RECOMMENDED_MODEL="qwen2.5:3b"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed."
  echo "Install it from: https://ollama.com/download"
  exit 1
fi

if ! curl -sSf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama service..."
  nohup ollama serve >/tmp/helpbot-ollama.log 2>&1 &
  sleep 3
fi

echo "Ensuring model is available: $MODEL"
if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$MODEL"; then
  ollama pull "$MODEL"
fi

if [ "$MODEL" != "$RECOMMENDED_MODEL" ]; then
  echo "Ensuring recommended multilingual model is available: $RECOMMENDED_MODEL"
  if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$RECOMMENDED_MODEL"; then
    ollama pull "$RECOMMENDED_MODEL"
  fi
fi

echo "Starting local API server..."
API_STARTED=0
if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8000 is already in use. Attempting to reuse existing server..."
  if ! curl -sSf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "Port 8000 is busy but /health is not responding."
    echo "Stop the existing process using port 8000, then run again."
    exit 1
  fi
  API_PID=""
else
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload >/tmp/helpbot-api.log 2>&1 &
  API_PID=$!
  API_STARTED=1
fi

cleanup() {
  echo "Stopping services..."
  if [ -n "${TUNNEL_PID:-}" ]; then kill "$TUNNEL_PID" >/dev/null 2>&1 || true; fi
  if [ "$API_STARTED" = "1" ] && [ -n "${API_PID:-}" ]; then kill "$API_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

for i in {1..40}; do
  if curl -sSf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -sSf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "API did not start correctly. Check /tmp/helpbot-api.log"
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing cloudflared with Homebrew..."
    brew install cloudflared
  else
    echo "cloudflared is required for public access."
    echo "Install it: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
  fi
fi

echo "Starting Cloudflare public tunnel..."
: > /tmp/helpbot-tunnel.log
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate >/tmp/helpbot-tunnel.log 2>&1 &
TUNNEL_PID=$!

PUBLIC_URL=""
for i in {1..80}; do
  PUBLIC_URL="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' /tmp/helpbot-tunnel.log | head -n1 || true)"
  if [ -n "$PUBLIC_URL" ]; then
    break
  fi
  sleep 0.5
done

if [ -z "$PUBLIC_URL" ]; then
  echo "Could not get public URL. Check /tmp/helpbot-tunnel.log"
  exit 1
fi

echo ""
echo "Local URL:  http://127.0.0.1:8000"
echo "Public URL: $PUBLIC_URL"
echo ""
echo "This single launcher now serves BOTH local and public access."
echo "Press Ctrl+C to stop."

(open "http://127.0.0.1:8000" >/dev/null 2>&1 &) || true
(open "$PUBLIC_URL" >/dev/null 2>&1 &) || true

wait "$TUNNEL_PID"
