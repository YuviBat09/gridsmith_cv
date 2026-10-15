#!/usr/bin/env bash
# One-command setup + launch for the GridSmith pole-inspection app (macOS / Linux).
#   ./run.sh                 first run: creates .venv, installs deps, downloads the model, starts the app
#   PORT=8080 ./run.sh       use another port
set -euo pipefail
cd "$(dirname "$0")"

MODEL_URL="https://github.com/YuviBat09/gridsmith_cv/releases/latest/download"
PORT="${PORT:-7860}"

# 1. Python 3.10+
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3.14 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10+ is required (https://www.python.org/downloads/)"; exit 1; }

# 2. Virtualenv + dependencies (reinstalled only when requirements.txt changes)
if [ ! -x .venv/bin/python ]; then
  echo "==> Creating virtualenv with $PY"
  "$PY" -m venv .venv
fi
if ! cmp -s requirements.txt .venv/.requirements.installed 2>/dev/null; then
  echo "==> Installing dependencies (first run takes a few minutes)"
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  cp requirements.txt .venv/.requirements.installed
fi

# 3. Trained model from the latest GitHub release
mkdir -p models
for f in best.pt metrics.json; do
  if [ ! -s "models/$f" ]; then
    echo "==> Downloading models/$f"
    curl -fL --progress-bar -o "models/$f.part" "$MODEL_URL/$f" && mv "models/$f.part" "models/$f"
  fi
done

# 4. Launch (opens the browser once the server answers)
URL="http://127.0.0.1:$PORT"
echo "==> Starting app at $URL  (Ctrl+C to stop)"
( for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL"; then
      { command -v open >/dev/null && open "$URL"; } || { command -v xdg-open >/dev/null && xdg-open "$URL"; } || true
      break
    fi
    sleep 1
  done ) >/dev/null 2>&1 &
PORT="$PORT" exec .venv/bin/python app/app.py
