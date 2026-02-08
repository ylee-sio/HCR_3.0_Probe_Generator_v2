#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
if ! python -c "import tkinter" >/dev/null 2>&1; then
  echo "Tkinter not found in Python. Installing python-tk via Homebrew..."
  brew install python-tk
fi
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

exec uvicorn app.main:app --reload --port 8000
