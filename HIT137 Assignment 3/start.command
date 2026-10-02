#!/bin/zsh
# Start the puzzle on macOS, installing its Python packages locally if needed.
set -e
cd "${0:A:h}"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

if ! .venv/bin/python -c 'import cv2, numpy, PIL' >/dev/null 2>&1; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi

if [[ "${1:-}" == "--check" ]]; then
  .venv/bin/python -c 'import cv2, numpy, PIL, tkinter; print("Dependencies are ready.")'
  exit 0
fi

exec .venv/bin/python app.py
