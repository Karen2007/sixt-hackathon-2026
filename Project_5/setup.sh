#!/usr/bin/env bash
# Project 5 setup: creates .venv, installs pinned deps, runs the toy-example check.
# Usage: PYTHON=python3.11 bash setup.sh   (default: python3)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/python -m pytest -q tests
echo "Setup OK. Activate with: source .venv/bin/activate"
