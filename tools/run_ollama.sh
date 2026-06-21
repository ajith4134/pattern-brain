#!/usr/bin/env bash
# Start the PROJECT-LOCAL Ollama server (Block 56) — separate from any system
# Ollama: its own binary (ollama-bin/), its own model store (ollama-models/), and
# its own port (11435), all inside the project folder (Rule 1). Independent of
# the rest of the VPS.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export OLLAMA_MODELS="$ROOT/ollama-models"
export OLLAMA_HOST="127.0.0.1:11435"          # server bind (host:port)
export LD_LIBRARY_PATH="$ROOT/ollama-bin/lib/ollama:${LD_LIBRARY_PATH:-}"
mkdir -p "$OLLAMA_MODELS"
echo "Starting project-local Ollama on $OLLAMA_HOST (models: $OLLAMA_MODELS)"
exec "$ROOT/ollama-bin/bin/ollama" serve
