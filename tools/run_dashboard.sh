#!/usr/bin/env bash
# Launch the dashboard wired to the project's LLM stack (Block 56):
#  - cloud keys from the gitignored .env (set -a so they export)
#  - the project-local Ollama client URL (port 11435) as the local fallback
# The agent then auto-selects: cloud-primary -> local Ollama -> offline heuristic.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export OLLAMA_HOST="http://127.0.0.1:11435"     # client URL for pattern_brain.llm
echo "Launching dashboard on http://127.0.0.1:8077 (cloud keys: $([ -f "$ROOT/.env" ] && echo loaded || echo none); local ollama: $OLLAMA_HOST)"
exec "$ROOT/.venv/bin/python" "$ROOT/dashboard/server.py"
