# Pattern Brain

A standalone research/architecture project — explicitly **not connected to the trading bot** (`/opt/trading-bot`). Started 2026-06-20, named by the owner.

**Scope:** researching ML/statistical model families for pattern-finding, sequence prediction, probability estimation, noise/signal separation, and synthetic data generation, plus the harder architectural problem of letting many heterogeneous models share input data and exchange outputs (a "universal data translator" + "universal belief space / connector intelligence" design).

**Status:** **implementation started 2026-06-20.** Build-tracker **steps 1 (starter model bank) and 2 (Connector Intelligence v0 + the §8 dashboard walking skeleton) are complete** — see `pattern_brain/` and `dashboard/`. The rest of the build order is in `PLAN.md`. See `DISCUSSION_NOTES.md` for the full, growing record of every idea discussed (owner's and Claude's). See `RULES.md` for how this project is run.

**Code (`pattern_brain/`):** a domain-independent model bank. Every node operates on a generic `(T, D)` sequence and emits a `Belief` through one common interface (`fit`/`predict`/`transform`/`process`) — zero candle/order-book coupling (Rule 23). **30 node types across 8 functional layers** (signal, noise, pattern, sequence, probability, equation, decision, rl). Run `python3 demo.py` to watch the whole bank work on a synthetic sequence, or `python3 tests/test_bank.py` for the contract + behavior suite. `pattern_brain/connector.py` adds Connector Intelligence v0: one fixed, hardcoded pathway (`difference -> hdbscan -> gaussian_hmm -> threshold_policy`) threaded end-to-end through the Universal Belief Space — `python3 run_pathway.py` for the terminal view, `python3 tests/test_connector.py` for its tests. Language policy is polyglot by stage (`PLAN.md` §0b): Python now, C/C++ for proven hot paths later. Deps: `requirements.txt` (numpy/scipy/scikit-learn).

**Dashboard (`dashboard/`):** the §8 living-graph dashboard, ratified stack (Block 43): a FastAPI backend + a single-file React/ReactFlow/Plotly frontend (loaded via esm.sh + an import map — no build step). Watches Connector v0's hardcoded pathway run, hop by hop, over a WebSocket — nodes glow as they fire, edges animate, a belief-space stream and per-node confidence chart update live, and clicking a node opens its Model-Genome metadata. Domain-agnostic (Rule 23): synthetic `(T, D)` data only, no candles. Setup (separate venv — these deps aren't in the model bank's `requirements.txt`):
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dashboard.txt
.venv/bin/python dashboard/server.py        # serves http://127.0.0.1:8077
.venv/bin/python tests/test_dashboard.py    # contract + behavior suite (no live server needed)
```

**Planned features (added 2026-06-20, spec in `DISCUSSION_NOTES.md` Block 14):**
1. **ML Model Creation + Mutation** — create new instances of existing model families and evolve/mutate already-trained models (hyperparameters, architecture tweaks, partial retraining, crossover between two trained models), gated by an automatic evaluator before any mutation replaces an incumbent.
2. **ML Algorithm Creation + Mutation** — a level above #1: discover and evolve genuinely new algorithms/equations (not just new settings within a fixed one), via evolutionary search over program/equation space (AutoML-Zero / FunSearch / AlphaEvolve-style), same automatic-evaluator gate.

**Tooling (proposed 2026-06-20, spec in `PLAN.md` §8 / `DISCUSSION_NOTES.md` Block 38):**
- **Dedicated visualization dashboard** — a project-only dashboard to watch the system work in detail; centerpiece is the *living graph* (nodes=models, edges=connections coloured by pathway-reputation, animated belief-flow), plus node inspector, pathway leaderboard, evaluator/test panel, regime ribbon, evolution feed, and belief-space stream. Domain-agnostic core + a separate stock-specific "Adapter View" tab (per Rule 23). 🟡 proposed, not yet ratified.

**Open decision:** the format of the "cpu ml models" deliverable (curated download-manifest+script vs. committed weight files) — owner will decide once the discussion phase is done.

**Files in this folder:**
- `README.md` — this file, project overview.
- `RULES.md` — the rules this project runs by (kept separate from content, per Rule 3).
- `DISCUSSION_NOTES.md` — single growing log of every discussion (Rule 2/4).
- `PLAN.md` — current decision state: what's actually decided/planned/proposed/open, and (once building starts) implementation progress in order (Rule 21/22).
- `pattern_brain/` — the model bank + Connector Intelligence v0 (build steps 1-2).
- `dashboard/` — the §8 living-graph dashboard (FastAPI + React/ReactFlow/Plotly).
- `tests/` — contract + behavior suites (`test_bank.py`, `test_connector.py`, `test_dashboard.py`).
- `requirements.txt` / `requirements-dashboard.txt` — model-bank deps vs. dashboard-only deps, kept separate so the core bank stays light.

All future files for this project — code, data, deliverables — go here too, per Rule 1 in `RULES.md`.

**Automation:** this repo (`https://github.com/ajith4134/pattern-brain`) is cloned daily (06:00 UTC) by a scheduled cloud routine that researches the weakest-understood open items in `PLAN.md` and opens a PR with proposed findings — it never marks anything decided or writes implementation code on its own (Rule 24, `DISCUSSION_NOTES.md` Block 34). Every local change also gets pushed to this remote per Rule 24.
