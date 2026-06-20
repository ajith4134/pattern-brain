# Pattern Brain

A standalone research/architecture project — explicitly **not connected to the trading bot** (`/opt/trading-bot`). Started 2026-06-20, named by the owner.

**Scope:** researching ML/statistical model families for pattern-finding, sequence prediction, probability estimation, noise/signal separation, and synthetic data generation, plus the harder architectural problem of letting many heterogeneous models share input data and exchange outputs (a "universal data translator" + "universal belief space / connector intelligence" design).

**Status:** still in the discussion phase — no code or deliverables built yet. See `DISCUSSION_NOTES.md` for the full, growing record of every idea discussed (owner's and Claude's). See `RULES.md` for how this project is run.

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

All future files for this project — code, data, deliverables — go here too, per Rule 1 in `RULES.md`.

**Automation:** this repo (`https://github.com/ajith4134/pattern-brain`) is cloned daily (06:00 UTC) by a scheduled cloud routine that researches the weakest-understood open items in `PLAN.md` and opens a PR with proposed findings — it never marks anything decided or writes implementation code on its own (Rule 24, `DISCUSSION_NOTES.md` Block 34). Every local change also gets pushed to this remote per Rule 24.
