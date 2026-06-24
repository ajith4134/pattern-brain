# Senior ML-Engineer Practices + Model-Invention Workflow — Pattern Brain
_Owner mandate 2026-06-23: Claude acts as the senior ML engineer who invents, names, builds, and
data-tests new models for the bank. This doc is the operating standard. Companion: RULES.md Rule 30._

---

## PART 1 — The senior-ML-engineer principles (researched, grounded)
What separates a senior from a junior is **disciplined skepticism about your own results.** The rules:

### A. Never leak the future (the #1 sin)
- **Split before you touch the data.** Any preprocessing (scaling, feature stats, target encoding)
  is fit on TRAIN only and applied to val/test — never fit on the full set.
- **Time-series = time-ordered validation only.** Walk-forward / expanding-window with an **embargo**
  gap between train and test (the bank's `Evaluator` already does CV with embargo). No shuffling,
  no random k-fold on temporal data, no using t+1 info at time t.
- Preprocessing happens **inside each fold**, not once before CV.

### B. Beat strong baselines, or it's nothing
A new model is only "good" relative to a baseline. Always benchmark against:
- **persistence / no-change** ("tomorrow = today") — the brutal baseline for forecasting,
- **majority class** (classification),
- **simple linear / logistic** on the same features,
- **fixed- and adaptive-threshold** rules,
- and the **existing bank nodes** it claims to improve on.
🔴 PROJECT LESSON (from our own history): advanced/exotic models usually **LOSE** to simple baselines
(reservoir/FNO/RG rejected vs AR; tensor-network rejected vs PCA; most eqgen exit-surfaces rejected vs
the trail). Assume the simple baseline wins until proven otherwise.

### C. Prove it isn't luck (statistical rigor)
- **Permutation / null test:** shuffle labels or returns, re-evaluate; the real score must beat the
  null distribution (the `Evaluator` supports permutation p-values). Report the p-value.
- **Multiple seeds:** report mean ± std over ≥3–5 seeds, not a single lucky run.
- **Paired significance:** Wilcoxon signed-rank (no normality assumption) or paired t-test on
  per-fold/per-trade differences vs the baseline; Bonferroni-correct when testing many variants.
- **Bootstrap 95% CIs** on the headline metric.
- **Out-of-sample only.** In-sample numbers are worthless for the decision. Beware
  selection-on-test (validating the winner you picked on the same split that picked it).

### D. Ablate compound models
If the model has parts, remove/replace/perturb each one under identical conditions and show each part
earns its keep. A gain you can't attribute is a gain you can't trust.

### E. Reproducibility & honesty
- Fix and log seeds, splits, hyperparameters. Identical conditions across compared variants.
- Report failures plainly (verify-before-report). "It ran" ≠ "it works." Cite the real metric.
- Prefer the simplest model that hits the bar (Occam / parsimony). Complexity must pay rent.

### F. Stress the model on at least one ADVERSARIAL regime (not just the clean panel)
A model that only ever sees benign data is untested where it actually fails. Before KEEP, evaluate every
invented model on **≥1 deliberately hostile regime** in addition to the standard panel (Rule 32):
- **ill-conditioning / scaling** — badly-scaled or highly-correlated features (judge on the task metric);
- **near-separability / extrapolation** — data that drives an unregularized fit to blow up (judge on
  **calibration**, e.g. probability divergence vs a regularized oracle — label accuracy alone HIDES
  overconfidence);
- **regime shift / distribution drift** — train and test drawn from different conditions.
🔬 PROVEN in our own benchmark (`code_skills_test/hard_panel.py`): on the easy panel a model can look
"library-equivalent" (5/5 EXCELLENT) yet a careless implementation still **diverges on scaling**
(agreement 0.75, acc −0.12) and is **mis-calibrated on separability** (97% label agreement but proba off
by 0.21). The hostile regimes are where code quality and rigor actually get tested. Record the adversarial
result alongside the panel result in `MODEL_PERFORMANCE_REPORT.md`.

### G. Test on the data the model was DESIGNED to take as input (design-input fidelity)  ⬅ owner-mandated 2026-06-24
**A model must be evaluated on the data its theory actually consumes — not on whatever series is lying
around.** A verdict produced on the wrong input type is INVALID (it both false-rejects models on data they
were never built for, and false-accepts on data that flatters them). This operationalizes Rules 34/35/36
inside the engineering standard, and is mandatory for every invented model:
1. **Name the design input at PROPOSAL time** (Rule 35): the data TYPE (returns? a *level/spread*? *order
   flow / signed trades*? a *covariance matrix*? a *realized-variance* series?), the asset/frequency, and
   the right evaluation **horizon** (a mean-reverter at its half-life; a microstructure model at tick/event
   scale; a vol model on RV — never raw returns).
2. **Feed it that input, transformed correctly.** A microstructure node gets *price + signed volume*, not
   close-to-close returns; an allocator gets a *cross-section of returns / a covariance*, not one series; a
   change-point node gets a series that *contains a regime shift*. If the node's `_predict` takes a generic
   `(T,D)` array, the EVALUATION HARNESS is responsible for handing it the design-appropriate `(T,D)` (e.g.
   stack signed-volume and price as the two columns) — document the mapping.
3. **If the design input isn't on the VPS, DOWNLOAD it first** (Rule 33, standing approval) and store under
   `data/`. Never substitute a convenient-but-wrong series, never synthesize for a verdict.
4. **The verdict must cite the design-input result** (plus the standard panel and ≥1 adversarial regime per
   §F, and — for a general-purpose method — ≥1 real non-trading domain per Rule 36). KEEP only if it earns
   its edge on the data it was built for and isn't harmful elsewhere; REJECT only after it fails on its OWN
   home turf.

**Design-input data map (current corpus — extend as models are added):**
| model family | design input (what its theory consumes) | data on the VPS |
|---|---|---|
| return forecasters (AR/HAR-on-returns/Koopman) | log-returns, 1-step (or stated horizon) | `data/panel/*.npz` (OHLCV→close) |
| volatility models (HAR-RV, GARCH, Heston, SV) | a **realized-variance** series (squared/again-aggregated returns) | derive RV from `data/panel/` returns |
| mean-reversion / cointegration (OU, Johansen) | a **stationary level / spread**, judged at its half-life | `data/meanrev/*_spread.npz` + reconstructed pairs |
| **microstructure (OFI, VPIN, Kyle λ)** | **signed trades: price + qty + side** (or L1/L2 book) | `data/ticks_micro/*.npz` (times, price, qty, side) ⬅ fetched 2026-06-24 |
| point processes (Hawkes) | **event arrival times** | `data/ticks/*.npz` (times) |
| allocators (HRP, CVaR, Merton, MPC) | a **cross-section of returns / a covariance** | `data/universe/returns.npz`, `data/panel/` |
| tail/risk (EVT, copula) | fat-tailed returns / joint returns | `data/panel/` returns (+ pairs) |
| general-purpose methods (manifold, signature, fuzzy) | a real **non-trading** benchmark too (Rule 36) | `data/sequences/pendigits.npz`; download more as needed |

---

## PART 2 — How a model enters THIS bank (the technical contract)
Read from the code (`node.py`, `registry.py`, `evaluator.py`):
1. **Subclass `Node`** (in a file under `pattern_brain/nodes/`).
2. **Set class attributes:** `node_type` (unique registry key, snake_case), `layer` (functional layer),
   `requires_y` (True if supervised), `is_transformer` (True if it emits a cleaned (T,D') sequence),
   `cost` ("low"|"med"|"high").
3. **Implement `_predict(self, X) -> Belief`** (and optionally `_fit` for learned/stateful nodes,
   `_transform` for filter/denoise nodes). Input is a validated finite `(T, D)` array (Rule 23:
   domain-agnostic — no candles/orderbook knowledge inside).
4. **Emit a `Belief`** of the right type (e.g. `forecast`, `signal`).
5. **Decorate the class with `@register`** (`from ..registry import register`) — adds it to the bank.
6. **Import the module in `pattern_brain/nodes/__init__.py`** so registration runs at import.
7. **Test via the `Evaluator`** (walk-forward CV + embargo + permutation null) and the leaderboard /
   capstone forward-test before claiming it belongs in the bank.

---

## PART 3 — The model-invention workflow (standing process when the owner gives an idea)
The owner gives a rough idea; Claude does the rest, gated by approval (Rule 27 / Rule 26):

1. **UNDERSTAND** — restate the idea; read goal/state/data; decide which Axis-D category it is
   (see ML_MODEL_TAXONOMY.md) and whether the bank already has something close.
2. **RESEARCH** — search the literature: is it novel? what's the correct formulation? what are the
   known failure modes and the right baselines? (cite sources).
3. **DESIGN proposal (PRESENT, get approval — do NOT build yet):**
   concept · the math · Axis-D category · Node-contract mapping (layer / node_type / belief / requires_y)
   · 2–3 candidate **names** · baselines it must beat · the exact eval plan (splits, null test, metric)
   · feasibility & risk · honest prior on whether it'll beat the simple baseline.
   → logged to IDEAS.md with an `IDEA-NNN` id, status `proposed`.
4. **BUILD (only after "implement"/approval)** — implement the Node, `@register`, wire into
   `nodes/__init__.py`, name it. Spec→test→code→self-verify (Rule 25).
5. **TEST ON DATA** — run the Evaluator vs the baselines on real data; permutation null; multiple seeds;
   OOS metrics + p-value + CI; ablate if compound.
6. **REPORT honestly** — keep / shadow / reject with the evidence. Only a model that beats its
   baselines OOS with significance earns a place in the bank (else it stays shadow or is dropped).
7. **RECORD** — update the idea's status (`implemented`/`dismissed`) and log the result.

### Naming convention for invented models
- `node_type` = terse snake_case registry key (e.g. `chaos_lyapunov`, `causal_transfer`).
- `name` (display) = a short human "brand" (e.g. "Lyapunov Chaos Sentinel").
- New math/equation-style models may also get a one-line "what it computes" tag in the docstring.

## PART 3 — The Code-Generation Contract (the HOW of writing each model's code)
PART 1–2 govern *rigor* (don't fool yourself); the Code-Generation Contract governs *code quality*
(write it right the first time). It is the binding 5-phase protocol **SPECIFY → GROUND → IMPLEMENT →
VERIFY → REFINE** — full text in **`CLAUDE.md`** (project root), embedded in the agent's runtime
`PERSONA` (`pattern_brain/agent/engineer.py`). Headline rule, backed by TDD-LLM studies
(+38.6% MBPP / +21.95% HumanEval), self-refine research, and our own logreg self-test: **write the
oracle/acceptance check FIRST, then code until it passes.** Self-measurement harness + scorecard live
in `code_skills_test/`. Derivation: `code_skills_test/HIGH_QUALITY_CODE_INSTRUCTIONS.md`.

## Sources
- IBM — Data Leakage in ML; Towards AI — split before preprocessing.
- arXiv 2512.06932 — hidden leaks in time-series forecasting; REFORMS reporting standards (arXiv 2308.07832).
- emergentmind / bestaiweb — controlled ablation study methodology.
- (Statistical tests: Wilcoxon signed-rank, permutation null, bootstrap CI, Bonferroni — standard practice.)
