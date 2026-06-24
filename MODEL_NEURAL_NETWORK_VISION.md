# NORTH-STAR VISION — The Model Neural Network (models-as-neurons, LLM-style flow)
_Owner-set goal, 2026-06-23. The project's destination for Claude-as-ML-engineer (Rule 30)._

## The goal in one line
A graph where **each ML model is a neuron**, **model groups are layers**, **connections are learned
weighted synapses**, a **router activates experts attention-style**, **outcomes update edge strengths +
reputation**, and an **evolution engine grows/prunes the graph** — so data flows through the model-graph
the way activations flow through a Transformer/LLM. Named field: **Mixture-of-Experts / modular routing
networks**. Pattern Brain is already an implementation of this; the goal is to fully realize it.

## What already exists (~70%) — DO NOT rebuild
- model=neuron → `Node`; group=layer → `StackedDAG`; weighted connection → combiner (`mixture` skill-weighted /
  `stacked` ridge meta-learner / `gated` MoE regime-gate); upstream→downstream → meta-layer `belief_features`;
  multi-path → the DAG; universal OUTPUT → `Belief`+interlingua; reward→reputation → Evaluator+GA fitness;
  grow/prune → `evolution.py` Evolver (mutate/crossover/prune pathways).

## The 3 real gaps (the roadmap, in order)
1. **Universal Market Embedding** — a learned encoder that fuses all raw sources (candles, orderbook, funding,
   OI, news, sentiment, options, on-chain) into ONE shared latent vector every node reads. Highest-leverage,
   most LLM-like missing piece. (Today: each node sees raw generic `(T,D)`.)
2. **Residual stream** — carry a running representation through layers (skip connections) so information isn't
   destroyed across depth (the `0.9²⁰≈0.12` info-loss problem). 
3. **Attention-style soft router + gradient-trained edges** — upgrade `gated` from regime-accuracy weights to a
   learned softmax-attention gate over experts (top-k activation), trained by backprop on outcomes.

## THE CONSTRAINT — and how the field SOLVES it (UPGRADED 2026-06-23 via Rule 31, see IMPASSE_SOLUTIONS.md IMP-001)
First take was too pessimistic. "LLM-style end-to-end backprop" is ~80–90% ACHIEVABLE:
- **Differentiable reimplementations exist** for many classical experts → they become TRUE gradient neurons:
  differentiable Kalman (akloss/differentiable_filters, torchEnKF, BackpropKF), differentiable HMM
  (30stomercury/hmm-backprop), soft decision trees, DDSP.
- **Non-differentiable ops** (argmax routing, HDBSCAN, PySR) are handled by **Gumbel-Softmax** (differentiable
  routing), **Straight-Through Estimator**, or **REINFORCE** gradient estimates; or by **distillation** into a
  differentiable surrogate. JAX can autodiff through whole classical algorithms (JAX MD).
- So the realistic design: **gradient-train** the router + edges + every expert that has a differentiable form;
  **estimator/distill** the few that don't; keep **evolution** for graph STRUCTURE (topology), gradients for weights.
SENIOR CAVEAT (still binding): differentiable rewrites cost effort, REINFORCE is high-variance, and with limited
market data a full end-to-end net can overfit — so adopt in STAGES and keep Rule 30 (beat baselines OOS). The
ceiling is high, but earn each step.

## Realism guardrail (Rule 30)
LLM scale works on enormous data + full differentiability. We have limited trades + non-differentiable experts.
Expect a **well-routed MoE ensemble**, not "GPT for markets." Every addition still must beat simple baselines
OOS with significance (persistence/linear/threshold) — the architecture being elegant does not exempt it.

## Phased plan (each phase = a Rule-30 proposal, approval-gated, NOT yet started)
- **P0** — formalize the contract: every node already speaks `Belief`; standardize a common decision vector
  (e.g. {bullish, bearish, volatility, confidence}) as an interlingua belief-type so any node connects to any.
- **P1** — Universal Market Embedding encoder (gap #1).
- **P2** — Attention soft-router + gradient-trained edge weights (gap #3), replacing/extending `gated`.
- **P3** — Residual stream in the DAG (gap #2).
- **P4** — Evolution engine drives structure (grow/prune) on top of P1–P3 (mostly exists; wire to the new parts).
- Throughout: Claude invents the expert NODES (Rule 30) to populate the layers across all Axis-D categories.

## THE FIVE STAGES OF LEARNING — the model-neuron network's training curriculum (owner request 2026-06-24)
🟡 PROPOSED. Owner's frame: the network should learn in the **5 canonical ML paradigms — as STAGES (a curriculum
/ lifecycle the network passes THROUGH), not 5 switchable modes.** This is exactly how a modern foundation model is
actually trained (self-supervised pretrain → supervised finetune → RL finetune); we fold all five into one pipeline.
Each paradigm is a different **training SIGNAL** that owns one TIER of the existing models-as-neurons graph. They are
defined GENERICALLY on whatever the adapter feeds (Rule 23) — "pretrain on the raw source vectors", never "on candles".

The pipeline (bottom→top of the graph; each stage feeds the next):

1. **SELF-SUPERVISED — build the substrate.** The Universal Market Embedding encoder (gap #1 / P1) learns from ALL
   *unlabeled* history via masked-reconstruction + contrastive objectives (TS2Vec, PatchTST, TimeDART, diffusion-SSL/
   TSDE). No outcome labels needed → directly defeats the "limited trades" data constraint by learning from every bar.
   OUTPUT: the one shared latent vector every neuron reads. This is foundation-model pretraining.
2. **UNSUPERVISED — discover the map.** On/beside the latent: regime discovery, clustering (HDBSCAN already exists),
   density + change-point. No labels. The network organizes its own latent BEFORE being told any answer; produces the
   "where are we" regime context the router/gate consumes.
3. **SUPERVISED — train the experts.** The expert prediction-head neurons train on realized *labeled* outcomes
   (direction / return / trade-win). This is what most of today's bank already does. Each neuron becomes a specialist
   with a verifiable skill. Rule 30/32/36 gate stays in force — a neuron only KEEPs if it beats baselines OOS.
4. **SEMI-SUPERVISED — stretch the scarce labels.** Bridge the FEW labeled outcomes to the VAST unlabeled bars via
   confidence-thresholded pseudo-labeling + consistency regularization (Mean-Teacher / FixMatch style). Expands
   effective training data far past realized-trade labels — the core market label-scarcity fix. RISK (flagged): bad
   pseudo-labels self-reinforce/snowball → hard confidence gate + OOS validation are mandatory.
5. **REINFORCEMENT — train the decision-maker.** The top tier: (a) the attention soft-router (gap #3 / P2) — choosing
   top-k experts IS a policy over experts → trainable by policy-gradient / Gumbel-Softmax / REINFORCE; and (b) the
   trade-lifecycle policy (size / SL / TP / exit / allocation) on CUMULATIVE reward. Optimizes the true objective (PnL),
   not a proxy. HARDEST + highest-variance (REINFORCE noise, non-stationary reward, sample-inefficiency) → staged LAST,
   stabilized with 2025 MoE-RL techniques: Router-Shift Policy Optimization (RSPO), routing-replay, importance-sampling
   correction (T2MIR token+task MoE-for-RL).

**Two views of one plan:** these 5 stages and the P0–P4 phases are the SAME roadmap seen differently — Stage 1 = P1
(embedding), Stage 5(a) router = P2 (attention router), and P3's residual stream is what CARRIES the Stage-1 embedding
up through the depth. **Per-neuron view (complementary):** every neuron also carries a *native-paradigm tag* — an
autoencoder neuron is self-supervised, a cluster neuron unsupervised, a classifier neuron supervised, the router an RL
neuron — so the global stages and the per-node tags are consistent.
**It's a LOOP, not a one-shot line:** after Stage 5, new unlabeled data re-enters Stage 1 (continual re-pretraining)
under drift/forgetting guards — the network keeps learning, it doesn't freeze after one pass.
**Rigor (Rule 30/36):** the curriculum is NOT exempt — the SSL embedding must beat raw features OOS; the RL router must
beat today's `gated` router OOS; semi-supervised must beat supervised-only OOS. Adopt in stages, earn each one.

## Dashboard: animated Model-Neural-Network visualization (owner request 2026-06-23)
GOAL: the dashboard should look/feel like the LLM "Transformer Explainer" visualizations — the model-graph
with data visibly FLOWING through it, neurons lighting up, weighted edges pulsing.

ALREADY EXISTS (build on it, don't replace): `dashboard/index.html` uses **React Flow** (`reactflow@11`) —
builds the graph from a `DAGSpec` (input → layers of model nodes → combiner → output), edges already
`animated: true`, per-belief-type edge colors, an edge-reputation view, and a **`/ws/run` WebSocket** that
streams the run hop-by-hop so the "living graph" can animate belief-flow.

THE DELTA to reach Transformer-Explainer polish:
1. **Particle/packet flow along edges** — signal packets traveling source→target (requestAnimationFrame /
   react-spring), instead of React Flow's dashed-line "animated". Speed/density ∝ activation.
2. **Node activation glow + edge thickness = weight** — nodes light up as they fire (driven by `/ws/run`);
   edge brightness/width = learned connection weight; dim the experts the router didn't pick (top-k).
3. **Attention overlay** — when the soft-router (gap #3) lands, show its softmax weights as edge intensity,
   exactly like attention-head visualizations.
4. **Layer-by-layer left→right reveal** + the residual stream drawn as a persistent spine (gap #2 made visible).
5. **Scale**: keep React Flow (SVG) for ≤~150 nodes; if it grows, move to d3-force layout, and only go GPU
   (three.js / react-three-fiber, optional 3D) past ~100k elements (we won't be near that).

RECOMMENDATION: stay on React Flow + drive everything from the existing `/ws/run` stream; add a custom
animated-edge component (particles) + node-glow state. Visual target reference: poloclub Transformer Explainer
(open source). This is a SEPARATE, parallel track — it visualizes the architecture, it doesn't change behavior.

PLAN PLACEMENT: **P-VIZ** (parallel to P0–P4). Can start early since the data source (`/ws/run` + DAGSpec graph)
already exists; richer once P1–P3 add embeddings/attention/residual to show.

## Sources
- Transformer data flow: poloclub Transformer Explainer; DataCamp "How Transformers Work"; Towards Data Science
  mechanistic-interpretability (residual stream).
- Models-as-experts: Mixture of Experts (Wikipedia); "Modular Deep Learning" (arXiv 2302.11529);
  routing networks (arXiv 1904.08324).
- Animated viz: poloclub Transformer Explainer; "Building a Real-Time Neural Network Visualizer with React
  Three Fiber" (erikjs.com); react-graph-gallery network chart (D3 + d3-force); D3.js.
