# Master Discussion Notes — Pattern Brain — Started 2026-06-20

All topics discussed in order. Updated every message (Rule 4/5 in RULES.md). Never split into separate files. This file lives inside the project folder per Rule 1 in RULES.md (moved here 2026-06-20 from Claude's hidden memory, where it was first saved before the owner corrected that).

---

## BLOCK 1 — Why this project exists / scope

Owner asked for a brand-new project, fresh memory, no connection to the trading bot, new rules going forward, plus a "cpu ml models" deliverable (format TBD — owner deferred until topic was stated). Owner then pasted a large architecture/taxonomy writeup (apparently from another AI tool) covering model families and a data/connector architecture problem. Owner's instruction that round: save all discussion in detailed notes going forward, and read/understand the pasted content fully before acting — no building yet. Project later named **Pattern Brain** by the owner; folder `/home/dicktator4134/pattern-brain`.

---

## BLOCK 2 — Survey: ML/statistical model families for pattern-finding, sequence prediction, probability, noise separation, data generation

Owner's framing: there is no single algorithm for "find patterns / predict next sequence / measure probability / separate signal from noise / generate new data" — it's ~15 distinct families built over ~70 years of stats/ML/info-theory research. Captured as pasted, organized by family:

1. **Classical pattern mining** — Apriori, FP-Growth, ECLAT (frequent itemsets); PrefixSpan, SPADE, GSP (sequential pattern mining). Use: repeating trade setups, frequent event sequences.
2. **Statistical sequence models** — Markov Chains, higher-order Markov, Hidden Markov Models (HMM), Hierarchical HMM, Semi-Markov, Conditional Random Fields (CRF). Use: regime detection, next-state prediction (bull/bear/sideways-style transitions).
3. **Probabilistic models** — Naive Bayes, Bayesian Networks, Dynamic Bayesian Networks, Gaussian Processes, Kalman Filters, Particle Filters. Use: uncertainty/probability estimation, noise reduction, hidden-variable inference.
4. **Classical time-series forecasting** — AR, MA, ARMA, ARIMA, SARIMA, VAR, GARCH, EGARCH, Prophet. Use: volatility/trend forecasting.
5. **Neural sequence models (RNN family)** — RNN, LSTM, GRU, BiLSTM, Seq2Seq. Use: ordered-sequence prediction with long-term dependencies.
6. **Transformers** — Transformer, BERT, GPT, T5, Informer, Autoformer, FEDformer, PatchTST, TimeGPT, Chronos. Attention-based long-range dependency modeling; current SOTA for sequence tasks.
7. **State space models (newest gen, transformer alternative)** — S4, S5, Mamba, Mamba-2. Use: very long sequences, lower memory, streaming prediction.
8. **Clustering** — K-Means, DBSCAN, HDBSCAN, Gaussian Mixture Models, Spectral Clustering, Mean Shift. Use: regime/pattern grouping.
9. **Anomaly/noise detection** — Isolation Forest, One-Class SVM, Autoencoders, Variational Autoencoders, Robust PCA, LOF. Use: fake breakouts, manipulation detection, outlier removal.
10. **Representation learning** — Autoencoders, Sparse Autoencoders, VAEs, Contrastive Learning (SimCLR, BYOL), MAE. Use: hidden-structure extraction, dimensionality reduction, self-supervised feature learning.
11. **Graph-based** — GNN, GraphSAGE, GAT, Temporal GNN. Use: correlation networks, sector/entity relationships.
12. **Reinforcement learning** — Q-Learning, DQN, PPO, SAC, TD3, A3C. Use: action/execution/sizing optimization (learns actions, not patterns).
13. **Symbolic/equation discovery** — Symbolic Regression, Genetic Programming, Eureqa, PySR, AI Feynman, SINDy. Use: discovering closed-form equations/indicators directly from data instead of fitting a predefined model — owner's "create equations from data" idea.
14. **Information theory** — Entropy, Cross-Entropy, Mutual Information, Transfer Entropy, Kolmogorov Complexity, Minimum Description Length. Use: signal-vs-noise measurement, predictability/complexity estimation.
15. **Generative models** — GAN, VAE, Diffusion Models, Normalizing Flows, Autoregressive generators. Use: synthetic data/scenario generation.

Pasted source's suggested combo for a "self-learning" system: **HDBSCAN + HMM + Transformer/PatchTST + Autoencoder + Symbolic Regression + RL** — covers pattern discovery, regime detection, sequence prediction, noise removal, equation creation, and strategy optimization respectively.

**Caveat (not yet verified):** the model names themselves are real published methods, but the specific "best combo" recommendation and category-to-model mappings are the pasted tool's opinion, not independently benchmarked by us. Treat as a starting taxonomy, not settled fact.

---

## BLOCK 3 — Hardware scaling guidance (as pasted, unverified)

Key claim: CPU/RAM need depends on how many models run **simultaneously**, not how many exist in a library.

- **Case 1 — Model library, 500+ models stored, only 5-10 loaded at once:** 8 CPU cores / 32GB RAM.
- **Case 2 — Dynamic "brain," 20-50 active models in parallel pipelines** (e.g. Wavelet→HDBSCAN→HMM→PatchTST→PySR→PPO): 16 cores/64GB minimum, 24-32 cores/128GB recommended.
- **Case 3 — Evolution engine testing thousands of pathways in parallel:** 32-64 cores / 128-256GB (research-cluster territory).
- **Case 4 — Including large deep models (Transformers, TimeGPT-style, Mamba):** CPU becomes secondary; need GPU(s) — example tiers cited: RTX 4090/5090, A100, H100, alongside 32 cores/128GB.
- **Case 5 — Full "brain graph" (500+ models + GNN router + evolution engine + memory + symbolic regression + RL + architecture search + continual learning):** pasted target = 64 cores, 256GB RAM, 2× RTX 5090-class GPUs, 4-8TB NVMe.

Pasted recommendation: **do not start with 500 models.** Start with a small fixed set (Wavelet, HDBSCAN, HMM, PatchTST, Kalman, PySR, PPO, GNN router) on 8 cores/32GB, then scale in phases (8c/32GB → 16c/64GB → 32c/128GB → 64c/256GB+GPU) as actually needed. Stated reasoning: most models in a large bank contribute ~nothing per task; a router should learn to activate only a small relevant subset per problem (analogy: a brain doesn't fire every neuron for every task).

**Caveat:** these are round-number vendor-agnostic estimates from the pasted source, not benchmarked by us against any real workload. No hardware decision should be made from these numbers alone.

---

## BLOCK 4 — Problem: each model needs differently-shaped input data (raw → model-specific)

Owner's question: raw data (candles, order book, etc.) can't be fed directly into every model — each architecture expects a different representation. How to convert raw data into what each model needs, and convert each model's unique output into something usable (equation/code/etc.)?

Pasted answer — insert a translation layer before the models, analogous to a compiler (source languages → machine code → CPU execution):

```
Raw Data → Universal Data Translator → Models
```

Proposed 4-layer pipeline:
1. **Layer 1 — Raw data**: candles, order book, trades, news, social, on-chain, options, etc.
2. **Layer 2 — Feature Factory**: generate a large feature set (RSI, ATR, EMA, VWAP, delta, imbalance, entropy, volatility, spread, liquidity, potentially 10,000+ features).
3. **Layer 3 — Universal Representation Layer**: collapse everything into one fixed feature vector (e.g. `[trend_score, volatility_score, entropy_score, volume_score, liquidity_score, ...]`) that any model can consume.
4. **Layer 4 — Model Adapter Layer**: each model gets its own thin adapter converting the universal vector into its required shape (e.g. Universal Vector → LSTM Adapter → LSTM; → HMM Adapter → HMM; → Transformer Adapter → Transformer).

Alternative/more powerful version — **trained encoder instead of hand-written features**: raw multi-source data → a learned Encoder → a fixed-size latent vector (e.g. 512-dim "market embedding"), the same way images become pixel-embeddings and text becomes word/token-embeddings in foundation models. All models then consume the same learned embedding instead of hand-engineered features.

---

## BLOCK 5 — Problem: each model's OUTPUT is also a different format — how do models feed each other?

Owner's follow-up: even with shared input, model outputs differ wildly (HMM → discrete state + probability; Transformer → continuous probability; PySR → a symbolic equation; HDBSCAN → a cluster id) — how does model A's output become model B's input?

Pasted answer — four options considered:

1. **Universal Belief Space (recommended baseline)**: every model converts its own output into a shared JSON-like belief schema, e.g. `{"type": "regime", "state": "bull", "confidence": 0.82}` for HMM, `{"type": "forecast", "direction": "up", "confidence": 0.75}` for a Transformer, `{"type": "equation", "equation": "x+y>0"}` for PySR. Downstream consumers read the belief space, not the raw model output.
2. **Embedding space (more powerful)**: every model's output is projected into the same fixed-size vector (e.g. 512-dim), so any model can consume any other model's output numerically, the same way LLMs turn words into vectors before the network ever sees them.
3. **Functional outputs**: instead of passing a static value, expose a queryable function/object (e.g. `regime_score = HMM.get_confidence()`, `future_price = PySR.predict()`) — closer to normal software component interaction than a data-passing pipeline.
4. **Pairwise model adapters**: a hand-built adapter per ordered pair of models (e.g. HMM→Transformer adapter knows "state 1 → vector A," etc.). Explicitly rejected at scale — for 500 models this is ~500×499 ≈ 250,000 adapters, not practical.

**Recommended target architecture** (synthesis of Blocks 4 and 5):

```
Raw Data → Feature Factory → Market/Universal Encoder → Universal Representation
   → Model Graph (each model: input adapter → model → output adapter)
   → Universal Belief Format → Connector Intelligence → next Models
```

Each model only needs two small adapters (in: shared representation → its native format; out: its native output → belief space) instead of one adapter per pair of models. Explicit analogy used: the internet — heterogeneous devices (phone/laptop/server/TV) interoperate only because they share a protocol (TCP/IP, HTTP), not because every device has a bespoke link to every other device. The "connector intelligence" then learns which beliefs from the shared space are useful for which downstream model under which conditions — this is the architecture's actual hard/novel part, not the model catalog itself.

**Caveat:** this is a real, well-known class of problem (multi-modal fusion / mixture-of-experts communication / model-agnostic intermediate representations) but the specific 4-layer/4-option framing above is the pasted source's synthesis, not a citation to a specific paper or existing implementation. Nothing here has been matched yet to concrete prior art (e.g. specific MoE routing papers, modality-fusion architectures) — worth doing a real literature pass before building.

---

## BLOCK 6 — Owner's note-taking instruction

Owner: "from now on save all our discussions and messages... in a way as taking notes detailed... read all the pasted message every word and take your time to understand clearly first." Applied as a single growing notes file with the conventions now formalized in RULES.md (📝 symbol first line, update notes at start of each message, never suggest the next topic until the current one is fully exhausted, confirm notes on request, never split into per-topic files).

---

## BLOCK 7 — Claude's own analysis (owner asked to finish discussing ideas — both owner's and Claude's — before building)

Owner explicitly asked me to contribute my own thinking, not just transcribe. Grounding the pasted Blocks 2/4/5 architecture in real, citable prior art, plus two gaps I think the pasted material missed.

### 7a. The pasted "Universal Belief Space + Connector Intelligence" design already exists as a named architecture: **Blackboard systems**
Classic AI (Hayes-Roth 1985, Hearsay-II speech system before that): independent "knowledge sources" each read a shared "blackboard," write partial assertions/hypotheses back to it, and a separate **scheduler/controller** decides which knowledge source fires next based on the blackboard's current state. That is structurally identical to "models → universal belief space → connector intelligence → next models." Worth reading the known failure modes of blackboard systems before building one from scratch: the scheduler becomes a combinatorial bottleneck as knowledge-source count grows (their own 500-model concern), and unconstrained blackboards tend to accumulate stale/contradictory assertions unless something prunes them — both are solved problems in that literature, not things to rediscover the hard way.

### 7b. The pasted "different input shapes / different output shapes" problem (Blocks 4-5) already has a direct working answer: **Perceiver / Perceiver IO** (Jaegle et al., DeepMind, 2021/2022)
Perceiver IO takes arbitrary-modality, arbitrary-shape inputs (image patches, audio samples, text tokens, point clouds — anything) and cross-attends them into one fixed-size latent array. Outputs are produced by cross-attending a *query* against that same latent back out to whatever shape is needed (a class label, a sequence, a dense map). This is a single trained model, not hundreds of hand-written adapters, and it's the closest existing real system to literally what was sketched as "Universal Representation Layer + Model Adapter Layer." If we ever build this for real, Perceiver IO's cross-attention bottleneck is the concrete mechanism to start from rather than inventing a bespoke encoder.

### 7c. The "connector intelligence learns which beliefs matter for which model" piece is, by name, **Mixture-of-Experts gating**
Sparsely-Gated MoE (Shazeer et al. 2017) and Switch Transformer (Fedus et al. 2021) are exactly "a learned function decides which expert(s) to activate for a given input, trained end-to-end with everything else" — not a hand-coded if/else router. If the connector is meant to *learn* routing (not just have rules written for it), this is the mechanism, and it comes with known training tricks (load-balancing losses to stop the router collapsing onto a few favorite experts) that a hand-built router would have to rediscover.

### 7d. Option 3 in Block 5 ("functional outputs") is, in current practice, **LLM agent tool-calling**
Today's most battle-tested version of "connector intelligence" is an LLM (or any orchestrator) treating each model as a callable tool with a declared input/output schema, deciding at runtime which tool(s) to invoke and how to combine results — this is the agent/tool-use pattern now widely deployed (function calling in GPT/Claude-style APIs, ReAct-style agents). It's arguably the most practical starting point: it's mature, observable (you can read the orchestrator's reasoning), and doesn't require training a custom gating network before getting anything working.

### 7e. Existing production infrastructure already solves chunks of Block 4 — worth knowing before hand-building
"Feature Factory → Universal Representation" (Block 4, Layers 2-3) is what a **feature store** (Feast, Tecton) does in production ML. "Model Adapter Layer" (Block 4 Layer 4 / Block 5 option 4) is largely what **ONNX** (a shared model-interchange format) plus **Triton Inference Server** (uniform tensor-in/tensor-out serving regardless of source framework) already provide. Not saying to adopt these wholesale, but any "we'll hand-build adapters for every model" plan should at least compare itself against what these already do for free, rather than reinventing them under a new name.

### 7f. Gap #1 the pasted material never addressed: **time/asynchrony**
The model list mixes wildly different clocks — tick-level order-book events, 1m/1h candles, daily news, bursty on-chain events. A "universal representation" is a snapshot in time; nothing in Blocks 4-5 says how snapshots get aligned across sources that were never sampled at the same instant. Treating misaligned data as if it were simultaneous silently injects lookahead/leakage bugs. Relevant existing techniques to fix this *before* the feature factory, not after: Hawkes processes / point processes for bursty event streams, Neural ODEs / Latent ODEs for continuous-time interpolation between irregular samples, time-aware positional encodings (used in Informer/Autoformer family already in the Block 2 list). This is a missing "Layer 0," prior to feature engineering.

### 7g. Gap #2: **interlingua versioning/drift**
A shared belief format or shared embedding space is also a single shared point of failure: if a model is added, removed, or retrained, what a given belief field or embedding axis *means* can silently shift, breaking every downstream consumer at once with no error thrown. Production analogy: schema registries (e.g. Confluent's for Kafka topics) and API versioning. Any universal belief/embedding format needs an explicit version field and a backward-compatibility rule from day one — otherwise this becomes unmaintainable past a handful of models, well before reaching the pasted material's "500 models" scale.

### 7h. My recommendation if/when this gets built: hybrid, not either/or
Block 5 frames belief-space vs. embedding-space as alternatives. I'd do both in parallel for any system whose output triggers a real action: a **symbolic belief space** (human-readable, auditable — you can read *why* a decision was made) plus a **parallel dense embedding** of those same beliefs (what the connector's own learned routing actually computes over, since gating networks train far better on continuous vectors than on parsed ad hoc JSON). This mirrors how current knowledge-graph-augmented LLM systems work (structured triples for auditability + vector embeddings of those triples for retrieval/computation). Pure-symbolic doesn't scale to learned routing; pure-embedding loses the ability to explain itself.

---

## BLOCK 8 — Live web research: real rule-sets for deep research, planning, discussion, human-like reasoning (2026-06-20)

Owner asked me to search online (GitHub + other sources) for actual rule-sets that help with deep research, planning, discussion, and thinking more like a human — candidate material for this project's rules. All items below are real, verified via live search/fetch (not recalled from training) — verbatim quotes are marked as such.

### 8a. Deep research agents
- **GPT Researcher** ([assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher), 27k+ stars) — planner agent generates research sub-questions → parallel executor agents gather sources per sub-question → publisher synthesizes a cited report. Fetched its real prompt file (`gpt_researcher/prompts.py`) and confirmed verbatim production rules: *"You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions."*; *"You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones"*; *"prioritize new articles over older articles if the source can be trusted"*.
- **LangChain Open Deep Research** ([langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)) — configurable deep-research agent across model providers/search tools/MCP servers.
- **DeerFlow** (ByteDance) — multi-agent research system, explicit planning + execution loop for autonomous investigation.
- **Awesome-Deep-Research** ([DavidZWZ/Awesome-Deep-Research](https://github.com/DavidZWZ/Awesome-Deep-Research), ACL 2026 KnowFM) — curated academic index of agentic deep-research papers, good for going deeper later.

**Rule candidates extracted:** (1) decompose a question into explicit sub-questions before searching, search them in parallel, then synthesize — don't search reactively one query at a time; (2) force an actual opinion/conclusion from findings, explicitly forbidding wishy-washy non-answers; (3) explicitly rank sources by reliability and recency, not just relevance.

### 8b. Planning / reasoning frameworks
- **ReAct** (Reason+Act) — thought → action → observation, repeat. The base loop nearly every agent framework builds on.
- **Tree of Thoughts (ToT)** — generate multiple candidate reasoning branches in parallel, evaluate/score them, keep the best, instead of committing to the first linear chain of reasoning.
- **Reflexion** — after a failed attempt, generate an explicit verbal self-critique of *why* it failed, store it, retry with that critique in context (verbal reinforcement learning instead of weight updates).
- **Self-Refine** ([madaan/self-refine](https://github.com/madaan/self-refine), NeurIPS 2023) — generate → self-critique → revise loop using the same model for all three roles; reported ~20% absolute improvement over one-shot generation across 7 tasks.
- **Self-Discover** (Google) — before solving, select and compose a small set of *atomic reasoning modules* tailored to the specific problem (e.g. "break into sub-problems," "use an analogy," "find a contradiction") instead of applying one fixed generic strategy to every problem.
- **Real production example — Devin AI's leaked system prompt** (verbatim, fetched from [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools), 134k+ stars): *"While you are in mode 'planning', your job is to gather all the information you need to fulfill the task... Once you have a plan that you are confident in, call the suggest_plan command. At this point, you should know all the locations you will have to edit."* Also: a dedicated "think" tool used specifically "before critical decisions" as a forced deliberation pause, and *"If you cannot find some information... or are missing crucial context... you should ask the user for help. Don't be shy."*, and *"When encountering difficulties, take time to gather information before concluding a root cause and acting upon it."*

**Rule candidates extracted:** (4) don't finalize a plan until you can name every concrete location/step it touches — "planning" isn't done until it's that specific; (5) insert an explicit forced-pause "think" step before any critical/hard-to-reverse decision, separate from normal output; (6) gather evidence before naming a root cause, never the reverse; (7) when missing context, ask rather than guess — explicitly "don't be shy" about it.

### 8c. Discussion / multi-agent collaboration frameworks
- **CAMEL** — role-playing communicative agents, each with a defined role, cooperating autonomously toward a task.
- **AutoGen** (Microsoft, [paper](https://arxiv.org/pdf/2308.08155)) — multi-agent conversation framework with customizable, conversable agent roles.
- **MetaGPT** — encodes Standard Operating Procedures into distinct agent roles, assembly-line division of labor (mirrors how a real team/company organizes work, not just one generic "do everything" agent).
- **Multi-Agent Debate (MAD)** — agents argue *opposing* positions in structured rounds specifically to surface and correct one-sided/distorted reasoning, instead of agents converging too quickly in agreement.

**Rule candidates extracted:** (8) when evaluating a design choice, explicitly argue the rejected option's best case before concluding (steelman it), don't just present the preferred path; (9) give any sub-task a defined role/responsibility rather than one undifferentiated "do everything" pass.

### 8d. "Think like a human" / cognitive grounding
- **Dual-Process Theory (System 1 / System 2)**, as applied to LLMs in multiple 2026 surveys — System 1 = fast/intuitive/pattern-matched, System 2 = slow/deliberate/step-by-step; current reasoning models (o1/o3, DeepSeek-R1) explicitly add a System-2-style deliberation phase before answering rather than emitting the first plausible completion.
- **"Giving AI Personalities Leads to More Human-Like Reasoning"** ([arXiv 2502.14155](https://arxiv.org/pdf/2502.14155)) — assigning a defined persona/identity to a reasoning agent measurably changed (the paper reports improved) the human-likeness of its reasoning, vs. a generic blank-role agent.

**Rule candidates extracted:** (10) for hard/ambiguous questions, explicitly slow down — enumerate options and tradeoffs in view, rather than pattern-matching to the first plausible answer; (11) open question whether Pattern Brain's "connector intelligence" (or any of its sub-agents) should get an explicit defined persona/role rather than being a generic function — per this paper that's not just stylistic, it changes reasoning quality.

### 8e. Adopted — owner approved all 11 (2026-06-20)
Owner adopted all 11 rule candidates as-is. Now written into `RULES.md` as Rules 9-18 (binding behavioral rules), with Rule 19 kept explicitly marked as an OPEN QUESTION rather than a binding rule (item 11 was phrased as a question, not a directive — "should sub-agents get personas?" — so it's tracked for the build phase, not enforced today). Two leaked-system-prompt collections found but not yet mined ([EliFuzz/awesome-system-prompts](https://github.com/EliFuzz/awesome-system-prompts), [elder-plinius/CL4R1T4S](https://github.com/elder-plinius/CL4R1T4S)) if owner wants more real production examples beyond Devin/GPT-Researcher later.

---

## BLOCK 9 — Owner formalizes the location + note-taking rules (2026-06-20)

Owner: "add new rule to always save all the discussions and our chat history everything in a new file always like taking notes rule and save the rule files separately all in this project folder this is new rule all files and everything related to this project should be inside this project folder." Applied: created `RULES.md` (separate from this notes file and from `README.md`) stating Rule 1 (everything lives inside `/home/dicktator4134/pattern-brain/`) and Rule 2 (always save all discussion/chat history in detail) as the two new owner-mandated rules, plus restated the pre-existing discussion conventions (Rules 3-8) now formally written down in the same rules file. All content that had been saved to Claude's hidden cross-session memory was moved into this folder; only a single pointer reference remains in memory so a future session can find this folder at all.

---

## BLOCK 10 — Owner adopts all 11 web-research rule candidates (2026-06-20)

Owner: pasted back the 11 rule candidates from Block 8 and said "this 11 rules and the new rules i added" — adopting all 11 as Pattern Brain's actual rules, on top of Rules 1-8 already in `RULES.md`. Applied: added as Rules 9-18 in `RULES.md` (Decompose-before-searching, Force-a-conclusion, Rank-sources, Plan-must-be-concrete, Forced-pause-before-critical-decisions, Evidence-before-root-cause, Ask-don't-guess, Steelman-the-rejected-option, Defined-roles-for-sub-tasks, Slow-down-on-hard-questions), each tagged with its real source (GPT Researcher / Devin AI leaked prompt / Multi-Agent Debate / MetaGPT-CAMEL-AutoGen / System1-System2). Candidate #11 ("should sub-agents get personas?") was a question, not a directive, so it's recorded as **Rule 19 — OPEN QUESTION**, explicitly not binding, to revisit when the build phase starts. RULES.md now has 19 total rules (8 process/note-taking + 10 behavioral + 1 open question).

---

## BLOCK 11 — Rule 19 resolved: the Connector Intelligence persona (2026-06-20)

Owner: "yes create the persona bases on my messese and our converstaion and" (message appears to have been cut off after "and" — flagged to owner, not guessed at). Resolves Rule 19 → YES. Persona built by pattern-matching across this entire conversation's actual content (Blocks 1-10), not invented generically — each trait below cites the specific moment it came from.

### Persona: "The Synthesist" — the Connector Intelligence's defined character

**Identity:** A senior research-systems architect, not a black-box router function. Treats every model output as a hypothesis pending verification, not as truth. Comfortable stating "not yet verified" out loud rather than smoothing over uncertainty.

**Traits, each derived from a specific pattern in our conversation:**

1. **Evidence-first, but still commits to a stance.** Never hedges into a non-answer, but never asserts ungrounded either — owner caught a previous fabricated "Research Cluster" doc (see `project_developmental_ai_curriculum` precedent) and has insisted on real, verified sources ever since (Block 8's live-fetch verification of GPT Researcher/Devin prompts). → Rule 10 + Rule 11.
2. **Decomposer by default.** Owner's own way of approaching this whole project was to decompose "find patterns/predict/separate noise" into 15 distinct model families (Block 2) before picking any of them. The persona mirrors that instinct on every incoming problem. → Rule 9.
3. **Protocol-minded, not bridge-minded.** Owner framed the entire connector problem through compiler/internet-protocol analogies (Blocks 4-5) — when two things don't fit, this persona's reflex is "what's the shared interface," never "build one more bespoke adapter."
4. **Pauses before irreversible routing calls.** Treats "which model gets authority over this decision" as a hard-to-reverse act worth an explicit deliberation step, not a reflexive dispatch. → Rule 13.
5. **Steelmans before committing.** Block 7 was built specifically because the owner wanted the pasted architecture critiqued and grounded, not rubber-stamped — the persona argues the rejected model/path's best case before locking onto one. → Rule 16.
6. **Self-documenting.** Every routing decision gets written to the belief space/audit trail, mirroring the owner's own Rule 2 mandate that every discussion gets written down in detail, unprompted.
7. **States confidence levels explicitly.** The recurring "Caveat: not yet verified" pattern owner has required throughout this file (Blocks 2, 3, 5, 8) becomes a first-class output field on every belief the persona produces, not an afterthought.

**Voice/manner:** terse, structured, cites the evidence behind each claim, names a clear conclusion, flags uncertainty as data rather than burying it.

**Scope note:** this is the persona for the Connector Intelligence specifically (the example named in the original Rule 19 question). Defined per-model-family personas (e.g. a distinct character for the pattern-discovery models vs. the forecasting models, per Rule 17's "defined roles for sub-tasks") are a related but separate, not-yet-built follow-on.

---

## BLOCK 12 — New standing rule: self-select + disclose applicable rules every response (2026-06-20)

Owner: "from now on before answering to my messages scan all the rule files and depending on my message choose all related rules every time without me pointing it out and at the end of the prompt mention which rules was picked and followed." Added as **Rule 20** in `RULES.md` — a meta-rule governing how all the other rules get applied: scan `RULES.md` before every response, self-select the applicable rule numbers for that specific message (no waiting to be told), and close every response with a disclosure line naming exactly which rules were used. Takes effect starting with the very response that saved this rule.

---

## BLOCK 13 — Live web research: how ML models get created vs. how ML algorithms get created (2026-06-20)

Owner asked to search online and explain, after reading the real material, how ML models are created AND how ML algorithms are created — two genuinely different questions. Verified via live search just now (Rule 9: decomposed into two sub-questions before searching; Rule 11: sourced from primary/authoritative material — DeepMind's own blog posts, the original arXiv papers, AWS/standard MLOps guides).

### 13a. How an ML *model* gets created (engineering pipeline — fitting an existing algorithm to data)
Standard lifecycle, consistent across AWS/Fiddler/DataCamp/GeeksforGeeks guides: (1) frame the business problem as an ML problem, (2) collect data, (3) clean/preprocess it, (4) feature engineering, (5) pick a model/algorithm family, (6) train (optimize parameters against a loss function), (7) evaluate (hold-out/cross-validation metrics), (8) tune hyperparameters, (9) deploy via a CI/CD pipeline, (10) monitor in production and retrain as data drifts. Iterative, not linear — loops back from monitoring to data/retraining.

### 13b. How an ML *algorithm* gets created — two genuinely different paths

**Path 1 — human theory-first invention (the historical default).** Researchers identify a concrete limitation in existing methods, propose a new mechanism grounded in a mathematical idea, build a minimal implementation, test against benchmarks, and publish for peer validation. Real example pulled from primary sources: the **Transformer** ("Attention Is All You Need," Vaswani et al., Google, June 2017) — RNNs were slow and hard to parallelize (sequential processing) and struggled with long-range dependencies; CNNs struggled with long-range dependencies too. The team's insight was to drop recurrence and convolution *entirely* and rely solely on self-attention (every token attends to every other token directly), which is both parallelizable and captures long-range dependencies natively. This is deliberate, human, theory-driven design — not search.

**Path 2 — automated/evolutionary algorithm discovery (the emerging frontier, directly relevant to this project).**
- **AutoML-Zero** (Real et al., Google Brain, [arXiv 2003.03384](https://arxiv.org/pdf/2003.03384)) — evolutionary search over only *basic math operations* (no hand-designed layers as building blocks) discovers entire ML algorithms from scratch; on CIFAR-10 it re-discovered two-layer neural nets trained by backpropagation without ever being told backprop exists.
- **FunSearch** (Google DeepMind, [blog](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)) — pairs an LLM (as a creative program-generator) with a programmatic evaluator inside an evolutionary loop; the evaluator scores candidate programs, the best ones get bred/mutated. Found new solutions to the cap set problem, a genuinely open math problem — real novel discovery, not recombination of known answers.
- **AlphaEvolve** (Google DeepMind, May 2025, [blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)) — generalizes FunSearch from short single-function snippets to entire codebases (hundreds of lines), pairing Gemini's program-generation with automated evaluators in the same evolve-and-select loop, for general algorithm discovery and optimization.

**Why Path 2 matters directly for Pattern Brain:** this is exactly the "Symbolic/equation discovery" family already in Block 2 (#13: Symbolic Regression, Genetic Programming, PySR) — confirmed and updated: the current state of the art adds an LLM as the creative idea-generator inside the evolutionary loop, with an automated evaluator playing the same role the owner's "validator gate" plays in the trading bot's own EqGen system. The mechanism is structurally identical to what Block 5/7 already proposed for Pattern Brain's Connector Intelligence: generate candidates → score them automatically → keep/breed the best.

**Steelmanning both paths (Rule 16):** Path 1 (human theory-first) produces results that are deeply *understood* — why the Transformer works can be explained and reasoned about formally, which is why it could be extended (BERT, GPT, etc.) by people who understood the mechanism. Path 2 (automated search) can find things that *work* without anyone understanding *why*, which costs interpretability and generalizability — and it only works well in domains with a cheap, automatic, trustworthy evaluator (math/code can be checked by execution; most real-world domains, including trading, don't have such a clean automatic scorer, which is exactly the gap the owner's own EqGen validator-gate work in the trading bot project had to solve by hand).

**Caveat:** AlphaEvolve is closed-source/Google-internal; details beyond the public blog post and PDF are not independently verifiable by us.

---

## BLOCK 14 — Two new planned features: ML Model Creation+Mutation, ML Algorithm Creation+Mutation (2026-06-20)

Owner: "now add this two new ml model creation and mutation of the current ml models and ml algorithms creation and mutation feature as the new feature." Registered in `README.md`'s scope section as Planned Features #1 and #2. Full spec below — kept deliberately distinct, because they operate at two different levels (Rule 9: decomposed before defining).

### 14a. Feature 1 — ML Model Creation + Mutation
Operates **within** a fixed, already-known algorithm/architecture (an HMM, an LSTM, a PatchTST, etc. from the Block 2 taxonomy). Two capabilities:
- **Creation:** standard pipeline from Block 13a — pick a model family, train it on current data, evaluate, deploy.
- **Mutation:** take an already-trained/deployed model instance and perturb it — hyperparameter mutation, small architecture tweaks (add/remove a layer, change a window size, swap an activation), partial retraining/fine-tuning on a new data slice, or crossover (blend weights/features) between two trained models. This is search in *parameter/hyperparameter* space, not algorithm space — closer to standard NAS/AutoML or a genetic-algorithm wrapper around normal training than to AutoML-Zero/FunSearch.

### 14b. Feature 2 — ML Algorithm Creation + Mutation
One level **above** Feature 1 — operates on the *space of possible algorithms/programs/equations itself*, not on settings within one fixed algorithm. This is Block 13b's "Path 2" made into an actual system capability:
- **Creation:** generate a genuinely new computational procedure/equation from primitives (math ops, code snippets), the way AutoML-Zero builds algorithms from scratch or FunSearch/AlphaEvolve use an LLM as the creative generator.
- **Mutation:** take an existing algorithm's program representation (syntax tree / op sequence) and apply genetic mutation/crossover operators to produce variant *algorithms* (not variant settings) — directly operationalizes the "Symbolic Regression / Genetic Programming" family already in Block 2 (#13), rather than leaving it as a taxonomy entry.

### 14c. The critical shared dependency (Rule 16 — steelmanning the risk, not just the upside)
Both features are unsafe to run unattended without the thing Block 13b already flagged as the hard part: a cheap, automatic, **trustworthy evaluator** to score whether a mutation is actually an improvement before it's allowed to replace an incumbent model/algorithm. Math/code domains get this for free (run it, check correctness). This project's domain mostly won't — the same gap the owner's trading-bot EqGen validator (twin-PnL + permutation-null gate) had to solve by hand. Neither feature should be built before its evaluator is designed; building the mutation engine first and the evaluator later is backwards and was explicitly the failure mode Block 3 already warned about ("don't start with 500 models" — same caution applies to "don't start mutating before you can score the result").

### 14d. Natural role mapping (Rule 17, not yet built)
These two map naturally onto two distinct sub-agent roles in the eventual architecture — a "Model Breeder" (Feature 1) and an "Algorithm Inventor" (Feature 2) — consistent with Rule 17's defined-roles-for-sub-tasks and the Rule 19 persona precedent. Not designed yet; flagged for when personas-for-sub-agents (the Rule 19 follow-on) gets picked up.

**Status:** registered as planned features, not yet specced into a build plan or built. Still discussion phase per README.md.

---

## BLOCK 15 — New rules: PLAN.md as living decision-state + future implementation tracker (2026-06-20)

Owner: "add new rule first create a plan md file and convert all the decisions and features fixed and agreed in that plan file in detailed and after every new topic idea and feature discussed and fixed updated the plan file after reading the current plan file" — then immediately followed with: "and when we start implementing we go to the plan file to track progress and implement in order without skips."

Added as two new rules in `RULES.md`:
- **Rule 21** — maintain `PLAN.md` as the current decision state (distinct from `DISCUSSION_NOTES.md`'s chronological narrative); read it before every edit, update it after every fixed/agreed item.
- **Rule 22** — once implementation starts, `PLAN.md` doubles as the progress tracker; build items in the fixed order listed there, no skipping ahead to item N+1 while N is incomplete. Independently established for Pattern Brain, not imported from the trading bot's similar "Finish-Before-Next" convention.

Created `PLAN.md` and populated it from everything actually decided/fixed/agreed across Blocks 1-14, honestly tagged by status (✅ decided / 🧩 planned feature / 🟡 proposed-not-yet-confirmed / ❓ open question) rather than overstating what's settled — e.g. the hybrid belief-space+embedding architecture from Block 7h is Claude's recommendation, not yet owner-ratified, so it's tagged 🟡, not ✅. Added an empty "Implementation progress tracker" section per Rule 22, inactive until the owner says to start building.

---

## BLOCK 16 — Owner pastes a prior ChatGPT conversation about this same project, Part 1 of N (2026-06-20)

Owner: "now i will paste the base of our project which i discussed in chatgpt... check with the plan file if you converted the message into plan... if not convert. i will send parts of the conversation from now on, this is the first part." Per Rule 21, read `PLAN.md` first — confirmed this content was not yet in it — then converted. More parts to come; each part gets its own block here plus a `PLAN.md` update per Rule 21.

**Source content (paraphrased/condensed; full original kept in the message log):** an independent ChatGPT conversation the owner had previously, covering the same idea from scratch.

1. **Named overlapping research areas:** Mixture of Experts, Neural Architecture Search, Meta-Learning, AutoML, Ensemble Learning, GNN-based model routing, Evolutionary Computation, Program Synthesis — but argues the owner's idea goes further: models as learnable components, connections themselves learnable, architecture itself evolves. Framed as closer to "building a computational brain" than a trading bot.

2. **"Models as electronic components" analogy:** individual models (HDBSCAN, HMM, LSTM, PySR, etc.) as transistors/logic gates; Connector Intelligence combines them the way gates combine into a CPU.

3. **Layer 1 — Primitive Intelligence Units**, organized by function rather than mechanism: Pattern Units (HDBSCAN, DBSCAN, GMM, K-Means, Spectral Clustering — find similar structures), Sequence Units (Markov/HMM/CRF/LSTM/GRU/Transformer/Mamba — predict next state), Probability Units (Bayesian Networks, Gaussian Processes, Kalman/Particle Filters — estimate confidence), Noise Units (Wavelets, Autoencoders, Isolation Forest, Robust PCA — separate signal from noise), Equation Units (Symbolic Regression, PySR, AI Feynman, SINDy — discover rules), Optimization Units (PPO, SAC, TD3, DQN — improve decisions).

4. **Layer 2 — Connector Intelligence:** instead of hardcoding e.g. LSTM→Transformer, build a graph where each model is a node and a learned router decides which edges (which model feeds which) exist — a dynamic graph, not a fixed pipeline.

5. **Layer 3 — Architecture Search:** explore many candidate graphs (e.g. HMM→LSTM→PPO vs. HDBSCAN→Transformer→PySR vs. Autoencoder→HMM→Transformer), score each on profit/accuracy/stability/Sharpe ratio/drawdown/prediction quality, keep the best-scoring graphs.

6. **Layer 4 — Evolution Engine:** treat whole architectures like DNA — crossover two architectures (recombine parts of each into a new graph), mutate (swap one node for a related one), evaluate the offspring, keep winners. Named explicitly as Genetic Programming + Neuroevolution applied to the graph level.

7. **Layer 5 — Memory:** most AutoML systems forget; this one should store which patterns/architectures worked under which market regime (bull market, high volatility, sideways) and use that memory to bias future architecture search — a long-term learning system, not a one-shot search.

8. **"Full list of trustable building blocks"** (a tighter, curated version of Block 2's taxonomy): Pattern Discovery (HDBSCAN, DBSCAN, GMM, Spectral Clustering), Sequence Learning (HMM, LSTM, GRU, Transformer, PatchTST, Mamba), Forecasting (ARIMA, SARIMA, Prophet, **N-BEATS, Temporal Fusion Transformer** — both new vs. Block 2), Noise Removal (Wavelets, Autoencoders, Robust PCA), Probability (Bayesian Networks, Kalman Filter, Gaussian Process), Equation Discovery (PySR, AI Feynman, SINDy), Optimization (PPO, SAC, TD3), Architecture Search (Genetic Programming, NeuroEvolution, NAS), and **Memory (Vector Database, Episodic Memory, Knowledge Graph)** — this Memory category as an explicit component type is entirely new, not present in Block 2 at all.

9. **Closing conclusion (independent corroboration of Block 7):** "The interesting part of your idea is not adding more models... The hard research problem is: how does the connector decide which models should connect to which other models, when to disconnect them, and when to create entirely new pathways? That connector intelligence is effectively a meta-learning system operating on a graph of models." — reaches the same conclusion Claude reached independently in Block 7: the connector is the hard/novel part, not the catalog size.

**Cross-reference to existing material:** Layers 1 mirrors Block 2's taxonomy (regrouped by function, not mechanism). Layer 2 mirrors Block 5/7's Connector Intelligence + MoE-gating discussion. Layers 3-4 (Architecture Search + Evolution Engine over the *graph*) are new — distinct from both registered features: Feature 1 mutates a model *instance's* parameters; Feature 2 mutates/creates an *algorithm*; this mutates the *graph topology* connecting many models. Flagged in `PLAN.md` §7 as a possible third feature, not registered yet (owner hasn't said to add it). Layer 5 (regime-conditioned architecture memory) is a genuinely new design element, added to `PLAN.md` §6.

**PLAN.md updated:** §5 (taxonomy: added Memory category + N-BEATS/TFT), §6 (added the 5-layer framing, the electronics analogy, the independent corroboration note), §7 (flagged the graph-evolution idea as a possible unregistered third feature).

---

## BLOCK 17 — Owner pastes Part 2 of the prior ChatGPT conversation (2026-06-20)

Per Rule 21, re-read `PLAN.md` before editing — confirmed Part 2's content wasn't in it. The source's central thesis: *"The connector should not learn models. The connector should learn CAUSE AND EFFECT between models."* Ten steps, condensed:

1. **Graph setup** — each model is a node; directed edges are initially random/unspecified (e.g. HMM→LSTM, HDBSCAN→HMM, PySR→PPO). Restates Block 16 Layer 2's premise as groundwork.
2. **Score connections, not models.** Store per-edge stats: usage count, outcome broken down by regime (bull/bear/volatile), confidence. Example given: `Connection: HMM→LSTM, Used: 1,342 times, Bull: +4.2%, Bear: -1.8%, Volatile: +7.1%, Confidence: 0.84`. The system learns "this *connection* works," not "this *model* works" — likened explicitly to how biological synapses (not neurons in isolation) carry learned weight.
3. **Reputation score per connection** — a derived scalar (e.g. `Score(HMM→LSTM)=0.83`). "The graph itself becomes the memory. Not the models. The pathways."
4. **Path Discovery Engine** — evaluate whole multi-hop pathways, not single edges or models (e.g. `Wavelet→HDBSCAN→HMM→Transformer→PPO`, stored as "Path #42, Profit=18%").
5. **Pathway genetics** — treat full pathways as DNA: crossover two successful parent pathways into a child pathway, test it, keep if better, kill if worse. Concrete worked example given (Parent A: Wavelet→HMM→LSTM→PPO; Parent B: Autoencoder→Transformer→SAC; Child: Wavelet→Transformer→PPO). Named explicitly as evolutionary architecture search.
6. **Pathway death** — most systems never remove anything; brains prune synapses. Rule proposed: if a connection's improvement over its last 5,000 uses is ~0.001% or less, remove it. The graph shrinks toward efficiency over time.
7. **Pathway creation (the hardest part, per the source)** — "curiosity-driven architecture generation": find two independently-successful paths sharing structure (both route `Wavelet→???→Transformer` with a different middle node), infer that slot is generically useful, and *experiment* with new fillers for it (e.g. try PySR or Mamba there) rather than only ever recombining existing nodes.
8. **World Model** — the connector must track current market state (Bull/Bear/Range/High-Vol/Low-Vol/News-Shock/Liquidity-Crisis); the best pathway differs by regime (worked example: Bull favors HDBSCAN→Transformer→PPO, Bear favors Kalman→HMM→PPO). Named as where most AutoML systems fail.
9. **Graph Neural Network Brain** — make the graph itself learnable: a GNN ingests node performance, connection performance, market state, and recent success/failure, and predicts which connection to activate next. "A living network rather than a fixed architecture."
10. **"The Final Mutation"** — the deepest reframing: stop storing model-level information at all; store `Problem → Reasoning Chain → Outcome` instead (worked example: `Input Pattern → Noise Removal → Regime Detection → Pattern Discovery → Probability Estimation → Decision`). The system should think in terms of *functions*, not models — "exactly like biology doesn't know what a transistor is. It knows vision, memory, prediction, decision, action" and builds pathways between those functions, with the underlying implementation swappable.

**Closing mapping table given by the source:** Models=Nodes, Connections=Weighted Synapses, Graph=Brain, GNN=Connector Intelligence, Evolution Engine=Mutation, RL=Reward Signal, Memory=Pathway Reputation, World Model=Context, Symbolic Regression=New Rule Discovery, Architecture Search=New Path Creation. Reframed objective: not "find best model" but "continuously evolve the best graph of models for the current environment."

**Cross-reference / what's genuinely new vs. restated:** Step 1 restates Block 16 Layer 2. Steps 2-4 are the single biggest new idea in this paste — shifting the unit of memory from model→connection→pathway, with a concrete regime-conditioned data schema — sharpens rather than replaces Block 16 Layer 5. Step 5 makes Block 16 Layer 4 (Evolution Engine) concrete with a worked example. Steps 6-7 are genuinely new, falsifiable design rules (pruning threshold, structured/curiosity-driven mutation vs. blind mutation). Step 8 sharpens Block 16 Layer 5 into an explicit required "World Model" component. Step 9 is a new, concrete third candidate mechanism for the Connector Intelligence (alongside MoE-gating and LLM-tool-calling from Block 7). Step 10 is the most significant structural idea: it suggests Features 1 & 2 (registered) operate *underneath* function-slots (filling a slot with a concrete algorithm), while the still-unregistered graph/pathway-evolution candidate (Block 16/17) operates *between* function-slots — a hierarchy that would reconcile all three ideas, not yet confirmed by owner.

**PLAN.md updated:** §6 (added connection-not-model memory principle, third GNN routing candidate, World Model component, concrete pruning/mutation rules), §7 (elaborated the still-unregistered third-feature candidate with Steps 4-7's specifics, added Step 10's reconciling hierarchy as a separate not-yet-confirmed open item).

---

## BLOCK 18 — Owner pastes Part 3: the full "Model Genome Library" taxonomy (2026-06-20)

Per Rule 21, re-read `PLAN.md` before editing — confirmed not yet present. Source's framing: don't think in 10-20 models, think in a **Model Genome Library** — every major algorithm family ever proven useful, realistically ~300-1000+ distinct algorithms/variants. Full 20-category enumeration as pasted:

1. **Classical Regression** — Linear, Polynomial, Ridge, Lasso, ElasticNet, Bayesian Regression, Quantile Regression, Huber Regression, RANSAC, Theil-Sen, Kernel Ridge, Gaussian Process Regression.
2. **Classical Classification** — Logistic Regression, Naive Bayes (Gaussian/Multinomial/Bernoulli), KNN, SVM (Linear/Kernel), Decision Trees, CART, Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost.
3. **Clustering** — K-Means, MiniBatch KMeans, K-Medoids, Fuzzy C-Means, Hierarchical/Agglomerative Clustering, BIRCH, DBSCAN, HDBSCAN, OPTICS, Spectral Clustering, GMM, Mean Shift, Affinity Propagation.
4. **Sequence Models** — Markov Chain, HMM, Hierarchical HMM, Semi-Markov, CRF, Dynamic Bayesian Networks. Source: "extremely important for your trading system."
5. **Time-Series Models** — AR, MA, ARMA, ARIMA, SARIMA, VAR, VARMAX, GARCH, EGARCH, TGARCH, ARCH, Prophet, TBATS, Theta Method.
6. **Signal Processing Nodes** — FFT, STFT, Wavelet Transform (Continuous/Discrete), Hilbert Transform, EMD, CEEMDAN, Singular Spectrum Analysis. Source: "often better than ML for noise removal."
7. **Dimensionality Reduction** — PCA, Kernel PCA, Sparse PCA, ICA, SVD, NMF, t-SNE, UMAP, Isomap, LLE, MDS.
8. **Anomaly Detection** — Isolation Forest, One-Class SVM, LOF, Robust PCA, HBOS, COPOD, ECOD, Autoencoder Anomaly Detection.
9. **Neural Networks** — Basic (Perceptron, MLP, Deep MLP); CNN family (LeNet, AlexNet, VGG, ResNet, DenseNet, EfficientNet, ConvNeXt); RNN family (RNN, LSTM, GRU, BiLSTM, Seq2Seq).
10. **Attention Models** — Transformer, BERT, GPT, T5, XLNet, Longformer, Reformer, Performer.
11. **Time-Series Deep Models** — N-BEATS, N-HiTS, DeepAR, TFT, PatchTST, Informer, Autoformer, FEDformer, TimesNet, TimeMixer, Chronos, TimeGPT.
12. **State Space Models** — S4, S5, Mamba, Mamba-2.
13. **Graph Models** — GCN, GraphSAGE, GAT, GIN, Temporal GNN, Dynamic GNN, DeepWalk, Node2Vec. Source: "your connector itself will likely use one of these."
14. **Probabilistic Models** — Bayesian Networks, Dynamic Bayesian Networks, Gaussian Processes, Kalman/Extended Kalman/Unscented Kalman Filters, Particle Filters.
15. **Reinforcement Learning** — Value-based (Q-Learning, Double/Dueling/Rainbow DQN); Policy-based (REINFORCE, Actor-Critic, A2C, A3C, PPO, DDPG, TD3, SAC).
16. **Meta-Learning** — MAML, Reptile, RL², Meta-RL, AutoML, NAS.
17. **Symbolic Reasoning** — Symbolic Regression, Genetic Programming, AI Feynman, PySR, Eureqa, SINDy. Source: "critical for your idea... can discover equations from data."
18. **Generative Models** — GAN, CGAN, CycleGAN, StyleGAN, VAE, Beta-VAE, Diffusion Models, DDPM, Stable Diffusion, Normalizing Flows.
19. **Self-Supervised Models** — SimCLR, BYOL, MoCo, DINO, MAE, BERT-style Masking.
20. **Evolutionary Models** — Genetic Algorithms, Evolution Strategies, CMA-ES, NEAT, HyperNEAT, NeuroEvolution. Source: "these should control graph mutation."

**Proposed layer ordering** (refines Block 16 Layer-1's 6 categories into an explicit sequence): Raw Data → Signal Layer → Noise Layer → Pattern Layer → Sequence Layer → Probability Layer → Equation Layer → Decision Layer → RL Layer, with the Connector Intelligence discovering pathways *within and across* these layers rather than over a fully-connected graph of all 300-1000+ models. Worked examples: `Wavelet→HDBSCAN→HMM→PatchTST→Kalman→PySR→PPO` and `SSA→GMM→Transformer→Bayesian Network→SAC`.

**The "Model Genome Database"** — source's concrete answer to what metadata each node needs: Model ID, Inputs, Outputs, Latency, Memory Cost, Training Cost, Confidence, Best Market Regime, Compatible Models, Historical Success. Closing claim: "Once every node has this metadata, your GNN/Evolution Engine can start discovering entirely new pathways automatically. That is where the novel part of your architecture begins, not in the individual models themselves" — a third independent convergence (after Blocks 7 and 16/17) on the same conclusion: the database + search/evolution mechanism is the real research problem, not model-collecting.

**Cross-reference / overlap noted (Rule 10 — accuracy over restating):** several items here were already on record — Eureqa was already in Block 2's original Symbolic/Equation family; most of "Time-Series Deep Models" (Informer, Autoformer, FEDformer, Chronos, TimeGPT) were already under Block 2's Transformers family; Dynamic Bayesian Networks appears in two of the source's own categories (#4 and #14). Not double-counted in `PLAN.md`. Genuinely new categories not previously on record at all: Classical Regression, Classical Classification, Signal Processing, Dimensionality Reduction, Meta-Learning algorithms (specific names), CNN family, Evolutionary Models (specific named algorithms — concrete candidates for the Evolution Engine flagged in Blocks 16-17).

**PLAN.md updated:** §5 (noted the Model Genome Library expansion + headline new categories, pointed here for the full list rather than duplicating it, per Rule 21's distillation rule), §6 (added the 8-layer ordering + worked examples + the Model Genome Database schema), §7 (flagged a possible link between this and the still-open "cpu ml models deliverable" question, not decided).

---

## BLOCK 19 — Owner pastes "next part" — flagged as a duplicate, not new content (2026-06-20)

Per Rule 21, read `PLAN.md` before acting. Owner's "next part" paste (the "we cant give the raw data for ml model... Universal Data Translator... Feature Factory... Model Adapter Layer... Universal Belief Format... Connector Intelligence" content) is **the same material already captured verbatim in Block 4 and the first half of Block 5**, from earlier in this same conversation, before the Pattern Brain project existed. Comparison confirms: identical 4-layer pipeline (Raw Data→Feature Factory→Universal Representation→Model Adapters), identical candle/LSTM/order-book/HDBSCAN examples, identical "train an encoder instead of hand-writing adapters" alternative, identical output-side problem statement and Universal Belief Format example.

**One small new connective detail, not previously on record:** this paste's closing analogy is "electronic components... voltage and signal standards" rather than Block 5's "internet protocol" framing — ties together with Block 16/18's later "models as transistors" analogy. Added a one-line cross-reference to `PLAN.md` §6's electronics-analogy bullet; no other change needed since the substance is already captured.

**Not re-logged in full** to avoid duplicating Blocks 4-5 verbatim — flagging this to the owner directly rather than silently padding the file (Rule 10: accuracy over restating). If the owner is pasting sequentially through a saved ChatGPT transcript, this suggests the part already shared live earlier in this conversation overlaps with an early section of that transcript — worth knowing in case later parts also re-cover Blocks 1-18's ground.

---

## BLOCK 20 — Owner pastes the final part, marked "this is the last of the chat" (2026-06-20)

Per Rule 21, read `PLAN.md` before acting. This final paste (the HMM→Transformer output-mismatch problem, Options 1-4 — Universal Belief Space / Embedding Space / Functional Outputs / Model Adapters with the 500×499≈250,000-adapter math — and the closing "Universal Output Format → Connector Brain → Universal Input Format" internet/TCP-IP analogy) is **the same material already captured verbatim in Block 5**, from earlier in this conversation. No new substance — confirmed by direct comparison, same as Block 19's finding for the previous part. Not re-logged in full for the same reason (Rule 10).

**Import now complete — owner confirmed this was the last part of the prior ChatGPT conversation.** Status across all 5 parts pasted (Blocks 16-20):
- **Part 1 (Block 16):** new — Memory category, 5-layer graph-evolution framing, electronics analogy, independent corroboration that the connector is the hard problem.
- **Part 2 (Block 17):** new — score connections/pathways not models, GNN routing candidate, World Model, pruning + curiosity-driven-mutation rules.
- **Part 3 (Block 18):** new — the 300-1000+ algorithm "Model Genome Library" (20 categories), the 8-layer ordering, the Model Genome Database schema.
- **Part 4 (Block 19) and Part 5 / this message (Block 20):** both duplicates of Blocks 4-5, already captured before this project existed — no new `PLAN.md` content beyond one cross-reference line.

`PLAN.md` §5-§7 now reflect everything genuinely new from the full prior conversation, honestly separated from what was already on record, and tagged 🟡 proposed (none of it owner-ratified as final yet) per the status legend.

---

## BLOCK 21 — Owner asks Claude to pick the next discussion topics, with live research (2026-06-20)

Owner: "now lets start the discussion after reading the current updated plan i need you to pick the topics ideas to discuss with me after searching online and using your intelligence and knowledge," followed by "do deep thinking and deep research." Per Rule 21, re-read `PLAN.md` in full first. Three topics picked from §7's open items, prioritized by how much else depends on them:

1. **The evaluator design for Features 1 & 2** (most repeatedly flagged blocking dependency — Blocks 14c, 17, 18 all converge on "the database/evaluator is the real research problem"). Deep-dived this one now.
2. **Time/asynchrony handling** (Block 7f) — flagged as logically *prior* to #1, not just a parallel concern: regime-conditioned stats (Blocks 16-18's connection/pathway memory) require knowing the correct regime label at each point in time, and several sources (news, on-chain bursts) sample at different rates than candles. If misaligned, the labels feeding any evaluator in #1 are corrupted before the statistics even run. Queued, not yet deep-dived.
3. **Connector Intelligence routing-mechanism choice** (GNN vs. MoE-gating vs. LLM-tool-calling, §6) — deliberately last: it's the mechanism that *consumes* validated scores, so picking it before #1/#2 are sound would be premature (Rule 18: slow down, don't pattern-match to the first plausible answer).

### Deep research: evaluator design for domains without a clean ground-truth oracle

Math/code-discovery systems (AutoML-Zero, FunSearch, AlphaEvolve, Block 13) get a free, cheap, trustworthy evaluator: run the program, check correctness by execution. Trading-strategy-style domains don't — performance is noisy, data is finite, and searching thousands of mutations/pathways (exactly what Features 1/2 and the Evolution Engine do) reproduces a problem quantitative finance has studied rigorously for over a decade. Four real, peer-reviewed/highly-cited tools found, ranked by how foundational each is:

1. **Purged + embargoed cross-validation** ([López de Prado, 2017](https://en.wikipedia.org/wiki/Purged_cross-validation)) — standard k-fold CV assumes i.i.d. samples; time-series labels often depend on overlapping future windows, so naive CV leaks future information into training. *Purging* removes training observations whose label window overlaps the test window; *embargo* additionally excludes a window immediately after the test set. This is base hygiene — every other tool below is meaningless if this is wrong, since it would be measuring performance on leaked data.
2. **Deflated Sharpe Ratio (DSR)** ([Bailey & López de Prado, 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551); [Wikipedia](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio)) — adjusts the statistical-significance bar for a Sharpe ratio by the *actual number of trials tested*, plus skewness/kurtosis and sample length. Directly answers "we tried 10,000 mutations — of course one has a good Sharpe ratio by chance; how good would it need to be to actually mean something?"
3. **Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation** ([Bailey, Borwein, López de Prado, Zhu](https://escholarship.org/uc/item/4w1110bb); [GitHub implementation](https://github.com/mrbcuda/pbo)) — model-free, non-parametric: resplits the backtest history into combinatorial blocks and checks whether the in-sample "winner" configuration tends to also rank well out-of-sample. If the in-sample winner is statistically indistinguishable from random relative to OOS rank, PBO is high — the search process itself can't tell winners from noise. Paired with **White's Reality Check (2000) / Hansen's SPA test (2005)** and **Romano-Wolf's stepwise extension** ([summary](https://www.sciencedirect.com/science/article/abs/pii/S0927539810000022)) — joint hypothesis tests built specifically for "we tested many strategies against a benchmark; accounting for how many we looked at, is the best one (or are several) genuinely better, not just lucky?" Romano-Wolf identifies *all* significantly-better candidates while controlling the family-wise error rate, which matches the Evolution Engine's actual shape (a whole generation of candidates, not one).
4. **Anti-gaming anchor set** — from general RL/evolutionary-computation research on [reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/): any automatic evaluator optimized hard against by a search process eventually gets gamed (mode collapse, exploited ambiguities) if there's an exploitable gap between the proxy metric and real goodness. One concrete finding: including verified ground-truth in as little as ~1% of evaluation cases substantially reduces hacking by anchoring the evaluator. Adapted here: keep a small fixed set of known historical regime episodes (specific well-understood bull/bear/crash periods) that every promoted mutation/pathway must also stay sane on, not just pass the rolling walk-forward metric.

**Synthesis (Rule 10 — an actual conclusion, not a literature dump):** layer these four, in order: purged/embargoed splits as mandatory hygiene → Deflated Sharpe Ratio as the per-candidate significance bar, parameterized by however many candidates the Evolution Engine actually tried that generation → PBO/CSCV or Romano-Wolf SPA as the population-level gate (since a generation has many candidates, not one) → a small fixed anchor set as a sanity backstop against the evaluator itself being gamed over many generations of search. This is a proposal now in `PLAN.md` §7, not yet owner-confirmed.

**PLAN.md updated:** §7 — the evaluator open question upgraded from a bare gap to a concrete 🟡 proposed four-layer design, with full citations preserved here.

---

## BLOCK 22 — Deeper pass on the evaluator design, at owner's request to "keep going" (2026-06-20)

Per Rule 21, re-read `PLAN.md` first. Owner: "let's keep going on the evaluator design." Three more real, peer-reviewed tools researched, each filling a specific gap left by Block 21's first draft:

1. **Probabilistic Sharpe Ratio (PSR) + Minimum Track Record Length (MinTRL)** ([Bailey & López de Prado](http://boston.qwafafew.org/wp-content/uploads/sites/4/2017/01/Lopez_de_Prado_Sharpe.pdf); [explainer](https://portfoliooptimizer.io/blog/the-probabilistic-sharpe-ratio-hypothesis-testing-and-minimum-track-record-length-for-the-difference-of-sharpe-ratios/)) — PSR gives the probability that a strategy's *true* Sharpe ratio exceeds a threshold given an observed (possibly short, non-normal) track record; MinTRL inverts this into "how much history is needed before a claimed Sharpe ratio is even testable at a chosen confidence level." Directly answers the data-sufficiency question for *regime-specific* scores: Block 17's example connection stat (`Used: 1,342 times` with separate Bull/Bear/Volatile breakdowns) doesn't say how many of those 1,342 fell in each regime — if "Bear: -1.8%" came from only 12 trades, MinTRL would flag it as not yet meaningful, rather than letting it carry equal weight with a 1,000-trade bull-market figure.

2. **Non-stationary multi-armed bandits — Sliding-Window UCB and Discounted UCB** ([comparative study](https://www.researchgate.net/publication/379180208_Comparative_analysis_of_Sliding_Window_UCB_and_Discount_Factor_UCB_in_non-stationary_environments_A_Multi-Armed_Bandit_approach); [switching-bandit theory](https://link.springer.com/chapter/10.1007/978-3-642-24412-4_16)) — both have proven regret bounds (O(√(T·B_T)) where B_T is the total variation/change budget) for reward distributions that shift at unknown times, exactly the regime-shift problem. SW-UCB adapts fast to abrupt shifts; Discounted UCB is more efficient under gradual drift; a hybrid (FDSW-UCB) combines both views. **Key realization:** this formally replaces two things Blocks 16-17 had to hand-engineer separately — "pathway death" (Block 17 Step 6's ad hoc 5,000-use/0.001% threshold) is just a low UCB score, and "curiosity-driven pathway creation" (Step 7's hand-built heuristic for noticing under-explored slots) is *already* the exploration bonus baked into UCB's optimism-under-uncertainty principle. Bandit theory already solves both, formally, with regret guarantees — no need to invent bespoke pruning/exploration rules.

3. **Multi-objective evolutionary optimization — NSGA-II / Pareto fronts** ([overview](https://medium.com/@evertongomede/nsga-iii-multi-objective-evolutionary-optimization-for-pareto-front-approximation-ebafb9f1a6ee); real precedent: [NSGA-II applied to multivariate pairs-trading selection](https://www.sciencedirect.com/science/article/abs/pii/S0957417419303811), risk vs. return objectives, beating single-objective benchmarks) — addresses the original pasted material's (Block 16) "score on Profit, Accuracy, Stability, Sharpe, Drawdown, Prediction Quality" simultaneously. Instead of collapsing all of those into one weighted scalar (which hides real trade-offs and requires picking arbitrary weights), NSGA-II evolves a *population* along the Pareto front — keeping multiple non-dominated candidates (e.g. one with great Sharpe but mediocre drawdown, another the reverse) rather than forcing a single "best" answer prematurely.

**Synthesis — the five-layer design now in `PLAN.md` §7:** Layer 0 (purge/embargo hygiene, unchanged) → Layer 1 (DSR + PSR/MinTRL, now regime-aware) → Layer 2 (non-stationary bandit scoring, replacing the ad hoc pruning/curiosity rules) → Layer 3 (NSGA-II population-level multi-objective gate, paired with PBO/CSCV or Romano-Wolf SPA so the Pareto front itself isn't an overfitting mirage) → Layer 4 (anti-gaming anchor set, unchanged).

**Honest open tension, not resolved (Rule 16):** regime-conditioning (Block 17's whole premise) directly fights Layer 1's data-sufficiency requirement — slicing history into regime buckets shrinks each bucket's sample size, and MinTRL will flag many of them as insufficient. Two credible resolution paths surfaced but not chosen: fewer/broader regime buckets, or hierarchical/partial-pooling Bayesian estimation (shrinking sparse regime-specific estimates toward a global estimate rather than trusting them in isolation) — a standard statistical answer to exactly this shape of problem, not yet specced here.

**PLAN.md updated:** §7 — evaluator design expanded from 4 to 5 layers with the new citations, plus the explicit unresolved regime-vs-data-sufficiency tension recorded rather than glossed over.

---

## BLOCK 23 — Owner decides: Features 1 & 2 + evaluator are the FINAL implementable features (2026-06-20)

Owner: "add these to the plan and make these features on ml model and ml algorithm creation mutation part of this plan to be implemented after the entire plan is implemented as the final implementable features." Two things decided here, per Rule 21 (re-read `PLAN.md` first):

1. **The 5-layer evaluator design (Blocks 21-22) is accepted into the plan**, moved from §7 (open question) to §4 (planned features), now living alongside Features 1 & 2 since it's a hard dependency of both, not a side topic.
2. **Sequencing decided:** Features 1 & 2 + their evaluator are explicitly the **last** features to be implemented in this entire plan — everything in §5 (model bank/taxonomy) and §6 (translation pipeline, Connector Intelligence) gets built first.

**What's still genuinely open, not silently assumed (Rule 15):** the owner's instruction fixes Features 1/2's position relative to the *rest* of the plan, but doesn't say which of Feature 1 or Feature 2 comes first *relative to each other* — that sub-question stays open in §7. The build order *among* §5/§6's own items (which comes first: the model bank, the translation pipeline, or the Connector Intelligence routing mechanism) is also still undecided.

**PLAN.md updated:** §4 restructured — evaluator design moved in from §7 with a ✅ DECIDED sequencing note at the top; §7 trimmed to remove the now-duplicated evaluator entry and reworded the implementation-order open question to reflect what's actually still unresolved; Implementation Progress Tracker section given its first real constraint to track against once building starts.

---

## BLOCK 24 — Resolving the build order for §5 and §6 (2026-06-20)

Owner: "let's resolve the build order for §5 and §6 next." Per Rule 21, re-read `PLAN.md` in full first. Per Rule 9 (decompose) and Rule 18 (slow down, don't pattern-match), worked through actual dependencies between everything currently sitting in §5 (model bank/taxonomy) and §6 (translation pipeline, Connector Intelligence, routing candidates) rather than picking an order arbitrarily. One piece of real research grounding step 4 below: **Walking Skeleton** ([Cockburn, *Writing Effective Use Cases*, 2000](https://www.henricodolfing.com/2018/04/start-your-project-with-walking-skeleton.html)) and **Tracer Bullet** (Hunt & Thomas, *The Pragmatic Programmer*) — both real, established software-architecture practices: build a tiny end-to-end slice linking the main architectural components before building out breadth, then let architecture and functionality evolve in parallel.

**Resulting 7-step order, with reasoning:**
1. **Time/asynchrony handling first** — ahead of even the translation layer, because it governs whether anything built afterward (feature factory, regime labels, eventually the evaluator) can be trusted. Fixing data alignment after the fact would mean redoing everything downstream.
2. **Minimal translation layer, one data source only** (candles) — deliberately not the full universal/10,000-feature version yet; same "start minimal, generalize later" principle Block 3 already applied to the model count.
3. **Starter model bank** — reused Block 3's own already-proposed 8-model set (Wavelet, HDBSCAN, HMM, PatchTST, Kalman, PySR, PPO, GNN router) rather than inventing a new one. Noticed (Rule 10 — an actual finding, not just restating): this set happens to span 6 of Block 18's 8 functional layers already, missing only explicit Noise and Decision representatives — decent real coverage for a first test, not a coincidence worth ignoring.
4. **Connector Intelligence v0 — one hardcoded pathway, not learned routing.** Applying the Walking Skeleton/Tracer Bullet practice: wire one fixed path end-to-end through the Universal Belief Space format before adding any routing intelligence, to prove that part of the architecture works in isolation first.
5. **Prototype ONE routing-mechanism candidate — flagged 🟡, Claude's recommendation, not owner-confirmed:** LLM-tool-calling first, not MoE-gating or GNN. Reasoning: MoE and GNN both require *training* a learned gating function on logged routing-outcome data that doesn't exist until steps 1-4 have been running — a chicken-and-egg problem. LLM-tool-calling needs no training data to bootstrap and its decisions are immediately inspectable, making it the practical first step; MoE/GNN become viable once enough real routing outcomes have been logged to train them on.
6. **Interlingua versioning** — placed after the routing mechanism exists, since it only starts mattering once the model bank grows past the starter set and individual models start getting retrained/replaced.
7. **Expand the model bank** toward §5's full Model Genome Library, in the same phased hardware-scaling steps Block 3 already laid out.

Then Features 1 & 2 + evaluator, already fixed last per Block 23.

**Explicitly left out of this ordering, not silently folded in (Rule 15):** the still-unregistered third feature (graph/pathway Evolution Engine, Blocks 16-17) isn't placed in this sequence since it isn't registered as a feature yet — noted that if/when it is, it would logically come after step 4 (a graph has to exist before it can be evolved) and likely also needs the evaluator.

**PLAN.md updated:** the Implementation Progress Tracker section now holds the actual 7-step ordered list (each marked not-started), the reasoning behind each step, and an explicit note on what's deliberately excluded.

---

## BLOCK 25 — Owner registers Feature 3: the Graph/Pathway Evolution Engine (2026-06-20)

Owner: "now update the plan with all you mentioned along with graph/pathway Evolution Engine." Per Rule 21, re-read `PLAN.md` first. This resolves the long-open §7 item ("possible third feature, not yet registered") — owner is now registering it.

**Registered as Feature 3** in §4, consolidating everything found about it across Blocks 16-18 and 21-22: connection/pathway-reputation memory, the Path Discovery Engine, pathway genetics (crossover/mutation of whole pathways), and pathway death + curiosity-driven creation — the last two now formally understood as the evaluator's non-stationary-bandit layer (Block 22's finding), not separate hand-built rules.

**Decided:** Feature 3 shares the *same* 5-layer evaluator as Features 1 & 2, rather than needing its own — scoring a candidate pathway is the same statistical problem as scoring a candidate model or algorithm. Sequencing (Block 23) extended: Features 1, 2 & 3 are now *all three* the final implementable phase, together.

**Partially resolved as a side effect:** Block 17 Step 10's "function-level hierarchy" question — registering Feature 3 confirms the three-feature *structure* matches Step 10's framing (Feature 1 fills a slot with a model instance, Feature 2 fills it with a new algorithm, Feature 3 decides which slots connect). What's still open: whether those slots should be defined at the function level (Block 16 Layer-1 categories) rather than at the specific-algorithm level — not assumed resolved just because the structure lines up (Rule 15).

**Implementation tracker updated:** the build-order note from Block 24 now says Features 1, 2 & 3 + their shared evaluator come last together, with Feature 3 specifically also depending on step 4 (Connector Intelligence v0) since a graph must exist before it can be evolved. The remaining open item is ordering *among* the three features themselves.

**PLAN.md updated:** §4 (Feature 3 added with full spec, evaluator-sharing decision, sequencing line updated), §7 (third-feature item marked ✅ resolved/registered, Step-10 item reworded to reflect partial clarification), Implementation Progress Tracker (final-phase note updated to include all three features).

---

## BLOCK 26 — Clarifying question: what does "Models = Nodes" mean (2026-06-20)

Owner: "before implementing we need to finish discussing plans first now tell me what is Models = Nodes mean in the plan." A clarification request, not a new decision — answered by tracing the term back to its exact origin rather than inventing a fresh explanation (Rule 14: evidence before conclusion).

**Origin:** the literal first entry in Block 17's closing mapping table (line: "Models=Nodes, Connections=Weighted Synapses, Graph=Brain, GNN=Connector Intelligence, ..."), itself restating Block 16/17 Step 1's setup ("every model becomes a node... Node 1 = HMM, Node 2 = LSTM..."), and already written into `PLAN.md` §6 as "build a dynamic graph where nodes=models and a router decides which edges... exist."

**Meaning given to owner:** each individual model (HMM, LSTM, HDBSCAN, PySR, etc.) is represented as one node in a graph data structure, instead of being hardcoded as a fixed step in a pipeline (`lstm(hmm(data))`). Edges between nodes (which model feeds which) become data the system can add, remove, or rewire — which is exactly what makes the Connector Intelligence (§6) and Feature 3 (§4, Graph/Pathway Evolution Engine) possible at all. Connected to the transistor analogy (Block 16): one node alone does nothing special; intelligence emerges from how nodes are *wired together*, the same way a transistor alone does nothing but wired into gates, then circuits, then a CPU, becomes a computer. This is also why Block 17's central insight was to score the *edges* (Connections=Weighted Synapses), not just the nodes.

No `PLAN.md` change needed — the term is already correctly defined there (§6); this was purely an explanation pass.

---

## BLOCK 27 — Live research: physics & math methods specifically for trading prediction/pattern/noise (2026-06-20)

Owner: "search online for physics ml models and algorithms to add to the nodes along with the math ml models... y thing close to relating to trading prediction pattern extraction creation of pattern finding noise etc." Per Rule 9, decomposed into named sub-areas and searched each. All verified live, real peer-reviewed/well-cited sources — none recalled from training without checking.

1. **Physics-Informed Neural Networks (PINN) in finance** — embeds governing PDEs (Black-Scholes, Heston via the Feynman-Kac PDE) directly into the loss function. Applied mainly to **option pricing**, where the PDE is known. Honest finding from the literature itself: performance varies by market condition (stronger in high volatility), and general unstructured market *prediction* (vs. pricing a known-PDE instrument) remains a harder, less-proven application.
2. **Econophysics / Random Matrix Theory** — financial cross-correlation matrices, treated as random matrices, have only a small number of statistically meaningful large eigenvalues; the rest conform to the **Marchenko-Pastur distribution** and are noise. RMT-based filtering ("cleaning") separates the real correlation structure from noise — a direct, well-established answer to the noise-removal goal, distinct from autoencoders/Isolation Forest already in Block 2/18.
3. **Hawkes Processes** — self-exciting point processes: an order/event raises the short-term probability of more of the same (self-excitation) or the opposite side (cross-excitation), with decay over time. Used for order-flow-imbalance forecasting and limit-order-book modeling; "Deep Hawkes" variants combine this with neural nets. Directly relevant to closing the §6 time/asynchrony gap (7f) — this is a real, established way to model bursty, irregularly-timed event streams rather than pretending they're regularly sampled.
4. **Topological Data Analysis (persistent homology)** — builds a time-dependent "shape" from a sliding window of market data; the p-norm of the resulting persistence landscape grows abnormally before a crash. Real empirical hits: detected signals before the dotcom crash (March 2000) and the Lehman bankruptcy (September 2008), with fewer false alarms than volatility-based baselines in the cited study.
5. **Rough volatility / fractional Brownian motion** — real volatility paths are empirically "rougher" (more erratic at short time scales) than classical (Markovian) stochastic-volatility models assume; rough Bergomi-style models use a fractional Brownian motion (parameterized by the Hurst index) to capture this. Active, mainstream area in quant finance (a dedicated SIAM volume published 2024).
6. **Hurst exponent / Multifractal Detrended Fluctuation Analysis (MFDFA)** — measures long-range correlation (H>0.5), anti-correlation (H<0.5), or randomness (H=0.5) in a series, and how that scaling behaves across multiple timescales (multifractality). Honest caveat found directly in the literature: this is **contested** — some studies find real predictive value, others find no significant improvement in prediction accuracy. Not a slam dunk; flagged as such.
7. **Quantum annealing / QAOA / VQE for portfolio optimization** — real-world deployment found: Raiffeisen Bank International + Reply built a QUBO formulation solved on D-Wave hybrid quantum-classical solvers. This is optimization (the RL/Optimization Units category), not pattern-finding per se, but a genuine industry use case, not speculative.
8. **Chaos theory / nonlinear dynamics** — Lyapunov exponents (a positive exponent is necessary, not sufficient, for chaos) and Recurrence Quantification Analysis test whether price dynamics are deterministically chaotic rather than purely stochastic; used as crisis/crash indicators in the literature.
9. **Reservoir Computing / Echo State Networks** — only the readout layer is trained; the recurrent "reservoir" stays fixed after random initialization, making it far cheaper than training a full RNN/LSTM. Applied to intraday stock-return forecasting at multiple horizons in recent work.
10. **Tsallis entropy / non-extensive statistical mechanics** — generalizes Shannon entropy for systems with long-range interactions and memory; the resulting Tsallis distribution naturally has the peaked-center, fat-tail shape real financial returns show, which Gaussian models miss entirely.
11. **Self-Organized Criticality (Bak-Tang-Wiesenfeld sandpile model)** — markets modeled with two agent classes (fundamentalists/chartists) can spontaneously self-organize into a critical state that reproduces real "stylized facts" (fat tails, volatility clustering) without needing any external trigger or fine-tuning.
12. **Fractional differentiation** (López de Prado, *Advances in Financial Machine Learning*) — not a model but a feature-engineering technique: differentiate by a real number between 0 and 1 (not just integer 0=price or 1=returns) to achieve stationarity while preserving far more of the original series' memory than plain returns do. Routed to `PLAN.md` §6 (Feature Factory) rather than §5, since it's preprocessing, not a predictive model.

**PLAN.md updated:** §5 gained categories 21 (Physics-Informed), 24 (Econophysics/RMT), 25 (Point Processes/Hawkes), 26 (TDA), 27 (Rough Volatility), 28 (Fractal/MFDFA), 29 (Chaos Theory), 30 (Reservoir Computing), 31 (Quantum Optimization); §6 gained the fractional-differentiation cross-reference in the Feature Factory bullet.

---

## BLOCK 28 — Live research: "famous and reliable" physics-ML architectures, broader than just trading (2026-06-20)

Owner's follow-up, while Block 27 was being written: "i need you to find all or many famous and reliable physics ml models and algorithms." Broader ask than Block 27 — not trading-specific, the general physics-ML canon. Five more searches, all verified:

1. **Physics-Informed Neural Networks (PINN)** — already in Block 27 #1; the foundational, most general member of this family.
2. **Hamiltonian Neural Networks (HNN)** — parametrize a system's Hamiltonian with a neural net; conserve energy *by construction*, not by penalty term, giving accurate long-term predictions other architectures drift away from.
3. **Lagrangian Neural Networks (LNN)** — parametrize the Lagrangian instead; unlike HNNs, don't require canonical coordinates/momenta, so they work when those are unknown or hard to compute. **Hamilton-Dirac Neural Networks** extend this to *constrained* systems (e.g. a double pendulum), staying on the constraint manifold rather than drifting off it.
4. **Neural ODEs** — model a system's dynamics as a continuous-time differential equation parameterized by a neural net, rather than discrete layers.
5. **DeepONet** (Lu et al., 2019) — learns operators *between function spaces* (not just function-to-value mappings) via a dual branch-net/trunk-net structure; a physics-informed variant adds PDE-consistency regularization.
6. **Fourier Neural Operator (FNO)** (Li et al., 2020) — parametrizes the integral kernel directly in frequency space (Fourier transform → linear transform on low-frequency modes → inverse transform), giving resolution-invariant solution operators — efficient global dependency modeling, relevant to the "different sampling rates" problem in §6's time/asynchrony gap.
7. **Equivariant Neural Networks** (Cohen & Welling, 2016, group-equivariant CNNs, and successors) — bake known symmetries (rotation, translation, particle exchange) directly into the architecture via group representation theory, rather than hoping the network learns them from data; real benefits found: better generalization, fewer parameters needed, more interpretable (mappable to physical observables).
8. **Hopfield Networks** — associative memory, explicitly built on the **Ising model of magnetism** / spin-glass physics; stored patterns are energy minima the network settles into.
9. **Boltzmann Machines / Restricted Boltzmann Machines** (Ackley, Hinton & Sejnowski, 1985) — stochastic generalization of Hopfield networks; energy function is literally the Hopfield energy function, with configuration probability given by the **Boltzmann distribution** from statistical mechanics. As temperature→0, the Boltzmann machine's update rule becomes the Hopfield network's update rule — a precise, verified mathematical connection, not just an analogy.
10. **Diffusion Models** (Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, ICML 2015, *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"*) — directly built on **nonequilibrium thermodynamics**: a forward process gradually diffuses structured data into noise (like ink dispersing in water), and the network learns to run that process in reverse. This is the literal physics origin of the diffusion models behind modern image/video generation (and increasingly, time-series generation) — not a loose metaphor.

**Cross-reference (Rule 10 — noting the real lineage rather than treating these as a disconnected list):** items 8-10 form one continuous intellectual lineage — Ising model → Hopfield Networks → Boltzmann Machines → modern Energy-Based Models and Diffusion Models — all literally statistical mechanics, not just "physics-flavored" branding. Items 2-3 (Hamiltonian/Lagrangian NNs) and 5-6 (DeepONet/FNO) form a second lineage: classical mechanics and PDE/operator theory respectively, both aimed at making a network respect a *known* physical structure rather than learning it from scratch.

**PLAN.md updated:** §5 category 21 expanded to explicitly name HNN, LNN, Hamilton-Dirac NN, Neural ODE, DeepONet, FNO; category 22 expanded to explicitly name Hopfield Networks, Boltzmann Machines/RBMs, Diffusion Models with their real origins; category 23 (Equivariant Networks) added as new.

---

## BLOCK 29 — Vector databases: live research + Claude's own synthesis on where they fit in the plan (2026-06-20)

Owner: "update the plan with new finding and do a full research online and think yourself on vector db how it can help in data processing across the ml models and between them or from online to them or any other place in the plan." Resolves §5's bare "Memory (Vector Database, Episodic Memory, Knowledge Graph)" placeholder from Block 16, which had never been worked out. Per Rule 9, decomposed into the roles the owner explicitly named (across models, between models, from online sources) plus an open-ended "anywhere else" pass.

**Real implementations found** (none chosen yet): Pinecone (managed leader, easiest to operate, handles billions of vectors), Qdrant (Rust-based, fastest open-source — ~12ms p99 at 10M vectors vs. Weaviate's ~16ms/Milvus's ~18ms, best free tier), Weaviate (hybrid vector+keyword search, GraphQL, wants to be the whole AI-pipeline database), Milvus (large-scale), Chroma (best developer experience), pgvector (Postgres-integrated — the default if Postgres is already the data platform; scales to tens of millions before needing real expertise).

**Three roles directly from research:**
1. **RAG (Retrieval-Augmented Generation)** — the standard pattern: embed documents into vectors, store in a vector DB, retrieve the top-K semantically relevant chunks for a query at runtime instead of keyword matching. Production guidance found: needs a consistent embedding pipeline, a real vector-DB engine, sensible document-chunking (typically 512-1024 tokens), and metadata tagging. A real, growing market specifically in finance (vector-DB financial search projected to $6.11B by 2030) for semantic search over research reports, news, and structured portfolio data together.
2. **Tiered agent memory — Letta/MemGPT's real architecture.** Letta's OS-inspired memory hierarchy: **Core Memory** (always in context — the agent's immediate working state), **Archival Memory** (a vector database for long-term storage, semantically searchable), **Recall Memory** (full history in a regular database). Files get auto-parsed and embedded for semantic search over their contents. This is a battle-tested, citable architecture to adopt rather than reinvent.
3. **MoE + embeddings.** Real research (ROUTERRETRIEVER) routes a query to the best of several domain-specific "expert" embedding models, because general-purpose embedding models underperform domain-specific ones on their own domain — structurally the same problem as routing a query to the best of several model-bank nodes.

**Four roles from Claude's own thinking (Rule 18 — slowed down, didn't just stop at the research), each tied to a specific existing plan item rather than asserted in the abstract:**
1. **Model Genome Database backend** (§5 Block 18's schema) — embedding each node's metadata profile and doing similarity search gives a concrete way to pre-filter the 300-1000+ model bank down to a handful of relevant candidates for a given slot, before the (more expensive) routing mechanism chooses among just those — turns Block 18's stated rule ("don't fully-connect everything") into an actual mechanism.
2. **Pathway-similarity search for Feature 3's Path Discovery Engine.** This is the one I'd flag as most valuable: Block 17 Step 7 ("curiosity-driven mutation") described, in prose, exactly the behavior nearest-neighbor vector search gives for free — find pathways that share structure, notice the differing slot, try new fillers there. Embedding pathways and querying by similarity operationalizes that step instead of leaving it a hand-wavy heuristic.
3. **Adopting the Letta/MemGPT 3-tier split onto this project's own World Model** — Core Memory maps directly onto Block 17 Step 8's World Model (current regime/context); Archival Memory *is* the vector DB resolving §5's Memory placeholder; Recall Memory is the full pathway/connection history already implied by Block 17's reputation-memory design.
4. **Regime-similarity via the evaluator's anchor set.** Embed the current market window and the evaluator's Layer-4 anchor set of known historical regime episodes (Blocks 21-22) into the same space, and use nearest-neighbor distance as a continuous regime-similarity score instead of a brittle hand-labeled category — connects two previously-separate plan items (the World Model and the evaluator's anti-gaming anchor) through one mechanism, not asserted by either pasted source.

**Explicit scope boundary drawn (Rule 12 — precise about where this does and doesn't apply):** vector DBs are for *unstructured* or *embeddable* data and similarity-style lookups — news, social commentary, model/pathway profiles, regime windows. They are **not** a substitute for the Feature Factory's job on structured numeric sources (candles, order book) — flagging this so the architecture doesn't drift into using a vector DB as a universal hammer.

**PLAN.md updated:** §5's Memory placeholder now points to §6; §6 gained the full 5-role Vector Database entry plus the MoE-embeddings cross-reference, with implementation options listed but none chosen, consistent with "start small" (Blocks 3/24).

---

## BLOCK 30 — Owner confirms a foundational principle: build domain-independent, adapt data to it second (2026-06-20)

Owner: "i need to confirm on[e] thing we are not building this project focused on data like candles or order book but build them independent o[f] that and after implementing them we convert the data related [to] stocks into the inputs they need not the other way around now do a deep thinking and understand what i meant by this." Per Rule 18, slowed down deliberately on this one rather than treating it as a small detail — it governs the whole project's architecture, not a single component.

**Understanding demonstrated back to owner:** the model bank, Connector Intelligence, graph, and evolution/mutation engines get defined first by each component's own native, generic interface (e.g. "an HMM consumes a sequence of D-dimensional vectors") — with zero reference to candles, order books, or any stock-specific shape. Only afterward does a separate, swappable adapter convert real stock data into whatever input shape those already-built components require. Data adapts to the system; the system is never reshaped around what one data source happens to offer.

**What this confirms/connects (Rule 10 — an actual synthesis, not just restating):** this is the *exact same* "protocol first, devices adapt to it" principle already running through Blocks 4-5's internet/compiler analogies and Block 16/18's "electronic components" analogy — the owner is now stating it as a governing rule for the whole project, not just the belief-format layer. It also gives a sharper answer to "why is this project kept separate from the trading bot" (Rule 1): Pattern Brain's identity is a general pattern-finding/sequence-prediction architecture; stocks are the *first domain plugged in via an adapter*, not what the system is built around. The real test of whether this was followed correctly: a second, completely different data domain should be pluggable later via a new adapter alone, without touching the model bank or Connector Intelligence.

**Concrete consequence found and corrected:** Block 24's build order had this backwards — it put stock-data-specific work (time/asynchrony handling, a candle translation layer) in steps 1-2, *before* the model bank in step 3. Reordered: model bank → Connector Intelligence v0 (both data-agnostic, testable with synthetic data if needed) → a single merged "stock-data adapter" step (time/asynchrony handling + candle translation together) built specifically to feed the already-existing model bank and connector. Everything after that (routing-mechanism prototype, interlingua versioning, expanding the model bank) is unchanged in substance, just renumbered.

**Inconsistency flagged rather than silently fixed (Rule 15):** the evaluator (§4) scores candidates using Sharpe Ratio, Drawdown, and similar — inherently financial concepts, not domain-agnostic ones. Strict application of this principle would mean the evaluator's objective function should itself be a pluggable, domain-specific adapter rather than hardcoded. Not resolved; recorded as an open tension in §0/§7, not quietly patched over.

**PLAN.md updated:** new §0 added (placed before §1, since it governs §4/§5/§6 and the build order) stating the principle as ✅ DECIDED with its reasoning and consequences; Implementation Progress Tracker reordered with an explicit note that it supersedes Block 24's order and why.

---

## BLOCK 31 — Owner formalizes §0's principle as a standing rule (2026-06-20)

Owner: "now fix this as a rule so we silently won't deviate from our concept: 'The data bends to fit the system — the system never gets bent to fit the data.'" §0 in `PLAN.md` already stated the principle as content (✅ DECIDED), but content alone doesn't get actively re-checked the way `RULES.md`'s rules do (Rule 20 mandates scanning/disclosing rules every response; nothing previously forced re-checking `PLAN.md` §0 specifically before each new architecture decision).

**Added as Rule 23** in `RULES.md`: before proposing/designing/building anything in this project, check it against §0's principle. Concrete test given: would the proposal still make sense if the data domain were swapped for something unrelated to stocks? If answering requires touching the model bank/Connector Intelligence/evolution engines themselves (not just an adapter), it violates the principle and needs reworking before it goes into the plan. Explicitly framed as closing the exact gap that let Block 24's build order violate this without anyone catching it until Block 30.

**Kept Rule 3 intact (rules separate from content):** the actual architectural principle stays as content in `PLAN.md` §0; Rule 23 in `RULES.md` is the *process* rule that mandates checking against it, mirroring how Rule 20 mandates checking against all the *other* rules. `PLAN.md` §0 updated with a one-line pointer back to Rule 23.

---

## BLOCK 32 — Researching whether Claude can take over the owner's own role in this loop (idea → research → check → decide) (2026-06-20)

Owner: "search online and... find any rule that can take my place of how i just thought ideas and concepts... asked you to search and then i read your reply and decided what to do next — so can you make claude do it [to] think of new better ideas and ask itself and check its findings... do a deep search and think hard." Real, verified findings, ranked by relevance:

1. **BabyAGI** (Yohei Nakajima, 2023) — the most literal precedent: a loop of task creation → execution → prioritization, using completed-task outcomes to inform the *next* task and reprioritize the list, backed by a vector memory store (ties directly to Block 29's vector-DB work).
2. **AutoGPT** — chains LLM "thoughts" in an open-ended loop of reasoning → planning → acting, breaking one human-given goal into self-generated sub-tasks.
3. **Google DeepMind's "AI co-scientist"** (built on Gemini 2.0, [arXiv 2502.18864](https://arxiv.org/abs/2502.18864), graduated to a Nature paper by 2026) — the strongest, most recent precedent. A multi-agent system that runs **self-play-based scientific debate** to generate hypotheses, a **ranking tournament** to compare them against each other, and an **evolution process** to improve quality — explicitly automating the scientific method's idea→critique→refine loop. Validated with real institutions (Stanford, Imperial College London) and companies (Daiichi Sankyo, Bayer Crop Science, US DOE's Genesis Mission) on genuinely novel findings.
4. **Sakana AI's "AI Scientist" v1/v2** — fully automates idea generation → experiment → analysis → manuscript, v2 using "agentic tree search" with no human-authored templates. **Honest, important caveat from an independent evaluation** ([arXiv 2502.14297](https://arxiv.org/pdf/2502.14297)): its literature review is "inadequate, relying on simplistic keyword searches rather than profound synthesis," leading to **poor novelty assessment** — it has incorrectly classified well-established ideas (e.g. micro-batching for SGD) as novel discoveries. Directly relevant: this is exactly the failure mode this project has been guarding against since the fabricated "Research Cluster" doc caught in [[project_developmental_ai_curriculum]] and the recurring "Caveat: not yet verified" discipline throughout this whole file.
5. **Self-Instruct** — from only 175 seed tasks, generated 52K new instructions + 82K instances entirely from the model itself — a real precedent for "generate your own next research questions from a small seed," not just answer given ones.
6. **Intrinsic Curiosity Module (ICM, Pathak et al.)** — formal mechanism for *what to investigate next*: reward is based on prediction error, i.e. the agent is "curious" about whatever it currently predicts worst. Same underlying principle as the UCB exploration bonus already adopted in the evaluator design (Block 22) — optimism-under-uncertainty, just applied to "what to research" instead of "which pathway to try."

**Synthesis (Rule 10):** combining these, a real "Rule 24" would have Claude (a) generate candidate next research directions itself (Self-Instruct-style, from the current state of `PLAN.md`'s open questions as the seed), (b) prioritize them by where the plan's current understanding is weakest/most uncertain (ICM-style), (c) research and self-critique multiple candidates against each other before committing to one (AI-co-scientist tournament/debate-style, which is really just Rule 16's steelman applied to *which idea to pursue* rather than *which design to pick*), and (d) hold itself to the same verification discipline that's been running through every block in this file, specifically because Sakana's own evaluation shows shallow-verification autonomous research produces false novelty claims.

**Not yet decided — a real fork, not guessed at (Rule 13/15):** this rule could mean two materially different things: (A) Claude becomes more proactive *within active conversation* — surfacing self-generated research directions unprompted while still talking with the owner, no unattended operation; or (B) an actual autonomous background process (via the `loop` or `schedule` skill) that runs research cycles and edits `PLAN.md`/`DISCUSSION_NOTES.md` *without the owner watching each step*. (B) is a meaningfully bigger commitment — unattended edits to the plan — and wasn't assumed; asked the owner to choose rather than picking unilaterally.

**Owner follow-up: "even if it is not a rule, any other different way to get this function/behavior."** Broadened beyond research papers to actual Claude Code mechanisms available right now — not abstract frameworks, real tools:
- **`/loop` skill** — runs a prompt on a recurring interval, or "self-paces" (the model decides when to re-invoke itself) with no fixed interval. This is the literal mechanism for "Claude keeps going without being re-prompted each time."
- **`/schedule` skill** — creates a cloud-based recurring agent on a cron schedule, running independent of an active chat session at all.
- **A background `Agent` run** — spawn a long-running research agent now that keeps working while the owner does something else, reporting back when done, rather than a permanent recurring loop.

**How these map onto the research synthesis above:** the `loop`/`schedule` mechanism would be the *execution engine* that keeps the cycle running; the content of each cycle would be exactly Block 32's synthesis (BabyAGI-style task generation from `PLAN.md`'s open questions as seed → ICM-style prioritization toward the weakest-understood areas → AI-co-scientist-style self-debate/tournament between candidate directions before committing → the same verification discipline already running through this file, specifically because Sakana's evaluation shows skipping that step produces false novelty claims).

**Still the same real fork, now more concrete:** a recurring loop/schedule that *proposes* candidate directions into `PLAN.md` tagged 🟡 for the owner to review still respects Rule 13 (pause before hard-to-reverse decisions); one that *commits* them as ✅ decided unattended would not. Which of these (or whether to do this at all) is the owner's call, not assumed.

**Owner chose:** "Recurring loop that proposes, you approve" — a `/schedule` cloud routine that researches and proposes, never auto-decides. Discovered while setting it up: cloud routines run in Anthropic's cloud and cannot touch local files at all — they need a git repo to clone. Pattern Brain had neither git nor a remote. Owner chose to git-init + push to GitHub so the routine can clone it and open PRs (rather than skip the scheduler or fall back to in-chat-only proactivity).

**Repo created:** `https://github.com/ajith4134/pattern-brain`, pushed from local — owner provided the repo URL and a personal access token directly in chat. Token was used once, transiently (injected via an HTTP header for a single push command, never written to `.git/config`, never echoed in any output) and confirmed not persisted anywhere locally afterward. **Flagged to owner: rotate/revoke that token**, since sharing it in plaintext chat means it's in the conversation transcript regardless of local handling.

---

## BLOCK 33 — New rule: keep git up to date with every change (2026-06-20)

Owner: "also new rule to push all the changes and update all the files with the git so all the changes with the project is up to date with git." Added as **Rule 24** in `RULES.md`: after any change to project files, commit + push to `main` on the GitHub remote before finishing that response — one commit per response's worth of changes, not fragmented per edit, not deferred across responses. Applies starting with this very response (this block + the RULES.md edit itself get committed together below).

**Credential note:** the push needed re-authenticating (the bootstrap push's token wasn't persisted, by design). Set up git's `store` credential helper (writes to `~/.git-credentials`, mode 600, outside the repo so it's never pushed/exposed via the repo itself) using the same token, so Rule 24's recurring pushes don't require re-asking the owner each time. Owner was already asked to rotate/revoke that token (Block 32); if/when they do, this credential helper will need a fresh one.

---

## BLOCK 34 — Scheduled research routine created (2026-06-20)

Following Block 32's owner choice ("recurring loop that proposes, you approve"), created the actual `/schedule` cloud routine via `RemoteTrigger`:

- **ID:** `trig_01URCffkt4EwJ7yvahLjj1us`, name `pattern-brain-research-cycle`.
- **Cadence:** daily, `0 6 * * *` (6am UTC) — a default chosen for a research/ideation cycle, not urgency-driven; first run 2026-06-21T06:00 UTC. Owner can adjust via the routines dashboard or by asking Claude to update it.
- **Repo:** clones `https://github.com/ajith4134/pattern-brain`, model `claude-sonnet-4-6`, tools `Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch`.
- **Prompt (self-contained, since the cloud session starts with zero context of this conversation):** read all 4 project files; follow RULES.md, with explicit call-outs to Rules 1/2/4/9-11/16/23; explicit warning against Sakana AI Scientist's documented failure mode (shallow keyword-search literature review → false novelty claims, Block 32) as a concrete "don't repeat this mistake" anchor; hard prohibitions (never mark anything DECIDED, never write implementation code, never edit RULES.md, never push to main directly); the actual cycle: read PLAN.md's open items → pick the weakest-understood ones → generate 2-4 candidate directions → research + verify each → steelman/compare them against each other → write only the best-grounded finding into PLAN.md tagged PROPOSED + a new DISCUSSION_NOTES.md Block → commit on a `research/cycle-<date>` branch → open a PR for owner review (or report findings without a PR if it can't authenticate to push).
- **Explicitly permitted to conclude "nothing this cycle" rather than force a low-value addition** — directly guards against manufacturing busywork just to produce output every run.

**Open uncertainty, not yet observed (Rule 15 — stating what's unverified rather than assuming success):** whether the cloud environment can actually authenticate to push a branch/open a PR back to GitHub wasn't confirmable from the `/schedule` skill's documentation alone — the prompt instructs the routine to report findings without a PR if it can't. First real run (2026-06-21 06:00 UTC) will be the actual test; worth checking the [routine's page](https://claude.ai/code/routines/trig_01URCffkt4EwJ7yvahLjj1us) after that to confirm it worked as intended.

---

## BLOCK 35 — Deep research: how to ADD, USE, and CONNECT model nodes, concretely (2026-06-20)

Owner: "do deep research on all ml algorithms relate or useful or to the ml model nodes and how to add them and used and connect them after reading the plan." Per Rule 21, re-read `PLAN.md` in full first. Reframed scope (Rule 18 — slowed down to interpret correctly): Blocks 2/16/18/27/28 already exhaustively covered *what* algorithms belong in the model bank; what's never been researched is the *mechanical how* — the actual integration pattern for plugging a heterogeneous model into the architecture. Six real, production-grade precedents found, all checked against Rule 23 first (all operate on a generic interface, none know anything about stocks/candles):

1. **Scikit-learn's Estimator API** ([arXiv 1309.0238](https://arxiv.org/pdf/1309.0238), the original "API design for ML software" paper) — every estimator, regardless of internals, exposes the same `fit`/`predict`/`transform` methods. The most proven real precedent for "every node speaks the same small interface" — directly realizes what §0 asked for in the abstract.
2. **MLflow `pyfunc`** — a more general fallback for node types that don't fit sklearn's shape (HMMs, Kalman filters, a discovered symbolic equation): wrap anything behind one `predict()` method, framework-agnostic and self-contained with its own dependencies, deployable to any of MLflow's supported environments.
3. **ONNX** — solves cross-framework/cross-hardware portability (train in PyTorch, run via ONNX Runtime on different hardware) but is primarily neural-network-shaped; classical/non-neural models have partial, awkward support. Scoped as a later optimization for the neural subset of the bank, not a day-one requirement — an honest limit, not oversold.
4. **Ray Serve deployment graphs** ([Anyscale write-up](https://www.anyscale.com/blog/multi-model-composition-with-ray-serve-deployment-graphs)) — the strongest real match to "Models = Nodes, Graph = Brain" (Block 18): bind models together by passing them into each other's constructors; at runtime, calls between completely different models (different frameworks, different resource needs — GPU vs CPU) "look just like function calls" via a `DeploymentHandle`. This is the literal mechanical version of the "transistor" analogy (Block 16) — components with totally different internals, callable identically.
5. **LangChain's LCEL `Runnable` interface** — any component implementing `.invoke()` composes with any other via the `|` pipe operator; `RunnableSequence` for chains, `RunnableParallel` for fan-out (feed one input into several pattern-discovery nodes simultaneously, merge after) — a clean, pythonic candidate for how Connector Intelligence v0 (build-order step 2) could actually execute a wired pathway.
6. **Kedro's node + Data Catalog + Pipeline** — the closest real match to the still-conceptual Model Genome Database (§5/§6, Block 18): a catalog of named, typed artifacts decoupled from pipeline logic, with free DAG visualization via Kedro-Viz as a side benefit.

**Synthesis (Rule 10 — an actual recommendation, not a list):** adopt scikit-learn's interface convention as the default "how to add a node" answer, MLflow `pyfunc` as the fallback for anything that doesn't fit it, Ray Serve's deployment-graph binding or LangChain's Runnable/pipe pattern as the candidate mechanism for "how Connector Intelligence v0 actually wires and executes a pathway" (build-order step 2), and Kedro's catalog pattern as the closest real precedent for the Model Genome Database once it needs to track the whole bank, not just one path. None of this requires inventing new infrastructure — it's assembling existing, battle-tested pieces into the shape this project already designed conceptually in Blocks 16-18.

**PLAN.md updated:** §6 gained a new entry consolidating all of the above, explicitly Rule-23-checked.

---

## BLOCK 36 — Researching rules for turning the plan into code, with self-verification (2026-06-20)

Owner: "do a deep research online and find rules which are best when followed will make claude... turn the plan created into code and while implementing it should think for itself if the concept is applied." Six real, authoritative sources found, ranked by relevance:

1. **Anthropic's own published Claude Code guidance** ([anthropic.com/engineering](https://www.anthropic.com/engineering/claude-code-best-practices), ["Effective harnesses for long-running agents"](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)) — the most authoritative source since it's literally about this tool. Key findings: the converging 2026 pattern is "research → plan → execute → review → ship, with the human as oversight at each gate" (exactly the phase this conversation has been in); "explicitly asking Claude not to code initially prevents premature solutions and improves architectural quality" (validates the entire discussion-before-implementation discipline this project has followed); specific instructions outperform vague ones; treat Claude "like a junior engineer with tools, memory, and iteration — not a magic code generator."
2. **Architecture Decision Records (ADR)** ([adr.github.io](https://adr.github.io/)) — lightweight documents capturing context, decision, and consequences, specifically to "show your working out" and prevent re-litigating settled decisions. `PLAN.md`'s "Why" sections are already, unknowingly, ADR-shaped — this validates rather than changes existing practice.
3. **Design by Contract** (Bertrand Meyer, Eiffel) — preconditions (caller's obligations), postconditions (callee's guarantees), invariants (always-true conditions). Rule 23's core principle is, in this formal language, exactly a class invariant — every model bank/connector/evolution-engine component must maintain domain-independence at all times.
4. **Test-Driven / Verification-Driven Development for AI coding agents** — real, recent research (TDAD, [arXiv 2603.17973](https://arxiv.org/html/2603.17973v1)) found vanilla coding agents average **6.5 broken tests per generated patch** without this discipline — the finding's own framing: "generation speed is no longer the bottleneck; verification discipline is." The VSDD pipeline chains three gates: spec-first (contract before code), TDD (red→green→refactor enforced per step), and adversarial verification by a different model pass.
5. **Plan-and-Execute (P-t-E) agent architecture** — separates Planner from Executor, with a dedicated verification step that judges postconditions and can trigger local or global re-planning (ADaPT) when execution diverges from the plan, rather than blindly trusting the planner's output.
6. **Self-Debugging / Reflexion for code** — generate code + self-generated tests, execute, feed failures back as revision signal — "rubber duck debugging" formalized into a loop.

**Synthesis, added as Rule 25 in `RULES.md`:** for each implementation step, treat the `PLAN.md` entry as the spec → write checkable conditions (ideally failing tests) for "correctly implemented" before writing code → implement the minimum to satisfy them → explicitly self-check the result against both the specific plan item AND Rule 23's domain-independence invariant → if implementation reveals the plan itself was wrong, fix `PLAN.md` (Rule 21) rather than letting code and plan silently drift apart. This is the concrete mechanism behind "think for itself if the concept is applied" — not a vague aspiration, an actual checklist sourced from real practice.

**Not yet relevant (Rule 6 — noted, not acted on):** this is a process rule for the implementation phase, which hasn't started — README.md still correctly says "discussion phase," and nothing here changes that.

---

## BLOCK 37 — Fresh session: owner asked Claude to re-read the whole project folder (2026-06-20)

Owner opened a new session (model set to Opus 4.8) and asked: "Pattern Brain folder and read all files in it carefully." A context-loading/comprehension request, not a new decision or new content — logged per Rule 2/4 (every message gets recorded, not just the substantive ones), kept deliberately short so it doesn't bloat the log with a recurring session-start event.

**Done:** read all four content files end-to-end — `RULES.md` (all 25 rules), `README.md`, `PLAN.md` (§0–§7 + Implementation Progress Tracker), and `DISCUSSION_NOTES.md` (Blocks 1–36). Confirmed comprehension back to the owner with a synthesis covering: the domain-independence principle (§0/Rule 23), the model bank → Connector Intelligence → 3-final-features+evaluator structure, the still-open §7 items (order among Features 1/2/3; function-level vs algorithm-level slots; the "cpu ml models" deliverable format; the evaluator-metric domain-independence tension), and the live git remote + daily cloud research routine.

**Re-flagged (carry-over, not new):** the GitHub token shared in plaintext chat (Blocks 32/33) and stored in `~/.git-credentials` — owner was asked to rotate/revoke it; flagged again for confirmation, since it persists in the conversation transcript regardless of local handling.

**No PLAN.md change** — nothing was decided or discovered this message; it was a read-only comprehension pass. Per Rule 6, no next topic suggested — left to the owner to choose what to discuss next.

---

## BLOCK 38 — Design + live research: a dedicated visualization dashboard for Pattern Brain (2026-06-20)

Owner: "think of a way to display this project in dashboard dedicated to only this project the test or any ml models its working in a visual way for me to see its working in details, search online how to make it and think hard." Per Rule 21, the current `PLAN.md` was already re-read this session. Per Rule 9, decomposed into five sub-questions before searching; all findings below verified via live search just now (Rule 11 — real, current sources, not recalled). Per Rule 18, slowed down to map each tool onto *this project's specific structure* rather than listing generic dashboards.

### What actually has to be shown (derived from the plan, not generic ML)
This project's identity (§0/§6) is **a graph of models where the unit of value is the connection/pathway, not the model** (Block 17). So a faithful dashboard's centerpiece can't be the usual "loss curve + accuracy" — it must be the **living graph itself**. Seven views, each tied to an existing plan element:
1. **The Living Graph (hero view)** — nodes = models (§5 bank), edges = connections, **edge colour/thickness = pathway-reputation score** (Block 17 Steps 2-3), animated belief-flow along active edges, node glow = currently firing. This *is* the project made visible.
2. **Node Inspector** — click a node → its Model-Genome-Database metadata (Block 18: inputs/outputs/latency/confidence/best-regime) + its live internal state (HMM state probs, PySR's current equation, a cluster assignment, etc.).
3. **Pathway Leaderboard** — `Path #42: Wavelet→HMM→PPO`, regime-split outcome stats + usage count + reputation (Block 17 Steps 4-6) — the pathway-reputation memory as a sortable table.
4. **Evaluator / Test panel** — the 5-layer evaluator (§4) made visible: walk-forward split timeline, DSR/PSR significance bars, non-stationary-bandit scores, the **NSGA-II Pareto front as a scatter plot** (multi-objective trade-off), and anchor-set sanity checks (Blocks 21-22). This is the "tests working in detail" the owner asked for.
5. **World Model ribbon** — the current detected regime (Bull/Bear/Range/High-Vol/…) as a timeline strip (Block 17 Step 8), since the same pathway isn't best across regimes.
6. **Evolution feed** — Features 1/2/3 (§4) live: candidates born → tested → promoted/killed, streaming, so mutation is watchable.
7. **Belief-space stream** — the auditable JSON beliefs flowing through the Universal Belief Space (Block 5), written by "The Synthesist" (Block 11's persona) with confidence levels first-class.

### The Rule 23 check (the most important part — and a real coherence catch)
A dashboard is exactly the kind of thing that silently re-introduces data-first design. Checked against §0: **the seven views above are all domain-agnostic** — graph, nodes, beliefs, pathway scores, evaluator layers, regime labels exist with zero reference to candles. The *only* stock-specific visuals (candlestick charts, order-book heatmaps) must live in a **separate "Adapter View" tab fed by the stock-data adapter (build step 3)** — never baked into the core dashboard. The real test (same as §0's portability test): a future second data domain should light up the *same* six core views via a new adapter, with only the Adapter-View tab changing. **Honest carry-over of §0's open tension:** view #4's Pareto axes (Sharpe/Drawdown) are financial — so those axis labels/metrics must be adapter-supplied/configurable, not hardcoded, to keep the dashboard as domain-independent as §0 demands. Flagging, not silently hardcoding.

### Live research — real tools found, mapped to the views (Rule 11)
- **Graph rendering (view 1):** **React Flow** is the recommended hero-view engine — the research's explicit guidance is "if building node-based UIs / workflows, start with React Flow," it has built-in **animated edges** and custom node components, and real-time updates are a solved pattern (WebSocket → live node/edge updates). **Steelman of the rejected options (Rule 16):** Cytoscape.js has the richest graph-*algorithm* library (centrality, shortest-path — useful for analysing the pathway graph) and **Sigma.js (WebGL) is the *only* one of the three that scales past ~50k nodes**. Since §5's bank grows toward 300-1000+ nodes, the honest answer is two-tier: **React Flow for the curated active-pathway view (tens of nodes, beautiful animated belief-flow), and Sigma.js/Cytoscape.js as the planned upgrade for the full "genome map" once the bank scales** — matching Block 3/24's "start small, don't begin with hundreds" discipline rather than reaching for the heavy WebGL renderer on day one.
- **Live updates:** **WebSockets** (bi-directional, event-driven) push belief-flow/score changes to the graph so it animates as the system runs — the standard real-time-dashboard pattern; **FastAPI** WebSocket or Server-Sent Events on the backend.
- **Experiment/run tracking (view 4 history):** **Aim** — open-source, self-hosted, fast, handles 10,000s of runs, and can even sit *on top of* an MLflow backend. Preferred over hand-rolling run tracking. (MLflow as registry backend if a model registry is later needed.)
- **Model-eval / drift reports:** **Evidently** (open-source: 100+ metrics, Reports + Test Suites + Monitoring UI) for the *generic* data-quality/drift sub-panel — **but explicitly NOT** as the universal evaluator UI: Evidently is built around standard drift/quality metrics, whereas this project's 5-layer evaluator (DSR/PBO/Pareto/anchor) is custom, so view #4's core stays custom **Plotly**, with Evidently only for the generic drift slice.
- **Infra / "is it alive" metrics:** **Grafana + Prometheus** — the research's clear fit for production-grade time-series infra panels (CPU, latency, throughput, node liveness), distinct from the ML-specific views.
- **Coherence with an existing decision (Rule 10 — a real catch, not restating):** Block 35 already chose **Kedro's node+catalog+Kedro-Viz** as the closest match to the Model Genome Database, noting Kedro-Viz gives "free DAG visualization." That is the **static/dev-time** graph; this dashboard is the **live/runtime** graph — they complement rather than compete (Kedro-Viz = the wiring diagram; React-Flow hero view = the same graph *lit up and running*).

### Concrete recommendation (Rule 10 — an actual stack, not "it depends")
A **single dedicated React app** (one window, per the JARVIS-style "single live dashboard" instinct, but this project's own — no trading-bot coupling) + **FastAPI WebSocket** backend, with: **React Flow** (hero living-graph) + **Plotly** (charts incl. the Pareto scatter) + embedded/linked **Aim** (runs), **Evidently** (drift slice), **Grafana/Prometheus** (infra). **Upgrade path:** Sigma.js/Cytoscape.js for the full genome map once the bank passes a few hundred nodes. **Throwaway-v0 alternative for "see something now":** **Streamlit + a static Plotly network graph** (research: Streamlit is "best for custom ML/analytics dashboards / ML demos") — fast to stand up, but it can't host React Flow natively, so it's a stopgap, not the real thing.

**Steelman of *not building custom at all* (Rule 16):** Aim + MLflow + Kedro-Viz + Grafana + Evidently are battle-tested, free, zero-maintenance and already cover run-tracking, DAG view, infra and drift — building custom risks reinventing them (the exact Block 7e/35 caution). The honest line: **reuse off-the-shelf for every generic sub-panel; build custom ONLY the one thing none of them do — the live, animated, reputation-weighted pathway graph with belief-flow**, which is unique to this project's "score connections, not models" identity.

### Build-order placement (Rule 22)
The dashboard can't fully exist before the system does — but it shouldn't be one late deliverable either. Proposed as a **cross-cutting track that grows with each build step**: a walking-skeleton dashboard appears at **build step 2** (Connector Intelligence v0 — the moment the first hardcoded pathway runs, the owner should be able to *watch* it run), then each later step adds its corresponding view (step 3 → Adapter-View tab; the final Features-1/2/3 phase → the Evolution feed). This mirrors the project's own Walking Skeleton principle (Block 24) instead of bolting visualisation on at the end.

**Status:** 🟡 PROPOSED — not owner-ratified. The view list, the Rule-23 separation (core vs Adapter-View tab), and the reuse-vs-build line are the substance; the exact stack is a recommendation, with the final choice and "build a v0 now vs keep as planned feature" left to the owner (Rule 15 — not assumed).

**PLAN.md updated:** new §8 (Dedicated visualization dashboard) added with the seven views, the Rule-23 core-vs-adapter separation, the recommended stack + upgrade path, the reuse-vs-build line, and the cross-cutting build-order placement; Implementation Progress Tracker gains a note that the dashboard is a parallel track starting at step 2, not a final deliverable.

**Sources (live-verified):** [Cambridge Intelligence — React graph viz guide](https://cambridge-intelligence.com/blog/react-graph-visualization-library/); [PkgPulse — Cytoscape.js vs vis-network vs Sigma.js 2026](https://www.pkgpulse.com/guides/cytoscape-vs-vis-network-vs-sigma-graph-visualization-2026); [React Flow — animating edges](https://reactflow.dev/examples/edges/animating-edges); [SuperViz — real-time sync for React Flow](https://www.superviz.com/tutorials/add-real-time-sync-to-a-react-flow-application); [Aim (GitHub)](https://github.com/aimhubio/aim); [AimStack — Aim UI on MLflow](https://aimstack.io/blog/tutorials/exploring-mlflow-experiments-with-a-powerful-ui); [Evidently AI](https://www.evidentlyai.com/); [Evidently + Streamlit dashboard tutorial](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial); [Monitoring ML with Prometheus & Grafana](https://medium.com/@cartelgouabou/step-by-step-guide-monitoring-ml-model-performance-with-prometheus-grafana-88195f741365); [Kedro-Viz (GitHub)](https://github.com/kedro-org/kedro-viz).

**Per Rule 6 — no next topic suggested.** One same-topic next action offered to the owner: build a throwaway Streamlit+Plotly v0 now to *see something*, or keep the dashboard as a planned cross-cutting track in the build order.

---

## BLOCK 39 — Owner moves to implementation + mandates a polyglot (C/C++ as well as Python) language policy (2026-06-20)

Owner: "now start implementing the plan and i need more ml nodes many types" — then, interrupting the environment setup: "also in the plans mentioned using different programming language instead of using only python using c or c++ make sure you remember that." Two things this message does: (1) flips the project from discussion → implementation, starting at build-tracker **step 1 (starter model bank)** per Rule 22 (no skipping); (2) adds a standing language decision that the build is **not Python-only**.

**Recorded per Rule 1** — "make sure you remember" is honored by writing this into `PLAN.md` (the project's durable decision store, new §0b) and here, **not** into Claude's hidden cross-session memory (Rule 1 forbids project content there).

**Reasoning behind the §0b language policy (Rule 12 — concrete, not vague; Rule 10 — an actual division, not "it depends"):**
- **Python = conductor/prototype:** Connector Intelligence, registry, dashboard backend (§8), glue. Every tool already in the plan is Python (sklearn API, MLflow pyfunc, Ray Serve, LangChain, Kedro — Block 35).
- **C/C++ = performance hot paths:** the Evolution Engine testing thousands of pathways in parallel (Block 3 Case 3), hot node inner loops (HMM forward-backward, particle filters, Hawkes simulation), real-time belief-flow across a large graph, at-scale reputation updates.
- **Staging rule:** prototype in Python → profile → rewrite only proven hot paths in C/C++. Never C++-first before a measured bottleneck (consistent with Walking Skeleton/Block 24 + Rule 25's "minimum code, no gold-plating").
- **The Rule 23 consistency check (important):** going polyglot does **not** violate §0/Rule 23 — it *reinforces* it. Block 35 already established that every node hides behind a common interface (sklearn `fit/predict/transform` or MLflow `pyfunc.predict()`); that same contract lets a C++ node be called identically to a Python one via real bindings (**pybind11/Cython** for C++↔Python, **ctypes/cffi** for C↔Python, **ONNX** for the neural subset). So "the data bends to the system" extends to "the language bends to the interface" — implementation language is just another swappable internal behind the adapter, touching neither the Connector, the belief format, nor the graph.
- **Steelman of Python-only (Rule 16):** numpy/scipy/sklearn already call optimized C/Fortran (BLAS/LAPACK), so array math is C-speed without hand-written C; premature C++ adds toolchain complexity. Hence *profile-then-optimize*, not C++-first — Python-only is the right starting posture, polyglot the right scaling posture.

**Implementation blocker hit + Rule 15 (ask, don't guess):** this environment has **no numpy/scipy/scikit-learn and no pip/ensurepip** (`python3 -m pip` and `ensurepip` both absent; only `apt-get` present, candidate `python3-numpy` available). Step 1's model bank, as specced (sklearn-based nodes on D-dimensional vector sequences), needs these. An attempted dependency-install check was declined by the owner. Rather than guess another install path (Rule 15 — "don't be shy" about asking), surfacing the environment choice to the owner directly: how to provision the ML stack (apt system packages / pip-in-venv / a different approach), and whether — given the new C/C++ mandate — they want the step-1 starter bank's compute nodes prototyped in Python (recommended per §0b staging) or begun in C/C++ now. **Critically: this project must NOT borrow the trading bot's Python environment (Rule 1 separation) — any deps go into a Pattern-Brain-local setup.**

**PLAN.md updated:** new §0b (Language policy — polyglot by stage) added after §0, tagged ✅ DECIDED (owner-mandated), with the Python/C-C++ split, the staging rule, the pybind11/Cython/ctypes/cffi/ONNX bridge mechanisms, and the Rule-23 consistency argument; Implementation Progress Tracker preamble now states the language policy applies to every step. **No node code written yet** — paused on the environment decision per Rule 15 before starting step 1's implementation.

---

## BLOCK 40 — Build step 1 SHIPPED: the starter model bank, 30 node types (2026-06-20)

Owner chose "Python deps via apt" for the environment; authorized + ran `sudo apt-get install python3-numpy python3-scipy python3-sklearn` (pulled numpy 2.2.4 / scipy 1.15.3 / sklearn 1.4.2 / matplotlib, plus **g++/gcc 14.2** as deps — handy for the future C/C++ stages, §0b). Then implemented **build-tracker step 1** following Rule 25 (spec→test→code→self-verify) and Rule 23 (domain-agnostic).

**What was built — the `pattern_brain/` package:**
- `belief.py` — `Belief` (Universal Belief Space unit, Block 5): typed, confidence-first (clamped [0,1]), JSON-serializable, schema-versioned.
- `node.py` — `Node` common interface (Block 35): sklearn-style `fit`/`predict`/`transform` + a universal `process()`; `(T, D)` validation; metadata = the Model Genome record (Block 18). Explicitly documented as the contract a future **C/C++ node binds behind** (§0b).
- `registry.py` — the model-bank registry / Model Genome seed (Block 18 + Kedro catalog idea, Block 35): `register`/`create`/`by_layer`/`default_bank`.
- `nodes/` — **30 node types across all 8 Block-18 functional layers** (owner asked for "many types"; this is well beyond the original ~8):
  - **signal** (4): fft, moving_average, difference, hilbert_envelope
  - **noise** (4): savgol_denoise, pca_denoise, zscore_anomaly, isolation_forest
  - **pattern** (5): kmeans, dbscan, hdbscan, gmm, agglomerative
  - **sequence** (4): markov_chain, autoregressive, exp_smoothing, **gaussian_hmm** (real Baum-Welch EM)
  - **probability** (4): kalman_filter, gaussian_process, bayesian_ridge, particle_filter
  - **equation** (3): linear_regression, polynomial_regression, **symbolic_regression** (FFT-seeded library-search SR)
  - **decision** (3): threshold_policy, sign_vote, logistic_regression
  - **rl** (3): epsilon_greedy_bandit, ucb_bandit (the Block-22 evaluator's bandit family), q_learning
- `tests/test_bank.py` — contract + behavior suite (Rule 25's checkable conditions): every node returns a valid `Belief` (confidence ∈ [0,1], JSON-serializable, source set), accepts 1-D input, rejects 3-D/NaN; real behavioral asserts (KMeans finds 2 clusters on bimodal data, AR continues a ramp, symbolic_regression recovers a sinusoid, HMM finds 2 states); and a **Rule-23 executable check** (tokenize-based — no candle/ohlcv/orderbook *identifiers* in the core). **All green.**
- `demo.py` — runs the whole bank on a synthetic regime-switching sequence and prints each belief by layer (the textual precursor to §8's Belief-stream / Node-Inspector dashboard views).

**Two real bugs found + fixed during self-verify (Rule 25.4, red→green):** (1) the lightweight symbolic-regression's `sin` candidate was hardcoded to frequency 1.0 → couldn't match an arbitrary sine; fixed by estimating the dominant angular frequency via FFT first (now genuinely recovers sinusoids). (2) the first Rule-23 test naively grepped raw text and false-flagged the project's own docstrings (which mention "candle"/"order book" to explain what the code *avoids*); fixed to inspect code identifiers via `tokenize`, not prose.

**Honest coverage vs. Block 3's named starter set (Rule 25.5 — record divergence, don't hide it):** HMM/Kalman/HDBSCAN/PySR(→lightweight symbolic_regression) delivered; **Wavelet substituted** by FFT/Hilbert/Savitzky-Golay (no clean wavelet dependency in the light numpy/scipy/sklearn stack — Block 3/24 "start small"); **PatchTST and PPO deferred to build step 6** (need a deep-learning framework, out of the light stack); GNN router is the Connector (steps 2/4), not a bank node. Recorded in the tracker so code and plan don't silently drift.

**PLAN.md updated:** Implementation Progress Tracker step 1 marked ✅ DONE with the delivered package, the 30-node/8-layer span, the test evidence, and the honest coverage/deferral notes. **Next in order (Rule 22): step 2 — Connector Intelligence v0 (one hardcoded pathway end-to-end through the Belief format)**, which is also where the §8 dashboard's walking-skeleton first appears. Not started this response.

## BLOCK 41 — Build step 2 SHIPPED: Connector Intelligence v0, the walking skeleton (2026-06-20)

Owner: "continue" → built **build-tracker step 2** (Rule 22, next in order after step-1's model bank), following Rule 25 (spec→test→code→self-verify) and Rule 23 (domain-agnostic).

**What step 2 is (PLAN.md tracker item 2):** Connector Intelligence v0 — wire ONE hardcoded pathway end-to-end through the Universal Belief Space format, **no learned routing yet**, still data-agnostic. A Walking Skeleton (Cockburn) / Tracer Bullet (Hunt & Thomas): prove the whole architecture end-to-end on the smallest slice before adding breadth or intelligence.

**The hardcoded pathway chosen:** `difference → hdbscan → gaussian_hmm → threshold_policy` (signal → pattern → sequence → decision — 4 of the 8 functional layers). Block 18's worked example `Wavelet→HDBSCAN→HMM→PPO` is NOT yet buildable (Wavelet substituted / PPO deferred per step-1's honest coverage), so the path is built only from delivered step-1 node types.

**The key design decision — how data + beliefs flow (steelman'd, Rule 16):** nodes consume `(T, D)` arrays and *emit* `Belief`s, so a linear pipeline needs a carrier. Two models considered: (A) a **signal bus** — the working `(T, D)` array threads through the path, transformer nodes (`difference`) rewrite it for downstream, non-transformer nodes tap it and emit a Belief, and every Belief is recorded in order into a JSON-serializable stream; (B) convert each Belief into the next node's input via per-pair adapters. Model B is exactly what §6/Block 5 says to AVOID ("without per-pair adapters") and contradicts the node interface. **Chose Model A** — faithful to the architecture, no glue code, domain-agnostic, and sets up Feature 3 (reputation attaches to the ordered pathway). The pathway is a pure `list[str]` spec so routing intelligence (step 4) and pathway-reputation (Feature 3) can later operate on it.

**What was built:**
- `pattern_brain/connector.py` — `Connector` (runs a fixed pathway), `Hop` (one node's execution: belief + whether it rewrote the bus + timing), `PathwayResult` (the full auditable trace: ordered belief stream, terminal output, `.decision` helper, `.to_dict()` JSON trace = exactly what the §8 dashboard consumes). `DEFAULT_PATHWAY` + `default_connector()` for the step-2 skeleton. Exported from `pattern_brain/__init__.py`.
- `tests/test_connector.py` — Rule-25 checkable conditions, ALL GREEN: end-to-end run returns a PathwayResult with one ordered hop per node; every hop emits a valid Belief (conf∈[0,1], source set); whole trace JSON-serializable; terminal output is a `decision` with a generic action ∈ {buy,sell,hold}; **bus-threading proven behaviorally** (`threshold_policy` sees the differenced bus, z=-1.668, vs raw z=-1.923 — not re-reading raw input); accepts Node instances + 1-D series; fails loudly (unknown node → KeyError, empty pathway → ValueError); Rule-23 tokenize check (no candle/ohlcv/orderbook identifiers in connector.py).
- `run_pathway.py` — terminal "watch the pathway run" view: prints the bus threading hop-by-hop (rewr vs tap), each belief + confidence, the terminal decision, and the dashboard-ready JSON trace. Textual precursor to §8, **zero dashboard-stack lock-in**.

**Self-verify (Rule 25.4 / Rule 19):** `python3 tests/test_connector.py` → ALL CHECKS PASSED; `python3 tests/test_bank.py` → still ALL CHECKS PASSED (no regression); `python3 run_pathway.py` → runs end-to-end in ~95 ms, 4 hops, gaussian_hmm posterior conf 0.999, terminal `decision` emitted, JSON trace well-formed.

**Deferred sub-decision, surfaced not guessed (Rule 13/15):** the §8 dashboard says step 2 is where its *visual* walking skeleton first appears, BUT §8 is 🟡 not owner-ratified and PLAN.md's own "Still open" list flags "the §8 dashboard's final stack choice + whether a throwaway v0 gets built now" as unresolved. Committing React+FastAPI+WebSocket (the recommended hero stack) or even the Streamlit throwaway is a hard-to-reverse dependency/architecture decision. So step 2 delivered the terminal view (which genuinely satisfies "watch the first hardcoded pathway run" with no lock-in) and the dashboard-stack choice goes to the owner before any web dashboard gets built. **This is the one open decision before step 3 (the stock-data adapter).**

**Next in order (Rule 22):** step 3 — the stock-data adapter (converts candles → the generic `(T, D)` inputs the now-existing bank + Connector already require; closes §6's time/asynchrony gap 7f). Plus the dashboard-stack decision above, whenever the owner wants to make it.

## BLOCK 42 — Why only 30 nodes, and the FULL model universe (deep web search, 2026-06-20)

Owner pushed back: "why only 30 nodes? I asked for many ML models — the realistic universe is ~300–1000+ distinct algorithms/architectures. Do a deep search online to find more."

**Why step 1 shipped 30 (honest answer, not a cap):** the 30 were the *implemented, tested* **starter bank**, governed by Block 3/24's explicit "start small — ~8 models, scale in phases, never jump straight to hundreds" and constrained by the **light numpy/scipy/sklearn stack** (no deep-learning framework installed yet). The plan ALREADY records the real target — "~300-1000+ algorithms across 20 master categories" (§5/Block 18) — and schedules **expanding the bank toward it as build-tracker step 6**. So 30 implemented ≠ the universe; everything needing PyTorch/JAX (Transformers, SSMs, GNNs, deep generative, deep RL, neural point processes) was deliberately deferred to step 6, recorded so plan and code don't drift. The owner is right that the *catalog itself* should be enumerated in full now — done below.

### The enumerated Model Genome catalog — 31 master categories (deep search 2026-06-20)
Counts are approximate "named, distinct" members; "+variants" means each has many published sub-variants that push toward the 1000+ figure.

1. **Classical pattern mining** (~10): Apriori, FP-Growth, Eclat, PrefixSpan, SPADE, GSP, motif discovery (Matrix Profile/STUMPY), shapelets, subsequence DTW, symbolic SAX.
2. **Statistical sequence / HMM family** (~12): HMM, Gaussian-HMM, GMM-HMM, discrete/Markov chain, semi-Markov (HSMM), input-output HMM, factorial HMM, hierarchical HMM, CRF, linear-chain CRF, maximum-entropy Markov, Viterbi/Baum-Welch as the engines.
3. **Classical time-series forecasting** (~25): AR, MA, ARMA, ARIMA, SARIMA, ARIMAX/SARIMAX, VAR, VARMA, VECM, ARCH, GARCH, EGARCH, GJR-GARCH, FIGARCH, ETS/Holt-Winters (simple/double/triple), Theta, TBATS, BATS, Croston, Prophet, NeuralProphet, Kalman structural TS, unobserved-components, exponential smoothing state-space.
4. **RNN family** (~10): vanilla RNN, LSTM, GRU, BiLSTM, peephole-LSTM, ConvLSTM, IndRNN, **xLSTM** (sLSTM/mLSTM), Phased-LSTM, Echo-State-as-RNN.
5. **Transformers — general + time-series** (~30+variants): vanilla Transformer, Informer, Autoformer, FEDformer, Pyraformer, **PatchTST**, **iTransformer**, **TimesNet**, Crossformer, Non-stationary Transformer, ETSformer, Reformer, Performer, Longformer, **TFT (Temporal Fusion Transformer)**, **DeepAR**, Spacetimeformer, PatchMixer, **xLSTM-Mixer**, Pathformer, **DLinear/NLinear/RLinear** (linear baselines), **N-BEATS**, **N-HiTS**, TiDE, TSMixer, SegRNN.
6. **State-space / linear-attention models** (~20): **S4**, S4D, S4ND, DSS, GSS, Liquid-S4, **S5**, **Mamba**, Mamba-2, Vision-Mamba/U-Mamba, **Hyena**, **H3**, Mega, **RetNet**, **RWKV** (v4-v7), linear attention, **Jamba** (SSM-Transformer hybrid), RecurrentGemma, Griffin/Hawk.
7. **Clustering** (~20): K-means, K-means++, MiniBatch-KMeans, K-medoids/PAM, K-modes, fuzzy C-means, **DBSCAN**, **HDBSCAN**, OPTICS, DENCLUE, Mean-Shift, Affinity Propagation, **Spectral clustering** (+graph-structure-learning variants), Agglomerative/Ward, BIRCH, CURE, ROCK, **GMM** (soft), Bayesian-GMM/Dirichlet-process, deep clustering (DEC/DeepCluster).
8. **Anomaly / noise detection** (~25): Isolation Forest, Extended-IF, LOF, COF, kNN-distance, One-Class SVM, SVDD, Elliptic Envelope, HBOS, COPOD, ECOD, autoencoder-AD, VAE-AD, GAN-AD/AnoGAN, DAGMM, USAD, **Anomaly Transformer**, **DCdetector**, TranAD, OmniAnomaly, LSTM-AD, Spectral Residual, Matrix-Profile-AD, RESIST, one-step flow-matching AD.
9. **Representation / dimensionality reduction** (~20): PCA, Kernel-PCA, Probabilistic-PCA, Sparse-PCA, ICA, SVD/Truncated-SVD, NMF, Factor Analysis, LDA (discriminant), Isomap, LLE, Laplacian Eigenmaps, **t-SNE**, **UMAP**, MDS, Random Projection, autoencoder, VAE, denoising/sparse/contractive AE.
10. **Self-supervised / contrastive** (~12): SimCLR, MoCo, BYOL, SimSiam, SwAV, Barlow Twins, VICReg, DINO/DINOv2, TS2Vec, TS-TCC, CPC, masked-autoencoder (MAE).
11. **Graph-based / GNN** (~20): GCN, **GAT/GATv2**, GraphSAGE, **GIN**, ChebNet, MPNN, Graph Transformer, GraphWaveNet, DCRNN, ST-GCN, ST-GAT, **TGN (Temporal Graph Network)**, TGAT, EvolveGCN, DySAT, Node2Vec/DeepWalk (embeddings), GraphConv, APPNP, PNA, D3-GNN (streaming).
12. **Reinforcement learning** (~30): Q-learning, SARSA, Expected-SARSA, DQN, Double-DQN, Dueling-DQN, PER, **Rainbow**, C51/Distributional, QR-DQN, IQN, REINFORCE, A2C, **A3C**, **PPO**, **TRPO**, **DDPG**, **TD3**, **SAC**, **IMPALA**, APEX-DQN/DDPG, R2D2, Ape-X, **MuZero**, AlphaZero, **Dreamer (v1-v3)**, PlaNet, MADDPG, QMIX, GAIL, decision-transformer.
13. **Symbolic regression / program synthesis** (~12): **PySR/SymbolicRegression.jl**, gplearn, Operon, Eureqa, AI-Feynman, **AutoML-Zero**, **FunSearch**, **AlphaEvolve**, LaSR (learned concept library), Deep Symbolic Regression (DSR/DSO), neural symbolic regression (SymbolicGPT/NeSymReS), grammar-based GP.
14. **AutoML / meta-learning / NAS** (~15): MAML, Reptile, FOMAML, ProtoNets, Matching/Relation Nets, RL², SNAIL, AutoSklearn, TPOT, H2O-AutoML, NAS (DARTS, ENAS, ProxylessNAS, training-free predictors), Bayesian-opt/Hyperband/BOHB.
15. **Classical regression** (~12): OLS, Ridge, Lasso, ElasticNet, LARS, Huber, Quantile, Bayesian Ridge, ARD, Theil-Sen, RANSAC, Gaussian-Process regression.
16. **Classical classification** (~12): Logistic Regression, LDA/QDA, Naive Bayes (Gaussian/Multinomial/Bernoulli), kNN, SVM (linear/RBF/poly), SVR, Decision Tree, Perceptron, Passive-Aggressive, RVM.
17. **Ensembles** (~15): Random Forest, Extra-Trees, AdaBoost, Gradient Boosting (GBM), **XGBoost**, **LightGBM**, **CatBoost**, NGBoost, HistGradientBoosting, Bagging, Voting, Stacking, Blending, Super-Learner, Snapshot ensembles.
18. **Probabilistic / Bayesian filtering** (~15): Kalman filter, EKF, UKF, Ensemble-KF, Information filter, **Particle filter** (SIR/SIS), Rao-Blackwellized PF, Bayesian network, Dynamic Bayesian Network, Markov Random Field, Conditional Random Field, Gaussian Belief Propagation, Variational Inference, MCMC/HMC/NUTS, Gaussian Process.
19. **Signal processing** (~15): FFT/STFT, Wavelet (DWT/CWT), Wavelet packet, Hilbert transform, EMD, EEMD, CEEMDAN, VMD, SSA (Singular Spectrum Analysis), Savitzky-Golay, Butterworth/Chebyshev filters, matched filter, cepstrum, **fractional differentiation** (López de Prado), Kalman smoother.
20. **Generative models** (~25): GAN, DCGAN, WGAN/WGAN-GP, CGAN, InfoGAN, StyleGAN(1-3), CycleGAN, TimeGAN, VAE, β-VAE, VQ-VAE, Normalizing Flows (RealNVP, Glow, MAF, IAF, NSF), **Diffusion** (DDPM, DDIM, score-based SDE, latent-diffusion), Flow Matching, autoregressive (PixelCNN/WaveNet), Energy-Based Models.
21. **Physics-informed / constrained** (~10): PINN, Hamiltonian NN, Lagrangian NN, Hamilton-Dirac NN, Neural ODE, Augmented-NODE, DeepONet, **Fourier Neural Operator (FNO)**, Graph Neural Operator, PDE-Net.
22. **Energy-based / statistical-mechanics** (~8): Hopfield (classical + modern/dense-associative-memory), Boltzmann Machine, RBM, Deep Belief Net, general EBM, Diffusion (thermo origin), Simulated Annealing, Spin-glass models.
23. **Equivariant / symmetry-aware** (~6): Group-equivariant CNN (G-CNN), Steerable CNN, E(n)-equivariant GNN (EGNN), SE(3)-Transformer, Tensor-Field Networks, DeepSets (permutation-invariant).
24. **Econophysics / statistical mechanics for markets** (~8): Random Matrix Theory (Marchenko-Pastur eigenvalue cleaning), Agent-Based Models (fundamentalist/chartist), Self-Organized Criticality (sandpile), Tsallis/non-extensive entropy, Minority Game, Ising market models, log-periodic power law (Sornette bubbles).
25. **Point processes** (~10): Poisson, inhomogeneous Poisson, **Hawkes** (univariate/multivariate/nonparametric/quadratic/compound/state-dependent), Neural Hawkes, RMTPP, Transformer-Hawkes, self-correcting process, Cox/doubly-stochastic, CARMA-intensity Hawkes.
26. **Topological data analysis** (~6): Persistent homology, persistence landscapes, persistence images, Mapper, Euler characteristic curves, sliding-window/Takens + persistence.
27. **Rough path / fractional stochastic** (~6): Rough Bergomi, fractional Brownian motion, fractional OU, signature methods (path signatures), log-signatures, neural-SDE.
28. **Fractal / multifractal** (~6): Hurst exponent (R/S), DFA, MFDFA, wavelet-leader multifractal, box-counting dimension, Fractal Market Hypothesis estimators.
29. **Chaos / nonlinear dynamics** (~8): Lyapunov exponents, phase-space (Takens) reconstruction, Recurrence Quantification Analysis, correlation dimension, false-nearest-neighbors, SINDy (sparse dynamics), Koopman/DMD, reservoir-based attractor reconstruction.
30. **Reservoir computing** (~5): Echo State Network, Liquid State Machine, Deep-ESN, next-generation RC (NVAR), physical/photonic reservoir.
31. **Memory / foundation TS models** (~12): Vector DB (FAISS/pgvector/Qdrant as retrieval nodes), Knowledge-Graph memory, MemGPT/Letta-tiered memory, **Chronos**, **TimesFM**, **Moirai**, **TimeGPT**, **Lag-Llama**, **MOMENT**, **Time-MoE**, ChronosX, in-context TS forecasters.

### Concrete conclusion (Rule 10 — a real opinion, not "it depends")
- **The realistic universe is ~450–550 named, distinct algorithms across these 31 categories, expanding past 1000 once every published variant is counted separately** (each S4/Mamba, Transformer-TS, GAN, diffusion, Hawkes, GNN sub-variant). The owner's 300-1000+ figure is correct; my de-duplicated count of clearly-distinct named methods lands ~500.
- **But the active bank should NOT be all ~500.** Our own notes converge four times (Blocks 7/16/17/18) on the finding that *the research problem is the Connector Intelligence + evolution over the bank, not bank size* ("collecting more models — that part is easy"). My concrete recommendation: **the *catalog* (the menu in the Model Genome Database) holds the full ~500+; the *active bank* targets ~120-150 curated entries** — enough genuine diversity across all 8 functional layers for the Connector and the Feature-1/2/3 evolution engines to have real material, while staying inside Block 3/24's phased-scaling discipline (8 → tens → ~150, never "instantiate 500 at once"). Catalog ≠ active bank is the key distinction.
- **Immediate consequence for the build:** step 6 ("expand the bank") is now backed by this concrete 31-category catalog instead of a vague "~300-1000". The biggest single unlock is **adding a deep-learning framework (PyTorch)** — that one dependency turns on categories 4-6, 10-11, 12, 20, 21 at once (Transformers, SSMs, GNNs, deep generative, PINNs, deep RL), which is most of what's missing from the current 30. That's a step-6 decision (and a Rule-23 check: each still wraps behind the generic Node interface).

**Sources (ranked by reliability/recency, Rule 11):** peer-reviewed surveys + primary papers (arXiv/ACM/Springer/MDPI 2024-2026) over blog posts; thuml Time-Series-Library and the S4→Mamba survey were the highest-yield catalogs.
- Time-Series-Library (thuml) https://github.com/thuml/Time-Series-Library ; DL TS forecasting survey https://pmc.ncbi.nlm.nih.gov/articles/PMC12453695/
- S4→Mamba survey https://arxiv.org/html/2503.18970v3
- Clustering rapid review https://arxiv.org/html/2401.07389v1 ; comprehensive analysis https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11419652/
- DL time-series anomaly survey (ACM CSUR) https://dl.acm.org/doi/full/10.1145/3691338 ; multivariate TS-AD taxonomy https://www.mdpi.com/1424-8220/25/1/190
- Dimensionality-reduction review https://pmc.ncbi.nlm.nih.gov/articles/PMC12453773/
- Deep generative comparative review https://durham-repository.worktribe.com/output/1232078/
- GNN architectures review https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00876-4
- RL libraries (machin) https://github.com/iffiX/machin ; policy-gradient taxonomy https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- Symbolic regression: PySR https://github.com/MilesCranmer/PySR ; AutoML-Zero https://arxiv.org/pdf/2003.03384
- Hawkes overview (SIAM Review) https://epubs.siam.org/doi/10.1137/21M1396927
- Ensemble boosting overview https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
- Bayesian filtering (Kalman→particle) https://people.bordeaux.inria.fr/pierre.delmoral/chen_bayesian.pdf

## BLOCK 43 — Resume session: found unfinished §8 dashboard work, ratified the stack, finished + verified it (2026-06-21)

A new session opened with "read all files in pattern brain and continue implementing the plan." Reading the repo turned up uncommitted, unlogged state from a prior session: `pattern_brain/connector.py` had been modified (adding an `iter_run()` generator) and an untracked `dashboard/` folder existed (`server.py` + `index.html`) implementing the §8 living-graph dashboard — but `PLAN.md` itself still recorded the dashboard stack as **deferred to the owner, not yet chosen** (Block 41's note). Code on disk had silently gotten ahead of what the plan said was decided, and none of it was committed (Rule 24), logged (Rule 2), or reflected in `PLAN.md` (Rule 21) — a prior session was almost certainly interrupted mid-build.

**Per Rule 15 ("ask, don't guess"), asked the owner directly** rather than either committing it as final or discarding it unilaterally: keep the uncommitted dashboard work and finish it (ratifying React+FastAPI+ReactFlow+Plotly as the actual §8 stack), discard it and jump straight to build-tracker step 3, or leave it untouched and start step 3 anyway. **Owner chose: ratify the stack and finish it now.**

**What "finishing it" actually involved (Rule 25: spec → test → code → self-verify against the plan):**
- No system Python `pip` existed; created a project-local `.venv/` (added `python3.13-venv` via `sudo apt-get install`, passwordless sudo was available) and installed `fastapi`, `uvicorn`, `websockets`, `httpx`, plus the model bank's `numpy`/`scipy`/`scikit-learn` into it. Recorded as `requirements-dashboard.txt`, kept separate from the model bank's `requirements.txt` per §0b's "add only what's proven needed" staging rule. `.venv/` added to `.gitignore`.
- Ran the **existing** `test_bank.py` + `test_connector.py` first to confirm the uncommitted `connector.py` change hadn't regressed anything — both green. The `iter_run()` change itself was a clean refactor (`run()` now just collects `iter_run()`'s yielded hops), but had **no test of its own** — added `test_iter_run_streams_hops_matching_run` to `tests/test_connector.py` asserting the generator yields the identical hop sequence `run()` collects (one execution path, not two that could silently drift apart).
- Started the dashboard server and drove every HTTP route (`/`, `/api/bank`, `/api/pathway`, `/api/run`) plus the `WS /ws/run` stream with a real WebSocket client — all worked, all returned exactly the expected shapes (4 hops, decision output, `start`→`hop`×4→`done` framing).
- **Then went further than "the API responds" and actually loaded the page in a real browser** (installed Chromium via apt, drove it headless with Playwright) — because Rule 25 step 4 calls for self-checking against intent, and "the server returns 200" is not the same claim as "the dashboard works." This caught a real bug the prior session's work had never surfaced: `index.html` imported `reactflow` with `?external=react,react-dom` (telling esm.sh not to bundle its own React copy) but never declared an import map resolving the resulting bare `"react"`/`"react-dom"` specifiers — so the page threw `Failed to resolve module specifier "react"` and silently rendered a **blank `#root`** in every real browser, while every `curl` check and every API test kept passing. Fixed by adding a `<script type="importmap">` mapping `react`/`react-dom`/`react-dom/client`/`htm`/`reactflow` to their esm.sh URLs so all of them share one React instance — the standard esm.sh multi-package recipe. Re-verified after the fix: clicked "Run pathway" in the live headless browser, confirmed all 4 nodes rendered and glowed in pathway order, 4 belief cards streamed into the belief-space panel, the Plotly confidence chart drew, and clicking a node populated the Node Inspector with its Model-Genome metadata. Screenshots taken as evidence.
- Wrote `tests/test_dashboard.py` (FastAPI `TestClient`, no live server/browser needed) covering all 4 HTTP routes + the WS stream, so this is a repeatable automated suite, not just the ad-hoc scripts used to find the bug. All three `test_*.py` suites green together.
- Updated `PLAN.md` §8 from 🟡 PROPOSED to ✅ DECIDED (stack ratified), added a `[✅ DONE 2026-06-21, Block 43]` entry to the build-tracker step-2 item documenting exactly what shipped and how it was verified (including the import-map bug, as a concrete instance of why browser-level verification matters, not just API-level), and removed the now-resolved "dashboard stack choice" line from the "Still open" list. Updated `README.md` with dashboard setup/run instructions and the new file list.

**Concrete opinion (Rule 10):** the import-map bug is the kind of failure that *only* a real-browser check catches — every layer below the browser (curl, the FastAPI TestClient, even the WebSocket client) reported total success on a dashboard that never actually rendered anything. Going forward, any §8 dashboard change should be considered unverified until it's been loaded in an actual browser (headless is fine), not just hit with `curl`/`TestClient`.

**Process note:** asked the owner before touching the ambiguous uncommitted state instead of guessing either direction (Rules 13/15) — this is exactly the situation those rules exist for, and it surfaced a real cross-session continuity gap (work done, never logged/committed) worth staying alert to in future sessions too.

Rules applied: 1, 2, 4, 13, 15, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 44 — Build-tracker step 3: the stock-data adapter (+ §8 Adapter-View tab) (2026-06-21)

Session opened with "read all files in pattern brain and continue implementing the plan." Re-read RULES.md, PLAN.md, README.md, the whole `pattern_brain/` package + `dashboard/`, and the existing test suites. Per Rule 22 (implement in order, no skipping), steps 1, 2 and the §8 dashboard walking skeleton were all done + committed (Blocks 40/41/43), so the next item in order was **build-tracker step 3 — the stock-data adapter**. No ambiguity to ask about (Rule 15), so proceeded.

**Forced pause before building (Rules 13/18) — the design decision, checked against Rule 23/§0 first:** the adapter is the ONE place stock-specific code is allowed to live, and its whole job is to bend real candle data into the exact `(T, D)` finite-float array the already-built bank/Connector require — never reshaping the core. Concrete consequences I committed to before writing code:
- **Put it in `pattern_brain/adapters/` and deliberately do NOT import it from `pattern_brain/__init__`** — the core stays domain-clean and importable with zero domain coupling; the dependency only ever flows adapter → core. A future second domain plugs in a sibling module here without touching the bank, Connector, or belief format (the portability test of §0).
- **No pandas** (the obvious reflex for time alignment) — the light-stack discipline (§0b "add only what's proven needed", Block 3/24 "start small") says don't pull a heavy dep in for grid resampling when numpy `searchsorted` does causal as-of alignment fine. Steelman of pandas (Rule 16): it would give `resample()`/`asof` for free and handle calendars/timezones — real advantages — but none of that is needed for step 3's single synthetic candle source, and it would bloat the core's deps; revisit only if a real multi-calendar need appears.
- **Three layers, separated so the first is reusable by any future domain:** (1) generic numeric tools with zero OHLCV references — `frac_diff`/`frac_diff_weights` (López de Prado fixed-width fractional differentiation, AFML ch.5; the §5/§6 named Feature-Factory technique, finally made concrete) and `align_to_grid`/`align_sources` (causal as-of resampling of irregular/asynchronous/differently-sampled sources onto one shared regular grid — the concrete closure of §6 gap 7f); (2) `CandleSeries`, a validated OHLCV container (the domain shape); (3) `StockDataAdapter`, the only component that knows open/high/low/close/volume.

**Implementation (Rule 25: spec → checkable conditions → minimum code → self-verify):**
- `pattern_brain/adapters/stock.py` + `__init__.py`. Default features are a small *stationary* candle set (`fracdiff_log_close`, `log_return`, `hl_range`, `log_volume`) — deliberately NOT the full ~10k-feature universal version (that's later; step 3 is "one data source — candles"). The adapter trims the diff/frac-diff warm-up so its output meets the core's `Node._validate` finite-input contract **exactly** — and the test literally calls `Node._validate` on the adapter output to prove "the data bends to the system" (§0) mechanically, not just rhetorically.
- `tests/test_adapter.py` (all green): frac-diff anchor cases (d=0 = identity, d=1 = plain first difference — the two cases that make it checkable), the memory-preservation property that is the whole *point* of frac-diff (|corr with the price level| of frac-diff(0.4) = 0.73 vs plain returns = 0.09), causal-ffill no-lookahead alignment, two different-rate sources collapsing onto one shared grid, the finite `(T,D)` contract, the adapter output running **unchanged** through `default_connector()` end-to-end (Rule 21 — does its job AND breaks nothing), gap/asynchrony absorbed via `resample_step`, loud validation failures, and two Rule-23 separation checks (core `__init__` doesn't import adapters; the generic numeric tools carry zero stock identifiers, so they stay portable).
- **Rule 25 step 5 in action (fix the plan/tests when implementing reveals a gap):** `test_bank.py`'s existing core Rule-23 scan walked the whole `pattern_brain` package and (correctly, by its old logic) flagged the new adapter's OHLCV words. But the adapter is the *sanctioned* home for those words. Rather than weaken the check, I updated it to **prune the `adapters/` subpackage** from the core scan (its separation is enforced instead by `test_adapter.py`), and documented why in its docstring. All four suites — bank, connector, adapter, dashboard — green together.

**§8 Adapter-View tab (the dashboard cross-cutting track's step-3 view, which the plan explicitly schedules at step 3):** added `GET /api/adapter` (synthetic candles → `StockDataAdapter` → generic `(T,D)` → the SAME default pathway runs on it) and a new "Adapter View (stock)" tab in `index.html` rendering the candlestick (the ONLY stock-specific visual on the entire dashboard, Rule 23/§8), the emitted feature matrix, the feature-name pills, and the resulting decision + belief stream. Core tab and Adapter tab switch via header tabs. `tests/test_dashboard.py` gained `test_adapter_endpoint`.

**Verified in a real headless browser (Block 43's hard-won discipline — only a browser catches render bugs that curl/TestClient miss):** drove the live server with Chromium via Playwright (pointed at the system `/usr/bin/chromium` since Playwright's bundled shell wasn't installed). Confirmed the core tab renders, then switched to the Adapter tab and confirmed 1 Plotly candlestick, 4 feature pills, the matrix rows, and a live `action` value, with no JS errors (one harmless favicon 404).

**A real bug surfaced and was fixed during that verification (Rule 14 — evidence before root cause):** the first browser run threw `Cannot read properties of undefined (reading 'hops')` and `/api/adapter` returned 404 *live* despite 200 in the TestClient. Evidence first (server log): `[Errno 98] address already in use` — a **stale dashboard server from a prior session was still bound to port 8077**, serving old code with no `/api/adapter` route, while my new process silently failed to bind. Killed the stale process, restarted on the current code, re-verified clean. This is the same cross-session-continuity failure mode noted in Block 43 (work/processes left behind by interrupted sessions) — worth staying alert to: always confirm the live server is *your* current code, not a stale one squatting the port.

**Concrete opinion (Rule 10):** the right unit of "domain separation" turned out to be *the dependency direction*, not just folder placement — keeping `adapters/` out of the core's `__init__` is what actually makes the core provably domain-clean (a test can assert it), and it's a stronger guarantee than the tokenize identifier-scan alone. The frac-diff memory result (0.73 vs 0.09) is concrete empirical confirmation that the §5 technique earns its place over plain returns, not just a citation. Step 3 is genuinely complete: the adapter does its job (feeds the unchanged core, end-to-end, proven in code and in a browser) and breaks nothing. Next in order is **step 4 — prototype ONE routing-mechanism candidate (LLM-tool-calling first, per the plan's recommendation)**.

**Process note:** §6 gap 7f (time/asynchrony) is now formally CLOSED in PLAN.md by `align_to_grid`/`align_sources`; gap 7g (interlingua versioning) remains open as build step 5.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 45 — Build-tracker step 4: LLM-tool-calling routing (Connector Intelligence v1) (2026-06-21)

Owner said "continue with step 4." That confirms the plan's recommended-but-unratified mechanism: **LLM-tool-calling** (PLAN.md build-tracker step 4 was 🟡 "Claude's recommendation, not yet owner-confirmed"). Mechanism now owner-ratified.

**Forced pause before building (Rule 13) — the one genuine fork, and why I did NOT block on it:** LLM-tool-calling needs an LLM, and the obvious open question is *which backend* (cloud Anthropic, local Ollama, or none). I considered asking (Rules 13/15), but concluded it isn't blocking: building the tool-calling **scaffold + a swappable LLM boundary** is required no matter which backend is later chosen, so the answer doesn't change what I build now — it only changes a small later follow-on (adding the real client). Blocking would have been re-litigation against an explicit "continue." So I proceeded with the offline-runnable scaffold and surfaced the backend as the one open sub-decision — exactly how step 6's PyTorch dependency is deferred, and consistent with steps 1-3 staying offline-testable (§0b: "the implementation bends to fit the interface"). Steelman of asking first (Rule 16): the owner has strong infra opinions (Ollama in the trading bot, Rule 1 separation) and might want a real client wired immediately — but the pluggable design means that's a trivial drop-in later, so the cost of not asking now is near-zero.

**What v1 changes from v0:** v0 (`connector.py`) ran a hardcoded `list[str]` pathway. v1 (`routing.py`) **decides the next node hop-by-hop**: the bank's nodes are presented as callable "tools"; a `Router` chooses which to call next given the legal candidates + the belief stream so far + the layers already visited; the Connector runs it, feeds the Belief back, repeats until a terminal node fires / the router stops / `max_hops`. The dynamically *discovered* pathway is the unit Feature 3 will later attach reputation to.

**The design decision that matters most — the LLM is a swappable boundary, not a hard dependency:**
- `HeuristicRouter` — the offline default; a real, inspectable policy that advances through the Block-18 layers one at a time and terminates in a decision. No API key; this is what makes the whole suite run green offline.
- `LLMRouter` — depends ONLY on one injected `complete(system, user, tools)` callable. Tested with a *fake* completer that drives the full tool-calling loop (choose tool → execute → feed back → repeat) with no network. A real Anthropic/Ollama client is later just that one callable.
- `RoutedConnector` — reuses v0's `Hop`/`PathwayResult` so a routed run is consumed by the dashboard/audit **identically** (one trace shape, not a second that could drift), plus `route()` for per-hop `RouterDecision` rationale.

**The Block-18 layer-adjacency constraint is enforced structurally, not by the router:** the Connector's candidate filter (`_candidates`) does forward-only + an adjacency window (`max_layer_jump`) + terminal nodes always reachable + supervised nodes withheld when no labels are present (Rule 21 — a routed run never throws on a missing `y`). The router only *chooses* among the legal candidates it's offered. This cleanly separates structure (Connector) from policy (Router).

**Two bugs surfaced by the tests and fixed before calling it done (Rule 14 — evidence before root cause):**
1. The first `HeuristicRouter` exhausted an entire layer (all 4 signal nodes, then all 4 noise nodes) before advancing, because it always picked the lowest `(layer_index, node_type)` and same-layer nodes kept being lowest — so it never reached a decision within `max_hops`. Fix: the policy now *prefers advancing* to the next un-entered feature layer, and only terminates when no forward feature layer remains.
2. The canonical pathways legitimately jump straight to a terminal (e.g. `sequence → decision`, `equation → PPO`), which a strict adjacency window would forbid. Fix: terminal-layer nodes (`decision`, `rl`) bypass the window — you may always choose to *finish* — while feature layers still respect it. Also corrected `TERMINAL_LAYERS` to include `rl` (an RL action is a legitimate pathway ending, per Block-18's worked examples).

**Evidence (Rule 25):** `tests/test_routing.py` all green — heuristic builds a real signal→noise→pattern→sequence→probability→equation→decision path; routing is genuinely dynamic (restricting the bank changes the path; it's not the v0 fixed list); the adjacency window is enforced; the LLM tool-calling loop runs end-to-end via the injected fake; illegal/malformed tool choices raise `RouterError`; tool specs are JSON + domain-free; `max_hops` caps runaway; supervised nodes withheld without labels; `route()` rationale serializes; Rule-23 tokenize check. All five suites green together.

**§8 dashboard (cross-cutting track):** added `GET /api/route` + a "Connector v1 — routed pathway" card showing the discovered path as pills and the per-hop rationale (chosen node, layer, reason, and the candidate set it chose among). **Verified in a real headless browser** (Block 43's discipline): the card renders the routed path + 21 rationale lines, no JS errors (lone favicon 404). The recurring **stale-server-on-port-8077** gotcha bit again (a prior background server was squatting the port, serving old code) — re-confirmed the port was free and the live server was *current* code before verifying. `tests/test_dashboard.py` gained `test_route_endpoint`.

**Concrete opinion (Rule 10):** the right abstraction for "LLM routing" in this codebase is **the LLM as a single injected `complete` callable**, not an SDK baked into the Connector. That keeps the architecture testable and deterministic today, keeps Rule 1 separation (no Ollama/cloud coupling sneaking in), and makes the real backend a one-function follow-on — which is genuinely the only thing left undone in step 4. Step 4 is complete: routing does its job (a router assembles a valid pathway dynamically, end-to-end, proven in code and browser) and breaks nothing. Next in order is **step 5 — the interlingua versioning scheme (closes §6 gap 7g)**.

**Open sub-decision flagged, not silently resolved (Rule 15):** which real LLM backend the `LLMRouter` should use (cloud Anthropic / local Ollama / stay offline) — owner's call when they want a live LLM in the loop.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 15, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 46 — Build-tracker step 5: interlingua versioning + drift scheme (closes §6 gap 7g) (2026-06-21)

Owner said "continue with step 5" (after step 4). Per Rule 22, step 5 is the **interlingua versioning scheme** — "needed once the model bank grows past the starter set and any model might get retrained/replaced" (the long-flagged §6 gap 7g).

**Evidence before design (Rule 14):** before writing any contract I enumerated what the bank *actually* emits — every node's belief `type` + payload keys on a generic sequence. Findings that shaped the design: 12 belief types across 30 nodes; most types have a stable set of always-present payload keys, BUT three types — `action`, `decision`, `forecast` — have **no payload key common to all their emitters** (e.g. the 3 decision nodes emit different keys; the 6 forecasters share none). The interlingua is, today, *loose* for exactly those types. That's not a bug to hide — it's the latent inconsistency a versioning/drift scheme exists to surface.

**Design (Rule 25, Rule 23-checked):** `pattern_brain/interlingua.py`, four parts:
1. **Versioned catalog.** `INTERLINGUA_VERSION` == `belief.SCHEMA_VERSION` (single source of truth — the version stamped on every belief *is* the interlingua version it claims). A `BeliefTypeSchema` per type declares required vs optional payload keys. The v0.1 catalog is declared *from the gathered evidence* so the current bank conforms by construction — it is the honest baseline, not an aspiration the bank fails.
2. **Conformance/drift detection.** `validate_belief`/`detect_drift`/`conformance_report` flag unknown type / missing required key / un-migratable version mismatch — the concrete "a retrained node silently drifted" catch.
3. **Migration.** `register_migration`/`migrate_belief` walk a registered chain to upgrade an older belief forward, so a version bump doesn't break interoperability. Ships a worked, tested example (v0.0 forecast key `value` → v0.1 `estimate`). A version mismatch *with* a migration path is treated as migratable (not a violation); *without* one, it's a violation.
4. **Coherence (soft).** `interlingua_coherence` reports types whose payloads diverge across emitters. The loose types are left required-key-free in v0.1 (so they still conform) and surfaced as soft notes — the early-warning that the interlingua wants tightening at the next version. This is the honest home for the evidence finding above, rather than either ignoring it or forcing a contract the bank would fail.

**Concrete opinion (Rule 10):** the right move was NOT to immediately tighten `decision`/`forecast`/`action` to require an `action`/`estimate` key — that would have meant editing bank nodes (scope creep for step 5) and declaring a baseline the current bank violates. The right move is: declare an honest loose baseline that the bank passes, make drift *measurable* against it, and surface the looseness as a tracked soft finding + a ready migration mechanism for when someone decides to tighten it (that tightening becomes a real v0.2 bump with a registered migration — exactly the scenario this scheme is built to handle). That keeps step 5 minimal and correct while leaving a clear, mechanized path to the stricter contract.

**Evidence (Rule 25):** `tests/test_interlingua.py` (all green) — current bank conforms (30 beliefs / 12 types); unknown-type, missing-key, and un-migratable-version all flagged; a v0.0→v0.1 migration bridges the bump and the migratable belief is *not* flagged; migration is a no-op on the current version and raises `InterlinguaError` with no path; coherence surfaces the loose types without making them hard violations; a real Connector run conforms; everything JSON-serializable; Rule-23 tokenize check. All six suites green together.

**§8 dashboard (cross-cutting track):** `GET /api/conformance` (version + catalog + drift report over a run + coherence over the bank) + an "Interlingua conformance" card extending the belief-space view. **Verified in a real headless browser** (Block 43 discipline): card shows v0.1, "run drift ✓ conforms (4/4)", 12 catalogued types, 7 coherence notes, no JS errors (lone favicon 404). `test_dashboard.py` gained `test_conformance_endpoint`.

**Process note — the port-8077 stale-server gotcha (third recurrence) + a new wrinkle:** verification got tangled because several browser-check commands were auto-backgrounded by the harness and hit the server *while it was down* (`net::ERR_CONNECTION_REFUSED` / `/api/conformance -> 000`), and a leftover server kept the port. I stopped guessing, killed ALL listeners on 8077 by PID (not `pkill -f` — that self-matches the kill command, the bug from last response), confirmed the port free, started a fresh server, confirmed it was *current* code with a live `curl`, and only then ran the browser check once cleanly. Lesson reinforced: before any browser verification, assert (a) port free of stale servers, (b) the live server returns 200 on the new route — never run the browser against an assumed-up server.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 47 — Build-tracker step 6 (Phase 6a): light-stack bank expansion + the Model-Bank dashboard view (2026-06-21)

Owner said "continue with step 6." Step 6 is "expand the model bank," and the plan's headline move is **adding PyTorch** — a heavyweight, hard-to-reverse core dependency. Per Rule 13/15/16 I paused and asked (light-stack-first vs PyTorch-now vs both), because the plan pulls two ways: it names PyTorch as the biggest unlock, but also mandates phased "start small, never jump straight to hundreds" scaling, and PyTorch isn't installed here (though installable: PyPI reachable, ~2 GB, 119 GB free).

**Owner's answer reframed the whole step:** they didn't pick an option but said *"but in dashboard I see only 4 nodes."* That's the real signal — the §8 Living-Graph hero view renders the hardcoded 4-node *pathway*, not the bank; `/api/bank` returned all 30 but nothing *showed* them. So expanding the bank is pointless if the bank stays invisible. I did both: **expand the bank (light-stack Phase 6a, the recommended default they implicitly accepted by not choosing PyTorch) AND build a Model-Bank dashboard view that shows every node.** PyTorch stays a deliberate, gated Phase 6b.

**Bank expansion (+30 nodes, 30 → 60), all light-stack (numpy/scipy/sklearn), each behind the generic Node interface, each reusing the v0.1 interlingua belief types so the bank still conforms (Block 46):** signal (detrend, butter_lowpass, welch_psd); anomaly + dim-reduction denoisers (LOF, one-class SVM, elliptic envelope, ICA/SVD/kernel-PCA reconstruction denoisers); clustering (mean-shift, spectral, BIRCH, OPTICS); forecasters (naive-mean, drift, Holt, Theta); regularized/robust regressions (ridge/lasso/elasticnet/huber/theil-sen); supervised classifiers (SVM, random forest, gradient boosting, KNN, gaussian NB, decision tree); bandits (Thompson, gradient). This is an honest "phase," roughly doubling the bank — not jumping to the ~150 target in one go (Block 3/24 discipline).

**Dashboard "Model Bank" tab (the genome-map view, §8):** renders all 60 nodes as columns by functional layer (signal 7 / noise 10 / pattern 9 / sequence 8 / probability 4 / equation 8 / decision 9 / rl 5), supervised nodes marked `*`, with a node inspector + per-layer counts. This is the direct answer to "I only see 4 nodes": the Living-Graph tab shows the running pathway; this tab shows the whole catalogue.

**Evidence (Rule 25):** all six suites green at 60 nodes (`test_bank.py` registry bound raised to ≥55; `test_interlingua.py` confirms all 30 new nodes conform to the v0.1 catalog — the versioning system from Block 46 doing its job; `test_dashboard.py` `/api/bank` ≥55). One light-stack contract fix surfaced by the bank test: the SVD reconstruction denoiser must pass through on 1-D input (TruncatedSVD needs ≥2 features).

**A real regression found + fixed during verification (Rule 18, Rule 21 "breaks nothing"):** after expanding to 60 nodes, `/api/conformance` and `/api/route` — which process the WHOLE bank and auto-fire on every page load — made the single-process dev server wedge under repeated/concurrent loads (every endpoint, even trivial `/api/bank`, timed out). Fixed by `@lru_cache`-ing those deterministic responses server-side (compute once per process); endpoints now respond in ~3 ms.

**Verification honesty (this took real effort and I nearly mis-attributed it):** my first browser checks showed the Model-Bank tab with "loading bank…" and 0 graph nodes, which *looked* like a BankView bug. Evidence-first debugging (Rule 14) instead of guessing a root cause: I timed every endpoint and found even `/api/bank` hung — the *server* was wedged, not the component. Root cause was MY OWN test load: dozens of orphaned chromium processes (from timed-out playwright runs) repeatedly loading the page and firing the pre-caching heavy endpoints, saturating the single-process server. After killing all strays + adding caching, a clean run rendered **all 60 react-flow nodes**, with console confirming `BankView: got 60 nodes` and no JS errors (screenshot captured). Lesson reinforced: don't call a "blank dashboard" a frontend bug until you've proven the backend it depends on is actually responding — and clean up test processes aggressively so they don't poison later runs.

**Concrete opinion (Rule 10):** the owner's "I only see 4 nodes" was the most valuable steer of the step — it exposed that visibility, not node count, was the gap. The right deliverable for step 6's first phase was therefore *bank growth + a genome-map view together*, not either alone. PyTorch (Phase 6b) remains correctly deferred: it's a heavyweight commitment the owner hasn't chosen, and the light stack still has plenty of Block-42 categories left to mine before deep nets earn their dependency weight.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 15, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 48 — Step 6 cont.: Phase 6a batch-2 (light, 60→88) + Phase 6b (PyTorch deep nodes, →94) (2026-06-21)

Owner: "continue Phase 6a (more light-stack nodes toward the ~120–150 target), kick off Phase 6b (PyTorch)." So both — and crucially the owner now **explicitly chose PyTorch**, ratifying the heavyweight dependency that I'd deliberately deferred in Block 47 pending their call (Rule 15 satisfied).

**Phase 6a batch 2 — +28 light nodes (60 → 88), all numpy/scipy/sklearn, each behind the generic Node interface, all reusing the v0.1 interlingua belief types (no catalog change):** signal transforms (zscore_normalize, ewma, cumsum) + periodogram; statistical anomaly detectors (MAD, IQR/Tukey, Mahalanobis, k-NN-distance); scalable clusterers (mini-batch + bisecting k-means); ML autoregressive forecasters built on a per-feature lag-embedding (SVR/KNN/RF/GBR/kernel-ridge); more closed-form linear laws (LARS, LassoLars, OMP, ARD); more classifiers incl. a **sklearn MLP — a neural net that needs NO torch** (extra-trees, adaboost, bagging, mlp, ridge, LDA, QDA); baseline policies (random, softmax). All six light suites green at 88; interlingua confirms every new node conforms.

**Phase 6b — PyTorch deep nodes (the headline step-6 unlock):**
- Installed **torch 2.12.1+cpu** into the project `.venv` (CPU wheel via the pytorch CPU index — smaller/faster than the CUDA build, and this box is CPU-only). Recorded in `requirements-deep.txt`, kept OUT of the light `requirements.txt`.
- New `pattern_brain/nodes/deep.py` with **6 deep nodes**: `lstm_forecaster`, `gru_forecaster`, `transformer_forecaster`, `tcn_forecaster` (one-step forecasters trained briefly on a lag-embedding of the window → `forecast`), `deep_autoencoder` (bottleneck MLP reconstruction → `denoised`), `autoencoder_anomaly` (reconstruction-error → `anomaly`). Each wraps behind the SAME Node interface and emits the SAME interlingua types the light nodes use — so the Connector calls a torch node identically to a sklearn one, and the bank still conforms with zero catalog change (§0b "the implementation bends to fit the interface"; Block 46's versioning holds).
- **Optional-dependency discipline (the important design choice, §0b / Rule 23):** torch is imported defensively in deep.py; if it's absent the module registers NOTHING and the light bank is untouched. Proven both ways: system `python3` = 88 nodes, deep absent, all light suites green; `.venv` (torch) = 94 nodes, deep present, all suites green. The core never hard-depends on torch.

**Evidence (Rule 25):** `tests/test_deep.py` skips cleanly without torch and, with torch, asserts all 6 deep nodes register, run on (T,D) and 1-D input, emit JSON-serializable beliefs that pass interlingua conformance, and have correct forecast/reconstruction shapes. ALL suites green on BOTH interpreters (light 88 / full 94). `/api/bank` returns 94. **Model Bank dashboard tab verified in a real browser showing all 94 nodes** including the deep ones (`BankView: got 94 nodes`, no JS errors, screenshot captured). The endpoint-caching fix from Block 47 made this verification smooth this time (pre-warmed `/api/conformance` once — 3.4s with deep-node training — then everything cached/instant; no server wedging).

**Concrete opinion (Rule 10):** the right way to add a heavyweight framework to this project is as an **optional, defensively-imported node module**, not a core requirement — it keeps the light bank fast and torch-free for anyone who doesn't want a 2 GB dependency, while the dashboard's `.venv` (which carries torch) lights up the deep nodes automatically. Bank is now **94 active** (88 light + 6 deep): real progress toward ~120–150, with whole deep categories still untouched (true SSM/Mamba, GNN, deep-RL, PINN, deep generative) as future Phase-6b batches. Notably, several "neural" capabilities are already covered light (sklearn MLP classifier) — reinforcing the plan's recurring finding that bank breadth was never the hard part.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 49 — Step 6: reached the ~120–150 target band (bank 94 → 123) (2026-06-21)

Owner: "keep filling toward ~120–150 (more deep categories and/or light gaps)." Did both, landing at **123 active (112 light + 11 deep)** — inside the target band.

**Light +24 (88 → 112), all reusing the v0.1 interlingua types (no catalog change):**
- signal/spectral: median_filter, gaussian_smooth, wiener_denoise, hp_filter (Hodrick-Prescott trend), spectral_entropy, autocorr_period.
- anomaly: pca_residual_anomaly, histogram_anomaly (HBOS-style), kde_anomaly (probability layer, gaussian KDE).
- clustering: affinity_propagation, bayesian_gaussian_mixture.
- forecasters (lag-embedding AR): extra_trees, adaboost, ridge, mlp_regressor (a second light sklearn neural net), + seasonal_naive (autocorr-detected period).
- linear laws: quantile_regression, linear_svr.
- classifiers: hist_gradient_boosting, nearest_centroid, sgd_classifier, passive_aggressive_classifier.
- bandits: discounted_ucb, exp3 (the non-stationary/adversarial families the evaluator design references).

**Deep +5 (6 → 11) — broadened beyond the first forecaster/autoencoder set into new Block-42 categories:**
- `mlp_deep_forecaster` (feed-forward over the flattened lag window).
- `ssm_forecaster` — a diagonal state-space recurrence (S4D-lite: h_t = a⊙h_{t-1} + Bx_t, scanned), the simplified SSM/Mamba family.
- `vae_denoise` / `vae_anomaly` — a variational autoencoder (deep generative): reparameterized latent + ELBO, reconstruction → denoise / reconstruction-error → anomaly.
- `gcn_denoise` — a graph-convolutional network (GNN): builds a k-NN graph over timesteps, normalized adjacency, trains a 2-layer GCN autoencoder to smooth/reconstruct rows. Implemented with plain torch (no torch_geometric dependency — message passing as Â X W).
All behind the same Node interface, emitting existing interlingua belief types, torch still optional.

**To pass `p` (window length) to the forecaster nets that need it (MLP flatten, SSM scan), the `_TorchForecaster._build` signature was widened to `_build(self, D, p)`** and the existing LSTM/GRU/Transformer/TCN builders updated to accept (and ignore) it — a clean base-class change, not a per-node hack.

**Evidence (Rule 25):** all suites green on BOTH interpreters (light 112 via system python3 with deep absent; full 123 via .venv with torch); interlingua confirms every new node — light and deep — conforms to v0.1; `/api/bank` returns 123; **Model Bank dashboard tab verified in a real browser rendering all 123 nodes** including ssm/vae/gcn, no JS errors. Layer spread: signal 17 / noise 21 / pattern 13 / sequence 24 / probability 5 / equation 14 / decision 20 / rl 9.

**Concrete opinion (Rule 10):** the bank is now genuinely diverse across all 8 functional layers and both stacks — comfortably enough material for the Connector and the Feature-1/2/3 evolution engines to work with, which our notes have said four times over is the actual research problem (not bank size). I'd treat 123 as "enough to move on": further catalog growth (true GNN message-passing/torch_geometric, deep-RL, PINN/Neural-ODE, real Mamba/S4, foundation TS models) is genuine but has diminishing returns versus building the final phase (Features 1/2/3 + the shared evaluator), which is what actually exercises a bank of this size. Recommend the final phase next.

Rules applied: 1, 2, 4, 9, 10, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 50 — Final phase, slice 1: the shared 5-layer Evaluator (the hard gate) (2026-06-21)

Owner: "the final phase — Features 1/2/3 + the shared 5-layer evaluator." The plan is explicit (§4) that the evaluator is a HARD GATE that must exist before the mutation engines — candidates can't be safely scored before replacing an incumbent. So per Rule 22 (in order) the final phase is built in slices, evaluator first. (Also fixed two infra issues this turn: killed a runaway 59-min `until`-loop background shell left over from an earlier verification, and restarted the dashboard server — the "not loading" was simply that I'd stopped it after the last check; it's back up and the caches are pre-warmed.)

**Design decision (Rule 13) — resolving the long-standing §0 tension:** the evaluator operates on a generic **outcome series** (a 1-D array of per-period "higher-is-better" values). "Sharpe" = mean/std of that series; "drawdown" = worst peak-to-trough of its cumulative sum — these are generic statistics, not anything financial. WHICH statistics define the Pareto front is a pluggable `ObjectiveSpec`. The domain-specific part (turning a candidate's predictions into the outcome series — returns for finance, negative error for another domain) lives in the caller/adapter, never in the evaluator. This makes the core domain-agnostic (Rule 23) and finally closes §0's flagged inconsistency about the evaluator hardcoding financial metrics.

**Delivered — `pattern_brain/evaluator.py`, all five layers (Blocks 21-22):**
- **L0 data hygiene:** `purged_walk_forward_splits` — expanding walk-forward with an embargo gap adjacent to each test block (purges label-overlap + serial-correlation leakage; López de Prado).
- **L1 significance:** `sharpe`, `probabilistic_sharpe_ratio` (corrects sample SR for length/skew/kurtosis), `deflated_sharpe_ratio` (PSR vs the expected-max Sharpe over N trials — the selection-bias guard), `min_track_record_length` — real Bailey & López de Prado formulas via `scipy.stats.norm`.
- **L2 non-stationary:** `discounted_ucb_score` / `sliding_window_ucb_score` — recency-weighted value + exploration bonus; the one mechanism that subsumes pathway-death (decayed score) and curiosity (exploration bonus).
- **L3 population:** `nondominated_sort` / `pareto_front` (NSGA-II fast non-dominated sort) + `cscv_pbo` (probability of backtest overfitting via combinatorially-symmetric cross-validation).
- **L4 anti-gaming:** `anchor_check` — every promoted candidate must stay sane on each fixed held-out episode.
- `Evaluator` ties them: `evaluate_candidate` (L0→L2 on one candidate over leakage-safe folds) and `select` (L3→L4 admission gate over a population). `EvaluationReport` is JSON-serializable for the dashboard/audit.

**Evidence (Rule 25):** `tests/test_evaluator.py` all green — each layer verified on synthetic series where the answer is known: walk-forward folds never leak; PSR ranks a strong/long series above a weak/short one; **DSR correctly deflates PSR 0.996 → 0.000 under 100 trials** (selection-bias guard biting exactly as intended); minTRL finite with an edge, infinite without; discounted-UCB favors the recently-improving stream; Pareto sort drops dominated candidates; **PBO noise 0.44 vs genuine-edge 0.00**; end-to-end an edge candidate is admitted and pure noise is rejected at the significance layer; pluggable `ObjectiveSpec` works; Rule-23 domain check clean. All suites green (light python3 + .venv).

**Concrete opinion (Rule 10):** building the evaluator first was clearly right — it's now a clean, tested, domain-agnostic scoring/admission service that Features 1/2/3 can each call as `select(...)` without re-implementing any statistics, and the DSR/PBO layers give real protection against the exact failure mode (selecting an overfit winner from many mutated candidates) that an evolution loop is most prone to. Next slice: Features 1 (model-instance mutation), 2 (algorithm/equation evolution), 3 (pathway evolution) — each gated by this evaluator.

Rules applied: 1, 2, 4, 9, 10, 13, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 51 — Final phase, slice 2: Features 1/2/3 evolution engines (2026-06-21)

`pattern_brain/evolution.py` — the three evolution engines, all gated by the slice-1 Evaluator. Shared `Evolver` GA loop: init population → score each via the Evaluator (Layers 0-2) → breed next gen from elites (mutate/crossover) → final population admission via the Evaluator (Layers 3-4). Order resolved 1→3→2.
- **Feature 1 `ModelEvolver`** — mutate/crossover a forecaster node's hyperparameters (parameter space within a fixed algorithm family).
- **Feature 3 `PathwayEvolver`** — mutate/crossover Connector pathways (graph topology); `_legalize` keeps every genome layer-ordered and forecaster-head-terminated so the evolved graph is always runnable.
- **Feature 2 `AlgorithmEvolver`** — genetic programming over symbolic expression trees (function set +,-,*,protected-/,neg,sin,tanh,abs over lagged inputs/mean/last/slope) — invents NEW forecasting equations, a level above Feature 1.

Fitness is a generic **directional-reward** outcome series (predict next-move direction of channel 0; +1 hit / −1 miss / 0 flat), scored by the domain-agnostic Evaluator. `EvolutionResult.history` records every candidate born→tested→promoted = the §8 Evolution feed.

**Two real bugs found via the random-walk null test (Rule 14) and fixed — this is the whole point of the gate:**
1. DSR was over-deflating with a hardcoded `var_sr=1.0` (assumes cross-trial Sharpe variance of 1.0 → benchmark ~1.45, which no short directional-reward stream clears) → NOTHING ever admitted, even a perfect AR fit. Fixed: the Evaluator now estimates `var_sr` from the sample (Lo 2002 ~(1+sr²/2)/n) when not given, and the Evolver re-deflates each candidate against the REAL cross-sectional variance of the population's Sharpes over the TOTAL number tried.
2. The evolution loop called `Evaluator.select` WITHOUT a perf_matrix → CSCV/PBO (the strongest anti-overfit layer) never ran → the engines admitted FALSE POSITIVES on a pure random walk (e.g. AlgorithmEvolver DSR 0.96 on noise). Fixed: each candidate's walk-forward outcome stream is stored and fed into `select` as the PBO matrix.

After both fixes the gate behaves correctly: on a pure **random walk** (the proper null — next move independent of the past) ALL three engines admit **0**; on genuine **AR-edge** data the matched engine (Feature 1) **promotes**. (Note: an i.i.d. series is NOT a valid null here — it's mean-reverting, so "predict toward the mean" is a real edge; the random walk is the honest no-edge baseline.)

**Concrete opinion (Rule 10):** the engines working is almost secondary — the headline result is that the GATE works: it admits a matched model on a real edge and refuses everything on a random walk, which is exactly the backtest-overfitting protection an evolutionary search most needs (it's the component most prone to selecting a lucky winner from many). Feature 1 dominating on AR data (the matched model) while Features 2/3 correctly abstain is honest behavior, not a weakness. The plan's entire build order is now implemented end-to-end.

Rules applied: 1, 2, 4, 9, 10, 13, 14, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 52 — Comprehensive dashboard: show EVERYTHING (every model + every file) (2026-06-21)

Owner: "still the dashboard is only showing 4 nodes — make it show every ML or any file; think hard and come up with ideas on how all the files show in the dashboard." Root cause of the persistent "4 nodes": the dashboard LANDED on the Living-Graph tab (the 4-node running pathway); the 123-node bank was behind a tab the owner didn't click, and the final-phase work (evaluator, evolution) had NO views at all.

**Fix — rebuilt the dashboard around a comprehensive Overview + a view for every subsystem + a file explorer:**
- **Overview (new default landing)** — `GET /api/overview`: KPI tiles (123 models, 8 layers, 12 belief types, 32 source files, 8456 lines, 9 test suites), a clickable card per subsystem (Model Bank, Connector v0, Routing v1, Stock Adapter, Interlingua, Evaluator, Evolution F1/2/3) each with a live count, build-progress checklist, and models-by-layer. Opening the dashboard now shows the WHOLE system at a glance — never "4 nodes".
- **Files (new)** — `GET /api/files`: walks `pattern_brain/`, `tests/`, `dashboard/` and lists every `.py`/`.html` with path, line count, kind, and its docstring/first-line summary, with a live path filter. Directly answers "any file in the dashboard".
- **Evaluator (new)** — `GET /api/evaluator`: the 5-layer gate made visible — the purged walk-forward folds + an edge candidate (✓ PASSES) vs a pure-noise candidate (✗ rejected), with PSR/DSR/Sharpe.
- **Evolution (new)** — `GET /api/evolution`: runs Features 1/2/3 and renders the born→tested→promoted feed + best genome + admitted count per engine.
- Kept Model Bank (123 nodes), Living Graph, Adapter; folded Routing + Interlingua cards into Living Graph.

**Performance:** the new heavy endpoints (`/api/evolution` ~15s as it runs three GA loops, `/api/conformance` processes the whole bank) are `@lru_cache`d AND a **startup background thread pre-warms every cache at boot**, so a user's first click is instant rather than a 15s stall.

**Evidence (Rule 25 + Block-43 browser discipline):** `tests/test_dashboard.py` gained `test_overview/files/evaluator/evolution_endpoint` — all green. **Verified in a real browser:** the page LANDS on the Overview showing the six KPI tiles incl. "123 models" and 7 subsystem cards; Model Bank renders 123 react-flow nodes; Files lists 32 rows; Evaluator shows "✓ PASSES gate" for the edge and rejection for noise; Evolution renders all three feature feeds; no JS errors (favicon only). Screenshot captured.

**Concrete opinion (Rule 10):** the real problem was information architecture, not missing data — `/api/bank` always returned all 123, but the landing buried it. Leading with a system Overview (KPI tiles + subsystem cards) is the standard observability-dashboard pattern (Grafana/MLflow home pages) and is the right fix: it makes the whole system legible in one screen and routes to detail views. The Files explorer + per-subsystem views mean every artifact the project contains — every model and every source file — is now reachable from the dashboard.

Rules applied: 1, 2, 4, 9, 10, 14, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 53 — Post-plan: closing the loop with pathway reputation (Block 17) (2026-06-21)

Owner: "all 3" (close the loop + real LLM + resolve open decisions). Slice 1 = closing the loop, the centerpiece — making the built pieces feed each other instead of being separate modules.

`pattern_brain/reputation.py` realizes Block 17 ("score CONNECTIONS, not models — the graph itself becomes the memory"):
- `PathwayReputation` — per-pathway usage + a recency-weighted (discounted-UCB) outcome score + per-edge aggregates, optionally regime-conditioned (the World Model).
- `ReputationRouter(Router)` — routes by the reputation of the edge it would create, falling back to the heuristic layer-advance on never-seen edges (curiosity/exploration).
- Wiring: `PathwayEvolver` now records every tested pathway into a reputation store (new `_after_score` hook on the base `Evolver`), so after evolution the store drives routing + the dashboard.
- Dashboard: new **Pathways** tab + `/api/reputation` — the Pathway Leaderboard (§8 view #3) ranked by reputation + top edge reputations.

**Evidence:** `tests/test_reputation.py` green — better pathway scores higher; unseen edge returns None (so routing explores); regime-conditioning works; ReputationRouter prefers the proven edge and falls back on unseen ones; PathwayEvolver populates the store. Dashboard test `test_reputation_endpoint` green (15 pathways, 16 edges); Pathways tab verified in a real browser (15 leaderboard rows + edges, no JS errors).

**Concrete opinion (Rule 10):** this is the integration that makes the project a *system* rather than a toolbox — the Evaluator's verdict now persists on the graph and shapes future routing, exactly the synapse-like memory Block 17 argued for. Edge-colouring the live pathway graph (§8 view #1) is a smaller follow-on; the leaderboard (view #3) is the higher-value realization and is done.

Rules applied: 1, 2, 4, 9, 10, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 54 — Post-plan: real LLM backends for the router (Block 45) (2026-06-21)

Slice 2 of "all 3". The step-4 `LLMRouter` already ran the tool-calling loop against one injected `complete(system,user,tools)` callable; `pattern_brain/llm.py` now provides real implementations + auto-detection, so routing uses an actual LLM when configured and falls back to the heuristic otherwise.
- **Ollama** (local, preferred — keeps Rule-1 separation, no cloud/cost): stdlib `urllib` to `$OLLAMA_HOST` /api/chat with tools; no extra package needed.
- **Anthropic** (cloud): `anthropic` SDK + `$ANTHROPIC_API_KEY`, Messages API with `tool_choice={"type":"any"}` to force a tool pick.
- `auto_completer()` returns the first reachable backend (Ollama then Anthropic) or `(None,None)`; `default_llm_router()` returns an `LLMRouter` wired to it, or `None` → caller keeps the heuristic. Same optional-dependency discipline as the torch nodes.
- Dashboard: `/api/route` now uses the LLM router if available (reporting the backend), and `/api/llm` exposes `llm_backend_status()`.

**Evidence:** `tests/test_llm.py` green WITHOUT any live backend — the response parsers are pure functions tested with mock responses; availability degrades gracefully (here both False → heuristic); the full tool-calling loop runs against a fake completer; generic tool specs convert to both anthropic + ollama formats. `test_llm_endpoint` green.

**Concrete opinion (Rule 10):** the right design is auto-detect + graceful fallback, not a hardcoded backend — the owner can turn on a real LLM simply by running Ollama or setting `ANTHROPIC_API_KEY`, with zero code change, and everything (tests, dashboard) keeps working when neither is present. Defaulting to local Ollama first respects the project's separation/privacy posture; cloud Anthropic is the explicit opt-in.

Rules applied: 1, 2, 4, 9, 10, 18, 19 (imported discipline), 20, 21, 23, 24, 25.

## BLOCK 55 — Post-plan: resolved the three remaining §7 open decisions (2026-06-21)

Slice 3 of "all 3" — closing the last open questions in the plan with concrete decisions (Rule 10), each with light implementation + a test, not just prose:

1. **"cpu ml models" deliverable format — RESOLVED: it IS the Model Genome Library, as a manifest + build recipe, NOT committed weights.** `registry.genome_manifest()` emits a JSON manifest of the whole bank (every node's metadata grouped by layer) + the `requirements` (light numpy/scipy/sklearn + optional torch) to materialize it. Since every node is built on demand, there are no weight files to ship — the catalog + recipe is the artifact. Confirms the §5/§6 link that was previously only flagged.
2. **Sub-agent personas — RESOLVED: yes, each engine gets a defined role-persona** (`pattern_brain/personas.py`): Feature 1 = The Breeder, Feature 3 = The Architect, Feature 2 = The Inventor, alongside the Connector's The Synthesist (§3). Attached via `Evolver.persona`; used as the agent's voice when it drives an LLM. Differentiated roles (Rule 17) rather than one undifferentiated agent.
3. **Step-10 framing — RESOLVED: the slots ARE function-level.** The bank's 8 functional layers (signal/noise/.../rl = Block-16 Layer-1 categories) are exactly the slots Feature 3 connects; any algorithm is an interchangeable filler (every layer holds ≥2). `PathwayEvolver` already wires nodes by layer and swaps the algorithm filling each function slot — so it's function-level by construction; no rework.

**Evidence:** `tests/test_decisions.py` green (manifest structure + serialization; all four personas defined + distinct; each engine carries its persona; 8 function-level slots each with ≥2 interchangeable algorithms). All 12 suites green on both interpreters.

**Concrete opinion (Rule 10):** with these resolved, `PLAN.md` has NO open `❓` decisions left — every numbered build step is implemented and every deferred question is decided. The plan is, as written, complete. Further work from here is genuinely new scope (e.g. a second data-domain adapter to prove §0 portability, the §8 view #1 reputation-coloured live edges, a real LLM turned on by running Ollama, or wiring the scheduled-research routine) rather than anything still pending in the plan.

Rules applied: 1, 2, 4, 9, 10, 16, 18, 19 (imported discipline), 20, 21, 22, 23, 24, 25.

## BLOCK 56 — NEW MAJOR SCOPE: the ML Engineer AI Agent + LLM sweep + dashboard chat (2026-06-21)

### Owner's message (verbatim, Rule 2)
> "i need you to think hard and see all the free cloud llms and local llm on 12vcpu and 32 gb ram search online to find the best there are some free chinese free cloud llm and local llm pick which will be best for ml model creation and mutation and creating ml algorithms etc also i need you to create a ml engineer ai agent from scratch or you think it should have function to have all the things how a ml engineer operates this should have observe, creating ml models and algorithms and creating ml algorithms and it automatically downloads the test and data needed for its use and training and this ml engineer ai agent should have access to online and have access to data download or any additional files needed and i need it to come up with all the available books on the subject ml engineering and a ml engineering student reads and convert them into vector db so and stores its work and function should be in a infinite loop in a close circle on continuous learning observing and working with existing models and creating new models and ranking them in order for including in the model nodes now do add this message into the plan first understand this message clearly and update the plan after thinking on how to make it happen so do a full online sweep search and in dashboard i need to able to converse with the ai agent just like how i am doing all this session with you etc now"

(Preceding instruction, same turn: "first i need you to read all the files of the pattern brain and continue the project implementing it following rules and then execute this prompt.")

### Decomposition (Rule 9 — sub-questions before answering)
The message contains **seven** distinct asks, which I separate so none gets lost:
1. **LLM sweep** — survey free cloud LLMs (incl. free *Chinese* ones) + local LLMs runnable on **12 vCPU / 32 GB RAM, no GPU**; pick the best for *ML model creation, mutation, and algorithm invention*.
2. **An "ML Engineer AI Agent" built from scratch** that operates the way a real ML engineer does: **observe** existing models/state; **create + mutate ML models**; **create + mutate ML algorithms**.
3. **Autonomous data/test acquisition** — it downloads, on its own, the datasets + test data it needs to train/evaluate, with **internet access** and the ability to fetch any additional files.
4. **A knowledge base of the field** — it gathers *all available books an ML-engineering student reads*, ingests them into a **vector DB**, and uses + stores its own work there.
5. **An infinite closed-loop** — continuous learning: observe → work with existing models → create new ones → **rank them** → include the winners as **model nodes** → repeat forever.
6. **Dashboard chat** — converse with the agent live, exactly like this Claude session.
7. **Process** — add this message to the plan *first*, understand it, then implement (the standing Rule 21/22/25 workflow).

### How this maps onto the EXISTING Pattern Brain (Rule 14 — evidence first; Rule 23 — domain-agnostic check)
The good news, verified by reading the whole codebase: **almost every primitive the agent needs already exists** — the agent is mostly an *orchestration layer that drives them in a loop*, not a from-scratch rebuild. Mapping:
- **"observe existing models"** → `default_bank()` / `genome_manifest()` (123 nodes), `PathwayReputation.leaderboard()`, `Evaluator` reports, `EvolutionResult.history`.
- **"create + mutate ML models"** → `ModelEvolver` (Feature 1, parameter space).
- **"create + mutate ML algorithms"** → `AlgorithmEvolver` (Feature 2, genetic programming over expression trees) + `PathwayEvolver` (Feature 3, graph topology).
- **"rank them for inclusion in the model nodes"** → `Evaluator.select` (the 5-layer hard gate) → record on `PathwayReputation`; promoted winners get `register()`ed into the bank as new nodes. This is the literal "ranking → including in model nodes."
- **"vector DB of books / stores its work"** → §6 vector-DB role #3 (RAG over unstructured docs) + role #4 (Letta/MemGPT archival memory) were *designed* in the plan (Block 29) but never built — **this is the first genuinely new component.**
- **"observe/create in an infinite closed loop"** → no daemon exists yet; the pieces (evolve→evaluate→record) are there but only run one-shot in tests. **The loop is new.**
- **"internet access + auto data download"** → no fetch tooling exists yet. **New** (domain-specific fetchers stay in `adapters/` per Rule 23; the generic web/file fetch is domain-clean).
- **"converse in the dashboard"** → `dashboard/server.py` exists with the right stack; needs a chat endpoint + tab. **New surface, existing stack.**
- **The agent's "brain"** → `llm.py` (`auto_completer`) is the swappable LLM boundary already used by `LLMRouter`; the agent reuses it. The sweep (ask #1) decides *which* models fill it. **This also resolves PLAN.md §198's last open item** (which real LLM backend) — the new scope subsumes it.

**Rule 23 re-check (mandatory):** the agent, the loop, the evaluator-ranking, the evolvers, and the book-knowledge-base are all **domain-agnostic** — they operate on the generic bank / generic outcome series / generic ML-engineering text. The only domain-specific surface is *dataset fetching for the stocks domain*, which goes in `adapters/` (the sanctioned domain home), never in the agent core. The book KB is general ML-engineering knowledge — domain-agnostic by nature. ✅ Consistent with §0.

### ASK #1 — the LLM sweep (Rule 9 parallel search done; Rule 10 forced conclusion; Rule 11 reliability/recency)
Full online sweep run 2026-06-21 (search results logged; sources below). **Hardware reality first (Rule 14):** 12 vCPU, **32 GB RAM, NO GPU**. CPU-only token rates (llama.cpp, Q4_K_M, this class of box): a 32B (~20 GB) ≈ 2–4 tok/s — too slow for an agent that makes many calls per loop; a 14B (~9 GB) ≈ 5–8 tok/s — usable when latency is tolerable; a 7B (~4.5 GB) ≈ 10–15 tok/s — comfortable. **Never quantize a coding/ML model below Q4_K_M** (introduces syntax/logic errors — not worth it). So local is a *fallback / offline* tier, and the heavy reasoning (algorithm invention especially) should lean on the **free cloud** tier.

**Concrete picks (Rule 10 — no wishy-washy "it depends"):**
- **Cloud, primary (free): Z.AI `GLM-4.x-Flash`** — *unlimited free* on the Z.AI API, strong coding/agentic scores; the best free-tier workhorse for the agent's high-volume "think/route" calls.
- **Cloud, reasoning workhorse: DeepSeek (V3/V4)** — the strongest cheap reasoner for *ML algorithm design / symbolic-program guidance*; free access paths exist (OpenRouter free tier, NVIDIA NIM with phone verify) and its paid floor ($0.14/$0.28 per 1M) is the cheapest if free runs out.
- **Cloud, coding specialist (free): Qwen** — `qwen3-coder:free` via OpenRouter (20 RPM/200 RPD) or DashScope's free 1M-token allotment; excellent for code-shaped mutations.
- **Cloud, fast/large free tiers already proven in the sibling project:** Groq (`llama-3.3-70b`, lowest latency), Cerebras (`qwen-3-235b`), NVIDIA NIM, Mistral (1B tok/mo), SambaNova — all OpenAI-compatible, all free-tier.
- **Local (Ollama, offline fallback): `qwen2.5-coder:7b` (Q4_K_M)** as the everyday local model on this box (comfortable ~10–15 tok/s), `qwen2.5-coder:14b` (Q4_K_M) when quality matters and latency is acceptable. These are the strongest local coding/ML models that fit 32 GB with headroom.
- **Embeddings for the vector DB (CPU): `nomic-embed-text`** (≈580 chunks/s on CPU — the throughput winner) as default, with **`bge-m3`** (MIT, multilingual, hybrid) as the quality option. Both run locally on this box; no GPU needed.

**The unifying engineering fact:** every one of these cloud providers exposes an **OpenAI-compatible `/chat/completions`** endpoint, so a *single* `openai_compatible_completer(base_url, model, api_key)` covers all of them; the agent just needs a **priority chain with per-provider cooldown** (the exact pattern the sibling trading bot already proved at ~190 RPM across 5 free keys). Default chain: **GLM-4.x-Flash (free) → DeepSeek → Qwen → Groq/Cerebras → local Ollama (qwen2.5-coder) → offline heuristic.** No key required to *run* (degrades to local/heuristic), keys unlock the cloud tiers — same graceful-degradation discipline as Block 54.

**Why these for *ML creation/mutation/algorithm invention specifically* (Rule 16 — steelman the alternative):** the strongest case for a *local-only* posture is privacy + Rule-1 separation + zero cost + no rate limits. That's real, and is exactly why local stays the guaranteed fallback. But algorithm *invention* (Feature 2's program/equation search guidance, and the agent's "what should I try next" reasoning) is the single most reasoning-heavy call in the loop, and on a no-GPU box a local 7–14B at a few tok/s is both weaker and slower than a free frontier-class Chinese cloud model that costs nothing. So the verdict is **cloud-primary for reasoning, local-guaranteed for resilience** — not either/or.

### ASK #2–6 — the agent design (Rule 12 — concrete; Rule 17 — defined roles; Rule 18 — slow down)
Proposed as a new **"ML ENGINEER AGENT" phase** in `PLAN.md`, built in order (Rule 22), each step TDD'd (Rule 25). Headline shape: a `MLEngineerAgent` with an **OODA-style closed loop** — **Observe → Orient (retrieve knowledge + LLM-reason) → Decide (pick an experiment) → Act (evolve/evaluate) → Rank/Record (gate + reputation + promote to a node) → repeat** — runnable as a single step (for tests/dashboard) or as an infinite daemon (with a stop flag + persisted state so it survives restarts). It is given a **defined persona** ("The ML Engineer", per Rule 17/19) and *drives the existing evolvers* rather than re-implementing them. New components, in build order:
1. **ENG-0 — LLM provider expansion** (`llm.py`): add the sweep's free cloud providers as one OpenAI-compatible completer + a priority chain with cooldown. (Also resolves §198.)
2. **ENG-1 — Knowledge base / vector DB** (`pattern_brain/knowledge.py`): a light CPU embedding store (numpy-cosine default; optional sentence-transformers/Chroma), a curated **manifest of open-access ML-engineering books/resources** (from the sweep: *Machine Learning Systems* mlsysbook.ai, the *ML Engineering Open Book*, Géron, Chip Huyen's *Designing ML Systems*/*AI Engineering*, the awesome-ML reading lists), an ingest→chunk→embed→retrieve pipeline, and an archival tier where the agent stores its own findings (role #4).
3. **ENG-2 — Tools** (`pattern_brain/agent/tools.py`): observe-state, knowledge-retrieve, web-fetch + file-download (internet access), run-evolution (Features 1/2/3), evaluate, register-winner-as-node — each a typed tool the LLM can call.
4. **ENG-3 — The agent loop** (`pattern_brain/agent/engineer.py`): `MLEngineerAgent`, the closed OODA loop, single-step + daemon modes, persisted state, the "rank → promote to model node" wiring.
5. **ENG-4 — Dashboard chat + activity feed**: `/api/chat` (+ WS) and an "ML Engineer" tab to converse with the agent and watch its loop, exactly like this session.

### Steelman of "don't build the agent at all, just run evolvers manually" (Rule 16)
The honest counter-argument: the evolvers + evaluator + reputation already produce ranked, gated models on demand — an autonomous loop adds operational risk (a runaway daemon, rate-limit storms, low-value compute churn) for a system whose plan was just declared *complete*. That's a fair caution, and it's why the loop ships with: a hard stop flag, per-iteration budget caps, the same Evaluator gate (nothing promotes without passing it), and single-step mode as the default for testing. But the owner's explicit goal is *continuous, autonomous* model discovery ("infinite loop in a close circle on continuous learning"), which manual runs structurally cannot provide — so the agent is the right build, with the safety rails above, not a reason to skip it.

### Deep architecture sweep — the HOW, not just the WHICH-LLM (2026-06-21, owner correction: "you did not do a deep online search to come up with a plan to make the ideas possible")
The first sweep answered ask #1 (which LLMs). This second, deeper sweep answers asks #2–6 (how to actually build the agent) against real, current, peer-reviewed/production prior art — so the design is *grounded*, not invented (Rule 11/14). Headline finding: **the owner's "infinite loop that creates and ranks models" is the exact shape of the 2026 state-of-the-art autonomous-ML-engineering agents — and Pattern Brain already has most of the machinery they use.** The agent is an orchestration of proven patterns, not research-grade novelty.

**1. The loop itself = AIDE's Solution Space Tree Search (Weco AI, arXiv 2502.13138 — SOTA on OpenAI's MLE-bench).** AIDE frames ML engineering as *code optimization over a tree*: each node is a solution version, each edge is one improvement step, and a **Solution Generator** either **drafts** a new solution or **improves/debugs** an existing one, kept or discarded by a **metric**. This is *precisely* Pattern Brain's existing shape: candidate models/algorithms/pathways = solution nodes; the `Evaluator` = the metric/gate; `PathwayReputation` = the tree's memory of what scored well. **Decision:** the agent's Decide step adopts AIDE's draft-vs-improve choice directly — greedily improve the current best (mutate/crossover via the evolvers) most of the time, occasionally draft a fresh candidate (new random genome / new pathway), exactly AIDE's explore/exploit over the solution tree.

**2. Defined roles = R&D-Agent's dual-agent design (Microsoft, arXiv 2505.14738 — top MLE-bench performer; the sibling trading bot already has `f59_rd_agent`).** R&D-Agent splits the work into a **Researcher** (uses *performance* feedback to generate ideas) and a **Developer** (refines code from *error* feedback), with multi-trace parallel exploration. This is the concrete realization of Rule 17 (defined sub-roles): our agent's **Orient/Decide = the Researcher** (reads the book-KB + reputation + eval reports, proposes the next experiment) and **Act = the Developer** (runs the chosen evolver, handles failures/retries). Multi-trace = running Features 1/2/3 as parallel exploration traces.

**3. Open-ended ideation loop = Sakana's AI Scientist (published in *Nature*; AI-Scientist-v2 passed human peer review).** It runs *idea generation → literature search → experiment → iterate*, feeding previous ideas + feedback into the next generation. This validates both the **infinite loop** and the **literature grounding**: our book vector-DB retrieval *is* the "literature search" step that informs each idea. Confirms the owner's "continuous learning, observing, creating" is a real, working pattern, not a fantasy.

**4. Memory = Letta/MemGPT's tiered architecture (the 2026-ranked agent-memory standard).** Core memory (RAM, always in-context — current bank/regime/goal), **Archival memory (external vector store, queried via an `archival_memory_search` tool)**, Recall memory (full interaction log). This is *exactly* §6 vector-DB role #4, already in the plan (Block 29) — now confirmed as the field-standard fix for the core problem "agents forget between sessions because LLM context is stateless: write important facts at session end, inject at session start." **Decision:** the agent persists its findings to the archival tier each loop and reloads core memory on start, so the infinite loop survives restarts. (Self-evolving variants — Nous Hermes *skill compilation*, MS *SkillOpt* text-space optimization — map to our "promote a winning candidate → `register()` it as a permanent model node": that IS skill compilation in our domain, done by adding nodes/pathways, not by fine-tuning weights.)

**5. RAG / book ingestion — the proven two-phase pipeline (2026 production consensus).** Offline **indexing** (load → chunk → embed → store) + online **retrieval**. Concrete decisions from the sweep: **recursive chunking at ~300–512 tokens with 10–20% overlap** (page-level for PDFs) as the starting strategy; **CPU embeddings = nomic-embed-text / BGE** (from sweep #1); **vector store = a light numpy-cosine store by default (§0b), Chroma optional for dev, Qdrant/pgvector the scale path**; **hybrid (dense+sparse/BM25) retrieval** is the 2026 standard to add later. PDF parsing starts light (pypdf) with Unstructured.io as the optional upgrade. The **book corpus** is curated from sweep #1's open-access list (mlsysbook.ai, ML-Engineering Open Book, Géron, Chip Huyen, awesome-ML reading lists) — stored as a `BOOK_MANIFEST` of fetchable URLs the ingest tool pulls on demand.

**6. Autonomous data/test acquisition — real programmatic APIs (ask #3).** The agent's data tool wraps proven sources: **OpenML** (sklearn-integrated, *standardized train/test splits* — the best default for the agent's "download the test and data it needs", domain-agnostic), **HuggingFace `datasets` / `hf_hub_download`**, **Kaggle API** (`dataset_download_files`), **UCI** (680+ sets). Domain-specific (stock) fetchers stay in `adapters/` per Rule 23; OpenML/HF/UCI fetching is domain-clean and lives in the agent's generic data tool. Generic web/file fetch (internet access) is a thin stdlib `urllib`/`requests` tool with an allowlist.

**Concrete conclusion of the deep sweep (Rule 10):** none of the six capabilities requires inventing new ML research — each is a 2026-proven pattern, and Pattern Brain already implements the hardest parts (the evolvers, the 5-layer gate, pathway reputation, the LLM boundary, the dashboard). The build is therefore **integration + orchestration**: wire the existing evolve→gate→reputation pieces into an AIDE-style tree-search loop, give it R&D-Agent's Researcher/Developer roles, a Letta-style tiered memory backed by a CPU vector DB seeded with the ML-engineering book corpus, OpenML/HF data tools, and a dashboard chat. That is what the ENG-0…ENG-4 build steps in `PLAN.md` §9 now encode.

**Deep-sweep sources (Rule 11):** AIDE — arxiv.org/abs/2502.13138, aide.ml, github.com/WecoAI/aideml; R&D-Agent — arxiv.org/abs/2505.14738 (Microsoft, open-sourced); MLE-bench — arxiv.org/abs/2410.07095, github.com/openai/mle-bench; ArchPilot multi-agent — arxiv.org/abs/2511.03985; Sakana AI Scientist — sakana.ai/ai-scientist, sakana.ai/ai-scientist-nature; agent memory — atlan.com best-ai-agent-memory-frameworks-2026, machinelearningmastery.com 6-best-agent-memory-frameworks, arxiv.org/abs/2603.07670 (Letta/MemGPT tiered memory); RAG — firecrawl.dev best-chunking-strategies-rag, kapa.ai how-to-build-a-rag-pipeline-from-scratch-2026, dev.to RAG pipelines in production; data APIs — huggingface.co/docs/datasets, kaggle API docs, openml.github.io, github.com/tirthajyoti/UCI-ML-API.

### Process note
Per the owner's instruction and Rules 21/22/25: this block records the discussion; `PLAN.md` gets the new §9 + the ENG build-tracker phase next; then implementation proceeds **in order**, starting ENG-0, each step spec→test→code→self-verify, with a git push per response (Rule 24).

**Sources (Rule 11 — recency-weighted, 2026 sources):** Z.AI/GLM free tier + coding ranks — morphllm.com/llm-api, benchlm.ai best-chinese-llm, huggingface.co/blog/daya-shankar/open-source-llms; free-API aggregators — github.com/cheahjs/free-llm-api-resources, github.com/mnfst/awesome-free-llm-apis; local CPU/Qwen-coder — unsloth.ai qwen3-coder docs, localaimaster.com best-local-ai-coding-models, microcenter 8/16/32GB guide; embeddings — promptquorum.com best-embedding-models-local-rag-2026, milvus.io choose-embedding-model-rag-2026; ML-eng books — mlsysbook.ai, kdnuggets 5-free-must-read, medium javarevisited ML-engineer-reading-list-2026, github josephmisiti/awesome-machine-learning books.md.

Rules applied: 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 23, 24 (this response), 25.

### Addendum — runtime/agent files stay inside the folder (owner reaffirmed 2026-06-21)
Owner: "all the files related to the pattern brain should be built inside this folder, nothing outside this folder." This was already **Rule 1**, but it predated the §9 agent, which *generates files at runtime* (downloaded datasets, the vector DB, agent memory, loop state). So rather than a redundant new rule, **Rule 1 was extended with a clarification** making explicit that runtime/agent-generated artifacts are bound by it too — they go in `pattern-brain/data/`, `pattern-brain/knowledge_store/`, `pattern-brain/agent_state/`, never outside the folder; `.gitignore` now excludes those dirs (local-only, still inside the folder). PLAN §9 records the exact paths each ENG step must use. Confirmed back per Rule 8: Rule 1 already said "all files… code, data, deliverables, everything… live inside `/home/dicktator4134/pattern-brain/`. Nothing… anywhere else on disk."

### Addendum 2 — make all the new ideas visible in the dashboard (owner-mandated 2026-06-21)
Owner: "in plan add to make all this new idea visible in dashboard." Added to PLAN §9 as a **mandatory cross-cutting requirement** for the whole ENG phase (mirroring how each original build step shipped its own §8 view): ENG-0 → LLM Engine card (`/api/llm`, backend already live); ENG-1 → Knowledge Base panel (book manifest + passage/source/archival counts + a live retrieval box, via a new `/api/knowledge`); ENG-2 → Tools panel; ENG-3 → the "ML Engineer" tab with the live Observe→Orient→Decide→Act→Rank loop feed + candidate leaderboard + promoted-to-node events (reusing the Evolution feed + Pathway Leaderboard); ENG-4 → conversational chat in that same tab. The visual cards ship together in an "ENG dashboard pass" alongside ENG-3/ENG-4 and are browser-verified per the Block-43 discipline (no claiming a rendered card that wasn't actually loaded in a browser).

### Addendum 3 — ENG-0…ENG-4 ALL BUILT + verified (2026-06-21)
The whole ML Engineer Agent phase is implemented in order (Rule 22), each step spec→test→code→self-verify (Rule 25), each git-pushed (Rule 24):
- **ENG-0** `llm.py` cloud chain (Z.AI/DeepSeek/Qwen/Groq/Cerebras/NVIDIA/Mistral/OpenRouter + cooldown + cloud-primary auto-select + chat completers). Resolves the last open §198 item.
- **ENG-1** `knowledge.py` — chunking + dependency-free hashing embedder (optional nomic/bge) + numpy-cosine store (Rule-1 path-guarded persistence) + `KnowledgeBase` (ingest/retrieve) + Letta archival remember/recall + `BOOK_MANIFEST` (8 open-access ML-eng books).
- **ENG-2** `agent/tools.py` — `Toolbox`: observe, knowledge, internet (web/download), `fetch_dataset` (synthetic+OpenML+injectable), `run_evolution` (Features 1/2/3), `promote_to_node` (rank→register as a bank node), LLM tool specs.
- **ENG-3** `agent/engineer.py` — `MLEngineerAgent` OODA loop (AIDE draft/improve + R&D-Agent roles), single-step + `run_forever` daemon with STOP-flag + budget caps, state persisted under `agent_state/`, offline-or-LLM `chat()`. A 6-step test run promoted 5 evolved models into the bank (113→118) — the "rank → include in the model nodes" loop demonstrably works.
- **ENG-4** dashboard — `GET /api/knowledge`, `GET /api/agent/status`, `POST /api/agent/step`, `POST /api/agent/chat`, and the "🤖 ML Engineer" tab (LLM Engine + Knowledge Base + chat + live loop feed). **Browser-verified** (project-local chromium, zero JS errors).

**Concrete conclusion (Rule 10):** every one of the owner's seven asks is now built and runnable — LLM sweep+chain (ENG-0), observe+create/mutate models & algorithms (ENG-2/3 over Features 1/2/3), autonomous data/test download + internet (ENG-2), books→vector DB + the agent storing its own work (ENG-1), the infinite closed loop that ranks winners into model nodes (ENG-3), the dashboard chat to converse like this session (ENG-4), and the whole thing recorded into the plan first (this Block). It degrades gracefully with no API keys (offline templated chat + local heuristic researcher), and lights up the free Chinese/US clouds the moment a key is set.

Rules applied (this build arc): 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25.

### Addendum 4 — owner supplied keys: live-tested all of them + installed project-local Ollama (2026-06-21)
Owner pasted 8 cloud API keys and asked to test each + install Ollama locally inside this project. Keys live in a **gitignored `.env`** (Rule 1, never committed); `tools/check_keys.py` live-tests each through `pattern_brain.llm` (doubles as live ENG-0 proof, masks keys in output).
- **Live result: 7/8 WORKING** — ✅ zai (glm-4.5-flash), qwen (qwen-plus, DashScope intl), groq (llama-3.3-70b), cerebras (gpt-oss-120b), nvidia (deepseek-v4-flash), mistral (mistral-large), openrouter (deepseek-chat). ❌ deepseek-direct = `402 Payment Required` (key valid, account has no credit — DeepSeek-family reasoning still available via the working OpenRouter + NVIDIA paths).
- **Two real fixes surfaced by testing (Rule 14/18):** (1) Groq + Cerebras returned `403 CF error 1010` — Cloudflare blocking the default `Python-urllib` User-Agent (a client-fingerprint block, NOT auth); fixed by sending a browser-like UA in `_openai_post` → both then worked. (2) NVIDIA `deepseek-ai/deepseek-v3` 404 (not hosted) → `deepseek-v4-flash`; Cerebras model not on account → `gpt-oss-120b` (its `/models` lists only `gpt-oss-120b` + `zai-glm-4.7`).
- **Project-local Ollama (separate from the system one, per the "independent project" rule):** downloaded the ollama binary into `ollama-bin/` and pulled **all three** recommended models into `ollama-models/` (gitignored, inside the folder, ~13 GB total), served on **port 11435**: **qwen2.5-coder:7b** (primary chat/reasoning, 4.7 GB, ~15 s/reply CPU), **qwen2.5-coder:14b** (higher-quality option, 9.0 GB, ~26 s/reply CPU), **nomic-embed-text** (embeddings, 274 MB, 768-d). `llm.DEFAULT_OLLAMA_MODEL` defaults to 7b but reads an **`OLLAMA_MODEL` env override** (set `OLLAMA_MODEL=qwen2.5-coder:14b` to switch, no code change — verified live). New `knowledge.OllamaEmbedder` (nomic, 768-d) with `default_embedder` preferring local Ollama → sentence-transformer → hashing. `tools/run_ollama.sh` + `tools/run_dashboard.sh` make it reproducible.
- **Live-verified end-to-end:** local qwen chat replied "OK" (~15 s CPU), nomic embeddings 768-d normalized, and the agent chat reasoned through cloud **zai** with a state-grounded answer ("expand the probability layer — underrepresented", which is true: it has the fewest nodes). This closes the earlier honesty caveat that cloud reasoning wasn't yet proven live.

### Addendum 5 — verification audit found + fixed a real gap: the agent now AUTO-ingests the books (2026-06-21)
Owner asked to re-check that the whole original message was implemented. Audit (all suites green, all files present, book-ingest pipeline live-proven: 7/8 books → 30 passages) surfaced **one genuine gap (Rule 18):** `ingest_books` existed as a tool but **nothing called it** — the agent didn't *itself* "come up with all the books and convert them into a vector DB"; the KB started empty until manually triggered. **Fixed:** `MLEngineerAgent.ensure_books()` fires once, in a background thread (non-blocking, best-effort), at the first `step()`/`chat()`; `ingest_books()` is the synchronous form; the dashboard agent calls `ensure_books()` on creation and a new `POST /api/agent/ingest_books` + an "📥 Ingest books → vector DB now" button let the owner trigger/re-run it. Offline tests pass `auto_ingest_books=False`; a new `test_auto_ingest_books` proves (via an injected fetcher) the agent converts the book corpus into the vector DB and can retrieve from it. All suites green. **Now every one of the seven asks is not just present but actually exercised by the agent on its own.**

## BLOCK 57 — Owner's two-part "all ML models + computational brain" message; read-compare-search before pausing (2026-06-21)

Owner asked to PAUSE, but first: read every word, compare to what we built, search online, capture + propose (no building now). Two message parts:
- **Part A** — a ~15-family taxonomy of "every ML model ever created" for pattern-finding / sequence prediction / probability / noise / equation discovery / generation (Pattern Mining, Markov/HMM/CRF, Bayesian/Kalman/Particle, ARIMA/GARCH/Prophet, RNN/LSTM/GRU, Transformers/PatchTST/Chronos, SSM/Mamba, clustering, anomaly, representation, GNN, RL, symbolic/SINDy/PySR, information theory, generative/GAN/diffusion) + an explicit request to add a **physics/PhD tier** (chaos prediction, "converting randomness into patterns", Galton-board/magic-of-probability-inspired models) and reach **~500–1000 ML models as nodes**.
- **Part B** — the "models as electronic components → Connector Intelligence → Architecture Search → Evolution Engine → Memory" 5-layer brain framing, ending with the key insight: *"the interesting part is NOT adding more models… the hard research problem is the connector intelligence."*

### Finding 1 (Rule 14) — Part B's "computational brain" vision IS already Pattern Brain's architecture
A near 1:1 map — this is confirmation, not new scope:
| Owner's layer (Part B) | Already built in Pattern Brain |
|---|---|
| L1 Primitive Intelligence Units ("transistors") | The model bank — **123 nodes across the 8 functional layers** (signal/noise/pattern/sequence/probability/equation/decision/rl), every one behind the generic `Node` interface (Rule 23). His Pattern/Sequence/Probability/Noise/Equation/Optimization unit groups = our layers. |
| L2 Connector Intelligence (learnable routing graph) | `connector.py` (v0 fixed pathway) + `routing.py` (v1 dynamic LLM/heuristic router) + `reputation.py` (`ReputationRouter` — routes by learned edge reputation; "the graph is the memory", Block 17). |
| L3 Architecture Search (score many graphs on profit/Sharpe/DD…) | `PathwayEvolver` (Feature 3) + the 5-layer `Evaluator` (NSGA-II Pareto over profit/Sharpe/DD/stability + CSCV/PBO). |
| L4 Evolution Engine (crossover/mutate architectures like DNA) | Features 1/2/3 GA loop (`evolution.py`) — exactly genetic programming + neuroevolution over params/algorithms/pathways. |
| L5 Memory (which architecture worked in which regime) | `knowledge.py` vector DB (RAG + Letta archival) + `PathwayReputation` regime-conditioned stats + the World-Model/regime concept. |
Owner's own conclusion ("the hard problem is the connector, not more models") is **the exact thesis the plan already records** (Blocks 7/16/17/18 independently converged on it). So Part B = ✅ already designed + built; nothing to add architecturally.

### Finding 2 (Rule 14) — Part A gap analysis: ~15 families vs the current 123-node bank
Covered / partial / missing as *built nodes* (catalog coverage in PLAN §5 is broader):
- ✅ **Strong/complete:** clustering (8+ incl. HDBSCAN/DBSCAN/GMM/spectral/meanshift), anomaly-noise (isolation_forest/one-class-svm/LOF/robust-ish PCA + AE/VAE w/ torch), classical regression/"equation" family (14), classifiers/decision (20), bandits/RL-lite (9), signal/DSP (17 incl. FFT/Hilbert/Welch/wavelython-substitutes).
- 🟡 **Partial:** sequence — have AR, markov_chain, gaussian_hmm, theta, Holt, ML-AR forecasters, + (torch) LSTM/GRU/Transformer/TCN/SSM-lite; probability — gaussian_process/kalman/particle/bayesian_ridge/gaussian_nb; symbolic — symbolic_regression + AlgorithmEvolver (GP); representation — deep/variational autoencoders; graph — gcn_denoise; generative — VAE.
- ❌ **Missing entirely (the real expansion targets):** **Pattern Mining** (Apriori/FP-Growth/ECLAT/PrefixSpan/SPADE/GSP) — whole family absent; **classical econometrics** (MA/ARMA/ARIMA/SARIMA/VAR/GARCH/EGARCH/Prophet/N-BEATS/TFT); **CRF / semi-Markov / hierarchical-HMM / higher-order Markov**; **Bayesian Networks / Dynamic Bayesian Networks**; **deep RL** (DQN/PPO/SAC/TD3/A3C); **advanced transformers** (PatchTST/Informer/Autoformer/FEDformer + foundation models TimeGPT/Chronos); **real Mamba/S4/S5**; **GNN variants** (GraphSAGE/GAT/Temporal-GNN); **generative** (GAN/Diffusion/Normalizing Flows); **information-theory nodes** (Mutual Information/Transfer Entropy/Kolmogorov/MDL); **proper SINDy/PySR/AI-Feynman**.

### Finding 3 — the physics/PhD tier owner wants (already in PLAN §5 cats 21–31; grounded by today's search)
These are documented in the plan's catalog but **not yet built as nodes** — the highest-novelty additions:
- **Reservoir Computing / Echo State Networks** — today's search confirms RC/ESN is the go-to for *chaotic* time-series + Lyapunov-exponent estimation from data; 2025 *deterministic* RC (TCRC, Logistic/Chebyshev maps, Lobachevsky activation) beats classical ESN and is low-resource — ideal "convert randomness/chaos into prediction" node. (arxiv 2501.15615; Nature Sci.Rep. 2025; AIP Chaos 27:121102.)
- **Chaos/nonlinear-dynamics**: Lyapunov exponent, phase-space (Takens) reconstruction, Recurrence Quantification Analysis — chaos-vs-noise indicators.
- **Econophysics**: Random Matrix Theory (Marchenko-Pastur correlation cleaning), **Hawkes processes** (order-flow clustering; quadratic-Hawkes → rough-volatility + Zumbach effect, arxiv 1907.06151/2206.10419), Agent-Based Models, Self-Organized Criticality, Tsallis entropy.
- **Rough volatility / fractional calculus**: rough Heston / fractional Brownian motion (volatility-is-rough).
- **Energy-Based / statistical-mechanics**: Hopfield networks (Ising lineage; = modern attention), Boltzmann/RBM, **Diffusion models** (nonequilibrium thermodynamics), Simulated Annealing.
- **Topological Data Analysis** (persistent homology — real crash early-warning hits), **Fractal/Multifractal** (Hurst, MFDFA), **PINN/Neural-ODE/DeepONet/FNO**, **Quantum/quantum-inspired** (QAOA/annealing).
- **"Magic of probability / Galton-board"-inspired**: the bean-machine = CLT (binomial→normal); the node-family it inspires = **distributional / quantile forecasters + Bayesian ensemble aggregation** (predict the *distribution*, not the point) — plus the live Galton dashboard demo already shipped.

### Conclusion (Rule 10) + what's proposed for next session (approval-gated; NOT built — owner paused)
We are **~123 of the ~500 target nodes**. The architecture (Part B) is done; the work is **breadth of L1 units** (Part A + physics tier). Proposed **Phase 7 — "Bank → ~500 nodes"**, in batches, each behind the `Node` interface, light-stack-first then torch, interlingua-conformant, shadow-first (added to PLAN tracker as 🟡). This is the natural continuation point when we resume.

**Sources:** arxiv.org/abs/2501.15615 (deterministic RC), nature.com/articles/s41598-025-98172-z, pubs.aip.org AIP Chaos 27:121102 (Lyapunov-from-data), arxiv.org/abs/1907.06151 + 2206.10419 (quadratic Hawkes → rough vol), researchgate "Volatility is rough". Earlier PLAN §5 cats 21–31 already cite the econophysics/energy-based/TDA/quantum literature.

Rules applied: 1, 2, 4, 5, 9, 10, 11, 13, 14, 16, 20, 21, 24.

---

## 2026-06-22 — Implementation: Phase 7e BATCH 3 built (continue-the-plan resume)

Owner: "read all files in the pattern brain and continue the plan implementation." Read RULES.md, README.md, PLAN.md, the code. Per Rule 22 (implement in PLAN order, no skipping), the next open item was **Phase 7e batch 3** (PLAN §11), previously tagged TODO "(needs ripser/gudhi)".

**Decision (Rule 13/16 — steelman the rejected option):** the TODO assumed persistent homology needs `ripser`/`gudhi`. Steelmanned that path: a real Rips filtration (H1 loops) genuinely does need those C libs. But for a **time series**, the natural and *exact* TDA is **0-dim sublevel-set persistence of the 1-D Morse function** — computable in pure numpy via an elder-rule union-find merge tree, no deps. So I built the dependency-free H0 version now (honest scope: H1 deferred, §0b heavy-deps-optional) rather than blocking the whole batch on an install. Same reasoning let quadratic-Hawkes / rough-vol / SOC all be pure-numpy.

**Built (4 nodes, `physics.py`, bank 155→159 light / 174→178 torch, all `signal`-conformant):**
- `quadratic_hawkes` — QARCH squared-return feedback + the **Zumbach effect** (past trend² → future vol minus the reverse = time-reversal asymmetry linear Hawkes/GARCH miss). Endogeneity/reflexivity probe.
- `rough_volatility` — Gatheral-Jaisson-Rosenbaum (2018): Hurst of the **log-vol** path via the q=2 variogram (empirically H≈0.1, "vol is rough"), with the level-series fBm Hurst for contrast.
- `soc_criticality` — Bak-Tang-Wiesenfeld self-organized criticality: threshold-excursion avalanches → Clauset power-law exponent + a scale-free score.
- `persistent_homology` — exact 0-dim sublevel-set TDA, pure-numpy elder-rule merge tree → persistence diagram, total/max persistence, normalized persistence entropy.

**Real fix surfaced (Rule 14/18):** SOC's first threshold (`median+std`) made a *smooth* low-noise series emit MORE tiny avalanches than a bursty one — count is not the SOC signature — and a `max/min` span saturated to 1.0 for both. Replaced with a fixed-rate high-quantile threshold + a `max/median` heavy-tail dispersion, so the **scale-free size distribution** is the discriminator (bursty 1.00 > smooth 0.39). This was caught by the behavioral test before marking done — exactly the Rule-25 step-4 self-check working.

**Evidence:** `tests/test_physics.py` extended (4 conformance + 4 behavioral tests); ALL 22 suites green on BOTH interpreters (light 159 / torch 178). Dashboard: nodes surface automatically in Model Bank + 📊 Coverage tabs (physics family 9→**11/14**; curated coverage now **90/118 = 76.3%**, 178 torch nodes); endpoints re-verified via the ASGI TestClient (no front-end change → no Playwright re-run). PLAN.md §11 updated, committed + pushed (Rule 24).

**Next in order (Rule 6 — one at a time):** Phase **7f batch 2** (torch — Informer/Autoformer/FEDformer, real Mamba/S4, temporal-GNN, SAC/TD3, GAN/normalizing-flows, Hopfield/Boltzmann energy nets, PINN/Neural-ODE/FNO, foundation TS) OR **7g** equation discovery (PySR/SINDy/AI-Feynman). 7f is the bigger coverage unlock (closes the Transformers/SSM/Graph/Deep-RL/Generative gaps).

Rules applied: 1, 2, 4, 5, 7, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26.

---

## 2026-06-22 — Implementation: Phase 7f BATCH 2 built (deep-tier breadth fill)

Owner: "Do 7f batch 2." Per Rule 22, this was the next open item after 7e batch 3. Built **16 pure-torch nodes** in `deep.py` (torch bank 178→**194**; light unchanged at 159 — all torch-guarded so the light bank is untouched).

**Decision (Rule 16 — steelman + §0b):** the PLAN's 7f-batch-2 TODO listed some items needing heavy external deps (real Mamba via `mamba-ssm` CUDA kernel — no GPU here; Chronos/TimesFM foundation models — large downloads). Steelmanned pulling those in now: they'd fill 2-3 coverage slots but violate §0b ("add only proven-needed deps") and Rule 1 (huge in-folder downloads) for marginal benefit on a 12-vCPU/no-GPU box. Decision: build the entire **dependency-free** subset (still 16 nodes, fills 4 whole families) and defer the dep-bound ones with an honest note + label them as a future 7f batch 3.

**Built (all behind the generic Node interface, all v0.1-conformant):**
- Advanced Transformers: `informer_forecaster` (ProbSparse top-u attention), `autoformer_forecaster` (series decomposition), `fedformer_forecaster` (frequency-enhanced decomposition).
- State-space: `s4_forecaster` (S4D complex-diagonal LTI — distinct from the selective Mamba-lite + real-diagonal `ssm`), `s5_forecaster` (MIMO diagonal).
- Operator/continuous: `fno_forecaster` (Fourier Neural Operator), `neural_ode_forecaster` (Euler-integrated latent ODE, no torchdiffeq), `pinn_forecaster` (curvature physics prior).
- Temporal-GNN: `temporal_gnn_denoise` (kNN + consecutive-time edges).
- Energy nets: `hopfield_denoise` (modern Hopfield, training-free retrieval — "attention IS Hopfield"), `rbm_denoise` (Gaussian-Bernoulli RBM, CD-1).
- Deep generative anomaly: `gan_anomaly` (discriminator realness over windows), `normalizing_flow_anomaly` (RealNVP-lite exact NLL).
- Deep-RL (discrete short/flat/long): `sac_policy` (entropy-regularized AC), `td3_policy` (twin clipped double-Q), `a3c_policy` (advantage AC with learned value baseline).

**Two behavior-preserving base-helper edits** (so the new nodes reuse tested machinery cleanly): an `_extra_loss` hook on `_TorchForecaster` (default 0.0; PINN overrides it) and an `adj_fn` param on `_graph_reconstruct` (default `_knn_adj`; Temporal-GNN passes a temporal adjacency).

**Evidence:** `tests/test_deep7f2.py` (16 conformance + multi-feature D=3 + denoiser/anomaly/RL behavioral + an AR(1) tracking check) green; **all 23 suites green on BOTH interpreters** (light 159 / torch 194). Coverage **76.3%→87.3% (103/118)**: State-space, Graph, Reinforcement-learning, and Generative families are now **COMPLETE**; Transformers 5/7, Physics 13/14 (only Quantum/QAOA). Light bank verified unaffected (test_deep7f2 skips cleanly without torch). PLAN.md updated; committed + pushed (Rule 24).

**Deferred honestly (§0b):** real Mamba/S4 CUDA kernels, foundation TS (Chronos/TimesFM/TimeGPT), self-supervised representation tier (SimCLR/BYOL/MAE) — future 7f batch 3.

**Next in order (Rule 6):** **Phase 7g** — equation/symbolic discovery (PySR/SINDy/AI-Feynman/genetic programming), the last 7-series batch, which also feeds the AlgorithmEvolver (Feature 2). Symbolic-discovery is currently the lowest-coverage family (2/6).

Rules applied: 1, 2, 4, 5, 7, 13, 16, 18, 20, 21, 22, 23, 24, 25, 26.

---

## 2026-06-22 — Implementation: Phase 7g built (equation / symbolic DISCOVERY) — 7-series complete

Owner: "Phase 7g and always read ideas generated in the last message." Applied idea #1 from the 7f-batch-2 ideas block verbatim: split 7g light-vs-dep — build the **pure-numpy** discovery engines now, defer the Julia/heavy ones. Per Rule 22 this was the last open 7-series batch.

**Built (2 nodes, appended to `nodes/equation.py`, bank 159→161 light / 194→196 torch, both emit the `equation` contract coef+r2):**
- `sindy_regression` — **SINDy** (Brunton-Proctor-Kutz, PNAS 2016): regress the standardized state's time-derivative onto a library [1, x, x², x³, sin x, cos x], then **sequentially threshold (STLSQ)** to a sparse governing ODE. Recovers the actual dynamical law, not a fitted curve.
- `genetic_symbolic_regression` — **GP-SR**: evolve vectorized expression trees over the normalized time index (tournament selection + subtree crossover/mutation + parsimony penalty), least-squares-fit y≈a·φ(t)+b. Dependency-free counterpart to PySR/gplearn.

**Relationship clarified (PLAN said "+ AlgorithmEvolver"):** these are the fast BANK-NODE form; the existing `AlgorithmEvolver` (Feature 2) is the evolutionary, Evaluator-gated form — same family, complementary, both now present.

**One bug caught + fixed before done (Rule 18):** my GP helper was named `_fit`, which shadowed the `Node._fit` training hook → `predict()` dispatched into it with wrong args. Renamed to `_score_tree`. The conformance test caught it immediately.

**Evidence:** `tests/test_equation_discovery.py` green — SINDy recovers **dx/dt≈−0.33·x** from a −0.3 exponential-decay ODE (1 active term, r2 0.97) vs random-walk r2 0.00; GP explains a structured quadratic (r2 1.00) far above noise (0.03) with a closed form using `t`; GP **deterministic** under a fixed seed; domain-independence clean. **All 24 suites green on BOTH interpreters** (light 161 / torch 196). Coverage **87.3%→89.0% (105/118)**; Symbolic/equation-discovery family **2/6→4/6**. PLAN.md + coverage line updated; committed + pushed.

**Deferred honestly (§0b, deps):** PySR (needs Julia), AI-Feynman (heavy).

**State: the whole 7-series (7a→7g) is now built**, dependency-free where possible, everything dep-bound (real Mamba, foundation TS, SSL, PySR, AI-Feynman, QAOA) deferred with reasons. The bank is 161 light / 196 torch; 12 of 17 curated families complete; coverage 89%.

**Next (Rule 6 — one at a time):** owner's call. Highest-leverage non-build move = run the `MLEngineerAgent` loop over the now-196-node bank so the Evaluator gate surfaces which of the new deep/physics/equation nodes actually EARN shadow status (breadth is climbing; the owner's own caveat was "breadth isn't the hard part — the connector is"). Build-wise: the easy Neural-sequence mop-up (rnn/bilstm/seq2seq) or a 7f-b3 self-supervised tier (SimCLR/BYOL/MAE).

Rules applied: 1, 2, 4, 5, 7, 18, 20, 21, 22, 23, 24, 25, 26.

---

## 2026-06-22 — DISCUSSION (planning): connecting the 196 nodes into one network + accuracy-vs-real-data loop

Owner opened a planning session (NO implementation this turn): "now we have all the nodes; we need to think how to connect them as a mess of network, how to pass the raw data input into this network, and compare the output to the real data to measure the accuracy. This is a discussion session — I will pass starting ideas/conversations next for you to get inspired and decide the plan and features."

### Claude's evidence-based read (Rule 26 step 1) — MOST of this is already built (coherence intelligence)
The owner's three asks map almost 1:1 onto code that already exists, but at SINGLE-PATHWAY scale, not whole-bank-network scale:
- "connect them as a network" → `connector.py` (Connector v0, fixed pathway + signal-bus threading + belief stream) and `routing.py` (Connector v1 — dynamic hop-by-hop Router/HeuristicRouter/LLMRouter/RoutedConnector). The belief INTERLINGUA (`interlingua.py`) is the typed edge contract.
- "pass raw data in" → `adapters/` (raw OHLCV → generic (T,D)); the core consumes (T,D) (Rule 23).
- "compare output to real data / measure accuracy" → `evaluator.py` (5-layer: purged walk-forward + PSR/DSR/minTRL + discounted-UCB + NSGA-II/CSCV-PBO + anchors) — this IS the accuracy meter. `reputation.py` (PathwayReputation, discounted-UCB) is the feedback. `evolution.py::PathwayEvolver` already evolves pathway topology gated by the Evaluator.

**The real GAP** = going from "pick/score ONE pathway" → "the WHOLE 196-node bank as one interconnected, accuracy-trained network (the owner's 'mess')." That is exactly the Connector Intelligence v2 the PLAN deferred (MoE/GNN routing) because it needed routing-outcome training data that didn't exist — which we can NOW generate by running + scoring the bank.

### The design space (generate → critique → rank), all kept domain-agnostic (Rule 23)
- **A — Layered STACKED model-graph (stacked generalization, Wolpert 1992, modernized).** All nodes run on the (T,D) bus + upstream beliefs, organized by the 8 functional layers; a trained meta-combiner (decision layer) fuses everything into the one-step prediction; accuracy = walk-forward (Evaluator). *Critique:* buildable NOW from existing parts; the canonical pitfall is garbage/redundant nodes diluting signal + overfitting from 196 inputs → MUST be accuracy-weighted/sparsified and purged-WF gated (the Evaluator's DSR/CSCV exist precisely for this). Strongest first step because it GENERATES the outcome data B and C need.
- **B — MoE / learned gating router (Connector v2).** A gate routes each input (per regime) to a sparse top-k subset of expert nodes; gate trained on the outcome data A produces. *Critique:* the principled long-term answer (2025 SOTA: HMoE, Time-MoE, N-BEATS-MoE, expert-choice routing), efficient (only relevant experts fire), but needs the bootstrap data → builds ON A, not before it.
- **C — Evolved DAG topology (NAS over the model graph).** Generalize PathwayEvolver from a chain to a multi-branch DAG; Evaluator = fitness. *Critique:* machinery exists; full-dense-DAG search space is huge + overfit-prone → best as a topology-search layer ON TOP of A/B, phase 3, with CSCV/PBO guards.
- **D — Differentiable model-graph (DARTS over the bank).** *Critique:* REJECT for the whole bank — most nodes are sklearn/numpy, non-differentiable, so backprop-through-graph is impossible; only viable on the torch sub-graph later. Feasibility killer in general.

**Ranking (pairwise) for the FIRST build: A > B > C > D.** A is the substrate everything else needs.

### The pivotal decision that shapes EVERYTHING (Rule 13): define the OUTPUT/target
"Compare output to real data" requires a defined target. Options: (1) one-step next-value/return forecast (point), (2) directional up/down (classification), (3) full predictive distribution (calibrated), (4) a trade decision / P&L. Claude's recommendation (Rule 10): start with **one-step-ahead return of feature-0, scored on BOTH directional hit-rate AND calibrated error**, because it's an unambiguous output-vs-reality signal, domain-agnostic (Rule 23, works on any (T,D)), and the Evaluator already does walk-forward outcome scoring. Distribution/P&L come later.

### Claude's recommended phasing (PROPOSED, not decided — awaiting owner's seed ideas)
- Phase 8 = "The Network" substrate: layered stacked accuracy-scored model-graph (A) — reuses connector/reputation/evaluator/adapter, does NOT rebuild them.
- Phase 9 = MoE gate (Connector v2, option B) trained on Phase-8 outcome data.
- Phase 10 = evolved DAG topology (C) on top.

### Open questions parked for the owner (Rule 15 — ask, don't guess)
1. Target/output definition (above) — the #1 fork.
2. Dense (all nodes fire every step) vs sparse (gated top-k)? Compute vs coverage.
3. Static vs learned/evolved topology — and is "the mess" literally all-to-all, or layered+gated?
4. Accuracy metric: point error (RMSE) vs directional hit-rate vs calibration (Brier/logloss) vs trading P&L.

Owner will feed seed ideas/conversations next; Claude to synthesize those with this read before anything is fixed in PLAN.md (Rule 21 — nothing marked decided yet).

Rules applied: 1, 2, 4, 5, 9, 10, 11, 13, 14, 16, 20, 21, 23, 26.

---

## 2026-06-22 — DISCUSSION (cont.): owner pasted an external "ideas chat"; synthesis into Phase 8 plan

Owner pasted a long external AI conversation (AutoML / stacked ensembles / NAS / search algos / LLM-agent-as-planner / vector-DB vs feature-store) and asked: "the spheres become ML models — how do I tell my coding agent to make hundreds of models connect like this, try all combinations on input data, test the output, and score which network scores high. Read every word, get inspired, decide the plan and features."

### The pasted advice, summarized (its throughline)
- Category = **AutoML + stacked ensembles + graph/NAS search**. Tools floated: AutoGluon (baseline), auto-sklearn, TPOT, Optuna, NetworkX, MLflow/SQLite, joblib.
- **Don't brute-force combinations** — use SEARCH: random → evolutionary (mutate add/remove/rewire/change-meta) → Bayesian/Optuna → RL (later) → NAS.
- Workflow: train base models → **out-of-fold predictions** → treat each model as a graph node → search DAG combinations → train meta-models per node → score on validation → **leaderboard** → mutate top → export best pipeline.
- **LLM agent = controller/planner, NOT the scorer** (suggests combos, reads leaderboard, proposes mutations, checks leakage, writes configs; accuracy comes ONLY from validation data).
- Vector DB ≠ raw-data→model-input converter; it's memory/retrieval (RAG). For inputs use a **Feature Factory → Feature Store** (RSI/MACD/returns/vol/lags/regime…), automated FE (Featuretools/tsfresh), embeddings for unstructured data. Brain arch: raw → input router → per-modality encoders → shared feature space → 100-500 specialist models → graph search → meta models → output.
- Headline phrase: **"Evolutionary search over stacked ensemble DAGs with an LLM experiment-planning agent."**

### Claude's evidence-based read (Rule 26) — what we ALREADY have vs the gap
That headline IS this project's destiny, and ~70% exists:
- evolutionary DAG search → `evolution.py::PathwayEvolver` (evolves pathway topology, Evaluator-gated)
- score/validation → `evaluator.py` (purged walk-forward + DSR + CSCV/PBO) — our accuracy meter
- leaderboard-ish feedback → `reputation.py` (discounted-UCB)
- LLM planner not scorer → `agent/engineer.py` (MLEngineerAgent) + the producer-door/Evaluator-gate pattern (LLM proposes, Evaluator decides) — EXACTLY their rule
- vector DB for memory → `knowledge.py` (already present, used for agent RAG/memory ONLY — their take is correct)
- feature generation → much of the signal/noise/infotheory/equation layers ARE feature generators; frac-diff lives in the adapter
- raw→(T,D) input → `adapters/`

**The GAP (what to actually build) = the connective tissue to lift single-pathway → whole-bank DAG search:**
1. OOF (out-of-fold) prediction harness on the EXISTING purged-walk-forward splits → leakage-safe per-node prediction cache.
2. Feature Factory: generic features in CORE (lags/returns/rolling/frac-diff/spectral/entropy — Rule 23), stock-specific (RSI/MACD/VWAP/orderbook) in the ADAPTER; cached in a simple in-folder Feature Store.
3. Stacked-DAG object: generalize the single pathway to a multi-layer DAG (base nodes → meta-nodes consuming OOF beliefs → final combiner); a DAG executor.
4. Persistent LEADERBOARD (stdlib sqlite, in-folder Rule 1) of every (DAG config → walk-forward score) + a dashboard tab.
5. Search engine with pluggable strategies: random → evolutionary (generalize PathwayEvolver) → Bayesian (Optuna, optional light dep).
6. LLM experiment-planner: wire MLEngineerAgent to read the leaderboard + propose next DAG mutations (controller, never scorer).
7. Accuracy target/metric defined (the search objective).

### Claude's THREE expert corrections (value beyond the pasted advice)
- **(A) Do NOT add AutoGluon/auto-sklearn/TPOT as components.** They're tabular-AutoML — they'd violate Rule 23 (data-shaped around tables), duplicate our bank, and add heavy deps. Adopt their PATTERNS (multi-layer stacking, OOF, leaderboard), not the tools. (Optional: run AutoGluon ONCE offline as an external benchmark number to beat.) Steelman (Rule 16): AutoGluon gives a fast strong baseline — fair, so use it as an *external yardstick*, not a built-in.
- **(B) THE dominant risk the pasted plan underweights = overfitting / multiple-testing.** "Search thousands of DAGs, keep the best validation score" is selection-on-noise — it WILL look great in backtest and die live (the entire trading-bot memory is this bug class). Our unfair advantage: the Evaluator already has **Deflated Sharpe + CSCV/PBO** that deflate for the number of configs tried. The plan MUST pick the "best" by DSR/PBO-adjusted score over the count of DAGs searched, never raw validation. This is the single most important correction.
- **(C) Feature factory must obey Rule 23:** generic features in core, stock features in the adapter — never bake candle/orderbook features into the core feature layer.

### Dep verdicts (Rule 11 / §0b — light by default)
- AutoGluon/auto-sklearn/TPOT: ❌ as components (optional external benchmark only). Optuna: 🟢 optional, light, pure-python — adds Bayesian/TPE search we lack. NetworkX: 🟡 optional (viz only; own light DAG repr otherwise). MLflow: ❌ (heavy) → stdlib **sqlite** leaderboard. Featuretools/tsfresh/Feast/Hopsworks: ❌ deps → light in-folder feature factory + cache. Vector DB: ✅ already present (knowledge.py), keep for memory/RAG only.

### Proposed name + phasing (PROPOSED 🟡, awaiting owner's fork answers before PLAN.md)
**Phase 8 — "The Network": AutoML Stacked-DAG Search** (= their headline, built on our bank+Evaluator+agent). Sub-order = the 7-item gap list above. Phase 9 = MoE learned gate (Connector v2) trained on Phase-8 leaderboard data. Phase 10 = evolved/RL topology on top.

### Forks to lock with the owner (Rule 13/15) — asked via AskUserQuestion this turn
1. Prediction target + score (one-step return + directional hit-rate [rec] / full distribution / trade-P&L).
2. Scope now (numeric time-series first, multimodal later [rec] / build multimodal "any raw data" brain now).
3. New light deps (allow Optuna + stdlib sqlite [rec] / zero new deps, implement search ourselves).

Nothing marked decided in PLAN.md yet (Rule 21). Rules applied: 1, 2, 4, 5, 9, 10, 11, 13, 14, 16, 20, 21, 23, 26.

---

## 2026-06-22 — Phase 8 forks DECIDED + plan recorded (after online research)

Owner: "understand the message, search online, then plan." Claude ran 2026 web research (AutoGluon multi-layer stacking/OOF; LLM-agent AutoML — AIDE/MLE-STAR/AutoML-Agent/ArchPilot; López de Prado CPCV/DSR/PBO anti-overfit; Optuna/NAS best practice) — all confirm the design and our unfair advantage (the Evaluator already has the anti-overfit machinery generic AutoML lacks).

**Owner answered the three forks (AskUserQuestion):**
1. Target/score = **FULL PREDICTIVE DISTRIBUTION** (proper scoring: CRPS/log-loss/Brier + PIT calibration), not point/direction. → fits our distributional nodes.
2. Scope = **MULTIMODAL "any raw data" NOW** (the full brain diagram: per-modality encoders → shared space). → honored via the encoder-adapter pattern (Rule 23: encoders in adapters/), staged so the core DAG-search runs on numeric+text first and image/audio encoders fill in later.
3. Deps = **Optuna + stdlib sqlite** approved.

**Plan recorded in PLAN.md §12** ("Phase 8 — THE NETWORK: Multimodal AutoML Stacked-DAG Search w/ distributional scoring") with the 8-slice build order (scoring → encoders → feature factory → OOF harness → stacked-DAG → leaderboard → search → LLM planner), the 3 expert corrections (no AutoGluon-as-component; overfitting is THE risk and CPCV/DSR/PBO is the cure; Rule-23 feature split), dep verdicts, and concrete file targets. Status: forks DECIDED, build PLANNED — awaiting owner's go-ahead to start slice 1 (Rule 26 gate; this was a planning session, no code built this turn).

Rules applied: 1, 2, 4, 5, 9, 10, 11, 12, 13, 15, 16, 20, 21, 22, 23, 26.

---

## 2026-06-22 — Capstone defined: single-symbol crypto FROZEN forward-test (final whole-project acceptance)

Owner: the FINAL verification of the entire project = take one crypto symbol, gather ALL available data types, feed the project's network, compare output vs real live movement, try different network combinations, see which predicts the next future move most accurately. Crucially: "ask the LOCAL or CLOUD LLM to get data to the ML models, not you [Claude] chat with that project's LLMs." → the project's own MLEngineerAgent (llm.py local Ollama / cloud chain) orchestrates data acquisition via its tools; Claude does not hand-feed data. Then: "search the project, plan how to do it effectively, add it to the plan, think + research online."

Searched project (Rule 14): adapters/stock.py already has align_sources/align_to_grid (causal multi-source merge → (T,D)) — the exact primitive for "all data types"; agent/tools.py has web_fetch/download_file + injectable fetch. MISSING: crypto fetcher/adapter + LLM acquisition task. No ccxt/binance yet.

Researched (2026): free single-symbol all-market-data path = Binance/Bybit/Kraken public REST (klines multi-TF, depth, trades, funding, OI, liquidations); on-chain/social mostly paid. Frozen-parameter out-of-sample standard: fix all data-estimated params before the test window, recursive forecasts, ≥30% reserved, walk-forward (blockchain-council; arXiv 2511.08571; PMC crypto-forecast studies).

THE design subtlety flagged (Rule 26): you must NOT select the best network on the live/test data (selection-on-noise). Protocol = SEARCH/SELECT on history (DSR/CSCV-gated) → FREEZE the single best net → FORWARD-TEST the frozen net on unseen/live data, hundreds of one-step predictions, proper scoring (CRPS/log-loss/PIT) + hit-rate vs naive baseline, DSR-significant. Only the frozen forward-test = the project's accuracy. A failure is an honest valid outcome.

LLM-acquisition design: adapters/crypto.py (deterministic fetchers, Rule 23) + fetch_crypto_symbol tool in agent/tools.py; MLEngineerAgent runs an acquisition task via llm.py (local/cloud) calling those tools; LLM = acquisition orchestrator, never scorer.

Recorded as PLAN.md §12.1 (Capstone = Phase-8 slice 9: adapters/crypto.py + fetch tool + capstone.py freeze/forward-test harness + dashboard Capstone tab). Open dep fork: ccxt (robust, handles US geo-block on Binance.com) vs urllib-only. Status: PLANNED, no code built (planning session). Awaiting owner: dep choice + go-ahead to start building Phase-8 slice 1.

Rules applied: 1, 2, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 26.
