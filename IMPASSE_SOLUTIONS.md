# Impasse → Solution Library — Pattern Brain
_Rule 31: every time an idea looks impossible to implement, search for how others solved it, borrow the
technique, and record it here so the same wall is never hit twice._

Format per entry: **Impasse** · **Who solved it / source** · **Technique** · **Did it unblock us?**

---

### IMP-001 · "Backprop like an LLM through a graph of heterogeneous ML models"
**Impasse:** experts (HMM, Kalman, HDBSCAN, PySR, GARCH) are not differentiable, so true end-to-end
gradient backprop through the whole model-graph looked impossible. Claude initially called it "not literal."

**Who solved it / sources:**
- Differentiable Kalman filters — akloss/differentiable_filters ("How to Train Your Differentiable Filter"),
  ymchen0/torchEnKF, tiboat/BackpropKF_Reproduction.
- Differentiable HMM — 30stomercury/hmm-backprop (differentiable forward-backward).
- Discrete/non-diff ops — Gumbel-Softmax (Jang et al.), Straight-Through Estimator, REINFORCE/score-function.
- Whole-system autodiff — JAX MD (backprop through entire classical simulations).
- Distillation — distill a non-diff model into a differentiable student.

**Technique (how to apply here):**
1. Reimplement experts that HAVE differentiable forms (Kalman, HMM, soft trees, DSP) in a diff framework
   (PyTorch/JAX) → they become true gradient neurons.
2. Make routing differentiable with **Gumbel-Softmax** (soft top-k expert selection) + **STE** for the
   argmax in the forward pass.
3. Wrap genuinely black-box experts (HDBSCAN, PySR) with **REINFORCE** gradient estimates, or **distill**
   them into a differentiable surrogate.
4. Keep **reputation/evolution** for structure (grow/prune) — gradients handle weights, evolution handles topology.

**Did it unblock us?** YES — verdict upgraded from "largely impossible" to "~80–90% achievable with
differentiable reimplementations + estimators." Staged adoption recommended (cost/variance/overfit caveats).
Feeds vision-doc gaps #1–#3 and ideas IDEA-015/016/017.

### IMP-002 · "BOCPD change-point probability comes out flat (always the hazard rate)"
**Impasse:** building `bocpd_break`, the textbook "change-point probability" P(run length=0 | data)
came out identically equal to the hazard rate at every step — a useless flat detector. Looked like
the implementation was broken.

**Who solved it / source:** Adams & MacKay (2007) "Bayesian Online Changepoint Detection"; the
run-length-posterior recursion + standard practitioner detectors (the bayesianchangepoint /
hildensia implementations plot the run-length posterior, not P(r=0)).

**Technique:** it is NOT a bug — it is the BOCPD normalisation identity: the changepoint mass is
H·Σ and the evidence normaliser is also Σ (growth (1−H)·Σ + change H·Σ), so P(rₜ=0)≡H always. The
real change-point signal is the **collapse of the run-length posterior**: track the expected run
length E[rₜ]=Σ r·R[t,r]; it grows ~linearly while a regime persists and drops sharply at a break.
Use the (clamped, normalised) **drop in E[rₜ]** as the per-step change-point score.

**Did it unblock us?** YES — switched the node's score to the fractional drop in E[rₜ]; it now
localizes a clean mean shift (oracle) and a real spliced regime break (t=300→detected t=305, ±5).
