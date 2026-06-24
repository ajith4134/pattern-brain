"""Reconcile CONCEPT_EQUATION_BANK.md against the live model bank (RULES Rule 30 support).

Adds a STATUS marker to every concept bullet so the bank becomes a live build-tracker
instead of a static menu. Re-run after adding nodes to refresh the column + tallies.

Status legend:
  [B] built as a registered Node          (in the bank, callable now)
  [M] built as a module/utility           (capability exists, not a registered Node)
  [ ] not built                           (an open model candidate)
  [F] foundational / no own node          (an underpinning concept, not meant to be a node)

The status of each concept is keyed by the candidate node name in its `-> `node`` tag
(hand-authored below from a registry diff on 2026-06-23). Edit STATUS to correct a call.
Usage:  PYTHONPATH=. python3 tools/reconcile_concept_bank.py        # rewrites the file in place
        PYTHONPATH=. python3 tools/reconcile_concept_bank.py --check # tally only, no write
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "CONCEPT_EQUATION_BANK.md"

# --- hand-authored build status per candidate node name (registry diff 2026-06-23) ---
# B = registered Node ; M = module/utility ; N = not built ; F = foundational/no-node
STATUS: dict[str, str] = {
    # D1 statistical/econometric
    "arfima_forecast": "B", "vecm_pairs": "B", "johansen_coint": "N", "egarch_vol": "B",
    "sv_particle": "N", "markov_switching": "B", "kalman_state": "B", "copula_dependence": "N",
    "evt_tail_risk": "N", "quantile_band": "B", "hawkes_intensity": "B", "hurst_dfa": "B",
    "bocpd_break": "N", "har_rv": "N",
    # D2 probabilistic/Bayesian
    "bayes_update": "N", "gp_forecast": "B", "hmm_regime": "B", "dp_mixture": "B",
    "vae_latent": "B", "bsts_forecast": "N", "smc_state": "B",
    # D3 information-theoretic
    "entropy_regime": "B", "mi_screen": "B", "transfer_entropy": "B", "mdl_compress": "B",
    "perm_entropy": "B",
    # D4 physics/SDE
    "gbm_baseline": "N", "ou_meanrevert": "B", "jump_diffusion": "N", "heston_vol": "N",
    "rough_vol": "B", "fokker_planck": "N", "langevin_sampler": "N", "ising_herding": "N",
    "soc_avalanche": "B", "percolation_risk": "N", "superstat_vol": "N",
    # D5 energy-based
    "maxent_dist": "N", "rbm_energy": "B", "active_inference": "N",
    # D6 geometric/topological
    "tda_persistence": "B", "infogeom_distance": "N", "wasserstein_shift": "N",
    "diffusion_map": "N", "path_signature": "M", "tensor_network": "N",
    # D7 dynamical/chaos
    "lyapunov_chaos": "B", "takens_embed": "B", "koopman_dmd": "N", "reservoir_esn": "B",
    "multifractal": "B", "rqa_determinism": "B",
    # D8 causal
    "granger_cause": "M", "ccm_cause": "N", "scm_intervene": "N", "pcmci_graph": "N",
    # D9 game-theoretic
    "kyle_impact": "N", "adversarial_robust": "N", "mfg_crowding": "N",
    # D10 control
    "hjb_control": "N", "merton_alloc": "N", "almgren_exec": "N", "mpc_position": "N",
    "rl_policy": "B",
    # D11 optimization/OR
    "markowitz": "M", "kelly_size": "M", "hrp_alloc": "N", "cvar_opt": "N", "bayes_opt": "M",
    # D12 bio-inspired
    "deep_seq": "B", "patchtst": "B", "mamba_ssm": "B", "tsfm_zeroshot": "N",
    # D13 symbolic
    "symbolic_regression": "B",
    # D14 soft computing
    "fuzzy_ts": "N", "grey_gm11": "N", "anfis": "N",
    # D15 quantum
    "quantum_kernel": "N", "qubo_portfolio": "N",
    # D16 linguistic/neuro-symbolic
    "news_sentiment": "N", "neuro_symbolic": "N",
    # D-signal
    "fft_cycles": "B", "wavelet": "B", "emd": "B", "ssa": "B", "hilbert_phase": "B",
    # microstructure
    "ofi": "N", "vpin": "N", "as_market_make": "N", "lob_hawkes": "N", "rough_hawkes_heston": "N",
    # ---- TIER-2 frontier (mostly deferred) ----
    "sig_kernel": "N", "neural_cde": "N", "neural_sde": "N", "deep_bsde": "N",
    "diffusion_synth": "N", "schrodinger_bridge": "N", "rmt_clean": "B", "rie_covariance": "M",
    "tracy_widom_test": "N", "spt_diversity": "N", "entropy_portfolio": "N", "ldp_rate": "N",
    "instanton_crash": "N", "wasserstein_dro": "N", "conformal_interval": "B",
    "distributional_rl": "N", "sinkhorn_dist": "N", "ntk_analysis": "N", "rg_multiscale": "N",
    "pid_synergy": "N", "predictive_info": "N", "neural_hawkes": "N", "marked_hawkes": "N",
    "optimal_stopping": "N", "deep_stopping": "N", "tda_zigzag": "N", "sheaf_consistency": "N",
    "hyperbolic_embed": "N", "nonlinear_filter": "N", "meanfield_control": "N", "hjb_isaacs": "N",
    "q_amplitude_est": "N", "q_reservoir": "N", "universal_portfolio": "N",
    "online_no_regret": "M", "bandit_alloc": "B", "pcmci_plus": "N", "irm_invariant": "N",
    # ---- TIER-3 timeless foundations ----
    "log_utility": "N", "momentum_kinematics": "N", "lagrange_opt": "N", "chebyshev_bound": "N",
    "diffusion": "F", "wiener_filter": "B", "ar_forecast": "B", "bs_pricing": "N",
}
MARK = {"B": "✅", "M": "🔬", "N": "⬜", "F": "▫️"}
LABEL = {"B": "built (Node)", "M": "built (module)", "N": "NOT built", "F": "foundational"}

NODE_RE = re.compile(r"→ `([a-z0-9_]+)`")
BULLET_RE = re.compile(r"^(\s*[-*] )(\*\*)")
EXISTING_MARK_RE = re.compile(r"^(\s*[-*] )(✅|🔬|⬜|▫️) ")


def classify(line: str) -> str | None:
    """Return a status code for a concept bullet, or None if not a concept bullet."""
    if not BULLET_RE.match(line):
        return None
    m = NODE_RE.search(line)
    if m and m.group(1) in STATUS:
        return STATUS[m.group(1)]
    if "(have" in line:           # author already marked it built
        return "B"
    return "F"                     # a concept with no node candidate → foundational/no-own-node


def process(text: str) -> tuple[str, dict[str, int]]:
    out, tally = [], {k: 0 for k in MARK}
    in_screened = False
    for line in text.splitlines():
        if line.startswith("## ⛔") or line.startswith("## Sources"):
            in_screened = True
        elif line.startswith("#") and not line.startswith("## ⛔"):
            in_screened = line.startswith("## Sources")
        line = EXISTING_MARK_RE.sub(r"\1", line)   # strip a prior run's marker (idempotent)
        code = None if in_screened else classify(line)
        if code:
            tally[code] += 1
            line = BULLET_RE.sub(rf"\g<1>{MARK[code]} \g<2>", line, count=1)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), tally


def main() -> None:
    text = DOC.read_text()
    new, tally = process(text)
    total = sum(tally.values())
    summary = (f"{tally['B']} built-Node ✅ · {tally['M']} built-module 🔬 · "
               f"{tally['N']} not-built ⬜ · {tally['F']} foundational ▫️  (of {total} concepts)")
    print("STATUS TALLY:", summary)
    for k in ("B", "M", "N", "F"):
        print(f"  {MARK[k]} {LABEL[k]:16s} {tally[k]}")
    if "--check" in sys.argv:
        return
    DOC.write_text(new)
    print(f"\nwrote status column to {DOC.name}")


if __name__ == "__main__":
    main()
