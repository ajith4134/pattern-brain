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
    "arfima_forecast": "B", "vecm_pairs": "B", "johansen_coint": "B", "egarch_vol": "B",
    "sv_particle": "N", "markov_switching": "B", "kalman_state": "B", "copula_dependence": "B",
    "evt_tail_risk": "B", "quantile_band": "B", "hawkes_intensity": "B", "hurst_dfa": "B",
    "bocpd_break": "B", "har_rv": "B",
    # D2 probabilistic/Bayesian  (TIER-3 build 2026-06-23: bayes_update now ✅)
    "bayes_update": "B", "gp_forecast": "B", "hmm_regime": "B", "dp_mixture": "B",
    "vae_latent": "B", "bsts_forecast": "N", "smc_state": "B",
    # D3 information-theoretic
    "entropy_regime": "B", "mi_screen": "B", "transfer_entropy": "B", "mdl_compress": "B",
    "perm_entropy": "B",
    # D4 physics/SDE  (TIER-3 build: gbm_baseline, langevin_sampler, fokker_planck now ✅)
    "gbm_baseline": "B", "ou_meanrevert": "B", "jump_diffusion": "N", "heston_vol": "N",
    "rough_vol": "B", "fokker_planck": "B", "langevin_sampler": "B", "ising_herding": "N",
    "soc_avalanche": "B", "percolation_risk": "N", "superstat_vol": "N",
    # D5 energy-based  (TIER-3 build: maxent_dist now ✅)
    "maxent_dist": "B", "rbm_energy": "B", "active_inference": "N",
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
    # D10 control  (TIER-3 build: hjb_control [Bellman/HJB], almgren_exec now ✅)
    "hjb_control": "B", "merton_alloc": "N", "almgren_exec": "B", "mpc_position": "N",
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
    # ---- TIER-3 timeless foundations  (TIER-3 build 2026-06-23: 5 below now ✅) ----
    "log_utility": "B", "momentum_kinematics": "B", "lagrange_opt": "B", "chebyshev_bound": "B",
    "diffusion": "F", "wiener_filter": "B", "ar_forecast": "B", "bs_pricing": "B",
}
MARK = {"B": "✅", "M": "🔬", "N": "⬜", "F": "▫️"}
LABEL = {"B": "built (Node)", "M": "built (module)", "N": "NOT built", "F": "foundational"}

NODE_RE = re.compile(r"→ `([a-z0-9_]+)`")
BULLET_RE = re.compile(r"^(\s*[-*] )(\*\*)")
EXISTING_MARK_RE = re.compile(r"^(\s*[-*] )(✅|🔬|⬜|▫️) ")


def classify(line: str) -> str | None:
    """Return a status code for a concept bullet, or None if not a concept bullet.

    A line may list several candidate nodes (e.g. `ising_herding`, `maxent_dist`); it counts
    as built if ANY of them exists. Priority best→worst: B > M > N > F.
    """
    if not BULLET_RE.match(line):
        return None
    tail = line.split("→", 1)[1] if "→" in line else ""   # all `node` names listed after the arrow
    names = re.findall(r"`([a-z0-9_]+)`", tail)
    codes = [STATUS[n] for n in names if n in STATUS]
    if codes:
        for best in ("B", "M", "N", "F"):
            if best in codes:
                return best
    if "(have" in line:           # author already marked it built
        return "B"
    return "F"                     # a concept with no node candidate → foundational/no-own-node


def process(text: str) -> tuple[str, dict[str, int]]:
    out, tally = [], {k: 0 for k in MARK}
    in_screened, started = False, False
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("## ⛔") and not line.startswith("## Sources"):
            started = True                          # first real concept section reached (skip the legend)
        if line.startswith("## ⛔") or line.startswith("## Sources"):
            in_screened = True
        elif line.startswith("#") and not line.startswith("## ⛔"):
            in_screened = line.startswith("## Sources")
        line = EXISTING_MARK_RE.sub(r"\1", line)   # strip a prior run's marker (idempotent)
        code = None if (in_screened or not started) else classify(line)
        if code:
            tally[code] += 1
            line = BULLET_RE.sub(rf"\g<1>{MARK[code]} \g<2>", line, count=1)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), tally


def summarize() -> dict:
    """Parse the already-marked doc into a dashboard payload (the doc is source of truth).

    Returns per-tier tallies + the flat 'to_implement' list (the ⬜ entries). Safe to call
    from the dashboard server; reads the committed markers, does not recompute or write.
    """
    text = DOC.read_text()
    mark_to_code = {v: k for k, v in MARK.items()}
    tiers = {"TIER-1": "TIER-1 — PhD / cutting-edge core",
             "TIER-2": "TIER-2 — research frontier (⭐)",
             "TIER-3": "TIER-3 — timeless foundations"}
    per_tier: dict[str, dict] = {t: {"name": n, "B": 0, "M": 0, "N": 0, "F": 0} for t, n in tiers.items()}
    tally = {k: 0 for k in MARK}
    to_implement: list[dict] = []
    tier, section = "TIER-1", ""
    bullet_mark = re.compile(r"^\s*[-*] (✅|🔬|⬜|▫️) \*\*(.+?)\*\*")  # .+? so titles w/ '*' (Kelly f*) match
    for line in text.splitlines():
        if line.startswith("# ===== TIER-2"):
            tier = "TIER-2"
        elif line.startswith("# ===== TIER-3"):
            tier = "TIER-3"
        elif line.startswith("## ") and not line.startswith("## ⛔") and not line.startswith("## Sources"):
            section = line[3:].strip()
        m = bullet_mark.match(line)
        if not m:
            continue
        code = mark_to_code[m.group(1)]
        tally[code] += 1
        per_tier[tier][code] += 1
        if code == "N":
            to_implement.append({"name": m.group(2).strip(), "tier": tier, "section": section,
                                 "frontier": "⭐" in line})
    buildable = tally["B"] + tally["M"] + tally["N"]
    done = tally["B"] + tally["M"]
    return {
        "total": sum(tally.values()),
        "built_node": tally["B"], "built_module": tally["M"],
        "not_built": tally["N"], "foundational": tally["F"],
        "buildable": buildable, "done": done,
        "done_pct": round(100 * done / buildable) if buildable else 0,
        "tiers": [dict(per_tier[t], buildable=per_tier[t]["B"] + per_tier[t]["M"] + per_tier[t]["N"]) for t in tiers],
        "to_implement": to_implement,
    }


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
