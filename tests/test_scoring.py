"""Tests for Phase 8 slice 1 — distributional proper scoring (PLAN.md §12).

Run: python3 tests/test_scoring.py  (light stack — numpy/scipy/sklearn).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain import scoring as S

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def crps_normal_closed_form(mu, sigma, y):
    """Closed-form CRPS for a Normal(mu, sigma) forecast (Gneiting & Raftery 2007)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * _normal_cdf(z) - 1) + 2 * _normal_pdf(z) - 1.0 / math.sqrt(math.pi))


def test_crps_matches_normal_closed_form():
    """The sample CRPS estimator must converge to the analytic Normal CRPS."""
    rng = np.random.default_rng(0)
    mu, sigma, y = 0.3, 1.2, 0.9
    samples = rng.normal(mu, sigma, size=40000)
    est = S.crps_ensemble(samples, y)
    exact = crps_normal_closed_form(mu, sigma, y)
    check(abs(est - exact) < 0.02, f"sample CRPS {est:.4f} != closed-form {exact:.4f}")
    print(f"  CRPS estimator matches closed-form Normal: {est:.4f} ≈ {exact:.4f}")


def test_crps_rewards_sharp_accurate_forecasts():
    """A sharp ensemble centered on y beats a wide one, which beats a biased one."""
    rng = np.random.default_rng(1)
    y = 0.0
    sharp = rng.normal(0.0, 0.2, 5000)
    wide = rng.normal(0.0, 2.0, 5000)
    biased = rng.normal(3.0, 0.2, 5000)
    cs, cw, cb = (S.crps_ensemble(sharp, y), S.crps_ensemble(wide, y),
                  S.crps_ensemble(biased, y))
    check(cs < cw < cb, f"CRPS ordering wrong: sharp {cs:.3f}, wide {cw:.3f}, biased {cb:.3f}")
    check(S.crps_ensemble(np.full(50, y), y) < 1e-9, "point-mass-at-y CRPS should be ~0")
    print(f"  CRPS rewards sharpness+accuracy: sharp {cs:.3f} < wide {cw:.3f} < biased {cb:.3f}")


def test_crps_quantile_consistency():
    """CRPS from a dense quantile grid ≈ CRPS from samples for the same forecast."""
    rng = np.random.default_rng(2)
    samples = rng.normal(0.5, 1.0, 20000)
    y = 1.0
    levels = np.linspace(0.01, 0.99, 99)
    qvals = np.quantile(samples, levels)
    c_s = S.crps_ensemble(samples, y)
    c_q = S.crps_from_quantiles(levels, qvals, y)
    check(abs(c_s - c_q) < 0.03, f"sample CRPS {c_s:.3f} vs quantile CRPS {c_q:.3f} diverge")
    print(f"  CRPS sample≈quantile form: {c_s:.3f} ≈ {c_q:.3f}")


def test_pinball_brier_logloss():
    # pinball symmetric at tau=0.5, asymmetric otherwise
    check(abs(S.pinball_loss(1.0, 0.5, 2.0) - 0.5) < 1e-9, "median pinball should be 0.5*|y-q|")
    check(S.pinball_loss(1.0, 0.9, 2.0) > S.pinball_loss(1.0, 0.1, 2.0),
          "under-prediction should cost more at the 0.9 quantile")
    # brier/log-loss: perfect vs worst
    check(S.brier_score([1.0, 0.0], [1, 0]) < 1e-9, "perfect Brier should be ~0")
    check(abs(S.log_loss([0.5, 0.5], [1, 0]) - math.log(2)) < 1e-6, "logloss(0.5)=ln2")
    check(S.log_loss([0.99, 0.01], [1, 0]) < S.log_loss([0.5, 0.5], [1, 0]),
          "confident-correct logloss should beat uninformed")
    print("  pinball/Brier/log-loss behave correctly")


def test_pit_calibration_detects_miscalibration():
    """A forecast sampling the TRUE distribution → PIT uniform (calibrated);
    an over-confident (too-narrow) forecast → PIT clusters at the tails (not)."""
    rng = np.random.default_rng(3)
    T, M = 600, 400
    mean = rng.normal(0.0, 1.0, T)                                 # latent mean per period
    y = rng.normal(mean, 1.0)                                      # realized = a DRAW ~ N(mean,1)
    calibrated = rng.normal(mean[:, None], 1.0, size=(T, M))       # forecast = the true dist
    overconf = rng.normal(mean[:, None], 0.25, size=(T, M))        # too narrow vs reality
    pit_cal = S.pit_values_ensemble(calibrated, y)
    pit_oc = S.pit_values_ensemble(overconf, y)
    cal = S.pit_calibration_error(pit_cal)
    oc = S.pit_calibration_error(pit_oc)
    check(cal["ks"] < oc["ks"], f"calibrated KS {cal['ks']:.3f} should be < overconfident {oc['ks']:.3f}")
    check(cal["calibrated"] and not oc["calibrated"],
          f"flags wrong: calibrated={cal['calibrated']} overconf={oc['calibrated']}")
    print(f"  PIT detects miscalibration: calibrated KS={cal['ks']:.3f} (ok) "
          f"vs overconfident KS={oc['ks']:.3f} (flagged)")


def test_skill_and_evaluator_bridge():
    """The per-period CRPS skill series must (a) reward a good model over a baseline
    and (b) run through the EXISTING Evaluator's outcome-scoring (purged-WF + DSR)."""
    rng = np.random.default_rng(4)
    T, M = 400, 300
    truth = rng.normal(0.0, 1.0, T)
    good = rng.normal(truth[:, None], 0.5, size=(T, M))           # informative
    naive = rng.normal(0.0, 1.0, size=(T, M))                     # climatology
    crps_good = S.crps_ensemble(good, truth)
    crps_naive = S.crps_ensemble(naive, truth)
    skill = S.crps_skill_score(crps_good, crps_naive)
    check(skill > 0, f"informative model should have positive CRPS skill, got {skill:.3f}")
    series = S.crps_outcome_series(crps_good, crps_naive)
    check(len(series) == T and np.mean(series) > 0, "skill outcome series should be net positive")

    from pattern_brain.evaluator import Evaluator
    ev = Evaluator(n_splits=4, embargo=0.02)
    rep = ev.evaluate_outcomes(series, n_trials=1)
    check(np.isfinite(rep.sharpe) and np.isfinite(rep.dsr),
          "Evaluator must score the distributional outcome series with finite Sharpe/DSR")
    print(f"  skill={skill:.3f}; Evaluator on the CRPS-skill series → Sharpe={rep.sharpe:.2f}, DSR={rep.dsr:.2f}")


def test_score_distribution_scorecard():
    rng = np.random.default_rng(5)
    T, M = 200, 200
    truth = rng.normal(0, 1, T)
    fc = rng.normal(truth[:, None], 0.6, size=(T, M))
    card = S.score_distribution(fc, truth, baseline_crps=S.crps_ensemble(rng.normal(0, 1, (T, M)), truth))
    check(set(["mean_crps", "calibration", "outcome_series", "crps_skill"]).issubset(card),
          f"scorecard missing keys: {card.keys()}")
    check(card["crps_skill"] > 0 and len(card["outcome_series"]) == T, "scorecard values off")
    print(f"  scorecard: mean_crps={card['mean_crps']:.3f}, skill={card['crps_skill']:.3f}")


def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book", "crypto", "binance")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "scoring.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    check(not hits, f"domain-coupling identifiers in scoring.py: {hits}")
    print("  domain-independence: scoring.py clean")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 1: distributional scoring tests")
    print("=" * 70)
    test_crps_matches_normal_closed_form()
    test_crps_rewards_sharp_accurate_forecasts()
    test_crps_quantile_consistency()
    test_pinball_brier_logloss()
    test_pit_calibration_detects_miscalibration()
    test_skill_and_evaluator_bridge()
    test_score_distribution_scorecard()
    test_domain_independence()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
