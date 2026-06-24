"""Oracle-first tests for the Tier-3 probability nodes (CONCEPT_EQUATION_BANK.md).

Per the Code-Generation Contract (CLAUDE.md Phase 4 + ML_ENGINEERING_PRACTICES.md):
the oracle/acceptance checks are written BEFORE the module and verify each node on
KNOWN closed-form answers, then on real panel data with the Rule-30 harness.

Nodes under test (all layer ``probability``):
  * ``bayes_update``    — conjugate Normal-Normal posterior mean → ``forecast`` belief.
  * ``maxent_dist``     — Gaussian Maximum-Entropy distribution → ``density`` belief.
  * ``chebyshev_bound`` — distribution-free P(|X-μ|≥kσ) ≤ 1/k² interval → ``forecast``.

Run: ``PYTHONPATH=. python3 -m pytest tests/test_tier3_probability.py -q``
or as a script: ``PYTHONPATH=. python3 tests/test_tier3_probability.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# importing the module registers the three nodes
from pattern_brain.nodes import tier3_probability  # noqa: F401
from pattern_brain.registry import create, all_node_types
from pattern_brain.interlingua import validate_belief

NODES = ["bayes_update", "maxent_dist", "chebyshev_bound"]


# --------------------------------------------------------------- registration
def test_all_registered_and_probability_layer():
    for n in NODES:
        assert n in all_node_types(), f"{n} not registered"
        node = create(n)
        assert node.layer == "probability", f"{n} layer is {node.layer!r}, expected 'probability'"


def test_beliefs_conform_to_interlingua():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200).reshape(-1, 1)
    for n in NODES:
        b = create(n).predict(x)
        viols = validate_belief(b)
        assert not viols, f"{n} emitted an off-contract belief: {viols}"


# ----------------------------------------------- bayes_update closed-form oracle
def test_bayes_update_matches_normal_normal_closed_form():
    """Conjugate Normal-Normal with prior N(0,1) and known likelihood variance 1.0:
    posterior mean of the mean after n iid obs is (Σx)/(n+1). The node uses an
    empirical-Bayes prior, so we test its INTERNAL closed-form solver directly
    against the textbook formula, which is the exact thing it must implement."""
    rng = np.random.default_rng(1)
    x = rng.normal(loc=2.0, scale=1.0, size=50)
    # textbook posterior: prior N(mu0, tau2), likelihood var sigma2, n obs
    mu0, tau2, sigma2 = 0.0, 1.0, 1.0
    n = x.size
    prec_post = 1.0 / tau2 + n / sigma2
    mu_post_oracle = (mu0 / tau2 + x.sum() / sigma2) / prec_post
    var_post_oracle = 1.0 / prec_post
    mu_hat, var_hat = tier3_probability.normal_normal_posterior(
        x, mu0=mu0, tau2=tau2, sigma2=sigma2)
    assert abs(mu_hat - mu_post_oracle) < 1e-10, (mu_hat, mu_post_oracle)
    assert abs(var_hat - var_post_oracle) < 1e-10, (var_hat, var_post_oracle)
    # specific closed form (Σx)/(n+1)
    assert abs(mu_hat - x.sum() / (n + 1)) < 1e-10


def test_bayes_update_posterior_between_prior_and_data():
    """Sanity post-condition: the posterior mean lies between the prior mean and the
    data mean (a defining property of conjugate updating)."""
    x = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    mu_hat, _ = tier3_probability.normal_normal_posterior(x, mu0=0.0, tau2=1.0, sigma2=1.0)
    assert 0.0 < mu_hat < 10.0


def test_bayes_update_emits_finite_forecast():
    rng = np.random.default_rng(2)
    x = rng.normal(size=120).reshape(-1, 1)
    b = create("bayes_update").predict(x)
    assert b.type == "forecast"
    pred = b.payload["next_vector"][0]
    assert np.isfinite(pred)
    assert b.payload["posterior_std"] > 0


# --------------------------------------------- maxent_dist Gaussian recovery oracle
def test_maxent_gaussian_recovers_mean_and_std():
    """Gaussian MaxEnt (mean+variance constraints) of N(0,1) data must recover
    mu≈0, sigma≈1 (the closed-form MaxEnt distribution under those two moments)."""
    rng = np.random.default_rng(3)
    x = rng.normal(loc=0.0, scale=1.0, size=20000)
    params = tier3_probability.gaussian_maxent(x)
    assert abs(params["mu"]) < 0.05, params["mu"]
    assert abs(params["sigma"] - 1.0) < 0.05, params["sigma"]


def test_maxent_gaussian_recovers_shifted_scaled():
    rng = np.random.default_rng(4)
    x = rng.normal(loc=7.0, scale=2.5, size=20000)
    params = tier3_probability.gaussian_maxent(x)
    assert abs(params["mu"] - 7.0) < 0.1, params["mu"]
    assert abs(params["sigma"] - 2.5) < 0.1, params["sigma"]


def test_maxent_pdf_integrates_to_one():
    """The returned density grid must integrate (trapezoid) to ≈1 — a hard
    post-condition for any distribution estimate."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=5000).reshape(-1, 1)
    p = create("maxent_dist").predict(x).payload
    grid = np.asarray(p["grid"])
    pdf = np.asarray(p["pdf"])
    mass = np.trapezoid(pdf, grid) if hasattr(np, "trapezoid") else np.trapz(pdf, grid)
    assert abs(mass - 1.0) < 1e-2, mass


def test_maxent_emits_density_belief():
    rng = np.random.default_rng(6)
    x = rng.normal(size=300).reshape(-1, 1)
    b = create("maxent_dist").predict(x)
    assert b.type == "density"
    assert not validate_belief(b), validate_belief(b)


# ------------------------------------------ chebyshev_bound coverage-guarantee oracle
def test_chebyshev_interval_formula():
    """At k, the half-width is exactly k·σ around the point μ (distribution-free)."""
    mu, sigma, k = 5.0, 2.0, 3.0
    lo, hi = tier3_probability.chebyshev_interval(mu, sigma, k)
    assert abs(lo - (mu - k * sigma)) < 1e-12
    assert abs(hi - (mu + k * sigma)) < 1e-12


def test_chebyshev_coverage_holds_on_heavy_tailed_data():
    """The Chebyshev guarantee is distribution-FREE: for ANY finite-variance dist,
    P(|X-μ|≥kσ) ≤ 1/k². Empirically, the fraction of held-out points inside the
    k=3 band must be ≥ 1 - 1/9 ≈ 88.9%. Test on a heavy-tailed (Student-t)
    sample where a Normal band would be too tight — Chebyshev must still hold."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(df=4, size=100000)  # heavy tails, finite variance
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    for k in (2.0, 3.0, 4.0):
        lo, hi = tier3_probability.chebyshev_interval(mu, sigma, k)
        covered = float(np.mean((x >= lo) & (x <= hi)))
        guarantee = 1.0 - 1.0 / (k * k)
        assert covered >= guarantee - 1e-3, (k, covered, guarantee)


def test_chebyshev_emits_forecast_with_interval():
    rng = np.random.default_rng(8)
    x = rng.normal(size=200).reshape(-1, 1)
    b = create("chebyshev_bound").predict(x)
    assert b.type == "forecast"
    lo, hi = b.payload["interval"]
    assert lo < b.payload["next_vector"][0] < hi
    assert 0.0 < b.payload["coverage_guarantee"] < 1.0


# --------------------------------------------------------- domain independence
def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "tier3_probability.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    assert not hits, f"domain-coupling identifiers found: {hits}"


# ------------------------------------------------------------------ run-as-script
def _main():
    import pytest
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))


if __name__ == "__main__":
    _main()
