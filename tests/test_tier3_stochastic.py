"""Tier-3 stochastic-process foundations — oracle-first tests (Rule 25 / CLAUDE.md Phase 4).

Run: PYTHONPATH=. python3 -m pytest tests/test_tier3_stochastic.py -q
 or: PYTHONPATH=. python3 tests/test_tier3_stochastic.py

These tests are written BEFORE the implementation (the CLAUDE.md headline rule:
write the oracle/acceptance check first, then code until it passes). They pin the
three nodes to closed-form ground truth:

* gbm_baseline      — recover GBM drift mu and vol sigma from simulated GBM.
* langevin_sampler  — recover OU gamma, mu from simulated OU AND prove the discrete
                      Langevin conditional mean is EXACTLY the AR(1)/OU form (<1e-8).
* fokker_planck     — the OU transition density integrates to 1 and its mean/variance
                      match the closed-form OU transition moments.

Domain-agnostic (Rule 23): generic (T,D) arrays, no candle/orderbook knowledge.
Scratch data is written under data/_test_tier3_stoch/ and cleaned up (Rule 1).
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief
from pattern_brain.nodes import tier3_stochastic as t3  # noqa: F401  (registers nodes)

TIER3 = ["gbm_baseline", "langevin_sampler", "fokker_planck"]


# --------------------------------------------------------------- simulators
def simulate_gbm(mu: float, sigma: float, x0: float, n: int, dt: float, seed: int) -> np.ndarray:
    """Exact GBM path: x_{t+1} = x_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    logx = np.log(x0) + np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return np.concatenate([[x0], np.exp(logx)])


def simulate_ou(gamma: float, mu: float, sigma: float, x0: float,
                n: int, dt: float, seed: int) -> np.ndarray:
    """Exact OU path via the transition: x_{t+1} = mu + (x_t-mu) e^{-gamma dt}
    + sqrt(sigma^2/(2 gamma) (1-e^{-2 gamma dt})) z."""
    rng = np.random.default_rng(seed)
    b = np.exp(-gamma * dt)
    sd = np.sqrt(sigma ** 2 / (2.0 * gamma) * (1.0 - b ** 2))
    x = np.empty(n + 1)
    x[0] = x0
    for t in range(n):
        x[t + 1] = mu + (x[t] - mu) * b + sd * rng.normal()
    return x


# ----------------------------------------------------- registration / contract
def test_all_registered_and_conform():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=300)).reshape(-1, 1) + 100.0   # positive level
    for n in TIER3:
        assert n in all_node_types(), f"{n} not registered"
        b = create(n).predict(x)
        assert not validate_belief(b), f"{n} belief off-contract: {validate_belief(b)}"
        assert np.isfinite(b.confidence), f"{n} non-finite confidence"


# ------------------------------------------------------------ gbm oracle
def test_gbm_recovers_drift_and_vol():
    """Estimated mu/sigma from the node must match the true GBM parameters within
    sampling tolerance (averaged over seeds to beat Monte-Carlo noise)."""
    mu_true, sigma_true, dt = 0.0005, 0.02, 1.0
    mu_hats, sig_hats = [], []
    for s in range(8):
        path = simulate_gbm(mu_true, sigma_true, x0=100.0, n=4000, dt=dt, seed=s)
        p = create("gbm_baseline").predict(path.reshape(-1, 1)).payload
        mu_hats.append(p["mu"])
        sig_hats.append(p["sigma"])
    mu_hat, sig_hat = float(np.mean(mu_hats)), float(np.mean(sig_hats))
    assert abs(sig_hat - sigma_true) < 0.002, f"sigma recovery off: {sig_hat} vs {sigma_true}"
    assert abs(mu_hat - mu_true) < 0.0015, f"mu recovery off: {mu_hat} vs {mu_true}"


def test_gbm_forecast_is_lognormal_mean():
    """The one-step level forecast must equal the closed-form GBM mean
    E[x_{t+1}|x_t] = x_t * exp(mu_hat) with mu_hat the per-step log-drift+convexity."""
    path = simulate_gbm(0.001, 0.015, x0=50.0, n=2000, dt=1.0, seed=11)
    b = create("gbm_baseline").predict(path.reshape(-1, 1))
    p = b.payload
    last = path[-1]
    r = np.diff(np.log(path))
    # unbiased (ddof=1) sigma is the documented convention used by the node
    expected = last * np.exp(r.mean() + 0.5 * r.var(ddof=1))   # lognormal mean of next level
    assert abs(p["next_vector"][0] - expected) < 1e-8, f"{p['next_vector'][0]} vs {expected}"
    assert abs(p["estimate"] - p["next_vector"][0]) < 1e-12, "estimate must mirror next_vector"
    assert p["std"] > 0, "forecast must carry a positive sigma band"


def test_gbm_bachelier_fallback_on_nonpositive():
    """A series with non-positive values can't take logs; the honest Bachelier
    (arithmetic BM) fallback must give x_t + mean(dx) and still emit a band."""
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(scale=1.0, size=500))          # centered at 0, crosses zero
    b = create("gbm_baseline").predict(x.reshape(-1, 1))
    p = b.payload
    expected = x[-1] + np.diff(x).mean()
    assert abs(p["next_vector"][0] - expected) < 1e-8, f"{p['next_vector'][0]} vs {expected}"
    assert p.get("model", "").lower().startswith("bachelier"), "should flag the Bachelier fallback"


# ---------------------------------------------------------- langevin oracle
def test_langevin_recovers_ou_params():
    """gamma and mu recovered from a simulated OU path within tolerance."""
    gamma_true, mu_true, sigma_true, dt = 0.05, 5.0, 0.3, 1.0
    g_hats, mu_hats = [], []
    for s in range(8):
        path = simulate_ou(gamma_true, mu_true, sigma_true, x0=5.0, n=6000, dt=dt, seed=s)
        p = create("langevin_sampler").predict(path.reshape(-1, 1)).payload
        g_hats.append(p["gamma"])
        mu_hats.append(p["mu"])
    g_hat, mu_hat = float(np.mean(g_hats)), float(np.mean(mu_hats))
    assert abs(g_hat - gamma_true) < 0.01, f"gamma recovery off: {g_hat} vs {gamma_true}"
    assert abs(mu_hat - mu_true) < 0.2, f"mu recovery off: {mu_hat} vs {mu_true}"


def test_langevin_conditional_mean_equals_ar1_identity():
    """The discrete Langevin one-step conditional mean must equal the AR(1) form
    a + b*x_t EXACTLY (to <1e-8) — this is the SDE-engine framing of OU/AR(1)."""
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=400)) + 10.0
    # independent OLS AR(1): x_{t+1} = a + b x_t
    x0, x1 = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(x0), x0])
    coef, *_ = np.linalg.lstsq(A, x1, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    ar1_pred = a + b * x[-1]
    node_pred = create("langevin_sampler").predict(x.reshape(-1, 1)).payload["next_vector"][0]
    assert abs(node_pred - ar1_pred) < 1e-8, f"Langevin cond.mean != AR(1): {node_pred} vs {ar1_pred}"


def test_langevin_matches_ou_node():
    """Honest overlap check: langevin_sampler's point forecast must equal the
    existing ou_mean_reversion node's (same underlying AR(1) math, SDE framing)."""
    rng = np.random.default_rng(6)
    x = (np.cumsum(rng.normal(size=350)) + 20.0).reshape(-1, 1)
    lp = create("langevin_sampler").predict(x).payload["next_vector"][0]
    op = create("ou_mean_reversion").predict(x).payload["next_vector"][0]
    assert abs(lp - op) < 1e-8, f"langevin {lp} != ou_mean_reversion {op}"


# --------------------------------------------------------- fokker-planck oracle
def test_fokker_planck_density_integrates_to_one():
    """The forward-evolved transition density must integrate to 1 over its grid."""
    rng = np.random.default_rng(8)
    x = (np.cumsum(rng.normal(scale=0.5, size=500)) + 30.0).reshape(-1, 1)
    p = create("fokker_planck").predict(x).payload
    grid = np.asarray(p["grid"])
    pdf = np.asarray(p["pdf"])
    mass = float(np.trapezoid(pdf, grid)) if hasattr(np, "trapezoid") else float(np.trapz(pdf, grid))
    assert abs(mass - 1.0) < 1e-3, f"density does not integrate to 1: {mass}"


def test_fokker_planck_moments_match_ou_transition():
    """For a simulated OU process, the Gaussian transition density's mean and
    variance must match the closed-form OU one-step transition moments."""
    gamma, mu, sigma, dt = 0.08, 3.0, 0.4, 1.0
    path = simulate_ou(gamma, mu, sigma, x0=3.0, n=6000, dt=dt, seed=10)
    p = create("fokker_planck").predict(path.reshape(-1, 1)).payload
    # closed-form OU transition from the LAST observed value
    b = np.exp(-gamma * dt)
    m_true = mu + (path[-1] - mu) * b
    v_true = sigma ** 2 / (2.0 * gamma) * (1.0 - b ** 2)
    assert abs(p["mean"] - m_true) < 0.15, f"transition mean off: {p['mean']} vs {m_true}"
    assert abs(p["variance"] - v_true) < 0.5 * v_true, f"transition var off: {p['variance']} vs {v_true}"
    # the analytic mean/var must match the numerically-integrated grid moments
    grid, pdf = np.asarray(p["grid"]), np.asarray(p["pdf"])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    grid_mean = float(trap(grid * pdf, grid))
    grid_var = float(trap((grid - grid_mean) ** 2 * pdf, grid))
    assert abs(grid_mean - p["mean"]) < 1e-2, f"grid mean != analytic: {grid_mean} vs {p['mean']}"
    assert abs(grid_var - p["variance"]) / p["variance"] < 0.05, "grid var != analytic var"


def test_fokker_planck_gbm_generator_density_normalizes():
    """On a positive (GBM-like) series the Fokker-Planck transition density (whichever
    generator the node fits) must still be a valid normalized pdf with a finite mode."""
    path = simulate_gbm(0.001, 0.02, x0=100.0, n=1500, dt=1.0, seed=12)
    p = create("fokker_planck").predict(path.reshape(-1, 1)).payload
    grid, pdf = np.asarray(p["grid"]), np.asarray(p["pdf"])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert abs(float(trap(pdf, grid)) - 1.0) < 1e-3, "GBM-generator density must normalize"
    assert np.all(pdf >= 0), "a density must be non-negative"
    assert 0.0 < p["max_proba"] <= 1.0, "max_proba must be a valid bin mass in (0,1]"


# --------------------------------------------------- domain-independence (Rule 23)
def test_domain_independence():
    import tokenize
    forbidden = ("candle", "ohlcv", "orderbook", "order_book")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pattern_brain", "nodes", "tier3_stochastic.py")
    hits = []
    with tokenize.open(path) as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.NAME and any(b in tok.string.lower() for b in forbidden):
                hits.append(tok.string)
    assert not hits, f"domain-coupling identifiers in tier3_stochastic.py: {hits}"


# --------------------------------------------------- scratch-dir hygiene (Rule 1)
def test_scratch_cleanup():
    scratch = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "_test_tier3_stoch")
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    assert not os.path.isdir(scratch)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print("=" * 60)
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
