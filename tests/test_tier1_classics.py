"""Oracle tests for TIER-1 directly-buildable classics — written BEFORE the code
(Code-Generation Contract, CLAUDE.md Phase 4). Built one node at a time.

Run: PYTHONPATH=. python3 -m pytest tests/test_tier1_classics.py -q
"""
from __future__ import annotations

import numpy as np

from pattern_brain.nodes.tier1_classics import (
    hill_tail_index, gpd_pot_fit, evt_var_es,
    adf_tstat, engle_granger, har_fit_forecast, empirical_tail_dependence,
    bocpd_gaussian,
)
from pattern_brain.registry import create
from pattern_brain.interlingua import validate_belief


# ---------------------------------------------------------------- EVT oracles
def test_hill_recovers_pareto_tail_index():
    """Hill estimator on Pareto(b=α) data recovers α (its defining property)."""
    rng = np.random.default_rng(0)
    for alpha in (2.0, 3.0, 4.0):
        # Pareto: P(X>x)=x^-alpha for x>=1  ->  tail index alpha
        x = (1.0 / rng.random(60_000)) ** (1.0 / alpha)
        est = hill_tail_index(x, k_frac=0.05)
        assert abs(est - alpha) < 0.4, f"Hill {est:.3f} vs true {alpha}"


def test_hill_separates_heavy_from_thin_tails():
    """A Gaussian (thin tail) yields a much larger tail index than Student-t(3)."""
    rng = np.random.default_rng(1)
    gauss = rng.standard_normal(40_000)
    student = rng.standard_t(3, 40_000)
    a_g = hill_tail_index(np.abs(gauss), k_frac=0.05)
    a_s = hill_tail_index(np.abs(student), k_frac=0.05)
    assert a_s < a_g, f"student tail index {a_s:.2f} should be < gaussian {a_g:.2f}"
    assert a_s < 5.0, f"student-t(3) tail index {a_s:.2f} should be heavy (small)"


def test_gpd_pot_recovers_shape():
    """GPD method-of-moments recovers the generalized-Pareto shape on simulated excesses."""
    from scipy.stats import genpareto
    rng = np.random.default_rng(2)
    for xi in (0.1, 0.25, 0.4):
        excess = genpareto.rvs(c=xi, scale=1.0, size=50_000, random_state=rng)
        xi_hat, beta_hat = gpd_pot_fit(excess)
        assert abs(xi_hat - xi) < 0.12, f"xi {xi_hat:.3f} vs true {xi}"
        assert beta_hat > 0


def test_var_es_monotone_and_ordered():
    """VaR rises with the confidence level, and ES >= VaR (it averages the worse tail)."""
    rng = np.random.default_rng(3)
    losses = rng.standard_t(4, 30_000)
    var95, es95 = evt_var_es(losses, p=0.95)
    var99, es99 = evt_var_es(losses, p=0.99)
    assert var99 > var95 > 0, f"VaR not monotone: {var95:.3f}, {var99:.3f}"
    assert es99 >= var99 and es95 >= var95, "ES must be >= VaR"


# ---------------------------------------------------------------- node contract
def test_evt_node_registers_and_conforms():
    node = create("evt_tail_risk")
    rng = np.random.default_rng(4)
    X = rng.standard_t(4, 500).reshape(-1, 1)
    b = node.process(X)
    assert b.type == "signal"
    assert validate_belief(b) == [], f"belief not conformant: {validate_belief(b)}"
    assert len(b.payload["series"]) == len(X)            # per-point extremeness signal
    assert np.all(np.isfinite(b.payload["series"]))
    assert b.payload["tail_index"] > 0 and 0.0 <= b.payload["var_99"]
    assert 0.0 <= b.confidence <= 1.0


def test_evt_node_flags_heavier_tail_with_lower_index():
    """The node's reported tail_index is smaller (heavier) for Student-t(3) than Gaussian."""
    rng = np.random.default_rng(5)
    g = create("evt_tail_risk").process(rng.standard_normal(4000).reshape(-1, 1))
    s = create("evt_tail_risk").process(rng.standard_t(3, 4000).reshape(-1, 1))
    assert s.payload["tail_index"] < g.payload["tail_index"]


def test_evt_node_handles_short_input_gracefully():
    node = create("evt_tail_risk")
    b = node.process(np.array([[0.1], [-0.2], [0.05]]))   # too short for a tail fit
    assert b.type == "signal" and np.all(np.isfinite(b.payload["series"]))
    assert b.confidence < 0.5                              # low confidence on thin data


# ============================================================ ADF unit-root oracles
def test_adf_random_walk_is_nonstationary():
    """A random walk has a unit root -> ADF t-stat is NOT very negative (> -2.86 @ 5%)."""
    rng = np.random.default_rng(10)
    rw = np.cumsum(rng.standard_normal(800))
    t = adf_tstat(rw, p=1)
    assert t > -2.86, f"random walk wrongly judged stationary (t={t:.2f})"


def test_adf_mean_reverting_is_stationary():
    """A strongly mean-reverting AR(1) (phi=0.2) -> very negative ADF t-stat (< -5)."""
    rng = np.random.default_rng(11)
    x = np.zeros(800)
    for i in range(1, 800):
        x[i] = 0.2 * x[i - 1] + rng.standard_normal()
    t = adf_tstat(x, p=1)
    assert t < -5.0, f"stationary series not detected (t={t:.2f})"


# ============================================================ Engle-Granger cointegration
def test_engle_granger_detects_cointegrated_pair():
    """y = 2x + stationary noise, x a random walk -> cointegrated, hedge ratio ~2."""
    rng = np.random.default_rng(12)
    x = np.cumsum(rng.standard_normal(800))
    y = 2.0 * x + rng.standard_normal(800)          # stationary residual
    res = engle_granger(np.column_stack([y, x]))
    assert res["cointegrated"], f"missed cointegration (adf={res['adf_stat']:.2f})"
    assert abs(res["hedge_ratio"] - 2.0) < 0.2, f"hedge ratio {res['hedge_ratio']:.3f}"


def test_engle_granger_rejects_independent_walks():
    """Two independent random walks share no equilibrium -> NOT cointegrated."""
    rng = np.random.default_rng(13)
    x = np.cumsum(rng.standard_normal(800))
    y = np.cumsum(rng.standard_normal(800))
    res = engle_granger(np.column_stack([y, x]))
    assert not res["cointegrated"], f"spurious cointegration (adf={res['adf_stat']:.2f})"


def test_johansen_node_conforms_and_signals():
    node = create("johansen_coint")
    rng = np.random.default_rng(14)
    x = np.cumsum(rng.standard_normal(400))
    y = 1.5 * x + rng.standard_normal(400)
    b = node.process(np.column_stack([y, x]))
    assert b.type == "signal" and validate_belief(b) == []
    assert len(b.payload["series"]) == 400
    assert b.payload["cointegrated"] in (True, False)


# ============================================================ HAR-RV volatility forecaster
def test_har_beats_naive_on_clustered_volatility():
    """On vol-clustering data (GARCH-like), HAR forecasts RV better (lower MSE) than
    the persistence baseline RV[t+1]=RV[t]."""
    rng = np.random.default_rng(15)
    n = 2000
    r = np.zeros(n); sig2 = np.ones(n)
    for i in range(1, n):
        sig2[i] = 0.02 + 0.1 * r[i - 1] ** 2 + 0.88 * sig2[i - 1]   # GARCH(1,1)
        r[i] = np.sqrt(sig2[i]) * rng.standard_normal()
    rv = r ** 2
    fc, naive, _ = har_fit_forecast(rv, train_frac=0.6)
    target = rv[-len(fc):]
    har_mse = np.mean((fc - target) ** 2)
    naive_mse = np.mean((naive - target) ** 2)
    assert har_mse < naive_mse, f"HAR {har_mse:.4g} not better than naive {naive_mse:.4g}"


def test_har_node_conforms():
    node = create("har_rv")
    rng = np.random.default_rng(16)
    x = np.cumsum(rng.standard_normal(600))         # a price-like level
    b = node.process(x.reshape(-1, 1))
    assert b.type == "forecast" and validate_belief(b) == []
    assert b.payload["rv_forecast"] >= 0.0


# ============================================================ Copula tail dependence
def test_t_copula_has_more_tail_dependence_than_gaussian():
    """t-copula (heavy joint tails) shows positive upper-tail dependence; a Gaussian
    copula's empirical tail dependence is near zero."""
    from scipy.stats import multivariate_t, multivariate_normal, t as tdist, norm
    rng = np.random.default_rng(17)
    rho = 0.5
    cov = np.array([[1.0, rho], [rho, 1.0]])
    # t-copula: t-sample -> uniforms via t-cdf
    z = multivariate_t(loc=[0, 0], shape=cov, df=3).rvs(20000, random_state=rng)
    Ut = tdist.cdf(z, df=3)
    # gaussian copula: normal-sample -> uniforms via normal-cdf
    g = multivariate_normal(mean=[0, 0], cov=cov).rvs(20000, random_state=rng)
    Ug = norm.cdf(g)
    _, up_t = empirical_tail_dependence(Ut[:, 0], Ut[:, 1], q=0.98)
    _, up_g = empirical_tail_dependence(Ug[:, 0], Ug[:, 1], q=0.98)
    assert up_t > up_g + 0.05, f"t-copula tail-dep {up_t:.3f} not > gaussian {up_g:.3f}"


def test_copula_node_conforms():
    node = create("copula_dependence")
    rng = np.random.default_rng(18)
    a = rng.standard_normal(500)
    b_ = 0.7 * a + 0.3 * rng.standard_normal(500)
    out = node.process(np.column_stack([a, b_]))
    assert out.type == "signal" and validate_belief(out) == []
    assert len(out.payload["series"]) == 500
    assert -1.0 <= out.payload["kendall_tau"] <= 1.0


# ============================================================ BOCPD change-point
def test_bocpd_detects_mean_shift():
    """A clean mean shift at t=100 in a 200-point series -> changepoint probability
    peaks within a small window of the true break."""
    rng = np.random.default_rng(19)
    x = np.concatenate([rng.standard_normal(100) * 0.5,
                        5.0 + rng.standard_normal(100) * 0.5])
    cp, _ = bocpd_gaussian(x, hazard=1.0 / 50.0)
    peak = int(np.argmax(cp[1:])) + 1                  # ignore the t=0 boundary
    assert abs(peak - 100) <= 8, f"change-point found at {peak}, expected ~100"


def test_bocpd_node_conforms():
    node = create("bocpd_break")
    rng = np.random.default_rng(20)
    x = np.concatenate([rng.standard_normal(80), 3.0 + rng.standard_normal(80)])
    b = node.process(x.reshape(-1, 1))
    assert b.type == "anomaly" and validate_belief(b) == []
    assert len(b.payload["scores"]) == 160 and len(b.payload["flags"]) == 160
