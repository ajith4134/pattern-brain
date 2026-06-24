"""Oracle tests for TIER-1 directly-buildable classics — written BEFORE the code
(Code-Generation Contract, CLAUDE.md Phase 4). Built one node at a time.

Run: PYTHONPATH=. python3 -m pytest tests/test_tier1_classics.py -q
"""
from __future__ import annotations

import numpy as np

from pattern_brain.nodes.tier1_classics import (
    hill_tail_index, gpd_pot_fit, evt_var_es,
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
