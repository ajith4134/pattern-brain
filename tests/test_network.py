"""Tests for Phase 8 slice 5 — the Stacked-DAG (PLAN.md §12).

Run: python3 tests/test_network.py  (light stack — numpy/scipy/sklearn).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_brain.network import DAGSpec, StackedDAG

FAILS = []


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _ar_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    return a.reshape(-1, 1)


def test_dag_runs_and_emits_distribution():
    X = _ar_series(180, 0)
    spec = DAGSpec(layers=[["drift_forecast", "theta_forecast"]], combiner="mixture")
    fc = StackedDAG(spec, min_train=40, n_samples=120).run(X)
    check(fc.index.size > 0, "DAG produced no evaluations")
    check(fc.samples.shape == (fc.index.size, 120), f"samples shape {fc.samples.shape}")
    check(np.all(np.isfinite(fc.samples)), "DAG samples non-finite")
    check(fc.index.max() < len(X), "DAG must not evaluate beyond the series")
    print(f"  DAG runs end-to-end → predictive distribution ({fc.samples.shape})")


def test_both_combiners_score():
    X = _ar_series(220, 1)
    for combiner in ("mixture", "stacked"):
        spec = DAGSpec(layers=[["drift_forecast", "theta_forecast", "naive_mean_forecast"]],
                       combiner=combiner)
        s = StackedDAG(spec, min_train=40, n_samples=150).score(X)
        for k in ("mean_crps", "crps_skill", "calib_ks", "n"):
            check(k in s and np.isfinite(s[k]), f"{combiner}: missing/non-finite {k}")
        check(s["n"] > 0 and len(s["outcome_series"]) == s["n"], f"{combiner}: outcome series wrong")
        print(f"  combiner '{combiner}': crps={s['mean_crps']:.3f}, skill={s['crps_skill']:.3f}, "
              f"calibrated={s['calibrated']}")


def test_models_feed_models_augmentation():
    """A 2-layer DAG must feed layer-0 outputs into layer-1's input matrix (the
    augmented width exceeds the raw width)."""
    X = _ar_series(160, 2)
    spec = DAGSpec(layers=[["drift_forecast", "theta_forecast"], ["var_forecast"]],
                   combiner="mixture")
    fc = StackedDAG(spec, min_train=40, n_samples=80).run(X)
    check(fc.diag["augmented_width"] > fc.diag["raw_width"],
          f"layer-1 should see augmented columns: {fc.diag}")
    check(fc.diag["augmented_width"] == fc.diag["raw_width"] + 2,
          "two base outputs should add exactly two columns")
    print(f"  models feed models: raw width {fc.diag['raw_width']} → augmented "
          f"{fc.diag['augmented_width']} (base outputs became meta inputs)")


def test_dag_is_leakage_safe():
    """Perturbing the FUTURE must not change the DAG's earlier predictions."""
    X = _ar_series(200, 3)
    spec = DAGSpec(layers=[["drift_forecast", "theta_forecast"]], combiner="stacked")
    dag = StackedDAG(spec, min_train=40, n_samples=100, seed=0)
    f1 = dag.run(X)
    X2 = X.copy(); X2[150:] += 30.0
    f2 = dag.run(X2)
    early = f1.index < 145
    check(np.allclose(f1.point[early], f2.point[early], atol=0, rtol=0),
          "LOOK-AHEAD: DAG predictions before t changed when the future was perturbed")
    print(f"  causality: DAG predictions at t≤145 unchanged by future perturbation")


def test_dagspec_serializable():
    spec = DAGSpec(layers=[["a", "b"], ["c"]], combiner="stacked")
    import json
    d = json.loads(spec.signature())
    check(d["combiner"] == "stacked" and d["layers"] == [["a", "b"], ["c"]], "spec round-trip wrong")
    check(spec.n_nodes == 3, "n_nodes wrong")
    print("  DAGSpec serializes (genome for the slice-6 search)")


def main():
    print("=" * 70)
    print("Pattern Brain — Phase 8 slice 5: Stacked-DAG: tests")
    print("=" * 70)
    test_dag_runs_and_emits_distribution()
    test_both_combiners_score()
    test_models_feed_models_augmentation()
    test_dag_is_leakage_safe()
    test_dagspec_serializable()
    print("=" * 70)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s):")
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
