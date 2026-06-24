"""Oracle-first tests for the TIER-1 soft-computing nodes (CLAUDE.md Code-Gen
Contract: the acceptance/oracle check is written FIRST, then the code is made to
pass). Light stack only. Each node is tested in the regime where its theory is
MEANT to apply (the "should work" oracle) plus a control, then on the actual
DESIGN-INPUT panel data (ML_ENGINEERING_PRACTICES.md §G) against persistence.

Nodes:
  * fuzzy_ts   — Chen (1996) Fuzzy Time Series. Oracle: tracks a smooth/monotone
                 level series with bounded error and reacts to up/down moves.
  * grey_gm11  — Grey System GM(1,1). Oracle (DATA-POOR by design): recovers the
                 growth of a near-exponential series and forecasts the next points
                 accurately from only ~4-6 points, BEATING persistence on a smooth
                 trend; flags a poor fit on noisy/oscillatory data.
  * anfis      — Adaptive Neuro-Fuzzy Inference System (Sugeno). Oracle: fits a
                 known NONLINEAR function with RMSE materially below ordinary linear
                 regression; reduces toward linear when the target is linear.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain.nodes.tier1_softcomputing  # noqa: F401  (runs @register)
from pattern_brain.registry import all_node_types, create
from pattern_brain.interlingua import validate_belief

FAILS = []
PANEL = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT",
         "LINKUSDT", "LTCUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "panel")


def check(cond, msg):
    if not bool(cond):
        FAILS.append(msg)
    return bool(cond)


def _close(sym):
    o = np.load(os.path.join(DATA_DIR, f"{sym}.npz"))["ohlcv"]
    return o[:, 3].astype(float)


# ============================================================ registration / contract
def test_all_registered_and_conform():
    for nt in ("fuzzy_ts", "grey_gm11", "anfis"):
        check(nt in all_node_types(), f"{nt} not registered")
    # fuzzy_ts / grey_gm11 are unsupervised level forecasters
    x = _close("BTCUSDT")[:200].reshape(-1, 1)
    for nt in ("fuzzy_ts", "grey_gm11"):
        b = create(nt).predict(x)
        v = validate_belief(b)
        check(not v, f"{nt} off-contract: {v}")
        check(np.isfinite(b.payload["estimate"]), f"{nt} non-finite estimate")
    # anfis is supervised
    rng = np.random.default_rng(0)
    Xa = rng.uniform(-2, 2, size=(200, 2))
    ya = np.sin(Xa[:, 0]) + Xa[:, 1] ** 2
    b = create("anfis").fit(Xa, ya).predict(Xa)
    v = validate_belief(b)
    check(not v, f"anfis off-contract: {v}")
    check(np.isfinite(b.payload["estimate"]), "anfis non-finite estimate")
    print("  fuzzy_ts, grey_gm11, anfis registered, run, conform")


# ============================================================ 1) fuzzy_ts ORACLE
def test_fuzzy_ts_tracks_smooth_level():
    """Chen-style FTS should track a smooth level series with bounded error and
    react to the direction of the last move (the classic benchmark behavior)."""
    t = np.arange(120)
    # smooth monotone-with-turns level (enrollment-style): rises then falls
    level = 1000.0 + 300.0 * np.sin(2 * np.pi * t / 120.0) + 0.5 * t
    fc = []
    node = create("fuzzy_ts")
    for k in range(20, len(level)):
        b = node.predict(level[:k].reshape(-1, 1))
        fc.append(b.payload["estimate"])
    fc = np.asarray(fc)
    actual = level[20:]
    # bounded tracking error: FTS forecast within one partition-width of the level.
    rng = float(level.max() - level.min())
    mae = float(np.mean(np.abs(fc - actual)))
    check(mae < 0.08 * rng, f"fuzzy_ts tracking MAE {mae:.2f} too large vs range {rng:.1f}")
    # reacts to direction: when the series is clearly rising, forecast > last seen
    # near the steep ascending part; check correlation of forecast with level.
    corr = float(np.corrcoef(fc, actual)[0, 1])
    check(corr > 0.95, f"fuzzy_ts forecast poorly correlated with level: {corr:.3f}")
    print(f"  fuzzy_ts ORACLE: tracks smooth level MAE={mae:.1f} (<8% range), corr={corr:.3f}")


def test_fuzzy_ts_control_flat_series():
    """Control: on a flat series the forecast collapses to the level (no spurious
    motion)."""
    x = np.full(60, 500.0) + np.random.default_rng(1).normal(0, 0.01, 60)
    b = create("fuzzy_ts").predict(x.reshape(-1, 1))
    check(abs(b.payload["estimate"] - 500.0) < 5.0,
          f"fuzzy_ts on flat series drifted to {b.payload['estimate']}")
    print(f"  fuzzy_ts CONTROL flat: estimate={b.payload['estimate']:.2f} ~500")


# ============================================================ 2) grey_gm11 ORACLE
def test_grey_gm11_recovers_exponential_growth_data_poor():
    """GM(1,1) is DATA-POOR by design: from only ~5-6 points of a near-exponential
    growth series it should recover the growth rate and forecast the next point
    accurately, BEATING persistence on the smooth trend."""
    rate = 0.10
    full = 100.0 * np.exp(rate * np.arange(8))
    train = full[:6]                       # only 6 points
    node = create("grey_gm11")
    b = node.predict(train.reshape(-1, 1))
    a = b.payload["a"]
    # the grey development coefficient -a recovers the continuous growth rate.
    check(abs(-a - rate) < 0.02, f"grey_gm11 growth coeff -a={-a:.4f} != {rate}")
    # one-step forecast accurate, and beats persistence (= last value).
    truth = full[6]
    gm_err = abs(b.payload["estimate"] - truth)
    persist_err = abs(train[-1] - truth)
    check(gm_err < persist_err, f"grey_gm11 err {gm_err:.2f} !< persistence {persist_err:.2f}")
    check(gm_err < 0.02 * truth, f"grey_gm11 forecast off by {gm_err/truth:.1%}")
    check(b.confidence > 0.6, f"grey_gm11 should be confident on a clean exp fit, got {b.confidence}")
    print(f"  grey_gm11 ORACLE: -a={-a:.4f}~{rate}, 1-step err={gm_err:.2f} < persist {persist_err:.2f}, conf={b.confidence:.2f}")


def test_grey_gm11_flags_poor_fit_on_noise():
    """Control: on a noisy/oscillatory series GM(1,1) should report a poor fit
    (low confidence) — it must not pretend to model noise."""
    rng = np.random.default_rng(7)
    osc = 100.0 + 30.0 * np.sin(np.arange(6)) + rng.normal(0, 20, 6)
    b = create("grey_gm11").predict(osc.reshape(-1, 1))
    smooth = create("grey_gm11").predict((100.0 * np.exp(0.1 * np.arange(6))).reshape(-1, 1))
    check(b.confidence < smooth.confidence,
          f"grey_gm11 noise conf {b.confidence:.2f} !< smooth conf {smooth.confidence:.2f}")
    print(f"  grey_gm11 CONTROL noise: conf={b.confidence:.2f} < smooth conf={smooth.confidence:.2f}")


# ============================================================ 3) anfis ORACLE
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_anfis_beats_linear_on_nonlinear_target():
    """ANFIS (Gaussian MFs + linear Sugeno consequents) should fit a known
    NONLINEAR target with test RMSE materially below ordinary linear regression,
    proving it captures the nonlinearity."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2.0, 2.0, size=(400, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2          # genuinely nonlinear
    ntr = 300
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    node = create("anfis").fit(Xtr, ytr)
    pred = np.array([node.predict(Xte[i:i + 1]).payload["estimate"] for i in range(len(Xte))])
    anfis_rmse = _rmse(pred, yte)

    # ordinary least-squares linear baseline
    A = np.column_stack([np.ones(ntr), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    lin_pred = np.column_stack([np.ones(len(Xte)), Xte]) @ coef
    lin_rmse = _rmse(lin_pred, yte)

    check(anfis_rmse < 0.5 * lin_rmse,
          f"anfis RMSE {anfis_rmse:.3f} not materially < linear {lin_rmse:.3f}")
    print(f"  anfis ORACLE nonlinear: RMSE={anfis_rmse:.3f} << linear {lin_rmse:.3f}")


def test_anfis_reduces_toward_linear_when_target_linear():
    """When the target IS linear, ANFIS should be no worse than (≈) linear
    regression — it must not overfit the premises into nonsense."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-2.0, 2.0, size=(400, 2))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3
    ntr = 300
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    pred = np.array([create("anfis").fit(Xtr, ytr).predict(Xte[i:i + 1]).payload["estimate"]
                     for i in range(1)])  # fit once
    node = create("anfis").fit(Xtr, ytr)
    pred = np.array([node.predict(Xte[i:i + 1]).payload["estimate"] for i in range(len(Xte))])
    anfis_rmse = _rmse(pred, yte)
    A = np.column_stack([np.ones(ntr), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    lin_rmse = _rmse(np.column_stack([np.ones(len(Xte)), Xte]) @ coef, yte)
    check(anfis_rmse < 3.0 * lin_rmse + 1e-6,
          f"anfis blew up on linear target: {anfis_rmse:.3f} vs linear {lin_rmse:.3f}")
    print(f"  anfis CONTROL linear: RMSE={anfis_rmse:.3f} ~ linear {lin_rmse:.3f}")


# ============================================================ DESIGN-INPUT verdicts
def test_design_data_fuzzy_grey_vs_persistence_levels():
    """REAL panel level series, rolling 1-step forecast vs persistence. Honest
    expectation: standalone level forecasters usually do NOT beat persistence on
    a near-random-walk close. Records the verdict; does not require a win."""
    wins = {"fuzzy_ts": 0, "grey_gm11": 0}
    detail = {}
    for sym in PANEL:
        c = _close(sym)
        for nt in ("fuzzy_ts", "grey_gm11"):
            node = create(nt)
            errs, perr = [], []
            # walk-forward, modest window; grey_gm11 uses a short window (data-poor)
            win = 6 if nt == "grey_gm11" else 60
            step = 5
            for k in range(win, len(c) - 1, step):
                seg = c[k - win:k]
                est = node.predict(seg.reshape(-1, 1)).payload["estimate"]
                errs.append(abs(est - c[k]))
                perr.append(abs(seg[-1] - c[k]))     # persistence
            m_node = float(np.mean(errs)); m_pers = float(np.mean(perr))
            if m_node < m_pers:
                wins[nt] += 1
            detail.setdefault(nt, []).append((sym, m_node, m_pers))
    for nt in ("fuzzy_ts", "grey_gm11"):
        print(f"  {nt} vs persistence on levels: beats on {wins[nt]}/{len(PANEL)} symbols")
    # sanity: both produced finite, comparable-magnitude errors everywhere
    for nt, rows in detail.items():
        for sym, mn, mp in rows:
            check(np.isfinite(mn) and np.isfinite(mp), f"{nt} non-finite error on {sym}")
    # record for the report (not an assertion of a win)
    test_design_data_fuzzy_grey_vs_persistence_levels.wins = wins


def test_design_data_anfis_returns_vs_persistence():
    """REAL panel: lagged returns + vol -> next return. Honest expectation: a
    standalone return forecaster usually loses to persistence (zero/last). Records
    the verdict."""
    wins = 0
    for sym in PANEL:
        c = _close(sym)
        r = np.diff(np.log(c))
        # features: 3 lagged returns + rolling vol; target next return
        L = 3
        feats, targ = [], []
        vol = np.array([r[max(0, i - 10):i].std() if i >= 2 else 0.0 for i in range(len(r))])
        for i in range(L, len(r) - 1):
            feats.append([r[i - 1], r[i - 2], r[i - 3], vol[i]])
            targ.append(r[i])
        F = np.asarray(feats); T = np.asarray(targ)
        ntr = int(0.7 * len(T))
        node = create("anfis").fit(F[:ntr], T[:ntr])
        pred = np.array([node.predict(F[ntr + j:ntr + j + 1]).payload["estimate"]
                         for j in range(len(T) - ntr)])
        yte = T[ntr:]
        anfis_rmse = _rmse(pred, yte)
        persist_rmse = _rmse(np.zeros_like(yte), yte)   # mean-return persistence ~ 0
        if anfis_rmse < persist_rmse:
            wins += 1
        check(np.isfinite(anfis_rmse), f"anfis non-finite on {sym}")
    print(f"  anfis returns vs zero-persistence: beats on {wins}/{len(PANEL)} symbols")
    test_design_data_anfis_returns_vs_persistence.wins = wins


# ============================================================ run summary
def test_zzz_no_failures():
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
    assert not FAILS, f"{len(FAILS)} checks failed"
