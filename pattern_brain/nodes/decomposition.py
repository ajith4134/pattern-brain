"""Decomposition layer (PLAN §17.6 — named-model coverage gaps: Wavelet, SSA, EMD).

The owner's vision repeatedly used decomposition front-ends ("Wavelet → HDBSCAN
→ …", "SSA → GMM → PySR → …"): the "feature factory" that splits a raw series
into interpretable components (trend / cycle / detail / residual) before the rest
of the graph consumes it. The bank had ``fft`` for *frequency* content but no
*time-frequency* / *data-adaptive* decomposition — this module fills that gap.

Four nodes, all behind the one generic ``Node`` interface (Rule 23 — generic
``(T, D)`` in, a cleaned/reconstructed ``(T, D)`` out + a conformant Belief):

* ``ssa_decompose``  — Singular Spectrum Analysis (trajectory-matrix SVD +
  diagonal averaging). **Pure numpy**, always available.
* ``wavelet_decompose`` — multi-level discrete wavelet transform + soft-threshold
  denoise. Uses **PyWavelets** if installed (richer wavelets); otherwise a
  **dependency-free Haar** fallback, so the node ALWAYS registers.
* ``emd_decompose`` — Empirical Mode Decomposition (cubic-spline envelope sifting
  into IMFs). Uses **scipy.interpolate** (already a dependency); always available.
* ``pysr_symbolic`` — PySR symbolic-regression backend (the owner named PySR
  specifically). **OPTIONAL-guarded** like the torch nodes: registers only if
  ``pysr`` is importable (needs Julia), so it's a no-op on a box without it while
  the integration path exists. The symbolic *capability* already exists via
  ``symbolic_regression`` / ``genetic_symbolic_regression`` / ``sindy_regression``.

Interlingua-conformant (no catalog change): the three signal nodes emit ``signal``
(required key ``series``); the PySR node emits ``equation`` (required ``coef``,
``r2``) — exactly like the existing equation nodes.
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

# --- optional richer wavelet backend (Haar fallback is dependency-free) ---
try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:  # pragma: no cover - pywt simply absent
    pywt = None
    _HAS_PYWT = False

from scipy.interpolate import CubicSpline  # scipy is already a core dependency


# =============================================================================
# Singular Spectrum Analysis (pure numpy)
# =============================================================================
def _hankelize(M: np.ndarray, N: int) -> np.ndarray:
    """Diagonal-average an (L, K) matrix back into a length-N series (the SSA
    reconstruction step)."""
    L, K = M.shape
    out = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        # anti-diagonal i contributes to out[i : i+K]
        out[i:i + K] += M[i, :]
        cnt[i:i + K] += 1.0
    return out / np.maximum(cnt, 1.0)


def _ssa_one(x: np.ndarray, L: int, r: int):
    """Reconstruct series ``x`` from its top-``r`` SSA components. Returns
    (reconstruction, singular-value energy ratios)."""
    N = len(x)
    if N < 4:
        return x.copy(), np.array([1.0])
    L = int(max(2, min(L, N - 1)))
    K = N - L + 1
    traj = np.column_stack([x[i:i + L] for i in range(K)])      # (L, K)
    try:
        U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    except np.linalg.LinAlgError:                              # pragma: no cover
        return x.copy(), np.array([1.0])
    r = int(max(1, min(r, len(S))))
    recon_mat = (U[:, :r] * S[:r]) @ Vt[:r, :]                 # rank-r approx
    rec = _hankelize(recon_mat, N)
    energy = S ** 2
    ratios = energy / (energy.sum() + 1e-12)
    return rec, ratios


@register
class SSADecomposeNode(Node):
    """Singular Spectrum Analysis: split the series into trend/oscillatory/noise
    components via trajectory-matrix SVD; transform = top-component reconstruction."""
    layer = "signal"
    node_type = "ssa_decompose"
    is_transformer = True

    def __init__(self, window: int = 20, rank: int = 3, **kw):
        super().__init__(window=window, rank=rank, **kw)
        self.window = window
        self.rank = rank

    def _transform(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for j in range(X.shape[1]):
            rec, _ = _ssa_one(X[:, j], self.window, self.rank)
            cols.append(rec)
        return np.column_stack(cols)

    def _predict(self, X: np.ndarray) -> Belief:
        rec, ratios = _ssa_one(X[:, 0], self.window, self.rank)
        recon = np.column_stack([_ssa_one(X[:, j], self.window, self.rank)[0]
                                 for j in range(X.shape[1])])
        resid = float(np.mean(np.abs(X - recon)))
        top = float(ratios[:self.rank].sum()) if len(ratios) else 0.0
        return Belief("signal",
                      {"series": recon.tolist(),
                       "energy_ratios": ratios[:10].tolist(),
                       "top_component_energy": top,
                       "mean_abs_change": resid,
                       "method": "ssa"},
                      float(max(0.0, min(1.0, top))), self.name)


# =============================================================================
# Wavelet decomposition + denoise (PyWavelets if present, else pure-numpy Haar)
# =============================================================================
def _haar_dwt(x: np.ndarray):
    """One level of the (orthonormal) Haar DWT. Returns (approx, detail)."""
    if len(x) % 2:                                            # pad to even length
        x = np.append(x, x[-1])
    even, odd = x[0::2], x[1::2]
    s = np.sqrt(2.0)
    return (even + odd) / s, (even - odd) / s


def _haar_idwt(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_haar_dwt`."""
    s = np.sqrt(2.0)
    even = (approx + detail) / s
    odd = (approx - detail) / s
    out = np.empty(2 * len(approx))
    out[0::2] = even
    out[1::2] = odd
    return out


def _wavelet_denoise_haar(x: np.ndarray, level: int):
    """Multi-level Haar decompose → soft-threshold the detail bands → reconstruct.
    Returns (denoised_series_same_length, energy_per_band)."""
    N = len(x)
    if N < 4:
        return x.copy(), [1.0]
    level = int(max(1, min(level, int(np.log2(N)))))
    approx = x.astype(float).copy()
    details = []
    for _ in range(level):
        if len(approx) < 2:
            break
        approx, d = _haar_dwt(approx)
        details.append(d)
    energies = [float(np.sum(d ** 2)) for d in details] + [float(np.sum(approx ** 2))]
    # universal soft-threshold on the finest detail band (robust sigma via MAD)
    if details:
        sigma = np.median(np.abs(details[0])) / 0.6745 + 1e-12
        thr = sigma * np.sqrt(2.0 * np.log(max(2, N)))
        details = [np.sign(d) * np.maximum(np.abs(d) - thr, 0.0) for d in details]
    # reconstruct
    rec = approx
    for d in reversed(details):
        rec = _haar_idwt(rec[:len(d)], d)
    return rec[:N], energies


def _wavelet_denoise_pywt(x: np.ndarray, level: int):
    N = len(x)
    wav = "db4"
    maxlev = pywt.dwt_max_level(N, pywt.Wavelet(wav).dec_len)
    level = int(max(1, min(level, max(1, maxlev))))
    coeffs = pywt.wavedec(x, wav, level=level, mode="periodization")
    energies = [float(np.sum(np.asarray(c) ** 2)) for c in coeffs]
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
    thr = sigma * np.sqrt(2.0 * np.log(max(2, N)))
    coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wav, mode="periodization")
    return np.asarray(rec)[:N], energies


def _wavelet_one(x: np.ndarray, level: int):
    if _HAS_PYWT:
        try:
            return _wavelet_denoise_pywt(x, level)
        except Exception:                                     # pragma: no cover
            pass
    return _wavelet_denoise_haar(x, level)


@register
class WaveletDecomposeNode(Node):
    """Multi-level discrete wavelet transform with soft-threshold denoising.
    Time-frequency localization the FFT can't give; transform = denoised series."""
    layer = "signal"
    node_type = "wavelet_decompose"
    is_transformer = True

    def __init__(self, level: int = 3, **kw):
        super().__init__(level=level, **kw)
        self.level = level

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([_wavelet_one(X[:, j], self.level)[0]
                                for j in range(X.shape[1])])

    def _predict(self, X: np.ndarray) -> Belief:
        clean = self._transform(X)
        _, energies = _wavelet_one(X[:, 0], self.level)
        tot = sum(energies) + 1e-12
        bands = [float(e / tot) for e in energies]
        resid = float(np.mean(np.abs(X - clean)))
        snr = float(np.var(clean) / (np.var(X - clean) + 1e-9))
        return Belief("signal",
                      {"series": clean.tolist(),
                       "band_energy": bands,
                       "n_levels": len(energies) - 1,
                       "mean_abs_change": resid,
                       "backend": "pywt" if _HAS_PYWT else "haar",
                       "method": "wavelet"},
                      float(snr / (1.0 + snr)), self.name)


# =============================================================================
# Empirical Mode Decomposition (scipy cubic-spline sifting)
# =============================================================================
def _extrema(x: np.ndarray):
    """Indices of local maxima and minima, with the endpoints anchored so the
    cubic-spline envelopes are well-defined at the boundaries."""
    n = len(x)
    d = np.diff(x)
    maxi = np.where((np.r_[d, -1] < 0) & (np.r_[1, d] > 0))[0]
    mini = np.where((np.r_[d, 1] > 0) & (np.r_[-1, d] < 0))[0]
    maxi = np.unique(np.r_[0, maxi, n - 1])
    mini = np.unique(np.r_[0, mini, n - 1])
    return maxi, mini


def _emd(x: np.ndarray, max_imf: int = 5, max_sift: int = 10):
    """Sift ``x`` into Intrinsic Mode Functions + a final residual (trend)."""
    n = len(x)
    if n < 8:
        return [x.copy()]
    t = np.arange(n)
    imfs = []
    resid = x.astype(float).copy()
    for _ in range(max_imf):
        d = np.diff(resid)
        if np.all(d >= 0) or np.all(d <= 0):                  # monotonic → trend
            break
        h = resid.copy()
        for _ in range(max_sift):
            maxi, mini = _extrema(h)
            if len(maxi) < 3 or len(mini) < 3:
                break
            upper = CubicSpline(maxi, h[maxi])(t)
            lower = CubicSpline(mini, h[mini])(t)
            mean = 0.5 * (upper + lower)
            h_new = h - mean
            denom = float(np.sum(h ** 2)) + 1e-12
            if float(np.sum((h - h_new) ** 2)) / denom < 1e-4:
                h = h_new
                break
            h = h_new
        imfs.append(h)
        resid = resid - h
    imfs.append(resid)                                        # the trend/residual
    return imfs


@register
class EMDDecomposeNode(Node):
    """Empirical Mode Decomposition: data-adaptive split into IMFs (fast→slow) +
    a residual trend. Transform = denoised reconstruction (drops the noisiest IMF)."""
    layer = "signal"
    node_type = "emd_decompose"
    is_transformer = True

    def __init__(self, max_imf: int = 5, **kw):
        super().__init__(max_imf=max_imf, **kw)
        self.max_imf = max_imf

    def _denoise_col(self, x: np.ndarray) -> np.ndarray:
        imfs = _emd(x, max_imf=self.max_imf)
        if len(imfs) <= 1:
            return x.copy()
        return np.sum(imfs[1:], axis=0)                       # drop the finest (noise) IMF

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self._denoise_col(X[:, j]) for j in range(X.shape[1])])

    def _predict(self, X: np.ndarray) -> Belief:
        imfs = _emd(X[:, 0], max_imf=self.max_imf)
        energies = [float(np.sum(c ** 2)) for c in imfs]
        tot = sum(energies) + 1e-12
        ratios = [float(e / tot) for e in energies]
        trend = imfs[-1]
        slope = float(np.polyfit(np.arange(len(trend)), trend, 1)[0]) if len(trend) > 1 else 0.0
        recon = self._transform(X)
        resid = float(np.mean(np.abs(X - recon)))
        return Belief("signal",
                      {"series": recon.tolist(),
                       "n_imfs": max(0, len(imfs) - 1),
                       "imf_energy": ratios,
                       "trend_slope": slope,
                       "mean_abs_change": resid,
                       "method": "emd"},
                      float(np.tanh(abs(slope)) if slope else 0.5), self.name)


# =============================================================================
# PySR symbolic-regression backend (OPTIONAL — registers only if pysr present)
# =============================================================================
try:
    from pysr import PySRRegressor  # type: ignore
    _HAS_PYSR = True
except Exception:  # pragma: no cover - pysr/Julia simply absent (the common case)
    PySRRegressor = None
    _HAS_PYSR = False

PYSR_AVAILABLE = _HAS_PYSR

if _HAS_PYSR:                                                 # pragma: no cover - needs Julia

    @register
    class PySRNode(Node):
        """PySR (Julia-backed) symbolic regression of feature 0 on its lags +
        time index. Emits the standard ``equation`` belief (coef + r2 + the
        discovered expression). Heavier than the light symbolic nodes — only
        registered when ``pysr`` is importable."""
        layer = "equation"
        node_type = "pysr_symbolic"

        def __init__(self, niterations: int = 20, n_lags: int = 3, **kw):
            super().__init__(niterations=niterations, n_lags=n_lags, **kw)
            self.niterations = niterations
            self.n_lags = n_lags
            self._model = None
            self._r2 = 0.0
            self._expr = ""

        def _design(self, x: np.ndarray):
            L = self.n_lags
            rows = len(x) - L
            feats = np.column_stack([x[i:i + rows] for i in range(L)]
                                    + [np.arange(rows, dtype=float)])
            target = x[L:]
            return feats, target

        def _fit(self, X: np.ndarray, y=None) -> None:
            x = X[:, 0].astype(float)
            if len(x) <= self.n_lags + 3:
                return
            feats, target = self._design(x)
            self._model = PySRRegressor(niterations=self.niterations,
                                        binary_operators=["+", "-", "*", "/"],
                                        unary_operators=["square", "cube"],
                                        progress=False, verbosity=0,
                                        model_selection="best")
            self._model.fit(feats, target)
            pred = self._model.predict(feats)
            ss_res = float(np.sum((target - pred) ** 2))
            ss_tot = float(np.sum((target - target.mean()) ** 2)) + 1e-12
            self._r2 = max(0.0, 1.0 - ss_res / ss_tot)
            try:
                self._expr = str(self._model.sympy())
            except Exception:
                self._expr = str(getattr(self._model, "equations_", ""))[:200]

        def _predict(self, X: np.ndarray) -> Belief:
            if self._model is None:
                self._fit(X)
            return Belief("equation",
                          {"coef": [], "r2": float(self._r2),
                           "expression": self._expr, "form": "pysr"},
                          float(max(0.0, min(1.0, self._r2))), self.name)
