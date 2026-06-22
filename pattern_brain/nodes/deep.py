"""Deep / PyTorch nodes (build-tracker step 6, Phase 6b).

These are the deep-learning bank entries the light numpy/scipy/sklearn stack
can't provide: recurrent (LSTM/GRU), attention (Transformer), temporal-conv
(TCN) forecasters, and a deep autoencoder (denoise + reconstruction-anomaly) —
the start of Block-42 categories 4-6/10-12/20-21 that PyTorch unlocks.

OPTIONAL DEPENDENCY (PLAN.md §0b "add only what's proven needed"): PyTorch is
imported defensively. If it is NOT installed, this module registers NOTHING and
the light-stack bank keeps working unchanged — so the core never hard-depends on
torch. Install with ``pip install torch`` (CPU wheel is enough); the deep nodes
then appear in the bank automatically (the dashboard's `.venv` carries torch).

Domain-agnostic (Rule 23): every node consumes a generic ``(T, D)`` sequence and
emits the SAME interlingua belief types the light nodes use (`forecast`,
`denoised`, `anomaly`) — no new belief type, no catalog change, nothing
candle/order-book specific. Each wraps behind the one common Node interface, so
the Connector calls a torch node identically to a sklearn one (§0b: "the
implementation bends to fit the interface").
"""
from __future__ import annotations

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch simply absent
    torch = None
    nn = None
    _HAS_TORCH = False

TORCH_AVAILABLE = _HAS_TORCH


if _HAS_TORCH:

    def _seed():
        torch.manual_seed(0)

    def _lag_dataset(X: np.ndarray, p: int):
        """Windows of length p -> next vector. Returns tensors (N, p, D), (N, D)."""
        T = X.shape[0]
        Xw = np.stack([X[i:i + p] for i in range(T - p)])      # (N, p, D)
        Yw = X[p:T]                                            # (N, D)
        return (torch.tensor(Xw, dtype=torch.float32),
                torch.tensor(Yw, dtype=torch.float32))

    class _TorchForecaster(Node):
        """Base: lag-embedding one-step forecaster trained briefly on the window.
        Subclasses build ``self._net(D)`` mapping (batch, p, D) -> (batch, D)."""
        layer = "sequence"

        def __init__(self, lag: int = 8, epochs: int = 60, hidden: int = 16, **kw):
            super().__init__(lag=lag, epochs=epochs, hidden=hidden, **kw)
            self.lag = lag
            self.epochs = epochs
            self.hidden = hidden

        def _build(self, D: int, p: int):  # pragma: no cover - overridden
            raise NotImplementedError

        def _extra_loss(self, pred, Xn, Yn):
            """Optional regularizer added to the MSE data loss (default: none).
            Overridden by physics-informed nodes (PINN) to inject a prior."""
            return 0.0

        def _predict(self, X: np.ndarray) -> Belief:
            T, D = X.shape
            p = max(2, min(self.lag, max(2, T // 4)))
            if T <= p + 2:
                nxt = X[-1]
                return _forecast(nxt, 0.0, self.name, p)
            _seed()
            Xw, Yw = _lag_dataset(X, p)
            # normalize for stable training
            mu = Xw.mean(dim=(0, 1), keepdim=True)
            sd = Xw.std(dim=(0, 1), keepdim=True) + 1e-6
            Xn = (Xw - mu) / sd
            Yn = (Yw - mu.squeeze(1)) / sd.squeeze(1)
            net = self._build(D, p)
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            lossf = nn.MSELoss()
            net.train()
            last = 0.0
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = net(Xn)
                data_loss = lossf(pred, Yn)
                loss = data_loss + self._extra_loss(pred, Xn, Yn)
                loss.backward()
                opt.step()
                last = float(data_loss.item())
            net.eval()
            with torch.no_grad():
                win = torch.tensor(X[T - p:T], dtype=torch.float32).unsqueeze(0)
                win = (win - mu) / sd
                out = net(win).squeeze(0) * sd.squeeze(1).squeeze(0) + mu.squeeze(1).squeeze(0)
                nxt = out.numpy()
            return _forecast(nxt, last, self.name, p)

    @register
    class LSTMForecastNode(_TorchForecaster):
        """LSTM recurrent one-step forecaster (Block-42 RNN family)."""
        node_type = "lstm_forecaster"

        def _build(self, D, p):
            return _RecurrentNet(D, self.hidden, kind="lstm")

    @register
    class GRUForecastNode(_TorchForecaster):
        """GRU recurrent one-step forecaster."""
        node_type = "gru_forecaster"

        def _build(self, D, p):
            return _RecurrentNet(D, self.hidden, kind="gru")

    @register
    class TransformerForecastNode(_TorchForecaster):
        """Transformer-encoder one-step forecaster (attention over the window)."""
        node_type = "transformer_forecaster"

        def _build(self, D, p):
            return _TransformerNet(D, self.hidden)

    @register
    class TCNForecastNode(_TorchForecaster):
        """Temporal Convolutional Network forecaster (dilated 1-D convolutions)."""
        node_type = "tcn_forecaster"

        def _build(self, D, p):
            return _TCNNet(D, self.hidden)

    @register
    class DeepAutoencoderNode(Node):
        """Deep MLP autoencoder denoiser: reconstruct each row through a bottleneck;
        the reconstruction is the cleaned series (emits 'denoised')."""
        layer = "noise"
        node_type = "deep_autoencoder"
        is_transformer = True

        def __init__(self, hidden: int = 8, bottleneck: int = 2, epochs: int = 120, **kw):
            super().__init__(hidden=hidden, bottleneck=bottleneck, epochs=epochs, **kw)
            self.hidden = hidden
            self.bottleneck = bottleneck
            self.epochs = epochs
            self._recon = None

        def _fit(self, X, y=None):
            self._recon = _autoencode(X, self.hidden, self.bottleneck, self.epochs)

        def _transform(self, X: np.ndarray) -> np.ndarray:
            if self._recon is None or self._recon.shape != X.shape:
                self._recon = _autoencode(X, self.hidden, self.bottleneck, self.epochs)
            return self._recon

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean.tolist(), "residual": resid},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    @register
    class AutoencoderAnomalyNode(Node):
        """Deep-autoencoder reconstruction-error anomaly detector (emits 'anomaly')."""
        layer = "noise"
        node_type = "autoencoder_anomaly"

        def __init__(self, hidden: int = 8, bottleneck: int = 2, epochs: int = 120,
                     quantile: float = 0.95, **kw):
            super().__init__(hidden=hidden, bottleneck=bottleneck, epochs=epochs,
                             quantile=quantile, **kw)
            self.hidden = hidden
            self.bottleneck = bottleneck
            self.epochs = epochs
            self.quantile = quantile

        def _predict(self, X: np.ndarray) -> Belief:
            recon = _autoencode(X, self.hidden, self.bottleneck, self.epochs)
            score = np.abs(X - recon).mean(axis=1)
            thr = float(np.quantile(score, self.quantile))
            flags = (score > thr).astype(int)
            frac = float(flags.mean())
            return Belief("anomaly",
                          {"n_anomalies": int(flags.sum()), "fraction": frac,
                           "scores": score.tolist(), "flags": flags.tolist()},
                          float(min(1.0, frac * 3)), self.name)

    # ------------------------------------------------------------------ nets
    class _RecurrentNet(nn.Module):
        def __init__(self, D, hidden, kind="lstm"):
            super().__init__()
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(input_size=D, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    class _TransformerNet(nn.Module):
        def __init__(self, D, hidden):
            super().__init__()
            self.proj = nn.Linear(D, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=1,
                                               dim_feedforward=hidden * 2,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):
            h = self.enc(self.proj(x))
            return self.head(h[:, -1, :])

    class _TCNNet(nn.Module):
        def __init__(self, D, hidden):
            super().__init__()
            self.c1 = nn.Conv1d(D, hidden, kernel_size=2, padding=1, dilation=1)
            self.c2 = nn.Conv1d(hidden, hidden, kernel_size=2, padding=2, dilation=2)
            self.relu = nn.ReLU()
            self.head = nn.Linear(hidden, D)

        def forward(self, x):                       # x: (B, p, D)
            z = x.transpose(1, 2)                   # (B, D, p)
            z = self.relu(self.c1(z))
            z = self.relu(self.c2(z))
            return self.head(z[:, :, -1])           # last time-step

    def _autoencode(X: np.ndarray, hidden: int, bottleneck: int, epochs: int) -> np.ndarray:
        """Train a tiny MLP autoencoder on the rows and return the reconstruction."""
        _seed()
        D = X.shape[1]
        mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
        Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
        bn = max(1, min(bottleneck, D))
        net = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, bn),
            nn.ReLU(), nn.Linear(bn, hidden), nn.ReLU(), nn.Linear(hidden, D),
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        lossf = nn.MSELoss()
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(Xn), Xn)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            recon = net(Xn).numpy() * sd + mu
        return recon

    def _forecast(next_vec, loss, name, lag):
        next_vec = np.asarray(next_vec, float).ravel()
        conf = float(1.0 / (1.0 + max(0.0, loss)))
        return Belief("forecast",
                      {"next_vector": next_vec.tolist(), "estimate": float(next_vec[0]),
                       "train_loss": float(loss), "lag": int(lag)},
                      max(0.0, min(1.0, conf)), name)


# ==========================================================================
# Build step 6 (Phase 6b, batch 2) — more deep categories: feed-forward + a
# diagonal state-space (S4D-lite) forecaster, a variational autoencoder
# (deep generative) denoiser/anomaly, and a graph-conv (GNN) denoiser.
# Re-enters the torch guard so nothing is defined when torch is absent.
# ==========================================================================
if _HAS_TORCH:

    @register
    class MLPForecastNode(_TorchForecaster):
        """Feed-forward MLP forecaster over the flattened lag window (deep, torch)."""
        node_type = "mlp_deep_forecaster"

        def _build(self, D, p):
            return _MLPNet(D, p, self.hidden)

    @register
    class SSMForecastNode(_TorchForecaster):
        """Diagonal state-space (S4D-lite) forecaster — a learned linear recurrence
        scanned over the window (the SSM/Mamba family, simplified)."""
        node_type = "ssm_forecaster"

        def _build(self, D, p):
            return _DiagSSMNet(D, self.hidden)

    @register
    class VAEDenoiseNode(Node):
        """Variational-autoencoder reconstruction denoiser (deep generative)."""
        layer = "noise"
        node_type = "vae_denoise"
        is_transformer = True

        def __init__(self, hidden: int = 8, latent: int = 2, epochs: int = 150, **kw):
            super().__init__(hidden=hidden, latent=latent, epochs=epochs, **kw)
            self.hidden = hidden; self.latent = latent; self.epochs = epochs
            self._recon = None

        def _fit(self, X, y=None):
            self._recon = _vae(X, self.hidden, self.latent, self.epochs)

        def _transform(self, X: np.ndarray) -> np.ndarray:
            if self._recon is None or self._recon.shape != X.shape:
                self._recon = _vae(X, self.hidden, self.latent, self.epochs)
            return self._recon

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean.tolist(), "residual": resid},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    @register
    class VAEAnomalyNode(Node):
        """VAE reconstruction-error anomaly detector (deep generative)."""
        layer = "noise"
        node_type = "vae_anomaly"

        def __init__(self, hidden: int = 8, latent: int = 2, epochs: int = 150,
                     quantile: float = 0.95, **kw):
            super().__init__(hidden=hidden, latent=latent, epochs=epochs,
                             quantile=quantile, **kw)
            self.hidden = hidden; self.latent = latent
            self.epochs = epochs; self.quantile = quantile

        def _predict(self, X: np.ndarray) -> Belief:
            recon = _vae(X, self.hidden, self.latent, self.epochs)
            score = np.abs(X - recon).mean(axis=1)
            thr = float(np.quantile(score, self.quantile))
            flags = (score > thr).astype(int)
            frac = float(flags.mean())
            return Belief("anomaly",
                          {"n_anomalies": int(flags.sum()), "fraction": frac,
                           "scores": score.tolist(), "flags": flags.tolist()},
                          float(min(1.0, frac * 3)), self.name)

    @register
    class GCNDenoiseNode(Node):
        """Graph-convolutional-network denoiser (GNN): smooth rows over a k-NN
        graph of timesteps via a trained 2-layer GCN autoencoder."""
        layer = "noise"
        node_type = "gcn_denoise"
        is_transformer = True

        def __init__(self, hidden: int = 8, k: int = 5, epochs: int = 100, **kw):
            super().__init__(hidden=hidden, k=k, epochs=epochs, **kw)
            self.hidden = hidden; self.k = k; self.epochs = epochs
            self._recon = None

        def _fit(self, X, y=None):
            self._recon = _gcn_denoise(X, self.hidden, self.k, self.epochs)

        def _transform(self, X: np.ndarray) -> np.ndarray:
            if self._recon is None or self._recon.shape != X.shape:
                self._recon = _gcn_denoise(X, self.hidden, self.k, self.epochs)
            return self._recon

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean.tolist(), "residual": resid},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    # ------------------------------------------------------------------ nets
    class _MLPNet(nn.Module):
        def __init__(self, D, p, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(p * D, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, D))

        def forward(self, x):                 # x: (B, p, D)
            return self.net(x)

    class _DiagSSMNet(nn.Module):
        """Diagonal linear recurrence h_t = a*h_{t-1} + B x_t ; y = C h_T + Dx_T."""
        def __init__(self, D, hidden):
            super().__init__()
            self.loga = nn.Parameter(torch.zeros(hidden))   # a = sigmoid(loga) in (0,1)
            self.B = nn.Linear(D, hidden, bias=False)
            self.C = nn.Linear(hidden, D)
            self.skip = nn.Linear(D, D)

        def forward(self, x):                 # x: (B, p, D)
            a = torch.sigmoid(self.loga)
            h = torch.zeros(x.shape[0], a.shape[0])
            for t in range(x.shape[1]):
                h = a * h + self.B(x[:, t, :])
            return self.C(h) + self.skip(x[:, -1, :])

    def _vae(X, hidden, latent, epochs):
        _seed()
        D = X.shape[1]
        mu_ = X.mean(0); sd_ = X.std(0) + 1e-6
        Xn = torch.tensor((X - mu_) / sd_, dtype=torch.float32)
        lat = max(1, min(latent, D))
        enc = nn.Sequential(nn.Linear(D, hidden), nn.ReLU())
        fmu = nn.Linear(hidden, lat); flv = nn.Linear(hidden, lat)
        dec = nn.Sequential(nn.Linear(lat, hidden), nn.ReLU(), nn.Linear(hidden, D))
        params = (list(enc.parameters()) + list(fmu.parameters())
                  + list(flv.parameters()) + list(dec.parameters()))
        opt = torch.optim.Adam(params, lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            h = enc(Xn); m = fmu(h); lv = torch.clamp(flv(h), -8, 8)
            z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
            recon = dec(z)
            loss = ((recon - Xn) ** 2).mean() - 0.0005 * torch.mean(1 + lv - m ** 2 - lv.exp())
            loss.backward(); opt.step()
        with torch.no_grad():
            recon = dec(fmu(enc(Xn))).numpy() * sd_ + mu_
        return recon

    def _gcn_denoise(X, hidden, k, epochs):
        _seed()
        from sklearn.neighbors import kneighbors_graph
        T, D = X.shape
        kk = max(1, min(k, T - 1))
        A = kneighbors_graph(X, kk, mode="connectivity", include_self=True).toarray()
        A = np.maximum(A, A.T)
        deg = A.sum(1)
        dinv = np.diag(1.0 / np.sqrt(deg + 1e-9))
        Ah = torch.tensor(dinv @ A @ dinv, dtype=torch.float32)
        mu_ = X.mean(0); sd_ = X.std(0) + 1e-6
        Xt = torch.tensor((X - mu_) / sd_, dtype=torch.float32)
        W1 = nn.Linear(D, hidden); W2 = nn.Linear(hidden, D)
        opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            H = torch.relu(Ah @ W1(Xt))
            Xr = Ah @ W2(H)
            loss = ((Xr - Xt) ** 2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            recon = (Ah @ W2(torch.relu(Ah @ W1(Xt)))).numpy() * sd_ + mu_
        return recon


# ==========================================================================
# Phase 7f — deep tier (PLAN.md §11, Block 57): the coverage gaps the meter
# flagged lowest — Transformers (PatchTST), SSM/Mamba (selective SSM), N-BEATS,
# Graph (GAT + GraphSAGE), Generative (diffusion-lite), deep-RL (DQN + PPO).
# All re-enter the torch guard so nothing defines when torch is absent.
# ==========================================================================
if _HAS_TORCH:

    # ---- forecasters (reuse the _TorchForecaster base: build (B,p,D)->(B,D)) ----
    class _PatchTSTNet(nn.Module):
        """PatchTST: split the window into patches, linear-embed each, transformer
        over patches, head -> next vector."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.ps = 2 if p >= 2 else 1
            self.npatch = max(1, p // self.ps)
            self.embed = nn.Linear(self.ps * D, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=1,
                                               dim_feedforward=hidden * 2, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden, D)
            self.p, self.D = p, D

        def forward(self, x):                       # (B, p, D)
            B = x.shape[0]
            usable = self.npatch * self.ps
            xp = x[:, :usable, :].reshape(B, self.npatch, self.ps * self.D)
            h = self.enc(self.embed(xp))
            return self.head(h[:, -1, :])

    class _MambaLiteNet(nn.Module):
        """Selective state-space (Mamba/S6-inspired, pure-torch — NOT the CUDA kernel):
        a diagonal recurrence whose decay/input gates depend on the input (selection)."""
        def __init__(self, D, hidden):
            super().__init__()
            self.proj = nn.Linear(D, hidden)
            self.dt = nn.Linear(hidden, hidden)      # input-dependent step (selective)
            self.Bp = nn.Linear(hidden, hidden)
            self.Cp = nn.Linear(hidden, hidden)
            self.A = nn.Parameter(torch.randn(hidden) * 0.1)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):                        # (B, p, D)
            B, p, _ = x.shape
            u = torch.tanh(self.proj(x))             # (B,p,H)
            h = torch.zeros(B, u.shape[-1])
            decay = -torch.nn.functional.softplus(self.A)   # negative -> stable
            for t in range(p):
                dt = torch.sigmoid(self.dt(u[:, t]))         # selective step size
                a = torch.exp(decay * dt)                    # input-dependent decay
                h = a * h + dt * self.Bp(u[:, t])
                y = self.Cp(h)
            return self.head(y)

    class _NBeatsNet(nn.Module):
        """N-BEATS-lite: stacked fully-connected blocks with backcast/forecast and
        residual stacking over the flattened lag window."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.p, self.D, self.insz = p, D, p * D
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "fc": nn.Sequential(nn.Linear(self.insz, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU()),
                    "back": nn.Linear(hidden, self.insz),
                    "fore": nn.Linear(hidden, D),
                }) for _ in range(2)])

        def forward(self, x):                        # (B, p, D)
            r = x.reshape(x.shape[0], -1)
            fc_sum = 0.0
            for blk in self.blocks:
                h = blk["fc"](r)
                r = r - blk["back"](h)               # residual backcast
                fc_sum = fc_sum + blk["fore"](h)
            return fc_sum

    @register
    class PatchTSTForecastNode(_TorchForecaster):
        """PatchTST patch-transformer forecaster (Transformers family)."""
        node_type = "patchtst_forecaster"

        def _build(self, D, p):
            return _PatchTSTNet(D, self.hidden, p)

    @register
    class MambaForecastNode(_TorchForecaster):
        """Selective state-space (Mamba-lite) forecaster (SSM family)."""
        node_type = "mamba_forecaster"

        def _build(self, D, p):
            return _MambaLiteNet(D, self.hidden)

    @register
    class NBeatsForecastNode(_TorchForecaster):
        """N-BEATS-lite deep forecaster (backcast/forecast residual stacks)."""
        node_type = "nbeats_forecaster"

        def _build(self, D, p):
            return _NBeatsNet(D, self.hidden, p)

    # ---- graph denoisers (kNN-of-timesteps graph) ----
    def _knn_adj(X: np.ndarray, k: int = 4):
        from scipy.spatial.distance import cdist
        d = cdist(X, X)
        np.fill_diagonal(d, np.inf)
        k = min(k, X.shape[0] - 1)
        idx = np.argsort(d, axis=1)[:, :k]
        A = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
        for i, nb in enumerate(idx):
            A[i, nb] = 1.0
        A = np.maximum(A, A.T)
        A = A + np.eye(X.shape[0], dtype=np.float32)     # self-loops
        return A / A.sum(1, keepdims=True)

    def _graph_reconstruct(X, attention, hidden=12, epochs=80, adj_fn=None):
        _seed()
        T, D = X.shape
        if T < 6:
            return X.copy()
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        A = torch.tensor((adj_fn or _knn_adj)(X), dtype=torch.float32)
        W1 = nn.Linear(D, hidden); W2 = nn.Linear(hidden, D)
        att = nn.Linear(2 * hidden, 1) if attention else None
        params = list(W1.parameters()) + list(W2.parameters()) + (list(att.parameters()) if att else [])
        opt = torch.optim.Adam(params, lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            h = torch.relu(W1(Xt))
            if att is not None:                          # GAT: attention-weighted aggregation
                T_ = h.shape[0]
                hi = h.unsqueeze(1).expand(T_, T_, hidden)
                hj = h.unsqueeze(0).expand(T_, T_, hidden)
                e = torch.nn.functional.leaky_relu(att(torch.cat([hi, hj], -1)).squeeze(-1))
                e = e.masked_fill(A == 0, -1e9)
                alpha = torch.softmax(e, dim=1)
                agg = alpha @ h
            else:                                        # GraphSAGE: mean aggregation
                agg = A @ h
            recon = W2(torch.relu(agg))
            loss = ((recon - Xt) ** 2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            h = torch.relu(W1(Xt))
            if att is not None:
                T_ = h.shape[0]
                e = torch.nn.functional.leaky_relu(
                    att(torch.cat([h.unsqueeze(1).expand(T_, T_, hidden),
                                   h.unsqueeze(0).expand(T_, T_, hidden)], -1)).squeeze(-1))
                e = e.masked_fill(A == 0, -1e9)
                agg = torch.softmax(e, 1) @ h
            else:
                agg = A @ h
            recon = (W2(torch.relu(agg)).numpy()) * sd + mu
        return recon

    class _GraphDenoise(Node):
        layer = "noise"
        is_transformer = True
        _attention = False

        def __init__(self, hidden: int = 12, epochs: int = 80, **kw):
            super().__init__(hidden=hidden, epochs=epochs, **kw)
            self.hidden, self.epochs = hidden, epochs

        def _transform(self, X: np.ndarray) -> np.ndarray:
            return _graph_reconstruct(X, self._attention, self.hidden, self.epochs)

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean[:, 0].tolist(), "residual": resid},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    @register
    class GATDenoiseNode(_GraphDenoise):
        """Graph Attention Network denoiser over a kNN-of-timesteps graph."""
        node_type = "gat_denoise"
        _attention = True

    @register
    class GraphSAGEDenoiseNode(_GraphDenoise):
        """GraphSAGE (mean-aggregation) denoiser over a kNN-of-timesteps graph."""
        node_type = "graphsage_denoise"
        _attention = False

    # ---- generative: diffusion-lite denoiser ----
    @register
    class DiffusionDenoiseNode(Node):
        """Diffusion-inspired denoiser: train a net to predict the Gaussian noise
        added to each row (DDPM reverse step), then subtract it (emits 'denoised')."""
        layer = "noise"
        node_type = "diffusion_denoise"
        is_transformer = True

        def __init__(self, hidden: int = 16, epochs: int = 120, noise: float = 0.4, **kw):
            super().__init__(hidden=hidden, epochs=epochs, noise=noise, **kw)
            self.hidden, self.epochs, self.noise = hidden, epochs, noise

        def _transform(self, X: np.ndarray) -> np.ndarray:
            _seed()
            T, D = X.shape
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
            net = nn.Sequential(nn.Linear(D, self.hidden), nn.ReLU(),
                                nn.Linear(self.hidden, D))
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                eps = torch.randn_like(Xt) * self.noise
                loss = ((net(Xt + eps) - eps) ** 2).mean()  # predict the noise
                loss.backward(); opt.step()
            with torch.no_grad():
                return (Xt - net(Xt)).numpy() * sd + mu

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean[:, 0].tolist(), "residual": resid,
                                       "model": "diffusion-lite"},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    # ---- deep RL policies (emit 'action') ----
    def _rl_dataset(x, p):
        """States = flattened windows; per-action 1-step direction rewards
        (action 0=short,1=flat,2=long)."""
        T = len(x)
        S, R = [], []
        for t in range(p, T - 1):
            S.append(x[t - p:t])
            up = np.sign(x[t + 1] - x[t])
            R.append([-up, 0.0, up])                       # short, flat, long
        return np.array(S, np.float32), np.array(R, np.float32)

    @register
    class DQNPolicyNode(Node):
        """Deep Q-Network (1-step) over windowed states; actions short/flat/long.
        Q(s,a) regresses the immediate directional reward; emits the greedy action."""
        layer = "rl"
        node_type = "dqn_policy"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 60, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, **kw)
            self.lag, self.hidden, self.epochs = lag, hidden, epochs

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x = X[:, 0].astype(float)
            p = max(2, min(self.lag, max(2, len(x) // 4)))
            S, R = _rl_dataset(x, p)
            if len(S) < 4:
                return Belief("action", {"action": 1, "q_values": [0, 0, 0]}, 0.1, self.name)
            mu, sd = S.mean(), S.std() + 1e-6
            St = torch.tensor((S - mu) / sd, dtype=torch.float32)
            Rt = torch.tensor(R, dtype=torch.float32)
            net = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad(); loss = ((net(St) - Rt) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad():
                q = net(torch.tensor(((x[-p:] - mu) / sd), dtype=torch.float32)).numpy()
            a = int(np.argmax(q))
            return Belief("action", {"action": a - 1, "q_values": [float(v) for v in q],
                                     "labels": ["short", "flat", "long"], "model": "DQN"},
                          float(max(0.0, min(1.0, abs(q[a]) ))), self.name)

    @register
    class PPOPolicyNode(Node):
        """PPO-lite policy-gradient over windowed states; actions short/flat/long.
        Emits the action distribution + chosen action."""
        layer = "rl"
        node_type = "ppo_policy"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 60, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, **kw)
            self.lag, self.hidden, self.epochs = lag, hidden, epochs

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x = X[:, 0].astype(float)
            p = max(2, min(self.lag, max(2, len(x) // 4)))
            S, R = _rl_dataset(x, p)
            if len(S) < 4:
                return Belief("action", {"action": 1, "policy": [0.33, 0.34, 0.33]}, 0.1, self.name)
            mu, sd = S.mean(), S.std() + 1e-6
            St = torch.tensor((S - mu) / sd, dtype=torch.float32)
            Rt = torch.tensor(R, dtype=torch.float32)
            adv = Rt - Rt.mean(0, keepdim=True)               # advantage baseline
            net = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                logp = torch.log_softmax(net(St), dim=1)
                loss = -(logp * adv).mean()                   # policy gradient (advantage-weighted)
                loss.backward(); opt.step()
            with torch.no_grad():
                probs = torch.softmax(net(torch.tensor(((x[-p:] - mu) / sd), dtype=torch.float32)), 0).numpy()
            a = int(np.argmax(probs))
            return Belief("action", {"action": a - 1, "policy": [float(v) for v in probs],
                                     "labels": ["short", "flat", "long"], "model": "PPO"},
                          float(max(0.0, min(1.0, probs[a]))), self.name)


# ==========================================================================
# Phase 7f — BATCH 2 (PLAN.md §11): the deep-tier breadth fill. All pure-torch
# (no new heavy deps): advanced Transformers (Informer/Autoformer/FEDformer),
# more state-space (S4/S5), Fourier Neural Operator, Neural-ODE + PINN,
# Temporal-GNN, energy nets (modern Hopfield / RBM), deep generative
# (GAN / normalizing-flow anomaly), and continuous-control-family deep-RL
# (SAC / TD3 / A3C, discrete short/flat/long adaptations). Each behind the
# generic Node interface, emitting EXISTING interlingua types (Rule 23 / Block 46).
# Deferred (need external deps/downloads, §0b): real Mamba/S4 CUDA kernels
# (mamba-ssm) and foundation TS models (Chronos / TimesFM).
# ==========================================================================
if _HAS_TORCH:

    def _series_decomp(x, k):
        """Autoformer/FEDformer trend-seasonal decomposition: trend = causal-padded
        moving average (odd kernel), seasonal = x - trend. x: (B, p, D)."""
        p = x.shape[1]
        k = max(3, min(k, p))
        k = k if k % 2 == 1 else k - 1
        pad = k // 2
        xt = nn.functional.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        trend = nn.functional.avg_pool1d(xt, k, stride=1).transpose(1, 2)[:, :p, :]
        return trend, x - trend

    # ---------- advanced Transformers ----------
    class _InformerNet(nn.Module):
        """Informer ProbSparse-lite: only the top-u 'active' queries (highest
        sparsity score max-mean) get full attention; the rest collapse to the mean
        value — the O(L log L) sparse-attention idea, in miniature."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.proj = nn.Linear(D, hidden)
            self.q = nn.Linear(hidden, hidden)
            self.k = nn.Linear(hidden, hidden)
            self.v = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, D)
            self.u = max(1, int(np.ceil(np.log2(p + 1))))

        def forward(self, x):                              # (B, p, D)
            h = torch.tanh(self.proj(x))
            Q, K, V = self.q(h), self.k(h), self.v(h)
            scores = Q @ K.transpose(1, 2) / (K.shape[-1] ** 0.5)   # (B,p,p)
            M = scores.max(-1).values - scores.mean(-1)            # query sparsity (B,p)
            u = min(self.u, scores.shape[1])
            topu = torch.topk(M, u, dim=1).indices                # (B,u)
            full = torch.softmax(scores, -1) @ V                  # (B,p,H)
            meanV = V.mean(1, keepdim=True).expand_as(full)
            mask = torch.zeros_like(M).scatter(1, topu, 1.0).unsqueeze(-1)
            out = mask * full + (1 - mask) * meanV
            return self.head(out[:, -1, :])

    class _AutoformerNet(nn.Module):
        """Autoformer-lite: series-decomposition block (its signature) — a
        transformer encodes the seasonal part, a linear head extrapolates the trend,
        and the two are summed."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.k = max(3, p // 4)
            self.proj = nn.Linear(D, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=1,
                                               dim_feedforward=hidden * 2, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden, D)
            self.trend_head = nn.Linear(D, D)

        def forward(self, x):                              # (B, p, D)
            trend, seasonal = _series_decomp(x, self.k)
            h = self.enc(self.proj(seasonal))
            return self.head(h[:, -1, :]) + self.trend_head(trend[:, -1, :])

    class _FEDformerNet(nn.Module):
        """FEDformer-lite: decomposition + a frequency-enhanced block — the seasonal
        part is mixed by learnable complex weights on its lowest Fourier modes
        (rFFT -> per-mode complex multiply -> iRFFT), then summed with the trend."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.k = max(3, p // 4)
            self.modes = max(1, min(p // 2, 6))
            self.proj = nn.Linear(D, hidden)
            self.wr = nn.Parameter(torch.randn(self.modes, hidden) * 0.05)
            self.wi = nn.Parameter(torch.randn(self.modes, hidden) * 0.05)
            self.head = nn.Linear(hidden, D)
            self.trend_head = nn.Linear(D, D)

        def forward(self, x):                              # (B, p, D)
            trend, seasonal = _series_decomp(x, self.k)
            h = torch.tanh(self.proj(seasonal))           # (B,p,H)
            f = torch.fft.rfft(h, dim=1)                  # (B,F,H) complex
            m = min(self.modes, f.shape[1])
            w = torch.complex(self.wr[:m], self.wi[:m]).unsqueeze(0)
            fout = torch.zeros_like(f)
            fout[:, :m] = f[:, :m] * w
            hf = torch.fft.irfft(fout, n=h.shape[1], dim=1)
            return self.head(hf[:, -1, :]) + self.trend_head(trend[:, -1, :])

    # ---------- more state-space ----------
    class _S4Net(nn.Module):
        """S4D-style diagonal COMPLEX state-space (LTI, non-selective) — distinct
        from the selective Mamba-lite and the real-diagonal ssm_forecaster: a stable
        complex pole per channel (decay + oscillation), HiPPO-inspired init."""
        def __init__(self, D, hidden):
            super().__init__()
            self.enc = nn.Linear(D, hidden)
            self.log_neg_re = nn.Parameter(torch.log(torch.linspace(0.1, 1.0, hidden)))
            self.theta = nn.Parameter(torch.linspace(0.0, 3.0, hidden))
            self.Bp = nn.Parameter(torch.ones(hidden) * 0.1)
            self.Cr = nn.Parameter(torch.randn(hidden) * 0.1)
            self.Ci = nn.Parameter(torch.randn(hidden) * 0.1)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):                              # (B, p, D)
            B, p, _ = x.shape
            u = torch.tanh(self.enc(x))
            mag = torch.exp(-torch.exp(self.log_neg_re))   # |pole| in (0,1) -> stable
            cos, sin = torch.cos(self.theta) * mag, torch.sin(self.theta) * mag
            hr = torch.zeros(B, u.shape[-1]); hi = torch.zeros(B, u.shape[-1])
            for t in range(p):
                inp = self.Bp * u[:, t]
                hr, hi = cos * hr - sin * hi + inp, sin * hr + cos * hi
            return self.head(self.Cr * hr - self.Ci * hi)  # Re(C x)

    class _S5Net(nn.Module):
        """S5-style: a single MIMO diagonal SSM (one shared state with full
        input/output mixing) rather than S4's bank of independent SISO SSMs."""
        def __init__(self, D, hidden):
            super().__init__()
            self.Bm = nn.Linear(D, hidden, bias=False)
            self.log_dec = nn.Parameter(torch.log(torch.linspace(0.1, 0.9, hidden)))
            self.Cm = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):                              # (B, p, D)
            B, p, _ = x.shape
            a = torch.exp(-torch.exp(self.log_dec))
            h = torch.zeros(B, a.shape[0])
            for t in range(p):
                h = a * h + self.Bm(x[:, t])
            return self.head(torch.tanh(self.Cm(h)))

    class _FNONet(nn.Module):
        """Fourier Neural Operator: lift -> a spectral conv (keep low modes, learnable
        per-mode complex weights, iFFT) + a pointwise residual -> project. Learns the
        forecasting map as an operator in frequency space."""
        def __init__(self, D, hidden, p):
            super().__init__()
            self.modes = max(1, min(p // 2, 8))
            self.lift = nn.Linear(D, hidden)
            self.wr = nn.Parameter(torch.randn(self.modes, hidden) * 0.05)
            self.wi = nn.Parameter(torch.randn(self.modes, hidden) * 0.05)
            self.w = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, D)

        def forward(self, x):                              # (B, p, D)
            h = self.lift(x)
            f = torch.fft.rfft(h, dim=1)
            m = min(self.modes, f.shape[1])
            w = torch.complex(self.wr[:m], self.wi[:m]).unsqueeze(0)
            fout = torch.zeros_like(f)
            fout[:, :m] = f[:, :m] * w
            spec = torch.fft.irfft(fout, n=h.shape[1], dim=1)
            h2 = torch.relu(spec + self.w(h))
            return self.head(h2[:, -1, :])

    class _NeuralODENet(nn.Module):
        """Latent Neural ODE: a GRU encodes the window to z0, then dz/dt = f(z) is
        integrated by explicit fixed-step Euler (no torchdiffeq dep), decode -> next."""
        def __init__(self, D, hidden):
            super().__init__()
            self.enc = nn.GRU(D, hidden, batch_first=True)
            self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
            self.head = nn.Linear(hidden, D)
            self.steps, self.dt = 4, 0.25

        def forward(self, x):                              # (B, p, D)
            _, h = self.enc(x)
            z = h[-1]
            for _ in range(self.steps):
                z = z + self.dt * self.f(z)
            return self.head(z)

    class _PINNNet(nn.Module):
        def __init__(self, D, hidden, p):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(p * D, hidden), nn.Tanh(), nn.Linear(hidden, D))

        def forward(self, x):                              # (B, p, D)
            return self.net(x.reshape(x.shape[0], -1))

    @register
    class InformerForecastNode(_TorchForecaster):
        """Informer (ProbSparse-attention) forecaster (Transformers family)."""
        node_type = "informer_forecaster"

        def _build(self, D, p):
            return _InformerNet(D, self.hidden, p)

    @register
    class AutoformerForecastNode(_TorchForecaster):
        """Autoformer (series-decomposition) forecaster (Transformers family)."""
        node_type = "autoformer_forecaster"

        def _build(self, D, p):
            return _AutoformerNet(D, self.hidden, p)

    @register
    class FEDformerForecastNode(_TorchForecaster):
        """FEDformer (frequency-enhanced decomposition) forecaster (Transformers)."""
        node_type = "fedformer_forecaster"

        def _build(self, D, p):
            return _FEDformerNet(D, self.hidden, p)

    @register
    class S4ForecastNode(_TorchForecaster):
        """S4 diagonal complex state-space forecaster (State-space family)."""
        node_type = "s4_forecaster"

        def _build(self, D, p):
            return _S4Net(D, self.hidden)

    @register
    class S5ForecastNode(_TorchForecaster):
        """S5 MIMO diagonal state-space forecaster (State-space family)."""
        node_type = "s5_forecaster"

        def _build(self, D, p):
            return _S5Net(D, self.hidden)

    @register
    class FNOForecastNode(_TorchForecaster):
        """Fourier Neural Operator forecaster (operator learning in frequency space)."""
        node_type = "fno_forecaster"

        def _build(self, D, p):
            return _FNONet(D, self.hidden, p)

    @register
    class NeuralODEForecastNode(_TorchForecaster):
        """Latent Neural-ODE forecaster (Euler-integrated continuous dynamics)."""
        node_type = "neural_ode_forecaster"

        def _build(self, D, p):
            return _NeuralODENet(D, self.hidden)

    @register
    class PINNForecastNode(_TorchForecaster):
        """Physics-Informed forecaster: MLP over the lag window + a finite-difference
        curvature prior (penalize large 2nd differences) added to the data loss — a
        generic 'smoothness physics' that regularizes the one-step prediction."""
        node_type = "pinn_forecaster"

        def __init__(self, phys: float = 0.1, **kw):
            super().__init__(**kw)
            self.phys = phys
            self.params["phys"] = phys

        def _build(self, D, p):
            return _PINNNet(D, self.hidden, p)

        def _extra_loss(self, pred, Xn, Yn):
            last2, last1 = Xn[:, -2, :], Xn[:, -1, :]
            return self.phys * ((pred - 2 * last1 + last2) ** 2).mean()

    # ---------- Temporal-GNN (graph over a temporal + kNN adjacency) ----------
    def _temporal_knn_adj(X, k=4):
        from scipy.spatial.distance import cdist
        T = X.shape[0]
        d = cdist(X, X); np.fill_diagonal(d, np.inf)
        k = min(k, T - 1)
        idx = np.argsort(d, axis=1)[:, :k]
        A = np.zeros((T, T), dtype=np.float32)
        for i, nb in enumerate(idx):
            A[i, nb] = 1.0
        A = np.maximum(A, A.T)
        for i in range(T - 1):                              # temporal chain edges
            A[i, i + 1] = A[i + 1, i] = 1.0
        A = A + np.eye(T, dtype=np.float32)
        return A / A.sum(1, keepdims=True)

    @register
    class TemporalGNNDenoiseNode(_GraphDenoise):
        """Temporal GNN denoiser: a graph conv over a graph that adds explicit
        consecutive-time edges to the kNN-of-timesteps graph (so temporal locality
        is a first-class relation, not just feature similarity)."""
        node_type = "temporal_gnn_denoise"
        _attention = False

        def _transform(self, X: np.ndarray) -> np.ndarray:
            return _graph_reconstruct(X, False, self.hidden, self.epochs,
                                      adj_fn=_temporal_knn_adj)

    # ---------- energy-based / associative-memory nets ----------
    @register
    class HopfieldDenoiseNode(Node):
        """Modern Hopfield network (Ramsauer et al. 2020 — the 'attention IS Hopfield'
        result) as an associative-memory denoiser: each (normalized) row is replaced
        by a softmax(beta * X Xᵢᵀ)-weighted average of all stored rows, completing
        each pattern toward the learned manifold. No training — pure retrieval."""
        layer = "noise"
        node_type = "hopfield_denoise"
        is_transformer = True

        def __init__(self, beta: float = 2.0, **kw):
            super().__init__(beta=beta, **kw)
            self.beta = beta

        def _transform(self, X: np.ndarray) -> np.ndarray:
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
            attn = torch.softmax(self.beta * (Xt @ Xt.T), dim=1)
            return (attn @ Xt).numpy() * sd + mu

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean[:, 0].tolist(), "residual": resid,
                                       "model": "modern-hopfield"},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    @register
    class RBMDenoiseNode(Node):
        """Restricted Boltzmann Machine (Gaussian-Bernoulli) denoiser trained by
        contrastive divergence (CD-1, manual updates); the one-Gibbs-step
        reconstruction is the cleaned series — an energy-based generative denoiser."""
        layer = "noise"
        node_type = "rbm_denoise"
        is_transformer = True

        def __init__(self, hidden: int = 8, epochs: int = 120, **kw):
            super().__init__(hidden=hidden, epochs=epochs, **kw)
            self.hidden, self.epochs = hidden, epochs

        def _transform(self, X: np.ndarray) -> np.ndarray:
            _seed()
            mu, sd = X.mean(0), X.std(0) + 1e-6
            V = torch.tensor((X - mu) / sd, dtype=torch.float32)
            D = V.shape[1]
            W = torch.randn(D, self.hidden) * 0.1
            hb, vb = torch.zeros(self.hidden), torch.zeros(D)
            n, lr = V.shape[0], 0.05
            for _ in range(self.epochs):
                ph = torch.sigmoid(V @ W + hb)              # positive phase
                h0 = (torch.rand_like(ph) < ph).float()
                vn = h0 @ W.T + vb                          # gaussian visible reconstruction
                phn = torch.sigmoid(vn @ W + hb)            # negative phase
                W += lr * (V.T @ ph - vn.T @ phn) / n       # CD-1 updates
                vb += lr * (V - vn).mean(0)
                hb += lr * (ph - phn).mean(0)
            ph = torch.sigmoid(V @ W + hb)
            return (ph @ W.T + vb).numpy() * sd + mu

        def _predict(self, X: np.ndarray) -> Belief:
            clean = self._transform(X)
            resid = float(np.mean(np.abs(X - clean)))
            denom = float(np.mean(np.abs(X))) + 1e-9
            return Belief("denoised", {"series": clean[:, 0].tolist(), "residual": resid,
                                       "model": "gaussian-bernoulli-RBM"},
                          float(max(0.0, 1.0 - resid / denom)), self.name)

    # ---------- deep generative anomaly (operate on lag-windows) ----------
    def _windows(x, p):
        return np.stack([x[i:i + p] for i in range(len(x) - p)]) if len(x) > p + 2 else None

    @register
    class GANAnomalyNode(Node):
        """GAN anomaly detector: a generator learns the distribution of lag-windows
        while a discriminator learns to rate 'realness'; windows the trained
        discriminator deems unlikely (low realness) are flagged anomalous."""
        layer = "noise"
        node_type = "gan_anomaly"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 80,
                     quantile: float = 0.9, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, quantile=quantile, **kw)
            self.lag, self.hidden, self.epochs, self.quantile = lag, hidden, epochs, quantile

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x = X[:, 0].astype(float)
            p = max(2, min(self.lag, max(2, len(x) // 4)))
            W = _windows(x, p)
            if W is None or len(W) < 6:
                return Belief("anomaly", {"n_anomalies": 0, "fraction": 0.0,
                                          "scores": [], "flags": []}, 0.1, self.name)
            mu, sd = W.mean(), W.std() + 1e-6
            Wt = torch.tensor((W - mu) / sd, dtype=torch.float32)
            G = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, p))
            Dn = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
            og = torch.optim.Adam(G.parameters(), lr=0.01)
            od = torch.optim.Adam(Dn.parameters(), lr=0.01)
            bce = nn.BCEWithLogitsLoss()
            ones, zeros = torch.ones(Wt.shape[0], 1), torch.zeros(Wt.shape[0], 1)
            for _ in range(self.epochs):
                z = torch.randn(Wt.shape[0], p)
                fake = G(z).detach()
                od.zero_grad()
                (bce(Dn(Wt), ones) + bce(Dn(fake), zeros)).backward()
                od.step()
                z = torch.randn(Wt.shape[0], p)
                og.zero_grad()
                bce(Dn(G(z)), ones).backward()
                og.step()
            with torch.no_grad():
                real = torch.sigmoid(Dn(Wt)).numpy().ravel()
            score = 1.0 - real
            thr = float(np.quantile(score, self.quantile))
            flags = (score > thr).astype(int)
            return Belief("anomaly", {"n_anomalies": int(flags.sum()), "fraction": float(flags.mean()),
                                      "scores": score.tolist(), "flags": flags.tolist(), "model": "GAN"},
                          float(min(1.0, float(flags.mean()) * 3)), self.name)

    class _CouplingLayer(nn.Module):
        """RealNVP affine coupling: x1 unchanged; x2 -> x2*exp(s(x1)) + t(x1)."""
        def __init__(self, dim, hidden):
            super().__init__()
            self.half = dim // 2
            self.s = nn.Sequential(nn.Linear(self.half, hidden), nn.Tanh(),
                                   nn.Linear(hidden, dim - self.half))
            self.t = nn.Sequential(nn.Linear(self.half, hidden), nn.Tanh(),
                                   nn.Linear(hidden, dim - self.half))

        def forward(self, x):
            x1, x2 = x[:, :self.half], x[:, self.half:]
            s = torch.tanh(self.s(x1))
            z2 = x2 * torch.exp(s) + self.t(x1)
            return torch.cat([x1, z2], 1), s.sum(1)

    @register
    class NormalizingFlowAnomalyNode(Node):
        """Normalizing-flow (RealNVP-lite) anomaly detector: two affine coupling
        layers (with a dimension flip between) map lag-windows to a standard-normal
        base; per-window negative log-likelihood is the anomaly score (low-density
        windows = anomalies). Exact likelihood, no adversarial training."""
        layer = "noise"
        node_type = "normalizing_flow_anomaly"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 120,
                     quantile: float = 0.9, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, quantile=quantile, **kw)
            self.lag, self.hidden, self.epochs, self.quantile = lag, hidden, epochs, quantile

        def _nll(self, flows, Wt):
            z, logdet = Wt, torch.zeros(Wt.shape[0])
            for i, fl in enumerate(flows):
                z, ld = fl(z)
                logdet = logdet + ld
                z = z.flip(1) if i < len(flows) - 1 else z      # permute dims between layers
            return 0.5 * (z ** 2).sum(1) - logdet               # -log p(x), up to const

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x = X[:, 0].astype(float)
            p = max(2, min(self.lag, max(2, len(x) // 4)))
            W = _windows(x, p)
            if W is None or len(W) < 6 or p < 2:
                return Belief("anomaly", {"n_anomalies": 0, "fraction": 0.0,
                                          "scores": [], "flags": []}, 0.1, self.name)
            mu, sd = W.mean(), W.std() + 1e-6
            Wt = torch.tensor((W - mu) / sd, dtype=torch.float32)
            flows = nn.ModuleList([_CouplingLayer(p, self.hidden) for _ in range(2)])
            opt = torch.optim.Adam(flows.parameters(), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                self._nll(flows, Wt).mean().backward()
                opt.step()
            with torch.no_grad():
                score = self._nll(flows, Wt).numpy()
            score = score - score.min()
            thr = float(np.quantile(score, self.quantile))
            flags = (score > thr).astype(int)
            return Belief("anomaly", {"n_anomalies": int(flags.sum()), "fraction": float(flags.mean()),
                                      "scores": score.tolist(), "flags": flags.tolist(),
                                      "model": "RealNVP"},
                          float(min(1.0, float(flags.mean()) * 3)), self.name)

    # ---------- continuous-control-family deep-RL (discrete short/flat/long) ----------
    def _rl_tensors(X, lag):
        x = X[:, 0].astype(float)
        p = max(2, min(lag, max(2, len(x) // 4)))
        S, R = _rl_dataset(x, p)
        return x, p, S, R

    @register
    class SACPolicyNode(Node):
        """Soft Actor-Critic (discrete adaptation): an entropy-regularized stochastic
        policy over short/flat/long, trained against a 1-step Q-critic with the SAC
        maximum-entropy bonus (favors decisive-but-not-overconfident policies)."""
        layer = "rl"
        node_type = "sac_policy"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 60,
                     alpha: float = 0.1, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, alpha=alpha, **kw)
            self.lag, self.hidden, self.epochs, self.alpha = lag, hidden, epochs, alpha

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x, p, S, R = _rl_tensors(X, self.lag)
            if len(S) < 4:
                return Belief("action", {"action": 1, "policy": [0.33, 0.34, 0.33]}, 0.1, self.name)
            mu, sd = S.mean(), S.std() + 1e-6
            St = torch.tensor((S - mu) / sd, dtype=torch.float32)
            Rt = torch.tensor(R, dtype=torch.float32)
            actor = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            critic = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                q = critic(St)
                logp = torch.log_softmax(actor(St), dim=1)
                probs = logp.exp()
                critic_loss = ((q - Rt) ** 2).mean()
                actor_loss = -(probs * (q.detach() - self.alpha * logp)).sum(1).mean()
                (critic_loss + actor_loss).backward()
                opt.step()
            with torch.no_grad():
                probs = torch.softmax(actor(torch.tensor((x[-p:] - mu) / sd, dtype=torch.float32)), 0).numpy()
            a = int(np.argmax(probs))
            return Belief("action", {"action": a - 1, "policy": [float(v) for v in probs],
                                     "labels": ["short", "flat", "long"], "model": "SAC"},
                          float(max(0.0, min(1.0, probs[a]))), self.name)

    @register
    class TD3PolicyNode(Node):
        """TD3 (discrete adaptation): twin Q-networks; the action is greedy w.r.t. the
        element-wise MINIMUM of the two critics (clipped double-Q — TD3's fix for the
        overestimation bias that single-critic DQN suffers)."""
        layer = "rl"
        node_type = "td3_policy"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 60, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, **kw)
            self.lag, self.hidden, self.epochs = lag, hidden, epochs

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x, p, S, R = _rl_tensors(X, self.lag)
            if len(S) < 4:
                return Belief("action", {"action": 1, "q_values": [0, 0, 0]}, 0.1, self.name)
            mu, sd = S.mean(), S.std() + 1e-6
            St = torch.tensor((S - mu) / sd, dtype=torch.float32)
            Rt = torch.tensor(R, dtype=torch.float32)
            q1 = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            q2 = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
            opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                (((q1(St) - Rt) ** 2).mean() + ((q2(St) - Rt) ** 2).mean()).backward()
                opt.step()
            with torch.no_grad():
                w = torch.tensor((x[-p:] - mu) / sd, dtype=torch.float32)
                q = torch.minimum(q1(w), q2(w)).numpy()         # clipped double-Q
            a = int(np.argmax(q))
            return Belief("action", {"action": a - 1, "q_values": [float(v) for v in q],
                                     "labels": ["short", "flat", "long"], "model": "TD3"},
                          float(max(0.0, min(1.0, abs(q[a])))), self.name)

    @register
    class A3CPolicyNode(Node):
        """A3C (synchronous single-worker): a shared trunk with a policy head and a
        value head; advantage = reward - learned V baseline drives the policy
        gradient (an actor-critic, vs PPO-lite's fixed mean baseline)."""
        layer = "rl"
        node_type = "a3c_policy"

        def __init__(self, lag: int = 6, hidden: int = 16, epochs: int = 60, **kw):
            super().__init__(lag=lag, hidden=hidden, epochs=epochs, **kw)
            self.lag, self.hidden, self.epochs = lag, hidden, epochs

        def _predict(self, X: np.ndarray) -> Belief:
            _seed()
            x, p, S, R = _rl_tensors(X, self.lag)
            if len(S) < 4:
                return Belief("action", {"action": 1, "policy": [0.33, 0.34, 0.33]}, 0.1, self.name)
            mu, sd = S.mean(), S.std() + 1e-6
            St = torch.tensor((S - mu) / sd, dtype=torch.float32)
            Rt = torch.tensor(R, dtype=torch.float32)
            trunk = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU())
            pi = nn.Linear(self.hidden, 3)
            vf = nn.Linear(self.hidden, 1)
            opt = torch.optim.Adam(list(trunk.parameters()) + list(pi.parameters())
                                   + list(vf.parameters()), lr=0.01)
            for _ in range(self.epochs):
                opt.zero_grad()
                h = trunk(St)
                v = vf(h)                                       # (N,1) baseline
                adv = Rt - v                                    # advantage per action
                logp = torch.log_softmax(pi(h), dim=1)
                policy_loss = -(logp * adv.detach()).mean()
                value_loss = (adv ** 2).mean()
                (policy_loss + value_loss).backward()
                opt.step()
            with torch.no_grad():
                probs = torch.softmax(pi(trunk(torch.tensor((x[-p:] - mu) / sd, dtype=torch.float32))), 0).numpy()
            a = int(np.argmax(probs))
            return Belief("action", {"action": a - 1, "policy": [float(v) for v in probs],
                                     "labels": ["short", "flat", "long"], "model": "A3C"},
                          float(max(0.0, min(1.0, probs[a]))), self.name)
