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

        def _build(self, D: int):  # pragma: no cover - overridden
            raise NotImplementedError

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
            net = self._build(D)
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            lossf = nn.MSELoss()
            net.train()
            last = 0.0
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = net(Xn)
                loss = lossf(pred, Yn)
                loss.backward()
                opt.step()
                last = float(loss.item())
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

        def _build(self, D):
            return _RecurrentNet(D, self.hidden, kind="lstm")

    @register
    class GRUForecastNode(_TorchForecaster):
        """GRU recurrent one-step forecaster."""
        node_type = "gru_forecaster"

        def _build(self, D):
            return _RecurrentNet(D, self.hidden, kind="gru")

    @register
    class TransformerForecastNode(_TorchForecaster):
        """Transformer-encoder one-step forecaster (attention over the window)."""
        node_type = "transformer_forecaster"

        def _build(self, D):
            return _TransformerNet(D, self.hidden)

    @register
    class TCNForecastNode(_TorchForecaster):
        """Temporal Convolutional Network forecaster (dilated 1-D convolutions)."""
        node_type = "tcn_forecaster"

        def _build(self, D):
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
