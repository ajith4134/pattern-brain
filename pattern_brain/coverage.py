"""Catalog-vs-built coverage meter (PLAN.md §11, Block 57).

Maps the owner's full ML taxonomy (the 15 families of Block 57 part A + the
physics/PhD tier + the Galton-board distributional family) to what is ACTUALLY
registered in the bank, so the road from ~150 → ~500 nodes is visible and the
Advisor/agent can target the ❌ gaps.

Honest framing: this tracks *curated key models per family*, not every published
variant — the full catalog is ~500+ when all variants are counted (PLAN §5 Block 42).
A family item is "built" if any registered node_type contains one of its keywords.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .registry import all_node_types

# family -> list of (label, [keyword substrings matched against node_type])
CATALOG: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    ("Pattern mining", [
        ("Apriori", ["apriori"]), ("FP-Growth", ["fpgrowth"]), ("ECLAT", ["eclat"]),
        ("PrefixSpan", ["prefixspan"]), ("SPADE", ["spade"]), ("GSP", ["gsp"])]),
    ("Statistical sequence (Markov/HMM)", [
        ("Markov chain", ["markov_chain"]), ("Higher-order Markov", ["higher_order_markov"]),
        ("HMM", ["hmm"]), ("Hierarchical HMM", ["hierarchical_hmm"]),
        ("Semi-Markov", ["semi_markov"]), ("CRF", ["crf"])]),
    ("Probabilistic", [
        ("Naive Bayes", ["gaussian_nb", "naive_bayes"]), ("Bayesian network", ["bayesian_network"]),
        ("Dynamic Bayesian network", ["dynamic_bayes"]), ("Gaussian process", ["gaussian_process"]),
        ("Kalman filter", ["kalman"]), ("Particle filter", ["particle"])]),
    ("Time-series forecasting", [
        ("AR", ["autoregressive"]), ("ARMA", ["arma_forecast"]), ("ARIMA", ["arima_forecast"]),
        ("SARIMA", ["sarima"]), ("VAR", ["var_forecast"]), ("GARCH", ["garch_forecast"]),
        ("EGARCH", ["egarch"]), ("EWMA vol", ["ewma_volatility"]), ("Prophet", ["prophet"]),
        ("Holt-Winters", ["holt_winters", "holt_linear"]), ("Theta", ["theta"]),
        ("N-BEATS", ["nbeats"]), ("TFT", ["temporal_fusion", "tft"])]),
    ("Neural sequence", [
        ("RNN", ["rnn"]), ("LSTM", ["lstm"]), ("GRU", ["gru"]), ("BiLSTM", ["bilstm"]),
        ("Seq2Seq", ["seq2seq"]), ("TCN", ["tcn"]), ("MLP", ["mlp"])]),
    ("Transformers", [
        ("Transformer", ["transformer"]), ("PatchTST", ["patchtst"]), ("Informer", ["informer"]),
        ("Autoformer", ["autoformer"]), ("FEDformer", ["fedformer"]),
        ("TimeGPT", ["timegpt"]), ("Chronos", ["chronos"])]),
    ("State-space models", [
        ("SSM (S4D-lite)", ["ssm"]), ("S4", ["s4_"]), ("S5", ["s5_"]), ("Mamba", ["mamba"])]),
    ("Clustering", [
        ("K-Means", ["kmeans"]), ("DBSCAN", ["dbscan"]), ("HDBSCAN", ["hdbscan"]),
        ("GMM", ["gmm", "gaussian_mixture"]), ("Spectral", ["spectral_clustering"]),
        ("Mean shift", ["mean_shift"]), ("OPTICS", ["optics"]), ("BIRCH", ["birch"])]),
    ("Anomaly / noise", [
        ("Isolation Forest", ["isolation_forest"]), ("One-Class SVM", ["one_class_svm"]),
        ("LOF", ["local_outlier_factor"]), ("Autoencoder", ["autoencoder"]),
        ("VAE", ["vae"]), ("Robust PCA", ["pca_denoise", "pca_residual"]),
        ("Elliptic envelope", ["elliptic"])]),
    ("Representation learning", [
        ("Autoencoder", ["deep_autoencoder", "autoencoder"]), ("Sparse AE", ["sparse_ae"]),
        ("VAE", ["vae"]), ("Contrastive/SimCLR", ["simclr", "contrastive"]),
        ("BYOL", ["byol"]), ("MAE", ["masked_ae", "mae_"]), ("PCA/ICA", ["ica_", "kernel_pca", "svd"])]),
    ("Graph", [
        ("GNN/GCN", ["gcn", "gnn"]), ("GraphSAGE", ["graphsage"]), ("GAT", ["gat_"]),
        ("Temporal GNN", ["temporal_gnn"])]),
    ("Reinforcement learning", [
        ("Q-learning", ["q_learning"]), ("Bandits", ["bandit", "ucb", "thompson"]),
        ("DQN", ["dqn"]), ("PPO", ["ppo"]), ("SAC", ["sac"]), ("TD3", ["td3"]), ("A3C", ["a3c"])]),
    ("Symbolic / equation discovery", [
        ("Symbolic regression", ["symbolic_regression"]), ("Genetic programming", ["genetic"]),
        ("PySR", ["pysr"]), ("AI Feynman", ["feynman"]), ("SINDy", ["sindy"]),
        ("Regression laws", ["linear_regression", "ridge_regression", "lasso"])]),
    ("Signal decomposition", [
        ("Wavelet (DWT)", ["wavelet"]), ("SSA", ["ssa_decompose", "ssa_"]),
        ("EMD", ["emd_decompose", "emd_"]), ("Fourier (FFT/Welch)", ["fft", "welch_psd", "periodogram"])]),
    ("Information theory", [
        ("Spectral entropy", ["spectral_entropy"]), ("Sample entropy", ["sample_entropy"]),
        ("Permutation entropy", ["permutation_entropy"]), ("Tsallis entropy", ["tsallis"]),
        ("Mutual information", ["mutual_information"]), ("Transfer entropy", ["transfer_entropy"]),
        ("Kolmogorov/MDL", ["kolmogorov", "mdl_"])]),
    ("Generative", [
        ("VAE", ["vae"]), ("GAN", ["gan_"]), ("Diffusion", ["diffusion"]),
        ("Normalizing flows", ["normalizing_flow", "flow_"]), ("Autoregressive", ["autoregressive"])]),
    ("Physics / chaos / econophysics (PhD tier)", [
        ("Reservoir computing (ESN)", ["esn_forecaster"]), ("Deterministic RC (TCRC)", ["deterministic_esn"]),
        ("Lyapunov exponent", ["lyapunov"]), ("Recurrence (RQA)", ["recurrence_rate"]),
        ("Phase-space (Takens)", ["phase_space"]), ("Hurst", ["hurst"]), ("MFDFA", ["mfdfa"]),
        ("RMT (Marchenko-Pastur)", ["rmt_"]), ("Hawkes", ["hawkes"]),
        ("Quadratic Hawkes / rough vol", ["quadratic_hawkes", "rough_vol"]),
        ("TDA / persistent homology", ["persistent_homology", "tda_"]),
        ("PINN / Neural-ODE", ["pinn", "neural_ode"]),
        ("Hopfield / Boltzmann (energy)", ["hopfield", "boltzmann", "rbm"]),
        ("Quantum / QAOA", ["qaoa", "quantum"])]),
    ("Probability / distributional (Galton)", [
        ("Quantile forecaster", ["quantile_forecaster"]), ("Conformal", ["conformal"]),
        ("Bayesian bootstrap", ["bayesian_bootstrap"]),
        ("Bayesian model averaging", ["bayesian_ar_ensemble"])]),
]


def _status(n_built: int, n_target: int) -> str:
    if n_built == 0:
        return "missing"
    if n_built >= n_target:
        return "complete"
    return "partial"


def coverage() -> Dict:
    """Compute per-family built/missing coverage against the registry."""
    registered = all_node_types()
    families = []
    tot_built = tot_target = 0
    for fam, items in CATALOG:
        built, missing = [], []
        for label, keys in items:
            hit = any(any(k in nt for nt in registered) for k in keys)
            (built if hit else missing).append(label)
        n_b, n_t = len(built), len(items)
        tot_built += n_b
        tot_target += n_t
        families.append({"family": fam, "built": built, "missing": missing,
                         "n_built": n_b, "n_target": n_t,
                         "pct": round(100 * n_b / n_t, 1), "status": _status(n_b, n_t)})
    return {
        "n_registered_nodes": len(registered),
        "curated_built": tot_built, "curated_target": tot_target,
        "curated_pct": round(100 * tot_built / tot_target, 1),
        "full_catalog_target": 500,
        "families": families,
    }
