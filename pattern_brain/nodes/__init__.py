"""Importing this package registers every built-in node with the registry.

Modules are grouped by Block 18's functional layers, in order.
"""
from . import signal      # noqa: F401
from . import noise       # noqa: F401
from . import pattern     # noqa: F401
from . import sequence    # noqa: F401
from . import probability # noqa: F401
from . import equation    # noqa: F401
from . import decision    # noqa: F401
from . import rl          # noqa: F401
from . import physics       # noqa: F401  (Phase 7e: physics/PhD-tier light-stack nodes)
from . import econometrics  # noqa: F401  (Phase 7a: classical econometrics + volatility)
from . import patternmining # noqa: F401  (Phase 7b: frequent + sequential pattern mining)
from . import probabilistic # noqa: F401  (Phase 7c: higher-order/semi-Markov, HHMM, CRF, Bayes nets)
from . import infotheory    # noqa: F401  (Phase 7d: MI, transfer entropy, Kolmogorov/MDL, cross/cond entropy)
from . import deep         # noqa: F401  (deep/PyTorch nodes — only register if torch is installed)
from . import decomposition # noqa: F401  (PLAN §17.6: Wavelet/SSA/EMD decomposition + optional PySR)
from . import tier3_probability  # noqa: F401  (TIER-3: Bayes update, MaxEnt, Chebyshev bound)
from . import tier3_stochastic   # noqa: F401  (TIER-3: GBM/Bachelier, Langevin, Fokker-Planck)
from . import tier3_control      # noqa: F401  (TIER-3: log-utility, Lagrange alloc, Bellman/HJB)
from . import tier3_pricing      # noqa: F401  (TIER-3: Newton kinematics, Almgren exec, Black-Scholes)
from . import tier1_classics     # noqa: F401  (TIER-1 directly-buildable classics, built one at a time)
from . import tier1_microstructure  # noqa: F401  (TIER-1 batch 3: OFI, VPIN, Kyle lambda)
from . import tier1_allocation      # noqa: F401  (TIER-1 batch 3: HRP, CVaR, Merton, MPC)
from . import tier1_stochastic      # noqa: F401  (TIER-1 batch 3: jump-diffusion, Heston, SV particle, percolation)
from . import tier1_dynamical       # noqa: F401  (TIER-1 batch 3: Koopman/DMD, BSTS, info-geom, diffusion-map)
from . import tier1_softcomputing   # noqa: F401  (TIER-1 batch 3: fuzzy-TS, grey GM(1,1), ANFIS)
from . import semisupervised        # noqa: F401  (Gap-batch A: self/co-training, label prop/spread)
from . import graph_embedding       # noqa: F401  (Gap-batch B: DeepWalk/Node2Vec random-walk + skip-gram embeddings)
from . import embedding             # noqa: F401  (VISION P1 / Stage-1: learned self-supervised universal market embedding)
