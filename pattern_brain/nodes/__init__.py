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
from . import physics     # noqa: F401  (Phase 7e: physics/PhD-tier light-stack nodes)
from . import deep        # noqa: F401  (deep/PyTorch nodes — only register if torch is installed)
