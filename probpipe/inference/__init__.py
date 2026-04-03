"""Inference algorithms for ProbPipe.

Provides MCMC sampling (gradient-free RWMH and nutpie-backed NUTS),
chain-structured empirical distributions, diagnostics, and the
inference method registry for ``condition_on`` dispatch.
"""

from ._diagnostics import (
    InferenceDiagnostics,
    MCMCDiagnostics,
    extract_arviz_diagnostics,
)
from ._mcmc_distribution import MCMCApproximateDistribution
from ._registry import (
    InferenceMethod,
    inference_method_registry,
)
from ._rwmh import rwmh
from ._nutpie import condition_on_nutpie

__all__ = [
    "InferenceDiagnostics",
    "MCMCDiagnostics",
    "extract_arviz_diagnostics",
    "MCMCApproximateDistribution",
    "InferenceMethod",
    "inference_method_registry",
    "rwmh",
    "condition_on_nutpie",
]


# ---------------------------------------------------------------------------
# Register built-in inference methods
# ---------------------------------------------------------------------------

# TFP-backed methods (always available since TFP is a core dependency)
from ._tfp_mcmc import TFPNutsMethod, TFPHmcMethod

inference_method_registry.register(TFPNutsMethod())
inference_method_registry.register(TFPHmcMethod())

# RWMH (always available)
from ._rwmh import TFPRWMHMethod

inference_method_registry.register(TFPRWMHMethod())

# Optional backends — registered only if their dependencies are importable

try:
    from ._nutpie import NutpieNutsMethod
    inference_method_registry.register(NutpieNutsMethod())
except ImportError:
    pass

try:
    from ._cmdstan_method import CmdStanNutsMethod
    inference_method_registry.register(CmdStanNutsMethod())
except ImportError:
    pass

try:
    from ._pymc_method import PyMCMCMCMethod, PyMCADVIMethod
    inference_method_registry.register(PyMCMCMCMethod())
    inference_method_registry.register(PyMCADVIMethod())
except ImportError:
    pass
