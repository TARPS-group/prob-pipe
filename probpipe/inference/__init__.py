"""Inference algorithms for ProbPipe.

Provides MCMC sampling (gradient-free RWMH and nutpie-backed NUTS),
chain-structured empirical distributions, and the inference method
registry for ``condition_on`` dispatch.
"""

from __future__ import annotations
from ._approximate_distribution import ApproximateDistribution
from ._registry import (
    InferenceMethod,
    inference_method_registry,
)
from ..core._registry import Method, MethodInfo, MethodRegistry
from ._rwmh import rwmh
from ._nutpie import condition_on_nutpie

__all__ = [
    "ApproximateDistribution",
    "Method",
    "MethodInfo",
    "MethodRegistry",
    "InferenceMethod",
    "inference_method_registry",
    "rwmh",
    "condition_on_nutpie",
    "sbi_learn_conditional",
    "sbi_learn_likelihood",
    "DirectSamplerSBIModel",
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
    from ._pymc_method import PyMCNutsMethod, PyMCADVIMethod
    inference_method_registry.register(PyMCNutsMethod())
    inference_method_registry.register(PyMCADVIMethod())
except ImportError:
    pass

try:
    from ._sbijax import (
        SbiSMCABCMethod,
        sbi_learn_conditional,
        sbi_learn_likelihood,
        DirectSamplerSBIModel,
    )
    inference_method_registry.register(SbiSMCABCMethod())
except ImportError:

    _SBI_INSTALL_MSG = (
        "SBI features require sbijax: pip install probpipe[sbi]"
    )

    def sbi_learn_conditional(*args, **kwargs):  # type: ignore[misc]
        """Placeholder that raises when sbijax is not installed."""
        raise ImportError(_SBI_INSTALL_MSG)

    def sbi_learn_likelihood(*args, **kwargs):  # type: ignore[misc]
        """Placeholder that raises when sbijax is not installed."""
        raise ImportError(_SBI_INSTALL_MSG)

    class DirectSamplerSBIModel:  # type: ignore[no-redef]
        """Placeholder that raises when sbijax is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(_SBI_INSTALL_MSG)
