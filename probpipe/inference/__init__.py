"""Inference algorithms for ProbPipe.

Provides MCMC sampling (gradient-based NUTS/HMC + gradient-free RWMH and
elliptical slice sampling — all BlackJAX-backed), chain-structured
empirical distributions, and the inference method registry for
``condition_on`` dispatch.
"""

from __future__ import annotations
from ._approximate_distribution import ApproximateDistribution
from ._registry import (
    InferenceMethod,
    inference_method_registry,
)
from ..core._registry import Method, MethodInfo, MethodRegistry
from ._blackjax_rwmh import rwmh
from ._blackjax_ess import elliptical_slice
from ._nutpie import condition_on_nutpie
from ._minibatch import MinibatchedDistribution

__all__ = [
    "ApproximateDistribution",
    "Method",
    "MethodInfo",
    "MethodRegistry",
    "InferenceMethod",
    "inference_method_registry",
    "rwmh",
    "elliptical_slice",
    "condition_on_nutpie",
    "MinibatchedDistribution",
    "sbi_learn_conditional",
    "sbi_learn_likelihood",
    "DirectSamplerSBIModel",
]


# ---------------------------------------------------------------------------
# Register built-in inference methods
# ---------------------------------------------------------------------------

# TFP-backed methods (always available since TFP is a core dependency).
# These are opt-in-only (priority 0) after the BlackJAX migration —
# auto-dispatch picks the BlackJAX-backed equivalents below.
from ._tfp_mcmc import TFPNutsMethod, TFPHmcMethod

inference_method_registry.register(TFPNutsMethod())
inference_method_registry.register(TFPHmcMethod())

# BlackJAX MCMC (gradient-based) — the auto-dispatch default for any
# JAX-traceable SupportsLogProb target.
from ._blackjax_mcmc import BlackJAXNutsMethod, BlackJAXHmcMethod

inference_method_registry.register(BlackJAXNutsMethod())
inference_method_registry.register(BlackJAXHmcMethod())

# BlackJAX gradient-free MCMC: RWMH (catch-all) and ESS (Gaussian-prior).
# ``TFPRWMHMethod`` is a deprecated alias kept for one minor release —
# the legacy ``tfp_rwmh`` name pre-dates the BlackJAX migration and was
# never actually TFP-backed (the original implementation was a
# hand-rolled Python loop).
from ._blackjax_rwmh import BlackJAXRWMHMethod, TFPRWMHMethod
from ._blackjax_ess import BlackJAXESSMethod

inference_method_registry.register(BlackJAXRWMHMethod())
inference_method_registry.register(BlackJAXESSMethod())
inference_method_registry.register(TFPRWMHMethod())

# BlackJAX SGMCMC
from ._blackjax_sgmcmc import BlackJAXSGLDMethod, BlackJAXSGHMCMethod

inference_method_registry.register(BlackJAXSGLDMethod())
inference_method_registry.register(BlackJAXSGHMCMethod())

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
