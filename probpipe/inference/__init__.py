"""Inference algorithms for ProbPipe.

Provides MCMC sampling (gradient-free RWMH and nutpie-backed NUTS),
chain-structured empirical distributions, and the inference method
registry for ``condition_on`` dispatch.
"""

from __future__ import annotations
from ._mcmc_distribution import ApproximateDistribution
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
    "train_sbi",
    "TrainedSBIModel",
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
    from ._sbijax_methods import SbiSMCABCMethod
    from ._sbijax_train import train_sbi
    from ._sbijax_distribution import TrainedSBIModel
    inference_method_registry.register(SbiSMCABCMethod())
except ImportError:
    train_sbi = None  # type: ignore[assignment]
    TrainedSBIModel = None  # type: ignore[assignment,misc]
