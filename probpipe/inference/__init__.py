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
from ..core._registry import (
    BaseDispatchMethod,
    BaseDispatchRegistry,
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ._blackjax_rwmh import rwmh
from ._blackjax_ess import elliptical_slice
from ._nutpie import condition_on_nutpie
from ._minibatch import MinibatchedDistribution
# Amortized SBI (optional ``[bayesflow]`` extra). keras/bayesflow load lazily on
# first call, so this eager import stays cheap; the trained estimator dispatches
# via ``SupportsConditioning`` and needs no inference-registry method.
from ._bayesflow import BayesFlowModel, learn_amortized_posterior

__all__ = [
    "ApproximateDistribution",
    "BaseDispatchMethod",
    "BaseDispatchRegistry",
    "BinaryDispatchMethod",
    "BinaryDispatchRegistry",
    "MethodInfo",
    "UnaryDispatchMethod",
    "UnaryDispatchRegistry",
    "InferenceMethod",
    "inference_method_registry",
    "rwmh",
    "elliptical_slice",
    "condition_on_nutpie",
    "MinibatchedDistribution",
    "learn_amortized_posterior",
    "BayesFlowModel",
]


# ---------------------------------------------------------------------------
# Register built-in inference methods
# ---------------------------------------------------------------------------

# TFP-backed MCMC — registered at priority 0 (opt-in only); BlackJAX
# methods below win auto-dispatch.
from ._tfp_mcmc import TFPNutsMethod, TFPHmcMethod

inference_method_registry.register(TFPNutsMethod())
inference_method_registry.register(TFPHmcMethod())

# BlackJAX MCMC (gradient-based) — auto-dispatch default for any
# JAX-traceable ``SupportsLogProb`` target.
from ._blackjax_mcmc import BlackJAXNutsMethod, BlackJAXHmcMethod

inference_method_registry.register(BlackJAXNutsMethod())
inference_method_registry.register(BlackJAXHmcMethod())

# BlackJAX gradient-free MCMC: RWMH (catch-all) and ESS (Gaussian-prior).
from ._blackjax_rwmh import BlackJAXRWMHMethod
from ._blackjax_ess import BlackJAXESSMethod

inference_method_registry.register(BlackJAXRWMHMethod())
inference_method_registry.register(BlackJAXESSMethod())

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
