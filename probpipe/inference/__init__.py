"""Inference algorithms for ProbPipe.

Provides MCMC sampling (gradient-based NUTS/HMC + gradient-free RWMH and
elliptical slice sampling — all BlackJAX-backed), chain-structured
empirical distributions, and the inference methods that normalize the targets
of ``condition_on``. The methods register into the inference-method registry,
which is defined with ``condition_on`` and re-exported here.
"""

from __future__ import annotations

from ..core._dispatch import (
    BaseDispatchMethod,
    BaseDispatchRegistry,
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    BinarySupportedTypes,
    Feasibility,
    MathematicalDomainError,
    MethodInfo,
    ResolutionError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
    UnarySupportedTypes,
)
from ..operations._condition import InferenceMethod, inference_method_registry
from ._bayesflow_likelihoods import (
    BayesFlowLikelihood,
    BayesFlowRatio,
    learn_amortized_likelihood,
    learn_amortized_ratio,
)

# Amortized SBI (optional ``[bayesflow]`` extra). keras/bayesflow load lazily on
# first call, so these eager imports stay cheap. The trained artifacts are
# kernels: an amortized posterior claims ``SupportsApproximateConditioning``, and
# a learned likelihood or ratio composes with a prior into a joint whose
# conditional the registered methods normalize.
from ._bayesflow_posteriors import learn_amortized_posterior
from ._blackjax_ess import elliptical_slice
from ._blackjax_rwmh import rwmh
from ._minibatch import MinibatchedDistribution
from ._nutpie import condition_on_nutpie

__all__ = [
    "BaseDispatchMethod",
    "BaseDispatchRegistry",
    "BayesFlowLikelihood",
    "BayesFlowRatio",
    "BinaryDispatchMethod",
    "BinaryDispatchRegistry",
    "BinarySupportedTypes",
    "Feasibility",
    "InferenceMethod",
    "MathematicalDomainError",
    "MethodInfo",
    "MinibatchedDistribution",
    "ResolutionError",
    "UnaryDispatchMethod",
    "UnaryDispatchRegistry",
    "UnarySupportedTypes",
    "condition_on_nutpie",
    "elliptical_slice",
    "inference_method_registry",
    "learn_amortized_likelihood",
    "learn_amortized_posterior",
    "learn_amortized_ratio",
    "rwmh",
]


# ---------------------------------------------------------------------------
# Register built-in inference methods
# ---------------------------------------------------------------------------

# ``probpipe.condition_on`` passes a model and its observed data to the registry,
# whose methods take the target this package forms from them.
from ..operations._condition import _install_observed_target
from ._inference_utils import observed_target

_install_observed_target(observed_target)

# TFP-backed MCMC — registered with ``priority=None`` (opt-in only); BlackJAX
# methods below win auto-dispatch.
from ._tfp_mcmc import TFPNutsMethod

inference_method_registry.register(TFPNutsMethod())

# BlackJAX MCMC (gradient-based) — auto-dispatch default for any
# JAX-traceable ``SupportsLogProb`` target.
from ._blackjax_mcmc import BlackJAXHmcMethod, BlackJAXNutsMethod

inference_method_registry.register(BlackJAXNutsMethod())
inference_method_registry.register(BlackJAXHmcMethod())

# BlackJAX gradient-free MCMC: RWMH (catch-all) and ESS (Gaussian-prior).
from ._blackjax_ess import BlackJAXESSMethod
from ._blackjax_rwmh import BlackJAXRWMHMethod

inference_method_registry.register(BlackJAXRWMHMethod())
inference_method_registry.register(BlackJAXESSMethod())

# Exact Bayes' rule for an empirical prior, which reweights its atoms.
from ._empirical_reweighting import EmpiricalReweightingMethod

inference_method_registry.register(EmpiricalReweightingMethod())

# BlackJAX SGMCMC
from ._blackjax_sgmcmc import BlackJAXSGHMCMethod, BlackJAXSGLDMethod

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
    from ._pymc_method import PyMCADVIMethod, PyMCNutsMethod

    inference_method_registry.register(PyMCNutsMethod())
    inference_method_registry.register(PyMCADVIMethod())
except ImportError:
    pass

try:
    import pyabc  # registration gated on the [pyabc] extra

    from ._pyabc import PyABCSMCMethod

    inference_method_registry.register(PyABCSMCMethod())
except ImportError:
    pass
