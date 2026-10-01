"""The distribution catalog: the concrete families the library ships.

Every family is an ordinary ``Distribution`` or ``ConditionalDistribution``,
and the catalog adds no base classes. Each module realizes one section of the
catalog.

Provides:
  - the resampling families: ``BootstrapReplicateDistribution``,
    ``BootstrapDistribution``, and ``KDEDistribution``, with the smoothing
    kernels ``SmoothingKernel``, ``GaussianKernel``, and ``EpanechnikovKernel``;
  - the mixture family, ``MixtureDistribution``;
  - the evaluation-result families, ``LinearPushforwardDistribution`` and
    ``BijectorTransformedDistribution``;
  - the Gaussian algebra's ``FactoredMultivariateGaussian`` and
    ``GaussianProcess``;
  - the conditional families: ``LinearGaussianConditional``, the response
    families ``GLMFamily``, ``GaussianFamily``, ``BernoulliFamily``, and
    ``PoissonFamily``, and ``glm_likelihood``;
  - the program-defined families ``StanModel``, ``PyMCModel``, and
    ``UnnormalizedDistribution``.
"""

from ._conditional import (
    BernoulliFamily,
    GaussianFamily,
    GLMFamily,
    LinearGaussianConditional,
    PoissonFamily,
    glm_likelihood,
)
from ._gaussian import FactoredMultivariateGaussian, GaussianProcess
from ._mixture import MixtureDistribution
from ._programs import PyMCModel, StanModel, UnnormalizedDistribution
from ._resampling import (
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    EpanechnikovKernel,
    GaussianKernel,
    KDEDistribution,
    SmoothingKernel,
)
from ._transformed import BijectorTransformedDistribution, LinearPushforwardDistribution

__all__ = [
    "BernoulliFamily",
    "BijectorTransformedDistribution",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "EpanechnikovKernel",
    "FactoredMultivariateGaussian",
    "GLMFamily",
    "GaussianFamily",
    "GaussianKernel",
    "GaussianProcess",
    "KDEDistribution",
    "LinearGaussianConditional",
    "LinearPushforwardDistribution",
    "MixtureDistribution",
    "PoissonFamily",
    "PyMCModel",
    "SmoothingKernel",
    "StanModel",
    "UnnormalizedDistribution",
    "glm_likelihood",
]
