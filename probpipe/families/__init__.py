"""The distribution catalog: the concrete families the library ships.

Every family is an ordinary ``Distribution`` or ``ConditionalDistribution``,
and the catalog adds no base classes. Each module realizes one section of the
catalog.

Provides:
  - the backend adapter ``TFPDistribution`` and the parametric families over
    it: the continuous ``Normal``, ``Beta``, ``Gamma``, ``InverseGamma``,
    ``Exponential``, ``LogNormal``, ``StudentT``, ``Uniform``, ``Cauchy``,
    ``Laplace``, ``HalfNormal``, ``HalfCauchy``, ``Pareto``, and
    ``TruncatedNormal``; the discrete ``Bernoulli``, ``Binomial``,
    ``Poisson``, ``Categorical``, and ``NegativeBinomial``; and the
    multivariate ``MultivariateNormal``, ``Dirichlet``, ``Multinomial``,
    ``Wishart``, and ``VonMisesFisher``;
  - the smoothing kernels of a kernel density estimate: ``SmoothingKernel``,
    ``GaussianKernel``, and ``EpanechnikovKernel``;
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

from ._backend import TFPDistribution
from ._conditional import (
    BernoulliFamily,
    GaussianFamily,
    GLMFamily,
    LinearGaussianConditional,
    PoissonFamily,
    glm_likelihood,
)
from ._continuous import (
    Beta,
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    StudentT,
    TruncatedNormal,
    Uniform,
)
from ._discrete import Bernoulli, Binomial, Categorical, NegativeBinomial, Poisson
from ._gaussian import FactoredMultivariateGaussian, GaussianProcess
from ._mixture import MixtureDistribution
from ._multivariate import Dirichlet, Multinomial, MultivariateNormal, VonMisesFisher, Wishart
from ._programs import PyMCModel, StanModel, UnnormalizedDistribution
from ._resampling import EpanechnikovKernel, GaussianKernel, SmoothingKernel
from ._transformed import BijectorTransformedDistribution, LinearPushforwardDistribution

__all__ = [
    "Bernoulli",
    "BernoulliFamily",
    "Beta",
    "BijectorTransformedDistribution",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Dirichlet",
    "EpanechnikovKernel",
    "Exponential",
    "FactoredMultivariateGaussian",
    "GLMFamily",
    "Gamma",
    "GaussianFamily",
    "GaussianKernel",
    "GaussianProcess",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "Laplace",
    "LinearGaussianConditional",
    "LinearPushforwardDistribution",
    "LogNormal",
    "MixtureDistribution",
    "Multinomial",
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "Poisson",
    "PoissonFamily",
    "PyMCModel",
    "SmoothingKernel",
    "StanModel",
    "StudentT",
    "TFPDistribution",
    "TruncatedNormal",
    "Uniform",
    "UnnormalizedDistribution",
    "VonMisesFisher",
    "Wishart",
    "glm_likelihood",
]
