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
  - the resampling families: ``BootstrapReplicateDistribution``,
    ``BootstrapDistribution``, and ``KDEDistribution``, with the smoothing
    kernels ``SmoothingKernel``, ``GaussianKernel``, and ``EpanechnikovKernel``;
  - the mixture family, ``MixtureDistribution``;
  - the evaluation-result families, ``LinearPushforwardDistribution`` and
    ``BijectorTransformedDistribution``;
  - the random functions and random measures, ``RandomFunction`` and
    ``RandomMeasure``;
  - the Gaussian algebra's ``FactoredMultivariateGaussian``,
    ``GaussianRandomFunction``, ``GaussianProcess``, and
    ``LinearBasisFunction``;
  - the conditional families: ``LinearGaussianConditional``, the response
    families ``GLMFamily``, ``GaussianFamily``, ``BernoulliFamily``, and
    ``PoissonFamily``, and ``glm_likelihood``;
  - the program-defined families ``StanModel`` and ``PyMCModel``.

Importing the package registers the shipped converters with the converter
registry (:mod:`._converters`).
"""

from . import _converters
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
from ._gaussian import (
    FactoredMultivariateGaussian,
    GaussianProcess,
    GaussianRandomFunction,
    LinearBasisFunction,
)
from ._mixture import MixtureDistribution
from ._multivariate import Dirichlet, Multinomial, MultivariateNormal, VonMisesFisher, Wishart
from ._programs import PyMCModel, StanModel
from ._random_functions import RandomFunction, RandomMeasure
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
    "Bernoulli",
    "BernoulliFamily",
    "Beta",
    "BijectorTransformedDistribution",
    "Binomial",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
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
    "GaussianRandomFunction",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "KDEDistribution",
    "Laplace",
    "LinearBasisFunction",
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
    "RandomFunction",
    "RandomMeasure",
    "SmoothingKernel",
    "StanModel",
    "StudentT",
    "TFPDistribution",
    "TruncatedNormal",
    "Uniform",
    "VonMisesFisher",
    "Wishart",
    "glm_likelihood",
]
