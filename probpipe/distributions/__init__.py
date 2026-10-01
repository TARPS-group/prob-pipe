from ..core._random_functions import ArrayRandomFunction, RandomFunction
from . import _composition
from ._batches import ConditionalDistributionBatch, DistributionBatch
from ._bijector_dispatch import (
    bijector_for,
    register_bijector,
)
from ._capabilities import (
    SupportsConditionalCovariance,
    SupportsConditionalExpectation,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalQuantile,
    SupportsConditionalRandomLogProb,
    SupportsConditionalRandomUnnormalizedLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
    SupportsMarginals,
)
from ._conditional import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
    ConditionalNumericDistribution,
    FullyNumericConditionalDistribution,
    NumericConditionalDistribution,
)
from ._distribution import Distribution, DistributionSpec, NumericDistribution
from ._empirical import EmpiricalDistribution
from ._factored import (
    FactoredConditionalDistribution,
    FactoredConditionalNumericDistribution,
    FactoredDistribution,
    FactoredFullyNumericConditionalDistribution,
    FactoredNumericConditionalDistribution,
    FactoredNumericDistribution,
    SupportsFactors,
)
from ._tfp_base import TFPDistribution
from ._views import FieldView
from .continuous import (
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
from .discrete import (
    Bernoulli,
    Binomial,
    Categorical,
    NegativeBinomial,
    Poisson,
)
from .gaussian_random_function import (
    GaussianRandomFunction,
    LinearBasisFunction,
)
from .joint import (
    JointGaussian,
    ProductDistribution,
    SequentialJointDistribution,
)
from .multivariate import (
    Dirichlet,
    Multinomial,
    MultivariateNormal,
    VonMisesFisher,
    Wishart,
)
from .transformed import TransformedDistribution

__all__ = [
    "ArrayRandomFunction",
    # Discrete
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "ConditionalDistribution",
    "ConditionalDistributionBatch",
    "ConditionalDistributionSpec",
    "ConditionalNumericDistribution",
    "Dirichlet",
    "Distribution",
    "DistributionBatch",
    "DistributionSpec",
    "EmpiricalDistribution",
    "Exponential",
    "FactoredConditionalDistribution",
    "FactoredConditionalNumericDistribution",
    "FactoredDistribution",
    "FactoredFullyNumericConditionalDistribution",
    "FactoredNumericConditionalDistribution",
    "FactoredNumericDistribution",
    "FieldView",
    "FullyNumericConditionalDistribution",
    "Gamma",
    "GaussianRandomFunction",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "JointGaussian",
    "Laplace",
    "LinearBasisFunction",
    "LogNormal",
    "Multinomial",
    # Multivariate
    "MultivariateNormal",
    "NegativeBinomial",
    # Univariate continuous
    "Normal",
    "NumericConditionalDistribution",
    "NumericDistribution",
    "Pareto",
    "Poisson",
    # Joint
    "ProductDistribution",
    # Random functions
    "RandomFunction",
    "SequentialJointDistribution",
    "StudentT",
    "SupportsConditionalCovariance",
    "SupportsConditionalExpectation",
    "SupportsConditionalLogProb",
    "SupportsConditionalMarginals",
    "SupportsConditionalMean",
    "SupportsConditionalQuantile",
    "SupportsConditionalRandomLogProb",
    "SupportsConditionalRandomUnnormalizedLogProb",
    "SupportsConditionalSampling",
    "SupportsConditionalUnnormalizedLogProb",
    "SupportsConditionalVariance",
    "SupportsFactors",
    "SupportsMarginals",
    # TFP base
    "TFPDistribution",
    # Transformed
    "TransformedDistribution",
    "TruncatedNormal",
    "Uniform",
    "VonMisesFisher",
    "Wishart",
    "bijector_for",
    "register_bijector",
]
