from importlib.metadata import version as _version
import warnings as _warnings

__version__ = _version("probpipe")

# Suppress known TFP-internal warnings that are harmless but noisy.
# These are upstream issues in tfp-nightly's JAX substrate:
#   - float64 requests truncated to float32 (TFP internals assume x64)
#   - deprecated jax.interpreters.xla API usage
#   - np.shape(None) deprecation in random generators
_warnings.filterwarnings(
    "ignore",
    message=r"Explicitly requested dtype.*float64.*",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"jax\.interpreters\.xla\.pytype_aval_mappings is deprecated",
    category=DeprecationWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"shape requires ndarray or scalar arguments, got <class 'NoneType'>",
    category=DeprecationWarning,
)

from probpipe.distributions import (
    # Base classes
    Distribution,
    PyTreeArrayDistribution,
    ArrayDistribution,
    FlattenedView,
    TFPDistribution,
    EmpiricalDistribution,
    Provenance,
    # Global settings
    BootstrapDistribution,
    DEFAULT_NUM_EVALUATIONS,
    RETURN_APPROX_DIST,
    set_default_num_evaluations,
    set_return_approx_dist,
    monte_carlo,
    # Constraints
    Constraint,
    real,
    positive,
    non_negative,
    non_negative_integer,
    boolean,
    unit_interval,
    simplex,
    positive_definite,
    sphere,
    interval,
    greater_than,
    integer_interval,
    # Continuous
    Normal,
    Beta,
    Gamma,
    InverseGamma,
    Exponential,
    LogNormal,
    StudentT,
    Uniform,
    Cauchy,
    Laplace,
    HalfNormal,
    HalfCauchy,
    Pareto,
    TruncatedNormal,
    # Discrete
    Bernoulli,
    Binomial,
    Poisson,
    Categorical,
    NegativeBinomial,
    # Multivariate
    MultivariateNormal,
    Dirichlet,
    Multinomial,
    Wishart,
    VonMisesFisher,
    # Transformed
    TransformedDistribution,
    # Joint
    JointDistribution,
    ProductDistribution,
    SequentialJointDistribution,
    JointEmpirical,
    JointGaussian,
    DistributionView,
    ConditionedComponent,
    # Random functions
    RandomFunction,
    ArrayRandomFunction,
    GaussianRandomFunction,
    LinearBasisFunction,
)
from probpipe.core.node import WorkflowFunction, Module, wf
from probpipe.provenance import provenance_ancestors, provenance_dag
from probpipe.surrogate import (
    EmulatorMixin,
)
from probpipe.converters import (
    converter_registry,
    ConversionInfo,
    ConversionMethod,
    Converter,
)

__all__ = [
    # Base classes
    "Distribution",
    "PyTreeArrayDistribution",
    "ArrayDistribution",
    "FlattenedView",
    "TFPDistribution",
    "EmpiricalDistribution",
    "Provenance",
    # Constraints
    "Constraint",
    "real",
    "positive",
    "non_negative",
    "non_negative_integer",
    "boolean",
    "unit_interval",
    "simplex",
    "positive_definite",
    "sphere",
    "interval",
    "greater_than",
    "integer_interval",
    # Continuous
    "Normal",
    "Beta",
    "Gamma",
    "InverseGamma",
    "Exponential",
    "LogNormal",
    "StudentT",
    "Uniform",
    "Cauchy",
    "Laplace",
    "HalfNormal",
    "HalfCauchy",
    "Pareto",
    "TruncatedNormal",
    # Discrete
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Categorical",
    "NegativeBinomial",
    # Multivariate
    "MultivariateNormal",
    "Dirichlet",
    "Multinomial",
    "Wishart",
    "VonMisesFisher",
    # Transformed
    "TransformedDistribution",
    # Joint
    "JointDistribution",
    "ProductDistribution",
    "SequentialJointDistribution",
    "JointEmpirical",
    "JointGaussian",
    "DistributionView",
    "ConditionedComponent",
    # WorkflowFunction
    "WorkflowFunction",
    "Module",
    "wf",
    # Provenance
    "provenance_ancestors",
    "provenance_dag",
    # Random functions
    "RandomFunction",
    "ArrayRandomFunction",
    "GaussianRandomFunction",
    "LinearBasisFunction",
    # Emulators
    "EmulatorMixin",
    # Converters
    "converter_registry",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
]
