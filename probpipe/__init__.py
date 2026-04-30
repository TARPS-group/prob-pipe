from importlib.metadata import version as _version
import warnings as _warnings

__version__ = _version("probpipe")

# Suppress known TFP-internal warnings that are harmless but noisy.
# These are upstream issues in tfp-nightly's JAX substrate:
#   - deprecated jax.interpreters.xla API usage
#   - np.shape(None) deprecation in random generators
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

from probpipe._weights import Weights
from probpipe.core.distribution import (
    Distribution,
    RecordDistribution,
    NumericRecordDistribution,
    FlattenedView,
    EmpiricalDistribution,
    RecordEmpiricalDistribution,
    BroadcastDistribution,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    RecordBootstrapReplicateDistribution,
    DEFAULT_NUM_EVALUATIONS,
    RETURN_APPROX_DIST,
    set_default_num_evaluations,
    set_return_approx_dist,
    # Random functions
    RandomFunction,
    ArrayRandomFunction,
    # Random measures
    RandomMeasure,
    NumericRandomMeasure,
)
from probpipe.core._distribution_array import DistributionArray
from probpipe.distributions import (
    # TFP base
    TFPDistribution,
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
    bijector_for,
    register_bijector,
    # Joint
    ProductDistribution,
    SequentialJointDistribution,
    JointEmpirical,
    NumericJointEmpirical,
    JointGaussian,
    # Gaussian random functions
    GaussianRandomFunction,
    LinearBasisFunction,
    # KDE
    KDEDistribution,
)
from probpipe.core.record import Record, RecordTemplate, NumericRecordTemplate
from probpipe.core._numeric_record import NumericRecord
from probpipe.core._record_array import RecordArray, NumericRecordArray
from probpipe.core._array_backend import (
    AuxHooks,
    aux_for,
    register_aux,
)
from probpipe.core.config import WorkflowKind, prefect_config
from probpipe.core.node import WorkflowFunction, Module, workflow_function, workflow_method, abstract_workflow_method
from probpipe.core.provenance import Provenance, provenance_ancestors, provenance_dag
from probpipe.core.constraints import (
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
)
from probpipe.core.transition import (
    iterate,
    with_conversion,
    with_resampling,
)
from probpipe.modeling import GLMLikelihood, Likelihood, GenerativeLikelihood, IncrementalConditioner
from probpipe.record import Design, FullFactorialDesign
from probpipe.inference import (
    ApproximateDistribution,
    inference_method_registry,
    rwmh,
    condition_on_nutpie,
    sbi_learn_conditional,
    sbi_learn_likelihood,
)
from probpipe.modeling import ProbabilisticModel, SimpleModel, SimpleGenerativeModel
from probpipe.validation import predictive_check
from probpipe.core.protocols import (
    SupportsExpectation,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsConditioning,
)
from probpipe.converters import (
    converter_registry,
    ConversionInfo,
    ConversionMethod,
    Converter,
)

__all__ = [
    # Record
    "Record",
    "RecordTemplate",
    "NumericRecordTemplate",
    "NumericRecord",
    "RecordArray",
    "NumericRecordArray",
    # Array backend / aux registry
    "AuxHooks",
    "aux_for",
    "register_aux",
    # Weights
    "Weights",
    # Base classes
    "Distribution",
    "RecordDistribution",
    "NumericRecordDistribution",
    "FlattenedView",
    "DistributionArray",
    "TFPDistribution",
    "EmpiricalDistribution",
    "RecordEmpiricalDistribution",
    "BroadcastDistribution",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "RecordBootstrapReplicateDistribution",
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
    "bijector_for",
    "register_bijector",
    # Joint
    "ProductDistribution",
    "SequentialJointDistribution",
    "JointEmpirical",
    "NumericJointEmpirical",
    "JointGaussian",
    # Configuration
    "WorkflowKind",
    "prefect_config",
    # WorkflowFunction
    "WorkflowFunction",
    "Module",
    "workflow_function",
    "workflow_method",
    "abstract_workflow_method",
    # Provenance
    "provenance_ancestors",
    "provenance_dag",
    # Random functions
    "RandomFunction",
    "ArrayRandomFunction",
    "GaussianRandomFunction",
    "LinearBasisFunction",
    # Random measures
    "RandomMeasure",
    "NumericRandomMeasure",
    # KDE
    "KDEDistribution",
    # Protocols
    "SupportsExpectation",
    "SupportsSampling",
    "SupportsUnnormalizedLogProb",
    "SupportsLogProb",
    "SupportsMean",
    "SupportsVariance",
    "SupportsCovariance",
    "SupportsRandomLogProb",
    "SupportsRandomUnnormalizedLogProb",
    "SupportsConditioning",
    # Transition / iteration
    "iterate",
    "with_conversion",
    "with_resampling",
    # Modeling
    "GLMLikelihood",
    "Likelihood",
    "GenerativeLikelihood",
    "IncrementalConditioner",
    "ProbabilisticModel",
    "SimpleModel",
    "SimpleGenerativeModel",
    # Record-based designs
    "Design",
    "FullFactorialDesign",
    # Inference
    "ApproximateDistribution",
    "inference_method_registry",
    "rwmh",
    "condition_on_nutpie",
    "sbi_learn_conditional",
    "sbi_learn_likelihood",
    # Validation
    "predictive_check",
    # Converters
    "converter_registry",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
]

# ---------------------------------------------------------------------------
# Standalone operations (plain functions + WorkflowFunction wrappers)
# ---------------------------------------------------------------------------
from probpipe.core.ops import (  # noqa: E402
    sample,
    log_prob,
    prob,
    unnormalized_log_prob,
    unnormalized_prob,
    mean,
    variance,
    cov,
    expectation,
    random_log_prob,
    random_unnormalized_log_prob,
    condition_on,
    from_distribution,
)
