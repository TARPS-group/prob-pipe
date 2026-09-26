import warnings as _warnings
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

# The importable ``probpipe`` package ships in the ``probpipe-core`` distribution
# (the friendly ``probpipe`` name is a code-less metapackage over it), so the
# version is read from ``probpipe-core``.
try:
    __version__ = _version("probpipe-core")
except _PackageNotFoundError:  # pragma: no cover - source tree with no install
    __version__ = "0.0.0+unknown"

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

# ``probpipe.distributions`` must initialize before any ``probpipe.core`` module
# that imports its base class. Importing the base initializes the whole package,
# whose families import such modules back, so a core module loaded first is still
# partially initialized when a family imports it.
from probpipe import distributions

# isort: split

from probpipe._weights import Weights
from probpipe.converters import (
    ConversionInfo,
    ConversionMethod,
    Converter,
    converter_registry,
)
from probpipe.core._array_backend import (
    ArrayBackend,
    array_backend_for,
    register_array_backend,
)
from probpipe.core._batch import Batch, BatchSpec
from probpipe.core._broadcast_distributions import BroadcastDistribution
from probpipe.core._dispatch import MathematicalDomainError, ResolutionError
from probpipe.core._distribution_array import DistributionArray
from probpipe.core._empirical import (
    BootstrapReplicateDistribution,
    EmpiricalDistribution,
    RecordBootstrapReplicateDistribution,
    RecordEmpiricalDistribution,
)
from probpipe.core._function_batch import FunctionBatch
from probpipe.core._numeric import Numeric
from probpipe.core._numeric_array import NumericArray
from probpipe.core._numeric_array_batch import NumericArrayBatch
from probpipe.core._numeric_record import NumericRecord
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._numeric_record_distribution import (
    BootstrapDistribution,
    FlatNumericRecordDistribution,
    FlattenedDistributionView,
    NumericRecordDistribution,
    NumericRecordDistributionView,
)
from probpipe.core._opaque import Opaque, OpaqueSpec
from probpipe.core._opaque_batch import OpaqueBatch
from probpipe.core._random_functions import ArrayRandomFunction, RandomFunction
from probpipe.core._random_measures import NumericRandomMeasure, RandomMeasure
from probpipe.core._record_batch import RecordBatch
from probpipe.core._record_distribution import RecordDistribution
from probpipe.core._specs import (
    FunctionSpec,
    InputSpec,
    NumericArraySpec,
    NumericRecordSpec,
    NumericSpec,
    OutputSpec,
    RecordSpec,
    TermSpec,
)
from probpipe.core._workflow_context import workflow_run
from probpipe.core._workflow_errors import (
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
    UnmanagedConcurrentWorkflowEntryError,
)
from probpipe.core._workflow_replay import replay_run
from probpipe.core.config import ProvenanceMode, WorkflowKind, prefect_config, provenance_config
from probpipe.core.constraints import (
    Constraint,
    boolean,
    greater_than,
    integer_interval,
    interval,
    non_negative,
    non_negative_integer,
    positive,
    positive_definite,
    real,
    simplex,
    sphere,
    unit_interval,
)
from probpipe.core.named_tree import NamedTree
from probpipe.core.node import (
    Function,
    Module,
    abstract_workflow_method,
    function,
    workflow_method,
)
from probpipe.core.protocols import (
    SupportsApproximateConditioning,
    SupportsArrayBackend,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from probpipe.core.provenance import ParentInfo, Provenance, provenance_ancestors, provenance_dag
from probpipe.core.record import (
    Record,
)
from probpipe.core.tracked import Annotated, TrackedTerm
from probpipe.core.transition import (
    iterate,
    with_conversion,
    with_resampling,
)
from probpipe.distributions import (
    # Discrete
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Dirichlet,
    Exponential,
    Gamma,
    # Gaussian random functions
    GaussianRandomFunction,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    JointEmpirical,
    JointGaussian,
    # KDE
    KDEDistribution,
    Laplace,
    LinearBasisFunction,
    LogNormal,
    Multinomial,
    # Multivariate
    MultivariateNormal,
    NegativeBinomial,
    # Continuous
    Normal,
    NumericJointEmpirical,
    Pareto,
    Poisson,
    # Joint
    ProductDistribution,
    SequentialJointDistribution,
    StudentT,
    # TFP base
    TFPDistribution,
    # Transformed
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
    VonMisesFisher,
    Wishart,
    bijector_for,
    register_bijector,
)
from probpipe.distributions._distribution import (
    DEFAULT_NUM_EVALUATIONS,
    RETURN_APPROX_DIST,
    Distribution,
    DistributionSpec,
    NumericDistribution,
    set_default_num_evaluations,
    set_return_approx_dist,
)
from probpipe.inference import (
    ApproximateDistribution,
    BayesFlowLikelihood,
    BayesFlowModel,
    BayesFlowRatio,
    MinibatchedDistribution,
    condition_on_nutpie,
    elliptical_slice,
    inference_method_registry,
    learn_amortized_likelihood,
    learn_amortized_posterior,
    learn_amortized_ratio,
    rwmh,
)
from probpipe.modeling import (
    ConditionallyIndependentLikelihood,
    GenerativeLikelihood,
    GLMLikelihood,
    IncrementalConditioner,
    Likelihood,
    ProbabilisticModel,
    SimpleGenerativeModel,
    SimpleModel,
)
from probpipe.record import Design, FullFactorialDesign
from probpipe.validation import predictive_check

__all__ = [
    "Annotated",
    "ApproximateDistribution",
    "ArrayBackend",
    "ArrayRandomFunction",
    "Batch",
    "BatchSpec",
    "BayesFlowLikelihood",
    "BayesFlowModel",
    "BayesFlowRatio",
    "Bernoulli",
    "Beta",
    "Binomial",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "BroadcastDistribution",
    "Categorical",
    "Cauchy",
    "ConditionallyIndependentLikelihood",
    "Constraint",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
    "Design",
    "Dirichlet",
    "Distribution",
    "DistributionArray",
    "DistributionSpec",
    "EmpiricalDistribution",
    "Exponential",
    "FlatNumericRecordDistribution",
    "FlattenedDistributionView",
    "FullFactorialDesign",
    "Function",
    "FunctionBatch",
    "FunctionSpec",
    "GLMLikelihood",
    "Gamma",
    "GaussianRandomFunction",
    "GenerativeLikelihood",
    "HalfCauchy",
    "HalfNormal",
    "IncrementalConditioner",
    "InputSpec",
    "InverseGamma",
    "JointEmpirical",
    "JointGaussian",
    "KDEDistribution",
    "Laplace",
    "Likelihood",
    "LinearBasisFunction",
    "LogNormal",
    "MathematicalDomainError",
    "MinibatchedDistribution",
    "Module",
    "Multinomial",
    "MultivariateNormal",
    "NamedTree",
    "NegativeBinomial",
    "Normal",
    "Numeric",
    "NumericArray",
    "NumericArrayBatch",
    "NumericArraySpec",
    "NumericDistribution",
    "NumericJointEmpirical",
    "NumericRandomMeasure",
    "NumericRecord",
    "NumericRecordBatch",
    "NumericRecordDistribution",
    "NumericRecordDistributionView",
    "NumericRecordSpec",
    "NumericSpec",
    "Opaque",
    "OpaqueBatch",
    "OpaqueSpec",
    "OutputSpec",
    "ParentInfo",
    "Pareto",
    "Poisson",
    "ProbabilisticModel",
    "ProductDistribution",
    "Provenance",
    "ProvenanceMode",
    "RandomFunction",
    "RandomMeasure",
    "Record",
    "RecordBatch",
    "RecordBootstrapReplicateDistribution",
    "RecordDistribution",
    "RecordEmpiricalDistribution",
    "RecordSpec",
    "ReplayCompatibilityError",
    "ReplayUnsupportedCallableError",
    "ResolutionError",
    "SequentialJointDistribution",
    "SimpleGenerativeModel",
    "SimpleModel",
    "StudentT",
    "SupportsApproximateConditioning",
    "SupportsArrayBackend",
    "SupportsCovariance",
    "SupportsExactConditioning",
    "SupportsExpectation",
    "SupportsLogProb",
    "SupportsMean",
    "SupportsQuantile",
    "SupportsRandomLogProb",
    "SupportsRandomUnnormalizedLogProb",
    "SupportsSampling",
    "SupportsUnnormalizedLogProb",
    "SupportsVariance",
    "TFPDistribution",
    "TermSpec",
    "TrackedTerm",
    "TransformedDistribution",
    "TruncatedNormal",
    "Uniform",
    "UnmanagedConcurrentWorkflowEntryError",
    "VonMisesFisher",
    "Weights",
    "Wishart",
    "WorkflowKind",
    "abstract_workflow_method",
    "array_backend_for",
    "bijector_for",
    "boolean",
    "condition_on_nutpie",
    "converter_registry",
    "elliptical_slice",
    "function",
    "greater_than",
    "inference_method_registry",
    "integer_interval",
    "interval",
    "iterate",
    "learn_amortized_likelihood",
    "learn_amortized_posterior",
    "learn_amortized_ratio",
    "non_negative",
    "non_negative_integer",
    "positive",
    "positive_definite",
    "predictive_check",
    "prefect_config",
    "provenance_ancestors",
    "provenance_config",
    "provenance_dag",
    "real",
    "register_array_backend",
    "register_bijector",
    "replay_run",
    "rwmh",
    "simplex",
    "sphere",
    "unit_interval",
    "with_conversion",
    "with_resampling",
    "workflow_method",
    "workflow_run",
]

# ---------------------------------------------------------------------------
# Standalone operations (plain functions + Function wrappers)
# ---------------------------------------------------------------------------
from probpipe.core.ops import (
    condition_on,
    cov,
    expectation,
    from_distribution,
    log_prob,
    mean,
    prob,
    quantile,
    random_log_prob,
    random_unnormalized_log_prob,
    sample,
    unnormalized_log_prob,
    unnormalized_prob,
    variance,
)
