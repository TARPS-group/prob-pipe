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
from probpipe.core._array_backend import (
    ArrayBackend,
    array_backend_for,
    register_array_backend,
)
from probpipe.core._batch import Batch, BatchSpec
from probpipe.core._dispatch import MathematicalDomainError, ResolutionError
from probpipe.core._function_batch import FunctionBatch
from probpipe.core._numeric import Numeric
from probpipe.core._numeric_array import NumericArray
from probpipe.core._numeric_array_batch import NumericArrayBatch
from probpipe.core._numeric_record import NumericRecord
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._opaque import Opaque, OpaqueSpec
from probpipe.core._opaque_batch import OpaqueBatch
from probpipe.core._record_batch import RecordBatch
from probpipe.core._specs import (
    InputSpec,
    NumericArraySpec,
    NumericRecordSpec,
    NumericSpec,
    OutputSpec,
    RecordSpec,
    TermSpec,
)
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
from probpipe.core.protocols import SupportsArrayBackend
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
from probpipe.distributions._batches import DistributionBatch
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
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
from probpipe.distributions._conversion import ConversionInfo, Converter, converter_registry
from probpipe.distributions._distribution import (
    DEFAULT_NUM_EVALUATIONS,
    Distribution,
    DistributionSpec,
    NumericDistribution,
    set_default_num_evaluations,
)
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.families import (
    Bernoulli,
    Beta,
    BijectorTransformedDistribution,
    Binomial,
    Categorical,
    Cauchy,
    Dirichlet,
    Exponential,
    Gamma,
    GaussianRandomFunction,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LinearBasisFunction,
    LogNormal,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    RandomFunction,
    RandomMeasure,
    StudentT,
    TFPDistribution,
    TruncatedNormal,
    Uniform,
    UnnormalizedDistribution,
    VonMisesFisher,
    Wishart,
)
from probpipe.families._resampling import (
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    KDEDistribution,
)
from probpipe.functions import (
    AbstractModule,
    Module,
    abstract_workflow_method,
    bijector_for,
    function,
    register_bijector,
    workflow_method,
)
from probpipe.functions._call import ApplicabilityError, CallReport
from probpipe.functions._context import workflow_run
from probpipe.functions._errors import (
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
    UnmanagedConcurrentWorkflowEntryError,
)
from probpipe.functions._replay import replay_run
from probpipe.functions._result import ResultKindError, ResultSchemaError
from probpipe.functions._rules import evaluation_rule_registry
from probpipe.inference import (
    ApproximateDistribution,
    BayesFlowLikelihood,
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
from probpipe.record import Design, FullFactorialDesign
from probpipe.validation import predictive_check
from probpipe.values import (
    Function,
    FunctionSpec,
    SupportsDifferentiation,
    SupportsInverse,
    SupportsLogDetJacobian,
    is_differentiable,
    is_invertible,
)

__all__ = [
    "AbstractModule",
    "Annotated",
    "ApplicabilityError",
    "ApproximateDistribution",
    "ArrayBackend",
    "Batch",
    "BatchSpec",
    "BayesFlowLikelihood",
    "BayesFlowRatio",
    "Bernoulli",
    "Beta",
    "BijectorTransformedDistribution",
    "Binomial",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "CallReport",
    "Categorical",
    "Cauchy",
    "Constraint",
    "ConversionInfo",
    "Converter",
    "Design",
    "Dirichlet",
    "Distribution",
    "DistributionBatch",
    "DistributionSpec",
    "EmpiricalDistribution",
    "Exponential",
    "FullFactorialDesign",
    "Function",
    "FunctionBatch",
    "FunctionSpec",
    "Gamma",
    "GaussianRandomFunction",
    "HalfCauchy",
    "HalfNormal",
    "InputSpec",
    "InverseGamma",
    "KDEDistribution",
    "Laplace",
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
    "NumericRecord",
    "NumericRecordBatch",
    "NumericRecordSpec",
    "NumericSpec",
    "Opaque",
    "OpaqueBatch",
    "OpaqueSpec",
    "OutputSpec",
    "ParentInfo",
    "Pareto",
    "Poisson",
    "Provenance",
    "ProvenanceMode",
    "PyMCModel",
    "RandomFunction",
    "RandomMeasure",
    "Record",
    "RecordBatch",
    "RecordSpec",
    "ReplayCompatibilityError",
    "ReplayUnsupportedCallableError",
    "ResolutionError",
    "ResultKindError",
    "ResultSchemaError",
    "StanModel",
    "StudentT",
    "SupportsApproximateConditioning",
    "SupportsArrayBackend",
    "SupportsCovariance",
    "SupportsDifferentiation",
    "SupportsExactConditioning",
    "SupportsExpectation",
    "SupportsInverse",
    "SupportsLogDetJacobian",
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
    "TruncatedNormal",
    "Uniform",
    "UnmanagedConcurrentWorkflowEntryError",
    "UnnormalizedDistribution",
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
    "evaluation_rule_registry",
    "expectation_method_registry",
    "function",
    "greater_than",
    "inference_method_registry",
    "integer_interval",
    "interval",
    "is_differentiable",
    "is_invertible",
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
from probpipe.operations import expectation, expectation_method_registry


def __getattr__(name: str):
    """The program-defined families, which ``probpipe`` exports lazily."""
    if name in ("PyMCModel", "StanModel"):
        from probpipe.families import _programs

        return getattr(_programs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
