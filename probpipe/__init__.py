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

from probpipe._weights import Weights
from probpipe.converters import (
    ConversionInfo,
    ConversionMethod,
    Converter,
    converter_registry,
)
from probpipe.core._array_backend import (
    AuxHooks,
    aux_for,
    register_aux,
)
from probpipe.core._distribution_array import DistributionArray
from probpipe.core._numeric_record import NumericRecord
from probpipe.core._record_array import NumericRecordArray, RecordArray
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
from probpipe.core.distribution import (
    DEFAULT_NUM_EVALUATIONS,
    RETURN_APPROX_DIST,
    ArrayRandomFunction,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    BroadcastDistribution,
    Distribution,
    EmpiricalDistribution,
    FlatNumericRecordDistribution,
    FlattenedDistributionView,
    NumericRandomMeasure,
    NumericRecordDistribution,
    NumericRecordDistributionView,
    # Random functions
    RandomFunction,
    # Random measures
    RandomMeasure,
    RecordBootstrapReplicateDistribution,
    RecordDistribution,
    RecordEmpiricalDistribution,
    set_default_num_evaluations,
    set_return_approx_dist,
)
from probpipe.core.event_template import (
    ArraySpec,
    DistributionSpec,
    EventTemplate,
    FunctionSpec,
    LeafSpec,
    NumericEventTemplate,
    OpaqueSpec,
)
from probpipe.core.node import (
    Module,
    WorkflowFunction,
    abstract_workflow_method,
    workflow_function,
    workflow_method,
)
from probpipe.core.protocols import (
    SupportsArrayBackend,
    SupportsConditioning,
    SupportsCovariance,
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
from probpipe.core.tracked import Annotated, Tracked
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
    # Identity & metadata mixins
    "Annotated",
    # Inference
    "ApproximateDistribution",
    "ArrayRandomFunction",
    "ArraySpec",
    # Array backend / aux registry
    "AuxHooks",
    "BayesFlowLikelihood",
    "BayesFlowModel",
    "BayesFlowRatio",
    # Discrete
    "Bernoulli",
    "Beta",
    "Binomial",
    "BootstrapDistribution",
    "BootstrapReplicateDistribution",
    "BroadcastDistribution",
    "Categorical",
    "Cauchy",
    "ConditionallyIndependentLikelihood",
    # Constraints
    "Constraint",
    "ConversionInfo",
    "ConversionMethod",
    "Converter",
    # Record-based designs
    "Design",
    "Dirichlet",
    # Base classes
    "Distribution",
    "DistributionArray",
    "DistributionSpec",
    "EmpiricalDistribution",
    "EventTemplate",
    "Exponential",
    "FlatNumericRecordDistribution",
    "FlattenedDistributionView",
    "FullFactorialDesign",
    "FunctionSpec",
    # Modeling
    "GLMLikelihood",
    "Gamma",
    "GaussianRandomFunction",
    "GenerativeLikelihood",
    "HalfCauchy",
    "HalfNormal",
    "IncrementalConditioner",
    "InverseGamma",
    "JointEmpirical",
    "JointGaussian",
    # KDE
    "KDEDistribution",
    "Laplace",
    "LeafSpec",
    "Likelihood",
    "LinearBasisFunction",
    "LogNormal",
    "MinibatchedDistribution",
    "Module",
    "Multinomial",
    # Multivariate
    "MultivariateNormal",
    "NegativeBinomial",
    # Continuous
    "Normal",
    "NumericEventTemplate",
    "NumericJointEmpirical",
    "NumericRandomMeasure",
    "NumericRecord",
    "NumericRecordArray",
    "NumericRecordDistribution",
    "NumericRecordDistributionView",
    "OpaqueSpec",
    "ParentInfo",
    "Pareto",
    "Poisson",
    "ProbabilisticModel",
    # Joint
    "ProductDistribution",
    "Provenance",
    "ProvenanceMode",
    # Random functions
    "RandomFunction",
    # Random measures
    "RandomMeasure",
    # Record
    "Record",
    "RecordArray",
    "RecordBootstrapReplicateDistribution",
    "RecordDistribution",
    "RecordEmpiricalDistribution",
    "SequentialJointDistribution",
    "SimpleGenerativeModel",
    "SimpleModel",
    "StudentT",
    "SupportsArrayBackend",
    "SupportsConditioning",
    "SupportsCovariance",
    # Protocols
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
    # Identity & metadata mixins
    "Tracked",
    # Transformed
    "TransformedDistribution",
    "TruncatedNormal",
    "Uniform",
    "VonMisesFisher",
    # Weights
    "Weights",
    "Wishart",
    # WorkflowFunction
    "WorkflowFunction",
    # Configuration
    "WorkflowKind",
    "abstract_workflow_method",
    "aux_for",
    "bijector_for",
    "boolean",
    "condition_on_nutpie",
    # Converters
    "converter_registry",
    "elliptical_slice",
    "greater_than",
    "inference_method_registry",
    "integer_interval",
    "interval",
    # Transition / iteration
    "iterate",
    "learn_amortized_likelihood",
    "learn_amortized_posterior",
    "learn_amortized_ratio",
    "non_negative",
    "non_negative_integer",
    "positive",
    "positive_definite",
    # Validation
    "predictive_check",
    "prefect_config",
    # Provenance
    "provenance_ancestors",
    "provenance_config",
    "provenance_dag",
    "real",
    "register_aux",
    "register_bijector",
    "rwmh",
    "simplex",
    "sphere",
    "unit_interval",
    "with_conversion",
    "with_resampling",
    "workflow_function",
    "workflow_method",
]

# ---------------------------------------------------------------------------
# Standalone operations (plain functions + WorkflowFunction wrappers)
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
