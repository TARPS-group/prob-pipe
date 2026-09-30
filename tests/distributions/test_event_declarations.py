"""Every distribution stores a complete event declaration, and draws what it declares."""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import pathlib
import pickle
import pkgutil
import sys
import tempfile
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.glm as tfp_glm

import probpipe
from probpipe import (
    Bernoulli,
    Beta,
    Binomial,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    Categorical,
    Cauchy,
    Dirichlet,
    Distribution,
    DistributionArray,
    EmpiricalDistribution,
    Exponential,
    FlatNumericRecordDistribution,
    Gamma,
    GLMLikelihood,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    JointEmpirical,
    JointGaussian,
    KDEDistribution,
    Laplace,
    LinearBasisFunction,
    LogNormal,
    MinibatchedDistribution,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    NumericDistribution,
    NumericRecordDistribution,
    NumericRecordSpec,
    NumericSpec,
    Pareto,
    Poisson,
    ProductDistribution,
    Record,
    RecordDistribution,
    SequentialJointDistribution,
    SimpleGenerativeModel,
    SimpleModel,
    StudentT,
    TFPDistribution,
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
    VonMisesFisher,
    Wishart,
    sample,
)
from probpipe.core._broadcast_distributions import (
    BroadcastDistribution,
    _ListMarginal,
    _make_mixture_marginal,
    _MixtureMarginal,
    _RecordMarginal,
)
from probpipe.core._empirical import (
    RecordBootstrapReplicateDistribution,
    RecordEmpiricalDistribution,
)
from probpipe.core._numeric_record_distribution import (
    FlattenedDistributionView,
    NumericRecordDistributionView,
)
from probpipe.core._random_measures import RandomMeasure
from probpipe.core._record_distribution import _RecordDistributionView
from probpipe.core._specs import RecordSpec
from probpipe.distributions import FactoredDistribution, FactoredNumericDistribution, FieldView
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.distributions._joint_empirical import NumericJointEmpirical
from probpipe.distributions._product import TFPProductDistribution
from probpipe.distributions._views import _RenamedDistribution
from probpipe.distributions.gaussian_random_function import (
    _IndependentSumGRF,
    _LinearMapGRF,
    _ScaledGRF,
    _ShiftedGRF,
)
from probpipe.inference._approximate_distribution import (
    ApproximateDistribution,
    make_posterior,
)
from probpipe.inference._bayesflow_posteriors import BayesFlowModel
from probpipe.inference._minibatch import (
    _FixedMinibatchDistribution,
    _MinibatchLogProbAtPoint,
    _RandomMinibatchLogProb,
)
from probpipe.modeling import PyMCModel, StanModel
from probpipe.modeling._likelihood import GenerativeLikelihood
from probpipe.modeling._stan import _UnconstrainedStanView

# -- Constructions ------------------------------------------------------------


# The callables a construction passes are module-level, so they pickle.
def _features(X, output_shape=()):
    phi = jnp.concatenate([X, X**2], -1)
    return jnp.stack([phi] * output_shape[0], -2) if output_shape else phi


def _basis_function(name: str = "f", output_shape: tuple[int, ...] = ()) -> LinearBasisFunction:
    width = 2 * max(1, int(np.prod(output_shape)))
    weights = MultivariateNormal("w", loc=jnp.zeros(width), cov=jnp.eye(width))
    return LinearBasisFunction(
        name,
        feature_map=functools.partial(_features, output_shape=output_shape),
        weights=weights,
        input_shape=(1,),
        output_shape=output_shape,
    )


def _measure() -> MinibatchedDistribution:
    X = jax.random.normal(jax.random.PRNGKey(0), (20, 2))
    y = (X[:, 0] > 0).astype(jnp.float32)
    prior = MultivariateNormal("theta", loc=jnp.zeros(2), cov=jnp.eye(2))
    likelihood = GLMLikelihood(tfp_glm.Bernoulli(), x=X, fit_intercept=False)
    return MinibatchedDistribution(
        "measure", prior, likelihood, Record("r", X=X, y=y), batch_size=5
    )


class _Likelihood:
    data_template = RecordSpec(y=(3,))

    def log_likelihood(self, params, data):
        return jnp.asarray(0.0)


class _Simulator(GenerativeLikelihood):
    def log_likelihood(self, params, data):
        return jnp.asarray(0.0)

    def generate_data(self, params, n_samples, *, key=None):
        return jnp.zeros((n_samples, 1))


def _pymc_model_fn(y=None):
    import pymc as pm

    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
        pm.Normal("slope", 0, 1, shape=2)
        pm.Normal("y", 0, 1, observed=y)
    return model


def _pymc_model() -> PyMCModel:
    pytest.importorskip("pymc")
    return PyMCModel("model", _pymc_model_fn)


def _stan_model() -> StanModel:
    pytest.importorskip("bridgestan")
    stan_file = pathlib.Path(tempfile.mkdtemp()) / "declared.stan"
    stan_file.write_text("parameters { real mu; } model { mu ~ normal(0, 1); }")
    return StanModel("model", str(stan_file))


def _conditional(z):
    return Normal("x", z, 1.0)


# One construction per concrete class, keyed by the class it represents.
_CONSTRUCTIONS: dict[type, Callable[[], Distribution]] = {
    Normal: lambda: Normal("x", 0.0, 1.0),
    Beta: lambda: Beta("x", 2.0, 3.0),
    Gamma: lambda: Gamma("x", 2.0, 1.0),
    InverseGamma: lambda: InverseGamma("x", 2.0, 1.0),
    Exponential: lambda: Exponential("x", 1.0),
    LogNormal: lambda: LogNormal("x", 0.0, 1.0),
    StudentT: lambda: StudentT("x", 3.0, 0.0, 1.0),
    Uniform: lambda: Uniform("x", -1.0, 2.0),
    Cauchy: lambda: Cauchy("x", 0.0, 1.0),
    Laplace: lambda: Laplace("x", 0.0, 1.0),
    HalfNormal: lambda: HalfNormal("x", 1.0),
    HalfCauchy: lambda: HalfCauchy("x", 0.5, 1.0),
    Pareto: lambda: Pareto("x", 2.0, 1.5),
    TruncatedNormal: lambda: TruncatedNormal("x", 0.0, 1.0, -1.0, 1.0),
    Bernoulli: lambda: Bernoulli("x", probs=0.3),
    Binomial: lambda: Binomial("x", 5, probs=0.3),
    Poisson: lambda: Poisson("x", 2.0),
    Categorical: lambda: Categorical("x", probs=[0.2, 0.3, 0.5]),
    NegativeBinomial: lambda: NegativeBinomial("x", 5.0, probs=0.3),
    MultivariateNormal: lambda: MultivariateNormal("x", jnp.zeros(3), cov=jnp.eye(3)),
    Dirichlet: lambda: Dirichlet("x", jnp.ones(3)),
    Multinomial: lambda: Multinomial("x", 4.0, probs=jnp.array([0.2, 0.3, 0.5])),
    Wishart: lambda: Wishart("x", 4.0, scale_tril=jnp.eye(2)),
    VonMisesFisher: lambda: VonMisesFisher("x", jnp.array([0.0, 1.0]), 2.0),
    KDEDistribution: lambda: KDEDistribution("k", jnp.zeros((10, 2))),
    TransformedDistribution: lambda: TransformedDistribution("t", Normal("x", 0.0, 1.0), tfb.Exp()),
    EmpiricalDistribution: lambda: EmpiricalDistribution("e", ["a", "b"]),
    RecordEmpiricalDistribution: lambda: EmpiricalDistribution("r", jnp.zeros((5, 2))),
    BootstrapReplicateDistribution: lambda: BootstrapReplicateDistribution(
        "b", Normal("x", 0.0, 1.0), replicate_size=3
    ),
    RecordBootstrapReplicateDistribution: lambda: BootstrapReplicateDistribution(
        "b", jnp.zeros((5, 2))
    ),
    BootstrapDistribution: lambda: BootstrapDistribution("expectation", jnp.zeros((10, 3))),
    ProductDistribution: lambda: ProductDistribution(
        a=Normal("a", 0.0, 1.0), e=EmpiricalDistribution("e", ["x", "y"])
    ),
    TFPProductDistribution: lambda: ProductDistribution(
        a=Normal("a", 0.0, 1.0), b=Gamma("b", 2.0, 1.0)
    ),
    SequentialJointDistribution: lambda: SequentialJointDistribution(
        z=Normal("z", 0.0, 1.0), x=_conditional
    ),
    JointGaussian: lambda: JointGaussian(mean=jnp.zeros(3), cov=jnp.eye(3), x=1, y=2),
    JointEmpirical: lambda: JointEmpirical(
        labels=np.array(["a", "b"], dtype=object), ids=np.array([0, 1])
    ),
    NumericJointEmpirical: lambda: JointEmpirical(u=np.ones((4, 2)), v=np.zeros(4)),
    DistributionArray: lambda: DistributionArray.from_batched_params(
        Normal, loc=jnp.zeros(3), scale=1.0, name="x"
    ),
    BroadcastDistribution: lambda: BroadcastDistribution(
        {"x": jnp.zeros(3)}, jnp.zeros(3), broadcast_args=["x"]
    ),
    _RecordMarginal: lambda: _RecordMarginal(jnp.zeros((4, 2)), name="m"),
    _MixtureMarginal: lambda: _make_mixture_marginal(
        [Normal("y", 0.0, 1.0), Normal("y", 1.0, 1.0)]
    ),
    _ListMarginal: lambda: _ListMarginal(["a", "b"]),
    FlattenedDistributionView: lambda: ProductDistribution(
        a=Normal("a", 0.0, 1.0), b=Normal("b", 0.0, 1.0)
    ).as_flat_distribution(),
    NumericRecordDistributionView: lambda: MultivariateNormal(
        "theta", jnp.zeros(3), cov=jnp.eye(3)
    ).as_record_distribution(template=NumericRecordSpec(a=(), b=(2,))),
    _RecordDistributionView: lambda: ProductDistribution(
        a=Normal("a", 0.0, 1.0), b=Normal("b", 0.0, 1.0)
    )["a"],
    RandomMeasure: lambda: RandomMeasure("m"),
    MinibatchedDistribution: _measure,
    _FixedMinibatchDistribution: lambda: _measure()._draw_one(jax.random.PRNGKey(0)),
    _RandomMinibatchLogProb: lambda: _measure()._random_unnormalized_log_prob(),
    _MinibatchLogProbAtPoint: lambda: _measure()._random_unnormalized_log_prob()(jnp.zeros(2)),
    LinearBasisFunction: _basis_function,
    _LinearMapGRF: lambda: jnp.eye(2) @ _basis_function(output_shape=(2,)),
    _ShiftedGRF: lambda: _basis_function() + 1.0,
    _ScaledGRF: lambda: 2.0 * _basis_function(),
    _IndependentSumGRF: lambda: _basis_function("f") + _basis_function("g"),
    ApproximateDistribution: lambda: make_posterior(
        [jnp.zeros((10, 2))],
        parents=(MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2)),),
        algorithm="test",
    ),
    SimpleModel: lambda: SimpleModel(Normal("theta", 0.0, 1.0), _Likelihood()),
    SimpleGenerativeModel: lambda: SimpleGenerativeModel(Normal("theta", 0.0, 1.0), _Simulator()),
    BayesFlowModel: lambda: BayesFlowModel(
        None, Normal("theta", 0.0, 1.0), _Simulator(), method="npe", data_dim=1
    ),
    PyMCModel: _pymc_model,
    StanModel: _stan_model,
    _UnconstrainedStanView: lambda: _stan_model().as_unconstrained_distribution(),
    FieldView: lambda: FieldView(
        ProductDistribution(a=Normal("a", 0.0, 1.0), b=Normal("b", 0.0, 1.0)), "a"
    ),
    FactoredDistribution: lambda: Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0),
    _RenamedDistribution: lambda: ProductDistribution(
        a=Normal("a", 0.0, 1.0), b=Normal("b", 0.0, 1.0)
    ).with_path_names(a="x"),
}

# Bases a concrete class specializes, constructed only through one.
_BASES = frozenset(
    {
        NumericDistribution,
        FactoredNumericDistribution,
        TFPDistribution,
        RecordDistribution,
        NumericRecordDistribution,
        FlatNumericRecordDistribution,
    }
)


def _rows(failures: dict[type, pytest.MarkDecorator] | None = None) -> list:
    """One case per construction, marked where the check fails for a known reason."""
    failures = failures or {}
    return [
        pytest.param(cls, make, id=cls.__name__, marks=failures.get(cls, ()))
        for cls, make in _CONSTRUCTIONS.items()
    ]


def _reachable(cls: type) -> bool:
    """Whether ``cls`` is found in its module under its qualified name."""
    target = sys.modules[cls.__module__]
    for part in cls.__qualname__.split("."):
        target = getattr(target, part, None)
    return target is cls


def _library_classes() -> set[type]:
    """Every concrete distribution class the library defines by name."""
    for module in pkgutil.walk_packages(probpipe.__path__, "probpipe."):
        try:
            importlib.import_module(module.name)
        except ImportError as exc:
            # A module whose optional backend is absent defines nothing to check.
            if (exc.name or "").partition(".")[0] == "probpipe":
                raise
    found: set[type] = set()
    pending = [Distribution]
    while pending:
        for cls in pending.pop().__subclasses__():
            if cls in found:
                continue
            found.add(cls)
            pending.append(cls)
    return {
        cls
        for cls in found
        if cls.__module__.startswith("probpipe.")
        # A class a factory creates at runtime is not reachable by its name.
        and _reachable(cls)
        and not inspect.isabstract(cls)
        and cls not in _BASES
    }


# ``sample`` stacks a tuple draw as rows instead of wrapping it as one opaque
# value.
_DRAW_FAILURES = {
    SimpleGenerativeModel: pytest.mark.xfail(
        raises=ValueError, strict=True, reason="sample stacks a tuple draw as rows"
    ),
    FactoredDistribution: pytest.mark.pending(reason="a joint samples through its factors"),
}

# Laws that do not pickle, by the exception each raises.
_TFP_BACKEND = pytest.mark.xfail(
    raises=pytest.RaisesExc(TypeError, match="missing a required argument"),
    strict=True,
    reason="a TFP backend built from another law does not unpickle",
)
_RUNTIME_CLASS = pytest.mark.xfail(
    raises=pickle.PicklingError,
    strict=True,
    reason="a class made at runtime does not pickle (#417)",
)
_BACKEND_MODEL = pytest.mark.xfail(
    raises=AttributeError, strict=True, reason="the backend's model object does not pickle"
)
_PICKLE_FAILURES = {
    PyMCModel: _BACKEND_MODEL,
    StanModel: _BACKEND_MODEL,
    _UnconstrainedStanView: _BACKEND_MODEL,
    MultivariateNormal: _TFP_BACKEND,
    KDEDistribution: _TFP_BACKEND,
    JointGaussian: _TFP_BACKEND,
    MinibatchedDistribution: _TFP_BACKEND,
    _FixedMinibatchDistribution: _TFP_BACKEND,
    _RandomMinibatchLogProb: _TFP_BACKEND,
    _MinibatchLogProbAtPoint: _TFP_BACKEND,
    LinearBasisFunction: _TFP_BACKEND,
    _LinearMapGRF: _TFP_BACKEND,
    _ShiftedGRF: _TFP_BACKEND,
    _ScaledGRF: _TFP_BACKEND,
    _IndependentSumGRF: _TFP_BACKEND,
    TransformedDistribution: _RUNTIME_CLASS,
    SequentialJointDistribution: _RUNTIME_CLASS,
    _MixtureMarginal: _RUNTIME_CLASS,
    FlattenedDistributionView: _RUNTIME_CLASS,
    NumericRecordDistributionView: _RUNTIME_CLASS,
    _RecordDistributionView: _RUNTIME_CLASS,
}


# The interim ``event_shape`` overrides: an empirical law over an array still
# draws a one-field record, and the Stan and PyMC models count flat parameters.
_EVENT_SHAPE_OVERRIDES = {
    "RecordEmpiricalDistribution",
    "RecordBootstrapReplicateDistribution",
    "PyMCModel",
    "StanModel",
    "_UnconstrainedStanView",
}


# -- Tests --------------------------------------------------------------------


class TestCoverage:
    def test_every_concrete_class_has_a_construction(self):
        missing = {cls.__qualname__ for cls in _library_classes() - _CONSTRUCTIONS.keys()}
        assert not missing, f"no construction declares an event for {sorted(missing)}"

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_each_construction_builds_its_class(self, cls, make):
        # A class made at runtime counts as the class it specializes.
        assert next(c for c in type(make()).__mro__ if c in _CONSTRUCTIONS) is cls

    def test_the_declaration_is_the_one_schema_source(self):
        _library_classes()  # imports every module, so every class is loaded
        classes: set[type] = set()
        pending = [Distribution]
        while pending:
            for cls in pending.pop().__subclasses__():
                if cls not in classes:
                    classes.add(cls)
                    pending.append(cls)
        for cls in classes:
            if not cls.__module__.startswith("probpipe."):
                continue
            defined = vars(cls)
            assert "event_template" not in defined, cls
            if cls is not NumericDistribution:
                assert not {"dtypes", "supports", "dtype", "support"} & defined.keys(), cls
            if cls.__name__ not in _EVENT_SHAPE_OVERRIDES:
                assert "event_shape" not in defined, cls


class TestDeclaration:
    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_a_rename_keeps_the_declaration(self, cls, make):
        law = make()
        renamed = law.with_name("renamed")
        assert renamed.name == "renamed"
        assert renamed.event_spec == law.event_spec

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_numeric_membership_follows_the_declaration(self, cls, make):
        law = make()
        assert isinstance(law, NumericDistribution) is isinstance(law.event_spec.spec, NumericSpec)

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_a_law_has_the_numeric_views_when_it_is_numeric(self, cls, make):
        law = make()
        numeric = isinstance(law, NumericDistribution)
        for view in ("dtypes", "supports", "dtype", "support"):
            assert hasattr(law, view) is numeric

    @pytest.mark.parametrize(("cls", "make"), _rows(_DRAW_FAILURES))
    def test_the_declaration_admits_the_draw(self, cls, make):
        law = make()
        if not isinstance(law, SupportsSampling):
            pytest.skip("the law does not sample")
        assert law.event_spec.spec.is_valid(sample(law, key=jax.random.PRNGKey(0)))


class TestRoundTrips:
    """A round trip of a renamed law keeps its name and its declaration."""

    @pytest.mark.parametrize(("cls", "make"), _rows(_PICKLE_FAILURES))
    def test_pickle(self, cls, make):
        law = make().with_name("renamed")
        restored = pickle.loads(pickle.dumps(law))
        assert (restored.name, restored.spec) == (law.name, law.spec)

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_copy(self, cls, make):
        law = make().with_name("renamed")
        restored = copy.copy(law)
        assert (restored.name, restored.spec) == (law.name, law.spec)

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_pytree(self, cls, make):
        law = make().with_name("renamed")
        leaves, treedef = jax.tree_util.tree_flatten(law)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert (restored.name, restored.spec) == (law.name, law.spec)
