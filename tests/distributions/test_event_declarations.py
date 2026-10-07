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
    EmpiricalDistribution,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
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
    NumericRecordBatch,
    NumericRecordSpec,
    NumericSpec,
    OpaqueBatch,
    OutputSpec,
    Pareto,
    Poisson,
    Record,
    StudentT,
    TFPDistribution,
    TruncatedNormal,
    Uniform,
    VonMisesFisher,
    Wishart,
    distribution,
    sample,
)
from probpipe.distributions import (
    FactoredDistribution,
    FactoredNumericDistribution,
    FieldView,
)
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.distributions._factored import _SoleField
from probpipe.distributions._from_functions import _FunctionLaw
from probpipe.distributions._views import _RenamedDistribution
from probpipe.families import (
    BernoulliFamily,
    BijectorTransformedDistribution,
    FactoredMultivariateGaussian,
    GaussianProcess,
    LinearPushforwardDistribution,
    MixtureDistribution,
    PoissonFamily,
    RandomMeasure,
    glm_likelihood,
)
from probpipe.families._conditional import _LogRatePoisson
from probpipe.families._gaussian import (
    _IndependentSumGRF,
    _LinearMapGRF,
    _ScaledGRF,
    _ShiftedGRF,
)
from probpipe.families._programs import (
    PyMCModel,
    StanModel,
    _StanPosterior,
    _UnconstrainedStanView,
)
from probpipe.inference._bayesflow_likelihoods import (
    BayesFlowLikelihood,
    BayesFlowRatio,
    _LearnedDensity,
    _LearnedLaw,
    _LearnedRatioLaw,
)
from probpipe.inference._bayesflow_posteriors import _AmortizedPosterior, _AmortizedPosteriorLaw
from probpipe.inference._minibatch import (
    _FixedMinibatchDistribution,
    _MinibatchLogProbAtPoint,
    _RandomMinibatchLogProb,
)
from probpipe.linalg import DenseLinOp
from probpipe.operations._condition import _unnormalized_conditional, _UnnormalizedConditional
from tests._stanc import require_stanc

# -- Constructions ------------------------------------------------------------


# The callables a construction passes are module-level, so they pickle.
def _features(X, output_shape=()):
    phi = jnp.concatenate([X, X**2], -1)
    return jnp.stack([phi] * output_shape[0], -2) if output_shape else phi


def _basis_function(name: str = "f", output_shape: tuple[int, ...] = ()) -> LinearBasisFunction:
    width = 2 * max(1, int(np.prod(output_shape)))
    weights = MultivariateNormal("w", loc=jnp.zeros(width), cov=jnp.eye(width))
    return LinearBasisFunction(
        name, functools.partial(_features, output_shape=output_shape), weights
    )


def _measure() -> MinibatchedDistribution:
    X = jax.random.normal(jax.random.PRNGKey(0), (20, 2))
    y = (X[:, 0] > 0).astype(jnp.float32)
    prior = MultivariateNormal("beta", loc=jnp.zeros(2), cov=jnp.eye(2))
    likelihood = glm_likelihood("y", BernoulliFamily(), X=X)
    return MinibatchedDistribution("measure", prior, likelihood, y, batch_size=5)


class _ZeroNetwork:
    """A stand-in for a trained posterior network, whose draws are zeros."""

    def sample(self, *, num_samples, conditions, seed):
        return {"theta_0": np.zeros((1, num_samples, 1))}


class _Simulator:
    """A stand-in for the simulator a learned kernel stores and never calls here."""


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


def _stan_model() -> _StanPosterior:
    require_stanc()
    stan_file = pathlib.Path(tempfile.mkdtemp()) / "declared.stan"
    stan_file.write_text("parameters { real mu; } model { mu ~ normal(0, 1); }")
    return StanModel("model", str(stan_file))


def _stan_view() -> _UnconstrainedStanView:
    pytest.importorskip("bridgestan")
    return _stan_model().as_unconstrained_distribution()


def _standard_normal_density(x):
    return -0.5 * jnp.sum(jnp.asarray(x) ** 2)


def _standard_normal_draw(key):
    return jax.random.normal(key, (2,))


def _zero_mean(X):
    return jnp.zeros(X.shape[0])


def _squared_exponential(X, Y):
    return jnp.exp(-0.5 * (X[:, None, 0] - Y[None, :, 0]) ** 2)


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
    KDEDistribution: lambda: KDEDistribution("kde", jnp.arange(6.0).reshape(3, 2)),
    EmpiricalDistribution: lambda: EmpiricalDistribution(
        "e", OpaqueBatch("labels", ["a", "b"], "e")
    ),
    BootstrapReplicateDistribution: lambda: BootstrapReplicateDistribution(
        "b", Normal("x", 0.0, 1.0), replicate_size=3
    ),
    BootstrapDistribution: lambda: BootstrapDistribution("measure", Normal("x", 0.0, 1.0), 3),
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
    _AmortizedPosteriorLaw: lambda: _AmortizedPosterior(
        _ZeroNetwork(), Normal("theta", 0.0, 1.0), _Simulator(), method="npe", data_dim=2
    )._condition_on({"observation": jnp.zeros(2)}),
    _LearnedDensity: lambda: BayesFlowLikelihood(
        None, Normal("theta", 0.0, 1.0), _Simulator(), data_dim=2
    )._condition_on({"theta": 0.0}),
    _LearnedRatioLaw: lambda: BayesFlowRatio(
        None, Normal("theta", 0.0, 1.0), _Simulator(), data_dim=2
    )._condition_on({"theta": 0.0}),
    PyMCModel: _pymc_model,
    _StanPosterior: _stan_model,
    _UnconstrainedStanView: _stan_view,
    _FunctionLaw: lambda: distribution(
        "u",
        sample=_standard_normal_draw,
        log_prob=_standard_normal_density,
        event_spec=OutputSpec(x=probpipe.NumericArraySpec((2,))),
    ),
    FieldView: lambda: FieldView(Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0), "a"),
    FactoredDistribution: lambda: Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0),
    _UnnormalizedConditional: lambda: _unnormalized_conditional(
        Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0), Record("given", {"a": 0.0})
    ),
    _LogRatePoisson: lambda: PoissonFamily()._build_canonical("y", jnp.zeros(3)),
    MixtureDistribution: lambda: MixtureDistribution(
        "m",
        [
            Normal("a", 0.0, 1.0, event_spec=OutputSpec(x=None)),
            Normal("b", 1.0, 1.0, event_spec=OutputSpec(x=None)),
        ],
        jnp.array([0.5, 0.5]),
    ),
    LinearPushforwardDistribution: lambda: LinearPushforwardDistribution(
        "y", MultivariateNormal("x", jnp.zeros(2), cov=jnp.eye(2)), DenseLinOp(jnp.eye(2))
    ),
    BijectorTransformedDistribution: lambda: BijectorTransformedDistribution(
        "y", Normal("x", 0.0, 1.0), tfb.Exp()
    ),
    FactoredMultivariateGaussian: lambda: Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0),
    GaussianProcess: lambda: GaussianProcess("f", _zero_mean, _squared_exponential),
    _SoleField: lambda: _SoleField(FactoredDistribution("record", [Normal("beta", 0.0, 1.0)])),
    _RenamedDistribution: lambda: EmpiricalDistribution(
        "e",
        NumericRecordBatch(
            "rows",
            {"a": jnp.zeros(2), "b": jnp.ones(2)},
            "row",
            element_spec=NumericRecordSpec(a=(), b=()),
        ),
    ).with_path_names({"a": "g/a"}),
}

# The catalog's families whose implementation has not merged construct by raising.
_STUB_CONSTRUCTIONS = {
    cls: pytest.mark.pending(reason=f"{cls.__name__} constructs")
    for cls in (LinearPushforwardDistribution,)
}

# Bases a concrete class specializes, constructed only through one.
#: The constructions that read a Stan program, so they need BridgeStan.
_STAN_CONSTRUCTIONS = frozenset({_StanPosterior, _UnconstrainedStanView})

_BASES = frozenset(
    {
        NumericDistribution,
        FactoredNumericDistribution,
        TFPDistribution,
        _LearnedLaw,
    }
)


def _rows(failures: dict[type, pytest.MarkDecorator] | None = None) -> list:
    """One case per construction, marked where the check fails for a known reason."""
    failures = {**_STUB_CONSTRUCTIONS, **(failures or {})}
    return [
        pytest.param(cls, make, id=cls.__name__, marks=_marks(cls, failures))
        for cls, make in _CONSTRUCTIONS.items()
    ]


def _marks(cls: type, failures: dict[type, pytest.MarkDecorator]) -> tuple:
    """The row's known-failure mark, and ``stan`` when its construction needs BridgeStan."""
    known = failures.get(cls, ())
    known = known if isinstance(known, tuple) else (known,)
    return (*known, pytest.mark.stan) if cls in _STAN_CONSTRUCTIONS else known


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


# Laws that do not pickle, by the exception each raises.
_RUNTIME_CLASS = pytest.mark.xfail(
    raises=pickle.PicklingError,
    strict=True,
    reason="a class made at runtime does not pickle (#417)",
)
_PICKLE_FAILURES = {}


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
            assert "event_shape" not in defined, cls


class TestDeclaration:
    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_a_rename_keeps_the_declaration(self, cls, make):
        law = make()
        renamed = law.with_label("renamed")
        assert renamed.label == "renamed"
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

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_the_declaration_admits_the_draw(self, cls, make):
        law = make()
        if not isinstance(law, SupportsSampling):
            pytest.skip("the law does not sample")
        assert law.event_spec.spec.is_valid(sample(law))


class TestRoundTrips:
    """A round trip of a renamed law keeps its name and its declaration."""

    @pytest.mark.parametrize(("cls", "make"), _rows(_PICKLE_FAILURES))
    def test_pickle(self, cls, make):
        law = make().with_label("renamed")
        restored = pickle.loads(pickle.dumps(law))
        assert (restored.label, restored.spec) == (law.label, law.spec)

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_copy(self, cls, make):
        law = make().with_label("renamed")
        restored = copy.copy(law)
        assert (restored.label, restored.spec) == (law.label, law.spec)

    @pytest.mark.parametrize(("cls", "make"), _rows())
    def test_pytree(self, cls, make):
        law = make().with_label("renamed")
        leaves, treedef = jax.tree_util.tree_flatten(law)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert (restored.label, restored.spec) == (law.label, law.spec)
