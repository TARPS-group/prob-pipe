"""Contracts of the parametric families (VII.1) and their backend adapter."""

from __future__ import annotations

import asyncio
import copy
import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe.families as F
from probpipe import (
    Distribution,
    DistributionBatch,
    MathematicalDomainError,
    NumericArraySpec,
    NumericDistribution,
    OutputSpec,
    real,
)
from probpipe.core._compilation import _eager_decision_keys
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)
from probpipe.families import TFPDistribution, _backend
from probpipe.families._backend import _allow_batched_tfp_init, _eager_sample
from probpipe.linalg import DiagonalLinOp, LinOp
from tests._compilation import RecordingProgram, count_compilations


def _families() -> dict[str, Distribution]:
    return {
        "Normal": F.Normal("a", 0.0, 1.0),
        "Beta": F.Beta("a", 2.0, 3.0),
        "Gamma": F.Gamma("a", 2.0, 1.0),
        "InverseGamma": F.InverseGamma("a", 3.0, 1.0),
        "Exponential": F.Exponential("a", 1.0),
        "LogNormal": F.LogNormal("a", 0.0, 1.0),
        "StudentT": F.StudentT("a", 3.0, 0.0, 1.0),
        "Uniform": F.Uniform("a", 0.0, 1.0),
        "Cauchy": F.Cauchy("a", 0.0, 1.0),
        "Laplace": F.Laplace("a", 0.0, 1.0),
        "HalfNormal": F.HalfNormal("a", 1.0),
        "HalfCauchy": F.HalfCauchy("a", 0.0, 1.0),
        "Pareto": F.Pareto("a", 3.0, 1.0),
        "TruncatedNormal": F.TruncatedNormal("a", 0.0, 1.0, -1.0, 1.0),
        "Bernoulli": F.Bernoulli("a", probs=0.3),
        "Binomial": F.Binomial("a", 5.0, probs=0.3),
        "Poisson": F.Poisson("a", 2.0),
        "Categorical": F.Categorical("a", probs=jnp.array([0.2, 0.8])),
        "NegativeBinomial": F.NegativeBinomial("a", 5.0, probs=0.3),
        "MultivariateNormal": F.MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2)),
        "Dirichlet": F.Dirichlet("a", jnp.ones(3)),
        "Multinomial": F.Multinomial("a", 5, probs=jnp.array([0.2, 0.8])),
        "Wishart": F.Wishart("a", 3.0, scale_tril=jnp.eye(2)),
        "VonMisesFisher": F.VonMisesFisher("a", jnp.array([0.0, 1.0]), 2.0),
    }


_NAMES = list(_families())

_MOMENTS = {SupportsMean, SupportsVariance, SupportsCovariance}

#: The capabilities each family's backend computes beyond sampling and the density.
_EXPECTED = {
    **{
        name: _MOMENTS | {SupportsQuantile}
        for name in (
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
            "TruncatedNormal",
            "MultivariateNormal",
        )
    },
    **{
        name: _MOMENTS
        for name in (
            "Pareto",
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Categorical",
            "NegativeBinomial",
            "Dirichlet",
            "Multinomial",
        )
    },
    "Wishart": {SupportsMean, SupportsVariance},
    "VonMisesFisher": {SupportsMean, SupportsVariance, SupportsCovariance},
}

#: The families whose mean, variance, and covariance do not exist at any parameters.
#: The families whose moments do not all converge: each moment that is undefined and
#: raises, and each that diverges and is ``inf``.
_UNDEFINED_MOMENTS = {
    ("Cauchy", "_mean"),
    ("Cauchy", "_variance"),
    ("Cauchy", "_cov"),
    ("HalfCauchy", "_variance"),
    ("HalfCauchy", "_cov"),
}
_INFINITE_MOMENTS = {("HalfCauchy", "_mean")}


def _dense_or_array(moment):
    return moment.to_dense() if isinstance(moment, LinOp) else moment


_PROTOCOLS = {
    "_mean": SupportsMean,
    "_variance": SupportsVariance,
    "_cov": SupportsCovariance,
    "_quantile": SupportsQuantile,
}


class TestTheEventDeclaration:
    def test_event_spec_names_the_output_and_the_label_stays(self):
        prior = F.Normal("beta", 0.0, 1.0, label="prior")
        assert prior.label == "prior"
        assert list(prior.event_spec.outputs) == ["beta"]
        assert prior.event_spec.term_spec.shape == ()

    def test_the_output_defaults_to_the_label_captured_at_construction(self):
        law = F.Normal("x", 0.0, 1.0)
        assert list(law.event_spec.outputs) == ["x"]
        assert list(law.with_label("y").event_spec.outputs) == ["x"]

    def test_the_event_spec_is_derived_from_the_parameters(self):
        law = F.MultivariateNormal("z", jnp.zeros(3), cov=jnp.eye(3))
        spec = law.event_spec.term_spec
        assert spec.shape == (3,)
        assert spec.dtype == jnp.zeros(3).dtype
        assert spec.support is not None

    def test_a_declared_type_must_agree_with_the_parameters(self):
        with pytest.raises(ValueError):
            F.Normal("x", 0.0, 1.0, event_spec=OutputSpec(x=NumericArraySpec((3,))))

    @pytest.mark.parametrize("name", _NAMES)
    def test_each_family_is_a_numeric_distribution(self, name):
        assert isinstance(_families()[name], NumericDistribution)


class TestTheAdapter:
    def test_the_adapter_wraps_a_backend_distribution(self):
        law = TFPDistribution("x", tfd.Normal(0.0, 1.0))
        assert law.event_spec.term_spec.shape == ()
        assert float(law._log_prob(0.0)) == pytest.approx(-0.9189385, rel=1e-6)
        assert not isinstance(law, SupportsMean)

    @pytest.mark.parametrize("name", _NAMES)
    def test_one_draw_matches_the_declaration(self, name):
        law = _families()[name]
        draw = law._sample(jax.random.PRNGKey(0))
        assert tuple(jnp.shape(draw)) == law.event_spec.term_spec.shape

    @pytest.mark.parametrize("name", _NAMES)
    def test_each_family_claims_exactly_what_its_backend_computes(self, name):
        law = _families()[name]
        claimed = {protocol for protocol in _PROTOCOLS.values() if isinstance(law, protocol)}
        assert claimed == _EXPECTED[name]
        assert type(law)._backend_capabilities == _EXPECTED[name]

    @pytest.mark.parametrize("name", _NAMES)
    @pytest.mark.parametrize("method", ["_mean", "_variance", "_cov"])
    def test_each_claimed_moment_computes(self, name, method):
        law = _families()[name]
        if not isinstance(law, _PROTOCOLS[method]):
            pytest.skip(f"{name} does not claim {method}")
        if (name, method) in _UNDEFINED_MOMENTS:
            # The claim answers that the moment is undefined (II.7).
            with pytest.raises(MathematicalDomainError):
                getattr(law, method)()
            return
        value = np.asarray(_dense_or_array(getattr(law, method)()))
        if (name, method) in _INFINITE_MOMENTS:
            assert np.all(np.isposinf(value))
            return
        assert np.all(np.isfinite(value))

    @pytest.mark.parametrize("name", [n for n in _NAMES if SupportsQuantile in _EXPECTED[n]])
    def test_the_quantiles_put_the_levels_first(self, name):
        law = _families()[name]
        quantiles = law._quantile(jnp.array([0.25, 0.5, 0.75]))
        assert quantiles.shape == (3, *law.event_spec.term_spec.shape)
        assert bool(jnp.all(jnp.diff(quantiles, axis=0) >= 0))

    def test_raw_is_the_backend_distribution(self):
        assert isinstance(F.Normal("x", 0.0, 1.0).raw, tfd.Normal)

    def test_the_adapter_claims_the_quantile(self):
        law = F.Normal("x", 0.0, 1.0)
        assert isinstance(law, SupportsQuantile)
        assert float(law._quantile(0.5)) == pytest.approx(0.0, abs=1e-6)

    def test_the_multivariate_normal_quantiles_are_those_of_its_marginals(self):
        loc = jnp.array([1.0, -2.0])
        cov = jnp.array([[4.0, 0.6], [0.6, 0.25]])
        law = F.MultivariateNormal("z", loc, cov=cov)
        levels = jnp.array([0.1, 0.9])
        expected = jnp.stack(
            [F.Normal("m", loc[i], jnp.sqrt(cov[i, i]))._quantile(levels) for i in range(2)],
            axis=-1,
        )
        np.testing.assert_allclose(law._quantile(levels), expected, rtol=1e-5)

    def test_the_covariance_is_the_operator_of_the_given_matrix(self):
        cov = jnp.array([[4.0, 0.6], [0.6, 0.25]])
        law = F.MultivariateNormal("z", jnp.zeros(2), cov=cov)
        assert isinstance(law._cov(), LinOp)
        np.testing.assert_allclose(law._cov().to_dense(), cov, rtol=1e-6)

    def test_the_covariance_from_a_scale_factor_is_its_outer_product(self):
        scale_tril = jnp.array([[2.0, 0.0], [0.3, 0.4]])
        law = F.MultivariateNormal("z", jnp.zeros(2), scale_tril)
        np.testing.assert_allclose(law._cov().to_dense(), scale_tril @ scale_tril.T, rtol=1e-6)

    def test_a_backend_covariance_is_the_backends_dense_matrix(self):
        law = F.Dirichlet("p", jnp.array([2.0, 3.0, 5.0]))
        expected = tfd.Dirichlet(jnp.array([2.0, 3.0, 5.0])).covariance()
        assert np.abs(np.asarray(expected) - np.diag(np.diag(expected))).max() > 0.01
        np.testing.assert_allclose(law._cov().to_dense(), expected, rtol=1e-6)

    def test_a_given_covariance_operator_keeps_its_structure(self):
        operator = DiagonalLinOp(jnp.array([1.0, 4.0]))
        law = F.MultivariateNormal("z", jnp.zeros(2), cov=operator)
        assert law._cov() is operator
        np.testing.assert_allclose(law.cov, jnp.diag(jnp.array([1.0, 4.0])), rtol=1e-6)

    def test_an_infinite_support_family_claims_no_expectation(self):
        assert not isinstance(F.Normal("x", 0.0, 1.0), SupportsExpectation)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: F.Bernoulli("b", probs=0.3),
            lambda: F.Binomial("b", 4.0, probs=0.3),
            lambda: F.Categorical("b", probs=jnp.array([0.2, 0.3, 0.5])),
        ],
    )
    def test_a_finite_support_law_over_one_coordinate_has_the_exact_expectation(self, make):
        law = make()
        assert isinstance(law, SupportsExpectation)
        assert float(law._expectation(lambda x: x)) == pytest.approx(float(law._mean()), rel=1e-5)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: F.Bernoulli("b", probs=jnp.array([0.3, 0.6])),
            lambda: F.Binomial("b", jnp.array([4.0, 5.0]), probs=0.3),
            lambda: F.Categorical("b", probs=jnp.array([[0.2, 0.8], [0.5, 0.5]])),
        ],
    )
    def test_a_law_over_several_coordinates_claims_no_exact_expectation(self, make):
        assert not isinstance(make(), SupportsExpectation)


class TestVectorParameters:
    """A scalar family given parameters with axes draws one array of independent coordinates."""

    def test_vector_parameters_draw_a_vector(self):
        law = F.Normal("x", jnp.arange(5.0), 1.0)
        assert law.event_shape == (5,)
        assert law._sample(jax.random.PRNGKey(0)).shape == (5,)
        assert law._sample(jax.random.PRNGKey(0), (3,)).shape == (3, 5)

    def test_the_density_is_the_product_over_the_coordinates(self):
        law = F.Normal("x", jnp.arange(3.0), 2.0)
        value = jnp.array([0.5, 1.5, -1.0])
        expected = sum(float(F.Normal("c", float(i), 2.0)._log_prob(value[i])) for i in range(3))
        assert float(law._log_prob(value)) == pytest.approx(expected, rel=1e-6)
        assert law._log_prob(jnp.zeros((4, 3))).shape == (4,)

    def test_the_coordinates_are_uncorrelated(self):
        law = F.Gamma("g", jnp.array([2.0, 3.0]), 1.0)
        covariance = law._cov()
        assert isinstance(covariance, DiagonalLinOp)
        np.testing.assert_allclose(covariance.to_dense(), jnp.diag(law._variance()), rtol=1e-6)

    def test_parameters_broadcast_to_the_event_shape(self):
        law = F.Beta("b", jnp.ones((2, 3)), 1.0)
        assert law.event_shape == (2, 3)
        assert law._mean().shape == (2, 3)
        assert law._quantile(jnp.array([0.5, 0.9])).shape == (2, 2, 3)

    def test_a_multivariate_family_given_parameters_for_rows_draws_independent_rows(self):
        d = 3
        law = F.MultivariateNormal(
            "z", jnp.zeros((2, d)), scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d))
        )
        assert law.event_shape == (2, d)
        assert law._sample(jax.random.PRNGKey(0), (4,)).shape == (4, 2, d)
        assert law._log_prob(jnp.zeros((2, d))).shape == ()

    def test_scalar_parameters_draw_a_scalar(self):
        law = F.Normal("x", 0.0, 1.0)
        assert law.event_shape == ()
        assert not hasattr(law, "batch_shape")

    def test_a_batch_of_separate_laws_is_a_distribution_batch(self):
        batch = DistributionBatch(
            [F.Normal("x", float(i), 1.0) for i in range(5)],
            "x",
            label="x",
        )
        assert batch.batch_shape == (5,)
        assert batch[0].label == "x[x=0]"
        assert batch[0].event_shape == ()


class TestTheSeparateLawsForm:
    """Inside ``_allow_batched_tfp_init`` a family keeps its backend's batch axes as separate laws."""

    def test_the_batch_axes_are_separate_laws_inside_the_block(self):
        with _allow_batched_tfp_init():
            law = F.Normal("x", jnp.zeros(5), 1.0)
        assert law.event_shape == ()
        assert tuple(law.raw.batch_shape) == (5,)
        assert law._log_prob(jnp.zeros(5)).shape == (5,)

    def test_the_form_is_scoped_to_the_block(self):
        with _allow_batched_tfp_init():
            F.Normal("x", jnp.zeros(3), 1.0)
        assert F.Normal("x", jnp.zeros(3), 1.0).event_shape == (3,)

    def test_a_nested_block_restores_the_outer_form(self):
        with _allow_batched_tfp_init():
            with _allow_batched_tfp_init():
                F.Normal("x", jnp.zeros(2), 1.0)
            assert F.Normal("x", jnp.zeros(2), 1.0).event_shape == ()
        assert F.Normal("x", jnp.zeros(2), 1.0).event_shape == (2,)

    def test_the_form_does_not_leak_across_asyncio_tasks(self):
        inside: list[tuple[int, ...]] = []
        outside: list[tuple[int, ...]] = []

        async def within_the_block():
            with _allow_batched_tfp_init():
                # Yield so the sibling task runs while this block is active.
                await asyncio.sleep(0)
                inside.append(F.Normal("x", jnp.zeros(2), 1.0).event_shape)

        async def outside_the_block():
            await asyncio.sleep(0)
            outside.append(F.Normal("y", jnp.zeros(2), 1.0).event_shape)

        async def main():
            await asyncio.gather(within_the_block(), outside_the_block())

        asyncio.run(main())
        assert inside == [()]
        assert outside == [(2,)]


class TestTheBackend:
    """The adapter takes the backend distribution it wraps at construction."""

    def test_the_adapter_takes_a_backend_distribution(self):
        with pytest.raises(TypeError, match="backend distribution"):
            TFPDistribution(
                "x", None, event_spec=OutputSpec(x=NumericArraySpec((), jnp.float32, real))
            )


#: A scalar family, two families over vector parameters, and a multivariate family.
_ROUND_TRIPS = {
    "scalar": lambda: F.Normal("x", 0.5, 2.0),
    "vector-parameters": lambda: F.Normal("x", jnp.array([0.0, 1.0]), jnp.array([1.0, 0.5])),
    "multivariate": lambda: F.MultivariateNormal(
        "x", jnp.array([1.0, -1.0]), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]])
    ),
    "discrete-vector-parameters": lambda: F.Bernoulli("x", probs=jnp.array([0.2, 0.7])),
}

_COPIES = {
    "pickle": lambda law: pickle.loads(pickle.dumps(law)),
    "deepcopy": copy.deepcopy,
}


class TestRecordedArguments:
    """A family rebuilds itself from its constructor arguments, so it keeps them as given."""

    def test_a_numpy_argument_is_read_only(self):
        loc = np.array([0.0, 1.0, 2.0])
        law = F.Normal("x", loc, np.ones(3))

        with pytest.raises(ValueError, match="read-only"):
            loc[0] = 5.0
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(law)).raw.mean(), [0.0, 1.0, 2.0])


class TestPickling:
    """A family pickles and deep-copies, rebuilt from its constructor arguments."""

    @pytest.mark.parametrize("how", list(_COPIES))
    @pytest.mark.parametrize("family", list(_ROUND_TRIPS))
    def test_a_family_round_trips(self, family, how):
        law = _ROUND_TRIPS[family]().with_label("renamed")
        copied = _COPIES[how](law)
        assert type(copied) is type(law)
        assert copied.label == "renamed"
        assert copied.event_spec == law.event_spec
        key = jax.random.PRNGKey(3)
        draws = law._sample(key, (4,))
        np.testing.assert_array_equal(copied._sample(key, (4,)), draws)
        np.testing.assert_allclose(copied._log_prob(draws), law._log_prob(draws), rtol=1e-6)

    @pytest.mark.parametrize("how", list(_COPIES))
    def test_a_family_in_the_separate_laws_form_keeps_its_form(self, how):
        with _allow_batched_tfp_init():
            law = F.Normal("x", jnp.zeros(3), 1.0)
        copied = _COPIES[how](law)
        assert copied.event_shape == ()
        assert tuple(copied.raw.batch_shape) == (3,)


class _NumpyNormal(tfd.Normal):
    """A normal backend that draws with a new NumPy generator, ignoring the key."""

    def _sample_n(self, n, seed=None):
        shape = (n, *np.shape(self.loc + self.scale))
        noise = np.random.default_rng().standard_normal(shape).astype(np.float32)
        return self.loc + self.scale * noise


def _ran_compiled(law: Distribution) -> bool:
    """Whether every call of *law* so far ran a compiled program."""
    memo = law._memo
    return memo["compilable"] and not _eager_decision_keys(memo)


#: The families whose quantiles their backend computes, one coordinate per draw.
_SCALAR_QUANTILES = [
    name
    for name, claimed in _EXPECTED.items()
    if SupportsQuantile in claimed and name != "MultivariateNormal"
]


class TestCompiledCalls:
    """The adapter samples, scores, and takes quantiles through compiled programs that later calls reuse."""

    @pytest.mark.parametrize("name", _NAMES)
    def test_the_compiled_calls_agree_with_the_backend_called_eagerly(self, name):
        law = _families()[name]
        key = jax.random.PRNGKey(0)
        draws = law._sample(key, (16,))
        # XLA fuses the arithmetic of a compiled program differently, which changes the
        # last float32 digit. Observed across 4 seeds and every family: at most 5.7e-7
        # absolute and 7e-7 relative away from zero.
        np.testing.assert_allclose(
            draws, law.raw.sample(seed=key, sample_shape=(16,)), rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            law._log_prob(draws), law.raw.log_prob(draws), rtol=2e-6, atol=2e-6
        )
        assert _ran_compiled(law)

    @pytest.mark.parametrize("name", _SCALAR_QUANTILES)
    def test_the_compiled_quantiles_agree_with_the_backend_called_eagerly(self, name):
        law = _families()[name]
        q = jnp.array([0.05, 0.5, 0.95])
        # The tolerance of the draws above, which the quantiles meet.
        np.testing.assert_allclose(law._quantile(q), law.raw.quantile(q), rtol=2e-6, atol=2e-6)
        assert _ran_compiled(law)

    @pytest.mark.parametrize("family", [F.Beta, F.Gamma])
    def test_a_warm_draw_and_quantile_compile_nothing(self, family):
        first, second = family("a", 2.0, 3.0), family("a", 4.0, 1.5)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        q = jnp.array([0.1, 0.9])
        first._sample(keys[0], (10,))
        first._quantile(q)
        with count_compilations() as compiled:
            second._sample(keys[1], (10,))
            second._quantile(q)
        assert compiled.count == 0

    @pytest.mark.parametrize("family", [F.Poisson, F.Binomial])
    def test_a_warm_draw_of_a_discrete_family_compiles_nothing(self, family):
        arguments = {F.Poisson: {"rate": 2.0}, F.Binomial: {"total_count": 5.0, "probs": 0.3}}
        law = family("a", **arguments[family])
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        law._sample(keys[0], (10,))
        with count_compilations() as compiled:
            law._sample(keys[1], (10,))
        assert compiled.count == 0

    @pytest.mark.parametrize(
        ("program", "eager", "call"),
        [
            (
                "_sample_program",
                "_eager_sample",
                lambda law: law._sample(jax.random.PRNGKey(1), (4,)),
            ),
            (
                "_log_prob_program",
                "_eager_log_prob",
                lambda law: law._log_prob(jnp.array([0.5, 1.5])),
            ),
            (
                "_quantile_program",
                "_eager_quantile",
                lambda law: law._quantile(jnp.array([0.25, 0.75])),
            ),
        ],
    )
    def test_a_call_runs_the_compiled_program(self, monkeypatch, program, eager, call):
        compiled, uncompiled = [], []
        monkeypatch.setattr(
            _backend, program, RecordingProgram(getattr(_backend, program), runs=compiled)
        )
        original_eager = getattr(_backend, eager)
        monkeypatch.setattr(
            _backend,
            eager,
            lambda *args: uncompiled.append(args) or original_eager(*args),
        )
        law = F.Gamma("a", 2.0, 3.0)
        call(law)
        call(law)
        assert len(compiled) == 2
        assert uncompiled == []

    @pytest.mark.parametrize(
        "backend",
        [
            _NumpyNormal(0.0, 1.0),
            tfd.Independent(_NumpyNormal(jnp.zeros(2), 1.0), 1),
            tfd.Blockwise(
                tfd.JointDistributionSequential(
                    [tfd.Normal(0.0, 1.0), lambda a: _NumpyNormal(a, 1.0)]
                )
            ),
        ],
        ids=["subclass", "subclass-inside-independent", "subclass-from-a-model-function"],
    )
    def test_a_backend_built_from_a_class_a_user_writes_runs_eagerly(self, backend):
        law = TFPDistribution("x", backend)
        first = law._sample(jax.random.PRNGKey(0), (4,))
        second = law._sample(jax.random.PRNGKey(0), (4,))
        # A compiled program would fix the NumPy draws of its trace.
        assert not np.array_equal(first, second)

    @pytest.mark.parametrize(
        "backend",
        [
            tfd.Beta(2.0, 3.0, validate_args=True),
            tfd.Beta(jnp.full(2, 2.0), 3.0, validate_args=True),
        ],
        ids=["scalar", "independent-coordinates"],
    )
    def test_a_backend_that_validates_its_arguments_runs_eagerly(self, backend):
        law = TFPDistribution("x", backend)
        with pytest.raises(ValueError, match="did not hold"):
            law._log_prob(jnp.full(law.event_shape, 1.5))

    def test_a_program_that_does_not_trace_falls_back_once_to_the_eager_call(self, monkeypatch):
        attempts = []

        @functools.partial(jax.jit, static_argnums=2)
        def masked(backend, key, sample_shape):
            draws = backend.sample(seed=key, sample_shape=sample_shape)
            return draws[draws > -jnp.inf]

        runs: list = []
        monkeypatch.setattr(
            _backend, "_sample_program", RecordingProgram(masked, runs=runs, traces=attempts)
        )
        law = F.Normal("x", 0.0, 1.0)
        key = jax.random.PRNGKey(2)
        draws = law._sample(key, (4,))
        law._sample(key, (4,))
        # The boolean mask needs concrete values, so the first call's trace raises
        # NonConcreteBooleanIndexError, an IndexError, and the second call skips it.
        assert len(attempts) == 1
        assert runs == []
        np.testing.assert_array_equal(draws, law.raw.sample(seed=key, sample_shape=(4,)))

    def test_an_error_the_eager_call_raises_too_propagates_and_keeps_the_program(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            _backend, "_eager_sample", lambda *args: calls.append(args) or _eager_sample(*args)
        )
        law = F.Gamma("x", 2.0, 3.0)
        with pytest.raises(TypeError, match="invalid PRNG key"):
            law._sample(jnp.zeros(3), (4,))
        calls.clear()
        law._sample(jax.random.PRNGKey(1), (4,))
        assert calls == []

    def test_a_call_inside_a_transformation_the_caller_opens_traces_with_it(self):
        law = F.Gamma("x", 2.0, 3.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        draws = jax.jit(jax.vmap(lambda key: law._sample(key, (5,))))(keys)
        assert draws.shape == (3, 5)
        # The log-density of Gamma(a, b) is (a - 1) log x - b x + const, so its
        # derivative in x is (a - 1) / x - b, which is -2 at x = 1.
        slope = jax.grad(lambda x: law._log_prob(x))(1.0)
        np.testing.assert_allclose(slope, -2.0, rtol=1e-6)
