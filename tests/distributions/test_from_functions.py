"""A law built from a sampling function, a log-density, or both (IV.4).

``distribution`` builds a ``Distribution`` that claims the capability of each
function it is given. Construction draws nothing and scores no value: it
evaluates each function that traces in JAX abstractly against the event
declaration, and it accepts a function that does not trace and an event of any
kind.
"""

from __future__ import annotations

import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import NumericArraySpec, OpaqueBatch, OpaqueSpec, OutputSpec, Record, RecordSpec
from probpipe.core.constraints import non_negative_integer, positive, real
from probpipe.distributions import (
    Distribution,
    NumericDistribution,
    conditional_distribution,
    distribution,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _is_normalized,
)
from tests._ops import condition_on, log_prob, sample, unnormalized_log_prob

REAL = NumericArraySpec((), jnp.float32, real)
POSITIVE = NumericArraySpec((), jnp.float32, positive)
VECTOR = NumericArraySpec((3,), jnp.float32)
COUNTS = NumericArraySpec((4,), jnp.int32, non_negative_integer)


def _normal_draw(key: Any) -> Any:
    return jax.random.normal(key, (3,))


def _normal_density(value: Any) -> Any:
    return -0.5 * jnp.sum(value**2) - 1.5 * jnp.log(2.0 * jnp.pi)


def _numpy_seed(key: Any) -> np.ndarray:
    """The seed of a NumPy generator, read from a typed or a raw PRNG key."""
    typed = jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    return np.asarray(jax.random.key_data(key) if typed else key)


def _numpy_draw(key: Any) -> np.ndarray:
    """A draw that does not trace: NumPy's generator, seeded from the key."""
    return np.random.default_rng(_numpy_seed(key)).normal(size=3).astype(np.float32)


def _law(**functions: Any) -> Distribution:
    return distribution("x", event_spec=VECTOR, **functions)


class TestTheClaims:
    @pytest.mark.parametrize(
        ("functions", "claimed", "normalized"),
        [
            ({"sample": _normal_draw}, {SupportsSampling}, True),
            (
                {"log_prob": _normal_density},
                {SupportsLogProb, SupportsUnnormalizedLogProb},
                True,
            ),
            ({"unnormalized_log_prob": _normal_density}, {SupportsUnnormalizedLogProb}, False),
            (
                {"sample": _normal_draw, "log_prob": _normal_density},
                {SupportsSampling, SupportsLogProb, SupportsUnnormalizedLogProb},
                True,
            ),
            (
                {"sample": _normal_draw, "unnormalized_log_prob": _normal_density},
                {SupportsSampling, SupportsUnnormalizedLogProb},
                True,
            ),
        ],
        ids=["sample", "log_prob", "unnormalized", "sample-log_prob", "sample-unnormalized"],
    )
    def test_the_law_claims_the_capability_of_each_function(self, functions, claimed, normalized):
        law = _law(**functions)
        capabilities = {SupportsSampling, SupportsLogProb, SupportsUnnormalizedLogProb}
        assert {protocol for protocol in capabilities if isinstance(law, protocol)} == claimed
        assert _is_normalized(law) is normalized

    def test_the_law_has_its_label_and_declaration(self):
        law = _law(sample=_normal_draw)
        assert isinstance(law, Distribution)
        assert isinstance(law, NumericDistribution)
        assert law.label == "x"
        assert law.event_spec == OutputSpec(x=VECTOR)

    def test_a_normalized_density_is_also_the_unnormalized_one(self):
        law = _law(log_prob=_normal_density)
        value = jnp.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(
            law._unnormalized_log_prob(value), _normal_density(value), rtol=1e-6
        )

    def test_the_package_exports_it(self):
        assert probpipe.distribution is distribution

    def test_a_law_of_module_functions_pickles(self):
        law = _law(sample=_normal_draw, log_prob=_normal_density)
        restored = pickle.loads(pickle.dumps(law))
        assert isinstance(restored, SupportsSampling)
        assert isinstance(restored, SupportsLogProb)
        assert restored.event_spec == law.event_spec
        key = jax.random.key(3)
        np.testing.assert_allclose(restored._sample(key, (2,)), law._sample(key, (2,)))

    def test_the_repr_names_the_functions(self):
        text = repr(_law(sample=_normal_draw, log_prob=_normal_density))
        assert "_normal_draw" in text
        assert "_normal_density" in text


class TestSampling:
    @pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
    def test_a_sample_shape_leads_the_draws(self, shape):
        draws = sample(_law(sample=_normal_draw), sample_shape=shape)
        assert np.shape(np.asarray(draws)) == (*shape, 3)

    @pytest.mark.parametrize("sampler", [_normal_draw, _numpy_draw], ids=["traces", "numpy"])
    def test_each_draw_is_the_samplers_at_a_key_split_from_the_draws_key(self, sampler):
        key = jax.random.key(7)
        draws = _law(sample=sampler)._sample(key, (2, 3))
        expected = np.stack([np.asarray(sampler(one)) for one in jax.random.split(key, 6)])
        assert draws.shape == (2, 3, 3)
        np.testing.assert_allclose(np.reshape(draws, (6, 3)), expected, rtol=1e-6)

    def test_one_draw_is_the_samplers_at_the_key(self):
        key = jax.random.key(8)
        np.testing.assert_allclose(_law(sample=_numpy_draw)._sample(key), _numpy_draw(key))

    def test_a_sampler_that_does_not_trace_samples_through_the_operation(self):
        draws = sample(_law(sample=_numpy_draw), sample_shape=(4,))
        assert np.shape(np.asarray(draws)) == (4, 3)

    @pytest.mark.parametrize("sampler", [_normal_draw, _numpy_draw], ids=["traces", "numpy"])
    def test_an_empty_sample_shape_axis_gives_no_draws(self, sampler):
        draws = _law(sample=sampler)._sample(jax.random.key(0), (0,))
        assert draws.shape == (0, 3)

    def test_a_record_event_draws_records(self):
        def draw(key: Any) -> dict[str, Any]:
            low, high = jax.random.split(key)
            return {"mu": jax.random.normal(low), "sigma": jnp.exp(jax.random.normal(high, (2,)))}

        law = distribution(
            "theta",
            sample=draw,
            event_spec=RecordSpec(mu=REAL, sigma=NumericArraySpec((2,), jnp.float32, positive)),
        )
        draws = sample(law, sample_shape=(5,))
        assert draws.batch_shape == (5,)
        assert np.shape(np.asarray(draws["sigma"])) == (5, 2)
        assert isinstance(sample(law), Record)


class TestDensities:
    def test_the_density_is_the_users(self):
        value = jnp.array([0.5, -1.0, 2.0])
        law = _law(log_prob=_normal_density)
        np.testing.assert_allclose(
            np.asarray(log_prob(law, value)), _normal_density(value), rtol=1e-6
        )

    def test_the_unnormalized_density_is_the_users(self):
        law = distribution(
            "u",
            unnormalized_log_prob=lambda x: -0.5 * jnp.sum(x**2),
            event_spec=OutputSpec(x=NumericArraySpec((2,))),
        )
        assert float(law._unnormalized_log_prob(jnp.ones(2))) == pytest.approx(-1.0)

    def test_a_batch_is_scored_along_its_leading_axes(self):
        values = jax.random.normal(jax.random.key(1), (2, 5, 3))
        scores = _law(log_prob=_normal_density)._log_prob(values)
        assert scores.shape == (2, 5)
        np.testing.assert_allclose(scores[1, 3], _normal_density(values[1, 3]), rtol=1e-6)

    def test_a_record_value_reaches_the_density_as_a_record(self):
        seen: list[type] = []

        def density(value: Any) -> Any:
            seen.append(type(value))
            return -0.5 * (value["a"] ** 2 + value["b"] ** 2)

        law = distribution(
            "pair", unnormalized_log_prob=density, event_spec=RecordSpec(a=REAL, b=REAL)
        )
        score = unnormalized_log_prob(law, {"a": 1.0, "b": 2.0})
        assert float(np.asarray(score)) == pytest.approx(-2.5)
        assert all(issubclass(kind, Record) for kind in seen)

    def test_a_batch_of_records_is_scored_along_its_leading_axes(self):
        law = distribution(
            "pair",
            unnormalized_log_prob=lambda value: -0.5 * (value["a"] ** 2 + value["b"] ** 2),
            event_spec=RecordSpec(a=REAL, b=REAL),
        )
        scores = law._unnormalized_log_prob({"a": jnp.arange(4.0), "b": jnp.ones(4)})
        np.testing.assert_allclose(scores, -0.5 * (jnp.arange(4.0) ** 2 + 1.0))


class TestTheArguments:
    def test_a_law_needs_a_function(self):
        with pytest.raises(TypeError, match="sample, log_prob, or unnormalized_log_prob"):
            distribution("x", event_spec=VECTOR)

    def test_a_function_that_is_not_callable_raises_naming_it(self):
        with pytest.raises(TypeError, match="log_prob of 'x' must be callable"):
            distribution("x", log_prob=1.0, event_spec=VECTOR)

    def test_both_densities_raise(self):
        with pytest.raises(TypeError, match="got both"):
            distribution(
                "x",
                log_prob=_normal_density,
                unnormalized_log_prob=_normal_density,
                event_spec=VECTOR,
            )

    def test_an_event_spec_that_is_no_spec_raises(self):
        with pytest.raises(TypeError, match="event_spec"):
            distribution("x", sample=_normal_draw, event_spec=(3,))

    def test_a_label_that_is_no_string_raises(self):
        with pytest.raises(TypeError, match="label"):
            distribution(None, sample=_normal_draw, event_spec=VECTOR)

    def test_a_symbolic_dimension_is_accepted(self):
        law = distribution("x", log_prob=_normal_density, event_spec=NumericArraySpec(("n",)))
        assert law.event_spec.spec.free_dims == frozenset({"n"})


class TestTheDeclarationCheck:
    def test_the_draw_completes_the_declaration(self):
        law = distribution("x", sample=_normal_draw, event_spec=NumericArraySpec((3,)))
        assert law.event_spec.spec == VECTOR

    def test_the_draw_fills_a_pending_type(self):
        law = distribution("x", sample=_normal_draw, event_spec=OutputSpec(y=None))
        assert law.event_spec == OutputSpec(y=VECTOR)

    def test_a_draw_of_another_shape_raises(self):
        with pytest.raises(ValueError, match=r"declaration check of 'x'.*\(3,\)"):
            distribution("x", sample=_normal_draw, event_spec=NumericArraySpec((4,)))

    def test_a_draw_of_another_dtype_raises(self):
        with pytest.raises(ValueError, match=r"declaration check of 'y'.*float32"):
            distribution("y", sample=lambda key: jax.random.normal(key, (4,)), event_spec=COUNTS)

    def test_an_array_for_a_record_event_raises(self):
        with pytest.raises(ValueError, match="declaration check"):
            distribution("x", sample=_normal_draw, event_spec=RecordSpec(a=REAL))

    def test_the_check_runs_under_jit(self):
        def build(scale: Any) -> Any:
            law = distribution(
                "x",
                sample=lambda key: scale * jax.random.normal(key, (3,)),
                event_spec=NumericArraySpec((4,)),
            )
            return law._sample(jax.random.key(1))

        with pytest.raises(ValueError, match="declaration check"):
            jax.jit(build)(1.0)

    def test_a_sampler_that_does_not_trace_keeps_the_declaration_as_given(self):
        law = distribution("x", sample=_numpy_draw, event_spec=NumericArraySpec((3,)))
        assert law.event_spec.spec == NumericArraySpec((3,))

    def test_a_density_that_returns_no_scalar_raises(self):
        with pytest.raises(ValueError, match="real scalar"):
            distribution("x", log_prob=lambda value: -0.5 * value**2, event_spec=VECTOR)

    def test_a_pending_type_with_no_sampler_raises(self):
        with pytest.raises(TypeError, match="pending type"):
            distribution("x", log_prob=_normal_density, event_spec=OutputSpec(x=None))

    def test_a_pending_type_with_a_sampler_that_does_not_trace_raises(self):
        with pytest.raises(TypeError, match="pending type"):
            distribution("x", sample=_numpy_draw, event_spec=OutputSpec(x=None))


class TestConstructionEvaluatesNothing:
    def test_construction_calls_no_function_at_a_concrete_value(self):
        concrete: list[str] = []

        def draw(key: Any) -> Any:
            if not isinstance(key, jax.core.Tracer):
                concrete.append("sample")
            return jax.random.normal(key, (3,))

        def density(value: Any) -> Any:
            if not isinstance(value, jax.core.Tracer):
                concrete.append("log_prob")
            return _normal_density(value)

        law = _law(sample=draw, log_prob=density)
        assert concrete == []
        law._sample(jax.random.key(0))
        law._log_prob(jnp.zeros(3))
        assert concrete == ["sample", "log_prob"]

    def test_a_sampler_that_ignores_its_key_is_accepted(self):
        law = _law(sample=lambda key: np.random.normal(size=3).astype(np.float32))
        assert isinstance(law, SupportsSampling)

    def test_draws_outside_the_support_are_accepted(self):
        law = distribution(
            "y", sample=lambda key: jax.random.poisson(key, 3.0, (4,)) - 5, event_spec=COUNTS
        )
        assert isinstance(law, SupportsSampling)


def _numpy_density(value: Any) -> float:
    """A density that does not trace: NumPy's, at a concrete value alone."""
    array = np.asarray(value, dtype=np.float64)
    return float(-0.5 * np.sum(array**2) - 1.5 * np.log(2.0 * np.pi))


class TestFunctionsThatDoNotTrace:
    def test_a_density_that_does_not_trace_is_accepted(self):
        law = _law(log_prob=_numpy_density)
        assert isinstance(law, SupportsLogProb)
        value = jnp.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(
            np.asarray(log_prob(law, value)), _normal_density(value), rtol=1e-6
        )

    def test_a_density_that_does_not_trace_scores_a_batch_one_value_at_a_time(self):
        values = np.random.default_rng(0).normal(size=(2, 4, 3)).astype(np.float32)
        scores = _law(unnormalized_log_prob=_numpy_density)._unnormalized_log_prob(values)
        assert scores.shape == (2, 4)
        np.testing.assert_allclose(scores[1, 2], _numpy_density(values[1, 2]), rtol=1e-6)


def _word(key: Any) -> str:
    """A draw that is no array: one of three words, chosen by the key."""
    return ("low", "mid", "high")[int(jax.random.randint(key, (), 0, 3))]


class TestEventsThatAreNotNumeric:
    def test_an_opaque_event_samples_one_key_at_a_time(self):
        law = distribution("w", sample=_word, event_spec=OpaqueSpec(str))
        assert not isinstance(law, NumericDistribution)
        key = jax.random.key(2)
        draws = law._sample(key, (2, 3))
        assert draws.shape == (2, 3)
        expected = [_word(one) for one in jax.random.split(key, 6)]
        assert list(draws.reshape(-1)) == expected

    def test_an_opaque_event_samples_through_the_operation(self):
        law = distribution("w", sample=_word, event_spec=OpaqueSpec(str))
        assert sample(law).value in {"low", "mid", "high"}
        batch = sample(law, sample_shape=(5,))
        assert isinstance(batch, OpaqueBatch)
        assert batch.batch_shape == (5,)

    def test_a_record_with_a_field_that_is_no_array_samples_into_a_batch_of_records(self):
        def draw(key: Any) -> dict[str, Any]:
            return {"word": _word(key), "x": jax.random.normal(key, ())}

        law = distribution("r", sample=draw, event_spec=RecordSpec(word=OpaqueSpec(str), x=REAL))
        batch = sample(law, sample_shape=(4,))
        assert batch.batch_shape == (4,)
        assert np.shape(np.asarray(batch["x"])) == (4,)

    def test_a_density_of_an_opaque_event_scores_the_value(self):
        law = distribution(
            "w",
            unnormalized_log_prob=lambda word: 0.0 if word == "mid" else -1.0,
            event_spec=OpaqueSpec(str),
        )
        assert float(law._unnormalized_log_prob("mid")) == 0.0


def _counts(rate: Any) -> Distribution:
    """A simulator's law at a rate: four Poisson counts, drawn by JAX."""
    return distribution(
        "y", sample=lambda key: jax.random.poisson(key, rate, (4,)), event_spec=COUNTS
    )


class TestKernels:
    def test_a_simulators_kernel_claims_conditional_sampling_and_no_density(self):
        kernel = conditional_distribution("y", _counts, given_spec={"rate": POSITIVE})
        assert isinstance(kernel, SupportsConditionalSampling)
        assert not isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(kernel, SupportsConditionalLogProb)
        assert kernel.event_spec == OutputSpec(y=COUNTS)

    def test_a_bound_simulator_samples(self):
        kernel = conditional_distribution("y", _counts, given_spec={"rate": POSITIVE})
        law = condition_on(kernel, {"rate": 3.0})
        draws = np.asarray(sample(law, sample_shape=(500,)))
        assert draws.shape == (500, 4)
        # Poisson(3) has standard deviation sqrt(3), so the mean of 2000 counts
        # is within 0.2 of 3 at about five standard errors.
        assert abs(draws.mean() - 3.0) < 0.2

    def test_a_joint_of_the_simulator_and_a_prior_samples(self):
        kernel = conditional_distribution("y", _counts, given_spec={"rate": POSITIVE})
        joint = kernel * probpipe.Gamma("rate", 4.0, 2.0)
        draws = sample(joint, sample_shape=(6,))
        assert np.shape(np.asarray(draws["y"])) == (6, 4)

    def test_a_binding_calls_the_sampler_at_no_concrete_key(self):
        concrete: list[Any] = []

        def counts(rate: Any) -> Distribution:
            def draw(key: Any) -> np.ndarray:
                concrete.append(key)
                rng = np.random.default_rng(_numpy_seed(key))
                return rng.poisson(float(rate), size=4).astype(np.int32)

            return distribution("y", sample=draw, event_spec=COUNTS)

        kernel = conditional_distribution("y", counts, given_spec={"rate": POSITIVE})
        laws = [condition_on(kernel, {"rate": rate}) for rate in (1.0, 2.0, 3.0)]
        assert all(isinstance(key, jax.core.Tracer) for key in concrete)
        assert np.asarray(sample(laws[-1], sample_shape=(3,))).shape == (3, 4)


class TestNormalization:
    def test_a_density_alone_samples_through_a_method_of_the_registry(self):
        law = distribution(
            "theta",
            unnormalized_log_prob=lambda x: -0.5 * jnp.sum(x**2),
            event_spec=OutputSpec(theta=NumericArraySpec((2,), jnp.float32)),
        )
        report = sample.check(law, sample_shape=(10,))
        assert (report.feasible, report.route, report.method) == (
            True,
            "normalize",
            "blackjax_nuts",
        )

    def test_a_sampler_beside_a_density_draws_without_a_method(self):
        law = distribution(
            "x",
            sample=lambda key: jax.random.normal(key, ()),
            unnormalized_log_prob=lambda x: -0.5 * x**2,
            event_spec=REAL,
        )
        assert sample.check(law).route == "exact"
