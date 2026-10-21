"""The inference target and the run key: what every inference method receives, and its contract.

A method registered in a fresh registry records the target and the key
``condition_on`` hands it, so the tests read the three forms of target: a model
at observed values of its fields, a law that is itself unnormalized, and a model
at data it does not declare as fields. Each is checked against the contract of
``InferenceTarget``: its parts, the flat layout of ``unflatten`` and
``log_density``, and ``factors``. The key is checked against the contract of
``InferenceMethod.execute``: one per run, derived from the enclosing workflow.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from probpipe import (
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
    condition_on,
    conditional_distribution,
    workflow_run,
)
from probpipe.core._dispatch import Feasibility
from probpipe.distributions import Distribution
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.inference import (
    InferenceMethod,
    InferenceTarget,
    ModelFactors,
    inference_method_registry,
)
from probpipe.inference._inference_utils import flat_vector
from probpipe.operations._condition import _inference_run_key, _InferenceMethodRegistry
from probpipe.operations._operation import _RegistryRoute

Y = jnp.array([0.4, 0.9, -0.1])


class _TargetRecorded(Exception):
    """Raised by the recording method once it holds the target, so no method runs."""


class _RecordingMethod(InferenceMethod):
    """A method that applies to every target and records the target it receives."""

    def __init__(self) -> None:
        self.received_targets: list[Any] = []
        self.received_keys: list[Any] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def description(self) -> str:
        return "Records its target."

    @property
    def priority(self) -> int:
        return 1

    def supported_types(self) -> tuple[type, ...]:
        return (object,)

    def check(self, target: Any, /, **options: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, target: Any, /, *, key: Any, **options: Any) -> Any:
        self.received_targets.append(target)
        self.received_keys.append(key)
        raise _TargetRecorded


@pytest.fixture
def recording_method(monkeypatch):
    """The recording method, alone in the registry that ``condition_on`` reads."""
    method = _RecordingMethod()
    registry = _InferenceMethodRegistry(name="recording", description="The recording method.")
    registry.register(method)
    method.registry = registry
    for route in condition_on.routes:
        if isinstance(route, _RegistryRoute):
            monkeypatch.setattr(route, "registry", registry)
    return method


@pytest.fixture
def received_target(recording_method):
    """The function that runs a registry call and returns the target the method received."""

    def target_of(call: Any, *args: Any) -> Any:
        with pytest.raises(_TargetRecorded):
            call(recording_method.registry, *args)
        return recording_method.received_targets[-1]

    return target_of


def _regression() -> Distribution:
    """``y ~ Normal(beta_0 + beta_1 + mu, 1)``, three times, with priors on ``beta`` and ``mu``."""
    likelihood = conditional_distribution(
        lambda beta, mu: Normal("y", (jnp.sum(beta) + mu) * jnp.ones(3), 1.0),
        given_spec={"beta": NumericArraySpec((2,)), "mu": NumericArraySpec(())},
        label="likelihood",
    )
    prior = MultivariateNormal("beta", jnp.zeros(2), jnp.eye(2)) * Normal("mu", 0.0, 2.0)
    return likelihood * prior


def _log_joint(beta: Any, mu: Any) -> Any:
    """The regression's joint log-density at *beta*, *mu*, and ``Y``, by hand."""
    return (
        multivariate_normal.logpdf(beta, jnp.zeros(2), jnp.eye(2))
        + norm.logpdf(mu, 0.0, 2.0)
        + jnp.sum(norm.logpdf(Y, jnp.sum(beta) + mu, 1.0))
    )


def _by_condition_on(registry: Any, model: Any, given: Any) -> Any:
    return condition_on(model, given)


def _by_the_registry(registry: Any, *args: Any) -> Any:
    return registry.execute(*args)


class _UnnormalizedMatrix(Distribution):
    """A law over a 2x2 matrix with only an unnormalized density, ``-|x|^2 / 2``."""

    def __init__(self) -> None:
        super().__init__(OutputSpec(x=NumericArraySpec((2, 2))), label="matrix")

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return -0.5 * jnp.sum(jnp.asarray(value) ** 2)


class _DataModel(Distribution):
    """A model over ``theta`` whose density reads data it does not declare: ``(theta, data)``."""

    def __init__(self) -> None:
        super().__init__(OutputSpec(theta=NumericArraySpec(())), label="data_model")

    def _unnormalized_log_prob(self, value: Any) -> Any:
        theta, data = value
        return jnp.sum(norm.logpdf(data, theta, 1.0))


class _OfUnknownLength(Distribution):
    """A law over an array whose length a name stands for, so no flat layout holds it."""

    def __init__(self) -> None:
        super().__init__(OutputSpec(z=NumericArraySpec(("n",))), label="unknown_length")

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return -jnp.sum(jnp.asarray(value) ** 2)


class _WithoutDensity(Distribution, SupportsSampling):
    """A law over a scalar that only samples."""

    def __init__(self) -> None:
        super().__init__(OutputSpec(s=NumericArraySpec(())), label="sampler_only")

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return jax.random.normal(key, sample_shape)


# ---------------------------------------------------------------------------
# What a method receives
# ---------------------------------------------------------------------------


class TestEveryMethodReceivesAnInferenceTarget:
    def test_a_model_at_values_of_its_fields(self, received_target):
        model = _regression()
        target = received_target(_by_condition_on, model, {"y": Y})
        assert isinstance(target, InferenceTarget)
        assert target.joint is model
        assert isinstance(target.given, Record)
        assert list(target.given.fields) == ["y"]
        assert set(target.event_spec.outputs) == {"beta", "mu"}
        assert target.label == model.label

    def test_a_law_that_is_itself_unnormalized(self, received_target):
        law = _UnnormalizedMatrix()
        target = received_target(_by_the_registry, law)
        assert isinstance(target, InferenceTarget)
        assert target.joint is law
        assert isinstance(target.given, Record)
        assert list(target.given.fields) == []
        assert target.event_spec == law.event_spec

    def test_a_model_at_data_it_does_not_declare(self, received_target):
        model, data = _DataModel(), jnp.array([1.0, 2.0])
        target = received_target(_by_the_registry, model, data)
        assert isinstance(target, InferenceTarget)
        assert target.joint is model
        assert target.given is data

    def test_a_target_is_not_wrapped_again(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        assert received_target(_by_the_registry, target) is target


# ---------------------------------------------------------------------------
# The flat layout
# ---------------------------------------------------------------------------


class TestTheFlatLayout:
    def test_a_record_event_lays_out_its_fields_in_the_events_order(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        position = jnp.array([1.0, 2.0, 3.0])
        draw = target.unflatten(position)
        assert isinstance(draw, Record)
        widths = {"beta": 2, "mu": 1}
        expected, offset = {}, 0
        for field in target.event_spec.outputs:
            expected[field] = position[offset : offset + widths[field]]
            offset += widths[field]
        np.testing.assert_array_equal(np.ravel(draw["beta"]), expected["beta"])
        np.testing.assert_array_equal(np.ravel(draw["mu"]), expected["mu"])
        np.testing.assert_array_equal(flat_vector(draw), position)

    def test_an_event_of_one_array_is_that_array_in_its_shape(self, received_target):
        target = received_target(_by_the_registry, _UnnormalizedMatrix())
        draw = target.unflatten(jnp.arange(4.0))
        np.testing.assert_array_equal(draw, jnp.arange(4.0).reshape(2, 2))

    def test_the_log_density_is_the_joints_at_the_position_and_the_observed_values(
        self, received_target
    ):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        position = jnp.array([0.3, -0.2, 0.5])
        draw = target.unflatten(position)
        expected = _log_joint(jnp.ravel(draw["beta"]), jnp.squeeze(draw["mu"]))
        np.testing.assert_allclose(target.log_density(position), expected, rtol=1e-5)
        assert jnp.shape(target.log_density(position)) == ()

    def test_the_log_density_compiles_and_differentiates_for_a_jax_model(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        position = jnp.array([0.3, -0.2, 0.5])
        np.testing.assert_allclose(
            jax.jit(target.log_density)(position), target.log_density(position), rtol=1e-6
        )
        gradient = jax.grad(target.log_density)(position)
        assert gradient.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_the_log_density_of_a_law_conditioned_on_nothing_is_its_own(self, received_target):
        target = received_target(_by_the_registry, _UnnormalizedMatrix())
        np.testing.assert_allclose(target.log_density(jnp.full(4, 2.0)), -8.0)

    def test_the_log_density_of_a_model_at_data_reads_the_data(self, received_target):
        model, data = _DataModel(), jnp.array([1.0, 2.0])
        target = received_target(_by_the_registry, model, data)
        np.testing.assert_allclose(
            target.log_density(jnp.array([1.5])), jnp.sum(norm.logpdf(data, 1.5, 1.0)), rtol=1e-6
        )

    def test_a_target_without_a_flat_layout_refuses_a_position(self, received_target):
        target = received_target(_by_the_registry, _OfUnknownLength())
        with pytest.raises(TypeError, match="cannot lay out the draws of _OfUnknownLength"):
            target.unflatten(jnp.zeros(3))
        with pytest.raises(TypeError, match="cannot lay out the draws"):
            target.log_density(jnp.zeros(3))

    def test_a_target_without_a_density_refuses_log_density(self, received_target):
        target = received_target(_by_the_registry, _WithoutDensity())
        with pytest.raises(TypeError, match="unnormalized log-density"):
            target.log_density(jnp.zeros(1))


# ---------------------------------------------------------------------------
# The factors
# ---------------------------------------------------------------------------


class TestTheFactors:
    def test_a_merged_model_gives_its_prior_likelihood_and_observed_value(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        factors = target.factors()
        assert isinstance(factors, ModelFactors)
        assert set(factors.prior.event_spec.outputs) == {"beta", "mu"}
        assert list(factors.likelihood.event_spec.outputs) == ["y"]
        np.testing.assert_array_equal(factors.observed, Y)

    def test_a_law_conditioned_on_nothing_has_no_factors(self, received_target):
        assert received_target(_by_the_registry, _UnnormalizedMatrix()).factors() is None

    def test_a_model_at_undeclared_data_has_no_factors(self, received_target):
        target = received_target(_by_the_registry, _DataModel(), jnp.array([1.0]))
        assert target.factors() is None

    def test_a_factor_that_produces_an_observed_and_an_unobserved_field_gives_none(
        self, received_target
    ):
        model = Normal("a", 0.0, 1.0) * MultivariateNormal("b", jnp.zeros(2), jnp.eye(2))
        joint = conditional_distribution(
            lambda a: Normal("c", a, 1.0) * Normal("d", a, 1.0),
            given_spec={"a": NumericArraySpec(())},
            label="pair",
        )
        target = received_target(_by_condition_on, joint * model, {"c": 0.2})
        assert target.factors() is None


# ---------------------------------------------------------------------------
# The run key
# ---------------------------------------------------------------------------


def _key_of_one_run(recording_method: Any, seed: int | None) -> Any:
    """The key the recording method receives for one ``condition_on`` of the regression."""
    with pytest.raises(_TargetRecorded):
        if seed is None:
            condition_on(_regression(), {"y": Y})
        else:
            with workflow_run(seed=seed):
                condition_on(_regression(), {"y": Y})
    return recording_method.received_keys[-1]


class TestTheRunKey:
    def test_a_run_receives_the_workflows_key_for_that_run_of_the_method(self, recording_method):
        with workflow_run(seed=3), pytest.raises(_TargetRecorded):
            recording_method.registry.execute(_regression(), {"y": Y})
        key = recording_method.received_keys[-1]
        with workflow_run(seed=3):
            expected = _inference_run_key("recording")
        assert jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(expected))

    def test_one_seed_gives_one_key_and_two_seeds_two(self, recording_method):
        first, again, other = (_key_of_one_run(recording_method, seed) for seed in (3, 3, 4))
        np.testing.assert_array_equal(jax.random.key_data(first), jax.random.key_data(again))
        assert not np.array_equal(jax.random.key_data(first), jax.random.key_data(other))

    def test_two_runs_in_one_workflow_receive_different_keys(self, recording_method):
        with workflow_run(seed=3):
            keys = []
            for _ in range(2):
                with pytest.raises(_TargetRecorded):
                    condition_on(_regression(), {"y": Y})
                keys.append(recording_method.received_keys[-1])
        assert not np.array_equal(jax.random.key_data(keys[0]), jax.random.key_data(keys[1]))

    def test_check_receives_no_key(self, recording_method):
        recorded_options = []

        def check(target: Any, /, **options: Any) -> Feasibility:
            recorded_options.append(options)
            return Feasibility(True)

        recording_method.check = check
        recording_method.registry.check(_regression(), {"y": Y}, num_draws=2)
        assert recorded_options == [{"num_draws": 2}]

    @pytest.mark.parametrize("method_name", inference_method_registry.list_methods())
    def test_every_registered_method_takes_the_key_as_a_required_keyword(self, method_name):
        import inspect

        execute = inference_method_registry.get_method(method_name).execute
        key = inspect.signature(execute).parameters["key"]
        assert key.kind is inspect.Parameter.KEYWORD_ONLY
        assert key.default is inspect.Parameter.empty

    def test_a_method_called_without_a_key_raises(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        method = inference_method_registry.get_method("blackjax_rwmh")
        with pytest.raises(TypeError, match="key"):
            method.execute(target, num_results=2, num_warmup=2)

    def test_a_method_called_directly_draws_from_the_key_it_is_given(self, received_target):
        target = received_target(_by_condition_on, _regression(), {"y": Y})
        method = inference_method_registry.get_method("blackjax_rwmh")
        options = {"num_results": 5, "num_warmup": 5, "num_chains": 1}
        first = method.execute(target, key=jax.random.key(0), **options)
        again = method.execute(target, key=jax.random.key(0), **options)
        other = method.execute(target, key=jax.random.key(1), **options)
        np.testing.assert_array_equal(first.atoms.raw["beta"], again.atoms.raw["beta"])
        assert not np.array_equal(first.atoms.raw["beta"], other.atoms.raw["beta"])

    @pytest.mark.parametrize(
        "function_name",
        [
            "rwmh",
            "elliptical_slice",
            "condition_on_nutpie",
            "learn_amortized_likelihood",
            "learn_amortized_ratio",
            "learn_amortized_posterior",
        ],
    )
    def test_a_function_that_runs_inference_outside_the_registry_takes_no_key(self, function_name):
        import inspect

        import probpipe.inference

        parameters = inspect.signature(getattr(probpipe.inference, function_name)).parameters
        assert "key" not in parameters
