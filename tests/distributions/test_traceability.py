"""The traceability a law or a kernel derives from its parts (design V.13, VI.6).

A family is traceable, a law built from functions is traceable when each of
its functions traced, the kernel of a function takes the traceability of the
law it returned when it was built, a joint takes the least of its factors', a
view or a rename takes its parent's, and any other law or kernel probes its
density once. An inference method reads it instead of tracing the target.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericArraySpec,
    OutputSpec,
    condition_on,
    conditional_distribution,
    distribution,
    inference_method_registry,
)
from probpipe.core._execution_claims import Traceability
from probpipe.distributions import ConditionalDistribution, SupportsConditionalSampling
from probpipe.distributions._capabilities import SupportsUnnormalizedLogProb
from probpipe.distributions._distribution import Distribution
from probpipe.inference._blackjax_rwmh import _target_traces
from tests.inference.canonical import ObservationKernel

Y = jnp.array([0.2, -0.4, 1.1])


def _observation_kernel() -> ObservationKernel:
    """``y ~ N(mu, 1)`` three times, a kernel whose methods are JAX code but state nothing."""
    return ObservationKernel(
        "y",
        {"mu": NumericArraySpec(())},
        NumericArraySpec((3,)),
        lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (3,)), 1.0), 1),
    )


class _HostObservationKernel(ObservationKernel):
    """The same kernel, scoring through NumPy, so JAX cannot trace its density."""

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        mu = float(np.asarray(given["mu"]))
        return jnp.asarray(-0.5 * float(np.sum((np.asarray(value) - mu) ** 2)))

    _conditional_unnormalized_log_prob = _conditional_log_prob


class _HostDensity(Distribution, SupportsUnnormalizedLogProb):
    """A law over ``mu`` whose density converts its value to NumPy."""

    def __init__(self) -> None:
        super().__init__(OutputSpec(mu=NumericArraySpec(())), label="host")

    def _unnormalized_log_prob(self, value: Any) -> Any:
        return jnp.asarray(-0.5 * float(np.asarray(value)) ** 2)


class TestDerivedFromParts:
    def test_a_family_is_traceable_without_a_probe(self, probe_attempts):
        assert Normal("mu", 0.0, 1.0)._traceability() is Traceability.TRACEABLE
        assert probe_attempts == []

    def test_a_kernel_that_states_nothing_probes_its_density_once(self, probe_attempts):
        kernel = _observation_kernel()
        assert kernel._traceability() is Traceability.TRACEABLE
        assert kernel._traceability() is Traceability.TRACEABLE
        assert len(probe_attempts) == 1

    def test_a_kernel_whose_event_has_an_unbound_dimension_is_probed_at_size_one(self):
        rows = ObservationKernel(
            "y",
            {"mu": NumericArraySpec(())},
            NumericArraySpec(("rows",)),
            lambda mu: tfd.Independent(tfd.Normal(jnp.reshape(mu, (1,)), 1.0), 1),
        )
        assert rows._traceability() is Traceability.TRACEABLE

    def test_a_kernel_whose_density_converts_to_numpy_is_host_only(self):
        assert (
            _HostObservationKernel(
                "y",
                {"mu": NumericArraySpec(())},
                NumericArraySpec((3,)),
                lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (3,)), 1.0), 1),
            )._traceability()
            is Traceability.HOST_ONLY
        )

    def test_a_law_whose_density_converts_to_numpy_is_host_only(self):
        assert _HostDensity()._traceability() is Traceability.HOST_ONLY

    def test_a_joint_takes_the_least_of_its_factors(self):
        traceable = _observation_kernel() * Normal("mu", 0.0, 1.0)
        host_only = _observation_kernel() * _HostDensity()
        assert traceable._traceability() is Traceability.TRACEABLE
        assert host_only._traceability() is Traceability.HOST_ONLY

    def test_a_view_takes_its_parents_traceability(self):
        host_only = _observation_kernel() * _HostDensity()
        assert host_only["mu"]._traceability() is Traceability.HOST_ONLY

    def test_the_kernel_of_a_function_takes_the_law_it_returned(self):
        def host_score(y):
            return jnp.asarray(-0.5 * float(np.asarray(y)) ** 2)

        traceable = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec(())}, label="k"
        )
        host_only = conditional_distribution(
            lambda mu: distribution(
                unnormalized_log_prob=host_score,
                event_spec=NumericArraySpec((), jnp.float32),
                output_name="y",
            ),
            given_spec={"mu": NumericArraySpec(())},
            label="k",
        )
        assert traceable._traceability() is Traceability.TRACEABLE
        assert host_only._traceability() is Traceability.HOST_ONLY


class TestLawsFromFunctions:
    def test_a_law_built_inside_a_kernel_is_probed_once_over_its_applications(self, probe_attempts):
        kernel = conditional_distribution(
            lambda mu: distribution(
                log_prob=lambda y: -0.5 * (y - mu) ** 2,
                event_spec=NumericArraySpec((), jnp.float32),
                output_name="y",
            ),
            given_spec={"mu": NumericArraySpec((), jnp.float32)},
            label="k",
        )
        # The kernel's construction evaluates its function at a traced given.
        attempts_at_construction = len(probe_attempts)
        for mu in (0.3, 1.0, 2.0):
            law = condition_on(kernel, {"mu": mu})
            assert law._traceability() is Traceability.TRACEABLE
        assert len(probe_attempts) == attempts_at_construction + 1

    def test_a_law_with_only_a_sampler_that_does_not_trace_is_host_only(self):
        law = distribution(
            sample=lambda key: np.float32(np.asarray(jax.random.key_data(key)).sum()),
            event_spec=NumericArraySpec((), jnp.float32),
            output_name="x",
        )
        assert law._traceability() is Traceability.HOST_ONLY

    def test_a_law_whose_density_traces_is_traceable_whatever_its_sampler_does(self):
        law = distribution(
            sample=lambda key: np.float32(np.asarray(jax.random.key_data(key)).sum()),
            log_prob=lambda x: -0.5 * x**2,
            event_spec=NumericArraySpec((), jnp.float32),
            output_name="mu",
        )
        assert law._traceability() is Traceability.TRACEABLE

    def test_laws_whose_samplers_read_a_captured_size_keep_their_own_event(self):
        def law_of_size(size):
            count = np.array(size)
            return distribution(
                sample=lambda key: jax.random.normal(key, (int(count),)),
                event_spec=OutputSpec(x=None),
            )

        assert law_of_size(3).event_spec.term_spec.shape == (3,)
        assert law_of_size(5).event_spec.term_spec.shape == (5,)
        assert law_of_size(3).event_spec.term_spec.shape == (3,)


class TestInferenceReadsTheTraceability:
    def test_nuts_refuses_a_target_whose_likelihood_is_host_only(self):
        model = _HostObservationKernel(
            "y",
            {"mu": NumericArraySpec(())},
            NumericArraySpec((3,)),
            lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (3,)), 1.0), 1),
        ) * Normal("mu", 0.0, 1.0)
        target = condition_on.using(method="unnormalized")(model, {"y": Y})
        report = inference_method_registry.get_method("blackjax_nuts").check(target)
        assert report.feasible is False
        assert report.description == "the log-density is not JAX-traceable"

    def test_nuts_check_traces_nothing_for_a_model_of_families(self, probe_attempts):
        model = conditional_distribution(
            lambda mu: Normal("y", mu * jnp.ones(3), 1.0),
            given_spec={"mu": NumericArraySpec(())},
            label="likelihood",
        ) * Normal("mu", 0.0, 1.0)
        target = condition_on.using(method="unnormalized")(model, {"y": Y})
        attempts_before = len(probe_attempts)
        report = inference_method_registry.get_method("blackjax_nuts").check(target)
        assert report.feasible is True
        assert len(probe_attempts) == attempts_before

    def test_elliptical_slice_refuses_a_likelihood_that_is_host_only(self):
        target = condition_on.using(method="unnormalized")(
            _host_observation_kernel() * Normal("mu", 0.0, 1.0), {"y": Y}
        )
        report = inference_method_registry.get_method("blackjax_elliptical_slice").check(target)
        assert report.feasible is False
        assert report.description == "the log-likelihood is not JAX-traceable"

    def test_elliptical_slice_refuses_a_likelihood_without_a_density(self):
        class _SamplingOnly(ConditionalDistribution, SupportsConditionalSampling):
            def __init__(self) -> None:
                super().__init__(
                    {"mu": NumericArraySpec(())},
                    OutputSpec(y=NumericArraySpec((3,))),
                    label="y",
                )

            def _condition_on(self, given: Any, /, **options: Any) -> Any:
                return Normal("y", given["mu"] * jnp.ones(3), 1.0)

            def _conditional_sample(self, given: Any, key: Any, sample_shape=()) -> Any:
                return self._condition_on(given)._sample(key, sample_shape)

        target = condition_on.using(method="unnormalized")(
            _SamplingOnly() * Normal("mu", 0.0, 1.0), {"y": Y}
        )
        report = inference_method_registry.get_method("blackjax_elliptical_slice").check(target)
        assert report.feasible is False
        assert "the likelihood must have an unnormalized log-density" in report.description

    def test_tfp_nuts_refuses_a_target_that_is_host_only(self):
        target = condition_on.using(method="unnormalized")(
            _host_observation_kernel() * Normal("mu", 0.0, 1.0), {"y": Y}
        )
        report = inference_method_registry.get_method("tfp_nuts").check(target)
        assert report.feasible is False
        assert report.description == "the log-density is not JAX-traceable"

    def test_rwmh_runs_a_host_only_target_in_its_python_loop(self):
        traceable = condition_on.using(method="unnormalized")(
            _observation_kernel() * Normal("mu", 0.0, 1.0), {"y": Y}
        )
        host_only = condition_on.using(method="unnormalized")(
            _host_observation_kernel() * Normal("mu", 0.0, 1.0), {"y": Y}
        )
        start = jnp.zeros(1)
        assert _target_traces(traceable, None, None, start) is True
        assert _target_traces(host_only, None, None, start) is False

    def test_empirical_reweighting_scores_a_host_only_likelihood_atom_by_atom(self):
        atoms = jnp.linspace(-2.0, 2.0, 9)
        weights = {}
        for name, kernel in (
            ("traceable", _observation_kernel()),
            ("host_only", _host_observation_kernel()),
        ):
            prior = EmpiricalDistribution(atoms, output_name="mu", label="prior")
            posterior = condition_on.using(method="empirical_reweighting")(kernel * prior, {"y": Y})
            weights[name] = np.asarray(posterior.weights)
        np.testing.assert_allclose(weights["host_only"], weights["traceable"], rtol=1e-5)


def _host_observation_kernel() -> _HostObservationKernel:
    """``y ~ N(mu, 1)`` three times, scored through NumPy."""
    return _HostObservationKernel(
        "y",
        {"mu": NumericArraySpec(())},
        NumericArraySpec((3,)),
        lambda mu: tfd.Independent(tfd.Normal(jnp.broadcast_to(mu, (3,)), 1.0), 1),
    )
