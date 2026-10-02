"""A kernel built from a function of its given values that returns a law (III.9).

``conditional_distribution`` builds a ``ConditionalDistribution`` from a
function, as ``function`` builds a ``Function``. Each parameter is a given
slot, declared by its annotation or by ``given_spec``. The event declaration is
the returned law's, or an explicit ``event_spec`` that agrees with it.
Conditioning, sampling, and the conditional density derive from the returned
law, each claimed only when that law claims it, and with the law's guard.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import (
    HalfCauchy,
    InputSpec,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
    RecordSpec,
    Uniform,
    workflow_run,
)
from probpipe.core.constraints import positive, real
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    conditional_distribution,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsSampling,
    _capability_guard,
)
from probpipe.families import UnnormalizedDistribution
from tests._ops import EmpiricalDistribution, condition_on, log_prob, sample
from tests.inference import canonical

REAL = NumericArraySpec((), jnp.float32, real)
POSITIVE = NumericArraySpec((), jnp.float32, positive)
SLOTS = {"mu": REAL, "tau": POSITIVE}


def _location(mu: Any, tau: Any) -> Distribution:
    return Normal("y", mu, tau)


def _annotated(mu: REAL, tau: POSITIVE) -> Distribution:
    return Normal("y", mu, tau)


class _GuardedSampler(Distribution, SupportsSampling):
    """A scalar law that samples, whose sampling guard rejects every call."""

    def __init__(self, name: str, loc: Any) -> None:
        super().__init__(name, REAL)
        self.loc = loc

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self.loc + jax.random.normal(key, tuple(sample_shape))

    def _sample_guard(self) -> bool:
        """The stand-in sampler rejects every call."""
        return False


class TestTheGivenSlots:
    def test_given_spec_declares_the_parameters_as_slots(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        assert isinstance(kernel, ConditionalDistribution)
        assert kernel.name == "y"
        assert kernel.given_spec == InputSpec(SLOTS)

    def test_an_annotation_declares_its_parameter(self):
        assert conditional_distribution("y", _annotated).given_spec == InputSpec(SLOTS)

    def test_given_spec_takes_precedence_over_an_annotation(self):
        kernel = conditional_distribution("y", _annotated, given_spec={"tau": REAL})
        assert kernel.given_spec["tau"] == REAL

    def test_a_parameter_that_declares_no_spec_raises(self):
        with pytest.raises(TypeError, match="'tau'"):
            conditional_distribution(
                "y", lambda mu, tau: Normal("y", mu, tau), given_spec={"mu": REAL}
            )

    def test_given_spec_naming_no_parameter_raises(self):
        with pytest.raises(TypeError, match="'sigma'"):
            conditional_distribution("y", _location, given_spec={**SLOTS, "sigma": POSITIVE})

    def test_a_variadic_parameter_raises(self):
        with pytest.raises(TypeError, match="'values'"):
            conditional_distribution("y", lambda **values: Normal("y", 0.0, 1.0))

    def test_a_slot_with_free_dimensions_raises(self):
        with pytest.raises(TypeError, match="free dimensions"):
            conditional_distribution(
                "y", lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec(("n",))}
            )


class TestTheEventDeclaration:
    def test_the_event_declaration_is_the_returned_laws(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        assert kernel.event_spec == Normal("y", 0.0, 1.0).event_spec

    def test_the_slots_shapes_shape_the_event(self):
        kernel = conditional_distribution(
            "y", lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec((3,))}
        )
        assert kernel.event_spec.spec.shape == (3,)

    def test_an_explicit_event_spec_is_completed_from_the_law(self):
        kernel = conditional_distribution(
            "y", _location, given_spec=SLOTS, event_spec=OutputSpec(y=None)
        )
        assert kernel.event_spec == Normal("y", 0.0, 1.0).event_spec

    def test_an_explicit_event_spec_naming_other_components_raises(self):
        with pytest.raises(ValueError, match="components"):
            conditional_distribution(
                "y", _location, given_spec=SLOTS, event_spec=OutputSpec(obs=None)
            )

    def test_a_support_that_depends_on_the_given_values_is_left_undeclared(self):
        kernel = conditional_distribution(
            "u", lambda low: Uniform("u", low, low + 1.0), given_spec={"low": REAL}
        )
        assert kernel.event_spec.spec.support is None
        assert kernel.event_spec.spec.shape == ()


class TestTheCapabilities:
    def test_a_law_that_samples_and_has_a_density_gives_both_twins(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        assert isinstance(kernel, SupportsConditionalSampling)
        assert isinstance(kernel, SupportsConditionalLogProb)

    def test_a_law_with_an_unnormalized_density_gives_that_twin_alone(self):
        kernel = conditional_distribution(
            "x",
            lambda mu: UnnormalizedDistribution(
                "x", lambda x: -0.5 * (x - mu) ** 2, OutputSpec(x=REAL)
            ),
            given_spec={"mu": REAL},
        )
        assert isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(kernel, SupportsConditionalLogProb)
        assert not isinstance(kernel, SupportsConditionalSampling)

    def test_a_law_that_only_samples_gives_conditional_sampling_alone(self):
        kernel = conditional_distribution(
            "x",
            lambda mu: EmpiricalDistribution("x", mu + jnp.arange(3.0)),
            given_spec={"mu": REAL},
        )
        assert isinstance(kernel, SupportsConditionalSampling)
        assert not isinstance(kernel, SupportsConditionalUnnormalizedLogProb)

    def test_the_guard_is_the_laws(self):
        kernel = conditional_distribution(
            "x", lambda mu: _GuardedSampler("x", mu), given_spec={"mu": REAL}
        )
        report = _capability_guard(kernel, "_conditional_sample")
        assert report.feasible is False
        assert "rejects every call" in report.description


class TestEvaluation:
    def test_the_given_values_arrive_by_name_at_their_slots_kinds(self):
        seen: dict[str, type] = {}

        def law(population: Any, scale: Any) -> Distribution:
            seen.update(population=type(population), scale=type(scale))
            return Normal("y", population["mu"], scale)

        kernel = conditional_distribution(
            "y", law, given_spec={"population": RecordSpec(mu=REAL), "scale": POSITIVE}
        )
        result = condition_on(kernel, {"population": {"mu": 1.0}, "scale": 2.0})
        assert issubclass(seen["population"], Record)
        assert isinstance(jnp.zeros(()), seen["scale"])
        np.testing.assert_allclose(result._mean(), 1.0)

    def test_condition_on_returns_the_law_the_function_returns(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        law = condition_on(kernel, {"mu": 1.0, "tau": 2.0})
        assert isinstance(law, Normal)
        np.testing.assert_allclose(law._mean(), 1.0)
        np.testing.assert_allclose(law._variance(), 4.0)

    def test_binding_some_slots_curries_the_kernel(self):
        curried = condition_on(
            conditional_distribution("y", _location, given_spec=SLOTS), {"mu": 1.0}
        )
        assert list(curried.given_spec) == ["tau"]
        assert isinstance(curried, SupportsConditionalSampling)
        np.testing.assert_allclose(condition_on(curried, {"tau": 2.0})._mean(), 1.0)

    def test_the_conditional_density_is_the_laws(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        np.testing.assert_allclose(
            kernel._conditional_log_prob({"mu": 1.0, "tau": 2.0}, 0.5),
            Normal("y", 1.0, 2.0)._log_prob(0.5),
            rtol=1e-6,
        )

    def test_a_given_that_does_not_conform_to_its_slot_raises(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        with pytest.raises(ValueError):
            kernel._condition_on({"mu": jnp.ones(3), "tau": 1.0})

    def test_a_returned_law_that_departs_from_the_declaration_raises(self):
        calls: list[Any] = []

        def law(mu: Any) -> Distribution:
            calls.append(mu)
            return Normal("y" if len(calls) == 1 else "z", mu, 1.0)

        kernel = conditional_distribution("y", law, given_spec={"mu": REAL})
        with pytest.raises(ValueError, match="declares"):
            kernel._condition_on({"mu": 0.0})

    def test_an_option_is_refused(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        with pytest.raises(TypeError, match="num_results"):
            kernel._condition_on({"mu": 0.0, "tau": 1.0}, num_results=3)


class TestTheForms:
    def test_the_decorator_names_the_kernel_after_the_function(self):
        @conditional_distribution(given_spec=SLOTS)
        def likelihood(mu: Any, tau: Any) -> Distribution:
            return Normal("y", mu, tau)

        assert isinstance(likelihood, ConditionalDistribution)
        assert likelihood.name == "likelihood"

    def test_the_bare_decorator_reads_the_annotations(self):
        @conditional_distribution
        def obs(mu: REAL, tau: POSITIVE) -> Distribution:
            return Normal("obs", mu, tau)

        assert obs.given_spec == InputSpec(SLOTS)
        assert list(obs.event_spec.components) == ["obs"]

    def test_the_decorator_takes_a_name(self):
        kernel = conditional_distribution("y_model", given_spec=SLOTS)(_location)
        assert kernel.name == "y_model"

    def test_it_is_exported_from_probpipe(self):
        assert probpipe.conditional_distribution is conditional_distribution


def _school_effects(mu: Any, tau: Any, theta_tilde: Any) -> Distribution:
    sigma = jnp.asarray(canonical.SCHOOL_ERRORS, jnp.float32)
    return Normal("y", mu + tau * theta_tilde, sigma)


class TestEightSchools:
    """The eight-schools likelihood, built from a function rather than a subclass."""

    def test_the_joint_agrees_with_the_canonical_model(self):
        model = canonical.eight_schools().model
        _, *priors = model.factors
        slots = {prior.name: prior.event_spec.spec for prior in priors}
        joint = conditional_distribution("y", _school_effects, given_spec=slots)
        for prior in priors:
            joint = joint * prior
        J = canonical.SCHOOL_EFFECTS.shape[0]
        value = {
            "y": jnp.asarray(canonical.SCHOOL_EFFECTS, jnp.float32),
            "theta_tilde": jnp.linspace(-1.0, 1.0, J),
            "tau": 2.0,
            "mu": 1.0,
        }
        np.testing.assert_allclose(
            jnp.asarray(log_prob(joint, value)), jnp.asarray(log_prob(model, value)), rtol=1e-5
        )
        with workflow_run(seed=0):
            draws = sample(joint, sample_shape=(4,))
        assert draws.batch_shape == (4,)
        assert draws.element_spec["y"].shape == (J,)

    def test_a_nested_prior_gives_record_slots(self):
        J = canonical.SCHOOL_EFFECTS.shape[0]
        sigma = jnp.asarray(canonical.SCHOOL_ERRORS, jnp.float32)
        flat = (
            Normal("mu", 0.0, 5.0)
            * HalfCauchy("tau", 0.0, 5.0)
            * Normal("theta_tilde", jnp.zeros(J), 1.0)
        )
        prior = flat.with_path_names(
            {"mu": "population/mu", "tau": "population/tau", "theta_tilde": "groups/theta_tilde"}
        )

        @conditional_distribution(given_spec=prior.event_spec.components)
        def y(population: Any, groups: Any) -> Distribution:
            return Normal("y", population["mu"] + population["tau"] * groups["theta_tilde"], sigma)

        assert list(y.given_spec) == ["population", "groups"]
        schools = y * prior
        given = {"population": {"mu": 1.0, "tau": 2.0}, "groups": {"theta_tilde": jnp.ones(J)}}
        np.testing.assert_allclose(condition_on(y, given)._mean(), jnp.full(J, 3.0))
        with workflow_run(seed=0):
            draws = sample(schools, sample_shape=(3,))
        assert draws.batch_shape == (3,)
        assert draws.element_spec["y"].shape == (J,)
