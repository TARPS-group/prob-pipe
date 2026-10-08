"""A kernel built from a function of its given values that returns a law (IV.4).

``conditional_distribution`` builds a ``ConditionalDistribution`` from a
function, as ``function`` builds a ``Function``. Each parameter is a given
slot, declared by its annotation or by ``given_spec``. The event declaration is
the returned law's, or an explicit ``event_spec`` that agrees with it.
Conditioning, sampling, and the conditional density derive from the returned
law, each claimed only when that law claims it, and with the law's guard.
"""

from __future__ import annotations

import pickle
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
    OpaqueSpec,
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
    distribution,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsSampling,
    _capability_guard,
)
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

    def __init__(self, label: str, loc: Any) -> None:
        super().__init__(label, REAL)
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
        assert kernel.label == "y"
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

    def test_the_refusal_shows_the_declaration_that_fixes_it(self):
        with pytest.raises(TypeError, match=r"given_spec=\{'tau': NumericArraySpec\(\(\)\)\}"):
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
        with pytest.raises(TypeError, match="unbound dimensions"):
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
        with pytest.raises(ValueError, match="declares the fields"):
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
            lambda mu: distribution(
                "x",
                unnormalized_log_prob=lambda x: -0.5 * (x - mu) ** 2,
                event_spec=OutputSpec(x=REAL),
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

    def test_a_kernel_and_its_curried_kernel_pickle(self):
        kernel = conditional_distribution("y", _location, given_spec=SLOTS)
        restored = pickle.loads(pickle.dumps(kernel))
        np.testing.assert_allclose(condition_on(restored, {"mu": 1.0, "tau": 2.0})._mean(), 1.0)
        curried = pickle.loads(pickle.dumps(condition_on(kernel, {"mu": 1.0})))
        assert list(curried.given_spec) == ["tau"]
        np.testing.assert_allclose(condition_on(curried, {"tau": 2.0})._mean(), 1.0)


#: An array default, which declares its slot's shape.
_OFFSETS = jnp.zeros(3)


def _scaled(mu: Any, scale: Any = 2.0) -> Distribution:
    return Normal("y", mu, scale)


class TestOptionalSlots:
    """A parameter with a default is an optional slot, which takes its default when left unbound."""

    def test_a_parameter_with_a_default_is_an_optional_slot(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        assert kernel.given_spec.required == ("mu",)
        assert kernel.given_spec.optional == {"scale"}
        assert kernel.given_spec["scale"] == NumericArraySpec(())

    def test_given_spec_declares_an_optional_slot(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL, "scale": POSITIVE})
        assert kernel.given_spec["scale"] == POSITIVE
        assert kernel.given_spec.optional == {"scale"}

    def test_binding_the_required_slots_takes_the_default(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        law = condition_on(kernel, {"mu": 1.0})
        assert isinstance(law, Normal)
        np.testing.assert_allclose(law._variance(), 4.0)

    def test_a_bound_optional_slot_replaces_the_default(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        np.testing.assert_allclose(condition_on(kernel, {"mu": 1.0, "scale": 3.0})._variance(), 9.0)

    def test_binding_an_optional_slot_alone_curries_the_kernel(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        curried = condition_on(kernel, {"scale": 3.0})
        assert curried.given_spec == InputSpec(mu=REAL)
        np.testing.assert_allclose(condition_on(curried, {"mu": 0.0})._variance(), 9.0)

    def test_construction_reads_the_law_at_the_defaults(self):
        """A default that fixes a shape reaches the function as its value."""
        kernel = conditional_distribution(
            "y", lambda mu, n=3: Normal("y", mu * jnp.ones(n), 1.0), given_spec={"mu": REAL}
        )
        assert kernel.event_spec.components["y"].shape == (3,)

    def test_a_kernel_with_an_optional_slot_pickles(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        restored = pickle.loads(pickle.dumps(kernel))
        assert restored.given_spec == kernel.given_spec
        np.testing.assert_allclose(condition_on(restored, {"mu": 1.0})._variance(), 4.0)
        curried = pickle.loads(pickle.dumps(condition_on(kernel, {"scale": 3.0})))
        np.testing.assert_allclose(condition_on(curried, {"mu": 1.0})._variance(), 9.0)

    def test_an_array_default_declares_its_shape(self):
        kernel = conditional_distribution(
            "y", lambda mu, loc=_OFFSETS: Normal("y", mu + loc, 1.0), given_spec={"mu": REAL}
        )
        assert kernel.given_spec["loc"].shape == (3,)
        assert kernel.event_spec.components["y"].shape == (3,)

    def test_an_annotation_declares_an_optional_slot(self):
        def law(mu: REAL, scale: POSITIVE = 2.0) -> Distribution:
            return Normal("y", mu, scale)

        kernel = conditional_distribution("y", law)
        assert kernel.given_spec == InputSpec(mu=REAL, scale=POSITIVE).with_optional("scale")

    def test_a_keyword_only_default_is_an_optional_slot(self):
        kernel = conditional_distribution(
            "y", lambda mu, *, scale=2.0: Normal("y", mu, scale), given_spec={"mu": REAL}
        )
        assert kernel.given_spec.optional == {"scale"}
        np.testing.assert_allclose(condition_on(kernel, {"mu": 0.0, "scale": 3.0})._variance(), 9.0)

    def test_a_default_that_is_no_array_is_an_opaque_slot(self):
        def law(mu: Any, family: Any = "narrow") -> Distribution:
            return Normal("y", mu, 1.0 if family == "narrow" else 2.0)

        kernel = conditional_distribution("y", law, given_spec={"mu": REAL})
        assert kernel.given_spec["family"] == OpaqueSpec(str)
        np.testing.assert_allclose(condition_on(kernel, {"mu": 0.0})._variance(), 1.0)
        np.testing.assert_allclose(
            condition_on(kernel, {"mu": 0.0, "family": "wide"})._variance(), 4.0
        )

    def test_a_none_default_takes_a_declared_spec_to_be_met(self):
        """``None`` declares a slot that only ``None`` conforms to, so a prior meets it once declared."""

        def law(mu: Any, shift: Any = None) -> Distribution:
            return Normal("y", mu + (0.0 if shift is None else shift), 1.0)

        prior = Normal("mu", 0.0, 1.0) * Normal("shift", 0.0, 1.0)
        undeclared = conditional_distribution("y", law, given_spec={"mu": REAL})
        assert undeclared.given_spec["shift"] == OpaqueSpec(type(None))
        with pytest.raises(ValueError, match="'shift'"):
            undeclared * prior
        declared = conditional_distribution("y", law, given_spec={"mu": REAL, "shift": REAL})
        assert declared.given_spec.optional == {"shift"}
        assert list((declared * prior).event_spec.components) == ["y", "mu", "shift"]

    def test_a_kernel_whose_slots_are_all_optional_binds_at_its_defaults(self):
        kernel = conditional_distribution("y", lambda mu=1.0: Normal("y", mu, 1.0))
        assert kernel.given_spec.required == ()
        np.testing.assert_allclose(kernel._condition_on({})._mean(), 1.0)
        np.testing.assert_allclose(condition_on(kernel, {"mu": 3.0})._mean(), 3.0)

    def test_binding_a_required_slot_curries_over_the_rest_and_keeps_them_optional(self):
        kernel = conditional_distribution(
            "y",
            lambda mu, tau, scale=1.0: Normal("y", mu, tau * scale),
            given_spec={"mu": REAL, "tau": POSITIVE},
        )
        curried = condition_on(kernel, {"mu": 0.0})
        expected = InputSpec(tau=POSITIVE, scale=NumericArraySpec(())).with_optional("scale")
        assert curried.given_spec == expected

    def test_a_value_that_does_not_conform_to_an_optional_slot_raises(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        with pytest.raises(ValueError):
            kernel._condition_on({"mu": 0.0, "scale": jnp.ones(3)})

    def test_the_conditional_density_and_draws_take_the_default(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        np.testing.assert_allclose(
            kernel._conditional_log_prob({"mu": 1.0}, 0.5),
            Normal("y", 1.0, 2.0)._log_prob(0.5),
            rtol=1e-6,
        )
        draws = kernel._conditional_sample({"mu": 0.0}, jax.random.PRNGKey(0), (4000,))
        np.testing.assert_allclose(float(jnp.std(draws)), 2.0, rtol=0.05)

    def test_a_renamed_optional_slot_stays_optional(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        renamed = kernel.with_path_names({"scale": "sigma"})
        assert renamed.given_spec.optional == {"sigma"}
        np.testing.assert_allclose(condition_on(renamed, {"mu": 0.0})._variance(), 4.0)
        np.testing.assert_allclose(
            condition_on(renamed, {"mu": 0.0, "sigma": 3.0})._variance(), 9.0
        )
        assert kernel.with_path_names({"mu": "m"}).given_spec.optional == {"scale"}

    def test_grouping_an_optional_slot_makes_it_required(self):
        kernel = conditional_distribution("y", _scaled, given_spec={"mu": REAL})
        grouped = kernel.with_path_names({"mu": "pars/mu", "scale": "pars/scale"})
        assert grouped.given_spec.required == ("pars",)
        assert grouped.given_spec.optional == frozenset()

    def test_the_posterior_fits_the_slots_a_prior_produces(self):
        """With the default scale the model is conjugate, so NUTS meets its posterior mean."""
        kernel = conditional_distribution(
            "y", lambda mu, scale=2.0: Normal("y", mu * jnp.ones(5), scale), given_spec={"mu": REAL}
        )
        y = jnp.array([1.0, 2.0, 0.5, 1.5, 2.5])
        budget = {"num_results": 500, "num_warmup": 500, "num_chains": 2}
        with workflow_run(seed=0):
            posterior = condition_on.with_options(method="blackjax_nuts", method_options=budget)(
                kernel * Normal("mu", 0.0, 1.0), {"y": y}
            )
        assert list(posterior.event_spec.components) == ["mu"]
        # Precision 1 + 5 / 4, so the mean is (sum(y) / 4) / 2.25.
        expected = float(jnp.sum(y)) / 4.0 / 2.25
        draws = np.asarray(sample(posterior, sample_shape=(2000,)))
        np.testing.assert_allclose(draws.mean(), expected, atol=0.1)


class TestTheForms:
    def test_the_decorator_names_the_kernel_after_the_function(self):
        @conditional_distribution(given_spec=SLOTS)
        def likelihood(mu: Any, tau: Any) -> Distribution:
            return Normal("y", mu, tau)

        assert isinstance(likelihood, ConditionalDistribution)
        assert likelihood.label == "likelihood"

    def test_the_bare_decorator_reads_the_annotations(self):
        @conditional_distribution
        def obs(mu: REAL, tau: POSITIVE) -> Distribution:
            return Normal("obs", mu, tau)

        assert obs.given_spec == InputSpec(SLOTS)
        assert list(obs.event_spec.components) == ["obs"]

    def test_the_decorator_takes_a_name(self):
        kernel = conditional_distribution("y_model", given_spec=SLOTS)(_location)
        assert kernel.label == "y_model"

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
        slots = {prior.label: prior.event_spec.spec for prior in priors}
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
