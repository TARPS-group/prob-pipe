"""A rename that gathers a joint's components under a node regroups their factors (IV.1).

Each gathering node becomes one factor of the renamed joint: a packaged
sub-joint of the factors that produce the gathered components, whose event is
the node as one component holding their record, labeled as ``*`` labels the
joint of those factors. A factor that consumes a gathered component conditions
on the sub-joint through its renamed given slot, which holds the node's whole
record. The joint stays factored, so conditioning on a node, and the factor,
the view, and the marginal at a node, take the factored routes. A gathering
whose groups condition on one another in a cycle renames at the joint's
boundary, which realizes the same law.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import HalfCauchy, LogNormal, Normal, OutputSpec, RecordSpec, workflow_run
from probpipe.distributions import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    SupportsFactors,
    conditional_distribution,
)
from probpipe.distributions._capabilities import (
    SupportsLogProb,
    SupportsMean,
    _capability_guard,
)
from probpipe.inference import ApproximateDistribution
from tests._ops import condition_on, factor, log_prob, marginal, mean, sample

J = 8

#: The renames that gather the non-centered prior's components into two groups.
NON_CENTERED = {"mu": "population/mu", "tau": "population/tau", "theta_tilde": "groups/theta_tilde"}

#: The renames that gather the centered prior's components into two groups.
CENTERED = {"mu": "population/mu", "tau": "population/tau", "theta": "groups/theta"}


def _mu() -> Normal:
    return Normal("mu", 0.0, 5.0)


def _tau() -> HalfCauchy:
    return HalfCauchy("tau", 0.0, 5.0)


def _non_centered() -> FactoredDistribution:
    return _mu() * _tau() * Normal("theta_tilde", jnp.zeros(J), 1.0)


def _school_effects(mu, tau) -> Normal:
    """The law ``theta | mu, tau ~ Normal(mu, tau)`` of the eight school effects."""
    return Normal("theta", mu * jnp.ones(J), tau)


def _centered() -> FactoredDistribution:
    """The centered prior, whose school effects condition on ``mu`` and ``tau``."""
    slots = {"mu": _mu().event_spec.spec, "tau": _tau().event_spec.spec}
    theta = conditional_distribution("theta", _school_effects, given_spec=slots)
    return theta * _mu() * _tau()


_MODELS = [
    pytest.param(_non_centered, NON_CENTERED, "theta_tilde", id="non-centered"),
    pytest.param(_centered, CENTERED, "theta", id="centered"),
]

_POPULATION = {"mu": 1.0, "tau": 2.0}

#: The renames that gather ``mu`` and ``theta`` of the chain ``theta | tau``, ``tau | mu``.
_CHAIN_RENAMES = {"mu": "g/mu", "theta": "g/theta"}


def _value(groups_field: str) -> dict:
    """A value of the renamed prior, with every group field at known values."""
    return {
        "population": {"mu": jnp.asarray(1.0), "tau": jnp.asarray(2.0)},
        "groups": {groups_field: jnp.linspace(-1.0, 1.0, J)},
    }


def _original(value: dict) -> dict:
    """The value of the original prior that *value*, a value of the renamed one, renames."""
    return {**value["population"], **value["groups"]}


class TestTheRegroupedJoint:
    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_each_gathering_node_is_one_factor(self, make, renames, field):
        prior = make().with_path_names(renames)
        assert isinstance(prior, FactoredDistribution)
        assert isinstance(prior, SupportsFactors)
        assert set(prior.event_spec.components) == {"population", "groups"}
        assert len(prior.factors) == 2
        population = factor(prior, "population")
        assert isinstance(population, FactoredDistribution)
        assert population.event_spec == OutputSpec(
            population=RecordSpec(mu=_mu().event_spec.spec, tau=_tau().event_spec.spec)
        )
        assert [part.label for part in population.factors] == ["mu", "tau"]
        assert population.label == "mu·tau"

    def test_a_group_of_one_factor_packages_it(self):
        groups = factor(_non_centered().with_path_names(NON_CENTERED), "groups")
        assert isinstance(groups, FactoredDistribution)
        assert not groups.event_spec.exposes_record
        assert list(groups.event_spec.components) == ["groups"]
        assert list(groups.event_spec.spec.children) == ["theta_tilde"]
        assert groups.label == "theta_tilde"

    def test_a_consumer_conditions_on_the_sub_joint_through_its_renamed_slot(self):
        groups = factor(_centered().with_path_names(CENTERED), "groups")
        assert isinstance(groups, FactoredConditionalDistribution)
        assert list(groups.given_spec) == ["population"]
        assert list(groups.given_spec["population"].children) == ["mu", "tau"]
        assert list(groups.event_spec.components) == ["groups"]

    def test_a_consumer_slot_follows_the_record_order_of_the_sub_joint(self):
        renames = {"tau": "population/tau", "mu": "population/mu", "theta": "groups/theta"}
        prior = _centered().with_path_names(renames)
        groups = factor(prior, "groups")
        population = factor(prior, "population")
        assert list(groups.given_spec["population"].children) == list(
            population.event_spec.spec.children
        )

    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_it_scores_as_the_original(self, make, renames, field):
        original = make()
        prior = original.with_path_names(renames)
        value = _value(field)
        np.testing.assert_allclose(
            jnp.asarray(log_prob(prior, value)),
            jnp.asarray(log_prob(original, _original(value))),
            rtol=1e-6,
        )

    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_it_pickles(self, make, renames, field):
        prior = make().with_path_names(renames)
        restored = pickle.loads(pickle.dumps(prior))
        assert restored.event_spec == prior.event_spec
        assert [part.event_spec for part in restored.factors] == [
            part.event_spec for part in prior.factors
        ]

    def test_a_nested_joint_is_built_one_group_at_a_time(self):
        prior = _non_centered()
        stepwise = prior.with_path_names({"mu": "population/mu", "tau": "population/tau"})
        stepwise = stepwise.with_path_names({"theta_tilde": "groups/theta_tilde"})
        at_once = prior.with_path_names(NON_CENTERED)
        assert isinstance(stepwise, FactoredDistribution)
        assert stepwise.event_spec == at_once.event_spec
        assert [part.event_spec for part in stepwise.factors] == [
            part.event_spec for part in at_once.factors
        ]

    def test_a_joint_of_a_packaged_factor_keeps_it_whole(self):
        population = factor(_non_centered().with_path_names(NON_CENTERED), "population")
        joint = population * Normal("sigma", 0.0, 1.0)
        assert [list(part.event_spec.components) for part in joint.factors] == [
            ["population"],
            ["sigma"],
        ]


class TestTheFactoredRoutes:
    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_conditioning_on_the_population_is_exact(self, make, renames, field):
        prior = make().with_path_names(renames)
        given = {"population": _POPULATION}
        report = condition_on.check(prior, given)
        assert (report.route, report.exact) == ("slice", True)
        posterior = condition_on(prior, given)
        assert not isinstance(posterior, ApproximateDistribution)
        assert "method" not in posterior.provenance.metadata
        assert list(posterior.event_spec.components) == ["groups"]
        expected = 1.0 if field == "theta" else 0.0
        np.testing.assert_allclose(jnp.asarray(posterior._mean()[field]), expected)

    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_the_factor_view_and_marginal_at_the_population_are_exact(self, make, renames, field):
        prior = make().with_path_names(renames)
        report = marginal.check(prior, "population")
        assert (report.route, report.exact) == ("exact", True)
        detached = marginal(prior, "population")
        value = {"mu": jnp.asarray(1.0), "tau": jnp.asarray(2.0)}
        expected = _mu()._log_prob(1.0) + _tau()._log_prob(2.0)
        np.testing.assert_allclose(detached._log_prob(value), expected, rtol=1e-6)
        view = prior["population"]
        assert isinstance(view, SupportsLogProb)
        np.testing.assert_allclose(view._log_prob(value), expected, rtol=1e-6)
        assert factor(prior, "population").factors[0].label == "mu"

    @pytest.mark.parametrize(("make", "renames", "field"), _MODELS)
    def test_the_marginal_inside_a_sub_joint_is_its_factor(self, make, renames, field):
        prior = make().with_path_names(renames)
        assert marginal.check(prior, "population/mu").exact is True
        np.testing.assert_allclose(
            marginal(prior, "population/mu")._log_prob(0.5), _mu()._log_prob(0.5), rtol=1e-6
        )
        assert isinstance(prior["population/mu"], SupportsLogProb)

    def test_the_groups_node_of_the_non_centered_prior_is_exact(self):
        prior = _non_centered().with_path_names(NON_CENTERED)
        assert marginal.check(prior, "groups").exact is True
        view = prior["groups"]
        assert isinstance(view, SupportsLogProb)
        value = {"theta_tilde": jnp.zeros(J)}
        expected = Normal("theta_tilde", jnp.zeros(J), 1.0)._log_prob(jnp.zeros(J))
        np.testing.assert_allclose(view._log_prob(value), expected, rtol=1e-6)
        np.testing.assert_allclose(marginal(prior, "groups")._mean()["theta_tilde"], 0.0)

    def test_the_groups_node_of_the_centered_prior_conditions_on_the_population(self):
        prior = _centered().with_path_names(CENTERED)
        report = _capability_guard(prior, "_marginal", "groups")
        assert report.feasible is False
        assert "population" in report.description
        assert isinstance(factor(prior, "groups"), FactoredConditionalDistribution)

    def test_the_population_view_of_the_centered_prior_takes_the_sub_joint_mean(self):
        """The centered prior claims no moment, so the view's mean is its marginal's."""
        prior = _centered().with_path_names(CENTERED)
        assert not isinstance(prior, SupportsMean)
        assert isinstance(prior["population/mu"], SupportsMean)
        np.testing.assert_allclose(jnp.asarray(prior["population/mu"]._mean()), 0.0)

    def test_the_scale_of_the_centered_prior_has_an_infinite_mean(self):
        """The half-Cauchy root factor's mean diverges, so the view's mean is ``inf``."""
        prior = _centered()
        assert float(jnp.asarray(mean(prior["tau"]))) == jnp.inf
        renamed = prior.with_path_names(CENTERED)
        assert float(jnp.asarray(mean(renamed["population/tau"]))) == jnp.inf

    def test_sampling_the_renamed_prior_draws_records_of_each_group(self):
        prior = _centered().with_path_names(CENTERED)
        with workflow_run(seed=0):
            draws = sample(prior, sample_shape=(5,))
        assert draws.batch_shape == (5,)
        assert draws.element_spec["groups/theta"].shape == (J,)

    def test_the_draws_follow_the_original_law(self):
        """The renamed prior's draws and the original's agree in their means."""
        original = _non_centered()
        prior = original.with_path_names(NON_CENTERED)
        draws = prior._sample(jax.random.PRNGKey(4), (4000,))
        before = original._sample(jax.random.PRNGKey(5), (4000,))
        np.testing.assert_allclose(
            jnp.mean(draws["population"]["mu"]), jnp.mean(before["mu"]), atol=0.5
        )
        np.testing.assert_allclose(
            jnp.mean(draws["groups"]["theta_tilde"]), jnp.mean(before["theta_tilde"]), atol=0.1
        )


class TestACycle:
    """Gathering ``mu`` and ``theta`` when ``tau`` conditions on ``mu`` and ``theta`` on ``tau``."""

    @staticmethod
    def _chain() -> FactoredDistribution:
        theta = conditional_distribution(
            "theta",
            lambda tau: Normal("theta", 0.0, tau),
            given_spec={"tau": LogNormal("tau", 0.0, 1.0).event_spec.spec},
        )
        tau = conditional_distribution(
            "tau",
            lambda mu: LogNormal("tau", mu, 1.0),
            given_spec={"mu": Normal("mu", 0.0, 1.0).event_spec.spec},
        )
        return theta * tau * Normal("mu", 0.0, 1.0)

    def test_it_keeps_the_boundary_rename(self):
        renamed = self._chain().with_path_names(_CHAIN_RENAMES)
        assert not isinstance(renamed, SupportsFactors)
        assert set(renamed.event_spec.components) == {"g", "tau"}

    def test_it_samples_and_scores_as_the_original(self):
        original = self._chain()
        renamed = original.with_path_names(_CHAIN_RENAMES)
        key = jax.random.PRNGKey(5)
        draw, before = renamed._sample(key), original._sample(key)
        assert jnp.allclose(draw["g"]["theta"], before["theta"])
        assert jnp.allclose(draw["tau"], before["tau"])
        value = {"g": {"mu": 0.3, "theta": -0.2}, "tau": 1.5}
        np.testing.assert_allclose(
            jnp.asarray(log_prob(renamed, value)),
            jnp.asarray(log_prob(original, {"mu": 0.3, "theta": -0.2, "tau": 1.5})),
            rtol=1e-6,
        )
