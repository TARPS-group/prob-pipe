"""Contract tests of the density operations: log-densities, densities, and random log-densities."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import ApplicabilityError, NumericArray, NumericArrayBatch, NumericArraySpec, Record
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.core.constraints import non_negative
from probpipe.distributions._batches import DistributionBatch
from probpipe.distributions._distribution import Distribution
from probpipe.operations import RouteSource
from probpipe.operations._density import (
    log_prob,
    prob,
    random_log_prob,
    random_unnormalized_log_prob,
    unnormalized_log_prob,
    unnormalized_prob,
)

from ._laws import (
    Bare,
    Coin,
    CountedVector,
    Gaussian,
    OneField,
    Polymorphic,
    RandomDensity,
    Unnormalized,
)


class TestLogProb:
    def test_log_prob_returns_the_normalized_log_density(self):
        score = log_prob(Gaussian("g", 1.0, 2.0), 0.5)
        assert isinstance(score, NumericArray)
        assert score.label == "g"
        np.testing.assert_allclose(
            float(jnp.asarray(score)), jax.scipy.stats.norm.logpdf(0.5, 1.0, 2.0), rtol=1e-6
        )

    def test_log_prob_requires_the_normalized_capability(self):
        with pytest.raises(ResolutionError, match="does not claim SupportsLogProb"):
            log_prob(Unnormalized("u"), 0.0)

    def test_unnormalized_log_prob_needs_only_the_unnormalized_capability(self):
        score = unnormalized_log_prob(Unnormalized("u"), 0.0)
        np.testing.assert_allclose(
            float(jnp.asarray(score)), jax.scipy.stats.norm.logpdf(0.0) + np.log(2.0), rtol=1e-6
        )

    def test_a_normalized_density_is_also_an_unnormalized_one(self):
        law = Gaussian("g")
        assert float(jnp.asarray(unnormalized_log_prob(law, 0.3))) == pytest.approx(
            float(jnp.asarray(log_prob(law, 0.3)))
        )

    def test_a_law_with_no_density_raises_resolution_error(self):
        with pytest.raises(ResolutionError):
            log_prob(Bare("b"), 0.0)

    def test_a_discrete_value_scores_at_its_atom(self):
        assert float(jnp.asarray(log_prob(Coin("c", 0.25), 1))) == pytest.approx(np.log(0.25))


class TestTheScoredValue:
    def test_a_value_of_another_shape_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="does not conform"):
            log_prob(Gaussian("g"), jnp.zeros(3))

    def test_a_one_field_record_event_takes_a_record(self):
        score = log_prob(OneField("o"), Record("v", {"x": jnp.float32(0.0)}))
        assert float(jnp.asarray(score)) == pytest.approx(float(jax.scipy.stats.norm.logpdf(0.0)))

    def test_a_one_field_record_event_refuses_a_bare_array(self):
        with pytest.raises(ApplicabilityError, match="does not conform"):
            log_prob(OneField("o"), jnp.float32(0.0))

    def test_a_scored_value_binds_the_symbolic_dimensions_for_that_call_only(self):
        law = Polymorphic("poly")
        three = log_prob(law, jnp.zeros(3))
        five = log_prob(law, jnp.zeros(5))
        np.testing.assert_allclose(float(jnp.asarray(three)), 3 * jax.scipy.stats.norm.logpdf(0.0))
        np.testing.assert_allclose(float(jnp.asarray(five)), 5 * jax.scipy.stats.norm.logpdf(0.0))
        assert law.event_spec.spec.shape == ("n",)

    def test_a_batch_of_values_is_scored_at_its_levels(self):
        values = NumericArrayBatch(
            "values",
            jnp.linspace(-1.0, 1.0, 6).reshape(2, 3),
            ("rows", "cols"),
            element_spec=NumericArraySpec(()),
            axes_per_level=(1, 1),
        )
        scores = log_prob(Gaussian("g"), values)
        assert isinstance(scores, NumericArrayBatch)
        assert scores.level_names == ("rows", "cols")
        assert scores.batch_shape == (2, 3)
        np.testing.assert_allclose(
            np.asarray(scores.values),
            jax.scipy.stats.norm.logpdf(np.linspace(-1.0, 1.0, 6).reshape(2, 3)),
            rtol=1e-6,
        )


class TestABatchOfArraysIsScoredInOneMappedCall:
    @staticmethod
    def _points(n: int) -> NumericArrayBatch:
        return NumericArrayBatch(
            "points", jax.random.normal(jax.random.PRNGKey(n), (n, 3)), "point"
        )

    @pytest.mark.parametrize("score", [log_prob, unnormalized_log_prob])
    def test_the_density_runs_as_often_for_a_thousand_values_as_for_five(self, score):
        law = CountedVector("v")
        points = self._points(1000)
        scores = score(law, points)
        thousand = len(law.calls)
        law.calls.clear()
        score(law, self._points(5))
        assert len(law.calls) == thousand < 5
        assert all(jnp.shape(jnp.asarray(value)) == (3,) for value in law.calls)
        assert (scores.level_names, scores.batch_shape) == (("point",), (1000,))
        np.testing.assert_allclose(
            np.asarray(scores.values),
            np.sum(jax.scipy.stats.norm.logpdf(np.asarray(points.values)), axis=-1),
            rtol=1e-6,
        )

    @pytest.mark.parametrize("score", [log_prob, unnormalized_log_prob])
    def test_the_mapped_scores_equal_the_sequential_ones_at_every_level(self, score):
        points = NumericArrayBatch(
            "grid",
            jax.random.normal(jax.random.PRNGKey(1), (2, 4, 3)),
            ("rows", "cols"),
            element_spec=NumericArraySpec((3,)),
            axes_per_level=(1, 1),
        )
        mapped_law, sequential_law = CountedVector("v"), CountedVector("v")
        mapped = score(mapped_law, points)
        sequential = score.with_options(dispatch="sequential")(sequential_law, points)
        assert len(sequential_law.calls) == 8 > len(mapped_law.calls)
        assert mapped.level_names == sequential.level_names == ("rows", "cols")
        assert mapped.batch_shape == sequential.batch_shape == (2, 4)
        np.testing.assert_allclose(
            np.asarray(mapped.values), np.asarray(sequential.values), rtol=1e-6
        )

    def test_a_batch_of_values_of_another_shape_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="does not conform"):
            log_prob(CountedVector("v"), NumericArrayBatch("points", jnp.zeros((4, 2)), "point"))


class TestLiftedScores:
    def test_a_law_at_the_value_is_admitted_and_planned_at_its_event_kind(self):
        report = log_prob.check(Gaussian("g"), Gaussian("v"))
        assert (report.feasible, report.route, report.exact) == (True, "exact", True)
        assert report.lifted == ("value",)
        assert report.result == OutputSpec(log_prob=NumericArraySpec(()))
        assert isinstance(log_prob(Gaussian("g"), Gaussian("v")), Distribution)

    def test_a_law_whose_draws_do_not_conform_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="does not conform"):
            log_prob.check(Gaussian("g"), OneField("v"))

    def test_a_swept_batch_of_laws_and_a_law_at_the_value_lift_together(self):
        laws = DistributionBatch("laws", [Gaussian("g", 1.0), Gaussian("g", 2.0)], "laws")
        report = log_prob.check(laws, Gaussian("v"))
        assert report.feasible is True
        assert report.lifted == ("d", "value")

    def test_the_identity_of_prob_is_checked_at_the_points_of_a_lift(self):
        report = prob.check(Gaussian("g"), Gaussian("v"))
        assert (report.feasible, report.route, report.lifted) == (
            True,
            "identity",
            ("value",),
        )
        assert prob.check(Bare("b"), Gaussian("v")).feasible is False


class TestDerivedDensities:
    def test_prob_is_derived_from_log_prob(self):
        assert prob.is_derived
        assert prob.identity == "jnp.exp(log_prob.with_options(raw=True)(d, value))"
        (route,) = prob.routes
        assert (route.name, route.source) == ("identity", RouteSource.FALLBACK)

    def test_prob_is_the_exponential_of_log_prob(self):
        law = Gaussian("g", 0.0, 1.5)
        density = prob(law, 0.7)
        assert density.label == "g"
        assert density.spec.support is non_negative
        assert float(jnp.asarray(density)) == pytest.approx(
            float(np.exp(jax.scipy.stats.norm.logpdf(0.7, 0.0, 1.5))), rel=1e-6
        )

    def test_unnormalized_prob_is_the_exponential_of_unnormalized_log_prob(self):
        assert unnormalized_prob.is_derived
        assert float(jnp.asarray(unnormalized_prob(Unnormalized("u"), 0.0))) == pytest.approx(
            2.0 * float(np.exp(jax.scipy.stats.norm.logpdf(0.0))), rel=1e-6
        )

    def test_prob_of_a_law_without_a_normalized_density_raises_resolution_error(self):
        with pytest.raises(ResolutionError):
            prob(Unnormalized("u"), 0.0)

    def test_the_identity_is_infeasible_where_log_prob_has_no_route(self):
        report = prob.check(Bare("b"), 0.0)
        assert report.feasible is False
        assert "does not claim SupportsLogProb" in report.description
        with pytest.raises(ResolutionError, match="does not claim SupportsLogProb"):
            prob(Bare("b"), 0.0)

    def test_the_unnormalized_identity_is_infeasible_where_its_constituent_has_no_route(self):
        report = unnormalized_prob.check(Bare("b"), 0.0)
        assert report.feasible is False
        assert "does not claim SupportsUnnormalizedLogProb" in report.description

    def test_the_identity_takes_the_exactness_of_the_route_log_prob_selects(self):
        report = prob.check(Gaussian("g"), 0.5)
        assert (report.feasible, report.route, report.exact) == (True, "identity", True)
        assert {info.method_name: info for info in report.routes}["identity"].exact is True

    def test_the_identity_route_states_its_constituent_as_its_condition(self):
        (route,) = prob.summary().routes
        assert route.condition == "``log_prob`` has a route for the law and the value."


class TestRandomLogDensities:
    def test_random_log_prob_returns_the_random_function_as_a_law(self):
        law = random_log_prob(RandomDensity("m"))
        assert isinstance(law, Distribution)
        assert law.loc == -1.0

    def test_random_unnormalized_log_prob_returns_its_own_random_function(self):
        assert random_unnormalized_log_prob(RandomDensity("m")).loc == -2.0

    def test_neither_takes_a_value(self):
        assert list(inspect.signature(random_log_prob).parameters) == ["M"]
        assert list(inspect.signature(random_unnormalized_log_prob).parameters) == ["M"]

    def test_a_law_without_a_random_density_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="SupportsRandomLogProb"):
            random_log_prob(Gaussian("g"))
