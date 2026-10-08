"""Factored distributions: joints that carry the ordered list of their factors.

A factored joint holds its factors in conditional-first order and derives the
rest from them. Its event declaration is an exposed record over the factors'
components; sampling and the densities are the intersection of the factors'
capabilities; the moments are present exactly when an edge-free joint's factors
all have them; and the marginal is exact per path, by the factor graph. A
conditional joint curries by binding its givens in every factor that names them.
"""

from __future__ import annotations

import copy
import pickle
import re
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from probpipe import (
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    ResolutionError,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from probpipe.core._dispatch import Feasibility
from probpipe.distributions import (
    ConditionalDistribution,
    ConditionalNumericDistribution,
    Distribution,
    FactoredConditionalDistribution,
    FactoredConditionalNumericDistribution,
    FactoredDistribution,
    FactoredFullyNumericConditionalDistribution,
    FactoredNumericConditionalDistribution,
    FactoredNumericDistribution,
    FieldView,
    NumericConditionalDistribution,
    NumericDistribution,
    SupportsConditionalCovariance,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
    SupportsFactors,
    SupportsMarginals,
)
from probpipe.distributions._capabilities import _capability_guard
from probpipe.distributions._distribution import _fixed_paths
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.distributions._factored import _is_named, _SoleField
from probpipe.linalg import DenseLinOp

SCALAR = NumericArraySpec(())
SYMBOLIC = NumericArraySpec(("n",))

#: The capabilities a joint may claim, for comparing two joints' claims.
_CAPABILITIES = (
    SupportsSampling,
    SupportsLogProb,
    SupportsUnnormalizedLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsExpectation,
    SupportsMarginals,
    SupportsConditionalSampling,
    SupportsConditionalLogProb,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalMean,
    SupportsConditionalVariance,
    SupportsConditionalCovariance,
    SupportsConditionalMarginals,
)

_CONDITIONAL_MARKERS = (
    FactoredConditionalNumericDistribution,
    FactoredNumericConditionalDistribution,
    FactoredFullyNumericConditionalDistribution,
)


# -- Kernels ------------------------------------------------------------------------


def _total(values: Mapping[str, Any]) -> Any:
    """The default location of a kernel: the sum of its given values."""
    return sum(jnp.asarray(value) for value in values.values())


def _exp_x(values: Mapping[str, Any]) -> Any:
    """The location ``exp(x)`` of the kernel ``y | x``."""
    return jnp.exp(jnp.asarray(values["x"]))


class NormalKernel(ConditionalDistribution):
    """``K(given, ·) = Normal(loc(given), scale)``, a law over the kernel's own whole-term event.

    Binding some of the given slots curries to the kernel over the rest, which
    keeps the values bound so far.
    """

    def __init__(
        self,
        label: str,
        given_spec: Mapping[str, Any],
        event_spec: OutputSpec,
        *,
        loc: Callable[[Mapping[str, Any]], Any] = _total,
        scale: float = 1.0,
        bound: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(label, given_spec, event_spec)
        self._loc = loc
        self._scale = scale
        self._bound = dict(bound or {})

    def _location(self, given: Any) -> Any:
        return jnp.asarray(self._loc({**self._bound, **dict(given.items())}))

    def _condition_on(self, given, /, **options):
        values = {**self._bound, **dict(given.items())}
        rest = {slot: spec for slot, spec in self.given_spec.items() if slot not in values}
        if rest:
            return type(self)(
                self.label, rest, self.event_spec, loc=self._loc, scale=self._scale, bound=values
            )
        (component,) = self.event_spec.components
        return Normal(
            self.label, self._loc(values), self._scale, event_spec=OutputSpec(**{component: None})
        )


class SamplingKernel(NormalKernel, SupportsConditionalSampling):
    """A normal kernel that draws from ``K(given, ·)``."""

    def _conditional_sample(self, given, key, sample_shape=()):
        loc = self._location(given)
        return loc + self._scale * jax.random.normal(key, (*sample_shape, *loc.shape))


class ScoringKernel(NormalKernel, SupportsConditionalLogProb):
    """A normal kernel with the normalized log-density of ``K(given, ·)``."""

    def _conditional_log_prob(self, given, value):
        return norm.logpdf(value, self._location(given), self._scale)


class NormKernel(NormalKernel, SupportsConditionalLogProb):
    """``y | beta ~ Normal(‖beta‖, sigma)``, whose location reduces over every axis of ``beta``.

    The scale is the given ``sigma`` when the kernel has that slot, and one
    otherwise. Called once on a batch of betas it would take the norm of the
    whole batch, so it scores a batch correctly only when called value by value.
    """

    def _conditional_log_prob(self, given, value):
        scale = given["sigma"] if "sigma" in self.given_spec else 1.0
        return norm.logpdf(value, jnp.linalg.norm(jnp.asarray(given["beta"])), scale)


class UnnormalizedKernel(NormalKernel, SupportsConditionalUnnormalizedLogProb):
    """A normal kernel with the log-density of ``K(given, ·)`` up to an additive constant."""

    def _conditional_unnormalized_log_prob(self, given, value):
        return -0.5 * ((value - self._location(given)) / self._scale) ** 2


class MomentKernel(NormalKernel, SupportsConditionalMean, SupportsConditionalVariance):
    """A normal kernel with the mean and variance of ``K(given, ·)``."""

    def _conditional_mean(self, given):
        return self._location(given)

    def _conditional_variance(self, given):
        return jnp.asarray(self._scale**2)


class FullKernel(SamplingKernel, ScoringKernel, MomentKernel):
    """A normal kernel that samples, scores, and has its mean and variance."""


class RenamingKernel(NormalKernel):
    """A kernel whose bound law declares a component other than the kernel's own."""

    def _condition_on(self, given, /, **kwargs):
        return Normal(self.label, 0.0, 1.0, event_spec=OutputSpec(elsewhere=None))


class MarginalKernel(NormalKernel, SupportsConditionalMarginals):
    """A kernel over an exposed record whose conditional marginal is exact at every path."""

    def _condition_on(self, given, /, **kwargs):
        return Law(self.name, self.event_spec)

    def _conditional_marginal(self, given, path):
        return Law(path, OutputSpec(**{path: SCALAR}))


# -- Laws ---------------------------------------------------------------------------


class Law(Distribution):
    """A law over its declared event that claims no capability."""


class MeanLaw(Law, SupportsMean):
    """A point mass at zero with a mean and no other moment."""

    def _mean(self):
        return jnp.zeros(())


class MomentLaw(
    Law, SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile, SupportsExpectation
):
    """A point mass at zero with every moment capability, its expectation exact."""

    def _mean(self):
        return jnp.zeros(())

    def _variance(self):
        return jnp.zeros(())

    def _cov(self):
        return DenseLinOp(jnp.zeros((1, 1)))

    def _quantile(self, q):
        return jnp.zeros(jnp.shape(q))

    def _expectation(self, f):
        return f(jnp.zeros(()))


class MarginalLaw(Law, SupportsMarginals):
    """A law whose marginal is exact at the paths it lists, and rejected at any other.

    It records each path it marginalizes, and the marginal is a law over the
    node under a component named by the path's final segment.
    """

    def __init__(self, label: str, event_spec: OutputSpec, *, exact: tuple[str, ...] = ()) -> None:
        super().__init__(label, event_spec)
        self.exact = frozenset(exact)
        self.marginalized: list[Any] = []

    def _marginal(self, path):
        self.marginalized.append(path)
        return Law(path, OutputSpec(**{path.rsplit("/", 1)[-1]: SCALAR}))

    def _marginal_guard(self, path):
        if path in self.exact:
            return Feasibility(True)
        return Feasibility(False, f"{self.label!r} has no exact marginal at {path!r}")


class TotalMarginalLaw(Law, SupportsMarginals):
    """A law whose marginal is exact at every path, so it defines no guard."""

    def _marginal(self, path):
        return Law(path, OutputSpec(**{path: SCALAR}))


class RecordingLaw(Law, SupportsLogProb):
    """A law whose log-density records each value it scores and returns zero."""

    def __init__(self, label: str, event_spec: OutputSpec) -> None:
        super().__init__(label, event_spec)
        self.scored: list[Any] = []

    def _log_prob(self, value):
        self.scored.append(value)
        return jnp.zeros(())


class PointLaw(Law, SupportsSampling):
    """A point mass at a fixed raw draw, which every sample returns."""

    def __init__(self, label: str, event_spec: OutputSpec, draw: Any) -> None:
        super().__init__(label, event_spec)
        self.draw = draw

    def _sample(self, key, sample_shape=()):
        return self.draw


class RejectingPointLaw(PointLaw):
    """A point mass whose sampling guard rejects."""

    def _sample_guard(self) -> bool:
        """Draws only from a finite point."""
        return False


class PairMomentLaw(Law, SupportsMean, SupportsVariance):
    """A law over the record ``{a, b}`` with mean ``{a: 1, b: 2}`` and variance ``{a: 3, b: 4}``."""

    def _mean(self):
        return {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)}

    def _variance(self):
        return {"a": jnp.asarray(3.0), "b": jnp.asarray(4.0)}


class ShiftedQuantileLaw(Law, SupportsQuantile):
    """A law whose quantile at each level ``q`` is ``q`` plus its shift."""

    def __init__(self, label: str, event_spec: OutputSpec, shift: float) -> None:
        super().__init__(label, event_spec)
        self.shift = shift

    def _quantile(self, q):
        return jnp.asarray(q) + self.shift


class OneFieldNormal(Law, SupportsSampling, SupportsLogProb):
    """A normal law of scale two over the one-field record ``{beta}``."""

    def _sample(self, key, sample_shape=()):
        return {"beta": 2.0 * jax.random.normal(key, sample_shape)}

    def _log_prob(self, value):
        return norm.logpdf(value["beta"], 0.0, 2.0)


#: The labels of :class:`LabelLaw`, whose lengths locate :class:`LengthKernel`.
_LABELS = ("a", "bb", "ccc")


class LabelLaw(Law, SupportsSampling, SupportsLogProb):
    """A uniform law over three labels, an opaque event, drawn as an object array in a batch."""

    def _sample(self, key, sample_shape=()):
        choices = np.asarray(jax.random.randint(key, sample_shape, 0, len(_LABELS)))
        if not sample_shape:
            return _LABELS[int(choices)]
        labels = np.empty(choices.shape, dtype=object)
        for position, choice in np.ndenumerate(choices):
            labels[position] = _LABELS[choice]
        return labels

    def _log_prob(self, value):
        return jnp.full(np.shape(value), -jnp.log(3.0))


class LengthKernel(NormalKernel, SupportsConditionalSampling, SupportsConditionalLogProb):
    """``y | label ~ Normal(len(label), 1)``, which reads one label per call.

    A draw is the location itself, so a draw shows which label it was given.
    """

    def _conditional_sample(self, given, key, sample_shape=()):
        return jnp.full(sample_shape, float(len(given["label"])))

    def _conditional_log_prob(self, given, value):
        return norm.logpdf(value, float(len(given["label"])), 1.0)


def _labeled_joint() -> FactoredDistribution:
    """``y | label`` composed with the law of ``label``, an opaque component."""
    kernel = LengthKernel("lik", {"label": OpaqueSpec()}, OutputSpec(y=SCALAR))
    return kernel * LabelLaw("labels", OutputSpec(label=OpaqueSpec()))


class UndecidedSamplingKernel(SamplingKernel):
    """A sampling kernel whose guard cannot decide before its values are known."""

    def _conditional_sample_guard(self) -> None:
        """Draws once its scale is known."""
        return None


class NumericJoint(FactoredNumericDistribution):
    """A joint class that claims a numeric event for every instance."""


# -- Constructions ------------------------------------------------------------------


def _likelihood(kernel: type[NormalKernel] = NormalKernel, **options: Any) -> NormalKernel:
    """``y | beta``: a kernel labeled ``lik`` that conditions on ``beta`` and produces ``y``."""
    return kernel("lik", {"beta": SCALAR}, OutputSpec(y=SCALAR), **options)


def _prior(scale: float = 1.0) -> Normal:
    """A normal law labeled ``prior`` whose component is ``beta``."""
    return Normal("prior", 0.0, scale, event_spec=OutputSpec(beta=None))


def _law(label: str, component: str, spec: Any = SCALAR, law: type[Law] = Law) -> Law:
    """A law labeled *label* whose whole-term component is *component*."""
    return law(label, OutputSpec(**{component: spec}))


def _pair(label: str = "pair", law: type[Law] = Law, **options: Any) -> Law:
    """A law over an exposed record of the fields ``a`` and ``b``."""
    return law(label, OutputSpec(RecordSpec(a=SCALAR, b=SCALAR)), **options)


def _exponential_location_model() -> FactoredDistribution:
    """``x ~ Normal(0, 1)`` and ``y | x ~ Normal(exp(x), 1)``, whose ``E[y]`` is ``e^{1/2}``."""
    kernel = FullKernel("y_given_x", {"x": SCALAR}, OutputSpec(y=SCALAR), loc=_exp_x)
    return kernel * Normal("x", 0.0, 1.0)


def _claimed(term: Any) -> set[type]:
    """The capabilities a joint may claim that *term* claims."""
    return {capability for capability in _CAPABILITIES if isinstance(term, capability)}


def _mentions(*fragments: str) -> str:
    """A pattern matching a message that contains every fragment, in any order."""
    return "".join(f"(?=.*{re.escape(fragment)})" for fragment in fragments)


# -- Construction -------------------------------------------------------------------


class TestConstruction:
    """A factored joint is built from an ordered list of factors under the composition rules."""

    def test_same_named_opaque_givens_unify_to_the_known_type(self):
        first = NormalKernel("first", {"s": OpaqueSpec()}, OutputSpec(y=SCALAR))
        second = NormalKernel("second", {"s": OpaqueSpec(type=str)}, OutputSpec(z=SCALAR))
        assert (first * second).given_spec["s"] == OpaqueSpec(type=str)
        with pytest.raises(ValueError, match="does not conform"):
            _ = (
                first
                * NormalKernel("third", {"s": OpaqueSpec(type=int)}, OutputSpec(w=SCALAR))
                * (second)
            )

    def test_the_joint_holds_its_factors_in_order(self):
        lik, prior = _likelihood(), _prior()
        joint = FactoredDistribution("model", [lik, prior])
        assert joint.label == "model"
        assert joint.factors == (lik, prior)

    def test_a_conditional_joint_holds_its_factors_in_order(self):
        lik, other = _likelihood(), _law("other", "c")
        joint = FactoredConditionalDistribution("model", [lik, other])
        assert joint.factors == (lik, other)
        assert list(joint.given_spec) == ["beta"]

    def test_composition_builds_the_joint_the_constructor_builds(self):
        lik, prior = _likelihood(), _prior()
        composed, constructed = lik * prior, FactoredDistribution("lik·prior", [lik, prior])
        assert type(composed) is type(constructed)
        assert (composed.label, composed.spec, composed.factors) == (
            constructed.label,
            constructed.spec,
            constructed.factors,
        )

    def test_the_factors_are_a_tuple(self):
        assert isinstance(FactoredDistribution("model", [_likelihood(), _prior()]).factors, tuple)

    @pytest.mark.parametrize("factor", [1.0, "beta", OutputSpec(beta=SCALAR)])
    def test_a_factor_is_a_distribution_or_a_kernel(self, factor):
        with pytest.raises(
            TypeError, match=_mentions("ConditionalDistribution", type(factor).__name__)
        ):
            FactoredDistribution("model", [_prior(), factor])

    def test_an_unmet_given_makes_the_joint_conditional(self):
        with pytest.raises(
            ValueError, match=_mentions("'model'", "'beta'", "FactoredConditionalDistribution")
        ):
            FactoredDistribution("model", [_likelihood()])

    def test_a_conditional_joint_leaves_a_given_unmet(self):
        with pytest.raises(ValueError, match=_mentions("'model'", "FactoredDistribution")):
            FactoredConditionalDistribution("model", [_likelihood(), _prior()])

    def test_the_constructor_applies_the_composition_rules(self):
        with pytest.raises(
            ValueError, match=_mentions("'lik'", "'beta'", "to its left in the product")
        ):
            FactoredDistribution("model", [_prior(), _likelihood()])
        with pytest.raises(ValueError, match=_mentions("'beta'", "'prior'")):
            FactoredDistribution("model", [_prior(), _prior()])


# -- The event declaration --------------------------------------------------------


class TestEventDeclaration:
    """An exposed record over the factors' components, in factor order, each factor's order kept."""

    def test_the_event_is_an_exposed_record_of_every_factor_component(self):
        joint = _likelihood() * _prior() * _law("other", "c")
        assert joint.event_spec.exposes_record
        assert isinstance(joint.event_spec.spec, RecordSpec)
        assert list(joint.event_spec.components) == ["y", "beta", "c"]

    def test_each_component_keeps_its_factor_spec(self):
        lik, prior = _likelihood(), _prior()
        components = (lik * prior).event_spec.components
        assert components["y"] == lik.event_spec.components["y"]
        assert components["beta"] == prior.event_spec.components["beta"]

    def test_each_factor_keeps_its_component_order(self):
        unsorted = Law("unsorted", OutputSpec(RecordSpec(zeta=SCALAR, alpha=SCALAR)))
        joint = _law("m", "m") * unsorted * _law("a", "a")
        assert list(joint.event_spec.components) == ["m", "zeta", "alpha", "a"]

    def test_a_single_whole_term_factor_is_exposed_as_a_record(self):
        joint = FactoredDistribution("model", [Normal("a", 0.0, 1.0)])
        assert joint.event_spec.exposes_record
        assert list(joint.event_spec.components) == ["a"]

    def test_a_whole_record_factor_contributes_one_component(self):
        params = RecordSpec(u=SCALAR, v=SCALAR)
        joint = _law("a", "a") * _law("p", "params", params)
        assert list(joint.event_spec.components) == ["a", "params"]
        assert joint.event_spec.components["params"] == params

    def test_the_joint_keeps_each_factor_packaging(self):
        exposed = Law("exposed", OutputSpec(RecordSpec(beta=SCALAR)))
        whole = Law("whole", OutputSpec(beta=SCALAR))
        via_exposed, via_whole = _likelihood() * exposed, _likelihood() * whole
        assert via_exposed.event_spec == via_whole.event_spec
        assert via_exposed.factors[1].event_spec.exposes_record
        assert not via_whole.factors[1].event_spec.exposes_record

    def test_a_factor_that_produces_nothing_is_kept_among_the_factors(self):
        potential = Law("potential", OutputSpec(RecordSpec({})))
        joint = _likelihood() * _prior() * potential
        assert joint.factors[-1] is potential
        assert list(joint.event_spec.components) == ["y", "beta"]

    def test_the_numeric_views_read_each_component_of_the_joint(self):
        from probpipe import Gamma, positive, real

        joint = Normal("a", 0.0, 1.0) * Gamma("g", 2.0, 1.0)
        assert isinstance(joint, NumericDistribution)
        assert joint.supports == {"a": real, "g": positive}
        assert joint.support is None
        assert set(joint.dtypes) == {"a", "g"}


# -- Factors ------------------------------------------------------------------------


class TestFactors:
    """Both factored kinds carry ``SupportsFactors`` and list their factors conditional-first."""

    def test_both_factored_kinds_support_factors(self):
        assert isinstance(_likelihood() * _prior(), SupportsFactors)
        assert isinstance(_likelihood() * _law("other", "c"), SupportsFactors)

    def test_an_ordinary_law_or_kernel_does_not_support_factors(self):
        assert not isinstance(_prior(), SupportsFactors)
        assert not isinstance(_likelihood(), SupportsFactors)

    def test_a_consumer_precedes_its_producer(self):
        lik, prior = _likelihood(), _prior()
        assert (lik * prior).factors == (lik, prior)

    def test_a_conditional_joint_lists_its_factors(self):
        lik, other = _likelihood(), _law("other", "c")
        assert (lik * other).factors == (lik, other)


# -- Sampling and density capabilities --------------------------------------------


class TestSamplingAndDensityCapabilities:
    """Sampling and the densities intersect the factors'; a kernel claims them through its twin."""

    @pytest.mark.parametrize(
        ("kernel", "samples"),
        [
            pytest.param(NormalKernel, False, id="no-capability"),
            pytest.param(ScoringKernel, False, id="scores"),
            pytest.param(SamplingKernel, True, id="samples"),
            pytest.param(FullKernel, True, id="samples-and-scores"),
        ],
    )
    def test_the_joint_samples_exactly_when_every_factor_samples(self, kernel, samples):
        assert isinstance(_likelihood(kernel) * _prior(), SupportsSampling) is samples

    def test_a_law_that_does_not_sample_leaves_the_joint_without_sampling(self):
        joint = _likelihood(SamplingKernel) * _law("prior", "beta")
        assert not isinstance(joint, SupportsSampling)

    @pytest.mark.parametrize(
        ("kernel", "normalized", "unnormalized"),
        [
            pytest.param(NormalKernel, False, False, id="no-density"),
            pytest.param(UnnormalizedKernel, False, True, id="unnormalized"),
            pytest.param(ScoringKernel, True, True, id="normalized"),
        ],
    )
    def test_the_joint_density_is_the_weakest_factor_density(
        self, kernel, normalized, unnormalized
    ):
        joint = _likelihood(kernel) * _prior()
        assert isinstance(joint, SupportsLogProb) is normalized
        assert isinstance(joint, SupportsUnnormalizedLogProb) is unnormalized

    def test_a_law_without_a_density_leaves_the_joint_without_one(self):
        joint = _likelihood(ScoringKernel) * _law("prior", "beta")
        assert not isinstance(joint, SupportsUnnormalizedLogProb)

    def test_a_conditional_joint_claims_the_conditional_twins(self):
        joint = _likelihood(FullKernel) * Normal("c", 0.0, 1.0)
        assert isinstance(joint, SupportsConditionalSampling)
        assert isinstance(joint, SupportsConditionalLogProb)
        assert not isinstance(joint, SupportsSampling)
        assert not isinstance(joint, SupportsUnnormalizedLogProb)

    def test_a_conditional_joint_samples_only_when_every_factor_samples(self):
        joint = _likelihood(ScoringKernel) * Normal("c", 0.0, 1.0)
        assert not isinstance(joint, SupportsConditionalSampling)
        assert isinstance(joint, SupportsConditionalLogProb)

    def test_an_unnormalized_factor_leaves_a_conditional_joint_unnormalized(self):
        joint = _likelihood(UnnormalizedKernel) * Normal("c", 0.0, 1.0)
        assert isinstance(joint, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(joint, SupportsConditionalLogProb)


# -- Moment capabilities ----------------------------------------------------------


_MOMENTS = (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)


class TestMomentCapabilities:
    """Edge-free joints have a moment exactly when every factor does; dependent joints have none."""

    def test_an_edge_free_joint_has_every_moment_its_factors_share(self):
        joint = _law("a", "a", law=MomentLaw) * _law("b", "b", law=MomentLaw)
        assert all(isinstance(joint, moment) for moment in _MOMENTS)

    def test_a_moment_one_factor_lacks_is_absent(self):
        joint = _law("a", "a", law=MomentLaw) * _law("b", "b", law=MeanLaw)
        assert isinstance(joint, SupportsMean)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS[1:])

    def test_a_dependent_joint_has_no_moment(self):
        joint = _likelihood(MomentKernel) * _law("prior", "beta", law=MomentLaw)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS)

    def test_the_exponential_location_model_has_no_moment(self):
        # E[y] = E[exp(x)] = e^{1/2}, which the factors' means, 0 and exp(x), do not give.
        joint = _exponential_location_model()
        assert isinstance(joint, SupportsSampling)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS)

    def test_an_edge_free_conditional_joint_has_the_moment_twins_its_factors_share(self):
        joint = MomentKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) * _law(
            "b", "b", law=MomentLaw
        )
        assert isinstance(joint, SupportsConditionalMean)
        assert isinstance(joint, SupportsConditionalVariance)
        assert not isinstance(joint, SupportsConditionalCovariance)
        assert not isinstance(joint, SupportsMean)

    def test_a_dependent_conditional_joint_has_no_moment_twin(self):
        kernel = MomentKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = kernel * _law("prior", "beta", law=MomentLaw)
        assert isinstance(joint, FactoredConditionalDistribution)
        assert not isinstance(joint, SupportsConditionalMean)
        assert not isinstance(joint, SupportsConditionalVariance)

    def test_the_joint_claims_no_exact_expectation(self):
        joint = _law("a", "a", law=MomentLaw) * _law("b", "b", law=MomentLaw)
        assert not isinstance(joint, SupportsExpectation)

    def test_an_edge_free_mean_is_the_record_of_the_factor_means(self):
        mean = (Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0))._mean()
        assert list(mean.keys()) == ["a", "b"]
        assert jnp.allclose(mean["a"], 0.0)
        assert jnp.allclose(mean["b"], 1.0)

    def test_an_edge_free_variance_is_the_record_of_the_factor_variances(self):
        variance = (Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0))._variance()
        assert list(variance.keys()) == ["a", "b"]
        assert jnp.allclose(variance["a"], 1.0)
        assert jnp.allclose(variance["b"], 4.0)

    def test_an_edge_free_covariance_is_block_diagonal_over_the_flat_event(self):
        cov = (Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0))._cov()
        assert jnp.allclose(cov.to_dense(), jnp.diag(jnp.array([1.0, 4.0])))

    def test_the_exponential_location_mean_is_not_a_composition_of_factor_means(self):
        draws = _exponential_location_model()._sample(jax.random.PRNGKey(0), (20_000,))
        mean = float(jnp.mean(jnp.asarray(draws["y"])))
        # exp(E[x]) = 1, while E[y] = e^{1/2}; sd(y) is about 2.4, a standard error of 0.017.
        assert abs(mean - float(jnp.exp(0.5))) < 0.1

    def test_each_factor_covariance_is_a_block_in_factor_order(self):
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        vector = MultivariateNormal("m", loc=jnp.zeros(2), cov=cov)
        joint_cov = (Normal("s", 0.0, 3.0) * vector)._cov()
        assert isinstance(joint_cov, DenseLinOp)
        expected = jnp.zeros((3, 3)).at[0, 0].set(9.0).at[1:, 1:].set(cov)
        assert jnp.allclose(joint_cov.to_dense(), expected)

    def test_a_record_factor_contributes_each_of_its_components(self):
        joint = PairMomentLaw("pair", OutputSpec(RecordSpec(a=SCALAR, b=SCALAR))) * Normal(
            "c", 5.0, 1.0
        )
        mean, variance = joint._mean(), joint._variance()
        assert list(mean) == list(variance) == ["a", "b", "c"]
        assert [float(mean[c]) for c in mean] == [1.0, 2.0, 5.0]
        assert [float(variance[c]) for c in variance] == [3.0, 4.0, 1.0]

    def test_an_edge_free_quantile_is_assembled_per_component(self):
        first = ShiftedQuantileLaw("first", OutputSpec(a=SCALAR), 1.0)
        second = ShiftedQuantileLaw("second", OutputSpec(b=SCALAR), 10.0)
        levels = jnp.array([0.25, 0.5])
        quantiles = (first * second)._quantile(levels)
        assert list(quantiles) == ["a", "b"]
        assert jnp.allclose(quantiles["a"], levels + 1.0)
        assert jnp.allclose(quantiles["b"], levels + 10.0)

    def test_the_quantiles_of_a_joint_of_empirical_laws_keep_each_leaf(self):
        atoms = NumericRecordBatch(
            "rows",
            {"b": jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), "a": jnp.array([1.0, 2.0, 3.0])},
            "row",
            element_spec=RecordSpec(b=(2,), a=()),
        )
        record = EmpiricalDistribution("post", atoms, jnp.array([0.5, 0.25, 0.25]))
        theta = EmpiricalDistribution("theta", jnp.array([4.0, 5.0, 6.0]))
        levels = jnp.array([0.25, 0.75])
        quantiles = FactoredDistribution("j", [record, theta])._quantile(levels)
        assert list(quantiles) == ["b", "a", "theta"]
        assert quantiles["b"].shape == (2, 2)
        expected = record._quantile(levels)
        assert jnp.allclose(quantiles["b"], expected["b"])
        assert jnp.allclose(quantiles["a"], expected["a"])
        assert jnp.allclose(quantiles["theta"], theta._quantile(levels))

    def test_the_law_of_the_one_field_of_an_empirical_record_has_its_quantiles(self):
        column = jnp.array([3.0, 1.0, 2.0, 4.0])
        atoms = NumericRecordBatch("rows", {"x": column}, "row", element_spec=RecordSpec(x=()))
        field = _SoleField(EmpiricalDistribution("one", atoms))
        levels = jnp.array([0.25, 0.5])
        expected = EmpiricalDistribution("x", column)._quantile(levels)
        assert jnp.allclose(field._quantile(levels), expected)
        assert jnp.shape(field._quantile(0.5)) == ()

    def test_an_edge_free_conditional_joint_has_its_moments_at_its_given(self):
        joint = MomentKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) * Normal("b", 1.0, 2.0)
        mean = joint._conditional_mean({"x": 0.5})
        variance = joint._conditional_variance({"x": 0.5})
        assert list(mean) == list(variance) == ["a", "b"]
        assert [float(mean[c]) for c in mean] == [0.5, 1.0]
        assert [float(variance[c]) for c in variance] == [1.0, 4.0]


# -- Marginals ----------------------------------------------------------------------


class TestMarginalCapability:
    """The unconditional joint claims marginals and resolves them per path."""

    def test_every_unconditional_joint_claims_marginals(self):
        assert isinstance(_likelihood() * _prior(), SupportsMarginals)
        assert isinstance(_law("a", "a") * _law("b", "b"), SupportsMarginals)

    def test_a_conditional_joint_claims_no_marginals(self):
        joint = _likelihood() * _law("other", "c")
        assert not isinstance(joint, SupportsConditionalMarginals)
        assert not isinstance(joint, SupportsMarginals)


class TestMarginalGuard:
    """``_marginal_guard(path)`` reports whether the marginal at the path is exact."""

    def test_a_root_factor_is_exact(self):
        assert (_likelihood() * _prior())._marginal_guard("beta").feasible is True

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("a", id="a"),
            pytest.param("c", id="c"),
            pytest.param(("a", "b"), id="a+b"),
            pytest.param(("a", "c"), id="a+c"),
            pytest.param(("a", "b", "c"), id="a+b+c"),
        ],
    )
    def test_every_group_of_whole_factors_of_an_edge_free_joint_is_exact(self, path):
        joint = _law("u", "a") * _law("v", "b") * _law("w", "c")
        assert joint._marginal_guard(path).feasible is True

    def test_an_ancestrally_closed_group_is_exact(self):
        joint = _likelihood() * _prior() * _law("other", "c")
        assert joint._marginal_guard(("y", "beta")).feasible is True

    def test_a_component_whose_ancestor_is_another_factor_is_rejected(self):
        report = (_likelihood() * _prior())._marginal_guard("y")
        assert report.feasible is False
        assert report.description

    def test_a_group_that_is_not_ancestrally_closed_is_rejected(self):
        joint = _likelihood() * _prior() * _law("other", "c")
        assert joint._marginal_guard(("y", "c")).feasible is False

    def test_a_path_inside_one_factor_follows_that_factor_guard(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        joint = pair * _law("other", "c")
        assert joint._marginal_guard("a") == pair._marginal_guard("a") == Feasibility(True)
        assert joint._marginal_guard("b") == pair._marginal_guard("b")
        assert joint._marginal_guard("b").feasible is False

    def test_a_path_inside_a_whole_record_follows_that_factor_guard(self):
        params = MarginalLaw(
            "p", OutputSpec(params=RecordSpec(u=SCALAR, v=SCALAR)), exact=("params/u",)
        )
        joint = _law("other", "c") * params
        assert joint._marginal_guard("params").feasible is True
        assert joint._marginal_guard("params/u").feasible is True
        assert joint._marginal_guard("params/v").feasible is False

    def test_a_root_factor_that_a_kernel_consumes_is_still_delegated(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        joint = NormalKernel("k", {"b": SCALAR}, OutputSpec(y=SCALAR)) * pair
        assert joint._marginal_guard("a").feasible is True

    def test_a_factor_without_a_guard_is_exact_at_every_path(self):
        joint = _pair(law=TotalMarginalLaw) * _law("other", "c")
        assert joint._marginal_guard("a").feasible is True

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("gamma", id="unknown-component"),
            pytest.param("a/zzz", id="below-a-leaf"),
            pytest.param(("a", "gamma"), id="in-a-selection"),
        ],
    )
    def test_a_path_that_is_not_an_event_path_is_rejected_with_a_reason(self, path):
        joint = _pair(law=TotalMarginalLaw) * _law("other", "c")
        report = joint._marginal_guard(path)
        assert report.feasible is False
        assert "not an event path" in report.description

    def test_a_path_inside_a_factor_without_marginals_is_rejected(self):
        report = (_pair() * _law("other", "c"))._marginal_guard("a")
        assert report.feasible is False
        assert report.description

    def test_a_target_spanning_independent_factors_follows_each_factor_guard(self):
        left = _pair("left", law=MarginalLaw, exact=("a",))
        right = MarginalLaw("right", OutputSpec(RecordSpec(c=SCALAR, d=SCALAR)), exact=("c",))
        joint = left * right
        assert joint._marginal_guard(("a", "c")).feasible is True
        assert joint._marginal_guard(("b", "c")).feasible is False

    @pytest.mark.pending(
        reason="a field no factor consumes integrates out within its kernel",
        raises=AssertionError,
    )
    def test_a_sibling_of_a_dependent_field_integrates_out_within_its_kernel(self):
        lik = MarginalKernel("lik", {"beta": SCALAR}, OutputSpec(RecordSpec(y1=SCALAR, y2=SCALAR)))
        assert (lik * _prior())._marginal_guard(("y1", "beta")).feasible is True


class TestFactorGuards:
    """A joint's derived capability takes every factor's guard of the call it makes."""

    def test_a_joint_of_unguarded_factors_is_feasible(self):
        joint = _likelihood(SamplingKernel) * _prior()
        assert _capability_guard(joint, "_sample") == Feasibility(True)

    def test_a_conditional_factor_is_guarded_through_its_twin(self):
        joint = _likelihood(UndecidedSamplingKernel) * _prior()
        assert _capability_guard(joint, "_sample") == Feasibility(
            None,
            pending=(
                "whether conditional_sample() is available for UndecidedSamplingKernel depends "
                "on values not yet known; it requires: draws once its scale is known",
            ),
        )

    def test_a_rejecting_factor_rejects_the_joint_whatever_the_others_report(self):
        point = RejectingPointLaw("point", OutputSpec(beta=SCALAR), jnp.zeros(()))
        joint = _likelihood(UndecidedSamplingKernel) * point
        assert _capability_guard(joint, "_sample") == Feasibility(
            False,
            "sample() is not available for RejectingPointLaw; it requires: draws only from a "
            "finite point",
        )

    def test_a_conditional_joint_takes_its_factors_guards(self):
        kernel = UndecidedSamplingKernel("k", {"s": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel * PointLaw("p", OutputSpec(b=SCALAR), jnp.zeros(()))
        assert isinstance(joint, FactoredConditionalDistribution)
        assert _capability_guard(joint, "_conditional_sample").unresolved


class TestMarginalValues:
    """The exact marginal is the sub-joint of the target's ancestor closure, reduced."""

    def test_the_marginal_of_a_root_factor_is_that_factor_law(self):
        prior = _prior(scale=2.0)
        marginal = (_likelihood() * prior)._marginal("beta")
        assert marginal.event_spec == prior.event_spec
        assert jnp.allclose(marginal._log_prob(0.7), prior._log_prob(0.7))

    def test_the_marginal_of_an_ancestrally_closed_group_is_the_sub_joint(self):
        lik, prior = _likelihood(), _prior()
        marginal = (lik * prior * _law("other", "c"))._marginal(("y", "beta"))
        assert isinstance(marginal, SupportsFactors)
        assert marginal.factors == (lik, prior)
        assert list(marginal.event_spec.components) == ["y", "beta"]

    def test_the_marginal_of_an_edge_free_group_is_an_exposed_record_of_it(self):
        joint = _law("u", "a") * _law("v", "b") * _law("w", "c")
        marginal = joint._marginal(("a", "c"))
        assert marginal.event_spec.exposes_record
        assert list(marginal.event_spec.components) == ["a", "c"]

    def test_a_selection_marginal_follows_the_order_of_its_paths(self):
        joint = Normal("x", 0.0, 1.0) * Normal("z", 5.0, 2.0)
        marginal = joint._marginal(("z", "x"))
        assert list(marginal.event_spec.components) == ["z", "x"]
        assert marginal.event_spec == FieldView(joint, ("z", "x")).event_spec
        assert list(marginal._mean()) == ["z", "x"]
        assert jnp.allclose(marginal._cov().to_dense(), jnp.diag(jnp.array([4.0, 1.0])))
        value = {"z": jnp.asarray(4.0), "x": jnp.asarray(0.2)}
        expected = norm.logpdf(4.0, 5.0, 2.0) + norm.logpdf(0.2, 0.0, 1.0)
        assert jnp.allclose(marginal._log_prob(value), expected)

    def test_a_sub_joint_selected_against_its_factor_order_follows_the_paths(self):
        marginal = (_likelihood() * _prior())._marginal(("beta", "y"))
        assert list(marginal.event_spec.components) == ["beta", "y"]

    def test_the_marginal_inside_one_factor_is_that_factor_marginal(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        marginal = (pair * _law("other", "c"))._marginal("a")
        assert pair.marginalized == ["a"]
        assert marginal.event_spec == OutputSpec(a=SCALAR)

    def test_the_marginal_of_a_group_keeps_the_joint_label(self):
        joint = _law("u", "a") * _law("v", "b") * _law("w", "c")
        assert joint._marginal(("a", "c")).label == joint.label

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("beta", id="kept-factor"),
            pytest.param(("y", "beta"), id="sub-joint"),
            pytest.param(("beta",), id="selection-of-one"),
            pytest.param("record", id="field-of-a-record"),
            pytest.param("params/u", id="reduced-factor"),
        ],
    )
    def test_a_marginal_keeps_the_joint_label_as_the_view_there_does(self, path):
        record = OneFieldNormal("one", OutputSpec(RecordSpec(record=SCALAR)))
        params = MarginalLaw("p", OutputSpec(params=RecordSpec(u=SCALAR)), exact=("params/u",))
        joint = _likelihood() * _prior() * record * params
        assert joint._marginal(path).label == FieldView(joint, path).label == joint.label

    def test_a_selection_of_one_whole_term_is_an_exposed_record(self):
        prior = _prior()
        marginal = (_likelihood() * prior)._marginal(("beta",))
        assert marginal.event_spec == OutputSpec(RecordSpec(beta=prior.event_spec.spec))
        assert marginal.factors == (prior,)

    def test_a_selection_of_a_record_factor_components_is_that_factor_under_the_paths(self):
        pair = _pair()
        joint = pair * _law("other", "c")
        marginal = joint._marginal(("a", "b"))
        assert type(marginal) is type(pair) and marginal.spec == pair.spec
        assert (marginal.label, pair.label) == (joint.label, "pair")

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("gamma", id="unknown-component"),
            pytest.param("beta/zzz", id="below-a-leaf"),
            pytest.param(("beta", "gamma"), id="in-a-selection"),
            pytest.param("", id="empty"),
        ],
    )
    def test_a_path_that_is_not_an_event_path_raises_key_error(self, path):
        joint = _likelihood() * _prior()
        with pytest.raises(KeyError, match="not an event path"):
            joint._marginal(path)

    def test_a_projection_onto_a_one_field_record_is_the_law_of_its_field(self):
        record = OneFieldNormal("record", OutputSpec(RecordSpec(beta=SCALAR)))
        joint = _likelihood() * record
        assert joint._marginal_guard("beta").feasible is True
        marginal = joint._marginal("beta")
        assert marginal.event_spec == OutputSpec(beta=SCALAR)
        assert jnp.allclose(marginal._log_prob(0.7), norm.logpdf(0.7, 0.0, 2.0))
        draws = marginal._sample(jax.random.PRNGKey(0), (3,))
        assert jnp.allclose(draws, record._sample(jax.random.PRNGKey(0), (3,))["beta"])

    def test_the_law_of_a_record_field_draws_the_nested_mapping_of_its_leaves(self):
        fields = OutputSpec(RecordSpec(params=RecordSpec(u=SCALAR, v=SCALAR)))
        draw = Record("draw", {"params": {"u": jnp.asarray(1.0), "v": jnp.asarray(2.0)}})
        joint = PointLaw("record", fields, draw) * _law("other", "c")
        value = joint._marginal("params")._sample(jax.random.PRNGKey(0))
        assert isinstance(value, dict) and list(value) == ["u", "v"]
        assert (float(value["u"]), float(value["v"])) == (1.0, 2.0)

    def test_a_marginal_the_guard_rejects_raises_naming_the_fields(self):
        with pytest.raises(ResolutionError, match=_mentions("['y']", "['beta']")):
            (_likelihood() * _prior())._marginal("y")


# -- Conditioning a conditional joint -----------------------------------------------


def _sigma_model() -> tuple[NormalKernel, Normal, FactoredConditionalDistribution]:
    """``y | beta, sigma`` composed with a prior on ``beta``, so ``sigma`` stays unmet."""
    lik = NormalKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
    prior = _prior()
    return lik, prior, lik * prior


class TestConditioning:
    """A conditional joint binds its givens in every factor that names them."""

    def test_binding_every_given_yields_a_factored_distribution(self):
        _, prior, joint = _sigma_model()
        bound = joint._condition_on({"sigma": 1.0})
        assert isinstance(bound, FactoredDistribution)
        assert list(bound.event_spec.components) == ["y", "beta"]
        curried, kept = bound.factors
        assert kept is prior
        assert isinstance(curried, ConditionalDistribution)
        assert list(curried.given_spec) == ["beta"]

    def test_binding_some_givens_curries_to_a_smaller_conditional_joint(self):
        kernel = NormalKernel("k", {"s1": SCALAR, "s2": SCALAR}, OutputSpec(a=SCALAR))
        bound = (kernel * _law("other", "b"))._condition_on({"s1": 1.0})
        assert isinstance(bound, FactoredConditionalDistribution)
        assert list(bound.given_spec) == ["s2"]
        assert list(bound.event_spec.components) == ["a", "b"]

    def test_currying_then_binding_the_rest_equals_binding_at_once(self):
        kernel = NormalKernel("k", {"s1": SCALAR, "s2": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel * _law("other", "b")
        stepwise = joint._condition_on({"s1": 1.0})._condition_on({"s2": 2.0})
        at_once = joint._condition_on({"s1": 1.0, "s2": 2.0})
        assert stepwise.spec == at_once.spec
        assert float(stepwise.factors[0].loc) == float(at_once.factors[0].loc) == 3.0

    def test_every_given_value_arrives_in_the_given(self):
        _, _, joint = _sigma_model()
        assert isinstance(joint._condition_on({"sigma": 1.0}), FactoredDistribution)

    def test_a_keyword_is_an_option_and_binds_no_slot(self):
        _, _, joint = _sigma_model()
        unbound = joint._condition_on({}, sigma=1.0)
        assert isinstance(unbound, FactoredConditionalDistribution)
        assert list(unbound.given_spec) == ["sigma"]

    def test_an_option_reaches_the_primitive_of_each_bound_factor(self):
        seen: list[dict[str, Any]] = []

        class RecordingKernel(NormalKernel):
            def _condition_on(self, given, /, **options):
                seen.append(options)
                return super()._condition_on(given)

        lik = RecordingKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        (lik * _prior())._condition_on({"sigma": 1.0}, budget=3)
        assert seen == [{"budget": 3}]

    def test_a_bound_value_binds_the_factor_that_names_it(self):
        kernel = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        bound = (kernel * _law("other", "b"))._condition_on({"x": 0.25})
        assert float(bound.factors[0].loc) == 0.25

    @pytest.mark.parametrize("name", ["beta", "y", "gamma"])
    def test_a_name_that_is_not_a_given_slot_of_the_joint_raises(self, name):
        _, _, joint = _sigma_model()
        with pytest.raises(KeyError, match=_mentions(f"'{name}'", "given slot")):
            joint._condition_on({name: 0.0})

    def test_the_bound_joint_derives_its_capabilities_from_the_bound_factors(self):
        joint = SamplingKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) * Normal("b", 0.0, 1.0)
        bound = joint._condition_on({"x": 0.0})
        assert isinstance(bound, SupportsSampling)
        assert isinstance(bound, SupportsMean)

    def test_a_bound_factor_that_breaks_its_kernel_declaration_raises(self):
        kernel = RenamingKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel * _law("other", "b")
        with pytest.raises(ValueError):
            joint._condition_on({"x": 0.0})


# -- Numeric markers --------------------------------------------------------------


class TestNumericMarkers:
    """Membership in each factored numeric marker is read from the joint's declarations."""

    def test_a_joint_with_a_numeric_event_is_a_factored_numeric_distribution(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        assert isinstance(joint, FactoredNumericDistribution)
        assert isinstance(joint, NumericDistribution)

    def test_a_non_numeric_component_leaves_the_joint_outside_the_marker(self):
        joint = Normal("a", 0.0, 1.0) * _law("o", "o", OpaqueSpec())
        assert not isinstance(joint, FactoredNumericDistribution)
        assert not isinstance(joint, NumericDistribution)

    def test_a_numeric_law_that_is_not_factored_is_outside_the_marker(self):
        assert not isinstance(Normal("a", 0.0, 1.0), FactoredNumericDistribution)

    @pytest.mark.parametrize(
        ("given", "event", "markers"),
        [
            pytest.param(SCALAR, SCALAR, set(_CONDITIONAL_MARKERS), id="both-numeric"),
            pytest.param(
                OpaqueSpec(), SCALAR, {FactoredConditionalNumericDistribution}, id="event-numeric"
            ),
            pytest.param(
                SCALAR, OpaqueSpec(), {FactoredNumericConditionalDistribution}, id="given-numeric"
            ),
            pytest.param(OpaqueSpec(), OpaqueSpec(), set(), id="neither"),
        ],
    )
    def test_each_conditional_marker_reads_its_side(self, given, event, markers):
        kernel = NormalKernel("k", {"x": given}, OutputSpec(a=event))
        joint = kernel * Normal("b", 0.0, 1.0)
        assert {marker for marker in _CONDITIONAL_MARKERS if isinstance(joint, marker)} == markers

    def test_an_unconditional_joint_is_in_no_conditional_marker(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        assert not any(isinstance(joint, marker) for marker in _CONDITIONAL_MARKERS)

    def test_a_conditional_joint_is_not_a_factored_numeric_distribution(self):
        joint = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) * Normal("b", 0.0, 1.0)
        assert not isinstance(joint, FactoredNumericDistribution)
        assert isinstance(joint, ConditionalNumericDistribution)
        assert isinstance(joint, NumericConditionalDistribution)

    def test_a_kernel_that_is_not_factored_is_in_no_factored_marker(self):
        kernel = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        assert isinstance(kernel, ConditionalNumericDistribution)
        assert not any(isinstance(kernel, marker) for marker in _CONDITIONAL_MARKERS)

    def test_a_class_inheriting_the_marker_constructs_as_itself(self):
        joint = NumericJoint("model", [Normal("a", 0.0, 1.0)])
        assert isinstance(joint, NumericJoint)
        assert joint.factors[0].label == "a"

    def test_a_class_inheriting_the_marker_has_its_claim_checked(self):
        with pytest.raises(TypeError, match="inherits FactoredNumericDistribution"):
            NumericJoint("model", [_law("o", "o", OpaqueSpec())])


# -- Dimension transforms ---------------------------------------------------------


def _symbolic_joint() -> FactoredDistribution:
    """Two laws over ``(n,)`` arrays and one over a scalar, with ``n`` free."""
    return _law("la", "a", SYMBOLIC) * _law("lb", "b", SYMBOLIC) * _law("lc", "c")


def _symbolic_conditional_joint() -> FactoredConditionalDistribution:
    """A kernel on an ``(n,)`` given beside a law over ``(n,)`` arrays, with ``n`` free."""
    kernel = NormalKernel("k", {"x": SYMBOLIC}, OutputSpec(a=SYMBOLIC))
    return kernel * _law("lb", "b", SYMBOLIC)


class TestDimensionTransforms:
    """``with_dim_sizes`` and ``with_dim_names`` apply to each factor declaring the dimension."""

    def test_binding_a_dimension_binds_it_in_every_factor(self):
        joint = _symbolic_joint()
        bound = joint.with_dim_sizes(n=3)
        assert isinstance(bound, FactoredDistribution)
        assert bound.label == joint.label
        assert [factor.event_spec.spec.shape for factor in bound.factors] == [(3,), (3,), ()]
        assert bound.event_spec.spec.free_dims == frozenset()

    def test_renaming_a_dimension_renames_it_in_every_factor(self):
        renamed = _symbolic_joint().with_dim_names(n="m")
        assert [factor.event_spec.spec.shape for factor in renamed.factors] == [("m",), ("m",), ()]
        assert renamed.event_spec.spec.free_dims == {"m"}

    def test_the_declaration_follows_the_transformed_factors(self):
        bound = _symbolic_joint().with_dim_sizes(n=3)
        assert bound.spec == FactoredDistribution(bound.label, bound.factors).spec

    def test_binding_a_dimension_that_is_not_free_raises(self):
        with pytest.raises(ValueError, match=_mentions("'k'", "free dimension")):
            _symbolic_joint().with_dim_sizes(k=3)

    def test_the_original_joint_is_unchanged(self):
        joint = _symbolic_joint()
        joint.with_dim_sizes(n=3)
        joint.with_dim_names(n="m")
        assert joint.event_spec.spec.free_dims == {"n"}
        assert [factor.event_spec.spec.shape for factor in joint.factors] == [("n",), ("n",), ()]

    def test_binding_a_conditional_joint_dimension_binds_it_in_every_factor(self):
        bound = _symbolic_conditional_joint().with_dim_sizes(n=3)
        assert bound.given_spec["x"].shape == (3,)
        assert [factor.event_spec.spec.shape for factor in bound.factors] == [(3,), (3,)]
        assert bound.factors[0].given_spec["x"].shape == (3,)

    def test_renaming_a_conditional_joint_dimension_renames_it_in_every_factor(self):
        renamed = _symbolic_conditional_joint().with_dim_names(n="m")
        assert renamed.given_spec["x"].shape == ("m",)
        assert [factor.event_spec.spec.shape for factor in renamed.factors] == [("m",), ("m",)]


class TestPathRenames:
    """A joint renames a component through the factors that produce and consume it."""

    def test_renaming_a_component_renames_its_producer_and_its_consumers(self):
        renamed = (_likelihood() * _prior()).with_path_names(beta="theta")
        assert isinstance(renamed, FactoredDistribution)
        assert list(renamed.event_spec.components) == ["y", "theta"]
        consumer, producer = renamed.factors
        assert list(consumer.given_spec) == ["theta"]
        assert list(producer.event_spec.components) == ["theta"]

    def test_renaming_an_unmet_given_renames_it_in_every_factor_that_names_it(self):
        lik = NormalKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        noise = NormalKernel("noise", {"sigma": SCALAR}, OutputSpec(e=SCALAR))
        renamed = (lik * noise * _prior()).with_path_names(sigma="s")
        assert isinstance(renamed, FactoredConditionalDistribution)
        assert list(renamed.given_spec) == ["s"]
        assert [sorted(getattr(f, "given_spec", {})) for f in renamed.factors] == [
            ["beta", "s"],
            ["s"],
            [],
        ]

    def test_a_rename_keeps_the_graph_and_the_label(self):
        joint = _likelihood() * _prior()
        renamed = joint.with_path_names(beta="theta", y="obs")
        assert renamed.label == joint.label
        assert [edge[:2] for edge in renamed._graph.edges] == [
            edge[:2] for edge in joint._graph.edges
        ]
        assert [info.label for info in renamed.provenance.parents] == [joint.label]

    def test_gathering_components_of_two_factors_regroups_them(self, key):
        """The node ``g`` is one factor: the sub-joint of the two factors, packaged as ``g``."""
        joint = Normal("a", 0.0, 1.0) * Normal("b", 2.0, 1.0)
        renamed = joint.with_path_names({"a": "g/a", "b": "g/b"})
        assert isinstance(renamed, SupportsFactors)
        assert list(renamed.event_spec.components) == ["g"]
        (group,) = renamed.factors
        assert [part.label for part in group.factors] == ["a", "b"]
        assert list(group.event_spec.components) == ["g"]
        assert not group.event_spec.exposes_record
        assert list(renamed._sample(key)["g"]) == ["a", "b"]


# -- Notation -----------------------------------------------------------------------


def _with_fixed_paths(term: Any, *paths: str) -> Any:
    """*term* holding *paths* fixed, as conditioning on them records."""
    object.__setattr__(term, "_fixed_paths", paths)
    return term


class TestNotation:
    """An unlabeled joint reads factor by factor, and a labeled one by its label."""

    def test_a_composed_joint_reads_factor_by_factor(self):
        joint = _likelihood() * _prior()
        assert str(joint) == joint.notation == "lik(y | beta)·prior(beta)"
        assert joint.label == "lik·prior"

    def test_a_relabeled_joint_reads_by_its_label_and_its_components(self):
        model = (_likelihood() * _prior()).with_label("model")
        assert model.notation == "model(y, beta)"

    def test_a_constructed_joint_reads_by_its_label(self):
        model = FactoredDistribution("model", [_likelihood(), _prior()])
        assert model.notation == "model(y, beta)"

    def test_a_joint_relabeled_with_its_own_label_reads_by_it(self):
        joint = _likelihood() * _prior()
        assert joint.with_label(joint.label).notation == "(lik·prior)(y, beta)"

    def test_a_conditional_joint_reads_as_an_unconditional_one_does(self):
        _, _, joint = _sigma_model()
        assert joint.notation == "lik(y | beta, sigma)·prior(beta)"
        assert joint.with_label("model").notation == "model(y, beta | sigma)"

    def test_a_chain_reads_one_factor_after_another_under_either_grouping(self):
        a, b, c = _law("a", "x"), _law("b", "z"), _law("c", "w")
        assert ((a * b) * c).notation == (a * (b * c)).notation == "a(x)·b(z)·c(w)"

    def test_a_labeled_factor_reads_by_its_own_notation(self):
        model = (_likelihood() * _prior()).with_label("model").with_path_names(y="obs")
        joint = _law("d", "d") * model
        assert joint.notation == "d(d)·lik(obs | beta)·prior(beta)"

    def test_the_repr_keeps_the_label_first(self):
        assert repr(_likelihood() * _prior()).startswith("FactoredDistribution(\n    'lik·prior',")

    def test_an_unlabeled_joint_that_holds_paths_fixed_reads_by_its_label(self):
        """Its factors' notations would leave out the fixed paths."""
        joint = _with_fixed_paths(_law("a", "x") * _law("b", "z"), "y")
        assert joint.notation == "(a·b)(x, z; y)"
        model = _with_fixed_paths((_law("a", "x") * _law("b", "z")).with_label("model"), "y")
        assert model.notation == "model(x, z; y)"


class TestTheLabeledFlag:
    """A law derived from a joint is labeled or unlabeled as the joint is."""

    @staticmethod
    def _joints() -> tuple[FactoredDistribution, FactoredDistribution]:
        joint = _symbolic_joint()
        return joint, joint.with_label("model")

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda j: j.with_dim_sizes(n=3), id="with_dim_sizes"),
            pytest.param(lambda j: j.with_dim_names(n="m"), id="with_dim_names"),
            pytest.param(lambda j: j.with_path_names(a="u"), id="rename-through-factors"),
            pytest.param(lambda j: j.with_path_names({"a": "g/a", "b": "g/b"}), id="regroup"),
            pytest.param(lambda j: j._marginal(("a", "b")), id="marginal-over-two-factors"),
            pytest.param(lambda j: copy.copy(j), id="copy"),
            pytest.param(lambda j: pickle.loads(pickle.dumps(j)), id="pickle"),
        ],
    )
    def test_a_derived_joint_keeps_the_flag(self, derive):
        unlabeled, labeled = self._joints()
        assert _is_named(derive(unlabeled)) is False
        assert _is_named(derive(labeled)) is True

    def test_a_packaged_sub_joint_is_labeled_as_the_joint_is(self):
        unlabeled, labeled = self._joints()
        for joint, named in ((unlabeled, False), (labeled, True)):
            regrouped = joint.with_path_names({"a": "g/a", "b": "g/b"})
            group = next(
                factor for factor in regrouped.factors if isinstance(factor, SupportsFactors)
            )
            assert _is_named(group) is named

    def test_binding_a_conditional_joint_keeps_the_flag(self):
        _, _, joint = _sigma_model()
        for kernel, named in ((joint, False), (joint.with_label("model"), True)):
            assert _is_named(kernel._condition_on({"sigma": 1.0})) is named

    def test_a_relabeled_derived_joint_is_labeled(self):
        derived = _symbolic_joint().with_dim_sizes(n=3)
        assert derived.with_label("model").notation == "model(a, b, c)"

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda j: j.with_dim_sizes(n=3), id="with_dim_sizes"),
            pytest.param(lambda j: j.with_dim_names(n="m"), id="with_dim_names"),
            pytest.param(lambda j: j.with_label("model"), id="with_label"),
            pytest.param(lambda j: j.with_path_names(a="u"), id="rename-through-factors"),
            pytest.param(lambda j: j.with_path_names({"a": "g/a", "b": "g/b"}), id="regroup"),
            pytest.param(lambda j: j._marginal(("a", "b")), id="marginal-over-two-factors"),
        ],
    )
    def test_a_derived_joint_keeps_the_fixed_paths(self, derive):
        joint = _with_fixed_paths(_symbolic_joint(), "y")
        assert _fixed_paths(derive(joint)) == ("y",)

    def test_a_bound_conditional_joint_keeps_the_fixed_paths(self):
        _, _, joint = _sigma_model()
        assert _fixed_paths(_with_fixed_paths(joint, "y")._condition_on({"sigma": 1.0})) == ("y",)


# -- Round trips ----------------------------------------------------------------------


_JOINTS = [
    pytest.param(lambda: _likelihood(FullKernel) * _prior(), id="dependent"),
    pytest.param(lambda: Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0), id="edge-free"),
    pytest.param(lambda: _likelihood(FullKernel) * Normal("c", 0.0, 1.0), id="conditional"),
    pytest.param(lambda: (_likelihood() * _prior()).with_label("posterior"), id="relabeled"),
    pytest.param(lambda: _symbolic_joint().with_dim_sizes(n=3), id="bound"),
]


def _assert_same_joint(restored: Any, joint: Any) -> None:
    assert type(restored) is type(joint)
    assert (restored.label, restored.spec, restored.notation) == (
        joint.label,
        joint.spec,
        joint.notation,
    )
    assert [(type(f), f.label, f.spec) for f in restored.factors] == [
        (type(f), f.label, f.spec) for f in joint.factors
    ]
    assert _claimed(restored) == _claimed(joint)


class TestRoundTrips:
    """A pickled or copied joint keeps its label, declarations, factors, and capabilities."""

    @pytest.mark.parametrize("make", _JOINTS)
    def test_pickle(self, make):
        joint = make()
        _assert_same_joint(pickle.loads(pickle.dumps(joint)), joint)

    @pytest.mark.parametrize("make", _JOINTS)
    def test_copy(self, make):
        joint = make()
        _assert_same_joint(copy.copy(joint), joint)

    def test_a_restored_conditional_joint_still_curries(self):
        _, _, joint = _sigma_model()
        bound = pickle.loads(pickle.dumps(joint))._condition_on({"sigma": 1.0})
        assert isinstance(bound, FactoredDistribution)
        assert list(bound.event_spec.components) == ["y", "beta"]


# -- Joint sampling and densities -------------------------------------------------


class TestJointSampling:
    """A draw extracts each factor's components from its one draw, in canonical factor order."""

    def test_a_draw_holds_every_component_in_canonical_order(self):
        joint = _likelihood(SamplingKernel) * _prior() * Normal("c", 0.0, 1.0)
        draw = joint._sample(jax.random.PRNGKey(0))
        assert joint.event_spec.spec.is_valid(draw)
        assert list(draw.keys()) == ["y", "beta", "c"]

    def test_a_draw_extracts_components_by_each_factor_packaging(self):
        fields = OutputSpec(RecordSpec(a=SCALAR, b=SCALAR))
        exposed = PointLaw("exposed", fields, {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)})
        params = OutputSpec(params=RecordSpec(u=SCALAR, v=SCALAR))
        whole = PointLaw("whole", params, {"u": jnp.asarray(3.0), "v": jnp.asarray(4.0)})
        draw = (exposed * whole)._sample(jax.random.PRNGKey(0))
        assert list(draw.keys()) == ["a", "b", "params"]
        assert (float(draw["a"]), float(draw["b"])) == (1.0, 2.0)
        assert (float(draw["params"]["u"]), float(draw["params"]["v"])) == (3.0, 4.0)

    def test_every_consumer_reads_the_draw_its_producer_reports(self):
        first = SamplingKernel("first", {"beta": SCALAR}, OutputSpec(y1=SCALAR), scale=1e-6)
        second = SamplingKernel("second", {"beta": SCALAR}, OutputSpec(y2=SCALAR), scale=1e-6)
        draw = (first * second * _prior())._sample(jax.random.PRNGKey(1))
        assert jnp.allclose(draw["y1"], draw["beta"], atol=1e-4)
        assert jnp.allclose(draw["y2"], draw["beta"], atol=1e-4)

    def test_a_sample_shape_prepends_batch_axes_to_every_component(self):
        draws = (_likelihood(SamplingKernel) * _prior())._sample(jax.random.PRNGKey(2), (4,))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (4,)

    def test_a_conditional_joint_samples_under_its_given(self):
        lik = SamplingKernel(
            "lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6
        )
        draw = (lik * _prior())._conditional_sample({"sigma": 10.0}, jax.random.PRNGKey(3))
        assert list(draw.keys()) == ["y", "beta"]
        assert jnp.allclose(draw["y"], draw["beta"] + 10.0, atol=1e-4)

    def test_one_key_reproduces_the_draw(self):
        joint = _likelihood(SamplingKernel) * _prior() * Normal("c", 0.0, 1.0)
        first, again = joint._sample(jax.random.PRNGKey(4)), joint._sample(jax.random.PRNGKey(4))
        other = joint._sample(jax.random.PRNGKey(5))
        assert all(jnp.array_equal(first[c], again[c]) for c in first)
        assert not any(jnp.array_equal(first[c], other[c]) for c in first)

    def test_each_factor_draws_with_its_own_key_of_one_split(self):
        lik, prior = _likelihood(SamplingKernel), _prior()
        key = jax.random.PRNGKey(6)
        draw = (lik * prior)._sample(key)
        lik_key, prior_key = jax.random.split(key, 2)
        assert jnp.array_equal(draw["beta"], prior._sample(prior_key))
        assert jnp.allclose(draw["y"], lik._conditional_sample({"beta": draw["beta"]}, lik_key))

    def test_under_a_sample_shape_each_draw_conditions_on_its_own_producer_draw(self):
        lik = SamplingKernel("lik", {"beta": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6)
        draws = (lik * _prior())._sample(jax.random.PRNGKey(7), (3, 2))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (3, 2)
        assert jnp.allclose(draws["y"], draws["beta"], atol=1e-4)
        assert len(set(jnp.ravel(draws["beta"]).tolist())) == 6

    def test_a_conditional_joint_samples_under_its_given_with_a_sample_shape(self):
        lik = SamplingKernel(
            "lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6
        )
        draws = (lik * _prior())._conditional_sample({"sigma": 10.0}, jax.random.PRNGKey(8), (5,))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (5,)
        assert jnp.allclose(draws["y"], draws["beta"] + 10.0, atol=1e-4)

    def test_a_kernel_on_an_opaque_component_draws_once_per_draw(self):
        joint = _labeled_joint()
        assert _capability_guard(joint, "_sample").feasible is True
        draws = joint._sample(jax.random.PRNGKey(9), (2, 3))
        assert draws["label"].dtype == object and draws["label"].shape == (2, 3)
        lengths = np.vectorize(len, otypes=[float])(draws["label"])
        np.testing.assert_array_equal(np.asarray(draws["y"]), lengths)


class TestJointDensity:
    """Scoring reconstructs each factor's event and sums the factors' log-densities."""

    def test_the_log_density_is_the_sum_of_the_factor_log_densities(self):
        lik, prior = _likelihood(ScoringKernel), _prior(scale=2.0)
        y, beta = jnp.asarray(0.3), jnp.asarray(-0.2)
        expected = lik._conditional_log_prob({"beta": beta}, y) + prior._log_prob(beta)
        assert jnp.allclose((lik * prior)._log_prob({"y": y, "beta": beta}), expected)

    def test_the_unnormalized_density_differs_from_the_factor_sum_by_a_constant(self):
        lik, prior = _likelihood(UnnormalizedKernel), _prior()
        joint = lik * prior

        def factor_sum(value):
            y, beta = value["y"], value["beta"]
            return lik._conditional_unnormalized_log_prob({"beta": beta}, y) + prior._log_prob(beta)

        first = {"y": jnp.asarray(0.3), "beta": jnp.asarray(-0.2)}
        second = {"y": jnp.asarray(-1.1), "beta": jnp.asarray(0.4)}
        difference = joint._unnormalized_log_prob(first) - joint._unnormalized_log_prob(second)
        assert jnp.allclose(difference, factor_sum(first) - factor_sum(second))

    def test_each_factor_scores_a_value_of_its_own_declared_kind(self):
        exposed = RecordingLaw("exposed", OutputSpec(RecordSpec(beta=SCALAR)))
        whole = RecordingLaw("whole", OutputSpec(gamma=SCALAR))
        joint = _likelihood(ScoringKernel) * exposed * whole
        joint._log_prob(
            {"y": jnp.asarray(0.3), "beta": jnp.asarray(-0.2), "gamma": jnp.asarray(0.5)}
        )
        (beta_value,) = exposed.scored
        (gamma_value,) = whole.scored
        assert exposed.event_spec.spec.is_valid(beta_value)
        assert not SCALAR.is_valid(beta_value)
        assert whole.event_spec.spec.is_valid(gamma_value)
        assert not exposed.event_spec.spec.is_valid(gamma_value)

    def test_a_conditional_joint_scores_under_its_given(self):
        lik = ScoringKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        prior = _prior()
        y, beta = jnp.asarray(0.3), jnp.asarray(-0.2)
        kernel_given = {"beta": beta, "sigma": 1.5}
        expected = lik._conditional_log_prob(kernel_given, y) + prior._log_prob(beta)
        value = {"y": y, "beta": beta}
        assert jnp.allclose((lik * prior)._conditional_log_prob({"sigma": 1.5}, value), expected)

    def test_the_log_density_agrees_with_a_hand_computed_sum(self):
        joint = _likelihood(ScoringKernel, scale=0.5) * _prior(scale=2.0)
        # log N(y; beta, 0.5^2) + log N(beta; 0, 2^2), written out.
        y, beta = 0.3, -0.2
        expected = (
            -0.5 * ((y - beta) / 0.5) ** 2
            - jnp.log(0.5)
            - 0.5 * (beta / 2.0) ** 2
            - jnp.log(2.0)
            - jnp.log(2.0 * jnp.pi)
        )
        value = {"y": jnp.asarray(y), "beta": jnp.asarray(beta)}
        assert jnp.allclose(joint._log_prob(value), expected)

    def test_a_batch_of_values_is_scored_value_by_value(self):
        joint = _likelihood(ScoringKernel) * _prior(scale=2.0)
        values = {"y": jnp.array([0.3, -1.0, 2.0]), "beta": jnp.array([-0.2, 0.5, 1.0])}
        scores = joint._log_prob(values)
        assert jnp.shape(scores) == (3,)
        for index in range(3):
            one = {name: column[index] for name, column in values.items()}
            assert jnp.allclose(scores[index], joint._log_prob(one))

    def test_a_conditional_joint_scores_a_batch_under_its_given(self):
        lik = ScoringKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = lik * _prior()
        values = {"y": jnp.array([0.3, 1.2]), "beta": jnp.array([-0.2, 0.4])}
        scores = joint._conditional_log_prob({"sigma": 1.5}, values)
        assert jnp.shape(scores) == (2,)
        for index in range(2):
            one = {name: column[index] for name, column in values.items()}
            assert jnp.allclose(scores[index], joint._conditional_log_prob({"sigma": 1.5}, one))

    def test_a_kernel_that_does_not_broadcast_scores_each_value_at_its_own_given(self):
        joint = _likelihood(NormKernel) * _prior(scale=2.0)
        y, beta = jnp.array([0.3, -1.0, 2.0]), jnp.array([-0.2, 0.5, 1.0])
        # log N(y_i; |beta_i|, 1) + log N(beta_i; 0, 2^2), written out.
        expected = (
            -0.5 * (y - jnp.abs(beta)) ** 2
            - 0.5 * (beta / 2.0) ** 2
            - jnp.log(2.0)
            - jnp.log(2.0 * jnp.pi)
        )
        assert jnp.allclose(joint._log_prob({"y": y, "beta": beta}), expected)

    def test_a_kernel_that_does_not_broadcast_scores_each_value_under_the_given(self):
        lik = NormKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        y, beta = jnp.array([0.3, 1.2]), jnp.array([-0.2, 0.4])
        scores = (lik * _prior())._conditional_log_prob({"sigma": 0.5}, {"y": y, "beta": beta})
        # log N(y_i; |beta_i|, 0.5^2) + log N(beta_i; 0, 1), written out.
        expected = (
            -0.5 * ((y - jnp.abs(beta)) / 0.5) ** 2
            - jnp.log(0.5)
            - 0.5 * beta**2
            - jnp.log(2.0 * jnp.pi)
        )
        assert jnp.allclose(scores, expected)

    def test_a_kernel_on_an_opaque_component_scores_value_by_value(self):
        y = jnp.array([1.0, 2.0, 0.0])
        values = {"y": y, "label": np.array(["a", "bb", "ccc"], dtype=object)}
        scores = _labeled_joint()._log_prob(values)
        # log N(y_i; len(label_i), 1) + log(1/3), with the lengths 1, 2, and 3.
        expected = -0.5 * (y - jnp.array([1.0, 2.0, 3.0])) ** 2 - 0.5 * jnp.log(2 * jnp.pi)
        assert jnp.allclose(scores, expected - jnp.log(3.0))

    def test_an_edge_free_density_is_the_sum_of_the_factor_densities(self):
        first, second = Normal("a", 0.0, 1.0), Normal("b", 1.0, 2.0)
        value = {"a": jnp.asarray(0.3), "b": jnp.asarray(-0.4)}
        expected = first._log_prob(value["a"]) + second._log_prob(value["b"])
        assert jnp.allclose((first * second)._log_prob(value), expected)


class TestFusedGiven:
    """A conditional joint's capabilities take a value for every given slot, and only those."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda joint, given: joint._conditional_sample(given, jax.random.PRNGKey(0)),
                id="sample",
            ),
            pytest.param(
                lambda joint, given: joint._conditional_log_prob(given, {"y": 0.0, "beta": 0.0}),
                id="log-prob",
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("given", "slot"),
        [
            pytest.param({}, "sigma", id="missing"),
            pytest.param({"sigma": 1.0, "tau": 1.0}, "tau", id="unknown"),
        ],
    )
    def test_a_given_that_omits_or_adds_a_slot_raises(self, call, given, slot):
        lik = FullKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        with pytest.raises(KeyError, match=_mentions(f"'{slot}'", "'lik·prior'")):
            call(lik * _prior(), given)

    def test_a_record_given_binds_by_its_fields(self):
        lik = FullKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = lik * _prior()
        value = {"y": jnp.asarray(0.3), "beta": jnp.asarray(-0.2)}
        from_record = joint._conditional_log_prob(Record("given", sigma=1.5), value)
        assert jnp.allclose(from_record, joint._conditional_log_prob({"sigma": 1.5}, value))
