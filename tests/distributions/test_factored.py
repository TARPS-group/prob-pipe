"""Factored distributions: joints that carry the ordered list of their factors.

A factored joint holds its factors in conditional-first order and derives the
rest from them. Its event spec takes the positional form, a record over the
factors' outputs; sampling and the densities are the intersection of the factors'
capabilities; the moments are present exactly when an edge-free joint's factors
all have them; and the marginal is exact per path, by the factor graph. A
conditional joint binds its givens in every factor that names them.
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
    Poisson,
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
    conditional_distribution,
)
from probpipe.core._compilation import _program_store_of
from probpipe.core._dispatch import Feasibility
from probpipe.core._specs import _has_record_outputs
from probpipe.distributions import (
    ConditionalDistribution,
    ConditionalNumericDistribution,
    CoupledMarginal,
    Distribution,
    FactoredConditionalDistribution,
    FactoredConditionalNumericDistribution,
    FactoredDistribution,
    FactoredFullyNumericConditionalDistribution,
    FactoredNumericConditionalDistribution,
    FactoredNumericDistribution,
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
from probpipe.families import MixtureDistribution
from probpipe.linalg import DenseLinOp
from probpipe.operations._marginal import marginal as marginal_operation
from tests._compilation import count_compilations
from tests._fixed_paths import with_fixed_paths

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

    Binding some of the inputs returns the kernel over the rest, which
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
        super().__init__(
            given_spec,
            event_spec,
            label=label,
        )
        self._loc = loc
        self._scale = scale
        self._bound = dict(bound or {})

    def _location(self, given: Any) -> Any:
        return jnp.asarray(self._loc({**self._bound, **dict(given.items())}))

    def _condition_on(self, given, /, **options):
        values = {**self._bound, **dict(given.items())}
        rest = {
            input_name: spec
            for input_name, spec in self.given_spec.items()
            if input_name not in values
        }
        if rest:
            return type(self)(
                self.label, rest, self.event_spec, loc=self._loc, scale=self._scale, bound=values
            )
        (output_name,) = self.event_spec.outputs
        return Normal(output_name, self._loc(values), self._scale, label=self.label)


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

    The scale is the given ``sigma`` when the kernel has that input, and one
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
    """A kernel whose bound law declares an output other than the kernel's own."""

    def _condition_on(self, given, /, **kwargs):
        return Normal("elsewhere", 0.0, 1.0, label=self.label)


class MarginalKernel(NormalKernel, SupportsConditionalMarginals):
    """A kernel over a record in the positional form.

    Its conditional marginal is exact at every path.
    """

    def _condition_on(self, given, /, **kwargs):
        return Law(
            self.event_spec,
            label=self.name,
        )

    def _conditional_marginal(self, given, path):
        return Law(
            OutputSpec(**{path: SCALAR}),
            label=path,
        )


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

    It records each path it marginalizes. The marginal at one path is a law
    over the node under an output named by the path's last name, and the
    marginal at a list of paths, exact when each is listed, is a law over the
    record that keeps each node at its path.
    """

    def __init__(self, label: str, event_spec: OutputSpec, *, exact: tuple[str, ...] = ()) -> None:
        super().__init__(
            event_spec,
            label=label,
        )
        self.exact = frozenset(exact)
        self.marginalized: list[Any] = []

    def _marginal(self, path):
        self.marginalized.append(path)
        if isinstance(path, tuple):
            return Law(
                OutputSpec(RecordSpec({each: SCALAR for each in path})),
                label=self.label,
            )
        return Law(
            OutputSpec(**{path.rsplit("/", 1)[-1]: SCALAR}),
            label=path,
        )

    def _marginal_guard(self, path):
        paths = path if isinstance(path, tuple) else (path,)
        if all(each in self.exact for each in paths):
            return Feasibility(True)
        return Feasibility(False, f"{self.label!r} has no exact marginal at {path!r}")


class TotalMarginalLaw(Law, SupportsMarginals):
    """A law whose marginal is exact at every path, so it defines no guard."""

    def _marginal(self, path):
        return Law(
            OutputSpec(**{path: SCALAR}),
            label=path,
        )


class RecordingLaw(Law, SupportsLogProb):
    """A law whose log-density records each value it scores and returns zero."""

    def __init__(self, label: str, event_spec: OutputSpec) -> None:
        super().__init__(
            event_spec,
            label=label,
        )
        self.scored: list[Any] = []

    def _log_prob(self, value):
        self.scored.append(value)
        return jnp.zeros(())


class PointLaw(Law, SupportsSampling):
    """A point mass at a fixed raw draw, which every sample returns."""

    def __init__(self, label: str, event_spec: OutputSpec, draw: Any) -> None:
        super().__init__(
            event_spec,
            label=label,
        )
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
        super().__init__(
            event_spec,
            label=label,
        )
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
    """``y | label`` merged with the law of ``label``, an opaque output."""
    kernel = LengthKernel("lik", {"label": OpaqueSpec()}, OutputSpec(y=SCALAR))
    return kernel & LabelLaw(
        OutputSpec(label=OpaqueSpec()),
        label="labels",
    )


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
    """A normal law labeled ``prior`` whose output is ``beta``."""
    return Normal("beta", 0.0, scale, label="prior")


def _law(label: str, output_name: str, spec: Any = SCALAR, law: type[Law] = Law) -> Law:
    """A law labeled *label* whose whole-term output is *output_name*."""
    return law(event_spec=OutputSpec(**{output_name: spec}), label=label)


def _pair(label: str = "pair", law: type[Law] = Law, **options: Any) -> Law:
    """A law over a record of the fields ``a`` and ``b``, in the positional form."""
    return law(event_spec=OutputSpec(RecordSpec(a=SCALAR, b=SCALAR)), label=label, **options)


def _exponential_location_model() -> FactoredDistribution:
    """``x ~ Normal(0, 1)`` and ``y | x ~ Normal(exp(x), 1)``, whose ``𝔼[y]`` is ``e^{1/2}``."""
    kernel = FullKernel("y_given_x", {"x": SCALAR}, OutputSpec(y=SCALAR), loc=_exp_x)
    return kernel & Normal("x", 0.0, 1.0)


def _claimed(term: Any) -> set[type]:
    """The capabilities a joint may claim that *term* claims."""
    return {capability for capability in _CAPABILITIES if isinstance(term, capability)}


def _mentions(*fragments: str) -> str:
    """A pattern matching a message that contains every fragment, in any order."""
    return "".join(f"(?=.*{re.escape(fragment)})" for fragment in fragments)


# -- Construction -------------------------------------------------------------------


class TestConstruction:
    """A factored joint is built from an ordered list of factors under the merge rules."""

    def test_same_named_opaque_givens_keep_the_first_type(self):
        first = NormalKernel("first", {"s": OpaqueSpec()}, OutputSpec(y=SCALAR))
        second = NormalKernel("second", {"s": OpaqueSpec(type=str)}, OutputSpec(z=SCALAR))
        assert (first & second).given_spec["s"] == OpaqueSpec()
        with pytest.raises(ValueError, match="does not conform"):
            _ = (
                first & NormalKernel("third", {"s": OpaqueSpec(type=int)}, OutputSpec(w=SCALAR))
            ) & (second)

    def test_the_joint_holds_its_factors_in_order(self):
        lik, prior = _likelihood(), _prior()
        joint = FactoredDistribution(
            [lik, prior],
            label="model",
        )
        assert joint.label == "model"
        assert joint.factors == (lik, prior)

    def test_a_conditional_joint_holds_its_factors_in_order(self):
        lik, other = _likelihood(), _law("other", "c")
        joint = FactoredConditionalDistribution(
            [lik, other],
            label="model",
        )
        assert joint.factors == (lik, other)
        assert list(joint.given_spec) == ["beta"]

    def test_merge_builds_the_joint_the_constructor_builds(self):
        lik, prior = _likelihood(), _prior()
        merged, constructed = (
            (lik & prior),
            FactoredDistribution(
                [lik, prior],
                label="lik·prior",
            ),
        )
        assert type(merged) is type(constructed)
        assert (merged.label, merged.spec, merged.factors) == (
            constructed.label,
            constructed.spec,
            constructed.factors,
        )

    def test_the_factors_are_a_tuple(self):
        assert isinstance(
            FactoredDistribution(
                [_likelihood(), _prior()],
                label="model",
            ).factors,
            tuple,
        )

    @pytest.mark.parametrize("factor", [1.0, "beta", OutputSpec(beta=SCALAR)])
    def test_a_factor_is_a_distribution_or_a_kernel(self, factor):
        with pytest.raises(
            TypeError, match=_mentions("ConditionalDistribution", type(factor).__name__)
        ):
            FactoredDistribution(
                [_prior(), factor],
                label="model",
            )

    def test_an_unmet_given_makes_the_joint_conditional(self):
        with pytest.raises(
            ValueError, match=_mentions("'model'", "'beta'", "FactoredConditionalDistribution")
        ):
            FactoredDistribution(
                [_likelihood()],
                label="model",
            )

    def test_a_conditional_joint_leaves_a_given_unmet(self):
        with pytest.raises(ValueError, match=_mentions("'model'", "FactoredDistribution")):
            FactoredConditionalDistribution(
                [_likelihood(), _prior()],
                label="model",
            )

    def test_the_constructor_applies_the_merge_rules(self):
        with pytest.raises(ValueError, match=_mentions("lik takes beta", "Write lik & prior")):
            FactoredDistribution(
                [_prior(), _likelihood()],
                label="model",
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "cannot merge factor 0 ('prior') and factor 1 ('prior'), since both have the "
                "output 'beta'. Rename it in one of them, as rename({'beta': 'beta_2'})"
            ),
        ):
            FactoredDistribution([_prior(), _prior()], label="model")


# -- The event spec ---------------------------------------------------------------


class TestEventDeclaration:
    """The positional form: the factors' outputs in factor order, each factor's own order kept."""

    def test_the_event_is_an_exposed_record_of_every_factor_output(self):
        joint = (_likelihood() & _prior()) & _law("other", "c")
        assert _has_record_outputs(joint.event_spec)
        assert isinstance(joint.event_spec.term_spec, RecordSpec)
        assert list(joint.event_spec.outputs) == ["y", "beta", "c"]

    def test_each_output_keeps_its_factor_spec(self):
        lik, prior = _likelihood(), _prior()
        outputs = (lik & prior).event_spec.outputs
        assert outputs["y"] == lik.event_spec.outputs["y"]
        assert outputs["beta"] == prior.event_spec.outputs["beta"]

    def test_each_factor_keeps_its_output_order(self):
        unsorted = Law(
            OutputSpec(RecordSpec(zeta=SCALAR, alpha=SCALAR)),
            label="unsorted",
        )
        joint = (_law("m", "m") & unsorted) & _law("a", "a")
        assert list(joint.event_spec.outputs) == ["m", "zeta", "alpha", "a"]

    def test_a_single_whole_term_factor_is_exposed_as_a_record(self):
        joint = FactoredDistribution(
            [Normal("a", 0.0, 1.0)],
            label="model",
        )
        assert _has_record_outputs(joint.event_spec)
        assert list(joint.event_spec.outputs) == ["a"]

    def test_a_whole_record_factor_contributes_one_output(self):
        params = RecordSpec(u=SCALAR, v=SCALAR)
        joint = _law("a", "a") & _law("p", "params", params)
        assert list(joint.event_spec.outputs) == ["a", "params"]
        assert joint.event_spec.outputs["params"] == params

    def test_the_joint_keeps_each_factor_form(self):
        exposed = Law(
            OutputSpec(RecordSpec(beta=SCALAR)),
            label="exposed",
        )
        whole = Law(
            OutputSpec(beta=SCALAR),
            label="whole",
        )
        via_exposed, via_whole = (_likelihood() & exposed), (_likelihood() & whole)
        assert via_exposed.event_spec == via_whole.event_spec
        assert _has_record_outputs(via_exposed.factors[1].event_spec)
        assert not _has_record_outputs(via_whole.factors[1].event_spec)

    def test_a_factor_that_produces_nothing_is_kept_among_the_factors(self):
        potential = Law(
            OutputSpec(RecordSpec({})),
            label="potential",
        )
        joint = (_likelihood() & _prior()) & potential
        assert joint.factors[-1] is potential
        assert list(joint.event_spec.outputs) == ["y", "beta"]

    def test_the_numeric_views_read_each_output_of_the_joint(self):
        from probpipe import Gamma, positive, real

        joint = Normal("a", 0.0, 1.0) & Gamma("g", 2.0, 1.0)
        assert isinstance(joint, NumericDistribution)
        assert joint.supports == {"a": real, "g": positive}
        assert joint.support is None
        assert set(joint.dtypes) == {"a", "g"}


# -- Factors ------------------------------------------------------------------------


class TestFactors:
    """Both factored kinds carry ``SupportsFactors`` and list their factors conditional-first."""

    def test_both_factored_kinds_support_factors(self):
        assert isinstance((_likelihood() & _prior()), SupportsFactors)
        assert isinstance((_likelihood() & _law("other", "c")), SupportsFactors)

    def test_an_ordinary_law_or_kernel_does_not_support_factors(self):
        assert not isinstance(_prior(), SupportsFactors)
        assert not isinstance(_likelihood(), SupportsFactors)

    def test_a_consumer_precedes_its_producer(self):
        lik, prior = _likelihood(), _prior()
        assert (lik & prior).factors == (lik, prior)

    def test_a_conditional_joint_lists_its_factors(self):
        lik, other = _likelihood(), _law("other", "c")
        assert (lik & other).factors == (lik, other)


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
        assert isinstance((_likelihood(kernel) & _prior()), SupportsSampling) is samples

    def test_a_law_that_does_not_sample_leaves_the_joint_without_sampling(self):
        joint = _likelihood(SamplingKernel) & _law("prior", "beta")
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
        joint = _likelihood(kernel) & _prior()
        assert isinstance(joint, SupportsLogProb) is normalized
        assert isinstance(joint, SupportsUnnormalizedLogProb) is unnormalized

    def test_a_law_without_a_density_leaves_the_joint_without_one(self):
        joint = _likelihood(ScoringKernel) & _law("prior", "beta")
        assert not isinstance(joint, SupportsUnnormalizedLogProb)

    def test_a_conditional_joint_claims_the_conditional_twins(self):
        joint = _likelihood(FullKernel) & Normal("c", 0.0, 1.0)
        assert isinstance(joint, SupportsConditionalSampling)
        assert isinstance(joint, SupportsConditionalLogProb)
        assert not isinstance(joint, SupportsSampling)
        assert not isinstance(joint, SupportsUnnormalizedLogProb)

    def test_a_conditional_joint_samples_only_when_every_factor_samples(self):
        joint = _likelihood(ScoringKernel) & Normal("c", 0.0, 1.0)
        assert not isinstance(joint, SupportsConditionalSampling)
        assert isinstance(joint, SupportsConditionalLogProb)

    def test_an_unnormalized_factor_leaves_a_conditional_joint_unnormalized(self):
        joint = _likelihood(UnnormalizedKernel) & Normal("c", 0.0, 1.0)
        assert isinstance(joint, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(joint, SupportsConditionalLogProb)


# -- Moment capabilities ----------------------------------------------------------


_MOMENTS = (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)


class TestMomentCapabilities:
    """Edge-free joints have a moment exactly when every factor does; dependent joints have none."""

    def test_an_edge_free_joint_has_every_moment_its_factors_share(self):
        joint = _law("a", "a", law=MomentLaw) & _law("b", "b", law=MomentLaw)
        assert all(isinstance(joint, moment) for moment in _MOMENTS)

    def test_a_moment_one_factor_lacks_is_absent(self):
        joint = _law("a", "a", law=MomentLaw) & _law("b", "b", law=MeanLaw)
        assert isinstance(joint, SupportsMean)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS[1:])

    def test_a_dependent_joint_has_no_moment(self):
        joint = _likelihood(MomentKernel) & _law("prior", "beta", law=MomentLaw)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS)

    def test_the_exponential_location_model_has_no_moment(self):
        # 𝔼[y] = 𝔼[exp(x)] = e^{1/2}, which the factors' means, 0 and exp(x), do not give.
        joint = _exponential_location_model()
        assert isinstance(joint, SupportsSampling)
        assert not any(isinstance(joint, moment) for moment in _MOMENTS)

    def test_an_edge_free_conditional_joint_has_the_moment_twins_its_factors_share(self):
        joint = MomentKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) & _law(
            "b", "b", law=MomentLaw
        )
        assert isinstance(joint, SupportsConditionalMean)
        assert isinstance(joint, SupportsConditionalVariance)
        assert not isinstance(joint, SupportsConditionalCovariance)
        assert not isinstance(joint, SupportsMean)

    def test_a_dependent_conditional_joint_has_no_moment_twin(self):
        kernel = MomentKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = kernel & _law("prior", "beta", law=MomentLaw)
        assert isinstance(joint, FactoredConditionalDistribution)
        assert not isinstance(joint, SupportsConditionalMean)
        assert not isinstance(joint, SupportsConditionalVariance)

    def test_the_joint_claims_no_exact_expectation(self):
        joint = _law("a", "a", law=MomentLaw) & _law("b", "b", law=MomentLaw)
        assert not isinstance(joint, SupportsExpectation)

    def test_an_edge_free_mean_is_the_record_of_the_factor_means(self):
        mean = (Normal("a", 0.0, 1.0) & Normal("b", 1.0, 2.0))._mean()
        assert list(mean.keys()) == ["a", "b"]
        assert jnp.allclose(mean["a"], 0.0)
        assert jnp.allclose(mean["b"], 1.0)

    def test_an_edge_free_variance_is_the_record_of_the_factor_variances(self):
        variance = (Normal("a", 0.0, 1.0) & Normal("b", 1.0, 2.0))._variance()
        assert list(variance.keys()) == ["a", "b"]
        assert jnp.allclose(variance["a"], 1.0)
        assert jnp.allclose(variance["b"], 4.0)

    def test_an_edge_free_covariance_is_block_diagonal_over_the_flat_event(self):
        cov = (Normal("a", 0.0, 1.0) & Normal("b", 1.0, 2.0))._cov()
        assert jnp.allclose(cov.to_dense(), jnp.diag(jnp.array([1.0, 4.0])))

    def test_the_exponential_location_mean_is_not_a_merge_of_factor_means(self):
        draws = _exponential_location_model()._sample(jax.random.PRNGKey(0), (20_000,))
        mean = float(jnp.mean(jnp.asarray(draws["y"])))
        # exp(𝔼[x]) = 1, while 𝔼[y] = e^{1/2}; sd(y) is about 2.4, a standard error of 0.017.
        assert abs(mean - float(jnp.exp(0.5))) < 0.1

    def test_each_factor_covariance_is_a_block_in_factor_order(self):
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        vector = MultivariateNormal("m", loc=jnp.zeros(2), cov=cov)
        joint_cov = (Normal("s", 0.0, 3.0) & vector)._cov()
        assert isinstance(joint_cov, DenseLinOp)
        expected = jnp.zeros((3, 3)).at[0, 0].set(9.0).at[1:, 1:].set(cov)
        assert jnp.allclose(joint_cov.to_dense(), expected)

    def test_a_record_factor_contributes_each_of_its_outputs(self):
        joint = PairMomentLaw(
            OutputSpec(RecordSpec(a=SCALAR, b=SCALAR)),
            label="pair",
        ) & Normal("c", 5.0, 1.0)
        mean, variance = joint._mean(), joint._variance()
        assert list(mean) == list(variance) == ["a", "b", "c"]
        assert [float(mean[c]) for c in mean] == [1.0, 2.0, 5.0]
        assert [float(variance[c]) for c in variance] == [3.0, 4.0, 1.0]

    def test_an_edge_free_quantile_is_assembled_per_output(self):
        first = ShiftedQuantileLaw("first", OutputSpec(a=SCALAR), 1.0)
        second = ShiftedQuantileLaw("second", OutputSpec(b=SCALAR), 10.0)
        levels = jnp.array([0.25, 0.5])
        quantiles = (first & second)._quantile(levels)
        assert list(quantiles) == ["a", "b"]
        assert jnp.allclose(quantiles["a"], levels + 1.0)
        assert jnp.allclose(quantiles["b"], levels + 10.0)

    def test_the_quantiles_of_a_joint_of_empirical_laws_keep_each_leaf(self):
        atoms = NumericRecordBatch(
            {"b": jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), "a": jnp.array([1.0, 2.0, 3.0])},
            "row",
            element=RecordSpec(b=(2,), a=()),
            label="rows",
        )
        record = EmpiricalDistribution(atoms, jnp.array([0.5, 0.25, 0.25]), label="post")
        theta = EmpiricalDistribution(jnp.array([4.0, 5.0, 6.0]), output_name="theta")
        levels = jnp.array([0.25, 0.75])
        quantiles = FactoredDistribution(
            [record, theta],
            label="j",
        )._quantile(levels)
        assert list(quantiles) == ["b", "a", "theta"]
        assert quantiles["b"].shape == (2, 2)
        expected = record._quantile(levels)
        assert jnp.allclose(quantiles["b"], expected["b"])
        assert jnp.allclose(quantiles["a"], expected["a"])
        assert jnp.allclose(quantiles["theta"], theta._quantile(levels))

    def test_the_law_of_the_one_field_of_an_empirical_record_has_its_quantiles(self):
        column = jnp.array([3.0, 1.0, 2.0, 4.0])
        atoms = NumericRecordBatch(
            {"x": column},
            "row",
            element=RecordSpec(x=()),
            label="rows",
        )
        field = _SoleField(EmpiricalDistribution(atoms, label="one"))
        levels = jnp.array([0.25, 0.5])
        expected = EmpiricalDistribution(column, output_name="x")._quantile(levels)
        assert jnp.allclose(field._quantile(levels), expected)
        assert jnp.shape(field._quantile(0.5)) == ()

    def test_an_edge_free_conditional_joint_has_its_moments_at_its_given(self):
        joint = MomentKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) & Normal("b", 1.0, 2.0)
        mean = joint._conditional_mean({"x": 0.5})
        variance = joint._conditional_variance({"x": 0.5})
        assert list(mean) == list(variance) == ["a", "b"]
        assert [float(mean[c]) for c in mean] == [0.5, 1.0]
        assert [float(variance[c]) for c in variance] == [1.0, 4.0]


# -- Marginals ----------------------------------------------------------------------


class TestMarginalCapability:
    """The unconditional joint claims marginals and resolves them per path."""

    def test_every_unconditional_joint_claims_marginals(self):
        assert isinstance((_likelihood() & _prior()), SupportsMarginals)
        assert isinstance((_law("a", "a") & _law("b", "b")), SupportsMarginals)

    def test_a_conditional_joint_claims_no_marginals(self):
        joint = _likelihood() & _law("other", "c")
        assert not isinstance(joint, SupportsConditionalMarginals)
        assert not isinstance(joint, SupportsMarginals)


class TestMarginalGuard:
    """``_marginal_guard(path)`` reports whether the marginal at the path is exact."""

    def test_a_root_factor_is_exact(self):
        assert (_likelihood() & _prior())._marginal_guard("beta").feasible is True

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
        joint = (_law("u", "a") & _law("v", "b")) & _law("w", "c")
        assert joint._marginal_guard(path).feasible is True

    def test_an_ancestrally_closed_group_is_exact(self):
        joint = (_likelihood() & _prior()) & _law("other", "c")
        assert joint._marginal_guard(("y", "beta")).feasible is True

    def test_an_output_whose_ancestor_is_another_factor_is_rejected(self):
        report = (_likelihood() & _prior())._marginal_guard("y")
        assert report.feasible is False
        assert report.description

    def test_a_group_that_is_not_ancestrally_closed_is_rejected(self):
        joint = (_likelihood() & _prior()) & _law("other", "c")
        assert joint._marginal_guard(("y", "c")).feasible is False

    def test_a_path_inside_one_factor_follows_that_factor_guard(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        joint = pair & _law("other", "c")
        assert joint._marginal_guard("a") == pair._marginal_guard("a") == Feasibility(True)
        assert joint._marginal_guard("b") == pair._marginal_guard("b")
        assert joint._marginal_guard("b").feasible is False

    def test_a_path_inside_a_whole_record_follows_that_factor_guard(self):
        params = MarginalLaw(
            "p", OutputSpec(params=RecordSpec(u=SCALAR, v=SCALAR)), exact=("params/u",)
        )
        joint = _law("other", "c") & params
        assert joint._marginal_guard("params").feasible is True
        assert joint._marginal_guard("params/u").feasible is True
        assert joint._marginal_guard("params/v").feasible is False

    def test_a_root_factor_that_a_kernel_consumes_is_still_delegated(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        joint = NormalKernel("k", {"b": SCALAR}, OutputSpec(y=SCALAR)) & pair
        assert joint._marginal_guard("a").feasible is True

    def test_a_factor_without_a_guard_is_exact_at_every_path(self):
        joint = _pair(law=TotalMarginalLaw) & _law("other", "c")
        assert joint._marginal_guard("a").feasible is True

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("gamma", id="unknown-output"),
            pytest.param("a/zzz", id="below-a-leaf"),
            pytest.param(("a", "gamma"), id="in-a-selection"),
        ],
    )
    def test_a_path_that_is_not_an_event_path_is_rejected_with_a_reason(self, path):
        joint = _pair(law=TotalMarginalLaw) & _law("other", "c")
        report = joint._marginal_guard(path)
        assert report.feasible is False
        assert "not an event path" in report.description

    def test_a_path_inside_a_factor_without_marginals_is_rejected(self):
        report = (_pair() & _law("other", "c"))._marginal_guard("a")
        assert report.feasible is False
        assert report.description

    def test_a_target_spanning_independent_factors_follows_each_factor_guard(self):
        left = _pair("left", law=MarginalLaw, exact=("a",))
        right = MarginalLaw("right", OutputSpec(RecordSpec(c=SCALAR, d=SCALAR)), exact=("c",))
        joint = left & right
        assert joint._marginal_guard(("a", "c")).feasible is True
        assert joint._marginal_guard(("b", "c")).feasible is False

    @pytest.mark.pending(
        reason="a field no factor consumes integrates out within its kernel",
        raises=AssertionError,
    )
    def test_a_sibling_of_a_dependent_field_integrates_out_within_its_kernel(self):
        lik = MarginalKernel("lik", {"beta": SCALAR}, OutputSpec(RecordSpec(y1=SCALAR, y2=SCALAR)))
        assert (lik & _prior())._marginal_guard(("y1", "beta")).feasible is True


class TestFactorGuards:
    """A joint's derived capability takes every factor's guard of the call it makes."""

    def test_a_joint_of_unguarded_factors_is_feasible(self):
        joint = _likelihood(SamplingKernel) & _prior()
        assert _capability_guard(joint, "_sample") == Feasibility(True)

    def test_a_conditional_factor_is_guarded_through_its_twin(self):
        joint = _likelihood(UndecidedSamplingKernel) & _prior()
        assert _capability_guard(joint, "_sample") == Feasibility(
            None,
            pending=(
                "whether conditional_sample() is available for UndecidedSamplingKernel depends "
                "on values not yet known; it requires: draws once its scale is known",
            ),
        )

    def test_a_rejecting_factor_rejects_the_joint_whatever_the_others_report(self):
        point = RejectingPointLaw("point", OutputSpec(beta=SCALAR), jnp.zeros(()))
        joint = _likelihood(UndecidedSamplingKernel) & point
        assert _capability_guard(joint, "_sample") == Feasibility(
            False,
            "sample() is not available for RejectingPointLaw; it requires: draws only from a "
            "finite point",
        )

    def test_a_conditional_joint_takes_its_factors_guards(self):
        kernel = UndecidedSamplingKernel("k", {"s": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel & PointLaw("p", OutputSpec(b=SCALAR), jnp.zeros(()))
        assert isinstance(joint, FactoredConditionalDistribution)
        assert _capability_guard(joint, "_conditional_sample").unresolved

    def test_given_call_keeps_a_factor_guard_unresolved(self):
        kernel = UndecidedSamplingKernel("k", {"s": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel & PointLaw("p", OutputSpec(b=SCALAR), jnp.zeros(()))
        assert _capability_guard(joint, "_conditional_sample", {"s": 1.0}).unresolved


class TestMarginalValues:
    """The exact marginal is the sub-joint of the target's ancestor closure, reduced."""

    def test_the_marginal_of_a_root_factor_is_that_factor_law(self):
        prior = _prior(scale=2.0)
        marginal = (_likelihood() & prior)._marginal("beta")
        assert marginal.event_spec == prior.event_spec
        assert jnp.allclose(marginal._log_prob(0.7), prior._log_prob(0.7))

    def test_the_marginal_of_an_ancestrally_closed_group_is_the_sub_joint(self):
        lik, prior = _likelihood(), _prior()
        marginal = ((lik & prior) & _law("other", "c"))._marginal(("y", "beta"))
        assert isinstance(marginal, SupportsFactors)
        assert marginal.factors == (lik, prior)
        assert list(marginal.event_spec.outputs) == ["y", "beta"]

    def test_the_marginal_of_an_edge_free_group_is_an_exposed_record_of_it(self):
        joint = (_law("u", "a") & _law("v", "b")) & _law("w", "c")
        marginal = joint._marginal(("a", "c"))
        assert _has_record_outputs(marginal.event_spec)
        assert list(marginal.event_spec.outputs) == ["a", "c"]

    def test_a_marginal_at_a_list_of_paths_follows_the_joint_order(self):
        joint = Normal("x", 0.0, 1.0) & Normal("z", 5.0, 2.0)
        marginal = joint._marginal(("z", "x"))
        assert list(marginal.event_spec.outputs) == ["x", "z"]
        assert marginal.event_spec == CoupledMarginal(joint, ("z", "x")).event_spec
        assert list(marginal._mean()) == ["x", "z"]
        assert jnp.allclose(marginal._cov().to_dense(), jnp.diag(jnp.array([1.0, 4.0])))
        value = {"z": jnp.asarray(4.0), "x": jnp.asarray(0.2)}
        expected = norm.logpdf(4.0, 5.0, 2.0) + norm.logpdf(0.2, 0.0, 1.0)
        assert jnp.allclose(marginal._log_prob(value), expected)

    def test_a_marginal_at_paths_in_two_groups_keeps_both_groups(self):
        joint = (Normal("t1", 0.0, 1.0) & Normal("t2", 0.0, 2.0)) & Normal("mu", 0.0, 1.0)
        hier = joint.rename({"t1": "s1/theta", "t2": "s2/theta"})
        marginal = hier._marginal(("s2/theta", "s1/theta"))
        real = Normal("x", 0.0, 1.0).event_spec.term_spec
        assert marginal.event_spec.term_spec == RecordSpec(
            s1=RecordSpec(theta=real), s2=RecordSpec(theta=real)
        )
        variance = marginal._variance()
        assert jnp.allclose(variance["s1"]["theta"], 1.0)
        assert jnp.allclose(variance["s2"]["theta"], 4.0)

    def test_a_marginal_inside_a_packaged_factor_keeps_the_factor_group(self):
        joint = (Normal("a", 0.0, 1.0) & Normal("c", 0.0, 3.0)) & Normal("e", 0.0, 1.0)
        grouped = joint.rename({"c": "g/c", "e": "g/e"})
        marginal = grouped._marginal(("a", "g/c"))
        real = Normal("x", 0.0, 1.0).event_spec.term_spec
        assert marginal.event_spec.term_spec == RecordSpec(a=real, g=RecordSpec(c=real))
        assert jnp.allclose(marginal._variance()["g"]["c"], 9.0)

    def test_a_sub_joint_selected_against_its_factor_order_keeps_the_factor_order(self):
        marginal = (_likelihood() & _prior())._marginal(("beta", "y"))
        assert list(marginal.event_spec.outputs) == ["y", "beta"]

    def test_the_marginal_inside_one_factor_is_that_factor_marginal(self):
        pair = _pair(law=MarginalLaw, exact=("a",))
        marginal = (pair & _law("other", "c"))._marginal("a")
        assert pair.marginalized == ["a"]
        assert marginal.event_spec == OutputSpec(a=SCALAR)

    def test_the_marginal_of_a_group_of_whole_factors_is_their_product_without_a_label(self):
        joint = ((_law("u", "a") & _law("v", "b")) & _law("w", "c")).with_label("model")
        marginal = joint._marginal(("a", "c"))
        assert (marginal.label, _is_named(marginal)) == ("u·w", False)
        assert marginal.notation == "u(a)·w(c)"

    def test_whole_factors_named_out_of_factor_order_form_a_product_in_factor_order(self):
        joint = ((_law("u", "a") & _law("v", "b")) & _law("w", "c")).with_label("model")
        marginal = joint._marginal(("c", "a"))
        assert isinstance(marginal, FactoredDistribution)
        assert [factor.label for factor in marginal.factors] == ["u", "w"]
        assert list(marginal.event_spec.outputs) == ["a", "c"]
        assert marginal.notation == "u(a)·w(c)"

    def test_a_dependent_pair_named_producer_first_keeps_the_factor_order(self):
        joint = (_likelihood() & _prior()).with_label("model")
        marginal = joint._marginal(("beta", "y"))
        assert list(marginal.event_spec.outputs) == ["y", "beta"]
        assert marginal.notation == "lik(y | beta)·prior(beta)"

    @pytest.mark.parametrize(
        ("path", "label"),
        [
            pytest.param("beta", "prior", id="kept-factor"),
            pytest.param(("y", "beta"), "lik·prior", id="sub-joint"),
            pytest.param(("beta",), "prior", id="selection-of-one"),
            pytest.param("record", "one", id="field-of-a-record"),
            pytest.param("params/u", "model", id="reduced-factor"),
        ],
    )
    def test_a_marginal_is_labeled_as_the_marginal_operation_labels_it(self, path, label):
        """The view at the path takes the marginal's label as well."""
        record = OneFieldNormal(
            OutputSpec(RecordSpec(record=SCALAR)),
            label="one",
        )
        params = MarginalLaw("p", OutputSpec(params=RecordSpec(u=SCALAR)), exact=("params/u",))
        joint = (((_likelihood() & _prior()) & record) & params).with_label("model")
        assert joint._marginal(path).label == label
        assert marginal_operation._derived_label({"d": joint, "paths": path}) == label
        assert CoupledMarginal(joint, path).label == label

    def test_a_selection_of_one_whole_term_is_an_exposed_record(self):
        prior = _prior()
        marginal = (_likelihood() & prior)._marginal(("beta",))
        assert marginal.event_spec == OutputSpec(RecordSpec(beta=prior.event_spec.term_spec))
        assert marginal.factors == (prior,)

    def test_a_selection_of_a_record_factor_outputs_is_that_factor_under_the_paths(self):
        pair = _pair()
        joint = pair & _law("other", "c")
        marginal = joint._marginal(("a", "b"))
        assert type(marginal) is type(pair) and marginal.spec == pair.spec
        assert marginal.label == pair.label == "pair"

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("gamma", id="unknown-output"),
            pytest.param("beta/zzz", id="below-a-leaf"),
            pytest.param(("beta", "gamma"), id="in-a-selection"),
            pytest.param("", id="empty"),
        ],
    )
    def test_a_path_that_is_not_an_event_path_raises_key_error(self, path):
        joint = _likelihood() & _prior()
        with pytest.raises(KeyError, match="not an event path"):
            joint._marginal(path)

    def test_a_projection_onto_a_one_field_record_is_the_law_of_its_field(self):
        record = OneFieldNormal(
            OutputSpec(RecordSpec(beta=SCALAR)),
            label="record",
        )
        joint = _likelihood() & record
        assert joint._marginal_guard("beta").feasible is True
        marginal = joint._marginal("beta")
        assert marginal.event_spec == OutputSpec(beta=SCALAR)
        assert jnp.allclose(marginal._log_prob(0.7), norm.logpdf(0.7, 0.0, 2.0))
        draws = marginal._sample(jax.random.PRNGKey(0), (3,))
        assert jnp.allclose(draws, record._sample(jax.random.PRNGKey(0), (3,))["beta"])

    def test_the_law_of_a_record_field_draws_the_nested_mapping_of_its_leaves(self):
        fields = OutputSpec(RecordSpec(params=RecordSpec(u=SCALAR, v=SCALAR)))
        draw = Record(
            {"params": {"u": jnp.asarray(1.0), "v": jnp.asarray(2.0)}},
            label="draw",
        )
        joint = PointLaw("record", fields, draw) & _law("other", "c")
        value = joint._marginal("params")._sample(jax.random.PRNGKey(0))
        assert isinstance(value, dict) and list(value) == ["u", "v"]
        assert (float(value["u"]), float(value["v"])) == (1.0, 2.0)

    def test_a_marginal_the_guard_rejects_raises_naming_the_fields(self):
        with pytest.raises(ResolutionError, match=_mentions("['y']", "['beta']")):
            (_likelihood() & _prior())._marginal("y")


# -- Conditioning a conditional joint -----------------------------------------------


def _sigma_model() -> tuple[NormalKernel, Normal, FactoredConditionalDistribution]:
    """``y | beta, sigma`` merged with a prior on ``beta``, so ``sigma`` stays unmet."""
    lik = NormalKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
    prior = _prior()
    return lik, prior, (lik & prior)


class TestConditioning:
    """A conditional joint binds its givens in every factor that names them."""

    def test_binding_every_given_yields_a_factored_distribution(self):
        _, prior, joint = _sigma_model()
        bound = joint._condition_on({"sigma": 1.0})
        assert isinstance(bound, FactoredDistribution)
        assert list(bound.event_spec.outputs) == ["y", "beta"]
        kernel, kept = bound.factors
        assert kept is prior
        assert isinstance(kernel, ConditionalDistribution)
        assert list(kernel.given_spec) == ["beta"]

    def test_binding_some_givens_returns_a_smaller_conditional_joint(self):
        kernel = NormalKernel("k", {"s1": SCALAR, "s2": SCALAR}, OutputSpec(a=SCALAR))
        bound = (kernel & _law("other", "b"))._condition_on({"s1": 1.0})
        assert isinstance(bound, FactoredConditionalDistribution)
        assert list(bound.given_spec) == ["s2"]
        assert list(bound.event_spec.outputs) == ["a", "b"]

    def test_binding_in_two_steps_equals_binding_at_once(self):
        kernel = NormalKernel("k", {"s1": SCALAR, "s2": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel & _law("other", "b")
        stepwise = joint._condition_on({"s1": 1.0})._condition_on({"s2": 2.0})
        at_once = joint._condition_on({"s1": 1.0, "s2": 2.0})
        assert stepwise.spec == at_once.spec
        assert float(stepwise.factors[0].loc) == float(at_once.factors[0].loc) == 3.0

    def test_every_given_value_arrives_in_the_given(self):
        _, _, joint = _sigma_model()
        assert isinstance(joint._condition_on({"sigma": 1.0}), FactoredDistribution)

    def test_a_keyword_is_an_option_and_binds_no_input(self):
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
        (lik & _prior())._condition_on({"sigma": 1.0}, budget=3)
        assert seen == [{"budget": 3}]

    def test_a_bound_value_binds_the_factor_that_names_it(self):
        kernel = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        bound = (kernel & _law("other", "b"))._condition_on({"x": 0.25})
        assert float(bound.factors[0].loc) == 0.25

    @pytest.mark.parametrize("name", ["beta", "y", "gamma"])
    def test_a_name_that_is_not_an_input_of_the_joint_raises(self, name):
        _, _, joint = _sigma_model()
        with pytest.raises(KeyError, match=_mentions(f"'{name}'", "input")):
            joint._condition_on({name: 0.0})

    def test_the_bound_joint_derives_its_capabilities_from_the_bound_factors(self):
        joint = SamplingKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) & Normal("b", 0.0, 1.0)
        bound = joint._condition_on({"x": 0.0})
        assert isinstance(bound, SupportsSampling)
        assert isinstance(bound, SupportsMean)

    def test_a_bound_factor_that_breaks_its_kernel_declaration_raises(self):
        kernel = RenamingKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        joint = kernel & _law("other", "b")
        with pytest.raises(ValueError):
            joint._condition_on({"x": 0.0})


# -- Numeric markers --------------------------------------------------------------


class TestNumericMarkers:
    """Membership in each factored numeric marker is read from the joint's specs."""

    def test_a_joint_with_a_numeric_event_is_a_factored_numeric_distribution(self):
        joint = Normal("a", 0.0, 1.0) & Normal("b", 0.0, 1.0)
        assert isinstance(joint, FactoredNumericDistribution)
        assert isinstance(joint, NumericDistribution)

    def test_a_non_numeric_output_leaves_the_joint_outside_the_marker(self):
        joint = Normal("a", 0.0, 1.0) & _law("o", "o", OpaqueSpec())
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
        joint = kernel & Normal("b", 0.0, 1.0)
        assert {marker for marker in _CONDITIONAL_MARKERS if isinstance(joint, marker)} == markers

    def test_an_unconditional_joint_is_in_no_conditional_marker(self):
        joint = Normal("a", 0.0, 1.0) & Normal("b", 0.0, 1.0)
        assert not any(isinstance(joint, marker) for marker in _CONDITIONAL_MARKERS)

    def test_a_conditional_joint_is_not_a_factored_numeric_distribution(self):
        joint = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR)) & Normal("b", 0.0, 1.0)
        assert not isinstance(joint, FactoredNumericDistribution)
        assert isinstance(joint, ConditionalNumericDistribution)
        assert isinstance(joint, NumericConditionalDistribution)

    def test_a_kernel_that_is_not_factored_is_in_no_factored_marker(self):
        kernel = NormalKernel("k", {"x": SCALAR}, OutputSpec(a=SCALAR))
        assert isinstance(kernel, ConditionalNumericDistribution)
        assert not any(isinstance(kernel, marker) for marker in _CONDITIONAL_MARKERS)

    def test_a_class_inheriting_the_marker_constructs_as_itself(self):
        joint = NumericJoint(
            [Normal("a", 0.0, 1.0)],
            label="model",
        )
        assert isinstance(joint, NumericJoint)
        assert tuple(joint.factors[0].event_spec.outputs) == ("a",)

    def test_a_class_inheriting_the_marker_has_its_claim_checked(self):
        with pytest.raises(TypeError, match="inherits FactoredNumericDistribution"):
            NumericJoint(
                [_law("o", "o", OpaqueSpec())],
                label="model",
            )


# -- Dimension transforms ---------------------------------------------------------


def _symbolic_joint() -> FactoredDistribution:
    """Two laws over ``(n,)`` arrays and one over a scalar, with ``n`` free."""
    return (_law("la", "a", SYMBOLIC) & _law("lb", "b", SYMBOLIC)) & _law("lc", "c")


def _symbolic_conditional_joint() -> FactoredConditionalDistribution:
    """A kernel on an ``(n,)`` given beside a law over ``(n,)`` arrays, with ``n`` free."""
    kernel = NormalKernel("k", {"x": SYMBOLIC}, OutputSpec(a=SYMBOLIC))
    return kernel & _law("lb", "b", SYMBOLIC)


class TestDimensionTransforms:
    """``with_dim_sizes`` and ``with_dim_names`` apply to each factor declaring the dimension."""

    def test_binding_a_dimension_binds_it_in_every_factor(self):
        joint = _symbolic_joint()
        bound = joint.with_dim_sizes(n=3)
        assert isinstance(bound, FactoredDistribution)
        assert bound.label == joint.label
        assert [factor.event_spec.term_spec.shape for factor in bound.factors] == [(3,), (3,), ()]
        assert bound.event_spec.term_spec.free_dims == frozenset()

    def test_renaming_a_dimension_renames_it_in_every_factor(self):
        renamed = _symbolic_joint().with_dim_names(n="m")
        assert [factor.event_spec.term_spec.shape for factor in renamed.factors] == [
            ("m",),
            ("m",),
            (),
        ]
        assert renamed.event_spec.term_spec.free_dims == {"m"}

    def test_the_declaration_follows_the_transformed_factors(self):
        bound = _symbolic_joint().with_dim_sizes(n=3)
        assert (
            bound.spec
            == FactoredDistribution(
                bound.factors,
                label=bound.label,
            ).spec
        )

    def test_binding_a_dimension_that_is_not_free_raises(self):
        with pytest.raises(ValueError, match=_mentions("'k'", "free dimension")):
            _symbolic_joint().with_dim_sizes(k=3)

    def test_the_original_joint_is_unchanged(self):
        joint = _symbolic_joint()
        joint.with_dim_sizes(n=3)
        joint.with_dim_names(n="m")
        assert joint.event_spec.term_spec.free_dims == {"n"}
        assert [factor.event_spec.term_spec.shape for factor in joint.factors] == [
            ("n",),
            ("n",),
            (),
        ]

    def test_binding_a_conditional_joint_dimension_binds_it_in_every_factor(self):
        bound = _symbolic_conditional_joint().with_dim_sizes(n=3)
        assert bound.given_spec["x"].shape == (3,)
        assert [factor.event_spec.term_spec.shape for factor in bound.factors] == [(3,), (3,)]
        assert bound.factors[0].given_spec["x"].shape == (3,)

    def test_renaming_a_conditional_joint_dimension_renames_it_in_every_factor(self):
        renamed = _symbolic_conditional_joint().with_dim_names(n="m")
        assert renamed.given_spec["x"].shape == ("m",)
        assert [factor.event_spec.term_spec.shape for factor in renamed.factors] == [("m",), ("m",)]


class TestPathRenames:
    """A joint renames an output through the factors that produce and consume it."""

    def test_renaming_an_output_renames_its_producer_and_its_consumers(self):
        renamed = (_likelihood() & _prior()).rename(beta="theta")
        assert isinstance(renamed, FactoredDistribution)
        assert list(renamed.event_spec.outputs) == ["y", "theta"]
        consumer, producer = renamed.factors
        assert list(consumer.given_spec) == ["theta"]
        assert list(producer.event_spec.outputs) == ["theta"]

    def test_renaming_an_unmet_given_renames_it_in_every_factor_that_names_it(self):
        lik = NormalKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        noise = NormalKernel("noise", {"sigma": SCALAR}, OutputSpec(e=SCALAR))
        renamed = ((lik & noise) & _prior()).rename(sigma="s")
        assert isinstance(renamed, FactoredConditionalDistribution)
        assert list(renamed.given_spec) == ["s"]
        assert [sorted(getattr(f, "given_spec", {})) for f in renamed.factors] == [
            ["beta", "s"],
            ["s"],
            [],
        ]

    def test_a_rename_keeps_the_graph_and_the_label(self):
        joint = _likelihood() & _prior()
        renamed = joint.rename(beta="theta", y="obs")
        assert renamed.label == joint.label
        assert [edge[:2] for edge in renamed._graph.edges] == [
            edge[:2] for edge in joint._graph.edges
        ]
        assert [info.label for info in renamed.provenance.parents] == [joint.label]

    def test_gathering_outputs_of_two_factors_regroups_them(self, key):
        """The node ``g`` is one factor: the sub-joint of the two factors, packaged as ``g``."""
        joint = Normal("a", 0.0, 1.0, label="a") & Normal("b", 2.0, 1.0, label="b")
        renamed = joint.rename({"a": "g/a", "b": "g/b"})
        assert isinstance(renamed, SupportsFactors)
        assert list(renamed.event_spec.outputs) == ["g"]
        (group,) = renamed.factors
        assert [part.label for part in group.factors] == ["a", "b"]
        assert list(group.event_spec.outputs) == ["g"]
        assert not _has_record_outputs(group.event_spec)
        assert list(renamed._sample(key)["g"]) == ["a", "b"]


# -- Notation -----------------------------------------------------------------------


class TestNotation:
    """An unlabeled joint reads factor by factor, and a labeled one by its label."""

    def test_a_merged_joint_reads_factor_by_factor(self):
        joint = _likelihood() & _prior()
        assert str(joint) == joint.notation == "lik(y | beta)·prior(beta)"
        assert joint.label == "lik·prior"

    def test_a_relabeled_joint_reads_by_its_label_and_its_outputs(self):
        model = (_likelihood() & _prior()).with_label("model")
        assert model.notation == "model(y, beta)"

    def test_a_constructed_joint_reads_by_its_label(self):
        model = FactoredDistribution(
            [_likelihood(), _prior()],
            label="model",
        )
        assert model.notation == "model(y, beta)"

    def test_a_joint_relabeled_with_its_own_label_reads_by_it(self):
        joint = _likelihood() & _prior()
        assert joint.with_label(joint.label).notation == "(lik·prior)(y, beta)"

    def test_a_conditional_joint_reads_as_an_unconditional_one_does(self):
        _, _, joint = _sigma_model()
        assert joint.notation == "lik(y | beta, sigma)·prior(beta)"
        assert joint.with_label("model").notation == "model(y, beta | sigma)"

    def test_a_chain_reads_one_factor_after_another_under_either_grouping(self):
        a, b, c = _law("a", "x"), _law("b", "z"), _law("c", "w")
        assert ((a & b) & c).notation == (a & (b & c)).notation == "a(x)·b(z)·c(w)"

    def test_a_labeled_factor_reads_by_its_own_notation(self):
        model = (_likelihood() & _prior()).with_label("model").rename(y="obs")
        joint = _law("d", "d") & model
        assert joint.notation == "d(d)·model(obs, beta)"

    def test_the_repr_reads_as_the_constructor_call_without_the_derived_label(self):
        """The constructor derives the label from the factors again, so the repr leaves it out."""
        text = repr(_likelihood() & _prior())
        assert text.startswith("FactoredDistribution(\n    factors=(")
        assert "label='lik·prior'" not in text
        assert text.endswith("),\n)")

    def test_the_repr_shows_an_alias_as_a_keyword(self):
        text = repr((_likelihood() & _prior()).with_label("model"))
        assert text.endswith("    label='model',\n)")

    def test_an_unlabeled_joint_that_holds_paths_fixed_reads_by_its_label(self):
        """Its factors' notations would leave out the fixed paths."""
        joint = with_fixed_paths((_law("a", "x") & _law("b", "z")), "y")
        assert joint.notation == "(a·b)(x, z; y)"
        model = with_fixed_paths((_law("a", "x") & _law("b", "z")).with_label("model"), "y")
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
            pytest.param(lambda j: j.rename(a="u"), id="rename-through-factors"),
            pytest.param(lambda j: j.rename({"a": "g/a", "b": "g/b"}), id="regroup"),
            pytest.param(lambda j: copy.copy(j), id="copy"),
            pytest.param(lambda j: pickle.loads(pickle.dumps(j)), id="pickle"),
        ],
    )
    def test_a_derived_joint_keeps_the_flag(self, derive):
        unlabeled, labeled = self._joints()
        assert _is_named(derive(unlabeled)) is False
        assert _is_named(derive(labeled)) is True

    def test_a_marginal_over_several_whole_factors_is_unlabeled_whether_or_not_the_joint_is(
        self,
    ):
        for joint in self._joints():
            assert _is_named(joint._marginal(("a", "b"))) is False

    def test_a_packaged_sub_joint_is_labeled_as_the_joint_is(self):
        unlabeled, labeled = self._joints()
        for joint, named in ((unlabeled, False), (labeled, True)):
            regrouped = joint.rename({"a": "g/a", "b": "g/b"})
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
            pytest.param(lambda j: j.rename(a="u"), id="rename-through-factors"),
            pytest.param(lambda j: j.rename({"a": "g/a", "b": "g/b"}), id="regroup"),
            pytest.param(lambda j: j._marginal(("a", "b")), id="marginal-over-two-factors"),
        ],
    )
    def test_a_derived_joint_keeps_the_fixed_paths(self, derive):
        joint = with_fixed_paths(_symbolic_joint(), "y")
        assert _fixed_paths(derive(joint)) == ("y",)

    def test_a_bound_conditional_joint_keeps_the_fixed_paths(self):
        _, _, joint = _sigma_model()
        assert _fixed_paths(with_fixed_paths(joint, "y")._condition_on({"sigma": 1.0})) == ("y",)


# -- Round trips ----------------------------------------------------------------------


_JOINTS = [
    pytest.param(lambda: _likelihood(FullKernel) & _prior(), id="dependent"),
    pytest.param(lambda: Normal("a", 0.0, 1.0) & Normal("b", 1.0, 2.0), id="edge-free"),
    pytest.param(lambda: _likelihood(FullKernel) & Normal("c", 0.0, 1.0), id="conditional"),
    pytest.param(lambda: (_likelihood() & _prior()).with_label("posterior"), id="relabeled"),
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
    """A pickled or copied joint keeps its label, specs, factors, and capabilities."""

    @pytest.mark.parametrize("make", _JOINTS)
    def test_pickle(self, make):
        joint = make()
        _assert_same_joint(pickle.loads(pickle.dumps(joint)), joint)

    @pytest.mark.parametrize("make", _JOINTS)
    def test_copy(self, make):
        joint = make()
        _assert_same_joint(copy.copy(joint), joint)

    def test_a_restored_conditional_joint_still_binds_some_givens(self):
        _, _, joint = _sigma_model()
        bound = pickle.loads(pickle.dumps(joint))._condition_on({"sigma": 1.0})
        assert isinstance(bound, FactoredDistribution)
        assert list(bound.event_spec.outputs) == ["y", "beta"]


# -- Joint sampling and densities -------------------------------------------------


class TestJointSampling:
    """A draw extracts each factor's outputs from its one draw, in canonical factor order."""

    def test_a_draw_holds_every_output_in_canonical_order(self):
        joint = (_likelihood(SamplingKernel) & _prior()) & Normal("c", 0.0, 1.0)
        draw = joint._sample(jax.random.PRNGKey(0))
        assert joint.event_spec.term_spec.is_valid(draw)
        assert list(draw.keys()) == ["y", "beta", "c"]

    def test_a_draw_extracts_outputs_by_each_factor_form(self):
        fields = OutputSpec(RecordSpec(a=SCALAR, b=SCALAR))
        exposed = PointLaw("exposed", fields, {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)})
        params = OutputSpec(params=RecordSpec(u=SCALAR, v=SCALAR))
        whole = PointLaw("whole", params, {"u": jnp.asarray(3.0), "v": jnp.asarray(4.0)})
        draw = (exposed & whole)._sample(jax.random.PRNGKey(0))
        assert list(draw.keys()) == ["a", "b", "params"]
        assert (float(draw["a"]), float(draw["b"])) == (1.0, 2.0)
        assert (float(draw["params"]["u"]), float(draw["params"]["v"])) == (3.0, 4.0)

    def test_every_consumer_reads_the_draw_its_producer_reports(self):
        first = SamplingKernel("first", {"beta": SCALAR}, OutputSpec(y1=SCALAR), scale=1e-6)
        second = SamplingKernel("second", {"beta": SCALAR}, OutputSpec(y2=SCALAR), scale=1e-6)
        draw = ((first & second) & _prior())._sample(jax.random.PRNGKey(1))
        assert jnp.allclose(draw["y1"], draw["beta"], atol=1e-4)
        assert jnp.allclose(draw["y2"], draw["beta"], atol=1e-4)

    def test_a_sample_shape_prepends_batch_axes_to_every_output(self):
        draws = (_likelihood(SamplingKernel) & _prior())._sample(jax.random.PRNGKey(2), (4,))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (4,)

    def test_a_conditional_joint_samples_under_its_given(self):
        lik = SamplingKernel(
            "lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6
        )
        draw = (lik & _prior())._conditional_sample({"sigma": 10.0}, jax.random.PRNGKey(3))
        assert list(draw.keys()) == ["y", "beta"]
        assert jnp.allclose(draw["y"], draw["beta"] + 10.0, atol=1e-4)

    def test_one_key_reproduces_the_draw(self):
        joint = (_likelihood(SamplingKernel) & _prior()) & Normal("c", 0.0, 1.0)
        first, again = joint._sample(jax.random.PRNGKey(4)), joint._sample(jax.random.PRNGKey(4))
        other = joint._sample(jax.random.PRNGKey(5))
        assert all(jnp.array_equal(first[c], again[c]) for c in first)
        assert not any(jnp.array_equal(first[c], other[c]) for c in first)

    def test_each_factor_draws_with_its_own_key_of_one_split(self):
        lik, prior = _likelihood(SamplingKernel), _prior()
        key = jax.random.PRNGKey(6)
        draw = (lik & prior)._sample(key)
        lik_key, prior_key = jax.random.split(key, 2)
        assert jnp.array_equal(draw["beta"], prior._sample(prior_key))
        assert jnp.allclose(draw["y"], lik._conditional_sample({"beta": draw["beta"]}, lik_key))

    def test_under_a_sample_shape_each_draw_conditions_on_its_own_producer_draw(self):
        lik = SamplingKernel("lik", {"beta": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6)
        draws = (lik & _prior())._sample(jax.random.PRNGKey(7), (3, 2))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (3, 2)
        assert jnp.allclose(draws["y"], draws["beta"], atol=1e-4)
        assert len(set(jnp.ravel(draws["beta"]).tolist())) == 6

    def test_a_conditional_joint_samples_under_its_given_with_a_sample_shape(self):
        lik = SamplingKernel(
            "lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6
        )
        draws = (lik & _prior())._conditional_sample({"sigma": 10.0}, jax.random.PRNGKey(8), (5,))
        assert jnp.shape(draws["y"]) == jnp.shape(draws["beta"]) == (5,)
        assert jnp.allclose(draws["y"], draws["beta"] + 10.0, atol=1e-4)

    def test_a_kernel_on_an_opaque_output_draws_once_per_draw(self):
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
        assert jnp.allclose((lik & prior)._log_prob({"y": y, "beta": beta}), expected)

    def test_the_unnormalized_density_differs_from_the_factor_sum_by_a_constant(self):
        lik, prior = _likelihood(UnnormalizedKernel), _prior()
        joint = lik & prior

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
        joint = (_likelihood(ScoringKernel) & exposed) & whole
        joint._log_prob(
            {"y": jnp.asarray(0.3), "beta": jnp.asarray(-0.2), "gamma": jnp.asarray(0.5)}
        )
        (beta_value,) = exposed.scored
        (gamma_value,) = whole.scored
        assert exposed.event_spec.term_spec.is_valid(beta_value)
        assert not SCALAR.is_valid(beta_value)
        assert whole.event_spec.term_spec.is_valid(gamma_value)
        assert not exposed.event_spec.term_spec.is_valid(gamma_value)

    def test_a_conditional_joint_scores_under_its_given(self):
        lik = ScoringKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        prior = _prior()
        y, beta = jnp.asarray(0.3), jnp.asarray(-0.2)
        kernel_given = {"beta": beta, "sigma": 1.5}
        expected = lik._conditional_log_prob(kernel_given, y) + prior._log_prob(beta)
        value = {"y": y, "beta": beta}
        assert jnp.allclose((lik & prior)._conditional_log_prob({"sigma": 1.5}, value), expected)

    def test_the_log_density_agrees_with_a_hand_computed_sum(self):
        joint = _likelihood(ScoringKernel, scale=0.5) & _prior(scale=2.0)
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
        joint = _likelihood(ScoringKernel) & _prior(scale=2.0)
        values = {"y": jnp.array([0.3, -1.0, 2.0]), "beta": jnp.array([-0.2, 0.5, 1.0])}
        scores = joint._log_prob(values)
        assert jnp.shape(scores) == (3,)
        for index in range(3):
            one = {name: column[index] for name, column in values.items()}
            assert jnp.allclose(scores[index], joint._log_prob(one))

    def test_a_conditional_joint_scores_a_batch_under_its_given(self):
        lik = ScoringKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = lik & _prior()
        values = {"y": jnp.array([0.3, 1.2]), "beta": jnp.array([-0.2, 0.4])}
        scores = joint._conditional_log_prob({"sigma": 1.5}, values)
        assert jnp.shape(scores) == (2,)
        for index in range(2):
            one = {name: column[index] for name, column in values.items()}
            assert jnp.allclose(scores[index], joint._conditional_log_prob({"sigma": 1.5}, one))

    def test_a_kernel_that_does_not_broadcast_scores_each_value_at_its_own_given(self):
        joint = _likelihood(NormKernel) & _prior(scale=2.0)
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
        scores = (lik & _prior())._conditional_log_prob({"sigma": 0.5}, {"y": y, "beta": beta})
        # log N(y_i; |beta_i|, 0.5^2) + log N(beta_i; 0, 1), written out.
        expected = (
            -0.5 * ((y - jnp.abs(beta)) / 0.5) ** 2
            - jnp.log(0.5)
            - 0.5 * beta**2
            - jnp.log(2.0 * jnp.pi)
        )
        assert jnp.allclose(scores, expected)

    def test_a_kernel_on_an_opaque_output_scores_value_by_value(self):
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
        assert jnp.allclose((first & second)._log_prob(value), expected)


def _counts_model() -> FactoredDistribution:
    """A population path over 8 steps, written with ``lax.scan``, observed as Poisson counts."""
    prior = Normal("r", 0.4, 0.05) & Normal("K", 100.0, 5.0)

    @conditional_distribution(given_spec=prior.event_spec.term_spec, label="counts")
    def counts(r, K, n0=20.0):
        def step(n, _):
            following = n * jnp.exp(r * (1.0 - n / K))
            return following, following

        _, path = jax.lax.scan(step, jnp.full((), n0, jnp.result_type(r)), None, length=8)
        return Poisson("y", path)

    return counts & prior


def _offset_kernel(
    kernel: type[NormalKernel], spec: Any, offset: Callable[[Any], Any], calls: list
) -> NormalKernel:
    """``y | beta, extra ~ Normal(beta + offset(extra), 1e-6)``, recording each call of its location."""

    def location(values: Mapping[str, Any]) -> Any:
        calls.append(values)
        return values["beta"] + offset(values["extra"])

    given = {"beta": SCALAR, "extra": spec}
    return kernel("lik", given, OutputSpec(y=SCALAR), loc=location, scale=1e-6)


class _Holder:
    """A mutable object an unmet given holds."""

    def __init__(self, values: list[float]) -> None:
        self.values = values


def _ran_uncompiled(joint: FactoredDistribution) -> bool:
    """Whether a conditional factor of *joint* records a program that does not trace."""
    return any(
        _program_store_of(factor).runs_eagerly(("mapped", method))
        for factor in joint.factors
        if isinstance(factor, ConditionalDistribution)
        for method in ("_conditional_sample", "_conditional_log_prob")
    )


class TestCompiledFactorCalls:
    """A joint maps a kernel over draws through one compiled program, which the kernel keeps."""

    def test_a_warm_draw_compiles_nothing(self):
        model = _counts_model()
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        model._sample(keys[0], (16,))
        with count_compilations() as compiled:
            model._sample(keys[1], (16,))
        assert compiled.count == 0

    def test_a_warm_density_of_a_batch_compiles_nothing(self):
        model = _counts_model()
        first, second = (
            model._sample(key, (16,)) for key in jax.random.split(jax.random.PRNGKey(1))
        )
        model._log_prob(first)
        with count_compilations() as compiled:
            model._log_prob(second)
        assert compiled.count == 0

    def test_the_compiled_draws_and_densities_equal_the_uncompiled_ones(self):
        model = _counts_model()
        key = jax.random.PRNGKey(2)
        draws = model._sample(key, (32,))
        with jax.disable_jit():
            uncompiled = model._sample(key, (32,))
            uncompiled_scores = model._log_prob(draws)
        # XLA's fusion changes the last float32 digit: observed across 6 seeds, at most a
        # relative 1.5e-7, and every count equal. The tolerance is the family tests', which
        # a wrong key or parameter, off by order 1, exceeds. A count can change where a
        # rate differs in its last digit, so most counts, not all, must agree.
        assert np.mean(np.asarray(draws["y"]) == np.asarray(uncompiled["y"])) >= 0.95
        for output_name in ("r", "K"):
            np.testing.assert_allclose(
                draws[output_name], uncompiled[output_name], rtol=2e-6, atol=2e-6
            )
        np.testing.assert_allclose(model._log_prob(draws), uncompiled_scores, rtol=2e-6, atol=2e-6)

    @pytest.mark.parametrize("held", [float, jnp.asarray], ids=["number", "array"])
    def test_each_draw_reads_the_unmet_given_of_its_own_call(self, held):
        lik = SamplingKernel(
            "lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR), scale=1e-6
        )
        joint = lik & _prior()
        for sigma in (10.0, -3.0, 0.5):
            draws = joint._conditional_sample({"sigma": held(sigma)}, jax.random.PRNGKey(3), (4,))
            np.testing.assert_allclose(draws["y"], draws["beta"] + sigma, atol=1e-4)

    @pytest.mark.parametrize("held", [float, jnp.asarray], ids=["number", "array"])
    def test_a_new_unmet_given_does_not_trace_the_kernel_again(self, held):
        calls: list = []
        joint = _offset_kernel(SamplingKernel, SCALAR, lambda extra: extra, calls) & _prior()
        joint._conditional_sample({"extra": held(1.0)}, jax.random.PRNGKey(4), (4,))
        calls.clear()
        draws = joint._conditional_sample({"extra": held(2.0)}, jax.random.PRNGKey(5), (4,))
        assert calls == []
        np.testing.assert_allclose(draws["y"], draws["beta"] + 2.0, atol=1e-4)

    def test_a_string_given_is_a_constant_of_the_program(self):
        calls: list = []
        joint = _offset_kernel(SamplingKernel, OpaqueSpec(), len, calls) & _prior()
        for tag in ("abc", "abcdef", "abc"):
            draws = joint._conditional_sample({"extra": tag}, jax.random.PRNGKey(6), (4,))
            np.testing.assert_allclose(draws["y"], draws["beta"] + len(tag), atol=1e-4)
        # One trace for each distinct string: the second "abc" reuses its program.
        assert len(calls) == 2

    def test_a_string_array_given_runs_uncompiled_and_leaves_the_program_enabled(self):
        calls: list = []
        joint = _offset_kernel(SamplingKernel, OpaqueSpec(), len, calls) & _prior()
        key = jax.random.PRNGKey(6)
        draws = joint._conditional_sample({"extra": np.array(["a", "b"])}, key, (4,))
        np.testing.assert_allclose(draws["y"], draws["beta"] + 2.0, atol=1e-4)
        joint._conditional_sample({"extra": "abc"}, key, (4,))
        assert not _ran_uncompiled(joint)

    def test_an_object_given_is_read_at_every_call(self):
        calls: list = []
        joint = _offset_kernel(SamplingKernel, OpaqueSpec(), lambda h: sum(h.values), calls)
        joint = joint & _prior()
        holder = _Holder([1.0, 2.0])
        first = joint._conditional_sample({"extra": holder}, jax.random.PRNGKey(7), (4,))
        holder.values = [10.0, 20.0]
        second = joint._conditional_sample({"extra": holder}, jax.random.PRNGKey(7), (4,))
        np.testing.assert_allclose(first["y"], first["beta"] + 3.0, atol=1e-4)
        np.testing.assert_allclose(second["y"], second["beta"] + 30.0, atol=1e-4)

    def test_a_copy_holding_other_values_draws_with_its_own(self):
        prior = _prior()

        @conditional_distribution(given_spec={"beta": SCALAR, "shift": SCALAR}, label="lik")
        def lik(beta, shift):
            return Normal("y", beta + shift, 1e-6)

        (lik & prior)._conditional_sample({"shift": 0.0}, jax.random.PRNGKey(8), (4,))
        shifted = lik._condition_on({"shift": jnp.asarray(5.0)})
        draws = (shifted & prior)._sample(jax.random.PRNGKey(9), (4,))
        np.testing.assert_allclose(draws["y"], draws["beta"] + 5.0, atol=1e-4)
        # The copy builds its own program, so the kernel's still traces and compiles.
        assert shifted._memo is not lik._memo
        assert not _ran_uncompiled(lik & prior)

    def test_a_warm_density_with_a_new_unmet_given_does_not_trace_the_kernel_again(self):
        calls: list = []
        joint = _offset_kernel(ScoringKernel, SCALAR, lambda extra: extra, calls) & _prior()
        values = {"y": jnp.array([0.3, 1.2]), "beta": jnp.array([-0.2, 0.4])}
        joint._conditional_log_prob({"extra": jnp.asarray(0.5)}, values)
        calls.clear()
        scores = joint._conditional_log_prob({"extra": jnp.asarray(0.7)}, values)
        assert calls == []
        expected = norm.logpdf(values["y"], values["beta"] + 0.7, 1e-6) + norm.logpdf(
            values["beta"]
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-5)

    def test_a_family_check_of_a_parameter_computed_in_the_program_does_not_run(self):
        @conditional_distribution(given_spec={"beta": SCALAR, "w": SCALAR}, label="lik")
        def lik(beta, w):
            components = [Normal("y", beta, 1.0), Normal("y", beta + 1.0, 1.0)]
            return MixtureDistribution(components, jnp.stack([w, 1.0 - w]))

        with pytest.raises(ValueError, match="nonnegative"):
            lik._condition_on({"beta": 0.0, "w": -0.5})
        values = {"y": jnp.zeros(3), "beta": jnp.zeros(3)}
        scores = (lik & _prior())._conditional_log_prob({"w": jnp.asarray(-0.5)}, values)
        assert np.all(np.isnan(scores))

    def test_a_call_that_reuses_the_program_reads_a_global_as_it_was_compiled(self):
        shift = {"value": 0.0}

        @conditional_distribution(given_spec={"beta": SCALAR}, label="lik")
        def lik(beta):
            return Normal("y", beta + shift["value"], 1e-6)

        joint = lik & _prior()
        key = jax.random.PRNGKey(12)
        joint._sample(key, (4,))
        shift["value"] = 100.0
        reused = joint._sample(key, (4,))
        recompiled = joint._sample(key, (5,))
        np.testing.assert_allclose(reused["y"], reused["beta"], atol=1e-4)
        np.testing.assert_allclose(recompiled["y"], recompiled["beta"] + 100.0, atol=1e-3)

    @pytest.mark.parametrize(
        "how",
        [lambda term: pickle.loads(pickle.dumps(term)), copy.deepcopy],
        ids=["pickle", "deepcopy"],
    )
    def test_a_joint_that_has_drawn_round_trips(self, how):
        lik = SamplingKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = lik & _prior()
        key = jax.random.PRNGKey(10)
        draws = joint._conditional_sample({"sigma": 1.0}, key, (4,))
        restored = how(joint)._conditional_sample({"sigma": 1.0}, key, (4,))
        for output_name in draws:
            np.testing.assert_array_equal(restored[output_name], draws[output_name])

    @pytest.mark.parametrize(
        "offset",
        [float, lambda extra: jnp.sum(extra[extra > 0])],
        ids=["python-number", "boolean-mask"],
    )
    def test_a_kernel_that_does_not_trace_draws_uncompiled_and_is_not_traced_again(self, offset):
        calls: list = []
        joint = _offset_kernel(SamplingKernel, SCALAR, offset, calls) & _prior()
        first = joint._conditional_sample({"extra": jnp.asarray(2.0)}, jax.random.PRNGKey(11), (4,))
        assert len(calls) == 2
        second = joint._conditional_sample(
            {"extra": jnp.asarray(3.0)}, jax.random.PRNGKey(11), (4,)
        )
        # The first call traces the kernel, which needs the value of the unmet given, and
        # then maps it uncompiled. The second call maps it uncompiled without tracing it.
        assert len(calls) == 3
        np.testing.assert_allclose(first["y"], first["beta"] + 2.0, atol=1e-4)
        np.testing.assert_allclose(second["y"], second["beta"] + 3.0, atol=1e-4)

    def test_a_kernel_density_that_does_not_trace_scores_uncompiled(self):
        calls: list = []
        joint = _offset_kernel(ScoringKernel, SCALAR, float, calls) & _prior()
        values = {"y": jnp.array([0.3, 1.2]), "beta": jnp.array([-0.2, 0.4])}
        scores = joint._conditional_log_prob({"extra": jnp.asarray(0.5)}, values)
        joint._conditional_log_prob({"extra": jnp.asarray(0.5)}, values)
        assert len(calls) == 3
        expected = norm.logpdf(values["y"], values["beta"] + 0.5, 1e-6) + norm.logpdf(
            values["beta"]
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-5)


class TestFusedGiven:
    """A conditional joint's capabilities take a value for every input, and only those."""

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
        ("given", "input_name"),
        [
            pytest.param({}, "sigma", id="missing"),
            pytest.param({"sigma": 1.0, "tau": 1.0}, "tau", id="unknown"),
        ],
    )
    def test_a_given_that_omits_or_adds_an_input_raises(self, call, given, input_name):
        lik = FullKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        with pytest.raises(KeyError, match=_mentions(f"'{input_name}'", "'lik·prior'")):
            call((lik & _prior()), given)

    def test_a_record_given_binds_by_its_fields(self):
        lik = FullKernel("lik", {"beta": SCALAR, "sigma": SCALAR}, OutputSpec(y=SCALAR))
        joint = lik & _prior()
        value = {"y": jnp.asarray(0.3), "beta": jnp.asarray(-0.2)}
        from_record = joint._conditional_log_prob(
            Record(
                {"sigma": 1.5},
                label="given",
            ),
            value,
        )
        assert jnp.allclose(from_record, joint._conditional_log_prob({"sigma": 1.5}, value))
