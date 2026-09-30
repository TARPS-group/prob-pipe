"""The ``*`` operator: composing distributions and kernels into one flat joint.

``A * B`` is conditional-first: the left operand may condition on a component
the right operand produces. An operand is characterized by its produced slots,
the keys of its event declaration's components, and its unmet given slots. The
bound, unmet, and require rules decide the joint's connections and its kind, the
operands share one dimension scope, and the joint's label joins the operands'
labels with a middle dot. Composition reads component and slot names, never a
label.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    LinearBasisFunction,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    RecordSpec,
)
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    FactoredConditionalDistribution,
    FactoredDistribution,
    GaussianRandomFunction,
    SupportsFactors,
)

SCALAR = NumericArraySpec(())
VECTOR = NumericArraySpec((3,))
SYMBOLIC = NumericArraySpec(("n",))


# -- Laws and kernels -----------------------------------------------------------


def _total(values: Mapping[str, Any]) -> Any:
    """The location of a kernel: the sum of its given values."""
    return sum(jnp.asarray(value) for value in values.values())


class NormalKernel(ConditionalDistribution):
    """``K(given, ·) = Normal(loc(given), 1)``, a law over the kernel's own whole-term event.

    Binding some of the given slots curries to the kernel over the rest, which
    keeps the values bound so far.
    """

    def __init__(
        self,
        name: str,
        given_spec: Mapping[str, Any],
        event_spec: OutputSpec,
        *,
        loc: Callable[[Mapping[str, Any]], Any] = _total,
        bound: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name, given_spec, event_spec)
        self._loc = loc
        self._bound = dict(bound or {})

    def _condition_on(self, given, /, **kwargs):
        values = {**self._bound, **dict(given.items()), **kwargs}
        rest = {slot: spec for slot, spec in self.given_spec.items() if slot not in values}
        if rest:
            return type(self)(self.name, rest, self.event_spec, loc=self._loc, bound=values)
        (component,) = self.event_spec.components
        return Normal(self.name, self._loc(values), 1.0, event_spec=OutputSpec(**{component: None}))


class Law(Distribution):
    """A law over its declared event that claims no capability."""


def _likelihood() -> NormalKernel:
    """``y | beta``: a kernel labeled ``lik`` that conditions on ``beta`` and produces ``y``."""
    return NormalKernel("lik", {"beta": SCALAR}, OutputSpec(y=SCALAR))


def _prior() -> Normal:
    """A law labeled ``prior`` whose component is ``beta``."""
    return Normal("prior", 0.0, 1.0, event_spec=OutputSpec(beta=None))


def _law(name: str, component: str, spec: Any = SCALAR) -> Law:
    """A law labeled *name* whose whole-term component is *component*."""
    return Law(name, OutputSpec(**{component: spec}))


def _kernel(name: str, given: Mapping[str, Any], component: str, spec: Any = SCALAR):
    """A kernel labeled *name* on the slots *given* that produces *component*."""
    return NormalKernel(name, dict(given), OutputSpec(**{component: spec}))


def _features(X):
    """The quadratic feature map of the random function below."""
    return jnp.concatenate([X, X**2], -1)


def _random_function() -> LinearBasisFunction:
    """A random function labeled ``f`` whose Gaussian weights multiply ``_features``."""
    weights = MultivariateNormal("w", loc=jnp.zeros(2), cov=jnp.eye(2))
    return LinearBasisFunction(
        "f", feature_map=_features, weights=weights, input_shape=(1,), output_shape=()
    )


def _structure(joint: Any) -> tuple:
    """What composition decides: the kind, the label, the declarations, and the factors."""
    return (
        type(joint),
        joint.name,
        joint.spec,
        tuple((type(factor), factor.name, factor.spec) for factor in joint.factors),
    )


def _mentions(*fragments: str) -> str:
    """A pattern matching a message that contains every fragment, in any order."""
    return "".join(f"(?=.*{re.escape(fragment)})" for fragment in fragments)


# -- The require rules ------------------------------------------------------------


class TestRequireRules:
    """``F_A ∩ F_B = ∅`` and ``G_B ∩ F_A = ∅``: each violation raises ValueError naming the fix."""

    def test_a_component_both_operands_produce_raises(self):
        first = Normal("first", 0.0, 1.0, event_spec=OutputSpec(a=None))
        second = Normal("second", 1.0, 1.0, event_spec=OutputSpec(a=None))
        with pytest.raises(
            ValueError, match=_mentions("'a'", "'first'", "'second'", "with_path_names")
        ):
            first * second

    def test_a_component_produced_inside_both_joint_operands_raises(self):
        left = _law("u", "a") * _law("v", "b")
        right = _law("w", "c") * _law("x", "b")
        with pytest.raises(ValueError, match=_mentions("'b'", "'v'", "'x'")):
            left * right

    def test_a_kernel_and_a_law_that_produce_one_component_raise(self):
        with pytest.raises(ValueError, match=_mentions("'a'", "'k'", "'l'")):
            _kernel("k", {"x": SCALAR}, "a") * _law("l", "a")

    def test_a_producer_on_the_left_of_its_consumer_raises(self):
        with pytest.raises(
            ValueError, match=_mentions("'lik'", "'beta'", "'prior'", "producer on the right")
        ):
            _prior() * _likelihood()

    def test_a_right_operand_whose_unmet_given_the_left_produces_raises(self):
        right = _likelihood() * _law("other", "c")
        assert "beta" in right.given_spec
        with pytest.raises(
            ValueError, match=_mentions("'lik'", "'beta'", "'prior'", "producer on the right")
        ):
            _prior() * right

    def test_the_producer_on_the_right_of_its_consumer_composes(self):
        assert isinstance(_likelihood() * _prior(), FactoredDistribution)


# -- The bound rule ---------------------------------------------------------------


class TestBoundRule:
    """``bound = G_A ∩ F_B``: a given of the left operand is met by a component on the right."""

    def test_a_given_the_right_operand_produces_is_met(self):
        joint = _likelihood() * _prior()
        assert not isinstance(joint, ConditionalDistribution)

    def test_a_given_a_factor_further_right_produces_is_met(self):
        joint = _likelihood() * (_law("other", "c") * _prior())
        assert isinstance(joint, FactoredDistribution)

    @pytest.mark.parametrize(
        "event_spec",
        [
            pytest.param(OutputSpec(beta=SCALAR), id="whole-term"),
            pytest.param(OutputSpec(RecordSpec(beta=SCALAR)), id="one-field-record"),
        ],
    )
    def test_a_component_meets_a_given_under_either_packaging(self, event_spec):
        joint = _likelihood() * Law("prior", event_spec)
        assert isinstance(joint, FactoredDistribution)
        assert list(joint.event_spec.components) == ["y", "beta"]

    def test_a_whole_record_component_meets_a_record_slot(self):
        params = RecordSpec(u=SCALAR, v=SCALAR)
        joint = _kernel("lik", {"params": params}, "y") * _law("prior", "params", params)
        assert isinstance(joint, FactoredDistribution)


# -- The unmet rule and the kind of the result -----------------------------------


class TestUnmetRule:
    """``unmet = (G_A − F_B) ∪ G_B``, and the result is conditional exactly when it is not empty."""

    def test_no_unmet_given_gives_a_factored_distribution(self):
        joint = _likelihood() * _prior()
        assert isinstance(joint, FactoredDistribution)
        assert isinstance(joint, Distribution)
        assert not isinstance(joint, ConditionalDistribution)

    def test_an_unmet_given_gives_a_factored_conditional_distribution(self):
        joint = _likelihood() * _law("other", "c")
        assert isinstance(joint, FactoredConditionalDistribution)
        assert isinstance(joint, ConditionalDistribution)
        assert not isinstance(joint, Distribution)

    def test_the_given_spec_is_exactly_the_unmet_givens(self):
        left = _kernel("left", {"beta": SCALAR, "sigma": SCALAR}, "y")
        right = _kernel("right", {"tau": VECTOR}, "beta")
        assert dict((left * right).given_spec) == {"sigma": SCALAR, "tau": VECTOR}

    def test_the_unmet_givens_follow_the_factor_order(self):
        joint = _kernel("k1", {"z": SCALAR}, "a") * _kernel("k2", {"x": SCALAR, "w": SCALAR}, "b")
        assert list(joint.given_spec) == ["z", "x", "w"]

    def test_the_kind_is_recomputed_when_a_later_operand_meets_the_givens(self):
        pair = _kernel("k1", {"x": SCALAR}, "a") * _kernel("k2", {"x": SCALAR}, "b")
        assert isinstance(pair, FactoredConditionalDistribution)
        assert isinstance(pair * _law("p", "x"), FactoredDistribution)


# -- Same-named unmet givens ------------------------------------------------------


class TestSameNamedGivens:
    """Same-named unmet givens are one slot, and binding it binds each factor that names it."""

    def test_same_named_givens_unify_whichever_dtype_is_listed_first(self):
        wide = _kernel("k1", {"x": NumericArraySpec((), np.float32)}, "a")
        narrow = _kernel("k2", {"x": NumericArraySpec((), np.int32)}, "b")
        assert list((wide * narrow).given_spec) == list((narrow * wide).given_spec) == ["x"]

    def test_same_named_unmet_givens_are_one_slot(self):
        joint = _kernel("k1", {"x": SCALAR}, "a") * _kernel("k2", {"x": SCALAR}, "b")
        assert dict(joint.given_spec) == {"x": SCALAR}

    def test_a_disagreement_between_their_specs_raises(self):
        with pytest.raises(ValueError, match=_mentions("'x'")):
            _kernel("k1", {"x": SCALAR}, "a") * _kernel("k2", {"x": VECTOR}, "b")

    def test_the_slot_carries_the_first_factor_spec(self):
        typed = NumericArraySpec((), "float32")
        joint = _kernel("k1", {"x": SCALAR}, "a") * _kernel("k2", {"x": typed}, "b")
        assert joint.given_spec["x"] == SCALAR

    def test_a_polymorphic_given_binds_to_a_concrete_one_listed_later(self):
        polymorphic = _kernel("k1", {"x": SYMBOLIC}, "a", SYMBOLIC)
        joint = polymorphic * _kernel("k2", {"x": VECTOR}, "b")
        assert joint.given_spec["x"] == VECTOR
        assert joint.factors[0].event_spec.spec == VECTOR

    def test_binding_the_slot_binds_it_in_every_factor_that_names_it(self):
        joint = _kernel("k1", {"x": SCALAR}, "a") * _kernel("k2", {"x": SCALAR}, "b")
        bound = joint._condition_on({"x": 0.5})
        assert isinstance(bound, FactoredDistribution)
        assert [float(factor.loc) for factor in bound.factors] == [0.5, 0.5]

    def test_same_named_polymorphic_givens_share_one_free_dimension(self):
        joint = _kernel("k1", {"x": SYMBOLIC}, "a") * _kernel("k2", {"x": SYMBOLIC}, "b")
        assert joint.given_spec["x"] == SYMBOLIC
        assert joint.spec.free_dims == {"n"}

    def test_the_unified_slot_does_not_depend_on_the_operand_order(self):
        polymorphic = _kernel("k1", {"x": SYMBOLIC}, "a", SYMBOLIC)
        concrete = _kernel("k2", {"x": VECTOR}, "b")
        left, right = polymorphic * concrete, concrete * polymorphic
        assert left.given_spec["x"] == right.given_spec["x"] == VECTOR

    def test_givens_that_are_different_quantities_are_renamed_apart_first(self):
        second = _kernel("k2", {"x": SCALAR}, "b").with_path_names(x="w")
        joint = _kernel("k1", {"x": SCALAR}, "a") * second
        assert list(joint.given_spec) == ["x", "w"]


# -- One dimension scope ----------------------------------------------------------


class TestOneDimensionScope:
    """The operands share one dimension scope: a name two factors use is one dimension."""

    def test_a_dimension_two_factors_declare_is_one_free_dimension(self):
        joint = _law("la", "a", SYMBOLIC) * _law("lb", "b", SYMBOLIC)
        assert joint.event_spec.spec.free_dims == {"n"}
        bound = joint.with_dim_sizes(n=3)
        assert [spec.shape for spec in bound.event_spec.components.values()] == [(3,), (3,)]

    def test_a_matched_slot_binds_the_dimension_in_every_factor_that_declares_it(self):
        consumer = _kernel("k", {"x": SYMBOLIC}, "a", SYMBOLIC)
        joint = consumer * _law("l", "b", SYMBOLIC) * _law("p", "x", VECTOR)
        shapes = {component: spec.shape for component, spec in joint.event_spec.components.items()}
        assert shapes == {"a": (3,), "b": (3,), "x": (3,)}
        assert [factor.event_spec.spec.shape for factor in joint.factors] == [(3,), (3,), (3,)]
        assert joint.factors[0].given_spec["x"] == VECTOR

    def test_a_dimension_bound_to_two_sizes_raises(self):
        first = _kernel("k1", {"x": SYMBOLIC}, "a")
        second = _kernel("k2", {"w": SYMBOLIC}, "b")
        producers = _law("p", "x", VECTOR) * _law("q", "w", NumericArraySpec((4,)))
        with pytest.raises(ValueError, match=_mentions("'n'", "3", "4")):
            first * second * producers

    def test_dimensions_renamed_apart_bind_separately(self):
        first = _kernel("k1", {"x": SYMBOLIC}, "a", SYMBOLIC)
        second = _kernel("k2", {"w": SYMBOLIC}, "b", SYMBOLIC).with_dim_names(n="m")
        producers = _law("p", "x", VECTOR) * _law("q", "w", NumericArraySpec((4,)))
        joint = first * second * producers
        assert joint.event_spec.components["a"].shape == (3,)
        assert joint.event_spec.components["b"].shape == (4,)

    def test_a_slot_and_its_producer_sharing_a_dimension_keep_it_free(self):
        joint = _kernel("lik", {"beta": SYMBOLIC}, "y", SYMBOLIC) * _law("prior", "beta", SYMBOLIC)
        assert joint.event_spec.spec.free_dims == {"n"}
        bound = joint.with_dim_sizes(n=3)
        assert [factor.event_spec.spec.shape for factor in bound.factors] == [(3,), (3,)]


# -- Matched specs -----------------------------------------------------------------


class TestMatchedSpecs:
    """A name match is not enough: the supplying component's spec must unify with the slot."""

    @pytest.mark.parametrize(
        "component_spec",
        [
            pytest.param(SCALAR, id="rank"),
            pytest.param(OpaqueSpec(), id="kind"),
            pytest.param(NumericArraySpec((3,), "float32"), id="dtype"),
        ],
    )
    def test_a_producer_whose_spec_does_not_unify_with_the_slot_raises(self, component_spec):
        consumer = _kernel("lik", {"beta": NumericArraySpec((3,), "int32")}, "y")
        with pytest.raises(ValueError, match=_mentions("'beta'")):
            consumer * _law("prior", "beta", component_spec)

    def test_a_polymorphic_slot_binds_to_its_producer(self):
        joint = _kernel("lik", {"beta": SYMBOLIC}, "y", SYMBOLIC) * _law("prior", "beta", VECTOR)
        assert joint.event_spec.components["y"] == VECTOR
        assert joint.factors[0].given_spec["beta"] == VECTOR

    def test_a_polymorphic_producer_binds_to_the_slot_it_meets(self):
        joint = _kernel("lik", {"beta": VECTOR}, "y") * _law("prior", "beta", SYMBOLIC)
        assert joint.event_spec.components["beta"] == VECTOR
        assert joint.factors[1].event_spec.spec == VECTOR


# -- Flattening --------------------------------------------------------------------


class TestFlattening:
    """Every operand enters as its flattened factors, so a chain is one flat joint."""

    def test_direct_construction_flattens_a_factored_factor(self):
        a, b, c = (_law(name, name) for name in "abc")
        joint = FactoredDistribution("j", [a * b, c])
        assert [factor.name for factor in joint.factors] == ["a", "b", "c"]

    def test_a_chain_of_three_operands_is_one_joint_of_three_factors(self):
        lik, prior, other = _likelihood(), _prior(), _law("other", "c")
        joint = lik * prior * other
        assert joint.factors == (lik, prior, other)
        assert not any(isinstance(factor, SupportsFactors) for factor in joint.factors)

    def test_joint_operands_contribute_their_factors_in_order(self):
        a, b, c, d = (_law(name, name) for name in "abcd")
        assert ((a * b) * (c * d)).factors == (a, b, c, d)

    def test_a_relabeled_joint_contributes_its_factors(self):
        lik, prior, other = _likelihood(), _prior(), _law("d", "d")
        joint = (lik * prior).with_name("posterior") * other
        assert joint.factors == (lik, prior, other)

    def test_the_components_follow_the_flattened_factor_order(self):
        joint = _likelihood() * (_prior() * _law("other", "c"))
        assert list(joint.event_spec.components) == ["y", "beta", "c"]

    def test_an_independent_factor_may_stand_anywhere_in_a_chain(self):
        lik, prior, other = _likelihood(), _prior(), _law("other", "c")
        for joint in (other * lik * prior, lik * other * prior, lik * prior * other):
            assert isinstance(joint, FactoredDistribution)


# -- Associativity -----------------------------------------------------------------


def _chain():
    return _likelihood(), _prior(), _law("other", "c")


def _shared_given():
    return _kernel("k1", {"x": SCALAR}, "a"), _kernel("k2", {"x": SCALAR}, "b"), _law("p", "x")


def _open_givens():
    return _kernel("k1", {"x": SCALAR}, "a"), _law("l", "b"), _kernel("k2", {"z": SCALAR}, "c")


def _bound_dimension():
    return (
        _kernel("k", {"x": SYMBOLIC}, "a", SYMBOLIC),
        _law("l", "b", SYMBOLIC),
        _law("p", "x", VECTOR),
    )


def _scope_carried():
    """A kernel over ``n`` bound by its producer, and a later law over the same ``n``."""
    return (
        _kernel("k", {"x": SYMBOLIC}, "a", SYMBOLIC),
        _law("p", "x", VECTOR),
        _law("l", "b", SYMBOLIC),
    )


def _consumer_after_producer():
    return _prior(), _law("other", "c"), _likelihood()


def _produced_twice():
    return _law("u", "a"), _law("v", "b"), _law("w", "a")


def _consumer_of_the_middle():
    return _law("u", "a"), _law("p", "x"), _kernel("k", {"x": SCALAR}, "c")


class TestAssociativity:
    """Under ``G_B ∩ F_A = ∅`` the two groupings agree on validity and build one joint."""

    def test_a_joint_carries_its_bound_dimensions_into_a_later_composition(self):
        a, b, c = _scope_carried()
        joint = (a * b) * c
        assert joint.event_spec.components["b"] == VECTOR

    @pytest.mark.parametrize(
        "operands",
        [
            pytest.param(_chain, id="chain"),
            pytest.param(_shared_given, id="shared-given"),
            pytest.param(_open_givens, id="open-givens"),
            pytest.param(_bound_dimension, id="bound-dimension"),
            pytest.param(_scope_carried, id="scope-carried"),
        ],
    )
    def test_both_groupings_build_one_joint(self, operands):
        a, b, c = operands()
        assert _structure((a * b) * c) == _structure(a * (b * c))

    @pytest.mark.parametrize(
        "operands",
        [
            pytest.param(_consumer_after_producer, id="consumer-after-producer"),
            pytest.param(_produced_twice, id="produced-twice"),
            pytest.param(_consumer_of_the_middle, id="consumer-of-the-middle"),
        ],
    )
    def test_both_groupings_are_invalid_together(self, operands):
        a, b, c = operands()
        with pytest.raises(ValueError):
            (a * b) * c
        with pytest.raises(ValueError):
            a * (b * c)


# -- Naming the result -------------------------------------------------------------


class TestLabels:
    """The joint's label joins the operands' current labels with ``·``, as ordinary strings."""

    def test_the_label_joins_the_operand_labels_with_a_middle_dot(self):
        assert (_likelihood() * _prior()).name == "lik·prior"

    def test_a_chain_has_one_label_under_either_grouping(self):
        a, b, c = (_law(name, name) for name in "abc")
        assert ((a * b) * c).name == (a * (b * c)).name == "a·b·c"

    def test_a_relabeled_joint_contributes_its_new_label(self):
        posterior = (_likelihood() * _prior()).with_name("posterior")
        assert (posterior * _law("d", "d")).name == "posterior·d"

    def test_labels_join_without_escaping(self):
        left = _law("x·y", "u") * _law("z", "v")
        right = _law("x", "u") * _law("y·z", "v")
        assert left.name == right.name == "x·y·z"

    def test_exchanging_independent_operands_changes_the_label_and_the_order(self):
        a, b = _law("a", "a"), _law("b", "b")
        ab, ba = a * b, b * a
        assert (ab.name, ba.name) == ("a·b", "b·a")
        assert list(ab.event_spec.components) == ["a", "b"]
        assert list(ba.event_spec.components) == ["b", "a"]
        assert (ab.factors, ba.factors) == ((a, b), (b, a))

    def test_exchanging_independent_operands_keeps_the_law(self):
        a, b = Normal("a", 0.0, 1.0), Normal("b", 1.0, 2.0)
        value = {"a": jnp.asarray(0.3), "b": jnp.asarray(-0.4)}
        assert jnp.allclose((a * b)._log_prob(value), (b * a)._log_prob(value))


class TestWithName:
    """``with_name`` changes only the text a joint contributes to a later label."""

    def test_relabeling_a_joint_keeps_its_factors_and_declaration(self):
        joint = _likelihood() * _prior()
        posterior = joint.with_name("posterior")
        assert posterior.name == "posterior"
        assert isinstance(posterior, FactoredDistribution)
        assert (posterior.spec, posterior.factors) == (joint.spec, joint.factors)

    def test_a_joint_relabeled_with_its_own_label_composes_as_the_same_joint(self):
        joint, other = _likelihood() * _prior(), _law("c", "c")
        assert _structure(joint.with_name(joint.name) * other) == _structure(joint * other)


class TestLabelsNeverDecideStructure:
    """Composition matches component and slot names; a label never meets a given."""

    def test_a_label_that_matches_a_given_does_not_meet_it(self):
        impostor = Normal("beta", 0.0, 1.0, event_spec=OutputSpec(theta=None))
        joint = _likelihood() * impostor
        assert isinstance(joint, FactoredConditionalDistribution)
        assert list(joint.given_spec) == ["beta"]

    def test_a_component_meets_a_given_whatever_its_label(self):
        joint = _likelihood() * Normal("anything", 0.0, 1.0, event_spec=OutputSpec(beta=None))
        assert isinstance(joint, FactoredDistribution)

    def test_a_law_keeps_its_component_after_with_name(self):
        joint = _likelihood() * Normal("beta", 0.0, 1.0).with_name("renamed")
        assert isinstance(joint, FactoredDistribution)
        assert list(joint.event_spec.components) == ["y", "beta"]

    def test_relabeling_the_operands_changes_only_the_label(self):
        lik, prior = _likelihood(), _prior()
        plain, relabeled = lik * prior, lik.with_name("L") * prior.with_name("P")
        assert relabeled.name == "L·P"
        assert type(relabeled) is type(plain)
        assert relabeled.spec == plain.spec
        assert [factor.spec for factor in relabeled.factors] == [
            factor.spec for factor in plain.factors
        ]

    def test_factor_labels_may_repeat(self):
        joint = _law("x", "a") * _law("x", "b")
        assert joint.name == "x·x"
        assert [factor.name for factor in joint.factors] == ["x", "x"]
        assert list(joint.event_spec.components) == ["a", "b"]


# -- Operand kinds -----------------------------------------------------------------


class TestOperandKinds:
    """Distribution and kernel operands compose; a scalar operand never does."""

    @pytest.mark.parametrize("scalar", [pytest.param(2.0, id="float"), pytest.param(2, id="int")])
    def test_a_law_times_a_scalar_does_not_compose(self, scalar):
        law = Normal("x", 0.0, 1.0)
        assert law.__mul__(scalar) is NotImplemented
        with pytest.raises(TypeError):
            law * scalar
        with pytest.raises(TypeError):
            scalar * law

    def test_a_kernel_times_a_scalar_does_not_compose(self):
        kernel = _likelihood()
        assert kernel.__mul__(2.0) is NotImplemented
        with pytest.raises(TypeError):
            kernel * 2.0

    def test_a_random_function_times_a_scalar_scales(self):
        scaled = _random_function() * 2.0
        assert isinstance(scaled, GaussianRandomFunction)
        assert not isinstance(scaled, SupportsFactors)

    def test_a_law_times_a_random_function_composes(self):
        joint = Normal("z", 0.0, 1.0) * _random_function()
        assert isinstance(joint, FactoredDistribution)
        assert list(joint.event_spec.components) == ["z", "f"]

    def test_a_random_function_times_a_law_composes(self):
        joint = _random_function() * Normal("z", 0.0, 1.0)
        assert isinstance(joint, FactoredDistribution)
        assert list(joint.event_spec.components) == ["f", "z"]
