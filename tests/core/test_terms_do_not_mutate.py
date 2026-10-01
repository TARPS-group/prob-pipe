"""Reading a tracked term does not modify it.

`design/05-operations.md` §V.1 promises an implementer's object is never
modified. A term that memoises a derived value fills a memo container assigned
at construction, so the attributes the term was built with stay untouched.

The rest of the class is an invariant rather than a regression: those terms wrote
their fields before the object reached a caller, which is construction by another
name. The tests keep watch on all of them, since the difference is invisible from
outside and easy to lose.
"""

import copy
import pickle
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal
from probpipe.core._specs import NumericArraySpec, OutputSpec
from probpipe.distributions import ConditionalDistribution, SupportsConditionalSampling
from tests._ops import condition_on


def assigned_state(term) -> dict:
    """What the term holds, in the two ways a read could change it.

    Each attribute maps to its value's identity — which catches an attribute
    being *rebound* — paired with a shallow census of that value where the value
    is a container, which catches one being edited *in place*. Identity alone
    misses the second, and it is the likelier of the two: a lazy read that fills
    a dictionary it already holds rebinds nothing.

    The census is deliberately shallow and structural: keys and element
    identities, not values. A distribution's leaves are jax arrays and other
    terms, which have no cheap value equality, so this asserts that the
    *arrangement* is untouched rather than that the numbers are.

    ``_memo`` is excluded, being the one store a read is meant to fill.
    """
    state = object.__getstate__(term)
    instance_dict, slots = state if isinstance(state, tuple) else (state, {})
    both = {**(instance_dict or {}), **(slots or {})}
    return {name: (id(value), _census(value)) for name, value in both.items() if name != "_memo"}


def _census(value):
    """A shallow, structural snapshot of a container, or ``None`` for a leaf."""
    if isinstance(value, Mapping):
        return ("mapping", tuple(value), tuple(id(v) for v in value.values()))
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        return ("sequence", len(value), tuple(id(v) for v in value))
    return None


class _ShiftKernel(ConditionalDistribution, SupportsConditionalSampling):
    """``x | z ~ Normal(z, 0.5)``, the dependent factor of a joint."""

    def __init__(self):
        spec = NumericArraySpec(())
        super().__init__("x", {"z": spec}, OutputSpec(x=spec))

    def _condition_on(self, given, /, **options):
        return Normal("x", given["z"], 0.5)

    def _conditional_sample(self, given, key, sample_shape=()):
        return Normal("x", given["z"], 0.5)._sample(key, sample_shape)


class TestTheCheckItself:
    """The helper has to catch the kind of mutation these tests are about.

    A lazy read that fills a container the term already holds rebinds nothing,
    so a check comparing attribute identities alone passes straight through it.
    """

    def test_it_catches_an_edit_that_rebinds_nothing(self):
        class _Holder:
            def __init__(self):
                self.cache = {"filled": object()}

        holder = _Holder()
        before = assigned_state(holder)
        holder.cache["injected"] = object()  # same dict object, new entry
        assert assigned_state(holder) != before, "an in-place edit to cache went unseen"

    def test_it_catches_a_rebound_attribute(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0)
        before = assigned_state(joint)
        object.__setattr__(joint, "_graph", None)
        assert assigned_state(joint) != before

    def test_it_ignores_the_memo(self):
        # The one store a read is meant to fill.
        from probpipe.inference._approximate_distribution import ApproximateDistribution

        posterior = ApproximateDistribution([np.zeros((4, 1)), np.ones((4, 1))], name="p")
        before = assigned_state(posterior)
        assert posterior._concat_chains() is not None  # fills the memo
        assert assigned_state(posterior) == before


class TestAQueryLeavesTheTermUnchanged:
    def test_an_approximate_distribution_concatenates_at_construction(self):
        # The constructor reads the concatenation, so the memo is filled before
        # a caller holds the object and no later read assigns anything.
        from probpipe.inference._approximate_distribution import ApproximateDistribution

        posterior = ApproximateDistribution([np.zeros((4, 1)), np.ones((4, 1))], name="p")
        before = assigned_state(posterior)
        first = posterior._concat_chains()
        assert assigned_state(posterior) == before
        assert posterior._concat_chains() is first

    def test_a_factored_joint_is_unchanged_by_a_field_view(self):
        # The view reads the joint's marginal report at the view's construction,
        # which fills nothing on the joint.
        joint = Normal("a", 0.0, 1.0) * Normal("b", 1.0, 2.0)
        before = assigned_state(joint)
        _ = joint["a"]
        assert assigned_state(joint) == before


class TestAnOperationDoesNotMutateItsResultAfterBuildingIt:
    def test_conditioning_a_dependent_joint(self):
        joint = _ShiftKernel() * Normal(loc=0.0, scale=1.0, name="z")
        conditioned = condition_on(joint, {"z": jnp.asarray(2.0)})
        # The result is complete when it is returned, and conditioning again
        # builds another result rather than editing this one.
        before = assigned_state(conditioned)
        again = condition_on(joint, {"z": jnp.asarray(3.0)})
        assert assigned_state(conditioned) == before
        assert again is not conditioned
        # The operand is untouched, which is what §V.1 promises.
        assert set(joint.event_spec.components) == {"z", "x"}


class TestEveryMemoHolderDropsItsMemoOnACopy:
    """Each memo-holding term, across each way of copying one.

    The mixin's own tests cover the mechanism; these cover the classes
    that opt into it, so dropping `_transient_state` from one of them, or
    breaking its rebuild path, fails here rather than passing quietly.

    Each case gives a term, a read that fills its memo, and what that read
    returns, so the assertions can check both halves: the copy does not inherit
    the memo, and it can still rebuild the value.
    """

    @staticmethod
    def _approximate():
        from probpipe.inference._approximate_distribution import ApproximateDistribution

        term = ApproximateDistribution([np.zeros((4, 1)), np.ones((4, 1))], name="p")
        return term, lambda d: d._concat_chains()

    @pytest.fixture(
        params=[
            pytest.param("_approximate", id="approximate-chains"),
        ]
    )
    def case(self, request):
        return getattr(self, request.param)()

    @pytest.fixture(
        params=[
            pytest.param(lambda t: t.with_name("renamed"), id="with_name"),
            pytest.param(copy.copy, id="copy"),
            pytest.param(copy.deepcopy, id="deepcopy"),
            pytest.param(lambda t: pickle.loads(pickle.dumps(t)), id="pickle"),
        ]
    )
    def copied(self, request):
        return request.param

    def test_the_copy_does_not_inherit_the_memo(self, case, copied):
        term, read = case
        read(term)  # fill it on the original
        assert getattr(copied(term), "_memo", {}) == {}

    def test_the_copy_still_rebuilds_the_value(self, case, copied):
        term, read = case
        read(term)
        assert read(copied(term)) is not None

    def test_the_memo_is_declared_transient(self, case):
        from probpipe.core._immutable import declared_state_names

        term, _ = case
        assert "_memo" in declared_state_names(type(term), "_transient_state")
