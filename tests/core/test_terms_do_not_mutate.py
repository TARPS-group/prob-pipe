"""Reading a tracked term does not modify it.

`design/05-operations.md` §V.1 promises an implementer's object is never
modified. Two terms broke that where a caller could see it: a
``BroadcastDistribution`` assigned its marginal on first ``marginalize()``, and a
backend-delegated ``DistributionArray`` assigned its components on first read.
Both now fill a memo container assigned at construction, so the attributes the
term was built with stay untouched.

The rest of the class is an invariant rather than a regression: those terms wrote
their fields before the object reached a caller, which is construction by another
name. The tests keep watch on all of them, since the difference is invisible from
outside and easy to lose.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from probpipe import (
    DistributionArray,
    Normal,
    ProductDistribution,
    SequentialJointDistribution,
    condition_on,
)
from probpipe.core._broadcast_distributions import BroadcastDistribution


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


class _ScalarBackend:
    """The smallest thing ``DistributionArray._from_backend`` accepts."""

    is_approximate = False

    def __init__(self, n: int):
        self.batch_shape = (n,)

    def cell(self, index: int) -> Normal:
        return Normal(float(index), 1.0, name=f"c{index}")


class TestTheCheckItself:
    """The helper has to catch the kind of mutation these tests are about.

    A lazy read that fills a container the term already holds rebinds nothing,
    so a check comparing attribute identities alone passes straight through it.
    """

    def test_it_catches_an_edit_that_rebinds_nothing(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        name, container = next((n, v) for n, v in vars(joint).items() if isinstance(v, dict) and v)
        before = assigned_state(joint)
        container["injected"] = object()  # same dict object, new entry
        assert assigned_state(joint) != before, f"an in-place edit to {name} went unseen"

    def test_it_catches_a_rebound_attribute(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        before = assigned_state(joint)
        object.__setattr__(joint, "_conditioned_names", ("z",))
        assert assigned_state(joint) != before

    def test_it_ignores_the_memo(self):
        # The one store a read is meant to fill.
        array = DistributionArray._from_backend(_ScalarBackend(3), name="x")
        before = assigned_state(array)
        assert array.components  # fills the memo
        assert assigned_state(array) == before


class TestAQueryLeavesTheTermUnchanged:
    def test_marginalizing_a_broadcast_distribution(self):
        broadcast = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 2)),
            weights=None,
            broadcast_args=["x"],
        )
        before = assigned_state(broadcast)
        first = broadcast.marginalize()
        assert assigned_state(broadcast) == before
        # Still memoised: the second read returns the first result.
        assert broadcast.marginalize() is first

    def test_reading_a_backend_delegated_array_s_components(self):
        # The backend-delegated array is the one that materialises on read; an
        # array built from a literal component list has them from the start.
        array = DistributionArray._from_backend(_ScalarBackend(3), name="x")
        before = assigned_state(array)
        first = array.components
        assert assigned_state(array) == before
        assert array.components is first

    def test_an_approximate_distribution_concatenates_at_construction(self):
        # The constructor reads the concatenation, so the memo is filled before
        # a caller holds the object and no later read assigns anything.
        from probpipe.inference._approximate_distribution import ApproximateDistribution

        posterior = ApproximateDistribution([np.zeros((4, 1)), np.ones((4, 1))], name="p")
        before = assigned_state(posterior)
        first = posterior._concat_chains()
        assert assigned_state(posterior) == before
        assert posterior._concat_chains() is first

    def test_a_tfp_product_distribution_builds_its_tfp_view_at_construction(self):
        # The combined TFP distribution is built once, by the constructor: no
        # read fills it in later.
        joint = ProductDistribution(
            a=Normal(0.0, 1.0, name="a"), b=Normal(1.0, 2.0, name="b"), name="j"
        )
        assert hasattr(joint, "_tfp_dist")
        before = assigned_state(joint)
        _ = joint.event_shape
        assert assigned_state(joint) == before


class TestAnOperationDoesNotMutateItsResultAfterBuildingIt:
    def test_conditioning_a_sequential_joint(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        conditioned = condition_on(joint, z=jnp.asarray(2.0))
        # The result is complete when it is returned, and conditioning again
        # builds another result rather than editing this one.
        before = assigned_state(conditioned)
        again = condition_on(joint, z=jnp.asarray(3.0))
        assert assigned_state(conditioned) == before
        assert again is not conditioned
        # The operand is untouched, which is what §V.1 promises.
        assert set(joint.components) == {"z", "x"}


class TestACopyDoesNotInheritAMemo:
    """A memo is per term, because what it holds can be per term.

    ``marginalize`` stamps the distribution's own provenance onto the marginal it
    builds, so a renamed copy sharing one memo with its original would hand
    whichever of them asked second the other's lineage — and which that is
    depends only on query order.
    """

    @staticmethod
    def _broadcast() -> BroadcastDistribution:
        return BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 2)),
            weights=None,
            broadcast_args=["x"],
        )

    def test_the_rename_does_not_share_the_original_s_memo(self):
        original = self._broadcast()
        renamed = original.with_name("renamed")
        assert getattr(renamed, "_memo", None) is not getattr(original, "_memo", None)

    def test_lineage_does_not_depend_on_which_is_marginalized_first(self):
        renamed_first = self._broadcast()
        renamed = renamed_first.with_name("renamed")
        from_rename = renamed.marginalize()
        from_original = renamed_first.marginalize()

        original_first = self._broadcast()
        also_from_original = original_first.marginalize()
        also_from_rename = original_first.with_name("renamed").marginalize()

        # The original's marginal carries the original's lineage in both orders,
        # and the rename's carries the rename's.
        assert from_original.provenance is None
        assert also_from_original.provenance is None
        assert from_rename.provenance.operation == "with_name"
        assert also_from_rename.provenance.operation == "with_name"
        assert from_original is not from_rename

    def test_each_still_memoises_its_own(self):
        original = self._broadcast()
        renamed = original.with_name("renamed")
        assert original.marginalize() is original.marginalize()
        assert renamed.marginalize() is renamed.marginalize()
