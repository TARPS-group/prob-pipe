"""Tests for `FunctionBatch` and `OpaqueBatch` — the batch forms that store objects.

These two are the first concrete `Batch` implementations, so the suite carries a
second burden beyond the classes themselves: showing that the ABC's storage
contract is satisfiable by real storage rather than only by the test doubles in
`test_batch.py`. The view-sharing and index-order assertions below are that
check.
"""

from __future__ import annotations

import copy
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ArraySpec,
    Batch,
    BatchSpec,
    EventTemplate,
    FunctionBatch,
    FunctionSpec,
    OpaqueBatch,
    OpaqueSpec,
    TrackedTerm,
)


@pytest.fixture
def functions():
    """Three callables under one `variant` level."""
    return FunctionBatch([lambda x: x, lambda x: 2 * x, lambda x: 3 * x], "variant", name="f")


@pytest.fixture
def labels():
    """Three opaque values under one `site` level."""
    return OpaqueBatch(["north", "east", "south"], "site", name="s")


@pytest.fixture
def grid():
    """A 2x3 opaque batch over two levels."""
    store = np.empty((2, 3), dtype=object)
    for chain in range(2):
        for draw in range(3):
            store[chain, draw] = f"c{chain}d{draw}"
    return OpaqueBatch(store, ["chain", "draw"], name="post")


class TestConstruction:
    def test_a_flat_sequence_is_one_axis(self, functions):
        assert functions.batch_shape == (3,)
        assert functions.batch_size == 3
        assert functions.axis_groups == ((3,),)
        assert functions.level_names == ("variant",)

    def test_an_object_array_carries_its_own_shape(self, grid):
        assert grid.batch_shape == (2, 3)
        assert grid.axis_groups == ((2,), (3,))
        assert grid.level_names == ("chain", "draw")

    def test_axis_groups_may_put_several_axes_in_one_level(self):
        store = np.empty((2, 3), dtype=object)
        store[...] = "x"
        batch = OpaqueBatch(store, "draw", axis_groups=[(2, 3)])

        assert batch.axis_groups == ((2, 3),)
        assert batch.level_names == ("draw",)
        assert batch.batch_shape == (2, 3)

    def test_a_single_string_names_a_single_level(self, labels):
        assert labels.level_names == ("site",)

    def test_an_unnamed_batch_takes_an_auto_name(self):
        batch = OpaqueBatch(["a"], "site")

        assert batch.name == "opaquebatch"
        assert batch.name_is_auto

    def test_a_given_name_is_not_auto(self, labels):
        assert labels.name == "s"
        assert not labels.name_is_auto


class TestConstructionRefusals:
    def test_a_non_callable_element_names_its_position(self):
        with pytest.raises(TypeError, match=r"element at 1 is a int"):
            FunctionBatch([lambda x: x, 3], "variant")

    def test_a_mapping_is_not_an_opaque_element(self):
        with pytest.raises(TypeError, match="denotes a subtree"):
            OpaqueBatch([{"a": 1}], "site")

    def test_a_multi_axis_position_is_reported_as_a_tuple(self):
        store = np.empty((2, 2), dtype=object)
        store[...] = "x"
        store[1, 0] = {"a": 1}

        with pytest.raises(TypeError, match=r"element at \(1, 0\)"):
            OpaqueBatch(store, ["chain", "draw"])

    def test_no_elements_is_refused(self):
        with pytest.raises(ValueError, match="at least one element"):
            OpaqueBatch([], "site")

    def test_a_single_object_is_not_a_batch(self):
        with pytest.raises(ValueError, match="at least one batch axis"):
            OpaqueBatch(np.array(None, dtype=object), "site")

    def test_a_numeric_array_is_not_object_storage(self):
        with pytest.raises(TypeError, match="dtype=object"):
            OpaqueBatch(np.zeros(3), "site")

    def test_naming_fewer_levels_than_axes_is_refused(self):
        with pytest.raises(ValueError, match="2 axes need 2 level names"):
            OpaqueBatch(np.empty((2, 2), dtype=object), "site")

    @pytest.mark.parametrize(
        ("cls", "spec", "match"),
        [
            (FunctionBatch, ArraySpec(()), "must be a FunctionSpec"),
            (OpaqueBatch, ArraySpec(()), "must be an OpaqueSpec"),
        ],
    )
    def test_the_element_spec_must_be_its_own_kind(self, cls, spec, match):
        with pytest.raises(TypeError, match=match):
            cls([lambda: 1], "variant", element_spec=spec)


class TestSpec:
    def test_the_batch_is_specified_at_the_family_kind(self, functions):
        assert isinstance(functions.spec, BatchSpec)
        assert functions.spec == BatchSpec(FunctionSpec(), ((3,),), ("variant",))

    def test_element_spec_is_a_view_of_the_declared_kind(self, functions, labels):
        assert isinstance(functions.element_spec, FunctionSpec)
        assert isinstance(labels.element_spec, OpaqueSpec)
        assert functions.element_spec is functions.spec.element_spec

    def test_an_element_spec_may_be_given(self):
        declared = FunctionSpec(EventTemplate(x=()), EventTemplate(y=()))
        batch = FunctionBatch([lambda x: x], "variant", element_spec=declared)

        assert batch.element_spec == declared
        assert batch.spec.element_spec == declared

    def test_a_batch_naming_no_kind_is_specified_all_the_same(self, labels):
        """The case `BatchSpec` exists for: `OpaqueSpec` names no kind."""
        assert not isinstance(labels.element_spec, type(labels.spec))
        assert labels.spec.is_valid(labels)

    def test_is_valid_reads_the_whole_multiplicity(self, functions):
        assert functions.spec.is_valid(functions)
        assert not functions.spec.is_valid(FunctionBatch([lambda x: x], "variant", name="other"))


class TestElements:
    def test_an_element_is_the_object_that_was_stored(self, labels):
        stored = ["north", "east", "south"]
        assert [labels[i] for i in range(3)] == stored

    def test_a_callable_element_is_callable(self, functions):
        assert functions[1](5) == 10

    def test_a_tracked_element_keeps_its_own_name(self):
        """A stored element is handed back as given, not renamed to its position.

        The identity rule applies to a batch that materializes an element per
        index; here the caller's own object comes back, so a name that means
        something is not replaced by one that states a position.
        """

        class _Named(TrackedTerm):
            __slots__ = ("_name", "_name_is_auto", "_provenance")

            def __init__(self, name):
                self._init_tracked(name)

            def __call__(self):
                return self._name

        element = _Named("alpha")
        batch = FunctionBatch([element], "variant", name="f")

        assert batch[0] is element
        assert batch[0].name == "alpha"

    def test_iteration_walks_the_leading_axis(self, functions):
        assert [f(2) for f in functions] == [2, 4, 6]

    def test_len_is_the_leading_axis(self, grid):
        assert len(grid) == 2
        assert [len(inner) for inner in grid] == [3, 3]

    def test_elements_holding_arrays_are_not_unpacked(self):
        """`np.asarray` would stack these into one numeric array."""
        batch = OpaqueBatch([jnp.zeros(3), jnp.ones(3)], "site")

        assert batch.batch_shape == (2,)
        np.testing.assert_array_equal(np.asarray(batch[1]), np.ones(3))

    def test_elements_holding_sequences_are_not_unpacked(self):
        batch = OpaqueBatch([[1, 2], [3, 4]], "site")

        assert batch.batch_shape == (2,)
        assert batch[0] == [1, 2]


class TestTheStorageContractIsSatisfiable:
    """A sub-batch is a view over the parent's store, in the order asked for.

    `Batch._sub_batch_at` contracts for both, and until now only test doubles
    answered it. These assertions are the ABC's contract checked against real
    storage.
    """

    def test_a_sub_batch_shares_the_parent_store(self, labels):
        assert np.shares_memory(labels[0:2]._store, labels._store)

    def test_a_view_of_a_view_still_shares(self, grid):
        assert np.shares_memory(grid[0:2][1:3]._store, grid._store)

    def test_a_descending_slice_is_presented_in_the_order_given(self, labels):
        assert [labels[::-1][i] for i in range(3)] == ["south", "east", "north"]

    def test_a_stepped_slice_is_presented_in_the_order_given(self, functions):
        assert [f(1) for f in functions[::2]] == [1, 3]

    def test_selecting_everything_is_a_view_and_not_the_batch_itself(self, labels):
        whole = labels.at_levels()

        assert whole is not labels
        assert np.shares_memory(whole._store, labels._store)
        assert [whole[i] for i in range(3)] == [labels[i] for i in range(3)]

    def test_a_view_carries_the_spec_the_abc_computed(self, grid):
        assert grid.at_levels(chain=0).spec == BatchSpec(OpaqueSpec(), ((3,),), ("draw",))


class TestNaming:
    def test_a_view_is_named_by_what_it_selects(self, grid):
        assert grid.at_levels(chain=0).name == "post[chain=0]"
        assert grid.at_levels(draw=slice(1, 3)).name == "post[draw=1:3]"

    def test_a_derived_name_is_auto(self, labels):
        assert labels[0:2].name_is_auto

    def test_positional_and_named_indexing_derive_one_name(self, grid):
        assert grid[0].name == grid.at_levels(chain=0).name

    def test_level_names_can_be_repinned(self, labels):
        renamed = labels.with_level_names(site="place")

        assert renamed.level_names == ("place",)
        assert renamed[0:2].name == "s[place=0:2]"

    def test_with_name_re_roots_a_view(self, labels):
        assert labels[0:2].with_name("q")[0:1].name == "q[site=0:1]"


class TestFieldKeys:
    def test_these_elements_have_no_fields(self, functions):
        with pytest.raises(TypeError, match="have no fields to address by name"):
            functions["anything"]

    def test_the_message_names_the_concrete_class(self, labels):
        with pytest.raises(TypeError, match="this OpaqueBatch"):
            labels["anything"]


class TestRoundTrips:
    @pytest.mark.parametrize("clone", [pickle.loads, copy.copy], ids=["pickle", "copy"])
    def test_a_batch_survives(self, labels, clone):
        restored = clone(pickle.dumps(labels)) if clone is pickle.loads else clone(labels)

        assert restored.name == labels.name
        assert restored.level_names == labels.level_names
        assert [restored[i] for i in range(3)] == ["north", "east", "south"]

    def test_a_view_survives_pickling(self, labels):
        restored = pickle.loads(pickle.dumps(labels[0:2]))

        assert restored.name == "s[site=0:2]"
        assert [restored[i] for i in range(2)] == ["north", "east"]

    def test_the_store_travels_without_being_declared(self, labels):
        """`Batch.__getstate__` walks the MRO, so `_store` needs no restating."""
        assert "_store" in labels.__getstate__()[1]


class TestImmutability:
    def test_assignment_is_refused(self, labels):
        with pytest.raises(AttributeError, match="OpaqueBatch is immutable"):
            labels._store = None

    def test_deletion_is_refused(self, labels):
        with pytest.raises(AttributeError, match="OpaqueBatch is immutable"):
            del labels._store


class TestTheseAreBatches:
    def test_both_are_batches(self, functions, labels):
        assert isinstance(functions, Batch)
        assert isinstance(labels, Batch)

    def test_both_are_tracked_terms(self, functions, labels):
        assert isinstance(functions, TrackedTerm)
        assert isinstance(labels, TrackedTerm)

    def test_neither_adds_public_interface_beyond_the_abc(self):
        """Design III.1: each carries its shared spec, adding no other interface."""
        added = {
            name
            for cls in (FunctionBatch, OpaqueBatch)
            for name in vars(cls)
            if not name.startswith("_")
        }
        assert added == {"element_spec"}

    def test_repr_reads_the_levels_and_no_elements(self, grid):
        assert repr(grid) == "OpaqueBatch(name='post', chain=2, draw=3)"
