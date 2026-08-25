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
    Batch,
    BatchSpec,
    EventTemplate,
    FunctionBatch,
    FunctionSpec,
    NumericArraySpec,
    OpaqueBatch,
    OpaqueSpec,
    Record,
    TermSpec,
    TrackedTerm,
)
from probpipe.core.provenance import Provenance


@pytest.fixture
def functions():
    """Three callables under one `variant` level."""
    return FunctionBatch([lambda x: x, lambda x: 2 * x, lambda x: 3 * x], "variant", name="f")


@pytest.fixture
def labels():
    """Three opaque values under one `site` level."""
    return OpaqueBatch(["north", "east", "south"], "site", name="s")


@pytest.fixture
def function_grid():
    """A 2x2 FunctionBatch, so multi-axis behavior is pinned for both classes."""
    store = np.empty((2, 2), dtype=object)
    for row in range(2):
        for column in range(2):
            store[row, column] = lambda x, r=row, c=column: 10 * r + c + x
    return FunctionBatch(store, ["chain", "draw"], name="fg")


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

    def test_a_given_name_can_still_be_marked_auto(self):
        """The shape an operation deriving a batch name needs: named, but re-derivable."""
        batch = OpaqueBatch(["a"], "site", name="given", name_is_auto=True)

        assert batch.name == "given"
        assert batch.name_is_auto

    def test_a_provenance_is_carried_as_given(self):
        record = Provenance.create("sample", parents=[])
        batch = OpaqueBatch(["a"], "site", provenance=record)

        assert batch.provenance is record

    def test_a_generator_is_materialized_into_one_axis(self):
        batch = OpaqueBatch((str(i) for i in range(3)), "site")

        assert batch.batch_shape == (3,)
        assert [batch[i] for i in range(3)] == ["0", "1", "2"]


class TestConstructionRefusals:
    def test_a_non_callable_element_names_its_position(self):
        with pytest.raises(TypeError, match=r"element at 1 is a int"):
            FunctionBatch([lambda x: x, 3], "variant")

    def test_a_mapping_is_not_an_opaque_element(self):
        with pytest.raises(TypeError, match="denotes a subtree"):
            OpaqueBatch([{"a": 1}], "site")

    @pytest.mark.parametrize(
        ("cls", "good", "bad"),
        [(OpaqueBatch, "x", {"a": 1}), (FunctionBatch, lambda x: x, 3)],
        ids=["opaque", "function"],
    )
    def test_a_multi_axis_position_is_reported_as_a_tuple(self, cls, good, bad):
        """Both classes render the position the same way, from the shared check."""
        store = np.empty((2, 2), dtype=object)
        store[...] = good
        store[1, 0] = bad

        with pytest.raises(TypeError, match=r"element at \(1, 0\)"):
            cls(store, ["chain", "draw"])

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
            (FunctionBatch, NumericArraySpec(()), "must be a FunctionSpec"),
            (OpaqueBatch, NumericArraySpec(()), "must be an OpaqueSpec"),
        ],
    )
    def test_the_element_spec_must_be_its_own_kind(self, cls, spec, match):
        with pytest.raises(TypeError, match=match):
            cls([lambda: 1], "variant", element_spec=spec)

    @pytest.mark.parametrize(
        ("axis_groups", "why"),
        [
            ([(5, 7)], "sizes disagree with the store"),
            ([(2,)], "an axis is left out"),
            ([(2,), (3,), (4,)], "an axis is invented"),
            ([(3,), (2,)], "the sizes are transposed"),
        ],
        ids=["wrong-sizes", "too-few-axes", "too-many-axes", "transposed"],
    )
    def test_axis_groups_must_tile_the_stored_shape(self, axis_groups, why):
        """A grouping that disagrees would make every accessor a false statement."""
        store = np.empty((2, 3), dtype=object)
        store[...] = "x"

        with pytest.raises(ValueError, match="must tile the shape"):
            OpaqueBatch(store, ["a", "b"][: len(axis_groups)], axis_groups=axis_groups)

    @pytest.mark.parametrize(
        ("elements", "match"),
        [
            ("north", "iterates into its parts"),
            (b"ab", "iterates into its parts"),
            ({"a": 1, "b": 2}, "iterates into its parts"),
            (np.zeros(3), "must have dtype=object"),
            (jnp.zeros(3), "must have dtype=object"),
        ],
        ids=["str", "bytes", "mapping", "ndarray", "jax"],
    )
    def test_a_container_that_iterates_into_its_parts_is_refused(self, elements, match):
        """Each would give a batch of pieces of one object, not a batch of objects."""
        with pytest.raises(TypeError, match=match):
            OpaqueBatch(elements, "site")

    def test_something_not_iterable_at_all_is_refused(self):
        with pytest.raises(TypeError, match="iterable of elements"):
            OpaqueBatch(3, "site")


class TestSpec:
    def test_the_batch_is_specified_at_the_family_kind(self, functions):
        assert isinstance(functions.spec, BatchSpec)
        assert functions.spec == BatchSpec(FunctionSpec(), ((3,),), ("variant",))

    def test_element_spec_is_a_view_of_the_declared_kind(self, functions, labels):
        assert isinstance(functions.element_spec, FunctionSpec)
        assert isinstance(labels.element_spec, OpaqueSpec)
        assert functions.element_spec is functions.spec.element_spec
        assert labels.element_spec is labels.spec.element_spec

    def test_an_element_spec_may_be_given(self):
        declared = FunctionSpec(EventTemplate(x=()), EventTemplate(y=()))
        batch = FunctionBatch([lambda x: x], "variant", element_spec=declared)

        assert batch.element_spec == declared
        assert batch.spec.element_spec == declared

    def test_a_batch_naming_no_kind_is_specified_all_the_same(self, labels):
        """The case `BatchSpec` exists for: `OpaqueSpec` names no kind, the batch does."""
        assert not isinstance(labels.element_spec, TermSpec)
        assert isinstance(labels.spec, TermSpec)

    def test_is_valid_reads_every_part_of_the_spec(self, functions):
        """All three fields: the element spec, the axis sizes, and the level names."""
        three = [lambda x: x, lambda x: 2 * x, lambda x: 3 * x]

        assert functions.spec.is_valid(functions)
        assert not functions.spec.is_valid(FunctionBatch(three[:1], "variant"))
        assert not functions.spec.is_valid(FunctionBatch(three, "flavor"))
        assert not functions.spec.is_valid(
            FunctionBatch(three, "variant", element_spec=FunctionSpec(EventTemplate(x=())))
        )


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

    def test_an_object_array_of_arrays_is_not_unpacked(self):
        """The other input path: a supplied object array, copied rather than built.

        `_as_object_array` has two branches, and this is the one numpy could
        silently re-stack — a plain `np.asarray` of an object array of equal-shaped
        arrays descends into them and returns one 2-d numeric array.
        """
        store = np.empty(2, dtype=object)
        store[0], store[1] = jnp.zeros(3), jnp.ones(3)

        batch = OpaqueBatch(store, "site")

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

    @pytest.mark.parametrize(
        "select",
        [
            lambda b: b[0:2],
            lambda b: b[0],
            lambda b: b.at_levels(draw=1),
            lambda b: b.at_levels(chain=0, draw=slice(1, 3)),
            lambda b: b[::-1],
        ],
        ids=["slice", "integer-outer", "integer-inner", "mixed", "reversed"],
    )
    def test_every_indexing_form_shares_the_store(self, grid, select):
        """Not slices alone: an integer index drops an axis and must still view."""
        assert np.shares_memory(select(grid)._store, grid._store)

    def test_a_function_batch_shares_too(self, function_grid):
        """The contract is the base's, so it holds for both classes."""
        assert np.shares_memory(function_grid[0]._store, function_grid._store)

    def test_a_descending_slice_is_presented_in_the_order_given(self, labels):
        assert [labels[::-1][i] for i in range(3)] == ["south", "east", "north"]

    def test_a_stepped_slice_is_presented_in_the_order_given(self, functions):
        assert [f(1) for f in functions[::2]] == [1, 3]

    def test_selecting_everything_is_a_view_and_not_the_batch_itself(self, labels):
        whole = labels.at_levels()

        assert whole is not labels
        assert np.shares_memory(whole._store, labels._store)
        assert [whole[i] for i in range(3)] == [labels[i] for i in range(3)]

    def test_the_store_cannot_be_written_through(self, labels):
        """A view shares the buffer, so a writable store would reach the parent."""
        with pytest.raises(ValueError, match="read-only"):
            labels._store[0] = {"a": 1}
        with pytest.raises(ValueError, match="read-only"):
            labels[0:2]._store[0] = {"a": 1}

    def test_the_array_a_caller_passes_is_copied(self):
        """Otherwise the caller could write past the per-element check."""
        store = np.empty(2, dtype=object)
        store[0], store[1] = "north", "south"
        batch = OpaqueBatch(store, "site")

        store[1] = {"a": 1}

        assert batch[1] == "south"
        assert batch.element_spec.is_valid(batch[1])

    def test_an_empty_selection_is_a_batch_of_nothing(self, labels):
        """Reachable by selection though the constructor refuses it — see Notes."""
        empty = labels[0:0]

        assert empty.batch_shape == (0,)
        assert empty.level_names == ("site",)
        assert len(empty) == 0
        assert list(empty) == []
        assert empty._store.size == 0

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
    @pytest.mark.parametrize(
        "clone",
        [lambda b: pickle.loads(pickle.dumps(b)), copy.copy, copy.deepcopy],
        ids=["pickle", "copy", "deepcopy"],
    )
    def test_a_batch_survives(self, labels, clone):
        restored = clone(labels)

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


class TestProvenance:
    """A stored element is not the batch's to attribute (design II.5)."""

    def test_a_view_inherits_the_batch_lineage(self, labels, full_provenance_mode):
        produced = labels.with_provenance(Provenance.create("collect", parents=[]))

        assert produced[0:2].provenance is produced.provenance

    def test_reading_an_element_leaves_the_caller_object_untouched(self, full_provenance_mode):
        """These batches store what they were given, so a read writes to nothing."""
        element = Record("r", x=1.0)
        batch = OpaqueBatch([element, Record("r2", x=2.0)], "site", name="recs")
        batch.with_provenance(Provenance.create("collect", parents=[]))

        assert batch[0] is element
        assert element.provenance is None

    def test_the_caller_can_still_set_its_own_provenance_afterwards(self, full_provenance_mode):
        """The write-once slot stays the caller's to spend."""
        element = Record("r", x=1.0)
        batch = OpaqueBatch([element], "site")
        batch[0]

        own = Provenance.create("fit", parents=[])
        assert element.with_provenance(own).provenance is own


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
            for ancestor in cls.__mro__[: cls.__mro__.index(Batch)]
            for name in vars(ancestor)
            if not name.startswith("_")
        }
        assert added == {"element_spec"}

    def test_repr_reads_the_levels_and_no_elements(self):
        """Load-bearing: `with_provenance` interpolates the batch into its own error."""

        class _Unreadable:
            def __repr__(self):
                raise AssertionError("repr must not read an element")

        store = np.empty(2, dtype=object)
        store[0] = store[1] = _Unreadable()
        batch = OpaqueBatch(store, "site", name="s")

        assert repr(batch) == "OpaqueBatch(name='s', site=2)"
