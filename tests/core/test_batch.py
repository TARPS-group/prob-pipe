"""Tests for the `Batch[E]` ABC — the level algebra, not any concrete storage.

These tests exercise the shared contract through two minimal
doubles: `_ListBatch`, whose elements are tracked terms, and `_BareBatch`, whose
elements are plain values and so carry no identity to derive.
"""

from __future__ import annotations

import copy
import itertools
import pickle
from typing import get_type_hints

import pytest

from probpipe import ArraySpec, EventTemplate, OpaqueSpec, TermSpec
from probpipe.core._batch import Batch, BatchSpec
from probpipe.core._fingerprint import fingerprint
from probpipe.core.tracked import TrackedTerm

# The doubles' element type is neither a Record, a Distribution, nor a Function,
# so no term spec fits it: OpaqueSpec is the honest element spec, and doubles as
# a check that an element spec naming no kind is well formed.
_ELEMENT_SPEC = OpaqueSpec()


def _spec(axis_groups, level_names, element_spec=_ELEMENT_SPEC):
    """A ``BatchSpec`` over the doubles' element type."""
    return BatchSpec(element_spec, axis_groups, level_names)


class _Leaf(TrackedTerm):
    """A minimal tracked element."""

    __slots__ = ("_name", "_name_is_auto", "_provenance", "value")

    def __init__(self, value, name="leaf", *, name_is_auto=False):
        object.__setattr__(self, "value", value)
        self._init_tracked(name, name_is_auto=name_is_auto)


class _ListBatch(Batch[_Leaf]):
    """A batch storing elements in a flat list, row-major over ``batch_shape``."""

    __slots__ = ("_store",)

    def __init__(self, store, spec, *, name="b", name_is_auto=False):
        object.__setattr__(self, "_store", list(store))
        self._init_batch(spec, name=name, name_is_auto=name_is_auto)

    # -- the storage seam --

    def _flat(self, index):
        offset = 0
        for position, size in zip(index, self.batch_shape, strict=True):
            offset = offset * size + position
        return offset

    def _element_at(self, index, *, name):
        return _Leaf(self._store[self._flat(index)], name=name, name_is_auto=True)

    def _sub_batch_at(self, index, *, spec, name):
        # The index is honoured in the order it presents its positions, so a
        # descending selection stores its elements descending.
        kept = [
            self._store[self._flat(position)]
            for position in itertools.product(*_selected(index, self.batch_shape))
        ]
        return type(self)(kept, spec, name=name, name_is_auto=True)


class _NestedBatch(_ListBatch):
    """A batch whose elements are batches, which take the name derived for them."""

    __slots__ = ()

    def _element_at(self, index, *, name):
        return self._store[self._flat(index)].with_name(name)


class _BareBatch(_ListBatch):
    """A batch whose elements are bare values; nothing to name."""

    __slots__ = ()

    def _element_at(self, index, *, name):
        return self._store[self._flat(index)]


def _selected(index, shape):
    """The positions each axis selects, in the order the index presents them."""
    return [
        [indexer] if isinstance(indexer, int) else list(range(*indexer.indices(size)))
        for indexer, size in zip(index, shape, strict=True)
    ]


@pytest.fixture
def flat():
    """A single-level batch: 4 elements on one ``draw`` axis."""
    return _ListBatch(range(4), _spec([(4,)], ["draw"]))


@pytest.fixture
def nested():
    """Two levels: ``chain`` of 2 over ``draw`` of 3."""
    return _ListBatch(range(6), _spec([(2,), (3,)], ["chain", "draw"]))


@pytest.fixture
def two_axis():
    """One two-axis level, to exercise partial level indexers."""
    return _ListBatch(range(6), _spec([(2, 3)], ["draw"]))


class TestShapeAndLevels:
    def test_batch_shape_is_the_flat_concatenation_of_the_levels(self, nested):
        assert nested.axis_groups == ((2,), (3,))
        assert nested.batch_shape == (2, 3)
        assert nested.batch_size == 6

    def test_a_multi_axis_level_keeps_its_axes_together(self, two_axis):
        assert two_axis.axis_groups == ((2, 3),)
        assert two_axis.batch_shape == (2, 3)
        assert two_axis.level_names == ("draw",)

    def test_len_and_iter_speak_only_about_the_leading_axis(self, nested):
        assert len(nested) == 2
        assert [len(inner) for inner in nested] == [3, 3]

    @pytest.mark.parametrize(
        ("groups", "names", "match"),
        [
            ([], [], "at least one batch axis"),
            ([()], ["draw"], "every level holds at least one axis"),
            ([(2,)], [], "name every level"),
            ([(2,), (3,)], ["draw"], "name every level"),
            ([(2,), (3,)], ["draw", "draw"], "unique within a batch"),
            ([(2,)], [""], "non-empty"),
            ([(-1,)], ["draw"], "non-negative"),
        ],
    )
    def test_level_invariants_are_checked_at_construction(self, groups, names, match):
        with pytest.raises(ValueError, match=match):
            _spec(groups, names)


class TestSpec:
    """The stored ``BatchSpec`` is the single source of a batch's type."""

    def test_the_level_accessors_are_views_on_the_stored_spec(self, nested):
        assert isinstance(nested.spec, BatchSpec)
        assert nested.spec is nested._spec
        assert (nested.axis_groups, nested.level_names) == (
            nested.spec.axis_groups,
            nested.spec.level_names,
        )
        assert (nested.batch_shape, nested.batch_size) == (
            nested.spec.batch_shape,
            nested.spec.batch_size,
        )

    def test_the_spec_names_the_batch_not_the_element(self, nested):
        assert nested.spec != nested.element_spec
        assert nested.element_spec == _ELEMENT_SPEC

    def test_an_element_spec_naming_no_kind_is_well_formed(self):
        """The case ``BatchSpec`` exists to cover: a batch of raw values."""
        bare = _BareBatch(range(3), _spec([(3,)], ["draw"], ArraySpec(shape=())))
        assert bare.element_spec == ArraySpec(shape=())
        assert isinstance(bare.spec, BatchSpec)

    def test_axis_groups_are_normalised_to_tuples(self):
        assert _spec([[2], [3]], ["chain", "draw"]).axis_groups == ((2,), (3,))

    def test_the_multiplicity_fields_are_declared_at_the_types_they_store(self):
        """The annotation is the post-construction guarantee, not the input sugar.

        The constructor takes any iterables; the fields are the tuples it stores,
        which is what makes a stored spec hashable. Mirrors the declaration
        fields on ``DistributionSpec`` and ``FunctionSpec``.
        """
        hints = get_type_hints(BatchSpec)
        assert hints["axis_groups"] == tuple[tuple[int, ...], ...]
        assert hints["level_names"] == tuple[str, ...]

    def test_specs_compare_and_hash_by_value(self):
        assert _spec([(4,)], ["draw"]) == _spec([(4,)], ["draw"])
        assert hash(_spec([(4,)], ["draw"])) == hash(_spec([(4,)], ["draw"]))
        assert _spec([(4,)], ["draw"]) != _spec([(4,)], ["chain"])

    def test_an_element_spec_must_be_a_value_spec(self):
        with pytest.raises(TypeError, match="must be a ValueSpec"):
            BatchSpec("not a spec", [(2,)], ["draw"])

    def test_a_batch_must_be_given_a_batch_spec(self):
        with pytest.raises(TypeError, match="specified by a BatchSpec"):
            _ListBatch([], _ELEMENT_SPEC)

    def test_a_sub_batch_view_keeps_the_element_spec_over_surviving_levels(self, nested):
        inner = nested[1]
        assert inner.element_spec == nested.element_spec
        assert inner.spec == _spec([(3,)], ["draw"])

    def test_a_sliced_view_narrows_the_axis_in_its_spec(self, nested):
        assert nested.at_levels(draw=slice(0, 2)).spec == _spec([(2,), (2,)], ["chain", "draw"])

    def test_renaming_replaces_the_spec_and_leaves_the_original_alone(self, nested):
        renamed = nested.with_level_names(chain="walker")
        assert renamed.spec == _spec([(2,), (3,)], ["walker", "draw"])
        assert nested.spec == _spec([(2,), (3,)], ["chain", "draw"])

    def test_is_valid_accepts_the_batch_it_specifies(self, flat):
        assert flat.spec.is_valid(flat)

    def test_is_valid_rejects_a_differently_shaped_batch(self, flat):
        assert not _spec([(3,)], ["draw"]).is_valid(flat)
        assert not _spec([(4,)], ["chain"]).is_valid(flat)

    def test_is_valid_rejects_a_non_batch(self, flat):
        assert not flat.spec.is_valid([0, 1, 2, 3])
        assert not flat.spec.is_valid(_Leaf(0))

    def test_a_batch_spec_is_a_term_spec(self, flat):
        assert isinstance(flat.spec, TermSpec)


class TestLevelNames:
    def test_with_level_names_renames_and_keeps_everything_else(self, nested):
        renamed = nested.with_level_names(chain="walker")
        assert renamed.level_names == ("walker", "draw")
        assert renamed.batch_shape == nested.batch_shape

    def test_renaming_shares_elements_and_preserves_identity(self, nested):
        """The default is a shallow copy: no storage is rebuilt, no identity minted."""
        renamed = nested.with_level_names(chain="walker")
        assert renamed._store is nested._store
        assert (renamed.name, renamed.name_is_auto) == (nested.name, nested.name_is_auto)

    def test_renaming_leaves_the_original_alone(self, nested):
        nested.with_level_names(chain="walker")
        assert nested.level_names == ("chain", "draw")

    def test_positional_mapping_and_keywords_both_work(self, nested):
        assert nested.with_level_names({"chain": "c"}, draw="d").level_names == ("c", "d")

    def test_renaming_onto_an_existing_level_raises(self, nested):
        with pytest.raises(ValueError, match="duplicate a level name"):
            nested.with_level_names(chain="draw")

    def test_swapping_two_names_is_allowed(self, nested):
        assert nested.with_level_names(chain="draw", draw="chain").level_names == ("draw", "chain")

    def test_renaming_an_unknown_level_raises(self, nested):
        with pytest.raises(KeyError, match="not levels of this batch"):
            nested.with_level_names(nope="x")

    def test_a_duplicate_level_name_is_rejected_not_altered(self):
        """A minted level takes a name of its own; a clash is an error.

        An operation adding a level is given the name to use, so a name already
        present is a mistake the caller resolves. Nothing silently alters it,
        which is what keeps a level name a statement about meaning rather than
        about the order levels were added in.
        """
        with pytest.raises(ValueError, match="must be unique within a batch"):
            _spec([(2,), (3,)], ["draw", "draw"])

    def test_the_duplicate_message_names_the_remedy(self):
        with pytest.raises(ValueError, match="name of its own"):
            _spec([(2,), (3,)], ["draw", "draw"])

    def test_renaming_onto_a_kept_level_raises_like_minting_does(self, nested):
        """Renaming and minting answer a clash the same way."""
        with pytest.raises(ValueError, match="would duplicate a level name"):
            nested.with_level_names(chain="draw")

    def test_an_empty_new_name_raises(self, nested):
        with pytest.raises(ValueError, match="must be non-empty"):
            nested.with_level_names(chain="")

    def test_two_renames_onto_one_name_raise(self, nested):
        with pytest.raises(ValueError, match="would duplicate a level name"):
            nested.with_level_names(chain="x", draw="x")

    def test_a_level_renamed_twice_by_mapping_and_keyword_raises(self, nested):
        with pytest.raises(ValueError, match="renamed twice"):
            nested.with_level_names({"chain": "a"}, chain="b")

    def test_the_same_new_name_from_both_forms_is_not_a_conflict(self, nested):
        assert nested.with_level_names({"chain": "a"}, chain="a").level_names == ("a", "draw")

    def test_a_level_name_must_be_an_identifier(self):
        with pytest.raises(ValueError, match="must be an identifier"):
            _spec([(2,)], ["my level"])

    def test_a_level_name_must_be_a_string(self):
        with pytest.raises(TypeError, match="level names are strings"):
            _spec([(2,)], [7])

    def test_renaming_a_level_repins_the_names_a_view_derives_from(self, nested):
        renamed = nested.with_level_names(chain="group")
        assert renamed[1].name == "b[group=1]"


class TestIndexing:
    def test_indexing_a_single_level_batch_yields_an_element(self, flat):
        assert flat[2].value == 2

    def test_indexing_a_multi_level_batch_yields_the_inner_level(self, nested):
        inner = nested[1]
        assert isinstance(inner, _ListBatch)
        assert inner.level_names == ("draw",)
        assert inner.batch_shape == (3,)
        assert [element.value for element in inner] == [3, 4, 5]

    def test_a_slice_keeps_the_axis_and_returns_a_sub_batch(self, flat):
        sub = flat[0:2]
        assert isinstance(sub, _ListBatch)
        assert sub.batch_shape == (2,)
        assert sub.level_names == ("draw",)

    def test_negative_indices_resolve(self, flat):
        assert flat[-1].value == 3

    def test_out_of_range_raises(self, flat):
        with pytest.raises(IndexError, match="out of range"):
            flat[4]

    def test_too_many_indices_raises(self, flat):
        with pytest.raises(IndexError, match="too many indices"):
            flat._at_axes((0, 0))


class TestAtLevels:
    def test_indexing_a_named_level_matches_positional_indexing(self, nested):
        assert nested.at_levels(chain=1).batch_shape == nested[1].batch_shape

    def test_an_unnamed_level_is_kept_whole(self, nested):
        assert nested.at_levels(draw=0).level_names == ("chain",)
        assert nested.at_levels(draw=0).batch_shape == (2,)

    def test_naming_every_level_reaches_an_element(self, nested):
        assert nested.at_levels(chain=1, draw=2).value == 5

    def test_none_means_the_whole_axis(self, nested):
        assert nested.at_levels(chain=None).batch_shape == (2, 3)

    def test_a_slice_keeps_its_level(self, nested):
        selected = nested.at_levels(draw=slice(0, 2))
        assert selected.batch_shape == (2, 2)
        assert selected.level_names == ("chain", "draw")

    def test_a_scalar_on_a_two_axis_level_fills_the_leading_axis(self, two_axis):
        # draw=1 means draw=(1, None): the first axis drops, the second is kept.
        selected = two_axis.at_levels(draw=1)
        assert selected.batch_shape == (3,)
        assert selected.level_names == ("draw",)

    def test_a_tuple_addresses_a_level_axes_in_order(self, two_axis):
        assert two_axis.at_levels(draw=(1, 2)).value == 5

    def test_too_many_indexers_for_a_level_raises(self, two_axis):
        with pytest.raises(ValueError, match="has 2 axes but got 3 indexers"):
            two_axis.at_levels(draw=(0, 0, 0))

    def test_an_unknown_level_raises(self, nested):
        with pytest.raises(KeyError, match="not levels of this batch"):
            nested.at_levels(nope=0)

    def test_no_indexers_returns_an_equivalent_whole_view(self, nested):
        assert nested.at_levels().batch_shape == (2, 3)


class TestElementIdentity:
    def test_an_element_derives_the_level_it_was_selected_at(self, flat):
        element = flat[2]
        assert element.name == "b[draw=2]"
        assert element.name_is_auto

    def test_nested_levels_name_every_level_selected(self, nested):
        assert nested[1][2].name == "b[chain=1, draw=2]"

    def test_at_levels_derives_the_same_name_as_positional_indexing(self, nested):
        assert nested.at_levels(chain=1, draw=2).name == nested[1][2].name

    def test_a_sub_batch_view_also_derives_its_name(self, nested):
        assert nested[1].name == "b[chain=1]"
        assert nested[1].name_is_auto

    def test_a_negative_index_names_the_position_it_resolves_to(self, flat):
        assert flat[-1].name == flat[3].name == "b[draw=3]"

    def test_a_slice_names_the_positions_it_spans(self, flat):
        assert flat[1:3].name == "b[draw=1:3]"

    def test_a_step_slice_names_its_step(self, flat):
        assert flat[0:3:2].name == "b[draw=0:3:2]"

    def test_slices_spanning_the_same_positions_name_alike(self, flat):
        """A name is a function of what is selected, not of how it was written."""
        assert flat[0:4:2].name == flat[0:3:2].name

    def test_a_multi_axis_level_names_its_axes_together(self, two_axis):
        assert two_axis.at_levels(draw=(1, slice(0, 2))).name == "b[draw=(1, 0:2)]"

    def test_selecting_the_whole_batch_derives_the_batch_s_own_name(self, nested):
        assert nested.at_levels().name == "b"
        assert nested[:].name == "b"

    def test_a_user_given_name_survives_a_no_op_selection(self, nested):
        assert not nested.at_levels().name_is_auto

    def test_a_renamed_batch_roots_the_names_of_its_own_views(self, nested):
        assert nested[1].with_name("inner")[2].name == "inner[draw=2]"

    def test_bare_elements_carry_no_identity(self):
        bare = _BareBatch(range(3), _spec([(3,)], ["draw"], ArraySpec(shape=())))
        assert bare[1] == 1
        assert not isinstance(bare[1], TrackedTerm)


class TestABC:
    def test_batch_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            Batch()  # type: ignore[abstract]

    def test_a_batch_is_a_tracked_term(self, flat):
        assert isinstance(flat, TrackedTerm)

    def test_the_storage_seam_is_abstract(self):
        assert set(Batch.__abstractmethods__) == {"_element_at", "_sub_batch_at"}


class TestDerivedNamesIdentifyTheObject:
    """A derived name is a function of what a view selects.

    Two routes to the same selection read alike, and two different selections of
    one batch never do — the property that lets a name be used to say which
    object is meant.
    """

    def test_indexing_two_levels_in_either_order_reads_alike(self, nested):
        one_call = nested.at_levels(chain=1, draw=2)
        chain_first = nested.at_levels(chain=1).at_levels(draw=2)
        draw_first = nested.at_levels(draw=2).at_levels(chain=1)
        positional = nested[1][2]
        assert (
            one_call.name
            == chain_first.name
            == draw_first.name
            == positional.name
            == "b[chain=1, draw=2]"
        )

    def test_the_same_element_is_reached_by_either_order(self, nested):
        assert (
            nested.at_levels(chain=1, draw=2).value
            == nested.at_levels(draw=2).at_levels(chain=1).value
            == nested[1][2].value
            == 5
        )

    def test_slicing_then_indexing_reads_as_the_position_selected(self, flat):
        """A position within a slice is named by where it sits in the batch."""
        assert flat[1:4][0].name == flat[1].name == "b[draw=1]"
        assert flat[1:4][0].value == 1

    def test_a_slice_of_a_slice_names_the_positions_it_still_spans(self, flat):
        assert flat[1:4][1:].name == "b[draw=2:4]"

    def test_selecting_different_levels_reads_differently(self, nested):
        """The collision a positional-only scheme cannot avoid."""
        assert nested.at_levels(chain=1).name == "b[chain=1]"
        assert nested.at_levels(draw=1).name == "b[draw=1]"
        assert nested.at_levels(chain=1).name != nested.at_levels(draw=1).name

    def test_distinct_elements_of_distinct_sub_batches_read_differently(self, nested):
        outer = nested.at_levels(draw=1)[0]
        inner = nested[1][0]
        assert outer.name != inner.name
        assert (outer.name, inner.name) == ("b[chain=0, draw=1]", "b[chain=1, draw=0]")

    def test_a_slice_view_does_not_borrow_the_batch_s_own_name(self, flat):
        assert flat[0:2].name != flat.name


class TestIndexerValidation:
    @pytest.mark.parametrize("indexer", ["draw", 1.5, object()])
    def test_an_indexer_must_be_an_integer_a_slice_or_none(self, flat, indexer):
        with pytest.raises(TypeError, match="indexed by an integer, a slice, or None"):
            flat[indexer]

    def test_at_levels_rejects_the_same_indexers(self, flat):
        with pytest.raises(TypeError, match="indexed by an integer, a slice, or None"):
            flat.at_levels(draw="first")

    def test_a_bool_is_not_an_index(self, flat):
        with pytest.raises(TypeError, match="not indexed by a bool"):
            flat[True]

    def test_a_tuple_addresses_the_leading_axes_in_order(self, nested):
        assert nested[1, 2].value == 5
        assert nested[1, 2].name == "b[chain=1, draw=2]"

    def test_a_partial_tuple_leaves_the_rest_whole(self, nested):
        assert nested[1,].name == "b[chain=1]"

    def test_too_many_indices_raises_through_the_public_index(self, flat):
        with pytest.raises(IndexError, match="too many indices"):
            flat[0, 0]

    def test_an_out_of_range_index_raises(self, flat):
        with pytest.raises(IndexError, match="out of range"):
            flat[4]


class TestViewProvenance:
    def test_a_view_records_the_indexing(self, nested, full_provenance_mode):
        view = nested[1]
        assert view.provenance is not None
        assert view.provenance.operation == "index"
        assert view.provenance.metadata["selection"] == "chain=1"

    def test_an_element_records_the_indexing(self, nested, full_provenance_mode):
        element = nested.at_levels(chain=1, draw=2)
        assert element.provenance is not None
        assert element.provenance.metadata["selection"] == "chain=1, draw=2"

    def test_the_selection_recorded_reads_against_the_parent_recorded(self, nested):
        """A two-step index records what was selected *from the batch it names*."""
        element = nested[1][2]
        assert element.provenance.metadata["selection"] == "draw=2"
        assert [parent.name for parent in element.provenance.parents] == ["b[chain=1]"]

    def test_selecting_the_whole_batch_records_no_indexing(self, nested):
        """Nothing happened, so no lineage node claims otherwise."""
        assert nested.at_levels().provenance is None
        assert nested[:].provenance is None

    def test_the_batch_indexed_is_the_parent(self, nested, full_provenance_mode):
        view = nested[1]
        assert [parent.name for parent in view.provenance.parents] == ["b"]

    def test_a_bare_element_has_nothing_to_record_on(self, full_provenance_mode):
        bare = _BareBatch(range(3), _spec([(3,)], ["draw"], ArraySpec(shape=())))
        assert bare[1] == 1


class TestImmutability:
    def test_a_batch_rejects_attribute_assignment(self, flat):
        with pytest.raises(AttributeError, match="immutable"):
            flat._spec = None

    def test_a_batch_rejects_attribute_deletion(self, flat):
        with pytest.raises(AttributeError, match="immutable"):
            del flat._spec

    def test_renaming_levels_shares_storage(self, nested):
        renamed = nested.with_level_names(chain="group")
        assert renamed.batch_shape == nested.batch_shape
        assert [element.value for element in renamed[0]] == [0, 1, 2]


class TestDegenerateAxes:
    def test_a_zero_length_axis_is_a_batch_of_nothing(self):
        empty = _ListBatch([], _spec([(0,)], ["draw"]))
        assert (len(empty), empty.batch_size) == (0, 0)
        assert list(empty) == []

    def test_indexing_an_empty_axis_raises(self):
        empty = _ListBatch([], _spec([(0,)], ["draw"]))
        with pytest.raises(IndexError, match="out of range"):
            empty[0]

    def test_a_slice_selecting_nothing_keeps_the_level(self, flat):
        none_of_it = flat[2:2]
        assert none_of_it.batch_shape == (0,)
        assert none_of_it.level_names == ("draw",)


class TestSpecValidation:
    def test_an_axis_size_must_be_integral(self):
        with pytest.raises(TypeError, match="axis sizes are integers"):
            _spec([(2.7,)], ["draw"])

    def test_a_string_is_not_an_axis_size(self):
        with pytest.raises(TypeError, match="axis sizes are integers"):
            _spec([("3",)], ["draw"])

    def test_replacing_a_field_revalidates_the_levels(self):
        from dataclasses import replace

        with pytest.raises(ValueError, match="must name every level"):
            replace(_spec([(2,)], ["draw"]), level_names=("a", "b"))


class TestDescendingSelections:
    """A negative step selects the same positions, in the reverse order.

    The positions a selection spans are carried as a ``range``, so nothing
    re-reads a slice's from-the-end bound and a reversal keeps its elements.
    """

    @pytest.mark.parametrize(
        ("indexer", "expected"),
        [
            (slice(None, None, -1), [3, 2, 1, 0]),
            (slice(2, None, -1), [2, 1, 0]),
            (slice(3, 1, -1), [3, 2]),
            (slice(None, None, -2), [3, 1]),
            (slice(None, None, 2), [0, 2]),
        ],
    )
    def test_a_reverse_slice_keeps_its_positions_in_order(self, flat, indexer, expected):
        view = flat[indexer]
        assert view.batch_shape == (len(expected),)
        assert [element.value for element in view] == expected

    def test_a_reverse_slice_keeps_its_level(self, flat):
        assert flat[::-1].level_names == ("draw",)

    @pytest.mark.parametrize(
        "indexer", [slice(None, None, -1), slice(2, None, -1), slice(3, 1, -1)]
    )
    def test_a_reverse_name_reads_back_as_the_same_selection(self, flat, indexer):
        """The rendered name is an index that reselects the same positions."""
        view = flat[indexer]
        spelled = view.name.removeprefix("b[draw=").removesuffix("]")
        start, stop, step = (part or None for part in spelled.split(":"))
        reselected = flat[slice(*(None if p is None else int(p) for p in (start, stop, step)))]
        assert reselected.name == view.name
        assert [e.value for e in reselected] == [e.value for e in view]

    def test_an_element_of_a_reverse_view_is_named_for_its_position(self, flat):
        element = flat[::-1][0]
        assert element.name == "b[draw=3]"
        assert element.value == 3

    def test_a_reverse_selection_composes_with_a_forward_one(self, flat):
        view = flat[::-1][1:3]
        assert [element.value for element in view] == [2, 1]
        assert view.name == flat[2:0:-1].name

    def test_reversing_twice_restores_the_order(self, flat):
        assert [element.value for element in flat[::-1][::-1]] == [0, 1, 2, 3]
        assert flat[::-1][::-1].name == "b"

    def test_a_reverse_selection_on_one_of_two_levels(self, nested):
        view = nested.at_levels(draw=slice(None, None, -1))
        assert view.batch_shape == (2, 3)
        assert view.name == "b[draw=2::-1]"
        assert [element.value for element in view[0]] == [2, 1, 0]


class TestSerialization:
    """A batch is immutable, and still round-trips through copy and pickle."""

    def test_a_batch_round_trips_through_pickle(self, nested):
        restored = pickle.loads(pickle.dumps(nested))
        assert restored.name == nested.name
        assert restored.spec == nested.spec
        assert [element.value for element in restored[0]] == [0, 1, 2]

    def test_a_view_round_trips_with_the_names_it_derives_from(self, nested):
        view = nested[1]
        restored = pickle.loads(pickle.dumps(view))
        assert restored.name == view.name == "b[chain=1]"
        assert restored[2].name == view[2].name == "b[chain=1, draw=2]"

    def test_a_batch_is_copyable(self, nested):
        assert copy.copy(nested).spec == nested.spec
        assert copy.deepcopy(nested).name == nested.name

    def test_the_immutability_message_names_the_concrete_class(self, flat):
        with pytest.raises(AttributeError, match="_ListBatch is immutable"):
            flat._spec = None


class TestRenamingAView:
    def test_a_renamed_view_and_its_own_views_read_the_level_alike(self, nested):
        renamed = nested[0:1].with_level_names(chain="group")
        assert renamed.name == "b[group=0:1]"
        assert renamed[0:1].name == "b[group=0:1]"
        assert renamed.at_levels().name == renamed.name

    def test_renaming_records_itself_and_leaves_room_for_more(self, nested, full_provenance_mode):
        view = nested[1]
        renamed = view.with_level_names(draw="d")
        assert renamed.provenance.operation == "with_level_names"
        assert [parent.name for parent in renamed.provenance.parents] == [view.name]
        assert renamed.provenance is not view.provenance

    def test_renaming_onto_a_dropped_root_level_name_says_why(self, nested):
        """The view's own levels allow it, but the name it derives from would not."""
        view = nested[1]
        assert view.level_names == ("draw",)
        with pytest.raises(ValueError, match="derives its name from but no longer carries"):
            view.with_level_names(draw="chain")

    def test_the_same_rename_is_fine_once_the_view_is_its_own_root(self, nested):
        assert nested[1].with_name("inner").with_level_names(draw="chain").level_names == ("chain",)


class TestBatchSpecFingerprint:
    def test_equal_specs_fingerprint_alike(self):
        one = _spec([(3,)], ["draw"], ArraySpec(shape=(2,)))
        two = _spec([(3,)], ["draw"], ArraySpec(shape=(2,)))
        assert fingerprint(one) == fingerprint(two)

    def test_the_multiplicity_is_part_of_the_digest(self):
        base = _spec([(3,)], ["draw"], ArraySpec(shape=(2,)))
        assert fingerprint(base) != fingerprint(_spec([(4,)], ["draw"], ArraySpec(shape=(2,))))
        assert fingerprint(base) != fingerprint(_spec([(3,)], ["chain"], ArraySpec(shape=(2,))))
        assert fingerprint(base) != fingerprint(_spec([(3,)], ["draw"], ArraySpec(shape=(5,))))

    def test_a_spec_in_a_template_fingerprints_by_content(self):
        one = EventTemplate(post=_spec([(3,)], ["draw"], ArraySpec(shape=(2,))), y=(2,))
        two = EventTemplate(post=_spec([(3,)], ["draw"], ArraySpec(shape=(2,))), y=(2,))
        assert fingerprint(one) == fingerprint(two)


class TestASlicedViewHoldsWhatItSelected:
    """Shapes and names agreeing is not enough: the elements must be the ones
    selected. These read values out of sliced views, so a selection that keeps
    the wrong positions cannot pass on shape alone."""

    def test_a_sliced_level_of_a_multi_level_batch(self, nested):
        view = nested.at_levels(draw=slice(0, 2))
        assert view.batch_shape == (2, 2)
        assert [[element.value for element in row] for row in view] == [[0, 1], [3, 4]]

    def test_a_sliced_outer_level_keeps_whole_inner_rows(self, nested):
        view = nested.at_levels(chain=slice(1, 2))
        assert [[element.value for element in row] for row in view] == [[3, 4, 5]]

    def test_a_slice_on_a_multi_axis_level(self, two_axis):
        view = two_axis.at_levels(draw=(slice(0, 2), slice(1, 3)))
        assert view.batch_shape == (2, 2)
        assert [element.value for element in view[0]] == [1, 2]
        assert [element.value for element in view[1]] == [4, 5]

    def test_a_scalar_fill_on_a_multi_axis_level_selects_that_row(self, two_axis):
        view = two_axis.at_levels(draw=1)
        assert [element.value for element in view] == [3, 4, 5]

    def test_a_partially_dropped_multi_axis_level_keeps_the_other_axis(self, two_axis):
        view = two_axis.at_levels(draw=(1, slice(0, 2)))
        assert view.level_names == ("draw",)
        assert view.batch_shape == (2,)
        assert [element.value for element in view] == [3, 4]

    def test_an_unnamed_level_keeps_all_of_its_elements(self, nested):
        view = nested.at_levels(chain=0)
        assert [element.value for element in view] == [0, 1, 2]

    def test_a_chain_of_mixed_selections_tracks_its_elements(self, nested):
        step = nested.at_levels(draw=slice(0, 2))
        assert [element.value for element in step[1]] == [3, 4]
        assert step[1][1].value == 4
        assert step[1][1].name == "b[chain=1, draw=1]"

    def test_the_same_element_by_three_routes(self, nested):
        one = nested.at_levels(chain=1, draw=2)
        two = nested.at_levels(draw=slice(1, 3))[1][1]
        three = nested[1][::-1][0]
        assert one.value == two.value == three.value == 5
        assert one.name == two.name == three.name == "b[chain=1, draw=2]"


class TestNamesStayOneFormPerSelection:
    """Every selection of one batch reads one way, and no two read alike."""

    def test_one_position_reads_alike_however_it_was_reached(self, flat):
        assert flat[0:1].name == flat[0:2:2].name == "b[draw=0:1]"
        assert flat[3:4].name == flat[3:2:-1].name == "b[draw=3:4]"

    def test_no_selection_reads_alike_however_it_was_written(self, flat):
        assert flat[2:2].name == flat[1:1].name == "b[draw=0:0]"

    def test_a_step_selection_is_not_read_as_the_whole_level(self, flat):
        """It spans 2 of 4 positions, so it cannot borrow the batch's own name."""
        assert flat[0:4:3].batch_shape == (2,)
        assert flat[0:4:3].name != flat.name

    def test_selections_and_names_correspond_one_to_one(self, flat):
        """Views spanning the same positions read alike; different ones differ."""
        views = [(f"{i}:{j}", flat[i:j]) for i in range(4) for j in range(i, 5)]
        by_name: dict[str, set[tuple[int, ...]]] = {}
        for _, view in views:
            content = tuple(element.value for element in view)
            by_name.setdefault(view.name, set()).add(content)
        assert all(len(contents) == 1 for contents in by_name.values())
        assert len(by_name) == len({tuple(e.value for e in v) for _, v in views})

    def test_an_element_and_a_one_element_sub_batch_read_differently(self, flat):
        """Same value, different objects, so the readings must not collide."""
        assert flat[1].name == "b[draw=1]"
        assert flat[1:2].name == "b[draw=1:2]"


@pytest.fixture
def three_level():
    """Three levels, so a middle one can be dropped with survivors either side."""
    return _ListBatch(range(24), _spec([(2,), (3,), (4,)], ["chain", "draw", "coord"]))


class TestALevelBetweenOthers:
    def test_dropping_a_middle_level_keeps_the_levels_either_side(self, three_level):
        view = three_level.at_levels(draw=1)
        assert view.axis_groups == ((2,), (4,))
        assert view.level_names == ("chain", "coord")
        assert view.name == "b[draw=1]"
        assert [element.value for element in view[0]] == [4, 5, 6, 7]

    def test_an_element_of_a_middle_dropped_view(self, three_level):
        element = three_level.at_levels(draw=1)[0][2]
        assert element.name == "b[chain=0, draw=1, coord=2]"
        assert element.value == 6

    def test_three_levels_reach_one_element_by_three_routes(self, three_level):
        one = three_level.at_levels(chain=1, draw=1, coord=3)
        two = three_level.at_levels(draw=1).at_levels(chain=1, coord=3)
        three = three_level[1, 1, 3]
        assert one.value == two.value == three.value == 19
        assert one.name == two.name == three.name == "b[chain=1, draw=1, coord=3]"

    def test_a_middle_level_selected_empty_is_kept(self, three_level):
        """A size-0 axis still survives, so the level stays and is named."""
        view = three_level.at_levels(draw=slice(2, 2))
        assert view.axis_groups == ((2,), (0,), (4,))
        assert view.at_levels(chain=1, coord=2).name == "b[chain=1, draw=0:0, coord=2]"


class TestLongerMixedChains:
    """Reverse, step, forward and whole selections composed, checked on both the
    derived name and the elements at the end of the chain."""

    @pytest.fixture
    def eight(self):
        return _ListBatch(range(8), _spec([(8,)], ["draw"]))

    @pytest.mark.parametrize(
        ("chain", "expected_name", "expected"),
        [
            (lambda b: b[::-1][1:6][::2], "b[draw=6:1:-2]", [6, 4, 2]),
            (lambda b: b[1:7][::-1][1:4][:], "b[draw=5:2:-1]", [5, 4, 3]),
            (lambda b: b[7:0:-2][1:][::-1], "b[draw=1:6:2]", [1, 3, 5]),
            (lambda b: b[:][::2][::-1], "b[draw=6::-2]", [6, 4, 2, 0]),
        ],
    )
    def test_a_chain_keeps_its_name_and_its_elements(self, eight, chain, expected_name, expected):
        view = chain(eight)
        assert view.name == expected_name
        assert [element.value for element in view] == expected

    def test_a_whole_selection_mid_chain_changes_nothing(self, eight):
        assert eight[:][::2][::-1][1].name == eight[::2][::-1][1].name

    def test_an_element_at_the_end_of_a_chain(self, eight):
        element = eight[::-1][1:6][::2][1]
        assert element.name == "b[draw=4]"
        assert element.value == 4


class TestDegenerateAxesInUse:
    def test_an_empty_view_can_be_indexed_further_and_stays_empty(self, flat):
        empty = flat[2:2]
        assert empty[0:0].name == empty[:].name == empty[::-1].name == "b[draw=0:0]"
        assert list(empty[:]) == []
        with pytest.raises(IndexError, match="out of range"):
            empty[0]

    def test_an_empty_inner_level_leaves_the_outer_one_iterable(self, nested):
        view = nested.at_levels(draw=slice(1, 1))
        assert (view.axis_groups, view.batch_size) == (((2,), (0,)), 0)
        assert [inner.name for inner in view] == ["b[chain=0, draw=0:0]", "b[chain=1, draw=0:0]"]

    def test_an_empty_level_can_still_be_renamed(self, nested):
        assert nested.at_levels(draw=slice(1, 1)).with_level_names(draw="d").name == "b[d=0:0]"

    def test_reversing_a_single_element_axis_selects_all_of_it(self):
        """One position in the same order is the whole axis, so nothing is derived."""
        one = _ListBatch([7], _spec([(1,)], ["draw"]))
        assert one[::-1].name == "b"
        assert one[::-1].provenance is None
        assert one[0].name == "b[draw=0]"

    def test_reversing_a_zero_length_axis_selects_all_of_it(self):
        empty = _ListBatch([], _spec([(2,), (0,)], ["chain", "draw"]))
        assert empty.at_levels(draw=slice(None, None, -1)).name == "b"


class TestMultiAxisLevelIndexers:
    def test_none_inside_a_level_tuple_keeps_that_axis(self, two_axis):
        view = two_axis.at_levels(draw=(None, 1))
        assert view.name == "b[draw=(0:2, 1)]"
        assert [element.value for element in view] == [1, 4]
        assert view.name == two_axis.at_levels(draw=(slice(0, 2), 1)).name

    def test_a_three_axis_level_addresses_its_axes_in_order(self):
        deep = _ListBatch(range(24), _spec([(2, 3, 4)], ["draw"]))
        view = deep.at_levels(draw=(1, None, 2))
        assert view.name == "b[draw=(1, 0:3, 2)]"
        assert [element.value for element in view] == [14, 18, 22]

    def test_a_partially_dropped_level_takes_the_indexers_it_has_left(self):
        deep = _ListBatch(range(24), _spec([(2, 3, 4)], ["draw"]))
        once = deep.at_levels(draw=1)
        assert once.axis_groups == ((3, 4),)
        assert once.at_levels(draw=(2, 3)).name == deep.at_levels(draw=(1, 2, 3)).name
        with pytest.raises(ValueError, match="has 2 axes but got 3 indexers"):
            once.at_levels(draw=(0, 0, 0))

    def test_a_multi_axis_level_beside_a_single_axis_one(self):
        both = _ListBatch(range(24), _spec([(2, 3), (4,)], ["draw", "coord"]))
        assert both[1, 2, 3].name == both.at_levels(draw=(1, 2), coord=3).name
        assert both[1, 2, 3].value == 23


class TestALevelNamedLikeAParameter:
    def test_a_level_may_be_named_self(self):
        """The receiver is positional-only, so no level name is out of reach."""
        batch = _ListBatch(range(3), _spec([(3,)], ["self"]))
        assert batch.at_levels(self=1).name == batch[1].name == "b[self=1]"
        assert batch.at_levels(self=1).value == 1

    def test_a_level_named_self_beside_another(self):
        batch = _ListBatch(range(6), _spec([(2,), (3,)], ["self", "draw"]))
        assert batch.at_levels(self=1, draw=2).value == 5


class TestABatchOfBatches:
    """Nesting needs no dedicated class: a batch is a tracked term, so a batch
    whose elements are batches is admitted as it stands."""

    @pytest.fixture
    def outer(self):
        inner_spec = _spec([(3,)], ["draw"])
        laws = [_ListBatch(range(i * 3, i * 3 + 3), inner_spec, name=f"law{i}") for i in range(2)]
        return _NestedBatch(laws, BatchSpec(inner_spec, [(2,)], ["law"]), name="outer")

    def test_the_element_spec_is_itself_a_batch_spec(self, outer):
        assert isinstance(outer.element_spec, BatchSpec)
        assert outer.spec.is_valid(outer)

    def test_an_element_is_the_inner_batch(self, outer):
        inner = outer[1]
        assert isinstance(inner, _ListBatch)
        assert inner.name == "outer[law=1]"
        assert [element.value for element in inner] == [3, 4, 5]

    def test_indexing_through_both_batches(self, outer):
        assert outer[1][2].name == "outer[law=1][draw=2]"
        assert outer[1][2].value == 5

    def test_a_whole_selection_of_the_outer_batch_keeps_its_name(self, outer):
        assert outer[0:2].name == "outer"
        assert outer[0:1].name == "outer[law=0:1]"
