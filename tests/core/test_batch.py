"""Tests for the `Batch[E]` ABC — the level algebra, not any concrete storage.

These tests exercise the shared contract through minimal doubles, each supplying
only what its case needs: `_ListBatch`, whose elements are tracked terms;
`_BareBatch`, whose elements are plain values and so carry no identity to derive;
`_NestedBatch`, whose elements are batches; `_FieldBatch`, whose elements answer a
named key; and `_ViewBatch`, which copies nothing and reads one shared store.
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
from probpipe.core.provenance import Provenance
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
        built = _Leaf(self._store[self._flat(index)], name=name, name_is_auto=True)
        return self._inherit_provenance(built)

    def _sub_batch_at(self, index, *, spec, name):
        # The index is honored in the order it presents its positions, so a
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
        return self._inherit_provenance(self._store[self._flat(index)].with_name(name))


class _BareBatch(_ListBatch):
    """A batch whose elements are bare values; nothing to name."""

    __slots__ = ()

    def _element_at(self, index, *, name):
        return self._store[self._flat(index)]


class _FieldBatch(_ListBatch):
    """A batch that answers a named key, as a batch of records will.

    The elements here have one field, ``value``, and a name reaches it across the
    whole batch. Supplying the field side of ``[]`` is all a concrete batch does;
    what a *position* means stays the ABC's.
    """

    __slots__ = ()

    def _at_fields(self, path):
        if path != ("value",):
            raise KeyError(f"{path!r} is not a field of these elements")
        return tuple(self._store)


class _ViewBatch(Batch[_Leaf]):
    """A batch that stores no elements of its own: one root store and a selection.

    A sub-batch is contracted to be a *view*, so this double copies nothing. It
    keeps the root's list, the root's shape, and which root position each of its
    axes now spans — an integer where an axis has been dropped — and resolves an
    element only when asked. That the level algebra is implementable without
    materializing anything is a fact about the ABC rather than about storage, so it
    is checked here rather than left to the concrete batches.
    """

    __slots__ = ("_root_shape", "_root_store", "_store_selection")

    def __init__(
        self, store, spec, *, name="b", name_is_auto=False, root_shape=None, store_selection=None
    ):
        object.__setattr__(self, "_root_store", store)
        object.__setattr__(self, "_root_shape", root_shape or spec.batch_shape)
        object.__setattr__(
            self,
            "_store_selection",
            store_selection
            if store_selection is not None
            else tuple(range(size) for size in spec.batch_shape),
        )
        self._init_batch(spec, name=name, name_is_auto=name_is_auto)

    def _offset(self, index):
        """Where this view's positional *index* lands in the root store."""
        positions = []
        view_axis = 0
        for entry in self._store_selection:
            if isinstance(entry, int):
                positions.append(entry)
                continue
            positions.append(entry[index[view_axis]])
            view_axis += 1
        offset = 0
        for position, size in zip(positions, self._root_shape, strict=True):
            offset = offset * size + position
        return offset

    # -- the storage seam --

    def _element_at(self, index, *, name):
        built = _Leaf(self._root_store[self._offset(index)], name=name, name_is_auto=True)
        return self._inherit_provenance(built)

    def _sub_batch_at(self, index, *, spec, name):
        # Nothing is read and nothing is copied: composing a range with the
        # indexer records which root positions the new view spans, and a range
        # sliced by a descending slice descends, so the order survives too.
        composed = []
        view_axis = 0
        for entry in self._store_selection:
            if isinstance(entry, int):
                composed.append(entry)
                continue
            composed.append(entry[index[view_axis]])
            view_axis += 1
        return type(self)(
            self._root_store,
            spec,
            name=name,
            name_is_auto=True,
            root_shape=self._root_shape,
            store_selection=tuple(composed),
        )


class _StoringBatch(Batch[_Leaf]):
    """A batch that hands back the very element the caller put in.

    The elements are stored, not built, so ``batch[i]`` is the caller's own
    object: it keeps the name and the provenance it arrived with, and nothing is
    copied. This is the storing side of the identity rule, which the doubles above
    cannot exercise — each of them builds a fresh element per index, so they would
    keep passing if the ABC ever renamed or re-attributed a borrowed object.
    """

    __slots__ = ("_store",)

    def __init__(self, elements, spec, *, name="b", name_is_auto=False):
        object.__setattr__(self, "_store", list(elements))
        self._init_batch(spec, name=name, name_is_auto=name_is_auto)

    def _element_at(self, index, *, name):
        return self._store[index[0]]

    def _sub_batch_at(self, index, *, spec, name):
        return type(self)(self._store[index[0]], spec, name=name, name_is_auto=True)


class _StringSlotsBatch(Batch[int]):
    """A double whose ``__slots__`` names its one slot as a bare string.

    It derives from :class:`Batch` directly, not from another double: a parent
    declaring the same slot in the tuple form would supply the name the string
    form fails to, and the double would pass whether or not the string was read
    correctly.
    """

    __slots__ = "_store"  # a bare string, deliberately: the point of the double

    def __init__(self, store, spec, *, name="b", name_is_auto=False):
        object.__setattr__(self, "_store", list(store))
        self._init_batch(spec, name=name, name_is_auto=name_is_auto)

    def _element_at(self, index, *, name):
        return self._store[index[0]]

    def _sub_batch_at(self, index, *, spec, name):
        return type(self)(self._store[index[0]], spec, name=name, name_is_auto=True)


class _DictBatch(_ListBatch):
    """A double that declares no ``__slots__``, so its instances carry a dict."""

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

    def test_a_flat_shape_is_not_a_grouping(self):
        """``batch_shape`` is the natural thing to reach for, and is one nesting short."""
        with pytest.raises(TypeError, match=r"is not a group"):
            _spec((4,), ("draw",))

    def test_a_bare_string_is_not_one_name_per_character(self):
        """``tuple("ab")`` is two names, which is never what a caller means."""
        with pytest.raises(TypeError, match="one name per character"):
            _spec([(2,)], "ab")


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

    def test_axis_groups_are_normalized_to_tuples(self):
        assert _spec([[2], [3]], ["chain", "draw"]).axis_groups == ((2,), (3,))

    def test_the_multiplicity_fields_are_declared_at_the_types_they_store(self):
        """The annotation is the post-construction guarantee, not the input sugar.

        The constructor takes any iterables; the fields are the tuples it stores,
        which is what makes a stored spec hashable. Mirrors the declaration
        fields on ``DistributionSpec`` and ``FunctionSpec``.
        """
        hints = get_type_hints(BatchSpec)
        assert hints["axis_groups"] == tuple[tuple[int | str, ...], ...]
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

    @pytest.mark.parametrize("new", [None, 0, [], 3.5], ids=["None", "zero", "empty-list", "float"])
    def test_a_new_name_that_is_not_a_string_is_reported_as_one(self, nested, new):
        """A falsy non-string is a wrong type, not an empty name.

        ``None``, ``0`` and ``[]`` are all falsy, so an emptiness check reached
        first would describe the wrong problem.
        """
        with pytest.raises(TypeError, match="level names are strings"):
            nested.with_level_names(chain=new)

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
    @pytest.mark.parametrize("indexer", [1.5, object()])
    def test_an_indexer_must_be_an_integer_or_a_slice(self, flat, indexer):
        with pytest.raises(TypeError, match="indexed by an integer or a slice"):
            flat[indexer]

    def test_at_levels_rejects_the_same_indexers(self, flat):
        with pytest.raises(TypeError, match="indexed by an integer or a slice"):
            flat.at_levels(draw="first")

    @pytest.mark.parametrize("bound", [2.5, "2"], ids=["float", "str"])
    def test_a_slice_bound_that_is_not_an_integer_is_placed(self, nested, bound):
        """A bound computed with ``/`` is a float, the ordinary way to arrive here."""
        with pytest.raises(TypeError, match=r"sliced by integers.*batch_shape \(2, 3\)"):
            nested[0:bound]
        with pytest.raises(TypeError, match=r"sliced by integers.*level 'draw'"):
            nested.at_levels(draw=slice(0, bound))

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
    """A selection inherits the lineage of the batch it came out of.

    Reading one position out of a collection computes nothing, so no node records
    the reading; which position it was is carried by the name.
    """

    @staticmethod
    def _from_an_operation(batch):
        """The batch as an operation would hand it over: carrying provenance."""
        return batch.with_provenance(Provenance.create("sample", parents=[]))

    def test_a_view_inherits_the_batch_provenance(self, nested, full_provenance_mode):
        produced = self._from_an_operation(nested)

        assert produced[1].provenance is produced.provenance

    def test_an_element_inherits_it_too(self, nested, full_provenance_mode):
        produced = self._from_an_operation(nested)

        assert produced.at_levels(chain=1, draw=2).provenance is produced.provenance

    def test_no_node_claims_the_indexing_happened(self, nested, full_provenance_mode):
        """The lineage names the operation that produced the batch, not the read."""
        produced = self._from_an_operation(nested)

        assert produced[1].provenance.operation == "sample"

    def test_a_view_of_a_batch_with_no_provenance_has_none(self, nested):
        """Nothing to inherit, and the reading itself was not an event."""
        assert nested.provenance is None
        assert nested[1].provenance is None
        assert nested.at_levels(chain=1, draw=2).provenance is None

    def test_selecting_the_whole_batch_inherits_the_same_way(self, nested, full_provenance_mode):
        """Nothing distinguishes the degenerate selection: it was never a special case."""
        produced = self._from_an_operation(nested)

        assert produced.at_levels().provenance is produced.provenance
        assert produced[:].provenance is produced.provenance

    def test_a_view_two_steps_down_still_reads_the_root_operation(
        self, nested, full_provenance_mode
    ):
        """Lineage does not lengthen with the route taken through the batch."""
        produced = self._from_an_operation(nested)

        assert produced[1][2].provenance is produced.provenance

    def test_a_bare_element_has_nowhere_to_carry_it(self, full_provenance_mode):
        bare = _BareBatch(range(3), _spec([(3,)], ["draw"], ArraySpec(shape=())))
        produced = self._from_an_operation(bare)

        assert produced[1] == 1

    def test_a_lineage_the_element_already_carries_is_not_overwritten(self, full_provenance_mode):
        """The batch adds nothing where the element brought its own record.

        `_NestedBatch` renames its element, and a rename carries its own
        provenance naming the original as parent — so the chain back to the
        element's origin stands, and the batch does not replace it.
        """
        inner = _ListBatch(range(2), _spec([(2,)], ["draw"]), name="inner")
        inner.with_provenance(Provenance.create("fit", parents=[]))
        outer = _NestedBatch([inner, inner], _spec([(2,)], ["chain"]), name="outer")
        self._from_an_operation(outer)

        element = outer[0]
        assert element.provenance.operation == "with_name"
        assert [parent.name for parent in element.provenance.parents] == ["inner"]


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

    def test_an_identifier_is_a_symbolic_axis_size(self):
        """A name defers a size, as an `ArraySpec` shape entry may."""
        assert _spec([("draws",)], ["draw"]).axis_groups == (("draws",),)

    def test_a_symbolic_axis_size_must_be_an_identifier(self):
        with pytest.raises(ValueError, match="must be an identifier"):
            _spec([("not an identifier",)], ["draw"])

    def test_a_numeric_string_is_not_a_size(self):
        """The likeliest slip: "3" is a name, and not one with_dims could bind."""
        with pytest.raises(ValueError, match="must be an identifier"):
            _spec([("3",)], ["draw"])

    def test_an_empty_name_is_refused(self):
        with pytest.raises(ValueError, match="must be an identifier"):
            _spec([("",)], ["draw"])

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

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(copy.copy, id="copy"),
            pytest.param(lambda b: pickle.loads(pickle.dumps(b)), id="pickle"),
            pytest.param(lambda b: b.with_name("renamed"), id="with_name"),
            pytest.param(lambda b: b.with_level_names(draw="step"), id="with_level_names"),
        ],
    )
    def test_a_string_slots_declaration_carries_its_storage(self, derive):
        """``__slots__`` may name one slot as a bare string, which is not a list of one.

        Walking ``__slots__`` by hand iterates that string into characters, none
        of which is an attribute, so the storage would be dropped in silence — a
        missing attribute being indistinguishable from an unassigned slot. Every
        way of deriving a new object from an old one has to read it: ``copy`` and
        ``pickle`` go through the state round-trip, while ``with_name`` and
        ``with_level_names`` go through ``_shallow_copy``.
        """
        batch = _StringSlotsBatch(range(3), _spec([(3,)], ["draw"]))

        derived = derive(batch)
        assert derived._store == [0, 1, 2]
        assert [element for element in derived] == [0, 1, 2]

    @pytest.mark.parametrize(
        "round_trip",
        [
            pytest.param(copy.copy, id="copy"),
            pytest.param(lambda b: pickle.loads(pickle.dumps(b)), id="pickle"),
            pytest.param(lambda b: b.with_name("renamed"), id="with_name"),
        ],
    )
    def test_a_subclass_without_slots_carries_its_instance_dict(self, round_trip):
        """A subclass that declares no ``__slots__`` keeps its attributes in a dict.

        No walk over ``__slots__`` would find them, so the state has two halves
        and both have to be restored.
        """
        batch = _DictBatch(range(3), _spec([(3,)], ["draw"]))
        object.__setattr__(batch, "extra", "kept in __dict__")

        restored = round_trip(batch)
        assert restored.extra == "kept in __dict__"
        assert restored._store == [0, 1, 2]


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


class TestIndexingDispatchesOnTheKeyType:
    """A name addresses a field within the elements, a position addresses the axes.

    The two namespaces never collide, an axis having no name and a field no
    position, which is what lets one ``[]`` serve both.
    """

    def test_a_name_reaches_the_elements_rather_than_an_axis(self, flat):
        with pytest.raises(TypeError, match="have no fields to address by name"):
            flat["x"]

    def test_a_path_of_names_reaches_the_elements_too(self, flat):
        with pytest.raises(TypeError, match=r"\('outer', 'a'\) indexes nothing"):
            flat["outer", "a"]

    def test_the_refusal_says_where_the_axes_are_addressed(self, flat):
        with pytest.raises(TypeError, match="at_levels addresses them by level name"):
            flat["x"]

    def test_a_batch_whose_elements_have_fields_answers_a_name(self):
        fields = _FieldBatch(range(6), _spec([(2,), (3,)], ["chain", "draw"]))
        assert fields["value"] == (0, 1, 2, 3, 4, 5)
        with pytest.raises(KeyError, match="not a field of these elements"):
            fields["nope"]

    def test_supplying_the_field_side_leaves_the_axis_side_alone(self):
        fields = _FieldBatch(range(6), _spec([(2,), (3,)], ["chain", "draw"]))
        assert fields[1, 2].value == 5
        assert fields[1].level_names == ("draw",)
        assert fields[:, 1].name == "b[draw=1]"

    def test_a_tuple_mixing_names_and_positions_addresses_neither(self, nested):
        with pytest.raises(TypeError, match="mixes field names with axis indexers"):
            nested[0, "x"]

    def test_a_mixed_tuple_is_blamed_on_the_mix_and_not_on_the_count(self, flat):
        # One axis and two entries, so a complaint about arity would fire first
        # and blame the count rather than the name that indexes no axis.
        with pytest.raises(TypeError, match="mixes field names"):
            flat[0, "x"]

    def test_an_empty_tuple_selects_the_whole_batch(self, nested):
        assert nested[()].batch_shape == (2, 3)
        assert nested[()].name == "b"


class TestNoneSpellsAWholeAxisByKeywordAlone:
    """``:`` is how a whole axis is written; ``None`` says it only in ``at_levels``."""

    def test_a_positional_none_is_refused(self, flat):
        with pytest.raises(TypeError, match="not indexed by None"):
            flat[None]

    def test_the_refusal_gives_the_spelling_to_use(self, flat):
        with pytest.raises(TypeError, match="write ':' for the whole axis"):
            flat[None]

    def test_a_none_inside_a_positional_tuple_is_refused(self, nested):
        # The case that motivates refusing it: an unset argument read as *all of
        # it* answers a question the caller never asked.
        with pytest.raises(TypeError, match="not indexed by None"):
            nested[0, None]

    def test_a_colon_keeps_the_axis_whole(self, nested):
        assert nested[:].name == "b"
        assert nested[:, 1].name == "b[draw=1]"
        assert nested[:, 1].batch_shape == (2,)

    def test_an_omitted_axis_is_still_kept_whole(self, nested):
        assert nested[1,].name == "b[chain=1]"

    def test_at_levels_still_spells_a_whole_axis_none(self, nested):
        assert nested.at_levels(chain=None).batch_shape == (2, 3)
        assert nested.at_levels(chain=None).name == "b"
        assert nested.at_levels(chain=None, draw=1).name == "b[draw=1]"

    def test_none_inside_a_level_tuple_is_a_whole_axis(self, two_axis):
        assert two_axis.at_levels(draw=(None, 1)).batch_shape == (2,)
        assert two_axis.at_levels(draw=(None, 1)).name == "b[draw=(0:2, 1)]"


class TestAnIndexerIsBlamedWhereItWasGiven:
    """An error names the position the caller addressed, in the caller's own terms."""

    def test_at_levels_names_the_level(self, nested):
        with pytest.raises(IndexError, match=r"out of range for level 'draw' of size 3"):
            nested.at_levels(draw=3)

    def test_at_levels_names_the_axis_within_a_multi_axis_level(self, two_axis):
        with pytest.raises(IndexError, match=r"axis 1 of level 'draw', axes \(2, 3\)"):
            two_axis.at_levels(draw=(0, 3))

    def test_a_positional_index_speaks_flat_axes(self, nested):
        with pytest.raises(IndexError, match=r"out of range for axis 1 of batch_shape \(2, 3\)"):
            nested[0, 3]

    def test_a_positional_index_does_not_claim_a_level(self, nested):
        with pytest.raises(IndexError, match="out of range") as raised:
            nested[0, 3]
        assert "level" not in str(raised.value)

    def test_a_type_error_is_placed_the_same_way(self, nested):
        with pytest.raises(TypeError, match=r"level 'chain' of size 2"):
            nested.at_levels(chain=1.5)


class TestASubBatchCanBeAView:
    """The storage hook presents a view: the elements are not copied to select them."""

    @pytest.fixture
    def viewed(self):
        return _ViewBatch(list(range(6)), _spec([(2,), (3,)], ["chain", "draw"]))

    def test_a_view_shares_the_store_it_was_taken_from(self, viewed):
        assert viewed[1]._root_store is viewed._root_store
        assert viewed[:, 1:]._root_store is viewed._root_store

    def test_selecting_the_whole_batch_copies_nothing(self, viewed):
        # Which is why the whole selection needs no short-circuit: reaching the
        # hook costs nothing once the hook returns a view.
        assert viewed[:]._root_store is viewed._root_store

    def test_a_view_reads_the_elements_it_selected(self, viewed):
        assert [leaf.value for leaf in viewed[1]] == [3, 4, 5]
        assert viewed.at_levels(chain=1, draw=2).value == 5

    def test_a_chain_of_views_composes_over_the_one_store(self, viewed):
        descending = viewed.at_levels(draw=slice(None, None, -1))
        assert [leaf.value for leaf in descending[0]] == [2, 1, 0]
        assert descending[0]._root_store is viewed._root_store

    @pytest.mark.parametrize(
        "index",
        [
            1,
            (0, 2),
            (slice(None), 1),
            (slice(None, None, -1), 0),
            (slice(1, 2), slice(None, None, -1)),
            (),
        ],
    )
    def test_a_view_and_a_copy_agree_on_names_and_elements(self, viewed, nested, index):
        # The algebra is the ABC's, so how a batch stores its elements cannot
        # change which elements a selection holds or what it is called.
        as_view, as_copy = viewed[index], nested[index]
        assert as_view.name == as_copy.name
        assert _values(as_view) == _values(as_copy)
        if isinstance(as_copy, Batch):
            assert as_view.batch_shape == as_copy.batch_shape
            assert as_view.level_names == as_copy.level_names


def _values(batch_or_element):
    """Every element value of a batch, flat, or the one value of an element."""
    if isinstance(batch_or_element, Batch):
        return [_values(batch_or_element[index]) for index in range(len(batch_or_element))]
    return batch_or_element.value


class TestRepr:
    def test_the_repr_names_the_class_the_batch_and_its_levels(self, nested):
        assert repr(nested) == "_ListBatch(name='b', chain=2, draw=3)"

    def test_a_multi_axis_level_shows_its_axes(self, two_axis):
        assert repr(two_axis) == "_ListBatch(name='b', draw=(2, 3))"

    def test_a_view_reprs_under_the_name_it_derived(self, nested):
        assert repr(nested[1]) == "_ListBatch(name='b[chain=1]', draw=3)"
        assert (
            repr(nested.at_levels(draw=slice(0, 2)))
            == "_ListBatch(name='b[draw=0:2]', chain=2, draw=2)"
        )

    def test_the_repr_reads_no_element(self):
        # A store too short for the shape: reading one would raise, and the repr
        # must not, since with_provenance interpolates the batch into an error.
        unreadable = _ListBatch([], _spec([(2,), (3,)], ["chain", "draw"]))
        assert repr(unreadable) == "_ListBatch(name='b', chain=2, draw=3)"
        with pytest.raises(IndexError):
            unreadable[0, 0]

    def test_renaming_a_level_shows_in_both_the_levels_and_the_name(self, nested):
        renamed = nested[1].with_level_names(draw="step")
        assert repr(renamed) == "_ListBatch(name='b[chain=1]', step=3)"


class TestTheTwoWaysOfIndexingCompose:
    """A name and a position address orthogonal things, so either can follow the other."""

    @pytest.fixture
    def fields(self):
        return _FieldBatch(range(6), _spec([(2,), (3,)], ["chain", "draw"]))

    def test_a_name_addresses_the_fields_of_a_view(self, fields):
        # The view is a _FieldBatch too, so it answers a name over just the
        # elements it selected.
        assert fields[1]["value"] == (3, 4, 5)
        assert fields[:, 1:]["value"] == (1, 2, 4, 5)

    def test_a_name_addresses_the_fields_of_a_descending_view(self, fields):
        assert fields.at_levels(draw=slice(None, None, -1))[0]["value"] == (2, 1, 0)

    def test_a_renamed_level_does_not_disturb_the_fields(self, fields):
        assert fields.with_level_names(draw="step")["value"] == (0, 1, 2, 3, 4, 5)

    def test_a_mixed_tuple_is_refused_whichever_comes_first(self, fields):
        for key in [(0, "value"), ("value", 0)]:
            with pytest.raises(TypeError, match="mixes field names with axis indexers"):
                fields[key]


class TestAViewOverSharedStorageBehavesLikeAnyBatch:
    """The ABC's promises hold for a batch that stores a selection rather than elements."""

    @pytest.fixture
    def viewed(self):
        return _ViewBatch(list(range(6)), _spec([(2,), (3,)], ["chain", "draw"]))

    def test_a_view_round_trips_through_pickle(self, viewed):
        # A second storage shape, to pin that the slot walk carries whatever a
        # subclass declares rather than the doubles' one list.
        view = viewed[1]
        restored = pickle.loads(pickle.dumps(view))
        assert restored.name == view.name == "b[chain=1]"
        assert [leaf.value for leaf in restored] == [3, 4, 5]

    def test_a_view_is_copyable(self, viewed):
        assert [leaf.value for leaf in copy.copy(viewed[1])] == [3, 4, 5]
        assert copy.deepcopy(viewed[1]).name == "b[chain=1]"

    def test_a_dropped_middle_level_still_resolves_the_axes_that_remain(self):
        # Three levels, the middle one indexed away: the axes on either side of
        # the gap must still line up with the store they are read from.
        spec = _spec([(2,), (2,), (2,)], ["law", "chain", "draw"])
        viewed = _ViewBatch(list(range(8)), spec)
        selected = viewed.at_levels(chain=1)
        assert selected.level_names == ("law", "draw")
        assert selected.name == "b[chain=1]"
        assert [[leaf.value for leaf in inner] for inner in selected] == [[2, 3], [6, 7]]
        assert selected[1, 0].value == 6
        assert selected._root_store is viewed._root_store

    def test_a_level_renamed_on_a_view_keeps_reading_the_same_store(self, viewed):
        renamed = viewed[1].with_level_names(draw="step")
        assert renamed.level_names == ("step",)
        assert [leaf.value for leaf in renamed] == [3, 4, 5]
        assert renamed._root_store is viewed._root_store

    def test_the_repr_names_the_concrete_class(self, viewed):
        assert repr(viewed) == "_ViewBatch(name='b', chain=2, draw=3)"


class TestAStoredElementKeepsItsOwnIdentity:
    """A batch that stores its elements hands one back exactly as it arrived.

    The other doubles build an element per index, so they say nothing about this:
    the ABC could start renaming or re-attributing a borrowed object and every one
    of them would still pass.
    """

    @pytest.fixture
    def stored(self):
        self.leaves = [
            _Leaf(value, name=f"given{value}").with_provenance(
                Provenance.create("author", parents=[])
            )
            for value in range(3)
        ]
        return _StoringBatch(self.leaves, _spec([(3,)], ["draw"]), name="b")

    def test_the_element_is_the_object_that_was_stored(self, stored):
        assert stored[1] is self.leaves[1]

    def test_the_element_keeps_the_name_it_arrived_with(self, stored):
        """Not ``b[draw=1]``: renaming it would mean handing back a copy."""
        assert stored[1].name == "given1"
        assert stored.at_levels(draw=2).name == "given2"

    def test_the_element_keeps_the_provenance_it_arrived_with(self, stored):
        assert stored[1].provenance.operation == "author"
        assert stored[1].provenance is self.leaves[1].provenance

    def test_selecting_twice_does_not_accumulate_anything(self, stored):
        """The borrowed object is not written to, so reading it again is the same."""
        once, twice = stored[1], stored[1]
        assert once is twice
        assert once.name == twice.name == "given1"
        assert once.provenance is twice.provenance

    def test_a_sub_batch_still_takes_a_derived_name(self, stored):
        """The view is the batch's own, so it is named by what it selects."""
        assert stored[0:2].name == "b[draw=0:2]"
        assert stored[0:2][0] is self.leaves[0]


class TestSymbolicMultiplicity:
    """An axis size may be a name, as an `ArraySpec` shape entry may.

    A *declaration* may defer how many elements a level holds — "returns a batch
    of `S` draws" before `S` is known. A live batch may not: it holds elements at
    positions.
    """

    def test_a_symbolic_axis_size_is_reported_as_free(self):
        spec = _spec([("S",)], ["draw"])

        assert spec.free_dims == frozenset({"S"})
        assert spec.batch_shape == ("S",)

    def test_free_dims_unions_the_element_schema_and_the_multiplicity(self):
        """Distinct names, so neither operand can pass for the union."""
        spec = BatchSpec(ArraySpec(shape=("d",)), [("S",)], ["draw"])

        assert spec.free_dims == frozenset({"S", "d"})
        assert spec.free_axis_dims == frozenset({"S"})

    def test_a_shared_name_declares_a_square_batch(self):
        """One scope: `("n",)` of arrays of shape `("n",)` is square by declaration."""
        spec = BatchSpec(ArraySpec(shape=("n",)), [("n",)], ["row"])

        assert spec.free_dims == frozenset({"n"})

    def test_only_the_multiplicity_must_be_concrete_for_a_live_batch(self):
        """How many elements there are is a different question from what one is."""
        spec = BatchSpec(ArraySpec(shape=("d",)), [(4,)], ["draw"])

        assert spec.free_axis_dims == frozenset()
        assert spec.batch_size == 4
        assert _ListBatch(range(4), spec).batch_shape == (4,)

    def test_a_concrete_multiplicity_is_free_of_dimensions(self):
        assert _spec([(2,), (3,)], ["chain", "draw"]).free_dims == frozenset()

    def test_batch_size_is_undefined_while_an_axis_is_symbolic(self):
        spec = _spec([("S",)], ["draw"])

        with pytest.raises(ValueError, match="undefined while an axis size is symbolic"):
            _ = spec.batch_size

    def test_a_live_batch_refuses_a_symbolic_axis(self):
        """A batch holds elements at positions, so its multiplicity is concrete."""
        with pytest.raises(ValueError, match="leaves the axis size S unbound"):
            _ListBatch([], _spec([("S",)], ["draw"]))

    def test_a_symbolic_axis_is_substitutable(self):
        spec = _spec([("S",)], ["draw"])

        assert spec.with_bound_dims({"S": 3}).axis_groups == ((3,),)
        assert spec.with_bound_dims({"S": 3}).free_dims == frozenset()

    def test_a_concrete_batch_still_builds(self, flat):
        assert flat.batch_shape == (4,)
        assert flat.batch_size == 4

    def test_a_symbolic_axis_binds_from_an_authoritative_spec(self):
        """What a spec reports it also binds — the third of the trio.

        The spec-to-spec path, which validates a declaration against another
        declaration rather than against a live batch.
        """
        bindings: dict[str, int] = {}

        assert _spec([("S",)], ["draw"]).bind_dims_from_spec(
            _spec([(3,)], ["draw"]), bindings, "path"
        )
        assert bindings == {"S": 3}

    def test_an_axis_and_an_element_dimension_share_one_scope(self):
        """A batch of `("n",)` over arrays of shape `("n",)` binds `n` once."""
        declared = BatchSpec(ArraySpec(shape=("n",)), [("n",)], ["row"])
        bindings: dict[str, int] = {}

        assert declared.bind_dims_from_spec(
            BatchSpec(ArraySpec(shape=(3,)), [(3,)], ["row"]), bindings, "path"
        )
        assert bindings == {"n": 3}

    def test_a_batch_that_is_not_square_is_refused(self):
        """The other half of declaring it square: 3 elements of length 5 is not."""
        declared = BatchSpec(ArraySpec(shape=("n",)), [("n",)], ["row"])
        actual = BatchSpec(ArraySpec(shape=(5,)), [(3,)], ["row"])

        with pytest.raises(ValueError, match=r"symbolic dimension 'n' to 5, .*already bound to 3"):
            declared.bind_dims_from_spec(actual, {}, "path")

    def test_binding_leaves_the_spec_unsubstituted(self):
        """Substitution waits for the closed scope, as it does for every leaf."""
        declared = _spec([("S",)], ["draw"])

        declared.bind_dims_from_spec(_spec([(3,)], ["draw"]), {}, "path")

        assert declared.axis_groups == (("S",),)

    def test_a_different_tiling_is_refused_rather_than_bound(self):
        """The tiling is structure: two levels do not bind against one."""
        declared = BatchSpec(OpaqueSpec(), [("S",), ("T",)], ["chain", "draw"])

        with pytest.raises(ValueError, match="has levels"):
            declared.bind_dims_from_spec(_spec([(3,)], ["draw"]), {}, "path")

    def test_a_name_repeated_within_one_level_binds_once(self):
        """`("n", "n")` on one level is a square grid, as it is in an array shape."""
        declared = BatchSpec(OpaqueSpec(), [("n", "n")], ["grid"])
        bindings: dict[str, int] = {}

        declared.bind_dims_from_spec(_spec([(3, 3)], ["grid"]), bindings, "path")
        assert bindings == {"n": 3}

        with pytest.raises(ValueError, match=r"'n' to 4, .*already bound to 3"):
            declared.bind_dims_from_spec(_spec([(3, 4)], ["grid"]), {}, "path")

    def test_levels_bind_their_own_dimensions(self):
        """Distinct names on distinct levels each take their own axis."""
        declared = BatchSpec(OpaqueSpec(), [("C",), ("D",)], ["chain", "draw"])
        bindings: dict[str, int] = {}

        declared.bind_dims_from_spec(_spec([(2,), (4,)], ["chain", "draw"]), bindings, "path")

        assert bindings == {"C": 2, "D": 4}

    def test_a_nested_batch_binds_at_every_level(self):
        """A batch of batches binds the outer axis and the inner one."""
        declared = BatchSpec(BatchSpec(OpaqueSpec(), [("i",)], ["inner"]), [("o",)], ["outer"])
        actual = BatchSpec(BatchSpec(OpaqueSpec(), [(5,)], ["inner"]), [(2,)], ["outer"])
        bindings: dict[str, int] = {}

        declared.bind_dims_from_spec(actual, bindings, "path")

        assert bindings == {"o": 2, "i": 5}

    def test_one_name_across_two_nesting_levels_is_one_dimension(self):
        """The outer axis and the inner one share a scope, so they must agree."""
        declared = BatchSpec(BatchSpec(OpaqueSpec(), [("n",)], ["inner"]), [("n",)], ["outer"])
        square = BatchSpec(BatchSpec(OpaqueSpec(), [(4,)], ["inner"]), [(4,)], ["outer"])
        oblong = BatchSpec(BatchSpec(OpaqueSpec(), [(5,)], ["inner"]), [(4,)], ["outer"])
        bindings: dict[str, int] = {}

        declared.bind_dims_from_spec(square, bindings, "path")
        assert bindings == {"n": 4}

        with pytest.raises(ValueError, match=r"'n' to 5, .*already bound to 4"):
            declared.bind_dims_from_spec(oblong, {}, "path")

    def test_an_element_dimension_binds_through_the_element_spec(self):
        """The element's own schema binds by the same rule one level in."""
        declared = BatchSpec(ArraySpec(shape=("d",)), [("n",)], ["item"])
        actual = BatchSpec(ArraySpec(shape=(7,)), [(3,)], ["item"])
        bindings: dict[str, int] = {}

        declared.bind_dims_from_spec(actual, bindings, "path")

        assert bindings == {"n": 3, "d": 7}
