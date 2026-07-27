"""Tests for the `Batch[E]` ABC — the level algebra, not any concrete storage.

The concrete batch types (`FunctionBatch`, `RecordBatch`, `DistributionBatch`)
land separately; these tests exercise the shared contract through two minimal
doubles: `_ListBatch`, whose elements are tracked terms, and `_BareBatch`, whose
elements are plain values and so carry no identity to derive.
"""

from __future__ import annotations

import pytest

from probpipe import ArraySpec, OpaqueSpec, TermSpec
from probpipe.core._batch import Batch, BatchSpec
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

    __slots__ = ("_name", "_name_is_auto", "_provenance", "_spec", "_store")

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
        kept = [
            self._store[self._flat(position)]
            for position in _positions(self.batch_shape)
            if _selected(position, index)
        ]
        return type(self)(kept, spec, name=name, name_is_auto=True)


class _BareBatch(_ListBatch):
    """A batch whose elements are bare values; nothing to name."""

    __slots__ = ()

    def _element_at(self, index, *, name):
        return self._store[self._flat(index)]


def _positions(shape):
    if not shape:
        yield ()
        return
    for head in range(shape[0]):
        for rest in _positions(shape[1:]):
            yield (head, *rest)


def _selected(position, index):
    return all(
        coordinate == indexer
        if isinstance(indexer, int)
        else coordinate in range(*indexer.indices(size))
        for coordinate, indexer, size in zip(position, index, _SHAPE_OF[len(index)], strict=True)
    )


_SHAPE_OF: dict[int, tuple[int, ...]] = {}


@pytest.fixture
def flat():
    """A single-level batch: 4 elements on one ``draw`` axis."""
    _SHAPE_OF[1] = (4,)
    return _ListBatch(range(4), _spec([(4,)], ["draw"]))


@pytest.fixture
def nested():
    """Two levels: ``chain`` of 2 over ``draw`` of 3."""
    _SHAPE_OF[2] = (2, 3)
    return _ListBatch(range(6), _spec([(2,), (3,)], ["chain", "draw"]))


@pytest.fixture
def two_axis():
    """One two-axis level, to exercise partial level indexers."""
    _SHAPE_OF[2] = (2, 3)
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
        _SHAPE_OF[1] = (3,)
        bare = _BareBatch(range(3), _spec([(3,)], ["draw"], ArraySpec(shape=())))
        assert bare.element_spec == ArraySpec(shape=())
        assert isinstance(bare.spec, BatchSpec)

    def test_axis_groups_are_normalised_to_tuples(self):
        assert _spec([[2], [3]], ["chain", "draw"]).axis_groups == ((2,), (3,))

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

    @pytest.mark.parametrize(
        ("existing", "candidate", "expected"),
        [
            ((), "draw", "draw"),
            (("draw",), "draw", "draw2"),
            (("draw", "draw2"), "draw", "draw3"),
            (("draw", "draw3"), "draw", "draw2"),
            (("chain",), "draw", "draw"),
        ],
    )
    def test_disambiguate_appends_the_smallest_free_suffix(self, existing, candidate, expected):
        assert Batch.disambiguate_level_name(existing, candidate) == expected


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
    def test_an_element_derives_name_index_and_is_marked_auto(self, flat):
        element = flat[2]
        assert element.name == "b[2]"
        assert element.name_is_auto

    def test_nested_levels_compose_the_scheme(self, nested):
        assert nested[1][2].name == "b[1][2]"

    def test_at_levels_derives_the_same_name_as_positional_indexing(self, nested):
        assert nested.at_levels(chain=1, draw=2).name == nested[1][2].name

    def test_a_sub_batch_view_also_derives_its_name(self, nested):
        assert nested[1].name == "b[1]"
        assert nested[1].name_is_auto

    def test_bare_elements_carry_no_identity(self):
        _SHAPE_OF[1] = (3,)
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
