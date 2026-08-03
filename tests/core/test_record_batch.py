"""Tests for `RecordBatch` / `NumericRecordBatch` — the columnar batch of records.

Two burdens beyond the classes themselves. This is the first batch whose elements
have *fields*, so it is the first to implement the ABC's `_at_fields` hook, and
the leaf-keyed column tests below are what pin that contract. It is also the
first batch that *materializes* its elements rather than storing them, so the
element-identity and spec-sharing assertions check the other side of the two
rules `Batch._element_at` states.
"""

from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ArraySpec,
    EventTemplate,
    FunctionBatch,
    FunctionSpec,
    NumericRecord,
    OpaqueBatch,
    Record,
    RecordSpec,
)
from probpipe.core._record_batch import NumericRecordBatch, RecordBatch

# A nested, all-numeric element: the shape issue 340 is about.
NESTED = EventTemplate(outer=EventTemplate(a=(), b=()), m=(2,))


def nested_batch(n: int = 3, **kwargs) -> NumericRecordBatch:
    """A `NumericRecordBatch` of *n* elements over `NESTED`."""
    return NumericRecordBatch(
        {
            "outer/a": jnp.arange(float(n)),
            "outer/b": jnp.arange(float(n)) * 2,
            "m": jnp.zeros((n, 2)),
        },
        "draw",
        element_spec=NESTED,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_columns_and_batch_shape(self):
        batch = nested_batch()
        assert batch.batch_shape == (3,)
        assert batch.batch_size == 3
        assert batch.level_names == ("draw",)
        assert batch.axis_groups == ((3,),)

    def test_a_nested_column_mapping_is_flattened(self):
        by_path = nested_batch()
        nested = NumericRecordBatch(
            {"outer": {"a": jnp.arange(3.0), "b": jnp.arange(3.0) * 2}, "m": jnp.zeros((3, 2))},
            "draw",
            element_spec=NESTED,
        )
        assert nested == by_path

    def test_event_shape_is_split_off_the_column(self):
        # ``m`` declares (2,), so a (3, 2) column is three elements, not six.
        batch = nested_batch()
        assert batch.batch_shape == (3,)
        assert batch["m"].shape == (3, 2)

    def test_multi_level_construction(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((4, 100, 2))},
            ("chain", "draw"),
            element_spec=EventTemplate(x=(2,)),
        )
        assert batch.batch_shape == (4, 100)
        assert batch.axis_groups == ((4,), (100,))

    def test_a_level_spanning_several_axes(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((2, 3, 5))},
            ("grid", "draw"),
            element_spec=EventTemplate(x=()),
            axis_groups=((2, 3), (5,)),
        )
        assert batch.axis_groups == ((2, 3), (5,))
        assert batch.batch_shape == (2, 3, 5)

    # -- refusals -----------------------------------------------------------

    def test_missing_and_unexpected_columns_are_named(self):
        with pytest.raises(ValueError, match=r"missing \['m'\]"):
            NumericRecordBatch(
                {"outer/a": jnp.zeros(3), "outer/b": jnp.zeros(3)}, "draw", element_spec=NESTED
            )
        with pytest.raises(ValueError, match=r"unexpected \['z'\]"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "z": jnp.zeros(3)}, "draw", element_spec=EventTemplate(x=())
            )

    def test_a_bad_column_shape_names_the_leaf_path(self):
        with pytest.raises(ValueError, match=r"the column at 'm' has shape \(3, 5\)"):
            NumericRecordBatch(
                {"outer/a": jnp.zeros(3), "outer/b": jnp.zeros(3), "m": jnp.zeros((3, 5))},
                "draw",
                element_spec=NESTED,
            )

    def test_columns_disagreeing_on_the_batch_axis_raise(self):
        with pytest.raises(ValueError, match="the column at 'y' has shape"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "y": jnp.zeros(4)},
                "draw",
                element_spec=EventTemplate(x=(), y=()),
            )

    def test_a_batch_needs_at_least_one_axis(self):
        with pytest.raises(ValueError, match="at least one batch axis"):
            NumericRecordBatch({"x": jnp.zeros(2)}, "draw", element_spec=EventTemplate(x=(2,)))

    def test_empty_columns_raise(self):
        with pytest.raises(ValueError, match="at least one column"):
            NumericRecordBatch({}, "draw", element_spec=EventTemplate(x=()))

    def test_a_non_mapping_columns_argument_raises(self):
        with pytest.raises(TypeError, match="columns must be a mapping"):
            NumericRecordBatch([jnp.zeros(3)], "draw", element_spec=EventTemplate(x=()))

    def test_axis_groups_must_tile_the_batch_shape(self):
        with pytest.raises(ValueError, match="must tile"):
            NumericRecordBatch(
                {"x": jnp.zeros((3, 4))},
                "draw",
                element_spec=EventTemplate(x=()),
                axis_groups=((3, 5),),
            )

    def test_a_missing_level_name_raises(self):
        with pytest.raises(ValueError, match="need 2 level names"):
            NumericRecordBatch({"x": jnp.zeros((3, 4))}, "draw", element_spec=EventTemplate(x=()))

    def test_element_spec_must_be_a_record_declaration(self):
        with pytest.raises(TypeError, match="RecordSpec or an EventTemplate"):
            NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=ArraySpec(shape=()))

    def test_numeric_batch_refuses_a_non_numeric_element_spec(self):
        with pytest.raises(TypeError, match="carries a NumericEventTemplate"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "label": np.array(["a", "b", "c"], dtype=object)},
                "draw",
                element_spec=EventTemplate(x=(), label=None),
            )

    def test_numeric_batch_refuses_a_non_numeric_column(self):
        with pytest.raises(TypeError, match="non-numeric dtype"):
            NumericRecordBatch(
                {"x": np.array(["a", "b", "c"])}, "draw", element_spec=EventTemplate(x=())
            )

    # -- the declaration, either form --------------------------------------

    def test_a_record_spec_declaration_is_accepted_and_stored(self):
        spec = RecordSpec(EventTemplate(x=(2,)))
        batch = NumericRecordBatch({"x": jnp.zeros((3, 2))}, "draw", element_spec=spec)
        assert batch.element_spec is spec
        assert batch.event_template is spec.event_template

    def test_the_two_declaration_forms_agree(self):
        template = EventTemplate(x=(2,))
        columns = {"x": jnp.zeros((3, 2))}
        assert NumericRecordBatch(dict(columns), "draw", element_spec=template) == (
            NumericRecordBatch(dict(columns), "draw", element_spec=RecordSpec(template))
        )

    def test_spec_accessors_are_views_on_one_object(self):
        batch = nested_batch()
        assert batch.element_spec is batch.spec.element_spec
        assert batch.event_template is batch.element_spec.event_template

    def test_default_name_is_the_class_name_marked_auto(self):
        batch = nested_batch()
        assert batch.name == "numericrecordbatch"
        assert batch.name_is_auto
        assert not nested_batch(name="post").name_is_auto


# ---------------------------------------------------------------------------
# Issue 340: leaf-keyed field columns
# ---------------------------------------------------------------------------


class TestLeafKeyedFieldColumns:
    """A nested field must be reachable, which is what issue 340 was about.

    Under the previous batch storage a nested top-level name was a subtree that
    ``[]`` refused, so a nested record could not be batched and read back.
    """

    def test_the_event_template_is_leaf_keyed(self):
        assert tuple(nested_batch().event_template.keys()) == ("outer/a", "outer/b", "m")

    def test_a_nested_leaf_path_yields_its_column(self):
        batch = nested_batch()
        np.testing.assert_array_equal(np.asarray(batch["outer/a"]), np.asarray([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(np.asarray(batch["outer/b"]), np.asarray([0.0, 2.0, 4.0]))

    def test_a_tuple_path_reaches_the_same_column(self):
        batch = nested_batch()
        np.testing.assert_array_equal(np.asarray(batch["outer", "a"]), np.asarray(batch["outer/a"]))

    def test_an_interior_path_yields_a_sub_batch(self):
        batch = nested_batch()
        sub = batch["outer"]
        assert isinstance(sub, NumericRecordBatch)
        assert sub.batch_shape == batch.batch_shape
        assert sub.level_names == batch.level_names
        # Re-keyed relative to the path: a batch of the sub-records, not of parents.
        assert tuple(sub.event_template.keys()) == ("a", "b")
        np.testing.assert_array_equal(np.asarray(sub["a"]), np.asarray(batch["outer/a"]))

    def test_an_interior_view_shares_the_parents_storage(self):
        batch = nested_batch()
        assert batch["outer"]._columns["a"] is batch._columns["outer/a"]

    def test_the_repro_from_the_issue(self):
        # A nested batched record, built from a flat matrix as the issue does.
        template = EventTemplate(outer=EventTemplate(a=(), b=()), m=())
        batch = NumericRecordBatch.from_vector(
            "post", template, jnp.arange(15.0).reshape(5, 3), level_name="draw"
        )
        assert tuple(batch.event_template.keys()) == ("outer/a", "outer/b", "m")
        assert batch["outer/a"].shape == (5,)
        assert isinstance(batch["outer"], NumericRecordBatch)
        assert batch["m"].shape == (5,)

    def test_an_unknown_path_names_the_fields_there_are(self):
        with pytest.raises(KeyError, match="neither a field nor an interior node"):
            nested_batch()["nope"]


# ---------------------------------------------------------------------------
# A column in the batch form its spec calls for
# ---------------------------------------------------------------------------


class TestColumnBatchForms:
    def test_an_array_field_yields_the_array_itself(self):
        batch = nested_batch()
        assert batch["m"] is batch._columns["m"]

    def test_a_callable_field_yields_a_function_batch(self):
        functions = np.empty(2, dtype=object)
        functions[0], functions[1] = (lambda x: x), (lambda x: 2 * x)
        batch = RecordBatch(
            {"f": functions, "x": jnp.zeros(2)},
            "variant",
            element_spec=EventTemplate({"f": FunctionSpec(), "x": ()}),
            name="fs",
        )
        column = batch["f"]
        assert isinstance(column, FunctionBatch)
        assert column.batch_shape == (2,)
        assert column.level_names == ("variant",)
        assert column[1](3) == 6

    def test_an_opaque_field_yields_an_opaque_batch(self):
        labels = np.empty(2, dtype=object)
        labels[0], labels[1] = "north", "south"
        batch = RecordBatch(
            {"site": labels, "x": jnp.zeros(2)},
            "row",
            element_spec=EventTemplate(site=None, x=()),
            name="design",
        )
        column = batch["site"]
        assert isinstance(column, OpaqueBatch)
        assert column[0] == "north"
        assert column.level_names == ("row",)

    def test_a_column_batch_carries_a_derived_name(self):
        labels = np.empty(2, dtype=object)
        labels[0], labels[1] = "north", "south"
        batch = RecordBatch(
            {"site": labels}, "row", element_spec=EventTemplate(site=None), name="design"
        )
        assert batch["site"].name == "design['site']"


# ---------------------------------------------------------------------------
# A batch is a collection, not a named tree
# ---------------------------------------------------------------------------


class TestCollectionNotTree:
    @pytest.mark.parametrize(
        "attribute",
        ["keys", "values", "items", "children", "at_path", "is_field", "fields", "map", "replace"],
    )
    def test_the_field_keyed_mapping_protocol_is_absent(self, attribute):
        assert not hasattr(nested_batch(), attribute)

    def test_len_and_iter_range_over_the_batch(self):
        batch = nested_batch(4)
        assert len(batch) == 4
        assert [element["outer/a"] for element in batch] == [0.0, 1.0, 2.0, 3.0]

    def test_len_is_the_leading_axis_of_a_multi_level_batch(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((4, 100))}, ("chain", "draw"), element_spec=EventTemplate(x=())
        )
        assert len(batch) == 4


# ---------------------------------------------------------------------------
# Elements: materialized, named, and sharing the batch's spec
# ---------------------------------------------------------------------------


class TestElements:
    def test_an_element_is_a_record_over_the_shared_template(self):
        element = nested_batch(name="post")[1]
        assert isinstance(element, NumericRecord)
        assert element["outer/a"] == 1.0
        assert element["outer/b"] == 2.0
        assert element.event_template == NESTED

    def test_an_element_takes_the_derived_name_marked_auto(self):
        element = nested_batch(name="post")[1]
        assert element.name == "post[draw=1]"
        assert element.name_is_auto

    def test_an_element_shares_the_batchs_spec_object(self):
        """Materializing a row must not allocate a declaration.

        ``is``, not ``==``: a record keeps a supplied ``RecordSpec`` verbatim, so
        batch and element agree structurally. An equality check would still pass
        if that reuse were lost, and every row inside a trace would allocate.
        """
        batch = nested_batch()
        assert batch[0].spec is batch.element_spec
        assert batch[2].spec is batch[0].spec

    def test_an_element_of_a_multi_level_batch_names_both_levels(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((2, 3))},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            name="post",
        )
        assert batch[1, 2].name == "post[chain=1, draw=2]"

    def test_an_element_inherits_the_batchs_provenance(self):
        from probpipe import Provenance

        batch = nested_batch(name="post")
        batch.with_provenance(Provenance.create("sample", parents=[]))
        assert batch[0].provenance is batch.provenance


# ---------------------------------------------------------------------------
# Level algebra over records
# ---------------------------------------------------------------------------


class TestLevels:
    def test_indexing_a_level_drops_it(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((2, 3))},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            name="post",
        )
        inner = batch[0]
        assert isinstance(inner, NumericRecordBatch)
        assert inner.level_names == ("draw",)
        assert inner.batch_shape == (3,)
        assert inner.name == "post[chain=0]"

    def test_at_levels_selects_by_name(self):
        batch = NumericRecordBatch(
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            name="post",
        )
        assert batch.at_levels(draw=2).level_names == ("chain",)
        assert batch.at_levels(chain=1, draw=2).name == "post[chain=1, draw=2]"
        assert float(np.asarray(batch.at_levels(chain=1, draw=2)["x"])) == 5.0

    def test_with_level_names_renames(self):
        renamed = nested_batch().with_level_names(draw="sample")
        assert renamed.level_names == ("sample",)
        assert renamed.batch_shape == (3,)

    def test_a_slice_keeps_the_level_and_shares_storage(self):
        batch = nested_batch(4)
        sliced = batch[1:3]
        assert isinstance(sliced, NumericRecordBatch)
        assert sliced.batch_shape == (2,)
        assert sliced.level_names == ("draw",)
        np.testing.assert_array_equal(np.asarray(sliced["outer/a"]), np.asarray([1.0, 2.0]))

    def test_selecting_an_empty_range_is_allowed(self):
        empty = nested_batch()[0:0]
        assert empty.batch_shape == (0,)

    def test_a_view_of_a_numeric_batch_is_a_numeric_batch(self):
        """The class is its own view type — there is no separate view class."""
        batch = nested_batch()
        assert type(batch[0:2]) is NumericRecordBatch
        assert type(batch["outer"]) is NumericRecordBatch


# ---------------------------------------------------------------------------
# select / select_all
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_returns_single_field_views(self):
        batch = nested_batch(name="post")
        selected = batch.select("m")
        assert set(selected) == {"m"}
        view = selected["m"]
        assert isinstance(view, RecordBatch)
        assert tuple(view.event_template.keys()) == ("m",)
        assert view._columns["m"] is batch._columns["m"]

    def test_a_view_carries_the_parents_levels(self):
        view = nested_batch().select_all()["m"]
        assert view.level_names == ("draw",)
        assert view.axis_groups == ((3,),)

    def test_select_all_covers_every_field_by_path(self):
        assert set(nested_batch().select_all()) == {"outer/a", "outer/b", "m"}

    def test_keywords_remap(self):
        selected = nested_batch().select(value="m")
        assert set(selected) == {"value"}
        assert tuple(selected["value"].event_template.keys()) == ("m",)

    def test_an_unknown_field_raises(self):
        with pytest.raises(KeyError, match="no field 'nope'"):
            nested_batch().select("nope")


# ---------------------------------------------------------------------------
# The flat layout
# ---------------------------------------------------------------------------


class TestFlatLayout:
    def test_to_vector_shape_and_order(self):
        batch = nested_batch()
        vec = batch.to_vector()
        assert vec.shape == (3, batch.event_template.vector_size) == (3, 4)
        # Canonical order: outer/a, outer/b, then m's two entries.
        np.testing.assert_array_equal(np.asarray(vec[1]), np.asarray([1.0, 2.0, 0.0, 0.0]))

    def test_round_trip_through_a_vector(self):
        batch = nested_batch()
        rebuilt = NumericRecordBatch.from_vector(
            batch.name, batch.event_template, batch.to_vector(), level_name="draw"
        )
        assert rebuilt == batch

    def test_from_vector_refuses_an_unbatched_vector(self):
        with pytest.raises(TypeError, match=r"NumericRecord\.from_vector"):
            NumericRecordBatch.from_vector(
                "v", EventTemplate(x=(2,)), jnp.zeros(2), level_name="draw"
            )

    def test_from_vector_checks_the_trailing_axis(self):
        with pytest.raises(ValueError, match="the trailing axis is 3, expected 2"):
            NumericRecordBatch.from_vector(
                "v", EventTemplate(x=(2,)), jnp.zeros((5, 3)), level_name="draw"
            )

    def test_multi_level_vectorization_flattens_outermost_first(self):
        batch = NumericRecordBatch(
            {"x": jnp.arange(24.0).reshape(2, 3, 4)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=(4,)),
        )
        # batch_shape is the flat concatenation, so to_vector keeps both levels.
        assert batch.to_vector().shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------


class TestStack:
    def test_stack_builds_one_named_level(self):
        records = [NumericRecord("r", x=jnp.asarray(float(i))) for i in range(3)]
        batch = NumericRecordBatch.stack(records, level_name="draw")
        assert batch.batch_shape == (3,)
        assert batch.level_names == ("draw",)
        np.testing.assert_array_equal(np.asarray(batch["x"]), np.asarray([0.0, 1.0, 2.0]))

    def test_stack_supports_a_nested_template(self):
        """The previous batch storage refused this; leaf-keyed columns make it free."""
        records = [
            Record(
                "r", {"outer/a": float(i), "outer/b": 0.0, "m": jnp.zeros(2)}, event_template=NESTED
            )
            for i in range(4)
        ]
        batch = NumericRecordBatch.stack(records, level_name="draw")
        assert batch.batch_shape == (4,)
        assert tuple(batch.event_template.keys()) == ("outer/a", "outer/b", "m")
        np.testing.assert_array_equal(
            np.asarray(batch["outer/a"]), np.asarray([0.0, 1.0, 2.0, 3.0])
        )

    def test_stack_refuses_an_empty_list(self):
        with pytest.raises(ValueError, match="at least one record"):
            NumericRecordBatch.stack([], level_name="draw")

    def test_stack_takes_the_spec_from_the_first_record(self):
        spec = RecordSpec(EventTemplate(x=ArraySpec(shape=(), dtype=jnp.float32)))
        records = [
            NumericRecord("r", {"x": jnp.asarray(1.0, dtype=jnp.float32)}, event_template=spec)
            for _ in range(2)
        ]
        assert NumericRecordBatch.stack(records, level_name="draw").element_spec == spec

    def test_stack_stores_a_non_array_field_as_an_object_column(self):
        records = [
            Record("r", {"site": s}, event_template=EventTemplate(site=None))
            for s in ("north", "south")
        ]
        batch = RecordBatch.stack(records, level_name="row")
        assert isinstance(batch["site"], OpaqueBatch)
        assert batch["site"][1] == "south"


# ---------------------------------------------------------------------------
# Equality, repr, copying
# ---------------------------------------------------------------------------


class TestEqualityAndCopying:
    def test_equal_columns_and_spec_compare_equal(self):
        assert nested_batch() == nested_batch()

    def test_a_different_level_name_compares_unequal(self):
        assert nested_batch() != nested_batch().with_level_names(draw="sample")

    def test_different_data_compares_unequal(self):
        assert nested_batch(3) != nested_batch(4)

    def test_a_different_class_is_not_comparable(self):
        assert nested_batch().__eq__(object()) is NotImplemented

    def test_reflexive_with_nan(self):
        batch = NumericRecordBatch(
            {"x": jnp.asarray([jnp.nan, 1.0])}, "draw", element_spec=EventTemplate(x=())
        )
        assert batch == batch

    def test_unhashable(self):
        with pytest.raises(TypeError):
            hash(nested_batch())

    def test_repr_reports_the_levels_without_reading_a_column(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros((4, 100))},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            name="post",
        )
        assert repr(batch) == "NumericRecordBatch(name='post', chain=4, draw=100)"

    def test_pickle_round_trip(self):
        batch = nested_batch(name="post")
        rebuilt = pickle.loads(pickle.dumps(batch))
        assert rebuilt == batch
        assert rebuilt.name == "post"

    def test_copy_round_trip(self):
        batch = nested_batch()
        assert copy.copy(batch) == batch
        assert copy.deepcopy(batch) == batch

    def test_immutability(self):
        batch = nested_batch()
        with pytest.raises(AttributeError, match="immutable"):
            batch._columns = {}


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_flatten_unflatten_round_trip(self):
        batch = nested_batch(name="post")
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        assert len(leaves) == 3
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == batch
        assert rebuilt.name == "post"

    def test_the_spec_rides_in_the_aux_by_identity(self):
        batch = nested_batch()
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        assert jax.tree_util.tree_unflatten(treedef, leaves).spec is batch.spec

    def test_batches_sharing_a_spec_share_a_treedef(self):
        assert jax.tree_util.tree_structure(nested_batch()) == jax.tree_util.tree_structure(
            nested_batch()
        )

    def test_a_different_spec_gives_a_different_treedef(self):
        other = NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=EventTemplate(x=()))
        assert jax.tree_util.tree_structure(nested_batch()) != jax.tree_util.tree_structure(other)

    def test_tree_map_preserves_the_batch(self):
        batch = nested_batch()
        doubled = jax.tree.map(lambda column: column * 2, batch)
        assert isinstance(doubled, NumericRecordBatch)
        np.testing.assert_array_equal(np.asarray(doubled["outer/a"]), np.asarray([0.0, 2.0, 4.0]))

    def test_vmap_strips_the_batch_axis_inside_the_trace(self):
        """Unflatten must not validate: a traced column's shape is transform-relative."""
        batch = nested_batch()
        seen: list[tuple[int, ...]] = []

        def row_sum(element):
            seen.append(tuple(element["m"].shape))
            return element["outer/a"] + element["outer/b"]

        result = jax.vmap(row_sum)(batch)
        np.testing.assert_array_equal(np.asarray(result), np.asarray([0.0, 3.0, 6.0]))
        assert seen == [(2,)]

    def test_jit_round_trips_the_batch(self):
        batch = nested_batch()

        @jax.jit
        def total(b):
            return b["outer/a"].sum()

        assert float(total(batch)) == 3.0

    def test_grad_through_a_column(self):
        batch = nested_batch()

        def loss(b):
            return (b["outer/b"] ** 2).sum()

        gradient = jax.grad(loss)(batch)
        assert isinstance(gradient, NumericRecordBatch)
        np.testing.assert_array_equal(np.asarray(gradient["outer/b"]), np.asarray([0.0, 4.0, 8.0]))
