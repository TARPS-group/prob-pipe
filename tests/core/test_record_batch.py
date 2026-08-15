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
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ArrayBackend,
    EventTemplate,
    FunctionBatch,
    FunctionSpec,
    NumericArraySpec,
    NumericRecord,
    OpaqueBatch,
    Record,
    RecordSpec,
    register_array_backend,
)
from probpipe.core import _array_backend
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._record_batch import RecordBatch
from probpipe.core.event_template import OpaqueSpec


@pytest.fixture
def clean_registry():
    """Snapshot and restore the backend registry around a test's registrations."""
    saved = dict(_array_backend._backend_registry)
    yield
    _array_backend._backend_registry.clear()
    _array_backend._backend_registry.update(saved)


# A nested, all-numeric element: the shape a nested field makes.
NESTED = EventTemplate(outer=EventTemplate(a=(), b=()), m=(2,))


def _object_column(values: list) -> np.ndarray:
    """*values* as an object array, without unpacking any of them."""
    column = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        column[position] = value
    return column


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
    def test_fields_and_batch_shape(self):
        batch = nested_batch()
        assert batch.batch_shape == (3,)
        assert batch.batch_size == 3
        assert batch.level_names == ("draw",)
        assert batch.axis_groups == ((3,),)

    def test_a_nested_field_mapping_is_flattened(self):
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

    def test_missing_and_unexpected_fields_are_named(self):
        with pytest.raises(ValueError, match=r"missing \['m'\]"):
            NumericRecordBatch(
                {"outer/a": jnp.zeros(3), "outer/b": jnp.zeros(3)}, "draw", element_spec=NESTED
            )
        with pytest.raises(ValueError, match=r"unexpected \['z'\]"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "z": jnp.zeros(3)}, "draw", element_spec=EventTemplate(x=())
            )

    def test_a_column_whose_trailing_axes_are_not_the_event_shape_is_named(self):
        # ``m`` declares (2,), so a (3, 5) column's trailing (5,) is not its event
        # shape — caught against the declaration, not merely against the batch axes.
        with pytest.raises(ValueError, match=r"the column at 'm' has shape \(3, 5\)"):
            NumericRecordBatch(
                {"outer/a": jnp.zeros(3), "outer/b": jnp.zeros(3), "m": jnp.zeros((3, 5))},
                "draw",
                element_spec=NESTED,
            )

    def test_fields_disagreeing_on_the_batch_axis_raise(self):
        with pytest.raises(ValueError, match=r"disagree on the batch axes — 'x' carries \(3,\)"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "y": jnp.zeros(4)},
                "draw",
                element_spec=EventTemplate(x=(), y=()),
            )

    def test_a_batch_needs_at_least_one_axis(self):
        with pytest.raises(ValueError, match="at least one batch axis"):
            NumericRecordBatch({"x": jnp.zeros(2)}, "draw", element_spec=EventTemplate(x=(2,)))

    def test_no_fields_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            NumericRecordBatch({}, "draw", element_spec=EventTemplate(x=()))

    def test_a_non_mapping_fields_argument_raises(self):
        with pytest.raises(TypeError, match="fields must be a mapping"):
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
            NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=NumericArraySpec(shape=()))

    def test_numeric_batch_refuses_a_non_numeric_element_spec(self):
        with pytest.raises(TypeError, match="carries a NumericEventTemplate"):
            NumericRecordBatch(
                {"x": jnp.zeros(3), "label": np.array(["a", "b", "c"], dtype=object)},
                "draw",
                element_spec=EventTemplate(x=(), label=None),
            )

    def test_numeric_batch_refuses_a_non_numeric_column(self):
        with pytest.raises(TypeError, match="its column is a numeric array"):
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
    """A nested field must be reachable, by leaf path and by the node above it.

    A batch keyed by top-level name cannot answer for a nested field at all: the
    name reaches a subtree, which is not a column. Keying by leaf path is what
    lets a nested record be batched and read back.
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

    def test_a_nested_template_round_trips_through_a_flat_matrix(self):
        template = EventTemplate(outer=EventTemplate(a=(), b=()), m=())
        batch = NumericRecordBatch.from_vector(
            "post", template, jnp.arange(15.0).reshape(5, 3), level_names="draw"
        )
        assert tuple(batch.event_template.keys()) == ("outer/a", "outer/b", "m")
        assert batch["outer/a"].shape == (5,)
        assert isinstance(batch["outer"], NumericRecordBatch)
        assert batch["m"].shape == (5,)

    def test_a_field_name_that_prefixes_another_is_not_mistaken_for_a_subtree(self):
        """``out`` is a field; ``outer/a`` is under a different subtree. The
        separator belongs to the prefix, or the two would collide."""
        template = EventTemplate({"out": (), "outer": EventTemplate(a=())})
        batch = NumericRecordBatch(
            {"out": jnp.zeros(2), "outer/a": jnp.ones(2)}, "draw", element_spec=template
        )
        assert tuple(batch.event_template.keys()) == ("out", "outer/a")
        np.testing.assert_array_equal(np.asarray(batch["out"]), np.asarray([0.0, 0.0]))
        sub = batch["outer"]
        assert isinstance(sub, NumericRecordBatch)
        np.testing.assert_array_equal(np.asarray(sub["a"]), np.asarray([1.0, 1.0]))

    def test_sibling_subtrees_may_reuse_a_leaf_name(self):
        template = EventTemplate(a=EventTemplate(c=()), b=EventTemplate(c=()))
        batch = NumericRecordBatch(
            {"a/c": jnp.zeros(2), "b/c": jnp.ones(2)}, "draw", element_spec=template
        )
        np.testing.assert_array_equal(np.asarray(batch["a"]["c"]), np.asarray([0.0, 0.0]))
        np.testing.assert_array_equal(np.asarray(batch["b"]["c"]), np.asarray([1.0, 1.0]))

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

    def test_a_field_with_no_batch_form_is_refused_at_construction(self):
        """A batch admits the element kinds it can present, and no more.

        Reading a field gives the batch of its element kind, and a distribution
        has none — so admitting the field and refusing the read would make a
        batch nobody can take a field from. The refusal moves to where the field
        is declared."""
        from probpipe import DistributionSpec, Normal

        law = Normal(0.0, 1.0, name="n")
        spec = EventTemplate({"d": DistributionSpec(law.event_template), "x": ()})
        with pytest.raises(TypeError, match="DistributionSpec, which has no batch form"):
            RecordBatch(
                {"d": _object_column([law, law]), "x": jnp.zeros(2)}, "row", element_spec=spec
            )

    def test_a_column_batch_carries_a_derived_name(self):
        labels = np.empty(2, dtype=object)
        labels[0], labels[1] = "north", "south"
        batch = RecordBatch(
            {"site": labels}, "row", element_spec=EventTemplate(site=None), name="design"
        )
        assert batch["site"].name == "design['site']"

    def test_an_object_column_is_presented_as_a_view_not_a_copy(self):
        """Reading a field is O(1), as it is for an array field.

        The object batch's public constructor copies the pointer array and walks
        every entry, both of which defend against a caller who owns the array —
        and neither of which this batch needs, having frozen the column and
        checked its entries against the same spec when it was built. Entering
        through it would make reading one field a walk over the whole batch.
        """
        batch = RecordBatch(
            {"f": _object_column([lambda x: x, lambda x: 2 * x])},
            "variant",
            element_spec=EventTemplate(f=FunctionSpec()),
        )
        first, second = batch["f"], batch["f"]
        assert first._store is batch._columns["f"]
        assert first._store is second._store
        # Shared, so still nobody's to write through.
        assert not first._store.flags.writeable

    def test_an_empty_batch_presents_an_empty_column(self):
        """A batch of no elements is a batch, so reading one of its fields is
        reading a field: the column comes back as the empty batch of its kind
        rather than being refused for holding nothing.
        """
        batch = RecordBatch(
            {"f": np.array([], dtype=object)},
            "variant",
            element_spec=EventTemplate(f=FunctionSpec()),
        )
        assert batch.batch_shape == (0,)
        column = batch["f"]
        assert isinstance(column, FunctionBatch)
        assert column.batch_shape == (0,)


class TestColumnEntryValidation:
    """A batch asserts its ``element_spec`` of every element, so a non-array
    column is walked entry by entry — the counterpart of what the object batches
    do for a batch that stores its elements."""

    def test_an_entry_the_field_spec_refuses_is_named_with_its_position(self):
        with pytest.raises(TypeError, match=r"the entry at 1 is a dict"):
            RecordBatch(
                {"o": _object_column(["fine", {"k": 1}, "fine"])},
                "row",
                element_spec=EventTemplate(o=None),
            )

    def test_a_callable_field_refuses_a_non_callable_entry(self):
        with pytest.raises(TypeError, match=r"the entry at 0 is a str"):
            RecordBatch(
                {"f": _object_column(["not callable", lambda x: x])},
                "row",
                element_spec=EventTemplate({"f": FunctionSpec()}),
            )

    def test_an_array_column_carries_no_entries_to_walk(self):
        # Its shape is the whole of what it declares, and that is already checked.
        assert nested_batch().batch_shape == (3,)


class TestColumnSpecConformance:
    """A column must satisfy the field it belongs to, not merely have the right
    shape: the batch asserts its ``element_spec`` of every element."""

    def test_a_cross_kind_dtype_is_refused(self):
        with pytest.raises(TypeError, match=r"has dtype float32, which its declared int32"):
            NumericRecordBatch(
                {"x": jnp.zeros(3, dtype=jnp.float32)},
                "draw",
                element_spec=EventTemplate(x=NumericArraySpec(shape=(), dtype=jnp.int32)),
            )

    @pytest.mark.parametrize(
        ("column_dtype", "declared"),
        [
            (jnp.float16, jnp.float32),
            (jnp.float32, jnp.float16),
            (jnp.int16, jnp.int32),
            (jnp.int32, jnp.int16),
        ],
        ids=["float-widening", "float-narrowing", "int-widening", "int-narrowing"],
    )
    def test_a_same_kind_cast_is_admitted(self, column_dtype, declared):
        # The rule ``NumericArraySpec.is_valid`` applies to one value: a widening or a
        # within-kind narrowing passes.
        #
        # The pairs stay inside 32 bits deliberately. ``jax_enable_x64`` is off by
        # default, so a 64-bit request is truncated to 32 with a warning — which
        # would collapse the narrowing case to a same-dtype one and leave the
        # direction under test unexercised. Promoting the warning to an error is
        # what keeps that from creeping back.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            column = jnp.zeros(3, dtype=column_dtype)
        assert column.dtype == column_dtype
        NumericRecordBatch(
            {"x": column},
            "draw",
            element_spec=EventTemplate(x=NumericArraySpec(shape=(), dtype=declared)),
        )

    @pytest.mark.parametrize("spec", [FunctionSpec(), None], ids=["function", "opaque"])
    def test_a_field_with_no_stacked_form_refuses_a_dense_column(self, spec):
        """A dense array under such a field would make its *entries* array
        elements rather than the values themselves, and the column could not be
        presented as the batch form its spec calls for."""
        with pytest.raises(TypeError, match="no stacked form"):
            RecordBatch({"f": jnp.zeros(3)}, "draw", element_spec=EventTemplate({"f": spec}))

    def test_an_element_of_a_validated_batch_conforms_to_its_own_spec(self):
        batch = nested_batch()
        assert batch.element_spec.is_valid(batch[0])


class TestConstructionRefusals:
    def test_a_column_reporting_no_shape_is_refused_in_any_position(self):
        # Not only in the first position: a shape is what a batch axis is read
        # from, so every column must have one.
        for columns in (
            {"o": ["a", "b"], "x": jnp.zeros((2, 2))},
            {"x": jnp.zeros((2, 2)), "o": ["a", "b"]},
        ):
            with pytest.raises(TypeError, match="reports no shape"):
                RecordBatch(columns, "row", element_spec=EventTemplate(o=None, x=(2,)))

    def test_a_column_too_short_for_its_event_shape_is_refused(self):
        with pytest.raises(ValueError, match="too short to carry"):
            NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=EventTemplate(x=(2, 2)))


class TestProvenance:
    def test_every_derived_view_inherits_the_batchs_provenance(self):
        from probpipe import Provenance

        batch = RecordBatch(
            {"outer/a": jnp.zeros(3), "site": _object_column(list("abc"))},
            "row",
            element_spec=EventTemplate(outer=EventTemplate(a=()), site=None),
            name="design",
        )
        batch.with_provenance(Provenance.create("sample", parents=[]))
        assert batch[0].provenance is batch.provenance
        assert batch[0:2].provenance is batch.provenance
        assert batch["site"].provenance is batch.provenance  # an OpaqueBatch column
        assert batch["outer"].provenance is batch.provenance  # a sub-batch view
        assert batch.select("outer/a")["outer/a"].provenance is batch.provenance


class TestPlainRecordBatch:
    """A non-numeric batch end to end — the base class, not the numeric one."""

    def test_elements_are_plain_records_and_iteration_works(self):
        batch = RecordBatch(
            {"site": _object_column(["north", "south"])},
            "row",
            element_spec=EventTemplate(site=None),
            name="design",
        )
        assert type(batch[0]) is Record
        assert [element["site"] for element in batch] == ["north", "south"]

    def test_it_has_no_flat_layout(self):
        batch = RecordBatch(
            {"site": _object_column(["north"])}, "row", element_spec=EventTemplate(site=None)
        )
        assert not hasattr(batch, "to_vector")

    def test_it_round_trips_through_pickle_and_the_pytree(self):
        batch = RecordBatch(
            {"site": _object_column(["north", "south"])},
            "row",
            element_spec=EventTemplate(site=None),
        )
        assert pickle.loads(pickle.dumps(batch)) == batch
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == batch


class TestStructuralTransforms:
    """The record transforms apply elementwise. A batch presents each field's
    values across the batch, so that is what they act on — the batch axes and the
    levels come through untouched.
    """

    def test_with_path_names_renames_within_every_element(self):
        batch = nested_batch(name="post")
        renamed = batch.with_path_names({"outer/a": "alpha"})
        assert tuple(renamed.event_template.keys()) == ("outer/alpha", "outer/b", "m")
        np.testing.assert_array_equal(
            np.asarray(renamed["outer/alpha"]), np.asarray(batch["outer/a"])
        )
        assert renamed.level_names == batch.level_names
        assert renamed.batch_shape == batch.batch_shape

    def test_with_path_names_takes_a_bare_name(self):
        assert "mass" in nested_batch().with_path_names(m="mass").event_template

    def test_the_two_name_spaces_are_independent(self):
        # Renaming a field never touches a level, or the reverse.
        batch = nested_batch()
        assert batch.with_path_names(m="mass").level_names == ("draw",)
        assert tuple(batch.with_level_names(draw="s").event_template.keys()) == tuple(
            batch.event_template.keys()
        )

    def test_without_drops_a_field_or_a_subtree(self):
        batch = nested_batch()
        assert tuple(batch.without("outer/b").event_template.keys()) == ("outer/a", "m")
        assert tuple(batch.without("outer").event_template.keys()) == ("m",)

    def test_without_everything_raises(self):
        with pytest.raises(ValueError):
            nested_batch().without("outer", "m")

    def test_replace_takes_a_fields_values_across_the_batch(self):
        batch = nested_batch()
        replaced = batch.replace({"m": jnp.ones((3, 2))})
        assert replaced.batch_shape == (3,)
        np.testing.assert_array_equal(np.asarray(replaced["m"]), np.asarray(jnp.ones((3, 2))))
        # An untouched field keeps its own values.
        np.testing.assert_array_equal(np.asarray(replaced["outer/a"]), np.asarray(batch["outer/a"]))

    def test_replace_re_infers_only_the_edited_field(self):
        batch = nested_batch()
        replaced = batch.replace({"m": jnp.ones((3, 5))})
        assert replaced.event_template["m"].shape == (5,)
        assert replaced.event_template["outer/a"] is batch.event_template["outer/a"]

    def test_replace_needs_the_batch_axes(self):
        with pytest.raises(ValueError, match="does not carry this batch's axes"):
            nested_batch().replace({"m": jnp.ones((9, 2))})

    def test_replace_edits_rather_than_adds(self):
        with pytest.raises(KeyError, match="replace edits, it does not add"):
            nested_batch().replace({"nope": jnp.zeros(3)})

    def test_merge_unions_the_fields(self):
        batch = nested_batch()
        other = NumericRecordBatch({"z": jnp.ones(3)}, "draw", element_spec=EventTemplate(z=()))
        merged = batch.merge(other)
        assert tuple(merged.event_template.keys()) == ("outer/a", "outer/b", "m", "z")
        np.testing.assert_array_equal(np.asarray(merged["z"]), np.asarray(jnp.ones(3)))

    def test_merge_pairs_elements_so_the_axes_must_agree(self):
        with pytest.raises(ValueError, match="span the same axes under the same names"):
            nested_batch(3).merge(
                NumericRecordBatch({"z": jnp.ones(4)}, "draw", element_spec=EventTemplate(z=()))
            )

    def test_merge_refuses_overlapping_fields(self):
        with pytest.raises(ValueError, match="overlapping field keys"):
            nested_batch().merge(nested_batch())

    def test_map_applies_to_each_fields_values_at_once(self):
        """One call per field, not per element: what a batch presents at a field
        is that field's values across the batch."""
        seen: list[tuple[int, ...]] = []

        def double(column):
            seen.append(tuple(column.shape))
            return column * 2

        doubled = nested_batch().map(double)
        assert seen == [(3,), (3,), (3, 2)]  # three calls, each the whole field
        np.testing.assert_array_equal(np.asarray(doubled["outer/a"]), np.asarray([0.0, 2.0, 4.0]))

    def test_map_with_keys_passes_the_leaf_path(self):
        scaled = nested_batch().map_with_keys(lambda key, column: column * (2 if key == "m" else 1))
        np.testing.assert_array_equal(np.asarray(scaled["outer/a"]), np.asarray([0.0, 1.0, 2.0]))

    def test_a_transform_re_derives_the_class_from_its_result(self):
        """The class follows the edited fields, not the object's history — which is
        also what makes a mixed ``merge`` give the same answer either way round."""
        numeric = NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=EventTemplate(x=()))
        plain = RecordBatch(
            {"s": _object_column(list("abc"))}, "draw", element_spec=EventTemplate(s=None)
        )
        assert type(numeric.replace({"x": _object_column(list("abc"))})) is RecordBatch
        assert type(plain.replace({"s": jnp.ones(3)})) is NumericRecordBatch
        assert type(numeric.merge(plain)) is type(plain.merge(numeric)) is RecordBatch

    def test_an_edited_field_is_typed_the_way_inference_would(self):
        plain = RecordBatch(
            {"f": _object_column([lambda x: x] * 2)},
            "draw",
            element_spec=EventTemplate({"f": FunctionSpec()}),
        )
        # A column of callables stays a function field rather than going opaque.
        edited = plain.replace({"f": _object_column([lambda x: 2 * x] * 2)})
        assert isinstance(edited.event_template["f"], FunctionSpec)

    def test_a_non_object_array_for_a_field_with_no_stacked_form_is_refused(self):
        """A unicode array's entries are numpy scalars, not the values the field
        holds, and it is not numeric either — so it is named rather than guessed at."""
        plain = RecordBatch(
            {"s": _object_column(list("ab"))}, "draw", element_spec=EventTemplate(s=None)
        )
        with pytest.raises(TypeError, match="no stacked form"):
            plain.replace({"s": np.array(["x", "y"])})

    def test_replace_accepts_what_field_access_hands_back(self):
        plain = RecordBatch(
            {"f": _object_column([lambda x: x] * 2), "x": jnp.zeros(2)},
            "draw",
            element_spec=EventTemplate({"f": FunctionSpec(), "x": ()}),
        )
        # ``batch["f"]`` is a FunctionBatch; putting one back must work.
        assert isinstance(plain.replace({"f": plain["f"]}), RecordBatch)

    def test_replace_refuses_both_update_forms_at_once(self):
        with pytest.raises(ValueError, match="not both"):
            nested_batch().replace({"m": jnp.ones((3, 2))}, m=jnp.zeros((3, 2)))

    def test_a_transform_keeps_a_user_given_name_and_re_derives_an_auto_one(self):
        assert nested_batch(name="post").without("m").name == "post"
        assert nested_batch().without("m").name_is_auto


# ---------------------------------------------------------------------------
# A batch is a collection, not a named tree
# ---------------------------------------------------------------------------


class TestCollectionNotTree:
    @pytest.mark.parametrize(
        "attribute",
        ["keys", "values", "items", "children", "at_path", "is_field", "fields"],
    )
    def test_the_field_keyed_mapping_protocol_is_absent(self, attribute):
        """Navigation goes; the structure-preserving transforms stay, since those
        act on the elements rather than treating the batch as a tree."""
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
        # The columns must ride along: renaming a level touches no data, and the
        # copy relies on the tracked base carrying the storage slot.
        np.testing.assert_array_equal(np.asarray(renamed["outer/a"]), np.asarray([0.0, 1.0, 2.0]))
        assert renamed[0]["outer/a"] == 0.0

    def test_a_renamed_batch_derives_view_names_under_the_new_name(self):
        renamed = nested_batch(name="post").with_level_names(draw="sample")
        assert renamed[0].name == "post[sample=0]"

    def test_a_descending_slice_is_presented_in_the_order_given(self):
        """The batch base requires storage to honor a reversed selection rather
        than re-sort it, since a view's derived name is stated in that order."""
        batch = nested_batch(4, name="post")
        reversed_view = batch[::-1]
        np.testing.assert_array_equal(
            np.asarray(reversed_view["outer/a"]), np.asarray([3.0, 2.0, 1.0, 0.0])
        )
        assert reversed_view[0]["outer/a"] == 3.0
        assert reversed_view.name == "post[draw=3::-1]"

    def test_a_stepped_slice_selects_every_other_element(self):
        batch = nested_batch(4)
        np.testing.assert_array_equal(np.asarray(batch[::2]["outer/a"]), np.asarray([0.0, 2.0]))

    def test_a_negative_index_names_the_position_it_resolves_to(self):
        batch = nested_batch(4, name="post")
        assert batch[-1].name == "post[draw=3]"
        assert batch[-1]["outer/a"] == 3.0

    def test_an_object_column_view_shares_its_parents_store(self):
        """Sharing is the object-column story: a JAX slice is a fresh array."""
        batch = RecordBatch(
            {"site": _object_column(["a", "b", "c", "d"])},
            "row",
            element_spec=EventTemplate(site=None),
        )
        assert np.shares_memory(batch[1:3]._columns["site"], batch._columns["site"])

    def test_an_object_column_cannot_be_written_through(self):
        column = _object_column(["a", "b"])
        batch = RecordBatch({"site": column}, "row", element_spec=EventTemplate(site=None))
        with pytest.raises(ValueError, match="read-only"):
            batch._columns["site"][0] = "MUTATED"
        # Nor can the caller's own handle reach in after construction.
        column[0] = "MUTATED"
        assert batch["site"][0] == "a"

    def test_a_slice_keeps_the_level_and_its_values(self):
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

    def test_select_all_covers_the_top_level_so_it_can_splat(self):
        """Keyed by top-level name, as ``Record.select_all`` is: a ``/``-path could
        not bind to a parameter, which is what ``select`` exists for."""
        selected = nested_batch().select_all()
        assert set(selected) == {"outer", "m"}
        # The interior name gives the sub-batch beneath it, so the parts cover
        # the batch between them.
        assert tuple(selected["outer"].event_template.keys()) == ("a", "b")
        assert tuple(selected["m"].event_template.keys()) == ("m",)

    def test_select_reaches_a_leaf_or_a_subtree_by_path(self):
        batch = nested_batch()
        assert tuple(batch.select("outer/a")["outer/a"].event_template.keys()) == ("outer/a",)
        assert tuple(batch.select("outer")["outer"].event_template.keys()) == ("a", "b")

    def test_keywords_remap(self):
        selected = nested_batch().select(value="m")
        assert set(selected) == {"value"}
        assert tuple(selected["value"].event_template.keys()) == ("m",)

    def test_an_unknown_path_raises(self):
        with pytest.raises(KeyError, match="neither a field nor an interior node"):
            nested_batch().select("nope")


class TestSingleFieldCoercion:
    """With one field, a batch forwards the array-conversion entry points to that
    field's values — narrower than the single-record shim, since a batch of values
    is not one scalar however few fields it has."""

    @staticmethod
    def one_field():
        return NumericRecordBatch(
            {"x": jnp.arange(6.0).reshape(3, 2)}, "draw", element_spec=EventTemplate(x=(2,))
        )

    def test_array_conversions_forward_to_the_sole_field(self):
        batch = self.one_field()
        np.testing.assert_array_equal(np.asarray(batch), np.asarray(batch["x"]))
        np.testing.assert_array_equal(np.asarray(jnp.asarray(batch)), np.asarray(batch["x"]))

    def test_introspection_forwards_to_the_sole_field(self):
        batch = self.one_field()
        assert batch.shape == (3, 2)
        assert batch.dtype == jnp.float32
        assert batch.ndim == 2

    @pytest.mark.parametrize("read", [lambda b: b.shape, lambda b: b.dtype, np.asarray])
    def test_more_than_one_field_is_refused(self, read):
        with pytest.raises(TypeError, match="is not array-like"):
            read(nested_batch())

    def test_a_batch_is_never_scalar_like(self):
        # The single-record shim offers these; a batch of values does not.
        batch = self.one_field()
        for entry in ("__float__", "__int__", "__bool__"):
            assert not hasattr(batch, entry)


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
            batch.name, batch.event_template, batch.to_vector(), level_names="draw"
        )
        assert rebuilt == batch

    def test_from_vector_refuses_an_unbatched_vector(self):
        with pytest.raises(TypeError, match=r"NumericRecord\.from_vector"):
            NumericRecordBatch.from_vector(
                "v", EventTemplate(x=(2,)), jnp.zeros(2), level_names="draw"
            )

    def test_from_vector_checks_the_trailing_axis(self):
        with pytest.raises(ValueError, match="the trailing axis is 3, expected 2"):
            NumericRecordBatch.from_vector(
                "v", EventTemplate(x=(2,)), jnp.zeros((5, 3)), level_names="draw"
            )

    def test_a_multi_level_batch_keeps_its_levels_as_leading_axes(self):
        batch = NumericRecordBatch(
            {"x": jnp.arange(24.0).reshape(2, 3, 4)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=(4,)),
        )
        vec = batch.to_vector()
        assert vec.shape == (2, 3, 4)
        # The flat dimension is the last axis; the levels stay outermost-first,
        # so a row of the result is that element's own flat vector.
        np.testing.assert_array_equal(np.asarray(vec[1, 2]), np.asarray(jnp.arange(24.0)[20:24]))
        np.testing.assert_array_equal(np.asarray(vec[1, 2]), np.asarray(batch[1, 2].to_vector()))

    def test_a_multi_level_batch_round_trips_when_both_levels_are_named(self):
        batch = NumericRecordBatch(
            {"x": jnp.arange(24.0).reshape(2, 3, 4)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=(4,)),
            name="post",
        )
        rebuilt = NumericRecordBatch.from_vector(
            "post", batch.event_template, batch.to_vector(), level_names=("chain", "draw")
        )
        assert rebuilt == batch

    def test_a_mixed_dtype_round_trip_restores_each_declared_dtype(self):
        """Concatenating promotes the fields to one dtype, so reconstruction must
        cast back — otherwise the result contradicts the template it was rebuilt
        from. Equality would not catch it: values compare, dtypes do not."""
        template = EventTemplate(
            {
                "i": NumericArraySpec(shape=(), dtype=jnp.int32),
                "f": NumericArraySpec(shape=(), dtype=jnp.float32),
            }
        )
        batch = NumericRecordBatch(
            {"i": jnp.arange(3, dtype=jnp.int32), "f": jnp.ones(3, dtype=jnp.float32)},
            "draw",
            element_spec=template,
        )
        assert batch.to_vector().dtype == jnp.float32  # the promotion
        rebuilt = NumericRecordBatch.from_vector(
            "b", batch.event_template, batch.to_vector(), level_names="draw"
        )
        assert rebuilt["i"].dtype == jnp.int32
        assert rebuilt["f"].dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(rebuilt["i"]), np.asarray([0, 1, 2]))

    def test_from_vector_takes_every_batch_axis_as_one_named_level(self):
        """One name is one level however many axes the flat vector carried: the
        draw it came from is one multiplicity, not one per axis."""
        template = EventTemplate(x=(2,))

        batch = NumericRecordBatch.from_vector(
            "v", template, jnp.zeros((4, 5, 2)), level_names="draw"
        )

        assert batch.level_names == ("draw",)
        assert batch.batch_shape == (4, 5)
        assert batch.axis_groups == ((4, 5),)

    def test_from_vector_gives_several_names_one_axis_each(self):
        template = EventTemplate(x=(2,))

        batch = NumericRecordBatch.from_vector(
            "v", template, jnp.zeros((4, 5, 2)), level_names=("chain", "draw")
        )

        assert batch.axis_groups == ((4,), (5,))

    def test_from_vector_refuses_more_names_than_axes(self):
        template = EventTemplate(x=(2,))
        with pytest.raises(ValueError, match="need 2 level names"):
            NumericRecordBatch.from_vector(
                "v", template, jnp.zeros((4, 5, 2)), level_names=("chain", "draw", "extra")
            )

    def test_to_vector_on_an_empty_selection(self):
        """The class advertises ``batch[0:0]``, so vectorizing one must work: the
        field's own flat width is used rather than an inferred axis."""
        batch = nested_batch()
        assert batch[0:0].to_vector().shape == (0, 4)


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

    def test_stack_names_a_record_whose_fields_are_missing(self):
        records = [NumericRecord("r", x=jnp.asarray(1.0)) for _ in range(2)]
        with pytest.raises(ValueError, match=r"the record at 0 .*missing \['z'\]"):
            NumericRecordBatch.stack(
                records, level_name="draw", element_spec=EventTemplate(x=(), z=())
            )

    def test_stack_names_a_record_with_extra_fields(self):
        """Extra fields must not be dropped silently: the batch's spec would
        become a false statement about that record."""
        spec = EventTemplate(x=())
        records = [
            NumericRecord("r", x=jnp.asarray(1.0)),
            NumericRecord("r", x=jnp.asarray(1.0), extra=jnp.asarray(2.0)),
        ]
        with pytest.raises(ValueError, match=r"the record at 1 .*unexpected \['extra'\]"):
            NumericRecordBatch.stack(records, level_name="draw", element_spec=spec)

    def test_stack_lets_a_ragged_numeric_field_fail_as_a_stacking_error(self):
        records = [
            NumericRecord("r", {"x": jnp.zeros(2)}, event_template=EventTemplate(x=(2,))),
            NumericRecord("r", {"x": jnp.zeros(3)}, event_template=EventTemplate(x=(3,))),
        ]
        # Declared a NumericArraySpec, so it stacks natively and the shapes must agree —
        # it is not quietly demoted to an object column.
        with pytest.raises((TypeError, ValueError)):
            NumericRecordBatch.stack(records, level_name="draw", element_spec=EventTemplate(x=(2,)))

    def test_stack_keeps_an_opaque_field_opaque_when_its_values_are_numeric(self):
        """The field's spec decides the column form, not the values.

        ``OpaqueSpec`` admits a number, so stacking must not read the values and
        conclude the field is an array — the column would come back as the wrong
        batch form and an element would come back as an array, not the int put in.
        """
        spec = EventTemplate(tag=None, x=(2,))
        records = [
            Record(f"r{i}", {"tag": i, "x": jnp.zeros(2)}, event_template=spec) for i in range(3)
        ]
        batch = RecordBatch.stack(records, level_name="draw")
        assert isinstance(batch["tag"], OpaqueBatch)
        assert batch["tag"][1] == 1
        assert batch[0]["tag"] == 0
        assert not isinstance(batch[0]["tag"], jnp.ndarray)

    def test_stack_refuses_an_empty_list(self):
        with pytest.raises(ValueError, match="at least one record"):
            NumericRecordBatch.stack([], level_name="draw")

    def test_stack_takes_the_spec_from_the_first_record(self):
        spec = RecordSpec(EventTemplate(x=NumericArraySpec(shape=(), dtype=jnp.float32)))
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

    def test_stack_converts_a_leaf_by_its_registered_backend(self, clean_registry):
        """Stacking goes through ``_to_jax_array``, the one conversion every
        compute boundary routes through, so a leaf type is converted by the rule
        its ``ArrayBackend`` states.

        A bare ``jnp.asarray`` would reach only the leaves that speak the numpy
        protocol, and a container registered *because* it does not — a lazy or
        disk-backed one — would fail to stack at all.
        """

        class Boxed:
            def __init__(self, value):
                self.value = value

        register_array_backend(
            Boxed,
            ArrayBackend(
                event_shape=lambda b: (),
                numpy_dtype=lambda b: np.dtype("float32"),
                to_jax=lambda b: jnp.asarray(b.value, dtype=jnp.float32),
                to_numpy=lambda b: np.asarray(b.value, dtype="float32"),
            ),
        )
        # Unconvertible by the numpy protocol on its own.
        with pytest.raises(TypeError):
            jnp.asarray(Boxed(1.0))

        template = EventTemplate(x=())
        records = [Record("r", {"x": Boxed(v)}, event_template=template) for v in (1.0, 2.0)]
        batch = RecordBatch.stack(records, level_name="row")
        assert batch.batch_shape == (2,)
        np.testing.assert_allclose(batch["x"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Equality, repr, copying
# ---------------------------------------------------------------------------


class TestEqualityAndCopying:
    def test_equal_columns_and_spec_compare_equal(self):
        assert nested_batch() == nested_batch()

    def test_same_spec_different_data_compares_unequal(self):
        """The case that makes every other ``==`` assertion in this file mean something.

        Comparing batches of *different length* proves nothing about column
        comparison: their specs differ, so ``__eq__`` answers before reading a
        column. These two share a spec exactly, so only the values can separate
        them.
        """
        spec = EventTemplate(x=())
        zeros = NumericRecordBatch({"x": jnp.zeros(3)}, "draw", element_spec=spec)
        ones = NumericRecordBatch({"x": jnp.ones(3)}, "draw", element_spec=spec)
        assert zeros.spec == ones.spec
        assert zeros != ones

    def test_a_single_differing_entry_compares_unequal(self):
        spec = EventTemplate(x=())
        left = NumericRecordBatch({"x": jnp.asarray([1.0, 2.0, 3.0])}, "draw", element_spec=spec)
        right = NumericRecordBatch({"x": jnp.asarray([1.0, 2.0, 4.0])}, "draw", element_spec=spec)
        assert left != right

    def test_an_object_column_compares_by_value(self):
        """The object-column path is the only one a non-numeric batch takes, and
        ``jnp.array_equal`` cannot walk it."""
        spec = EventTemplate(site=None)
        labels = ["north", "south"]
        left = RecordBatch({"site": _object_column(labels)}, "row", element_spec=spec)
        same = RecordBatch({"site": _object_column(labels)}, "row", element_spec=spec)
        other = RecordBatch({"site": _object_column(["north", "east"])}, "row", element_spec=spec)
        assert left == same
        assert left != other

    def test_an_object_column_of_arrays_compares_by_value(self):
        """Entries that are themselves arrays have no single truth value, so a
        vectorized comparison cannot answer; they are compared entry by entry."""
        spec = EventTemplate(cov=None)
        left = RecordBatch(
            {"cov": _object_column([np.zeros(2), np.zeros(3)])}, "row", element_spec=spec
        )
        same = RecordBatch(
            {"cov": _object_column([np.zeros(2), np.zeros(3)])}, "row", element_spec=spec
        )
        other = RecordBatch(
            {"cov": _object_column([np.zeros(2), np.ones(3)])}, "row", element_spec=spec
        )
        assert left == same
        assert left != other
        assert pickle.loads(pickle.dumps(left)) == left

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

    def test_vmap_hands_the_body_one_element_not_a_stale_batch(self):
        """``vmap`` strips the mapped axis, so what the body receives is an
        element. Rebuilding against the stored spec would leave an object whose
        ``batch_shape`` its own columns contradict, and every method reading that
        shape would be wrong — which reading a column alone does not reveal.
        """
        batch = nested_batch()
        seen: list[str] = []

        def body(row):
            seen.append(type(row).__name__)
            return row.to_vector()

        stacked = jax.vmap(body)(batch)
        assert seen == ["NumericRecord"]
        assert stacked.shape == (3, 4)
        np.testing.assert_array_equal(np.asarray(stacked), np.asarray(batch.to_vector()))

    @pytest.mark.parametrize("in_axes", [0, 1], ids=["leading", "named"])
    def test_raw_vmap_over_one_level_of_a_multi_level_batch_is_refused(self, in_axes):
        """Mapping *some* of a batch's axes leaves the survivors unnamed.

        Which axis ``vmap`` consumed is its own knowledge; the shape that arrives
        does not record it, and the leading axis is no more identifiable than any
        other — two levels of equal size explain the same removal either way. So
        the partial reduction is refused rather than guessed at, for the default
        ``in_axes`` as much as for a named one.
        """
        batch = NumericRecordBatch(
            {"x": jnp.zeros((2, 3, 4))}, ("chain", "draw"), element_spec=EventTemplate(x=(4,))
        )
        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.vmap(lambda inner: inner["x"].sum(), in_axes=in_axes)(batch)

    def test_raw_vmap_through_a_level_spanning_several_axes_is_refused(self):
        """A level holding several axes is no more recoverable: consuming one of
        them leaves the rest of that level standing, which is again a partial
        reduction the arriving shape cannot attribute."""
        batch = NumericRecordBatch(
            {"x": jnp.zeros((2, 3, 5))},
            ("grid", "draw"),
            element_spec=EventTemplate(x=()),
            axis_groups=((2, 3), (5,)),
        )
        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.vmap(lambda inner: inner["x"].sum(), in_axes=1)(batch)

    def test_an_untransformed_round_trip_is_still_a_batch(self):
        batch = nested_batch()
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == batch

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


class TestPyTreeRebuildContract:
    """Which raw pytree transformations a batch can be rebuilt under, and why the
    rest are refused.

    A ``Record`` threads its declaration through the pytree aux; a batch cannot,
    because ``vmap`` removes an axis the aux still names. What arrives is the only
    evidence, and **a shape is not a provenance** — so the rebuild supports two
    transformations and refuses the rest, rather than inferring which axis went.
    A ``BatchSpec`` naming the wrong level is not an approximation: it goes on to
    drive level-name alignment.
    """

    @staticmethod
    def _batch(shape, levels, **kwargs):
        return NumericRecordBatch(
            {"x": jnp.zeros(shape)}, levels, element_spec=EventTemplate(x=()), **kwargs
        )

    REFUSAL = "keeps every batch axis or removes all of them"

    # -- supported ----------------------------------------------------------

    def test_a_shape_preserving_map_reuses_the_spec(self):
        """The ordinary round trip: every batch axis still there, so the stored
        spec still describes the columns and is reused rather than rebuilt."""
        batch = self._batch((3, 2), ("chain", "draw"))
        doubled = jax.tree.map(lambda column: column * 2, batch)
        assert doubled.spec is batch.spec
        assert doubled.batch_shape == (3, 2)
        assert doubled.level_names == ("chain", "draw")

    def test_removing_every_batch_axis_gives_one_record(self):
        """No batch axis left means the value *is* an element, so a ``Record``
        comes back rather than a batch of nothing."""
        seen: list[str] = []
        batch = self._batch((3,), "draw")
        jax.vmap(lambda row: (seen.append(type(row).__name__), row["x"])[1])(batch)
        assert seen == ["NumericRecord"]

    # -- refused ------------------------------------------------------------

    def test_a_resized_axis_is_refused(self):
        """A resize keeps the axis count and changes what the level holds.

        Carrying the level names onto the new sizes would be right for a per-level
        slice and wrong for anything else that lands on the same shape — a
        transpose composed with a slice, say — so it is refused. Index the batch
        instead, which is told which positions it keeps.
        """
        batch = self._batch((3, 2), ("chain", "draw"))
        with pytest.raises(ValueError, match=self.REFUSAL):
            jax.tree.map(lambda column: column[:1], batch)
        # The supported route to the same value, exact because it carries the
        # selection rather than inferring it.
        assert batch[0:1].batch_shape == (1, 2)
        assert batch[0:1].level_names == ("chain", "draw")

    def test_a_mixed_drop_and_resize_is_refused(self):
        """The case a shape cannot distinguish at all: dropping one axis while
        resizing another lands on a shape a *different* single drop explains, so
        the surviving levels would be misnamed with nothing to signal it."""
        batch = self._batch((2, 3, 4), ("a", "b", "c"))
        with pytest.raises(ValueError, match=self.REFUSAL):
            jax.tree.map(lambda column: column[0, :2, :], batch)

    def test_an_unequal_permutation_is_refused(self):
        batch = self._batch((2, 3), ("chain", "draw"))
        with pytest.raises(ValueError, match=self.REFUSAL):
            jax.tree.map(lambda column: column.T, batch)

    def test_an_added_axis_is_refused(self):
        """``vmap`` stacking a batch-returning body inserts an axis no level
        names, and unflattening has no name to give one."""
        inner = self._batch((2,), "inner")
        with pytest.raises(ValueError, match="batch axes where its levels account for"):
            jax.vmap(lambda _: inner)(self._batch((3,), "outer"))

    def test_a_column_reporting_no_shape_is_refused(self):
        """A stored column is an array, so one reporting no shape came from the
        transform. Rebuilding at the old multiplicity would leave every positional
        read failing on a value that cannot hold it."""
        batch = self._batch((3,), "draw")
        with pytest.raises(ValueError, match="column reporting no shape"):
            jax.tree.map(lambda column: float(column.sum()), batch)

    def test_object_data_under_a_numeric_field_is_refused(self):
        """An ``NumericArraySpec`` requires numeric data whether or not it pins a dtype,
        so the kind is re-checked and not only the pinned dtype — otherwise a
        numeric batch comes back holding objects."""
        batch = self._batch((3,), "draw")
        with pytest.raises(TypeError, match="is declared an array, so its column is a numeric"):
            jax.tree.map(lambda _: np.array(["a", "b", "c"], dtype=object), batch)

    def test_a_cross_kind_dtype_is_refused(self):
        """The pinned-dtype rule is the constructor's: a widening or a within-kind
        narrowing passes, a change of kind does not. Float data under an
        integer-pinned field is the direction that fails."""
        batch = NumericRecordBatch(
            {"x": jnp.zeros(3, dtype=jnp.int32)},
            "draw",
            element_spec=EventTemplate(x=NumericArraySpec(shape=(), dtype=jnp.int32)),
        )
        with pytest.raises(TypeError, match="does not admit"):
            jax.tree.map(lambda column: column.astype(jnp.float32), batch)

    def test_retyping_a_callable_column_is_refused(self):
        """A shape-preserving transform can swap what a column holds without
        touching its rank, so a column of callables can come back as integers
        while the batch still declares ``FunctionSpec``. The element's kind is
        the element type's, not the transform's."""
        column = _object_column([lambda x: x, lambda x: 2 * x])
        batch = RecordBatch({"f": column}, "row", element_spec=EventTemplate(f=FunctionSpec()))
        with pytest.raises(TypeError, match="does not admit"):
            jax.tree.map(lambda _: np.array([1, 2], dtype=object), batch)

    def test_retyping_an_opaque_column_is_refused(self):
        """The same for a field whose spec narrows what an entry may be."""
        from probpipe.core.event_template import OpaqueSpec

        column = _object_column(["north", "south"])
        batch = RecordBatch(
            {"site": column}, "row", element_spec=EventTemplate(site=OpaqueSpec(meta="units"))
        )
        replacement = np.empty(2, dtype=object)
        replacement[0], replacement[1] = {"a": 1}, {"b": 2}
        # A mapping is the one thing an opaque field refuses, since it would slip
        # past the per-entry check the object batches make.
        with pytest.raises(TypeError, match=r"does not admit|transform left"):
            jax.tree.map(lambda _: replacement, batch)

    def test_an_object_column_replaced_by_an_array_is_refused(self):
        """A non-array field's column holds one entry per element; a dense array
        would make its *entries* array elements rather than the values."""
        column = _object_column([lambda x: x, lambda x: 2 * x])
        batch = RecordBatch({"f": column}, "row", element_spec=EventTemplate(f=FunctionSpec()))
        with pytest.raises(TypeError, match="object array"):
            jax.tree.map(lambda _: jnp.zeros(2), batch)

    def test_a_resized_event_axis_is_refused(self):
        """The batch axes are the transform's; the element's own are the element
        type's."""
        batch = NumericRecordBatch(
            {"x": jnp.zeros((3, 4))}, "draw", element_spec=EventTemplate(x=(4,))
        )
        with pytest.raises(ValueError, match="never the element's own"):
            jax.tree.map(lambda column: column[:, :2], batch)

    def test_columns_disagreeing_on_the_batch_axes_are_refused(self):
        batch = NumericRecordBatch(
            {"x": jnp.zeros(4), "y": jnp.zeros(4)}, "draw", element_spec=EventTemplate(x=(), y=())
        )
        leaves, treedef = jax.tree_util.tree_flatten(batch)
        with pytest.raises(ValueError, match="disagreeing batch axes"):
            jax.tree_util.tree_unflatten(treedef, [leaves[0][:2], leaves[1]])

    # -- unsupported, and undetectable --------------------------------------

    def test_an_equal_sized_permutation_is_undetectable(self):
        """Documents a limit rather than a behavior.

        A transpose of a square batch arrives at ``tree_unflatten`` with exactly
        the shapes a no-op round trip arrives with, so no rule here can tell them
        apart. Preserving every batch axis is a **precondition** on a raw pytree
        transform, not something this can check: the result below carries the
        original level names over transposed data, and that is why the precondition
        is stated rather than enforced.
        """
        batch = self._batch((2, 2), ("chain", "draw"))
        transposed = jax.tree.map(lambda column: column.T, batch)
        assert transposed.level_names == ("chain", "draw")
        assert transposed.batch_shape == (2, 2)


class TestRankZeroReconstruction:
    """Removing every batch axis yields one element, so what the columns hold has
    to come out of its storage."""

    @pytest.mark.parametrize(
        ("spec", "value"),
        [
            pytest.param(FunctionSpec(), (lambda x: x), id="function"),
            pytest.param(OpaqueSpec(meta="units"), "north", id="opaque"),
        ],
    )
    def test_a_rank_zero_object_column_yields_the_value_not_its_array(self, spec, value):
        """A non-array column is an object array even when it holds one entry, so
        handing it straight to the record would declare the field's kind over a
        zero-dimensional ndarray — a value its own spec refuses."""
        column = _object_column([value])
        batch = RecordBatch({"f": column}, "row", element_spec=EventTemplate(f=spec))
        element = jax.tree.map(lambda c: c.reshape(()), batch)
        assert isinstance(element, Record)
        assert spec.is_valid(element["f"])
