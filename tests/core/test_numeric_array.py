"""Tests for NumericArray and NumericArrayBatch — the numeric-array kind."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericArray, NumericArrayBatch, NumericArraySpec
from probpipe.core.provenance import Provenance


def _batch(values=None, level_names="draw", **kwargs) -> NumericArrayBatch:
    if values is None:
        values = jnp.arange(12.0).reshape(4, 3)
    kwargs.setdefault("element_spec", NumericArraySpec(shape=(3,), dtype=jnp.float32))
    kwargs.setdefault("name", "draws")
    return NumericArrayBatch(values, level_names, **kwargs)


class TestNumericArrayHoldsOneValue:
    def test_shape_is_the_event_shape(self):
        """A NumericArray carries no batch axes, so its shape is the event's."""
        value = NumericArray(jnp.zeros((2, 5)))

        assert value.shape == (2, 5)
        assert value.ndim == 2
        assert len(value) == 2

    def test_the_spec_is_derived_when_omitted(self):
        value = NumericArray(jnp.arange(3.0))

        assert value.spec == NumericArraySpec(shape=(3,), dtype=jnp.float32)

    def test_a_supplied_spec_is_checked(self):
        with pytest.raises(ValueError, match="does not satisfy its declaration"):
            NumericArray(jnp.arange(3.0), spec=NumericArraySpec(shape=(2,), dtype=jnp.float32))

    def test_a_value_that_is_not_an_array_is_refused(self):
        with pytest.raises(TypeError, match="holds one numeric array"):
            NumericArray(object())

    def test_a_spec_of_another_kind_is_refused(self):
        with pytest.raises(TypeError, match="must be a NumericArraySpec"):
            NumericArray(jnp.arange(3.0), spec="not a spec")


class TestNumericArrayStoresNativeForm:
    """Construction validates without converting, as `NumericRecord` does.

    A lazy or disk-backed value is not materialised merely to be named, and a
    container's own metadata is not discarded.
    """

    def test_an_array_is_stored_verbatim(self):
        raw = np.arange(3.0)

        assert NumericArray(raw).value is raw

    def test_a_container_keeps_its_own_metadata(self):
        xr = pytest.importorskip("xarray")
        data = xr.DataArray(np.arange(3.0), dims=["t"], coords={"t": [10, 20, 30]})

        stored = NumericArray(data).value

        assert isinstance(stored, xr.DataArray)
        assert list(stored.coords) == ["t"]

    def test_the_spec_is_read_from_metadata(self):
        assert NumericArray(np.arange(3.0)).shape == (3,)

    def test_a_bare_scalar_is_normalised(self):
        """It carries no metadata to read, so it is normalised."""
        assert isinstance(NumericArray(2.5).value, jax.Array)

    def test_conversion_happens_once_and_is_memoised(self):
        value = NumericArray(np.arange(3.0))

        assert isinstance(value.as_jax(), jax.Array)
        assert value.as_jax() is value.as_jax()

    def test_the_pytree_boundary_presents_a_bare_array(self):
        """A compute boundary, where native form converts."""
        (leaf,) = jax.tree_util.tree_leaves(NumericArray(np.arange(3.0)))

        assert isinstance(leaf, jax.Array)

    def test_dtype_reports_the_value_not_the_declaration(self):
        """An array's dtype describes its data; ``.spec`` carries the declaration.

        ``is_valid`` admits a same-kind cast, so the two can differ.
        """
        value = NumericArray(
            jnp.zeros(3, dtype=jnp.float32),
            spec=NumericArraySpec(shape=(3,), dtype=jnp.float64),
        )

        assert value.dtype == jnp.float32
        assert value.spec.dtype == jnp.float64

    def test_a_non_numeric_value_is_refused(self):
        with pytest.raises(TypeError, match="is not a numeric leaf"):
            NumericArray("not numeric")


class TestNumericArrayCarriesIdentity:
    def test_a_name_is_kept_and_marked_user_given(self):
        value = NumericArray(jnp.arange(3.0), name="draw")

        assert (value.name, value.name_is_auto) == ("draw", False)

    def test_an_omitted_name_is_auto_derived(self):
        assert NumericArray(jnp.arange(3.0)).name_is_auto is True

    def test_provenance_is_write_once(self):
        value = NumericArray(jnp.arange(3.0)).with_provenance(Provenance.create("test", parents=[]))

        assert value.provenance.operation == "test"
        with pytest.raises(RuntimeError, match="already set"):
            value.with_provenance(Provenance.create("again", parents=[]))

    def test_it_is_immutable(self):
        with pytest.raises(AttributeError, match="immutable"):
            NumericArray(jnp.arange(3.0))._value = jnp.zeros(3)

    def test_the_array_is_reachable_untracked(self):
        raw = jnp.arange(3.0)

        assert NumericArray(raw).value is raw

    def test_it_is_unhashable(self):
        """`__eq__` is elementwise, so a hash would promise more than it keeps."""
        with pytest.raises(TypeError):
            hash(NumericArray(jnp.arange(3.0)))


class TestNumericArrayComputesAsAnArray:
    """The full array surface, because with no fields `arr + 1` has one meaning."""

    @pytest.mark.parametrize(
        "compute",
        [
            lambda v: v + 1,
            lambda v: 1 + v,
            lambda v: v * v,
            lambda v: v - 1.0,
            lambda v: 2.0 / (v + 1),
            lambda v: -v,
            lambda v: abs(v),
            lambda v: v**2,
        ],
    )
    def test_arithmetic_yields_a_bare_array(self, compute):
        """Identity is attached by operations; arithmetic is not one."""
        result = compute(NumericArray(jnp.arange(3.0)))

        assert not isinstance(result, NumericArray)
        assert isinstance(result, jax.Array)

    def test_arithmetic_returns_the_stored_types_own_result(self):
        """The operators forward to the value, so numpy stays numpy."""
        result = NumericArray(np.arange(3.0)) + 1

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, jax.Array)

    def test_it_computes_the_same_values_as_the_array_it_holds(self):
        raw = jnp.arange(3.0)

        np.testing.assert_array_equal(
            np.asarray(NumericArray(raw) * 2 + 1), np.asarray(raw * 2 + 1)
        )

    def test_two_numeric_arrays_combine(self):
        pair = NumericArray(jnp.arange(3.0)) + NumericArray(jnp.ones(3))

        assert not isinstance(pair, NumericArray)
        np.testing.assert_array_equal(np.asarray(pair), np.asarray(jnp.arange(1.0, 4.0)))

    def test_the_reflected_operators_agree_with_the_forward_ones(self):
        """`1.0 - arr` is `arr`'s subtraction seen from the other side.

        Operand order is not a semantic distinction, so an array-like that
        computes one way and refuses the other would be a trap rather than a
        simplification.
        """
        value = NumericArray(jnp.arange(3.0))

        np.testing.assert_array_equal(np.asarray(1.0 - value), np.asarray(1.0 - jnp.arange(3.0)))
        np.testing.assert_array_equal(np.asarray(2.0 * value), np.asarray(value * 2.0))

    def test_an_in_place_operator_rebinds_to_a_bare_array(self):
        """An in-place operator on an immutable term is the out-of-place one.

        The name is rebound to the *result*, which is a bare array like any other
        arithmetic result — the term is not mutated, and does not survive.
        """
        value = NumericArray(jnp.arange(3.0), name="kept")
        original = value

        value += 1.0

        assert not isinstance(value, NumericArray)
        assert isinstance(original, NumericArray)
        assert original.name == "kept"
        np.testing.assert_array_equal(np.asarray(value), np.arange(1.0, 4.0))

    def test_comparison_is_elementwise(self):
        np.testing.assert_array_equal(
            np.asarray(NumericArray(jnp.arange(3.0)) == 1.0), np.array([False, True, False])
        )

    def test_it_converts_through_both_hooks(self):
        value = NumericArray(jnp.arange(3.0))

        np.testing.assert_array_equal(np.asarray(value), np.arange(3.0))
        assert isinstance(jnp.asarray(value), jax.Array)

    def test_it_traces_under_jit(self):
        """Registration is what carries it into a trace."""

        @jax.jit
        def double(x):
            return x * 2

        np.testing.assert_allclose(
            np.asarray(double(NumericArray(jnp.arange(3.0)))), np.arange(3.0) * 2
        )

    def test_a_scalar_converts_to_a_number(self):
        assert float(NumericArray(jnp.asarray(2.5))) == 2.5
        assert int(NumericArray(jnp.asarray(2))) == 2

    def test_a_scalar_is_truthy_by_its_value(self):
        assert bool(NumericArray(jnp.asarray(1.0))) is True
        assert bool(NumericArray(jnp.asarray(0.0))) is False

    def test_an_integer_scalar_indexes(self):
        """``__index__`` is what lets one stand in for a position."""
        assert [10, 20, 30][NumericArray(jnp.asarray(1))] == 20

    def test_indexing_and_iteration_reach_the_array(self):
        value = NumericArray(jnp.arange(3.0))

        assert float(value[1]) == 1.0
        assert [float(x) for x in value] == [0.0, 1.0, 2.0]


class TestNumericArrayBatchHoldsTheMultiplicity:
    def test_the_element_spec_splits_batch_axes_from_event_axes(self):
        batch = _batch()

        assert batch.batch_shape == (4,)
        assert batch.batch_size == 4
        assert tuple(batch.element_spec.shape) == (3,)

    def test_scalar_elements_leave_every_axis_to_the_batch(self):
        batch = NumericArrayBatch(
            jnp.arange(4.0),
            "draw",
            element_spec=NumericArraySpec(shape=(), dtype=jnp.float32),
            name="draws",
        )

        assert batch.batch_shape == (4,)

    def test_levels_tile_the_batch_shape(self):
        batch = _batch(
            jnp.arange(24.0).reshape(2, 4, 3),
            ("chain", "draw"),
        )

        assert batch.level_names == ("chain", "draw")
        assert batch.axis_groups == ((2,), (4,))
        assert batch.batch_shape == (2, 4)

    def test_the_stored_array_and_its_dtype_are_reachable(self):
        batch = _batch()

        assert batch.values.shape == (4, 3)
        assert batch.dtype == jnp.float32

    def test_the_repr_states_the_split(self):
        assert repr(_batch()) == (
            "NumericArrayBatch(batch_shape=(4,), levels=('draw',), event_shape=(3,))"
        )

    def test_one_level_may_span_several_axes(self):
        batch = _batch(jnp.arange(24.0).reshape(2, 4, 3), "cell", axis_groups=((2, 4),))

        assert batch.level_names == ("cell",)
        assert batch.axis_groups == ((2, 4),)


class TestNumericArrayBatchSelection:
    """Selection yields the element kind, as for every batch."""

    def test_an_element_is_a_numeric_array(self):
        element = _batch()[1]

        assert isinstance(element, NumericArray)
        np.testing.assert_array_equal(np.asarray(element), np.array([3.0, 4.0, 5.0]))

    def test_an_element_takes_the_derived_name(self):
        element = _batch(name="posterior")[1]

        assert element.name == "posterior[draw=1]"
        assert element.name_is_auto is True

    def test_an_element_inherits_the_batch_lineage(self):
        """Selecting computes nothing, so the element carries the batch's lineage."""
        batch = _batch().with_provenance(Provenance.create("sample", parents=[]))

        assert batch[1].provenance is batch.provenance

    def test_an_element_carries_the_batch_element_spec(self):
        assert _batch()[1].spec == _batch().element_spec

    def test_a_sub_batch_takes_the_declared_view_type(self):
        """`Batch._view_type` is the hook a subclass overrides to shed its state."""

        class _Derived(NumericArrayBatch):
            __slots__ = ()

            @property
            def _view_type(self) -> type:
                return NumericArrayBatch

        derived = _Derived(
            jnp.arange(12.0).reshape(4, 3),
            "draw",
            element_spec=NumericArraySpec(shape=(3,), dtype=jnp.float32),
            name="derived",
        )

        assert type(derived[1:3]) is NumericArrayBatch

    def test_a_slice_is_a_sub_batch(self):
        sub = _batch(name="posterior")[1:3]

        assert isinstance(sub, NumericArrayBatch)
        assert sub.batch_shape == (2,)
        assert sub.name == "posterior[draw=1:3]"

    def test_iteration_yields_elements(self):
        assert [type(e) for e in _batch()] == [NumericArray] * 4


class TestNumericArrayBatchOverNativeContainers:
    """Selection is positional, which `[]` is not on every container."""

    @staticmethod
    def _frame():
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(np.arange(12.0).reshape(4, 3))

    def test_an_element_selects_by_position_not_by_label(self):
        """``df[0]`` is a column, while the element is the row at position 0."""
        batch = NumericArrayBatch(
            self._frame(),
            "draw",
            element_spec=NumericArraySpec(shape=(3,), dtype=np.float64),
            name="draws",
        )

        element = batch[1]

        assert isinstance(element, NumericArray)
        assert element.shape == (3,)
        np.testing.assert_array_equal(np.asarray(element), np.array([3.0, 4.0, 5.0]))

    def test_a_sub_batch_selects_by_position(self):
        batch = NumericArrayBatch(
            self._frame(),
            "draw",
            element_spec=NumericArraySpec(shape=(3,), dtype=np.float64),
            name="draws",
        )

        sub = batch[1:3]

        assert isinstance(sub, NumericArrayBatch)
        assert sub.batch_shape == (2,)

    def test_the_container_is_still_stored_verbatim(self):
        pd = pytest.importorskip("pandas")
        frame = self._frame()

        batch = NumericArrayBatch(
            frame, "draw", element_spec=NumericArraySpec(shape=(3,), dtype=np.float64), name="draws"
        )

        assert isinstance(batch.values, pd.DataFrame)


class TestNumericArrayBatchRefusals:
    def test_a_stored_array_with_no_batch_axis_is_refused(self):
        """A batch has at least one batch axis; one value is a NumericArray."""
        with pytest.raises(ValueError, match="at least one batch axis"):
            NumericArrayBatch(
                jnp.zeros(3),
                "draw",
                element_spec=NumericArraySpec(shape=(3,), dtype=jnp.float32),
                name="draws",
            )

    def test_trailing_axes_that_are_not_the_event_shape_are_refused(self):
        with pytest.raises(ValueError, match="where its elements declare the event shape"):
            NumericArrayBatch(
                jnp.zeros((4, 5)),
                "draw",
                element_spec=NumericArraySpec(shape=(3,), dtype=jnp.float32),
                name="draws",
            )

    def test_a_symbolic_event_dimension_is_refused(self):
        """A symbolic size leaves the stored axes no split point."""
        with pytest.raises(ValueError, match="symbolic dimension"):
            NumericArrayBatch(
                jnp.zeros((4, 3)),
                "draw",
                element_spec=NumericArraySpec(shape=("n",), dtype=jnp.float32),
                name="draws",
            )

    def test_values_that_are_not_an_array_are_refused(self):
        with pytest.raises(TypeError, match="stores one array"):
            NumericArrayBatch(
                object(),
                "draw",
                element_spec=NumericArraySpec(shape=(), dtype=jnp.float32),
                name="draws",
            )

    def test_a_dtype_the_declaration_does_not_admit_is_refused(self):
        """The batch asserts the spec of every element, so it checks at build."""
        with pytest.raises(TypeError, match="does not admit"):
            NumericArrayBatch(
                jnp.zeros((2, 3), dtype=jnp.float32),
                "draw",
                element_spec=NumericArraySpec(shape=(3,), dtype=jnp.int32),
                name="draws",
            )

    def test_a_store_with_no_single_dtype_cannot_carry_a_pinned_one(self):
        """A backend may report no single dtype, which supports no pinned one."""
        from probpipe import ArrayBackend, register_array_backend

        class _NoSingleDtype:
            def __init__(self, array):
                self.array = array

        register_array_backend(
            _NoSingleDtype,
            ArrayBackend(
                event_shape=lambda o: o.array.shape,
                numpy_dtype=lambda o: None,
                to_jax=lambda o: jnp.asarray(o.array),
                to_numpy=lambda o: np.asarray(o.array),
                take=lambda o, index: _NoSingleDtype(o.array[index]),
            ),
        )
        store = _NoSingleDtype(np.zeros((4, 3)))

        with pytest.raises(TypeError, match="reports no single dtype"):
            NumericArrayBatch(
                store,
                "draw",
                element_spec=NumericArraySpec(shape=(3,), dtype=np.int32),
                name="draws",
            )

        # Declaring no dtype leaves nothing to substantiate, so it still builds.
        assert NumericArrayBatch(
            store, "draw", element_spec=NumericArraySpec(shape=(3,)), name="draws"
        ).batch_shape == (4,)

    def test_a_same_kind_dtype_is_admitted(self):
        """A widening or within-kind narrowing passes, as for a record."""
        batch = NumericArrayBatch(
            jnp.zeros((2, 3), dtype=jnp.float32),
            "draw",
            element_spec=NumericArraySpec(shape=(3,), dtype=jnp.float64),
            name="draws",
        )

        assert batch.batch_shape == (2,)

    def test_an_element_spec_of_another_kind_is_refused(self):
        with pytest.raises(TypeError, match="must be a NumericArraySpec"):
            NumericArrayBatch(jnp.zeros((4, 3)), "draw", element_spec="not a spec", name="draws")

    def test_axis_groups_must_tile_the_batch_shape(self):
        with pytest.raises(ValueError):
            _batch(jnp.arange(24.0).reshape(2, 4, 3), "cell", axis_groups=((2, 3),))


class TestNumericArrayIsAPyTree:
    """Registration is what lets it cross a transform boundary at all.

    A traced function reaches its arguments through the pytree registry, so
    registration is what carries a value into one.
    """

    def test_it_round_trips_through_flatten_and_unflatten(self):
        value = NumericArray(jnp.arange(3.0), name="draw")

        leaves, treedef = jax.tree_util.tree_flatten(value)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(rebuilt, NumericArray)
        assert (rebuilt.name, rebuilt.name_is_auto) == ("draw", False)
        np.testing.assert_array_equal(np.asarray(rebuilt), np.arange(3.0))

    def test_a_transform_that_changes_the_shape_keeps_the_declaration(self):
        """On this path a shape is transform-relative, so it states nothing.

        The value reports what it now holds; the spec keeps saying what was
        declared, which is the only thing the round trip can be faithful to.
        """
        stacked = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), NumericArray(jnp.arange(3.0)))

        assert stacked.shape == (2, 3)
        assert stacked.spec == NumericArraySpec(shape=(3,), dtype=jnp.float32)

    def test_a_declared_dtype_is_not_re_read_off_the_value(self):
        """`is_valid` admits a same-kind cast, so the two can differ.

        Deriving the spec from what arrives would quietly restate a float64
        declaration as float32 — the declaration is what the aux carries.
        """
        value = NumericArray(
            jnp.arange(3.0, dtype=jnp.float32),
            spec=NumericArraySpec(shape=(3,), dtype=np.float64),
        )

        leaves, treedef = jax.tree_util.tree_flatten(value)

        assert jax.tree_util.tree_unflatten(treedef, leaves).spec.dtype == np.float64

    def test_a_skeleton_still_carries_its_declaration(self):
        """`spec` is typed as one, so a rebuilt value has to have one."""
        skeleton = jax.tree_util.tree_map(lambda x: None, NumericArray(jnp.arange(3.0)))

        assert skeleton.spec == NumericArraySpec(shape=(3,), dtype=jnp.float32)

    def test_a_declared_support_rides_along(self):
        """The declaration is the aux, support included."""
        from probpipe import positive

        value = NumericArray(
            jnp.arange(1.0, 4.0),
            spec=NumericArraySpec(shape=(3,), dtype=jnp.float32, support=positive),
        )

        rebuilt = jax.tree_util.tree_map(lambda x: x, value)

        assert rebuilt.spec.support == positive

    def test_a_skeleton_rebuilds_rather_than_raising(self):
        """JAX unflattens with whatever it carries, and a skeleton is not an array."""
        value = NumericArray(jnp.arange(3.0), name="draw")

        skeleton = jax.tree_util.tree_map(lambda x: None, value)

        assert isinstance(skeleton, NumericArray)
        assert skeleton.name == "draw"

    def test_a_sentinel_child_rebuilds(self):
        _, treedef = jax.tree_util.tree_flatten(NumericArray(jnp.arange(3.0)))

        rebuilt = jax.tree_util.tree_unflatten(treedef, [object()])

        assert isinstance(rebuilt, NumericArray)

    def test_provenance_does_not_survive_the_boundary(self):
        """As for `Record`: lineage rides on the function layer, not the treedef."""
        value = NumericArray(jnp.arange(3.0)).with_provenance(
            Provenance.create("sample", parents=[])
        )

        rebuilt = jax.tree_util.tree_map(lambda x: x, value)

        assert rebuilt.provenance is None


class TestABatchIsNamed:
    """A batch's name is required, as a `Record`'s and an `Opaque`'s are."""

    def test_a_name_is_required(self):
        with pytest.raises(TypeError, match="name"):
            NumericArrayBatch(
                jnp.arange(12.0).reshape(4, 3),
                "draw",
                element_spec=NumericArraySpec(shape=(3,), dtype=jnp.float32),
            )

    def test_a_given_name_is_marked_user_given(self):
        batch = _batch(name="posterior")

        assert (batch.name, batch.name_is_auto) == ("posterior", False)

    def test_a_derived_name_says_so(self):
        """A view derives its name, and marks it, rather than defaulting."""
        sub = _batch(name="posterior")[1:3]

        assert (sub.name, sub.name_is_auto) == ("posterior[draw=1:3]", True)


class TestNumericArrayBatchIsAPyTree:
    """The two-transformation contract `RecordBatch` states, over one column.

    Registration is what lets a batch reach a traced function.
    """

    def test_it_round_trips_unchanged(self):
        batch = _batch(name="posterior")

        rebuilt = jax.tree_util.tree_map(lambda x: x, batch)

        assert isinstance(rebuilt, NumericArrayBatch)
        assert rebuilt.batch_shape == (4,)
        assert rebuilt.level_names == ("draw",)

    def test_removing_every_batch_axis_yields_the_element(self):
        """The value is one element, so it comes back as one.

        Observed through ``tree_map``: a ``vmap`` restacks, so the removal is
        visible only on the way in, inside the trace.
        """
        out = jax.tree_util.tree_map(lambda x: x[0], _batch())

        assert isinstance(out, NumericArray)
        assert out.shape == (3,)

    def test_a_vmap_hands_the_body_an_element_and_stacks_what_it_returns(self):
        """Which is why the batch's own levels are read rather than inferred.

        Unflattening a slice removes every batch axis, so the body receives a
        `NumericArray`, and returning it stacks into one with the mapped axis
        prepended. That value carries no levels, so its spec re-derives exactly
        from the arriving shape.
        """
        out = jax.vmap(lambda element: element)(_batch())

        assert isinstance(out, NumericArray)
        assert out.shape == (4, 3)

    def test_a_partial_or_resized_rank_is_refused(self):
        """A shape is not a provenance: no reading says which level survived."""
        with pytest.raises(ValueError, match="belongs to no level"):
            jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), _batch())

    def test_a_skeleton_rebuilds_rather_than_raising(self):
        skeleton = jax.tree_util.tree_map(lambda x: None, _batch())

        assert isinstance(skeleton, NumericArrayBatch)

    def test_it_reaches_a_traced_function(self):
        total = jax.jit(lambda b: jnp.sum(b))(_batch())

        assert float(total) == float(jnp.sum(jnp.arange(12.0)))


class TestNumericArrayBatchArrayShim:
    """Single-column, so it forwards from its store."""

    def test_shape_is_the_whole_store(self):
        batch = _batch()

        assert batch.shape == (4, 3)
        assert batch.ndim == 2
        assert batch.batch_shape == (4,)

    def test_it_converts_through_both_hooks(self):
        batch = _batch()

        assert np.asarray(batch).shape == (4, 3)
        assert isinstance(jnp.asarray(batch), jax.Array)

    def test_dtype_reports_the_store(self):
        assert _batch().dtype == jnp.float32
