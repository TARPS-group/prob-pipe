"""Tests for ``_RecordArrayView`` and the parent-identity sweep grouping.

A ``_RecordArrayView`` is a thin single-field wrapper around a
``RecordArray`` column, carrying the parent RA as shared-identity
metadata. The ``WorkflowFunction`` sweep layer recognises sibling
views (views of the same parent) and iterates them in lockstep (zip)
rather than cartesian-producting them.

Views are constructed via ``RecordArray.view("field")`` or via
``Design.select_all()``; ``ra["field"]`` still returns the raw column.
These tests cover:

- View construction + attribute/conversion forwarding.
- Parent-identity grouping in the sweep layer:
  * Views from the same parent → zip (one axis block).
  * Views from different parents → product (independent blocks).
  * Views + plain ``RecordArray`` / ``DistributionArray`` compose
    correctly (zip within sibling group, product across groups).
- Pattern-B parity: ``f(**design.select_all()) ≡ f(p=design)``.
- Raw columns (``ra["field"]``) still behave as plain ndarrays — no
  sweep grouping, independent axes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    FullFactorialDesign,
    Normal,
    NumericRecord,
    NumericRecordArray,
    Record,
    RecordArray,
    DistributionArray,
    workflow_function,
)
from probpipe.core._record_array import _RecordArrayView
from probpipe.core.record import RecordTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_ra():
    """A NumericRecordArray with two numeric fields, batch_shape=(4,)."""
    return NumericRecordArray.stack(
        [NumericRecord(x=float(i), y=float(i * 2)) for i in range(4)]
    )


@pytest.fixture
def mixed_ra():
    """A plain RecordArray with one numeric and one string field."""
    tpl = RecordTemplate(method=None, scale=())
    return RecordArray(
        {
            "method": np.asarray(["nutpie", "pymc", "stan"], dtype=object),
            "scale": jnp.array([0.5, 1.0, 2.0]),
        },
        batch_shape=(3,),
        template=tpl,
    )


# ---------------------------------------------------------------------------
# View construction + basic behavior
# ---------------------------------------------------------------------------


class TestViewConstruction:
    """``ra.view(field)`` returns a single-field view; ``ra["field"]``
    still returns the raw column. The view aliases the parent's data —
    no copy."""

    def test_view_is_recordarray_subclass(self, numeric_ra):
        v = numeric_ra.view("x")
        assert isinstance(v, _RecordArrayView)
        assert isinstance(v, RecordArray)

    def test_view_parent_identity(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v.parent is numeric_ra
        assert v.field == "x"

    def test_view_single_field(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v.fields == ("x",)

    def test_view_batch_shape_matches_parent(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v.batch_shape == numeric_ra.batch_shape == (4,)

    def test_view_aliases_underlying_column(self, numeric_ra):
        v = numeric_ra.view("x")
        # No copy — the underlying array is the parent's store entry.
        assert v._store["x"] is numeric_ra._store["x"]

    def test_view_unknown_field_raises(self, numeric_ra):
        with pytest.raises(KeyError):
            numeric_ra.view("not_a_field")

    def test_ra_getitem_still_returns_raw(self, numeric_ra):
        """The ``ra[field]`` contract is unchanged — it returns the
        raw column. Views are opt-in via ``ra.view(field)`` or
        ``Design.select_all()``."""
        assert isinstance(numeric_ra["x"], jnp.ndarray)
        assert not isinstance(numeric_ra["x"], _RecordArrayView)

    def test_mixed_field_view_object_dtype(self, mixed_ra):
        """View works on non-numeric (object-dtype) fields too."""
        v = mixed_ra.view("method")
        assert isinstance(v, _RecordArrayView)
        assert v.parent is mixed_ra
        assert v.shape == (3,)


# ---------------------------------------------------------------------------
# View attribute / conversion forwarding
# ---------------------------------------------------------------------------


class TestViewForwarding:
    """Views forward a minimal set of array-like accessors: conversion
    (``__array__`` / ``__jax_array__``), shape / dtype / ndim, indexing,
    iteration, len. Arithmetic / reductions require explicit
    ``jnp.asarray(view)`` — consistent with the explicit-conversion
    policy used elsewhere for single-field shims."""

    def test_shape_forwards_to_column(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v.shape == (4,)

    def test_dtype_forwards_to_column(self, numeric_ra):
        v = numeric_ra.view("x")
        # Default float dtype: float32 normally, float64 with jax_enable_x64.
        assert v.dtype == jnp.zeros((), dtype=float).dtype

    def test_ndim_forwards_to_column(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v.ndim == 1

    def test_jnp_asarray_returns_underlying_column(self, numeric_ra):
        v = numeric_ra.view("x")
        arr = jnp.asarray(v)
        np.testing.assert_allclose(np.asarray(arr), [0.0, 1.0, 2.0, 3.0])

    def test_np_asarray_returns_underlying_column(self, numeric_ra):
        v = numeric_ra.view("x")
        arr = np.asarray(v)
        np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0])

    def test_view_indexing_slices_column(self, numeric_ra):
        """``view[i]`` forwards to the underlying column's i-th row —
        matching the raw-array intuition users have for ``ra["x"][i]``."""
        v = numeric_ra.view("x")
        assert float(v[0]) == 0.0
        assert float(v[2]) == 2.0

    def test_view_string_index_is_idempotent(self, numeric_ra):
        v = numeric_ra.view("x")
        assert v["x"] is v

    def test_view_string_index_wrong_field_raises(self, numeric_ra):
        v = numeric_ra.view("x")
        with pytest.raises(KeyError):
            v["y"]

    def test_view_len_forwards(self, numeric_ra):
        v = numeric_ra.view("x")
        assert len(v) == 4

    def test_view_iteration_forwards(self, numeric_ra):
        v = numeric_ra.view("x")
        collected = [float(x) for x in v]
        assert collected == [0.0, 1.0, 2.0, 3.0]

    def test_jnp_ops_need_explicit_conversion(self, numeric_ra):
        """Recent JAX rejects ``__jax_array__`` during abstractification,
        so ``jnp.sum(view)`` raises. Users must convert explicitly."""
        v = numeric_ra.view("x")
        assert float(jnp.sum(jnp.asarray(v))) == pytest.approx(6.0)
        assert float(jnp.mean(jnp.asarray(v))) == pytest.approx(1.5)

    def test_arithmetic_requires_explicit_conversion(self, numeric_ra):
        """Arithmetic operators are *not* forwarded; users must
        ``jnp.asarray(view)`` before arithmetic. Documents the
        explicit-conversion policy."""
        v = numeric_ra.view("x")
        with pytest.raises(TypeError):
            v + 1           # no __add__ on view
        np.testing.assert_allclose(
            np.asarray(jnp.asarray(v) + 1),
            [1.0, 2.0, 3.0, 4.0],
        )


# ---------------------------------------------------------------------------
# Sweep grouping — parent identity drives zip vs product
# ---------------------------------------------------------------------------


class TestSweepGroupingByParent:
    """The ``WorkflowFunction`` sweep layer groups array-valued args by
    parent identity. Views sharing a parent iterate in lockstep (zip);
    distinct parents (including two independent ``RecordArray``s or
    two views of different parents) product."""

    def test_two_views_same_parent_zip(self, numeric_ra):
        """Two views of ``numeric_ra`` zip into a single (4,) axis."""

        @workflow_function
        def f(x, y):
            return x + y

        out = f(x=numeric_ra.view("x"), y=numeric_ra.view("y"))
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        # Row i: x[i] + y[i] = i + 2i = 3i
        np.testing.assert_allclose(
            np.asarray(out["f"]),
            [0.0, 3.0, 6.0, 9.0],
        )

    def test_two_views_different_parents_product(self):
        """Views from two different ``RecordArray``s carry different
        parent ids → distinct groups → cartesian product."""
        ra_a = NumericRecordArray.stack(
            [NumericRecord(a=float(i)) for i in range(3)]
        )
        ra_b = NumericRecordArray.stack(
            [NumericRecord(b=float(j * 10)) for j in range(2)]
        )

        @workflow_function
        def f(a, b):
            return a + b

        out = f(a=ra_a.view("a"), b=ra_b.view("b"))
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (3, 2)   # cartesian product

    def test_view_plus_plain_ra_products(self, numeric_ra):
        """A view and an independent ``RecordArray`` are in different
        groups (the view's parent ≠ the independent RA) → product."""
        other = NumericRecordArray.stack(
            [NumericRecord(z=float(k * 100)) for k in range(2)]
        )

        @workflow_function
        def f(x, p):
            return x + p["z"]

        out = f(x=numeric_ra.view("x"), p=other)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4, 2)

    def test_three_views_same_parent_zip_to_single_axis(self):
        """A 3-field NumericRecordArray with three sibling views
        collapses to one (n,) axis, not (n, n, n)."""
        ra = NumericRecordArray.stack(
            [NumericRecord(a=float(i), b=float(i * 2), c=float(i * 10))
             for i in range(5)]
        )

        @workflow_function
        def g(a, b, c):
            return a + b + c

        out = g(a=ra.view("a"), b=ra.view("b"), c=ra.view("c"))
        assert out.batch_shape == (5,)
        # Row i: i + 2i + 10i = 13i
        np.testing.assert_allclose(
            np.asarray(out["g"]),
            [0.0, 13.0, 26.0, 39.0, 52.0],
        )


# ---------------------------------------------------------------------------
# Pattern A ≡ Pattern B on Designs
# ---------------------------------------------------------------------------


class TestPatternParity:
    """``Design.select_all()`` returns sibling views — splatting them
    through a ``WorkflowFunction`` produces the same output as passing
    the whole Design as a single ``Record``-typed arg."""

    def test_parity_on_full_factorial(self):
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])

        @workflow_function
        def fit_a(p: NumericRecord):
            return p["r"] * p["K"]

        @workflow_function
        def fit_b(r, K):
            return r * K

        out_a = fit_a(p=ff)
        out_b = fit_b(**ff.select_all())
        assert out_a.batch_shape == out_b.batch_shape == (6,)
        np.testing.assert_allclose(
            np.asarray(out_a["fit_a"]),
            np.asarray(out_b["fit_b"]),
        )

    def test_parity_on_mixed_fields(self):
        """Works with mixed numeric/categorical marginals too."""
        ff = FullFactorialDesign(
            method=["nutpie", "pymc"], scale=[0.5, 1.0],
        )

        @workflow_function
        def label_a(p: Record):
            return f'{p["method"]}-{float(p["scale"]):.1f}'

        @workflow_function
        def label_b(method, scale):
            return f'{method}-{float(scale):.1f}'

        out_a = label_a(p=ff)
        out_b = label_b(**ff.select_all())
        assert out_a.batch_shape == out_b.batch_shape == (4,)
        assert list(out_a["label_a"]) == list(out_b["label_b"])

    def test_select_all_returns_views(self):
        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        cols = ff.select_all()
        # Sibling views — share the Design as their parent.
        assert all(isinstance(v, _RecordArrayView) for v in cols.values())
        assert all(v.parent is ff for v in cols.values())


# ---------------------------------------------------------------------------
# Raw-column behaviour stays unchanged
# ---------------------------------------------------------------------------


class TestRawColumnsUnchanged:
    """Raw columns (``ra["field"]``) carry no parent-identity signal,
    so two raw columns from the same RA behave as **independent**
    arrays in a WF call — JAX broadcasts the body once and the result
    is a single ``NumericRecord``, not a per-cell sweep."""

    def test_raw_columns_go_through_body_once(self, numeric_ra):
        @workflow_function
        def f(x, y):
            return x + y

        out = f(x=numeric_ra["x"], y=numeric_ra["y"])
        # Raw arrays aren't sweep sources under the scaled-back contract;
        # body runs once with full columns and JAX broadcasts.
        assert isinstance(out, NumericRecord)
        assert out["f"].shape == (4,)


# ---------------------------------------------------------------------------
# Composition with DistributionArray and Distribution args
# ---------------------------------------------------------------------------


class TestMixedGroupingComposition:
    """Views + ``DistributionArray`` + ``Distribution`` args compose:
    within-parent views zip, distinct-parent array args product, and
    Distribution args MC-marginalise per outer cell (nested regime)."""

    def test_view_plus_distribution_array(self, numeric_ra):
        """A view group (zip on (4,)) crossed with a
        ``DistributionArray`` of shape (2,) products to batch (4, 2).
        Each cell sees scalar ``x`` and one component Distribution
        (``d``), so the inner body can call ``d.mean()``-style ops."""
        from probpipe import mean
        da = DistributionArray(
            [Normal(loc=float(i), scale=1.0, name=f"n{i}") for i in range(2)],
        )

        @workflow_function
        def f(x, d):
            # d is a scalar Distribution per cell — add x to its mean.
            return x + float(d._mean())

        out = f(x=numeric_ra.view("x"), d=da)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4, 2)

    def test_view_plus_scalar_distribution_nested(self, numeric_ra):
        """View sweep + scalar Distribution arg (in a scalar slot) →
        nested regime: outer stack over view, inner MC-marginalise.
        Output is a DistributionArray of (4,) per-row marginals."""
        noise = Normal(loc=0.0, scale=0.1, name="noise")

        @workflow_function(n_broadcast_samples=20, vectorize="loop")
        def f(x, noise: float):
            return x + noise

        out = f(x=numeric_ra.view("x"), noise=noise)
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (4,)
