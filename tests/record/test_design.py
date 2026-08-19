"""Tests for ``probpipe.record.design``.

A ``Design`` is a ``RecordBatch`` whose rows are materialised from
per-field marginals according to a subclass-specific rule. This file
covers :class:`FullFactorialDesign`; other subclasses land in
follow-up PRs.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    FullFactorialDesign,
    NumericArray,
    NumericArrayBatch,
    NumericRecord,
    NumericRecordBatch,
    OpaqueBatch,
    Record,
    RecordBatch,
    function,
)

# Some assertions use NumericRecord / NumericRecordBatch — these only
# appear as Function outputs, not as Design types. A Design is
# always a plain RecordBatch subclass; the columns themselves are
# jnp.ndarray for numeric marginals.


# ---------------------------------------------------------------------------
# FullFactorialDesign — construction + shape invariants
# ---------------------------------------------------------------------------


class TestFullFactorial:
    """A FullFactorialDesign materialises the Cartesian product of its
    marginals into a RecordBatch whose ``batch_shape`` is
    ``(prod(sizes),)`` and whose rows sweep the axes in marginal-
    insertion order (row-major)."""

    def test_two_numeric_marginals(self):
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        assert isinstance(ff, RecordBatch)
        assert ff.batch_shape == (6,)
        # Fields come back in insertion order.
        assert ff.event_template.fields == ("r", "K")
        # Numeric-only marginals produce ``jnp.ndarray`` column leaves.
        assert isinstance(ff["r"], jnp.ndarray)
        assert isinstance(ff["K"], jnp.ndarray)

    def test_row_order_is_lexicographic(self):
        """With insertion-order axes ``r`` (outer) and ``K`` (inner),
        row order is (r=1.5, K=60), (r=1.5, K=80), (r=1.8, K=60), ...
        """
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        np.testing.assert_allclose(
            np.asarray(ff["r"]),
            [1.5, 1.5, 1.8, 1.8, 2.0, 2.0],
        )
        np.testing.assert_allclose(
            np.asarray(ff["K"]),
            [60.0, 80.0, 60.0, 80.0, 60.0, 80.0],
        )

    def test_single_marginal_edge_case(self):
        ff = FullFactorialDesign(method=["pymc"])
        # Categorical-only falls back to RecordBatch (non-numeric leaf).
        assert isinstance(ff, RecordBatch)
        assert not isinstance(ff, NumericRecordBatch)
        assert ff.batch_shape == (1,)
        assert ff.event_template.fields == ("method",)

    def test_mixed_numeric_and_categorical(self):
        """String marginals produce ``dtype=object`` columns; the
        design falls back to the permissive ``RecordBatch`` base."""
        ff = FullFactorialDesign(
            method=["nutpie", "pymc"],
            scale=[0.5, 1.0],
        )
        assert isinstance(ff, RecordBatch)
        assert not isinstance(ff, NumericRecordBatch)
        assert ff.batch_shape == (4,)
        # Insertion order: method outer, scale inner.
        assert list(ff["method"]) == ["nutpie", "nutpie", "pymc", "pymc"]
        np.testing.assert_allclose(
            np.asarray(ff["scale"]),
            [0.5, 1.0, 0.5, 1.0],
        )

    def test_three_axes_shape_and_count(self):
        ff = FullFactorialDesign(
            a=[1, 2, 3],
            b=[10, 20],
            c=[100, 200, 300, 400],
        )
        assert ff.batch_shape == (3 * 2 * 4,)

    def test_empty_marginals_raises(self):
        with pytest.raises(ValueError, match="at least one marginal"):
            FullFactorialDesign()

    def test_empty_marginal_column_raises(self):
        with pytest.raises(ValueError, match="must each be non-empty"):
            FullFactorialDesign(r=[1.0, 2.0], K=[])

    def test_marginals_introspection(self):
        """``.marginals`` returns the original per-field sequences."""
        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        marginals = ff.marginals
        assert set(marginals) == {"r", "K"}
        assert list(marginals["r"]) == [1.5, 1.8]
        assert list(marginals["K"]) == [60.0, 80.0]

    def test_single_row_record_indexing(self):
        """Integer-indexing a Design returns a single Record (scalar
        row), matching the RecordBatch contract."""
        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        # Insertion order: r outer, K inner. Second row → (r=1.5, K=80).
        row = ff[1]
        assert isinstance(row, Record)
        assert float(row["r"]) == pytest.approx(1.5)
        assert float(row["K"]) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Sweep integration — the main point of Designs
# ---------------------------------------------------------------------------


class TestDesignAsSweep:
    """The two idiomatic ways to pipe a Design into a Function.

    Pattern A (general): ``f(p=design)`` with ``f(p)`` taking a
    single ``Record`` arg — the WF sweep path runs one inner call per
    row.

    Pattern B (convenience): ``f(**design.select_all())`` with
    per-field scalar args — runs ``f`` once with full column arrays
    and relies on JAX broadcasting. Does not trigger the WF sweep.
    """

    def test_single_record_arg_triggers_sweep(self):
        @function
        def fit(p: NumericRecord):
            return p["r"] * p["K"]

        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        out = fit(p=ff)
        assert isinstance(out, NumericArrayBatch)
        assert out.batch_shape == (6,)
        # Insertion order: r outer, K inner.
        np.testing.assert_allclose(
            out.values,
            [1.5 * 60, 1.5 * 80, 1.8 * 60, 1.8 * 80, 2.0 * 60, 2.0 * 80],
        )

    def test_select_all_splat_triggers_zip_sweep(self):
        """Splatting ``**design.select_all()`` yields sibling views of
        the same Design. The WF sweep layer groups them by parent
        identity and iterates in lockstep — one inner call per row —
        producing an aggregate identical to the single Record-arg pattern
        (``fit(p=design)``)."""

        @function
        def product(r, K):
            return r["r"] * K["K"]

        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        out = product(**ff.select_all())
        assert isinstance(out, NumericArrayBatch)
        assert out.batch_shape == (6,)
        # Insertion order: r outer, K inner.
        np.testing.assert_allclose(
            out.values,
            [1.5 * 60, 1.5 * 80, 1.8 * 60, 1.8 * 80, 2.0 * 60, 2.0 * 80],
        )

    def test_patterns_a_and_b_are_equivalent(self):
        """Pattern A (``f(p=design)``) and Pattern B
        (``f(**design.select_all())``) produce identical outputs."""
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])

        @function
        def fit_a(p: NumericRecord):
            return p["r"] * p["K"]

        @function
        def fit_b(r, K):
            return r["r"] * K["K"]

        out_a = fit_a(p=ff)
        out_b = fit_b(**ff.select_all())
        assert out_a.batch_shape == out_b.batch_shape == (6,)
        np.testing.assert_allclose(out_a.values, out_b.values)

    def test_raw_fields_still_cartesian_product(self):
        """Passing raw columns (``design["r"]``, ``design["K"]``) gives
        the expected independent-arrays behaviour: they cartesian-product
        because they carry no parent-identity signal the WF layer can
        use to zip them."""

        @function
        def product(r, K):
            return r * K

        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        # Raw columns → two independent jnp.ndarrays. With no type
        # hints they're passed to the body wholesale and JAX broadcasts
        # the arithmetic to a (6,)-array; WF wraps as NumericRecord.
        out = product(r=ff["r"], K=ff["K"])
        # Confirm the output is a single value carrying the arithmetic
        # result, not a swept NumericArrayBatch.
        assert isinstance(out, NumericArray)
        assert out.shape == (6,)

    def test_mixed_field_sweep_uses_record_arg_pattern(self):
        """Categorical fields can't ride JAX broadcasting — the single
        Record arg pattern is the only one that works when any marginal
        is string-valued."""

        @function
        def label(p: Record):
            return f"{p['method']}-{float(p['scale']):.1f}"

        ff = FullFactorialDesign(
            method=["nutpie", "pymc"],
            scale=[0.5, 1.0],
        )
        out = label(p=ff)
        # A string row is an opaque value, so the rows batch at that kind and
        # the elements are reached by position rather than by a field name.
        assert isinstance(out, OpaqueBatch)
        assert out.batch_shape == (4,)
        assert [out[i] for i in range(4)] == [
            "nutpie-0.5",
            "nutpie-1.0",
            "pymc-0.5",
            "pymc-1.0",
        ]


# ---------------------------------------------------------------------------
# Introspection + select_all
# ---------------------------------------------------------------------------


class TestSelectAll:
    def test_select_all_returns_views(self):
        """``select_all()`` returns single-field views that share the
        Design as their parent. Sibling views passed to a
        ``Function`` zip rather than cartesian-product — the
        mechanism behind ``f(**design.select_all()) ≡ f(p=design)``."""

        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        cols = ff.select_all()
        assert set(cols) == {"r", "K"}
        # A view is a plain batch, not a Design: it holds none of the marginals.
        assert type(cols["r"]) is RecordBatch
        assert type(cols["K"]) is RecordBatch
        # It carries the design's own level, which is what the sweep zips on.
        assert cols["r"].level_names == cols["K"].level_names == ("design",)
        assert cols["r"].batch_shape == cols["K"].batch_shape == (4,)
        assert list(cols["r"].event_template) == ["r"]
        assert list(cols["K"].event_template) == ["K"]
