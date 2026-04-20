"""Tests for ``probpipe.record.design``.

A ``Design`` is a ``RecordArray`` whose rows are materialised from
per-field marginals according to a subclass-specific rule. This file
covers :class:`FullFactorialDesign`; other subclasses land in
follow-up PRs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    NumericRecord,
    NumericRecordArray,
    Record,
    RecordArray,
    workflow_function,
)
from probpipe import FullFactorialDesign

# Some assertions use NumericRecord / NumericRecordArray — these only
# appear as WorkflowFunction outputs, not as Design types. A Design is
# always a plain RecordArray subclass; the columns themselves are
# jnp.ndarray for numeric marginals.


# ---------------------------------------------------------------------------
# FullFactorialDesign — construction + shape invariants
# ---------------------------------------------------------------------------


class TestFullFactorial:
    """A FullFactorialDesign materialises the Cartesian product of its
    marginals into a RecordArray whose ``batch_shape`` is
    ``(prod(sizes),)`` and whose rows sweep the sorted axes in
    lexicographic (row-major) order."""

    def test_two_numeric_marginals(self):
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        assert isinstance(ff, RecordArray)
        assert ff.batch_shape == (6,)
        # Fields come back sorted.
        assert ff.fields == ("K", "r")
        # Numeric-only marginals produce ``jnp.ndarray`` column leaves.
        assert isinstance(ff["r"], jnp.ndarray)
        assert isinstance(ff["K"], jnp.ndarray)

    def test_row_order_is_lexicographic(self):
        """With sorted axes K (outer) and r (inner), row order is
        (K=60, r=1.5), (K=60, r=1.8), (K=60, r=2.0), (K=80, r=1.5), ...
        """
        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        np.testing.assert_allclose(
            np.asarray(ff["K"]), [60., 60., 60., 80., 80., 80.],
        )
        np.testing.assert_allclose(
            np.asarray(ff["r"]), [1.5, 1.8, 2.0, 1.5, 1.8, 2.0],
        )

    def test_single_marginal_edge_case(self):
        ff = FullFactorialDesign(method=["pymc"])
        # Categorical-only falls back to RecordArray (non-numeric leaf).
        assert isinstance(ff, RecordArray)
        assert not isinstance(ff, NumericRecordArray)
        assert ff.batch_shape == (1,)
        assert ff.fields == ("method",)

    def test_mixed_numeric_and_categorical(self):
        """String marginals produce ``dtype=object`` columns; the
        design falls back to the permissive ``RecordArray`` base."""
        ff = FullFactorialDesign(
            method=["nutpie", "pymc"], scale=[0.5, 1.0],
        )
        assert isinstance(ff, RecordArray)
        assert not isinstance(ff, NumericRecordArray)
        assert ff.batch_shape == (4,)
        np.testing.assert_allclose(
            np.asarray(ff["scale"]), [0.5, 1.0, 0.5, 1.0],
        )
        # sorted fields: ("method", "scale"); method varies slowest
        assert list(ff["method"]) == ["nutpie", "nutpie", "pymc", "pymc"]

    def test_three_axes_shape_and_count(self):
        ff = FullFactorialDesign(
            a=[1, 2, 3], b=[10, 20], c=[100, 200, 300, 400],
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
        row), matching the RecordArray contract."""
        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        row = ff[1]  # second row in lex order → (K=60, r=1.8)
        assert isinstance(row, Record)
        assert float(row["r"]) == pytest.approx(1.8)
        assert float(row["K"]) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Sweep integration — the main point of Designs
# ---------------------------------------------------------------------------


class TestDesignAsSweep:
    """The two idiomatic ways to pipe a Design into a WorkflowFunction.

    Pattern A (general): ``f(p=design)`` with ``f(p)`` taking a
    single ``Record`` arg — the WF sweep path runs one inner call per
    row.

    Pattern B (convenience): ``f(**design.select_all())`` with
    per-field scalar args — runs ``f`` once with full column arrays
    and relies on JAX broadcasting. Does not trigger the WF sweep.
    """

    def test_single_record_arg_triggers_sweep(self):
        @workflow_function
        def fit(p: NumericRecord):
            return p["r"] * p["K"]

        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        out = fit(p=ff)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (6,)
        np.testing.assert_allclose(
            np.asarray(out["fit"]),
            [1.5 * 60, 1.8 * 60, 2.0 * 60, 1.5 * 80, 1.8 * 80, 2.0 * 80],
        )

    def test_select_all_splat_for_jax_vectorized_body(self):
        @workflow_function
        def product(r, K):
            return r * K

        ff = FullFactorialDesign(r=[1.5, 1.8, 2.0], K=[60.0, 80.0])
        out = product(**ff.select_all())
        # Splat into a vectorizable body: WF runs once, JAX broadcasts
        # to a (6,)-array, WF wraps as NumericRecord with the function
        # name as the single field.
        assert isinstance(out, NumericRecord)
        assert "product" in out.fields
        np.testing.assert_allclose(
            np.asarray(out["product"]),
            [1.5 * 60, 1.8 * 60, 2.0 * 60, 1.5 * 80, 1.8 * 80, 2.0 * 80],
        )

    def test_mixed_field_sweep_uses_record_arg_pattern(self):
        """Categorical fields can't ride JAX broadcasting — the single
        Record arg pattern is the only one that works when any marginal
        is string-valued."""

        @workflow_function
        def label(p: Record):
            return f'{p["method"]}-{float(p["scale"]):.1f}'

        ff = FullFactorialDesign(
            method=["nutpie", "pymc"], scale=[0.5, 1.0],
        )
        out = label(p=ff)
        assert isinstance(out, RecordArray)
        assert out.batch_shape == (4,)
        assert list(out["label"]) == [
            "nutpie-0.5", "nutpie-1.0", "pymc-0.5", "pymc-1.0",
        ]


# ---------------------------------------------------------------------------
# Introspection + select_all
# ---------------------------------------------------------------------------


class TestSelectAll:
    def test_select_all_returns_column_arrays(self):
        ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
        cols = ff.select_all()
        assert set(cols) == {"r", "K"}
        # Each column is the raw field array (not a single-field RecordArray).
        assert not isinstance(cols["r"], RecordArray)
        assert cols["r"].shape == (4,)
        assert cols["K"].shape == (4,)
