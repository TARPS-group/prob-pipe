"""Tests for RecordArray-based parameter-sweep broadcasting (issue #130).

The four regimes (no broadcast, distribution-only, RecordArray-only,
nested) cover the parity matrix in the issue: 4 inner-return types
(scalar, ndarray, Record/NumericRecord, Distribution) × 2 broadcast
types (Distribution, RecordArray) = 8 core cases plus edge cases.

Provenance tests confirm every broadcast output carries a source node
pointing back to its input RecordArray / Distribution parents.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    DistributionArray,
    Normal,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
    Record,
    RecordArray,
    workflow_function,
)
from probpipe.core._distribution_base import Distribution


# ---------------------------------------------------------------------------
# Detection: _find_broadcast_args splits args into (dist, ra) groups
# ---------------------------------------------------------------------------


class TestRecordArrayDetection:
    """``_find_broadcast_args`` classifies inputs and validates shapes."""

    def test_recordarray_triggers_broadcast(self):
        @workflow_function
        def f(p: NumericRecord) -> float:
            return p["x"]

        ra = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(4)]
        )
        out = f(p=ra)
        assert out.batch_shape == (4,)

    def test_recordarray_hint_skips_broadcast(self):
        """When the hint is ``RecordArray``, the batched value flows
        through as-is — the function expects the batch."""

        @workflow_function
        def f(ra: NumericRecordArray):
            return ra["x"].sum()

        ra = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(4)]
        )
        out = f(ra=ra)
        # Single inner call — output is the unwrapped float result.
        assert float(out) == 6.0

    def test_zero_dim_recordarray_passes_through(self):
        """batch_shape=() means no batch axis, so don't iterate."""
        from probpipe.core.record import RecordTemplate

        @workflow_function
        def f(p: NumericRecord) -> float:
            return p["x"]

        tpl = RecordTemplate(x=())
        scalar_ra = NumericRecordArray({"x": jnp.asarray(7.0)},
                                       batch_shape=(), template=tpl)
        # No broadcasting — one call with the batch_shape=() RecordArray
        # passed through.
        out = f(p=scalar_ra)
        assert float(out) == 7.0

    def test_mismatched_batch_shapes_raise(self):
        @workflow_function
        def f(a: NumericRecord, b: NumericRecord) -> float:
            return a["x"] + b["y"]

        a = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(3)]
        )
        b = NumericRecordArray.stack(
            [NumericRecord(y=float(i)) for i in range(5)]
        )
        with pytest.raises(ValueError, match="batch_shapes"):
            f(a=a, b=b)


# ---------------------------------------------------------------------------
# Pure sweep (RecordArray-only) — stacked outputs, no marginalisation
# ---------------------------------------------------------------------------


class TestPureSweepParityMatrix:
    """4 inner-return types → stacked outputs (see issue #130 table)."""

    @pytest.fixture
    def sweep(self):
        return NumericRecordArray.stack(
            [NumericRecord(r=float(i), k=10.0 * i) for i in range(1, 5)]
        )

    def test_scalar_return_wraps_as_numericrecordarray(self, sweep):
        @workflow_function
        def f(p: NumericRecord) -> float:
            return p["r"] * p["k"]

        out = f(p=sweep)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        np.testing.assert_allclose(out["result"], [10.0, 40.0, 90.0, 160.0])

    def test_ndarray_return_preserves_event_shape(self, sweep):
        @workflow_function
        def f(p: NumericRecord) -> jnp.ndarray:
            return jnp.array([p["r"], p["k"], p["r"] + p["k"]])

        out = f(p=sweep)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        assert out["result"].shape == (4, 3)

    def test_numericrecord_return_stacks(self, sweep):
        @workflow_function
        def f(p: NumericRecord) -> NumericRecord:
            return NumericRecord(prod=p["r"] * p["k"], sum=p["r"] + p["k"])

        out = f(p=sweep)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        np.testing.assert_allclose(out["prod"], [10.0, 40.0, 90.0, 160.0])
        np.testing.assert_allclose(out["sum"], [11.0, 22.0, 33.0, 44.0])

    def test_record_with_string_stacks_as_recordarray(self, sweep):
        """Heterogeneous records (numeric + string) produce a plain
        ``RecordArray`` rather than the numeric-only variant."""

        @workflow_function
        def f(p: NumericRecord) -> Record:
            return Record(value=p["r"] * p["k"], label="row")

        out = f(p=sweep)
        assert isinstance(out, RecordArray)
        assert not isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        assert list(out["label"]) == ["row"] * 4

    def test_distribution_return_gives_distribution_array(self, sweep):
        @workflow_function
        def f(p: NumericRecord) -> Distribution:
            return Normal(loc=p["r"], scale=1.0, name="out")

        out = f(p=sweep)
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (4,)
        # Each component has the expected mean from its sweep row.
        means = jnp.stack([out[i]._mean() for i in range(4)])
        np.testing.assert_allclose(means, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Distribution-only (smoke check — refactor didn't break anything)
# ---------------------------------------------------------------------------


class TestDistributionOnlyPath:
    def test_scalar_output_still_marginal(self):
        @workflow_function(n_broadcast_samples=50, vectorize="loop")
        def f(x: float) -> float:
            return x * 2

        from probpipe.core._broadcast_distributions import _ArrayMarginal
        out = f(x=Normal(loc=2.0, scale=0.1, name="x"))
        assert isinstance(out, _ArrayMarginal)
        # Mean of 2 * N(2, 0.1) is ~4.
        from probpipe import mean as pmean
        assert abs(float(pmean(out)) - 4.0) < 0.1


# ---------------------------------------------------------------------------
# Nested (RecordArray + Distribution) — per-row marginal, stacked
# ---------------------------------------------------------------------------


class TestNestedSweepPlusDistribution:
    """Each argument keeps its semantics: RecordArray stacks, Distribution
    marginalises. Output is always a DistributionArray of per-row
    marginals (satisfies the Record | RecordArray | Distribution
    output-type contract)."""

    @pytest.fixture
    def sweep(self):
        return NumericRecordArray.stack(
            [NumericRecord(bias=float(i)) for i in range(3)]
        )

    def test_scalar_inner_gives_distribution_array(self, sweep):
        @workflow_function(n_broadcast_samples=50, vectorize="loop")
        def f(p: NumericRecord, noise: float) -> float:
            return p["bias"] + noise

        out = f(p=sweep, noise=Normal(loc=0.0, scale=0.5, name="noise"))
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (3,)
        # Per-row mean ≈ bias + 0
        from probpipe import mean as pmean
        means = pmean(out)
        np.testing.assert_allclose(means, [0.0, 1.0, 2.0], atol=0.2)

    def test_record_inner_gives_distribution_array(self, sweep):
        @workflow_function(n_broadcast_samples=30, vectorize="loop")
        def f(p: NumericRecord, noise: float) -> NumericRecord:
            return NumericRecord(x=p["bias"] + noise, y=p["bias"] - noise)

        out = f(p=sweep, noise=Normal(loc=0.0, scale=0.1, name="noise"))
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (3,)

    def test_distribution_inner_gives_distribution_array(self, sweep):
        @workflow_function(n_broadcast_samples=20, vectorize="loop")
        def f(p: NumericRecord, noise: float) -> Distribution:
            return Normal(loc=p["bias"] + noise, scale=1.0, name="out")

        out = f(p=sweep, noise=Normal(loc=0.0, scale=0.1, name="noise"))
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (3,)


# ---------------------------------------------------------------------------
# Vectorisation — vmap path vs loop path agree on pure-sweep outputs
# ---------------------------------------------------------------------------


class TestVectorization:
    """The JAX-vmap and Python-loop execution paths should produce the
    same stacked output (modulo numerical noise) when both are feasible."""

    def test_vmap_and_loop_agree_on_scalar_output(self):
        @workflow_function(vectorize="loop")
        def f_loop(p: NumericRecord) -> float:
            return p["r"] * p["k"]

        @workflow_function(vectorize="jax")
        def f_jax(p: NumericRecord) -> float:
            return p["r"] * p["k"]

        sweep = NumericRecordArray.stack(
            [NumericRecord(r=float(i), k=10.0 * i) for i in range(1, 5)]
        )
        out_loop = f_loop(p=sweep)
        out_jax = f_jax(p=sweep)
        np.testing.assert_allclose(out_loop["result"], out_jax["result"])


# ---------------------------------------------------------------------------
# Provenance — outputs carry a source pointing back to inputs
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    def test_pure_sweep_provenance(self):
        @workflow_function
        def f(p: NumericRecord) -> float:
            return p["x"]

        sweep = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(3)]
        )
        out = f(p=sweep)
        assert out.source is not None
        assert out.source.operation == "workflow.stack"
        assert sweep in out.source.parents
        assert out.source.metadata["n"] == 3
        assert out.source.metadata["k"] == 0
        assert out.source.metadata["func"] == "f"

    def test_nested_provenance_includes_both(self):
        @workflow_function(n_broadcast_samples=10, vectorize="loop")
        def f(p: NumericRecord, noise: float) -> float:
            return p["x"] + noise

        sweep = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(3)]
        )
        noise_dist = Normal(loc=0.0, scale=1.0, name="noise")
        out = f(p=sweep, noise=noise_dist)
        assert out.source.operation == "workflow.nested"
        assert sweep in out.source.parents
        assert noise_dist in out.source.parents
        assert out.source.metadata["k"] == 10


# ---------------------------------------------------------------------------
# Auto-wrap field name
# ---------------------------------------------------------------------------


class TestAutoWrapFieldName:
    def test_scalar_return_uses_result_field(self):
        from probpipe.core._broadcast_distributions import AUTO_WRAP_FIELD

        @workflow_function
        def f(p: NumericRecord) -> float:
            return p["x"] * 2

        sweep = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(3)]
        )
        out = f(p=sweep)
        assert out.fields == (AUTO_WRAP_FIELD,)
        assert AUTO_WRAP_FIELD == "result"  # documents the current default
