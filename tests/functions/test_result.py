"""The row aggregator and the result wrap of the return step (design V.10).

``_make_stack`` aggregates the rows of a sweep, the evaluations of a lift, and
a batch of draws at the kind of the rows, on the levels the caller mints, and
``_coerce_output`` gives a result its label and provenance.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Provenance


class TestMakeStack:
    """Dispatch rules for wrapping n inner-function outputs as a
    shape-(n,) aggregate. Every case is a parameter-sweep-like scenario
    where row identity must survive; there is no marginalisation."""

    def test_the_shape_is_given_exactly_once(self):
        """``batch_shape`` and ``n`` are two spellings of one thing."""
        from probpipe.functions._result import _make_stack

        with pytest.raises(TypeError, match="requires either batch_shape or n"):
            _make_stack([1.0], field_name="demo", level_names=("sweep",))
        with pytest.raises(TypeError, match="batch_shape OR n, not both"):
            _make_stack([1.0], batch_shape=(1,), n=1, field_name="demo", level_names=("sweep",))

    def test_one_level_name_per_group_of_axes(self):
        """A level name that names no group would name nothing."""
        from probpipe.functions._result import _make_stack

        with pytest.raises(ValueError, match="mints one level per group"):
            _make_stack(
                [1.0, 2.0, 3.0, 4.0],
                batch_shape=(2, 2),
                axis_groups=((2,), (2,)),
                field_name="demo",
                level_names=("only_one",),
            )

    def test_list_of_scalars_wraps_as_numeric_array_batch(self):
        """Numeric rows aggregate at their own kind, with no field to address."""
        from probpipe import NumericArrayBatch
        from probpipe.functions._result import _make_stack

        out = _make_stack([1.0, 2.0, 3.0, 4.0], n=4, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericArrayBatch)
        assert (out.batch_shape, out.level_names) == ((4,), ("sweep",))
        assert tuple(out.element_spec.shape) == ()
        np.testing.assert_allclose(out.values, [1.0, 2.0, 3.0, 4.0])

    def test_list_of_arrays_preserves_event_shape(self):
        """The rows' own shape is the element's; only the sweep axis is a level."""
        from probpipe import NumericArrayBatch
        from probpipe.functions._result import _make_stack

        values = [jnp.arange(3.0) + 10.0 * i for i in range(4)]
        out = _make_stack(values, n=4, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericArrayBatch)
        assert out.batch_shape == (4,)
        assert tuple(out.element_spec.shape) == (3,)
        assert out.values.shape == (4, 3)

    def test_list_of_numeric_records_promotes_to_numeric_array(self):
        from probpipe import NumericRecord, NumericRecordBatch
        from probpipe.functions._result import _make_stack

        records = [
            NumericRecord(
                {"a": float(i), "b": float(i) * 2},
                label="nr",
            )
            for i in range(5)
        ]
        out = _make_stack(records, n=5, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericRecordBatch)
        assert out.batch_shape == (5,)
        np.testing.assert_allclose(out["a"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(out["b"], [0, 2, 4, 6, 8])

    def test_list_of_mixed_records_falls_back_to_record_batch(self):
        """Records with a string (non-numeric) leaf can't go through
        ``NumericRecordBatch.stack``. The fallback path builds each
        field independently — numeric leaves via ``jnp.stack``,
        opaque leaves via ``np.asarray(dtype=object)``."""
        from probpipe import NumericRecordBatch, Record, RecordBatch
        from probpipe.functions._result import _make_stack

        records = [
            Record(
                {"a": float(i), "label": f"row{i}"},
                label="r",
            )
            for i in range(3)
        ]
        out = _make_stack(records, n=3, field_name="demo", level_names=("sweep",))
        assert isinstance(out, RecordBatch)
        assert not isinstance(out, NumericRecordBatch)
        np.testing.assert_allclose(out["a"], [0.0, 1.0, 2.0])
        assert [element.value for element in out["label"]] == ["row0", "row1", "row2"]

    def test_bfloat16_field_inferred_numeric_not_opaque(self):
        """The broadcast-template builder shares the numeric-dtype gate, so an
        ml_dtypes (bfloat16) field stacks into a NumericArraySpec column rather than
        being mislabeled opaque."""
        from probpipe import Record, RecordBatch
        from probpipe.core._opaque import OpaqueSpec
        from probpipe.core._specs import NumericArraySpec
        from probpipe.functions._result import _make_stack

        records = [
            Record(
                {"x": jnp.ones(2, dtype=jnp.bfloat16), "label": f"r{i}"},
                label="r",
            )
            for i in range(3)
        ]
        out = _make_stack(records, n=3, field_name="demo", level_names=("sweep",))
        assert isinstance(out, RecordBatch)
        assert out["x"].dtype == jnp.bfloat16
        assert out.event_template["x"] == NumericArraySpec((2,))  # numeric, not None/opaque
        assert out.event_template["label"] == OpaqueSpec(type=str)

    def test_list_of_distributions_gives_distribution_batch(self):
        from probpipe import DistributionBatch, Normal
        from probpipe.functions._result import _make_stack

        comps = [Normal("d", loc=float(i), scale=1.0) for i in range(3)]
        out = _make_stack(comps, n=3, field_name="demo", level_names=("sweep",))
        assert isinstance(out, DistributionBatch)
        assert (out.batch_shape, out.level_names) == ((3,), ("sweep",))
        assert out[0].label == "demo[sweep=0]"
        assert out[0]._tfp_dist is comps[0]._tfp_dist

    def test_distributions_that_declare_different_events_do_not_stack(self):
        from probpipe import Normal
        from probpipe.functions._result import _make_stack

        comps = [Normal("a", loc=0.0, scale=1.0), Normal("b", loc=0.0, scale=1.0)]
        with pytest.raises(TypeError, match="element 1 of the DistributionBatch"):
            _make_stack(comps, n=2, field_name="demo", level_names=("sweep",))

    def test_list_of_record_batches_nests_batch_shape(self):
        """Each inner RecordBatch has its own batch_shape (m,). Stacking
        n of them produces a RecordBatch with batch_shape (n, m)."""
        from probpipe import NumericRecord, NumericRecordBatch
        from probpipe.functions._result import _make_stack

        inner = [
            NumericRecordBatch.stack(
                [
                    NumericRecord(
                        {"x": float(i * 10 + j)},
                        label="nr",
                    )
                    for j in range(4)
                ],
                level_name="draw",
            )
            for i in range(3)
        ]
        out = _make_stack(inner, n=3, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericRecordBatch)
        assert out.batch_shape == (3, 4)
        np.testing.assert_allclose(out["x"][0], [0, 1, 2, 3])
        np.testing.assert_allclose(out["x"][2], [20, 21, 22, 23])

    def test_vmap_ndarray_wraps_as_numeric_array_batch(self):
        """A bare ``jnp.ndarray`` with leading axis n (typical ``jax.vmap``
        output for scalar-returning fns) wraps without unstacking.

        The mapped path agrees with the row-wise one above: same kind, same
        split of the sweep axis from the element's.
        """
        from probpipe import NumericArrayBatch
        from probpipe.functions._result import _make_stack

        arr = jnp.arange(12.0).reshape(4, 3)
        out = _make_stack(arr, n=4, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericArrayBatch)
        assert out.batch_shape == (4,)
        assert tuple(out.element_spec.shape) == (3,)
        assert out.values.shape == (4, 3)

    def test_declared_vmap_array_requires_single_leaf_template(self):
        from probpipe import RecordSpec
        from probpipe.functions._result import _make_stack

        with pytest.raises(ValueError, match=r"bare array.*single-leaf"):
            _make_stack(
                jnp.ones((4, 2)),
                n=4,
                field_name="demo",
                output_template=RecordSpec(left=(2,), right=(2,)),
                level_names=("sweep",),
            )

    def test_declared_vmap_array_preserves_nested_single_leaf_path(self):
        from probpipe import RecordSpec
        from probpipe.functions._result import _make_stack

        values = jnp.arange(8.0).reshape(4, 2)
        template = RecordSpec(stats=RecordSpec(value=(2,)))

        out = _make_stack(
            values,
            n=4,
            field_name="demo",
            output_template=template,
            level_names=("sweep",),
        )

        assert out.event_template == template
        np.testing.assert_allclose(out["stats/value"], values)

    def test_declared_vmap_array_supports_multidimensional_batch_shape(self):
        from probpipe import RecordSpec
        from probpipe.functions._result import _make_stack

        values = jnp.arange(12.0).reshape(6, 2)
        template = RecordSpec(stats=RecordSpec(value=(2,)))

        out = _make_stack(
            values,
            batch_shape=(2, 3),
            field_name="demo",
            output_template=template,
            level_names=("sweep",),
        )

        assert out.batch_shape == (2, 3)
        assert out.event_template == template
        np.testing.assert_allclose(out["stats/value"], values.reshape(2, 3, 2))

    def test_vmap_record_with_batched_leaves_promotes_to_ra(self):
        """``jax.vmap`` of a Record-returning fn produces a Record whose
        leaves are already batched along a leading axis. That's the
        input form for the pytree branch of ``_make_stack``."""
        from probpipe import NumericRecordBatch, Record
        from probpipe.functions._result import _make_stack

        rec = Record(
            {"x": jnp.arange(5.0), "y": jnp.arange(5.0) + 10},
            label="r",
        )
        out = _make_stack(rec, n=5, field_name="demo", level_names=("sweep",))
        assert isinstance(out, NumericRecordBatch)
        assert out.batch_shape == (5,)

    def test_length_mismatch_raises(self):
        from probpipe.functions._result import _make_stack

        with pytest.raises(ValueError, match=r"expected prod\(batch_shape\)=5"):
            _make_stack([1.0, 2.0, 3.0], n=5, field_name="demo", level_names=("sweep",))

    def test_ndarray_leading_axis_mismatch_raises(self):
        from probpipe.functions._result import _make_stack

        with pytest.raises(ValueError, match="expected leading axis"):
            _make_stack(jnp.arange(6.0), n=4, field_name="demo", level_names=("sweep",))


# ===========================================================================
# _coerce_output — attaches provenance to broadcast outputs
# ===========================================================================


class TestCoerceOutput:
    """``_coerce_output`` is the single entry point where broadcast
    outputs pick up their provenance. Non-broadcast values pass through
    unchanged (scalars / ndarrays / callables stay usable in idiomatic
    arithmetic / attribute access)."""

    def test_wrap_mode_with_no_provenance_wraps_scalar(self):
        from probpipe.functions import _result

        out = _result._coerce_output(
            3.14,
            broadcast_mode=_result.BROADCAST_WRAP,
            provenance=None,
            field_name="f",
        )

        np.testing.assert_allclose(float(out), 3.14)
        assert out.provenance is None

    def test_wrap_mode_explodes_nested_dict_return(self):
        # A workflow return of a nested dict denotes tree structure: it wraps
        # into a nested Record (mappings are never leaves), not a TypeError.
        from probpipe import Record
        from probpipe.functions._result import _coerce_output

        out = _coerce_output(
            {"summary": {"mean": 1.0, "count": 2.0}, "x": 3.0},
            broadcast_mode="wrap",
            provenance=None,
            field_name="f",
        )
        assert isinstance(out, Record)
        assert list(out.keys()) == ["summary/mean", "summary/count", "x"]

    def test_stack_mode_attaches_to_record_batch(self):
        from probpipe import NumericRecord, NumericRecordBatch
        from probpipe.functions._result import _coerce_output

        ra = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"x": float(i)},
                    label="nr",
                )
                for i in range(3)
            ],
            level_name="draw",
        )
        assert ra.provenance is None
        prov = Provenance("sweep", parents=())
        out = _coerce_output(ra, broadcast_mode="stack", provenance=prov, field_name="f")
        assert out is not ra
        assert out.label == "f"
        assert out.provenance.operation == "sweep"
        assert ra.provenance is None

    def test_attaches_to_distribution_batch(self):
        from probpipe import DistributionBatch, Normal
        from probpipe.functions._result import _coerce_output, _make_stack

        da = _make_stack(
            [Normal("d", loc=0.0, scale=1.0) for _ in range(3)],
            n=3,
            field_name="demo",
            level_names=("sweep",),
        )
        assert isinstance(da, DistributionBatch)
        assert da.provenance is None
        prov = Provenance("nested", parents=())
        out = _coerce_output(da, broadcast_mode="nested", provenance=prov, field_name="f")
        assert out.label == "f"
        assert out.provenance.operation == "nested"
        assert da.provenance is None

    def test_existing_provenance_is_not_overwritten(self):
        """A result that already carries its own provenance keeps it.

        ``_coerce_output`` copies the term and attaches the call's provenance to
        the copy, so the original's record is untouched."""
        from probpipe import NumericRecord
        from probpipe.functions._result import _coerce_output

        nr = NumericRecord(
            {"x": 1.0},
            label="nr",
        ).with_provenance(Provenance("inner", parents=()))
        # Second set would normally raise RuntimeError; _coerce_output
        # swallows it.
        _coerce_output(
            nr,
            broadcast_mode="stack",
            provenance=Provenance("outer", parents=()),
            field_name="f",
        )
        assert nr.provenance.operation == "inner"
