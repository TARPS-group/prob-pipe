"""Tests that the package's consumers of a batched record accept a `RecordBatch`.

A `RecordBatch` is deliberately not a `Record`, so every consumer that recognised
a batch by `isinstance(x, Record)` — or by a `RecordArray` subclass check, or by
duck-typing on `.fields` — stops recognising one when the batch arrives. Those
gates do not raise; they take the other branch, which is why widening them needs
its own tests rather than the producers' own.

Each test here drives a batch built by hand through one such consumer, so the
gates are pinned before any producer starts returning batches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ArraySpec,
    EventTemplate,
    Function,
    Normal,
    NumericRecord,
    OpaqueBatch,
    ProductDistribution,
    Record,
    function,
)
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._record_batch import RecordBatch
from probpipe.core.event_template import NumericEventTemplate

ELEMENT = NumericEventTemplate(a=(), b=(2,))


def _draws(n: int = 4, *, name: str = "draws") -> NumericRecordBatch:
    """*n* draws of a two-field numeric element, over a single ``draw`` level."""
    return NumericRecordBatch(
        {"a": jnp.arange(n, dtype=float), "b": jnp.ones((n, 2))},
        "draw",
        element_spec=ELEMENT,
        axis_groups=((n,),),
        name=name,
    )


def _one_field(n: int = 4, *, name: str = "draws") -> NumericRecordBatch:
    return NumericRecordBatch(
        {"x": jnp.arange(n, dtype=float)},
        "draw",
        element_spec=NumericEventTemplate(x=()),
        axis_groups=((n,),),
        name=name,
    )


class TestFunctionBoundary:
    """The wrap boundary keeps a returned batch as the batch it is."""

    def test_a_returned_batch_is_not_rewrapped(self):
        batch = _draws()

        f = Function(func=lambda: batch)

        result = f()

        assert isinstance(result, NumericRecordBatch)
        assert result.batch_shape == (4,)
        assert result.level_names == ("draw",)
        assert list(result.event_template.keys()) == ["a", "b"]

    def test_a_returned_batch_becomes_an_independent_result(self):
        batch = _draws()

        f = Function(func=lambda: batch)

        result = f()

        assert result is not batch
        assert result.provenance is not None
        assert batch.provenance is None

    def test_a_declared_output_template_retypes_a_returned_batch(self):
        batch = _draws()

        f = Function(func=lambda: batch, output_template=EventTemplate(a=(), b=(2,)))

        result = f()

        assert isinstance(result, NumericRecordBatch)
        assert result.element_spec.event_template == batch.event_template

    def test_a_declared_output_template_checks_a_batch_column(self):
        batch = _draws()
        declared = EventTemplate(a=ArraySpec((), dtype=jnp.int32), b=(2,))

        f = Function(func=lambda: batch, output_template=declared)

        with pytest.raises(ValueError, match="does not conform to"):
            f()


class TestBroadcastPlanning:
    """A batch passed as a workflow input broadcasts over its rows."""

    def test_a_batch_argument_sweeps(self):
        double = Function(func=lambda v: 2.0 * v["x"])

        result = double(_one_field(3))

        assert result.batch_shape == (3,)

    def test_a_batch_argument_reaches_the_body_as_an_element(self):
        seen: list = []

        def note_row(v):
            seen.append(v)
            return 0.0

        Function(func=note_row)(_one_field(3))

        # The sweep traces the body rather than running it per row, so what
        # matters is the *kind* it is handed: an element, never the batch.
        assert seen
        assert all(isinstance(row, Record) for row in seen)
        assert not any(isinstance(row, RecordBatch) for row in seen)


class TestFieldExtraction:
    """A field view reads its column out of a batch."""

    def test_a_field_view_extracts_its_column_from_a_batch(self):
        joint = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Normal(loc=0.0, scale=1.0, name="b"),
            name="joint",
        )
        batch = NumericRecordBatch(
            {"a": jnp.arange(4.0), "b": jnp.ones(4)},
            "draw",
            element_spec=NumericEventTemplate(a=(), b=()),
            axis_groups=((4,),),
        )

        assert np.allclose(joint["a"]._extract(batch), batch["a"])


class TestFlatVectorBoundary:
    """The distribution-level flatten accepts a batch."""

    def test_flatten_value_ravels_a_batch(self):
        from probpipe.core._numeric_record_distribution import NumericRecordDistribution

        batch = _draws(3)

        flat = NumericRecordDistribution.flatten_value(batch)

        assert np.allclose(flat, batch.to_vector())


class TestMinibatching:
    """Minibatching reads a batch's row count and gathers its rows."""

    def test_data_size_reads_the_leading_axis(self):
        from probpipe.inference._minibatch import _data_size

        assert _data_size(_draws(7)) == 7

    def test_indexing_gathers_the_named_columns(self):
        from probpipe.inference._minibatch import _index_along_leading

        batch = _draws(5)

        picked = _index_along_leading(batch, jnp.array([0, 2, 4]))

        assert isinstance(picked, Record)
        assert list(picked.event_template.keys()) == ["a", "b"]
        assert np.allclose(picked["a"], jnp.array([0.0, 2.0, 4.0]))


class TestDesignCoercion:
    """A GLM design coerces a batch the way it coerces a record."""

    def test_a_single_field_batch_coerces_to_its_column(self):
        from probpipe.modeling._glm import _coerce_array

        batch = _one_field(4)

        assert np.allclose(_coerce_array(batch), batch["x"])

    def test_a_multi_field_batch_stacks_its_columns(self):
        from probpipe.modeling._glm import _coerce_array

        batch = _one_field(4).merge(
            NumericRecordBatch(
                {"y": jnp.ones(4)},
                "draw",
                element_spec=NumericEventTemplate(y=()),
                axis_groups=((4,),),
            )
        )

        assert _coerce_array(batch).shape == (4, 2)


class TestBroadcastComponents:
    """The broadcast helpers gather and unwrap a batch's rows."""

    def test_taking_rows_keeps_the_batch_and_its_levels(self):
        from probpipe.core._broadcast_distributions import _take_rows

        batch = _draws(5)

        taken = _take_rows(batch, jnp.array([1, 3]))

        assert isinstance(taken, NumericRecordBatch)
        assert taken.batch_shape == (2,)
        assert taken.level_names == ("draw",)
        assert np.allclose(taken["a"], jnp.array([1.0, 3.0]))

    def test_taking_rows_keeps_a_trailing_axis_of_the_same_level(self):
        from probpipe.core._broadcast_distributions import _take_rows

        batch = NumericRecordBatch(
            {"a": jnp.zeros((5, 2))},
            "draw",
            element_spec=NumericEventTemplate(a=()),
            axis_groups=((5, 2),),
        )

        taken = _take_rows(batch, jnp.array([1, 3]))

        assert taken.batch_shape == (2, 2)
        assert taken.level_names == ("draw",)

    def test_one_row_of_a_batch_is_a_record(self):
        from probpipe.core._broadcast_distributions import _one_row

        row = _one_row(_draws(5))

        assert isinstance(row, NumericRecord)
        assert not isinstance(row, RecordBatch)

    def test_the_row_count_of_a_batch_reads_its_batch_shape(self):
        from probpipe.core._broadcast_distributions import _row_count

        assert _row_count(_draws(6)) == 6

    def test_a_batch_marginal_peels_the_rows_axis(self):
        from probpipe.core._broadcast_distributions import _RecordMarginal

        batch = _draws(4)

        marginal = _RecordMarginal(batch, None)

        assert marginal.event_template == batch.event_template
        assert marginal.num_atoms == 4


class TestDistributionBroadcastIndexing:
    """Indexing one row of a batched per-argument sample gives a record."""

    def test_indexing_a_batch_sample_gives_one_record(self):
        from probpipe.core._workflow_distribution_broadcast import _index_sample

        row = _index_sample(_draws(4), 2)

        assert isinstance(row, Record)
        assert np.allclose(row["a"], 2.0)

    def test_indexing_a_single_field_batch_sample_gives_its_scalar(self):
        from probpipe.core._workflow_distribution_broadcast import _index_sample

        assert np.allclose(_index_sample(_one_field(4), 3), 3.0)


class TestDiagnosticsBridge:
    """The ArviZ bridge reads a batch of draws by its schema, not by ``.fields``."""

    def test_draws_returning_a_batch_yields_one_variable_per_column(self):
        from probpipe.diagnostics._arviz_bridge import extract_draws

        batch = _draws(4)

        class Posterior:
            def draws(self):
                return batch

        extracted = extract_draws(Posterior())

        assert sorted(extracted) == ["a", "b"]
        assert extracted["a"].shape == (4,)
        assert extracted["b"].shape == (4, 2)


class TestSiblingViewsZipThroughACall:
    def test_select_all_views_sweep_zipped_not_producted(self):
        """Two views of one batch are two readings of one multiplicity — they
        share its level — so a call over both zips rows rather than forming the
        (n, n) product a per-object grouping would."""
        batch = NumericRecordBatch(
            {"x": jnp.arange(3.0), "y": jnp.arange(3.0) * 10},
            "draw",
            element_spec=EventTemplate(x=(), y=()),
        )
        views = batch.select_all()

        @function
        def add(x, y):
            return jnp.asarray(x["x"]) + jnp.asarray(y["y"])

        out = add(x=views["x"], y=views["y"])

        assert out.batch_shape == (3,)
        np.testing.assert_allclose(np.asarray(out["add"]), [0.0, 11.0, 22.0])


class TestOpaqueColumnsAreRearrangedRaw:
    """A non-array field presents as its own object batch, so an operation that
    rearranges the *storage* must read the column itself. Reading the presented
    form instead breaks every field that is not an array."""

    @staticmethod
    def _mixed():
        return RecordBatch(
            {
                "tag": np.array(["a", "b", "c"], dtype=object),
                "x": jnp.arange(3.0),
            },
            "draw",
            element_spec=EventTemplate(tag=None, x=()),
        )

    def test_presented_and_raw_columns_differ_for_an_opaque_field(self):
        batch = self._mixed()

        assert isinstance(batch["tag"], OpaqueBatch)
        assert isinstance(batch._raw_column("tag"), np.ndarray)
        # An array field is its column either way.
        assert batch._raw_column("x") is batch["x"]

    def test_gathering_rows_keeps_an_opaque_column(self):
        from probpipe.core._broadcast_distributions import _take_rows

        gathered = _take_rows(self._mixed(), jnp.array([2, 0]))

        assert list(gathered._raw_column("tag")) == ["c", "a"]
        np.testing.assert_array_equal(np.asarray(gathered._raw_column("x")), [2.0, 0.0])

    def test_indexing_a_minibatch_keeps_an_opaque_column(self):
        from probpipe.inference._minibatch import _index_along_leading

        indexed = _index_along_leading(self._mixed(), jnp.array([1, 2]))

        assert list(indexed["tag"]) == ["b", "c"]
        np.testing.assert_array_equal(np.asarray(indexed["x"]), [1.0, 2.0])


class TestRetypingADeclaredOutputKeepsColumnsWithTheirKeys:
    def test_a_reordered_declaration_does_not_swap_columns(self):
        """A batch flattens in its spec's leaf order, so retyping the spec alone
        would pair every value with the wrong key on the next unflatten."""
        from probpipe.core._workflow_result import _copy_result_term

        batch = NumericRecordBatch(
            {"a": jnp.arange(3.0), "b": jnp.arange(3.0) * 10},
            "draw",
            element_spec=EventTemplate(a=(), b=()),
        )

        retyped = _copy_result_term(batch, output_template=EventTemplate(b=(), a=()))
        roundtripped = jax.jit(lambda x: x)(retyped)

        np.testing.assert_array_equal(np.asarray(roundtripped["a"]), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(np.asarray(roundtripped["b"]), [0.0, 10.0, 20.0])
