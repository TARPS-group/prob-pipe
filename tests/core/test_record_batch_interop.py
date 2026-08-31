"""Tests that the package's consumers of a batched record accept a `RecordBatch`.

A `RecordBatch` is deliberately not a `Record`, so every consumer that recognised
a batch by `isinstance(x, Record)` — or by a `RecordBatch` subclass check, or by
duck-typing on `.fields` — stops recognising one when the batch arrives. Those
gates do not raise; they take the other branch, which is why widening them needs
its own tests rather than the producers' own.

Each test here drives a batch built by hand through one such consumer, so the
gates are pinned before any producer starts returning batches.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EventTemplate,
    Function,
    FunctionBatch,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    OpaqueBatch,
    ProductDistribution,
    Record,
    function,
)
from probpipe.core._batch import _ranks_of
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._record_batch import RecordBatch
from probpipe.core.event_template import NumericEventTemplate

ELEMENT = NumericEventTemplate(a=(), b=(2,))


def _draws(n: int = 4, *, name: str = "draws") -> NumericRecordBatch:
    """*n* draws of a two-field numeric element, over a single ``draw`` level."""
    return NumericRecordBatch(
        name,
        {"a": jnp.arange(n, dtype=float), "b": jnp.ones((n, 2))},
        "draw",
        element_spec=ELEMENT,
        axes_per_level=(1,),
    )


def _one_field(n: int = 4, *, name: str = "draws") -> NumericRecordBatch:
    return NumericRecordBatch(
        name,
        {"x": jnp.arange(n, dtype=float)},
        "draw",
        element_spec=NumericEventTemplate(x=()),
        axes_per_level=(1,),
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
        declared = EventTemplate(a=NumericArraySpec((), dtype=jnp.int32), b=(2,))

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
            "batch",
            {"a": jnp.arange(4.0), "b": jnp.ones(4)},
            "draw",
            element_spec=NumericEventTemplate(a=(), b=()),
            axes_per_level=(1,),
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

    def test_indexing_gathers_the_named_columns_into_a_batch(self):
        """A minibatch of records is a collection of them, so gathering rows
        gives a *batch*. Handing back a plain ``Record`` of gathered columns
        would state the batch's shape as one element's — the false type a
        per-datum transform then reads."""
        from probpipe.inference._minibatch import _index_along_leading

        batch = _draws(5)

        picked = _index_along_leading(batch, jnp.array([0, 2, 4]))

        assert isinstance(picked, RecordBatch)
        assert picked.batch_shape == (3,)
        assert list(picked.event_template.keys()) == ["a", "b"]
        assert np.allclose(picked["a"], jnp.array([0.0, 2.0, 4.0]))
        # The element declaration is the source's, not one re-read off the rows.
        assert picked.element_spec == batch.element_spec


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
                "batch",
                {"y": jnp.ones(4)},
                "draw",
                element_spec=NumericEventTemplate(y=()),
                axes_per_level=(1,),
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
            "batch",
            {"a": jnp.zeros((5, 2))},
            "draw",
            element_spec=NumericEventTemplate(a=()),
            axes_per_level=(2,),
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
            "batch",
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
        np.testing.assert_allclose(out.values, [0.0, 11.0, 22.0])


class TestOpaqueColumnsAreRearrangedRaw:
    """A non-array field presents as its own object batch, so an operation that
    rearranges the *storage* must read the column itself. Reading the presented
    form instead breaks every field that is not an array."""

    @staticmethod
    def _mixed():
        return RecordBatch(
            "batch",
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
            "batch",
            {"a": jnp.arange(3.0), "b": jnp.arange(3.0) * 10},
            "draw",
            element_spec=EventTemplate(a=(), b=()),
        )

        retyped = _copy_result_term(batch, output_template=EventTemplate(b=(), a=()))
        roundtripped = jax.jit(lambda x: x)(retyped)

        np.testing.assert_array_equal(np.asarray(roundtripped["a"]), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(np.asarray(roundtripped["b"]), [0.0, 10.0, 20.0])


class TestATransformCannotAddAnUnnamedLevel:
    def test_an_added_batch_axis_is_refused(self):
        """``vmap`` adds an axis on the way out, and unflattening has no name to
        give the level it would belong to. Taking the stored spec instead would
        return a batch whose own ``batch_shape`` its columns contradict."""

        def body(x):
            return NumericRecordBatch(
                "batch",
                {"s": x + jnp.zeros(2)},
                "inner",
                element_spec=EventTemplate(s=()),
                axes_per_level=(1,),
            )

        with pytest.raises(ValueError, match="An added axis belongs to no level"):
            jax.vmap(body)(jnp.arange(3.0))

    def test_dropping_one_of_two_batch_axes_is_refused_too(self):
        """An added axis is refused because no level names it; a *partially*
        dropped one is refused because no shape says which level went. Removing
        every axis is the case that works, and it yields a ``Record``."""
        batch = NumericRecordBatch(
            "batch",
            {"s": jnp.zeros((3, 2))},
            ("outer", "inner"),
            element_spec=EventTemplate(s=()),
            axes_per_level=(1, 1),
        )

        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.vmap(lambda b: jnp.zeros(()))(batch)

        single = NumericRecordBatch(
            "batch",
            {"s": jnp.zeros(3)},
            "outer",
            element_spec=EventTemplate(s=()),
        )
        seen: list[Any] = []
        jax.vmap(lambda b: seen.append(type(b).__name__) or jnp.zeros(()))(single)
        assert seen == ["NumericRecord"]


class TestBatchFingerprinting:
    """A batch's multiplicity is part of its type, so it is hashed."""

    @staticmethod
    def _one(level: str = "draw", groups=((3,),)):
        return NumericRecordBatch(
            "batch",
            {"x": jnp.arange(3.0)},
            (level,),
            element_spec=EventTemplate(x=()),
            axes_per_level=_ranks_of(groups),
        )

    def test_a_multi_field_batch_fingerprints(self):
        from probpipe.core._fingerprint import fingerprint

        batch = NumericRecordBatch(
            "batch",
            {"a": jnp.arange(3.0), "b": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(a=(), b=()),
        )

        assert isinstance(fingerprint(batch), str)

    def test_level_names_change_the_fingerprint(self):
        from probpipe.core._fingerprint import fingerprint

        assert fingerprint(self._one("draw")) != fingerprint(self._one("chain"))

    def test_a_single_field_batch_is_not_its_column(self):
        from probpipe.core._fingerprint import fingerprint

        assert fingerprint(self._one()) != fingerprint(jnp.arange(3.0))

    def test_equal_multi_field_batches_hash_equal(self):
        from probpipe.core._fingerprint import fingerprint

        def build():
            return NumericRecordBatch(
                "batch",
                {"a": jnp.arange(3.0), "b": jnp.arange(3.0) * 10},
                "draw",
                element_spec=EventTemplate(a=(), b=()),
            )

        assert fingerprint(build()) == fingerprint(build())

    def test_swapping_two_columns_values_changes_the_hash(self):
        from probpipe.core._fingerprint import fingerprint

        ab = NumericRecordBatch(
            "batch",
            {"a": jnp.arange(3.0), "b": jnp.arange(3.0) * 10},
            "draw",
            element_spec=EventTemplate(a=(), b=()),
        )
        ba = NumericRecordBatch(
            "batch",
            {"a": jnp.arange(3.0) * 10, "b": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(a=(), b=()),
        )

        assert fingerprint(ab) != fingerprint(ba)

    def test_axis_grouping_changes_the_fingerprint(self):
        from probpipe.core._fingerprint import fingerprint

        split = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((2, 3))},
            ("a", "b"),
            element_spec=EventTemplate(x=()),
            axes_per_level=(1, 1),
        )
        joined = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((2, 3))},
            "a",
            element_spec=EventTemplate(x=()),
            axes_per_level=(2,),
        )

        assert fingerprint(split) != fingerprint(joined)


class TestMultiLevelSweeps:
    def test_a_two_level_batch_sweeps_its_full_grid(self):
        """The sweep addresses an element by position, one indexer per batch
        axis; a flat index would read the leading axis alone and run off its
        end at the third cell."""
        grid = NumericRecordBatch(
            "batch",
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            axes_per_level=(1, 1),
        )

        @function
        def scale(p):
            return jnp.asarray(p["x"]) * 10.0

        out = scale(p=grid)

        assert out.batch_shape == (2, 3)
        np.testing.assert_allclose(out.values, np.arange(6.0).reshape(2, 3) * 10.0)

    def test_two_groups_and_a_returned_batch_keep_their_partition(self):
        """The aggregate mints one level per swept group, then the rows' own
        levels: collapsing the sweep into one group would leave more names than
        levels and refuse."""
        a = NumericRecordBatch(
            "batch",
            {"a": jnp.arange(2.0)},
            "outer",
            element_spec=EventTemplate(a=()),
        )
        b = NumericRecordBatch(
            "batch",
            {"b": jnp.arange(3.0)},
            "inner",
            element_spec=EventTemplate(b=()),
        )

        @function
        def rows(a, b):
            return NumericRecordBatch(
                "batch",
                {"s": jnp.asarray(a["a"]) + jnp.asarray(b["b"]) + jnp.zeros(2)},
                "rows",
                element_spec=EventTemplate(s=()),
                axes_per_level=(1,),
            )

        out = rows(a=a, b=b)

        assert out.level_names == ("outer", "inner", "rows")
        assert out.axis_groups == ((2,), (3,), (2,))
        expected = (jnp.arange(2.0)[:, None] + jnp.arange(3.0)[None, :])[..., None] + jnp.zeros(2)
        np.testing.assert_allclose(np.asarray(out["s"]), np.asarray(expected))


class TestAutoDispatchFallsBackForABatchReturningBody:
    def test_the_default_dispatch_produces_the_sequential_result(self):
        """The probe traces the vmap the dispatch is choosing, so a body vmap
        cannot run — one returning a batch, whose added axis no level names —
        resolves to sequential instead of failing mid-call. Dispatch paths
        agree on results by contract, so the fallback changes speed alone."""
        source = NumericRecordBatch(
            "batch",
            {"x": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(x=()),
        )

        def body(v):
            return NumericRecordBatch(
                "batch",
                {"s": jnp.asarray(v["x"]) + jnp.zeros(2)},
                "inner",
                element_spec=EventTemplate(s=()),
                axes_per_level=(1,),
            )

        out = Function(func=body)(v=source)

        assert out.level_names == ("draw", "inner")
        assert out.batch_shape == (3, 2)
        # The fallback and an explicit sequential dispatch agree on values —
        # the dispatch-equivalence contract — and both match the independently
        # computed result, so agreement is not two wrongs agreeing.
        explicit = Function(func=body, dispatch="sequential")(v=source)
        np.testing.assert_allclose(np.asarray(out["s"]), np.asarray(explicit["s"]))
        np.testing.assert_allclose(np.asarray(out["s"]), [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    def test_a_numeric_body_is_unaffected(self):
        source = NumericRecordBatch(
            "batch",
            {"x": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(x=()),
        )

        @function
        def double(v):
            return jnp.asarray(v["x"]) * 2.0

        out = double(v=source)

        assert out.batch_shape == (3,)
        np.testing.assert_allclose(out.values, [0.0, 2.0, 4.0])


class TestSameRankTransformsCannotLieEither:
    def test_a_resizing_transform_is_refused(self):
        """Slicing keeps every axis and changes what the level holds. Carrying the
        names onto the new sizes is right for a per-level slice and wrong for
        anything else landing on the same shape, so it is refused; indexing is the
        route that carries its selection instead of inferring it."""
        batch = NumericRecordBatch(
            "batch",
            {"x": jnp.arange(2.0), "y": jnp.arange(2.0) * 10},
            "draw",
            element_spec=EventTemplate(x=(), y=()),
        )

        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.tree.map(lambda leaf: leaf[:1], batch)

        sliced = batch[0:1]
        assert sliced.batch_shape == (1,)
        assert sliced.level_names == ("draw",)
        assert sliced._raw_column("x").shape == (1,)

    def test_columns_that_disagree_are_refused(self):
        import jax.tree_util as jtu

        batch = NumericRecordBatch(
            "batch",
            {"x": jnp.arange(2.0), "y": jnp.arange(2.0)},
            "draw",
            element_spec=EventTemplate(x=(), y=()),
        )
        _, treedef = jtu.tree_flatten(batch)

        with pytest.raises(ValueError, match="disagreeing batch axes"):
            jtu.tree_unflatten(treedef, [jnp.zeros(2), jnp.zeros(3)])


class TestOpaqueBatchesStack:
    def test_rows_holding_opaque_fields_stack_by_raw_column(self):
        """One row's opaque field presents as an OpaqueBatch; stacking the
        presented form hands wrappers to jnp.stack. The columns stack, through
        numpy, so the objects ride as they are."""
        from probpipe.core._broadcast_distributions import _make_stack

        rows = [
            RecordBatch(
                "batch",
                {
                    "tag": np.array([f"{i}a", f"{i}b"], dtype=object),
                    "x": jnp.arange(2.0) + i,
                },
                "inner",
                element_spec=EventTemplate(tag=None, x=()),
            )
            for i in range(3)
        ]

        out = _make_stack(rows, n=3, field_name="demo", level_names=("sweep",))

        assert out.level_names == ("sweep", "inner")
        assert out.batch_shape == (3, 2)
        assert list(out._raw_column("tag")[2]) == ["2a", "2b"]
        np.testing.assert_allclose(np.asarray(out._raw_column("x")[1]), [1.0, 2.0])


class TestObjectValuedMarginals:
    def test_an_object_batch_takes_the_list_marginal(self):
        """The record marginal is empirical over numeric leaves, so an object
        batch routes to the general list marginal: atoms and weights, no
        numeric pretence."""
        from probpipe.core._broadcast_distributions import _ListMarginal, _make_marginal

        batch = RecordBatch(
            "batch",
            {"tag": np.array(["a", "b", "c"], dtype=object), "x": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(tag=None, x=()),
        )

        marginal = _make_marginal(batch)

        assert isinstance(marginal, _ListMarginal)
        assert marginal.num_atoms == 3
        assert [row["tag"] for row in marginal.items] == ["a", "b", "c"]


class TestAnEmptySweepAnswersToItsTemplate:
    def test_zero_rows_still_build_the_declared_fields(self):
        source = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((0,))},
            "design",
            element_spec=EventTemplate(x=()),
        )

        @function(output_template=EventTemplate(y=()))
        def fit(p):
            return {"y": jnp.asarray(p["x"]) * 2.0}

        out = fit(p=source)

        assert list(out.event_template) == ["y"]
        assert out.batch_shape == (0,)
        assert np.asarray(out["y"]).shape == (0,)


class TestATransformCannotResizeTheElement:
    """The batch axes are a transform's to drop or resize; the element's own
    axes are the element type's."""

    @staticmethod
    def _vector_batch():
        return NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((3, 2))},
            "draw",
            element_spec=EventTemplate(x=(2,)),
        )

    def test_slicing_an_event_axis_is_refused(self):
        with pytest.raises(ValueError, match="never the element's own"):
            jax.tree.map(lambda leaf: leaf[:, :1], self._vector_batch())

    def test_transposing_batch_and_event_axes_is_refused(self):
        with pytest.raises(ValueError, match="never the element's own"):
            jax.tree.map(lambda leaf: leaf.T, self._vector_batch())

    def test_reducing_below_the_event_rank_is_refused(self):
        with pytest.raises(ValueError, match="fewer axes"):
            jax.tree.map(jnp.sum, self._vector_batch())


class TestAnEmptySweepIsNotAMissingOutput:
    def test_zero_expected_rows_build_the_declared_fields(self):
        from probpipe.core._broadcast_distributions import _make_stack

        out = _make_stack(
            [],
            batch_shape=(0,),
            field_name="fit",
            level_names=("design",),
            event_template=EventTemplate(y=()),
        )

        assert list(out.event_template) == ["y"]
        assert np.asarray(out["y"]).shape == (0,)

    def test_missing_outputs_are_an_error_not_a_fabrication(self):
        """An empty list where rows were expected reports the count mismatch;
        fabricating the declared fields would hide a swallowed failure."""
        from probpipe.core._broadcast_distributions import _make_stack

        with pytest.raises(ValueError, match="got 0 outputs but expected"):
            _make_stack(
                [],
                n=3,
                field_name="fit",
                level_names=("s",),
                event_template=EventTemplate(y=()),
            )


class TestZeroWidthEventsUnderExplicitJax:
    def test_the_probe_accepts_what_the_executor_runs(self):
        """A (0,)-event column reshapes at the stated flat size; a ``-1`` cannot
        be inferred over zero width, so the probe states the size exactly as the
        executor does."""
        source = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((3, 0)), "row": jnp.arange(3.0)},
            "draw",
            element_spec=EventTemplate(x=(0,), row=()),
        )

        out = Function(
            func=lambda v: jnp.sum(jnp.asarray(v["x"])) + jnp.asarray(v["row"]),
            dispatch="jax",
        )(v=source)

        assert out.batch_shape == (3,)
        # Row identities survive, so the zero-width column really was carried
        # per row rather than the output being indistinguishable zeros.
        np.testing.assert_allclose(out.values, [0.0, 1.0, 2.0])


class TestShapeCannotRecoverAxisProvenance:
    """Which axis a transform took is not readable off sizes alone, so a batch is
    rebuilt only where no axis has to be identified: all of them kept, or all
    removed. A partial reduction is refused whatever the sizes are — distinct
    sizes narrow the candidates without establishing which axis the transform
    actually consumed."""

    @staticmethod
    def _grid(chain: int, draw: int):
        return NumericRecordBatch(
            "batch",
            {"x": jnp.arange(float(chain * draw)).reshape(chain, draw)},
            ("chain", "draw"),
            element_spec=EventTemplate(x=()),
            axes_per_level=(1, 1),
        )

    @pytest.mark.parametrize(("chain", "draw"), [(2, 2), (2, 3)], ids=["equal", "distinct"])
    def test_removing_one_of_two_axes_is_refused(self, chain, draw):
        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.vmap(lambda v: 0.0, in_axes=1)(self._grid(chain, draw))

    def test_an_unequal_permutation_is_refused(self):
        """A transpose changes the batch shape here, so it is caught by the same
        gate a resize is. An *equal*-sized transpose changes nothing about the
        shape and is undetectable — see the batch's own contract tests."""
        with pytest.raises(ValueError, match="keeps every batch axis or removes all of them"):
            jax.tree.map(lambda leaf: leaf.T, self._grid(2, 3))


class TestATransformCannotRetypeTheElement:
    def test_a_changed_kind_is_refused(self):
        batch = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros(3, dtype=jnp.float32)},
            "draw",
            element_spec=EventTemplate(x=NumericArraySpec((), dtype=jnp.float32)),
        )

        with pytest.raises(TypeError, match="does not admit"):
            jax.tree.map(lambda leaf: leaf.astype(jnp.complex64), batch)

    def test_a_same_kind_widening_is_admitted(self):
        """The constructor admits same-kind casts, so the transform guard does
        too — the two must agree on what conforms."""
        batch = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros(3, dtype=jnp.float16)},
            "draw",
            element_spec=EventTemplate(x=NumericArraySpec((), dtype=jnp.float32)),
        )

        widened = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), batch)

        assert widened._raw_column("x").dtype == jnp.float32


class TestZeroRowsAgreeAcrossDispatch:
    def test_every_dispatch_takes_the_same_empty_aggregation(self):
        """Zero rows run nothing, so there is no body to trace and no per-row
        output for the paths to disagree over; the output schema must not
        depend on how the rows would have been executed."""
        source = NumericRecordBatch(
            "batch",
            {"x": jnp.zeros((2, 0))},
            ("a", "b"),
            element_spec=EventTemplate(x=()),
            axes_per_level=(1, 1),
        )

        def body(v):
            return jnp.sum(jnp.asarray(v["x"]))

        results = {
            name: Function(func=body, dispatch=name)(v=source)
            for name in ("auto", "sequential", "jax")
        }

        types = {type(r).__name__ for r in results.values()}
        specs = {repr(r.element_spec) for r in results.values()}
        assert len(types) == 1
        assert len(specs) == 1


class TestFunctionValuedColumnsStack:
    def test_rows_holding_callable_fields_stack_by_raw_column(self):
        """A callable field presents as a FunctionBatch; the raw columns are
        what stack, and the presentation survives the aggregation."""
        from probpipe.core._broadcast_distributions import _make_stack
        from probpipe.core._function_batch import FunctionBatch
        from probpipe.core.event_template import FunctionSpec

        rows = [
            RecordBatch(
                "batch",
                {"f": np.array([(lambda i=i, j=j: i * 10 + j) for j in range(2)], dtype=object)},
                "inner",
                element_spec=EventTemplate(f=FunctionSpec()),
            )
            for i in range(3)
        ]

        out = _make_stack(rows, n=3, field_name="demo", level_names=("sweep",))

        assert out.level_names == ("sweep", "inner")
        assert isinstance(out["f"], FunctionBatch)
        assert out._raw_column("f")[2, 1]() == 21


class TestAnEmpiricalTakesABatch:
    def test_a_batch_routes_to_the_record_empirical(self):
        """An empirical over a batch of records is an empirical over its rows:
        the batch peels to the leaf-rows form the class stores, raw columns and
        all, and resampling keeps rows paired."""
        from probpipe import EmpiricalDistribution, sample
        from probpipe.core._empirical import RecordEmpiricalDistribution

        data = NumericRecordBatch(
            "batch",
            {"X": jnp.arange(4.0), "y": jnp.arange(4.0) * 10},
            "obs",
            element_spec=EventTemplate(X=(), y=()),
        )

        empirical = EmpiricalDistribution(data)

        assert isinstance(empirical, RecordEmpiricalDistribution)
        assert empirical.num_atoms == 4
        drawn = sample(empirical, key=jax.random.PRNGKey(0), sample_shape=(16,))
        stored = {(float(i), float(i * 10)) for i in range(4)}
        seen = set(zip(np.asarray(drawn["X"]).tolist(), np.asarray(drawn["y"]).tolist()))
        assert seen <= stored


class TestBatchValuedRowAggregation:
    """A swept body returns one kind of row, and the aggregate is not its rows' class."""

    @staticmethod
    def _rows(n: int = 3):
        return NumericRecordBatch(
            "batch",
            {"x": jnp.arange(float(n))},
            "row",
            element_spec=EventTemplate(x=NumericArraySpec(shape=())),
        )

    @staticmethod
    def _inner(n: int, level: str = "inner"):
        return NumericRecordBatch(
            "batch",
            {"y": jnp.zeros(n)},
            level,
            element_spec=EventTemplate(y=NumericArraySpec(shape=())),
        )

    def test_mixing_batch_and_non_batch_rows_is_refused(self):
        def body(x):
            return self._inner(2) if float(x["x"]) < 1.5 else Record("r", y=0.0)

        with pytest.raises(TypeError, match="some rows returned a batch and some did not"):
            Function(func=body, dispatch="sequential")(x=self._rows())

    @pytest.mark.parametrize(
        "second",
        [
            pytest.param(lambda s: s._inner(3), id="batch-shape"),
            pytest.param(lambda s: s._inner(2, level="other"), id="level-names"),
        ],
    )
    def test_rows_disagreeing_on_their_multiplicity_are_refused(self, second):
        def body(x):
            return self._inner(2) if float(x["x"]) < 0.5 else second(self)

        with pytest.raises(ValueError, match="returned batches that disagree"):
            Function(func=body, dispatch="sequential")(x=self._rows())

    def test_rows_disagreeing_on_their_element_spec_are_refused(self):
        def body(x):
            if float(x["x"]) < 0.5:
                return self._inner(2)
            return NumericRecordBatch(
                "batch",
                {"z": jnp.zeros(2)},
                "inner",
                element_spec=EventTemplate(z=NumericArraySpec(shape=())),
            )

        with pytest.raises(ValueError, match="returned batches that disagree"):
            Function(func=body, dispatch="sequential")(x=self._rows())

    def test_compatible_batch_rows_stack_with_the_sweep_in_front(self):
        out = Function(func=lambda x: self._inner(2), dispatch="sequential")(x=self._rows())
        assert out.batch_shape == (3, 2)
        assert out.level_names == ("row", "inner")

    @staticmethod
    def _inner_array(n: int, level: str = "inner"):
        return NumericArrayBatch(
            "inner",
            jnp.zeros(n),
            level,
            element_spec=NumericArraySpec(shape=()),
        )

    def test_array_batch_rows_stack_with_the_sweep_in_front(self):
        """A row's own levels survive whichever batch kind it returns.

        Read as event shape instead, the rows' axis would say each cell holds
        one 2-vector where it holds two elements on a level.
        """
        out = Function(func=lambda x: self._inner_array(2), dispatch="sequential")(x=self._rows())

        assert isinstance(out, NumericArrayBatch)
        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "inner"))
        assert tuple(out.element_spec.shape) == ()

    def test_array_batch_rows_stack_from_their_native_store(self):
        """A row's store is in native form, so stacking goes through its backend.

        ``jnp.stack`` sees only what the duck path recognises, and would refuse a
        registered container that converts perfectly well through ``to_jax``.
        """
        import pandas as pd

        from probpipe.core._broadcast_distributions import _make_stack

        rows = [
            NumericArrayBatch(
                "inner",
                pd.Series([1.0 * i, 2.0 * i]),
                "inner",
                element_spec=NumericArraySpec(shape=()),
            )
            for i in range(1, 4)
        ]

        out = _make_stack(rows, n=3, field_name="f", level_names=("sweep",))

        assert (out.batch_shape, out.level_names) == ((3, 2), ("sweep", "inner"))
        np.testing.assert_allclose(np.asarray(out.values)[2], [3.0, 6.0])

    def test_array_batch_rows_disagreeing_on_their_multiplicity_are_refused(self):
        def body(x):
            return self._inner_array(2) if float(x["x"]) < 0.5 else self._inner_array(3)

        with pytest.raises(ValueError, match="returned batches that disagree"):
            Function(func=body, dispatch="sequential")(x=self._rows())

    def test_mixing_array_batch_and_record_batch_rows_is_refused(self):
        """The two kinds hold different things, so there is no one aggregate."""

        def body(x):
            return self._inner_array(2) if float(x["x"]) < 0.5 else self._inner(2)

        with pytest.raises(TypeError, match="one kind for every row"):
            Function(func=body, dispatch="sequential")(x=self._rows())

    def test_a_returned_design_aggregates_as_a_plain_batch(self):
        """The aggregate is not itself a design: a subclass with its own
        constructor cannot be rebuilt from columns."""
        from probpipe.record.design import FullFactorialDesign

        out = Function(func=lambda x: FullFactorialDesign(a=[1.0, 2.0]), dispatch="sequential")(
            x=self._rows()
        )
        assert type(out) is NumericRecordBatch
        assert out.batch_shape == (3, 2)


class TestDeclaredOpaqueOutputAcrossDispatches:
    """A field the declaration calls opaque holds one value per row, whatever
    those values look like — on every dispatch.

    Stacking them numerically instead would turn each row's own axes into batch
    axes the levels never named, and the batch refuses its own output. The
    sequential and JAX paths build the columns by different routes, so the
    equivalence is asserted rather than assumed.
    """

    @pytest.mark.parametrize("dispatch", ["sequential", "jax", "auto"])
    def test_an_opaque_field_of_arrays_is_one_object_per_row(self, dispatch):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", i=jnp.asarray(1)), NumericRecord("row", i=jnp.asarray(2))],
            level_name="row",
        )

        @function(output_template=EventTemplate(y=None), dispatch=dispatch)
        def make_vector(row):
            return {"y": jnp.array([row["i"], row["i"] + 1])}

        result = make_vector(row=rows)

        assert result.batch_shape == (2,)
        assert result.event_template == EventTemplate(y=None)
        column = result._raw_column("y")
        assert column.dtype == object
        assert column.shape == (2,)
        assert [int(v) for v in result[0]["y"]] == [1, 2]
        assert [int(v) for v in result[1]["y"]] == [2, 3]


class TestEveryBatchIsAnOperand:
    """A batch is swept because it holds a multiplicity, not because it holds records.

    The planner recognised only `RecordBatch` and `DistributionArray`, so the
    other batch kinds were handed to a body whole. A body written for one element
    then saw the whole collection, and the levels collapsed into the value's
    shape on the way out.
    """

    @staticmethod
    def _numeric(n: int = 3):
        return NumericArrayBatch(
            "rows",
            jnp.arange(float(n)),
            "row",
            element_spec=NumericArraySpec(shape=()),
        )

    def test_a_numeric_array_batch_reaches_the_body_as_an_element(self):
        seen: list = []

        Function(func=lambda v: (seen.append(v), 0.0)[1], name="f", dispatch="sequential")(
            v=self._numeric()
        )

        assert seen
        assert all(isinstance(row, NumericArray) for row in seen)
        assert not any(isinstance(row, NumericArrayBatch) for row in seen)

    def test_sweeping_a_numeric_array_batch_keeps_its_level(self):
        out = Function(func=lambda v: jnp.asarray(v) * 2.0, name="double", dispatch="sequential")(
            v=self._numeric()
        )

        assert (out.batch_shape, out.level_names) == ((3,), ("row",))

    def test_an_opaque_batch_hands_the_body_its_stored_element(self):
        """`OpaqueBatch` stores rather than materializes, so the body sees the
        caller's own object."""
        seen: list = []

        Function(func=lambda v: (seen.append(v), 0.0)[1], name="f", dispatch="sequential")(
            v=OpaqueBatch(
                "rows",
                ["a", "b"],
                "row",
            )
        )

        assert seen == ["a", "b"]

    def test_a_function_batch_is_swept_too(self):
        out = Function(func=lambda f: float(f()), name="call", dispatch="sequential")(
            f=FunctionBatch(
                "rows",
                [lambda: 1.0, lambda: 2.0],
                "row",
            )
        )

        assert (out.batch_shape, out.level_names) == ((2,), ("row",))
        np.testing.assert_allclose(out.values, [1.0, 2.0])


class TestSweepingASingleStoreBatchAgreesAcrossDispatch:
    """A row that returns a single-store batch keeps its levels, as a record row does."""

    @staticmethod
    def _rows(n: int = 3):
        return NumericArrayBatch(
            "rows",
            jnp.arange(float(n)),
            "row",
            element_spec=NumericArraySpec(shape=()),
        )

    @staticmethod
    def _inner(v):
        return NumericArrayBatch(
            "i",
            jnp.stack([jnp.asarray(v), jnp.asarray(v)]),
            "inner",
            element_spec=NumericArraySpec(shape=()),
        )

    @pytest.mark.parametrize("dispatch", ["auto", "sequential"])
    def test_the_rows_own_level_survives(self, dispatch):
        out = Function(func=self._inner, name="f", dispatch=dispatch)(v=self._rows())

        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "inner"))

    def test_explicit_jax_says_what_it_cannot_do(self):
        """The probe builds one leaf per field, which a single-store batch lacks.

        Declining beats mis-reading it as a law: `auto` sweeps it correctly, and
        an explicit `jax` should not silently differ from that.
        """
        with pytest.raises(TypeError, match="cannot vectorize over NumericArrayBatch"):
            Function(func=self._inner, name="f", dispatch="jax")(v=self._rows())
