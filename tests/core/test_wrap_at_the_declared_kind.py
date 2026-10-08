"""The Function output boundary wraps a raw return into its own kind."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    NumericArray,
    Opaque,
    Record,
    function,
    log_prob,
    mean,
    sample,
)


class TestARawReturnWrapsIntoItsOwnKind:
    """No kind is presented by wrapping it in another."""

    def test_an_array_return_is_a_numeric_array(self):
        @function
        def total(x):
            return x * 2

        result = total(jnp.arange(3.0))

        assert isinstance(result, NumericArray)
        np.testing.assert_array_equal(np.asarray(result), np.arange(3.0) * 2)

    def test_a_scalar_return_is_a_numeric_array(self):
        @function
        def half(x):
            return x / 2

        assert float(half(5.0)) == 2.5

    def test_a_mapping_return_is_a_record(self):
        @function
        def both(x):
            return {"lo": x - 1, "hi": x + 1}

        result = both(jnp.asarray(1.0))

        assert isinstance(result, Record)
        assert result.fields == ("lo", "hi")

    def test_a_callable_return_is_a_function(self):
        """Callable because it is the function kind."""

        @function
        def make_adder(k):
            return lambda x: x + k

        adder = make_adder(jnp.asarray(2.0))

        assert isinstance(adder, Function)
        assert float(adder(jnp.asarray(1.0))) == 3.0

    def test_any_other_return_is_opaque(self):
        @function
        def label(x):
            return f"value-{int(x)}"

        result = label(jnp.asarray(2))

        assert isinstance(result, Opaque)
        assert result.value == "value-2"

    def test_the_result_is_named_after_the_function(self):
        @function
        def scaled(x):
            return x * 3

        assert scaled(jnp.asarray(1.0)).label == "scaled"


class TestTheKindsAreOrderedNotDisjoint:
    def test_a_callable_takes_the_function_kind_before_the_opaque_fallback(self):
        """A callable is also a non-mapping value, and the specific rule wins."""

        @function
        def make(x):
            del x
            return lambda: 1

        assert not isinstance(make(jnp.asarray(0.0)), Opaque)

    def test_every_tracked_term_keeps_its_kind(self):
        """Whatever the kind, a term keeps it."""
        inner = Function(fn=lambda x: x + 1, label="inner")
        outer = Function(fn=lambda: inner, label="outer")

        assert isinstance(outer(), Function)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: NumericArray(
                "held",
                jnp.arange(3.0),
            ),
            lambda: Opaque("held", object()),
            lambda: Record("held", {"x": jnp.asarray(1.0)}),
        ],
    )
    def test_the_rule_is_the_same_for_every_kind(self, make):
        kind = type(make())
        returning = Function(fn=make, label="returning")

        assert isinstance(returning(), kind)


class TestTheOperationsReturnTheirDeclaredKind:
    def test_log_prob_is_a_numeric_array(self):
        law = Normal(loc=0.0, scale=1.0, label="x")

        assert isinstance(log_prob(law, 0.0), NumericArray)

    def test_mean_of_a_scalar_law_is_a_numeric_array(self):
        assert isinstance(mean(Normal(loc=2.0, scale=1.0, label="x")), NumericArray)

    def test_a_scalar_draw_is_a_numeric_array(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, label="x"))

        assert isinstance(drawn, NumericArray)

    def test_a_numeric_array_result_still_computes(self):
        """A result computes directly, which is what the array surface is for."""
        law = Normal(loc=0.0, scale=1.0, label="x")

        assert float(log_prob(law, 0.0) * 2) == pytest.approx(
            float(np.asarray(log_prob(law, 0.0))) * 2
        )


class TestASampleShapeGetsADrawLevel:
    """Design V.2: the leading dimensions go on a level named `draw`."""

    def test_no_sample_shape_is_one_value(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, label="x"))

        assert isinstance(drawn, NumericArray)

    @pytest.mark.parametrize("sample_shape", [(5,), (2, 3)])
    def test_draws_land_on_one_draw_level(self, sample_shape):
        from probpipe import NumericArrayBatch

        drawn = sample(Normal(loc=0.0, scale=1.0, label="x"), sample_shape=sample_shape)

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == sample_shape
        assert drawn.level_names == ("sample",)

    def test_the_event_shape_is_kept_out_of_the_draw_level(self):
        """A vector law draws vectors, so its event axes stay with the element."""
        from probpipe import MultivariateNormal, NumericArrayBatch

        law = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="v")

        drawn = sample(law, sample_shape=(5,))

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == (5,)
        assert tuple(drawn.element_spec.shape) == (3,)
        assert drawn.shape == (5, 3)

    def test_an_element_is_one_draw(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, label="x"), sample_shape=(5,))

        assert isinstance(drawn[2], NumericArray)
        assert drawn[2].shape == ()

    def test_a_joint_draws_a_batch_of_records_under_its_declaration(self):
        """A law whose draws are a mapping of columns draws the batch its declaration names."""
        from probpipe import NumericRecordBatch

        law = Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b")

        drawn = sample(law, sample_shape=(4,))

        assert isinstance(drawn, NumericRecordBatch)
        assert (drawn.batch_shape, drawn.level_names) == ((4,), ("sample",))
        assert drawn.element_spec == law.event_spec.spec

    def test_one_draw_of_a_joint_is_a_record_under_its_declaration(self):
        law = Normal(loc=0.0, scale=1.0, label="a") * Normal(loc=0.0, scale=1.0, label="b")

        drawn = sample(law)

        assert isinstance(drawn, Record)
        assert drawn.spec == law.event_spec.spec


class TestAnEmptyReturnKeepsItsHostsKind:
    """The kind follows the host's type, and having no entries does not change it.

    A mapping is a tree and a sequence an opaque value whether or not anything is
    in it. Reading the kind off the *cardinality* instead would give a function
    returning a dict a result type that varies with its data.
    """

    @staticmethod
    def _returned(value):
        return Function(fn=lambda: value, label="f")()

    def test_an_empty_mapping_is_an_empty_record(self):
        result = self._returned({})

        assert isinstance(result, Record)
        assert list(result.event_template) == []

    @pytest.mark.parametrize("sequence", [[], (), set()], ids=["list", "tuple", "set"])
    def test_an_empty_sequence_is_opaque(self, sequence):
        result = self._returned(sequence)

        assert isinstance(result, Opaque)
        assert result.value == sequence

    def test_an_empty_array_is_still_an_array(self):
        """Distinct from an empty container: the kind was never in doubt."""
        result = self._returned(jnp.array([]))

        assert isinstance(result, NumericArray)
        assert result.shape == (0,)


class TestAReturnedSequenceIsOpaque:
    """A list, a tuple, or a set is one Opaque, and a batch is declared through
    ``output_spec``."""

    @staticmethod
    def _returned(value, **declaration):
        return Function(fn=lambda: value, label="f", **declaration)()

    @pytest.mark.parametrize(
        "value",
        [[1.0, 2.0], ("a", "b"), [lambda: 1, lambda: 2], {1, 2}],
        ids=["numeric", "strings", "callables", "set"],
    )
    def test_a_sequence_is_one_opaque_under_the_functions_name(self, value):
        result = self._returned(value)

        assert isinstance(result, Opaque)
        assert result.label == "f"
        assert result.value == value

    def test_a_declared_batch_takes_the_sequence_as_its_elements(self):
        from probpipe import BatchSpec, NumericArrayBatch, NumericArraySpec

        declared = BatchSpec(NumericArraySpec(()), item="n")
        result = self._returned([1.0, 2.0, 3.0], output_spec=declared)

        assert isinstance(result, NumericArrayBatch)
        assert (result.batch_shape, result.level_names) == ((3,), ("item",))
        np.testing.assert_array_equal(np.asarray(result.values), [1.0, 2.0, 3.0])

    def test_a_declared_batch_of_opaque_elements_stores_each_one(self):
        from probpipe import BatchSpec, OpaqueBatch, OpaqueSpec

        declared = BatchSpec(OpaqueSpec(), item=2)
        result = self._returned(["a", "b"], output_spec=declared)

        assert isinstance(result, OpaqueBatch)
        assert [result[0].value, result[1].value] == ["a", "b"]


class TestAnEmptyRecordHasNoBatch:
    """An empty record is legal; a batch of them is not, and that is not an accident.

    A batch derives its `batch_shape` from a column, and a zero-field element
    supplies none — there is nothing to read the multiplicity from. Representing
    one would need a second source of truth for the shape, so the refusal stands
    and is stated here rather than left to be discovered.
    """

    def test_an_empty_record_is_legal(self):
        assert list(Record("r").event_template) == []

    def test_stacking_empty_records_is_refused(self):
        from probpipe import RecordBatch

        with pytest.raises(ValueError, match="at least one field"):
            RecordBatch.stack([Record("r"), Record("r")], level_name="x")

    def test_a_zero_column_batch_is_refused(self):
        from probpipe import RecordBatch, RecordSpec

        with pytest.raises(ValueError, match="at least one field"):
            RecordBatch(
                "batch",
                {},
                "x",
                element_spec=RecordSpec(),
            )


class TestEachSweptRowTakesItsOwnKind:
    """A row is one call's return, so the rule that names a single return's kind
    is the rule that names a row's.

    Every case runs under both dispatches. Which executor a sweep picks is a
    performance decision, so a row's kind cannot depend on it: the mapped path
    reads its rows through the same rule the row-wise path does, and the two
    agree here rather than in prose.
    """

    @pytest.fixture(params=["auto", "sequential"])
    def dispatch(self, request):
        return request.param

    @staticmethod
    def _rows(n: int = 3):
        from probpipe import NumericRecordBatch
        from probpipe.core._specs import NumericRecordSpec

        return NumericRecordBatch(
            "rows",
            {"x": jnp.arange(float(n))},
            "row",
            element_spec=NumericRecordSpec(x=()),
        )

    def _swept(self, body, dispatch):
        return Function(fn=body, label="f", dispatch=dispatch)(v=self._rows())

    def test_a_mapping_row_gives_a_batch_of_records(self, dispatch):
        out = self._swept(lambda v: {"y": jnp.asarray(v["x"]) * 2}, dispatch)

        assert list(out.event_template) == ["y"]
        assert (out.batch_shape, out.level_names) == ((3,), ("row",))
        np.testing.assert_array_equal(np.asarray(out["y"]), np.arange(3.0) * 2)

    def test_a_nested_mapping_row_keeps_its_subtree(self, dispatch):
        out = self._swept(
            lambda v: {"lo": jnp.asarray(v["x"]) - 1, "grp": {"hi": jnp.asarray(v["x"]) + 1}},
            dispatch,
        )

        assert list(out.event_template) == ["lo", "grp/hi"]
        np.testing.assert_array_equal(np.asarray(out["grp/hi"]), np.arange(3.0) + 1)

    def test_any_mapping_counts_not_only_dict(self, dispatch):
        from collections import OrderedDict

        out = self._swept(lambda v: OrderedDict(y=jnp.asarray(v["x"])), dispatch)

        assert list(out.event_template) == ["y"]

    @pytest.mark.parametrize(
        "body",
        [
            lambda v: [jnp.asarray(v["x"]), jnp.asarray(v["x"])],
            lambda v: [object(), object()],
            lambda v: (lambda z: z, lambda z: z),
            lambda v: [],
            lambda v: [[jnp.asarray(v["x"])], [jnp.asarray(v["x"])]],
        ],
        ids=["arrays", "objects", "callables", "empty", "nested"],
    )
    def test_a_sequence_row_is_one_opaque_element(self, dispatch, body):
        """A returned sequence is an Opaque, so each row stores one element and
        the sweep's level is the only one."""
        from probpipe import OpaqueBatch

        out = self._swept(body, dispatch)

        assert isinstance(out, OpaqueBatch)
        assert (out.batch_shape, out.level_names) == ((3,), ("row",))

    def test_a_batch_row_keeps_the_level_it_named(self, dispatch):
        """A row that names its own level keeps that name inside the sweep's."""
        from probpipe import NumericRecordBatch
        from probpipe.core._specs import NumericRecordSpec

        def body(v):
            x = jnp.asarray(v["x"])
            return NumericRecordBatch(
                "parts",
                {"y": jnp.stack([x, x * 2])},
                "part",
                element_spec=NumericRecordSpec(y=()),
            )

        out = self._swept(body, dispatch)

        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "part"))

    def test_rows_of_differing_numeric_shape_are_refused(self, dispatch):
        """An object column would record the disagreement as if it were the answer."""
        with pytest.raises(ValueError, match="differing shapes"):
            self._swept(lambda v: jnp.ones(int(jnp.asarray(v["x"])) + 1), dispatch)


class TestASweptEmptyMappingHitsTheSameWall:
    """Per-row wrapping makes a `{}` row a `Record`, so the sweep reaches the
    field guard and says what the direct routes say."""

    @pytest.mark.parametrize("dispatch", ["auto", "sequential"])
    def test_a_swept_body_returning_an_empty_mapping_is_refused(self, dispatch):
        from probpipe import NumericRecordBatch
        from probpipe.core._specs import NumericRecordSpec

        rows = NumericRecordBatch(
            "rows",
            {"x": jnp.arange(3.0)},
            "row",
            element_spec=NumericRecordSpec(x=()),
        )

        with pytest.raises(ValueError, match="at least one field"):
            Function(fn=lambda v: {}, label="f", dispatch=dispatch)(v=rows)
