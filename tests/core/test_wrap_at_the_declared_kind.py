"""The Function output boundary wraps a raw return into its own kind."""

from __future__ import annotations

import jax
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

        assert scaled(jnp.asarray(1.0)).name == "scaled"


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
        inner = Function(func=lambda x: x + 1, name="inner")
        outer = Function(func=lambda: inner, name="outer")

        assert isinstance(outer(), Function)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: NumericArray(jnp.arange(3.0), name="held"),
            lambda: Opaque("held", object()),
            lambda: Record("held", {"x": jnp.asarray(1.0)}, name_is_auto=True),
        ],
    )
    def test_the_rule_is_the_same_for_every_kind(self, make):
        kind = type(make())
        returning = Function(func=make, name="returning")

        assert isinstance(returning(), kind)


class TestTheOperationsReturnTheirDeclaredKind:
    def test_log_prob_is_a_numeric_array(self):
        law = Normal(loc=0.0, scale=1.0, name="x")

        assert isinstance(log_prob(law, 0.0), NumericArray)

    def test_mean_of_a_scalar_law_is_a_numeric_array(self):
        assert isinstance(mean(Normal(loc=2.0, scale=1.0, name="x")), NumericArray)

    def test_a_scalar_draw_is_a_numeric_array(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, name="x"), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArray)

    def test_a_numeric_array_result_still_computes(self):
        """A result computes directly, which is what the array surface is for."""
        law = Normal(loc=0.0, scale=1.0, name="x")

        assert float(log_prob(law, 0.0) * 2) == pytest.approx(
            float(np.asarray(log_prob(law, 0.0))) * 2
        )


class TestASampleShapeGetsADrawLevel:
    """Design V.2: the leading dimensions go on a level named `draw`."""

    def test_no_sample_shape_is_one_value(self):
        drawn = sample(Normal(loc=0.0, scale=1.0, name="x"), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArray)

    @pytest.mark.parametrize("sample_shape", [(5,), (2, 3)])
    def test_draws_land_on_one_draw_level(self, sample_shape):
        from probpipe import NumericArrayBatch

        drawn = sample(
            Normal(loc=0.0, scale=1.0, name="x"),
            sample_shape=sample_shape,
            key=jax.random.PRNGKey(0),
        )

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == sample_shape
        assert drawn.level_names == ("sample",)

    def test_the_event_shape_is_kept_out_of_the_draw_level(self):
        """A vector law draws vectors, so its event axes stay with the element."""
        from probpipe import MultivariateNormal, NumericArrayBatch

        law = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="v")

        drawn = sample(law, sample_shape=(5,), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericArrayBatch)
        assert drawn.batch_shape == (5,)
        assert tuple(drawn.element_spec.shape) == (3,)
        assert drawn.shape == (5, 3)

    def test_an_element_is_one_draw(self):
        drawn = sample(
            Normal(loc=0.0, scale=1.0, name="x"), sample_shape=(5,), key=jax.random.PRNGKey(0)
        )

        assert isinstance(drawn[2], NumericArray)
        assert drawn[2].shape == ()

    def test_a_record_law_still_draws_its_own_batch(self):
        """A record law builds its own batch."""
        from probpipe import NumericRecordBatch, ProductDistribution

        law = ProductDistribution(
            Normal(loc=0.0, scale=1.0, name="a"), Normal(loc=1.0, scale=1.0, name="b")
        )

        drawn = sample(law, sample_shape=(5,), key=jax.random.PRNGKey(0))

        assert isinstance(drawn, NumericRecordBatch)

    def test_a_law_that_does_not_prepend_its_draws_is_left_alone(self):
        """Rather than mis-splitting axes the law never laid out that way."""
        from probpipe.core.ops import _drawn_at_its_batch_form

        # Leading axes that are not the requested sample_shape: nothing here can
        # say which axes are draws, so the value is handed back untouched.
        drawn = _drawn_at_its_batch_form(jnp.zeros((2, 7)), (5,), name="law", name_is_auto=False)

        assert not isinstance(drawn, NumericArray)
        assert drawn.shape == (2, 7)


class TestAnEmptyReturnKeepsItsHostsKind:
    """The kind follows the host's type, and having no entries does not change it.

    A mapping is a tree and a sequence a multiplicity whether or not anything is
    in it. Reading the kind off the *cardinality* instead would give a function
    returning a dict a result type that varies with its data.
    """

    @staticmethod
    def _returned(value):
        return Function(func=lambda: value, name="f")()

    def test_an_empty_mapping_is_an_empty_record(self):
        result = self._returned({})

        assert isinstance(result, Record)
        assert list(result.event_template) == []

    @pytest.mark.parametrize("sequence", [[], ()], ids=["list", "tuple"])
    def test_an_empty_sequence_is_a_batch_of_no_elements(self, sequence):
        """No element to read a kind off, so the batch claims the least it can.

        Every element spec holds vacuously of no elements, which is why the
        opaque one is not a narrowing here.
        """
        from probpipe import OpaqueBatch

        result = self._returned(sequence)

        assert isinstance(result, OpaqueBatch)
        assert (result.batch_shape, result.level_names) == ((0,), ("f",))

    def test_an_empty_array_is_still_an_array(self):
        """Distinct from an empty container: the kind was never in doubt."""
        result = self._returned(jnp.array([]))

        assert isinstance(result, NumericArray)
        assert result.shape == (0,)


class TestASequenceAggregatesAtItsRowsKind:
    """The multiplicity side of the same table: rows batch at their own kind.

    Numeric rows had a batch form and opaque or callable ones did not, so they
    fell to a single-field `RecordBatch` keyed by the function's name — the
    burial this boundary otherwise stopped doing.
    """

    @staticmethod
    def _returned(value):
        return Function(func=lambda: value, name="f")()

    def test_numeric_rows_batch_as_arrays(self):
        from probpipe import NumericArrayBatch

        assert isinstance(self._returned([1.0, 2.0]), NumericArrayBatch)

    def test_opaque_rows_batch_as_opaque(self):
        from probpipe import OpaqueBatch

        result = self._returned(["a", "b"])

        assert isinstance(result, OpaqueBatch)
        assert [result[0], result[1]] == ["a", "b"]

    def test_callable_rows_batch_as_functions(self):
        from probpipe import FunctionBatch

        assert isinstance(self._returned([lambda: 1, lambda: 2]), FunctionBatch)

    def test_every_kind_takes_the_functions_name(self):
        """The last-ditch branch alone used to leave the aggregate auto-named."""
        for value in ([1.0, 2.0], ["a", "b"], [lambda: 1], []):
            assert self._returned(value).name == "f"


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
        from probpipe import EventTemplate, RecordBatch

        with pytest.raises(ValueError, match="at least one field"):
            RecordBatch({}, "x", element_spec=EventTemplate(), name="batch")


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
        from probpipe.core.event_template import NumericEventTemplate

        return NumericRecordBatch(
            {"x": jnp.arange(float(n))},
            "row",
            element_spec=NumericEventTemplate(x=()),
            name="rows",
        )

    def _swept(self, body, dispatch):
        return Function(func=body, name="f", dispatch=dispatch)(v=self._rows())

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

    def test_a_sequence_row_keeps_its_own_level(self, dispatch):
        """The row's multiplicity is a level, not part of the element's shape."""
        out = self._swept(lambda v: [jnp.asarray(v["x"]), jnp.asarray(v["x"])], dispatch)

        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "f"))

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (lambda v: [object(), object()], "OpaqueBatch"),
            (lambda v: [lambda z: z, lambda z: z], "FunctionBatch"),
        ],
        ids=["opaque", "callable"],
    )
    def test_a_row_of_unstackable_elements_keeps_its_level_too(self, dispatch, body, expected):
        """The level does not depend on what the elements are.

        These rows were stored whole: the row's own batch became one opaque
        element of the aggregate, so its multiplicity vanished and a batch of
        callables came back as opaque.
        """
        out = self._swept(body, dispatch)

        assert type(out).__name__ == expected
        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "f"))

    def test_an_empty_sequence_row_counts_zero_on_its_level(self, dispatch):
        """A batch of nothing is still a batch, as it is for a single return."""
        out = self._swept(lambda v: [], dispatch)

        assert (out.batch_shape, out.level_names) == ((3, 0), ("row", "f"))

    def test_two_nested_anonymous_levels_are_refused(self, dispatch):
        """Both would take the function's name, and a clash is not resolved by
        suffixing it."""
        with pytest.raises(ValueError, match="level names must be unique"):
            self._swept(lambda v: [[jnp.asarray(v["x"])], [jnp.asarray(v["x"])]], dispatch)

    def test_a_batch_row_keeps_the_level_it_named(self, dispatch):
        """A row that names its own level keeps that name inside the sweep's."""
        from probpipe import NumericRecordBatch
        from probpipe.core.event_template import NumericEventTemplate

        def body(v):
            x = jnp.asarray(v["x"])
            return NumericRecordBatch(
                {"y": jnp.stack([x, x * 2])},
                "part",
                element_spec=NumericEventTemplate(y=()),
                name="parts",
            )

        out = self._swept(body, dispatch)

        assert (out.batch_shape, out.level_names) == ((3, 2), ("row", "part"))

    def test_rows_of_differing_numeric_shape_are_refused(self, dispatch):
        """An object column would record the disagreement as if it were the answer."""
        with pytest.raises(ValueError, match="differing shapes"):
            self._swept(lambda v: jnp.ones(int(jnp.asarray(v["x"])) + 1), dispatch)

    def test_a_disagreement_inside_a_returned_sequence_is_not_swallowed(self):
        """The stack has a batch form for every element kind, so what raises here
        is the rows disagreeing — which the caller should see."""
        with pytest.raises(ValueError, match="differing shapes"):
            Function(func=lambda: [jnp.ones(1), jnp.ones(2)], name="g")()


class TestASweptEmptyMappingHitsTheSameWall:
    """Per-row wrapping makes a `{}` row a `Record`, so the sweep reaches the
    field guard and says what the direct routes say."""

    @pytest.mark.parametrize("dispatch", ["auto", "sequential"])
    def test_a_swept_body_returning_an_empty_mapping_is_refused(self, dispatch):
        from probpipe import NumericRecordBatch
        from probpipe.core.event_template import NumericEventTemplate

        rows = NumericRecordBatch(
            {"x": jnp.arange(3.0)},
            "row",
            element_spec=NumericEventTemplate(x=()),
            name="rows",
        )

        with pytest.raises(ValueError, match="at least one field"):
            Function(func=lambda v: {}, name="f", dispatch=dispatch)(v=rows)
