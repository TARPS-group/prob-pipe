"""Lifting, step 4 of the stack (design V.5).

The annotation decides what a parameter expects, and each argument that differs
from it is lifted:

- a distribution where a value is expected: a broadcast;
- a batch where one element is expected: a sweep;
- both at once: a nested sweep of broadcasts;
- neither: a plain call.

Lifted arguments group by root ancestor, so members of one group co-sample, and
swept batches align by level name. A parameter annotated with a class of raw
values receives each argument, element, or draw in its raw form when that form
is an instance of the class.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from probpipe import (
    ApplicabilityError,
    Batch,
    Distribution,
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    Record,
    ResolutionError,
    function,
    workflow_run,
)
from probpipe.distributions import DistributionBatch
from probpipe.distributions._capabilities import SupportsLogProb

from ._design_helpers import atom_leaves, error_of, one_field_law, record_law, standard_normal

SCALAR = NumericArraySpec(())


def _batch(values, level: str = "row") -> NumericArrayBatch:
    return NumericArrayBatch("rows", jnp.asarray(values), level, element_spec=SCALAR)


def _kind_of(body_annotation: Any, argument: Any) -> Any:
    """The result of calling a body annotated *body_annotation* on *argument*."""

    def body(x):
        return 0.0

    body.__annotations__ = {"x": body_annotation}
    wrapped = Function("body", body, n_broadcast_samples=6, dispatch="sequential")
    with workflow_run(seed=0):
        return wrapped(argument)


class TestTheTrigger:
    def test_a_distribution_at_an_unannotated_parameter_is_lifted(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert isinstance(result, Distribution)
        assert result.num_atoms == 6

    @pytest.mark.parametrize("annotation", [float, jnp.ndarray])
    def test_a_distribution_at_a_value_annotated_parameter_is_lifted(self, annotation):
        assert isinstance(_kind_of(annotation, standard_normal()), Distribution)

    def test_a_parameter_annotated_any_passes_a_law_whole(self):
        seen = []

        def accept(x: Any):
            seen.append(x)
            return 0.0

        law = standard_normal()
        Function("accept", accept)(law)

        assert seen == [law]

    @pytest.mark.parametrize(
        "annotation",
        [Distribution, Normal, SupportsLogProb, Distribution | None],
        ids=["Distribution", "subclass", "capability", "optional"],
    )
    def test_a_parameter_that_names_a_distribution_consumes_it(self, annotation):
        seen = []

        def consume(x):
            seen.append(x)
            return 0.0

        consume.__annotations__ = {"x": annotation}
        law = standard_normal()
        Function("consume", consume)(law)

        assert seen == [law]

    @pytest.mark.pending(
        reason="a parameter annotated Function expects a value and lifts a law", raises=TypeError
    )
    def test_a_parameter_annotated_function_expects_a_value(self):
        assert isinstance(_kind_of(Function, standard_normal()), Distribution)

    def test_a_batch_where_one_element_is_expected_is_swept(self):
        @function(dispatch="sequential")
        def double(x):
            return 2.0 * x

        result = double(_batch([1.0, 2.0, 3.0]))

        assert isinstance(result, Batch)
        assert result.level_names == ("row",)
        np.testing.assert_allclose(np.asarray(result.values), [2.0, 4.0, 6.0])

    def test_a_parameter_that_names_the_batch_consumes_it(self):
        seen = []

        def consume(x: NumericArrayBatch):
            seen.append(x)
            return 0.0

        rows = _batch([1.0, 2.0])
        Function("consume", consume)(rows)

        assert seen == [rows]

    def test_a_batch_and_a_distribution_give_a_sweep_of_broadcasts(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def shift(a, z):
            return a + z

        with workflow_run(seed=0):
            result = shift(_batch([0.0, 10.0]), standard_normal())

        assert isinstance(result[0], Distribution)
        assert float(np.mean(atom_leaves(result[1])[0])) > 5.0

    def test_a_nested_sweep_returns_a_batch_of_laws(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def shift(a, z):
            return a + z

        with workflow_run(seed=0):
            result = shift(_batch([0.0, 10.0]), standard_normal())

        assert isinstance(result, DistributionBatch)

    def test_neither_gives_a_plain_call(self):
        @function
        def double(x):
            return 2.0 * x

        result = double(jnp.ones(2))

        assert not isinstance(result, (Distribution, Batch))


class TestTheDraw:
    def test_a_record_draw_arrives_as_a_record(self):
        seen = []

        @function(n_broadcast_samples=6, dispatch="sequential")
        def record(theta):
            seen.append(theta)
            return 0.0

        with workflow_run(seed=0):
            record(record_law())

        assert all(isinstance(draw, Record) for draw in seen)

    def test_a_one_field_record_draw_arrives_as_a_record(self):
        seen = []

        @function(n_broadcast_samples=6, dispatch="sequential")
        def record(theta):
            seen.append(theta)
            return 0.0

        with workflow_run(seed=0):
            record(one_field_law())

        assert all(isinstance(draw, Record) for draw in seen)

    def test_explicit_binding_reads_the_receiving_parameter_not_the_component(self):
        seen = []

        @function(n_broadcast_samples=6, dispatch="sequential")
        def weigh(theta):
            seen.append(theta)
            return theta["a"] + theta["b"]

        law = record_law()
        with workflow_run(seed=0):
            result = weigh(theta=law)

        assert "theta" not in law.event_spec.components
        assert isinstance(result, Distribution)
        assert all(set(draw.keys()) == {"a", "b"} for draw in seen)


def _receiving(annotations: dict[str, Any], **controls: Any) -> tuple[Function, list[Any]]:
    """A function of ``x`` annotated by *annotations*, and the list of what its body receives."""
    seen: list[Any] = []

    def body(x):
        seen.append(x)
        return 0.0

    body.__annotations__ = annotations
    return Function("body", body, **controls), seen


class TestARawClassAnnotation:
    def test_a_function_written_for_pandas_runs_on_a_record_field(self):
        pd = pytest.importorskip("pandas")

        def differences(c):
            return jnp.asarray(c.diff().dropna().to_numpy())

        differences.__annotations__ = {"c": pd.Series, "return": jax.Array}
        record = Record("r", c=pd.Series([1.0, 2.0, 4.0]))

        result = Function("differences", differences)(record["c"])

        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0])

    def test_a_series_parameter_receives_the_series_a_record_field_holds(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series([1.0, 2.0, 4.0])
        wrapped, seen = _receiving({"x": pd.Series})

        wrapped(Record("r", c=series)["c"])

        assert len(seen) == 1 and seen[0] is series

    def test_a_data_array_parameter_receives_the_data_array_a_record_field_holds(self):
        xr = pytest.importorskip("xarray")
        data = xr.DataArray(np.arange(3.0), dims=("t",))
        wrapped, seen = _receiving({"x": xr.DataArray})

        wrapped(Record("r", d=data)["d"])

        assert len(seen) == 1 and seen[0] is data

    def test_a_jax_array_parameter_receives_the_array_a_numeric_array_holds(self):
        value = jnp.arange(3.0)
        wrapped, seen = _receiving({"x": jax.Array})

        wrapped(NumericArray("x", value))

        assert len(seen) == 1 and seen[0] is value

    def test_apply_presents_the_raw_form_as_a_call_does(self):
        value = jnp.arange(3.0)
        wrapped, seen = _receiving({"x": jax.Array})

        wrapped.apply(NumericArray("x", value))

        assert len(seen) == 1 and seen[0] is value

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_each_element_of_a_swept_batch_arrives_in_its_raw_form(self, dispatch):
        wrapped, seen = _receiving({"x": jax.Array}, dispatch=dispatch)

        wrapped(_batch([1.0, 2.0, 3.0]))

        assert seen and all(isinstance(element, jax.Array) for element in seen)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_each_draw_of_a_lifted_law_arrives_as_a_jax_array(self, dispatch):
        wrapped, seen = _receiving({"x": jax.Array}, n_broadcast_samples=6, dispatch=dispatch)

        with workflow_run(seed=0):
            wrapped(standard_normal())

        assert seen and all(isinstance(draw, jax.Array) for draw in seen)

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [(jax.Array | None, jnp.arange(3.0)), (npt.NDArray[np.float64], np.arange(3.0))],
        ids=["optional", "parametrized"],
    )
    def test_an_annotation_names_the_class_of_its_arm_or_its_origin(self, annotation, value):
        wrapped, seen = _receiving({"x": annotation})

        wrapped(NumericArray("x", value))

        assert len(seen) == 1 and seen[0] is value

    def test_each_argument_a_variadic_parameter_collects_arrives_in_its_raw_form(self):
        values = (jnp.arange(2.0), jnp.arange(3.0))
        seen: list[Any] = []

        def body(*xs):
            seen.extend(xs)
            return 0.0

        body.__annotations__ = {"xs": jax.Array}
        Function("body", body)(*(NumericArray(f"x{i}", value) for i, value in enumerate(values)))

        assert len(seen) == 2 and all(got is value for got, value in zip(seen, values))

    @pytest.mark.parametrize(
        "annotations",
        [{}, {"x": Any}, {"x": NumericArray}, {"x": object}, {"x": np.ndarray}],
        ids=["unannotated", "Any", "NumericArray", "object", "ndarray"],
    )
    def test_any_other_argument_arrives_as_it_is(self, annotations):
        argument = NumericArray("x", jnp.arange(3.0))
        wrapped, seen = _receiving(annotations)

        wrapped(argument)

        assert len(seen) == 1 and seen[0] is argument


class TestGrouping:
    def test_one_distribution_passed_twice_is_one_group(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def difference(x, y):
            return x - y

        law = standard_normal()
        with workflow_run(seed=0):
            result = difference(law, law)

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_sibling_views_of_one_parent_co_sample(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def residual(a, b):
            return b - 2.0 * a

        law = record_law()
        with workflow_run(seed=0):
            result = residual(law["a"], law["b"])

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_a_parent_and_its_own_view_co_sample(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def residual(record, a):
            return record["a"] - a

        law = record_law()
        with workflow_run(seed=0):
            result = residual(law, law["a"])

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_laws_with_no_common_ancestor_sample_independently(self):
        @function(n_broadcast_samples=16, dispatch="sequential")
        def difference(x, y):
            return x - y

        with workflow_run(seed=0):
            result = difference(standard_normal("x"), standard_normal("y"))

        assert not np.allclose(atom_leaves(result)[0], 0.0)

    def test_a_lifted_law_that_does_not_sample_raises_resolution_error(self):
        class Unsampled(Distribution):
            pass

        @function
        def identity(x):
            return x

        law = Unsampled("bare", SCALAR)

        assert isinstance(error_of(lambda: identity(law)), ResolutionError)


class TestAlignment:
    def test_batches_on_one_level_zip(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = add(_batch([1.0, 2.0]), _batch([10.0, 20.0]))

        assert result.batch_shape == (2,)
        np.testing.assert_allclose(np.asarray(result.values), [11.0, 22.0])

    def test_batches_on_different_levels_form_a_product(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = add(_batch([1.0, 2.0], "left"), _batch([10.0, 20.0, 30.0], "right"))

        assert result.batch_shape == (2, 3)
        assert result.level_names == ("left", "right")

    def test_misaligned_levels_raise_applicability_error_naming_the_level(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        error = error_of(lambda: add(_batch([1.0, 2.0]), _batch([1.0, 2.0, 3.0])))

        assert isinstance(error, ApplicabilityError)
        assert "row" in str(error)

    @pytest.mark.pending(reason="a level of size one broadcasts", raises=AssertionError)
    def test_a_level_of_size_one_broadcasts(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = error_of(lambda: add(_batch([1.0]), _batch([10.0, 20.0])))

        assert result is None
