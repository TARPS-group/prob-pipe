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

import functools
from typing import TYPE_CHECKING, Annotated, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from probpipe import (
    ApplicabilityError,
    Batch,
    Distribution,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    OutputSpec,
    Record,
    RecordBatch,
    ResolutionError,
    function,
    mean,
    workflow_run,
)
from probpipe.distributions import DistributionBatch
from probpipe.distributions._capabilities import SupportsLogProb

from ._design_helpers import atom_leaves, error_of, one_field_law, record_law, standard_normal

if TYPE_CHECKING:
    from pandas import DataFrame

SCALAR = NumericArraySpec(())
# NumPy 2.5 defines NDArray as a type alias, and _ArrayAlias stands in for it
# because CI does not install NumPy 2.5.
type _ArrayAlias[Scalar: np.generic] = np.ndarray[Any, np.dtype[Scalar]]
type _OptionalArrayAlias = np.ndarray | None
type _DistributionAlias[Event] = Distribution
type _OptionalNormalAlias = Normal | None
type _CapabilityAlias = SupportsLogProb
type _BatchAlias = NumericArrayBatch
type _UnresolvedAlias = DataFrame
type _RecursiveAlias = _RecursiveAlias | None
type _AnyAlias = Any
type _MaybeAlias[T] = T | None
type _OptionalOfAlias[T] = _MaybeAlias[T]
type _SecondOfAlias[First, Second] = Second | None


def _batch(values, level: str = "row") -> NumericArrayBatch:
    return NumericArrayBatch(
        jnp.asarray(values),
        level,
        element_spec=SCALAR,
        label="rows",
    )


def _kind_of(body_annotation: Any, argument: Any) -> Any:
    """The result of calling a body annotated *body_annotation* on *argument*."""

    def body(x):
        return 0.0

    body.__annotations__ = {"x": body_annotation}
    wrapped = Function(
        body,
        max_evaluations=6,
        dispatch="sequential",
        label="body",
        output_spec=OutputSpec(body=None),
    )
    with workflow_run(seed=0):
        return wrapped(argument)


class TestTheTrigger:
    def test_a_distribution_at_an_unannotated_parameter_is_lifted(self):
        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(square=None))
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert isinstance(result, Distribution)
        assert result.num_atoms == 6

    @pytest.mark.parametrize("annotation", [float, jnp.ndarray])
    def test_a_distribution_at_a_value_annotated_parameter_is_lifted(self, annotation):
        assert isinstance(_kind_of(annotation, standard_normal()), Distribution)

    @pytest.mark.parametrize("annotation", [Any, _AnyAlias], ids=["Any", "alias"])
    def test_a_parameter_annotated_any_passes_a_law_whole(self, annotation):
        seen = []

        def accept(x):
            seen.append(x)
            return 0.0

        accept.__annotations__ = {"x": annotation}
        law = standard_normal()
        Function(
            accept,
            label="accept",
        )(law)

        assert seen == [law]

    @pytest.mark.parametrize(
        "annotation",
        [
            Distribution,
            Normal,
            SupportsLogProb,
            Distribution | None,
            _DistributionAlias[float],
            _OptionalNormalAlias,
            _CapabilityAlias,
            _MaybeAlias[Normal],
            _OptionalOfAlias[Normal],
            _SecondOfAlias[float, Normal],
        ],
        ids=[
            "Distribution",
            "subclass",
            "capability",
            "optional",
            "aliased-origin",
            "aliased-union",
            "aliased-capability",
            "generic-alias",
            "nested-generic-alias",
            "two-parameter-alias",
        ],
    )
    def test_a_parameter_that_names_a_distribution_consumes_it(self, annotation):
        seen = []

        def consume(x):
            seen.append(x)
            return 0.0

        consume.__annotations__ = {"x": annotation}
        law = standard_normal()
        Function(
            consume,
            label="consume",
        )(law)

        assert seen == [law]

    def test_a_parameter_annotated_function_expects_a_value(self):
        assert isinstance(_kind_of(Function, standard_normal()), Distribution)

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto", "jax"])
    def test_a_function_annotation_lifts_in_check_and_execution(self, dispatch):
        @function(dispatch=dispatch, max_evaluations=6)
        def square(x: Function):
            return x * x

        law = standard_normal()
        report = square.check(law)
        assert report.feasible is True
        assert report.lifted == ("x",)
        with workflow_run(seed=0):
            result = square(law)
        assert isinstance(result, Distribution)
        assert result.num_atoms == 6

    def test_a_function_annotation_sweeps_values(self):
        @function(dispatch="sequential")
        def double(x: Function):
            return 2.0 * x

        np.testing.assert_array_equal(double(_batch([1.0, 2.0])).raw, [2.0, 4.0])

    def test_a_batch_where_one_element_is_expected_is_swept(self):
        @function(dispatch="sequential")
        def double(x):
            return 2.0 * x

        result = double(_batch([1.0, 2.0, 3.0]))

        assert isinstance(result, Batch)
        assert result.level_names == ("row",)
        np.testing.assert_allclose(np.asarray(result.raw), [2.0, 4.0, 6.0])

    @pytest.mark.parametrize(
        "annotation",
        [NumericArrayBatch, _BatchAlias, _MaybeAlias[NumericArrayBatch]],
        ids=["class", "alias", "generic-alias"],
    )
    def test_a_parameter_that_names_the_batch_consumes_it(self, annotation):
        seen = []

        def consume(x):
            seen.append(x)
            return 0.0

        consume.__annotations__ = {"x": annotation}
        rows = _batch([1.0, 2.0])
        Function(
            consume,
            label="consume",
        )(rows)

        assert seen == [rows]

    @pytest.mark.parametrize("annotation", [Any, _AnyAlias], ids=["Any", "alias"])
    def test_a_parameter_annotated_any_passes_a_batch_whole(self, annotation):
        seen = []

        def accept(x):
            seen.append(x)
            return 0.0

        accept.__annotations__ = {"x": annotation}
        rows = _batch([1.0, 2.0])
        Function(accept, label="accept")(rows)

        assert seen == [rows]

    def test_a_batch_and_a_distribution_give_a_sweep_of_broadcasts(self):
        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(shift=None))
        def shift(a, z):
            return a + z

        with workflow_run(seed=0):
            result = shift(_batch([0.0, 10.0]), standard_normal())

        assert isinstance(result[0], Distribution)
        assert float(np.mean(atom_leaves(result[1])[0])) > 5.0

    def test_a_nested_sweep_returns_a_batch_of_laws(self):
        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(shift=None))
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

        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(record=None))
        def record(theta):
            seen.append(theta)
            return 0.0

        with workflow_run(seed=0):
            record(record_law())

        assert all(isinstance(draw, Record) for draw in seen)

    def test_a_one_field_record_draw_arrives_as_a_record(self):
        seen = []

        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(record=None))
        def record(theta):
            seen.append(theta)
            return 0.0

        with workflow_run(seed=0):
            record(one_field_law())

        assert all(isinstance(draw, Record) for draw in seen)

    def test_explicit_binding_reads_the_receiving_parameter_not_the_component(self):
        seen = []

        @function(max_evaluations=6, dispatch="sequential", output_spec=OutputSpec(weigh=None))
        def weigh(theta):
            seen.append(theta)
            return theta["a"] + theta["b"]

        law = record_law()
        with workflow_run(seed=0):
            result = weigh(theta=law)

        assert "theta" not in law.event_spec.components
        assert isinstance(result, Distribution)
        assert all(set(draw.keys()) == {"a", "b"} for draw in seen)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_a_record_passed_beside_a_law_arrives_as_the_call_passed_it(self, dispatch):
        seen = []
        weights = NumericRecord({"a": 2.0, "b": 1.0}, label="weights")

        @function(dispatch=dispatch)
        def affine(weights, mu):
            seen.append(weights)
            return weights["a"] * mu + weights["b"]

        law = EmpiricalDistribution(jnp.array([1.0, 2.0, 4.0]), component="mu", label="prior")
        result = affine(weights, law)

        assert all(received is weights for received in seen)
        assert str(result) == "affine(weights, mu ~ prior)"
        np.testing.assert_allclose(atom_leaves(result)[0], [3.0, 5.0, 9.0])


def _receiving(annotations: dict[str, Any], **controls: Any) -> tuple[Function, list[Any]]:
    """A function of ``x`` annotated by *annotations*, and the list of what its body receives."""
    seen: list[Any] = []

    def body(x):
        seen.append(x)
        return 0.0

    body.__annotations__ = annotations
    return Function(
        body,
        **controls,
        label="body",
        output_spec=OutputSpec(body=None),
    ), seen


class TestARawClassAnnotation:
    def test_a_function_written_for_pandas_runs_on_a_record_field(self):
        pd = pytest.importorskip("pandas")

        def differences(c):
            return jnp.asarray(c.diff().dropna().to_numpy())

        differences.__annotations__ = {"c": pd.Series, "return": jax.Array}
        record = Record(
            {"c": pd.Series([1.0, 2.0, 4.0])},
            label="r",
        )

        result = Function(
            differences,
            label="differences",
        )(record["c"])

        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0])

    def test_a_series_parameter_receives_the_series_a_record_field_holds(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series([1.0, 2.0, 4.0])
        wrapped, seen = _receiving({"x": pd.Series})

        wrapped(
            Record(
                {"c": series},
                label="r",
            )["c"]
        )

        assert len(seen) == 1 and seen[0] is series

    def test_a_data_array_parameter_receives_the_data_array_a_record_field_holds(self):
        xr = pytest.importorskip("xarray")
        data = xr.DataArray(np.arange(3.0), dims=("t",))
        wrapped, seen = _receiving({"x": xr.DataArray})

        wrapped(
            Record(
                {"d": data},
                label="r",
            )["d"]
        )

        assert len(seen) == 1 and seen[0] is data

    def test_a_jax_array_parameter_receives_the_array_a_numeric_array_holds(self):
        value = jnp.arange(3.0)
        wrapped, seen = _receiving({"x": jax.Array})

        wrapped(
            NumericArray(
                value,
                label="x",
            )
        )

        assert len(seen) == 1 and seen[0] is value

    def test_apply_presents_the_raw_form_as_a_call_does(self):
        value = jnp.arange(3.0)
        wrapped, seen = _receiving({"x": jax.Array})

        wrapped.apply(
            NumericArray(
                value,
                label="x",
            )
        )

        assert len(seen) == 1 and seen[0] is value

    def test_each_view_of_a_swept_batch_arrives_as_the_tfp_marginal(self):
        """A view's raw form is its marginal's, so a TFP parameter receives the TFP law.

        A view passed alone is broadcast instead, since ``tfd.Distribution``
        names no ProbPipe law, so the body receives its draws.
        """
        tfd = pytest.importorskip("tensorflow_probability.substrates.jax.distributions")
        views = [(Normal("a", loc, 1.0) * Normal("b", 0.0, 1.0))["a"] for loc in (0.0, 2.0)]
        wrapped, seen = _receiving({"x": tfd.Distribution})

        wrapped(
            DistributionBatch(
                views,
                "row",
                label="views",
            )
        )

        assert [type(law) for law in seen] == [tfd.Normal, tfd.Normal]
        assert [float(law.loc) for law in seen] == [0.0, 2.0]

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_each_element_of_a_swept_batch_arrives_in_its_raw_form(self, dispatch):
        wrapped, seen = _receiving({"x": jax.Array}, dispatch=dispatch)

        wrapped(_batch([1.0, 2.0, 3.0]))

        assert seen and all(isinstance(element, jax.Array) for element in seen)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_each_draw_of_a_lifted_law_arrives_as_a_jax_array(self, dispatch):
        wrapped, seen = _receiving({"x": jax.Array}, max_evaluations=6, dispatch=dispatch)

        with workflow_run(seed=0):
            wrapped(standard_normal())

        assert seen and all(isinstance(draw, jax.Array) for draw in seen)

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [
            (jax.Array | None, jnp.arange(3.0)),
            (npt.NDArray[np.float64], np.arange(3.0)),
            (npt.NDArray, np.arange(3.0)),
            (_ArrayAlias, np.arange(3.0)),
            (np.ndarray[Any, np.dtype[np.float64]], np.arange(3.0)),
            (_ArrayAlias[np.float64], np.arange(3.0)),
            (_ArrayAlias[np.float64] | None, np.arange(3.0)),
            (Annotated[_ArrayAlias[np.float64], "array"], np.arange(3.0)),
            (_OptionalArrayAlias, np.arange(3.0)),
            (_MaybeAlias[np.ndarray], np.arange(3.0)),
            (_MaybeAlias[_ArrayAlias[np.float64]], np.arange(3.0)),
            (_SecondOfAlias[jax.Array, np.ndarray], np.arange(3.0)),
        ],
        ids=[
            "optional",
            "numpy",
            "bare-numpy",
            "bare-alias",
            "ndarray-generic",
            "alias",
            "optional-alias",
            "annotated-alias",
            "aliased-union",
            "generic-alias",
            "generic-alias-of-alias",
            "two-parameter-alias",
        ],
    )
    def test_an_annotation_names_the_class_of_its_arm_or_its_origin(self, annotation, value):
        wrapped, seen = _receiving({"x": annotation})

        wrapped(
            NumericArray(
                value,
                label="x",
            )
        )

        assert len(seen) == 1 and seen[0] is value

    def test_a_typing_extensions_alias_names_the_class_of_its_value(self):
        # Before Python 3.14, typing_extensions defines a TypeAliasType class of its own.
        typing_extensions = pytest.importorskip("typing_extensions")
        alias = typing_extensions.TypeAliasType("Array", np.ndarray)
        value = np.arange(3.0)
        wrapped, seen = _receiving({"x": alias})

        wrapped(NumericArray(value, label="x"))

        assert len(seen) == 1 and seen[0] is value

    def test_a_typing_extensions_generic_alias_names_the_class_of_its_argument(self):
        typing_extensions = pytest.importorskip("typing_extensions")
        T = typing_extensions.TypeVar("T")
        maybe = typing_extensions.TypeAliasType("Maybe", T | None, type_params=(T,))
        value = np.arange(3.0)
        wrapped, seen = _receiving({"x": maybe[np.ndarray]})

        wrapped(NumericArray(value, label="x"))

        assert len(seen) == 1 and seen[0] is value

    def test_an_annotated_hint_of_a_callable_object_names_the_class_it_wraps(self):
        # A callable object presents its hints without get_type_hints removing
        # the Annotated metadata.
        seen: list[Any] = []

        class Receive:
            def __call__(self, x):
                seen.append(x)
                return 0.0

        Receive.__call__.__annotations__ = {"x": Annotated[jax.Array, "metadata"]}
        value = jnp.arange(3.0)
        Function(Receive(), label="receive")(NumericArray(value, label="x"))

        assert len(seen) == 1 and seen[0] is value

    def test_an_annotated_distribution_hint_of_a_partial_consumes_the_law(self):
        seen: list[Any] = []

        def consume(d, scale):
            seen.append(d)
            return 0.0

        consume.__annotations__ = {"d": Annotated[Normal, "prior"], "scale": float}
        law = standard_normal()
        Function(functools.partial(consume, scale=1.0), label="consume")(law)

        assert seen == [law]

    def test_each_argument_a_variadic_parameter_collects_arrives_in_its_raw_form(self):
        values = (jnp.arange(2.0), jnp.arange(3.0))
        seen: list[Any] = []

        def body(*xs):
            seen.extend(xs)
            return 0.0

        body.__annotations__ = {"xs": jax.Array}
        Function(
            body,
            label="body",
        )(
            *(
                NumericArray(
                    value,
                    label=f"x{i}",
                )
                for i, value in enumerate(values)
            )
        )

        assert len(seen) == 2 and all(got is value for got, value in zip(seen, values))

    @pytest.mark.parametrize(
        "annotations",
        [
            {},
            {"x": Any},
            {"x": NumericArray},
            {"x": object},
            {"x": np.ndarray},
            {"x": _UnresolvedAlias},
            {"x": _RecursiveAlias},
            {"x": _AnyAlias},
        ],
        ids=[
            "unannotated",
            "Any",
            "NumericArray",
            "object",
            "ndarray",
            "unresolved-alias",
            "recursive-alias",
            "Any-alias",
        ],
    )
    def test_any_other_argument_arrives_as_it_is(self, annotations):
        argument = NumericArray(
            jnp.arange(3.0),
            label="x",
        )
        wrapped, seen = _receiving(annotations)

        wrapped(argument)

        assert len(seen) == 1 and seen[0] is argument


class TestGrouping:
    def test_one_distribution_passed_twice_is_one_group(self):
        @function(
            max_evaluations=8,
            dispatch="sequential",
            output_spec=OutputSpec(difference=None),
        )
        def difference(x, y):
            return x - y

        law = standard_normal()
        with workflow_run(seed=0):
            result = difference(law, law)

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_sibling_views_of_one_parent_co_sample(self):
        @function(
            max_evaluations=8,
            dispatch="sequential",
            output_spec=OutputSpec(residual=None),
        )
        def residual(a, b):
            return b - 2.0 * a

        law = record_law()
        with workflow_run(seed=0):
            result = residual(law["a"], law["b"])

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_a_parent_and_its_own_view_co_sample(self):
        @function(
            max_evaluations=8,
            dispatch="sequential",
            output_spec=OutputSpec(residual=None),
        )
        def residual(record, a):
            return record["a"] - a

        law = record_law()
        with workflow_run(seed=0):
            result = residual(law, law["a"])

        for leaf in atom_leaves(result):
            np.testing.assert_array_equal(leaf, 0.0)

    def test_laws_with_no_common_ancestor_sample_independently(self):
        @function(
            max_evaluations=16,
            dispatch="sequential",
            output_spec=OutputSpec(difference=None),
        )
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

        law = Unsampled(
            OutputSpec(bare=SCALAR),
            label="bare",
        )

        assert isinstance(error_of(lambda: identity(law)), ResolutionError)


class TestAlignment:
    def test_batches_on_one_level_zip(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = add(_batch([1.0, 2.0]), _batch([10.0, 20.0]))

        assert result.batch_shape == (2,)
        np.testing.assert_allclose(np.asarray(result.raw), [11.0, 22.0])

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

    def test_a_level_of_size_one_broadcasts(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = add(_batch([1.0]), _batch([10.0, 20.0]))

        assert result.level_names == ("row",)
        assert result.axis_groups == ((2,),)
        np.testing.assert_allclose(result.raw, [11.0, 21.0], atol=0, rtol=0)


@pytest.fixture(params=["auto", "jax", "sequential", "thread"])
def alignment_dispatch(request):
    return request.param


class TestSingletonAlignment:
    def test_a_later_singleton_and_three_operands_broadcast(self, alignment_dispatch):
        @function(dispatch=alignment_dispatch)
        def add(x, y, z):
            return x + y + z

        result = add(_batch([10.0, 20.0]), _batch([1.0]), _batch([100.0]))

        assert result.axis_groups == ((2,),)
        np.testing.assert_allclose(result.raw, [111.0, 121.0], atol=0, rtol=0)

    def test_multi_axis_levels_broadcast_from_the_right(self, alignment_dispatch):
        @function(dispatch=alignment_dispatch)
        def add(x, y):
            return x + y

        left = NumericArrayBatch(
            jnp.array([[1.0], [2.0]]), "grid", axes_per_level=(2,), label="left"
        )
        right = _batch([10.0, 20.0, 30.0], "grid")
        result = add(left, right)

        assert result.level_names == ("grid",)
        assert result.axis_groups == ((2, 3),)
        np.testing.assert_allclose(
            result.raw, [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]], atol=0, rtol=0
        )

    def test_each_level_broadcasts_without_crossing_its_boundary(self, alignment_dispatch):
        @function(dispatch=alignment_dispatch)
        def add(x, y):
            return x + y

        left = NumericArrayBatch(
            jnp.array([[1.0], [2.0]]), ("chain", "draw"), axes_per_level=(1, 1), label="left"
        )
        right_values = jnp.arange(12.0).reshape(1, 2, 2, 3)
        right = NumericArrayBatch(
            right_values, ("chain", "draw"), axes_per_level=(2, 2), label="right"
        )
        result = add(left, right)

        assert result.level_names == ("chain", "draw")
        assert result.axis_groups == ((1, 2), (2, 3))
        np.testing.assert_allclose(
            result.raw,
            np.asarray(right_values) + np.array([1.0, 2.0]).reshape(1, 2, 1, 1),
            atol=0,
            rtol=0,
        )

    def test_two_partitions_of_one_array_broadcast_within_each_level(self, alignment_dispatch):
        """Each level's shapes broadcast by NumPy's rule, so two partitions of one
        array combine when their shapes for each level broadcast."""

        @function(dispatch=alignment_dispatch)
        def add(x, y):
            return x + y

        data = jnp.arange(36.0).reshape(3, 3, 4)
        left = NumericArrayBatch(data, ("a", "b"), axes_per_level=(1, 2), label="left")
        right = NumericArrayBatch(data, ("a", "b"), axes_per_level=(2, 1), label="right")
        result = add(left, right)

        assert left.axis_groups == ((3,), (3, 4))
        assert right.axis_groups == ((3, 3), (4,))
        assert result.level_names == ("a", "b")
        assert result.axis_groups == ((3, 3), (3, 4))
        assert result.batch_shape == (3, 3, 3, 4)
        # Left reads level a at its trailing coordinate a1, and right reads level b
        # at its trailing coordinate b1.
        for a0, a1, b0, b1 in [(0, 0, 0, 0), (2, 1, 0, 3), (1, 2, 2, 1)]:
            np.testing.assert_allclose(
                result.raw[a0, a1, b0, b1],
                data[a1, b0, b1] + data[a0, a1, b1],
                atol=0,
                rtol=0,
            )

    def test_a_singleton_record_broadcasts_with_an_array(self, alignment_dispatch):
        @function(dispatch=alignment_dispatch)
        def add(record, x):
            return record["offset"] + x

        records = RecordBatch.stack([Record({"offset": 1.0}, label="offsets")], level_name="row")
        result = add(records, _batch([10.0, 20.0]))

        assert result.axis_groups == ((2,),)
        np.testing.assert_allclose(result.raw, [11.0, 21.0], atol=0, rtol=0)

    def test_a_singleton_batch_of_laws_broadcasts_with_an_array(self, alignment_dispatch):
        @function(dispatch=alignment_dispatch)
        def add(law: Distribution, x):
            return mean(law) + x

        laws = DistributionBatch([Normal("z", 1.0, 1.0)], "row", label="laws")
        if alignment_dispatch == "jax":
            # JAX vectorizes only batches of records and arrays.
            with pytest.raises(
                ValueError, match="can only sweep over batches of records or arrays"
            ):
                add(laws, _batch([10.0, 20.0]))
            return
        result = add(laws, _batch([10.0, 20.0]))

        assert result.axis_groups == ((2,),)
        np.testing.assert_allclose(result.raw, [11.0, 21.0], atol=0, rtol=0)

    def test_an_empty_level_broadcasts_with_one_without_running_the_body(self, alignment_dispatch):
        calls = []

        @function(dispatch=alignment_dispatch, output_spec=OutputSpec(total=SCALAR))
        def add(x, y):
            calls.append((x, y))
            return x + y

        result = add(_batch([1.0]), _batch([]))

        assert calls == []
        assert result.axis_groups == ((0,),)
        assert result.batch_shape == (0,)
        assert result.element_spec.shape == ()

    def test_a_later_incompatible_size_raises_naming_each_shape(self):
        @function(dispatch="sequential")
        def add(x, y, z):
            return x + y + z

        with pytest.raises(
            ApplicabilityError,
            match=(
                r"^level 'row' has shapes \(1,\) for 'x', \(2,\) for 'y', and \(3,\) for 'z', "
                r"which do not broadcast$"
            ),
        ):
            add(_batch([1.0]), _batch([10.0, 20.0]), _batch([1.0, 2.0, 3.0]))

    def test_a_singleton_forms_a_product_with_a_differently_named_level(self):
        @function(dispatch="sequential")
        def add(x, y):
            return x + y

        result = add(_batch([1.0], "left"), _batch([10.0, 20.0], "right"))

        assert result.level_names == ("left", "right")
        assert result.axis_groups == ((1,), (2,))
        np.testing.assert_allclose(result.raw, [[11.0, 21.0]], atol=0, rtol=0)

    @pytest.mark.parametrize("dispatch", ["auto", "jax", "sequential"])
    def test_a_nested_broadcast_repeats_singleton_values_per_sweep_cell(self, dispatch):
        @function(max_evaluations=6, dispatch=dispatch)
        def add(x, y, z):
            return x + y + z

        law = EmpiricalDistribution(jnp.array([0.0, 2.0]), component="z")
        with workflow_run(seed=7):
            result = add(_batch([1.0]), _batch([10.0, 20.0]), law)

        assert isinstance(result, DistributionBatch)
        assert result.axis_groups == ((2,),)
        for index in range(2):
            leaves = atom_leaves(result[index])
            assert len(leaves) == 1
            np.testing.assert_allclose(
                leaves[0], [11.0 + 10.0 * index, 13.0 + 10.0 * index], atol=0, rtol=0
            )
