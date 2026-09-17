"""Tests for Function sweep execution helpers."""

from __future__ import annotations

import inspect
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    DistributionArray,
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    Record,
    RecordBatch,
    RecordSpec,
    log_prob,
    mean,
)
from probpipe.core import _workflow_call, _workflow_execution, _workflow_sweep
from probpipe.core._record_batch import _MappedBatchColumns
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan
from probpipe.core.constraints import positive


def _numeric_record_batch(
    field: str, values: range, *, level_name: str = "draw"
) -> NumericRecordBatch:
    return NumericRecordBatch.stack(
        [NumericRecord("nr", **{field: float(value)}) for value in values],
        level_name=level_name,
    )


def _ref(name: str) -> _workflow_call.WorkflowInputRef:
    return _workflow_call.WorkflowInputRef(name)


def _plan(values):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(signature)
    return build_broadcast_plan(values=values, signature_info=signature_info)


def _stochastic_plan(values, n_broadcast_samples):
    return build_stochastic_plan(values, _plan(values), n_broadcast_samples)


def _unexpected_distribution_broadcast(*args, **kwargs):
    raise AssertionError("distribution broadcast should not run")


def _require_not_called(*args, **kwargs):
    raise AssertionError("JAX traceability should not be required")


class TestSliceSweepValues:
    def test_views_from_same_parent_zip(self):
        parent = NumericRecordBatch.stack(
            [NumericRecord("nr", x=float(i), y=float(10 + i)) for i in range(3)], level_name="draw"
        )
        views = parent.select_all()
        values = {"x": views["x"], "y": views["y"]}
        plan = _plan(values)

        observed = [
            _workflow_sweep.slice_sweep_values(
                values=values,
                index=i,
                array_groups=plan.array_groups,
            )
            for i in range(plan.n_sweep)
        ]

        assert [(float(row["x"]), float(row["y"])) for row in observed] == [
            (0.0, 10.0),
            (1.0, 11.0),
            (2.0, 12.0),
        ]

    def test_arrays_from_different_parents_use_row_major_product(self):
        values = {
            "a": _numeric_record_batch("a", range(2), level_name="outer"),
            "b": _numeric_record_batch("b", range(3), level_name="inner"),
        }
        plan = _plan(values)

        observed = [
            _workflow_sweep.slice_sweep_values(
                values=values,
                index=i,
                array_groups=plan.array_groups,
            )
            for i in range(plan.n_sweep)
        ]

        assert [(float(row["a"]["a"]), float(row["b"]["b"])) for row in observed] == [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
        ]

    def test_distribution_array_cell_uses_flat_component(self):
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2,),
            loc=jnp.asarray([3.0, 4.0]),
            scale=jnp.asarray([1.0, 1.0]),
            name="d",
        )
        values = {"d": da}
        plan = _plan(values)

        first = _workflow_sweep.slice_sweep_values(
            values=values,
            index=0,
            array_groups=plan.array_groups,
        )
        second = _workflow_sweep.slice_sweep_values(
            values=values,
            index=1,
            array_groups=plan.array_groups,
        )

        assert isinstance(first["d"], Normal)
        assert isinstance(second["d"], Normal)
        assert float(first["d"].loc) == 3.0
        assert float(second["d"].loc) == 4.0


class TestExecuteSweep:
    def test_row_wise_sweep_uses_execution_request(self, monkeypatch):
        values = {"p": _numeric_record_batch("x", range(3))}
        plan = _plan(values)
        execution = _workflow_execution.WorkflowExecutionConfig(
            mode="thread",
            max_workers=2,
            name="double",
        )
        seen = {}

        def double(p):
            return 2.0 * p["x"]

        def resolve_dispatch(values, array_args, *, jax_supported):
            return "thread"

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**item.call_values()) for item in request.work_items]

        monkeypatch.setattr(
            _workflow_sweep._workflow_execution,
            "execute_many",
            fake_execute_many,
        )

        result = _workflow_sweep.execute_sweep(
            func=double,
            values=values,
            plan=plan,
            stochastic_plan=None,
            make_execution_config=lambda: execution,
            requested_dispatch="thread",
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=_require_not_called,
            distribution_broadcast=_unexpected_distribution_broadcast,
            workflow_name="double",
        )

        request = seen["request"]
        assert request.execution is execution
        assert request.func is double
        assert [float(item.call_values()["p"]["x"]) for item in request.work_items] == [
            0.0,
            1.0,
            2.0,
        ]
        np.testing.assert_allclose(result.values, jnp.asarray([0.0, 2.0, 4.0]))

    def test_include_inputs_is_rejected_for_sweep(self):
        values = {"p": _numeric_record_batch("x", range(1))}
        plan = _plan(values)
        execution = _workflow_execution.WorkflowExecutionConfig(
            mode="sequential",
            name="identity",
        )

        with pytest.raises(NotImplementedError, match="include_inputs=True"):
            _workflow_sweep.execute_sweep(
                func=lambda p: p["x"],
                values=values,
                plan=plan,
                stochastic_plan=None,
                make_execution_config=lambda: execution,
                requested_dispatch="sequential",
                resolve_dispatch=lambda *args, **kwargs: "sequential",
                require_jax_traceable=_require_not_called,
                distribution_broadcast=_unexpected_distribution_broadcast,
                workflow_name="identity",
                include_inputs=True,
            )

    def test_nested_sweep_calls_distribution_broadcast_and_marginalizes(self):
        values = {
            "p": _numeric_record_batch("x", range(2)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }
        plan = _plan(values)
        stochastic_plan = _stochastic_plan(values, 7)
        execution = _workflow_execution.WorkflowExecutionConfig(
            mode="sequential",
            name="nested",
        )
        calls = []

        def distribution_broadcast(
            row_values,
            received_plan,
            logical_unit,
            include_inputs,
        ):
            calls.append(
                {
                    "x": float(row_values["p"]["x"]),
                    "plan": received_plan,
                    "logical_unit": logical_unit,
                    "include_inputs": include_inputs,
                }
            )
            loc = float(row_values["p"]["x"])
            return BroadcastDistribution(
                input_samples={"noise": jnp.asarray([0.0])},
                output_samples=jnp.asarray([loc]),
                output_distributions=[Normal(loc=loc, scale=1.0, name=f"row_{int(loc)}")],
                weights=None,
                broadcast_args=["noise"],
            )

        result = _workflow_sweep.execute_sweep(
            func=lambda p, noise: p["x"] + noise,
            values=values,
            plan=plan,
            stochastic_plan=stochastic_plan,
            make_execution_config=lambda: execution,
            requested_dispatch="sequential",
            resolve_dispatch=lambda *args, **kwargs: "sequential",
            require_jax_traceable=_require_not_called,
            distribution_broadcast=distribution_broadcast,
            workflow_name="nested",
        )

        assert result.batch_shape == (2,)
        assert [float(mean(component)) for component in result.components] == [
            0.0,
            1.0,
        ]
        assert calls == [
            {
                "x": 0.0,
                "plan": stochastic_plan,
                "logical_unit": stochastic_plan.logical_units[0],
                "include_inputs": True,
            },
            {
                "x": 1.0,
                "plan": stochastic_plan,
                "logical_unit": stochastic_plan.logical_units[1],
                "include_inputs": True,
            },
        ]
        assert result.provenance.operation == "workflow.nested"
        assert result.provenance.metadata["k"] == 7


class TestASweptBodyThatReturnsABatch:
    """A body returning a batch vectorizes instead of falling back.

    ``vmap`` adds an output axis that ``RecordBatch``'s unflatten hook cannot
    name — *a shape is not a provenance* — so the executor hands the transform
    raw columns and rebuilds the batch itself afterwards, from the levels it
    swept. The sequential path is the oracle throughout: it builds the same
    aggregate one row at a time.
    """

    @staticmethod
    def _rows(n: int, *, level_name: str = "row") -> RecordBatch:
        return RecordBatch.stack(
            [Record("p", {"x": jnp.asarray(float(i))}) for i in range(n)],
            level_name=level_name,
        )

    @staticmethod
    def _body(p):
        return RecordBatch.stack(
            [Record("r", {"y": p["x"] * k}) for k in (1.0, 2.0, 3.0)],
            level_name="k",
        )

    def test_the_sweep_takes_the_mapped_path(self, monkeypatch):
        """The mapped executor runs, with no fall back to row-wise dispatch.

        The discriminating assertion: every other test in this class passes on
        the sequential path too, so only this one distinguishes vectorizing from
        agreeing with the oracle.
        """
        reached = []
        real = _workflow_sweep.execute_sweep_rows_jax

        def spy(**kwargs):
            reached.append(1)
            return real(**kwargs)

        monkeypatch.setattr(_workflow_sweep, "execute_sweep_rows_jax", spy)
        Function(func=self._body, name="swept")(self._rows(4))

        assert reached == [1]

    def test_the_levels_are_the_sweeps_then_the_bodys(self):
        out = Function(func=self._body, name="swept")(self._rows(4))

        assert out.level_names == ("row", "k")
        assert out.axis_groups == ((4,), (3,))

    def test_the_shape_agrees_with_the_columns_it_holds(self):
        """A batch whose spec its own columns contradict is the failure to avoid."""
        out = Function(func=self._body, name="swept")(self._rows(4))

        assert out.batch_shape == (4, 3)
        assert out.batch_size == 12
        assert np.shape(out._raw_column("y")) == (4, 3)

    def test_it_matches_sequential_dispatch(self):
        mapped = Function(func=self._body, name="swept")(self._rows(4))
        sequential = Function(func=self._body, name="swept", dispatch="sequential")(self._rows(4))

        assert mapped.element_spec == sequential.element_spec
        assert mapped.level_names == sequential.level_names
        assert mapped.axis_groups == sequential.axis_groups
        np.testing.assert_allclose(np.asarray(mapped["y"]), np.asarray(sequential["y"]))

    def test_a_multi_axis_level_reshapes_on_the_mapped_path(self, monkeypatch):
        """One level spanning two axes, swept under the map.

        The mapped executor flattens the sweep to a single axis of
        ``prod(batch_shape)`` and the carrier restores its shape afterwards, so
        a sweep of rank greater than one is what exercises that reshape. It
        takes a single argument on purpose: two array arguments make two zip
        groups, which sets ``jax_supported`` false and would quietly measure
        row-wise dispatch instead.
        """
        reached = []
        real = _workflow_sweep.execute_sweep_rows_jax

        def spy(**kwargs):
            reached.append(1)
            return real(**kwargs)

        monkeypatch.setattr(_workflow_sweep, "execute_sweep_rows_jax", spy)

        grid = RecordBatch(
            "batch",
            {"x": jnp.arange(6.0).reshape(2, 3)},
            "cell",
            element_spec=RecordSpec(x=()),
            axes_per_level=(2,),
        )

        mapped = Function(func=self._body, name="swept")(grid)
        sequential = Function(func=self._body, name="swept", dispatch="sequential")(grid)

        assert reached == [1]
        assert mapped.level_names == ("cell", "k")
        assert mapped.axis_groups == ((2, 3), (3,))
        assert mapped.batch_shape == (2, 3, 3)
        assert np.shape(mapped._raw_column("y")) == (2, 3, 3)
        np.testing.assert_allclose(np.asarray(mapped["y"]), np.asarray(sequential["y"]))

    def test_two_zip_groups_sweep_as_a_product(self):
        """Two groups product, so the aggregate carries both sweep levels first.

        Group structure, not only total size: collapsing the sweep onto one flat
        leading axis agrees on ``batch_size`` and loses which axis belongs to
        which level. Two array arguments take row-wise dispatch, so this pins
        the aggregation rather than the mapped path.
        """

        def body(p, q):
            return RecordBatch.stack(
                [Record("r", {"z": p["x"] * q["w"] * k}) for k in (1.0, 2.0)],
                level_name="k",
            )

        first = self._rows(2, level_name="a")
        second = RecordBatch.stack(
            [Record("q", {"w": jnp.asarray(float(i))}) for i in range(3)],
            level_name="b",
        )

        mapped = Function(func=body, name="swept")(first, second)
        sequential = Function(func=body, name="swept", dispatch="sequential")(first, second)

        assert mapped.level_names == ("a", "b", "k")
        assert mapped.axis_groups == ((2,), (3,), (2,))
        assert mapped.batch_shape == (2, 3, 2)
        np.testing.assert_allclose(np.asarray(mapped["z"]), np.asarray(sequential["z"]))

    def test_a_nested_element_survives_the_transform(self):
        """Columns are leaf-keyed, so a nested record needs no special case."""

        def body(p):
            return RecordBatch.stack(
                [Record("r", {"inner": {"y": p["x"] * k}}) for k in (1.0, 2.0)],
                level_name="k",
            )

        mapped = Function(func=body, name="swept")(self._rows(3))
        sequential = Function(func=body, name="swept", dispatch="sequential")(self._rows(3))

        assert mapped.level_names == ("row", "k")
        assert mapped.element_spec == sequential.element_spec
        np.testing.assert_allclose(
            np.asarray(mapped._raw_column("inner/y")),
            np.asarray(sequential._raw_column("inner/y")),
        )

    def test_the_carrier_does_not_reach_the_caller(self):
        """It is wrapped and unwrapped inside one call, by construction."""
        out = Function(func=self._body, name="swept")(self._rows(4))

        assert not isinstance(out, _MappedBatchColumns)
        assert isinstance(out, RecordBatch)

    def test_a_raw_vmap_returning_a_batch_is_still_refused(self):
        """The hook is routed around, not softened.

        Only a caller that knows which axis it added, and what to call the level
        it stands for, may rebuild across one. The executor knows both; a raw
        ``vmap`` knows neither, and is refused.
        """
        with pytest.raises(ValueError, match="belongs to no level"):
            jax.vmap(
                lambda v: RecordBatch.stack(
                    [Record("r", {"y": v * k}) for k in (1.0, 2.0)],
                    level_name="k",
                )
            )(jnp.arange(4.0))


@pytest.fixture(
    params=[
        ((3,), ("row",), (1,)),
        ((2, 3), ("chain", "draw"), (1, 1)),
        ((2, 3, 2), ("chain", "draw"), (1, 2)),
    ],
    ids=["one-level", "two-levels", "multi-axis-level"],
)
def numeric_sweep_source(request):
    shape, levels, axes_per_level = request.param
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 4
    return NumericRecordBatch(
        "inputs",
        {"x": values},
        levels,
        element_spec=RecordSpec(x=NumericArraySpec((), dtype=np.float32)),
        axes_per_level=axes_per_level,
    )


class TestNumericArraySweep:
    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    @pytest.mark.parametrize(
        "declared_shape, event_shape",
        [(("d",), (2,)), (("d", "d"), (2, 2)), (("d", 3), (2, 3))],
        ids=["symbolic", "repeated-symbol", "partially-symbolic"],
    )
    def test_symbolic_numeric_rows_bind_their_event_dimensions(
        self, numeric_sweep_source, dispatch, declared_shape, event_shape
    ):
        source = numeric_sweep_source
        native = np.arange(1, np.prod(event_shape) + 1, dtype=np.float32).reshape(event_shape)
        declared = NumericArraySpec(declared_shape, dtype=np.float64, support=positive)
        value = NumericArray("original", native, spec=declared)

        result = Function(func=lambda row: value, name="repeated", dispatch=dispatch)(source)

        assert isinstance(result, NumericArrayBatch)
        assert result.element_spec == NumericArraySpec(
            event_shape, dtype=np.float64, support=positive
        )
        assert result.values.dtype == np.float32
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(
            np.asarray(result), np.broadcast_to(native, (*source.batch_shape, *event_shape))
        )
        assert value.spec is declared
        assert value.spec.shape == declared_shape
        assert value.value is native
        assert value.name == "original"
        assert value.provenance is None

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    def test_symbolic_and_concrete_numeric_rows_agree_after_binding(self, reverse):
        declarations = [
            NumericArraySpec(shape, dtype=np.float32, support=positive)
            for shape in (("d",), ("other",), (2,))
        ]
        outputs = [
            NumericArray("row", jnp.full((2,), i + 1, dtype=jnp.float32), spec=spec)
            for i, spec in enumerate(declarations)
        ]
        if reverse:
            outputs.reverse()
        source = _numeric_record_batch("x", range(3))

        result = Function(
            func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
        )(source)

        assert result.element_spec == NumericArraySpec((2,), dtype=np.float32, support=positive)
        assert result.values.dtype == np.float32
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        expected = np.repeat([[1.0], [2.0], [3.0]], 2, axis=1)
        np.testing.assert_array_equal(np.asarray(result), expected[::-1] if reverse else expected)
        assert [output.spec for output in outputs] == (
            declarations[::-1] if reverse else declarations
        )

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    @pytest.mark.parametrize("representation", ["raw", "tracked", "declared"])
    @pytest.mark.parametrize("event_shape", [(), (2,)], ids=["scalar", "vector"])
    def test_numeric_results_keep_their_declaration_and_levels(
        self, numeric_sweep_source, dispatch, representation, event_shape
    ):
        source = numeric_sweep_source
        declared = NumericArraySpec(event_shape, dtype=np.float64, support=positive)

        def body(row):
            value = row["x"] * 2 + 1
            if event_shape:
                value = jnp.stack([value, value + 1])
            if representation == "raw":
                return value
            return NumericArray(
                "row-result", value, spec=declared if representation == "declared" else None
            )

        result = Function(func=body, name="shift", dispatch=dispatch)(source)

        expected = np.asarray(source["x"]) * 2 + 1
        if event_shape:
            expected = np.stack([expected, expected + 1], axis=-1)
        expected_spec = (
            declared
            if representation == "declared"
            else NumericArraySpec(event_shape, dtype=np.float32)
        )
        assert isinstance(result, NumericArrayBatch)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        assert result.element_spec == expected_spec
        assert result.values.dtype == np.float32
        assert result.provenance is not None
        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_nested_density_results_compose_after_the_sweep(self, numeric_sweep_source, dispatch):
        source = numeric_sweep_source
        law = Normal(0.0, 1.0, name="x")
        result = Function(
            func=lambda row: log_prob(law, row["x"]), name="score", dispatch=dispatch
        )(source)

        expected = -0.5 * np.asarray(source["x"], dtype=np.float64) ** 2 - 0.5 * np.log(2 * np.pi)
        assert isinstance(result, NumericArrayBatch)
        assert result.axis_groups == source.axis_groups
        assert result.level_names == source.level_names
        assert result.element_spec == NumericArraySpec((), dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-7, atol=1e-7)

        shifted = Function(func=lambda value: value + 1, dispatch="sequential")(result)
        assert shifted.axis_groups == source.axis_groups
        assert shifted.level_names == source.level_names
        np.testing.assert_allclose(np.asarray(shifted), expected + 1, rtol=2e-7, atol=1e-7)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_native_numeric_results_keep_their_source(self, numeric_sweep_source, dispatch):
        pd = pytest.importorskip("pandas")
        native = pd.Series([1.0, 2.0], index=[5, 7])
        declared = NumericArraySpec((2,), dtype=np.float64, support=positive)
        value = NumericArray("original", native, spec=declared)

        result = Function(func=lambda row: value, name="repeated", dispatch=dispatch)(
            numeric_sweep_source
        )

        assert result.element_spec == declared
        assert result.axis_groups == numeric_sweep_source.axis_groups
        assert result.level_names == numeric_sweep_source.level_names
        np.testing.assert_array_equal(
            np.asarray(result), np.broadcast_to([1.0, 2.0], (*numeric_sweep_source.batch_shape, 2))
        )
        assert value.value is native
        assert value.name == "original"
        assert value.spec == declared
        assert value.provenance is None

    def test_disagreeing_numeric_row_supports_are_refused(self):
        declared = NumericArraySpec((), dtype=np.float64, support=positive)
        first = NumericArray("first", jnp.asarray(1.0), spec=declared)
        second = NumericArray(
            "second", jnp.asarray(2.0), spec=NumericArraySpec((), dtype=np.float64)
        )
        source = _numeric_record_batch("x", range(2))

        with pytest.raises(ValueError, match=r"different: numeric.*declarations"):
            Function(
                func=lambda row: first if float(row["x"]) == 0 else second,
                name="different",
                dispatch="sequential",
            )(source)

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    @pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
    @pytest.mark.parametrize("event_shape", [(), (2,)], ids=["scalar", "vector"])
    def test_native_and_jax_numeric_rows_promote_their_declared_dtypes(
        self, reverse, x64, event_shape
    ):
        with jax.enable_x64(x64):
            native = np.full(event_shape, 1.25, dtype=np.float64)
            if not event_shape:
                native = native[()]
            first = NumericArray("native", native)
            second = NumericArray("jax", jnp.full(event_shape, 2.5, dtype=jnp.float32))
            outputs = [second, first] if reverse else [first, second]
            source = _numeric_record_batch("x", range(2))

            result = Function(
                func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
            )(source)

        expected = np.stack([np.full(event_shape, 1.25), np.full(event_shape, 2.5)])
        if reverse:
            expected = expected[::-1]
        assert result.element_spec == NumericArraySpec(event_shape, dtype=np.float64)
        assert result.dtype == np.dtype(np.float64 if x64 else np.float32)
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)
        assert first.value is native
        assert first.spec.dtype == np.dtype(np.float64)
        assert second.spec.dtype == np.dtype(np.float32)

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    @pytest.mark.parametrize(
        "other_dtype, support, expected_dtype",
        [(np.float32, positive, np.float64), (None, positive, None)],
        ids=["promoted", "unspecified"],
    )
    def test_numeric_dtype_promotion_preserves_shared_support(
        self, reverse, other_dtype, support, expected_dtype
    ):
        first = NumericArray(
            "first",
            jnp.asarray(1.0),
            spec=NumericArraySpec((), dtype=np.float64, support=support),
        )
        second = NumericArray(
            "second",
            jnp.asarray(2.0),
            spec=NumericArraySpec((), dtype=other_dtype, support=support),
        )
        outputs = [second, first] if reverse else [first, second]

        result = Function(
            func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
        )(_numeric_record_batch("x", range(2)))

        assert result.element_spec == NumericArraySpec((), dtype=expected_dtype, support=support)
        np.testing.assert_array_equal(np.asarray(result), [2.0, 1.0] if reverse else [1.0, 2.0])

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    @pytest.mark.parametrize(
        "other_dtype, expected_dtype",
        [(jnp.float16, np.float32), (jnp.int32, jnp.bfloat16)],
        ids=["float16", "int32"],
    )
    def test_jax_extended_numeric_dtypes_promote(self, reverse, other_dtype, expected_dtype):
        outputs = [
            NumericArray(
                "row",
                jnp.asarray(value, dtype=dtype),
                spec=NumericArraySpec((), dtype=dtype, support=positive),
            )
            for value, dtype in ((1, jnp.bfloat16), (2, other_dtype))
        ]
        if reverse:
            outputs.reverse()
        source = _numeric_record_batch("x", range(2))

        result = Function(
            func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
        )(source)

        assert result.element_spec == NumericArraySpec((), dtype=expected_dtype, support=positive)
        assert result.values.dtype == np.dtype(expected_dtype)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), [2, 1] if reverse else [1, 2])

    @pytest.mark.parametrize(
        "dtypes",
        list(permutations(("int8", "uint8", "float16"))),
        ids=lambda dtypes: "-".join(dtypes),
    )
    def test_numeric_dtype_promotion_is_independent_of_row_order(self, dtypes):
        source = _numeric_record_batch("x", range(3), level_name="row")
        batches = []
        for order in (dtypes, ("int8", "float16", "uint8")):
            outputs = [
                NumericArray(
                    "row",
                    jnp.asarray(1, dtype=dtype),
                    spec=NumericArraySpec((), dtype=dtype, support=positive),
                )
                for dtype in order
            ]
            batch = Function(
                func=lambda row, outputs=outputs: outputs[int(row["x"])],
                name="mixed",
                dispatch="sequential",
            )(source)
            batches.append(batch)
            assert batch.element_spec == NumericArraySpec((), dtype=np.float16, support=positive)
            assert batch.values.dtype == np.float16
            assert batch.level_names == source.level_names
            assert batch.axis_groups == source.axis_groups
            np.testing.assert_array_equal(np.asarray(batch), [1, 1, 1])

        result = Function(
            func=lambda row: batches[int(row["x"])], name="combined", dispatch="sequential"
        )(_numeric_record_batch("x", range(2), level_name="outer"))

        assert result.element_spec == NumericArraySpec((), dtype=np.float16, support=positive)
        assert result.values.dtype == np.float16
        assert result.batch_shape == (2, 3)
        assert result.level_names == ("outer", "row")
        assert result.axis_groups == ((2,), (3,))
        np.testing.assert_array_equal(np.asarray(result), np.ones((2, 3)))

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    @pytest.mark.parametrize("tracked_float", [False, True], ids=["raw-float", "tracked-float"])
    def test_integer_and_float_numeric_rows_promote_without_losing_values(
        self, reverse, tracked_float
    ):
        integer = NumericArray(
            "integer",
            jnp.asarray(1, dtype=jnp.int32),
            spec=NumericArraySpec((), dtype=np.int32),
        )
        floating = jnp.asarray(2.5, dtype=jnp.float32)
        outputs = [integer, NumericArray("float", floating) if tracked_float else floating]
        if reverse:
            outputs.reverse()

        result = Function(
            func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
        )(_numeric_record_batch("x", range(2)))

        assert result.element_spec == NumericArraySpec(
            (), dtype=np.float64 if tracked_float else np.float32
        )
        np.testing.assert_array_equal(np.asarray(result), [2.5, 1.0] if reverse else [1.0, 2.5])

    @pytest.mark.parametrize("raw_value", [-1.0, 1.0], ids=["negative", "positive"])
    @pytest.mark.parametrize("tracked_first", [False, True], ids=["raw-first", "tracked-first"])
    @pytest.mark.parametrize("event_shape", [(), (2,)], ids=["scalar", "vector"])
    def test_mixed_numeric_rows_infer_the_aggregate_spec(
        self, raw_value, tracked_first, event_shape
    ):
        declared = NumericArraySpec(event_shape, dtype=np.float64, support=positive)
        value = NumericArray("held", jnp.full(event_shape, 2.0), spec=declared)
        raw = jnp.full(event_shape, raw_value) if event_shape else raw_value
        source = _numeric_record_batch("x", range(2))
        result = Function(
            func=lambda row: value if (float(row["x"]) == 0) == tracked_first else raw,
            name="mixed",
            dispatch="sequential",
        )(source)

        expected = np.stack([np.full(event_shape, raw_value), np.full(event_shape, 2.0)])
        if tracked_first:
            expected = expected[::-1]
        assert result.element_spec == NumericArraySpec(event_shape, dtype=result.values.dtype)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("raw_position", [0, 1, 2], ids=["raw-first", "raw-middle", "raw-last"])
    def test_mixed_numeric_rows_do_not_adopt_conflicting_partial_declarations(self, raw_position):
        outputs = [
            NumericArray(
                "positive",
                jnp.asarray(2.0),
                spec=NumericArraySpec((), dtype=np.float64, support=positive),
            ),
            NumericArray("unconstrained", jnp.asarray(3.0), spec=NumericArraySpec(())),
        ]
        outputs.insert(raw_position, -1.0)
        source = _numeric_record_batch("x", range(3))

        result = Function(
            func=lambda row: outputs[int(row["x"])], name="mixed", dispatch="sequential"
        )(source)

        expected = [2.0, 3.0]
        expected.insert(raw_position, -1.0)
        assert result.element_spec == NumericArraySpec((), dtype=result.values.dtype)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)
