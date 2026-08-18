"""Tests for Function sweep execution helpers."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    DistributionArray,
    EventTemplate,
    Function,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    Record,
    RecordBatch,
    mean,
)
from probpipe.core import _workflow_call, _workflow_execution, _workflow_sweep
from probpipe.core._record_batch import _MappedBatchColumns
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan


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
            [Record("p", {"x": jnp.asarray(float(i))}, name_is_auto=True) for i in range(n)],
            level_name=level_name,
        )

    @staticmethod
    def _body(p):
        return RecordBatch.stack(
            [Record("r", {"y": p["x"] * k}, name_is_auto=True) for k in (1.0, 2.0, 3.0)],
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
            {"x": jnp.arange(6.0).reshape(2, 3)},
            "cell",
            element_spec=EventTemplate(x=()),
            axis_groups=((2, 3),),
            name="batch",
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
                [Record("r", {"z": p["x"] * q["w"] * k}, name_is_auto=True) for k in (1.0, 2.0)],
                level_name="k",
            )

        first = self._rows(2, level_name="a")
        second = RecordBatch.stack(
            [Record("q", {"w": jnp.asarray(float(i))}, name_is_auto=True) for i in range(3)],
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
                [Record("r", {"inner": {"y": p["x"] * k}}, name_is_auto=True) for k in (1.0, 2.0)],
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
                    [Record("r", {"y": v * k}, name_is_auto=True) for k in (1.0, 2.0)],
                    level_name="k",
                )
            )(jnp.arange(4.0))
