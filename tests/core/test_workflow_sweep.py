"""Tests for Function sweep execution helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    DistributionArray,
    Normal,
    NumericRecord,
    NumericRecordArray,
    mean,
)
from probpipe.core import _workflow_execution, _workflow_sweep
from probpipe.core._workflow_plan import build_broadcast_plan


def _numeric_record_array(field: str, values: range) -> NumericRecordArray:
    return NumericRecordArray.stack(
        [NumericRecord("nr", **{field: float(value)}) for value in values]
    )


def _unexpected_distribution_broadcast(*args, **kwargs):
    raise AssertionError("distribution broadcast should not run")


def _require_not_called(*args, **kwargs):
    raise AssertionError("JAX traceability should not be required")


class TestSliceSweepValues:
    def test_views_from_same_parent_zip(self):
        parent = NumericRecordArray.stack(
            [NumericRecord("nr", x=float(i), y=float(10 + i)) for i in range(3)]
        )
        values = {"x": parent.view("x"), "y": parent.view("y")}
        plan = build_broadcast_plan(values=values, hints={})

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
            "a": _numeric_record_array("a", range(2)),
            "b": _numeric_record_array("b", range(3)),
        }
        plan = build_broadcast_plan(values=values, hints={})

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
        plan = build_broadcast_plan(values=values, hints={})

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
        values = {"p": _numeric_record_array("x", range(3))}
        plan = build_broadcast_plan(values=values, hints={})
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
            return [request.func(**call_values) for call_values in request.call_value_list]

        monkeypatch.setattr(
            _workflow_sweep._workflow_execution,
            "execute_many",
            fake_execute_many,
        )

        result = _workflow_sweep.execute_sweep(
            func=double,
            values=values,
            plan=plan,
            make_execution_config=lambda: execution,
            requested_dispatch="thread",
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=_require_not_called,
            distribution_broadcast=_unexpected_distribution_broadcast,
            workflow_name="double",
            n_broadcast_samples=5,
        )

        request = seen["request"]
        assert request.execution is execution
        assert request.func is double
        assert [float(row["p"]["x"]) for row in request.call_value_list] == [
            0.0,
            1.0,
            2.0,
        ]
        np.testing.assert_allclose(result["double"], jnp.asarray([0.0, 2.0, 4.0]))

    def test_include_inputs_is_rejected_for_sweep(self):
        values = {"p": _numeric_record_array("x", range(1))}
        plan = build_broadcast_plan(values=values, hints={})
        execution = _workflow_execution.WorkflowExecutionConfig(
            mode="sequential",
            name="identity",
        )

        with pytest.raises(NotImplementedError, match="include_inputs=True"):
            _workflow_sweep.execute_sweep(
                func=lambda p: p["x"],
                values=values,
                plan=plan,
                make_execution_config=lambda: execution,
                requested_dispatch="sequential",
                resolve_dispatch=lambda *args, **kwargs: "sequential",
                require_jax_traceable=_require_not_called,
                distribution_broadcast=_unexpected_distribution_broadcast,
                workflow_name="identity",
                n_broadcast_samples=5,
                include_inputs=True,
            )

    def test_nested_sweep_calls_distribution_broadcast_and_marginalizes(self):
        values = {
            "p": _numeric_record_array("x", range(2)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }
        plan = build_broadcast_plan(values=values, hints={})
        execution = _workflow_execution.WorkflowExecutionConfig(
            mode="sequential",
            name="nested",
        )
        calls = []

        def distribution_broadcast(
            row_values,
            dist_args,
            n_broadcast_samples,
            include_inputs,
        ):
            calls.append(
                {
                    "x": float(row_values["p"]["x"]),
                    "dist_args": tuple(dist_args),
                    "n": n_broadcast_samples,
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
            make_execution_config=lambda: execution,
            requested_dispatch="sequential",
            resolve_dispatch=lambda *args, **kwargs: "sequential",
            require_jax_traceable=_require_not_called,
            distribution_broadcast=distribution_broadcast,
            workflow_name="nested",
            n_broadcast_samples=7,
        )

        assert result.batch_shape == (2,)
        assert [float(mean(component)) for component in result.components] == [
            0.0,
            1.0,
        ]
        assert calls == [
            {
                "x": 0.0,
                "dist_args": ("noise",),
                "n": 7,
                "include_inputs": True,
            },
            {
                "x": 1.0,
                "dist_args": ("noise",),
                "n": 7,
                "include_inputs": True,
            },
        ]
        assert result.provenance.operation == "workflow.nested"
        assert result.provenance.metadata["k"] == 7
