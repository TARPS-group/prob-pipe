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
from probpipe.core._workflow_plan import build_broadcast_plan


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
            "p": _numeric_record_batch("x", range(2)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }
        plan = _plan(values)
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
                "dist_args": (_ref("noise"),),
                "n": 7,
                "include_inputs": True,
            },
            {
                "x": 1.0,
                "dist_args": (_ref("noise"),),
                "n": 7,
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
    swept. The sequential path is the oracle throughout: it always produced the
    right answer, only slowly.
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
        """The point of the exercise: this used to fail the probe and fall back."""
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

    def test_a_multi_axis_sweep_keeps_the_groups_apart(self):
        """Group structure, not only flat shape: ((2, 3), (2,)) is not ((2,), (3, 2)).

        Two zip groups sweep as a product, so the aggregate carries two sweep
        levels before the body's own — the case a reshape to a flat leading axis
        would silently get wrong.
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
        """The hook is routed around, not softened — issue 406 rests on it.

        Only the executor, which knows the level it swept, may rebuild across an
        added axis. A raw ``vmap`` still has no name to give one and still says
        so.
        """
        with pytest.raises(ValueError, match="belongs to no level"):
            jax.vmap(
                lambda v: RecordBatch.stack(
                    [Record("r", {"y": v * k}, name_is_auto=True) for k in (1.0, 2.0)],
                    level_name="k",
                )
            )(jnp.arange(4.0))
