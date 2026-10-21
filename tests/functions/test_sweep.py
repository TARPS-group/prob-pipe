"""Tests for Function sweep execution helpers."""

from __future__ import annotations

import inspect
import logging
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OpaqueBatch,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    log_prob,
    mean,
    workflow_run,
)
from probpipe.core._record_batch import _MappedBatchColumns
from probpipe.core.constraints import positive
from probpipe.functions import _execution, _function, _sweep
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding

from ._design_helpers import atom_leaves


def _numeric_record_batch(
    field: str, values: range, *, level_name: str = "draw"
) -> NumericRecordBatch:
    return NumericRecordBatch.stack(
        [
            NumericRecord(
                {**{field: float(value)}},
                label="nr",
            )
            for value in values
        ],
        level_name=level_name,
    )


def _ref(name: str) -> _binding.FunctionInputRef:
    return _binding.FunctionInputRef(name)


def _plan(values):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(signature)
    return build_broadcast_plan(values=values, signature_info=signature_info)


def _stochastic_plan(values, max_evaluations):
    return build_stochastic_plan(values, _plan(values), max_evaluations)


def _unexpected_distribution_broadcast(*args, **kwargs):
    raise AssertionError("distribution broadcast should not run")


def _require_not_called(*args, **kwargs):
    raise AssertionError("JAX traceability should not be required")


def _record_probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool]]:
    """Record each JAX traceability probe as two flags.

    The first flag says whether the probe traced swept values as dynamic inputs.
    The second says whether the probe succeeded.
    """
    probes: list[tuple[bool, bool]] = []
    original = _function._jax_traceability_error

    def probe(*args: object, **kwargs: object) -> Exception | None:
        error = original(*args, **kwargs)
        probes.append((bool(kwargs.get("dynamic_refs")), error is None))
        return error

    monkeypatch.setattr(_function, "_jax_traceability_error", probe)
    return probes


class TestSliceSweepValues:
    def test_views_from_same_parent_zip(self):
        parent = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"x": float(i), "y": float(10 + i)},
                    label="nr",
                )
                for i in range(3)
            ],
            level_name="draw",
        )
        views = parent.select_all()
        values = {"x": views["x"], "y": views["y"]}
        plan = _plan(values)

        observed = [
            _sweep.slice_sweep_values(
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
            _sweep.slice_sweep_values(
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

    def test_a_distribution_batch_cell_is_a_view_of_its_law(self):
        laws = DistributionBatch(
            [Normal("x", 3.0, 1.0), Normal("x", 4.0, 1.0)],
            "law",
            label="d",
        )
        values = {"d": laws}
        plan = _plan(values)

        first = _sweep.slice_sweep_values(
            values=values,
            index=0,
            array_groups=plan.array_groups,
        )
        second = _sweep.slice_sweep_values(
            values=values,
            index=1,
            array_groups=plan.array_groups,
        )

        assert isinstance(first["d"], Normal)
        assert isinstance(second["d"], Normal)
        assert float(first["d"].loc) == 3.0
        assert float(second["d"].loc) == 4.0
        assert (first["d"].label, second["d"].label) == ("d[law=0]", "d[law=1]")


class TestExecuteSweep:
    def test_row_wise_sweep_uses_execution_request(self, monkeypatch):
        values = {"p": _numeric_record_batch("x", range(3))}
        plan = _plan(values)
        execution = _execution.WorkflowExecutionConfig(
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
            return [
                request.func(**evaluation_point.call_arguments())
                for evaluation_point in request.evaluation_points
            ]

        monkeypatch.setattr(
            _sweep._execution,
            "execute_many",
            fake_execute_many,
        )

        result = _sweep.execute_sweep(
            func=double,
            values=values,
            plan=plan,
            stochastic_plan=None,
            make_execution_config=lambda: execution,
            requested_dispatch="thread",
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=_require_not_called,
            distribution_broadcast=_unexpected_distribution_broadcast,
            function_name="double",
        )

        request = seen["request"]
        assert request.execution is execution
        assert request.func is double
        assert [
            float(evaluation_point.call_arguments()["p"]["x"])
            for evaluation_point in request.evaluation_points
        ] == [
            0.0,
            1.0,
            2.0,
        ]
        np.testing.assert_allclose(result.raw, jnp.asarray([0.0, 2.0, 4.0]))

    def test_include_inputs_is_rejected_for_sweep(self):
        values = {"p": _numeric_record_batch("x", range(1))}
        plan = _plan(values)
        execution = _execution.WorkflowExecutionConfig(
            mode="sequential",
            name="identity",
        )

        with pytest.raises(NotImplementedError, match="include_inputs=True"):
            _sweep.execute_sweep(
                func=lambda p: p["x"],
                values=values,
                plan=plan,
                stochastic_plan=None,
                make_execution_config=lambda: execution,
                requested_dispatch="sequential",
                resolve_dispatch=lambda *args, **kwargs: "sequential",
                require_jax_traceable=_require_not_called,
                distribution_broadcast=_unexpected_distribution_broadcast,
                function_name="identity",
                include_inputs=True,
            )

    def test_nested_sweep_stacks_the_law_each_row_lifts(self):
        values = {
            "p": _numeric_record_batch("x", range(2)),
            "noise": Normal("noise", loc=0.0, scale=1.0),
        }
        plan = _plan(values)
        stochastic_plan = _stochastic_plan(values, 7)
        execution = _execution.WorkflowExecutionConfig(
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
            return Normal("row", loc=float(row_values["p"]["x"]), scale=1.0)

        result = _sweep.execute_sweep(
            func=lambda p, noise: p["x"] + noise,
            values=values,
            plan=plan,
            stochastic_plan=stochastic_plan,
            make_execution_config=lambda: execution,
            requested_dispatch="sequential",
            resolve_dispatch=lambda *args, **kwargs: "sequential",
            require_jax_traceable=_require_not_called,
            distribution_broadcast=distribution_broadcast,
            function_name="nested",
        )

        assert isinstance(result, DistributionBatch)
        assert result.batch_shape == (2,)
        assert [float(mean(output_name)) for output_name in result] == [
            0.0,
            1.0,
        ]
        assert calls == [
            {
                "x": 0.0,
                "plan": stochastic_plan,
                "logical_unit": stochastic_plan.logical_units[0],
                "include_inputs": False,
            },
            {
                "x": 1.0,
                "plan": stochastic_plan,
                "logical_unit": stochastic_plan.logical_units[1],
                "include_inputs": False,
            },
        ]
        assert result.provenance.operation == "workflow.nested"
        assert result.provenance.metadata["max_evaluations"] == 7


class TestPairedNumericSweeps:
    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    @pytest.mark.parametrize("shape", [(12,), (2, 3)])
    @pytest.mark.parametrize("record", [False, True])
    def test_paired_inputs_vectorize_and_match_sequential(self, dispatch, shape, record):
        a = NumericArrayBatch(
            jnp.arange(np.prod(shape)).reshape(shape),
            "row",
            label="a",
            element_spec=NumericArraySpec(()),
            axes_per_level=(len(shape),),
        )
        b = NumericArrayBatch(
            2 * a.as_jax(),
            "row",
            label="b",
            element_spec=NumericArraySpec(()),
            axes_per_level=(len(shape),),
        )
        if record:
            b = NumericRecordBatch(
                {"x": b.as_jax()},
                "row",
                label="b",
                element_spec=RecordSpec(x=()),
                axes_per_level=(len(shape),),
            )
        calls = []

        def body(a, b):
            calls.append(1)
            return a + (b["x"] if record else b)

        mapped = Function(body, label="sum", dispatch=dispatch)(a, b)
        assert len(calls) == 2
        sequential = Function(body, label="sum", dispatch="sequential")(a, b)
        np.testing.assert_array_equal(np.asarray(mapped), 3 * a.as_jax())
        np.testing.assert_array_equal(np.asarray(mapped), np.asarray(sequential))
        assert mapped.element_spec == sequential.element_spec
        assert mapped.level_names == sequential.level_names == ("row",)
        assert mapped.axis_groups == sequential.axis_groups == (shape,)

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_outer_product_remains_rowwise_or_is_refused(self, dispatch):
        a = NumericArrayBatch(jnp.arange(3.0), "a", label="a")
        b = NumericArrayBatch(jnp.arange(2.0), "b", label="b")
        wrapped = Function(lambda a, b: a + b, label="sum", dispatch=dispatch)
        if dispatch == "jax":
            with pytest.raises(ValueError, match="same levels"):
                wrapped(a, b)
        else:
            np.testing.assert_array_equal(
                np.asarray(wrapped(a, b)), np.arange(3.0)[:, None] + np.arange(2.0)
            )

    @pytest.mark.parametrize("record", [False, True])
    def test_mapped_storage_expands_each_level_before_flattening(self, record):
        data = jnp.arange(6.0).reshape(3, 1, 2)
        if record:
            batch = NumericRecordBatch(
                {"x": data}, ("site", "replicate"), label="data", element_spec=RecordSpec(x=(2,))
            )
        else:
            batch = NumericArrayBatch(
                data, ("site", "replicate"), label="data", element_spec=NumericArraySpec((2,))
            )
        actual = _sweep.mapped_storage(batch, 24, axis_groups=((2, 3), (4, 1)))
        if record:
            actual = actual["x"]
        expected = np.broadcast_to(np.asarray(data).reshape(1, 3, 1, 1, 2), (2, 3, 4, 1, 2))
        np.testing.assert_array_equal(actual, expected.reshape(24, 2))

    def test_mapped_storage_validates_row_count_for_empty_events(self):
        batch = NumericArrayBatch(
            jnp.empty((2, 0)), "row", label="empty", element_spec=NumericArraySpec((0,))
        )
        with pytest.raises(ValueError, match="row count 3"):
            _sweep.mapped_storage(batch, 3)


class TestNestedSweepProbes:
    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_numeric_sweep_probes_once_and_keeps_per_cell_draws(self, monkeypatch, dispatch):
        probes = _record_probes(monkeypatch)
        rows = NumericArrayBatch(jnp.arange(4.0), "row", label="shift")
        law = Normal("noise", 0.0, 1.0)
        wrapped = Function(
            lambda shift, noise: shift + noise,
            label="shift",
            dispatch=dispatch,
            max_evaluations=7,
        )
        with workflow_run(seed=12):
            mapped = wrapped(rows, law)
        assert probes == [(True, True)]
        with workflow_run(seed=12):
            sequential = wrapped.using(dispatch="sequential")(rows, law)
        for index in range(4):
            np.testing.assert_array_equal(
                np.asarray(mapped[index].atoms), np.asarray(sequential[index].atoms)
            )
        assert not np.array_equal(np.asarray(mapped[0].atoms), np.asarray(mapped[1].atoms) - 1)

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_fixed_numeric_branch_retains_per_cell_dispatch(self, monkeypatch, dispatch):
        probes = _record_probes(monkeypatch)

        def body(shift, noise):
            return noise + shift if float(shift) > 0 else noise - shift

        rows = NumericArrayBatch(jnp.array([-1.0, 2.0]), "row", label="shift")
        law = Normal("noise", 0.0, 1.0)
        wrapped = Function(body, label="branch", dispatch=dispatch, max_evaluations=7)
        with workflow_run(seed=12):
            actual = wrapped(rows, law)
        # The branch fails the probe with dynamic shifts, and each cell traces
        # with its own shift fixed.
        assert probes == [(True, False), (False, True), (False, True)]
        with workflow_run(seed=12):
            expected = wrapped.using(dispatch="sequential")(rows, law)
        with workflow_run(seed=12):
            noise = Function(
                lambda shift, noise: noise,
                label="noise",
                dispatch="sequential",
                max_evaluations=7,
            )(rows, law)
        # The else branch adds 1 at the shift -1, and the if branch adds 2 at the shift 2.
        for index, offset in enumerate([1.0, 2.0]):
            np.testing.assert_array_equal(
                np.asarray(actual[index].atoms), np.asarray(expected[index].atoms)
            )
            np.testing.assert_allclose(
                np.asarray(actual[index].atoms), np.asarray(noise[index].atoms) + offset
            )

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_untraceable_numeric_sweep_shares_its_first_cell_check(
        self, monkeypatch, caplog, dispatch
    ):
        probes = _record_probes(monkeypatch)
        rows = NumericArrayBatch(jnp.arange(5.0), "row", label="shift")
        law = Normal("noise", 0.0, 1.0)
        wrapped = Function(
            lambda shift, noise: float(noise) + float(shift),
            label="untraceable",
            dispatch=dispatch,
            max_evaluations=7,
        )
        if dispatch == "jax":
            with pytest.raises(ValueError, match="failed while tracing"):
                wrapped(rows, law)
        else:
            with (
                caplog.at_level(logging.INFO, logger="probpipe.functions._function"),
                workflow_run(seed=12),
            ):
                actual = wrapped(rows, law)
            logged = [record for record in caplog.records if "not JAX-traceable" in record.message]
            assert len(logged) == 1
            with workflow_run(seed=12):
                expected = wrapped.using(dispatch="sequential")(rows, law)
            for index in range(5):
                np.testing.assert_array_equal(
                    np.asarray(actual[index].atoms), np.asarray(expected[index].atoms)
                )
        # One probe with dynamic shifts and one with the first cell's shift fixed.
        assert probes == [(True, False), (False, False)]

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_object_cells_keep_their_own_traceability_check(self, monkeypatch, dispatch):
        probes = _record_probes(monkeypatch)

        def body(mode, noise):
            return float(noise) if mode.raw == "python" else noise + 1

        rows = OpaqueBatch(["mapped", "python"], "row", label="mode")
        law = Normal("noise", 0.0, 1.0)
        wrapped = Function(body, label="branch", dispatch=dispatch, max_evaluations=7)
        if dispatch == "jax":
            with pytest.raises(ValueError, match="failed while tracing"):
                wrapped(rows, law)
        else:
            with workflow_run(seed=12):
                actual = wrapped(rows, law)
            with workflow_run(seed=12):
                expected = wrapped.using(dispatch="sequential")(rows, law)
            for index in range(2):
                np.testing.assert_array_equal(
                    np.asarray(actual[index].atoms), np.asarray(expected[index].atoms)
                )
        assert probes == [(False, True), (False, False)]


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
            [
                Record(
                    {"x": jnp.asarray(float(i))},
                    label="p",
                )
                for i in range(n)
            ],
            level_name=level_name,
        )

    @staticmethod
    def _body(p):
        return RecordBatch.stack(
            [
                Record(
                    {"y": p["x"] * k},
                    label="r",
                )
                for k in (1.0, 2.0, 3.0)
            ],
            level_name="k",
        )

    def test_the_sweep_takes_the_mapped_path(self, monkeypatch):
        """The mapped executor runs, with no fall back to row-wise dispatch.

        The discriminating assertion: every other test in this class passes on
        the sequential path too, so only this one distinguishes vectorizing from
        agreeing with the oracle.
        """
        reached = []
        real = _sweep.execute_sweep_rows_jax

        def spy(**kwargs):
            reached.append(1)
            return real(**kwargs)

        monkeypatch.setattr(_sweep, "execute_sweep_rows_jax", spy)
        Function(self._body, label="swept")(self._rows(4))

        assert reached == [1]

    def test_the_levels_are_the_sweeps_then_the_bodys(self):
        out = Function(self._body, label="swept")(self._rows(4))

        assert out.level_names == ("row", "k")
        assert out.axis_groups == ((4,), (3,))

    def test_the_shape_agrees_with_the_columns_it_holds(self):
        """A batch whose spec its own columns contradict is the failure to avoid."""
        out = Function(self._body, label="swept")(self._rows(4))

        assert out.batch_shape == (4, 3)
        assert out.batch_size == 12
        assert np.shape(out._raw_column("y")) == (4, 3)

    def test_it_matches_sequential_dispatch(self):
        mapped = Function(self._body, label="swept")(self._rows(4))
        sequential = Function(self._body, label="swept", dispatch="sequential")(self._rows(4))

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
        real = _sweep.execute_sweep_rows_jax

        def spy(**kwargs):
            reached.append(1)
            return real(**kwargs)

        monkeypatch.setattr(_sweep, "execute_sweep_rows_jax", spy)

        grid = RecordBatch(
            {"x": jnp.arange(6.0).reshape(2, 3)},
            "cell",
            element_spec=RecordSpec(x=()),
            axes_per_level=(2,),
            label="batch",
        )

        mapped = Function(self._body, label="swept")(grid)
        sequential = Function(self._body, label="swept", dispatch="sequential")(grid)

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
                [
                    Record(
                        {"z": p["x"] * q["w"] * k},
                        label="r",
                    )
                    for k in (1.0, 2.0)
                ],
                level_name="k",
            )

        first = self._rows(2, level_name="a")
        second = RecordBatch.stack(
            [
                Record(
                    {"w": jnp.asarray(float(i))},
                    label="q",
                )
                for i in range(3)
            ],
            level_name="b",
        )

        mapped = Function(body, label="swept")(first, second)
        sequential = Function(body, label="swept", dispatch="sequential")(first, second)

        assert mapped.level_names == ("a", "b", "k")
        assert mapped.axis_groups == ((2,), (3,), (2,))
        assert mapped.batch_shape == (2, 3, 2)
        np.testing.assert_allclose(np.asarray(mapped["z"]), np.asarray(sequential["z"]))

    def test_a_nested_element_survives_the_transform(self):
        """Columns are leaf-keyed, so a nested record needs no special case."""

        def body(p):
            return RecordBatch.stack(
                [
                    Record(
                        {"inner": {"y": p["x"] * k}},
                        label="r",
                    )
                    for k in (1.0, 2.0)
                ],
                level_name="k",
            )

        mapped = Function(body, label="swept")(self._rows(3))
        sequential = Function(body, label="swept", dispatch="sequential")(self._rows(3))

        assert mapped.level_names == ("row", "k")
        assert mapped.element_spec == sequential.element_spec
        np.testing.assert_allclose(
            np.asarray(mapped._raw_column("inner/y")),
            np.asarray(sequential._raw_column("inner/y")),
        )

    def test_the_carrier_does_not_reach_the_caller(self):
        """It is wrapped and unwrapped inside one call, by construction."""
        out = Function(self._body, label="swept")(self._rows(4))

        assert not isinstance(out, _MappedBatchColumns)
        assert isinstance(out, RecordBatch)

    def test_a_raw_vmap_returning_a_batch_is_still_refused(self):
        """The hook is routed around, not softened.

        Only a caller that knows which axis it added, and what to call the level
        it stands for, may rebuild across one. The executor knows both; a raw
        ``vmap`` knows neither, and is refused.
        """
        with pytest.raises(ValueError, match="To add a level, build a new"):
            jax.vmap(
                lambda v: RecordBatch.stack(
                    [
                        Record(
                            {"y": v * k},
                            label="r",
                        )
                        for k in (1.0, 2.0)
                    ],
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
        {"x": values},
        levels,
        element_spec=RecordSpec(x=NumericArraySpec((), dtype=np.float32)),
        axes_per_level=axes_per_level,
        label="inputs",
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
        value = NumericArray(
            native,
            spec=declared,
            label="original",
        )

        result = Function(lambda row: value, label="repeated", dispatch=dispatch)(source)

        assert isinstance(result, NumericArrayBatch)
        # The rows are stored as one JAX array, whose dtype the declaration records.
        assert result.element_spec == NumericArraySpec(
            event_shape, dtype=np.float32, support=positive
        )
        assert result.raw.dtype == np.float32
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(
            np.asarray(result), np.broadcast_to(native, (*source.batch_shape, *event_shape))
        )
        assert value.spec is declared
        assert value.spec.shape == declared_shape
        assert value.raw is native
        assert value.label == "original"
        assert value.provenance is None

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    def test_symbolic_and_concrete_numeric_rows_agree_after_binding(self, reverse):
        declarations = [
            NumericArraySpec(shape, dtype=np.float32, support=positive)
            for shape in (("d",), ("other",), (2,))
        ]
        outputs = [
            NumericArray(
                jnp.full((2,), i + 1, dtype=jnp.float32),
                spec=spec,
                label="row",
            )
            for i, spec in enumerate(declarations)
        ]
        if reverse:
            outputs.reverse()
        source = _numeric_record_batch("x", range(3))

        result = Function(lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential")(
            source
        )

        assert result.element_spec == NumericArraySpec((2,), dtype=np.float32, support=positive)
        assert result.raw.dtype == np.float32
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
                value,
                spec=declared if representation == "declared" else None,
                label="row-result",
            )

        result = Function(body, label="shift", dispatch=dispatch)(source)

        expected = np.asarray(source["x"]) * 2 + 1
        if event_shape:
            expected = np.stack([expected, expected + 1], axis=-1)
        # The rows are stored as one JAX array, whose dtype the declaration records.
        expected_spec = NumericArraySpec(
            event_shape,
            dtype=np.float32,
            support=positive if representation == "declared" else None,
        )
        assert isinstance(result, NumericArrayBatch)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        assert result.element_spec == expected_spec
        assert result.raw.dtype == np.float32
        assert result.provenance is not None
        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_nested_density_results_compose_after_the_sweep(self, numeric_sweep_source, dispatch):
        source = numeric_sweep_source
        law = Normal("x", 0.0, 1.0)
        result = Function(lambda row: log_prob(law, row["x"]), label="score", dispatch=dispatch)(
            source
        )

        expected = -0.5 * np.asarray(source["x"], dtype=np.float64) ** 2 - 0.5 * np.log(2 * np.pi)
        assert isinstance(result, NumericArrayBatch)
        assert result.axis_groups == source.axis_groups
        assert result.level_names == source.level_names
        # log_prob declares only the shape, so completion takes the produced dtype (II.2).
        assert result.element_spec == NumericArraySpec((), dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-7, atol=1e-7)

        shifted = Function(lambda value: value + 1, label="function", dispatch="sequential")(result)
        assert shifted.axis_groups == source.axis_groups
        assert shifted.level_names == source.level_names
        np.testing.assert_allclose(np.asarray(shifted), expected + 1, rtol=2e-7, atol=1e-7)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_native_numeric_results_keep_their_source(self, numeric_sweep_source, dispatch):
        pd = pytest.importorskip("pandas")
        native = pd.Series([1.0, 2.0], index=[5, 7])
        declared = NumericArraySpec((2,), dtype=np.float64, support=positive)
        value = NumericArray(
            native,
            spec=declared,
            label="original",
        )

        result = Function(lambda row: value, label="repeated", dispatch=dispatch)(
            numeric_sweep_source
        )

        # The rows are stored as one JAX array, whose dtype the declaration records.
        assert result.raw.dtype == np.float32
        assert result.element_spec == NumericArraySpec((2,), dtype=np.float32, support=positive)
        assert result.axis_groups == numeric_sweep_source.axis_groups
        assert result.level_names == numeric_sweep_source.level_names
        np.testing.assert_array_equal(
            np.asarray(result), np.broadcast_to([1.0, 2.0], (*numeric_sweep_source.batch_shape, 2))
        )
        assert value.raw is native
        assert value.label == "original"
        assert value.spec == declared
        assert value.provenance is None

    def test_disagreeing_numeric_row_supports_are_refused(self):
        declared = NumericArraySpec((), dtype=np.float64, support=positive)
        first = NumericArray(
            jnp.asarray(1.0),
            spec=declared,
            label="first",
        )
        second = NumericArray(
            jnp.asarray(2.0),
            spec=NumericArraySpec((), dtype=np.float64),
            label="second",
        )
        source = _numeric_record_batch("x", range(2))

        with pytest.raises(
            ValueError, match=r"different\(nr\): the rows returned arrays with different specs"
        ):
            Function(
                lambda row: first if float(row["x"]) == 0 else second,
                label="different",
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
            first = NumericArray(
                native,
                label="native",
            )
            second = NumericArray(
                jnp.full(event_shape, 2.5, dtype=jnp.float32),
                label="jax",
            )
            outputs = [second, first] if reverse else [first, second]
            source = _numeric_record_batch("x", range(2))

            result = Function(
                lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential"
            )(source)

        expected = np.stack([np.full(event_shape, 1.25), np.full(event_shape, 2.5)])
        if reverse:
            expected = expected[::-1]
        stored = np.dtype(np.float64 if x64 else np.float32)
        assert result.element_spec == NumericArraySpec(event_shape, dtype=stored)
        assert result.dtype == stored
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)
        assert first.raw is native
        assert first.spec.dtype == np.dtype(np.float64)
        assert second.spec.dtype == np.dtype(np.float32)

    @pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
    @pytest.mark.parametrize(
        "other_dtype, support, expected_dtype",
        [(np.float32, positive, np.float32), (None, positive, np.float32)],
        ids=["promoted", "unspecified"],
    )
    def test_numeric_dtype_promotion_preserves_shared_support(
        self, reverse, other_dtype, support, expected_dtype
    ):
        first = NumericArray(
            jnp.asarray(1.0),
            spec=NumericArraySpec((), dtype=np.float64, support=support),
            label="first",
        )
        second = NumericArray(
            jnp.asarray(2.0),
            spec=NumericArraySpec((), dtype=other_dtype, support=support),
            label="second",
        )
        outputs = [second, first] if reverse else [first, second]

        result = Function(lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential")(
            _numeric_record_batch("x", range(2))
        )

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
                jnp.asarray(value, dtype=dtype),
                spec=NumericArraySpec((), dtype=dtype, support=positive),
                label="row",
            )
            for value, dtype in ((1, jnp.bfloat16), (2, other_dtype))
        ]
        if reverse:
            outputs.reverse()
        source = _numeric_record_batch("x", range(2))

        result = Function(lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential")(
            source
        )

        assert result.element_spec == NumericArraySpec((), dtype=expected_dtype, support=positive)
        assert result.raw.dtype == np.dtype(expected_dtype)
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
                    jnp.asarray(1, dtype=dtype),
                    spec=NumericArraySpec((), dtype=dtype, support=positive),
                    label="row",
                )
                for dtype in order
            ]
            batch = Function(
                lambda row, outputs=outputs: outputs[int(row["x"])],
                label="mixed",
                dispatch="sequential",
            )(source)
            batches.append(batch)
            assert batch.element_spec == NumericArraySpec((), dtype=np.float16, support=positive)
            assert batch.raw.dtype == np.float16
            assert batch.level_names == source.level_names
            assert batch.axis_groups == source.axis_groups
            np.testing.assert_array_equal(np.asarray(batch), [1, 1, 1])

        result = Function(
            lambda row: batches[int(row["x"])], label="combined", dispatch="sequential"
        )(_numeric_record_batch("x", range(2), level_name="outer"))

        assert result.element_spec == NumericArraySpec((), dtype=np.float16, support=positive)
        assert result.raw.dtype == np.float16
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
            jnp.asarray(1, dtype=jnp.int32),
            spec=NumericArraySpec((), dtype=np.int32),
            label="integer",
        )
        floating = jnp.asarray(2.5, dtype=jnp.float32)
        outputs = [
            integer,
            NumericArray(
                floating,
                label="float",
            )
            if tracked_float
            else floating,
        ]
        if reverse:
            outputs.reverse()

        result = Function(lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential")(
            _numeric_record_batch("x", range(2))
        )

        # The rows are stored as one JAX array, whose dtype the declaration records.
        assert result.element_spec == NumericArraySpec((), dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(result), [2.5, 1.0] if reverse else [1.0, 2.5])

    @pytest.mark.parametrize("raw_value", [-1.0, 1.0], ids=["negative", "positive"])
    @pytest.mark.parametrize("tracked_first", [False, True], ids=["raw-first", "tracked-first"])
    @pytest.mark.parametrize("event_shape", [(), (2,)], ids=["scalar", "vector"])
    def test_mixed_numeric_rows_infer_the_aggregate_spec(
        self, raw_value, tracked_first, event_shape
    ):
        declared = NumericArraySpec(event_shape, dtype=np.float64, support=positive)
        value = NumericArray(
            jnp.full(event_shape, 2.0),
            spec=declared,
            label="held",
        )
        raw = jnp.full(event_shape, raw_value) if event_shape else raw_value
        source = _numeric_record_batch("x", range(2))
        result = Function(
            lambda row: value if (float(row["x"]) == 0) == tracked_first else raw,
            label="mixed",
            dispatch="sequential",
        )(source)

        expected = np.stack([np.full(event_shape, raw_value), np.full(event_shape, 2.0)])
        if tracked_first:
            expected = expected[::-1]
        assert result.element_spec == NumericArraySpec(event_shape, dtype=result.raw.dtype)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)

    @pytest.mark.parametrize("raw_position", [0, 1, 2], ids=["raw-first", "raw-middle", "raw-last"])
    def test_mixed_numeric_rows_do_not_adopt_conflicting_partial_declarations(self, raw_position):
        outputs = [
            NumericArray(
                jnp.asarray(2.0),
                spec=NumericArraySpec((), dtype=np.float64, support=positive),
                label="positive",
            ),
            NumericArray(
                jnp.asarray(3.0),
                spec=NumericArraySpec(()),
                label="unconstrained",
            ),
        ]
        outputs.insert(raw_position, -1.0)
        source = _numeric_record_batch("x", range(3))

        result = Function(lambda row: outputs[int(row["x"])], label="mixed", dispatch="sequential")(
            source
        )

        expected = [2.0, 3.0]
        expected.insert(raw_position, -1.0)
        assert result.element_spec == NumericArraySpec((), dtype=result.raw.dtype)
        assert result.batch_shape == source.batch_shape
        assert result.level_names == source.level_names
        assert result.axis_groups == source.axis_groups
        np.testing.assert_array_equal(np.asarray(result), expected)


class TestASweptBatchOfArrays:
    @staticmethod
    def _grid() -> NumericArrayBatch:
        return NumericArrayBatch(
            jnp.arange(24.0).reshape(2, 4, 3),
            ("rows", "cols"),
            element_spec=NumericArraySpec((3,)),
            axes_per_level=(1, 1),
            label="grid",
        )

    def test_the_mapped_storage_flattens_the_levels_in_row_major_order(self):
        storage = _sweep.mapped_storage(self._grid(), 8)
        np.testing.assert_array_equal(np.asarray(storage), np.arange(24.0).reshape(8, 3))

    def test_a_batch_of_records_maps_one_column_per_field(self):
        source = NumericRecordBatch(
            {"a": jnp.arange(4.0).reshape(2, 2), "b": jnp.ones((2, 2, 3))},
            ("rows", "cols"),
            axes_per_level=(1, 1),
            label="rows",
        )
        storage = _sweep.mapped_storage(source, 4)
        assert list(storage) == ["a", "b"]
        assert (storage["a"].shape, storage["b"].shape) == ((4,), (4, 3))

    def test_a_batch_that_stores_objects_raises_type_error(self):
        laws = DistributionBatch(
            [Normal("x", 0.0, 1.0), Normal("x", 1.0, 1.0)],
            "law",
            label="laws",
        )
        with pytest.raises(TypeError, match="DistributionBatch stores objects"):
            _sweep.mapped_storage(laws, 2)

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_every_dispatch_sweeps_it_alike_at_its_levels(self, dispatch):
        rows: list = []

        def norm(v):
            rows.append(v)
            return jnp.sum(jnp.asarray(v) ** 2)

        result = Function(norm, label="norm", dispatch=dispatch)(self._grid())

        assert all(isinstance(row, NumericArray) for row in rows)
        assert len(rows) == 8 if dispatch == "sequential" else len(rows) < 8
        assert (result.level_names, result.axis_groups) == (("rows", "cols"), ((2,), (4,)))
        np.testing.assert_allclose(
            np.asarray(result), np.sum(np.arange(24.0).reshape(2, 4, 3) ** 2, axis=-1)
        )


class TestANestedSweepOfRecordRows:
    """A call over a batch of records and a law lifts the law at each record row.

    The body receives each row as a record and reads its fields. The rows are
    fixed within each row's lift, so the auto and jax dispatches map the law's
    atoms or draws at each row.
    """

    @staticmethod
    def _rows(fields: tuple[str, ...] = ("x", "y")) -> NumericRecordBatch:
        columns = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])}
        return NumericRecordBatch(
            {field: columns[field] for field in fields}, ("row",), label="rows"
        )

    @staticmethod
    def _shift(row, mu):
        return row["x"] * mu + row["y"]

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "jax"])
    def test_each_row_lifts_the_law_through_its_fields(self, dispatch):
        seen = []

        def shift(row, mu):
            seen.append((type(row), isinstance(mu, jax.core.Tracer)))
            return self._shift(row, mu)

        law = EmpiricalDistribution(jnp.array([1.0, 2.0, 4.0]), output_name="mu", label="prior")
        result = Function(shift, label="shift", dispatch=dispatch)(self._rows(), law)

        assert isinstance(result, DistributionBatch)
        assert (result.batch_shape, result.level_names) == ((3,), ("row",))
        # Each row's atoms are x * mu + y at the law's atoms 1, 2, and 4.
        expected = [[11.0, 12.0, 14.0], [22.0, 24.0, 28.0], [33.0, 36.0, 42.0]]
        np.testing.assert_allclose([atom_leaves(row_law)[0] for row_law in result], expected)
        assert {row_type for row_type, _ in seen} == {NumericRecord}
        assert {traced for _, traced in seen} == {dispatch != "sequential"}

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_a_mapped_dispatch_draws_the_atoms_sequential_dispatch_draws(self, dispatch):
        def lift(dispatch):
            with workflow_run(seed=0):
                return Function(self._shift, label="shift", dispatch=dispatch, max_evaluations=6)(
                    self._rows(), Normal("mu", 0.0, 1.0)
                )

        mapped, sequential = lift(dispatch), lift("sequential")

        assert (mapped.batch_shape, mapped.level_names) == (sequential.batch_shape, ("row",))
        np.testing.assert_allclose(
            [atom_leaves(row_law)[0] for row_law in mapped],
            [atom_leaves(row_law)[0] for row_law in sequential],
            rtol=1e-6,
        )

    @pytest.mark.parametrize("fields", [("x",), ("x", "y")], ids=["one-field", "two-fields"])
    def test_a_row_displays_as_the_call_of_its_record(self, fields):
        law = Normal("mu", 0.0, 1.0, label="prior")
        result = Function(lambda row, mu: row["x"] * mu, label="scale", dispatch="sequential")(
            self._rows(fields), law
        )

        assert [str(row_law) for row_law in result] == [
            f"scale(rows[row={i}], mu ~ prior)" for i in range(3)
        ]


class TestARecordedDtypeIsTheStoredDtype:
    """A sweep or a lift stacks its rows into one array, whose dtype its declaration records.

    JAX's 64-bit mode is off in the suite, so a 64-bit row is stored as a 32-bit one.
    """

    @staticmethod
    def _rows():
        return NumericRecordBatch(
            {"x": jnp.arange(3.0)},
            "row",
            element_spec=RecordSpec(x=()),
            label="rows",
        )

    @staticmethod
    def _bodies():
        return {
            "raw": lambda row: np.arange(3.0),
            "tracked": lambda row: NumericArray(
                np.arange(3.0),
                label="stored",
            ),
        }

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax"])
    @pytest.mark.parametrize("body", ["raw", "tracked"])
    @pytest.mark.parametrize(
        "declaration",
        [
            None,
            NumericArraySpec((3,)),
            OutputSpec(value=None),
            NumericArraySpec((3,), dtype="float64"),
        ],
        ids=["undeclared", "shape-only", "type-hole", "float64"],
    )
    def test_a_sweep_of_64_bit_rows_records_the_stored_dtype(self, dispatch, body, declaration):
        assert not jax.config.jax_enable_x64
        wrapped = Function(
            self._bodies()[body],
            output_spec=declaration
            if isinstance(declaration, OutputSpec) or declaration is None
            else OutputSpec.default(declaration, output_name="f"),
            dispatch=dispatch,
            label="f",
        )
        result = wrapped(self._rows())
        assert np.asarray(result.raw).dtype == np.float32
        assert result.element_spec == NumericArraySpec((3,), dtype="float32")

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax"])
    @pytest.mark.parametrize(
        "declaration",
        [None, NumericArraySpec((3,)), NumericArraySpec((3,), dtype="float64")],
        ids=["undeclared", "shape-only", "float64"],
    )
    def test_a_lift_of_64_bit_draws_records_the_stored_dtype(self, dispatch, declaration):
        wrapped = Function(
            lambda x: np.arange(3.0),
            output_spec=OutputSpec.default(declaration, output_name="f"),
            dispatch=dispatch,
            max_evaluations=5,
            label="f",
        )
        result = wrapped(EmpiricalDistribution(jnp.arange(3.0), output_name="x"))
        assert np.asarray(result.atoms.raw).dtype == np.float32
        assert result.atoms.element_spec.dtype == np.float32
        assert result.event_spec.outputs["f"].dtype == np.float32

    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "jax"])
    def test_a_declared_64_bit_record_field_records_the_stored_dtype(self, dispatch):
        wrapped = Function(
            lambda row: {"y": np.arange(3.0)},
            output_spec=RecordSpec(y=NumericArraySpec((3,), dtype="float64")),
            dispatch=dispatch,
            label="f",
        )
        result = wrapped(self._rows())
        assert result._raw_column("y").dtype == np.float32
        assert result.element_spec["y"] == NumericArraySpec((3,), dtype="float32")

    @pytest.mark.parametrize("output", ["array", "record"])
    def test_sequential_and_jax_sweeps_agree_on_a_dtype_record_leaves_leave_unset(self, output):
        """A record whose leaves declare no dtype gives the body views that declare none."""
        rows = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"value": jnp.ones((2,)) * i},
                    label="row",
                )
                for i in range(3)
            ],
            level_name="draw",
        )
        assert rows.element_spec["value"].dtype is None

        def body(row):
            value = row["value"] + 1
            return value if output == "array" else {"y": value}

        specs = {
            dispatch: Function(
                body,
                dispatch=dispatch,
                label="f",
            )(rows).element_spec
            for dispatch in ("sequential", "thread", "jax")
        }
        assert specs["sequential"] == specs["thread"] == specs["jax"]
        if output == "array":
            assert specs["jax"] == NumericArraySpec((2,), dtype="float32")
