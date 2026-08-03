"""Tests for Function distribution-only broadcast helpers."""

from __future__ import annotations

import inspect
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    EmpiricalDistribution,
    Normal,
    ProductDistribution,
    Record,
)
from probpipe.core import _workflow_call, _workflow_distribution_broadcast, _workflow_execution
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan
from probpipe.core.config import WorkflowKind


def _execution_config(
    *,
    mode: _workflow_execution.WorkflowExecutionMode = "sequential",
    max_workers: int | None = None,
    name: str = "workflow",
) -> _workflow_execution.WorkflowExecutionConfig:
    return _workflow_execution.WorkflowExecutionConfig(
        mode=mode,
        max_workers=max_workers,
        name=name,
    )


def _key_source(seed: int = 0, events=None):
    key = jax.random.PRNGKey(seed)

    def get_key(event):
        nonlocal key
        if events is not None:
            events.append(event)
        key, subkey = jax.random.split(key)
        return subkey

    return get_key


def _require_not_called(*args, **kwargs):
    raise AssertionError("JAX traceability should not be required")


def _resolve_to(dispatch: str):
    def resolve_dispatch(values, broadcast_args, *, jax_supported):
        return dispatch

    return resolve_dispatch


def _ref(name: str) -> _workflow_call.WorkflowInputRef:
    return _workflow_call.WorkflowInputRef(name)


def _stochastic_plan(values, n_broadcast_samples):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(signature)
    broadcast_plan = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast_plan, n_broadcast_samples)


class _RecordingNormal(Normal):
    def __init__(self, sample_calls, *, name):
        self.sample_calls = sample_calls
        super().__init__(loc=0.0, scale=1.0, name=name)

    def _sample(self, key, sample_shape=()):
        self.sample_calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class TestExecuteDistributionBroadcast:
    def test_direct_aliases_sample_one_root_and_stay_diagonal(self):
        sample_calls = []
        events = []
        shared = _RecordingNormal(sample_calls, name="shared")
        values = {"first": shared, "second": shared}

        plan = _stochastic_plan(values, 12)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda first, second: first - second,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(17, events),
            make_execution_config=lambda: _execution_config(name="difference"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="difference",
            workflow_kind=WorkflowKind.OFF,
        )

        assert len(sample_calls) == 1
        assert len(events) == 1
        np.testing.assert_array_equal(result.input_samples["first"], result.input_samples["second"])
        np.testing.assert_allclose(result.samples, 0.0)

    def test_equal_but_distinct_sources_sample_independently(self):
        first_calls = []
        second_calls = []
        events = []
        values = {
            "first": _RecordingNormal(first_calls, name="same"),
            "second": _RecordingNormal(second_calls, name="same"),
        }

        plan = _stochastic_plan(values, 12)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda first, second: first - second,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(19, events),
            make_execution_config=lambda: _execution_config(name="difference"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="difference",
            workflow_kind=WorkflowKind.OFF,
        )

        assert len(first_calls) == len(second_calls) == 1
        assert len(events) == 2
        assert not np.array_equal(result.input_samples["first"], result.input_samples["second"])

    def test_root_and_nested_view_use_the_same_sampled_realization(self):
        joint = ProductDistribution(
            nested={"leaf": Normal(loc=0.0, scale=1.0, name="leaf")},
            other=Normal(loc=3.0, scale=1.0, name="other"),
        )
        values = {"root": joint, "leaf": joint["nested"]["leaf"]}

        plan = _stochastic_plan(values, 8)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda root, leaf: root["nested/leaf"] - leaf,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(23),
            make_execution_config=lambda: _execution_config(name="difference"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="difference",
            workflow_kind=WorkflowKind.OFF,
        )

        np.testing.assert_allclose(result.samples, 0.0)

    def test_weighted_empirical_aliases_enumerate_once(self):
        shared = EmpiricalDistribution(
            jnp.asarray([1.0, 4.0]),
            weights=jnp.asarray([0.2, 0.8]),
            name="shared",
        )
        values = {"first": shared, "second": shared}

        plan = _stochastic_plan(values, 8)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda first, second: first - second,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_require_not_called,
            make_execution_config=lambda: _execution_config(name="difference"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="difference",
            workflow_kind=WorkflowKind.OFF,
        )

        assert result.num_atoms == 2
        np.testing.assert_array_equal(result.input_samples["first"], jnp.asarray([1.0, 4.0]))
        np.testing.assert_array_equal(result.input_samples["first"], result.input_samples["second"])
        np.testing.assert_allclose(result.samples, 0.0)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.2, 0.8]))

    def test_weighted_record_root_and_view_enumerate_once(self):
        shared = EmpiricalDistribution(
            Record(
                "draws",
                x=jnp.asarray([1.0, 4.0]),
                y=jnp.asarray([10.0, 40.0]),
            ),
            weights=jnp.asarray([0.3, 0.7]),
            name="shared",
        )
        values = {"root": shared, "x": shared["x"]}

        plan = _stochastic_plan(values, 8)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda root, x: root["x"] - x,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_require_not_called,
            make_execution_config=lambda: _execution_config(name="difference"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="difference",
            workflow_kind=WorkflowKind.OFF,
        )

        assert result.num_atoms == 2
        np.testing.assert_allclose(result.samples, 0.0)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.3, 0.7]))

    def test_sample_path_uses_execution_request(self, monkeypatch):
        values = {
            "x": Normal(loc=0.0, scale=1.0, name="x"),
            "offset": 2.0,
        }
        execution = _execution_config(mode="thread", max_workers=2, name="shift")
        plan = _stochastic_plan(values, 5)
        seen = {}

        def shift(x, offset):
            return x + offset

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**call_values) for call_values in request.call_value_list]

        monkeypatch.setattr(
            _workflow_distribution_broadcast._workflow_execution,
            "execute_many",
            fake_execute_many,
        )

        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=shift,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(0),
            make_execution_config=lambda: execution,
            requested_dispatch="thread",
            resolve_dispatch=_resolve_to("thread"),
            require_jax_traceable=_require_not_called,
            workflow_name="shift",
            workflow_kind=WorkflowKind.OFF,
        )

        request = seen["request"]
        assert isinstance(result, BroadcastDistribution)
        assert request.func is shift
        assert request.execution is execution
        assert len(request.call_value_list) == 5
        assert all(call_values["offset"] == 2.0 for call_values in request.call_value_list)
        assert all(
            not isinstance(call_values["x"], Normal) for call_values in request.call_value_list
        )
        assert result.provenance.metadata == {
            "dispatch": "thread",
            "orchestrate": "off",
            "n_samples": 5,
            "func": "shift",
            "broadcast_args": ["x"],
        }

    def test_empirical_enumeration_preserves_alignment_and_weights(self):
        values = {
            "x": EmpiricalDistribution(
                jnp.asarray([[1.0], [2.0]]),
                weights=jnp.asarray([0.25, 0.75]),
                name="x",
            ),
            "y": EmpiricalDistribution(
                jnp.asarray([[10.0], [20.0]]),
                weights=jnp.asarray([0.4, 0.6]),
                name="y",
            ),
        }

        def add(x, y):
            return x + y

        plan = _stochastic_plan(values, 10)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=add,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_require_not_called,
            make_execution_config=lambda: _execution_config(name="add"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="add",
            workflow_kind=WorkflowKind.OFF,
        )

        assert result.num_atoms == 4
        np.testing.assert_allclose(
            result.input_samples["x"],
            jnp.asarray([[1.0], [1.0], [2.0], [2.0]]),
        )
        np.testing.assert_allclose(
            result.input_samples["y"],
            jnp.asarray([[10.0], [20.0], [10.0], [20.0]]),
        )
        np.testing.assert_allclose(
            result.samples,
            jnp.asarray([[11.0], [21.0], [12.0], [22.0]]),
        )
        np.testing.assert_allclose(
            result.weights,
            jnp.asarray([0.1, 0.15, 0.3, 0.45]),
            atol=1e-6,
        )

    def test_jax_path_vectorizes_samples_and_outputs(self):
        values = {"x": Normal(loc=1.0, scale=0.5, name="x")}
        seen = {"required": False}

        def double(x):
            return 2.0 * x

        def require_jax_traceable(values, broadcast_args):
            seen["required"] = True

        plan = _stochastic_plan(values, 6)
        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=double,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(2),
            make_execution_config=lambda: _execution_config(name="double"),
            requested_dispatch="jax",
            resolve_dispatch=_resolve_to("jax"),
            require_jax_traceable=require_jax_traceable,
            workflow_name="double",
            workflow_kind=WorkflowKind.OFF,
        )

        assert seen["required"] is True
        assert result.num_atoms == 6
        np.testing.assert_allclose(result.samples, result.input_samples["x"] * 2.0)

    def test_jax_prefect_path_requires_prefect(self, monkeypatch):
        values = {"x": Normal(loc=1.0, scale=0.5, name="x")}
        monkeypatch.setattr(_workflow_distribution_broadcast, "task", None)
        monkeypatch.setattr(_workflow_distribution_broadcast, "flow", None)
        plan = _stochastic_plan(values, 6)

        with pytest.raises(
            RuntimeError,
            match="Prefect task or flow execution was requested",
        ):
            _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values=values,
                stochastic_plan=plan,
                logical_unit=plan.logical_units[0],
                include_inputs=True,
                get_key=_key_source(2),
                make_execution_config=lambda: _execution_config(name="identity"),
                requested_dispatch="jax",
                resolve_dispatch=_resolve_to("jax"),
                require_jax_traceable=lambda values, broadcast_args: None,
                workflow_name="identity",
                workflow_kind=WorkflowKind.TASK,
            )

    def test_same_parent_views_share_parent_sample(self):
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
        )
        view_x = joint["x"]
        values = {"a": view_x, "b": view_x}

        plan = _stochastic_plan(values, 8)
        sampled = _workflow_distribution_broadcast._sample_planned_source_groups(
            plan,
            plan.source_groups,
            (8,),
            plan.logical_units[0],
            _key_source(3),
        )

        np.testing.assert_allclose(sampled[_ref("a")], sampled[_ref("b")])

    def test_each_sampled_source_group_claims_and_samples_once(self):
        first_calls = []
        second_calls = []
        values = {
            "first": _RecordingNormal(first_calls, name="first"),
            "second": _RecordingNormal(second_calls, name="second"),
        }
        plan = _stochastic_plan(values, 11)
        assert plan.sample_shape is not None
        events = []

        sampled = _workflow_distribution_broadcast._sample_planned_source_groups(
            plan,
            plan.source_groups,
            plan.sample_shape,
            plan.logical_units[0],
            _key_source(4, events),
        )

        assert tuple(sampled) == (_ref("first"), _ref("second"))
        assert [sample_shape for _key, sample_shape in first_calls] == [(11,)]
        assert [sample_shape for _key, sample_shape in second_calls] == [(11,)]
        assert events == list(plan.random_events)

    def test_mixed_plan_claims_only_the_sampled_source_event(self):
        sampled_calls = []
        values = {
            "exact": EmpiricalDistribution(
                jnp.asarray([1.0, 2.0]),
                name="exact",
            ),
            "sampled": _RecordingNormal(sampled_calls, name="sampled"),
        }
        plan = _stochastic_plan(values, 5)
        events = []

        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=lambda exact, sampled: exact + sampled,
            values=values,
            stochastic_plan=plan,
            logical_unit=plan.logical_units[0],
            include_inputs=True,
            get_key=_key_source(6, events),
            make_execution_config=lambda: _execution_config(name="add"),
            requested_dispatch="sequential",
            resolve_dispatch=_resolve_to("sequential"),
            require_jax_traceable=_require_not_called,
            workflow_name="add",
            workflow_kind=WorkflowKind.OFF,
        )

        assert result.num_atoms == 4
        assert [sample_shape for _key, sample_shape in sampled_calls] == [(4,)]
        assert events == list(plan.random_events)
        assert events[0].stochastic_source_id == ("source-group", 1)

    @pytest.mark.parametrize(
        ("n_broadcast_samples", "error_type", "message"),
        [
            (True, TypeError, "n_broadcast_samples must be an integer"),
            (False, TypeError, "n_broadcast_samples must be an integer"),
            (2.5, TypeError, "n_broadcast_samples must be an integer"),
            (0, ValueError, "n_broadcast_samples must be a positive integer"),
            (-1, ValueError, "n_broadcast_samples must be a positive integer"),
        ],
    )
    def test_invalid_n_broadcast_samples_raise(
        self,
        n_broadcast_samples,
        error_type,
        message,
    ):
        values = {"x": Normal(loc=0.0, scale=1.0, name="x")}
        invalid_plan = replace(
            _stochastic_plan(values, 5),
            n_broadcast_samples=n_broadcast_samples,
        )

        with pytest.raises(error_type, match=message):
            _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values=values,
                stochastic_plan=invalid_plan,
                logical_unit=invalid_plan.logical_units[0],
                include_inputs=True,
                get_key=_key_source(4),
                make_execution_config=lambda: _execution_config(name="identity"),
                requested_dispatch="sequential",
                resolve_dispatch=_resolve_to("sequential"),
                require_jax_traceable=_require_not_called,
                workflow_name="identity",
                workflow_kind=WorkflowKind.OFF,
            )

    def test_low_n_broadcast_samples_warns(self):
        values = {"x": Normal(loc=0.0, scale=1.0, name="x")}
        plan = _stochastic_plan(values, 3)
        with pytest.warns(UserWarning, match="n_broadcast_samples=3 is too low"):
            result = _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values=values,
                stochastic_plan=plan,
                logical_unit=plan.logical_units[0],
                include_inputs=True,
                get_key=_key_source(5),
                make_execution_config=lambda: _execution_config(name="identity"),
                requested_dispatch="sequential",
                resolve_dispatch=_resolve_to("sequential"),
                require_jax_traceable=_require_not_called,
                workflow_name="identity",
                workflow_kind=WorkflowKind.OFF,
            )

        assert isinstance(result, BroadcastDistribution)
        assert result.num_atoms == 3

    def test_executor_has_no_empirical_replanning_helper(self):
        assert not hasattr(_workflow_distribution_broadcast, "_split_empirical_args")


class TestIndexSampleHelper:
    """Direct unit tests for the module-level ``_index_sample`` helper."""

    def test_bare_array(self):
        s = jnp.arange(20.0).reshape(5, 4)
        for i in range(5):
            np.testing.assert_array_equal(
                _workflow_distribution_broadcast._index_sample(s, i),
                s[i],
            )

    def test_bare_array_1d(self):
        s = jnp.arange(10.0)

        assert float(_workflow_distribution_broadcast._index_sample(s, 3)) == 3.0

    def test_single_field_record_unwraps(self):
        from probpipe import Record

        s = Record("r", x=jnp.arange(15.0).reshape(5, 3))

        for i in range(5):
            row = _workflow_distribution_broadcast._index_sample(s, i)
            assert not hasattr(row, "fields")
            np.testing.assert_array_equal(row, s["x"][i])

    def test_multi_field_record_returns_per_row_numeric_record(self):
        from probpipe import NumericRecord, Record

        s = Record(
            "r",
            mu=jnp.arange(5.0),
            sigma=jnp.arange(5.0) + 100.0,
        )

        row = _workflow_distribution_broadcast._index_sample(s, 2)

        assert isinstance(row, NumericRecord)
        assert row.fields == ("mu", "sigma")
        assert float(row["mu"]) == 2.0
        assert float(row["sigma"]) == 102.0

    def test_multi_field_record_with_nontrivial_event_shapes(self):
        from probpipe import NumericRecord, Record

        s = Record(
            "r",
            scalar=jnp.arange(4.0),
            vec=jnp.arange(12.0).reshape(4, 3),
        )

        row = _workflow_distribution_broadcast._index_sample(s, 1)

        assert isinstance(row, NumericRecord)
        assert float(row["scalar"]) == 1.0
        np.testing.assert_array_equal(row["vec"], jnp.array([3.0, 4.0, 5.0]))
