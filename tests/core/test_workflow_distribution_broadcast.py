"""Tests for Function distribution-only broadcast helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import BroadcastDistribution, EmpiricalDistribution, Normal, ProductDistribution
from probpipe.core import _workflow_distribution_broadcast, _workflow_execution
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


def _key_source(seed: int = 0):
    key = jax.random.PRNGKey(seed)

    def get_key():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    return get_key


def _require_not_called(*args, **kwargs):
    raise AssertionError("JAX traceability should not be required")


def _resolve_to(dispatch: str):
    def resolve_dispatch(values, broadcast_args, *, jax_supported):
        return dispatch

    return resolve_dispatch


class TestExecuteDistributionBroadcast:
    def test_sample_path_uses_execution_request(self, monkeypatch):
        values = {
            "x": Normal(loc=0.0, scale=1.0, name="x"),
            "offset": 2.0,
        }
        execution = _execution_config(mode="thread", max_workers=2, name="shift")
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
            broadcast_args=["x"],
            n_broadcast_samples=5,
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

        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=add,
            values=values,
            broadcast_args=["x", "y"],
            n_broadcast_samples=10,
            include_inputs=True,
            get_key=_key_source(1),
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

        result = _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=double,
            values=values,
            broadcast_args=["x"],
            n_broadcast_samples=6,
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

        with pytest.raises(
            RuntimeError,
            match="Prefect task or flow execution was requested",
        ):
            _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values=values,
                broadcast_args=["x"],
                n_broadcast_samples=6,
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

        sampled = _workflow_distribution_broadcast._sample_broadcast_args(
            values,
            ["a", "b"],
            8,
            jax.random.PRNGKey(3),
        )

        np.testing.assert_allclose(sampled["a"], sampled["b"])

    @pytest.mark.parametrize(
        ("n_broadcast_samples", "error_type", "message"),
        [
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
        with pytest.raises(error_type, match=message):
            _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values={"x": Normal(loc=0.0, scale=1.0, name="x")},
                broadcast_args=["x"],
                n_broadcast_samples=n_broadcast_samples,
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
        with pytest.warns(UserWarning, match="n_broadcast_samples=3 is too low"):
            result = _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values={"x": Normal(loc=0.0, scale=1.0, name="x")},
                broadcast_args=["x"],
                n_broadcast_samples=3,
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
