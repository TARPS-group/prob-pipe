"""Tests for Function distribution-only broadcast helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    EmpiricalDistribution,
    Function,
    Normal,
    ProductDistribution,
)
from probpipe.core import _workflow_call, _workflow_distribution_broadcast, _workflow_execution
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


def _ref(name: str) -> _workflow_call.WorkflowInputRef:
    return _workflow_call.WorkflowInputRef(name)


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
            broadcast_args=[_ref("x")],
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
            broadcast_args=[_ref("x"), _ref("y")],
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
            broadcast_args=[_ref("x")],
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
                broadcast_args=[_ref("x")],
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
            [_ref("a"), _ref("b")],
            8,
            jax.random.PRNGKey(3),
        )

        np.testing.assert_allclose(sampled[_ref("a")], sampled[_ref("b")])

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
                broadcast_args=[_ref("x")],
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
                broadcast_args=[_ref("x")],
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


class TestCoSamplingGroups:
    """One joint draw per co-sampling group, per design IV.2.

    Arguments are grouped by root ancestor: the same distribution passed twice,
    sibling views of one parent, and a parent passed alongside its own view all
    fall in one group, and each group is drawn once. Arguments with no common
    root are drawn independently, which samples the product of their laws.
    """

    @staticmethod
    def _joint():
        return ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
        )

    @staticmethod
    def _sample(values, names, *, n=8, seed=3):
        return _workflow_distribution_broadcast._sample_broadcast_args(
            values, [_ref(name) for name in names], n, jax.random.PRNGKey(seed)
        )

    def test_the_same_distribution_passed_twice_is_drawn_once(self):
        """The alias case: two references to one law denote one random variable."""
        dist = Normal(loc=0.0, scale=1.0, name="x")
        sampled = self._sample({"a": dist, "b": dist}, ("a", "b"))

        np.testing.assert_array_equal(sampled[_ref("a")], sampled[_ref("b")])

    def test_a_parent_and_its_own_view_share_one_draw(self):
        """The view's values are the parent draw's projection, not a second draw."""
        joint = self._joint()
        sampled = self._sample({"a": joint, "b": joint["x"]}, ("a", "b"))

        np.testing.assert_array_equal(sampled[_ref("a")]["x"], sampled[_ref("b")])

    def test_grouping_does_not_depend_on_argument_order(self):
        """A group is a set of references, so the view may come first."""
        joint = self._joint()
        parent_first = self._sample({"a": joint, "b": joint["x"]}, ("a", "b"))
        view_first = self._sample({"a": joint["x"], "b": joint}, ("a", "b"))

        np.testing.assert_array_equal(view_first[_ref("b")]["x"], view_first[_ref("a")])
        np.testing.assert_array_equal(parent_first[_ref("b")], view_first[_ref("a")])

    def test_sibling_views_come_from_one_parent_draw(self):
        """Distinct fields differ, but both project the same joint draw."""
        joint = self._joint()
        sampled = self._sample({"a": joint["x"], "b": joint["y"], "c": joint}, ("a", "b", "c"))

        parent = sampled[_ref("c")]
        np.testing.assert_array_equal(sampled[_ref("a")], parent["x"])
        np.testing.assert_array_equal(sampled[_ref("b")], parent["y"])
        assert not np.array_equal(sampled[_ref("a")], sampled[_ref("b")])

    def test_arguments_with_no_common_root_are_drawn_independently(self):
        """Separate groups sample the product law, one subkey per group in order.

        Pinning the subkeys, not just their difference: a group of one consumes
        exactly one split, so an aliased group cannot perturb the stream that
        unrelated arguments already receive.
        """
        first = Normal(loc=0.0, scale=1.0, name="x")
        second = Normal(loc=0.0, scale=1.0, name="y")
        sampled = self._sample({"a": first, "b": second}, ("a", "b"))

        key = jax.random.PRNGKey(3)
        key, subkey_a = jax.random.split(key)
        _key, subkey_b = jax.random.split(key)
        np.testing.assert_array_equal(sampled[_ref("a")], first._sample(subkey_a, (8,)))
        np.testing.assert_array_equal(sampled[_ref("b")], second._sample(subkey_b, (8,)))


class TestCoSamplingThroughACall:
    """The same contract as seen by a caller of a lifted ``Function``."""

    @staticmethod
    def _difference(**controls):
        return Function(
            func=lambda a, b: a - b,
            dispatch="sequential",
            n_broadcast_samples=controls.pop("n_broadcast_samples", 8),
            seed=0,
            **controls,
        )

    def test_a_law_passed_twice_approximates_f_of_one_variable(self):
        """``f(d, d)`` is ``X - X``, not ``X1 - X2``."""
        dist = Normal(loc=0.0, scale=1.0, name="x")
        result = self._difference()(dist, dist)

        np.testing.assert_array_equal(np.asarray(result.samples), np.zeros(8))

    def test_include_inputs_reports_one_realization_under_both_names(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        result = self._difference(include_inputs=True)(dist, dist)

        np.testing.assert_array_equal(
            np.asarray(result.input_samples["a"]), np.asarray(result.input_samples["b"])
        )

    def test_unrelated_laws_still_sample_the_product(self):
        """The complementary case: independence must survive the fix."""
        result = self._difference()(
            Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=0.0, scale=1.0, name="y")
        )

        assert not np.allclose(np.asarray(result.samples), 0.0)

    @pytest.mark.parametrize("n_broadcast_samples", [16, 8, 3])
    def test_an_empirical_passed_twice_enumerates_one_axis(self, n_broadcast_samples):
        """One enumeration axis per group: the diagonal, not the squared grid.

        Parameterized across the budget because the old behaviour degraded
        differently as ``n_broadcast_samples`` fell below the product size —
        enumerating both, then enumerating one and sampling the other.
        """
        empirical = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="e")
        result = self._difference(n_broadcast_samples=n_broadcast_samples)(empirical, empirical)

        samples = np.asarray(result.samples).ravel()
        assert samples.size == 3
        np.testing.assert_array_equal(samples, np.zeros(3))

    def test_an_aliased_empirical_counts_its_weight_once(self):
        """Weights are per group, so an alias does not square them."""
        empirical = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="e")
        result = self._difference(include_inputs=True)(empirical, empirical)

        np.testing.assert_allclose(np.asarray(result.weights), np.full(3, 1 / 3))


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
