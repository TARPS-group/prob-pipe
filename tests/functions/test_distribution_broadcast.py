"""Tests for Function distribution-only broadcast helpers."""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    MultivariateNormal,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    OpaqueBatch,
    Record,
    RecordBatch,
    ResultSchemaError,
    function,
    positive,
    sample,
    workflow_run,
)
from probpipe.core._record_batch import _batch_class_for
from probpipe.core._record_spec import _reshaped_template
from probpipe.core._specs import OutputSpec
from probpipe.core.config import WorkflowKind
from probpipe.distributions import (
    ConditionalDistribution,
    FactoredDistribution,
    SupportsConditionalSampling,
)
from probpipe.functions import (
    _broadcast,
    _context,
    _execution,
)
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding


class _ShiftKernel(ConditionalDistribution, SupportsConditionalSampling):
    """``x | z ~ Normal(z, 0.01)``, the dependent factor of a joint."""

    def __init__(self):
        spec = NumericArraySpec((), "float32")
        super().__init__(
            {"z": spec},
            OutputSpec(x=spec),
            label="x",
        )

    def _condition_on(self, given, /, **options):
        return Normal("x", given["z"], 0.01)

    def _conditional_sample(self, given, key, sample_shape=()):
        return Normal("x", given["z"], 0.01)._sample(key, sample_shape)


def _empirical_of_rows(label: str, rows: Record, weights=None) -> EmpiricalDistribution:
    """The empirical law of *rows*, a record whose leaves stack the atoms along their leading axis."""
    element = _reshaped_template(rows.event_template, lambda shape: shape[1:])
    columns = {path: rows[path] for path in rows.event_template}
    atoms = _batch_class_for(element)(columns, "atom", element_spec=element, label=label)
    return EmpiricalDistribution(atoms, weights, label=label)


def _drawn(law: EmpiricalDistribution, path: str) -> np.ndarray:
    """The atoms of *law* at the event path *path*, along one leading axis."""
    return np.asarray(law._rows[path])


def _execution_config(
    *,
    mode: _execution.WorkflowExecutionMode = "sequential",
    max_workers: int | None = None,
    name: str = "workflow",
) -> _execution.WorkflowExecutionConfig:
    return _execution.WorkflowExecutionConfig(
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


def _ref(name: str) -> _binding.FunctionInputRef:
    return _binding.FunctionInputRef(name)


def _stochastic_plan(values, n_broadcast_samples):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(signature)
    broadcast_plan = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast_plan, n_broadcast_samples)


class _RecordingNormal(Normal):
    def __init__(self, sample_calls, *, label, **attributes):
        self.sample_calls = sample_calls
        for attribute, value in attributes.items():
            setattr(self, attribute, value)
        super().__init__(label, loc=0.0, scale=1.0, label=label)

    def _sample(self, key, sample_shape=()):
        self.sample_calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


def _identity(value):
    return value


class TestExecuteDistributionBroadcast:
    def test_direct_aliases_sample_one_root_and_stay_diagonal(self):
        sample_calls = []
        events = []
        shared = _RecordingNormal(sample_calls, label="shared")
        values = {"first": shared, "second": shared}

        plan = _stochastic_plan(values, 12)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="difference",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(difference=None),
        )

        assert len(sample_calls) == 1
        assert len(events) == 1
        np.testing.assert_array_equal(_drawn(result, "first"), _drawn(result, "second"))
        np.testing.assert_allclose(_drawn(result, "difference"), 0.0)

    def test_equal_but_distinct_sources_sample_independently(self):
        first_calls = []
        second_calls = []
        events = []
        values = {
            "first": _RecordingNormal(first_calls, label="same"),
            "second": _RecordingNormal(second_calls, label="same"),
        }

        plan = _stochastic_plan(values, 12)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="difference",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(difference=None),
        )

        assert len(first_calls) == len(second_calls) == 1
        assert len(events) == 2
        assert not np.array_equal(_drawn(result, "first"), _drawn(result, "second"))

    @pytest.mark.parametrize("lookalike_attribute", ["parent", "base"])
    def test_unregistered_descendant_lookalikes_remain_independent(self, lookalike_attribute):
        first_calls = []
        second_calls = []
        first = _RecordingNormal(first_calls, label="first")
        second = _RecordingNormal(second_calls, label="second", **{lookalike_attribute: first})
        workflow = Function(
            lambda left, right: left - right,
            label="function",
            dispatch="sequential",
            n_broadcast_samples=12,
            output_spec=OutputSpec(function=None),
        )

        with workflow_run(seed=19):
            result = workflow(left=first, right=second)

        plan = result.provenance.controls["replay"]["plan"]["canonical_fields"]
        assert len(plan["source_groups"]) == 2
        assert len(first_calls) == len(second_calls) == 1
        assert not np.array_equal(first_calls[0][0], second_calls[0][0])

    def test_root_and_nested_view_use_the_same_sampled_realization(self):
        joint = (
            Normal("leaf", loc=0.0, scale=1.0) * Normal("other", loc=3.0, scale=1.0)
        ).with_path_names({"leaf": "nested/leaf"})
        values = {"root": joint, "leaf": joint["nested/leaf"]}

        plan = _stochastic_plan(values, 8)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="difference",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(difference=None),
        )

        np.testing.assert_allclose(_drawn(result, "difference"), 0.0)

    def test_weighted_empirical_aliases_enumerate_once(self):
        shared = EmpiricalDistribution(
            jnp.asarray([1.0, 4.0]),
            weights=jnp.asarray([0.2, 0.8]),
            component="shared",
        )
        values = {"first": shared, "second": shared}

        plan = _stochastic_plan(values, 8)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="difference",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(difference=None),
        )

        assert result.num_atoms == 2
        np.testing.assert_array_equal(_drawn(result, "first"), jnp.asarray([1.0, 4.0]))
        np.testing.assert_array_equal(_drawn(result, "first"), _drawn(result, "second"))
        np.testing.assert_allclose(_drawn(result, "difference"), 0.0)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.2, 0.8]))

    def test_weighted_record_root_and_view_enumerate_once(self):
        shared = _empirical_of_rows(
            "shared",
            Record(
                {"x": jnp.asarray([1.0, 4.0]), "y": jnp.asarray([10.0, 40.0])},
                label="draws",
            ),
            weights=jnp.asarray([0.3, 0.7]),
        )
        values = {"root": shared, "x": shared["x"]}

        plan = _stochastic_plan(values, 8)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="difference",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(difference=None),
        )

        assert result.num_atoms == 2
        np.testing.assert_allclose(_drawn(result, "difference"), 0.0)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.3, 0.7]))

    def test_sample_path_uses_execution_request(self, monkeypatch):
        values = {
            "x": Normal("x", loc=0.0, scale=1.0),
            "offset": 2.0,
        }
        execution = _execution_config(mode="thread", max_workers=2, name="shift")
        plan = _stochastic_plan(values, 5)
        seen = {}

        def shift(x, offset):
            return x + offset

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**item.call_values()) for item in request.work_items]

        monkeypatch.setattr(
            _broadcast._execution,
            "execute_many",
            fake_execute_many,
        )

        result = _broadcast.execute_distribution_broadcast(
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
            function_name="shift",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(shift=None),
        )

        request = seen["request"]
        assert isinstance(result, EmpiricalDistribution)
        assert request.func is shift
        assert request.execution is execution
        assert len(request.work_items) == 5
        assert all(item.call_values()["offset"] == 2.0 for item in request.work_items)
        assert all(not isinstance(item.call_values()["x"], Normal) for item in request.work_items)
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
                component="x",
            ),
            "y": EmpiricalDistribution(
                jnp.asarray([[10.0], [20.0]]),
                weights=jnp.asarray([0.4, 0.6]),
                component="y",
            ),
        }

        def add(x, y):
            return x + y

        plan = _stochastic_plan(values, 10)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="add",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(add=None),
        )

        assert result.num_atoms == 4
        np.testing.assert_allclose(
            _drawn(result, "x"),
            jnp.asarray([[1.0], [1.0], [2.0], [2.0]]),
        )
        np.testing.assert_allclose(
            _drawn(result, "y"),
            jnp.asarray([[10.0], [20.0], [10.0], [20.0]]),
        )
        np.testing.assert_allclose(
            _drawn(result, "add"),
            jnp.asarray([[11.0], [21.0], [12.0], [22.0]]),
        )
        np.testing.assert_allclose(
            result.weights,
            jnp.asarray([0.1, 0.15, 0.3, 0.45]),
            atol=1e-6,
        )

    def test_exact_empirical_size_must_match_the_frozen_plan(self):
        empirical = EmpiricalDistribution(jnp.asarray([1.0, 2.0, 3.0]), component="x")
        values = {"x": empirical}
        plan = _stochastic_plan(values, 8)
        atoms = NumericArrayBatch(
            jnp.asarray([1.0, 2.0]),
            "x",
            element_spec=NumericArraySpec(()),
            label="x",
        )
        object.__setattr__(empirical, "_atoms", atoms)

        with pytest.raises(RuntimeError, match="exact empirical size changed after planning"):
            _broadcast.execute_distribution_broadcast(
                func=lambda x: x,
                values=values,
                stochastic_plan=plan,
                logical_unit=plan.logical_units[0],
                include_inputs=False,
                get_key=_require_not_called,
                make_execution_config=lambda: _execution_config(name="identity"),
                requested_dispatch="sequential",
                resolve_dispatch=_resolve_to("sequential"),
                require_jax_traceable=_require_not_called,
                function_name="identity",
                workflow_kind=WorkflowKind.OFF,
            )

    def test_jax_path_vectorizes_samples_and_outputs(self):
        values = {"x": Normal("x", loc=1.0, scale=0.5)}
        seen = {"required": False}

        def double(x):
            return 2.0 * x

        def require_jax_traceable(values, broadcast_args):
            seen["required"] = True

        plan = _stochastic_plan(values, 6)
        result = _broadcast.execute_distribution_broadcast(
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
            function_name="double",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(double=None),
        )

        assert seen["required"] is True
        assert result.num_atoms == 6
        np.testing.assert_allclose(_drawn(result, "double"), _drawn(result, "x") * 2.0)

    def test_jax_prefect_path_requires_prefect(self, monkeypatch):
        values = {"x": Normal("x", loc=1.0, scale=0.5)}
        monkeypatch.setattr(_broadcast, "task", None)
        monkeypatch.setattr(_broadcast, "flow", None)
        plan = _stochastic_plan(values, 6)

        with pytest.raises(
            RuntimeError,
            match="Prefect task or flow execution was requested",
        ):
            _broadcast.execute_distribution_broadcast(
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
                function_name="identity",
                workflow_kind=WorkflowKind.TASK,
            )

    @pytest.mark.parametrize(
        ("dispatch", "workflow_kind", "route_module"),
        [
            pytest.param("sequential", WorkflowKind.TASK, _execution, id="row-wise"),
            pytest.param("jax", WorkflowKind.FLOW, _broadcast, id="jax"),
        ],
    )
    def test_prefect_route_failure_precedes_sampling_and_commit(
        self,
        monkeypatch,
        dispatch,
        workflow_kind,
        route_module,
    ):
        sample_calls = []
        commits = []
        source = _RecordingNormal(sample_calls, label="x")
        workflow = Function(
            _identity,
            label="identity",
            dispatch=dispatch,
            workflow_kind=workflow_kind,
            n_broadcast_samples=5,
        )
        commit_invocation = _context._commit_stochastic_invocation

        def record_commit(occurrence_kind="invocation"):
            commits.append(occurrence_kind)
            return commit_invocation(occurrence_kind)

        def reject_route(*args, **kwargs):
            raise ValueError("invalid Prefect route")

        monkeypatch.setattr(_context, "_commit_stochastic_invocation", record_commit)
        monkeypatch.setattr(route_module, "flow", reject_route)

        with workflow_run(seed=7), pytest.raises(ValueError, match="invalid Prefect route"):
            workflow(source)

        assert sample_calls == []
        assert commits == []

    def test_same_parent_views_share_parent_sample(self):
        joint = Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0)
        view_x = joint["x"]
        values = {"a": view_x, "b": view_x}

        plan = _stochastic_plan(values, 8)
        sampled = _broadcast._sample_planned_source_groups(
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
            "first": _RecordingNormal(first_calls, label="first"),
            "second": _RecordingNormal(second_calls, label="second"),
        }
        plan = _stochastic_plan(values, 11)
        assert plan.sample_shape is not None
        events = []

        sampled = _broadcast._sample_planned_source_groups(
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
                component="exact",
            ),
            "sampled": _RecordingNormal(sampled_calls, label="sampled"),
        }
        plan = _stochastic_plan(values, 5)
        events = []

        result = _broadcast.execute_distribution_broadcast(
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
            function_name="add",
            workflow_kind=WorkflowKind.OFF,
            output_spec=OutputSpec(add=None),
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
        values = {"x": Normal("x", loc=0.0, scale=1.0)}
        invalid_plan = replace(
            _stochastic_plan(values, 5),
            n_broadcast_samples=n_broadcast_samples,
        )

        with pytest.raises(error_type, match=message):
            _broadcast.execute_distribution_broadcast(
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
                function_name="identity",
                workflow_kind=WorkflowKind.OFF,
            )

    def test_low_n_broadcast_samples_warns(self):
        values = {"x": Normal("x", loc=0.0, scale=1.0)}
        plan = _stochastic_plan(values, 3)
        with pytest.warns(UserWarning, match="n_broadcast_samples=3 is too low"):
            result = _broadcast.execute_distribution_broadcast(
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
                function_name="identity",
                workflow_kind=WorkflowKind.OFF,
                output_spec=OutputSpec(identity=None),
            )

        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 3

    def test_executor_has_no_empirical_replanning_helper(self):
        assert not hasattr(_broadcast, "_split_empirical_args")


class TestCoSamplingGroups:
    """One joint draw per co-sampling group, per design IV.2.

    Arguments are grouped by root ancestor: the same distribution passed twice,
    sibling views of one parent, and a parent passed alongside its own view all
    fall in one group, and each group is drawn once. Arguments with no common
    root are drawn independently, which samples the product of their laws.
    """

    @staticmethod
    def _joint():
        return Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0)

    @staticmethod
    def _sample(values, names, *, n=8, seed=3):
        selected = {name: values[name] for name in names}
        plan = _stochastic_plan(selected, n)
        assert plan is not None
        assert plan.sample_shape is not None
        return _broadcast._sample_planned_source_groups(
            plan,
            plan.source_groups,
            plan.sample_shape,
            plan.logical_units[0],
            _key_source(seed),
        )

    def test_the_same_distribution_passed_twice_is_drawn_once(self):
        """The alias case: two references to one law denote one random variable."""
        dist = Normal("x", loc=0.0, scale=1.0)
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
        """Separate groups sample the product law through distinct planned events."""
        first = Normal("x", loc=0.0, scale=1.0)
        second = Normal("y", loc=0.0, scale=1.0)
        values = {"a": first, "b": second}
        plan = _stochastic_plan(values, 8)
        assert plan is not None
        assert plan.sample_shape is not None
        events = []
        sampled = _broadcast._sample_planned_source_groups(
            plan,
            plan.source_groups,
            plan.sample_shape,
            plan.logical_units[0],
            _key_source(3, events),
        )

        assert [event.stochastic_source_id for event in events] == [
            ("source-group", 0),
            ("source-group", 1),
        ]
        assert not np.array_equal(sampled[_ref("a")], sampled[_ref("b")])


class TestCoSamplingThroughACall:
    """The same contract as seen by a caller of a lifted ``Function``."""

    @staticmethod
    def _difference(**controls):
        controls.setdefault("output_spec", OutputSpec(function=None))
        return Function(
            lambda a, b: a - b,
            label="function",
            dispatch=controls.pop("dispatch", "sequential"),
            n_broadcast_samples=controls.pop("n_broadcast_samples", 8),
            **controls,
        )

    @staticmethod
    def _run(workflow, *args, **kwargs):
        with workflow_run(seed=0):
            return workflow(*args, **kwargs)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_a_law_passed_twice_approximates_f_of_one_variable(self, dispatch):
        """``f(d, d)`` is ``X - X``, not ``X1 - X2``.

        Both dispatches, because the grouping lives in the sampler all three
        execution paths share: a divergence here would mean one backend silently
        answering a different question from another.
        """
        dist = Normal("x", loc=0.0, scale=1.0)
        result = self._run(self._difference(dispatch=dispatch), dist, dist)

        np.testing.assert_array_equal(np.asarray(result.atoms), np.zeros(8))

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_include_inputs_reports_one_realization_under_both_names(self, dispatch):
        dist = Normal("x", loc=0.0, scale=1.0)
        result = self._run(
            self._difference(dispatch=dispatch, include_inputs=True),
            dist,
            dist,
        )

        np.testing.assert_array_equal(
            np.asarray(_drawn(result, "a")), np.asarray(_drawn(result, "b"))
        )

    def test_identical_but_distinct_laws_are_distinct_roots(self):
        """A group is object identity, not structural equality.

        Two separately constructed laws are two random variables however alike
        their parameters and names, so they sample the product; only a shared
        object is one variable.
        """
        first = Normal("x", loc=0.0, scale=1.0)
        second = Normal("x", loc=0.0, scale=1.0)

        assert not np.allclose(
            np.asarray(self._run(self._difference(), first, second).atoms),
            0.0,
        )
        np.testing.assert_array_equal(
            np.asarray(self._run(self._difference(), first, first).atoms),
            np.zeros(8),
        )

    def test_unrelated_laws_still_sample_the_product(self):
        """The complementary case: independence must survive the fix."""
        result = self._run(
            self._difference(),
            Normal("x", loc=0.0, scale=1.0),
            Normal("y", loc=0.0, scale=1.0),
        )

        assert not np.allclose(np.asarray(result.atoms), 0.0)

    @pytest.mark.parametrize("n_broadcast_samples", [16, 8, 3])
    def test_an_empirical_passed_twice_enumerates_one_axis(self, n_broadcast_samples):
        """One enumeration axis per group: the diagonal, not the squared grid.

        Parameterized across the budget because the old behaviour degraded
        differently as ``n_broadcast_samples`` fell below the product size —
        enumerating both, then enumerating one and sampling the other.
        """
        empirical = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), component="e")
        result = self._run(
            self._difference(n_broadcast_samples=n_broadcast_samples),
            empirical,
            empirical,
        )

        samples = np.asarray(result.atoms).ravel()
        assert samples.size == 3
        np.testing.assert_array_equal(samples, np.zeros(3))

    def test_a_record_valued_law_can_be_lifted(self):
        """Assembly counts rows by ``batch_shape``, which a record batch answers.

        Its ``len`` is the field count and its ``shape`` raises, so the row count
        had to come from somewhere that means one thing for every batched value.
        """
        joint = Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0)
        lifted = Function(
            lambda a: a["x"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        assert np.asarray(self._run(lifted, joint).atoms).shape[0] == 8

    def test_a_parent_and_its_own_view_lift_together(self):
        """The remaining IV.2 case, end to end: ``f(d, d["x"])`` is one draw."""
        joint = Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0)
        lifted = Function(
            lambda a, b: a["x"] - b,
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(lifted, joint, joint["x"]).atoms),
            np.zeros(8),
        )

    def test_a_record_valued_empirical_enumerates(self):
        """Enumerated rows stack per argument, and a record row is not an array.

        A record-valued empirical's atoms reach the body as ``Record``s, which
        ``jnp.stack`` cannot take; they stack through ``RecordBatch.stack`` instead.
        """
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])},
                label="r",
            ),
        )
        lifted = Function(
            lambda a: a["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(lifted, empirical).atoms).ravel(),
            np.array([10.0, 20.0, 30.0]),
        )

    def test_a_renamed_empirical_law_enumerates_its_atoms(self):
        """A rename moves the atoms' fields, so the renamed law enumerates as the law does."""
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])},
                label="r",
            ),
        )
        renamed = empirical.with_path_names({"x": "group/x", "y": "group/y"})
        lifted = Function(
            lambda a: a.at_path("group")["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        assert lifted.check(renamed).selected.method_name == "empirical_enumeration"
        np.testing.assert_array_equal(
            np.asarray(self._run(lifted, renamed).atoms).ravel(),
            np.array([10.0, 20.0, 30.0]),
        )

    def test_a_renamed_law_lifts_together_with_its_parent(self):
        """A renamed law reads its parent's draws, so the two are one draw per repetition."""
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])},
                label="r",
            ),
        )
        renamed = empirical.with_path_names({"x": "group/x", "y": "group/y"})
        lifted = Function(
            lambda a, b: a["y"] - b.at_path("group")["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(lifted, empirical, renamed).atoms).ravel(), np.zeros(3)
        )

    def test_a_record_valued_lift_can_be_resampled(self):
        """The joint over a record-valued input is a distribution, so it samples.

        Reading ``.atoms`` goes through the output marginal and says nothing
        about the joint: resampling gathers rows from every component, and a
        record-valued input carries its rows in fields rather than along a shape.
        """
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])},
                label="r",
            ),
        )
        lifted = Function(
            lambda a: a["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            include_inputs=True,
            output_spec=OutputSpec(function=None),
        )

        joint = self._run(lifted, empirical)
        with workflow_run(seed=0):
            drawn = sample(joint, sample_shape=(6,))

        # Every drawn row is one atom of the empirical, and the output is that
        # atom's own ``y`` — the pairing a joint exists to preserve.
        x, y = np.asarray(drawn["a"]["x"]), np.asarray(drawn["a"]["y"])
        np.testing.assert_allclose(y, x * 10)
        np.testing.assert_allclose(np.asarray(drawn["function"]).ravel(), y)
        assert set(x.tolist()) <= {1.0, 2.0, 3.0}

        with workflow_run(seed=0):
            one = sample(joint)
        assert np.asarray(one["a/x"]).shape == ()
        np.testing.assert_allclose(
            float(np.asarray(one["function"])), float(np.asarray(one["a/y"]))
        )

    def test_a_record_valued_empirical_bigger_than_the_budget_samples(self):
        """Too many atoms to enumerate, so the group routes to sampling.

        That path hands back a plain record batched on its leaves rather than a
        record batch, which reports no ``batch_shape`` — the rows are on a leaf.
        """
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.arange(10.0), "y": jnp.arange(10.0) * 10},
                label="r",
            ),
        )
        lifted = Function(
            lambda a: a["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=5,
            output_spec=OutputSpec(function=None),
        )

        result = self._run(lifted, empirical)
        assert result.num_atoms == 5
        assert np.asarray(result.atoms).shape == (5,)

    def test_a_mixed_record_stacks_with_an_object_column(self):
        """Columns are leaf-keyed and typed per field, so a record mixing a
        numeric leaf with an opaque one stacks — the refusal this test used to
        pin died with the class that refused."""
        rows = [
            Record(
                {"x": jnp.array(1.0), "tag": "a"},
                label="r",
            ),
            Record(
                {"x": jnp.array(2.0), "tag": "b"},
                label="r",
            ),
        ]

        stacked = _broadcast._stack_rows(rows)

        np.testing.assert_allclose(np.asarray(stacked["x"]), [1.0, 2.0])
        assert list(stacked._raw_column("tag")) == ["a", "b"]

    def test_a_nested_record_valued_empirical_lifts(self):
        """A column is keyed by leaf path, so a nested record batches like a
        flat one."""
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"group": {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])}},
                label="r",
            ),
        )
        lifted = Function(
            lambda a: a["group/y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=6,
            include_inputs=True,
            output_spec=OutputSpec(function=None),
        )

        joint = self._run(lifted, empirical)
        with workflow_run(seed=0):
            drawn = sample(joint, sample_shape=(4,))
        np.testing.assert_allclose(
            np.asarray(drawn["function"]).ravel(), np.asarray(drawn["a"]["group/y"])
        )

    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "thread"])
    def test_a_sampled_nested_record_valued_law_lifts_rowwise(self, dispatch):
        """Nested records are supported up to the row-wise dispatch boundary."""
        nested = (
            (Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0))
            .with_path_names({"x": "group/x", "y": "group/y"})
            .with_label("nested")
        )
        lifted = Function(
            lambda a: a["group/y"],
            label="function",
            dispatch=dispatch,
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        assert np.asarray(self._run(lifted, nested).atoms).shape == (8,)

    def test_a_sampled_nested_record_valued_law_matches_sequential_under_jax(self):
        """The draw supplies nested record structure before either body is mapped."""
        nested = (
            (Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=10.0, scale=1.0))
            .with_path_names({"x": "group/x", "y": "group/y"})
            .with_label("nested")
        )
        mapped = Function(
            lambda a: a["group/y"],
            label="function",
            dispatch="jax",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            lambda a: a["group/y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(mapped, nested).atoms),
            np.asarray(self._run(sequential, nested).atoms),
        )

    def test_a_record_valued_empirical_passed_twice_shares_its_atom(self):
        empirical = _empirical_of_rows(
            "e",
            Record(
                {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([10.0, 20.0, 30.0])},
                label="r",
            ),
        )
        lifted = Function(
            lambda a, b: a["y"] - b["y"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(lifted, empirical, empirical).atoms),
            np.zeros(3),
        )

    def test_an_aliased_empirical_counts_its_weight_once(self):
        """Weights are per group, so an alias does not square them."""
        empirical = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), component="e")
        result = self._run(self._difference(include_inputs=True), empirical, empirical)

        np.testing.assert_allclose(np.asarray(result.weights), np.full(3, 1 / 3))


class TestTheDrawsOfALargeEmpiricalLaw:
    """A lift draws an empirical argument's atoms with a lower variance than independent draws."""

    def test_equally_weighted_atoms_are_drawn_without_replacement(self):
        @function(output_spec=OutputSpec(identity=None))
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(EmpiricalDistribution(jnp.arange(1000.0), component="x"))
        values = np.asarray(result.atoms).ravel()
        assert result.num_atoms == Function.DEFAULT_N_BROADCAST_SAMPLES
        assert len(np.unique(values)) == result.num_atoms

    def test_more_draws_than_atoms_take_each_atom_equally_often(self):
        """Two 20-atom laws: one is enumerated and the other drawn 240 times, 12 per atom."""

        @function(output_spec=OutputSpec(pair=None))
        def pair(x, y):
            return jnp.stack([x, y])

        with workflow_run(seed=0):
            result = pair(
                EmpiricalDistribution(jnp.arange(20.0), component="a"),
                EmpiricalDistribution(100.0 + jnp.arange(20.0), component="b"),
            )
        sampled = np.asarray(result.atoms)[:, 1]
        assert result.num_atoms == 240
        assert set(np.unique(sampled, return_counts=True)[1].tolist()) == {12}

    def test_weighted_atoms_are_drawn_by_stratified_resampling(self):
        """Each atom is drawn within one stratum of its expected number of times on either side."""

        @function(output_spec=OutputSpec(identity=None))
        def identity(x):
            return x

        weights = jax.random.uniform(jax.random.PRNGKey(1), (1000,))
        weights = weights / weights.sum()
        with workflow_run(seed=0):
            result = identity(
                EmpiricalDistribution(jnp.arange(1000.0), weights=weights, component="w")
            )
        drawn = np.bincount(np.asarray(result.atoms).ravel().astype(int), minlength=1000)
        expected = result.num_atoms * np.asarray(weights)
        assert np.all(np.abs(drawn - expected) < 2.0)


class TestIndexSampleHelper:
    """Direct unit tests for the module-level ``_index_sample`` helper."""

    def test_bare_array(self):
        s = jnp.arange(20.0).reshape(5, 4)
        for i in range(5):
            np.testing.assert_array_equal(
                _broadcast._index_sample(s, i),
                s[i],
            )

    def test_bare_array_1d(self):
        s = jnp.arange(10.0)

        assert float(_broadcast._index_sample(s, 3)) == 3.0

    def test_single_field_record_returns_per_row_numeric_record(self):
        from probpipe import NumericRecord, Record

        s = Record(
            {"x": jnp.arange(15.0).reshape(5, 3)},
            label="r",
        )

        for i in range(5):
            row = _broadcast._index_sample(s, i)
            assert isinstance(row, NumericRecord)
            assert row.fields == ("x",)
            np.testing.assert_array_equal(row["x"], s["x"][i])

    def test_multi_field_record_returns_per_row_numeric_record(self):
        from probpipe import NumericRecord, Record

        s = Record(
            {"mu": jnp.arange(5.0), "sigma": jnp.arange(5.0) + 100.0},
            label="r",
        )

        row = _broadcast._index_sample(s, 2)

        assert isinstance(row, NumericRecord)
        assert row.fields == ("mu", "sigma")
        assert float(row["mu"]) == 2.0
        assert float(row["sigma"]) == 102.0

    def test_multi_field_record_with_nontrivial_event_shapes(self):
        from probpipe import NumericRecord, Record

        s = Record(
            {"scalar": jnp.arange(4.0), "vec": jnp.arange(12.0).reshape(4, 3)},
            label="r",
        )

        row = _broadcast._index_sample(s, 1)

        assert isinstance(row, NumericRecord)
        assert float(row["scalar"]) == 1.0
        np.testing.assert_array_equal(row["vec"], jnp.array([3.0, 4.0, 5.0]))


class TestTheProbeModelsItsExecutorsTransform:
    """``_broadcast_jax`` maps over draws, so the probe that gates it must too.

    Probing without the transform passes a body to an executor that then fails
    inside it, where no fallback is left to take.
    """

    @staticmethod
    def _run(workflow, *args, **kwargs):
        with workflow_run(seed=0):
            return workflow(*args, **kwargs)

    @staticmethod
    def _returns_a_batch(**controls):
        def body(x):
            return RecordBatch.stack(
                [
                    Record(
                        {"y": x * k},
                        label="r",
                    )
                    for k in (1.0, 2.0, 3.0)
                ],
                level_name="k",
            )

        return Function(
            body,
            label="body",
            dispatch=controls.pop("dispatch", "auto"),
            n_broadcast_samples=controls.pop("n_broadcast_samples", 8),
            **controls,
        )

    @pytest.mark.pending(
        reason="the empirical law of a lifted function that returns a batch",
        raises=NotImplementedError,
    )
    def test_a_batch_returning_body_falls_back_rather_than_failing_in_the_executor(self):
        """The regression: this raised the pytree rank error out of ``vmap``."""
        dist = Normal("x", loc=0.0, scale=1.0)

        result = self._run(self._returns_a_batch(), dist)

        assert result is not None

    @pytest.mark.pending(
        reason="the empirical law of a lifted function that returns a batch",
        raises=NotImplementedError,
    )
    def test_the_fallback_is_what_ran(self, caplog):
        dist = Normal("x", loc=0.0, scale=1.0)

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            self._run(self._returns_a_batch(), dist)

        assert any("not JAX-traceable" in record.message for record in caplog.records)

    @pytest.mark.pending(
        reason="the empirical law of a lifted function that returns a batch",
        raises=NotImplementedError,
    )
    def test_the_fallback_agrees_with_explicit_sequential(self):
        """Falling back costs speed, never the answer."""
        dist = Normal("x", loc=0.0, scale=1.0)

        fell_back = self._run(self._returns_a_batch(dispatch="auto"), dist)
        sequential = self._run(self._returns_a_batch(dispatch="sequential"), dist)

        np.testing.assert_array_equal(np.asarray(fell_back.atoms), np.asarray(sequential.atoms))

    def test_requesting_jax_reports_the_dispatch_rather_than_the_pytree(self):
        """The refusal names the choice the caller made and can change."""
        dist = Normal("x", loc=0.0, scale=1.0)

        with pytest.raises(ValueError, match="dispatch='jax' failed while tracing"):
            self._run(self._returns_a_batch(dispatch="jax"), dist)

    def test_a_body_that_survives_the_transform_still_takes_jax(self, caplog):
        """The probe gained a transform, not a blanket refusal."""
        dist = Normal("x", loc=0.0, scale=1.0)
        doubles = Function(
            lambda x: x * 2.0,
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            self._run(doubles, dist)

        assert not any("not JAX-traceable" in record.message for record in caplog.records)

    @pytest.mark.pending(
        reason="the empirical law of a lifted function that returns a batch",
        raises=NotImplementedError,
    )
    def test_the_mapped_probe_covers_several_distribution_arguments(self):
        """The probe builds a tuple of draws, one per broadcast argument.

        Covers the multiple-reference path without isolating the second
        argument: ``jax.make_jaxpr`` abstracts everything it is given, so no
        body can tell "mapped" from "traced" by its arguments alone.
        """

        def body(x, y):
            return RecordBatch.stack(
                [
                    Record(
                        {"z": x * k + y},
                        label="r",
                    )
                    for k in (1.0, 2.0)
                ],
                level_name="k",
            )

        broadcast = Function(body, label="body", n_broadcast_samples=8)
        sequential = Function(body, label="body", dispatch="sequential", n_broadcast_samples=8)
        first = Normal("x", loc=0.0, scale=1.0)
        second = Normal("y", loc=3.0, scale=1.0)

        np.testing.assert_array_equal(
            np.asarray(self._run(broadcast, first, second).atoms),
            np.asarray(self._run(sequential, first, second).atoms),
        )

    def test_the_mapped_slice_carries_the_declared_event_shape(self, caplog):
        """The slice is event-shaped, as the bare probe's dummy was.

        ``v[2]`` is rank-sensitive where a reduction would not be, and a dummy
        of the wrong shape silently leaves the JAX path — so staying on it is
        the assertion.
        """
        vector = MultivariateNormal("v", loc=jnp.zeros(3), cov=jnp.eye(3))
        third = Function(
            lambda v: v[2],
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            lambda v: v[2],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            mapped = self._run(third, vector)

        assert not any("not JAX-traceable" in record.message for record in caplog.records)
        np.testing.assert_allclose(
            np.asarray(mapped.atoms),
            np.asarray(self._run(sequential, vector).atoms),
            rtol=1e-6,
        )

    def test_an_aliased_argument_still_reads_as_one_variable(self):
        """Co-sampling is the sampler's, not the probe's: ``f(d, d)`` is ``X - X``.

        The probe's independent dummy per reference must not be mistaken for
        the executor's grouping.
        """
        dist = Normal("x", loc=0.0, scale=1.0)
        difference = Function(
            lambda a, b: a - b,
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        np.testing.assert_array_equal(
            np.asarray(self._run(difference, dist, dist).atoms),
            np.zeros(8),
        )

    def test_the_views_of_a_dependent_joint_are_probed(self, caplog):
        """The root's resolved component metadata supplies the probe dtypes."""
        joint = _ShiftKernel() * Normal("z", loc=0.0, scale=1.0)
        difference = Function(
            lambda a, b: a - b,
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            lambda a, b: a - b,
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            mapped = self._run(difference, a=joint["z"], b=joint["x"])

        assert not any("not JAX-traceable" in record.message for record in caplog.records)
        np.testing.assert_allclose(
            np.asarray(mapped.atoms),
            np.asarray(self._run(sequential, a=joint["z"], b=joint["x"]).atoms),
        )

    def test_a_multi_field_law_vectorizes(self, caplog):
        """A guard, not a regression: this law answers both, so it always could.

        Kept because the draw-based probe must not narrow what it accepts.
        """
        law = Normal("a", loc=0.0, scale=1.0) * Normal("b", loc=1.0, scale=1.0)
        totals = Function(
            lambda r: r["a"] + r["b"],
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            lambda r: r["a"] + r["b"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            mapped = self._run(totals, law)

        assert not any("not JAX-traceable" in record.message for record in caplog.records)
        np.testing.assert_array_equal(
            np.asarray(mapped.atoms), np.asarray(self._run(sequential, law).atoms)
        )

    def test_an_enumerated_empirical_law_is_probed_and_mapped(self, caplog):
        """The probe stands in for one atom of an enumerated law as for one draw of a sampled one."""
        law = _empirical_of_rows(
            "law",
            Record(
                {"a": jnp.arange(6.0), "b": jnp.arange(6.0) + 10.0},
                label="r",
            ),
        )
        totals = Function(
            lambda r: r["a"] + r["b"],
            label="function",
            n_broadcast_samples=6,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            lambda r: r["a"] + r["b"],
            label="function",
            dispatch="sequential",
            n_broadcast_samples=6,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            mapped = self._run(totals, law)

        assert not any("not JAX-traceable" in record.message for record in caplog.records)
        assert mapped.provenance.metadata["dispatch"] == "jax"
        np.testing.assert_allclose(
            np.asarray(mapped.atoms), np.asarray(self._run(sequential, law).atoms)
        )

    @pytest.mark.pending(
        reason="the empirical law of a lifted function that returns a batch",
        raises=NotImplementedError,
    )
    def test_a_batch_returning_body_survives_the_nested_regime(self, caplog):
        """A sweep crossed with a law still produces a result.

        ``_broadcast_jax`` is not reached in this regime at all, so this pins
        the composition rather than the mapped-draw probe.
        """

        def body(p, x):
            return RecordBatch.stack(
                [
                    Record(
                        {"y": p["a"] * x * k},
                        label="r",
                    )
                    for k in (1.0, 2.0)
                ],
                level_name="k",
            )

        rows = RecordBatch.stack(
            [
                Record(
                    {"a": jnp.asarray(float(i))},
                    label="p",
                )
                for i in range(3)
            ],
            level_name="row",
        )
        nested = Function(body, label="body", n_broadcast_samples=8)

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            result = self._run(nested, rows, Normal("x", loc=0.0, scale=1.0))

        assert result is not None
        assert any("not JAX-traceable" in record.message for record in caplog.records)

    def test_a_one_field_record_law_presents_a_record_to_every_dispatch(self, caplog):
        """A one-field draw presents as a record, whichever dispatch runs.

        The law draws a batch of records, so mapping it yields a record per draw,
        and the row-wise paths index the same record. A body that reads the field
        therefore runs under both, and the mapped executor still vectorizes it.
        """
        law = FactoredDistribution(
            [Normal("x", loc=0.0, scale=1.0)],
            label="law",
        )
        kinds = []

        def double(x):
            kinds.append(type(x))
            return x["x"] * 2

        doubles = Function(
            double,
            label="function",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )
        sequential = Function(
            double,
            label="function",
            dispatch="sequential",
            n_broadcast_samples=8,
            output_spec=OutputSpec(function=None),
        )

        with caplog.at_level(logging.INFO, logger="probpipe.functions._function"):
            mapped = self._run(doubles, law)

        # Consistent *and* still vectorized, rather than consistent by retreat.
        assert not any("not JAX-traceable" in record.message for record in caplog.records)
        np.testing.assert_array_equal(
            np.asarray(mapped.atoms), np.asarray(self._run(sequential, law).atoms)
        )
        assert kinds and all(issubclass(kind, Record) for kind in kinds)


class TestAnEnumerationRunsInOneMappedCall:
    """An enumerated lift maps its combinations of atoms as the sampling lift maps its draws.

    Each body below records its calls, so a count that does not grow with the
    atoms shows the body was traced rather than run once per atom.
    """

    @staticmethod
    def _records(n: int, weights=None) -> EmpiricalDistribution:
        a = jnp.linspace(0.5, 1.5, n)
        return _empirical_of_rows(
            "theta",
            Record(
                {"a": a, "b": a + 2.0},
                label="r",
            ),
            weights,
        )

    @staticmethod
    def _arrays(n: int, weights=None) -> EmpiricalDistribution:
        return EmpiricalDistribution(
            jnp.linspace(0.5, 1.5, 2 * n).reshape(n, 2), weights, component="theta"
        )

    @staticmethod
    def _rate(theta) -> jax.Array:
        """A scalar read from one atom, a record or an array."""
        return theta["a"] if isinstance(theta, Record) else theta[0]

    @classmethod
    def _counted(cls, calls: list, **controls) -> Function:
        """A function whose body runs a ``jax.lax.scan`` and records each of its calls."""

        def trajectory(theta):
            calls.append(theta)
            rate = 2.0 + cls._rate(theta)

            def step(n, _):
                n = rate * n * jnp.exp(-n)
                return n, n

            return jax.lax.scan(step, jnp.asarray(1.0), None, length=22)[1]

        controls.setdefault("output_spec", OutputSpec(trajectory=None))
        return Function(trajectory, label="trajectory", **controls)

    @pytest.mark.parametrize("make", ["_records", "_arrays"])
    def test_the_body_runs_as_often_for_a_thousand_atoms_as_for_five(self, make):
        make = getattr(self, make)
        calls: list = []
        result = self._counted(calls, n_broadcast_samples=1000)(make(1000))
        thousand = len(calls)
        calls.clear()
        self._counted(calls, n_broadcast_samples=1000)(make(5))

        assert len(calls) == thousand < 5
        assert result.num_atoms == 1000
        assert result.provenance.metadata["route"] == "empirical_enumeration"
        assert result.provenance.metadata["dispatch"] == "jax"

    @pytest.mark.parametrize("make", ["_records", "_arrays"])
    def test_the_atoms_and_weights_equal_the_sequential_loops(self, make):
        """Two weighted laws enumerate their product, each combination weighted by both atoms."""
        theta = getattr(self, make)(4, jnp.array([0.1, 0.2, 0.3, 0.4]))
        scale = EmpiricalDistribution(
            jnp.array([1.0, 2.0, 3.0]), jnp.array([0.5, 0.3, 0.2]), component="scale"
        )
        mapped_calls: list = []
        sequential_calls: list = []

        def scaled(calls):
            def body(theta, scale):
                calls.append(theta)
                return scale * jnp.sin(self._rate(theta))

            return body

        mapped = Function(
            scaled(mapped_calls),
            n_broadcast_samples=12,
            label="scaled",
            output_spec=OutputSpec(scaled=None),
        )(theta, scale)
        sequential = Function(
            scaled(sequential_calls),
            n_broadcast_samples=12,
            dispatch="sequential",
            label="scaled",
            output_spec=OutputSpec(scaled=None),
        )(theta, scale)

        assert len(sequential_calls) == 12 > len(mapped_calls)
        assert mapped.num_atoms == sequential.num_atoms == 12
        np.testing.assert_allclose(
            np.asarray(mapped.atoms), np.asarray(sequential.atoms), rtol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(mapped.weights), np.asarray(sequential.weights))

    def test_include_inputs_reports_the_enumerated_atoms(self):
        theta = self._records(6, jnp.arange(1.0, 7.0))
        joint = {}
        for dispatch in ("auto", "sequential"):
            calls: list = []
            joint[dispatch] = self._counted(
                calls, n_broadcast_samples=6, dispatch=dispatch, include_inputs=True
            )(theta)

        assert list(joint["auto"].event_spec.components) == ["theta", "trajectory"]
        for path in ("theta/a", "theta/b", "trajectory"):
            np.testing.assert_allclose(
                _drawn(joint["auto"], path), _drawn(joint["sequential"], path), rtol=1e-6
            )
        np.testing.assert_array_equal(
            np.asarray(joint["auto"].weights), np.asarray(joint["sequential"].weights)
        )

    def test_a_plan_that_also_samples_maps_with_the_same_draws(self):
        """An empirical law enumerated beside a sampled law maps with the loop's draws."""
        theta = self._arrays(3)
        noise = Normal("noise", loc=0.0, scale=1.0)
        laws = {}
        counts = {}
        for dispatch in ("auto", "sequential"):
            calls: list = []

            def body(theta, noise, calls=calls):
                calls.append(theta)
                return self._rate(theta) + noise

            with workflow_run(seed=3):
                laws[dispatch] = Function(
                    body,
                    n_broadcast_samples=30,
                    dispatch=dispatch,
                    label="f",
                    output_spec=OutputSpec(f=None),
                )(theta, noise)
            counts[dispatch] = len(calls)

        assert counts["sequential"] == 30 > counts["auto"]
        np.testing.assert_allclose(
            np.asarray(laws["auto"].atoms), np.asarray(laws["sequential"].atoms), rtol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(laws["auto"].weights), np.asarray(laws["sequential"].weights)
        )

    @pytest.mark.parametrize(
        ("n_broadcast_samples", "route"),
        [(8, "empirical_enumeration"), (4, "sampling_lift")],
    )
    def test_atoms_whose_declaration_leaves_the_dtype_open_map(self, n_broadcast_samples, route):
        """The probe reads an open dtype from the stored atoms, whether the lift enumerates or samples."""
        theta = self._records(8)
        assert set(theta.dtypes.values()) == {None}
        calls: list = []

        with workflow_run(seed=0):
            result = self._counted(calls, n_broadcast_samples=n_broadcast_samples)(theta)

        assert len(calls) < n_broadcast_samples
        assert result.provenance.metadata["route"] == route
        assert result.provenance.metadata["dispatch"] == "jax"

    def test_a_nested_lift_maps_the_enumeration_in_each_cell(self):
        theta = self._arrays(3, jnp.array([0.2, 0.3, 0.5]))
        cells = NumericArrayBatch(
            jnp.arange(4.0),
            "cell",
            label="c",
        )
        laws = {}
        counts = {}
        for dispatch in ("auto", "sequential"):
            calls: list = []

            def body(c, theta, calls=calls):
                calls.append(theta)
                return c * self._rate(theta)

            laws[dispatch] = Function(
                body,
                n_broadcast_samples=3,
                dispatch=dispatch,
                label="f",
                output_spec=OutputSpec(f=None),
            )(cells, theta)
            counts[dispatch] = len(calls)

        assert counts["sequential"] == 12 > counts["auto"]
        for index in range(4):
            np.testing.assert_allclose(
                np.asarray(laws["auto"][index].atoms),
                np.asarray(laws["sequential"][index].atoms),
                rtol=1e-6,
            )
            np.testing.assert_array_equal(
                np.asarray(laws["auto"][index].weights),
                np.asarray(laws["sequential"][index].weights),
            )

    @staticmethod
    def _object_atoms(kind: str) -> EmpiricalDistribution:
        if kind == "opaque":
            return EmpiricalDistribution(
                OpaqueBatch(
                    ["a", "bb", "ccc"],
                    "atom",
                    label="labels",
                ),
                component="s",
            )
        laws = [Normal("x", loc=float(loc), scale=1.0) for loc in range(3)]
        return EmpiricalDistribution(
            DistributionBatch(
                laws,
                "law",
                label="laws",
            ),
            component="laws",
        )

    @pytest.mark.parametrize("kind", ["opaque", "laws"])
    def test_object_atoms_enumerate_by_the_loop(self, kind):
        calls: list = []

        def body(atom):
            calls.append(atom)
            return jnp.asarray(1.0)

        law = self._object_atoms(kind)
        result = Function(body, n_broadcast_samples=3, label="f", output_spec=OutputSpec(f=None))(
            law
        )

        # The body receives each stored atom itself, once.
        atoms = law._atoms_at(np.arange(3))
        assert len(calls) == 3
        assert all(call is atom for call, atom in zip(calls, atoms, strict=True))
        assert result.provenance.metadata["dispatch"] == "sequential"
        np.testing.assert_array_equal(np.asarray(result.atoms), np.ones(3))

    def test_a_body_that_does_not_trace_enumerates_by_the_loop(self):
        theta = self._arrays(4, jnp.array([0.1, 0.2, 0.3, 0.4]))

        def untraceable(theta):
            return jnp.asarray(float(theta[0]) ** 2)

        result = Function(
            untraceable, n_broadcast_samples=4, label="f", output_spec=OutputSpec(f=None)
        )(theta)
        sequential = Function(
            untraceable,
            n_broadcast_samples=4,
            dispatch="sequential",
            label="f",
            output_spec=OutputSpec(f=None),
        )(theta)

        assert result.provenance.metadata["dispatch"] == "sequential"
        np.testing.assert_array_equal(np.asarray(result.atoms), np.asarray(sequential.atoms))
        np.testing.assert_array_equal(np.asarray(result.weights), np.asarray(sequential.weights))

    @pytest.mark.parametrize("kind", ["opaque", "untraceable"])
    def test_jax_dispatch_refuses_an_enumeration_that_does_not_trace(self, kind):
        if kind == "opaque":
            law, body = self._object_atoms("opaque"), lambda atom: jnp.asarray(1.0)
        else:
            law, body = self._arrays(4), lambda theta: jnp.asarray(float(theta[0]))

        with pytest.raises(ValueError, match="dispatch='jax' failed while tracing"):
            Function(
                body,
                n_broadcast_samples=4,
                dispatch="jax",
                label="f",
            )(law)

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    @pytest.mark.parametrize(("shift", "holds"), [(1.0, True), (-1.0, False)])
    def test_a_declared_support_is_checked_on_the_mapped_atoms(self, dispatch, shift, holds):
        """The atoms run from 0.5 to 1.5, so a shift of -1.0 leaves some outside the support."""
        calls: list = []

        def body(theta):
            calls.append(theta)
            return self._rate(theta) + shift

        wrapped = Function(
            body,
            output_spec=OutputSpec(f=NumericArraySpec((), support=positive)),
            n_broadcast_samples=50,
            dispatch=dispatch,
            label="f",
        )

        if holds:
            result = wrapped(self._arrays(50))
            assert len(calls) < 5
            assert bool(jnp.all(jnp.asarray(result.atoms.values) > 0))
        else:
            with pytest.raises(ResultSchemaError, match="support positive"):
                wrapped(self._arrays(50))
            assert len(calls) < 5


class TestARecordReturnLiftsToARecordLaw:
    """A record-returning function lifts to an empirical law over the record (V.6, V.10)."""

    @pytest.fixture
    def transform(self):
        @function(n_broadcast_samples=128, dispatch="sequential")
        def transform(x, y):
            return Record(
                {"sum": x + y, "diff": x - y},
                label="r",
            )

        return transform

    @staticmethod
    def _laws():
        return {
            "x": Normal("x", loc=1.0, scale=0.1),
            "y": Normal("y", loc=2.0, scale=0.1),
        }

    def test_the_result_is_an_empirical_law_over_the_record(self, transform):
        with workflow_run(seed=0):
            result = transform(**self._laws())
        assert isinstance(result, EmpiricalDistribution)
        assert result.event_spec.exposes_record
        assert list(result.event_spec.components) == ["sum", "diff"]

    def test_the_mean_and_variance_are_per_field(self, transform):
        with workflow_run(seed=0):
            result = transform(**self._laws())
        means, variances = result._mean(), result._variance()
        # sum ~ N(3, 0.02) and diff ~ N(-1, 0.02) for independent x and y.
        tolerance = 3.0 * np.sqrt(0.02) / np.sqrt(Function.DEFAULT_N_BROADCAST_SAMPLES)
        np.testing.assert_allclose(float(means["sum"]), 3.0, atol=tolerance)
        np.testing.assert_allclose(float(means["diff"]), -1.0, atol=tolerance)
        np.testing.assert_allclose(float(variances["sum"]), 0.02, atol=0.02)
        np.testing.assert_allclose(float(variances["diff"]), 0.02, atol=0.02)

    def test_draws_are_a_batch_of_records_on_the_sample_level(self, transform):
        with workflow_run(seed=0):
            result = transform(**self._laws())
            drawn = sample(result, sample_shape=(5,))
        assert (drawn.batch_shape, drawn.level_names) == ((5,), ("sample",))
        assert list(drawn.event_template) == ["sum", "diff"]
        assert drawn["sum"].shape == drawn["diff"].shape == (5,)
