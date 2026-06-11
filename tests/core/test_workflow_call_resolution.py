"""Characterization tests for WorkflowFunction call resolution.

These tests lock down the public-call boundary before WorkflowFunction
internals are split into smaller private modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from probpipe import BroadcastDistribution, Normal
from probpipe.core import _workflow_call
from probpipe.core.node import Module, Node, WorkflowFunction, workflow_method


@pytest.fixture
def add_func():
    def add(x, y):
        return x + y

    return add


@pytest.fixture
def identity_func():
    def identity(x):
        return x

    return identity


@pytest.fixture
def kwargs_recorder():
    seen = []

    def identity(x, **kwargs):
        seen.append(kwargs)
        return x

    return identity, seen


@pytest.fixture
def normal_dist():
    return Normal(loc=0.0, scale=1.0, name="x")


@pytest.fixture
def affine_func():
    def affine(x, offset=10.0, scale=1.0):
        return (x + offset) * scale

    return affine


class DataNode(Node):
    pass


class CallResolutionModule(Module):
    @workflow_method
    def step(self, dep: DataNode, x, y=7.0):
        return x + y


def _resolve_call(
    func,
    *args,
    bind=None,
    module=None,
    default_n_broadcast_samples=20,
    default_include_inputs=False,
    **call_inputs,
):
    info = _workflow_call.make_signature_info(func)
    return _workflow_call.resolve_workflow_call(
        info,
        args,
        call_inputs,
        bind=bind or {},
        module=module,
        dependency_type=Node,
        workflow_name=getattr(func, "__name__", "workflow"),
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
    )


class TestWorkflowCallHelpers:
    def test_positional_and_mixed_arguments_bind_like_python_calls(self, add_func):
        assert _resolve_call(add_func, 1.0, 2.0).values == {"x": 1.0, "y": 2.0}
        assert _resolve_call(add_func, 1.0, y=2.0).values == {"x": 1.0, "y": 2.0}

    def test_duplicate_positional_and_keyword_argument_raises(self, add_func):
        with pytest.raises(TypeError, match="multiple values"):
            _resolve_call(add_func, 1.0, x=2.0)

    def test_var_keyword_expands_extra_keywords(self, kwargs_recorder):
        identity, _ = kwargs_recorder

        call = _resolve_call(identity, x=1.0, scale=2.0)

        assert call.values == {"x": 1.0, "scale": 2.0}

    def test_literal_kwargs_argument_is_not_unpacked(self, kwargs_recorder):
        identity, _ = kwargs_recorder

        call = _resolve_call(identity, x=1.0, kwargs={"scale": 2.0})

        assert call.values == {"x": 1.0, "kwargs": {"scale": 2.0}}

    def test_legacy_options_are_not_function_inputs_when_unbindable(self, identity_func):
        call = _resolve_call(
            identity_func,
            "value",
            n_broadcast_samples=6,
            include_inputs=True,
            seed=123,
        )

        assert call.values == {"x": "value"}
        assert call.overrides.n_broadcast_samples == 6
        assert call.overrides.include_inputs is True
        assert call.overrides.seed == 123

    def test_workflow_option_names_bind_when_declared(self):
        def identity(x, n_broadcast_samples, include_inputs, seed):
            return x

        call = _resolve_call(
            identity,
            "value",
            n_broadcast_samples=6,
            include_inputs=True,
            seed=123,
        )

        assert call.values == {
            "x": "value",
            "n_broadcast_samples": 6,
            "include_inputs": True,
            "seed": 123,
        }
        assert call.overrides.n_broadcast_samples == 20
        assert call.overrides.include_inputs is False
        assert call.overrides.seed is None

    def test_bind_module_and_function_defaults_resolve_in_precedence_order(self):
        dep = DataNode()
        module = SimpleNamespace(child_nodes={"dep": dep}, inputs={"x": 5.0})

        def step(dep: DataNode, x, y=7.0, scale=1.0):
            return (x + y) * scale

        call = _resolve_call(step, module=module, bind={"scale": 2.0})
        override = _resolve_call(step, module=module, bind={"scale": 2.0}, y=3.0)

        assert call.values == {"dep": dep, "x": 5.0, "y": 7.0, "scale": 2.0}
        assert override.values == {"dep": dep, "x": 5.0, "y": 3.0, "scale": 2.0}

    def test_module_wired_dependency_cannot_be_overridden_at_call_time(self):
        def step(dep: DataNode):
            return dep

        module = SimpleNamespace(child_nodes={"dep": DataNode()}, inputs={})

        with pytest.raises(TypeError, match="cannot be overridden"):
            _resolve_call(step, module=module, dep=DataNode())

    def test_dependency_typed_parameter_requires_node_instance(self):
        def use_dep(dep: DataNode):
            return dep

        with pytest.raises(TypeError, match="expects dependency 'dep:"):
            _resolve_call(use_dep, dep=object())


class TestArgumentBinding:
    def test_positional_and_mixed_arguments_bind_like_python_calls(self, add_func):
        wf = WorkflowFunction(func=add_func, dispatch="sequential")

        assert float(wf(jnp.asarray(1.0), jnp.asarray(2.0))) == 3.0
        assert float(wf(jnp.asarray(1.0), y=jnp.asarray(2.0))) == 3.0

    def test_duplicate_positional_and_keyword_argument_raises(self, add_func):
        wf = WorkflowFunction(func=add_func, dispatch="sequential")

        with pytest.raises(TypeError, match="multiple values"):
            wf(jnp.asarray(1.0), x=jnp.asarray(2.0))

    def test_var_keyword_expands_extra_keywords(self, kwargs_recorder):
        identity, seen = kwargs_recorder
        wf = WorkflowFunction(func=identity, dispatch="sequential")

        assert float(wf(x=1.0, scale=2.0)) == 1.0
        assert seen == [{"scale": 2.0}]

    def test_literal_kwargs_argument_is_not_unpacked(self, kwargs_recorder):
        identity, seen = kwargs_recorder
        wf = WorkflowFunction(func=identity, dispatch="sequential")

        assert float(wf(x=1.0, kwargs={"scale": 2.0})) == 1.0
        assert seen == [{"kwargs": {"scale": 2.0}}]

    def test_bind_values_and_function_defaults_are_resolved_before_call(self, affine_func):
        default_wf = WorkflowFunction(func=affine_func, dispatch="sequential")
        bound_wf = WorkflowFunction(
            func=affine_func,
            dispatch="sequential",
            bind={"offset": 3.0},
            scale=2.0,
        )

        assert float(default_wf(x=1.0)) == 11.0
        assert float(bound_wf(x=1.0)) == 8.0
        assert float(bound_wf(x=1.0, offset=4.0)) == 10.0

    def test_missing_required_input_raises_after_all_resolution_sources_fail(self, add_func):
        wf = WorkflowFunction(func=add_func, dispatch="sequential")

        with pytest.raises(TypeError, match="Missing required input 'y'"):
            wf(x=1.0)


class TestModuleResolution:
    def test_module_inputs_dependencies_and_defaults_resolve_for_workflow_methods(self):
        dep = DataNode()
        module = CallResolutionModule(dep=dep, x=5.0)

        assert module.child_nodes["dep"] is dep
        assert float(module.step()) == 12.0
        assert float(module.step(y=2.0)) == 7.0

    def test_module_wired_dependency_cannot_be_overridden_at_call_time(self):
        module = CallResolutionModule(dep=DataNode(), x=5.0)

        with pytest.raises(TypeError, match="cannot be overridden"):
            module.step(dep=DataNode())

    def test_dependency_typed_parameter_requires_node_instance(self):
        def use_dep(dep: DataNode):
            return 1.0

        wf = WorkflowFunction(func=use_dep, dispatch="sequential")

        with pytest.raises(TypeError, match="expects dependency 'dep:"):
            wf(dep=object())


class TestCallOptions:
    def test_with_options_controls_broadcast_execution(
        self,
        identity_func,
        normal_dist,
    ):
        wf = WorkflowFunction(
            func=identity_func,
            n_broadcast_samples=20,
            dispatch="sequential",
            seed=42,
        )

        result = wf.with_options(
            n_broadcast_samples=6,
            include_inputs=True,
            seed=42,
        )(normal_dist)

        assert isinstance(result, BroadcastDistribution)
        assert result.num_atoms == 6

    def test_seed_override_restarts_sampling_state_for_a_call(
        self,
        identity_func,
        normal_dist,
    ):
        wf = WorkflowFunction(
            func=identity_func,
            n_broadcast_samples=8,
            dispatch="sequential",
            seed=42,
        )

        first = wf.with_options(seed=42)(normal_dist)
        second = wf.with_options(seed=42)(normal_dist)

        assert jnp.allclose(first.samples, second.samples)
