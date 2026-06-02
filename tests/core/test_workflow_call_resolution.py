"""Characterization tests for WorkflowFunction call resolution.

These tests lock down the public-call boundary before WorkflowFunction
internals are split into smaller private modules.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import BroadcastDistribution, Normal
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


class TestReservedOverrides:
    def test_reserved_call_time_overrides_control_broadcast_execution(
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

        result = wf(normal_dist, n_broadcast_samples=6, include_inputs=True, seed=42)

        assert isinstance(result, BroadcastDistribution)
        assert result.n == 6

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

        first = wf(normal_dist, seed=42)
        second = wf(normal_dist, seed=42)

        assert jnp.allclose(first.samples, second.samples)
