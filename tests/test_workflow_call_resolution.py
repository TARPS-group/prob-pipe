"""Characterization tests for WorkflowFunction call resolution.

These tests lock down the public-call boundary before WorkflowFunction
internals are split into smaller private modules.
"""

import jax.numpy as jnp
import pytest

from probpipe import BroadcastDistribution, Normal
from probpipe.core.node import Module, Node, WorkflowFunction, workflow_method


def test_positional_and_mixed_arguments_bind_like_python_calls():
    def add(x, y):
        return x + y

    wf = WorkflowFunction(func=add, vectorize="loop")

    assert float(wf(jnp.asarray(1.0), jnp.asarray(2.0))) == 3.0
    assert float(wf(jnp.asarray(1.0), y=jnp.asarray(2.0))) == 3.0


def test_duplicate_positional_and_keyword_argument_raises():
    def add(x, y):
        return x + y

    wf = WorkflowFunction(func=add, vectorize="loop")

    with pytest.raises(TypeError, match="multiple values"):
        wf(jnp.asarray(1.0), x=jnp.asarray(2.0))


def test_var_keyword_expands_extra_keywords():
    seen = []

    def identity(x, **kwargs):
        seen.append(kwargs)
        return x

    wf = WorkflowFunction(func=identity, vectorize="loop")

    assert float(wf(x=1.0, scale=2.0)) == 1.0
    assert seen == [{"scale": 2.0}]


def test_literal_kwargs_argument_is_not_unpacked():
    seen = []

    def identity(x, **kwargs):
        seen.append(kwargs)
        return x

    wf = WorkflowFunction(func=identity, vectorize="loop")

    assert float(wf(x=1.0, kwargs={"scale": 2.0})) == 1.0
    assert seen == [{"kwargs": {"scale": 2.0}}]


def test_bind_values_and_function_defaults_are_resolved_before_call():
    def affine(x, offset=10.0, scale=1.0):
        return (x + offset) * scale

    default_wf = WorkflowFunction(func=affine, vectorize="loop")
    bound_wf = WorkflowFunction(
        func=affine,
        vectorize="loop",
        bind={"offset": 3.0},
        scale=2.0,
    )

    assert float(default_wf(x=1.0)) == 11.0
    assert float(bound_wf(x=1.0)) == 8.0
    assert float(bound_wf(x=1.0, offset=4.0)) == 10.0


def test_missing_required_input_raises_after_all_resolution_sources_fail():
    def add(x, y):
        return x + y

    wf = WorkflowFunction(func=add, vectorize="loop")

    with pytest.raises(TypeError, match="Missing required input 'y'"):
        wf(x=1.0)


class DataNode(Node):
    pass


class CallResolutionModule(Module):
    @workflow_method
    def step(self, dep: DataNode, x, y=7.0):
        assert dep is self.child_nodes["dep"]
        return x + y


def test_module_inputs_dependencies_and_defaults_resolve_for_workflow_methods():
    dep = DataNode()
    module = CallResolutionModule(dep=dep, x=5.0)

    assert float(module.step()) == 12.0
    assert float(module.step(y=2.0)) == 7.0


def test_module_wired_dependency_cannot_be_overridden_at_call_time():
    module = CallResolutionModule(dep=DataNode(), x=5.0)

    with pytest.raises(TypeError, match="cannot be overridden"):
        module.step(dep=DataNode())


def test_dependency_typed_parameter_requires_node_instance():
    def use_dep(dep: DataNode):
        return 1.0

    wf = WorkflowFunction(func=use_dep, vectorize="loop")

    with pytest.raises(TypeError, match="expects dependency 'dep:"):
        wf(dep=object())


def test_reserved_call_time_overrides_control_broadcast_execution():
    def identity(x):
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=20,
        vectorize="loop",
        seed=0,
    )
    dist = Normal(loc=0.0, scale=1.0, name="x")

    result = wf(dist, n_broadcast_samples=6, include_inputs=True, seed=123)

    assert isinstance(result, BroadcastDistribution)
    assert result.n == 6


def test_seed_override_restarts_sampling_state_for_a_call():
    def identity(x):
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=8,
        vectorize="loop",
        seed=0,
    )
    dist = Normal(loc=0.0, scale=1.0, name="x")

    first = wf(dist, seed=42)
    second = wf(dist, seed=42)

    assert jnp.allclose(first.samples, second.samples)
