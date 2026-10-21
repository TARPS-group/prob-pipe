"""Characterization tests for Function call resolution.

These tests lock down the public-call boundary before Function
internals are split into smaller private modules.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import EmpiricalDistribution, Function, Normal, OutputSpec, function, workflow_run
from probpipe.core._specs import _has_record_outputs
from probpipe.functions import _call
from probpipe.values import _binding


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
    return Normal("x", loc=0.0, scale=1.0)


@pytest.fixture
def affine_func():
    def affine(x, offset=10.0, scale=1.0):
        return (x + offset) * scale

    return affine


type _AnyAlias = Any


def _resolve_call(
    func,
    *args,
    bind=None,
    **call_inputs,
):
    info = _binding.make_signature_info(func)
    return _call.resolve_function_call(
        info,
        args,
        call_inputs,
        bind=bind or {},
        function_name=getattr(func, "__name__", "workflow"),
    )


class TestWorkflowCallHelpers:
    def test_input_refs_use_one_subscript_for_variadic_inputs(self):
        def collect(head, *items, **extras):
            return head, items, extras

        info = _binding.make_signature_info(collect)
        values = {
            "head": 1,
            "items": (2, 3),
            "extras": {"tail": 4},
        }

        refs = _binding.iter_input_refs(info, values)

        assert refs == (
            _binding.FunctionInputRef("head"),
            _binding.FunctionInputRef("items", subscript=0),
            _binding.FunctionInputRef("items", subscript=1),
            _binding.FunctionInputRef("extras", subscript="tail"),
        )
        assert tuple(ref.label for ref in refs) == (
            "head",
            "*items[0]",
            "*items[1]",
            "**extras['tail']",
        )
        assert tuple(_binding.input_ref_value(values, ref) for ref in refs) == (
            1,
            2,
            3,
            4,
        )

    @pytest.mark.parametrize("annotation", [Any, _AnyAlias], ids=["Any", "alias"])
    def test_variadic_any_is_not_a_planner_pass_through_hint(self, annotation):
        def collect(head, *items, **extras):
            return head, items, extras

        collect.__annotations__ = {"head": annotation, "items": annotation, "extras": annotation}
        info = _binding.make_signature_info(collect)

        assert _binding.input_ref_hint(info, _binding.FunctionInputRef("head")) is annotation
        assert (
            _binding.input_ref_hint(
                info,
                _binding.FunctionInputRef("items", subscript=0),
            )
            is None
        )
        assert (
            _binding.input_ref_hint(
                info,
                _binding.FunctionInputRef("extras", subscript="tail"),
            )
            is None
        )

    def test_input_ref_replacement_preserves_signature_shaped_containers(self):
        values = {
            "head": 1,
            "items": (2, 3),
            "extras": {"tail": 4},
        }
        head = _binding.FunctionInputRef("head")
        first_item = _binding.FunctionInputRef("items", subscript=0)
        tail = _binding.FunctionInputRef("extras", subscript="tail")

        singly_replaced = _binding.replace_input_ref(values, first_item, 20)
        jointly_replaced = _binding.replace_input_refs(
            values,
            {
                head: 10,
                first_item: 20,
                tail: 40,
            },
        )

        assert singly_replaced == {
            "head": 1,
            "items": (20, 3),
            "extras": {"tail": 4},
        }
        assert jointly_replaced == {
            "head": 10,
            "items": (20, 3),
            "extras": {"tail": 40},
        }

    def test_positional_and_mixed_arguments_bind_like_python_calls(self, add_func):
        assert _resolve_call(add_func, 1.0, 2.0) == {"x": 1.0, "y": 2.0}
        assert _resolve_call(add_func, 1.0, y=2.0) == {"x": 1.0, "y": 2.0}

    def test_duplicate_positional_and_keyword_argument_raises(self, add_func):
        with pytest.raises(TypeError, match="multiple values"):
            _resolve_call(add_func, 1.0, x=2.0)

    def test_var_keyword_preserves_signature_shaped_mapping(self, kwargs_recorder):
        identity, _ = kwargs_recorder

        call = _resolve_call(identity, x=1.0, scale=2.0)

        assert call == {"x": 1.0, "kwargs": {"scale": 2.0}}

    def test_literal_kwargs_argument_is_not_unpacked(self, kwargs_recorder):
        identity, _ = kwargs_recorder

        call = _resolve_call(identity, x=1.0, kwargs={"scale": 2.0})

        assert call == {"x": 1.0, "kwargs": {"kwargs": {"scale": 2.0}}}

    def test_unbindable_workflow_control_name_raises_like_python(self, identity_func):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _resolve_call(identity_func, "value", max_evaluations=6)

    def test_control_like_names_bind_when_declared(self):
        def identity(x, max_evaluations, include_inputs, seed):
            return x

        call = _resolve_call(
            identity,
            "value",
            max_evaluations=6,
            include_inputs=True,
            seed=123,
        )

        assert call == {
            "x": "value",
            "max_evaluations": 6,
            "include_inputs": True,
            "seed": 123,
        }

    def test_bind_and_defaults_resolve_in_precedence_order(self):
        def step(x, y=7.0, scale=1.0):
            return (x + y) * scale

        assert _resolve_call(step, bind={"x": 5.0, "scale": 2.0}) == {
            "x": 5.0,
            "y": 7.0,
            "scale": 2.0,
        }
        assert _resolve_call(step, bind={"x": 5.0, "scale": 2.0}, x=9.0) == {
            "x": 9.0,
            "y": 7.0,
            "scale": 2.0,
        }


class TestArgumentBinding:
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_function_values_bind_and_allow_call_overrides(self, wrapped):
        def increment(x):
            return x + 1.0

        def apply(f: Function, x):
            return f(x)

        def unannotated(f, x):
            return f(x)

        f = Function(increment, dispatch="sequential") if wrapped else increment
        for body in (apply, unannotated):
            bound = Function(body, bind={"f": f, "x": 3.0})
            assert bound.check().feasible is True
            assert float(bound()) == 4.0
            assert float(bound(x=4.0)) == 5.0
            assert float(bound(f=Function(lambda x: 2.0 * x))) == 6.0

    def test_positional_and_mixed_arguments_bind_like_python_calls(self, add_func):
        wf = Function(add_func, label="add_func", dispatch="sequential")

        assert float(wf(jnp.asarray(1.0), jnp.asarray(2.0))) == 3.0
        assert float(wf(jnp.asarray(1.0), y=jnp.asarray(2.0))) == 3.0

    def test_duplicate_positional_and_keyword_argument_raises(self, add_func):
        wf = Function(add_func, label="add_func", dispatch="sequential")

        with pytest.raises(TypeError, match="multiple values"):
            wf(jnp.asarray(1.0), x=jnp.asarray(2.0))

    def test_var_keyword_expands_extra_keywords(self, kwargs_recorder):
        identity, seen = kwargs_recorder
        wf = Function(identity, label="identity", dispatch="sequential")

        assert float(wf(x=1.0, scale=2.0)) == 1.0
        assert seen == [{"scale": 2.0}]

    def test_literal_kwargs_argument_is_not_unpacked(self, kwargs_recorder):
        identity, seen = kwargs_recorder
        wf = Function(identity, label="identity", dispatch="sequential")

        assert float(wf(x=1.0, kwargs={"scale": 2.0})) == 1.0
        assert seen == [{"kwargs": {"scale": 2.0}}]

    def test_bind_values_and_function_defaults_are_resolved_before_call(self, affine_func):
        default_wf = Function(affine_func, label="affine_func", dispatch="sequential")
        bound_wf = Function(
            affine_func,
            label="affine_func",
            dispatch="sequential",
            bind={"offset": 3.0, "scale": 2.0},
        )

        assert float(default_wf(x=1.0)) == 11.0
        assert float(bound_wf(x=1.0)) == 8.0
        assert float(bound_wf(x=1.0, offset=4.0)) == 10.0

    def test_missing_required_input_raises_after_all_resolution_sources_fail(self, add_func):
        wf = Function(add_func, label="add_func", dispatch="sequential")

        with pytest.raises(TypeError, match=r"add_func\(\) missing required argument 'y'"):
            wf(x=1.0)


class TestCallOptions:
    def test_using_controls_broadcast_execution(
        self,
        identity_func,
        normal_dist,
    ):
        wf = Function(
            identity_func,
            label="identity_func",
            max_evaluations=20,
            dispatch="sequential",
            output_spec=OutputSpec(identity_func=None),
        )

        with workflow_run(seed=42):
            result = wf.using(
                max_evaluations=6,
                include_inputs=True,
            )(normal_dist)

        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 6
        assert _has_record_outputs(result.event_spec)

    def test_workflow_run_reproduces_sampling_state_for_a_call(
        self,
        identity_func,
        normal_dist,
    ):
        wf = Function(
            identity_func,
            label="identity_func",
            max_evaluations=8,
            dispatch="sequential",
            output_spec=OutputSpec(identity_func=None),
        )

        with workflow_run(seed=42):
            first = wf(normal_dist)
        with workflow_run(seed=42):
            second = wf(normal_dist)

        assert jnp.allclose(first.atoms, second.atoms)


@pytest.mark.parametrize("engine", ["installed", "plain"])
def test_a_function_keyword_reaches_the_wrapped_callable(engine, monkeypatch):
    """The engine takes the Function positionally, so a parameter named function binds."""
    from probpipe.values import _function_base

    if engine == "plain":
        monkeypatch.setattr(_function_base, "_call_engine", _function_base._plain_call)
    wrapped = Function(
        lambda function, x: function(x),
        label="apply_fn",
    )

    result = wrapped(function=jnp.sin, x=1.0)

    assert float(result) == pytest.approx(float(jnp.sin(1.0)))
    assert float(wrapped.apply(function=jnp.sin, x=1.0)) == pytest.approx(float(result))


class TestOrdinaryCallableBinding:
    @pytest.mark.parametrize("evaluate", ["apply", "call"])
    def test_bound_method_uses_python_binding(self, evaluate):
        class Offset:
            def __init__(self, value):
                self.value = value

            def add(self, x, scale=1.0):
                return (x + self.value) * scale

        wrapped = Function(Offset(3.0).add, bind={"scale": 2.0})
        invoke = wrapped.apply if evaluate == "apply" else wrapped
        assert float(invoke(1.0)) == 8.0
        assert float(invoke(1.0, scale=3.0)) == 12.0

    @pytest.mark.parametrize("evaluate", ["apply", "call", "check"])
    def test_function_annotation_accepts_a_callable_value(self, evaluate):
        def evaluate_at(fn: Function, x):
            return fn.apply(x)

        wrapped = Function(evaluate_at)
        supplied = Function(lambda x: x + 2.0)
        if evaluate == "check":
            assert wrapped.check(supplied, 3.0).feasible is True
        else:
            invoke = wrapped.apply if evaluate == "apply" else wrapped
            assert float(invoke(supplied, 3.0)) == 5.0

    @pytest.mark.parametrize("constructor", [Function, function])
    def test_module_is_not_a_constructor_option(self, constructor):
        with pytest.raises(TypeError, match="module"):
            constructor(lambda: 1.0, module=object())

    def test_module_remains_an_ordinary_callable_parameter(self):
        def size(module):
            return module + 1.0

        wrapped = Function(size, bind={"module": 2.0})
        assert float(wrapped()) == 3.0
        assert float(wrapped(module=4.0)) == 5.0
        assert float(wrapped.apply(module=4.0)) == 5.0
