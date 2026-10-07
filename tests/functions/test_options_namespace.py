"""Tests for separating workflow controls from user kwargs."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import pytest

import probpipe
import probpipe.functions as node
from probpipe import EmpiricalDistribution, Function, Normal, WorkflowKind, function, workflow_run


def test_function_is_the_only_public_wrapper_api():
    assert probpipe.Function is Function
    assert probpipe.function is function
    assert node.Function is Function
    assert node.function is function
    assert "WorkflowFunction" not in probpipe.__all__
    assert "workflow_function" not in probpipe.__all__
    assert not hasattr(probpipe, "WorkflowFunction")
    assert not hasattr(probpipe, "workflow_function")
    assert "WorkflowFunction" not in node.__all__
    assert "workflow_function" not in node.__all__
    assert not hasattr(node, "WorkflowFunction")
    assert not hasattr(node, "workflow_function")


def test_function_decorator_sets_construction_defaults():
    @function(
        n_broadcast_samples=7,
        dispatch="sequential",
    )
    def identity(x):
        return x

    with workflow_run(seed=0):
        result = identity(Normal(loc=0.0, scale=1.0, label="x"))

    assert result.num_atoms == 7


def test_function_has_no_options_alias():
    assert not hasattr(function, "options")


def test_function_rng_seed_controls_are_removed():
    def identity(x):
        return x

    wf = Function(label="identity", fn=identity, dispatch="sequential")

    assert "seed" not in inspect.signature(Function.__init__).parameters
    assert "seed" not in inspect.signature(wf.with_options).parameters
    with pytest.warns(FutureWarning, match="seed.*ignored"):
        deprecated = Function(label="identity", fn=identity, dispatch="sequential", seed=42)
    assert float(deprecated(3)) == 3
    with pytest.warns(FutureWarning, match="seed.*ignored"):

        @function(seed=42)
        def decorated_identity(x):
            return x

    assert float(decorated_identity(3)) == 3
    with pytest.raises(TypeError, match="seed"):
        wf.with_options(seed=42)


def test_function_construction_seed_warns_without_binding_user_parameter():
    def add_seed(x, seed):
        return x + seed

    with pytest.warns(FutureWarning, match="seed.*ignored"):
        wrapped = Function(label="add_seed", fn=add_seed, dispatch="sequential", seed=42)
    with pytest.raises(TypeError, match="seed"):
        wrapped(1)
    assert float(wrapped(1, seed=2)) == 3


def test_decorator_construction_seed_warns_without_binding_variadic_user_kwargs():
    with pytest.warns(FutureWarning, match="seed.*ignored"):

        @function(seed=42)
        def collect_seed(x, **kwargs):
            return x + kwargs.get("seed", 0)

    assert float(collect_seed(1)) == 1
    assert float(collect_seed(1, seed=2)) == 3


@pytest.mark.parametrize("option", ["input_template", "output_template"])
def test_removed_templates_warn_without_installing_declarations(option):
    with pytest.warns(FutureWarning, match=option):
        wrapped = Function("identity", lambda x: x, **{option: object()})
    assert wrapped.input_spec is None
    assert wrapped.output_spec is None
    assert float(wrapped(3)) == 3


def test_legacy_func_alias_warns_and_uses_the_replacement_signature():
    with pytest.warns(FutureWarning, match="func aliases fn"):
        wrapped = Function("replace", lambda x: x, func=lambda y: y + 1)
    assert tuple(wrapped.signature.parameters) == ("y",)
    assert float(wrapped(y=2)) == 3


def test_legacy_func_alias_validates_the_effective_callable():
    with (
        pytest.warns(FutureWarning, match="func aliases fn"),
        pytest.raises(TypeError, match="fn must be callable"),
    ):
        Function("invalid", lambda: 1, func=3)


@pytest.mark.parametrize("entrypoint", ["constructor", "decorator"])
@pytest.mark.parametrize("option", ["seed", "input_template", "output_template", "func"])
def test_legacy_option_warning_points_to_the_user_call(option, entrypoint):
    def identity(x):
        return x

    value = (lambda x: x + 1) if option == "func" else object()
    frame = inspect.currentframe()
    assert frame is not None
    with pytest.warns(FutureWarning) as caught:
        if entrypoint == "decorator":
            decorate = function(**{option: value})
            line = frame.f_lineno + 1
            wrapped = decorate(identity)
        else:
            line = frame.f_lineno + 1
            wrapped = Function("identity", identity, **{option: value})
    assert len(caught) == 1
    assert repr([option]) in str(caught[0].message)
    assert caught[0].filename == __file__
    assert caught[0].lineno == line
    assert float(wrapped(3)) == (4 if option == "func" else 3)


def test_one_warning_names_every_legacy_option():
    with pytest.warns(FutureWarning) as caught:
        Function(
            "identity",
            lambda x: x,
            func=lambda x: x,
            seed=1,
            input_template=object(),
            output_template=object(),
        )
    assert len(caught) == 1
    assert "['func', 'input_template', 'output_template', 'seed']" in str(caught[0].message)


def test_func_does_not_replace_the_required_fn_argument():
    with pytest.raises(TypeError, match="required positional argument: 'fn'"):
        Function(label="identity", func=lambda x: x)


def test_function_bind_can_still_supply_user_seed_parameter():
    def add_seed(x, seed):
        return x + seed

    wf = Function(label="add_seed", fn=add_seed, dispatch="sequential", bind={"seed": 42})

    assert float(wf(1.0)) == 43.0


def test_bare_decorator_forms_wrap_functions():
    @function
    def bare(x):
        return x

    @function()
    def bare_parentheses(x):
        return x

    assert isinstance(bare, Function)
    assert isinstance(bare_parentheses, Function)
    assert float(bare(1.0)) == 1.0
    assert float(bare_parentheses(2.0)) == 2.0


def test_with_options_controls_sample_count_and_include_inputs():
    def identity(x):
        return x

    wf = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
    )

    with workflow_run(seed=0):
        result = wf.with_options(
            n_broadcast_samples=6,
            include_inputs=True,
        )(Normal(loc=0.0, scale=1.0, label="x"))

    assert isinstance(result, EmpiricalDistribution)
    assert result.num_atoms == 6
    assert "x" in result.event_spec.components


def test_workflow_run_reproduces_one_lifted_call():
    def identity(x):
        return x

    wf = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=8,
        dispatch="sequential",
    )
    normal = Normal(loc=0.0, scale=1.0, label="x")

    with workflow_run(seed=42):
        first = wf(normal)
    with workflow_run(seed=42):
        second = wf(normal)

    assert jnp.allclose(first.atoms, second.atoms)


def test_workflow_seed_is_separate_from_user_seed_parameter():
    def identity(x):
        return x

    def add_user_seed(x, seed):
        return x + seed

    normal = Normal(loc=0.0, scale=1.0, label="x")
    base = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=8,
        dispatch="sequential",
    )
    wf = Function(
        label="add_user_seed",
        fn=add_user_seed,
        n_broadcast_samples=8,
        dispatch="sequential",
    )

    with workflow_run(seed=42):
        base_result = base(normal)
    with workflow_run(seed=42):
        first = wf(normal, seed=7.0)
    with workflow_run(seed=42):
        second = wf(normal, seed=7.0)

    assert jnp.allclose(first._rows, second._rows)
    assert jnp.allclose(first._rows, base_result._rows + 7.0)


def test_workflow_control_names_are_user_parameters():
    @function
    def collect(seed, n_broadcast_samples, include_inputs, name, dispatch):
        return f"{seed}:{n_broadcast_samples}:{include_inputs}:{name}:{dispatch}"

    result = collect(
        seed=1,
        n_broadcast_samples=2,
        include_inputs=True,
        name="model",
        dispatch="local",
    )

    assert result.value == "1:2:True:model:local"


def test_var_keyword_receives_workflow_control_names():
    seen = []

    def identity(x, **kwargs):
        seen.append(kwargs)
        return x

    wf = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
    )
    normal = Normal(loc=0.0, scale=1.0, label="x")

    with workflow_run(seed=0):
        result = wf.with_options(n_broadcast_samples=5)(
            x=normal,
            seed=42,
            n_broadcast_samples=99,
            include_inputs=True,
        )

    assert result.num_atoms == 5
    assert (
        seen
        == [
            {"seed": 42, "n_broadcast_samples": 99, "include_inputs": True},
        ]
        * 5
    )


def test_unbindable_call_time_control_name_is_rejected():
    def identity(x):
        return x

    wf = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        wf(Normal(loc=0.0, scale=1.0, label="x"), n_broadcast_samples=6)


def test_bindable_workflow_control_name_does_not_override():
    def identity(x, n_broadcast_samples):
        return x + n_broadcast_samples

    wf = Function(
        label="identity",
        fn=identity,
        n_broadcast_samples=5,
        dispatch="sequential",
    )
    normal = Normal(loc=0.0, scale=1.0, label="x")

    with workflow_run(seed=0):
        result = wf(x=normal, n_broadcast_samples=4)

    assert result.num_atoms == 5


def test_with_options_clears_workers_and_resets_sample_count():
    wrapped = Function(
        "identity", lambda x: x, dispatch="thread", max_workers=2, n_broadcast_samples=7
    )
    unchanged = wrapped.with_options(include_inputs=True)
    assert unchanged.options["max_workers"] == 2
    assert unchanged.options["n_broadcast_samples"] == 7

    reset = wrapped.with_options(max_workers=None, n_broadcast_samples=None)
    defaults = Function("defaults", lambda x: x, dispatch="thread")
    assert reset.options == defaults.options
    assert wrapped.options["max_workers"] == 2
    assert wrapped.options["n_broadcast_samples"] == 7
    with workflow_run(seed=3):
        result = reset(Normal("x", 0, 1))
    assert result.num_atoms == Function.DEFAULT_N_BROADCAST_SAMPLES


@pytest.mark.parametrize(
    ("control", "default", "error", "message"),
    [
        ("dispatch", "auto", ValueError, "dispatch must"),
        ("workflow_kind", WorkflowKind.DEFAULT, TypeError, "WorkflowKind"),
    ],
)
def test_a_view_resets_a_control_that_construction_refuses_as_none(
    control, default, error, message
):
    """A view's None resets a control to its default (V.2); construction refuses it."""
    wrapped = Function("identity", lambda x: x, dispatch="sequential")
    configured = wrapped.with_options(workflow_kind=WorkflowKind.OFF)
    assert configured.with_options(**{control: None}).options[control] == default
    with pytest.raises(error, match=message):
        Function("identity", lambda x: x, **{control: None})


@pytest.mark.parametrize("entrypoint", ["constructor", "decorator", "with_options"])
def test_max_workers_warning_points_to_the_user_call(entrypoint):
    def identity(x):
        return x

    wrapped = Function("identity", identity)
    frame = inspect.currentframe()
    assert frame is not None
    with pytest.warns(UserWarning, match="max_workers configures only") as caught:
        if entrypoint == "constructor":
            line = frame.f_lineno + 1
            Function("identity", identity, max_workers=2)
        elif entrypoint == "decorator":
            decorate = function(max_workers=2)
            line = frame.f_lineno + 1
            decorate(identity)
        else:
            line = frame.f_lineno + 1
            wrapped.with_options(max_workers=2)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == line
