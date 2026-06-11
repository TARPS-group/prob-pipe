"""Tests for separating workflow controls from user kwargs."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from probpipe import BroadcastDistribution, Normal, WorkflowFunction, workflow_function


def test_workflow_function_options_sets_construction_defaults():
    @workflow_function.options(
        n_broadcast_samples=7,
        dispatch="sequential",
        seed=0,
    )
    def identity(x):
        return x

    result = identity(Normal(loc=0.0, scale=1.0, name="x"))

    assert result.num_atoms == 7


def test_legacy_decorator_options_warn():
    with pytest.warns(DeprecationWarning, match="workflow_function.options"):

        @workflow_function(n_broadcast_samples=5, dispatch="sequential", seed=0)
        def identity(x):
            return x

    result = identity(Normal(loc=0.0, scale=1.0, name="x"))
    assert result.num_atoms == 5


def test_bare_decorator_forms_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @workflow_function
        def bare(x):
            return x

        @workflow_function()
        def bare_parentheses(x):
            return x

    assert not any(
        issubclass(warning.category, DeprecationWarning)
        for warning in caught
    )
    assert bare(1.0)["bare"] == 1.0
    assert bare_parentheses(2.0)["bare_parentheses"] == 2.0


def test_with_options_controls_sample_count_and_include_inputs():
    def identity(x):
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
        seed=0,
    )

    result = wf.with_options(
        n_broadcast_samples=6,
        include_inputs=True,
    )(Normal(loc=0.0, scale=1.0, name="x"))

    assert isinstance(result, BroadcastDistribution)
    assert result.num_atoms == 6
    assert "x" in result.input_samples


def test_with_options_seed_restarts_sampling_state_for_one_call():
    def identity(x):
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=8,
        dispatch="sequential",
        seed=0,
    )
    normal = Normal(loc=0.0, scale=1.0, name="x")

    first = wf.with_options(seed=42)(normal)
    second = wf.with_options(seed=42)(normal)

    assert jnp.allclose(first.samples, second.samples)


def test_with_options_seed_is_separate_from_user_seed_parameter():
    def identity(x):
        return x

    def add_user_seed(x, seed):
        return x + seed

    normal = Normal(loc=0.0, scale=1.0, name="x")
    base = WorkflowFunction(
        func=identity,
        n_broadcast_samples=8,
        dispatch="sequential",
        seed=0,
    ).with_options(seed=42)(normal)
    wf = WorkflowFunction(
        func=add_user_seed,
        n_broadcast_samples=8,
        dispatch="sequential",
        seed=0,
    )

    first = wf.with_options(seed=42)(normal, seed=7.0)
    second = wf.with_options(seed=42)(normal, seed=7.0)

    assert jnp.allclose(first.samples["marginal"], second.samples["marginal"])
    assert jnp.allclose(first.samples["marginal"], base.samples["marginal"] + 7.0)


def test_formerly_reserved_names_are_user_parameters():
    @workflow_function
    def collect(seed, n_broadcast_samples, include_inputs, name, dispatch):
        return f"{seed}:{n_broadcast_samples}:{include_inputs}:{name}:{dispatch}"

    result = collect(
        seed=1,
        n_broadcast_samples=2,
        include_inputs=True,
        name="model",
        dispatch="local",
    )

    assert result["collect"] == "1:2:True:model:local"


def test_var_keyword_receives_formerly_reserved_names():
    seen = []

    def identity(x, **kwargs):
        seen.append(kwargs)
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
        seed=0,
    )
    normal = Normal(loc=0.0, scale=1.0, name="x")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wf.with_options(n_broadcast_samples=5)(
            x=normal,
            seed=42,
            n_broadcast_samples=99,
            include_inputs=True,
        )

    assert result.num_atoms == 5
    assert not any(
        issubclass(warning.category, DeprecationWarning)
        for warning in caught
    )
    assert seen == [
        {"seed": 42, "n_broadcast_samples": 99, "include_inputs": True},
    ] * 5


def test_legacy_call_time_override_warns_when_name_cannot_bind():
    def identity(x):
        return x

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=20,
        dispatch="sequential",
        seed=0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wf(Normal(loc=0.0, scale=1.0, name="x"), n_broadcast_samples=6)

    legacy_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert len(legacy_warnings) == 1
    assert "with_options" in str(legacy_warnings[0].message)
    assert legacy_warnings[0].filename.endswith("test_workflow_options_namespace.py")
    assert result.num_atoms == 6


def test_bindable_formerly_reserved_name_does_not_warn_or_override():
    def identity(x, n_broadcast_samples):
        return x + n_broadcast_samples

    wf = WorkflowFunction(
        func=identity,
        n_broadcast_samples=5,
        dispatch="sequential",
        seed=0,
    )
    normal = Normal(loc=0.0, scale=1.0, name="x")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wf(x=normal, n_broadcast_samples=4)

    assert result.num_atoms == 5
    assert not any(
        issubclass(warning.category, DeprecationWarning)
        for warning in caught
    )
