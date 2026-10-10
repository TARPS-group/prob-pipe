"""Automatic mathematical symbols and literal aliases have separate contracts."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArraySpec,
    OutputSpec,
    conditional_distribution,
    cov,
    expectation,
    mean,
    quantile,
    variance,
    workflow_run,
)
from probpipe.core._fingerprint import fingerprint


@pytest.mark.parametrize(
    "operation,symbol,component",
    [
        (mean, "𝔼", "mean"),
        (variance, "𝕍", "variance"),
        (cov, "ℂ", "cov"),
        (quantile, "ℚ", "quantile"),
    ],
)
def test_summary_symbols_preserve_ascii_components(operation, symbol, component):
    law = Normal("mu", 0.0, 1.0, label="prior")
    args = (law, 0.5) if operation is quantile else (law,)
    result = operation(*args)
    assert result.label == f"{symbol}[mu ~ prior]"
    assert tuple(operation.check(*args).result.components) == (f"{component}(mu)",)
    if component == "quantile":
        plural = quantile(law, jnp.array([0.1, 0.9]))
        assert plural.level_names == ("quantile",)


def test_anonymous_defaults_and_literal_names():
    anonymous = Function(lambda x: x + 1)
    assert anonymous.notation == "𝒻(x)"
    assert expectation(Normal("mu", 0, 1, label="prior"), lambda x: x).label == "𝔼[𝒻(mu ~ prior)]"
    kernel = conditional_distribution(
        lambda mu: Normal("y", mu, 1), given_spec={"mu": NumericArraySpec(())}
    )
    assert kernel.notation == "ℙ(y | mu)"
    assert EmpiricalDistribution(jnp.arange(3.0), component="mu").notation == "ℙ(mu)"

    def f(x):
        return x + 1

    assert Function(f).notation == "f(x)"
    for label in ("p", "f", "E", "Q", "Var", "Cov"):
        assert anonymous.with_label(label).label == label
        assert Function(lambda x: x, label=label).label == label
        assert Function(lambda x: x, output_label=label)(1).label == label
        assert Normal("mu", 0, 1, label=label).label == label


def test_anonymous_display_is_independent_of_schema_and_random_stream():
    automatic = Function(lambda x: x + 1, n_broadcast_samples=5)
    fn = automatic.raw()
    ascii_alias = Function(fn, label="f", n_broadcast_samples=5)
    assert fingerprint(automatic) == fingerprint(ascii_alias)
    with workflow_run(seed=42):
        first = automatic(Normal("mu", 0, 1))
    with workflow_run(seed=42):
        second = ascii_alias(Normal("mu", 0, 1))
    assert tuple(first.event_spec.components) == ("f",)
    assert tuple(second.event_spec.components) == ("f",)
    np.testing.assert_array_equal(first.atoms.raw(), second.atoms.raw())
    typed = Function(fn, output_spec=NumericArraySpec(()))
    assert typed.output_spec == OutputSpec(f=NumericArraySpec(()))
