"""Contract tests of evaluate, whose evaluation-rule registry the engine provides."""

from __future__ import annotations

import pytest

from probpipe.operations import RouteSource
from probpipe.operations._evaluate import evaluate
from probpipe.values import Function

from ._laws import Gaussian


def test_the_route_is_the_evaluation_rule_registry():
    (route,) = evaluate.summary().routes
    assert (route.name, route.source, route.exact) == (
        "evaluation_rules",
        RouteSource.REGISTRY,
        None,
    )


@pytest.mark.pending(reason="the evaluation-rule registry realizes evaluate")
def test_a_map_evaluates_at_a_value():
    assert float(evaluate(Function("f", lambda x: x + 1.0), 1.0)) == 2.0


@pytest.mark.pending(reason="the evaluation-rule registry realizes evaluate")
def test_a_map_pushes_a_distribution_forward():
    assert evaluate(Function("f", lambda x: 2.0 * x), Gaussian("g")) is not None
