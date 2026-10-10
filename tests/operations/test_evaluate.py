"""Contract tests of evaluate, whose evaluation-rule registry the engine provides."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import (
    ApplicabilityError,
    EmpiricalDistribution,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    ResolutionError,
    workflow_run,
)
from probpipe.functions import _rules
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


def test_a_map_evaluates_at_a_value():
    assert float(evaluate(Function("f", lambda x: x + 1.0), 1.0)) == 2.0


def test_a_map_pushes_a_distribution_forward():
    with workflow_run(seed=0):
        law = evaluate(Function("f", lambda x: 2.0 * x), Gaussian("g"))
    assert isinstance(law, EmpiricalDistribution)
    assert law.provenance.operation == "workflow.evaluate"


def test_a_maps_own_sample_count_survives_evaluate():
    """evaluate forwards a control only when its caller set it, so the map keeps its own count."""
    double = Function("double", lambda x: 2.0 * x, n_broadcast_samples=9, dispatch="sequential")
    with workflow_run(seed=0):
        assert evaluate(double, Gaussian("g")).num_atoms == 9
        assert evaluate.with_options(n_broadcast_samples=5)(double, Gaussian("g")).num_atoms == 5


def test_the_result_takes_the_maps_output_name():
    double = Function("double", lambda x: 2.0 * x, output_label="doubled")
    with workflow_run(seed=0):
        assert evaluate(double, Gaussian("g")).label == "doubled"
    assert evaluate(double, 1.0).label == "doubled"


def test_the_registry_is_exported_beside_the_converter_registry():
    import probpipe

    assert probpipe.evaluation_rule_registry is _rules.evaluation_rule_registry
    assert probpipe.functions.evaluation_rule_registry is _rules.evaluation_rule_registry


def test_the_fixed_arguments_bind_the_other_parameters():
    def shift(x, offset):
        return x + offset

    value = evaluate(Function("shift", shift), 1.0, fixed_args={"offset": 2.0})
    assert float(value) == 3.0


def test_a_map_left_with_two_open_parameters_is_refused():
    def add(x, y):
        return x + y

    with pytest.raises(ApplicabilityError, match="exactly one parameter"):
        evaluate(Function("add", add), 1.0)


def test_a_map_left_with_no_open_parameter_is_refused_without_naming_others():
    def add(x, y):
        return x + y

    with pytest.raises(ApplicabilityError, match="no parameter of 'add' is left open") as info:
        evaluate(Function("add", add), 1.0, fixed_args={"x": 1.0, "y": 2.0})
    assert "pass the others" not in str(info.value)


def test_a_batch_is_swept_elementwise():
    rows = NumericArrayBatch("rows", jnp.arange(3.0), "row", element_spec=NumericArraySpec(()))
    swept = evaluate(Function("f", lambda x: x + 1.0), rows)
    assert isinstance(swept, NumericArrayBatch)
    assert swept.level_names == ("row",)


def _weighted_atoms():
    return EmpiricalDistribution(
        jnp.array([0.0, 1.0, 2.0]), weights=jnp.array([0.2, 0.3, 0.5]), component="e"
    )


@pytest.mark.parametrize("method", ["sampling_lift", "evaluation_rules/sampling_lift"])
def test_a_named_rule_runs_as_the_direct_call_runs_it(method):
    square = Function("square", lambda t: t * t)
    with workflow_run(seed=0):
        evaluated = evaluate.with_options(method=method, n_broadcast_samples=8)(
            square, _weighted_atoms()
        )
    with workflow_run(seed=0):
        direct = square.with_options(method="sampling_lift", n_broadcast_samples=8)(
            _weighted_atoms()
        )
    assert evaluated.num_atoms == direct.num_atoms == 8


def test_check_reports_the_rule_a_method_names():
    square = Function("square", lambda t: t * t)
    report = evaluate.with_options(method="sampling_lift").check(square, _weighted_atoms())
    assert (report.route, report.method, report.exact) == (
        "evaluation_rules",
        "sampling_lift",
        False,
    )


def test_a_rule_named_for_a_value_is_refused_as_the_direct_call_refuses_it():
    square = Function("square", lambda t: t * t)
    with pytest.raises(ResolutionError, match="lifts nothing"):
        square.with_options(method="sampling_lift")(3.0)
    with pytest.raises(ResolutionError, match="lifts nothing"):
        evaluate.with_options(method="sampling_lift")(square, 3.0)


class TestThePushforwardOfAView:
    def test_the_identity_over_a_field_view_gives_one_atom_per_draw(self):
        """The map returns its operand, so the stacked draws' term reaches the result step."""
        view = (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0))["a"]
        law = evaluate.with_options(n_broadcast_samples=3)(lambda x: x, view)
        assert law.num_atoms == 3
        assert law.atoms.element_spec.shape == ()

    def test_a_plain_callable_is_admitted_as_the_map(self):
        law = evaluate.with_options(n_broadcast_samples=4)(lambda x: 2.0 * x, Normal("x", 0.0, 1.0))
        assert law.num_atoms == 4
