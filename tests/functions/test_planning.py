"""Planning, step 5 of the stack (design V.6).

Planning unifies the available declarations and computes the result
declaration as far as they determine it, deferring what only the return can
settle:

- a bare output spec: completes to ``OutputSpec.default``;
- a type hole: is filled per call, and the declaration keeps it;
- a lift: plans a distribution over the output declaration;
- ``include_inputs``: returns the joint law of the lifted inputs and the output.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import (
    ApplicabilityError,
    EmpiricalDistribution,
    Function,
    NumericArraySpec,
    OutputSpec,
    RecordSpec,
    function,
    workflow_run,
)
from probpipe.core.constraints import positive

from ._design_helpers import error_of, one_field_law, standard_normal

_RATE = OutputSpec(rate=NumericArraySpec(("obs",), jnp.float32, positive))


def _rate_function() -> Function:
    @function(output_spec=_RATE)
    def rate(x):
        return jnp.exp(x)

    return rate


def _lifted_rate() -> Function:
    @function(output_spec=_RATE, n_broadcast_samples=6, dispatch="sequential")
    def rate(x, z):
        return jnp.exp(x + z)

    return rate


class TestTheDeclaredOutput:
    def test_a_bare_term_spec_completes_to_the_default_declaration(self):
        spec = NumericArraySpec(())
        wrapped = Function("f", lambda x: x, output_label="value", output_spec=spec)

        assert wrapped.output_spec == OutputSpec.default(spec, component="value")

    def test_a_bare_record_spec_exposes_its_fields(self):
        spec = RecordSpec(a=NumericArraySpec(()))
        wrapped = Function("f", lambda x: {"a": x}, output_spec=spec)

        assert wrapped.output_spec.exposes_record
        assert set(wrapped.output_spec.components) == {"a"}

    def test_a_type_hole_is_filled_per_call_and_the_declaration_keeps_it(self):
        wrapped = Function("f", lambda x: x, output_spec=OutputSpec(mean=None))

        first = wrapped(jnp.ones(2))
        second = wrapped(jnp.ones(3))

        assert first.shape == (2,) and second.shape == (3,)
        assert wrapped.output_spec.spec is None

    def test_output_only_dimensions_bind_from_the_returned_term_on_each_call(self):
        rate = _rate_function()

        assert rate(jnp.zeros(4)).spec.shape == (4,)
        assert rate(jnp.zeros(2)).spec.shape == (2,)

    def test_the_declaration_carries_the_support_inference_would_not(self):
        assert _rate_function()(jnp.zeros(3)).spec.support is positive


class TestTheResultOfALift:
    def test_a_lift_keeps_the_declared_components(self):
        with workflow_run(seed=0):
            result = _lifted_rate()(jnp.zeros(4), standard_normal())

        assert set(result.event_spec.components) == {"rate"}

    def test_a_lift_plans_a_distribution_over_the_output_declaration(self):
        with workflow_run(seed=0):
            result = _lifted_rate()(jnp.zeros(4), standard_normal())

        assert not result.event_spec.exposes_record
        assert isinstance(result.event_spec.spec, NumericArraySpec)
        assert result.event_spec.spec.support is positive

    def test_an_undeclared_array_return_lifts_to_a_whole_term_under_output_name(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert set(result.event_spec.components) == {"square"}
        assert not result.event_spec.exposes_record

    def test_the_sampling_lift_constructs_an_empirical_approximation(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert isinstance(result, EmpiricalDistribution)

    @pytest.mark.pending(
        reason="an empty sweep needs a declaration sufficient to construct its result",
        raises=AssertionError,
    )
    def test_an_empty_sweep_without_a_declaration_is_refused(self):
        from probpipe import NumericArrayBatch

        @function(dispatch="sequential")
        def double(x):
            return 2.0 * x

        empty = NumericArrayBatch("rows", jnp.zeros((0,)), "row", element_spec=NumericArraySpec(()))

        assert error_of(lambda: double(empty)) is not None


class TestIncludingTheInputs:
    def _predict(self) -> Function:
        @function(
            output_label="prediction",
            output_spec=OutputSpec(mean=None),
            include_inputs=True,
            n_broadcast_samples=6,
            dispatch="sequential",
        )
        def predict(theta, x):
            return x * theta

        return predict

    def test_plain_inputs_contribute_no_fields(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert "x" not in result.event_spec.components
        assert result.label == "prediction"

    def test_the_joint_is_an_empirical_law(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert isinstance(result, EmpiricalDistribution)

    def test_each_lifted_parameter_and_each_exposed_output_component_is_a_field(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert set(result.event_spec.components) == {"theta", "mean"}

    def test_a_one_field_record_draw_remains_nested(self):
        @function(include_inputs=True, n_broadcast_samples=6, dispatch="sequential")
        def weigh(theta):
            return 0.0

        with workflow_run(seed=0):
            result = weigh(theta=one_field_law())

        theta = result.event_spec.components["theta"]
        assert isinstance(theta, RecordSpec) and set(theta) == {"beta"}

    def test_a_parameter_named_like_an_output_component_raises(self):
        @function(include_inputs=True, n_broadcast_samples=6, dispatch="sequential")
        def x(x):
            return x

        with workflow_run(seed=0):
            error = error_of(lambda: x(standard_normal()))

        assert isinstance(error, (ApplicabilityError, ValueError))


class TestFailures:
    def test_arguments_that_do_not_unify_raise_applicability_error(self):
        wrapped = Function(
            "add",
            lambda x, y: x + y,
            input_spec={"x": NumericArraySpec(("n",)), "y": NumericArraySpec(("n",))},
        )

        assert isinstance(error_of(lambda: wrapped(jnp.ones(3), jnp.ones(4))), ApplicabilityError)
