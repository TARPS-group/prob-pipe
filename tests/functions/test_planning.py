"""Planning, step 5 of the stack (design V.6).

Planning unifies the available specs and computes the result's
output spec as far as they determine it, deferring what only the return can
settle:

- a bare output spec: completes to ``OutputSpec.default``;
- an unspecified output type: is filled per call, and the output spec keeps it;
- a lift: plans a distribution over the output spec;
- ``include_inputs``: returns the joint law of the lifted inputs and the output.
- ``check``: reports the call's output spec, whose term spec is the spec of the
  batch or law that a lifted call returns.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    ApplicabilityError,
    BatchSpec,
    DistributionBatch,
    DistributionSpec,
    EmpiricalDistribution,
    Function,
    InputSpec,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    RecordSpec,
    function,
    mean,
    workflow_run,
)
from probpipe.core._specs import _has_record_outputs
from probpipe.core.constraints import positive

from ._design_helpers import error_of, one_field_law, standard_normal

_RATE = OutputSpec(rate=NumericArraySpec(("obs",), jnp.float32, positive))


def _rate_function() -> Function:
    @function(output_spec=_RATE)
    def rate(x):
        return jnp.exp(x)

    return rate


def _lifted_rate() -> Function:
    @function(output_spec=_RATE, max_evaluations=6, executor="sequential")
    def rate(x, z):
        return jnp.exp(x + z)

    return rate


class TestTheDeclaredOutput:
    def test_a_bare_term_spec_completes_to_the_default_declaration(self):
        spec = NumericArraySpec(())
        wrapped = Function(
            lambda x: x,
            output_label="value",
            output_spec=OutputSpec.default(spec, output_name="value"),
            label="f",
        )

        assert wrapped.output_spec == OutputSpec.default(spec, output_name="value")

    def test_a_bare_record_spec_exposes_its_fields(self):
        spec = RecordSpec(a=NumericArraySpec(()))
        wrapped = Function(
            lambda x: {"a": x},
            output_spec=spec,
            label="f",
        )

        assert _has_record_outputs(wrapped.output_spec)
        assert set(wrapped.output_spec.outputs) == {"a"}

    def test_a_type_hole_is_filled_per_call_and_the_declaration_keeps_it(self):
        wrapped = Function(
            lambda x: x,
            output_spec=OutputSpec(mean=None),
            label="f",
        )

        first = wrapped(jnp.ones(2))
        second = wrapped(jnp.ones(3))

        assert first.shape == (2,) and second.shape == (3,)
        assert wrapped.output_spec.term_spec is None

    def test_output_only_dimensions_bind_from_the_returned_term_on_each_call(self):
        rate = _rate_function()

        assert rate(jnp.zeros(4)).spec.shape == (4,)
        assert rate(jnp.zeros(2)).spec.shape == (2,)

    def test_the_declaration_carries_the_support_inference_would_not(self):
        assert _rate_function()(jnp.zeros(3)).spec.support is positive


class TestTheResultOfALift:
    def test_a_lift_keeps_the_declared_outputs(self):
        with workflow_run(seed=0):
            result = _lifted_rate()(jnp.zeros(4), standard_normal())

        assert set(result.event_spec.outputs) == {"rate"}

    def test_a_lift_plans_a_distribution_over_the_output_declaration(self):
        with workflow_run(seed=0):
            result = _lifted_rate()(jnp.zeros(4), standard_normal())

        assert not _has_record_outputs(result.event_spec)
        assert isinstance(result.event_spec.term_spec, NumericArraySpec)
        assert result.event_spec.term_spec.support is positive

    def test_an_undeclared_array_return_lifts_to_a_whole_term_under_output_name(self):
        @function(max_evaluations=6, executor="sequential", output_spec=OutputSpec(square=None))
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert set(result.event_spec.outputs) == {"square"}
        assert not _has_record_outputs(result.event_spec)

    def test_the_sampling_lift_constructs_an_empirical_approximation(self):
        @function(max_evaluations=6, executor="sequential", output_spec=OutputSpec(square=None))
        def square(x):
            return x * x

        with workflow_run(seed=0):
            result = square(standard_normal())

        assert isinstance(result, EmpiricalDistribution)

    def test_an_empty_sweep_without_a_declaration_is_refused(self):
        @function(executor="sequential")
        def double(x):
            return 2.0 * x

        empty = NumericArrayBatch(
            jnp.zeros((0,)),
            "row",
            element=NumericArraySpec(()),
            label="rows",
        )

        assert error_of(lambda: double(empty)) is not None


#: Each executor of an empty sweep, with the claim that makes it the one that
#: would run: JAX maps a traceable function whatever executor is named, so the
#: one-at-a-time executors take ``traceable=False``, and None selects automatically.
_EXECUTORS = [
    pytest.param("sequential", False, id="sequential"),
    pytest.param("thread", False, id="thread"),
    pytest.param(None, None, id="automatic"),
    pytest.param("jax", None, id="jax"),
]


class TestEmptySweepPlanning:
    @pytest.mark.parametrize("executor, traceable", _EXECUTORS)
    @pytest.mark.parametrize(
        "declaration",
        [None, OutputSpec(value=None), NumericArraySpec(("n",)), RecordSpec(x=("n",))],
        ids=["undeclared", "unspecified-type", "unbound-array", "unbound-record"],
    )
    @pytest.mark.parametrize("check", [False, True], ids=["execute", "check"])
    def test_an_incomplete_declaration_is_refused_without_running_the_body(
        self, executor, traceable, declaration, check
    ):
        @function(executor=executor, traceable=traceable, output_spec=declaration)
        def unexpected(x):
            raise AssertionError("An empty sweep must not evaluate the body")

        empty = NumericArrayBatch(jnp.zeros((0,)), "row", label="rows")
        call = unexpected.check if check else unexpected
        with pytest.raises(ValueError, match="empty sweep requires a concrete output_spec"):
            call(empty)

    @pytest.mark.parametrize("executor, traceable", _EXECUTORS)
    def test_input_dimensions_complete_an_empty_sweeps_output(self, executor, traceable):
        @function(
            executor=executor,
            traceable=traceable,
            input_spec=InputSpec(x=NumericArraySpec(("n",))),
            output_spec=NumericArraySpec(("n",)),
        )
        def unexpected(x):
            raise AssertionError("An empty sweep must not evaluate the body")

        empty = NumericArrayBatch(jnp.zeros((0, 2)), "row", label="rows")
        planned = unexpected.check(empty).result.term_spec
        assert planned.level_names == ("row",)
        assert planned.element_spec.term_spec.shape == (2,)
        result = unexpected(empty)
        assert isinstance(result, NumericArrayBatch)
        assert result.batch_shape == (0,)
        assert result.element_spec.term_spec.shape == (2,)
        assert result.raw.shape == (0, 2)

    def test_an_operation_derives_the_empty_sweeps_declaration(self):
        import numpy as np

        empty = DistributionBatch(
            np.empty((0,), dtype=object),
            "row",
            element=DistributionSpec(OutputSpec(x=NumericArraySpec(()))),
            label="laws",
        )
        planned = mean.check(empty).result.term_spec
        assert planned.level_names == ("row",)
        assert planned.element_spec.term_spec.shape == ()
        result = mean(empty)
        assert isinstance(result, NumericArrayBatch)
        assert result.batch_shape == (0,)
        assert result.element_spec.term_spec.shape == ()


#: A scalar output declared at the dtype the stored results take. A planned spec
#: that states every dtype equals the returned term's spec, which ``is_valid``
#: compares by equality; return completes a dtype the declaration leaves open.
_FLOAT = NumericArraySpec((), jnp.float32)
_SCALAR = OutputSpec(f=_FLOAT)


def _scalar_function(**controls) -> Function:
    @function(output_spec=_SCALAR, max_evaluations=6, **controls)
    def f(x):
        return 2.0 * x

    return f


def _items() -> NumericArrayBatch:
    return NumericArrayBatch(jnp.array([1.0, 2.0]), "item", label="X")


class TestTheCheckedResult:
    """``check`` reports the call's output spec, whose term spec is the returned term's."""

    def test_a_plain_call_reports_the_functions_output_spec(self):
        assert _scalar_function().check(1.0).result == _SCALAR

    @pytest.mark.parametrize("executor, traceable", _EXECUTORS)
    def test_a_sweep_reports_a_batch_on_the_swept_levels(self, executor, traceable):
        f = _scalar_function(executor=executor, traceable=traceable)
        planned = f.check(_items()).result
        assert planned == OutputSpec(f=BatchSpec(_FLOAT, item=2))
        result = f(_items())
        assert planned.term_spec.is_valid(result)
        assert planned.term_spec.level_names == result.level_names

    def test_return_completes_what_a_swept_declaration_leaves_open(self):
        @function(output_spec=OutputSpec(f=NumericArraySpec(("n",))))
        def spread(x):
            return x * jnp.ones(3)

        planned = spread.check(_items()).result
        assert planned == OutputSpec(f=BatchSpec(NumericArraySpec(("n",)), item=2))
        returned = spread(_items()).spec
        assert returned.level_names == planned.term_spec.level_names
        assert returned.element_spec.term_spec.shape == (3,)
        assert returned.element_spec.term_spec.dtype == jnp.float32

    def test_a_swept_record_keeps_the_positional_form(self):
        record = RecordSpec(a=_FLOAT, b=_FLOAT)

        @function(output_spec=record)
        def pair(x):
            return {"a": x, "b": 2.0 * x}

        planned = pair.check(_items()).result
        assert planned == OutputSpec(BatchSpec(record, item=2))
        assert planned.term_spec.is_valid(pair(_items()))

    def test_a_returned_batch_adds_its_levels_after_the_swept_ones(self):
        @function(output_spec=OutputSpec(f=BatchSpec(_FLOAT, inner=3)))
        def spread(x):
            return NumericArrayBatch(x * jnp.ones(3), "inner", label="spread")

        planned = spread.check(_items()).result
        assert planned == OutputSpec(f=BatchSpec(_FLOAT, item=2, inner=3))
        assert planned.term_spec.is_valid(spread(_items()))

    def test_a_returned_batch_on_a_swept_level_raises_as_the_call_does(self):
        @function(output_spec=OutputSpec(f=BatchSpec(_FLOAT, item=3)))
        def spread(x):
            return NumericArrayBatch(x * jnp.ones(3), "item", label="spread")

        checked, called = (
            error_of(lambda: spread.check(_items())),
            error_of(lambda: spread(_items())),
        )
        assert isinstance(checked, ValueError) and isinstance(called, ValueError)
        assert str(checked) == str(called)
        assert "cannot sweep 'spread': its returned batch repeats the swept level 'item'" in str(
            checked
        )

    def test_a_broadcast_reports_a_law_over_the_outputs(self):
        f = _scalar_function()
        planned = f.check(standard_normal()).result
        assert planned == OutputSpec(DistributionSpec(_SCALAR))
        with workflow_run(seed=0):
            result = f(standard_normal())
        assert planned.term_spec.is_valid(result)
        assert tuple(planned.term_spec.event_spec.outputs) == tuple(result.event_spec.outputs)

    def test_a_broadcast_of_returned_laws_holds_each_law_under_the_output_name(self):
        # The event holds each law in the keyword form, an interim form of the
        # one-field record that design V.6 names.
        law = DistributionSpec(OutputSpec(y=NumericArraySpec(())))

        @function(output_spec=OutputSpec(law), max_evaluations=6)
        def centered(x):
            return Normal("y", x, 1.0)

        planned = centered.check(standard_normal()).result
        assert planned == OutputSpec(DistributionSpec(OutputSpec(centered=law)))
        with workflow_run(seed=0):
            result = centered(standard_normal())
        assert planned.term_spec.is_valid(result)

    def test_including_the_inputs_reports_the_joint_law(self):
        f = _scalar_function(include_inputs=True)
        planned = f.check(standard_normal()).result
        with workflow_run(seed=0):
            result = f(standard_normal())
        assert tuple(planned.term_spec.event_spec.outputs) == ("x", "f")
        assert tuple(result.event_spec.outputs) == ("x", "f")
        assert planned.term_spec.is_valid(result)

    @pytest.mark.parametrize(
        "declared", [NumericArraySpec(()), None], ids=["specified", "unspecified"]
    )
    def test_including_an_input_named_like_an_output_raises_at_check(self, declared):
        @function(include_inputs=True, max_evaluations=6, output_spec=OutputSpec(x=declared))
        def double(x):
            return 2.0 * x

        with pytest.raises(ApplicabilityError, match="cannot add parameter 'x' to the output"):
            double.check(standard_normal())

    def test_a_nested_sweep_reports_a_batch_of_laws(self):
        @function(output_spec=_SCALAR, max_evaluations=6)
        def f(x, scale):
            return x * scale

        planned = f.check(standard_normal(), _items()).result
        assert planned == OutputSpec(BatchSpec(DistributionSpec(_SCALAR), item=2))
        with workflow_run(seed=0):
            result = f(standard_normal(), _items())
        assert isinstance(result, DistributionBatch)
        assert planned.term_spec.is_valid(result)

    @pytest.mark.parametrize(
        "declaration, event",
        [
            (_SCALAR, _SCALAR),
            (OutputSpec(RecordSpec(a=_FLOAT)), OutputSpec(RecordSpec(a=_FLOAT))),
            (
                OutputSpec(DistributionSpec(_SCALAR)),
                OutputSpec(f=DistributionSpec(_SCALAR)),
            ),
        ],
        ids=["array", "record", "law"],
    )
    def test_an_empty_nested_sweep_returns_the_planned_batch_of_laws(self, declaration, event):

        @function(output_spec=declaration, traceable=False)
        def f(x, scale):
            raise AssertionError("An empty nested sweep must not evaluate the body")

        empty = NumericArrayBatch(jnp.zeros(0), "item", label="empty")
        planned = f.check(standard_normal(), empty).result
        with workflow_run(seed=0):
            result = f(standard_normal(), empty)
        assert isinstance(result, DistributionBatch)
        assert result.batch_shape == (0,)
        assert result.event_spec == event
        assert planned.term_spec.is_valid(result)

    def test_an_operation_swept_over_laws_reports_a_batch(self):
        laws = DistributionBatch([standard_normal(), Normal("z", 1.0, 1.0)], "row", label="laws")
        planned = mean.check(laws).result
        assert planned.term_spec.level_names == ("row",)
        assert planned.term_spec.is_valid(mean(laws))

    @pytest.mark.parametrize("argument", [_items, standard_normal], ids=["sweep", "broadcast"])
    def test_a_lift_of_an_unspecified_type_reports_no_result(self, argument):
        @function(output_spec=OutputSpec(f=None), max_evaluations=6)
        def f(x):
            return 2.0 * x

        report = f.check(argument())
        assert report.result is None
        assert report.deferred

    def test_a_plain_call_of_an_unspecified_type_reports_the_named_output(self):
        @function(output_spec=OutputSpec(f=None))
        def f(x):
            return 2.0 * x

        assert f.check(1.0).result == OutputSpec(f=None)

    def test_a_conversion_that_decides_the_lift_leaves_the_result_open(self):
        report = _scalar_function().check(tfd.Normal(0.0, 1.0))
        assert report.selected is None and report.routes[0].unresolved
        assert report.result is None

    @pytest.mark.parametrize("annotation", [Normal, Any], ids=["distribution", "any"])
    @pytest.mark.parametrize("swept", [False, True], ids=["plain", "sweep"])
    def test_a_conversion_consumed_whole_preserves_the_planned_result(
        self, annotation, swept, monkeypatch
    ):
        from probpipe.distributions._conversion import converter_registry

        def refuse(*args, **kwargs):
            raise AssertionError("check must not construct the conversion")

        monkeypatch.setattr(converter_registry, "convert", refuse)

        def consume(x, scale):
            raise AssertionError("check must not evaluate the body")

        consume.__annotations__["x"] = annotation
        f = Function(consume, output_spec=_SCALAR, traceable=False)
        report = f.check(tfd.Normal(0.0, 1.0), _items() if swept else 1.0)
        assert report.selected is None and report.routes[0].unresolved
        assert report.result == (OutputSpec(f=BatchSpec(_FLOAT, item=2)) if swept else _SCALAR)

    @pytest.mark.parametrize(
        ("role", "passes_whole"),
        [((OpaqueSpec, DistributionSpec), True), ((OpaqueSpec, NumericArraySpec), False)],
        ids=["admits-a-law", "admits-its-draws"],
    )
    def test_a_role_decides_whether_a_conversion_is_consumed_whole(self, role, passes_whole):
        class Consumer(Function):
            _roles = MappingProxyType({"x": role})

        def consume(x: Normal, scale):
            raise AssertionError("check must not evaluate the body")

        f = Consumer(consume, output_spec=_SCALAR, traceable=False)
        report = f.check(tfd.Normal(0.0, 1.0), 1.0)
        assert report.selected is None and report.routes[0].unresolved
        assert report.result == (_SCALAR if passes_whole else None)

    def test_one_unknown_conversion_keeps_the_result_open(self):
        @function(output_spec=_SCALAR, traceable=False)
        def f(x: Normal, y):
            raise AssertionError("check must not evaluate the body")

        report = f.check(tfd.Normal(0.0, 1.0), tfd.Normal(0.0, 1.0))
        assert report.result is None

    def test_a_call_without_a_feasible_route_reports_its_planned_result(self):
        report = _scalar_function().using(method="sampling_lift").check(_items())
        assert report.selected.feasible is False
        assert report.result == OutputSpec(f=BatchSpec(_FLOAT, item=2))


class TestIncludingTheInputs:
    def _predict(self) -> Function:
        @function(
            output_label="prediction",
            output_spec=OutputSpec(mean=None),
            include_inputs=True,
            max_evaluations=6,
            executor="sequential",
        )
        def predict(theta, x):
            return x * theta

        return predict

    def test_plain_inputs_contribute_no_fields(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert "x" not in result.event_spec.outputs
        assert result.label == "prediction"

    def test_the_joint_is_an_empirical_law(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert isinstance(result, EmpiricalDistribution)

    def test_each_lifted_parameter_and_each_exposed_output_is_a_field(self):
        with workflow_run(seed=0):
            result = self._predict()(theta=standard_normal(), x=jnp.ones(3))

        assert set(result.event_spec.outputs) == {"theta", "mean"}

    def test_a_one_field_record_draw_remains_nested(self):
        @function(
            include_inputs=True,
            max_evaluations=6,
            executor="sequential",
            output_spec=OutputSpec(weigh=None),
        )
        def weigh(theta):
            return 0.0

        with workflow_run(seed=0):
            result = weigh(theta=one_field_law())

        theta = result.event_spec.outputs["theta"]
        assert isinstance(theta, RecordSpec) and set(theta) == {"beta"}

    def test_a_parameter_named_like_an_output_raises(self):
        @function(include_inputs=True, max_evaluations=6, executor="sequential")
        def x(x):
            return x

        with workflow_run(seed=0):
            error = error_of(lambda: x(standard_normal()))

        assert isinstance(error, (ApplicabilityError, ValueError))


class TestFailures:
    def test_arguments_that_do_not_unify_raise_applicability_error(self):
        wrapped = Function(
            lambda x, y: x + y,
            input_spec={"x": NumericArraySpec(("n",)), "y": NumericArraySpec(("n",))},
            label="add",
        )

        assert isinstance(error_of(lambda: wrapped(jnp.ones(3), jnp.ones(4))), ApplicabilityError)
