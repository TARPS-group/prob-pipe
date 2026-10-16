"""Contract tests of pushforward, whose evaluation-rule registry the engine provides."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import (
    ApplicabilityError,
    Distribution,
    EmpiricalDistribution,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    OutputSpec,
    Record,
    conditional_distribution,
    expectation,
    log_prob,
    mean,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, Feasibility
from probpipe.distributions._batches import DistributionBatch
from probpipe.families import BootstrapDistribution
from probpipe.functions import _rules
from probpipe.operations import RouteSource
from probpipe.operations._pushforward import pushforward
from probpipe.values import Function

from ._laws import REAL, Gaussian

#: The draws of a Monte Carlo estimate.
_DRAWS = 4000


def _value(term: Any) -> float:
    return float(jnp.asarray(term))


def _bootstrap() -> BootstrapDistribution:
    """A random measure: a draw is the empirical law of three draws of the atoms 0, 1, 2."""
    return BootstrapDistribution("D", EmpiricalDistribution(jnp.arange(3.0), component="y"), 3)


def test_the_route_is_the_evaluation_rule_registry():
    (route,) = pushforward.summary().routes
    assert (route.name, route.source, route.exact) == (
        "evaluation_rules",
        RouteSource.REGISTRY,
        None,
    )


def test_a_map_pushes_a_law_forward():
    with workflow_run(seed=0):
        law = pushforward(
            Gaussian("g"),
            Function(lambda x: 2.0 * x, label="f", output_spec=OutputSpec(f=None)),
        )
    assert isinstance(law, EmpiricalDistribution)
    assert law.provenance.operation == "workflow.pushforward"


def test_a_maps_own_max_evaluations_survives_pushforward():
    """pushforward forwards a control only when its caller set it, so the map keeps its own count."""
    double = Function(
        lambda x: 2.0 * x,
        max_evaluations=9,
        dispatch="sequential",
        label="double",
        output_spec=OutputSpec(double=None),
    )
    with workflow_run(seed=0):
        assert pushforward(Gaussian("g"), double).num_atoms == 9
        assert pushforward.using(max_evaluations=5)(Gaussian("g"), double).num_atoms == 5


def test_the_result_takes_the_maps_output_name():
    double = Function(
        lambda x: 2.0 * x,
        output_label="doubled",
        label="double",
        output_spec=OutputSpec(doubled=None),
    )
    with workflow_run(seed=0):
        assert pushforward(Gaussian("g"), double).label == "doubled"


def test_a_nameless_map_needs_a_labeled_function():
    double = partial(lambda x, k: k * x, k=2.0)
    with pytest.raises(
        TypeError,
        match=r"^pushforward needs an explicit label for a partial, which has no name; "
        r"wrap it as Function\(f, label=\.\.\.\)$",
    ):
        pushforward(Gaussian("g"), double)
    with workflow_run(seed=0):
        law = pushforward(Gaussian("g"), Function(double, label="double"))
    assert isinstance(law, EmpiricalDistribution)


def test_the_registry_is_exported_beside_the_converter_registry():
    import probpipe

    assert probpipe.evaluation_rule_registry is _rules.evaluation_rule_registry
    assert probpipe.functions.evaluation_rule_registry is _rules.evaluation_rule_registry


def test_evaluate_is_no_longer_exported():
    import probpipe

    assert not hasattr(probpipe, "evaluate")
    assert "pushforward" in probpipe.__all__


def test_the_fixed_arguments_bind_the_other_parameters():
    def shift(x, offset):
        return x + offset

    atoms = EmpiricalDistribution(jnp.array([0.0, 1.0]), component="x")
    law = pushforward(atoms, shift, fixed_args={"offset": 2.0})
    assert _value(mean(law)) == pytest.approx(2.5)


def test_a_map_left_with_two_open_parameters_is_refused():
    def add(x, y):
        return x + y

    with pytest.raises(ApplicabilityError, match="exactly one parameter"):
        pushforward(Gaussian("g"), Function(add, label="add"))


def test_a_map_left_with_no_open_parameter_is_refused_without_naming_others():
    def add(x, y):
        return x + y

    with pytest.raises(ApplicabilityError, match="no parameter of 'add' is left open") as info:
        pushforward(Gaussian("g"), Function(add, label="add"), fixed_args={"x": 1.0, "y": 2.0})
    assert "pass the others" not in str(info.value)


def _weighted_atoms():
    return EmpiricalDistribution(
        jnp.array([0.0, 1.0, 2.0]), weights=jnp.array([0.2, 0.3, 0.5]), component="e"
    )


@pytest.mark.parametrize("method", ["sampling_lift", "evaluation_rules/sampling_lift"])
def test_a_named_rule_runs_as_the_direct_call_runs_it(method):
    square = Function(lambda t: t * t, label="square", output_spec=OutputSpec(square=None))
    with workflow_run(seed=0):
        pushed = pushforward.using(method=method, max_evaluations=8)(_weighted_atoms(), square)
    with workflow_run(seed=0):
        direct = square.using(method="sampling_lift", max_evaluations=8)(_weighted_atoms())
    assert pushed.num_atoms == direct.num_atoms == 8


def test_an_empirical_law_is_enumerated_exactly():
    law = pushforward(_weighted_atoms(), lambda t: t * t)
    metadata = law.provenance.metadata
    assert (metadata["method"], metadata["exact"]) == ("empirical_enumeration", True)
    assert _value(mean(law)) == pytest.approx(0.3 + 0.5 * 4.0)


def test_check_reports_the_rule_a_method_names():
    square = Function(lambda t: t * t, label="square")
    report = pushforward.using(method="sampling_lift").check(_weighted_atoms(), square)
    assert (report.route, report.method, report.exact) == (
        "evaluation_rules",
        "sampling_lift",
        False,
    )


class TestTheMapReceivesDraws:
    """The map receives draws of the law whatever its parameter's annotation."""

    def test_a_parameter_annotated_any_receives_draws(self):
        received: list[Any] = []

        def square(x: Any) -> Any:
            received.append(x)
            return x * x

        law = pushforward(_weighted_atoms(), square)
        assert received and not any(isinstance(x, Distribution) for x in received)
        assert _value(mean(law)) == pytest.approx(0.3 + 0.5 * 4.0)

    def test_a_parameter_annotated_distribution_receives_the_draws_of_a_random_measure(self):
        received: list[type] = []

        def statistic(D: Distribution) -> Any:
            received.append(type(D))
            return mean(D)

        with workflow_run(seed=0):
            law = pushforward.using(max_evaluations=1000)(_bootstrap(), statistic)
        assert law.num_atoms == 1000
        assert received and all(issubclass(kind, EmpiricalDistribution) for kind in received)
        # The mean of three draws of the atoms 0, 1, 2 has mean 1 and variance
        # (2/3)/3 = 2/9. Over seeds 0 to 3 the estimates were within 0.017 and
        # 0.005 of these.
        assert _value(mean(law)) == pytest.approx(1.0, abs=0.06)
        assert _value(variance(law)) == pytest.approx(2.0 / 9.0, abs=0.04)

    def test_an_optional_distribution_receives_the_draws_of_a_random_measure(self):
        def statistic(D: Distribution | None) -> Any:
            return mean(D)

        with workflow_run(seed=0):
            law = pushforward.using(max_evaluations=4)(_bootstrap(), statistic)
        assert law.num_atoms == 4

    def test_an_operation_receives_draws(self):
        with workflow_run(seed=0):
            law = pushforward.using(max_evaluations=400)(_bootstrap(), mean)
        assert law.num_atoms == 400
        # The replicate means have mean 1 and standard deviation sqrt(2/9), so
        # 400 of them give a standard error of about 0.024; over seeds 0 to 3
        # the estimate was within 0.024 of 1.
        assert _value(mean(law)) == pytest.approx(1.0, abs=0.1)
        assert _value(variance(law)) > 0.0

    def test_an_operation_receives_draws_where_its_parameter_samples(self):
        """log_prob's value parameter samples the law, so its rule runs on a map of its own."""
        with workflow_run(seed=0):
            law = pushforward.using(max_evaluations=_DRAWS)(
                Gaussian("g"), log_prob, {"d": Gaussian("g")}
            )
        # E[log N(X; 0, 1)] = -(log(2 pi) + 1) / 2 for X ~ N(0, 1). Over seeds 0
        # to 3 the estimate from 4000 draws was within 0.019 of it.
        expected = -0.5 * (jnp.log(2.0 * jnp.pi) + 1.0)
        assert _value(mean(law)) == pytest.approx(float(expected), abs=0.05)

    @pytest.mark.parametrize("method", ["empirical_enumeration", "sampling_lift"])
    def test_a_positional_only_parameter_receives_draws(self, method):
        law = pushforward.using(method=method)(_weighted_atoms(), jnp.exp)
        assert isinstance(law, EmpiricalDistribution)

    def test_a_maps_construction_bind_is_kept(self):
        def shift(x, a):
            return x + a

        bound = Function(shift, label="shift", bind={"a": 10.0})
        assert _value(mean(pushforward(_weighted_atoms(), bound))) == pytest.approx(11.3)

    def test_a_plain_callable_is_admitted_as_the_map(self):
        law = pushforward.using(max_evaluations=4)(
            Normal("x", 0.0, 1.0), lambda x: {"doubled": 2.0 * x}
        )
        assert law.num_atoms == 4

    def test_the_identity_over_a_field_view_gives_one_atom_per_draw(self):
        """The map returns its operand, so the stacked draws' term reaches the result step."""
        view = (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0))["a"]
        law = pushforward.using(max_evaluations=3)(
            view, Function(lambda x: x, output_spec=OutputSpec(a=None))
        )
        assert law.num_atoms == 3
        assert law.atoms.element_spec.shape == ()


class _Keyed(Function):
    """A map type that a rule keys on."""


class _KeyedRule(BinaryDispatchMethod):
    """An exact rule for the map type ``_Keyed``, which returns the point mass at 7."""

    name = "keyed"
    exact = True
    priority = 50

    def supported_types(self):
        return ((_Keyed,), (Distribution,))

    def check(self, f, operand, /, **call):
        return Feasibility(True)

    def execute(self, f, operand, /, **call):
        return EmpiricalDistribution(jnp.array([7.0]), component="keyed")


def test_a_rule_keyed_on_the_maps_type_applies_where_the_direct_call_samples(monkeypatch):
    registry = type(_rules.evaluation_rule_registry)()
    for rule in (
        _rules._SamplingLift(),
        _rules._ElementwiseSweep(),
        _rules._EmpiricalEnumeration(),
        _KeyedRule(),
    ):
        registry.register(rule)
    monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)
    (route,) = pushforward._route_table.routes
    monkeypatch.setattr(route, "registry", registry)
    law = pushforward(Gaussian("g"), _Keyed(lambda x: x, label="keyed_map"))
    assert law.provenance.metadata["method"] == "keyed"
    assert _value(mean(law)) == 7.0


class TestTheOperand:
    def test_a_value_is_refused_at_admission(self):
        with pytest.raises(ApplicabilityError, match="parameter 'd' accepts DistributionSpec"):
            pushforward(2.0, lambda x: x * x)

    def test_a_batch_of_values_is_refused_at_admission(self):
        rows = NumericArrayBatch(
            jnp.arange(3.0), "row", element_spec=NumericArraySpec(()), label="rows"
        )
        with pytest.raises(ApplicabilityError, match="parameter 'd' accepts DistributionSpec"):
            pushforward(rows, lambda x: x + 1.0)

    def test_a_batch_of_laws_gives_the_batch_of_their_pushforwards(self):
        laws = DistributionBatch(
            [
                EmpiricalDistribution(jnp.array([0.0, 1.0]), component="x"),
                EmpiricalDistribution(jnp.array([2.0, 4.0]), component="x"),
            ],
            "law",
            label="laws",
        )
        pushed = pushforward(laws, lambda x: x + 1.0)
        assert isinstance(pushed, DistributionBatch)
        assert [_value(mean(law)) for law in pushed] == [1.5, 4.0]

    def test_a_kernel_with_given_is_conditioned_first(self):
        kernel = conditional_distribution(
            lambda mu: EmpiricalDistribution(jnp.stack([mu, mu + 2.0]), component="y"),
            given_spec={"mu": REAL},
        )
        with workflow_run(seed=0):
            law = pushforward(kernel, lambda y: 2.0 * y, given={"mu": 1.0})
        assert _value(mean(law)) == pytest.approx(4.0, abs=0.5)

    def test_a_kernel_without_given_is_refused(self):
        kernel = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec={"mu": REAL})
        with pytest.raises(ApplicabilityError, match="requires given values"):
            pushforward(kernel, lambda y: y)


def _the_law(law: Distribution) -> Any:
    return law


class TestTheDomainCondition:
    """The map's open parameter must admit a draw of the law."""

    @pytest.mark.parametrize("call", ["call", "check"])
    def test_a_parameter_taking_a_law_refuses_a_numeric_draw(self, call):
        refusal = (
            r"^pushforward: cannot pass a draw of argument 'd' to '_the_law', since its "
            r"parameter 'law' takes a distribution, but the draw is an array$"
        )
        with pytest.raises(ApplicabilityError, match=refusal):
            getattr(pushforward, "__call__" if call == "call" else "check")(Gaussian("g"), _the_law)

    def test_the_body_does_not_run_and_nothing_is_drawn(self):
        calls: list[Any] = []

        def recorded(law: Distribution) -> Any:
            calls.append(law)
            return law

        class _Counting(Gaussian):
            draws = 0

            def _sample(self, key, sample_shape=()):
                type(self).draws += 1
                return super()._sample(key, sample_shape)

        with pytest.raises(ApplicabilityError):
            pushforward(_Counting("g"), recorded)
        assert calls == [] and _Counting.draws == 0

    def test_an_optional_law_refuses_a_numeric_draw(self):
        def optional(law: Distribution | None) -> Any:
            return law

        with pytest.raises(ApplicabilityError, match="takes a distribution"):
            pushforward(Gaussian("g"), optional)

    def test_an_operation_whose_role_takes_laws_refuses_a_numeric_draw(self):
        with pytest.raises(ApplicabilityError, match=r"'mean'.*takes a distribution"):
            pushforward(Gaussian("g"), mean)

    def test_a_term_class_annotation_refuses_another_kind(self):
        def fields(r: Record) -> Any:
            return r

        with pytest.raises(ApplicabilityError, match="takes a record, but the draw is an array"):
            pushforward(Gaussian("g"), fields)

    def test_an_input_spec_refuses_a_draw_of_another_shape(self):
        norm = Function(
            lambda v: jnp.sum(v * v),
            label="norm",
            input_spec={"v": NumericArraySpec((3,))},
        )
        with pytest.raises(ApplicabilityError, match="input_spec of parameter 'v'"):
            pushforward(Gaussian("g"), norm)

    def test_a_bare_none_annotation_admits_nothing(self):
        def nothing(x: None) -> Any:
            return x

        with pytest.raises(ApplicabilityError, match="takes no value"):
            pushforward(Gaussian("g"), nothing)

    def test_a_mismatched_fixed_argument_is_named_in_the_refusal(self):
        dot = Function(
            lambda x, w: jnp.dot(x, w),
            label="dot",
            input_spec={"x": NumericArraySpec(("n",)), "w": NumericArraySpec(("n",))},
        )
        laws = EmpiricalDistribution(jnp.ones((3, 2)), component="x")
        with pytest.raises(
            ApplicabilityError, match=r"fixed_args\['w'\] does not match the input_spec"
        ):
            pushforward(laws, dot, {"w": jnp.ones(3)})

    def test_a_kernel_given_values_is_refused_the_same_way(self):
        kernel = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec={"mu": REAL})
        with pytest.raises(ApplicabilityError, match="takes a distribution"):
            pushforward(kernel, _the_law, given={"mu": 1.0})

    @pytest.mark.parametrize("annotation", [float, jnp.ndarray, Any])
    def test_an_annotation_that_names_no_kind_admits_the_call(self, annotation):
        def identity(x):
            return x

        identity.__annotations__ = {"x": annotation}
        assert pushforward.check(_weighted_atoms(), identity).feasible


class TestFixedArgumentsAreConstants:
    def _shift(self, x, a):
        return x + a

    @pytest.mark.parametrize("call", ["call", "check"])
    def test_a_law_is_refused(self, call):
        shift = Function(self._shift, label="shift")
        with pytest.raises(
            ApplicabilityError,
            match=r"^pushforward: fixed_args\['a'\] must be a fixed value, but got "
            r"Normal 'Normal', which is a distribution$",
        ):
            getattr(pushforward, "__call__" if call == "call" else "check")(
                Gaussian("g"), shift, {"a": Normal("a", 100.0, 1.0)}
            )

    @pytest.mark.parametrize("call", ["call", "check"])
    def test_a_batch_is_refused(self, call):
        shift = Function(self._shift, label="shift")
        batch = NumericArrayBatch(jnp.array([0.0, 1.0]), "i", label="a")
        with pytest.raises(ApplicabilityError, match="got NumericArrayBatch 'a', which is a batch"):
            getattr(pushforward, "__call__" if call == "call" else "check")(
                Gaussian("g"), shift, {"a": batch}
            )

    def test_a_law_at_a_parameter_taking_a_law_is_a_constant(self):
        def scored(x, reference: Distribution):
            return x + mean(reference)

        law = pushforward(_weighted_atoms(), scored, {"reference": Normal("r", 10.0, 1.0)})
        assert _value(mean(law)) == pytest.approx(1.3 + 10.0)


def test_include_inputs_is_not_forwarded():
    """The pushforward is the law of the map's output alone, whatever the map's own control."""
    double = Function(lambda x: 2.0 * x, label="double", include_inputs=True)
    law = pushforward(_weighted_atoms(), double)
    assert _value(mean(law)) == pytest.approx(2.0 * 1.3)


def test_describe_lists_the_evaluation_rules():
    from probpipe import operation_registry

    text = operation_registry.describe("pushforward")
    assert text.startswith("pushforward(d, f, fixed_args, given) — primitive")
    assert "evaluation_rules (registry, exactness delegated)" in text


@pytest.mark.parametrize("operation", [pushforward, expectation])
@pytest.mark.parametrize("method", [None, "empirical_enumeration", "sampling_lift"])
@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize(
    "bound",
    [
        Normal("a", 10.0, 1.0),
        NumericArrayBatch(jnp.array([10.0, 20.0]), "i", label="a"),
    ],
)
def test_bound_arguments_must_be_constants(operation, method, check, bound):
    calls = []

    def shift(x, a):
        calls.append(x)
        return x + a

    mapped = Function(shift, label="shift", bind={"a": bound})
    op = operation.using(method=method)
    with pytest.raises(ApplicabilityError, match=r"fixed_args\['a'\] must be a fixed value"):
        (op.check if check else op)(_weighted_atoms(), mapped)
    assert calls == []


@pytest.mark.parametrize("operation", [pushforward, expectation])
def test_explicit_constant_overrides_a_bound_law(operation):
    mapped = Function(lambda x, a: x + a, label="shift", bind={"a": Normal("a", 10.0, 1.0)})
    result = operation(_weighted_atoms(), mapped, {"a": 2.0})
    assert _value(mean(result) if operation is pushforward else result) == pytest.approx(3.3)


@pytest.mark.parametrize("dispatch", ["jax", "sequential"])
@pytest.mark.parametrize("output_size", [2, 3])
def test_any_map_preserves_shared_input_output_dimensions(dispatch, output_size):
    def mapped(x: Any):
        return jnp.ones(output_size)

    fn = Function(
        mapped,
        label="mapped",
        input_spec={"x": NumericArraySpec(("n",))},
        output_spec=NumericArraySpec(("n",)),
    )
    law = EmpiricalDistribution(jnp.ones((2, 2)), component="x")
    op = pushforward.using(dispatch=dispatch)
    if output_size == 3:
        with pytest.raises(ValueError) as error:
            op(law, fn)
        cause = error.value.__cause__ if dispatch == "jax" else error.value
        assert "already bound to 2" in str(cause)
    else:
        result = op(law, fn)
        assert result.event_spec.spec.shape == (2,)
        assert jnp.array_equal(mean(result).raw(), jnp.ones(2))
