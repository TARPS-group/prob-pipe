"""Resolution, step 6 of the stack (design V.7).

A plain call has its body as its one route. A lifted application resolves
through the evaluation-rule registry, which is a binary dispatch registry keyed
on the map's and the operand's types. Its floors are the sampling lift on a law
that samples and the elementwise sweep on a batch, and the selected route
records its fidelity in provenance.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import (
    Batch,
    Distribution,
    DistributionBatch,
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    Opaque,
    OutputSpec,
    ResolutionError,
    SupportsSampling,
    function,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, BinaryDispatchRegistry, Feasibility
from probpipe.functions import _resolution, _rules
from probpipe.functions._rules import FLOOR_PRIORITY, evaluation_rule_registry

from ._design_helpers import error_of, standard_normal

SCALAR = NumericArraySpec(())


def _identity(x):
    return x


def _rows() -> NumericArrayBatch:
    return NumericArrayBatch(
        jnp.arange(3.0),
        "row",
        element_spec=SCALAR,
        label="rows",
    )


def _normals() -> DistributionBatch:
    return DistributionBatch(
        [Normal("x", 0.0, 1.0), Normal("x", 0.0, 1.0)],
        "law",
        label="normals",
    )


def _recording(annotation: Any) -> tuple[Function, list[Any]]:
    """A function of ``x``, annotated *annotation* unless None, that records what it receives."""
    seen: list[Any] = []

    def body(x):
        seen.append(x)
        return 0.0

    if annotation is not None:
        body.__annotations__ = {"x": annotation}
    return Function(
        body,
        n_broadcast_samples=6,
        dispatch="sequential",
        label="body",
        output_spec=OutputSpec(result=None),
    ), seen


class _Unsampled(Distribution):
    """A law that declares its event and cannot sample."""


class _UnsampledView(Distribution):
    """A view that cannot sample itself, over the law it views."""

    def __init__(self, parent: Distribution) -> None:
        super().__init__(
            OutputSpec(view=SCALAR),
            label="view",
        )
        object.__setattr__(self, "_viewed", parent)

    @property
    def parent(self) -> Distribution:
        return self._viewed


class _ClosedForm(BinaryDispatchMethod):
    """A rule that returns a marker in place of the pushforward, at a chosen rank."""

    def __init__(self, *, exact: bool, priority: int, operand: type = Distribution) -> None:
        self._exact = exact
        self._priority = priority
        self._operand = operand

    @property
    def name(self) -> str:
        return "closed_form"

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int:
        return self._priority

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (self._operand,))

    def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
        return "closed form"


class TestTheRegistry:
    def test_the_registry_is_a_binary_dispatch_registry(self):
        assert isinstance(evaluation_rule_registry, BinaryDispatchRegistry)

    def test_the_sampling_lift_is_the_floor_on_a_law_that_samples(self):
        wrapped = Function(
            _identity,
            label="identity",
        )

        info = evaluation_rule_registry.check(wrapped, standard_normal(), parameter="x")

        assert info.feasible is True
        assert info.method_name == "sampling_lift"
        assert info.exact is False

    def test_the_elementwise_sweep_is_the_floor_on_a_batch(self):
        wrapped = Function(
            _identity,
            label="identity",
        )
        rows = NumericArrayBatch(
            jnp.arange(3.0),
            "row",
            element_spec=SCALAR,
            label="rows",
        )

        info = evaluation_rule_registry.check(wrapped, rows, parameter="x")

        assert info.feasible is True
        assert info.method_name == "elementwise_sweep"
        assert info.exact is True

    def test_the_sampling_lift_declines_a_law_that_cannot_sample(self):
        wrapped = Function(
            _identity,
            label="identity",
        )

        info = evaluation_rule_registry.check(
            wrapped,
            _Unsampled(
                OutputSpec(bare=SCALAR),
                label="bare",
            ),
            parameter="x",
        )

        assert info.feasible is False
        assert "SupportsSampling" in info.description

    def test_the_sampling_lift_declines_a_parameter_that_consumes_the_law(self):
        def consume(x: Distribution):
            return 0.0

        info = evaluation_rule_registry.check(
            Function(
                consume,
                label="consume",
            ),
            standard_normal(),
            parameter="x",
        )

        assert info.feasible is False

    def test_a_view_samples_through_its_parent(self):
        view = _UnsampledView(standard_normal())

        info = evaluation_rule_registry.check(
            Function(
                _identity,
                label="identity",
            ),
            view,
            parameter="x",
        )

        assert not isinstance(view, SupportsSampling)
        assert info.feasible is True
        assert info.method_name == "sampling_lift"

    def test_a_view_over_a_law_that_cannot_sample_is_declined(self):
        view = _UnsampledView(
            _Unsampled(
                OutputSpec(bare=SCALAR),
                label="bare",
            )
        )

        info = evaluation_rule_registry.check(
            Function(
                _identity,
                label="identity",
            ),
            view,
            parameter="x",
        )

        assert info.feasible is False
        assert "SupportsSampling" in info.description


class TestTheFloorTier:
    """A floor ranks below every other rule on its domain, whatever their exactness."""

    @pytest.mark.parametrize(
        ("floor", "operand"),
        [("sampling_lift", standard_normal), ("elementwise_sweep", _rows)],
        ids=["approximate-floor", "exact-floor"],
    )
    @pytest.mark.parametrize("priority", [100, FLOOR_PRIORITY - 1], ids=["above", "below"])
    def test_an_approximate_rule_is_selected_over_the_floor(self, floor, operand, priority):
        value = operand()
        registry = type(evaluation_rule_registry)()
        registry.register(evaluation_rule_registry.get_method(floor))
        rule_operand = Batch if isinstance(value, Batch) else Distribution
        registry.register(_ClosedForm(exact=False, priority=priority, operand=rule_operand))

        info = registry.check(
            Function(
                _identity,
                label="identity",
            ),
            value,
            parameter="x",
        )

        assert info.method_name == "closed_form"
        assert registry.list_methods() == ["closed_form", floor]

    def test_a_floor_raised_in_priority_stays_below_every_rule(self):
        registry = type(evaluation_rule_registry)()
        registry.register(evaluation_rule_registry.get_method("elementwise_sweep"))
        registry.register(_ClosedForm(exact=False, priority=0, operand=Batch))

        registry.set_priorities(elementwise_sweep=10**6)
        info = registry.check(
            Function(
                _identity,
                label="identity",
            ),
            _rows(),
            parameter="x",
        )

        assert info.method_name == "closed_form"

    def test_the_exact_floor_ranks_first_on_the_domain_the_floors_share(self):
        info = evaluation_rule_registry.check(
            Function(
                _identity,
                label="identity",
            ),
            _normals(),
            parameter="x",
        )

        assert info.method_name == "elementwise_sweep"


class TestTheFloorsAgreeWithTheDirectCall:
    """Each floor is feasible where the direct call takes it and nowhere else."""

    @pytest.mark.parametrize(
        ("annotation", "operand", "controls", "swept"),
        [
            (None, _rows, {}, True),
            (NumericArrayBatch, _rows, {}, False),
            (Any, _rows, {}, False),
            (None, _rows, {"include_inputs": True}, False),
            (None, _normals, {}, True),
            (DistributionBatch, _normals, {}, False),
        ],
        ids=[
            "unannotated",
            "consumes-the-batch",
            "any",
            "include-inputs",
            "distribution-batch",
            "consumes-the-batch-of-laws",
        ],
    )
    def test_the_sweep_is_feasible_where_the_call_sweeps(
        self, annotation, operand, controls, swept
    ):
        wrapped, seen = _recording(annotation)
        wrapped = wrapped.with_options(**controls)
        value = operand()

        info = evaluation_rule_registry.check(
            wrapped, value, parameter="x", controls=wrapped.options
        )
        error = error_of(lambda: wrapped(value))

        assert (info.feasible is True and info.method_name == "elementwise_sweep") is swept
        assert (error is None and bool(seen) and all(x is not value for x in seen)) is swept

    @pytest.mark.parametrize(
        ("annotation", "operand", "sampled"),
        [
            (None, standard_normal, True),
            (Distribution, standard_normal, False),
            (Any, standard_normal, False),
            (None, _normals, False),
            (Any, _normals, False),
        ],
        ids=[
            "unannotated",
            "consumes-the-law",
            "any-passes-the-law-whole",
            "distribution-batch",
            "distribution-batch-at-any",
        ],
    )
    def test_the_sampling_lift_is_feasible_where_the_call_samples(
        self, annotation, operand, sampled
    ):
        wrapped, seen = _recording(annotation)
        value = operand()

        info = evaluation_rule_registry.check(
            wrapped, value, parameter="x", controls=wrapped.options
        )
        with workflow_run(seed=0):
            error = error_of(lambda: wrapped(value))

        assert (info.feasible is True and info.method_name == "sampling_lift") is sampled
        drawn = error is None and len(seen) == wrapped.options["n_broadcast_samples"]
        assert (drawn and not any(isinstance(x, Distribution) for x in seen)) is sampled


class TestTheDirectCall:
    def test_a_direct_call_takes_a_registered_exact_rule(self, monkeypatch):
        registry = type(evaluation_rule_registry)()
        registry.register(_rules._SamplingLift())
        registry.register(_ClosedForm(exact=True, priority=0))
        monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)

        @function(n_broadcast_samples=6, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(standard_normal())

        assert isinstance(result, Opaque)
        assert result.value == "closed form"
        assert result.provenance.metadata["route"] == "closed_form"

    def test_exact_only_leaves_no_route_for_a_sampled_lift(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.with_options(exact_only=True)(standard_normal())

    def test_a_method_that_names_no_route_raises_resolution_error(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.with_options(method="quadrature")(standard_normal())

    def test_a_lift_with_no_feasible_route_names_the_missing_requirement(self):
        @function
        def identity(x):
            return x

        error = error_of(
            lambda: identity(
                _Unsampled(
                    OutputSpec(bare=SCALAR),
                    label="bare",
                )
            )
        )

        assert isinstance(error, ResolutionError)
        assert "SupportsSampling" in str(error)

    def test_the_selected_route_is_recorded_in_provenance(self):
        @function(
            n_broadcast_samples=6,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(standard_normal())

        recorded = str(dict(result.provenance.controls)) + str(dict(result.provenance.metadata))
        assert "sampling_lift" in recorded


class _Route:
    def __init__(self, label: str):
        self.label = label


class TestNoRouteMessage:
    def test_lists_each_route_tried(self):
        message = _resolution._no_route(
            "op",
            {},
            [
                (_Route("a"), Feasibility(False, "x is not a kernel")),
                (_Route("b"), Feasibility(False, "y")),
            ],
        )
        assert message == "op: no route applies. Tried: a: x is not a kernel; b: y"

    def test_leads_with_the_first_actionable_reason(self):
        probed = [
            (_Route("a"), Feasibility(False, "Normal is not a ConditionalDistribution")),
            (_Route("b"), Feasibility(False, "unknown field 'x'", actionable=True)),
        ]
        message = _resolution._no_route("condition_on", {}, probed)
        assert message == (
            "condition_on: unknown field 'x'. Routes tried: "
            "a: Normal is not a ConditionalDistribution; b: unknown field 'x'"
        )
