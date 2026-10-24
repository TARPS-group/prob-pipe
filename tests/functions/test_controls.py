"""Controls, resolved apart from the arguments of the wrapped function (design V.2).

A Function keeps the wrapped function's arguments and the framework's controls
in two namespaces. Each control resolves from the framework's default, then
the decorator or constructor, then a ``using`` view, which leaves the
original unchanged. A control that is unknown or inadmissible raises at the
decorator or at ``using``, before any call.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import (
    Distribution,
    EmpiricalDistribution,
    Function,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    SupportsMean,
    WorkflowKind,
    config,
    function,
    mean,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, Feasibility
from probpipe.functions import _rules

from ._design_helpers import error_of, standard_normal

#: The controls the framework defines, each with its default.
_FRAMEWORK_CONTROLS = {
    "max_evaluations": config.sampling.max_evaluations,
    "num_samples": config.sampling.num_samples,
    "include_inputs": False,
    "method": None,
    "exact_only": False,
    "conversions": {},
    "method_options": {},
    "raw": False,
    "executor": None,
    "executor_options": {},
    "workflow_kind": WorkflowKind.DEFAULT,
}


def _identity(x):
    return x


class TestTwoNamespaces:
    @pytest.mark.parametrize("name", ["max_evaluations", "raw", "method", "seed", "key"])
    def test_a_call_keyword_binds_to_the_authored_signature(self, name):
        def body(x, **kwargs):
            return jnp.asarray(float(kwargs[name]))

        result = Function(
            body,
            label="body",
        )(1.0, **{name: 3})

        assert float(result.raw) == 3.0

    def test_a_call_keyword_named_like_a_control_does_not_set_it(self):
        def body(x, max_evaluations=None, raw=None):
            return x

        wrapped = Function(
            body, executor="sequential", label="body", output_spec=OutputSpec(body=None)
        )
        with workflow_run(seed=0):
            lifted = wrapped(standard_normal(), max_evaluations=3, raw=True)

        assert lifted.num_atoms == config.sampling.max_evaluations
        assert lifted.provenance is not None

    @pytest.mark.parametrize("name", ["name", "output_label", "output_spec", "input_spec"])
    def test_construction_metadata_is_not_a_control(self, name):
        wrapped = Function(
            _identity,
            label="identity",
        )

        with pytest.raises(TypeError, match="unknown Function option"):
            wrapped.using(**{name: "x"})

    @pytest.mark.parametrize("name", ["seed", "key"])
    def test_there_is_no_framework_key_or_seed_control(self, name):
        wrapped = Function(
            _identity,
            label="identity",
        )

        with pytest.raises(TypeError, match="unknown Function option"):
            wrapped.using(**{name: 0})

    def test_a_wrapped_functions_own_seed_parameter_is_an_ordinary_argument(self):
        @function
        def draw(x, seed):
            return jnp.asarray(x + seed)

        assert float(draw(1.0, seed=2).raw) == 3.0


class TestResolution:
    def test_every_control_has_a_default(self):
        options = Function(
            _identity,
            label="identity",
        ).options

        for name, default in _FRAMEWORK_CONTROLS.items():
            assert options[name] == default, name

    def test_the_decorator_overrides_the_framework_default(self):
        @function(max_evaluations=7)
        def identity(x):
            return x

        assert identity.options["max_evaluations"] == 7

    def test_a_view_overrides_the_decorator_and_leaves_the_original_unchanged(self):
        @function(max_evaluations=7)
        def identity(x):
            return x

        view = identity.using(max_evaluations=11)

        assert view.options["max_evaluations"] == 11
        assert identity.options["max_evaluations"] == 7
        assert view.label == identity.label
        assert view.spec is identity.spec

    def test_the_resolved_max_evaluations_governs_the_lift(self):
        @function(
            max_evaluations=7,
            executor="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.using(max_evaluations=9)(standard_normal())

        assert result.num_atoms == 9

    def test_an_unset_control_reads_the_default_when_it_is_read(self, monkeypatch):
        wrapped = Function(
            _identity,
            label="identity",
        )
        view = wrapped.using(raw=True)
        monkeypatch.setattr(config.sampling, "max_evaluations", 17)

        assert wrapped.options["max_evaluations"] == 17
        assert view.options["max_evaluations"] == 17

    def test_a_set_control_keeps_its_value_when_the_default_changes(self, monkeypatch):
        wrapped = Function(
            _identity,
            max_evaluations=7,
            label="identity",
        )
        monkeypatch.setattr(config.sampling, "max_evaluations", 17)

        assert wrapped.options["max_evaluations"] == 7
        assert wrapped.using(raw=True).options["max_evaluations"] == 7

    def test_the_default_read_at_call_time_governs_the_lift(self, monkeypatch):
        @function(executor="sequential", output_spec=OutputSpec(identity=None))
        def identity(x):
            return x

        monkeypatch.setattr(config.sampling, "max_evaluations", 6)
        with workflow_run(seed=0):
            result = identity(standard_normal())

        assert result.num_atoms == 6

    @pytest.mark.parametrize(
        ("control", "value"),
        [
            ("max_evaluations", 7),
            ("num_samples", 7),
            ("executor", "thread"),
            ("executor_options", {"max_workers": 2}),
            ("include_inputs", True),
            ("method", "sampling_lift"),
            ("exact_only", True),
            ("conversions", {"x": {"exact_only": True}}),
            ("method_options", {"num_results": 5}),
            ("raw", True),
        ],
    )
    def test_none_in_a_view_resets_a_control_to_its_default(self, control, value):
        wrapped = Function(
            _identity,
            **{control: value},
            label="identity",
        )
        reset = wrapped.using(**{control: None})

        assert (
            reset.options[control]
            == Function(
                _identity,
                label="identity",
            ).options[control]
        )
        other = "exact_only" if control == "raw" else "raw"
        assert wrapped.using(**{other: None}).options[control] == wrapped.options[control]

    def test_a_reset_reads_the_default_when_it_is_read(self, monkeypatch):
        reset = Function(
            _identity,
            max_evaluations=7,
            label="identity",
        ).using(max_evaluations=None)
        monkeypatch.setattr(config.sampling, "max_evaluations", 17)

        assert reset.options["max_evaluations"] == 17

    def test_a_declaration_is_kept_by_a_view(self):
        wrapped = Function(
            _identity,
            output_spec=OutputSpec(v=NumericArraySpec(())),
            label="value",
        )

        assert wrapped.using(raw=True).output_spec is wrapped.output_spec


class TestAdmissibility:
    def test_an_unknown_control_raises_at_the_using_call(self):
        with pytest.raises(TypeError, match="unknown Function option"):
            Function(
                _identity,
                label="identity",
            ).using(budget=3)

    @pytest.mark.parametrize(
        ("controls", "error"),
        [
            ({"max_evaluations": 0}, ValueError),
            ({"max_evaluations": -1}, ValueError),
            ({"max_evaluations": 2.5}, TypeError),
            ({"max_evaluations": True}, TypeError),
            ({"num_samples": 0}, ValueError),
            ({"num_samples": -1}, ValueError),
            ({"num_samples": 2.5}, TypeError),
            ({"num_samples": True}, TypeError),
            ({"executor": 3}, TypeError),
            ({"executor": ""}, TypeError),
            ({"executor_options": 3}, TypeError),
            ({"executor_options": {"": 1}}, TypeError),
            ({"include_inputs": "yes"}, TypeError),
            ({"method": 3}, TypeError),
            ({"method": ""}, TypeError),
            ({"exact_only": 1}, TypeError),
            ({"raw": "no"}, TypeError),
            ({"conversions": {"x": "tfp"}}, TypeError),
            ({"conversions": {"y": {}}}, ValueError),
        ],
    )
    def test_an_inadmissible_value_raises_at_the_using_call(self, controls, error):
        wrapped = Function(
            _identity,
            label="identity",
        )

        with pytest.raises(error):
            wrapped.using(**controls)

    @pytest.mark.parametrize(
        ("controls", "error"),
        [({"max_evaluations": 0}, ValueError), ({"raw": "no"}, TypeError)],
    )
    def test_an_inadmissible_value_raises_at_the_decorator(self, controls, error):
        with pytest.raises(error):
            function(**controls)(_identity)

    def test_conversions_settings_are_frozen_by_parameter(self):
        view = Function(
            _identity,
            label="identity",
        ).using(conversions={"x": {"exact_only": True}})

        assert dict(view.options["conversions"]["x"]) == {"exact_only": True}
        with pytest.raises(TypeError):
            view.options["conversions"]["x"]["exact_only"] = False

    def test_a_decorator_keyword_that_is_no_control_raises(self):
        def body(x, **kwargs):
            return x

        assert isinstance(error_of(lambda: function(budget=3)(body)), TypeError)

    def test_an_unknown_control_raises_at_the_decorator_before_wrapping_naming_it(self):
        with pytest.raises(TypeError, match="budget"):
            function(budget=3)

    @pytest.mark.parametrize("body", [lambda x, y=0: x + y, lambda x, **kwargs: x])
    def test_an_unknown_control_raises_at_construction_naming_it(self, body):
        with pytest.raises(TypeError, match=r"unknown Function option 'y'.*bind=\{'y': \.\.\.\}"):
            Function(
                body,
                y=2,
                label="add",
            )

    @pytest.mark.parametrize(
        "make",
        [
            lambda: Function(_identity, label="identity").using(n_broadcast_samples=5),
            lambda: Function(_identity, label="identity", n_broadcast_samples=5),
            lambda: function(n_broadcast_samples=5),
        ],
        ids=["using", "constructor", "decorator"],
    )
    def test_the_removed_sample_count_is_an_unknown_control(self, make):
        """``max_evaluations`` and ``num_samples`` replace it, with no alias."""
        with pytest.raises(
            TypeError, match="unknown Function option 'n_broadcast_samples'"
        ) as raised:
            make()
        assert "'max_evaluations'" in str(raised.value)
        assert "'num_samples'" in str(raised.value)

    def test_an_operation_refuses_the_removed_sample_count_as_an_unknown_control(self):
        with pytest.raises(
            TypeError, match=r"mean\.using\(\): unknown control 'n_broadcast_samples'"
        ):
            mean.using(n_broadcast_samples=5)

    def test_an_unknown_control_lists_the_controls(self):
        with pytest.raises(TypeError, match=r"available Function options: \[.*'executor'"):
            Function(
                _identity,
                label="identity",
            ).using(budget=3)

    @pytest.mark.parametrize("name", ["dispatch", "max_workers"])
    @pytest.mark.parametrize(
        "make",
        [
            lambda name: Function(_identity, label="identity").using(**{name: 2}),
            lambda name: Function(_identity, label="identity", **{name: 2}),
        ],
        ids=["using", "constructor"],
    )
    def test_the_removed_execution_controls_are_unknown_controls(self, make, name):
        """``executor`` and ``executor_options`` replace them, with no alias."""
        with pytest.raises(TypeError, match=f"unknown Function option '{name}'") as raised:
            make(name)
        assert "'executor'" in str(raised.value)
        assert "'executor_options'" in str(raised.value)

    def test_a_differentiability_claim_is_refused_naming_the_keyword(self):
        with pytest.raises(
            NotImplementedError, match=r"Function\(differentiable=\.\.\.\) is not supported yet"
        ):
            Function(
                _identity,
                differentiable=NumericArraySpec(()),
                label="identity",
            )

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"method": 3}, r"conversions\['x'\]\['method'\] must be a string; got 3"),
            ({"exact_only": "yes"}, r"conversions\['x'\]\['exact_only'\] must be a bool"),
        ],
    )
    def test_an_inadmissible_conversions_entry_names_the_setting(self, entry, message):
        def body(x: SupportsMean):
            return 0.0

        wrapped = Function(
            body,
            conversions={"x": entry},
            label="body",
        )

        with pytest.raises(TypeError, match=message):
            wrapped(standard_normal())

    def test_an_argument_binds_at_construction_through_bind(self):
        wrapped = Function(
            lambda x, y: x + y,
            bind={"y": 2.0},
            label="add",
        )

        @function(bind={"y": 2.0})
        def add(x, y):
            return x + y

        assert float(wrapped(1.0).raw) == float(add(1.0).raw) == 3.0

    def test_a_registered_rule_reads_its_budget_from_method_options(self, monkeypatch):
        seen: list[dict[str, Any]] = []

        class _Quadrature(BinaryDispatchMethod):
            """A rule whose numerical budget, the number of nodes, is a method option."""

            @property
            def name(self) -> str:
                return "quadrature"

            @property
            def exact(self) -> bool:
                return False

            @property
            def priority(self) -> int:
                return 0

            def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
                return ((Function,), (Distribution,))

            def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
                return Feasibility(True)

            def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
                seen.append(dict(call["controls"]["method_options"]))
                return jnp.float32(0.0)

        registry = type(_rules.evaluation_rule_registry)()
        registry.register(_Quadrature())
        monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)

        wrapped = Function(
            _identity,
            method_options={"n_nodes": 32},
            label="identity",
        )
        wrapped.using(method="quadrature")(standard_normal())

        assert wrapped.options["method_options"] == {"n_nodes": 32}
        assert seen == [{"n_nodes": 32}]
        with pytest.raises(TypeError, match="n_nodes"):
            Function(
                _identity,
                n_nodes=32,
                label="identity",
            )

    @pytest.mark.parametrize("method_options", [{"": 1}, [("n_nodes", 1)]], ids=["empty", "list"])
    def test_method_options_map_option_names_to_values(self, method_options):
        with pytest.raises(TypeError, match="method_options"):
            Function(
                _identity,
                method_options=method_options,
                label="identity",
            )


class TestControlsThatSelectTheRoute:
    def test_method_names_the_route_of_a_lifted_call(self):
        @function(
            max_evaluations=8,
            executor="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.using(method="sampling_lift")(standard_normal())

        assert result.num_atoms == 8

    def test_exact_only_excludes_the_approximate_sampling_lift(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.using(exact_only=True)(standard_normal())

    def test_conversions_configure_the_conversion_of_their_parameter(self):
        @function(
            max_evaluations=8,
            executor="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.using(conversions={"x": {"exact_only": True}})(standard_normal())

        assert result.num_atoms == 8


class TestTheCountOfALift:
    """``max_evaluations`` bounds the evaluations of a lift, and ``num_samples`` plays no part."""

    @pytest.mark.parametrize("num_samples", [1, 5000])
    def test_a_sampled_lift_follows_max_evaluations_and_ignores_num_samples(self, num_samples):
        identity = Function(_identity, label="identity", executor="sequential")
        with workflow_run(seed=0):
            law = identity.using(max_evaluations=9, num_samples=num_samples)(standard_normal())
        assert law.num_atoms == 9
        assert law.provenance.metadata["max_evaluations"] == 9
        assert "num_samples" not in law.provenance.metadata

    @pytest.mark.parametrize(
        ("max_evaluations", "route", "atoms"),
        [
            (6, "empirical_enumeration", 6),
            (100, "empirical_enumeration", 6),
            (5, "sampling_lift", 5),
        ],
    )
    def test_enumeration_fits_under_max_evaluations(self, max_evaluations, route, atoms):
        law = EmpiricalDistribution(jnp.arange(6.0), output_name="x")
        identity = Function(_identity, label="identity", executor="sequential")
        with workflow_run(seed=0):
            pushed = identity.using(max_evaluations=max_evaluations, num_samples=1)(law)
        assert pushed.provenance.metadata["route"] == route
        assert pushed.num_atoms == atoms

    def test_the_mixed_case_allocates_max_evaluations_over_the_combinations(self):
        """Four enumerated atoms share ten evaluations: two draws of ``b`` for each atom."""
        calls = []

        def add(a, b):
            calls.append(None)
            return a + b

        added = Function(add, label="add", traceable=False, executor="sequential")
        a = EmpiricalDistribution(jnp.arange(4.0), output_name="a")
        with workflow_run(seed=0):
            law = added.using(max_evaluations=10, num_samples=3)(a, standard_normal("b"))
        assert law.num_atoms == len(calls) == 8

    @pytest.mark.parametrize("max_evaluations", [5, 6, 7, 13, 24])
    def test_a_lift_evaluates_its_function_at_most_max_evaluations_times(self, max_evaluations):
        calls = []

        def total(a, b, c):
            calls.append(None)
            return a + b + c

        lifted = Function(total, label="total", traceable=False, executor="sequential")
        a = EmpiricalDistribution(jnp.arange(3.0), output_name="a")
        b = EmpiricalDistribution(jnp.arange(2.0), output_name="b")
        with workflow_run(seed=0):
            law = lifted.using(max_evaluations=max_evaluations)(a, b, standard_normal("c"))
        assert law.num_atoms == len(calls) <= max_evaluations
