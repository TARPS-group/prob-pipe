"""Controls, resolved apart from the arguments of the wrapped function (design V.2).

A Function keeps the wrapped function's arguments and the framework's controls
in two namespaces. Each control resolves from the framework's default, then
the decorator or constructor, then a ``with_options`` view, which leaves the
original unchanged. A control that is unknown or inadmissible raises at the
decorator or at ``with_options``, before any call.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import (
    Distribution,
    Function,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    WorkflowKind,
    function,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, Feasibility
from probpipe.functions import _rules

from ._design_helpers import error_of, standard_normal

#: The controls the framework defines, each with its default.
_FRAMEWORK_CONTROLS = {
    "n_broadcast_samples": Function.DEFAULT_N_BROADCAST_SAMPLES,
    "include_inputs": False,
    "method": None,
    "exact_only": False,
    "conversions": {},
    "raw": False,
    "dispatch": "auto",
    "max_workers": None,
    "workflow_kind": WorkflowKind.DEFAULT,
}


def _identity(x):
    return x


class TestTwoNamespaces:
    @pytest.mark.parametrize("name", ["n_broadcast_samples", "raw", "method", "seed", "key"])
    def test_a_call_keyword_binds_to_the_authored_signature(self, name):
        def body(x, **kwargs):
            return jnp.asarray(float(kwargs[name]))

        result = Function("body", body)(1.0, **{name: 3})

        assert float(result.value) == 3.0

    def test_a_call_keyword_named_like_a_control_does_not_set_it(self):
        def body(x, n_broadcast_samples=None, raw=None):
            return x

        wrapped = Function("body", body, dispatch="sequential")
        with workflow_run(seed=0):
            lifted = wrapped(standard_normal(), n_broadcast_samples=3, raw=True)

        assert lifted.num_atoms == Function.DEFAULT_N_BROADCAST_SAMPLES
        assert lifted.provenance is not None

    @pytest.mark.parametrize("name", ["name", "output_name", "output_spec", "input_spec"])
    def test_construction_metadata_is_not_a_control(self, name):
        wrapped = Function("identity", _identity)

        with pytest.raises(TypeError, match="Unknown Function controls"):
            wrapped.with_options(**{name: "x"})

    @pytest.mark.parametrize("name", ["seed", "key"])
    def test_there_is_no_framework_key_or_seed_control(self, name):
        wrapped = Function("identity", _identity)

        with pytest.raises(TypeError, match="Unknown Function controls"):
            wrapped.with_options(**{name: 0})

    def test_a_wrapped_functions_own_seed_parameter_is_an_ordinary_argument(self):
        @function
        def draw(x, seed):
            return jnp.asarray(x + seed)

        assert float(draw(1.0, seed=2).value) == 3.0


class TestResolution:
    def test_every_control_has_a_default(self):
        options = Function("identity", _identity).options

        for name, default in _FRAMEWORK_CONTROLS.items():
            assert options[name] == default, name

    def test_the_decorator_overrides_the_framework_default(self):
        @function(n_broadcast_samples=7)
        def identity(x):
            return x

        assert identity.options["n_broadcast_samples"] == 7

    def test_a_view_overrides_the_decorator_and_leaves_the_original_unchanged(self):
        @function(n_broadcast_samples=7)
        def identity(x):
            return x

        view = identity.with_options(n_broadcast_samples=11)

        assert view.options["n_broadcast_samples"] == 11
        assert identity.options["n_broadcast_samples"] == 7
        assert view.name == identity.name
        assert view.spec is identity.spec

    def test_the_resolved_sample_count_governs_the_lift(self):
        @function(n_broadcast_samples=7, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.with_options(n_broadcast_samples=9)(standard_normal())

        assert result.num_atoms == 9

    def test_an_unset_control_reads_the_default_when_it_is_read(self, monkeypatch):
        wrapped = Function("identity", _identity)
        view = wrapped.with_options(raw=True)
        monkeypatch.setattr(Function, "DEFAULT_N_BROADCAST_SAMPLES", 17)

        assert wrapped.options["n_broadcast_samples"] == 17
        assert view.options["n_broadcast_samples"] == 17

    def test_a_set_control_keeps_its_value_when_the_default_changes(self, monkeypatch):
        wrapped = Function("identity", _identity, n_broadcast_samples=7)
        monkeypatch.setattr(Function, "DEFAULT_N_BROADCAST_SAMPLES", 17)

        assert wrapped.options["n_broadcast_samples"] == 7
        assert wrapped.with_options(raw=True).options["n_broadcast_samples"] == 7

    def test_the_default_read_at_call_time_governs_the_lift(self, monkeypatch):
        @function(dispatch="sequential")
        def identity(x):
            return x

        monkeypatch.setattr(Function, "DEFAULT_N_BROADCAST_SAMPLES", 6)
        with workflow_run(seed=0):
            result = identity(standard_normal())

        assert result.num_atoms == 6

    def test_a_declaration_is_kept_by_a_view(self):
        wrapped = Function("value", _identity, output_spec=OutputSpec(v=NumericArraySpec(())))

        assert wrapped.with_options(raw=True).output_spec is wrapped.output_spec


class TestAdmissibility:
    def test_an_unknown_control_raises_at_with_options(self):
        with pytest.raises(TypeError, match="Unknown Function controls"):
            Function("identity", _identity).with_options(budget=3)

    @pytest.mark.parametrize(
        ("controls", "error"),
        [
            ({"n_broadcast_samples": 0}, ValueError),
            ({"n_broadcast_samples": -1}, ValueError),
            ({"n_broadcast_samples": 2.5}, TypeError),
            ({"n_broadcast_samples": True}, TypeError),
            ({"dispatch": "gpu"}, ValueError),
            ({"include_inputs": "yes"}, TypeError),
            ({"method": 3}, TypeError),
            ({"method": ""}, TypeError),
            ({"exact_only": 1}, TypeError),
            ({"raw": "no"}, TypeError),
            ({"conversions": {"x": "tfp"}}, TypeError),
            ({"conversions": {"y": {}}}, ValueError),
        ],
    )
    def test_an_inadmissible_value_raises_at_with_options(self, controls, error):
        wrapped = Function("identity", _identity)

        with pytest.raises(error):
            wrapped.with_options(**controls)

    @pytest.mark.parametrize(
        ("controls", "error"),
        [({"n_broadcast_samples": 0}, ValueError), ({"raw": "no"}, TypeError)],
    )
    def test_an_inadmissible_value_raises_at_the_decorator(self, controls, error):
        with pytest.raises(error):
            function(**controls)(_identity)

    def test_conversions_settings_are_frozen_by_parameter(self):
        view = Function("identity", _identity).with_options(conversions={"x": {"exact_only": True}})

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
        with pytest.raises(TypeError, match=r"Unknown Function controls: \['y'\]"):
            Function("add", body, y=2)

    def test_an_argument_binds_at_construction_through_bind(self):
        wrapped = Function("add", lambda x, y: x + y, bind={"y": 2.0})

        @function(bind={"y": 2.0})
        def add(x, y):
            return x + y

        assert float(wrapped(1.0).value) == float(add(1.0).value) == 3.0

    @pytest.mark.pending(
        reason="a control that a registered method declares is admitted", raises=TypeError
    )
    def test_a_control_a_registered_method_defines_is_admitted(self, monkeypatch):
        class _Quadrature(BinaryDispatchMethod):
            """A rule that declares its own numerical budget, the number of nodes."""

            @property
            def name(self) -> str:
                return "quadrature"

            @property
            def exact(self) -> bool:
                return False

            @property
            def priority(self) -> int:
                return 0

            @property
            def controls(self) -> dict[str, Any]:
                return {"n_nodes": 16}

            def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
                return ((Function,), (Distribution,))

            def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
                return Feasibility(True)

            def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
                return None

        registry = type(_rules.evaluation_rule_registry)()
        registry.register(_Quadrature())
        monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)

        wrapped = Function("identity", _identity, n_nodes=32)

        assert wrapped.options["n_nodes"] == 32
        assert wrapped.with_options(n_nodes=8).options["n_nodes"] == 8


class TestControlsThatSelectTheRoute:
    def test_method_names_the_route_of_a_lifted_call(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.with_options(method="sampling_lift")(standard_normal())

        assert result.num_atoms == 8

    def test_exact_only_excludes_the_approximate_sampling_lift(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.with_options(exact_only=True)(standard_normal())

    @pytest.mark.pending(reason="conversion planning under the conversions control")
    def test_conversions_configure_the_conversion_of_their_parameter(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.with_options(conversions={"x": {"exact_only": True}})(
                standard_normal()
            )

        assert result.num_atoms == 8
