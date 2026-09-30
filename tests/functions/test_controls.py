"""Controls, resolved apart from the arguments of the wrapped function (design V.2).

A Function keeps the wrapped function's arguments and the framework's controls
in two namespaces. Each control resolves from the framework's default, then
the decorator or constructor, then a ``with_options`` view, which leaves the
original unchanged. A control that is unknown or inadmissible raises at the
decorator or at ``with_options``, before any call.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import (
    Function,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    WorkflowKind,
    function,
    workflow_run,
)

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

    @pytest.mark.pending(
        reason="a keyword at the decorator is a control or a declaration, never an argument",
        raises=AssertionError,
    )
    def test_a_decorator_keyword_that_is_no_control_raises(self):
        def body(x, **kwargs):
            return x

        assert isinstance(error_of(lambda: function(budget=3)(body)), TypeError)


class TestControlsThatSelectTheRoute:
    @pytest.mark.pending(reason="route selection by the method control")
    def test_method_names_the_route_of_a_lifted_call(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity.with_options(method="sampling_lift")(standard_normal())

        assert result.num_atoms == 8

    @pytest.mark.pending(reason="route selection under exact_only")
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
