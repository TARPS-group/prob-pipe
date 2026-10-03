"""Binding and normalization, steps 2 and 3 of the stack (design V.3 and V.4).

The arguments bind to the signature by Python's rules; a tracked argument
becomes a dependency and any other a plain input. Normalization then runs three
sub-steps on each argument:

1. wrap: the argument is wrapped into its kind;
2. plan a conversion: where its parameter names another distribution class;
3. admit: the result is checked against what the parameter accepts, and a
   violation of the call contract raises ApplicabilityError.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    ApplicabilityError,
    Beta,
    Distribution,
    DistributionSpec,
    Function,
    Gamma,
    InputSpec,
    Laplace,
    Normal,
    NumericArray,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    function,
    workflow_run,
)
from probpipe.distributions import ConditionalDistribution
from probpipe.distributions._capabilities import SupportsExactConditioning

from ._design_helpers import error_of, standard_normal

SCALAR = NumericArraySpec(())


class _Law(Distribution):
    """A law that declares its event and implements no capability."""


class _Unconvertible(Distribution):
    """A distribution class no converter produces."""


class _Kernel(ConditionalDistribution):
    """A kernel from a scalar ``mu`` to a law over a scalar ``y``."""

    def _condition_on(self, given, /, **kwargs):
        return _Law(self.name, self.event_spec)


def _kernel() -> _Kernel:
    return _Kernel("lik", {"mu": SCALAR}, OutputSpec(y=SCALAR))


class TestBinding:
    def test_arguments_bind_by_pythons_rules(self):
        @function
        def affine(x, scale=2.0, *, shift=0.0):
            return scale * x + shift

        assert float(affine(1.0).value) == 2.0
        assert float(affine(1.0, 3.0, shift=1.0).value) == 4.0

    def test_an_absent_input_spec_adds_no_schema_constraints(self):
        @function
        def kind(x):
            return type(x).__name__

        assert kind(jnp.ones(3)).value == "ArrayImpl"
        assert kind("text").value == "str"

    def test_tracked_arguments_are_dependencies_and_others_inputs_by_parameter(self):
        @function
        def add(x, y):
            return x + y

        tracked = NumericArray("a", jnp.ones(2))
        result = add(tracked, 3.0)

        assert [parent.label for parent in result.provenance.parents] == ["add", "a"]
        assert set(result.provenance.inputs) == {"y"}

    def test_an_argument_that_binds_to_no_parameter_raises_pythons_error(self):
        @function
        def identity(x):
            return x

        with pytest.raises(TypeError, match="y"):
            identity(1.0, y=2.0)
        with pytest.raises(TypeError):
            identity(1.0, 2.0)


class TestWrap:
    def test_a_raw_argument_reaches_the_body_as_it_was_passed(self):
        seen = []

        @function
        def record(x):
            seen.append(x)
            return 0.0

        array = jnp.ones(2)
        record(array)

        assert seen[0] is array

    def test_a_raw_collection_wraps_as_opaque_and_fails_a_numeric_slot(self):
        wrapped = Function("f", lambda x: x, input_spec={"x": NumericArraySpec((2,))})

        error = error_of(lambda: wrapped([1.0, 2.0, 3.0]))

        assert isinstance(error, ApplicabilityError)
        assert "x" in str(error)


class TestConversionPlanning:
    def test_a_backend_distribution_at_a_value_parameter_is_converted_and_lifted(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(tfd.Normal(0.0, 1.0))

        assert isinstance(result, Distribution)
        assert result.num_atoms == 6

    @pytest.mark.parametrize("annotation", [Normal, Normal | None], ids=["class", "optional"])
    def test_a_law_of_another_class_is_converted_to_the_class_the_parameter_names(self, annotation):
        seen = []

        def consume(d):
            seen.append(d)
            return 0.0

        consume.__annotations__ = {"d": annotation}
        Function("consume", consume)(Laplace("g", 2.0, 1.0))

        assert isinstance(seen[0], Normal)

    @pytest.mark.parametrize(
        "annotation", [Distribution, Distribution | None], ids=["class", "optional"]
    )
    def test_a_backend_law_at_a_distribution_parameter_enters_probpipe(self, annotation):
        seen = []

        def consume(d):
            seen.append(d)
            return 0.0

        consume.__annotations__ = {"d": annotation}
        Function("consume", consume)(tfd.Normal(0.0, 1.0))

        assert isinstance(seen[0], Distribution)

    def test_a_union_of_several_distribution_classes_refuses_to_choose_a_conversion(self):
        def consume(d: Normal | Gamma):
            return 0.0

        error = error_of(lambda: Function("consume", consume)(Beta("b", 2.0, 2.0)))

        assert isinstance(error, ApplicabilityError)
        assert "'d'" in str(error) and "Normal" in str(error) and "Beta" in str(error)

    def test_a_union_of_several_distribution_classes_admits_a_law_of_one_of_them(self):
        seen = []

        def consume(d: Normal | Gamma):
            seen.append(d)
            return 0.0

        law = Gamma("g", 2.0, 1.0)
        Function("consume", consume)(law)

        assert seen == [law]

    def test_a_parameter_class_with_no_converter_raises_resolution_error(self):
        @function
        def consume(d: _Unconvertible):
            return 0.0

        assert isinstance(error_of(lambda: consume(standard_normal())), ResolutionError)

    def test_a_capability_no_converter_establishes_raises_resolution_error(self):
        @function
        def consume(d: SupportsExactConditioning):
            return 0.0

        assert isinstance(error_of(lambda: consume(standard_normal())), ResolutionError)

    def test_a_probe_plans_the_conversion_without_constructing_it(self, monkeypatch):
        from probpipe.distributions._conversion import converter_registry

        def refuse(*args, **kwargs):
            raise AssertionError("a probe constructed a conversion")

        monkeypatch.setattr(converter_registry, "convert", refuse)

        @function
        def identity(x):
            return x

        identity.check(tfd.Normal(0.0, 1.0))


class TestAdmission:
    def test_applicability_error_is_a_type_error(self):
        assert issubclass(ApplicabilityError, TypeError)

    def test_a_kernel_at_a_value_parameter_is_refused(self):
        calls = []

        @function
        def body(x):
            calls.append(x)
            return 0.0

        with pytest.raises(ApplicabilityError, match=r"'x'.*ConditionalDistribution"):
            body(_kernel())
        assert calls == []

    @pytest.mark.parametrize("annotation", [jnp.ndarray, float])
    def test_a_kernel_at_a_value_annotated_parameter_is_refused(self, annotation):
        def body(x):
            return 0.0

        body.__annotations__ = {"x": annotation}

        with pytest.raises(ApplicabilityError):
            Function("body", body)(_kernel())

    def test_a_kernel_at_a_parameter_that_consumes_kernels_passes(self):
        seen = []

        @function
        def consume(kernel: ConditionalDistribution):
            seen.append(kernel)
            return 0.0

        kernel = _kernel()
        consume(kernel)

        assert seen == [kernel]

    @pytest.mark.parametrize("slot", ["parameter", "variadic positional", "variadic keyword"])
    def test_a_kernel_at_a_slot_annotated_any_passes(self, slot):
        seen = []

        def at_parameter(proposal: Any):
            seen.append(proposal)
            return 0.0

        def at_args(*proposals: Any):
            seen.extend(proposals)
            return 0.0

        def at_options(**options: Any):
            seen.extend(options.values())
            return 0.0

        kernel = _kernel()
        if slot == "variadic keyword":
            Function("body", at_options)(proposal=kernel)
        else:
            Function("body", at_parameter if slot == "parameter" else at_args)(kernel)

        assert len(seen) == 1 and seen[0] is kernel

    def test_a_kernel_at_an_unannotated_variadic_slot_is_refused(self):
        def at_options(**options):
            return 0.0

        with pytest.raises(ApplicabilityError, match=r"proposal"):
            Function("body", at_options)(proposal=_kernel())

    def test_a_distribution_over_the_accepted_kind_is_admitted_for_lifting(self):
        wrapped = Function(
            "double",
            lambda x: 2.0 * x,
            input_spec={"x": SCALAR},
            n_broadcast_samples=6,
            dispatch="sequential",
        )

        with workflow_run(seed=0):
            result = wrapped(standard_normal())

        assert isinstance(result, Distribution)

    def test_a_declared_kind_mismatch_names_the_parameter_what_it_accepts_and_what_arrived(self):
        wrapped = Function("f", lambda x: x, input_spec=InputSpec({"x": NumericArraySpec((2,))}))

        error = error_of(lambda: wrapped("text"))

        assert isinstance(error, ApplicabilityError)
        assert "x" in str(error) and "NumericArraySpec" in str(error) and "str" in str(error)

    def test_a_bare_object_passes_only_a_parameter_that_accepts_opaque(self):
        class Sampler:
            def _sample(self, key, sample_shape=()):
                return 0.0

        law_spec = DistributionSpec(OutputSpec(v=SCALAR))
        wrapped = Function("f", lambda d: 0.0, input_spec={"d": law_spec})

        assert isinstance(error_of(lambda: wrapped(Sampler())), ApplicabilityError)
