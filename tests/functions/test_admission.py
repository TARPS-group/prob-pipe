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

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    ApplicabilityError,
    Distribution,
    DistributionSpec,
    Function,
    InputSpec,
    NumericArray,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    function,
    workflow_run,
)
from probpipe.distributions import ConditionalDistribution, SupportsMarginals

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

        assert [parent.name for parent in result.provenance.parents] == ["add", "a"]
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

    @pytest.mark.pending(
        reason="admission reads the kind a raw argument wraps as", raises=AssertionError
    )
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

    @pytest.mark.pending(
        reason="a conversion with no converter raises ResolutionError", raises=AssertionError
    )
    def test_a_parameter_class_with_no_converter_raises_resolution_error(self):
        @function
        def consume(d: _Unconvertible):
            return 0.0

        assert isinstance(error_of(lambda: consume(standard_normal())), ResolutionError)

    @pytest.mark.pending(
        reason="an unsatisfiable capability target raises rather than passing the law through",
        raises=AssertionError,
    )
    def test_a_capability_no_converter_establishes_raises_resolution_error(self):
        @function
        def consume(d: SupportsMarginals):
            return 0.0

        assert isinstance(error_of(lambda: consume(standard_normal())), ResolutionError)

    @pytest.mark.pending(reason="the engine's probe of steps 1 to 6")
    def test_a_probe_plans_the_conversion_without_constructing_it(self, monkeypatch):
        from probpipe.converters import converter_registry

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

    @pytest.mark.pending(
        reason="a declared-kind mismatch raises ApplicabilityError", raises=AssertionError
    )
    def test_a_declared_kind_mismatch_names_the_parameter_what_it_accepts_and_what_arrived(self):
        wrapped = Function("f", lambda x: x, input_spec=InputSpec({"x": NumericArraySpec((2,))}))

        error = error_of(lambda: wrapped("text"))

        assert isinstance(error, ApplicabilityError)
        assert "x" in str(error) and "NumericArraySpec" in str(error) and "str" in str(error)

    @pytest.mark.pending(
        reason="an object with methods passes only a parameter accepting Opaque",
        raises=AssertionError,
    )
    def test_a_bare_object_passes_only_a_parameter_that_accepts_opaque(self):
        class Sampler:
            def _sample(self, key, sample_shape=()):
                return 0.0

        law_spec = DistributionSpec(OutputSpec(v=SCALAR))
        wrapped = Function("f", lambda d: 0.0, input_spec={"d": law_spec})

        assert isinstance(error_of(lambda: wrapped(Sampler())), ApplicabilityError)
