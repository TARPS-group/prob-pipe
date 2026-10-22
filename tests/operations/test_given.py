"""Conditional operation arguments, direct twins, and execution-time delegation."""

from __future__ import annotations

import copy
import inspect
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    condition_on,
    conditional_distribution,
    convert,
    cov,
    distribution,
    expectation,
    factor,
    log_prob,
    marginal,
    mean,
    prob,
    quantile,
    random_log_prob,
    random_unnormalized_log_prob,
    sample,
    standard_deviation,
    unnormalized_log_prob,
    unnormalized_prob,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalCovariance,
    SupportsConditionalExpectation,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalQuantile,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
)
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.linalg import DenseLinOp
from probpipe.operations._given import _GivenLaw

REAL = NumericArraySpec(())


class _Direct(ConditionalDistribution, SupportsConditionalLogProb, SupportsConditionalSampling):
    """A normal kernel whose conditional primitives must serve calls directly."""

    def __init__(self) -> None:
        super().__init__({"mu": REAL}, OutputSpec(y=REAL), label="lik")
        self.keys: list[Any] = []

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        raise AssertionError("the direct conditional primitive must be used")

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        self.keys.append(key)
        return given["mu"] + jax.random.normal(key, sample_shape)

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        return jax.scipy.stats.norm.logpdf(value, given["mu"])


class _Fallback(ConditionalDistribution):
    """A kernel exposing its capabilities only through the law it constructs."""

    def __init__(self) -> None:
        super().__init__({"mu": REAL}, OutputSpec(y=REAL), label="lik")
        self.calls: list[Any] = []

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        self.calls.append(given)
        return Normal("y", given["mu"], 1.0)


class _Moments(
    _Direct,
    SupportsConditionalMean,
    SupportsConditionalVariance,
    SupportsConditionalCovariance,
    SupportsConditionalQuantile,
    SupportsConditionalExpectation,
    SupportsConditionalMarginals,
):
    """A kernel declaring direct moment, quantile, and marginal capabilities."""

    def _conditional_mean(self, given):
        return given["mu"]

    def _conditional_variance(self, given):
        return jnp.float32(1.0)

    def _conditional_cov(self, given):
        return DenseLinOp(jnp.ones((1, 1)))

    def _conditional_quantile(self, given, q):
        return given["mu"] + jax.scipy.special.ndtri(q)

    def _conditional_expectation(self, given, f):
        return f(given["mu"])

    def _conditional_marginal(self, given, path):
        return Normal("y", given["mu"], 1.0)

    def _conditional_marginal_guard(self, given, path):
        """The marginal is of the kernel's y output at a positive mean."""
        return path == "y" and given["mu"] > 0


class _GuardedDensity(_Direct):
    def _conditional_log_prob_guard(self, given):
        return given["mu"] > 0


class _OpaqueInputs(ConditionalDistribution):
    def __init__(self, data_spec=None):
        if data_spec is None:
            data_spec = OpaqueSpec()
        super().__init__(
            {"theta": RecordSpec(data=data_spec, offset=REAL)}, OutputSpec(y=REAL), label="opaque"
        )
        self.calls = []

    def _condition_on(self, given, /, **kwargs):
        self.calls.append(given)
        theta = given["theta"]
        return Normal("y", jnp.sum(theta["data"]) + theta["offset"], 1.0)


class _Approximate(_Fallback, SupportsApproximateConditioning):
    """Binding this kernel is an approximate conditioning capability."""


class TestConditionalOperations:
    @pytest.mark.parametrize(
        "operation",
        [
            sample,
            log_prob,
            unnormalized_log_prob,
            prob,
            unnormalized_prob,
            random_log_prob,
            random_unnormalized_log_prob,
            mean,
            variance,
            standard_deviation,
            cov,
            quantile,
            expectation,
            marginal,
            factor,
            convert,
        ],
    )
    def test_given_is_keyword_only(self, operation):
        parameter = inspect.signature(operation).parameters["given"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_the_view_claims_only_the_kernels_twins(self):
        view = _GivenLaw(_Direct(), {"mu": 2.0})
        assert isinstance(view, SupportsSampling)
        assert isinstance(view, SupportsLogProb)
        assert not isinstance(view, SupportsMean)
        assert not isinstance(_GivenLaw(_Fallback(), {"mu": 2.0}), SupportsSampling)

    @pytest.mark.parametrize("kernel_type", [_Direct, _Fallback])
    @pytest.mark.parametrize(
        "round_trip", [copy.copy, lambda value: pickle.loads(pickle.dumps(value))]
    )
    def test_view_round_trips_preserve_declarations_and_membership(self, kernel_type, round_trip):
        original = _GivenLaw(kernel_type(), {"mu": 2.0}).with_label("renamed")
        restored = round_trip(original)
        assert (restored.label, restored.spec) == (original.label, original.spec)
        assert isinstance(restored, SupportsSampling) == isinstance(original, SupportsSampling)
        assert restored.given == original.given

    @pytest.mark.parametrize(
        "operation", [log_prob, unnormalized_log_prob, prob, unnormalized_prob]
    )
    def test_density_twins_and_derived_operations(self, operation):
        result = operation(_Direct(), 1.0, given={"mu": 2.0})
        expected = jax.scipy.stats.norm.logpdf(1.0, 2.0)
        if operation in (prob, unnormalized_prob):
            expected = jnp.exp(expected)
        np.testing.assert_allclose(result, expected)
        assert "mu" in result.label

    @pytest.mark.parametrize("operation", [mean, variance, expectation])
    @pytest.mark.parametrize("method", [None, "monte_carlo"])
    def test_sampling_twins_serve_monte_carlo_without_constructing_a_law(self, operation, method):
        kernel = _Direct()
        args = (lambda y: y,) if operation is expectation else ()
        options = {"num_samples": 64}
        if method is not None:
            options["method"] = "sampling_lift" if operation is expectation else method
        view = operation.using(**options)
        expected_route = "pushforward_mean" if operation is expectation else "monte_carlo"
        assert view.check(kernel, *args, given={"mu": 2.0}).route == expected_route
        assert not kernel.keys
        with workflow_run(seed=42):
            result = view(kernel, *args, given={"mu": 2.0})
        assert jnp.isfinite(result)
        assert kernel.keys

    def test_the_standard_deviation_takes_the_variances_monte_carlo_route(self):
        kernel = _Direct()
        view = standard_deviation.using(num_samples=64)
        report = view.check(kernel, given={"mu": 2.0})
        assert (report.route, report.exact) == ("derived", False)
        with workflow_run(seed=42):
            result = view(kernel, given={"mu": 2.0})
        assert jnp.isfinite(result)
        assert kernel.keys

    def test_check_does_not_apply_a_fallback_kernel(self):
        kernel = _Fallback()
        report = mean.check(kernel, given={"mu": 2.0})
        assert report.feasible is True
        assert report.route == "conditional"
        assert report.exact is False
        assert report.deferred
        assert not kernel.calls
        np.testing.assert_allclose(mean(kernel, given={"mu": 2.0}), 2.0)
        assert kernel.calls

    @pytest.mark.parametrize("exact_only", [False, True])
    def test_fallback_check_never_applies_an_unnormalized_kernel(self, exact_only):
        class Unnormalized(_Fallback, SupportsConditionalUnnormalizedLogProb):
            def _conditional_unnormalized_log_prob(self, given, value):
                raise AssertionError("checking must not evaluate a density")

            def _condition_on(self, given, /, **kwargs):
                self.calls.append(given)
                return distribution(
                    unnormalized_log_prob=lambda value: -(value**2),
                    event_spec=OutputSpec(y=REAL),
                )

        kernel = Unnormalized()
        report = mean.using(exact_only=exact_only).check(kernel, given={"mu": 2.0})
        assert report.feasible is True
        assert report.route == "conditional"
        assert report.deferred
        assert not kernel.calls

    def test_an_explicit_method_is_kept_through_delegation(self):
        np.testing.assert_allclose(
            mean.using(method="closed_form", exact_only=True)(_Fallback(), given={"mu": 3.0}),
            3.0,
        )

    def test_a_functional_kernel_accepts_given_in_its_cached_sampling_guard(self):
        kernel = conditional_distribution(lambda mu: Normal("y", mu, 1.0), given_spec={"mu": REAL})
        assert sample.check(kernel, given={"mu": 2.0}).feasible is True
        with workflow_run(seed=42):
            result = sample(kernel, sample_shape=(4,), given={"mu": 2.0})
        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))

    def test_sampling_uses_a_workflow_owned_key_and_replays(self):
        kernel = _Direct()
        with workflow_run(seed=42):
            first = sample(kernel, sample_shape=(4,), given={"mu": 2.0})
        with workflow_run(seed=42):
            second = sample(kernel, sample_shape=(4,), given={"mu": 2.0})
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, 2.0 + jax.random.normal(kernel.keys[0], (4,)))
        assert first.shape == (4,)
        assert first.level_names == ("sample",)

    def test_record_batch_givens_are_swept_at_their_levels(self):
        given = NumericRecordBatch({"mu": jnp.array([1.0, 2.0])}, ("cases",), label="givens")
        result = log_prob(_Direct(), 0.0, given=given)
        np.testing.assert_allclose(result, jax.scipy.stats.norm.logpdf(0.0, jnp.array([1.0, 2.0])))
        assert result.level_names == ("cases",)

    def test_an_empty_given_sweep_uses_declarations_without_drawing(self):
        kernel = _Direct()
        given = NumericRecordBatch({"mu": jnp.empty((0,))}, ("cases",), label="givens")
        report = log_prob.check(kernel, 0.0, given=given)
        assert report.feasible is True
        result = log_prob(kernel, 0.0, given=given)
        assert result.shape == (0,)
        assert result.level_names == ("cases",)
        assert not kernel.keys

    def test_an_empty_given_sweep_does_not_probe_a_value_dependent_guard(self):
        given = NumericRecordBatch({"mu": jnp.empty((0,))}, ("cases",), label="givens")
        report = marginal.check(_Moments(), "y", given=given)
        assert report.feasible is True
        assert not report.routes

    def test_a_law_over_givens_defers_a_value_dependent_guard_without_sampling(self):
        def forbidden_draw(key):
            raise AssertionError("a spec-only check must not draw")

        given = distribution(sample=forbidden_draw, event_spec=RecordSpec(mu=REAL))
        report = marginal.check(_Moments(), "y", given=given)
        assert report.feasible is None
        assert any(route.pending for route in report.routes)

    def test_a_law_over_givens_defers_the_guard_of_a_delegating_default(self):
        def forbidden_draw(key):
            raise AssertionError("a spec-only check must not draw")

        given = distribution(sample=forbidden_draw, event_spec=RecordSpec(mu=REAL))
        report = unnormalized_log_prob.check(_GuardedDensity(), 0.0, given=given)
        assert report.feasible is None
        assert any(route.pending for route in report.routes)

    def test_a_law_over_givens_defers_fallback_without_applying_or_drawing(self):
        def forbidden_draw(key):
            raise AssertionError("a spec-only check must not draw")

        given = distribution(sample=forbidden_draw, event_spec=RecordSpec(mu=REAL))
        kernel = _Fallback()
        report = mean.check(kernel, given=given)
        assert report.feasible is None
        assert any(route.pending for route in report.routes)
        assert not kernel.calls

    def test_a_law_over_givens_executes_fallback_at_each_draw(self):
        given = distribution(sample=lambda key: {"mu": 2.0}, event_spec=RecordSpec(mu=REAL))
        kernel = _Fallback()
        with workflow_run(seed=42):
            result = mean.using(num_samples=8)(kernel, given=given)
        np.testing.assert_allclose(mean(result), 2.0)
        assert kernel.calls

    @pytest.mark.parametrize("sequence", [list, tuple])
    def test_conditioning_preserves_numeric_sequence_inputs_for_the_kernel(self, sequence):
        class SequenceKernel(ConditionalDistribution):
            def __init__(self):
                super().__init__(
                    {"data": NumericArraySpec(("n",))},
                    OutputSpec(y=NumericArraySpec(("n",))),
                    label="sequence",
                )
                self.calls = []

            def _condition_on(self, given, /, **kwargs):
                self.calls.append(given)
                return Normal("y", np.asarray(given["data"]), 1.0)

        kernel = SequenceKernel()
        data = sequence([1.0, 2.0, 3.0])
        law = condition_on(kernel, {"data": data})
        assert kernel.calls[0]["data"] is data
        assert law.event_spec.term_spec.shape == (3,)
        np.testing.assert_allclose(mean(law), [1.0, 2.0, 3.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            mean(kernel, given={"data": data}), [1.0, 2.0, 3.0], rtol=0.0, atol=0.0
        )
        assert kernel.calls[-1]["data"] is data

    @pytest.mark.parametrize("sequence", [list, tuple])
    @pytest.mark.parametrize("record", [dict, Record])
    @pytest.mark.parametrize("direct", [False, True])
    def test_nested_numeric_sequences_bind_dimensions_without_replacing_values(
        self, sequence, record, direct
    ):
        vector = NumericArraySpec(("n",))

        class Nested(ConditionalDistribution):
            def __init__(self):
                super().__init__(
                    {"theta": RecordSpec(group=RecordSpec(a=vector, b=vector))},
                    OutputSpec(y=vector),
                    label="nested",
                )
                self.calls = []

            def _condition_on(self, given, /, **kwargs):
                self.calls.append(given)
                return Normal("y", self._data(given), 1.0)

            def _data(self, given):
                theta = given["theta"]
                data = theta["group/a"].raw if isinstance(theta, Record) else theta["group"]["a"]
                return np.asarray(data)

        class Direct(Nested, SupportsConditionalMean):
            def _conditional_mean(self, given):
                self.calls.append(given)
                return self._data(given)

        kernel = Direct() if direct else Nested()
        data = sequence([1.0, 2.0, 3.0])
        theta = record({"group": {"a": data, "b": np.ones(3)}})
        given = {"theta": theta}
        law = condition_on(kernel, given)
        assert law.event_spec.term_spec.shape == (3,)
        report = mean.check(kernel, given=given)
        assert report.result.term_spec.shape == (3,)
        np.testing.assert_allclose(mean(kernel, given=given), data, rtol=0.0, atol=0.0)
        assert all(call["theta"] is theta for call in kernel.calls)
        nested_data = theta["group/a"].raw if record is Record else theta["group"]["a"]
        assert nested_data is data
        assert kernel.event_spec.term_spec.free_dims == frozenset({"n"})
        bad = record({"group": {"a": data, "b": np.ones(2)}})
        with pytest.raises((ApplicabilityError, ValueError)):
            mean(kernel, given={"theta": bad})

    @pytest.mark.parametrize("data_spec", [OpaqueSpec(), OpaqueSpec(type=np.ndarray)])
    def test_nested_opaque_inputs_admit_numeric_values_at_binding(self, data_spec):
        kernel = _OpaqueInputs(data_spec)
        given = {"theta": {"data": np.ones(3), "offset": 2.0}}
        law = condition_on(kernel, given)
        np.testing.assert_allclose(mean(law), 5.0)
        np.testing.assert_allclose(mean(kernel, given=given), 5.0)

    @pytest.mark.parametrize("data_spec", [OpaqueSpec(), OpaqueSpec(type=np.ndarray)])
    def test_nested_opaque_inputs_keep_value_admission_open_during_lifted_check(self, data_spec):
        def forbidden_draw(key):
            raise AssertionError("a spec-only check must not draw")

        given = distribution(
            sample=forbidden_draw,
            event_spec=RecordSpec(theta=RecordSpec(data=NumericArraySpec((3,)), offset=REAL)),
        )
        kernel = _OpaqueInputs(data_spec)
        assert mean.check(kernel, given=given).feasible is None
        assert not kernel.calls

    @pytest.mark.parametrize("fused", [False, True])
    def test_typed_opaque_inputs_still_validate_concrete_values(self, fused):
        kernel = _OpaqueInputs(OpaqueSpec(type=str))
        given = {"theta": {"data": np.ones(3), "offset": 2.0}}
        with pytest.raises((ApplicabilityError, ValueError)):
            if fused:
                mean(kernel, given=given)
            else:
                condition_on(kernel, given)
        assert not kernel.calls

    def test_a_given_binds_symbolic_event_dimensions_before_sampling(self):
        class Vector(ConditionalDistribution, SupportsConditionalSampling):
            def __init__(self):
                super().__init__(
                    {"mu": NumericArraySpec(("n",))},
                    OutputSpec(y=NumericArraySpec(("n",))),
                    label="vector",
                )

            def _condition_on(self, given, /, **kwargs):
                raise AssertionError("no conditioning")

            def _conditional_sample(self, given, key, sample_shape=()):
                return jnp.broadcast_to(given["mu"], (*sample_shape, *given["mu"].shape))

        kernel = Vector()
        assert sample.check(
            kernel, sample_shape=(2,), given={"mu": jnp.ones(3)}
        ).result.term_spec.element_spec.term_spec.shape == (3,)
        assert sample(kernel, sample_shape=(2,), given={"mu": jnp.ones(3)}).shape == (2, 3)
        assert kernel.event_spec.term_spec.free_dims == frozenset({"n"})

    @pytest.mark.parametrize("given", [{}, {"wrong": 1.0}, {"mu": jnp.ones(2)}])
    def test_invalid_given_fails_before_a_kernel_runs(self, given):
        kernel = _Fallback()
        with pytest.raises(ApplicabilityError):
            mean(kernel, given=given)
        assert not kernel.calls

    @pytest.mark.parametrize(
        "operation, args, expected",
        [
            (mean, (), 2.0),
            (variance, (), 1.0),
            (standard_deviation, (), 1.0),
            (quantile, (0.5,), 2.0),
            (expectation, (lambda y: 2 * y,), 4.0),
        ],
    )
    def test_moment_twins(self, operation, args, expected):
        kernel = _Moments()
        np.testing.assert_allclose(operation(kernel, *args, given={"mu": 2.0}), expected)
        assert operation.check(kernel, *args, given={"mu": 2.0}).exact is True

    def test_covariance_and_marginal_twins(self):
        kernel = _Moments()
        np.testing.assert_allclose(cov(kernel, given={"mu": 2.0}), [[1.0]])
        np.testing.assert_allclose(mean(marginal(kernel, "y", given={"mu": 2.0})), 2.0)
        report = marginal.check(kernel, "y", given={"mu": -2.0})
        assert report.route == "conditional"
        assert report.exact is False

    def test_apply_supports_the_fused_argument(self):
        np.testing.assert_allclose(
            log_prob.apply(_Direct(), 1.0, given={"mu": 2.0}), jax.scipy.stats.norm.logpdf(1.0, 2.0)
        )
        np.testing.assert_allclose(mean.apply(_Fallback(), given={"mu": 2.0}), 2.0)

    def test_exact_only_rejects_approximate_binding_before_it_runs(self):
        kernel = _Approximate()
        with pytest.raises(ResolutionError):
            mean.using(exact_only=True)(kernel, given={"mu": 2.0})
        assert not kernel.calls

    def test_an_unconditional_operand_rejects_given(self):
        with pytest.raises(ApplicabilityError, match="ConditionalDistribution"):
            sample(Normal("y", 0.0, 1.0), given={"mu": 1.0})


class TestPartialInputs:
    @pytest.mark.parametrize("scale", [None, 3.0])
    def test_fused_partial_binding_keeps_optional_input_defaults(self, scale):
        kernel = conditional_distribution(
            lambda theta, scale=2.0: Normal("y", theta["a"] + theta["b"], scale),
            given_spec={"theta": RecordSpec(a=REAL, b=REAL)},
        )
        partial = condition_on(kernel, {"theta/a": 2.0})
        assert partial.given_spec.optional == {"scale"}
        given = {"theta": {"b": 3.0}}
        if scale is not None:
            given["scale"] = scale
        law = condition_on(partial, given)
        np.testing.assert_allclose(log_prob(partial, 0.0, given=given), log_prob(law, 0.0))
        with workflow_run(seed=42):
            fused = sample(partial, sample_shape=(4,), given=given)
        with workflow_run(seed=42):
            conditioned = sample(law, sample_shape=(4,))
        np.testing.assert_array_equal(fused, conditioned)

    def test_bound_leaves_are_reassembled_with_the_residual_input(self):
        kernel = conditional_distribution(
            lambda theta: Normal("y", theta["a"] + theta["b"], 1.0),
            given_spec={"theta": RecordSpec(a=REAL, b=REAL)},
        )
        partial = condition_on(kernel, {"theta/a": 2.0})
        assert tuple(partial.given_spec["theta"]) == ("b",)
        law = condition_on(partial, {"theta": {"b": 3.0}})
        np.testing.assert_allclose(mean(law), 5.0)
        assert "theta/a" in law.notation

    def test_binding_every_leaf_dissolves_the_input(self):
        kernel = conditional_distribution(
            lambda theta: Normal("y", theta["a"] + theta["b"], 1.0),
            given_spec={"theta": RecordSpec(a=REAL, b=REAL)},
        )
        law = condition_on(kernel, {"theta/a": 2.0, "theta/b": 3.0})
        np.testing.assert_allclose(mean(law), 5.0)
        assert not isinstance(law, ConditionalDistribution)

    def test_overlapping_paths_are_refused(self):
        kernel = conditional_distribution(
            lambda theta: Normal("y", theta["a"] + theta["b"], 1.0),
            given_spec={"theta": RecordSpec(a=REAL, b=REAL)},
        )
        with pytest.raises(
            (ApplicabilityError, ResolutionError, ValueError), match=r"overlap|path prefix"
        ):
            condition_on(kernel, {"theta": {"a": 1.0, "b": 2.0}, "theta/a": 3.0})

    def test_partial_binding_projects_shared_dimensions_in_check_and_result(self):
        vector = NumericArraySpec(("n",))

        class Shared(ConditionalDistribution):
            def __init__(self):
                super().__init__(
                    {"theta": RecordSpec(a=vector, b=vector)}, OutputSpec(y=vector), label="shared"
                )

            def _condition_on(self, given, /, **kwargs):
                theta = given["theta"]
                return Normal("y", theta["a"] + theta["b"], 1.0)

        kernel = Shared()
        given = {"theta/a": jnp.ones(3)}
        report = condition_on.check(kernel, given)
        declared = report.result.term_spec
        partial = condition_on(kernel, given)
        assert declared.given_spec["theta"]["b"].shape == (3,)
        assert declared.event_spec == partial.event_spec
        assert partial.event_spec.term_spec.shape == (3,)

    def test_shared_dimensions_are_checked_across_bound_leaves(self):
        class Shared(_Direct):
            def __init__(self):
                ConditionalDistribution.__init__(
                    self,
                    {"theta": RecordSpec(a=NumericArraySpec(("n",)), b=NumericArraySpec(("n",)))},
                    OutputSpec(y=REAL),
                    label="shared",
                )

        kernel = Shared()
        with pytest.raises((ApplicabilityError, ResolutionError, ValueError)):
            condition_on(kernel, {"theta/a": jnp.ones(2), "theta/b": jnp.ones(3)})
