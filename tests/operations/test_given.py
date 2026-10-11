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
    OutputSpec,
    RecordSpec,
    condition_on,
    conditional_distribution,
    convert,
    cov,
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
        """The marginal is of the kernel's y component at a positive mean."""
        return path == "y" and given["mu"] > 0


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
        options = {"n_broadcast_samples": 64}
        if method is not None:
            options["method"] = "sampling_lift" if operation is expectation else method
        view = operation.with_options(**options)
        expected_route = "evaluation_rules" if operation is expectation else "monte_carlo"
        assert view.check(kernel, *args, given={"mu": 2.0}).route == expected_route
        assert not kernel.keys
        with workflow_run(seed=42):
            result = view(kernel, *args, given={"mu": 2.0})
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

    def test_an_explicit_method_is_kept_through_delegation(self):
        np.testing.assert_allclose(
            mean.with_options(method="closed_form", exact_only=True)(
                _Fallback(), given={"mu": 3.0}
            ),
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
        ).result.spec.element_spec.shape == (3,)
        assert sample(kernel, sample_shape=(2,), given={"mu": jnp.ones(3)}).shape == (2, 3)
        assert kernel.event_spec.spec.free_dims == frozenset({"n"})

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
            mean.with_options(exact_only=True)(kernel, given={"mu": 2.0})
        assert not kernel.calls

    def test_an_unconditional_operand_rejects_given(self):
        with pytest.raises(ApplicabilityError, match="ConditionalDistribution"):
            sample(Normal("y", 0.0, 1.0), given={"mu": 1.0})


class TestPartialSlots:
    def test_bound_leaves_are_reassembled_with_the_residual_slot(self):
        kernel = conditional_distribution(
            lambda theta: Normal("y", theta["a"] + theta["b"], 1.0),
            given_spec={"theta": RecordSpec(a=REAL, b=REAL)},
        )
        partial = condition_on(kernel, {"theta/a": 2.0})
        assert tuple(partial.given_spec["theta"]) == ("b",)
        law = condition_on(partial, {"theta": {"b": 3.0}})
        np.testing.assert_allclose(mean(law), 5.0)
        assert "theta/a" in law.notation

    def test_binding_every_leaf_dissolves_the_slot(self):
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
        declared = report.result.spec
        partial = condition_on(kernel, given)
        assert declared.given_spec["theta"]["b"].shape == (3,)
        assert declared.event_spec == partial.event_spec
        assert partial.event_spec.spec.shape == (3,)

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
