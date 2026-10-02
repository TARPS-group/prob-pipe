"""Workflow RNG ownership tests for the shipped converters.

A conversion takes no key: each draw it takes is a workflow-owned random
event, whose key the workflow scope derives, so a seeded scope reproduces the
draws. A conversion that takes no draws claims no event, and one that does
claims one batched event, after validating its sample count.
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    BijectorTransformedDistribution,
    Binomial,
    ConversionInfo,
    Converter,
    Distribution,
    EmpiricalDistribution,
    Laplace,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    convert,
    converter_registry,
    workflow_run,
)
from probpipe.distributions import ConverterRegistry
from probpipe.distributions._empirical import _coordinates
from probpipe.functions import _context
from probpipe.linalg.linear_operator import DenseLinOp


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, label="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _RecordingMultivariateNormal(MultivariateNormal):
    def __init__(self, calls, loc):
        self.calls = calls
        super().__init__(loc=jnp.asarray(loc), cov=jnp.eye(len(loc)), label="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _VectorSource(Distribution):
    """A law over a vector with a closed-form mean and no closed-form covariance."""

    def __init__(self, calls):
        super().__init__("x", NumericArraySpec((2,)))
        self.calls = calls

    def _mean(self):
        return jnp.asarray([0.0, 0.0])

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return jax.random.normal(key, (*sample_shape, 2))


class _CovarianceSource(_VectorSource):
    """The vector law with a closed-form covariance too."""

    def _cov(self):
        return DenseLinOp(jnp.eye(2))


def _flat_samples(dist):
    return np.asarray(_coordinates(dist))


class TestBuiltInConversionRandomness:
    def test_exact_and_analytic_paths_claim_no_event(self):
        source = Normal(loc=0.0, scale=1.0, label="x")
        analytic_source = Laplace(loc=9.0, scale=1.0, label="g")

        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
        ):
            assert converter_registry.convert(source, Normal) is source
            converted = converter_registry.convert(analytic_source, Normal)

        assert isinstance(converted, Normal)
        commit.assert_not_called()

    def test_sampled_conversion_is_seeded_and_claims_one_batched_event(self):
        source = Normal(loc=0.0, scale=1.0, label="x")

        def run(num_samples):
            with (
                patch(
                    "probpipe.functions._context._commit_stochastic_invocation",
                    wraps=_context._commit_stochastic_invocation,
                ) as commit,
                workflow_run(seed=7),
            ):
                result = converter_registry.convert(
                    source,
                    EmpiricalDistribution,
                    num_samples=num_samples,
                )
            return _flat_samples(result), commit.call_args_list

        first, first_commits = run(8)
        second, second_commits = run(8)
        numpy_count, numpy_commits = run(np.int64(8))
        larger, larger_commits = run(32)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(numpy_count, first)
        assert first_commits == second_commits == numpy_commits == larger_commits
        assert len(first_commits) == 1
        assert first_commits[0].args == ("operation",)
        assert larger.shape == (32, 1)

    @pytest.mark.parametrize("num_samples", [True, 0, -1, 1.5])
    def test_invalid_sample_count_fails_before_event_commit(self, num_samples):
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            converter_registry.convert(
                Normal(loc=0.0, scale=1.0, label="x"),
                EmpiricalDistribution,
                num_samples=num_samples,
            )

        commit.assert_not_called()

    def test_a_conversion_takes_no_key(self):
        """A key is no converter option; a seeded scope reproduces the draws instead."""
        calls = []
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            pytest.raises(TypeError, match="reads the options"),
        ):
            converter_registry.convert(
                _RecordingNormal(calls),
                EmpiricalDistribution,
                key=jax.random.key(11),
                num_samples=8,
            )
        commit.assert_not_called()
        assert calls == []

    @pytest.mark.parametrize("closed_form", [True, False], ids=["closed-form", "sampled"])
    def test_the_covariance_is_sampled_only_without_a_closed_form(self, closed_form):
        calls = []
        source = _CovarianceSource(calls) if closed_form else _VectorSource(calls)

        info = converter_registry.check(source, MultivariateNormal, num_samples=16)
        assert info.samples is not closed_form
        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                wraps=_context.derive_event_key_words_from_encoded,
            ) as derive,
            workflow_run(seed=7),
        ):
            converter_registry.convert(
                source,
                MultivariateNormal,
                check_support=False,
                num_samples=16,
            )

        assert len(calls) == (0 if closed_form else 1)
        assert derive.call_count == (0 if closed_form else 1)

    def test_convert_uses_the_function_broker_once(self):
        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                wraps=_context.derive_event_key_words_from_encoded,
            ) as derive,
            workflow_run(seed=7),
        ):
            result = convert.with_options(method_options={"num_samples": 8})(
                Normal(loc=0.0, scale=1.0, label="x"), EmpiricalDistribution
            )

        assert result.num_atoms == 8
        assert derive.call_count == 1

    def test_mc_moment_conversion_reuses_one_sample_batch(self):
        calls = []
        root = _RecordingNormal(calls)
        descendant = BijectorTransformedDistribution("descendant", root, tfb.Exp())

        info = converter_registry.check(descendant, Normal, num_samples=16, check_support=False)
        assert (info.method_name, info.samples) == ("moment_match", True)
        with (
            patch(
                "probpipe.functions._context._commit_stochastic_invocation",
                wraps=_context._commit_stochastic_invocation,
            ) as commit,
            patch.object(
                _context._WorkflowInvocation,
                "key_for",
                autospec=True,
                wraps=_context._WorkflowInvocation.key_for,
            ) as key_for,
            workflow_run(seed=41),
        ):
            converted = converter_registry.convert(
                descendant,
                Normal,
                num_samples=16,
                check_support=False,
            )

        commit.assert_called_once_with("operation")
        assert key_for.call_count == 1
        assert [shape for _key, shape in calls] == [(16,)]

        root_key = calls[0][0]
        root_samples = Normal._sample(root, root_key, (16,))
        expected = jnp.exp(root_samples)
        np.testing.assert_allclose(converted._mean(), jnp.mean(expected, axis=0), rtol=1e-6)
        np.testing.assert_allclose(converted._variance(), jnp.var(expected, axis=0), rtol=1e-5)

    def test_mc_covariance_conversion_reuses_the_moment_batch(self):
        calls = []
        root = _RecordingMultivariateNormal(calls, [0.0, 1.0])
        descendant = BijectorTransformedDistribution("descendant", root, tfb.Exp())

        with workflow_run(seed=47):
            converted = converter_registry.convert(
                descendant,
                MultivariateNormal,
                num_samples=16,
                check_support=False,
            )

        assert [shape for _key, shape in calls] == [(16,)]
        root_key = calls[0][0]
        root_samples = MultivariateNormal._sample(root, root_key, (16,))
        expected = jnp.exp(root_samples)
        expected_mean = jnp.mean(expected, axis=0)
        diff = expected - expected_mean
        expected_cov = jnp.einsum("ni,nj->ij", diff, diff) / expected.shape[0]
        expected_cov = expected_cov + 1e-6 * jnp.eye(expected_cov.shape[0])
        np.testing.assert_allclose(converted._mean(), expected_mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(converted._cov().to_dense(), expected_cov, rtol=1e-6, atol=1e-6)

    def test_mc_moment_target_preflight_fails_before_randomness(self):
        calls = []
        root = _RecordingNormal(calls)
        descendant = BijectorTransformedDistribution("descendant", root, tfb.Exp())

        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            pytest.raises(ValueError, match="total_count is required"),
        ):
            converter_registry.convert(descendant, Binomial, num_samples=16)

        urandom.assert_not_called()
        commit.assert_not_called()
        assert calls == []


class TestCustomConverterRandomness:
    def test_a_converter_receives_no_key_and_claims_no_event(self):
        seen = []

        class Source:
            pass

        class DeclaredConverter(Converter):
            @property
            def name(self):
                return "declared"

            @property
            def exact(self):
                return False

            @property
            def priority(self):
                return 1

            def supported_types(self):
                return ((Source,), (Normal,))

            def check(self, source, target_type, **options):
                return ConversionInfo(
                    feasible=True,
                    method_name=self.name,
                    exact=False,
                    target_spec=Normal(loc=0.0, scale=1.0, label="x").spec,
                )

            def execute(self, source, target_type, **options):
                seen.append(options)
                return Normal(loc=0.0, scale=1.0, label="x")

        registry = ConverterRegistry()
        registry.register(DeclaredConverter())

        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            registry.convert(Source(), Normal)

        assert seen == [{}]
        commit.assert_not_called()


class TestExternalProviderAdapters:
    def test_unknown_tfp_conversion_claims_one_event(self):
        source = tfd.VonMises(loc=0.0, concentration=1.0)

        with (
            patch(
                "probpipe.functions._context._commit_stochastic_invocation",
                wraps=_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
        ):
            result = converter_registry.convert(
                source,
                EmpiricalDistribution,
                num_samples=16,
            )

        assert result.num_atoms == 16
        commit.assert_called_once_with("operation")

    def test_unknown_tfp_sampling_is_seeded(self):
        source = tfd.VonMises(loc=0.0, concentration=1.0)

        def run():
            with workflow_run(seed=7):
                return converter_registry.convert(
                    source,
                    EmpiricalDistribution,
                    num_samples=16,
                )

        np.testing.assert_array_equal(_flat_samples(run()), _flat_samples(run()))

    def test_unknown_scipy_sampling_uses_seeded_adapter(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        source = scipy_stats.chi2(df=3)

        def run():
            with workflow_run(seed=7):
                return converter_registry.convert(
                    source,
                    EmpiricalDistribution,
                    num_samples=16,
                )

        np.testing.assert_array_equal(_flat_samples(run()), _flat_samples(run()))

    def test_unknown_scipy_conversion_claims_one_event(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        source = scipy_stats.chi2(df=3)

        with (
            patch(
                "probpipe.functions._context._commit_stochastic_invocation",
                wraps=_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
        ):
            result = converter_registry.convert(
                source,
                EmpiricalDistribution,
                num_samples=16,
            )

        assert result.num_atoms == 16
        commit.assert_called_once_with("operation")
