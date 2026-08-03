"""Workflow RNG ownership tests for distribution converters."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    ConversionInfo,
    ConversionMethod,
    Converter,
    Gamma,
    MultivariateNormal,
    Normal,
    RecordEmpiricalDistribution,
    converter_registry,
    from_distribution,
    workflow_run,
)
from probpipe.converters import ConverterRegistry
from probpipe.core import _workflow_context
from probpipe.core.distribution import Distribution


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _VectorSource(Distribution):
    def __init__(self, *, covariance_works: bool, calls):
        super().__init__(name="x")
        self._covariance_works = covariance_works
        self.calls = calls

    def _mean(self):
        return jnp.asarray([0.0, 0.0])

    def _cov(self):
        if not self._covariance_works:
            raise NotImplementedError
        return jnp.eye(2)

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return jax.random.normal(key, (*sample_shape, 2))


def _flat_samples(dist):
    return np.asarray(dist.flat_samples)


class TestBuiltInConversionPlanning:
    def test_exact_and_analytic_paths_claim_no_event(self):
        source = Normal(loc=0.0, scale=1.0, name="x")
        analytic_source = Gamma(concentration=9.0, rate=1.0, name="g")

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
        ):
            assert converter_registry.convert(source, Normal) is source
            converted = converter_registry.convert(analytic_source, Normal)

        assert isinstance(converted, Normal)
        commit.assert_not_called()

    def test_sampled_conversion_is_seeded_and_claims_one_batched_event(self):
        source = Normal(loc=0.0, scale=1.0, name="x")

        def run(num_samples):
            with (
                patch(
                    "probpipe.core._workflow_context._commit_stochastic_invocation",
                    wraps=_workflow_context._commit_stochastic_invocation,
                ) as commit,
                workflow_run(seed=7),
            ):
                result = converter_registry.convert(
                    source,
                    RecordEmpiricalDistribution,
                    num_samples=num_samples,
                )
            return _flat_samples(result), commit.call_args_list

        first, first_commits = run(8)
        second, second_commits = run(8)
        larger, larger_commits = run(32)

        np.testing.assert_array_equal(first, second)
        assert first_commits == second_commits == larger_commits
        assert len(first_commits) == 1
        assert first_commits[0].args == ("operation",)
        assert larger.shape == (32, 1)

    @pytest.mark.parametrize("num_samples", [True, 0, -1, 1.5])
    def test_invalid_sample_count_fails_before_event_commit(self, num_samples):
        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            converter_registry.convert(
                Normal(loc=0.0, scale=1.0, name="x"),
                RecordEmpiricalDistribution,
                num_samples=num_samples,
            )

        commit.assert_not_called()

    @pytest.mark.parametrize(
        "key_factory", [lambda: jax.random.key(11), lambda: jax.random.PRNGKey(11)]
    )
    def test_explicit_key_is_preserved_and_does_not_shift_automatic_conversion(self, key_factory):
        calls = []
        source = _RecordingNormal(calls)
        explicit = key_factory()

        with workflow_run(seed=7):
            expected = converter_registry.convert(
                source,
                RecordEmpiricalDistribution,
                num_samples=8,
            )

        calls.clear()
        with workflow_run(seed=7):
            converter_registry.convert(
                source,
                RecordEmpiricalDistribution,
                key=explicit,
                num_samples=8,
            )
            actual = converter_registry.convert(
                source,
                RecordEmpiricalDistribution,
                num_samples=8,
            )

        assert calls[0][0] is explicit
        np.testing.assert_array_equal(_flat_samples(actual), _flat_samples(expected))

    @pytest.mark.parametrize("covariance_works", [True, False])
    def test_covariance_fallback_claims_only_when_sampling(self, covariance_works):
        calls = []
        source = _VectorSource(covariance_works=covariance_works, calls=calls)

        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=_workflow_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
        ):
            converter_registry.convert(
                source,
                MultivariateNormal,
                check_support=False,
                num_samples=16,
            )

        assert len(calls) == (0 if covariance_works else 1)
        assert commit.call_count == (0 if covariance_works else 1)

    def test_from_distribution_uses_the_function_broker_once(self):
        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=_workflow_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
        ):
            result = from_distribution(
                Normal(loc=0.0, scale=1.0, name="x"),
                RecordEmpiricalDistribution,
                num_samples=8,
            )

        assert result.num_atoms == 8
        commit.assert_called_once_with("invocation")


class TestConverterCertification:
    @pytest.mark.parametrize("method", [ConversionMethod.EXACT, ConversionMethod.MOMENT_MATCH])
    def test_declared_nonrandom_converter_receives_none(self, method):
        seen = []

        class Source:
            pass

        class DeclaredConverter(Converter):
            def source_types(self):
                return (Source,)

            def target_types(self):
                return (Normal,)

            def check(self, source, target_type):
                return ConversionInfo(feasible=True, method=method)

            def convert(self, source, target_type, *, key=None, **kwargs):
                seen.append(key)
                return Normal(loc=0.0, scale=1.0, name="x")

        registry = ConverterRegistry()
        registry.register(DeclaredConverter())

        with patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit:
            registry.convert(Source(), Normal)

        assert seen == [None]
        commit.assert_not_called()

    def test_uncertified_sample_converter_requires_explicit_key(self):
        seen = []

        class Source:
            pass

        class SamplingConverter(Converter):
            def source_types(self):
                return (Source,)

            def target_types(self):
                return (Normal,)

            def check(self, source, target_type):
                return ConversionInfo(feasible=True, method=ConversionMethod.SAMPLE)

            def convert(self, source, target_type, *, key=None, **kwargs):
                seen.append(key)
                return Normal(loc=0.0, scale=1.0, name="x")

        registry = ConverterRegistry()
        registry.register(SamplingConverter())

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            pytest.raises(TypeError, match="explicit key"),
        ):
            registry.convert(Source(), Normal)

        commit.assert_not_called()
        assert seen == []

        explicit = jax.random.key(13)
        registry.convert(Source(), Normal, key=explicit)
        assert seen == [explicit]


class TestExternalProviderAdapters:
    def test_unknown_tfp_sampling_is_seeded(self):
        source = tfd.VonMises(loc=0.0, concentration=1.0)

        def run():
            with workflow_run(seed=7):
                return converter_registry.convert(
                    source,
                    RecordEmpiricalDistribution,
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
                    RecordEmpiricalDistribution,
                    num_samples=16,
                )

        np.testing.assert_array_equal(_flat_samples(run()), _flat_samples(run()))
