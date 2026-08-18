"""Workflow RNG ownership tests for distribution converters."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Binomial,
    ConversionInfo,
    ConversionMethod,
    Converter,
    Gamma,
    MultivariateNormal,
    Normal,
    RecordEmpiricalDistribution,
    TransformedDistribution,
    converter_registry,
    from_distribution,
    workflow_run,
)
from probpipe.converters import ConverterRegistry, _probpipe, _scipy, _tfp
from probpipe.converters._probpipe import ProbPipeConverter
from probpipe.converters._tfp import TFPConverter
from probpipe.core import _workflow_context
from probpipe.core.distribution import Distribution


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _RecordingEmpirical(RecordEmpiricalDistribution):
    def __init__(self, calls, values=None):
        self.calls = calls
        if values is None:
            values = [-1.0, 0.0, 1.0, 2.0]
        super().__init__(jnp.asarray(values), name="x")

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
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                wraps=_workflow_context.derive_event_key_words_from_encoded,
            ) as derive,
            workflow_run(seed=7),
        ):
            converter_registry.convert(
                source,
                MultivariateNormal,
                check_support=False,
                num_samples=16,
            )

        assert len(calls) == (0 if covariance_works else 1)
        assert derive.call_count == (0 if covariance_works else 1)

    def test_from_distribution_uses_the_function_broker_once(self):
        with (
            patch(
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                wraps=_workflow_context.derive_event_key_words_from_encoded,
            ) as derive,
            workflow_run(seed=7),
        ):
            result = from_distribution(
                Normal(loc=0.0, scale=1.0, name="x"),
                RecordEmpiricalDistribution,
                num_samples=8,
            )

        assert result.num_atoms == 8
        assert derive.call_count == 1

    def test_sampled_probpipe_conversion_uses_captured_root_and_forward(self):
        calls = []
        root = _RecordingNormal(calls)
        descendant = TransformedDistribution(root, tfb.Exp())

        with (
            patch.object(
                type(descendant),
                "_sample",
                side_effect=AssertionError("sampled descendant directly"),
            ),
            workflow_run(seed=31),
        ):
            converted = converter_registry.convert(
                descendant,
                RecordEmpiricalDistribution,
                num_samples=12,
            )

        assert [shape for _key, shape in calls] == [(12,)]
        root_key = calls[0][0]
        expected = jnp.exp(Normal._sample(root, root_key, (12,)))
        np.testing.assert_allclose(
            converted.flat_samples[:, 0],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_from_distribution_recipe_keeps_the_captured_descendant_plan(self):
        descendant = TransformedDistribution(
            Normal(loc=0.0, scale=1.0, name="root"),
            tfb.Exp(),
        )

        with workflow_run(seed=31):
            converted = from_distribution(
                descendant,
                RecordEmpiricalDistribution,
                num_samples=12,
            )

        effect = converted.provenance.controls["replay"]["plan"]["expected_effects"][0]
        assert effect["operation_kind"] == "conversion"
        assert effect["descendant_descriptor"][0] == "transformed-descendant"

    def test_unsupported_descendant_conversion_fails_before_entropy(self):
        calls = []
        root = _RecordingNormal(calls)
        descendant = TransformedDistribution(root, tfb.Tanh())

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(),
            pytest.raises(TypeError, match="does not support this bijector type"),
        ):
            converter_registry.convert(
                descendant,
                RecordEmpiricalDistribution,
                num_samples=8,
            )

        urandom.assert_not_called()
        commit.assert_not_called()
        assert calls == []

    def test_explicit_key_conversion_keeps_direct_descendant_sampling(self):
        root = Normal(loc=0.0, scale=1.0, name="base")
        descendant = TransformedDistribution(root, tfb.Tanh())
        explicit = jax.random.key(37)

        converted = converter_registry.convert(
            descendant,
            RecordEmpiricalDistribution,
            key=explicit,
            num_samples=8,
        )

        expected = descendant._sample(explicit, (8,))
        np.testing.assert_allclose(converted.flat_samples[:, 0], expected)

    def test_mc_moment_conversion_plans_and_reuses_one_sample_batch(self):
        calls = []
        root = _RecordingEmpirical(calls)
        descendant = TransformedDistribution(root, tfb.Exp())
        converter = ProbPipeConverter()

        plan = converter._workflow_plan_conversion(
            descendant,
            Normal,
            {"num_samples": 16},
        )

        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=_workflow_context._commit_stochastic_invocation,
            ) as commit,
            patch.object(
                _probpipe,
                "_sampled_moment_plan",
                wraps=_probpipe._sampled_moment_plan,
            ) as planner,
            patch.object(
                _workflow_context._WorkflowInvocation,
                "key_for",
                autospec=True,
                wraps=_workflow_context._WorkflowInvocation.key_for,
            ) as key_for,
            workflow_run(seed=41),
        ):
            converted = converter_registry.convert(
                descendant,
                Normal,
                num_samples=16,
                check_support=False,
            )

        assert plan.execution_mode == "sampled"
        assert plan.sample_shape == (16,)
        planner.assert_called_once()
        commit.assert_called_once_with("operation")
        assert key_for.call_count == 1
        assert [shape for _key, shape in calls] == [(16,)]

        root_key = calls[0][0]
        root_samples = RecordEmpiricalDistribution._sample(root, root_key, (16,))
        expected = descendant.bijector.forward(root_samples)
        np.testing.assert_allclose(converted._mean(), jnp.mean(expected, axis=0))
        np.testing.assert_allclose(converted._variance(), jnp.var(expected, axis=0))

    @pytest.mark.parametrize(
        "key_factory", [lambda: jax.random.key(43), lambda: jax.random.PRNGKey(43)]
    )
    def test_mc_moment_conversion_preserves_explicit_key(self, key_factory):
        calls = []
        root = _RecordingEmpirical(calls)
        descendant = TransformedDistribution(root, tfb.Exp())
        explicit = key_factory()

        with patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit:
            converter_registry.convert(
                descendant,
                Normal,
                key=explicit,
                num_samples=16,
                check_support=False,
            )

        commit.assert_not_called()
        assert [shape for _key, shape in calls] == [(16,)]
        assert calls[0][0] is explicit

    def test_mc_covariance_conversion_reuses_the_moment_batch(self):
        calls = []
        root = _RecordingEmpirical(
            calls,
            values=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        )
        descendant = TransformedDistribution(root, tfb.Exp())

        with workflow_run(seed=47):
            converted = converter_registry.convert(
                descendant,
                MultivariateNormal,
                num_samples=16,
                check_support=False,
            )

        assert [shape for _key, shape in calls] == [(16,)]
        root_key = calls[0][0]
        root_samples = RecordEmpiricalDistribution._sample(root, root_key, (16,))
        expected = descendant.bijector.forward(root_samples)
        expected_mean = jnp.mean(expected, axis=0)
        diff = expected - expected_mean
        expected_cov = jnp.einsum("ni,nj->ij", diff, diff) / expected.shape[0]
        expected_cov = expected_cov + 1e-6 * jnp.eye(expected_cov.shape[0])
        np.testing.assert_allclose(converted._mean(), expected_mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(converted._cov(), expected_cov, rtol=1e-6, atol=1e-6)

    def test_mc_moment_target_preflight_fails_before_randomness(self):
        calls = []
        root = _RecordingEmpirical(calls)
        descendant = TransformedDistribution(root, tfb.Exp())

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            pytest.raises(ValueError, match="total_count is required"),
        ):
            converter_registry.convert(descendant, Binomial, num_samples=16)

        urandom.assert_not_called()
        commit.assert_not_called()
        assert calls == []


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

    @pytest.mark.parametrize("num_samples", [True, 0, -1, 1.5])
    def test_explicit_key_does_not_bypass_declared_sample_count_validation(
        self,
        num_samples,
    ):
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
                seen.append((key, kwargs))
                return Normal(loc=0.0, scale=1.0, name="x")

        registry = ConverterRegistry()
        registry.register(SamplingConverter())

        with pytest.raises((TypeError, ValueError), match="num_samples"):
            registry.convert(
                Source(),
                Normal,
                key=jax.random.key(13),
                num_samples=num_samples,
            )

        assert seen == []


class TestExternalProviderAdapters:
    def test_unknown_tfp_registry_conversion_plans_once(self):
        source = tfd.VonMises(loc=0.0, concentration=1.0)

        with (
            patch.object(
                _tfp,
                "_sampled_conversion_plan",
                wraps=_tfp._sampled_conversion_plan,
            ) as planner,
            workflow_run(seed=7),
        ):
            result = converter_registry.convert(
                source,
                RecordEmpiricalDistribution,
                num_samples=16,
            )

        assert result.num_atoms == 16
        planner.assert_called_once()

    def test_unknown_tfp_direct_conversion_plans_once(self):
        source = tfd.VonMises(loc=0.0, concentration=1.0)

        with (
            patch.object(
                _tfp,
                "_sampled_conversion_plan",
                wraps=_tfp._sampled_conversion_plan,
            ) as planner,
            workflow_run(seed=7),
        ):
            result = TFPConverter().convert(
                source,
                RecordEmpiricalDistribution,
                num_samples=16,
            )

        assert result.num_atoms == 16
        planner.assert_called_once()

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

    def test_unknown_scipy_registry_conversion_plans_once(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        source = scipy_stats.chi2(df=3)

        with (
            patch.object(
                _scipy,
                "_sampled_conversion_plan",
                wraps=_scipy._sampled_conversion_plan,
            ) as planner,
            workflow_run(seed=7),
        ):
            result = converter_registry.convert(
                source,
                RecordEmpiricalDistribution,
                num_samples=16,
            )

        assert result.num_atoms == 16
        planner.assert_called_once()
