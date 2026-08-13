"""Workflow RNG ownership tests for validation operations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    GLMLikelihood,
    MultivariateNormal,
    Normal,
    RecordEmpiricalDistribution,
    predictive_check,
    workflow_run,
)
from probpipe.core import _workflow_context
from probpipe.validation import (
    Reference,
    score_posterior,
    simulation_based_calibration,
)


class _OpaqueLikelihood:
    def generate_data(self, params, num_observations, *, key=None):
        noise = jax.random.normal(key, (*jnp.shape(params), num_observations))
        return jnp.asarray(params)[..., None] + noise


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


def _glm_validation_setup():
    x = jnp.linspace(-1.0, 1.0, 6)[:, None]
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="beta")
    likelihood = GLMLikelihood(tfp_glm.Normal(), x=x)
    return prior, likelihood


class TestPredictiveCheckBroker:
    def test_certified_provider_is_seeded_and_claims_one_event(self):
        prior, likelihood = _glm_validation_setup()

        def run(num_replications):
            with (
                patch(
                    "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                    wraps=_workflow_context.derive_event_key_words_from_encoded,
                ) as derive,
                workflow_run(seed=7),
            ):
                result = predictive_check(
                    prior,
                    likelihood,
                    test_fn=jnp.mean,
                    num_observations=6,
                    num_replications=num_replications,
                )
            return np.asarray(result["replicated_statistics"].flat_samples), derive

        first, first_derive = run(8)
        second, second_derive = run(8)
        larger, larger_derive = run(16)

        np.testing.assert_array_equal(first, second)
        assert first_derive.call_count == 1
        assert second_derive.call_count == 1
        assert larger_derive.call_count == 1
        assert larger.shape == (16, 1)

    def test_opaque_provider_requires_explicit_key_before_sampling(self):
        calls = []
        prior = _RecordingNormal(calls)

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(TypeError, match="explicit key"),
        ):
            predictive_check(
                prior,
                _OpaqueLikelihood(),
                test_fn=jnp.mean,
                num_observations=4,
                num_replications=3,
            )

        assert calls == []
        commit.assert_not_called()

        explicit = jax.random.key(11)
        with patch(
            "probpipe.core._workflow_context._commit_stochastic_invocation"
        ) as explicit_commit:
            predictive_check(
                prior,
                _OpaqueLikelihood(),
                test_fn=jnp.mean,
                num_observations=4,
                num_replications=3,
                key=explicit,
            )
        assert len(calls) == 1
        explicit_commit.assert_not_called()

    def test_numpy_integer_counts_are_normalized_before_event_commit(self):
        prior, likelihood = _glm_validation_setup()

        with workflow_run(seed=7):
            result = predictive_check(
                prior,
                likelihood,
                test_fn=jnp.mean,
                num_observations=np.int64(6),
                num_replications=np.int64(3),
            )

        assert result["replicated_statistics"].num_atoms == 3

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("num_replications", True),
            ("num_replications", 0),
            ("num_replications", 1.5),
            ("num_observations", True),
            ("num_observations", 0),
            ("num_observations", 1.5),
        ],
    )
    def test_invalid_counts_fail_before_event_commit(self, argument, value):
        prior, likelihood = _glm_validation_setup()
        kwargs = {"num_observations": 6, "num_replications": 3, argument: value}

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            predictive_check(prior, likelihood, test_fn=jnp.mean, **kwargs)

        commit.assert_not_called()

    def test_instance_method_override_is_not_certified(self):
        prior, likelihood = _glm_validation_setup()
        likelihood.generate_data = _OpaqueLikelihood().generate_data

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(TypeError, match="explicit key"),
        ):
            predictive_check(
                prior,
                likelihood,
                test_fn=jnp.mean,
                num_observations=6,
                num_replications=3,
            )

        commit.assert_not_called()

    def test_glm_without_design_matrix_fails_before_event_commit(self):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="beta")
        likelihood = GLMLikelihood(tfp_glm.Normal())

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="design matrix"),
        ):
            predictive_check(
                prior,
                likelihood,
                test_fn=jnp.mean,
                num_observations=3,
                num_replications=2,
            )

        commit.assert_not_called()


class TestSimulationBasedCalibrationBroker:
    @staticmethod
    def _model():
        x = jnp.ones((3, 1))
        return SimpleNamespace(
            prior=MultivariateNormal(loc=jnp.zeros(1), cov=jnp.eye(1), name="beta"),
            likelihood=GLMLikelihood(tfp_glm.Normal(), x=x, fit_intercept=False),
        )

    def test_seeded_sbc_claims_one_event_and_derives_inference_seeds(self, monkeypatch):
        inference_seeds = []

        def fake_condition_on(
            model,
            data,
            *,
            method,
            num_results,
            random_seed,
            **kwargs,
        ):
            del model, data, method, kwargs
            inference_seeds.append(random_seed)
            return RecordEmpiricalDistribution(
                jnp.zeros((num_results, 1)),
                name="beta",
            )

        monkeypatch.setattr(
            "probpipe.validation._calibration.condition_on",
            fake_condition_on,
        )

        def run(num_simulations):
            inference_seeds.clear()
            with (
                patch(
                    "probpipe.core._workflow_context._commit_stochastic_invocation",
                    wraps=_workflow_context._commit_stochastic_invocation,
                ) as commit,
                workflow_run(seed=7),
            ):
                result = simulation_based_calibration(
                    self._model(),
                    num_simulations=num_simulations,
                    num_posterior_draws=4,
                    num_observations=3,
                )
            return result.ranks.copy(), tuple(inference_seeds), commit

        first_ranks, first_seeds, first_commit = run(2)
        second_ranks, second_seeds, second_commit = run(2)
        _, larger_seeds, larger_commit = run(5)

        np.testing.assert_array_equal(first_ranks, second_ranks)
        assert first_seeds == second_seeds
        assert len(larger_seeds) == 5
        first_commit.assert_called_once_with("operation")
        second_commit.assert_called_once_with("operation")
        larger_commit.assert_called_once_with("operation")

        with patch(
            "probpipe.core._workflow_context._commit_stochastic_invocation"
        ) as explicit_commit:
            simulation_based_calibration(
                self._model(),
                num_simulations=2,
                num_posterior_draws=4,
                num_observations=3,
                key=jax.random.key(11),
            )
        explicit_commit.assert_not_called()

    def test_numpy_integer_counts_are_normalized_before_event_commit(self, monkeypatch):
        def fake_condition_on(
            model,
            data,
            *,
            num_results,
            **kwargs,
        ):
            del model, data, kwargs
            return RecordEmpiricalDistribution(
                jnp.zeros((num_results, 1)),
                name="beta",
            )

        monkeypatch.setattr(
            "probpipe.validation._calibration.condition_on",
            fake_condition_on,
        )

        result = simulation_based_calibration(
            self._model(),
            num_simulations=np.int64(2),
            num_posterior_draws=np.int64(4),
            num_observations=np.int64(3),
            key=jax.random.key(11),
        )

        assert result.ranks.shape == (2, 1)

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("num_simulations", True),
            ("num_simulations", 0),
            ("num_posterior_draws", True),
            ("num_posterior_draws", 0),
            ("num_observations", True),
            ("num_observations", 0),
        ],
    )
    def test_invalid_counts_fail_before_event_commit(self, argument, value):
        kwargs = {
            "num_simulations": 2,
            "num_posterior_draws": 4,
            "num_observations": 3,
            argument: value,
        }
        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            simulation_based_calibration(self._model(), **kwargs)

        commit.assert_not_called()

    def test_opaque_provider_requires_explicit_key(self):
        model = SimpleNamespace(
            prior=Normal(loc=0.0, scale=1.0, name="x"),
            likelihood=_OpaqueLikelihood(),
        )
        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(TypeError, match="explicit key"),
        ):
            simulation_based_calibration(
                model,
                num_simulations=2,
                num_posterior_draws=4,
                num_observations=3,
            )

        commit.assert_not_called()


class TestPosteriorScoreBroker:
    @staticmethod
    def _inputs():
        approx = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
        reference_draws = jax.random.normal(jax.random.PRNGKey(1), (32, 2))
        return approx, Reference.from_draws(reference_draws)

    def test_sliced_wasserstein_claims_only_one_seeded_event(self):
        approx, reference = self._inputs()

        def run():
            with (
                patch(
                    "probpipe.core._workflow_context._commit_stochastic_invocation",
                    wraps=_workflow_context._commit_stochastic_invocation,
                ) as commit,
                workflow_run(seed=7),
            ):
                result = score_posterior(
                    approx,
                    reference,
                    metrics=("sliced_wasserstein",),
                )
            return np.asarray(result["sliced_wasserstein"]), commit

        first, first_commit = run()
        second, second_commit = run()

        np.testing.assert_array_equal(first, second)
        first_commit.assert_called_once_with("operation")
        second_commit.assert_called_once_with("operation")

    def test_nonrandom_or_unavailable_metrics_claim_no_event(self):
        approx, reference = self._inputs()
        moments = Reference.from_moments(jnp.zeros(2), jnp.eye(2))

        with patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit:
            score_posterior(approx, reference, metrics=("mmd",))
            score_posterior(approx, moments, metrics=("sliced_wasserstein",))

        commit.assert_not_called()

    def test_full_metric_preflight_happens_before_event_commit(self):
        approx, reference = self._inputs()

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="unknown metric"),
        ):
            score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein", "bogus"),
            )

        commit.assert_not_called()

    def test_invalid_sliced_wasserstein_inputs_fail_before_event_commit(self):
        reference = Reference(draws=jnp.zeros((8, 2)))

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="n, d"),
        ):
            score_posterior(
                jnp.zeros((8, 2, 1)),
                reference,
                metrics=("sliced_wasserstein",),
            )

        commit.assert_not_called()

    def test_explicit_key_does_not_shift_later_automatic_score(self):
        approx, reference = self._inputs()
        explicit = jax.random.key(11)

        with workflow_run(seed=7):
            expected = score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
            )

        with workflow_run(seed=7):
            score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
                key=explicit,
            )
            actual = score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
            )

        np.testing.assert_array_equal(
            actual["sliced_wasserstein"],
            expected["sliced_wasserstein"],
        )

    def test_explicit_key_reaches_sliced_wasserstein_unchanged(self):
        approx, reference = self._inputs()
        explicit = jax.random.key(11)

        with (
            patch(
                "probpipe.validation._comparison.sliced_wasserstein",
                return_value=jnp.asarray(0.0),
            ) as metric,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
        ):
            score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
                key=explicit,
            )

        assert metric.call_args.kwargs["key"] is explicit
        commit.assert_not_called()
