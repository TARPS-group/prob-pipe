"""Workflow RNG ownership tests for validation operations."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    ReplayCompatibilityError,
    conditional_distribution,
    predictive_check,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.families import GaussianFamily, glm_likelihood
from probpipe.functions import _context
from probpipe.inference._inference_utils import integer_seed, run_seed
from probpipe.validation import (
    Reference,
    score_posterior,
    simulation_based_calibration,
)


class _OpaqueLikelihood:
    def generate_data(self, params, num_observations, *, key=None):
        noise = jax.random.normal(key, (*jnp.shape(params), num_observations))
        return jnp.asarray(params)[..., None] + noise


def _check_setup():
    """A normal kernel of six observations and its normal prior."""
    prior = Normal("mu", 0.0, 1.0)
    likelihood = conditional_distribution(
        "y_given_mu",
        lambda mu: Normal("y", mu * jnp.ones(6), 1.0),
        given_spec=prior.event_spec.components,
    )
    return likelihood, prior


class TestPredictiveCheckBroker:
    def test_a_seeded_check_is_reproducible_and_claims_one_event(self):
        likelihood, prior = _check_setup()

        def run(num_replications):
            with (
                patch(
                    "probpipe.functions._context.derive_event_key_words_from_encoded",
                    wraps=_context.derive_event_key_words_from_encoded,
                ) as derive,
                workflow_run(seed=7),
            ):
                result = predictive_check(
                    likelihood, prior, jnp.mean, num_replications=num_replications
                )
            return np.asarray(result["replicated_statistics"].atoms.values), derive

        first, first_derive = run(8)
        second, second_derive = run(8)
        larger, larger_derive = run(16)

        np.testing.assert_array_equal(first, second)
        assert first_derive.call_count == 1
        assert second_derive.call_count == 1
        assert larger_derive.call_count == 1
        assert larger.shape == (16,)

    def test_several_statistics_claim_one_event(self):
        likelihood, prior = _check_setup()

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                wraps=_context.derive_event_key_words_from_encoded,
            ) as derive,
            workflow_run(seed=7),
        ):
            predictive_check(likelihood, prior, [jnp.mean, jnp.max], num_replications=4)

        assert derive.call_count == 1

    def test_an_explicit_key_claims_no_event(self):
        likelihood, prior = _check_setup()

        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            predictive_check(
                likelihood, prior, jnp.mean, num_replications=3, key=jax.random.key(11)
            )

        commit.assert_not_called()

    def test_a_check_outside_a_workflow_run_draws_a_fresh_key(self):
        likelihood, prior = _check_setup()

        result = predictive_check(likelihood, prior, jnp.mean, num_replications=3)

        assert result["replicated_statistics"].num_atoms == 3

    def test_numpy_integer_counts_are_normalized_before_event_commit(self):
        likelihood, prior = _check_setup()

        with workflow_run(seed=7):
            result = predictive_check(likelihood, prior, jnp.mean, num_replications=np.int64(3))

        assert result["replicated_statistics"].num_atoms == 3

    @pytest.mark.parametrize("value", [True, 0, 1.5])
    def test_invalid_counts_fail_before_event_commit(self, value):
        likelihood, prior = _check_setup()

        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            predictive_check(likelihood, prior, jnp.mean, num_replications=value)

        commit.assert_not_called()

    def test_a_missing_given_slot_fails_before_event_commit(self):
        likelihood, _ = _check_setup()
        other = Normal("tau", 0.0, 1.0)

        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(ValueError, match=r"given slots \['mu'\]"),
        ):
            predictive_check(likelihood, other, jnp.mean, num_replications=3)

        commit.assert_not_called()


class _FakeConditionOn:
    """A stand-in for ``condition_on`` that records its observed values and returns zero draws.

    A fit also records the seed it reads through ``run_seed``, as an inference
    method does, so the seed is a key of the workflow scope the fit runs in.
    The evaluation of a kernel reads no seed.
    """

    def __init__(self):
        self.seeds = []
        self.observations = []

    def __call__(self, d, given):
        self.observations.append(np.asarray(given["y"]))
        return EmpiricalDistribution("beta", jnp.zeros((4, 1)))

    def with_options(self, *, method=None, method_options=None):
        def fit(d, given):
            self.seeds.append(integer_seed(run_seed(method_options or {}, "fake")))
            return self(d, given)

        return fit


def _conjugate_calibration(**options):
    """Calibration of the exact posterior kernel of a normal mean given three observations."""
    prior = Normal("mu", 0.0, 1.0)
    likelihood = conditional_distribution(
        "y_given_mu",
        lambda mu: Normal("y", mu * jnp.ones(3), 1.0),
        given_spec=prior.event_spec.components,
    )
    exact = conditional_distribution(
        "posterior",
        lambda y: Normal("mu", jnp.sum(y) / 4.0, 0.5),
        given_spec={"y": NumericArraySpec((3,))},
    )
    return simulation_based_calibration(
        likelihood * prior,
        observed="y",
        posterior=exact,
        num_posterior_draws=10,
        **{"num_simulations": 4, **options},
    )


class TestSimulationBasedCalibrationBroker:
    @staticmethod
    def _model():
        x = jnp.ones((3, 1))
        prior = MultivariateNormal(loc=jnp.zeros(1), cov=jnp.eye(1), label="beta")
        return glm_likelihood("y", GaussianFamily(), X=x, dispersion=1.0) * prior

    def test_a_call_in_a_seeded_scope_reproduces_its_ranks(self):
        with workflow_run(seed=7):
            first = _conjugate_calibration().ranks
        with workflow_run(seed=7):
            second = _conjugate_calibration().ranks

        np.testing.assert_array_equal(first, second)

    def test_another_seed_changes_the_ranks(self):
        with workflow_run(seed=7):
            first = _conjugate_calibration().ranks
        with workflow_run(seed=8):
            other = _conjugate_calibration().ranks

        assert not np.array_equal(first, other)

    def test_an_unscoped_call_draws_afresh(self):
        # Eight ranks in {0, …, 10} agree by chance with probability about 5e-9.
        first = _conjugate_calibration(num_simulations=8).ranks
        second = _conjugate_calibration(num_simulations=8).ranks

        assert not np.array_equal(first, second)

    def test_each_replication_fit_reads_its_own_seed(self, monkeypatch):
        fake_condition_on = _FakeConditionOn()
        monkeypatch.setattr("probpipe.validation._calibration.condition_on", fake_condition_on)

        def seeds():
            fake_condition_on.seeds.clear()
            with workflow_run(seed=7):
                simulation_based_calibration(
                    self._model(), observed="y", num_simulations=5, num_posterior_draws=4
                )
            return tuple(fake_condition_on.seeds)

        first = seeds()

        assert len(set(first)) == 5
        assert seeds() == first

    def test_the_replications_observe_the_first_draw_of_the_call_whatever_the_posterior(
        self, monkeypatch
    ):
        """Every replication's ``y`` is a row of the draw of the joint that the call claims first.

        A fit reads a seed and a kernel's evaluation does not, so the two routes
        claim different numbers of events per replication and still see the same data.
        """
        fake_condition_on = _FakeConditionOn()
        monkeypatch.setattr("probpipe.validation._calibration.condition_on", fake_condition_on)
        kernel = conditional_distribution(
            "posterior",
            lambda y: MultivariateNormal(loc=y[:1], cov=jnp.eye(1), label="beta"),
            given_spec={"y": NumericArraySpec((3,))},
        )
        with workflow_run(seed=7):
            first_draw = sample.with_options(raw=True)(self._model(), sample_shape=(3,))

        for posterior in (None, kernel):
            fake_condition_on.observations.clear()
            with workflow_run(seed=7):
                simulation_based_calibration(
                    self._model(),
                    observed="y",
                    posterior=posterior,
                    num_simulations=3,
                    num_posterior_draws=4,
                )
            np.testing.assert_array_equal(
                np.stack(fake_condition_on.observations), np.asarray(first_draw["y"])
            )

    def test_a_key_keyword_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'key'"):
            _conjugate_calibration(key=jax.random.key(0))

    def test_a_call_inside_replay_run_raises(self):
        with workflow_run(seed=7):
            recorded = sample(Normal("z", 0.0, 1.0))

        with pytest.raises(ReplayCompatibilityError), replay_run(recorded.provenance):
            _conjugate_calibration()

    def test_numpy_integer_counts_are_normalized_before_event_commit(self, monkeypatch):
        monkeypatch.setattr("probpipe.validation._calibration.condition_on", _FakeConditionOn())

        with workflow_run(seed=11):
            result = simulation_based_calibration(
                self._model(),
                observed="y",
                num_simulations=np.int64(2),
                num_posterior_draws=np.int64(4),
            )

        assert result.ranks.shape == (2, 1)
        assert type(result.num_posterior_draws) is int

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("num_simulations", True),
            ("num_simulations", 0),
            ("num_posterior_draws", True),
            ("num_posterior_draws", 0),
            ("method_options", [("num_warmup", 10)]),
        ],
    )
    def test_invalid_arguments_fail_before_event_commit(self, argument, value):
        kwargs = {
            "observed": "y",
            "num_simulations": 2,
            "num_posterior_draws": 4,
            argument: value,
        }
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            simulation_based_calibration(self._model(), **kwargs)

        commit.assert_not_called()

    def test_a_posterior_whose_slots_miss_the_observed_fields_fails_before_event_commit(self):
        other_slot = conditional_distribution(
            "posterior",
            lambda x, z: Normal("beta", x + z, 1.0),
            given_spec={"x": NumericArraySpec(()), "z": NumericArraySpec(())},
        )
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="given slots"),
        ):
            simulation_based_calibration(
                self._model(),
                observed="y",
                posterior=other_slot,
                num_simulations=2,
                num_posterior_draws=4,
            )

        commit.assert_not_called()

    def test_a_model_that_does_not_sample_fails_before_event_commit(self):
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises(TypeError, match="does not support SBC joint sampling"),
        ):
            simulation_based_calibration(
                _OpaqueLikelihood(),
                observed="y",
                num_simulations=2,
                num_posterior_draws=4,
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
                    "probpipe.functions._context._commit_stochastic_invocation",
                    wraps=_context._commit_stochastic_invocation,
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

        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            score_posterior(approx, reference, metrics=("mmd",))
            score_posterior(approx, moments, metrics=("sliced_wasserstein",))

        commit.assert_not_called()

    def test_full_metric_preflight_happens_before_event_commit(self):
        approx, reference = self._inputs()

        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
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
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
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
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
        ):
            score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
                key=explicit,
            )

        assert metric.call_args.kwargs["key"] is explicit
        commit.assert_not_called()

    def test_missing_resolved_sliced_wasserstein_key_is_an_internal_error(self):
        approx, reference = self._inputs()

        with (
            patch(
                "probpipe.validation._comparison._resolve_validation_key",
                return_value=None,
            ),
            patch("probpipe.validation._comparison.sliced_wasserstein") as metric,
            pytest.raises(RuntimeError, match="resolved PRNG key"),
        ):
            score_posterior(
                approx,
                reference,
                metrics=("sliced_wasserstein",),
            )

        metric.assert_not_called()
