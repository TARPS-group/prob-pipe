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
from probpipe.functions import _context, _rng
from probpipe.inference._inference_utils import integer_seed, run_seed
from probpipe.validation import (
    Reference,
    score_posterior,
    simulation_based_calibration,
    sliced_wasserstein,
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
            pytest.raises(ValueError, match=r"does not produce \['mu'\]"),
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
            self.seeds.append(integer_seed(run_seed("fake")))
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
            pytest.raises(TypeError, match="model must be a distribution that can be sampled"),
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

    @pytest.mark.parametrize("certified", [False, True])
    @pytest.mark.parametrize("closed_over", [False, True])
    def test_jit_rejects_random_scoring_before_any_event_or_key(
        self, monkeypatch, certified, closed_over
    ):
        approx, reference = self._inputs()
        monkeypatch.setattr(_rng._JAX_KEY_ADAPTER_STATE, "certified", False)
        if certified:
            _rng.jax_key_from_words((0, 0))

        def score(draws):
            return score_posterior(draws, reference, metrics=("sliced_wasserstein",))

        compiled = jax.jit(lambda: score(approx)) if closed_over else jax.jit(score)
        args = () if closed_over else (approx,)
        with (
            workflow_run(seed=7),
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            patch("probpipe.functions._context.jax_key_from_words") as adapt,
        ):
            for _ in range(2):
                with pytest.raises(RuntimeError, match="'score-posterior' operation"):
                    compiled(*args)

        commit.assert_not_called()
        adapt.assert_not_called()

    @pytest.mark.parametrize("certified", [False, True])
    @pytest.mark.parametrize("transform", ["grad", "vmap", "vmap-grad", "grad-vmap"])
    def test_unstaged_scoring_rejects_before_any_event_or_key(
        self, monkeypatch, certified, transform
    ):
        approx, reference = self._inputs()
        monkeypatch.setattr(_rng._JAX_KEY_ADAPTER_STATE, "certified", False)
        if certified:
            _rng.jax_key_from_words((0, 0))

        def score(scale):
            return score_posterior(approx * scale, reference, metrics=("sliced_wasserstein",))[
                "sliced_wasserstein"
            ]

        def apply(metric):
            if transform == "grad":
                return jax.grad(metric)(jnp.asarray(0.7))
            scales = jnp.array([0.7, 1.3])
            if transform == "vmap":
                return jax.vmap(metric)(scales)
            if transform == "vmap-grad":
                return jax.vmap(jax.grad(metric))(scales)
            return jax.grad(lambda s: jnp.sum(jax.vmap(metric)(s)))(scales)

        with (
            workflow_run(seed=7),
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            patch("probpipe.functions._context.jax_key_from_words") as adapt,
            pytest.raises(RuntimeError, match="'score-posterior' operation"),
        ):
            apply(score)

        commit.assert_not_called()
        adapt.assert_not_called()
        assert _rng._JAX_KEY_ADAPTER_STATE.certified is certified

    @pytest.mark.parametrize(
        "transform",
        [
            "jit-grad",
            "grad-jit",
            "jit-vmap",
            "vmap-jit",
            "jit-vmap-grad",
            "jit-grad-vmap",
            "scan",
            "cond",
            "make-jaxpr",
        ],
    )
    def test_staged_scoring_rejects_nested_transformations_before_any_event_or_key(self, transform):
        approx, reference = self._inputs()

        def score(scale):
            return score_posterior(approx * scale, reference, metrics=("sliced_wasserstein",))[
                "sliced_wasserstein"
            ]

        with (
            workflow_run(seed=7),
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            patch("probpipe.functions._context.jax_key_from_words") as adapt,
            pytest.raises(RuntimeError, match="'score-posterior' operation"),
        ):
            if transform == "jit-grad":
                jax.jit(jax.grad(score))(1.0)
            elif transform == "grad-jit":
                jax.grad(jax.jit(score))(1.0)
            elif transform == "jit-vmap":
                jax.jit(jax.vmap(score))(jnp.array([0.7, 1.3]))
            elif transform == "vmap-jit":
                jax.vmap(jax.jit(score))(jnp.array([0.7, 1.3]))
            elif transform == "jit-vmap-grad":
                jax.jit(jax.vmap(jax.grad(score)))(jnp.array([0.7, 1.3]))
            elif transform == "jit-grad-vmap":
                jax.jit(jax.grad(lambda scales: jax.vmap(score)(scales).sum()))(
                    jnp.array([0.7, 1.3])
                )
            elif transform == "scan":
                jax.lax.scan(lambda carry, scale: (carry, score(scale)), 0, jnp.ones(2))
            elif transform == "cond":
                jax.lax.cond(True, score, lambda scale: scale, 1.0)
            else:
                jax.make_jaxpr(score)(1.0)

        commit.assert_not_called()
        adapt.assert_not_called()

    @pytest.mark.parametrize("transform", [jax.jit, jax.grad, jax.vmap])
    def test_rejected_score_preserves_the_next_workflow_draw(self, transform):
        approx, reference = self._inputs()

        def score(scale):
            return score_posterior(approx * scale, reference, metrics=("sliced_wasserstein",))[
                "sliced_wasserstein"
            ]

        transformed = transform(score)
        argument = jnp.ones(2) if transform is jax.vmap else 1.0
        law = Normal("z", 0.0, 1.0)
        with workflow_run(seed=7):
            expected = sample.with_options(raw=True)(law)
        with workflow_run(seed=7):
            for _ in range(2):
                with pytest.raises(RuntimeError, match="'score-posterior' operation"):
                    transformed(argument)
            actual = sample.with_options(raw=True)(law)

        np.testing.assert_array_equal(actual, expected)

    def test_jit_skips_unavailable_random_metric_without_claiming_an_event(self):
        approx, _ = self._inputs()
        reference = Reference.from_moments(jnp.zeros(2), jnp.eye(2))
        compiled = jax.jit(
            lambda draws: score_posterior(draws, reference, metrics=("sliced_wasserstein",))
        )
        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            assert compiled(approx) == {}
        commit.assert_not_called()

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

    def test_sliced_wasserstein_inside_the_callers_jit_raises(self, monkeypatch):
        monkeypatch.setattr(_rng, "_JAX_KEY_ADAPTER_STATE", _rng._JAXKeyAdapterState())
        approx, reference = self._inputs()

        def score(draws):
            return score_posterior(draws, reference, metrics=("sliced_wasserstein",))

        with pytest.raises(RuntimeError, match="'score-posterior' operation"):
            jax.jit(score)(approx)

        assert not _rng._JAX_KEY_ADAPTER_STATE.certified

    def test_sliced_wasserstein_with_an_explicit_key_runs_inside_the_callers_jit(self):
        approx, reference = self._inputs()
        key = jax.random.key(3)

        def score(draws, key):
            return sliced_wasserstein(draws, reference.draws, key=key)

        np.testing.assert_allclose(
            np.asarray(jax.jit(score)(approx, key)),
            np.asarray(score(approx, key)),
            rtol=1e-5,
        )

    @pytest.mark.parametrize("independent", [False, True])
    def test_explicit_keys_select_shared_or_independent_mapped_projections(self, independent):
        approx, reference = self._inputs()
        key = jax.random.key(3)
        scales = jnp.array([0.7, 0.7])
        keys = jax.random.split(key, len(scales)) if independent else key

        def score(scale, projection_key):
            return sliced_wasserstein(approx * scale, reference.draws, key=projection_key)

        mapped = jax.jit(jax.vmap(score, in_axes=(0, 0 if independent else None)))
        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            actual = mapped(scales, keys)

        expected = jnp.stack(
            [
                score(scale, keys[index] if independent else key)
                for index, scale in enumerate(scales)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        if independent:
            assert actual[0] != actual[1]
        else:
            np.testing.assert_array_equal(actual[0], actual[1])
        commit.assert_not_called()

    def test_explicit_key_compiled_gradient_matches_the_one_dimensional_shift(self):
        draws = jnp.arange(8, dtype=jnp.float32)[:, None]

        def score(shift, key):
            return sliced_wasserstein(draws + shift, draws, key=key)

        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            value, gradient = jax.jit(jax.value_and_grad(score))(0.5, jax.random.key(3))

        # Translating an empirical law by a positive shift gives W2 = shift and derivative 1.
        np.testing.assert_allclose(value, 0.5, rtol=1e-6)
        np.testing.assert_allclose(gradient, 1.0, rtol=1e-6)
        commit.assert_not_called()

    @pytest.mark.parametrize("transform", [jax.jit, jax.grad, jax.vmap])
    def test_deterministic_scoring_supports_caller_transformations(self, transform):
        approx, reference = self._inputs()

        def score(scale):
            return score_posterior(approx * scale, reference, metrics=("standardized_mean_error",))[
                "standardized_mean_error"
            ]

        argument = jnp.array([0.7, 1.3]) if transform is jax.vmap else 0.7
        with patch("probpipe.functions._context._commit_stochastic_invocation") as commit:
            result = transform(score)(argument)

        assert np.all(np.isfinite(np.asarray(result)))
        assert result.shape == ((2,) if transform is jax.vmap else ())
        commit.assert_not_called()

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
