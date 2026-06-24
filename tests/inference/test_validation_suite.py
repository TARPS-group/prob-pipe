"""Dogfood the validation metrics on real fits (issue #301).

NUTS recovers the conjugate model's closed-form posterior (closing #301's
acceptance: a method reproduces the analytic reference within measured
tolerances); vanilla fixed-step ``blackjax_sgld`` exhibits the covariance bias
that #304's calibration is meant to remove. Tolerances are measured across
seeds 0–2 per STYLE_GUIDE §8.6.
"""

from __future__ import annotations

from probpipe import condition_on
from probpipe.core.record import Record
from probpipe.validation import relative_cov_error, score_posterior


class TestNUTSReproducesReference:
    def test_moment_metrics_within_tolerance(
        self, conjugate_linear_model, conjugate_nuts_posterior
    ):
        # score_posterior on an analytic (moments-only) reference scores the moment
        # metrics and skips the sample/score ones. Measured across seeds 0–2:
        # relative_cov_error ∈ [0.04, 0.08], standardized_mean_error ∈ [0.007, 0.021].
        card = score_posterior(conjugate_nuts_posterior, conjugate_linear_model.reference)
        assert {"ksd", "mmd", "sliced_wasserstein"}.isdisjoint(card)  # no draws/score_fn
        assert float(card["relative_cov_error"]) < 0.15
        assert float(card["standardized_mean_error"]) < 0.1


class TestSGLDCovarianceBias:
    def test_sgld_overdisperses_vs_nuts(self, conjugate_linear_model, conjugate_nuts_posterior):
        m = conjugate_linear_model
        sgld = condition_on(
            m.model,
            Record(X=m.design, y=m.data),
            method="blackjax_sgld",
            batch_size=20,
            num_results=5000,
            num_warmup=2000,
            step_size=1e-3,
            random_seed=0,
        )
        rce_nuts = float(relative_cov_error(conjugate_nuts_posterior, m.reference))
        rce_sgld = float(relative_cov_error(sgld, m.reference))
        # Vanilla fixed-step SGLD mis-estimates the posterior covariance. Measured
        # across seeds 0–2: SGLD rce ∈ [0.18, 0.35] vs NUTS [0.04, 0.08], a 2.7–6×
        # gap — the bias #304's calibration must remove.
        assert rce_sgld > 2.0 * rce_nuts
        assert rce_sgld > 0.12


class TestNUTSReproducesNonGaussianReference:
    def test_recovers_skewed_beta_posterior(
        self, beta_bernoulli_model, beta_bernoulli_nuts_posterior
    ):
        m = beta_bernoulli_model
        # The reference is genuinely non-Gaussian (a right-skewed Beta posterior),
        # so good behavior here cannot be coming from Gaussianity alone.
        assert m.posterior_skewness > 0.3  # Gaussian skewness is 0; measured ≈ 0.54
        # The reference carries draws, so score_posterior also runs the sample-based
        # distances — small mmd / sliced_W confirm NUTS captures the skewed *shape*,
        # not just the moments. Measured across seeds 0–2: rce ≤ 0.11, sme ≤ 0.06,
        # mmd ≤ 0.001, sliced_W ≤ 0.008.
        card = score_posterior(beta_bernoulli_nuts_posterior, m.reference)
        assert float(card["relative_cov_error"]) < 0.2
        assert float(card["standardized_mean_error"]) < 0.15
        assert float(card["mmd"]) < 0.01
        assert float(card["sliced_wasserstein"]) < 0.02
