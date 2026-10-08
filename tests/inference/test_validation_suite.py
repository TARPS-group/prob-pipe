"""Dogfood the validation metrics on real fits.

NUTS recovers the conjugate model's closed-form posterior within measured
tolerances, and the metrics detect the covariance bias of vanilla fixed-step
``blackjax_sgld``. Tolerances are measured across four workflow seeds per
STYLE_GUIDE §8.6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probpipe import condition_on, workflow_run
from probpipe.validation import relative_cov_error, score_posterior


class TestNUTSReproducesReference:
    def test_moment_metrics_within_tolerance(
        self, conjugate_linear_model, conjugate_nuts_posterior
    ):
        # score_posterior on an analytic (moments-only) reference scores the moment
        # metrics and skips the sample/score ones. Measured across four workflow
        # seeds: relative_cov_error ∈ [0.03, 0.05], standardized_mean_error ∈
        # [0.014, 0.020].
        card = score_posterior(conjugate_nuts_posterior, conjugate_linear_model.reference)
        assert {"ksd", "mmd", "sliced_wasserstein"}.isdisjoint(card)  # no draws/score_fn
        assert float(card["relative_cov_error"]) < 0.15
        assert float(card["standardized_mean_error"]) < 0.1


class TestSGLDCovarianceBias:
    def test_sgld_overdisperses_vs_nuts(self, conjugate_linear_model, conjugate_nuts_posterior):
        m = conjugate_linear_model
        with workflow_run(seed=0):
            sgld = condition_on.with_options(
                method="blackjax_sgld",
                method_options={
                    "batch_size": 20,
                    "num_results": 5000,
                    "num_warmup": 2000,
                    "step_size": 1e-3,
                },
            )(m.model, {"y": m.data})
        rce_nuts = float(relative_cov_error(conjugate_nuts_posterior, m.reference))
        rce_sgld = float(relative_cov_error(sgld, m.reference))
        # Vanilla fixed-step SGLD mis-estimates the posterior covariance. Measured
        # across four workflow seeds: SGLD rce ∈ [0.22, 0.43] vs NUTS [0.03, 0.05],
        # a 4.8–11× gap.
        assert rce_sgld > 2.0 * rce_nuts
        assert rce_sgld > 0.12


class TestNUTSReproducesNonGaussianReference:
    def test_recovers_skewed_beta_posterior(
        self, beta_bernoulli_model, beta_bernoulli_nuts_posterior
    ):
        m = beta_bernoulli_model
        # The reference is markedly non-Gaussian — a right-skewed Beta posterior.
        assert m.posterior_skewness > 0.5  # Gaussian skewness is 0; measured ≈ 0.7
        # NUTS captures the shape: the distributional metrics sit near the sampling
        # floor. Measured across seeds 0–2: mmd ≤ 0.001, sliced_W ≤ 0.009.
        with workflow_run(seed=0):
            nuts = score_posterior(beta_bernoulli_nuts_posterior, m.reference)
        assert float(nuts["mmd"]) < 0.004
        assert float(nuts["sliced_wasserstein"]) < 0.014
        assert float(nuts["relative_cov_error"]) < 0.2

        # Negative control: a Gaussian with the *same mean and variance* matches the
        # moments (small relative_cov_error) yet is rejected by mmd — proving the
        # metric actually sees the skew, so NUTS passing above is meaningful and not
        # an artifact of Beta(3,12) being ~Gaussian. Measured: Gaussian mmd ≈ 0.011,
        # well above NUTS's 0.004 bound (≈13× the NUTS value).
        mean = m.reference.mean
        sd = jnp.sqrt(jnp.diag(m.reference.cov))
        gaussian = mean + sd * jax.random.normal(jax.random.PRNGKey(7), (5000, mean.shape[0]))
        with workflow_run(seed=0):
            control = score_posterior(gaussian, m.reference)
        assert float(control["relative_cov_error"]) < 0.05  # moments match
        assert float(control["mmd"]) > 0.004  # but the non-Gaussian shape is rejected
