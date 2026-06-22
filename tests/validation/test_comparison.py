"""Posterior-vs-reference comparison metrics (issue #301, PR 1).

Each metric is checked against an independent baseline (closed form, a known
identity, or a clearly-separated case), with measured tolerances per
STYLE_GUIDE §8.6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.validation import (
    Reference,
    ksd,
    mmd,
    relative_cov_error,
    score_posterior,
    sliced_wasserstein,
    standardized_mean_error,
    std_ratios,
)


def _mvn(key, n, mean, cov):
    return jax.random.multivariate_normal(key, jnp.asarray(mean), jnp.asarray(cov), shape=(n,))


class TestMomentMetrics:
    def test_standardized_mean_error_zero_when_matched(self):
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(2))
        approx = jnp.zeros((10, 2))  # sample mean == ref mean
        assert float(standardized_mean_error(approx, ref)) == pytest.approx(0.0, abs=1e-5)

    def test_standardized_mean_error_mahalanobis_value(self):
        # Σ_ref = diag(4, 9) → L = diag(2, 3); μ̂ = (2, 3), μ_ref = 0;
        # ‖L⁻¹ μ̂‖ = ‖(1, 1)‖ = √2.
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.diag(jnp.array([4.0, 9.0])))
        approx = jnp.broadcast_to(jnp.array([2.0, 3.0]), (50, 2))
        assert float(standardized_mean_error(approx, ref)) == pytest.approx(np.sqrt(2.0), abs=1e-5)

    def test_relative_cov_error_zero_when_matched(self):
        draws = _mvn(jax.random.PRNGKey(0), 500, jnp.zeros(3), jnp.eye(3))
        ref = Reference.from_draws(draws)
        # Σ̂ recomputed from the same draws equals Σ_ref → ‖I − Σ⁻¹Σ̂‖₂ = 0.
        assert float(relative_cov_error(draws, ref)) == pytest.approx(0.0, abs=1e-4)

    def test_relative_cov_error_operator_norm_value(self):
        # Σ_ref = I; approx ~ N(0, diag(4, 1)) → M ≈ diag(4, 1); ‖I − M‖₂ ≈ 3.
        approx = _mvn(jax.random.PRNGKey(1), 20000, jnp.zeros(2), jnp.diag(jnp.array([4.0, 1.0])))
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(2))
        # Observed across seeds ~2.9–3.1 at n=20000.
        assert float(relative_cov_error(approx, ref)) == pytest.approx(3.0, abs=0.2)

    def test_std_ratios(self):
        approx = _mvn(jax.random.PRNGKey(2), 20000, jnp.zeros(2), jnp.diag(jnp.array([4.0, 4.0])))
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.diag(jnp.array([1.0, 4.0])))
        # σ̂ ≈ (2, 2); σ_ref = (1, 2) → ratios ≈ (2, 1).
        np.testing.assert_allclose(np.asarray(std_ratios(approx, ref)), [2.0, 1.0], atol=0.1)

    def test_missing_reference_piece_raises(self):
        # A draws-only Reference lacks (mean, cov), so the moment metrics raise.
        draws = _mvn(jax.random.PRNGKey(0), 100, jnp.zeros(2), jnp.eye(2))
        ref = Reference(draws=draws)
        with pytest.raises(ValueError, match="mean"):
            standardized_mean_error(draws, ref)


class TestDistances:
    def test_mmd_same_distribution_near_zero(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        x = _mvn(k1, 400, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 400, jnp.zeros(2), jnp.eye(2))
        # Independent same-distribution samples → unbiased MMD² ≈ 0.
        assert abs(float(mmd(x, y))) < 0.02

    def test_mmd_separated_distributions_positive(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(4))
        x = _mvn(k1, 400, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 400, jnp.array([3.0, 3.0]), jnp.eye(2))
        assert float(mmd(x, y)) > 0.1

    def test_sliced_wasserstein_mean_shift_1d(self):
        # d = 1: sliced W₂ reduces to 1-D W₂ ≈ |mean shift| for equal-variance Gaussians.
        k1, k2 = jax.random.split(jax.random.PRNGKey(5))
        x = jax.random.normal(k1, (2000, 1))
        y = jax.random.normal(k2, (2000, 1)) + 2.0
        sw = float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(6)))
        assert sw == pytest.approx(2.0, abs=0.1)

    def test_sliced_wasserstein_same_distribution_near_zero(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        x = jax.random.normal(k1, (2000, 1))
        y = jax.random.normal(k2, (2000, 1))
        assert float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(8))) < 0.15


class TestKSD:
    def test_ksd_matched_small(self):
        x = _mvn(jax.random.PRNGKey(9), 500, jnp.zeros(2), jnp.eye(2))
        # Samples match the target N(0, I) (score = -x) → KSD small.
        assert float(ksd(x, lambda t: -t)) < 0.3

    def test_ksd_mismatch_larger(self):
        x = _mvn(jax.random.PRNGKey(10), 500, jnp.zeros(2), jnp.eye(2))
        matched = float(ksd(x, lambda t: -t))
        mismatch = float(ksd(x, lambda t: -(t - 3.0)))  # score of N(3, I); samples are N(0, I)
        assert mismatch > matched
        assert mismatch > 0.5


class TestScorePosterior:
    def test_scorecard_keys_and_skips(self):
        draws = _mvn(jax.random.PRNGKey(11), 400, jnp.zeros(2), jnp.eye(2))
        approx = _mvn(jax.random.PRNGKey(12), 400, jnp.zeros(2), jnp.eye(2))
        # Full reference → every metric present.
        full = Reference.from_draws(draws, score_fn=lambda t: -t)
        card = score_posterior(approx, full, key=jax.random.PRNGKey(13))
        assert set(card) >= {
            "standardized_mean_error",
            "relative_cov_error",
            "sliced_wasserstein",
            "mmd",
            "ksd",
        }
        # Moments-only reference → sample/score metrics skipped, not errored.
        moments = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(2))
        card2 = score_posterior(approx, moments, key=jax.random.PRNGKey(14))
        assert "standardized_mean_error" in card2
        assert {"ksd", "mmd", "sliced_wasserstein"}.isdisjoint(card2)
