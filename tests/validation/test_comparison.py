"""Posterior-vs-reference comparison metrics (issue #301).

Each metric is checked against an independent baseline (closed form, a known
identity, or a clearly-separated case), with measured tolerances per
STYLE_GUIDE §8.6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EmpiricalDistribution
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
        # Σ_ref = I; approx ~ N(0, diag(4, 1)) → M ≈ diag(3.95, 0.99); ‖I − M‖₂ ≈ 3.
        approx = _mvn(jax.random.PRNGKey(1), 20000, jnp.zeros(2), jnp.diag(jnp.array([4.0, 1.0])))
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(2))
        # = 2.97 at this seed (n=20000); ~2.9–3.15 across seeds.
        assert float(relative_cov_error(approx, ref)) == pytest.approx(3.0, abs=0.15)

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

    def test_relative_cov_error_non_diagonal(self):
        # Non-diagonal Σ_ref with a non-commuting Σ̂ (a case the diagonal tests miss):
        # the symmetric whitened form equals the worst variance-ratio deviation
        # max_d|1 − λ_d| exactly, where the λ_d are the generalized eigenvalues of
        # (Σ̂, Σ_ref) — computed here via an independent numpy eigendecomposition.
        sigma_ref = jnp.array([[2.0, 0.8], [0.8, 1.0]])
        approx = _mvn(
            jax.random.PRNGKey(40), 40000, jnp.zeros(2), jnp.array([[1.0, -0.5], [-0.5, 3.0]])
        )
        got = float(relative_cov_error(approx, Reference.from_moments(jnp.zeros(2), sigma_ref)))
        lam = np.linalg.eigvals(
            np.linalg.solve(np.asarray(sigma_ref), np.cov(np.asarray(approx), rowvar=False))
        ).real
        np.testing.assert_allclose(got, np.max(np.abs(1 - lam)), rtol=1e-4)

    def test_moment_metrics_reject_dimension_mismatch(self):
        approx = _mvn(jax.random.PRNGKey(0), 100, jnp.zeros(2), jnp.eye(2))  # d = 2
        ref = Reference.from_moments(mean=jnp.zeros(3), cov=jnp.eye(3))  # d = 3
        with pytest.raises(ValueError, match="reference"):
            standardized_mean_error(approx, ref)
        with pytest.raises(ValueError, match="reference"):
            relative_cov_error(approx, ref)

    def test_from_moments_validates_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(3))  # cov not (d, d)
        with pytest.raises(ValueError, match="shape"):
            Reference.from_moments(mean=jnp.zeros((2, 2)), cov=jnp.eye(2))  # mean not 1-D


class TestDistances:
    def test_mmd_same_distribution_near_zero(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        x = _mvn(k1, 400, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 400, jnp.zeros(2), jnp.eye(2))
        # Independent same-distribution samples → unbiased MMD² ≈ 0
        # (|MMD²| = 0.005 at this seed; ≤ 0.006 across seeds, n=400).
        assert abs(float(mmd(x, y))) < 0.015

    def test_mmd_separated_distributions_positive(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(4))
        x = _mvn(k1, 400, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 400, jnp.array([3.0, 3.0]), jnp.eye(2))
        # Means 3√2 apart → unbiased MMD² clearly positive (~0.88 at this seed).
        assert float(mmd(x, y)) > 0.1

    def test_mmd_float_bandwidth_hand_computed(self):
        # Exercises the non-median (float-bandwidth) path against a closed form.
        # For x = y = {0, 1} and ℓ² = 1: off_xx = off_yy = e⁻¹, cross-mean =
        # (1 + e⁻¹)/2, so MMD²_u = 2e⁻¹ − (1 + e⁻¹) = e⁻¹ − 1.
        pts = jnp.array([[0.0], [1.0]])
        assert float(mmd(pts, pts, bandwidth=1.0)) == pytest.approx(np.exp(-1.0) - 1.0, abs=1e-5)

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
        # Same distribution → sliced W₂ ≈ 0 (0.05 at this seed, n=2000).
        assert float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(8))) < 0.15

    def test_sliced_wasserstein_exact_1d_vs_pot(self):
        # The exact 1-D W₂ — including unequal sample sizes — matches POT's
        # wasserstein_1d (which returns W₂², hence the sqrt).
        ot = pytest.importorskip("ot")
        for i, (n, m) in enumerate([(2000, 2000), (1000, 3000)]):
            kx, ky = jax.random.split(jax.random.PRNGKey(30 + i))
            x = jax.random.normal(kx, (n, 1))
            y = jax.random.normal(ky, (m, 1)) + 1.5
            ours = float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(0)))
            pot = float(np.sqrt(ot.wasserstein_1d(np.asarray(x[:, 0]), np.asarray(y[:, 0]), p=2)))
            assert ours == pytest.approx(pot, abs=1e-3)

    def test_sliced_wasserstein_multivariate_vs_pot(self):
        # The full projection pipeline tracks POT's sliced_wasserstein_distance.
        ot = pytest.importorskip("ot")
        k1, k2 = jax.random.split(jax.random.PRNGKey(20))
        x = _mvn(k1, 1500, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 1500, jnp.array([1.5, -0.5]), jnp.diag(jnp.array([2.0, 0.5])))
        ours = float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(21), n_projections=400))
        pot = float(
            ot.sliced_wasserstein_distance(np.asarray(x), np.asarray(y), n_projections=400, seed=0)
        )
        # Independent projection sets; both converge to the true sliced-W₂.
        assert ours == pytest.approx(pot, rel=0.05)


class TestKSD:
    def test_ksd_matched_small(self):
        x = _mvn(jax.random.PRNGKey(9), 500, jnp.zeros(2), jnp.eye(2))
        # Samples match the target N(0, I) (score = -x) → KSD ≈ 0
        # (0.0 at this seed; ≤ 0.04 across seeds, n=500).
        assert float(ksd(x, lambda t: -t)) < 0.05

    def test_ksd_mismatch_larger(self):
        x = _mvn(jax.random.PRNGKey(10), 500, jnp.zeros(2), jnp.eye(2))
        matched = float(ksd(x, lambda t: -t))
        mismatch = float(ksd(x, lambda t: -(t - 3.0)))  # score of N(3, I); samples are N(0, I)
        assert mismatch > matched
        # mismatch ≈ 2.8, matched ≈ 0.02 at this seed (~2.7–2.85 across seeds).
        assert mismatch > 2.0

    def test_ksd_float_bandwidth(self):
        # Exercises the non-median (float c²) path: matched score → ≈ 0,
        # mismatched score → clearly larger (0.0 vs 2.86 at this seed, c² = 2.25).
        xm = _mvn(jax.random.PRNGKey(9), 500, jnp.zeros(2), jnp.eye(2))
        xs = _mvn(jax.random.PRNGKey(10), 500, jnp.zeros(2), jnp.eye(2))
        assert float(ksd(xm, lambda t: -t, bandwidth=1.5)) < 0.05
        assert float(ksd(xs, lambda t: -(t - 3.0), bandwidth=1.5)) > 2.0


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

    def test_skips_absent_moment_pieces(self):
        # A draws-only reference lacks (mean, cov): score_posterior skips the
        # moment metrics rather than raising (unlike the direct metric calls).
        draws = _mvn(jax.random.PRNGKey(15), 400, jnp.zeros(2), jnp.eye(2))
        approx = _mvn(jax.random.PRNGKey(16), 400, jnp.zeros(2), jnp.eye(2))
        ref = Reference(draws=draws)
        card = score_posterior(
            approx,
            ref,
            metrics=("standardized_mean_error", "relative_cov_error", "std_ratios", "mmd"),
            key=jax.random.PRNGKey(17),
        )
        assert "mmd" in card  # draws present
        assert {"standardized_mean_error", "relative_cov_error", "std_ratios"}.isdisjoint(card)

    def test_custom_metric_selection_and_unknown(self):
        approx = _mvn(jax.random.PRNGKey(18), 200, jnp.zeros(2), jnp.eye(2))
        ref = Reference.from_moments(mean=jnp.zeros(2), cov=jnp.eye(2))
        assert set(score_posterior(approx, ref, metrics=("std_ratios",))) == {"std_ratios"}
        with pytest.raises(ValueError, match="unknown metric"):
            score_posterior(approx, ref, metrics=("bogus",))


class TestInputHandling:
    def test_rejects_non_2d_input(self):
        bad = jnp.zeros((300, 2, 3))  # 3-D: not an (n, d) draws matrix
        with pytest.raises(ValueError, match="n, d"):
            mmd(bad, bad)

    def test_mismatched_dimension_raises(self):
        x = _mvn(jax.random.PRNGKey(0), 100, jnp.zeros(2), jnp.eye(2))
        y = _mvn(jax.random.PRNGKey(1), 100, jnp.zeros(3), jnp.eye(3))
        with pytest.raises(ValueError, match="share a dimension"):
            mmd(x, y)

    def test_too_few_draws_raises(self):
        one, two = jnp.zeros((1, 2)), jnp.zeros((2, 2))
        with pytest.raises(ValueError, match=">= 2"):
            mmd(one, two)
        with pytest.raises(ValueError, match=">= 2"):
            ksd(one, lambda t: -t)

    def test_accepts_distribution_input(self):
        # A distribution exposing flat_samples scores identically to its raw draws.
        draws = _mvn(jax.random.PRNGKey(2), 400, jnp.zeros(2), jnp.eye(2))
        emp = EmpiricalDistribution(draws, name="z")
        ref = Reference.from_moments(mean=jnp.array([0.1, -0.2]), cov=jnp.eye(2))
        from_dist = float(standardized_mean_error(emp, ref))
        from_array = float(standardized_mean_error(draws, ref))
        assert from_dist == pytest.approx(from_array, abs=1e-6)

    def test_metrics_jit_compatible(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        x = _mvn(k1, 300, jnp.zeros(2), jnp.eye(2))
        y = _mvn(k2, 300, jnp.zeros(2), jnp.eye(2))
        assert float(jax.jit(mmd)(x, y)) == pytest.approx(float(mmd(x, y)), abs=1e-5)
        ksd_j = jax.jit(lambda a: ksd(a, lambda t: -t))
        assert float(ksd_j(x)) == pytest.approx(float(ksd(x, lambda t: -t)), abs=1e-5)
        sw_j = jax.jit(lambda a, b: sliced_wasserstein(a, b, key=jax.random.PRNGKey(0)))
        eager = float(sliced_wasserstein(x, y, key=jax.random.PRNGKey(0)))
        assert float(sw_j(x, y)) == pytest.approx(eager, abs=1e-5)
