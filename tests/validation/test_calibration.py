"""Simulation-based calibration and interval coverage."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import GLMLikelihood, MultivariateNormal, SimpleModel
from probpipe.validation import SBCResult, interval_coverage, simulation_based_calibration
from probpipe.validation._calibration import _ks_uniform


def _gaussian_glm(p: int = 2, n: int = 12, seed: int = 7):
    """A well-specified Gaussian linear model: sampleable prior + generative GLM."""
    X = jax.random.normal(jax.random.PRNGKey(seed), (n, p - 1))
    prior = MultivariateNormal(loc=jnp.zeros(p), cov=jnp.eye(p), name="beta")
    return SimpleModel(prior, GLMLikelihood(tfp_glm.Normal(), x=X)), n


class TestIntervalCoverage:
    def test_contains_center_excludes_far(self):
        draws = jax.random.normal(jax.random.PRNGKey(0), (5000, 2))  # ~ N(0, I)
        cov = interval_coverage(draws, jnp.zeros(2), levels=(0.5, 0.9))
        for level in (0.5, 0.9):
            assert bool(jnp.all(cov[level]))  # truth = 0 is central → covered
        far = interval_coverage(draws, jnp.array([5.0, -5.0]), levels=(0.9,))
        assert not bool(jnp.any(far[0.9]))  # truth deep in the tails → not covered

    def test_frequentist_rate_matches_nominal(self):
        # truth and draws from the same N(0, 1): a fresh truth lands in the
        # central-`level` interval with probability ≈ level.
        draws = jax.random.normal(jax.random.PRNGKey(1), (4000, 1))
        truths = jax.random.normal(jax.random.PRNGKey(2), (400,))
        for level in (0.5, 0.9):
            hits = np.mean(
                [
                    bool(interval_coverage(draws, jnp.array([t]), levels=(level,))[level][0])
                    for t in truths
                ]
            )
            # 400 Bernoulli(level) draws → SE ≈ 0.015–0.025; abs=0.05 is ~2–3 SE.
            assert hits == pytest.approx(level, abs=0.05)

    def test_returns_per_level_per_param(self):
        draws = jax.random.normal(jax.random.PRNGKey(3), (1000, 3))
        cov = interval_coverage(draws, jnp.zeros(3), levels=(0.8, 0.95))
        assert set(cov) == {0.8, 0.95}
        assert cov[0.8].shape == (3,)


class TestKSUniform:
    def test_flags_uniform_vs_skewed(self):
        big_l, s = 100, 200
        uniform = np.linspace(0, big_l, s).astype(int)[:, None]  # evenly spread ranks
        assert float(_ks_uniform(uniform, big_l)[1][0]) > 0.05  # not rejected
        skewed = (np.arange(s) % (big_l // 10))[:, None]  # all in the bottom decile
        assert float(_ks_uniform(skewed, big_l)[1][0]) < 0.05  # rejected


class TestSBC:
    def test_well_specified_ranks_uniform(self):
        model, n = _gaussian_glm()
        res = simulation_based_calibration(
            model,
            num_simulations=32,
            num_posterior_draws=100,
            num_observations=n,
            num_warmup=100,
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(res, SBCResult)
        assert res.ranks.shape == (32, 2)
        assert res.ranks.min() >= 0 and res.ranks.max() <= res.num_posterior_draws
        assert res.param_names == ("beta",)
        assert isinstance(res.passed, bool)
        # Well-specified model + NUTS → ranks ~ uniform → comfortably not rejected.
        assert float(res.ks_pvalue.min()) > 0.01
        # Rank histogram: (num_params, num_bins), each row sums to num_simulations.
        hist = res.rank_histogram(num_bins=10)
        assert hist.shape == (2, 10)
        assert np.all(hist.sum(axis=1) == 32)
