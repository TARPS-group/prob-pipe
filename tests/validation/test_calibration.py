"""Simulation-based calibration and interval coverage.

Tolerances are measured (independent baselines / known calibration properties)
per STYLE_GUIDE §8.6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import EmpiricalDistribution, GLMLikelihood, MultivariateNormal, SimpleModel
from probpipe.core.record import Record
from probpipe.validation import SBCResult, interval_coverage, simulation_based_calibration
from probpipe.validation._calibration import _component_names, _flatten_point, _ks_uniform


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

    def test_accepts_distribution_input(self):
        # A distribution exposing flat_samples scores identically to its raw draws.
        draws = jax.random.normal(jax.random.PRNGKey(4), (2000, 2))
        emp = EmpiricalDistribution(draws, name="z")
        from_dist = interval_coverage(emp, jnp.array([0.3, -0.4]), levels=(0.9,))
        from_array = interval_coverage(draws, jnp.array([0.3, -0.4]), levels=(0.9,))
        assert bool(jnp.all(from_dist[0.9] == from_array[0.9]))

    def test_default_levels(self):
        draws = jax.random.normal(jax.random.PRNGKey(5), (1000, 2))
        assert set(interval_coverage(draws, jnp.zeros(2))) == {0.5, 0.8, 0.9, 0.95}

    def test_rejects_dimension_mismatch(self):
        draws = jax.random.normal(jax.random.PRNGKey(6), (500, 2))
        with pytest.raises(ValueError, match="dimension"):
            interval_coverage(draws, jnp.zeros(3))


class TestKSUniform:
    def test_flags_uniform_vs_skewed(self):
        big_l, s = 100, 200
        uniform = np.linspace(0, big_l, s).astype(int)[:, None]  # evenly spread ranks
        assert float(_ks_uniform(uniform, big_l)[1][0]) > 0.05  # not rejected
        skewed = (np.arange(s) % (big_l // 10))[:, None]  # all in the bottom decile
        assert float(_ks_uniform(skewed, big_l)[1][0]) < 0.05  # rejected


class TestFlattening:
    """The multi-field θ★ flattening + component-naming that ranks rely on."""

    def test_flatten_point_honors_field_order(self):
        point = Record(a=jnp.array([1.0, 2.0]), b=jnp.array([3.0]))
        np.testing.assert_array_equal(
            np.asarray(_flatten_point(point, ("a", "b"))), [1.0, 2.0, 3.0]
        )
        # The posterior's field order is authoritative (b before a).
        np.testing.assert_array_equal(
            np.asarray(_flatten_point(point, ("b", "a"))), [3.0, 1.0, 2.0]
        )

    def test_component_names_expand_per_field(self):
        # A length-k field becomes field[0..k-1]; a scalar field keeps its name —
        # in posterior field order, matching flat_samples columns.
        emp = EmpiricalDistribution(Record(a=jnp.zeros((10, 2)), b=jnp.zeros((10,))), name="m")
        assert _component_names(emp) == ("a[0]", "a[1]", "b")


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
        # param_names are per flattened component, aligned with the rank columns.
        assert res.param_names == ("beta[0]", "beta[1]")
        assert len(res.param_names) == res.ranks.shape[1]
        # Well-specified model + NUTS → ranks ~ uniform. Measured across seeds 0–2
        # (S=32, L=100): mean normalized rank ∈ [0.46, 0.52], median ks_pvalue ∈
        # [0.58, 0.77]. Assert stable statistics, not a tail bound on the min.
        u = (res.ranks + 0.5) / (res.num_posterior_draws + 1)
        assert np.all((u.mean(axis=0) > 0.35) & (u.mean(axis=0) < 0.65))
        assert float(np.median(res.ks_pvalue)) > 0.2
        # `passed` reflects every per-parameter p-value clearing the default alpha.
        assert isinstance(res.passed, bool)
        assert res.passed == bool(res.ks_pvalue.min() > 0.05)
        # Rank histogram: (num_params, num_bins), each row sums to num_simulations.
        hist = res.rank_histogram(num_bins=10)
        assert hist.shape == (2, 10)
        assert np.all(hist.sum(axis=1) == 32)

    def test_rejects_bad_num_simulations(self):
        model, n = _gaussian_glm()
        with pytest.raises(ValueError, match="num_simulations"):
            simulation_based_calibration(
                model, num_simulations=0, num_posterior_draws=50, num_observations=n
            )

    def test_rejects_random_seed_in_infer_kwargs(self):
        model, n = _gaussian_glm()
        with pytest.raises(ValueError, match="random_seed"):
            simulation_based_calibration(
                model, num_simulations=2, num_posterior_draws=50, num_observations=n, random_seed=0
            )
