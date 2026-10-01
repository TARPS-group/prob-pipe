"""Tests for the multivariate parametric families."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    MathematicalDomainError,
    NumericDistribution,
    TFPDistribution,
    cov,
    log_prob,
    mean,
    sample,
    variance,
)
from probpipe.distributions._capabilities import SupportsVariance
from probpipe.families import Dirichlet, Multinomial, MultivariateNormal, VonMisesFisher, Wishart

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(
    params=[
        pytest.param(
            lambda: Dirichlet(concentration=[1.0, 2.0, 3.0], name="d"),
            id="Dirichlet",
        ),
        pytest.param(
            lambda: Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="m"),
            id="Multinomial",
        ),
        pytest.param(
            lambda: Wishart(df=5.0, scale_tril=jnp.eye(3), name="w"),
            id="Wishart",
        ),
        pytest.param(
            lambda: VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="v"),
            id="VonMisesFisher",
        ),
    ]
)
def multivariate_dist(request):
    return request.param()


# ---------------------------------------------------------------------------
# Expected event shapes for each distribution
# ---------------------------------------------------------------------------

EXPECTED_EVENT_SHAPES = {
    "Dirichlet": (3,),
    "Multinomial": (3,),
    "Wishart": (3, 3),
    "VonMisesFisher": (3,),
}


# ---------------------------------------------------------------------------
# Generic tests
# ---------------------------------------------------------------------------


class TestGeneric:
    def test_is_distribution(self, multivariate_dist):
        assert isinstance(multivariate_dist, TFPDistribution)
        assert isinstance(multivariate_dist, NumericDistribution)

    def test_event_shape(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        expected = EXPECTED_EVENT_SHAPES[name]
        assert multivariate_dist.event_shape == expected

    def test_sample_shape(self, multivariate_dist, key):
        samples = sample(multivariate_dist, key=key, sample_shape=(5,))
        expected = (5, *multivariate_dist.event_shape)
        assert samples.shape == expected

    def test_log_prob_shape(self, multivariate_dist, key):
        s = sample(multivariate_dist, key=key)
        lp = log_prob(multivariate_dist, s)
        assert lp.shape == ()

    def test_mean_finite(self, multivariate_dist):
        m = mean(multivariate_dist)
        assert jnp.all(jnp.isfinite(m))

    def test_repr(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        assert name in repr(multivariate_dist)

    def test_name_required(self, multivariate_dist):
        assert multivariate_dist.name is not None

    def test_name_set(self):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], name="alpha")
        assert d.name == "alpha"

        m = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="counts")
        assert m.name == "counts"

        w = Wishart(df=5.0, scale_tril=jnp.eye(3), name="sigma")
        assert w.name == "sigma"

        v = VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="dir")
        assert v.name == "dir"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestDirichlet:
    def test_samples_sum_to_one(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], name="d")
        samples = sample(d, key=key, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_samples_positive(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], name="d")
        samples = jnp.asarray(sample(d, key=key, sample_shape=(100,)))
        assert jnp.all(samples > 0)


class TestMultinomial:
    def test_samples_nonnegative_integers(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="m")
        samples = jnp.asarray(sample(d, key=key, sample_shape=(100,)))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))

    def test_samples_sum_to_total_count(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="m")
        samples = sample(d, key=key, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 10.0)

    def test_probs_logits_validation(self):
        # Must provide exactly one of probs or logits.
        with pytest.raises(ValueError, match="Exactly one"):
            Multinomial(total_count=10, name="m")

        with pytest.raises(ValueError, match="Exactly one"):
            Multinomial(
                total_count=10,
                probs=[0.2, 0.3, 0.5],
                logits=[0.0, 0.0, 0.0],
                name="m",
            )

        # Using logits instead of probs should work.
        d = Multinomial(total_count=10, logits=[0.0, 0.0, 0.0], name="m")
        assert d.logits is not None
        assert d.probs is None


class TestWishart:
    def test_accepts_scale_tril(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), name="w")
        s = sample(d, key=key)
        assert s.shape == (3, 3)

    def test_accepts_scale(self, key):
        d = Wishart(df=5.0, scale=jnp.eye(3), name="w")
        s = sample(d, key=key)
        assert s.shape == (3, 3)

    def test_error_if_both_given(self):
        with pytest.raises(ValueError, match="exactly one"):
            Wishart(df=5.0, scale_tril=jnp.eye(3), scale=jnp.eye(3), name="w")

    def test_samples_positive_semi_definite(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), name="w")
        samples = jnp.asarray(sample(d, key=key, sample_shape=(10,)))
        # Diagonal elements of a positive semi-definite matrix are >= 0.
        for i in range(10):
            diag = jnp.diag(samples[i])
            assert jnp.all(diag >= 0)


class TestVonMisesFisher:
    def test_samples_unit_norm(self, key):
        d = VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="v")
        samples = sample(d, key=key, sample_shape=(100,))
        norms = jnp.linalg.norm(samples, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_the_variance_is_the_covariance_diagonal(self, key):
        d = VonMisesFisher("v", jnp.array([0.0, 0.6, 0.8]), 4.0)
        assert isinstance(d, SupportsVariance)
        np.testing.assert_allclose(variance(d), jnp.diagonal(cov(d)), rtol=1e-6)
        draws = np.asarray(d._sample(key, (200_000,)))
        np.testing.assert_allclose(variance(d), draws.var(axis=0), atol=5e-3)


def _rank_three_covariance(n=10):
    """``Φ Φᵀ`` for quadratic features at n points, a covariance of rank 3."""
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.stack([jnp.ones_like(x), x, x**2], axis=-1)
    return phi @ phi.T


class TestASingularMultivariateNormal:
    """A positive semidefinite, singular covariance draws through a rank-tolerant root."""

    def test_draws_are_finite_and_have_the_covariance(self, key):
        cov = _rank_three_covariance()
        d = MultivariateNormal("z", jnp.zeros(10), cov=cov)
        draws = np.asarray(d._sample(key, (200_000,)))
        assert np.isfinite(draws).all()
        np.testing.assert_allclose(np.cov(draws, rowvar=False), cov, atol=0.03)

    def test_draws_lie_on_the_affine_support(self, key):
        d = MultivariateNormal("z", jnp.array([1.0, -1.0]), cov=jnp.array([[1.0, 1.0], [1.0, 1.0]]))
        draws = np.asarray(d._sample(key, (1000,)))
        np.testing.assert_allclose(draws[:, 0] - draws[:, 1], 2.0, atol=1e-4)
        assert draws[:, 0].std() > 0.5

    def test_the_scale_factor_is_a_finite_triangular_root(self):
        cov = _rank_three_covariance()
        factor = MultivariateNormal("z", jnp.zeros(10), cov=cov).scale_tril
        np.testing.assert_array_equal(factor, jnp.tril(factor))
        np.testing.assert_allclose(factor @ factor.T, cov, atol=1e-5)

    def test_the_moments_are_the_given_covariances(self):
        cov = _rank_three_covariance()
        d = MultivariateNormal("z", jnp.arange(10.0), cov=cov)
        np.testing.assert_allclose(variance(d), jnp.diagonal(cov), rtol=1e-6)
        np.testing.assert_allclose(d.cov, cov, rtol=1e-6)
        np.testing.assert_allclose(mean(d), jnp.arange(10.0), rtol=1e-6)

    def test_there_is_no_density(self):
        d = MultivariateNormal("z", jnp.zeros(2), cov=jnp.array([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(MathematicalDomainError, match="singular"):
            d._log_prob(jnp.zeros(2))
        with pytest.raises(MathematicalDomainError, match="singular"):
            log_prob(d, jnp.zeros(2))

    def test_a_singular_scale_factor_has_no_density(self):
        d = MultivariateNormal("z", jnp.zeros(2), jnp.array([[1.0, 0.0], [1.0, 0.0]]))
        with pytest.raises(MathematicalDomainError, match="singular"):
            d._log_prob(jnp.zeros(2))

    def test_a_traced_singular_covariance_draws_finite_values_and_scores_nan(self, key):
        cov = jnp.array([[1.0, 1.0], [1.0, 1.0]])

        def draws_and_density(c):
            d = MultivariateNormal("z", jnp.zeros(2), cov=c)
            return d._sample(key, (500,)), d._log_prob(jnp.zeros(2))

        draws, density = jax.jit(draws_and_density)(cov)
        assert np.isfinite(np.asarray(draws)).all()
        assert np.isnan(float(density))

    def test_a_positive_definite_covariance_keeps_its_cholesky_factor(self, key):
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        d = MultivariateNormal("z", jnp.array([1.0, -1.0]), cov=cov)
        np.testing.assert_allclose(d.scale_tril, jnp.linalg.cholesky(cov), rtol=1e-6)
        backend = tfd.MultivariateNormalTriL(jnp.array([1.0, -1.0]), jnp.linalg.cholesky(cov))
        np.testing.assert_array_equal(d._sample(key, (5,)), backend.sample(5, seed=key))
        np.testing.assert_allclose(
            d._log_prob(jnp.zeros(2)), backend.log_prob(jnp.zeros(2)), rtol=1e-6
        )

    def test_a_traced_positive_definite_covariance_is_differentiable(self):
        def log_density(scale):
            d = MultivariateNormal("z", jnp.zeros(3), cov=scale * jnp.eye(3))
            return d._log_prob(jnp.ones(3))

        # log N(1; 0, s I) = -1.5 log(2π s) - 1.5 / s, with derivative -1.5 / s + 1.5 / s².
        for differentiate in (jax.grad(log_density), jax.jit(jax.grad(log_density))):
            assert float(differentiate(2.0)) == pytest.approx(-0.375, rel=1e-5)


# ---------------------------------------------------------------------------
# Numerical baselines — validate mean/variance/cov against analytical formulas
# ---------------------------------------------------------------------------


class TestMultivariateMoments:
    """Analytical mean/variance/cov must match closed-form identities."""

    # -- Dirichlet ---------------------------------------------------------

    def test_dirichlet_mean(self):
        """Dirichlet mean: α_i / α_0 where α_0 = Σα."""
        alpha = np.array([1.0, 2.0, 3.0])
        d = Dirichlet(concentration=alpha, name="d")
        np.testing.assert_allclose(mean(d), alpha / alpha.sum(), rtol=1e-6)

    def test_dirichlet_variance(self):
        """Dirichlet variance: α_i(α_0 - α_i) / (α_0² (α_0 + 1))."""
        alpha = np.array([1.0, 2.0, 3.0])
        alpha_0 = alpha.sum()
        d = Dirichlet(concentration=alpha, name="d")
        expected = alpha * (alpha_0 - alpha) / (alpha_0**2 * (alpha_0 + 1))
        np.testing.assert_allclose(variance(d), expected, rtol=1e-6)

    def test_dirichlet_cov_matches_scipy(self):
        """Full covariance vs scipy.stats.dirichlet."""
        alpha = np.array([1.0, 2.0, 3.0])
        d = Dirichlet(concentration=alpha, name="d")
        np.testing.assert_allclose(cov(d), scipy.stats.dirichlet(alpha).cov(), rtol=1e-5)

    def test_dirichlet_sample_mean_and_cov(self, key):
        """50k-sample mean and cov must match analytical values."""
        alpha = np.array([1.0, 2.0, 3.0])
        d = Dirichlet(concentration=alpha, name="d")
        draws = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), np.asarray(mean(d)), atol=0.005)
        np.testing.assert_allclose(np.cov(draws, rowvar=False), np.asarray(cov(d)), atol=0.002)

    def test_dirichlet_marginal_ks(self, key):
        """Each Dirichlet marginal X_i ~ Beta(α_i, α_0 - α_i)."""
        alpha = np.array([1.0, 2.0, 3.0])
        alpha_0 = alpha.sum()
        d = Dirichlet(concentration=alpha, name="d")
        draws = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        for i in range(3):
            scipy_marginal = scipy.stats.beta(alpha[i], alpha_0 - alpha[i])
            _, p = scipy.stats.kstest(draws[:, i], scipy_marginal.cdf)
            assert p > 0.001, f"KS failed for marginal {i}: p={p:.4e}"

    # -- Multinomial -------------------------------------------------------

    def test_multinomial_mean(self):
        """Multinomial mean: n * p."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, name="m")
        np.testing.assert_allclose(mean(d), 10 * probs, rtol=1e-6)

    def test_multinomial_variance(self):
        """Multinomial variance (diagonal): n * p_i * (1 - p_i)."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, name="m")
        np.testing.assert_allclose(variance(d), 10 * probs * (1 - probs), rtol=1e-6)

    def test_multinomial_cov_matches_scipy(self):
        """Full covariance: n*diag(p) - n*pp'."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, name="m")
        expected = 10 * (np.diag(probs) - np.outer(probs, probs))
        np.testing.assert_allclose(cov(d), expected, rtol=1e-6)

    def test_multinomial_sample_mean_and_cov(self, key):
        """50k-sample mean and cov must match analytical values."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, name="m")
        draws = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), np.asarray(mean(d)), atol=0.05)
        expected_cov = 10 * (np.diag(probs) - np.outer(probs, probs))
        np.testing.assert_allclose(np.cov(draws, rowvar=False), expected_cov, atol=0.1)

    # -- Wishart -----------------------------------------------------------

    def test_wishart_mean_matches_analytical(self):
        """Wishart mean: df * S where S = L L'."""
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), name="w")
        np.testing.assert_allclose(mean(d), 5.0 * np.eye(3), rtol=1e-5)

    def test_wishart_sample_mean(self, key):
        """50k-sample mean of Wishart(5, I) must match 5*I."""
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), name="w")
        draws = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), 5.0 * np.eye(3), atol=0.05)

    # -- Von Mises-Fisher --------------------------------------------------

    def test_vonmisesfisher_mean_direction(self):
        """VMF mean direction must be parallel to the mean_direction parameter."""
        direction = np.array([1.0, 0.0, 0.0])
        d = VonMisesFisher(mean_direction=direction.tolist(), concentration=5.0, name="v")
        m = np.asarray(mean(d))
        np.testing.assert_allclose(m / np.linalg.norm(m), direction, atol=1e-5)

    def test_vonmisesfisher_sample_mean_direction(self, key):
        """50k-sample mean direction must be parallel to mean_direction."""
        direction = np.array([1.0, 0.0, 0.0])
        d = VonMisesFisher(mean_direction=direction.tolist(), concentration=10.0, name="v")
        draws = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        sample_mean = draws.mean(0)
        sample_dir = sample_mean / np.linalg.norm(sample_mean)
        np.testing.assert_allclose(sample_dir, direction, atol=0.01)
