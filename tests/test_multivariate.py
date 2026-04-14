"""Tests for probpipe.distributions.multivariate."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from probpipe.distributions import (
    Dirichlet,
    Multinomial,
    Wishart,
    VonMisesFisher,
)
from probpipe.core.distribution import ArrayDistribution
from probpipe import cov, log_prob, mean, sample, variance


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
            lambda: VonMisesFisher(
                mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="v"
            ),
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
        assert isinstance(multivariate_dist, ArrayDistribution)

    def test_event_shape(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        expected = EXPECTED_EVENT_SHAPES[name]
        assert multivariate_dist.event_shape == expected

    def test_sample_shape(self, multivariate_dist, key):
        samples = sample(multivariate_dist, key=key, sample_shape=(5,))
        expected = (5,) + multivariate_dist.event_shape
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

        v = VonMisesFisher(
            mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="dir"
        )
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
        samples = sample(d, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)


class TestMultinomial:
    def test_samples_nonnegative_integers(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="m")
        samples = sample(d, key=key, sample_shape=(100,))
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
        samples = sample(d, key=key, sample_shape=(10,))
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
        expected = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1))
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
