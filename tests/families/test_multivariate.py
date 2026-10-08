"""Tests for the multivariate parametric families."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
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
    workflow_run,
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
            lambda: Dirichlet(concentration=[1.0, 2.0, 3.0], label="d"),
            id="Dirichlet",
        ),
        pytest.param(
            lambda: Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], label="m"),
            id="Multinomial",
        ),
        pytest.param(
            lambda: Wishart(df=5.0, scale_tril=jnp.eye(3), label="w"),
            id="Wishart",
        ),
        pytest.param(
            lambda: VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, label="v"),
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
        samples = sample(multivariate_dist, sample_shape=(5,))
        expected = (5, *multivariate_dist.event_shape)
        assert samples.shape == expected

    def test_log_prob_shape(self, multivariate_dist, key):
        s = sample(multivariate_dist)
        lp = log_prob(multivariate_dist, s)
        assert lp.shape == ()

    def test_mean_finite(self, multivariate_dist):
        m = mean(multivariate_dist)
        assert jnp.all(jnp.isfinite(m))

    def test_repr(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        assert name in repr(multivariate_dist)

    def test_name_required(self, multivariate_dist):
        assert multivariate_dist.label is not None

    def test_name_set(self):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], label="alpha")
        assert d.label == "alpha"

        m = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], label="counts")
        assert m.label == "counts"

        w = Wishart(df=5.0, scale_tril=jnp.eye(3), label="sigma")
        assert w.label == "sigma"

        v = VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, label="dir")
        assert v.label == "dir"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestDirichlet:
    def test_samples_sum_to_one(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], label="d")
        samples = sample(d, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_samples_positive(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], label="d")
        samples = jnp.asarray(sample(d, sample_shape=(100,)))
        assert jnp.all(samples > 0)


class TestMultinomial:
    def test_samples_nonnegative_integers(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], label="m")
        samples = jnp.asarray(sample(d, sample_shape=(100,)))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))

    def test_samples_sum_to_total_count(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], label="m")
        samples = sample(d, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 10.0)

    def test_probs_logits_validation(self):
        # Must provide exactly one of probs or logits.
        with pytest.raises(ValueError, match="exactly one of probs or logits"):
            Multinomial(total_count=10, label="m")

        with pytest.raises(ValueError, match="exactly one of probs or logits"):
            Multinomial(
                total_count=10,
                probs=[0.2, 0.3, 0.5],
                logits=[0.0, 0.0, 0.0],
                label="m",
            )

        # Using logits instead of probs should work.
        d = Multinomial(total_count=10, logits=[0.0, 0.0, 0.0], label="m")
        assert d.logits is not None
        assert d.probs is None


class TestWishart:
    def test_accepts_scale_tril(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), label="w")
        s = sample(d)
        assert s.shape == (3, 3)

    def test_accepts_scale(self, key):
        d = Wishart(df=5.0, scale=jnp.eye(3), label="w")
        s = sample(d)
        assert s.shape == (3, 3)

    def test_error_if_both_given(self):
        with pytest.raises(ValueError, match="cannot both be given"):
            Wishart(df=5.0, scale_tril=jnp.eye(3), scale=jnp.eye(3), label="w")

    def test_samples_positive_semi_definite(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), label="w")
        samples = jnp.asarray(sample(d, sample_shape=(10,)))
        # Diagonal elements of a positive semi-definite matrix are >= 0.
        for i in range(10):
            diag = jnp.diag(samples[i])
            assert jnp.all(diag >= 0)


class TestVonMisesFisher:
    def test_samples_unit_norm(self, key):
        d = VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0, label="v")
        samples = sample(d, sample_shape=(100,))
        norms = jnp.linalg.norm(samples, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)

    def test_the_variance_is_the_covariance_diagonal(self, key):
        d = VonMisesFisher("v", jnp.array([0.0, 0.6, 0.8]), 4.0)
        assert isinstance(d, SupportsVariance)
        np.testing.assert_allclose(variance(d), jnp.diagonal(cov(d)), rtol=1e-6)
        draws = np.asarray(d._sample(key, (200_000,)))
        np.testing.assert_allclose(variance(d), draws.var(axis=0), atol=5e-3)


def _block_diagonal(blocks):
    """The block-diagonal matrix of *blocks*, over the row-major flattening of the rows."""
    return scipy.linalg.block_diag(*np.asarray(blocks))


#: Three rows of a bivariate normal, with one covariance each.
_ROW_LOC = jnp.array([[0.0, 1.0], [2.0, -1.0], [-1.0, 0.5]])
_ROW_COV = jnp.array([[2.0, 0.5], [0.5, 1.0]])
_ROW_COVS = jnp.stack([_ROW_COV, 2.0 * _ROW_COV, 0.5 * _ROW_COV])


def _normal_rows(how):
    """A multivariate normal over three rows, and each row's covariance."""
    if how == "shared-cov":
        return MultivariateNormal("z", _ROW_LOC, cov=_ROW_COV), jnp.stack([_ROW_COV] * 3)
    if how == "cov-per-row":
        return MultivariateNormal("z", _ROW_LOC, cov=_ROW_COVS), _ROW_COVS
    factors = jnp.linalg.cholesky(_ROW_COVS)
    return MultivariateNormal("z", _ROW_LOC, factors), _ROW_COVS


_HOWS = ["shared-cov", "cov-per-row", "scale-factor-per-row"]


class TestIndependentRows:
    """Parameters with more axes than one law needs give one law over independent rows (VII.1)."""

    @pytest.mark.parametrize("how", _HOWS)
    def test_a_multivariate_normal_draws_independent_rows(self, how, key):
        d, covs = _normal_rows(how)
        assert d.event_shape == (3, 2)
        draws = np.asarray(d._sample(key, (100_000,)))
        assert draws.shape == (100_000, 3, 2)
        flat_cov = np.cov(draws.reshape(100_000, 6), rowvar=False)
        np.testing.assert_allclose(flat_cov, _block_diagonal(covs), atol=0.05)
        np.testing.assert_allclose(draws.mean(axis=0), _ROW_LOC, atol=0.03)

    @pytest.mark.parametrize("how", _HOWS)
    def test_its_density_sums_the_rows_densities(self, how):
        d, covs = _normal_rows(how)
        x = jnp.array([[0.5, 0.5], [1.0, -2.0], [0.0, 0.0]])
        expected = sum(
            float(MultivariateNormal("r", _ROW_LOC[i], cov=covs[i])._log_prob(x[i]))
            for i in range(3)
        )
        assert float(d._log_prob(x)) == pytest.approx(expected, rel=1e-5)
        assert d._log_prob(jnp.stack([x, x])).shape == (2,)

    @pytest.mark.parametrize("how", _HOWS)
    def test_its_moments_and_quantiles_are_per_row(self, how):
        d, covs = _normal_rows(how)
        scales = np.sqrt(np.diagonal(np.asarray(covs), axis1=-2, axis2=-1))
        np.testing.assert_allclose(mean(d), _ROW_LOC, rtol=1e-6)
        np.testing.assert_allclose(variance(d), scales**2, rtol=1e-5)
        np.testing.assert_allclose(cov(d), _block_diagonal(covs), rtol=1e-5, atol=1e-6)
        levels = jnp.array([0.1, 0.9])
        expected = np.asarray(_ROW_LOC) + scales * scipy.stats.norm.ppf([0.1, 0.9])[:, None, None]
        np.testing.assert_allclose(d._quantile(levels), expected, rtol=1e-4, atol=1e-5)

    def test_one_location_with_a_covariance_per_row_draws_rows(self):
        d = MultivariateNormal("z", jnp.zeros(2), cov=_ROW_COVS)
        assert d.event_shape == (3, 2)
        np.testing.assert_allclose(mean(d), jnp.zeros((3, 2)))
        np.testing.assert_allclose(cov(d), _block_diagonal(_ROW_COVS), rtol=1e-5, atol=1e-6)

    def test_a_dirichlet_over_rows(self):
        concentration = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 1.0]])
        d = Dirichlet("p", concentration)
        assert d.event_shape == (2, 3)
        x = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
        rows = [Dirichlet("r", concentration[i]) for i in range(2)]
        expected = sum(float(row._log_prob(x[i])) for i, row in enumerate(rows))
        assert float(d._log_prob(x)) == pytest.approx(expected, rel=1e-5)
        np.testing.assert_allclose(mean(d), concentration / concentration.sum(-1, keepdims=True))
        blocks = [row._cov().to_dense() for row in rows]
        np.testing.assert_allclose(cov(d), _block_diagonal(blocks), rtol=1e-5, atol=1e-7)

    def test_a_multinomial_over_rows(self):
        probs = jnp.array([[0.2, 0.8], [0.5, 0.5]])
        d = Multinomial("m", jnp.array([5.0, 10.0]), probs=probs)
        assert d.event_shape == (2, 2)
        np.testing.assert_allclose(mean(d), jnp.array([[1.0, 4.0], [5.0, 5.0]]), rtol=1e-6)
        x = jnp.array([[1.0, 4.0], [3.0, 7.0]])
        expected = float(Multinomial("r", 5.0, probs=probs[0])._log_prob(x[0])) + float(
            Multinomial("r", 10.0, probs=probs[1])._log_prob(x[1])
        )
        assert float(d._log_prob(x)) == pytest.approx(expected, rel=1e-5)

    def test_a_wishart_over_rows(self):
        factors = jnp.stack([jnp.eye(2), 2.0 * jnp.eye(2)])
        d = Wishart("w", jnp.array([3.0, 4.0]), scale_tril=factors)
        assert d.event_shape == (2, 2, 2)
        expected = jnp.stack([3.0 * jnp.eye(2), 4.0 * 4.0 * jnp.eye(2)])
        np.testing.assert_allclose(mean(d), expected, rtol=1e-6)

    def test_a_von_mises_fisher_over_rows(self):
        directions = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        d = VonMisesFisher("v", directions, jnp.array([2.0, 5.0]))
        assert d.event_shape == (2, 2)
        rows = [VonMisesFisher("r", directions[i], c) for i, c in enumerate([2.0, 5.0])]
        np.testing.assert_allclose(mean(d), jnp.stack([mean(row) for row in rows]), rtol=1e-6)
        blocks = [row._cov().to_dense() for row in rows]
        np.testing.assert_allclose(cov(d), _block_diagonal(blocks), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(variance(d), jnp.stack([variance(row) for row in rows]))


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
        d = Dirichlet(concentration=alpha, label="d")
        np.testing.assert_allclose(mean(d), alpha / alpha.sum(), rtol=1e-6)

    def test_dirichlet_variance(self):
        """Dirichlet variance: α_i(α_0 - α_i) / (α_0² (α_0 + 1))."""
        alpha = np.array([1.0, 2.0, 3.0])
        alpha_0 = alpha.sum()
        d = Dirichlet(concentration=alpha, label="d")
        expected = alpha * (alpha_0 - alpha) / (alpha_0**2 * (alpha_0 + 1))
        np.testing.assert_allclose(variance(d), expected, rtol=1e-6)

    def test_dirichlet_cov_matches_scipy(self):
        """Full covariance vs scipy.stats.dirichlet."""
        alpha = np.array([1.0, 2.0, 3.0])
        d = Dirichlet(concentration=alpha, label="d")
        np.testing.assert_allclose(cov(d), scipy.stats.dirichlet(alpha).cov(), rtol=1e-5)

    def test_dirichlet_sample_mean_and_cov(self):
        """50k-sample mean and cov must match analytical values."""
        alpha = np.array([1.0, 2.0, 3.0])
        d = Dirichlet(concentration=alpha, label="d")
        with workflow_run(seed=0):
            draws = np.asarray(sample(d, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), np.asarray(mean(d)), atol=0.005)
        np.testing.assert_allclose(np.cov(draws, rowvar=False), np.asarray(cov(d)), atol=0.002)

    def test_dirichlet_marginal_ks(self):
        """Each Dirichlet marginal X_i ~ Beta(α_i, α_0 - α_i)."""
        alpha = np.array([1.0, 2.0, 3.0])
        alpha_0 = alpha.sum()
        d = Dirichlet(concentration=alpha, label="d")
        with workflow_run(seed=1):
            draws = np.asarray(sample(d, sample_shape=(50_000,)))
        for i in range(3):
            scipy_marginal = scipy.stats.beta(alpha[i], alpha_0 - alpha[i])
            _, p = scipy.stats.kstest(draws[:, i], scipy_marginal.cdf)
            assert p > 0.001, f"KS failed for marginal {i}: p={p:.4e}"

    # -- Multinomial -------------------------------------------------------

    def test_multinomial_mean(self):
        """Multinomial mean: n * p."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, label="m")
        np.testing.assert_allclose(mean(d), 10 * probs, rtol=1e-6)

    def test_multinomial_variance(self):
        """Multinomial variance (diagonal): n * p_i * (1 - p_i)."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, label="m")
        np.testing.assert_allclose(variance(d), 10 * probs * (1 - probs), rtol=1e-6)

    def test_multinomial_cov_matches_scipy(self):
        """Full covariance: n*diag(p) - n*pp'."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, label="m")
        expected = 10 * (np.diag(probs) - np.outer(probs, probs))
        np.testing.assert_allclose(cov(d), expected, rtol=1e-6)

    def test_multinomial_sample_mean_and_cov(self):
        """50k-sample mean and cov must match analytical values."""
        probs = np.array([0.2, 0.3, 0.5])
        d = Multinomial(total_count=10, probs=probs, label="m")
        with workflow_run(seed=2):
            draws = np.asarray(sample(d, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), np.asarray(mean(d)), atol=0.05)
        expected_cov = 10 * (np.diag(probs) - np.outer(probs, probs))
        np.testing.assert_allclose(np.cov(draws, rowvar=False), expected_cov, atol=0.1)

    # -- Wishart -----------------------------------------------------------

    def test_wishart_mean_matches_analytical(self):
        """Wishart mean: df * S where S = L L'."""
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), label="w")
        np.testing.assert_allclose(mean(d), 5.0 * np.eye(3), rtol=1e-5)

    def test_wishart_sample_mean(self, key):
        """50k-sample mean of Wishart(5, I) must match 5*I."""
        d = Wishart(df=5.0, scale_tril=jnp.eye(3), label="w")
        draws = np.asarray(sample(d, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), 5.0 * np.eye(3), atol=0.05)

    # -- Von Mises-Fisher --------------------------------------------------

    def test_vonmisesfisher_mean_direction(self):
        """VMF mean direction must be parallel to the mean_direction parameter."""
        direction = np.array([1.0, 0.0, 0.0])
        d = VonMisesFisher(mean_direction=direction.tolist(), concentration=5.0, label="v")
        m = np.asarray(mean(d))
        np.testing.assert_allclose(m / np.linalg.norm(m), direction, atol=1e-5)

    def test_vonmisesfisher_sample_mean_direction(self, key):
        """50k-sample mean direction must be parallel to mean_direction."""
        direction = np.array([1.0, 0.0, 0.0])
        d = VonMisesFisher(mean_direction=direction.tolist(), concentration=10.0, label="v")
        draws = np.asarray(sample(d, sample_shape=(50_000,)))
        sample_mean = draws.mean(0)
        sample_dir = sample_mean / np.linalg.norm(sample_mean)
        np.testing.assert_allclose(sample_dir, direction, atol=0.01)
