
from probpipe import distributions


import numpy as np
import pytest


# ---------------------- Helpers for from_distribution ----------------------

class _NoSample:
    def sample(self, n: int):
        raise NotImplementedError


class _FixedSampler2D:
    """
    Deterministic 'distribution' whose sample(n) repeats a fixed matrix of shape (m, d)
    row-wise until length n.
    """
    def __init__(self, data_2d):
        data_2d = np.asarray(data_2d, dtype=float)
        assert data_2d.ndim == 2
        self.data = data_2d

    def sample(self, n: int):
        m, d = self.data.shape
        reps = int(np.ceil(n / m))
        tiled = np.tile(self.data, (reps, 1))
        return tiled[:n, :].astype(float)


# --------------------------------- Tests -----------------------------------

def test_init_shape_validation_errors():
    mu = np.array([0.0, 1.0])
    cov_bad_shape = np.eye(3)
    with pytest.raises(ValueError):
        MvNormal(mean=mu, cov=cov_bad_shape)

    cov_mismatch = np.array([[1.0]])
    with pytest.raises(ValueError):
        MvNormal(mean=mu, cov=cov_mismatch)

    with pytest.raises(ValueError):
        MvNormal(mean=np.array([[0.0, 1.0]]), cov=np.eye(2))


def test_init_handles_singular_covariance_with_jitter():
    # Rank-1 covariance (singular). Your implementation adds jitter to make it SPD.
    mu = np.array([0.0, 0.0])
    cov_singular = np.array([[1.0, 2.0],
                             [2.0, 4.0]])  # det == 0
    dist = MvNormal(mean=mu, cov=cov_singular, rng=np.random.default_rng(0))
    # Should be able to sample and compute log_density without errors
    xs = dist.sample(5)
    _ = dist.log_density(xs)
    assert xs.shape == (5, 2)


def test_dimension_property_and_accessors():
    mu = np.array([1.0, -2.0, 3.0])
    cov = np.diag([0.5, 1.0, 2.0])
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(1))
    assert dist.dimension == 3
    assert np.allclose(dist.mean(), mu)
    assert np.allclose(dist.cov(), cov)


def test_sample_shape_and_empirical_stats_close():
    rng = np.random.default_rng(123)
    mu = np.array([1.5, -0.5])
    cov = np.array([[1.2, 0.3],
                    [0.3, 2.0]])
    dist = MvNormal(mean=mu, cov=cov, rng=rng)

    n = 20_000
    X = dist.sample(n)
    assert X.shape == (n, 2)

    # LLN: empirical mean and covariance should be close
    emp_mu = X.mean(axis=0)
    emp_cov = np.cov(X, rowvar=False, ddof=1)

    assert np.allclose(emp_mu, mu, atol=0.05)
    assert np.allclose(emp_cov, cov, atol=0.08)


def test_log_density_and_density_consistency_and_values_at_mean():
    mu = np.array([0.2, -0.7, 1.3])
    cov = np.array([[1.0, 0.1, 0.0],
                    [0.1, 1.5, 0.2],
                    [0.0, 0.2, 0.8]])
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(2))

    # At the mean, Mahalanobis term is 0 → log p = -0.5*(d log(2π) + log|Σ|)
    d = mu.size
    L = np.linalg.cholesky(cov)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    expected_logp_mu = -0.5 * (d * np.log(2.0 * np.pi) + log_det)

    logp_mu = dist.log_density(mu)
    p_mu = dist.density(mu)

    assert np.isclose(logp_mu, expected_logp_mu, rtol=0, atol=1e-10)
    assert np.isclose(np.exp(logp_mu), p_mu, rtol=1e-12, atol=1e-12)

    # A far-away point should have strictly lower log-density
    x_far = mu + np.array([6.0, -6.0, 6.0])
    assert dist.log_density(x_far) < logp_mu


def test_density_and_log_density_vectorized_shapes():
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(3))

    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])
    p = dist.density(X)
    lp = dist.log_density(X)
    assert p.shape == (3,)
    assert lp.shape == (3,)
    assert np.allclose(np.exp(lp), p, rtol=1e-12, atol=1e-12)


def test_expectation_scalar_component_returns_Normal1D_close_to_true_mean():
    mu = np.array([2.0, -1.0, 0.5])
    cov = np.array([[1.0, 0.2, 0.0],
                    [0.2, 1.5, 0.0],
                    [0.0, 0.0, 0.7]])
    rng = np.random.default_rng(42)
    dist = MvNormal(mean=mu, cov=cov, rng=rng)

    # f(X) = first coordinate
    out = dist.expectation(lambda x: x[:, 0])
    assert isinstance(out, Normal1D)
    assert np.isclose(out.mu, mu[0], atol=0.06)
    # CLT: the returned sigma must be smaller than the parent std of that component
    parent_std = np.sqrt(cov[0, 0])
    assert 0.0 < out.sigma < parent_std


def test_expectation_vector_identity_returns_MvNormal_with_small_cov():
    mu = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.4],
                    [0.4, 1.0]])
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(7))

    # f(X) = X → distribution over the Monte-Carlo mean
    out = dist.expectation(lambda x: x)
    assert isinstance(out, MvNormal)

    # Mean close to true mean; covariance much smaller than original (by ~1/n_mc)
    assert np.allclose(out.mean(), mu, atol=0.1)
    assert out.cov().shape == cov.shape
    assert np.trace(out.cov()) < np.trace(cov)


def test_from_distribution_fits_mean_and_cov_on_fixed_sampler():
    data = np.array([
        [1.0, 2.0],
        [3.0, 5.0],
        [0.0, -1.0],
        [2.0,  4.0],
    ], dtype=float)

    src = _FixedSampler2D(data)
    fitted = MvNormal.from_distribution(src, n=20_000)

    expected_mu = data.mean(axis=0)
    # population covariance of the 4 points (the generator repeats this pattern)
    expected_cov = np.cov(data, rowvar=False, ddof=0)

    assert np.allclose(fitted.mean(), expected_mu, atol=0.02)
    assert np.allclose(fitted.cov(), expected_cov, atol=0.03)


def test_from_distribution_requires_sample_implemented():
    with pytest.raises(NotImplementedError):
        MvNormal.from_distribution(_NoSample(), n=100)


# -------------------- SciPy-related tests: cdf and inv_cdf --------------------

@pytest.mark.skipif(pytest.importorskip("scipy", reason="SciPy required for cdf/inv_cdf") is None,
                    reason="SciPy not available")
def test_cdf_matches_product_of_univariate_for_diagonal_cov():
    from scipy.stats import norm

    mu = np.array([0.5, -1.0, 1.5])
    sig = np.array([1.0, 2.0, 0.5])
    cov = np.diag(sig**2)
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(0))

    X = np.array([
        [0.5, -1.0, 1.5],          # at mean
        [0.5 + 1.0, -1.0, 1.5-0.5],
        [0.5 - 2.0, -1.0 + 2.0, 1.5 + 1.0],
    ], dtype=float)

    # For independent components, joint CDF = product of marginals
    expected = np.prod(norm.cdf((X - mu) / sig), axis=1)
    got = dist.cdf(X)
    assert got.shape == (X.shape[0],)
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(pytest.importorskip("scipy", reason="SciPy required for inv_cdf") is None,
                    reason="SciPy not available")
def test_inv_cdf_matches_closed_form_for_diagonal_cov():
    from scipy.stats import norm

    mu = np.array([1.0, -2.0])
    sig = np.array([0.7, 1.3])
    cov = np.diag(sig**2)
    dist = MvNormal(mean=mu, cov=cov, rng=np.random.default_rng(11))

    # Draw uniform u and map; for independent case:
    # inv_cdf(u) = mu + diag(sig) * Φ^{-1}(u)
    U = np.array([[0.1, 0.9],
                  [0.5, 0.5],
                  [0.8, 0.2]], dtype=float)

    X = dist.inv_cdf(U)
    X_expected = mu + norm.ppf(U) * sig
    assert np.allclose(X, X_expected, rtol=1e-12, atol=1e-12)