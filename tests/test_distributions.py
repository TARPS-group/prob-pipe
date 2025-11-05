import numpy as np
import pytest

from probpipe.core.distributions import EmpiricalDistribution
from probpipe.core.multivariate import Normal1D, MvNormal


# Initialization tests: EmpiricalDistribution
# --------------------------------------------------------------------

def test_init_with_uniform_weights(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    n, d = simple_samples.shape

    assert ed.n == n
    assert ed.d == d
    np.testing.assert_allclose(ed.weights, np.ones(n) / n)
    assert np.allclose(ed.mean(), np.mean(simple_samples, axis=0))


def test_init_with_custom_weights(simple_samples, simple_weights):
    ed = EmpiricalDistribution(simple_samples, simple_weights)
    assert np.isclose(ed.weights.sum(), 1.0)
    assert np.all(ed.weights >= 0)
    np.testing.assert_allclose(ed.mean(), (simple_samples.T @ ed.weights).ravel())


@pytest.mark.parametrize("bad_weights", [
    np.array([-0.1, 0.5, 0.6]),   # negative
    np.array([0.0, 0.0, 0.0]),    # zero sum
    np.array([0.5, 0.5])          # wrong length
])
def test_init_raises_with_invalid_weights(simple_samples, bad_weights):
    with pytest.raises(ValueError):
        EmpiricalDistribution(simple_samples, bad_weights)



# Sampling behavior: EmpiricalDistribution
# --------------------------------------------------------------------

def test_sample_with_replacement(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    n, d = simple_samples.shape

    draws = ed.sample(n, replace=True)
    assert draws.shape == (n, d)
    assert np.all(np.isin(draws, simple_samples))


def test_sample_without_replacement(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    draws = ed.sample(ed.n, replace=False)
    assert draws.shape == (ed.n, ed.d)
    assert np.unique(draws, axis=0).shape[0] == ed.n

    # Sampling more than n points should raise
    with pytest.raises(ValueError):
        ed.sample(ed.n + 1, replace=False)

    # Sampling with replacement can exceed n
    draws_replace = ed.sample(ed.n * 2, replace=True)
    assert draws_replace.shape == (ed.n * 2, ed.d)



# Density / log-density not implemented: EmpiricalDistribution
# --------------------------------------------------------------------

def test_density_not_implemented(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    with pytest.raises(NotImplementedError):
        ed.density(np.array([[1.0]]))

def test_log_density_not_implemented(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    with pytest.raises(NotImplementedError):
        ed.log_density(np.array([[1.0]]))



# Expectation estimation: EmpiricalDistribution
# --------------------------------------------------------------------

def test_expectation_returns_normal1d(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    f = lambda x: x.ravel() ** 2
    out = ed.expectation(f, n_mc=100)
    assert isinstance(out, Normal1D)
    assert out.mu > 0
    assert out.sigma > 0

def test_expectation_returns_mvnormal():
    # Create an arbitrary 2D dataset
    X = np.stack([np.arange(5.0), np.arange(5.0, 10.0)], axis=1)  # (5, 2)
    ed = EmpiricalDistribution(X)

    f = lambda x: x * 2.0
    out = ed.expectation(f, n_mc=200)
    d = X.shape[1]

    assert isinstance(out, MvNormal)
    assert out._mean.shape == (d,)
    assert out._cov.shape == (d, d)
    assert np.all(np.isfinite(out._cov))
    # Ensuring covariance is symmetric positive semidefinite
    np.testing.assert_allclose(out._cov, out._cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(out._cov)
    assert np.all(eigvals >= -1e-10), "Covariance not positive semidefinite"



# from_distribution: EmpiricalDistribution
# --------------------------------------------------------------------

def test_from_distribution_uses_sample():
    # Defining a dummy distribution returning constant samples
    class DummyDist:
        def __init__(self, constant=7.0):
            self.constant = constant

        def sample(self, n):
            return np.full((n, 1), self.constant, dtype=float)

    num_samples = 8
    constant_val = 4.2

    ed = EmpiricalDistribution.from_distribution(DummyDist(constant_val), num_samples=num_samples)

    # Behavioral checks
    assert isinstance(ed, EmpiricalDistribution)
    assert ed.samples.shape == (num_samples, 1)
    np.testing.assert_allclose(ed.samples, constant_val)
    np.testing.assert_allclose(ed.weights.sum(), 1.0)
    assert ed.n == num_samples


# Normal1D
# --------------------------------------------------------------------

def test_normal1d_sample_and_moments():
    mu, sigma = 2.0, 0.5
    num_samples = 5000

    n = Normal1D(mu=mu, sigma=sigma)
    x = n.sample(num_samples)

    # Checking shape: (n, 1)
    assert x.shape == (num_samples, 1)

    # Computing empirical stats
    sample_mean = np.mean(x)
    sample_std = np.std(x)

    # Expected sampling variability
    mean_tolerance = 3 * sigma / np.sqrt(num_samples)
    std_tolerance = 0.1 * sigma  # 10% relative tolerance

    # Behavioral assertions
    assert abs(sample_mean - mu) < mean_tolerance, (
        f"Sample mean {sample_mean:.3f} deviates too far from true mean {mu:.3f}"
    )
    assert abs(sample_std - sigma) < std_tolerance, (
        f"Sample std {sample_std:.3f} deviates too far from true sigma {sigma:.3f}"
    )



def test_normal1d_log_density_matches_density():
    # Parameters (easy to vary later)
    mu, sigma = 1.5, 0.7
    n = Normal1D(mu, sigma)

    # Generating test inputs covering a few standard deviations
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 10).reshape(-1, 1)

    # Evaluating density and log-density
    d = n.density(x)
    logd = n.log_density(x)

    # Basic shape sanity
    assert d.shape == logd.shape == (x.shape[0], 1)
    assert np.all(d > 0), "Density returned non-positive values"

    # Numerical consistency check
    np.testing.assert_allclose(
        logd,
        np.log(d),
        rtol=1e-10,
        atol=1e-10,
        err_msg="log_density() not matching log(density())"
    )


def test_from_distribution_returns_empirical():
    mu, sigma = 1.0, 0.5
    num_samples = 75

    src = Normal1D(mu, sigma)
    emp = EmpiricalDistribution.from_distribution(src, num_samples=num_samples)

    # Behavioral checks
    assert isinstance(emp, EmpiricalDistribution)
    assert emp.samples.shape == (num_samples, 1)
    assert emp.n == num_samples
    assert emp.d == 1

    # Empirical mean should roughly match the true mean within a reasonable tolerance
    sample_mean = np.mean(emp.samples)
    assert abs(sample_mean - mu) < 3 * sigma, (
        f"Empirical mean {sample_mean:.3f} deviates too far from true mean {mu:.3f}"
    )


def test_roundtrip_conversion_normal_to_empirical_and_back():
    # Define true parameters and sample size
    mu, sigma = 1.5, 0.8
    num_samples = 2000

    # Generating source Normal
    n = Normal1D(mu, sigma)

    # Converting to EmpiricalDistribution
    e = EmpiricalDistribution.from_distribution(n, num_samples=num_samples)

    # Fitting back to Normal
    fitted = Normal1D.from_distribution(e)

    # Behavioral checks
    assert isinstance(fitted, Normal1D)
    mean_error = abs(fitted.mu - mu)
    sigma_error = abs(fitted.sigma - sigma)

    # Expected Monte-Carlo variability
    mean_tolerance = 4 * sigma / np.sqrt(num_samples)
    sigma_tolerance = 0.2 * sigma  # 20 % relative tolerance

    assert mean_error < mean_tolerance, (
        f"Mean drift too large: {mean_error:.3f} > {mean_tolerance:.3f}"
    )
    assert sigma_error < sigma_tolerance, (
        f"Std-dev drift too large: {sigma_error:.3f} > {sigma_tolerance:.3f}"
    )


# MvNormal
# --------------------------------------------------------------------
def test_mvnormal_sampling_shape():
    dim = 3
    mean = np.arange(dim, dtype=float)
    cov = np.eye(dim) * 2.0
    num_samples = 2000

    m = MvNormal(mean, cov)
    X = m.sample(num_samples)

    # Check shape
    assert X.shape == (num_samples, dim)

    # Mean should approximately match within Monte Carlo tolerance
    sample_mean = np.mean(X, axis=0)
    mean_tolerance = 3 * np.sqrt(np.diag(cov) / num_samples)
    np.testing.assert_allclose(sample_mean, mean, atol=np.max(mean_tolerance))

    # Covariance should roughly match diagonal values
    sample_cov = np.cov(X.T)
    rel_tol = 0.2  # 20% relative tolerance
    np.testing.assert_allclose(
        np.diag(sample_cov),
        np.diag(cov),
        rtol=rel_tol,
        err_msg="Empirical covariance diagonal differs too much from true covariance",
    )


def test_mvnormal_covariance_reconstruction():
    dim = 2
    mean = np.zeros(dim)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    num_samples = 3000

    m = MvNormal(mean, cov)
    X = m.sample(num_samples)

    emp_cov = np.cov(X.T, ddof=1)

    #Shape and symmetry checks
    assert emp_cov.shape == (dim, dim)
    np.testing.assert_allclose(emp_cov, emp_cov.T, atol=1e-12)

    # Diagonal consistency
    np.testing.assert_allclose(
        np.diag(emp_cov), np.diag(cov), rtol=0.2, err_msg="Diagonal variance mismatch"
    )

    # Off-diagonal consistency
    corr_emp = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
    corr_true = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(corr_emp - corr_true) < 0.2, (
        f"Empirical correlation {corr_emp:.3f} deviates too far from true {corr_true:.3f}"
    )




