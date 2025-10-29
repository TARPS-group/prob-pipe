import numpy as np
import pytest

from probpipe.core.distributions import EmpiricalDistribution
from probpipe.core.multivariate import Normal1D, MvNormal


# Initialization tests: EmpiricalDistribution
# --------------------------------------------------------------------

def test_init_with_uniform_weights(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    assert ed.n == 3
    assert ed.d == 1
    np.testing.assert_allclose(ed.weights, np.ones(3) / 3)
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
    draws = ed.sample(5, replace=True)
    assert draws.shape == (5, 1)
    assert np.all(np.isin(draws, simple_samples))


def test_sample_without_replacement(simple_samples):
    ed = EmpiricalDistribution(simple_samples)
    draws = ed.sample(3, replace=False)
    assert np.unique(draws).shape[0] <= ed.n

    with pytest.raises(ValueError):
        ed.sample(10, replace=False)



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
    # 2D version
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    ed = EmpiricalDistribution(X)
    f = lambda x: x * 2.0
    out = ed.expectation(f, n_mc=200)
    assert isinstance(out, MvNormal)
    assert out._mean.shape == (2,) 
    assert out._cov.shape == (2, 2)
    assert np.all(np.isfinite(out._cov))



# from_distribution: EmpiricalDistribution
# --------------------------------------------------------------------

def test_from_distribution_uses_sample(monkeypatch):
    class DummyDist:
        def sample(self, n):  # minimal fake
            return np.ones((n, 1)) * 7.0

    ed = EmpiricalDistribution.from_distribution(DummyDist(), num_samples=5)
    assert isinstance(ed, EmpiricalDistribution)
    assert ed.samples.shape == (5, 1)
    np.testing.assert_allclose(ed.samples, 7.0)


# Normal1D
# --------------------------------------------------------------------

def test_normal1d_sample_and_moments():
    n = Normal1D(mu=0, sigma=1)
    x = n.sample(10000)
    assert x.shape == (10000,1)
    # empirical mean and std should roughly match true parameters
    assert abs(np.mean(x) - 0) < 0.1
    assert abs(np.std(x) - 1) < 0.1



def test_normal1d_log_density_matches_density():
    n = Normal1D(0, 1)
    x = np.linspace(-1, 1, 5)
    d = n.density(x)
    logd = n.log_density(x)
    np.testing.assert_allclose(logd, np.log(d), atol=1e-10)


def test_from_distribution_returns_empirical():
    src = Normal1D(1, 0.5)
    emp = EmpiricalDistribution.from_distribution(src, num_samples=50)
    assert isinstance(emp, EmpiricalDistribution)
    assert emp.samples.shape[0] == 50


def test_roundtrip_conversion_normal_to_empirical_and_back():
    n = Normal1D(0, 1)
    e = EmpiricalDistribution.from_distribution(n, num_samples=1000)
    fitted = Normal1D.from_distribution(e)
    assert abs(fitted.mu - 0) < 0.1
    assert abs(fitted.sigma - 1) < 0.1


# MvNormal
# --------------------------------------------------------------------
def test_mvnormal_sampling_shape():
    mean = np.zeros(2)
    cov = np.eye(2)
    m = MvNormal(mean, cov)
    X = m.sample(1000)
    assert X.shape == (1000, 2)
    assert np.allclose(np.mean(X, axis=0), mean, atol=0.1)

def test_mvnormal_covariance_reconstruction():
    mean = np.zeros(2)
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    m = MvNormal(mean, cov)
    X = m.sample(2000)
    emp_cov = np.cov(X.T)
    # diagonal within ~20% tolerance (Monte Carlo)
    assert np.allclose(np.diag(emp_cov), np.diag(cov), rtol=0.2)




