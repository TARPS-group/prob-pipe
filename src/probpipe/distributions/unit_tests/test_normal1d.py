# test_normal1d.py
# Adjust this import to your project layout:
# from your_module.distributions import Normal1D
from distributions.distributions import Normal1D  # <-- change to your actual module path

import numpy as np
import pytest
import os
import sys
import inspect


# ---------------------- Helpers for from_distribution ----------------------

class _NoSample:
    def sample(self, n: int):
        raise NotImplementedError


class _FixedSampler:
    """Deterministic 'distribution' whose sample(n) repeats a fixed vector."""
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def sample(self, n: int):
        # np.resize repeats the array as needed and truncates to length n
        return np.resize(self.data, n).astype(float)


# --------------------------------- Tests -----------------------------------

def test_init_invalid_sigma_raises():
    with pytest.raises(ValueError):
        Normal1D(mu=0.0, sigma=0.0)
    with pytest.raises(ValueError):
        Normal1D(mu=0.0, sigma=-1.0)


def test_sample_shape_and_basic_stats():
    rng = np.random.default_rng(123)
    mu, sigma = 1.5, 2.0
    dist = Normal1D(mu=mu, sigma=sigma, rng=rng)

    n = 10_000
    xs = dist.sample(n)

    assert xs.shape == (n,)
    # Law of large numbers — loose tolerances to avoid flakiness
    assert np.isclose(xs.mean(), mu, atol=0.1)
    assert np.isclose(xs.std(ddof=1), sigma, atol=0.1)


def test_density_matches_analytic_formula():
    mu, sigma = 0.0, 1.25
    dist = Normal1D(mu=mu, sigma=sigma, rng=np.random.default_rng(0))

    x = np.array([-2.0, 0.0, 2.0], dtype=float)
    z = (x - mu) / sigma
    expected = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)

    got = dist.density(x)
    assert got.shape == x.shape
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_log_density_consistent_with_density():
    mu, sigma = -0.7, 0.8
    dist = Normal1D(mu=mu, sigma=sigma, rng=np.random.default_rng(1))

    x = np.linspace(-3, 3, 21)
    logp = dist.log_density(x)
    p = dist.density(x)

    assert np.allclose(np.exp(logp), p, rtol=1e-12, atol=1e-12)
    # sanity: log-density at the mean should be the maximum among nearby points
    idx_mu = np.argmin(np.abs(x - mu))
    assert logp[idx_mu] == pytest.approx(logp.max(), rel=0, abs=1e-12)


def test_density_and_log_density_shapes():
    dist = Normal1D(mu=0.0, sigma=1.0, rng=np.random.default_rng(2))
    x_vec = np.array([0.0, 1.0, 2.0])
    assert dist.density(x_vec).shape == x_vec.shape
    assert dist.log_density(x_vec).shape == x_vec.shape


def test_expectation_identity_returns_normal_close_to_mu():
    mu, sigma = 3.0, 2.0
    rng = np.random.default_rng(42)
    dist = Normal1D(mu=mu, sigma=sigma, rng=rng)

    # f(x) = x  -> E[f(X)] = mu ; Var(mean of f) = sigma^2 / n_mc
    out = dist.expectation(lambda x: x)

    # Should return a Normal1D (distribution over the sample mean)
    assert isinstance(out, Normal1D)

    # Mean of the returned distribution should be close to true mu
    assert np.isclose(out.mu, mu, atol=0.12)

    # CLT reduces variance: sigma_out should be smaller than parent sigma
    assert 0.0 < out.sigma < sigma


def test_expectation_quadratic_matches_sigma_squared():
    # For X ~ N(mu, sigma^2), E[(X - mu)^2] = sigma^2
    mu, sigma = -1.0, 1.6
    dist = Normal1D(mu=mu, sigma=sigma, rng=np.random.default_rng(7))

    out = dist.expectation(lambda x: (x - mu) ** 2)
    assert isinstance(out, Normal1D)
    assert np.isclose(out.mu, sigma ** 2, atol=0.2)  # MC error tolerance
    assert out.sigma > 0.0


def test_from_distribution_fits_mean_and_sigma_on_fixed_sampler():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    src = _FixedSampler(data)
    fitted = Normal1D.from_distribution(src, n=10_000)

    expected_mu = data.mean()
    expected_sigma = data.std(ddof=0) #ddof=0 because n is 10_000. So close to population.

    assert np.isclose(fitted.mu, expected_mu, atol=1e-2)
    assert np.isclose(fitted.sigma, expected_sigma, atol=1e-2)


def test_from_distribution_requires_sample_implemented():
    with pytest.raises(NotImplementedError):
        Normal1D.from_distribution(_NoSample(), n=100)


if __name__ == "__main__":

    # Optional manual runner. Use: `python test_normal1d.py --manual`
    def _manual_run() -> int:
        fails = 0
        for name, obj in sorted(globals().items()):
            if name.startswith("test_") and inspect.isfunction(obj):
                try:
                    obj()
                    print(f"PASS {name}")
                except AssertionError as e:
                    print(f"FAIL {name}: {e}")
                    fails += 1
                except Exception as e:
                    print(f"ERROR {name}: {e.__class__.__name__}: {e}")
                    fails += 1
        return 1 if fails else 0

    args = sys.argv[1:]

    # If you pass --manual, run tests by directly calling the test functions.
    if "--manual" in args:
        sys.exit(_manual_run())

    # Otherwise, run via pytest (recommended). This *does* discover & execute test_*
    try:
        import pytest  # already imported at top too; here for clarity
    except Exception:
        print("pytest is required unless you use --manual. Install with `pip install pytest`.")
        sys.exit(1)

    # If no args, run just this file quietly; else forward args to pytest.
    pytest_args = args or ["-q", os.path.abspath(__file__)]
    sys.exit(pytest.main(pytest_args))
