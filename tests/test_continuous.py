"""Tests for continuous univariate distributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as _scipy

from probpipe.distributions import (
    Normal,
    Beta,
    Gamma,
    InverseGamma,
    Exponential,
    LogNormal,
    StudentT,
    Uniform,
    Cauchy,
    Laplace,
    HalfNormal,
    HalfCauchy,
    Pareto,
    TruncatedNormal,
)
from probpipe import ArrayDistribution, log_prob, mean, sample, variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# Map of (class, kwargs) for every continuous distribution under test.
_CONTINUOUS_DISTS = {
    "Normal": (Normal, dict(loc=0.0, scale=1.0, name="x")),
    "Beta": (Beta, dict(alpha=2.0, beta=5.0, name="x")),
    "Gamma": (Gamma, dict(concentration=3.0, rate=1.0, name="x")),
    "InverseGamma": (InverseGamma, dict(concentration=3.0, scale=1.0, name="x")),
    "Exponential": (Exponential, dict(rate=2.0, name="x")),
    "LogNormal": (LogNormal, dict(loc=0.0, scale=1.0, name="x")),
    "StudentT": (StudentT, dict(df=5.0, loc=0.0, scale=1.0, name="x")),
    "Uniform": (Uniform, dict(low=0.0, high=1.0, name="x")),
    "Cauchy": (Cauchy, dict(loc=0.0, scale=1.0, name="x")),
    "Laplace": (Laplace, dict(loc=0.0, scale=1.0, name="x")),
    "HalfNormal": (HalfNormal, dict(scale=1.0, name="x")),
    "HalfCauchy": (HalfCauchy, dict(loc=0.0, scale=1.0, name="x")),
    "Pareto": (Pareto, dict(concentration=3.0, scale=1.0, name="x")),
    "TruncatedNormal": (
        TruncatedNormal,
        dict(loc=0.0, scale=1.0, low=-2.0, high=2.0, name="x"),
    ),
}


@pytest.fixture(params=list(_CONTINUOUS_DISTS.keys()))
def continuous_dist(request):
    """Create each continuous distribution with valid parameters."""
    cls, kwargs = _CONTINUOUS_DISTS[request.param]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Generic tests for ALL continuous distributions
# ---------------------------------------------------------------------------


class TestContinuousGeneric:
    def test_is_distribution(self, continuous_dist):
        assert isinstance(continuous_dist, ArrayDistribution)

    def test_event_shape(self, continuous_dist):
        assert isinstance(continuous_dist.event_shape, tuple)

    def test_sample_shape(self, continuous_dist, key):
        s = sample(continuous_dist, key=key, sample_shape=(5,))
        assert s.shape == (5,) + continuous_dist.event_shape

    def test_log_prob_shape(self, continuous_dist, key):
        s = sample(continuous_dist, key=key, sample_shape=(5,))
        lp = log_prob(continuous_dist, s)
        assert lp.shape == (5,)

    def test_mean_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            pytest.skip("Cauchy/HalfCauchy have no finite mean")
        m = mean(continuous_dist)
        assert jnp.all(jnp.isfinite(m))

    def test_variance_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            pytest.skip("Cauchy/HalfCauchy have no finite variance")
        if isinstance(continuous_dist, StudentT):
            # StudentT with df <= 2 has infinite variance
            if float(continuous_dist.df) <= 2.0:
                pytest.skip("StudentT with df<=2 has no finite variance")
        v = variance(continuous_dist)
        assert jnp.all(jnp.isfinite(v))

    def test_repr(self, continuous_dist):
        r = repr(continuous_dist)
        class_name = type(continuous_dist).__name__
        assert class_name in r

    def test_name(self, continuous_dist):
        assert continuous_dist.name == "x"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestBeta:
    def test_samples_in_unit_interval(self, key):
        d = Beta(alpha=2.0, beta=5.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestGammaDist:
    def test_samples_nonnegative(self, key):
        d = Gamma(concentration=3.0, rate=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestInverseGammaDist:
    def test_samples_nonnegative(self, key):
        d = InverseGamma(concentration=3.0, scale=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestExponentialDist:
    def test_samples_nonnegative(self, key):
        d = Exponential(rate=2.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestHalfNormalDist:
    def test_samples_nonnegative(self, key):
        d = HalfNormal(scale=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestHalfCauchyDist:
    def test_samples_nonnegative(self, key):
        d = HalfCauchy(loc=0.0, scale=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestParetoDist:
    def test_samples_nonnegative(self, key):
        d = Pareto(concentration=3.0, scale=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestUniformDist:
    def test_samples_in_bounds(self, key):
        d = Uniform(low=0.0, high=1.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestTruncatedNormalDist:
    def test_samples_in_bounds(self, key):
        d = TruncatedNormal(loc=0.0, scale=1.0, low=-2.0, high=2.0, name="x")
        s = sample(d, key=key, sample_shape=(1000,))
        assert jnp.all(s >= -2.0)
        assert jnp.all(s <= 2.0)


class TestNormalDist:
    def test_has_loc_and_scale(self):
        d = Normal(loc=0.0, scale=1.0, name="x")
        assert hasattr(d, "loc")
        assert hasattr(d, "scale")
        assert float(d.loc) == 0.0
        assert float(d.scale) == 1.0


# ---------------------------------------------------------------------------
# Numerical baselines — validate mean/variance against scipy.stats
# ---------------------------------------------------------------------------


# scipy equivalents of _CONTINUOUS_DISTS.  Cauchy and HalfCauchy omitted
# (mean/variance undefined).
_SCIPY_EQUIVALENTS = {
    "Normal": _scipy.norm(loc=0.0, scale=1.0),
    "Beta": _scipy.beta(a=2.0, b=5.0),
    "Gamma": _scipy.gamma(a=3.0, scale=1.0),        # scipy scale = 1 / rate
    "InverseGamma": _scipy.invgamma(a=3.0, scale=1.0),
    "Exponential": _scipy.expon(scale=0.5),         # scipy scale = 1 / rate
    "LogNormal": _scipy.lognorm(s=1.0, scale=1.0),  # s = sigma, scale = exp(loc)
    "StudentT": _scipy.t(df=5.0, loc=0.0, scale=1.0),
    "Uniform": _scipy.uniform(loc=0.0, scale=1.0),  # scipy scale = high - low
    "Laplace": _scipy.laplace(loc=0.0, scale=1.0),
    "HalfNormal": _scipy.halfnorm(scale=1.0),
    "Pareto": _scipy.pareto(b=3.0, scale=1.0),
    "TruncatedNormal": _scipy.truncnorm(a=-2.0, b=2.0, loc=0.0, scale=1.0),
}


class TestContinuousMoments:
    """Analytical mean/variance must match scipy; samples must pass KS test."""

    @pytest.mark.parametrize("name", list(_SCIPY_EQUIVALENTS))
    def test_mean_matches_scipy(self, name):
        cls, kwargs = _CONTINUOUS_DISTS[name]
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        np.testing.assert_allclose(
            float(mean(cls(**kwargs))), float(scipy_dist.mean()),
            rtol=1e-5, atol=1e-6,
        )

    @pytest.mark.parametrize("name", list(_SCIPY_EQUIVALENTS))
    def test_variance_matches_scipy(self, name):
        cls, kwargs = _CONTINUOUS_DISTS[name]
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        np.testing.assert_allclose(
            float(variance(cls(**kwargs))), float(scipy_dist.var()),
            rtol=1e-5, atol=1e-6,
        )

    @pytest.mark.parametrize("name", list(_SCIPY_EQUIVALENTS))
    def test_samples_pass_ks_test(self, name, key):
        """Two-sided KS test: samples must be consistent with the scipy CDF."""
        cls, kwargs = _CONTINUOUS_DISTS[name]
        our_dist = cls(**kwargs)
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        draws = np.asarray(sample(our_dist, key=key, sample_shape=(50_000,)))
        stat, p = _scipy.kstest(draws, scipy_dist.cdf)
        assert p > 0.001, f"KS test failed for {name}: stat={stat:.4f}, p={p:.4e}"
