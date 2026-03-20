"""Tests for continuous univariate distributions."""

import jax
import jax.numpy as jnp
import pytest

from probpipe.distributions import (
    Distribution,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# Map of (class, kwargs) for every continuous distribution under test.
_CONTINUOUS_DISTS = {
    "Normal": (Normal, dict(loc=0.0, scale=1.0)),
    "Beta": (Beta, dict(alpha=2.0, beta=5.0)),
    "Gamma": (Gamma, dict(concentration=3.0, rate=1.0)),
    "InverseGamma": (InverseGamma, dict(concentration=3.0, scale=1.0)),
    "Exponential": (Exponential, dict(rate=2.0)),
    "LogNormal": (LogNormal, dict(loc=0.0, scale=1.0)),
    "StudentT": (StudentT, dict(df=5.0, loc=0.0, scale=1.0)),
    "Uniform": (Uniform, dict(low=0.0, high=1.0)),
    "Cauchy": (Cauchy, dict(loc=0.0, scale=1.0)),
    "Laplace": (Laplace, dict(loc=0.0, scale=1.0)),
    "HalfNormal": (HalfNormal, dict(scale=1.0)),
    "HalfCauchy": (HalfCauchy, dict(loc=0.0, scale=1.0)),
    "Pareto": (Pareto, dict(concentration=3.0, scale=1.0)),
    "TruncatedNormal": (
        TruncatedNormal,
        dict(loc=0.0, scale=1.0, low=-2.0, high=2.0),
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
        assert isinstance(continuous_dist, Distribution)

    def test_event_shape(self, continuous_dist):
        assert isinstance(continuous_dist.event_shape, tuple)

    def test_sample_shape(self, continuous_dist, key):
        s = continuous_dist.sample(key, sample_shape=(5,))
        assert s.shape == (5,) + continuous_dist.event_shape

    def test_log_prob_shape(self, continuous_dist, key):
        s = continuous_dist.sample(key, sample_shape=(5,))
        lp = continuous_dist.log_prob(s)
        assert lp.shape == (5,)

    def test_mean_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            pytest.skip("Cauchy/HalfCauchy have no finite mean")
        m = continuous_dist.mean()
        assert jnp.all(jnp.isfinite(m))

    def test_variance_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            pytest.skip("Cauchy/HalfCauchy have no finite variance")
        if isinstance(continuous_dist, StudentT):
            # StudentT with df <= 2 has infinite variance
            if float(continuous_dist.df) <= 2.0:
                pytest.skip("StudentT with df<=2 has no finite variance")
        v = continuous_dist.variance()
        assert jnp.all(jnp.isfinite(v))

    def test_repr(self, continuous_dist):
        r = repr(continuous_dist)
        class_name = type(continuous_dist).__name__
        assert class_name in r

    def test_name_none_by_default(self, continuous_dist):
        assert continuous_dist.name is None

    def test_name_set(self):
        """Every distribution should accept and store a name."""
        for cls, kwargs in _CONTINUOUS_DISTS.values():
            d = cls(**kwargs, name="test")
            assert d.name == "test"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestBeta:
    def test_samples_in_unit_interval(self, key):
        d = Beta(alpha=2.0, beta=5.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestGammaDist:
    def test_samples_nonnegative(self, key):
        d = Gamma(concentration=3.0, rate=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestInverseGammaDist:
    def test_samples_nonnegative(self, key):
        d = InverseGamma(concentration=3.0, scale=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestExponentialDist:
    def test_samples_nonnegative(self, key):
        d = Exponential(rate=2.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestHalfNormalDist:
    def test_samples_nonnegative(self, key):
        d = HalfNormal(scale=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestHalfCauchyDist:
    def test_samples_nonnegative(self, key):
        d = HalfCauchy(loc=0.0, scale=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestParetoDist:
    def test_samples_nonnegative(self, key):
        d = Pareto(concentration=3.0, scale=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)


class TestUniformDist:
    def test_samples_in_bounds(self, key):
        d = Uniform(low=0.0, high=1.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestTruncatedNormalDist:
    def test_samples_in_bounds(self, key):
        d = TruncatedNormal(loc=0.0, scale=1.0, low=-2.0, high=2.0)
        s = d.sample(key, sample_shape=(1000,))
        assert jnp.all(s >= -2.0)
        assert jnp.all(s <= 2.0)


class TestNormalDist:
    def test_has_loc_and_scale(self):
        d = Normal(loc=0.0, scale=1.0)
        assert hasattr(d, "loc")
        assert hasattr(d, "scale")
        assert float(d.loc) == 0.0
        assert float(d.scale) == 1.0
