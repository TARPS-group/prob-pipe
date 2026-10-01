"""Tests for continuous univariate distributions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as _scipy

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
from probpipe.distributions._capabilities import _capability_guard
from probpipe.families import (
    Beta,
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    StudentT,
    TruncatedNormal,
    Uniform,
)
from tests._ops import mean as modeled_mean
from tests._ops import variance as modeled_variance

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
        assert isinstance(continuous_dist, TFPDistribution)
        assert isinstance(continuous_dist, NumericDistribution)

    def test_event_shape(self, continuous_dist):
        assert isinstance(continuous_dist.event_shape, tuple)

    def test_sample_shape(self, continuous_dist, key):
        s = sample(continuous_dist, key=key, sample_shape=(5,))
        assert s.shape == (5, *continuous_dist.event_shape)

    def test_log_prob_shape(self, continuous_dist, key):
        s = sample(continuous_dist, key=key, sample_shape=(5,))
        lp = log_prob(continuous_dist, s)
        assert lp.shape == (5,)

    def test_mean_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            with pytest.raises(MathematicalDomainError, match="mean"):
                mean(continuous_dist)
            return
        m = mean(continuous_dist)
        assert jnp.all(jnp.isfinite(m))

    def test_variance_finite(self, continuous_dist):
        if isinstance(continuous_dist, (Cauchy, HalfCauchy)):
            with pytest.raises(MathematicalDomainError, match="variance"):
                variance(continuous_dist)
            return
        v = variance(continuous_dist)
        assert jnp.all(jnp.isfinite(v))

    def test_repr(self, continuous_dist):
        r = repr(continuous_dist)
        class_name = type(continuous_dist).__name__
        assert class_name in r

    def test_name(self, continuous_dist):
        assert continuous_dist.name == "x"


# ---------------------------------------------------------------------------
# Moments that do not exist
# ---------------------------------------------------------------------------

#: A law with a moment known not to exist, and that moment.
_NONEXISTENT = {
    "cauchy-mean": (lambda: Cauchy("x", 0.0, 1.0), "mean"),
    "cauchy-variance": (lambda: Cauchy("x", 0.0, 1.0), "variance"),
    "half-cauchy-mean": (lambda: HalfCauchy("x", 0.0, 1.0), "mean"),
    "half-cauchy-variance": (lambda: HalfCauchy("x", 0.0, 1.0), "variance"),
    "student-t-mean-at-df-1": (lambda: StudentT("x", 1.0, 0.0, 1.0), "mean"),
    "student-t-variance-at-df-0.5": (lambda: StudentT("x", 0.5, 0.0, 1.0), "variance"),
    "student-t-variance-at-df-1.5": (lambda: StudentT("x", 1.5, 0.0, 1.0), "variance"),
    "student-t-variance-at-df-2": (lambda: StudentT("x", 2.0, 0.0, 1.0), "variance"),
    "student-t-mean-of-one-coordinate": (
        lambda: StudentT("x", jnp.array([0.5, 3.0]), 0.0, 1.0),
        "mean",
    ),
    "inverse-gamma-mean-at-1": (lambda: InverseGamma("x", 1.0, 1.0), "mean"),
    "inverse-gamma-variance-at-2": (lambda: InverseGamma("x", 2.0, 1.0), "variance"),
    "pareto-mean-at-1": (lambda: Pareto("x", 1.0, 1.0), "mean"),
    "pareto-variance-at-2": (lambda: Pareto("x", 2.0, 1.0), "variance"),
}

#: The exported operation, the operation model's, and the capability of each moment.
_MOMENTS = {
    "mean": (mean, modeled_mean, "_mean"),
    "variance": (variance, modeled_variance, "_variance"),
}


class TestMomentsThatDoNotExist:
    """A moment known not to exist raises ``MathematicalDomainError`` (II.7)."""

    @pytest.mark.parametrize("case", list(_NONEXISTENT))
    def test_the_moment_raises(self, case):
        make, moment = _NONEXISTENT[case]
        exported, modeled, capability = _MOMENTS[moment]
        for compute in (exported, modeled, lambda law: getattr(law, capability)()):
            with pytest.raises(MathematicalDomainError, match=moment):
                compute(make())

    @pytest.mark.parametrize(
        "case", [case for case, (_, moment) in _NONEXISTENT.items() if moment == "variance"]
    )
    def test_the_covariance_raises_where_the_variance_does(self, case):
        make, _ = _NONEXISTENT[case]
        with pytest.raises(MathematicalDomainError, match="variance"):
            make()._cov()
        with pytest.raises(MathematicalDomainError, match="variance"):
            cov(make())

    def test_the_operation_does_not_estimate_the_moment_by_sampling(self):
        law = Cauchy("x", 0.0, 1.0)
        object.__setattr__(law, "_sample", lambda *args, **kwargs: pytest.fail("sampled"))
        with pytest.raises(MathematicalDomainError, match="mean"):
            modeled_mean(law)

    @pytest.mark.parametrize(
        ("make", "expected_mean", "expected_variance"),
        [
            (lambda: StudentT("x", 3.0, 2.0, 2.0), 2.0, 12.0),
            (lambda: InverseGamma("x", 3.0, 2.0), 1.0, 1.0),
            (lambda: Pareto("x", 3.0, 2.0), 3.0, 3.0),
        ],
        ids=["student-t", "inverse-gamma", "pareto"],
    )
    def test_the_moments_that_exist_are_computed(self, make, expected_mean, expected_variance):
        assert float(mean(make())) == pytest.approx(expected_mean, rel=1e-5)
        assert float(variance(make())) == pytest.approx(expected_variance, rel=1e-5)

    def test_a_student_t_has_a_mean_where_its_variance_does_not_exist(self):
        law = StudentT("x", 1.5, 2.0, 1.0)
        assert float(mean(law)) == pytest.approx(2.0)
        with pytest.raises(MathematicalDomainError, match="variance"):
            variance(law)

    def test_a_traced_parameter_leaves_existence_to_the_computation(self):
        reports = []

        def mean_of(df):
            law = StudentT("x", df, 0.0, 1.0)
            reports.append(_capability_guard(law, "_mean"))
            return law._mean()

        assert float(jax.jit(mean_of)(3.0)) == 0.0
        assert reports[0].feasible is None
        assert "needs values not yet known" in reports[0].pending[0]


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestBeta:
    def test_samples_in_unit_interval(self, key):
        d = Beta(alpha=2.0, beta=5.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestGammaDist:
    def test_samples_nonnegative(self, key):
        d = Gamma(concentration=3.0, rate=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestInverseGammaDist:
    def test_samples_nonnegative(self, key):
        d = InverseGamma(concentration=3.0, scale=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestExponentialDist:
    def test_samples_nonnegative(self, key):
        d = Exponential(rate=2.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestHalfNormalDist:
    def test_samples_nonnegative(self, key):
        d = HalfNormal(scale=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestHalfCauchyDist:
    def test_samples_nonnegative(self, key):
        d = HalfCauchy(loc=0.0, scale=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestParetoDist:
    def test_samples_nonnegative(self, key):
        d = Pareto(concentration=3.0, scale=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)


class TestUniformDist:
    def test_samples_in_bounds(self, key):
        d = Uniform(low=0.0, high=1.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
        assert jnp.all(s >= 0.0)
        assert jnp.all(s <= 1.0)


class TestTruncatedNormalDist:
    def test_samples_in_bounds(self, key):
        d = TruncatedNormal(loc=0.0, scale=1.0, low=-2.0, high=2.0, name="x")
        s = jnp.asarray(sample(d, key=key, sample_shape=(1000,)))
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
    "Gamma": _scipy.gamma(a=3.0, scale=1.0),  # scipy scale = 1 / rate
    "InverseGamma": _scipy.invgamma(a=3.0, scale=1.0),
    "Exponential": _scipy.expon(scale=0.5),  # scipy scale = 1 / rate
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
            float(mean(cls(**kwargs))),
            float(scipy_dist.mean()),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("name", list(_SCIPY_EQUIVALENTS))
    def test_variance_matches_scipy(self, name):
        cls, kwargs = _CONTINUOUS_DISTS[name]
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        np.testing.assert_allclose(
            float(variance(cls(**kwargs))),
            float(scipy_dist.var()),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("name", list(_SCIPY_EQUIVALENTS))
    def test_samples_pass_ks_test(self, name, key):
        """Two-sided KS test: samples must be consistent with the scipy CDF."""
        cls, kwargs = _CONTINUOUS_DISTS[name]
        our_dist = cls(**kwargs)
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        draws = np.asarray(sample(our_dist, key=key, sample_shape=(50_000,)))
        stat, p = _scipy.kstest(draws, scipy_dist.cdf)
        # Project-wide goodness-of-fit threshold: p > 0.001 (~3σ).
        # Strict enough to catch bugs, loose enough to avoid xdist flakes.
        assert p > 0.001, f"KS test failed for {name}: stat={stat:.4f}, p={p:.4e}"


# ---------------------------------------------------------------------------
# prob() op (exp(log_prob)) — scipy baseline
# ---------------------------------------------------------------------------


from probpipe import prob


class TestProb:
    """prob(dist, x) == exp(log_prob) and matches scipy.stats.X.pdf."""

    @pytest.mark.parametrize("name", ["Normal", "Beta", "Gamma", "Exponential"])
    def test_prob_matches_scipy_pdf(self, name, key):
        cls, kwargs = _CONTINUOUS_DISTS[name]
        scipy_dist = _SCIPY_EQUIVALENTS[name]
        # Evaluate at a few points inside the support
        if name == "Beta":
            xs = jnp.array([0.1, 0.5, 0.9])
        elif name in ("Gamma", "Exponential"):
            xs = jnp.array([0.5, 1.0, 2.0])
        else:
            xs = jnp.array([-1.0, 0.0, 1.0])
        our_dist = cls(**kwargs)
        ours = np.asarray(prob(our_dist, xs))
        expected = scipy_dist.pdf(np.asarray(xs))
        np.testing.assert_allclose(ours, expected, rtol=1e-4)

    def test_prob_equals_exp_logprob(self, key):
        """prob(dist, x) must equal exp(log_prob(dist, x))."""
        from probpipe import log_prob as log_prob_op

        d = Normal(loc=0.0, scale=1.0, name="x")
        xs = jnp.array([-1.0, 0.5, 1.2])
        np.testing.assert_allclose(
            np.asarray(prob(d, xs)),
            np.asarray(jnp.exp(log_prob_op(d, xs))),
            rtol=1e-6,
        )
