"""Tests for discrete distributions in probpipe.distributions.discrete."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from probpipe.distributions import (
    Bernoulli,
    Binomial,
    Poisson,
    Categorical,
    NegativeBinomial,
)
from probpipe import NumericRecordDistribution, log_prob, mean, sample, variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(
    params=[
        pytest.param(lambda: Bernoulli(probs=0.7, name="x"), id="Bernoulli"),
        pytest.param(
            lambda: Binomial(total_count=10, probs=0.3, name="x"), id="Binomial"
        ),
        pytest.param(lambda: Poisson(rate=5.0, name="x"), id="Poisson"),
        pytest.param(
            lambda: Categorical(probs=[0.2, 0.3, 0.5], name="x"), id="Categorical"
        ),
        pytest.param(
            lambda: NegativeBinomial(total_count=5, probs=0.4, name="x"),
            id="NegativeBinomial",
        ),
    ]
)
def discrete_dist(request):
    return request.param()


# ---------------------------------------------------------------------------
# Generic tests
# ---------------------------------------------------------------------------


class TestGeneric:
    def test_is_distribution(self, discrete_dist):
        assert isinstance(discrete_dist, NumericRecordDistribution)

    def test_event_shape(self, discrete_dist):
        assert isinstance(discrete_dist.event_shape, tuple)

    def test_sample_shape(self, discrete_dist, key):
        samples = sample(discrete_dist, key=key, sample_shape=(5,))
        assert samples.shape == (5,) + discrete_dist.event_shape

    def test_log_prob_shape(self, discrete_dist, key):
        samples = sample(discrete_dist, key=key, sample_shape=(5,))
        lp = log_prob(discrete_dist, samples)
        assert lp.shape == (5,) + discrete_dist.batch_shape

    def test_repr(self, discrete_dist):
        r = repr(discrete_dist)
        assert type(discrete_dist).__name__ in r

    def test_name(self, discrete_dist):
        assert discrete_dist.name == "x"


_NAMED_DISTS = {
    "Bernoulli": lambda name: Bernoulli(probs=0.5, name=name),
    "Binomial": lambda name: Binomial(total_count=10, probs=0.3, name=name),
    "Poisson": lambda name: Poisson(rate=5.0, name=name),
    "Categorical": lambda name: Categorical(probs=[0.2, 0.3, 0.5], name=name),
    "NegativeBinomial": lambda name: NegativeBinomial(
        total_count=5, probs=0.4, name=name
    ),
}


@pytest.mark.parametrize("name", list(_NAMED_DISTS))
def test_name_set(name):
    """Every discrete distribution must store the ``name`` constructor arg."""
    dist = _NAMED_DISTS[name](name="my_dist")
    assert dist.name == "my_dist"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestBernoulli:
    def test_samples_zero_or_one(self, key):
        dist = Bernoulli(probs=0.7, name="x")
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all((samples == 0) | (samples == 1))

    def test_works_with_probs(self, key):
        dist = Bernoulli(probs=0.7, name="x")
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_works_with_logits(self, key):
        dist = Bernoulli(logits=0.0, name="x")
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_error_if_both_probs_and_logits(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Bernoulli(probs=0.5, logits=0.0, name="x")


class TestBinomial:
    def test_samples_nonneg_leq_total_count(self, key):
        dist = Binomial(total_count=10, probs=0.3, name="x")
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 10)

    def test_samples_are_integers(self, key):
        dist = Binomial(total_count=10, probs=0.3, name="x")
        samples = sample(dist, key=key, sample_shape=(100,))
        assert jnp.allclose(samples, jnp.round(samples))


class TestPoisson:
    def test_samples_nonneg_integers(self, key):
        dist = Poisson(rate=5.0, name="x")
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))


class TestCategorical:
    def test_samples_are_valid_indices(self, key):
        probs = [0.2, 0.3, 0.5]
        dist = Categorical(probs=probs, name="x")
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < len(probs))

    def test_samples_are_integers(self, key):
        dist = Categorical(probs=[0.2, 0.3, 0.5], name="x")
        samples = sample(dist, key=key, sample_shape=(100,))
        assert jnp.allclose(samples, jnp.round(samples))


class TestNegativeBinomial:
    def test_samples_nonneg_integers(self, key):
        dist = NegativeBinomial(total_count=5, probs=0.4, name="x")
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))

    def test_works_with_probs(self, key):
        dist = NegativeBinomial(total_count=5, probs=0.4, name="x")
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_works_with_logits(self, key):
        dist = NegativeBinomial(total_count=5, logits=0.0, name="x")
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_error_if_both_probs_and_logits(self):
        with pytest.raises(ValueError, match="Exactly one"):
            NegativeBinomial(total_count=5, probs=0.4, logits=0.0, name="x")


# ---------------------------------------------------------------------------
# Probs/logits validation: both given or neither given
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs_both,kwargs_neither",
    [
        pytest.param(
            Bernoulli,
            {"probs": 0.5, "logits": 0.0, "name": "x"},
            {"name": "x"},
            id="Bernoulli",
        ),
        pytest.param(
            Binomial,
            {"total_count": 10, "probs": 0.3, "logits": 0.0, "name": "x"},
            {"total_count": 10, "name": "x"},
            id="Binomial",
        ),
        pytest.param(
            Categorical,
            {"probs": [0.5, 0.5], "logits": [0.0, 0.0], "name": "x"},
            {"name": "x"},
            id="Categorical",
        ),
        pytest.param(
            NegativeBinomial,
            {"total_count": 5, "probs": 0.4, "logits": 0.0, "name": "x"},
            {"total_count": 5, "name": "x"},
            id="NegativeBinomial",
        ),
    ],
)
class TestProbsLogitsValidation:
    def test_error_both_provided(self, cls, kwargs_both, kwargs_neither):
        with pytest.raises(ValueError, match="Exactly one"):
            cls(**kwargs_both)

    def test_error_neither_provided(self, cls, kwargs_both, kwargs_neither):
        with pytest.raises(ValueError, match="Exactly one"):
            cls(**kwargs_neither)


# ---------------------------------------------------------------------------
# Numerical baselines — validate mean/variance against analytical formulas
# ---------------------------------------------------------------------------


def _chi2_discrete(observed_counts, expected_probs, min_expected=5):
    """Chi-squared goodness-of-fit for discrete samples.

    Normalizes expected_probs to sum to 1 (handles truncated PMFs),
    merges bins with expected count < ``min_expected`` into one overflow
    bin, then runs scipy.stats.chisquare.
    """
    expected_probs = expected_probs / expected_probs.sum()
    n = observed_counts.sum()
    expected = expected_probs * n
    keep = expected >= min_expected
    obs = observed_counts[keep]
    exp = expected[keep]
    if (~keep).any():
        obs = np.append(obs, observed_counts[~keep].sum())
        exp = np.append(exp, expected[~keep].sum())
    return scipy.stats.chisquare(obs, exp)


class TestDiscreteMoments:
    """Mean/variance match scipy; samples pass chi-squared goodness-of-fit."""

    def test_bernoulli_mean_and_variance(self):
        d = Bernoulli(probs=0.7, name="x")
        np.testing.assert_allclose(float(mean(d)), 0.7, rtol=1e-6)
        np.testing.assert_allclose(float(variance(d)), 0.7 * 0.3, rtol=1e-6)

    def test_bernoulli_samples_chi2(self, key):
        """Bernoulli(0.7) samples must pass a chi-squared test."""
        d = Bernoulli(probs=0.7, name="x")
        s = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        counts = np.bincount(s.astype(int), minlength=2)
        _, p = _chi2_discrete(counts, np.array([0.3, 0.7]))
        assert p > 0.001, f"chi2 failed: p={p:.4e}"

    def test_binomial_mean_and_variance(self):
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p, name="x")
        np.testing.assert_allclose(float(mean(d)), n * p, rtol=1e-6)
        np.testing.assert_allclose(float(variance(d)), n * p * (1 - p), rtol=1e-6)

    def test_binomial_samples_chi2(self, key):
        """Binomial(10, 0.3) samples must pass a chi-squared test."""
        d = Binomial(total_count=10, probs=0.3, name="x")
        s = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        counts = np.bincount(s.astype(int), minlength=11)
        expected_probs = scipy.stats.binom.pmf(np.arange(11), 10, 0.3)
        _, p = _chi2_discrete(counts, expected_probs)
        assert p > 0.001, f"chi2 failed: p={p:.4e}"

    def test_poisson_mean_and_variance(self):
        d = Poisson(rate=5.0, name="x")
        np.testing.assert_allclose(float(mean(d)), 5.0, rtol=1e-6)
        np.testing.assert_allclose(float(variance(d)), 5.0, rtol=1e-6)

    def test_poisson_samples_chi2(self, key):
        """Poisson(5) samples must pass a chi-squared test."""
        d = Poisson(rate=5.0, name="x")
        s = np.asarray(sample(d, key=key, sample_shape=(50_000,)))
        max_k = int(s.max()) + 1
        counts = np.bincount(s.astype(int), minlength=max_k)
        expected_probs = scipy.stats.poisson.pmf(np.arange(max_k), 5.0)
        _, p = _chi2_discrete(counts, expected_probs)
        assert p > 0.001, f"chi2 failed: p={p:.4e}"

    def test_poisson_log_prob_matches_scipy(self):
        """log_prob must match scipy.stats.poisson.logpmf."""
        d = Poisson(rate=5.0, name="x")
        k = jnp.array([0, 1, 5, 10])
        np.testing.assert_allclose(
            np.asarray(log_prob(d, k)),
            scipy.stats.poisson.logpmf(np.asarray(k), 5.0),
            rtol=1e-5,
        )

    def test_binomial_log_prob_matches_scipy(self):
        """log_prob must match scipy.stats.binom.logpmf."""
        d = Binomial(total_count=10, probs=0.3, name="x")
        k = jnp.array([0, 3, 5, 10])
        np.testing.assert_allclose(
            np.asarray(log_prob(d, k)),
            scipy.stats.binom.logpmf(np.asarray(k), 10, 0.3),
            rtol=1e-5,
        )
