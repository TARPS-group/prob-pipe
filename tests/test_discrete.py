"""Tests for discrete distributions in probpipe.distributions.discrete."""

import jax
import jax.numpy as jnp
import pytest

from probpipe.distributions import (
    Bernoulli,
    Binomial,
    Poisson,
    Categorical,
    NegativeBinomial,
)
from probpipe import ArrayDistribution, log_prob, sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(
    params=[
        pytest.param(lambda: Bernoulli(probs=0.7), id="Bernoulli"),
        pytest.param(lambda: Binomial(total_count=10, probs=0.3), id="Binomial"),
        pytest.param(lambda: Poisson(rate=5.0), id="Poisson"),
        pytest.param(lambda: Categorical(probs=[0.2, 0.3, 0.5]), id="Categorical"),
        pytest.param(
            lambda: NegativeBinomial(total_count=5, probs=0.4),
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
        assert isinstance(discrete_dist, ArrayDistribution)

    def test_event_shape(self, discrete_dist):
        assert isinstance(discrete_dist.event_shape, tuple)

    def test_sample_shape(self, discrete_dist, key):
        samples = sample(discrete_dist, key=key, sample_shape=(5,))
        assert samples.shape[:1] == (5,)

    def test_log_prob_shape(self, discrete_dist, key):
        samples = sample(discrete_dist, key=key, sample_shape=(5,))
        lp = log_prob(discrete_dist, samples)
        assert lp.shape == (5,) + discrete_dist.batch_shape

    def test_repr(self, discrete_dist):
        r = repr(discrete_dist)
        assert type(discrete_dist).__name__ in r

    def test_name_none_by_default(self, discrete_dist):
        assert discrete_dist.name is None

    def test_name_set(self):
        dist = Bernoulli(probs=0.5, name="my_bernoulli")
        assert dist.name == "my_bernoulli"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestBernoulli:
    def test_samples_zero_or_one(self, key):
        dist = Bernoulli(probs=0.7)
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all((samples == 0) | (samples == 1))

    def test_works_with_probs(self, key):
        dist = Bernoulli(probs=0.7)
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_works_with_logits(self, key):
        dist = Bernoulli(logits=0.0)
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_error_if_both_probs_and_logits(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Bernoulli(probs=0.5, logits=0.0)


class TestBinomial:
    def test_samples_nonneg_leq_total_count(self, key):
        dist = Binomial(total_count=10, probs=0.3)
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 10)

    def test_samples_are_integers(self, key):
        dist = Binomial(total_count=10, probs=0.3)
        samples = sample(dist, key=key, sample_shape=(100,))
        assert jnp.allclose(samples, jnp.round(samples))


class TestPoisson:
    def test_samples_nonneg_integers(self, key):
        dist = Poisson(rate=5.0)
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))


class TestCategorical:
    def test_samples_are_valid_indices(self, key):
        probs = [0.2, 0.3, 0.5]
        dist = Categorical(probs=probs)
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples < len(probs))

    def test_samples_are_integers(self, key):
        dist = Categorical(probs=[0.2, 0.3, 0.5])
        samples = sample(dist, key=key, sample_shape=(100,))
        assert jnp.allclose(samples, jnp.round(samples))


class TestNegativeBinomial:
    def test_samples_nonneg_integers(self, key):
        dist = NegativeBinomial(total_count=5, probs=0.4)
        samples = sample(dist, key=key, sample_shape=(1000,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))

    def test_works_with_probs(self, key):
        dist = NegativeBinomial(total_count=5, probs=0.4)
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_works_with_logits(self, key):
        dist = NegativeBinomial(total_count=5, logits=0.0)
        samples = sample(dist, key=key, sample_shape=(10,))
        assert samples.shape == (10,)

    def test_error_if_both_probs_and_logits(self):
        with pytest.raises(ValueError, match="Exactly one"):
            NegativeBinomial(total_count=5, probs=0.4, logits=0.0)


# ---------------------------------------------------------------------------
# Probs/logits validation: both given or neither given
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs_both,kwargs_neither",
    [
        pytest.param(
            Bernoulli,
            {"probs": 0.5, "logits": 0.0},
            {},
            id="Bernoulli",
        ),
        pytest.param(
            Binomial,
            {"total_count": 10, "probs": 0.3, "logits": 0.0},
            {"total_count": 10},
            id="Binomial",
        ),
        pytest.param(
            Categorical,
            {"probs": [0.5, 0.5], "logits": [0.0, 0.0]},
            {},
            id="Categorical",
        ),
        pytest.param(
            NegativeBinomial,
            {"total_count": 5, "probs": 0.4, "logits": 0.0},
            {"total_count": 5},
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
