"""Tests for probpipe.distributions.multivariate."""

import jax
import jax.numpy as jnp
import pytest

from probpipe.distributions import (
    Dirichlet,
    Multinomial,
    Wishart,
    VonMisesFisher,
)
from probpipe.core.distribution import ArrayDistribution
from probpipe import log_prob, mean, sample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture(
    params=[
        pytest.param(
            lambda: Dirichlet(concentration=[1.0, 2.0, 3.0]),
            id="Dirichlet",
        ),
        pytest.param(
            lambda: Multinomial(total_count=10, probs=[0.2, 0.3, 0.5]),
            id="Multinomial",
        ),
        pytest.param(
            lambda: Wishart(df=5.0, scale_tril=jnp.eye(3)),
            id="Wishart",
        ),
        pytest.param(
            lambda: VonMisesFisher(
                mean_direction=[1.0, 0.0, 0.0], concentration=5.0
            ),
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
        assert isinstance(multivariate_dist, ArrayDistribution)

    def test_event_shape(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        expected = EXPECTED_EVENT_SHAPES[name]
        assert multivariate_dist.event_shape == expected

    def test_sample_shape(self, multivariate_dist, key):
        samples = sample(multivariate_dist, key=key, sample_shape=(5,))
        expected = (5,) + multivariate_dist.event_shape
        assert samples.shape == expected

    def test_log_prob_shape(self, multivariate_dist, key):
        s = sample(multivariate_dist, key=key)
        lp = log_prob(multivariate_dist, s)
        assert lp.shape == ()

    def test_mean_finite(self, multivariate_dist):
        m = mean(multivariate_dist)
        assert jnp.all(jnp.isfinite(m))

    def test_repr(self, multivariate_dist):
        name = type(multivariate_dist).__name__
        assert name in repr(multivariate_dist)

    def test_name_none_by_default(self, multivariate_dist):
        assert multivariate_dist.name is None

    def test_name_set(self):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0], name="alpha")
        assert d.name == "alpha"

        m = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5], name="counts")
        assert m.name == "counts"

        w = Wishart(df=5.0, scale_tril=jnp.eye(3), name="sigma")
        assert w.name == "sigma"

        v = VonMisesFisher(
            mean_direction=[1.0, 0.0, 0.0], concentration=5.0, name="dir"
        )
        assert v.name == "dir"


# ---------------------------------------------------------------------------
# Distribution-specific tests
# ---------------------------------------------------------------------------


class TestDirichlet:
    def test_samples_sum_to_one(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0])
        samples = sample(d, key=key, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_samples_positive(self, key):
        d = Dirichlet(concentration=[1.0, 2.0, 3.0])
        samples = sample(d, key=key, sample_shape=(100,))
        assert jnp.all(samples > 0)


class TestMultinomial:
    def test_samples_nonnegative_integers(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5])
        samples = sample(d, key=key, sample_shape=(100,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.round(samples))

    def test_samples_sum_to_total_count(self, key):
        d = Multinomial(total_count=10, probs=[0.2, 0.3, 0.5])
        samples = sample(d, key=key, sample_shape=(100,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 10.0)

    def test_probs_logits_validation(self):
        # Must provide exactly one of probs or logits.
        with pytest.raises(ValueError, match="Exactly one"):
            Multinomial(total_count=10)

        with pytest.raises(ValueError, match="Exactly one"):
            Multinomial(
                total_count=10,
                probs=[0.2, 0.3, 0.5],
                logits=[0.0, 0.0, 0.0],
            )

        # Using logits instead of probs should work.
        d = Multinomial(total_count=10, logits=[0.0, 0.0, 0.0])
        assert d.logits is not None
        assert d.probs is None


class TestWishart:
    def test_accepts_scale_tril(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3))
        s = sample(d, key=key)
        assert s.shape == (3, 3)

    def test_accepts_scale(self, key):
        d = Wishart(df=5.0, scale=jnp.eye(3))
        s = sample(d, key=key)
        assert s.shape == (3, 3)

    def test_error_if_both_given(self):
        with pytest.raises(ValueError, match="exactly one"):
            Wishart(df=5.0, scale_tril=jnp.eye(3), scale=jnp.eye(3))

    def test_samples_positive_semi_definite(self, key):
        d = Wishart(df=5.0, scale_tril=jnp.eye(3))
        samples = sample(d, key=key, sample_shape=(10,))
        # Diagonal elements of a positive semi-definite matrix are >= 0.
        for i in range(10):
            diag = jnp.diag(samples[i])
            assert jnp.all(diag >= 0)


class TestVonMisesFisher:
    def test_samples_unit_norm(self, key):
        d = VonMisesFisher(mean_direction=[1.0, 0.0, 0.0], concentration=5.0)
        samples = sample(d, key=key, sample_shape=(100,))
        norms = jnp.linalg.norm(samples, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5)
