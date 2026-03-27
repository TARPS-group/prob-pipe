from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.distributions import (
    ArrayDistribution, EmpiricalDistribution, MultivariateNormal,
    Normal, Beta, Gamma, InverseGamma, Exponential, LogNormal,
    StudentT, Uniform, Cauchy, Laplace, HalfNormal, HalfCauchy,
    Pareto, TruncatedNormal,
    Bernoulli, Binomial, Poisson, Categorical, NegativeBinomial,
    Dirichlet, Multinomial, Wishart, VonMisesFisher,
    Provenance,
)
from probpipe.core.distribution import (
    Constraint, real, positive, non_negative, non_negative_integer,
    boolean, unit_interval, simplex, positive_definite, sphere,
    interval, greater_than, integer_interval,
    _supports_compatible,
)


# ── Section 1: Constraint tests ──────────────────────────────────────────────


class TestConstraints:
    def test_real_check(self):
        assert jnp.all(real.check(jnp.array([-1.0, 0.0, 1.0])))

    def test_positive_check(self):
        assert jnp.all(positive.check(jnp.array([0.1, 1.0, 100.0])))
        assert not jnp.all(positive.check(jnp.array([-1.0, 0.0, 1.0])))

    def test_non_negative_check(self):
        assert jnp.all(non_negative.check(jnp.array([0.0, 1.0])))
        assert not jnp.all(non_negative.check(jnp.array([-1.0, 0.0])))

    def test_boolean_check(self):
        assert jnp.all(boolean.check(jnp.array([0.0, 1.0])))
        assert not jnp.all(boolean.check(jnp.array([0.0, 0.5])))

    def test_unit_interval_check(self):
        assert jnp.all(unit_interval.check(jnp.array([0.0, 0.5, 1.0])))
        assert not jnp.all(unit_interval.check(jnp.array([-0.1, 0.5])))

    def test_interval_check(self):
        c = interval(-2.0, 3.0)
        assert jnp.all(c.check(jnp.array([-2.0, 0.0, 3.0])))
        assert not jnp.all(c.check(jnp.array([-3.0, 0.0])))

    def test_greater_than_check(self):
        c = greater_than(5.0)
        assert jnp.all(c.check(jnp.array([5.1, 10.0])))
        assert not jnp.all(c.check(jnp.array([4.9, 10.0])))

    def test_simplex_check(self):
        assert simplex.check(jnp.array([0.3, 0.3, 0.4]))
        assert not simplex.check(jnp.array([0.5, 0.5, 0.5]))

    def test_constraint_equality(self):
        assert real == real
        assert interval(0, 1) == interval(0, 1)
        assert interval(0, 1) != interval(0, 2)
        assert real != positive

    def test_constraint_repr(self):
        assert repr(real) == "real"
        assert repr(positive) == "positive"
        assert "interval" in repr(interval(0, 1))


# ── Section 2: Support compatibility tests ────────────────────────────────────


class TestSupportCompatibility:
    def test_identical_supports(self):
        assert _supports_compatible(real, real)
        assert _supports_compatible(positive, positive)

    def test_subset_relations(self):
        assert _supports_compatible(positive, real)
        assert _supports_compatible(unit_interval, real)
        assert _supports_compatible(boolean, real)
        assert _supports_compatible(non_negative, real)

    def test_incompatible(self):
        assert not _supports_compatible(real, positive)
        assert not _supports_compatible(real, unit_interval)

    def test_interval_subset(self):
        assert _supports_compatible(interval(0, 1), interval(-1, 2))
        assert not _supports_compatible(interval(-1, 2), interval(0, 1))


# ── Section 3: Support properties on distributions ────────────────────────────


class TestDistributionSupport:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_normal_support(self):
        assert Normal(0.0, 1.0).support == real

    def test_beta_support(self):
        assert Beta(2.0, 5.0).support == unit_interval

    def test_gamma_support(self):
        assert Gamma(3.0, 1.0).support == positive

    def test_uniform_support(self):
        assert Uniform(low=-1.0, high=2.0).support == interval(-1.0, 2.0)

    def test_bernoulli_support(self):
        assert Bernoulli(probs=0.5).support == boolean

    def test_poisson_support(self):
        assert Poisson(rate=3.0).support == non_negative_integer

    def test_dirichlet_support(self):
        assert Dirichlet([1.0, 2.0]).support == simplex

    def test_wishart_support(self):
        assert Wishart(df=5.0, scale_tril=jnp.eye(3)).support == positive_definite

    def test_vonmisesfisher_support(self):
        assert VonMisesFisher([1.0, 0.0, 0.0], 5.0).support == sphere

    def test_mvn_support(self):
        assert MultivariateNormal(jnp.zeros(2), cov=jnp.eye(2)).support == real

    def test_empirical_support(self):
        ed = EmpiricalDistribution(jnp.ones((5, 2)))
        assert ed.support == real


# ── Section 4: from_distribution tests ────────────────────────────────────────


class TestFromDistribution:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    # -- same-class copy --
    def test_normal_from_normal(self, key):
        n = Normal(loc=3.0, scale=2.0)
        n2 = Normal.from_distribution(n, key=key)
        assert jnp.isclose(n2.loc, 3.0, atol=0.01)

    def test_beta_from_beta(self, key):
        b = Beta(alpha=2.0, beta=5.0)
        b2 = Beta.from_distribution(b, key=key)
        assert jnp.isclose(b2.alpha, 2.0, atol=0.01)

    # -- moment-matching --
    def test_normal_from_gamma(self, key):
        """Gamma -> Normal via moment matching (check_support=False needed)."""
        g = Gamma(concentration=9.0, rate=1.0)
        n = Normal.from_distribution(g, key=key, check_support=False, num_samples=5000)
        # Gamma(9,1) has mean=9, var=9
        assert jnp.isclose(n.loc, 9.0, atol=1.0)

    def test_gamma_from_normal(self, key):
        """Normal -> Gamma should fail support check by default."""
        n = Normal(loc=5.0, scale=1.0)
        with pytest.raises(ValueError, match="support"):
            Gamma.from_distribution(n, key=key)

    def test_gamma_from_normal_override(self, key):
        """Normal -> Gamma with check_support=False should work."""
        n = Normal(loc=5.0, scale=1.0)
        g = Gamma.from_distribution(n, key=key, check_support=False, num_samples=5000)
        assert jnp.isclose(float(g.concentration * 1.0 / g.rate), 5.0, atol=1.0)

    def test_beta_from_uniform(self, key):
        """Uniform(0,1) -> Beta should work (compatible support)."""
        u = Uniform(low=0.0, high=1.0)
        b = Beta.from_distribution(u, key=key, num_samples=5000)
        # Uniform(0,1) has mean=0.5, var=1/12 -> alpha~=beta~=1
        assert float(b.alpha) > 0
        assert float(b.beta) > 0

    # -- discrete --
    def test_bernoulli_from_bernoulli(self, key):
        b = Bernoulli(probs=0.7)
        b2 = Bernoulli.from_distribution(b, key=key)
        assert jnp.isclose(b2.probs, 0.7, atol=0.01)

    def test_poisson_from_poisson(self, key):
        p = Poisson(rate=5.0)
        p2 = Poisson.from_distribution(p, key=key)
        assert jnp.isclose(p2.rate, 5.0, atol=0.01)

    def test_binomial_requires_total_count(self, key):
        """Binomial.from_distribution from non-Binomial needs total_count."""
        p = Poisson(rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            Binomial.from_distribution(p, key=key, check_support=False)

    def test_binomial_from_poisson(self, key):
        p = Poisson(rate=3.0)
        b = Binomial.from_distribution(p, key=key, check_support=False, total_count=10, num_samples=5000)
        # mean ~ 3, so probs ~ 0.3
        assert b.probs is not None

    # -- multivariate --
    def test_mvn_from_empirical(self, key):
        samples = jax.random.normal(key, (100, 3))
        ed = EmpiricalDistribution(samples)
        mvn = MultivariateNormal.from_distribution(ed)
        assert mvn.dim == 3

    def test_dirichlet_from_dirichlet(self, key):
        d = Dirichlet(concentration=jnp.array([1.0, 2.0, 3.0]))
        d2 = Dirichlet.from_distribution(d, key=key)
        assert jnp.allclose(d2.concentration, d.concentration)

    # -- provenance --
    def test_from_distribution_same_class_returns_source(self, key):
        """Same-class conversion returns the source object (no-op)."""
        n = Normal(loc=0.0, scale=1.0)
        n2 = Normal.from_distribution(n, key=key)
        assert n2 is n

    def test_from_distribution_cross_class_provenance(self, key):
        """Cross-class conversion attaches provenance."""
        g = Gamma(concentration=3.0, rate=1.0)
        n = Normal.from_distribution(g, key=key, check_support=False)
        assert n.source is not None
        assert n.source.operation == "from_distribution"

    # -- empirical from anything --
    def test_empirical_from_normal(self, key):
        n = Normal(loc=0.0, scale=1.0)
        ed = EmpiricalDistribution.from_distribution(n, key=key, num_samples=100)
        assert ed.n == 100
