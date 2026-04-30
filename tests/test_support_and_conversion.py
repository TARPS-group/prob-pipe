from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import from_distribution
from probpipe import (
    NumericRecordDistribution, NumericEmpiricalDistribution, EmpiricalDistribution,
    Provenance,
)
from probpipe.distributions import (
    MultivariateNormal,
    Normal, Beta, Gamma, InverseGamma, Exponential, LogNormal,
    StudentT, Uniform, Cauchy, Laplace, HalfNormal, HalfCauchy,
    Pareto, TruncatedNormal,
    Bernoulli, Binomial, Poisson, Categorical, NegativeBinomial,
    Dirichlet, Multinomial, Wishart, VonMisesFisher,
)
from probpipe.core.constraints import (
    Constraint, real, positive, non_negative, non_negative_integer,
    boolean, unit_interval, simplex, positive_definite, sphere,
    interval, greater_than, integer_interval,
    _supports_compatible,
)


# ── Section 1: Constraint tests ──────────────────────────────────────────────


class TestConstraints:
    def test_real_check(self):
        assert jnp.all(real.check(jnp.array([-1.0, 0.0, 1.0])))

    def test_real_check_extreme_values(self):
        """real accepts any finite float, including extreme magnitudes."""
        assert jnp.all(real.check(jnp.array([-1e30, -1.0, 0.0, 1.0, 1e30])))

    def test_real_check_rejects_nonfinite(self):
        """NaN / inf are outside the real support (finite floats)."""
        # Real constraint on NaN/inf: must return False per row.
        assert not bool(real.check(jnp.asarray(float("nan"))))
        assert not bool(real.check(jnp.asarray(float("inf"))))
        assert not bool(real.check(jnp.asarray(float("-inf"))))

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

    def test_parameterized_equality_array_bounds(self):
        # __eq__ on parameterized constraints must use jnp.array_equal
        # so it doesn't crash on multi-element JAX-array bounds.
        a = interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5]))
        b = interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5]))
        c = interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 2.0]))
        assert a == b
        assert a != c
        assert greater_than(jnp.array([1.0, 2.0])) == greater_than(jnp.array([1.0, 2.0]))
        assert greater_than(jnp.array([1.0, 2.0])) != greater_than(jnp.array([1.0, 3.0]))
        assert integer_interval(jnp.array([0, 1]), jnp.array([5, 6])) == integer_interval(
            jnp.array([0, 1]), jnp.array([5, 6])
        )

    def test_parameterized_hash_array_bounds(self):
        # __hash__ on parameterized constraints must not raise for
        # array-valued bounds (0-d or higher).
        hash(interval(0.0, 1.0))
        hash(interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5])))
        hash(greater_than(jnp.array([1.0, 2.0])))
        hash(integer_interval(jnp.array([0, 1]), jnp.array([5, 6])))

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

    def test_interval_subset_array_bounds(self):
        # Per-dim bounds: each source dim must lie within the
        # corresponding target dim.
        src = interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5]))
        tgt_super = interval(jnp.array([-1.0, 0.0]), jnp.array([2.0, 2.0]))
        tgt_partial = interval(jnp.array([-1.0, 1.0]), jnp.array([2.0, 2.0]))
        assert _supports_compatible(src, tgt_super)
        # src dim 1 starts at 0.5, which is below tgt_partial dim 1's 1.0.
        assert not _supports_compatible(src, tgt_partial)

    def test_greater_than_subset_array_bounds(self):
        src = greater_than(jnp.array([1.0, 2.0]))
        tgt_super = greater_than(jnp.array([0.0, 1.0]))
        tgt_partial = greater_than(jnp.array([0.0, 3.0]))
        assert _supports_compatible(src, tgt_super)
        assert not _supports_compatible(src, tgt_partial)

    def test_integer_interval_subset_array_bounds(self):
        src = integer_interval(jnp.array([1, 2]), jnp.array([5, 6]))
        tgt_super = integer_interval(jnp.array([0, 1]), jnp.array([10, 10]))
        tgt_partial = integer_interval(jnp.array([0, 3]), jnp.array([10, 10]))
        assert _supports_compatible(src, tgt_super)
        # src dim 1 starts at 2, which is below tgt_partial dim 1's 3.
        assert not _supports_compatible(src, tgt_partial)


# ── Section 3: Support properties on distributions ────────────────────────────


class TestDistributionSupport:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    def test_normal_support(self):
        assert Normal(0.0, 1.0, name="x").support == real

    def test_beta_support(self):
        assert Beta(2.0, 5.0, name="b").support == unit_interval

    def test_gamma_support(self):
        assert Gamma(3.0, 1.0, name="g").support == positive

    def test_uniform_support(self):
        assert Uniform(low=-1.0, high=2.0, name="u").support == interval(-1.0, 2.0)

    def test_uniform_support_array_bounds(self):
        # Regression: ``float(self._low)`` previously crashed for
        # non-scalar low/high. Per-dim bounds should produce a working
        # interval whose ``.check`` returns a per-dim boolean.
        u = Uniform(
            low=jnp.array([0.0, -1.0]),
            high=jnp.array([1.0, 2.0]),
            name="u_arr",
        )
        c = u.support
        assert jnp.array_equal(c.check(jnp.array([0.5, 0.5])), jnp.array([True, True]))
        assert jnp.array_equal(c.check(jnp.array([2.0, 0.5])), jnp.array([False, True]))

    def test_half_cauchy_support_array_bounds(self):
        hc = HalfCauchy(
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 1.0]),
            name="hc_arr",
        )
        c = hc.support
        assert jnp.array_equal(c.check(jnp.array([0.5, 1.5])), jnp.array([True, True]))
        assert jnp.array_equal(c.check(jnp.array([-0.5, 0.5])), jnp.array([False, False]))

    def test_pareto_support_array_bounds(self):
        p = Pareto(
            concentration=jnp.array([2.0, 2.0]),
            scale=jnp.array([1.0, 2.0]),
            name="p_arr",
        )
        c = p.support
        assert jnp.array_equal(c.check(jnp.array([1.5, 2.5])), jnp.array([True, True]))
        assert jnp.array_equal(c.check(jnp.array([0.5, 2.5])), jnp.array([False, True]))

    def test_truncated_normal_support_array_bounds(self):
        tn = TruncatedNormal(
            loc=jnp.array([0.0, 0.0]),
            scale=jnp.array([1.0, 1.0]),
            low=jnp.array([-1.0, 0.0]),
            high=jnp.array([1.0, 2.0]),
            name="tn_arr",
        )
        c = tn.support
        assert jnp.array_equal(c.check(jnp.array([0.0, 1.0])), jnp.array([True, True]))
        assert jnp.array_equal(c.check(jnp.array([-2.0, 1.0])), jnp.array([False, True]))

    def test_binomial_support_array_total_count(self):
        # Regression: ``int(self._total_count)`` previously crashed for
        # array-valued total_count. Per-dim total_count should produce a
        # working integer_interval.
        b = Binomial(
            total_count=jnp.array([5, 10]),
            probs=jnp.array([0.3, 0.5]),
            name="b_arr",
        )
        c = b.support
        assert jnp.array_equal(c.check(jnp.array([3, 7])), jnp.array([True, True]))
        assert jnp.array_equal(c.check(jnp.array([6, 7])), jnp.array([False, True]))

    def test_bernoulli_support(self):
        assert Bernoulli(probs=0.5, name="d").support == boolean

    def test_poisson_support(self):
        assert Poisson(rate=3.0, name="p").support == non_negative_integer

    def test_dirichlet_support(self):
        assert Dirichlet([1.0, 2.0], name="d").support == simplex

    def test_wishart_support(self):
        assert Wishart(df=5.0, scale_tril=jnp.eye(3), name="w").support == positive_definite

    def test_vonmisesfisher_support(self):
        assert VonMisesFisher([1.0, 0.0, 0.0], 5.0, name="v").support == sphere

    def test_mvn_support(self):
        assert MultivariateNormal(jnp.zeros(2), cov=jnp.eye(2), name="z").support == real

    def test_empirical_support(self):
        ed = NumericEmpiricalDistribution(jnp.ones((5, 2)))
        assert ed.support == real


# ── Section 4: from_distribution tests ────────────────────────────────────────


class TestFromDistribution:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    # -- same-class copy --
    def test_normal_from_normal(self, key):
        n = Normal(loc=3.0, scale=2.0, name="n")
        n2 = from_distribution(n, Normal, key=key)
        assert jnp.isclose(n2.loc, 3.0, atol=0.01)

    def test_beta_from_beta(self, key):
        b = Beta(alpha=2.0, beta=5.0, name="b")
        b2 = from_distribution(b, Beta, key=key)
        assert jnp.isclose(b2.alpha, 2.0, atol=0.01)

    # -- moment-matching --
    def test_normal_from_gamma(self, key):
        """Gamma -> Normal via moment matching (check_support=False needed)."""
        g = Gamma(concentration=9.0, rate=1.0, name="g")
        n = from_distribution(g, Normal, key=key, check_support=False, num_samples=5000)
        # Gamma(9,1) has mean=9, var=9
        assert jnp.isclose(n.loc, 9.0, atol=1.0)

    def test_gamma_from_normal(self, key):
        """Normal -> Gamma should fail support check by default."""
        n = Normal(loc=5.0, scale=1.0, name="n")
        with pytest.raises(ValueError, match="support"):
            from_distribution(n, Gamma, key=key)

    def test_gamma_from_normal_override(self, key):
        """Normal -> Gamma with check_support=False should work."""
        n = Normal(loc=5.0, scale=1.0, name="n")
        g = from_distribution(n, Gamma, key=key, check_support=False, num_samples=5000)
        assert jnp.isclose(float(g.concentration * 1.0 / g.rate), 5.0, atol=1.0)

    def test_beta_from_uniform(self, key):
        """Uniform(0,1) -> Beta should work (compatible support)."""
        u = Uniform(low=0.0, high=1.0, name="u")
        b = from_distribution(u, Beta, key=key, num_samples=5000)
        # Uniform(0,1) has mean=0.5, var=1/12 -> alpha~=beta~=1
        assert float(b.alpha) > 0
        assert float(b.beta) > 0

    # -- discrete --
    def test_bernoulli_from_bernoulli(self, key):
        b = Bernoulli(probs=0.7, name="b")
        b2 = from_distribution(b, Bernoulli, key=key)
        assert jnp.isclose(b2.probs, 0.7, atol=0.01)

    def test_poisson_from_poisson(self, key):
        p = Poisson(rate=5.0, name="p")
        p2 = from_distribution(p, Poisson, key=key)
        assert jnp.isclose(p2.rate, 5.0, atol=0.01)

    def test_binomial_requires_total_count(self, key):
        """Binomial.from_distribution from non-Binomial needs total_count."""
        p = Poisson(rate=3.0, name="p")
        with pytest.raises(ValueError, match="total_count"):
            from_distribution(p, Binomial, key=key, check_support=False)

    def test_binomial_from_poisson(self, key):
        p = Poisson(rate=3.0, name="p")
        b = from_distribution(p, Binomial, key=key, check_support=False, total_count=10, num_samples=5000)
        # mean ~ 3, so probs ~ 0.3
        assert b.probs is not None

    # -- multivariate --
    def test_mvn_from_empirical(self, key):
        samples = jax.random.normal(key, (100, 3))
        ed = NumericEmpiricalDistribution(samples)
        mvn = from_distribution(ed, MultivariateNormal)
        assert mvn.dim == 3

    def test_dirichlet_from_dirichlet(self, key):
        d = Dirichlet(concentration=jnp.array([1.0, 2.0, 3.0]), name="d")
        d2 = from_distribution(d, Dirichlet, key=key)
        assert jnp.allclose(d2.concentration, d.concentration)

    # -- provenance --
    def test_from_distribution_same_class_returns_source(self, key):
        """Same-class conversion returns the source object (no-op)."""
        n = Normal(loc=0.0, scale=1.0, name="n")
        n2 = from_distribution(n, Normal, key=key)
        assert n2 is n

    def test_from_distribution_cross_class_provenance(self, key):
        """Cross-class conversion attaches provenance."""
        g = Gamma(concentration=3.0, rate=1.0, name="g")
        n = from_distribution(g, Normal, key=key, check_support=False)
        assert n.source is not None
        assert n.source.operation == "from_distribution"

    # -- empirical from anything --
    def test_empirical_from_normal(self, key):
        n = Normal(loc=0.0, scale=1.0, name="n")
        ed = from_distribution(n, NumericEmpiricalDistribution, key=key, num_samples=100)
        assert ed.n == 100
