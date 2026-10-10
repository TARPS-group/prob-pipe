from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    NumericArrayBatch,
    NumericArraySpec,
    ResolutionError,
    convert,
    converter_registry,
)
from probpipe.core.constraints import (
    _supports_compatible,
    boolean,
    greater_than,
    integer_interval,
    interval,
    non_negative,
    non_negative_integer,
    positive,
    positive_definite,
    real,
    simplex,
    sphere,
    unit_interval,
)
from probpipe.families import (
    Bernoulli,
    Beta,
    Binomial,
    Dirichlet,
    Gamma,
    MultivariateNormal,
    Normal,
    Poisson,
    Uniform,
    VonMisesFisher,
    Wishart,
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
        # Cross-type and non-Constraint comparisons must short-circuit
        # via the type guard, not raise.
        assert interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5])) != greater_than(
            jnp.array([0.0, 0.5])
        )
        assert interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5])) != None  # noqa: E711
        # Shape-mismatched bounds compare unequal rather than raising.
        assert interval(jnp.array([0.0, 0.5]), jnp.array([1.0, 1.5])) != interval(
            jnp.array([0.0]), jnp.array([1.0])
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


# Each real-valued constraint with one real value inside its support.
REAL_SUPPORT_MEMBERS = [
    pytest.param(real, 1.0, id="real"),
    pytest.param(positive, 2.0, id="positive"),
    pytest.param(non_negative, 0.0, id="non_negative"),
    pytest.param(non_negative_integer, 3.0, id="non_negative_integer"),
    pytest.param(boolean, 1.0, id="boolean"),
    pytest.param(unit_interval, 0.5, id="unit_interval"),
    pytest.param(interval(-2.0, 3.0), 1.0, id="interval"),
    pytest.param(greater_than(5.0), 6.0, id="greater_than"),
    pytest.param(integer_interval(0, 4), 2.0, id="integer_interval"),
    pytest.param(simplex, [0.3, 0.7], id="simplex"),
    pytest.param(sphere, [0.6, 0.8], id="sphere"),
    pytest.param(positive_definite, [[2.0, 0.0], [0.0, 1.0]], id="positive_definite"),
]


class TestComplexValues:
    """A real-valued support contains a complex value only where it is real and inside."""

    @pytest.mark.parametrize(("constraint", "member"), REAL_SUPPORT_MEMBERS)
    def test_a_complex_value_with_zero_imaginary_part_is_inside(self, constraint, member):
        assert bool(jnp.all(constraint.check(jnp.asarray(member, dtype=jnp.complex64))))

    @pytest.mark.parametrize(("constraint", "member"), REAL_SUPPORT_MEMBERS)
    def test_a_nonzero_imaginary_part_is_outside(self, constraint, member):
        value = jnp.asarray(member, dtype=jnp.complex64) + 0.5j
        assert not bool(jnp.any(constraint.check(value)))

    def test_an_ordered_support_rejects_a_purely_imaginary_value(self):
        assert not bool(positive.check(jnp.asarray(1j)))
        assert not bool(real.check(jnp.asarray(1j)))

    @pytest.mark.parametrize(("constraint", "member"), REAL_SUPPORT_MEMBERS)
    def test_the_result_has_the_shape_of_the_real_check(self, constraint, member):
        real_values = jnp.stack([jnp.asarray(member)] * 3)
        offset = jnp.zeros(3).at[1].set(0.5).reshape((3,) + (1,) * (real_values.ndim - 1))
        result = constraint.check(real_values + 1j * offset)
        assert result.shape == constraint.check(real_values).shape
        assert result.tolist() == [True, False, True]

    @pytest.mark.parametrize(("constraint", "member"), REAL_SUPPORT_MEMBERS)
    def test_the_check_is_traceable_under_jit(self, constraint, member):
        real_values = jnp.stack([jnp.asarray(member)] * 2)
        offset = jnp.array([0.0, 0.5]).reshape((2,) + (1,) * (real_values.ndim - 1))
        result = jax.jit(constraint.check)(real_values + 1j * offset)
        assert result.tolist() == [True, False]

    def test_a_declared_positive_output_rejects_an_imaginary_value(self):
        from probpipe import Function

        load = Function(
            "load", lambda: jnp.asarray(1j), output_spec=NumericArraySpec((), support=positive)
        )
        with pytest.raises(ValueError, match="output/load does not conform to declared support"):
            load()


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
        assert Normal("x", 0.0, 1.0).support == real

    def test_beta_support(self):
        assert Beta("b", 2.0, 5.0).support == unit_interval

    def test_gamma_support(self):
        assert Gamma("g", 3.0, 1.0).support == positive

    def test_uniform_support(self):
        assert Uniform("u", low=-1.0, high=2.0).support == interval(-1.0, 2.0)

    # NOTE: A family of "support with array bounds" tests was removed.
    # Each exercised a legacy batched constructor:
    # ``Uniform(low=arr, high=arr)``, ``HalfCauchy(loc=arr, scale=arr)``,
    # ``Pareto(concentration=arr, scale=arr)``,
    # ``TruncatedNormal(loc=arr, scale=arr, low=arr, high=arr)``,
    # ``Binomial(total_count=arr, probs=arr)``. The framework
    # hierarchy ("one random variable per Distribution") no longer
    # permits these forms; migrate to
    # a ``DistributionBatch`` of separate laws for batched
    # constructions, and use ``Constraint`` directly for per-element
    # support checks (those are a property of the ``Constraint``
    # type, not of a batched ``Distribution``).

    def test_bernoulli_support(self):
        assert Bernoulli("d", probs=0.5).support == boolean

    def test_poisson_support(self):
        assert Poisson("p", rate=3.0).support == non_negative_integer

    def test_dirichlet_support(self):
        assert Dirichlet("d", [1.0, 2.0]).support == simplex

    def test_wishart_support(self):
        assert Wishart("w", df=5.0, scale_tril=jnp.eye(3)).support == positive_definite

    def test_vonmisesfisher_support(self):
        assert VonMisesFisher("v", [1.0, 0.0, 0.0], 5.0).support == sphere

    def test_mvn_support(self):
        assert MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2)).support == real

    def test_empirical_support_is_what_its_atoms_declare(self):
        atoms = NumericArrayBatch(
            "x", jnp.ones((5, 2)), "atom", element_spec=NumericArraySpec((2,), support=real)
        )
        assert EmpiricalDistribution(atoms, component="x").support == real
        assert EmpiricalDistribution(jnp.ones((5, 2)), component="x").support is None


# ── Section 4: from_distribution tests ────────────────────────────────────────


class TestConvert:
    """Conversions through ``from_distribution``, whose draws are workflow-owned."""

    # -- same-class copy --
    def test_normal_from_normal(self):
        n = Normal("n", loc=3.0, scale=2.0)
        n2 = convert(n, Normal)
        assert jnp.isclose(n2.loc, 3.0, atol=0.01)

    def test_beta_from_beta(self):
        b = Beta("b", alpha=2.0, beta=5.0)
        b2 = convert(b, Beta)
        assert jnp.isclose(b2.alpha, 2.0, atol=0.01)

    # -- moment-matching --
    def test_normal_from_gamma(self):
        """Gamma -> Normal via moment matching (check_support=False needed)."""
        g = Gamma("g", concentration=9.0, rate=1.0)
        n = converter_registry.convert(g, Normal, check_support=False, num_samples=5000)
        # Gamma(9,1) has mean=9, var=9
        assert jnp.isclose(n.loc, 9.0, atol=1.0)

    def test_gamma_from_normal(self):
        """Normal -> Gamma is refused by default, since the fit's support is not the source's."""
        n = Normal("n", loc=5.0, scale=1.0)
        with pytest.raises(ResolutionError, match="check_support=False"):
            convert(n, Gamma)

    def test_gamma_from_normal_override(self):
        """Normal -> Gamma with check_support=False should work."""
        n = Normal("n", loc=5.0, scale=1.0)
        g = converter_registry.convert(n, Gamma, check_support=False, num_samples=5000)
        assert jnp.isclose(float(g.concentration * 1.0 / g.rate), 5.0, atol=1.0)

    def test_beta_from_uniform(self):
        """Uniform(0,1) -> Beta should work (compatible support)."""
        u = Uniform("u", low=0.0, high=1.0)
        b = convert.with_options(method_options={"num_samples": 5000})(u, Beta)
        # Uniform(0,1) has mean=0.5, var=1/12 -> alpha~=beta~=1
        assert float(b.alpha) > 0
        assert float(b.beta) > 0

    # -- discrete --
    def test_bernoulli_from_bernoulli(self):
        b = Bernoulli("b", probs=0.7)
        b2 = convert(b, Bernoulli)
        assert jnp.isclose(b2.probs, 0.7, atol=0.01)

    def test_poisson_from_poisson(self):
        p = Poisson("p", rate=5.0)
        p2 = convert(p, Poisson)
        assert jnp.isclose(p2.rate, 5.0, atol=0.01)

    def test_binomial_requires_total_count(self):
        """Binomial.from_distribution from non-Binomial needs total_count."""
        p = Poisson("p", rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, Binomial, check_support=False)

    def test_binomial_from_poisson(self):
        p = Poisson("p", rate=3.0)
        b = converter_registry.convert(
            p, Binomial, check_support=False, total_count=10, num_samples=5000
        )
        # mean ~ 3, so probs ~ 0.3
        assert b.probs is not None

    # -- multivariate --
    def test_mvn_from_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(42), (100, 3))
        ed = EmpiricalDistribution(samples, component="x")
        mvn = convert(ed, MultivariateNormal)
        assert mvn.dim == 3

    def test_dirichlet_from_dirichlet(self):
        d = Dirichlet("d", concentration=jnp.array([1.0, 2.0, 3.0]))
        d2 = convert(d, Dirichlet)
        assert jnp.allclose(d2.concentration, d.concentration)

    # -- provenance --
    def test_convert_same_class_returns_the_source_under_fresh_identity(self):
        n = Normal("n", loc=0.0, scale=1.0)
        n2 = convert(n, Normal)
        assert n2 is not n
        assert float(n2._loc) == float(n._loc)
        assert n2.provenance.operation == "workflow.convert"

    def test_convert_cross_class_provenance(self):
        """Cross-class conversion attaches provenance."""
        g = Gamma("g", concentration=3.0, rate=1.0)
        n = converter_registry.convert(g, Normal, check_support=False)
        assert n.provenance is not None
        assert n.provenance.operation == "convert"

    # -- empirical from anything --
    def test_empirical_from_normal(self):
        n = Normal("n", loc=0.0, scale=1.0)
        ed = convert.with_options(method_options={"num_samples": 100})(n, EmpiricalDistribution)
        assert ed.num_atoms == 100
