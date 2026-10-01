"""Tests for the converter registry and built-in converters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Bernoulli,
    Beta,
    Categorical,
    ConversionInfo,
    ConversionMethod,
    Converter,
    EmpiricalDistribution,
    Exponential,
    Gamma,
    KDEDistribution,
    MultivariateNormal,
    Normal,
    NumericRecordBatch,
    NumericRecordSpec,
    OpaqueBatch,
    Poisson,
    converter_registry,
    from_distribution,
)
from probpipe.families._continuous import (
    Cauchy,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    LogNormal,
    Pareto,
    StudentT,
    TruncatedNormal,
    Uniform,
)
from probpipe.families._discrete import Binomial, NegativeBinomial
from probpipe.families._multivariate import Dirichlet, Multinomial, VonMisesFisher, Wishart

# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


class TestConverterRegistry:
    def test_check_returns_conversioninfo(self):
        info = converter_registry.check(Normal("x", 0, 1), Normal)
        assert isinstance(info, ConversionInfo)
        assert info.feasible

    def test_check_infeasible_for_unknown_target(self):
        info = converter_registry.check(Normal("x", 0, 1), int)
        assert not info.feasible

    def test_convert_raises_for_unknown(self):
        with pytest.raises(TypeError):
            converter_registry.convert(42, Normal)

    def test_is_distribution_type_probpipe(self):
        assert converter_registry.is_distribution_type(Normal("x", 0, 1))
        assert converter_registry.is_distribution_type(EmpiricalDistribution("x", jnp.ones((5, 1))))

    def test_is_distribution_type_tfp(self):
        assert converter_registry.is_distribution_type(tfd.Normal(0, 1))

    def test_is_distribution_type_non_dist(self):
        assert not converter_registry.is_distribution_type(42)
        assert not converter_registry.is_distribution_type("hello")


# ---------------------------------------------------------------------------
# ProbPipe ↔ ProbPipe
# ---------------------------------------------------------------------------


class TestProbPipeConverter:
    def test_same_class_exact(self):
        n = Normal(loc=2.0, scale=0.5, name="x")
        info = converter_registry.check(n, Normal)
        assert info.method == ConversionMethod.EXACT
        assert info.estimated_time == 0.0

        result = converter_registry.convert(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)
        np.testing.assert_allclose(float(result._scale), 0.5)

    def test_cross_family_moment_match(self):
        g = Gamma(concentration=9.0, rate=1.0, name="g")
        info = converter_registry.check(g, Normal)
        assert info.method == ConversionMethod.MOMENT_MATCH

        result = converter_registry.convert(g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_support_mismatch_raises_by_default(self):
        n = Normal(loc=0.5, scale=0.1, name="x")
        with pytest.raises(ValueError, match="support"):
            converter_registry.convert(n, Beta)

    def test_support_mismatch_override(self):
        n = Normal(loc=0.5, scale=0.1, name="x")
        result = converter_registry.convert(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_to_empirical(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        emp = converter_registry.convert(n, EmpiricalDistribution, num_samples=100)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.num_atoms == 100
        assert emp.event_spec == n.event_spec
        assert emp.atoms.level_names == ("sample",)

    def test_provenance_attached(self):
        g = Gamma(concentration=3.0, rate=1.0, name="prior")
        result = converter_registry.convert(g, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "from_distribution"
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].name == "prior"

    def test_same_class_returns_source(self):
        """Same-class conversion returns the source object itself."""
        n = Normal(loc=1.0, scale=2.0, name="x")
        result = converter_registry.convert(n, Normal)
        assert result is n


class TestAnEmpiricalSourceAgainstTheTargetSupport:
    """An empirical law's array atoms declare no support, so its atoms are checked instead."""

    def test_atoms_outside_the_target_support_raise(self):
        source = EmpiricalDistribution("x", jnp.array([-5.0, -1.0, 2.0, -3.0]))
        with pytest.raises(ValueError, match="support"):
            from_distribution(source, Exponential)

    def test_atoms_inside_the_target_support_convert(self):
        source = EmpiricalDistribution("x", jnp.array([0.5, 1.0, 2.0, 3.0]))
        result = from_distribution(source, Exponential)
        assert isinstance(result, Exponential)
        np.testing.assert_allclose(float(result._mean()), 1.625, rtol=1e-6)

    def test_the_check_can_be_overridden(self):
        source = EmpiricalDistribution("x", jnp.array([-5.0, -1.0, 2.0, -3.0]))
        assert isinstance(from_distribution(source, Exponential, check_support=False), Exponential)


# ---------------------------------------------------------------------------
# Cross-family moment-matching (exercises all _convert_to_* functions)
# ---------------------------------------------------------------------------


class TestAllCrossFamilyConversions:
    """Exercise every _convert_to_* path with a cross-family source."""

    key = jax.random.PRNGKey(42)

    @pytest.mark.parametrize(
        "target_cls",
        [
            Normal,
            Beta,
            InverseGamma,
            Exponential,
            LogNormal,
            Uniform,
            Cauchy,
            Laplace,
            HalfNormal,
            HalfCauchy,
            Pareto,
            TruncatedNormal,
            StudentT,
        ],
    )
    def test_continuous_from_gamma(self, target_cls):
        """Convert Gamma(9,1) to each continuous type (skip support issues)."""
        g = Gamma(concentration=9.0, rate=1.0, name="g")
        result = converter_registry.convert(g, target_cls, check_support=False, num_samples=500)
        assert isinstance(result, target_cls)
        assert result.provenance is not None  # cross-family: provenance attached

    def test_bernoulli_from_poisson(self):
        p = Poisson(rate=0.5, name="p")
        result = converter_registry.convert(p, Bernoulli, check_support=False)
        assert isinstance(result, Bernoulli)

    def test_binomial_from_poisson(self):
        p = Poisson(rate=3.0, name="p")
        result = converter_registry.convert(p, Binomial, check_support=False, total_count=10)
        assert isinstance(result, Binomial)

    def test_binomial_requires_total_count(self):
        p = Poisson(rate=3.0, name="p")
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, Binomial, check_support=False)

    def test_poisson_from_bernoulli(self):
        b = Bernoulli(probs=0.3, name="b")
        result = converter_registry.convert(b, Poisson, check_support=False)
        assert isinstance(result, Poisson)

    def test_categorical_from_bernoulli(self):
        b = Bernoulli(probs=0.7, name="b")
        result = converter_registry.convert(b, Categorical, check_support=False, num_samples=500)
        assert isinstance(result, Categorical)

    def test_negativebinomial_from_poisson(self):
        p = Poisson(rate=3.0, name="p")
        result = converter_registry.convert(p, NegativeBinomial, check_support=False, total_count=5)
        assert isinstance(result, NegativeBinomial)

    def test_negativebinomial_requires_total_count(self):
        p = Poisson(rate=3.0, name="p")
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, NegativeBinomial, check_support=False)

    def test_dirichlet_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.array([0.3, 0.5, 0.2]), cov=0.01 * jnp.eye(3), name="z")
        result = converter_registry.convert(mvn, Dirichlet, check_support=False, num_samples=500)
        assert isinstance(result, Dirichlet)

    def test_multinomial_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.array([3.0, 5.0, 2.0]), cov=jnp.eye(3), name="z")
        result = converter_registry.convert(
            mvn, Multinomial, check_support=False, total_count=10, num_samples=500
        )
        assert isinstance(result, Multinomial)

    def test_multinomial_requires_total_count(self):
        mvn = MultivariateNormal(loc=jnp.array([3.0, 5.0]), cov=jnp.eye(2), name="z")
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(mvn, Multinomial, check_support=False)

    def test_wishart_from_mvn(self):
        # Wishart samples are matrices; use Wishart as source for itself
        w = Wishart(df=5.0, scale_tril=jnp.eye(2), name="w")
        result = converter_registry.convert(w, Wishart)
        assert result is w  # same-class

    def test_wishart_to_empirical(self):
        w = Wishart(df=5.0, scale_tril=jnp.eye(2), name="w")
        result = converter_registry.convert(w, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 50

    def test_vonmisesfisher_same_class(self):
        vmf = VonMisesFisher(
            mean_direction=jnp.array([1.0, 0.0, 0.0]), concentration=5.0, name="vmf"
        )
        result = converter_registry.convert(vmf, VonMisesFisher)
        assert result is vmf

    def test_mvn_from_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        emp = EmpiricalDistribution("x", samples)
        result = converter_registry.convert(emp, MultivariateNormal)
        assert isinstance(result, MultivariateNormal)
        assert result.loc.shape == (3,)

    def test_mvn_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        result = converter_registry.convert(mvn, MultivariateNormal)
        assert result is mvn


# ---------------------------------------------------------------------------
# TFP ↔ ProbPipe
# ---------------------------------------------------------------------------


class TestTFPConverter:
    def test_tfp_normal_to_probpipe(self):
        tfp_n = tfd.Normal(loc=2.0, scale=0.5)
        info = converter_registry.check(tfp_n, Normal)
        assert info.feasible
        assert info.method == ConversionMethod.EXACT

        result = converter_registry.convert(tfp_n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)
        np.testing.assert_allclose(float(result._scale), 0.5)

    def test_tfp_beta_to_probpipe(self):
        tfp_b = tfd.Beta(concentration1=2.0, concentration0=5.0)
        result = converter_registry.convert(tfp_b, Beta)
        assert isinstance(result, Beta)
        np.testing.assert_allclose(float(result._alpha), 2.0)
        np.testing.assert_allclose(float(result._beta), 5.0)

    def test_tfp_mvn_to_probpipe(self):
        loc = jnp.array([1.0, 2.0])
        tril = jnp.array([[1.0, 0.0], [0.3, 0.9]])
        tfp_mvn = tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril)
        result = converter_registry.convert(tfp_mvn, MultivariateNormal)
        assert isinstance(result, MultivariateNormal)
        np.testing.assert_allclose(result.loc, loc, atol=1e-5)

    def test_probpipe_to_tfp(self):
        n = Normal(loc=3.0, scale=1.0, name="x")
        result = converter_registry.convert(n, tfd.Normal)
        assert isinstance(result, tfd.Normal)
        np.testing.assert_allclose(float(result.loc), 3.0)
        np.testing.assert_allclose(float(result.scale), 1.0)

    def test_probpipe_to_tfp_beta(self):
        b = Beta(alpha=2.0, beta=5.0, name="b")
        result = converter_registry.convert(b, tfd.Beta)
        assert isinstance(result, tfd.Beta)
        np.testing.assert_allclose(float(result.concentration1), 2.0)
        np.testing.assert_allclose(float(result.concentration0), 5.0)

    def test_tfp_to_probpipe_provenance(self):
        result = converter_registry.convert(tfd.Normal(0, 1), Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert_from_tfp"

    def test_unknown_tfp_to_empirical(self):
        """Unknown TFP types fall back to sampling → EmpiricalDistribution."""
        # Use a TFP distribution we haven't mapped
        tfp_dist = tfd.VonMises(loc=0.0, concentration=1.0)
        result = converter_registry.convert(tfp_dist, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 50

    def test_probpipe_mvn_to_tfp(self):
        loc = jnp.array([1.0, 2.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 2.0]])
        mvn = MultivariateNormal(loc=loc, cov=cov, name="z")
        result = converter_registry.convert(mvn, tfd.MultivariateNormalTriL)
        assert isinstance(result, tfd.MultivariateNormalTriL)
        np.testing.assert_allclose(result.loc, loc, atol=1e-5)

    def test_probpipe_to_tfp_round_trip(self):
        n = Normal(loc=5.0, scale=2.0, name="x")
        tfp_n = converter_registry.convert(n, tfd.Normal)
        n2 = converter_registry.convert(tfp_n, Normal)
        np.testing.assert_allclose(float(n2._loc), 5.0)
        np.testing.assert_allclose(float(n2._scale), 2.0)

    def test_tfp_cross_family_chain(self):
        """TFP Gamma → ProbPipe Normal via chained conversion."""
        tfp_g = tfd.Gamma(concentration=9.0, rate=1.0)
        result = converter_registry.convert(tfp_g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_probpipe_gamma_to_tfp(self):
        g = Gamma(concentration=3.0, rate=1.0, name="g")
        result = converter_registry.convert(g, tfd.Gamma)
        assert isinstance(result, tfd.Gamma)
        np.testing.assert_allclose(float(result.concentration), 3.0)

    def test_probpipe_exponential_to_tfp(self):
        e = Exponential(rate=2.0, name="e")
        result = converter_registry.convert(e, tfd.Exponential)
        assert isinstance(result, tfd.Exponential)
        np.testing.assert_allclose(float(result.rate), 2.0)

    def test_probpipe_bernoulli_to_tfp(self):
        b = Bernoulli(probs=0.3, name="b")
        result = converter_registry.convert(b, tfd.Bernoulli)
        assert isinstance(result, tfd.Bernoulli)

    def test_probpipe_dirichlet_to_tfp(self):
        d = Dirichlet(concentration=jnp.array([2.0, 3.0, 1.0]), name="d")
        result = converter_registry.convert(d, tfd.Dirichlet)
        assert isinstance(result, tfd.Dirichlet)

    def test_tfp_poisson_to_probpipe(self):
        result = converter_registry.convert(tfd.Poisson(rate=3.0), Poisson)
        assert isinstance(result, Poisson)
        np.testing.assert_allclose(float(result._rate), 3.0)

    def test_tfp_categorical_to_probpipe(self):
        probs = jnp.array([0.2, 0.3, 0.5])
        result = converter_registry.convert(tfd.Categorical(probs=probs), Categorical)
        assert isinstance(result, Categorical)
        np.testing.assert_allclose(result._probs, probs, atol=1e-5)

    def test_tfp_dirichlet_to_probpipe(self):
        conc = jnp.array([2.0, 3.0])
        result = converter_registry.convert(tfd.Dirichlet(concentration=conc), Dirichlet)
        assert isinstance(result, Dirichlet)

    def test_tfp_mvn_diag_to_probpipe(self):
        result = converter_registry.convert(
            tfd.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2)),
            MultivariateNormal,
        )
        assert isinstance(result, MultivariateNormal)

    def test_is_distribution_type_for_tfp(self):
        assert converter_registry.is_distribution_type(tfd.Gamma(1.0, 1.0))


# ---------------------------------------------------------------------------
# Scipy ↔ ProbPipe (optional)
# ---------------------------------------------------------------------------


class TestScipyConverter:
    @pytest.fixture(autouse=True)
    def _check_scipy(self):
        pytest.importorskip("scipy")

    def test_scipy_norm_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.norm(loc=1.0, scale=2.0), Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 1.0)
        np.testing.assert_allclose(float(result._scale), 2.0)

    def test_scipy_beta_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.beta(2.0, 5.0), Beta)
        assert isinstance(result, Beta)
        np.testing.assert_allclose(float(result._alpha), 2.0)
        np.testing.assert_allclose(float(result._beta), 5.0)

    def test_scipy_gamma_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.gamma(3.0, scale=2.0), Gamma)
        assert isinstance(result, Gamma)
        np.testing.assert_allclose(float(result._concentration), 3.0)
        np.testing.assert_allclose(float(result._rate), 0.5)  # rate = 1/scale

    def test_probpipe_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen

        n = Normal(loc=3.0, scale=1.0, name="x")
        result = converter_registry.convert(n, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 3.0)
        np.testing.assert_allclose(result.std(), 1.0)

    def test_scipy_provenance(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.norm(0, 1), Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert_from_scipy"

    def test_scipy_norm_positional_args(self):
        """Scipy norm created with positional args should still extract correctly."""
        import scipy.stats as ss

        result = converter_registry.convert(ss.norm(1.0, 2.0), Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 1.0)
        np.testing.assert_allclose(float(result._scale), 2.0)

    def test_scipy_beta_keyword_args(self):
        """Scipy beta created with keyword args should extract correctly."""
        import scipy.stats as ss

        result = converter_registry.convert(ss.beta(a=2.0, b=5.0), Beta)
        assert isinstance(result, Beta)
        np.testing.assert_allclose(float(result._alpha), 2.0)
        np.testing.assert_allclose(float(result._beta), 5.0)

    def test_scipy_expon_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.expon(scale=2.0), Exponential)
        assert isinstance(result, Exponential)
        np.testing.assert_allclose(float(result._rate), 0.5)

    def test_scipy_uniform_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.uniform(loc=1.0, scale=3.0), Uniform)
        assert isinstance(result, Uniform)
        np.testing.assert_allclose(float(result._low), 1.0)
        np.testing.assert_allclose(float(result._high), 4.0)

    def test_scipy_laplace_to_probpipe(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.laplace(loc=2.0, scale=0.5), Laplace)
        assert isinstance(result, Laplace)
        np.testing.assert_allclose(float(result._loc), 2.0)
        np.testing.assert_allclose(float(result._scale), 0.5)

    def test_probpipe_gamma_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen

        g = Gamma(concentration=3.0, rate=0.5, name="g")
        result = converter_registry.convert(g, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 6.0, atol=0.01)

    def test_probpipe_exponential_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen

        e = Exponential(rate=2.0, name="e")
        result = converter_registry.convert(e, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_probpipe_beta_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen

        b = Beta(alpha=2.0, beta=5.0, name="b")
        result = converter_registry.convert(b, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 2.0 / 7.0, atol=0.01)

    def test_unknown_scipy_fallback_to_sampling(self):
        """Unknown scipy distribution type falls back to sampling."""
        import scipy.stats as ss

        # Use a scipy distribution we haven't mapped (e.g., chi2)
        result = converter_registry.convert(ss.chi2(df=3), EmpiricalDistribution, num_samples=100)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 100

    def test_scipy_check_unknown_type(self):
        import scipy.stats as ss

        info = converter_registry.check(ss.chi2(df=3), Normal)
        assert info.feasible
        assert info.method == ConversionMethod.SAMPLE

    def test_is_distribution_type_scipy(self):
        import scipy.stats as ss

        assert converter_registry.is_distribution_type(ss.norm(0, 1))


# ---------------------------------------------------------------------------
# Custom converter registration
# ---------------------------------------------------------------------------


class TestCustomConverter:
    def test_register_custom_converter(self):
        class DummyDist:
            def __init__(self, val):
                self.val = val

        class DummyConverter(Converter):
            def source_types(self):
                return (DummyDist,)

            def target_types(self):
                return (Normal,)

            def check(self, source, target_type):
                if isinstance(source, DummyDist) and target_type is Normal:
                    return ConversionInfo(feasible=True, method=ConversionMethod.EXACT)
                return ConversionInfo(feasible=False)

            def convert(self, source, target_type, *, key=None, **kwargs):
                return Normal(loc=source.val, scale=1.0, name="x")

            @property
            def priority(self):
                return 10

        converter_registry.register(DummyConverter())
        try:
            d = DummyDist(42.0)
            assert converter_registry.is_distribution_type(d)
            result = converter_registry.convert(d, Normal)
            assert isinstance(result, Normal)
            np.testing.assert_allclose(float(result._loc), 42.0)
        finally:
            # Clean up: remove the dummy converter
            converter_registry._converters = [
                c for c in converter_registry._converters if not isinstance(c, DummyConverter)
            ]
            converter_registry._type_cache.clear()


# ---------------------------------------------------------------------------
# from_distribution() backward compatibility
# ---------------------------------------------------------------------------


class TestFromDistributionDelegation:
    """Verify from_distribution() delegates to the registry."""

    def test_from_distribution_same_class(self):
        n = Normal(loc=2.0, scale=0.5, name="x")
        result = from_distribution(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)

    def test_from_distribution_cross_family(self):
        g = Gamma(concentration=9.0, rate=1.0, name="g")
        result = from_distribution(g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_from_distribution_support_check(self):
        n = Normal(loc=0.5, scale=0.1, name="x")
        with pytest.raises(ValueError, match="support"):
            from_distribution(n, Beta)

    def test_from_distribution_check_support_false(self):
        n = Normal(loc=0.5, scale=0.1, name="x")
        result = from_distribution(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_from_distribution_to_empirical(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        emp = from_distribution(n, EmpiricalDistribution, num_samples=50)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.num_atoms == 50

    def test_empirical_to_empirical_preserves_source_only_for_raw_apply(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        emp = EmpiricalDistribution("orig", samples)
        raw = from_distribution.apply(emp, EmpiricalDistribution)
        emp2 = from_distribution(emp, EmpiricalDistribution)
        assert raw is emp
        assert emp2 is not emp
        assert emp2.provenance.operation == "workflow.from_distribution"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConversionProvenance:
    """A conversion records its source in provenance."""

    def test_empirical_moments_record_the_source(self):
        """Moment matching an empirical law records the law as its parent."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        emp = EmpiricalDistribution("x", samples)
        result = converter_registry.convert(emp, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "from_distribution"

    def test_same_class_records_nothing(self):
        """Same-class conversion returns source directly, no provenance."""
        n = Normal(loc=2.0, scale=0.5, name="x")
        result = converter_registry.convert(n, Normal)
        assert result is n  # same object, no conversion

    def test_cross_family_provenance_attached(self):
        """Cross-family conversion attaches provenance with source as parent."""
        g = Gamma(concentration=9.0, rate=1.0, name="g")
        result = converter_registry.convert(g, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "from_distribution"
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].name == "g"


class TestEdgeCases:
    def test_convert_none_raises(self):
        with pytest.raises(TypeError):
            converter_registry.convert(None, Normal)

    def test_convert_non_type_target_raises(self):
        with pytest.raises(TypeError, match="No converter"):
            converter_registry.convert(Normal("x", 0, 1), str)

    def test_check_infeasible_non_type_target(self):
        info = converter_registry.check(Normal("x", 0, 1), "not a type")
        assert not info.feasible


# ---------------------------------------------------------------------------
# Protocol-based conversion
# ---------------------------------------------------------------------------

from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)


class TestProtocolConversion:
    """Tests for converting distributions to satisfy a protocol."""

    def test_already_satisfies_returns_same(self):
        """Distribution that already satisfies the protocol is returned unchanged."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        result = converter_registry.convert(n, SupportsLogProb)
        assert result is n

    def test_scalar_empirical_to_supports_log_prob(self):
        """Scalar EmpiricalDistribution converts to KDE via SupportsLogProb."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (300,))
        emp = EmpiricalDistribution("x", samples)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        np.testing.assert_allclose(float(result._mean()), float(emp._mean()), atol=0.01)
        np.testing.assert_allclose(
            float(result._variance()),
            float(emp._variance()),
            atol=0.3,
        )

    def test_multivariate_empirical_to_supports_log_prob(self):
        """Multivariate (single-field with d-dim event) RecordEmpirical → KDE."""
        samples = jax.random.normal(jax.random.PRNGKey(1), (300, 4))
        emp = EmpiricalDistribution("x", samples)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        np.testing.assert_allclose(
            np.array(result._mean()),
            np.array(emp._mean()),
            atol=0.2,
        )

    def test_multi_field_record_empirical_to_kde(self):
        """An empirical law over a record converts to a KDE over its atoms."""
        n = 200
        rows = NumericRecordBatch(
            "r",
            {
                "mu": jax.random.normal(jax.random.PRNGKey(2), (n,)),
                "log_sigma": jax.random.normal(jax.random.PRNGKey(3), (n,)),
            },
            "row",
            element_spec=NumericRecordSpec(mu=(), log_sigma=()),
        )
        emp = EmpiricalDistribution("emp", rows)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.num_atoms == n
        # log_prob of one record returns a scalar.
        lp = result._log_prob({"mu": 0.0, "log_sigma": 0.0})
        assert lp.shape == ()

    def test_single_field_record_empirical_to_kde_unchanged(self):
        """An empirical law over scalars converts to a KDE whose event is a scalar."""
        samples = jax.random.normal(jax.random.PRNGKey(4), (150,))
        emp = EmpiricalDistribution("theta", samples)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.event_shape == ()

    def test_weighted_single_field_empirical_to_kde_preserves_weights(self):
        """An empirical law with non-uniform weights converts to a KDE with the same weights."""
        n = 80
        samples = jax.random.normal(jax.random.PRNGKey(5), (n,))
        weights = jnp.linspace(0.1, 1.0, n)
        emp = EmpiricalDistribution("x", samples, weights=weights)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        np.testing.assert_allclose(
            np.asarray(result._w.normalized), np.asarray(emp.weights), atol=1e-6
        )
        assert not result._w.is_uniform

    def test_object_array_empirical_to_kde_rejected(self):
        """An empirical law over opaque atoms does not convert to a KDE, which smooths numbers."""
        emp = EmpiricalDistribution("emp", OpaqueBatch("labels", ["a", "b", "c"], "site"))
        with pytest.raises(TypeError, match="numeric"):
            converter_registry.convert(emp, KDEDistribution)

    def test_check_protocol_already_satisfied(self):
        """check() returns EXACT when protocol is already satisfied."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        info = converter_registry.check(n, SupportsLogProb)
        assert info.feasible
        assert info.method == ConversionMethod.EXACT

    def test_check_protocol_needs_conversion(self):
        """check() returns feasible when conversion is possible."""
        samples = jax.random.normal(jax.random.PRNGKey(2), (100,))
        emp = EmpiricalDistribution("x", samples)
        info = converter_registry.check(emp, SupportsLogProb)
        assert info.feasible
        assert info.method == ConversionMethod.MOMENT_MATCH

    def test_unregistered_protocol_raises(self):
        """Unregistered protocol raises TypeError when source doesn't satisfy it."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class SupportsSomethingUnregistered(Protocol):
            def _something_unregistered(self) -> None: ...

        samples = jax.random.normal(jax.random.PRNGKey(3), (50,))
        emp = EmpiricalDistribution("x", samples)
        # The protocol is not a registered conversion target, and the
        # empirical distribution does not satisfy it either.
        with pytest.raises(TypeError):
            converter_registry.convert(emp, SupportsSomethingUnregistered)

    def test_from_distribution_with_protocol(self):
        """from_distribution() works with protocol targets."""
        samples = jax.random.normal(jax.random.PRNGKey(4), (200,))
        emp = EmpiricalDistribution("x", samples)
        result = from_distribution(emp, SupportsLogProb)
        assert isinstance(result, SupportsLogProb)

    def test_protocol_conversion_preserves_provenance(self):
        """Protocol-based conversion attaches provenance."""
        samples = jax.random.normal(jax.random.PRNGKey(5), (200,))
        emp = EmpiricalDistribution("posterior", samples)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert result.provenance is not None
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].name == "posterior"

    def test_multi_field_empirical_preserves_template_through_kde(self):
        """An empirical law over a record converts to a KDE over that record's fields."""
        n = 200
        rows = NumericRecordBatch(
            "r",
            {
                "intercept": jax.random.normal(jax.random.PRNGKey(0), (n,)),
                "slope": jax.random.normal(jax.random.PRNGKey(1), (n,)),
            },
            "row",
            element_spec=NumericRecordSpec(intercept=(), slope=()),
        )
        emp = EmpiricalDistribution("emp", rows)
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.event_spec.spec.fields == ("intercept", "slope")

    def test_approximate_distribution_preserves_template_through_kde(self):
        """An inference result converts to a KDE over its target's record.

        An :class:`IncrementalConditioner` update beyond its first batch reads the
        record's fields.
        """
        from probpipe.inference._approximate_distribution import (
            ApproximateDistribution,
        )

        n_draws = 100
        chain_intercept = jax.random.normal(jax.random.PRNGKey(0), (n_draws,))
        chain_slope = jax.random.normal(jax.random.PRNGKey(1), (n_draws,))
        # Stack into per-chain (num_draws, total_dim) layout
        chains = [jnp.stack([chain_intercept, chain_slope], axis=-1)]
        approx = ApproximateDistribution(
            chains,
            name="posterior",
            event_spec=NumericRecordSpec(intercept=(), slope=()),
        )
        result = converter_registry.convert(approx, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.event_spec.spec.fields == ("intercept", "slope")
        assert list(result.event_spec.components) == list(approx.event_spec.components)


# ---------------------------------------------------------------------------
# KDEDistribution
# ---------------------------------------------------------------------------


class TestKDEDistribution:
    """Tests for KDEDistribution."""

    def test_scalar_construction(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution("kde", samples)
        assert kde.event_shape == ()
        assert not hasattr(kde, "batch_shape")
        assert kde.num_atoms == 100

    def test_multivariate_construction(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        kde = KDEDistribution("kde", samples)
        assert kde.event_shape == (3,)
        assert kde.num_atoms == 100

    def test_log_prob_finite(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        kde = KDEDistribution("kde", samples)
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_log_prob_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 2))
        kde = KDEDistribution("kde", samples)
        lp = kde._log_prob(jnp.zeros(2))
        assert jnp.isfinite(lp)

    def test_sample_shape_scalar(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution("kde", samples)
        s = kde._sample(jax.random.PRNGKey(1), (5,))
        assert s.shape == (5,)

    def test_sample_shape_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        kde = KDEDistribution("kde", samples)
        s = kde._sample(jax.random.PRNGKey(1), (5,))
        assert s.shape == (5, 3)

    def test_mean_close_to_sample_mean(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (500,))
        kde = KDEDistribution("kde", samples)
        np.testing.assert_allclose(
            float(kde._mean()),
            float(jnp.mean(samples)),
            atol=0.01,
        )

    def test_variance_larger_than_sample_variance(self):
        """KDE variance = sample variance + bandwidth^2, so should be larger."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (500,))
        kde = KDEDistribution("kde", samples)
        sample_var = float(jnp.var(samples))
        kde_var = float(kde._variance())
        assert kde_var > sample_var

    def test_weighted_construction(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        weights = jnp.ones(100)
        weights = weights.at[0].set(10.0)
        kde = KDEDistribution("kde", samples, weights=weights)
        assert kde.num_atoms == 100
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_custom_bandwidth(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution("kde", samples, bandwidth=0.5)
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_supports_protocols(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution("kde", samples)
        assert isinstance(kde, SupportsLogProb)
        assert isinstance(kde, SupportsSampling)
        assert isinstance(kde, SupportsMean)
        assert isinstance(kde, SupportsVariance)
        assert isinstance(kde, SupportsCovariance)

    def test_convert_empirical_to_kde(self):
        """from_distribution(empirical, KDEDistribution) works."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        emp = EmpiricalDistribution("x", samples)
        kde = converter_registry.convert(emp, KDEDistribution)
        assert isinstance(kde, KDEDistribution)
        assert kde.num_atoms == 200
        np.testing.assert_allclose(
            float(kde._mean()),
            float(emp._mean()),
            atol=0.01,
        )

    def test_convert_normal_to_kde(self):
        """Converting a parametric distribution to KDE works via sampling."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        kde = converter_registry.convert(n, KDEDistribution, num_samples=500)
        assert isinstance(kde, KDEDistribution)
        assert kde.num_atoms == 500

    def test_cov_scalar(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        kde = KDEDistribution("kde", samples)
        cov = kde._cov()
        assert cov.shape == (1, 1)

    def test_cov_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        kde = KDEDistribution("kde", samples)
        cov = kde._cov()
        assert cov.shape == (3, 3)

    def test_repr(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (50,))
        kde = KDEDistribution("test_kde", samples)
        r = repr(kde)
        assert "KDEDistribution" in r
        assert "num_atoms=50" in r
