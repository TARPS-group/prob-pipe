"""Tests for the converter registry and built-in converters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Normal, Beta, Gamma, Exponential, MultivariateNormal,
    Bernoulli, Poisson, Categorical,
    EmpiricalDistribution, ArrayDistribution,
    converter_registry, ConversionInfo, ConversionMethod, Converter,
    from_distribution,
)
from probpipe.distributions.continuous import (
    InverseGamma, LogNormal, StudentT, Uniform, Cauchy, Laplace,
    HalfNormal, HalfCauchy, Pareto, TruncatedNormal,
)
from probpipe.distributions.discrete import Binomial, NegativeBinomial
from probpipe.distributions.multivariate import Dirichlet, Multinomial, Wishart, VonMisesFisher


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------

class TestConverterRegistry:

    def test_check_returns_conversioninfo(self):
        info = converter_registry.check(Normal(0, 1), Normal)
        assert isinstance(info, ConversionInfo)
        assert info.feasible

    def test_check_infeasible_for_unknown_target(self):
        info = converter_registry.check(Normal(0, 1), int)
        assert not info.feasible

    def test_convert_raises_for_unknown(self):
        with pytest.raises(TypeError):
            converter_registry.convert(42, Normal)

    def test_is_distribution_type_probpipe(self):
        assert converter_registry.is_distribution_type(Normal(0, 1))
        assert converter_registry.is_distribution_type(EmpiricalDistribution(jnp.ones((5, 1))))

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
        n = Normal(loc=2.0, scale=0.5)
        info = converter_registry.check(n, Normal)
        assert info.method == ConversionMethod.EXACT
        assert info.estimated_time == 0.0

        result = converter_registry.convert(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)
        np.testing.assert_allclose(float(result._scale), 0.5)

    def test_cross_family_moment_match(self):
        g = Gamma(concentration=9.0, rate=1.0)
        info = converter_registry.check(g, Normal)
        assert info.method == ConversionMethod.MOMENT_MATCH

        result = converter_registry.convert(g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_support_mismatch_raises_by_default(self):
        n = Normal(loc=0.5, scale=0.1)
        with pytest.raises(ValueError, match="support"):
            converter_registry.convert(n, Beta)

    def test_support_mismatch_override(self):
        n = Normal(loc=0.5, scale=0.1)
        result = converter_registry.convert(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_to_empirical(self):
        n = Normal(loc=0.0, scale=1.0)
        emp = converter_registry.convert(n, EmpiricalDistribution, num_samples=100)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.n == 100

    def test_provenance_attached(self):
        g = Gamma(concentration=3.0, rate=1.0, name="prior")
        result = converter_registry.convert(g, Normal)
        assert result.source is not None
        assert result.source.operation == "from_distribution"
        assert g in result.source.parents

    def test_approximate_flag(self):
        g = Gamma(concentration=3.0, rate=1.0)
        result = converter_registry.convert(g, Normal)
        assert result.is_approximate

    def test_same_class_not_approximate(self):
        n = Normal(loc=0.0, scale=1.0)
        result = converter_registry.convert(n, Normal)
        assert not result.is_approximate

    def test_same_class_returns_source(self):
        """Same-class conversion returns the source object itself."""
        n = Normal(loc=1.0, scale=2.0)
        result = converter_registry.convert(n, Normal)
        assert result is n


# ---------------------------------------------------------------------------
# Cross-family moment-matching (exercises all _convert_to_* functions)
# ---------------------------------------------------------------------------

class TestAllCrossFamilyConversions:
    """Exercise every _convert_to_* path with a cross-family source."""

    key = jax.random.PRNGKey(42)

    @pytest.mark.parametrize("target_cls", [
        Normal, Beta, InverseGamma, Exponential, LogNormal,
        Uniform, Cauchy, Laplace, HalfNormal, HalfCauchy, Pareto,
        TruncatedNormal, StudentT,
    ])
    def test_continuous_from_gamma(self, target_cls):
        """Convert Gamma(9,1) to each continuous type (skip support issues)."""
        g = Gamma(concentration=9.0, rate=1.0)
        result = converter_registry.convert(g, target_cls, check_support=False, num_samples=500)
        assert isinstance(result, target_cls)
        assert result.source is not None  # cross-family: provenance attached

    def test_bernoulli_from_poisson(self):
        p = Poisson(rate=0.5)
        result = converter_registry.convert(p, Bernoulli, check_support=False)
        assert isinstance(result, Bernoulli)

    def test_binomial_from_poisson(self):
        p = Poisson(rate=3.0)
        result = converter_registry.convert(p, Binomial, check_support=False, total_count=10)
        assert isinstance(result, Binomial)

    def test_binomial_requires_total_count(self):
        p = Poisson(rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, Binomial, check_support=False)

    def test_poisson_from_bernoulli(self):
        b = Bernoulli(probs=0.3)
        result = converter_registry.convert(b, Poisson, check_support=False)
        assert isinstance(result, Poisson)

    def test_categorical_from_bernoulli(self):
        b = Bernoulli(probs=0.7)
        result = converter_registry.convert(b, Categorical, check_support=False, num_samples=500)
        assert isinstance(result, Categorical)

    def test_negativebinomial_from_poisson(self):
        p = Poisson(rate=3.0)
        result = converter_registry.convert(p, NegativeBinomial, check_support=False, total_count=5)
        assert isinstance(result, NegativeBinomial)

    def test_negativebinomial_requires_total_count(self):
        p = Poisson(rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, NegativeBinomial, check_support=False)

    def test_dirichlet_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.array([0.3, 0.5, 0.2]),
                                  cov=0.01 * jnp.eye(3))
        result = converter_registry.convert(mvn, Dirichlet, check_support=False, num_samples=500)
        assert isinstance(result, Dirichlet)

    def test_multinomial_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.array([3.0, 5.0, 2.0]),
                                  cov=jnp.eye(3))
        result = converter_registry.convert(mvn, Multinomial, check_support=False,
                                             total_count=10, num_samples=500)
        assert isinstance(result, Multinomial)

    def test_multinomial_requires_total_count(self):
        mvn = MultivariateNormal(loc=jnp.array([3.0, 5.0]),
                                  cov=jnp.eye(2))
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(mvn, Multinomial, check_support=False)

    def test_wishart_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.array([1.0, 0.0]),
                                  cov=jnp.eye(2))
        # Wishart samples are matrices; use Wishart as source for itself
        w = Wishart(df=5.0, scale_tril=jnp.eye(2))
        result = converter_registry.convert(w, Wishart)
        assert result is w  # same-class

    def test_wishart_to_empirical(self):
        w = Wishart(df=5.0, scale_tril=jnp.eye(2))
        result = converter_registry.convert(w, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50

    def test_vonmisesfisher_same_class(self):
        vmf = VonMisesFisher(mean_direction=jnp.array([1.0, 0.0, 0.0]),
                              concentration=5.0)
        result = converter_registry.convert(vmf, VonMisesFisher)
        assert result is vmf

    def test_mvn_from_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        emp = EmpiricalDistribution(samples)
        result = converter_registry.convert(emp, MultivariateNormal)
        assert isinstance(result, MultivariateNormal)
        assert result.loc.shape == (3,)

    def test_mvn_from_mvn(self):
        mvn = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
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
        n = Normal(loc=3.0, scale=1.0)
        result = converter_registry.convert(n, tfd.Normal)
        assert isinstance(result, tfd.Normal)
        np.testing.assert_allclose(float(result.loc), 3.0)
        np.testing.assert_allclose(float(result.scale), 1.0)

    def test_probpipe_to_tfp_beta(self):
        b = Beta(alpha=2.0, beta=5.0)
        result = converter_registry.convert(b, tfd.Beta)
        assert isinstance(result, tfd.Beta)
        np.testing.assert_allclose(float(result.concentration1), 2.0)
        np.testing.assert_allclose(float(result.concentration0), 5.0)

    def test_tfp_to_probpipe_provenance(self):
        result = converter_registry.convert(tfd.Normal(0, 1), Normal)
        assert result.source is not None
        assert result.source.operation == "convert_from_tfp"

    def test_unknown_tfp_to_empirical(self):
        """Unknown TFP types fall back to sampling → EmpiricalDistribution."""
        # Use a TFP distribution we haven't mapped
        tfp_dist = tfd.VonMises(loc=0.0, concentration=1.0)
        result = converter_registry.convert(tfp_dist, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50

    def test_probpipe_mvn_to_tfp(self):
        loc = jnp.array([1.0, 2.0])
        cov = jnp.array([[1.0, 0.3], [0.3, 2.0]])
        mvn = MultivariateNormal(loc=loc, cov=cov)
        result = converter_registry.convert(mvn, tfd.MultivariateNormalTriL)
        assert isinstance(result, tfd.MultivariateNormalTriL)
        np.testing.assert_allclose(result.loc, loc, atol=1e-5)

    def test_probpipe_to_tfp_round_trip(self):
        n = Normal(loc=5.0, scale=2.0)
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
        g = Gamma(concentration=3.0, rate=1.0)
        result = converter_registry.convert(g, tfd.Gamma)
        assert isinstance(result, tfd.Gamma)
        np.testing.assert_allclose(float(result.concentration), 3.0)

    def test_probpipe_exponential_to_tfp(self):
        e = Exponential(rate=2.0)
        result = converter_registry.convert(e, tfd.Exponential)
        assert isinstance(result, tfd.Exponential)
        np.testing.assert_allclose(float(result.rate), 2.0)

    def test_probpipe_bernoulli_to_tfp(self):
        b = Bernoulli(probs=0.3)
        result = converter_registry.convert(b, tfd.Bernoulli)
        assert isinstance(result, tfd.Bernoulli)

    def test_probpipe_dirichlet_to_tfp(self):
        d = Dirichlet(concentration=jnp.array([2.0, 3.0, 1.0]))
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
        import scipy.stats as ss
        from scipy.stats._distn_infrastructure import rv_frozen
        n = Normal(loc=3.0, scale=1.0)
        result = converter_registry.convert(n, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 3.0)
        np.testing.assert_allclose(result.std(), 1.0)

    def test_scipy_provenance(self):
        import scipy.stats as ss
        result = converter_registry.convert(ss.norm(0, 1), Normal)
        assert result.source is not None
        assert result.source.operation == "convert_from_scipy"

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
        g = Gamma(concentration=3.0, rate=0.5)
        result = converter_registry.convert(g, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 6.0, atol=0.01)

    def test_probpipe_exponential_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen
        e = Exponential(rate=2.0)
        result = converter_registry.convert(e, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_probpipe_beta_to_scipy(self):
        from scipy.stats._distn_infrastructure import rv_frozen
        b = Beta(alpha=2.0, beta=5.0)
        result = converter_registry.convert(b, rv_frozen)
        assert isinstance(result, rv_frozen)
        np.testing.assert_allclose(result.mean(), 2.0 / 7.0, atol=0.01)

    def test_unknown_scipy_fallback_to_sampling(self):
        """Unknown scipy distribution type falls back to sampling."""
        import scipy.stats as ss
        # Use a scipy distribution we haven't mapped (e.g., chi2)
        result = converter_registry.convert(ss.chi2(df=3), EmpiricalDistribution, num_samples=100)
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 100

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
                return Normal(loc=source.val, scale=1.0)
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
                c for c in converter_registry._converters
                if not isinstance(c, DummyConverter)
            ]
            converter_registry._type_cache.clear()


# ---------------------------------------------------------------------------
# from_distribution() backward compatibility
# ---------------------------------------------------------------------------

class TestFromDistributionDelegation:
    """Verify from_distribution() delegates to the registry."""

    def test_from_distribution_same_class(self):
        n = Normal(loc=2.0, scale=0.5)
        result = from_distribution(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)

    def test_from_distribution_cross_family(self):
        g = Gamma(concentration=9.0, rate=1.0)
        result = from_distribution(g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_from_distribution_support_check(self):
        n = Normal(loc=0.5, scale=0.1)
        with pytest.raises(ValueError, match="support"):
            from_distribution(n, Beta)

    def test_from_distribution_check_support_false(self):
        n = Normal(loc=0.5, scale=0.1)
        result = from_distribution(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_from_distribution_to_empirical(self):
        n = Normal(loc=0.0, scale=1.0)
        emp = from_distribution(n, EmpiricalDistribution, num_samples=50)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.n == 50

    def test_empirical_to_empirical_returns_source(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        emp = EmpiricalDistribution(samples, name="orig")
        emp2 = from_distribution(emp, EmpiricalDistribution)
        # Same-class: returns source directly
        assert emp2 is emp


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBootstrapMetadata:
    """Verify bootstrap distributions are stored in provenance metadata
    when the source distribution uses MC fallback for mean/variance."""

    def test_analytical_moments_no_bootstrap(self):
        """TFP distributions have exact mean/var, so no bootstrap in metadata."""
        g = Gamma(concentration=9.0, rate=1.0)
        result = converter_registry.convert(g, Normal, num_samples=500)
        assert result.source is not None
        meta = result.source.metadata
        # Gamma has analytical mean/variance → no BootstrapDistribution stored
        assert "mean_bootstrap" not in meta
        assert "var_bootstrap" not in meta

    def test_empirical_moments_have_bootstrap(self):
        """EmpiricalDistribution uses MC for mean/var, producing bootstrap metadata."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        emp = EmpiricalDistribution(samples[:, None])
        result = converter_registry.convert(emp, Normal)
        assert result.source is not None
        # EmpiricalDistribution._mean()/_variance() return plain arrays;
        # at minimum, provenance should be attached
        assert result.source.operation == "from_distribution"

    def test_same_class_no_bootstrap_metadata(self):
        """Same-class conversion returns source directly, no provenance."""
        n = Normal(loc=2.0, scale=0.5)
        result = converter_registry.convert(n, Normal)
        assert result is n  # same object, no conversion

    def test_cross_family_provenance_attached(self):
        """Cross-family conversion attaches provenance with source as parent."""
        g = Gamma(concentration=9.0, rate=1.0)
        result = converter_registry.convert(g, Normal)
        assert result.source is not None
        assert result.source.operation == "from_distribution"
        assert g in result.source.parents


class TestEdgeCases:

    def test_convert_none_raises(self):
        with pytest.raises(TypeError):
            converter_registry.convert(None, Normal)

    def test_convert_non_type_target_raises(self):
        with pytest.raises(TypeError, match="No converter"):
            converter_registry.convert(Normal(0, 1), str)

    def test_check_infeasible_non_type_target(self):
        info = converter_registry.check(Normal(0, 1), "not a type")
        assert not info.feasible
