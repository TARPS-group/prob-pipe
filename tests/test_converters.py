"""Tests for the converter registry and built-in converters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Normal, Beta, Gamma, Exponential, MultivariateNormal,
    Bernoulli, Poisson, Categorical,
    EmpiricalDistribution, Distribution,
    converter_registry, ConversionInfo, ConversionMethod, Converter,
)


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
        assert info.estimated_error == 0.0

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
        result = Normal.from_distribution(n)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)

    def test_from_distribution_cross_family(self):
        g = Gamma(concentration=9.0, rate=1.0)
        result = Normal.from_distribution(g, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_from_distribution_support_check(self):
        n = Normal(loc=0.5, scale=0.1)
        with pytest.raises(ValueError, match="support"):
            Beta.from_distribution(n)

    def test_from_distribution_check_support_false(self):
        n = Normal(loc=0.5, scale=0.1)
        result = Beta.from_distribution(n, check_support=False)
        assert isinstance(result, Beta)

    def test_from_distribution_to_empirical(self):
        n = Normal(loc=0.0, scale=1.0)
        emp = EmpiricalDistribution.from_distribution(n, num_samples=50)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.n == 50

    def test_empirical_to_empirical_returns_source(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        emp = EmpiricalDistribution(samples, name="orig")
        emp2 = EmpiricalDistribution.from_distribution(emp)
        # Same-class: returns source directly
        assert emp2 is emp


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

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
