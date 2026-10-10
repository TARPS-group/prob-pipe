"""Tests for the shipped converters and the global converter registry.

The registry returns a law that already satisfies the target as it is. The
shipped converters bring TFP and SciPy distributions into ProbPipe exactly, and
fit a family by its moments, sample an empirical law, or smooth a kernel
density estimate approximately; every conversion carries the source's event
declaration and records the converter in the result's provenance.
"""

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
    Converter,
    EmpiricalDistribution,
    Exponential,
    Gamma,
    KDEDistribution,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OpaqueBatch,
    OutputSpec,
    Poisson,
    ResolutionError,
    convert,
    converter_registry,
    function,
    workflow_run,
)
from probpipe.distributions import ConverterRegistry
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
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
from probpipe.families._converters import _MomentMatching
from probpipe.families._discrete import Binomial, NegativeBinomial
from probpipe.families._multivariate import Dirichlet, Multinomial, VonMisesFisher, Wishart
from tests._posterior import posterior_of

# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


class TestConverterRegistry:
    def test_the_shipped_converters_are_registered(self):
        assert set(converter_registry.list_methods()) >= {
            "tfp",
            "moment_match",
            "empirical",
            "kde",
        }

    def test_check_returns_conversioninfo(self):
        info = converter_registry.check(Normal("x", 0, 1), Normal)
        assert isinstance(info, ConversionInfo)
        assert info.feasible

    def test_check_infeasible_for_unknown_target(self):
        info = converter_registry.check(Normal("x", 0, 1), int)
        assert not info.feasible

    def test_convert_raises_for_unknown(self):
        with pytest.raises(ResolutionError, match=r"\(int, Normal\)"):
            converter_registry.convert(42, Normal)

    def test_is_distribution_type_probpipe(self):
        assert converter_registry.is_distribution_type(Normal("x", 0, 1))
        assert converter_registry.is_distribution_type(
            EmpiricalDistribution(jnp.ones((5, 1)), component="x")
        )

    def test_is_distribution_type_tfp(self):
        assert converter_registry.is_distribution_type(tfd.Normal(0, 1))

    def test_is_distribution_type_non_dist(self):
        assert not converter_registry.is_distribution_type(42)
        assert not converter_registry.is_distribution_type("hello")

    def test_event_spec_for_a_probpipe_law_raises_type_error(self):
        with pytest.raises(
            TypeError, match=r"'Normal' is a ProbPipe distribution .* remove the event_spec"
        ):
            converter_registry.convert(
                Normal("a", 0.0, 1.0), EmpiricalDistribution, event_spec=OutputSpec(a=None)
            )

    def test_a_backend_law_of_another_family_names_its_family(self):
        with pytest.raises(ResolutionError, match="the ProbPipe family for TFP's Normal is Normal"):
            converter_registry.convert(tfd.Normal(1.0, 1.0), Gamma)

    def test_an_option_no_converter_reads_raises_type_error(self):
        with pytest.raises(
            TypeError, match=r"unknown option 'bandwidth'.*\(converter 'moment_match'\)"
        ):
            converter_registry.convert(Laplace("g", 9.0, 1.0), Normal, bandwidth=0.5)


# ---------------------------------------------------------------------------
# Moment matching between ProbPipe laws
# ---------------------------------------------------------------------------


class TestMomentMatching:
    def test_same_class_needs_no_converter(self):
        n = Normal("x", loc=2.0, scale=0.5)
        info = converter_registry.check(n, Normal)
        assert (info.method_name, info.exact, info.samples) == (None, True, False)

        result = converter_registry.convert(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)
        np.testing.assert_allclose(float(result._scale), 0.5)

    def test_cross_family_moment_match(self):
        g = Laplace("g", loc=9.0, scale=1.0)
        info = converter_registry.check(g, Normal)
        assert (info.method_name, info.exact, info.samples) == ("moment_match", False, False)

        result = converter_registry.convert(g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_the_fit_keeps_the_source_label_and_component(self):
        g = Laplace("theta", loc=9.0, scale=1.0, label="g")
        result = converter_registry.convert(g, Normal)
        assert result.label == "g"
        assert list(result.event_spec.components) == ["theta"]

    def test_support_mismatch_raises_by_default(self):
        """A fit to a family on another support is infeasible at check, before any fitting."""
        n = Normal("x", loc=0.5, scale=0.1)
        assert converter_registry.check(n, Beta).feasible is False
        with pytest.raises(ResolutionError, match="check_support=False"):
            converter_registry.convert(n, Beta)

    def test_a_wider_support_is_refused_as_well(self):
        with pytest.raises(ResolutionError, match="declares the support positive"):
            converter_registry.convert(Gamma("g", 9.0, 1.0), Normal)

    def test_support_mismatch_override(self):
        n = Normal("x", loc=0.5, scale=0.1)
        result = converter_registry.convert(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_to_empirical(self):
        n = Normal("x", loc=0.0, scale=1.0)
        info = converter_registry.check(n, EmpiricalDistribution)
        assert (info.method_name, info.exact, info.samples) == ("empirical", False, True)
        emp = converter_registry.convert(n, EmpiricalDistribution, num_samples=100)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.num_atoms == 100
        assert emp.event_spec == n.event_spec
        assert emp.atoms.level_names == ("sample",)

    def test_provenance_attached(self):
        g = Laplace("x", loc=3.0, scale=1.0, label="prior")
        result = converter_registry.convert(g, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert"
        assert result.provenance.metadata == {"converter": "moment_match", "exact": False}
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].label == "prior"

    def test_same_class_returns_source(self):
        """Same-class conversion returns the source object itself."""
        n = Normal("x", loc=1.0, scale=2.0)
        result = converter_registry.convert(n, Normal)
        assert result is n

    def test_a_target_that_names_no_family_is_not_moment_matched(self):
        """Moment matching fits a parametric family, which a protocol is not."""
        emp = EmpiricalDistribution(jnp.array([0.5, 1.0, 2.0]), component="x")
        info = converter_registry.check(emp, SupportsLogProb, method="moment_match")
        assert info.feasible is False
        assert "SupportsLogProb is not a parametric family" in info.description


class TestAnEmpiricalSourceAgainstTheTargetSupport:
    """An empirical law's array atoms declare no support, so its atoms are checked instead."""

    def test_atoms_outside_the_target_support_raise(self):
        source = EmpiricalDistribution(jnp.array([-5.0, -1.0, 2.0, -3.0]), component="x")
        with pytest.raises(ValueError, match="support"):
            convert(source, Exponential)

    def test_atoms_inside_the_target_support_convert(self):
        source = EmpiricalDistribution(jnp.array([0.5, 1.0, 2.0, 3.0]), component="x")
        result = convert(source, Exponential)
        assert isinstance(result, Exponential)
        np.testing.assert_allclose(float(result._mean()), 1.625, rtol=1e-6)

    def test_the_check_can_be_overridden(self):
        source = EmpiricalDistribution(jnp.array([-5.0, -1.0, 2.0, -3.0]), component="x")
        assert isinstance(
            converter_registry.convert(source, Exponential, check_support=False), Exponential
        )


class TestARecordSource:
    """A family draws one array, so a law over a record converts through a field's law."""

    @staticmethod
    def _posterior() -> EmpiricalDistribution:
        lam = jnp.array([2.2, 2.5, 2.7, 2.4])
        spec = NumericRecordSpec(lam=NumericArraySpec((), lam.dtype))
        atoms = NumericRecordBatch(
            {"lam": lam},
            "draw",
            element_spec=spec,
            label="atoms",
        )
        return EmpiricalDistribution(atoms, label="posterior")

    def test_a_record_law_does_not_convert_to_a_family(self):
        """The conversion would change the packaging, which a conversion preserves."""
        with pytest.raises(
            ResolutionError,
            match=r"draws a record with fields \['lam'\]; convert one field instead",
        ):
            converter_registry.convert(self._posterior(), Normal)

    def test_its_field_law_matches_a_scalar_family(self):
        result = converter_registry.convert(self._posterior()["lam"], Normal)
        lam = jnp.array([2.2, 2.5, 2.7, 2.4])
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), float(jnp.mean(lam)), rtol=1e-6)
        np.testing.assert_allclose(float(result._scale), float(jnp.std(lam)), rtol=1e-5)

    def test_its_field_law_draws_fit_a_scalar_family(self):
        with workflow_run(seed=0):
            result = converter_registry.convert(
                self._posterior()["lam"], HalfCauchy, num_samples=200
            )
        assert isinstance(result, HalfCauchy)
        assert 2.2 <= float(result._scale) <= 2.7

    def test_a_parameter_naming_the_family_converts_the_field_law(self):
        @function
        def location(d: Normal):
            return d._loc

        lam = jnp.array([2.2, 2.5, 2.7, 2.4])
        np.testing.assert_allclose(
            float(location(self._posterior()["lam"])), float(jnp.mean(lam)), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Cross-family moment-matching (exercises every family's fit)
# ---------------------------------------------------------------------------


class TestAllCrossFamilyConversions:
    """Exercise every family's fit with a cross-family source."""

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
        g = Gamma("g", concentration=9.0, rate=1.0)
        result = converter_registry.convert(g, target_cls, check_support=False, num_samples=500)
        assert isinstance(result, target_cls)
        assert result.provenance is not None  # cross-family: provenance attached

    def test_bernoulli_from_poisson(self):
        p = Poisson("p", rate=0.5)
        result = converter_registry.convert(p, Bernoulli, check_support=False)
        assert isinstance(result, Bernoulli)

    def test_binomial_from_poisson(self):
        p = Poisson("p", rate=3.0)
        result = converter_registry.convert(p, Binomial, check_support=False, total_count=10)
        assert isinstance(result, Binomial)

    def test_binomial_requires_total_count(self):
        p = Poisson("p", rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, Binomial, check_support=False)

    def test_poisson_from_bernoulli_is_infeasible(self):
        """A Poisson draws floats, which do not cast to the Bernoulli's integers."""
        b = Bernoulli("b", probs=0.3)
        with pytest.raises(ResolutionError, match="does not cast"):
            converter_registry.convert(b, Poisson, check_support=False)

    def test_a_fit_whose_draws_do_not_cast_is_reported_infeasible_at_check(self):
        """A Normal draws floats, which do not cast to the Bernoulli's integers."""
        b = Bernoulli("b", probs=0.3)
        info = converter_registry.check(b, Normal)
        assert info.feasible is False
        assert "moment_match" in info.description
        assert "does not cast" in info.description
        assert info.target_spec is None
        with pytest.raises(ResolutionError, match="does not cast"):
            converter_registry.convert(b, Normal)

    def test_the_registry_tries_the_next_converter_after_an_infeasible_fit(self):
        """A converter that follows moment matching in selection order is selected."""

        class FollowingConverter(Converter):
            @property
            def name(self):
                return "following"

            @property
            def exact(self):
                return False

            @property
            def priority(self):
                return 0

            def supported_types(self):
                return ((Bernoulli,), (Normal,))

            def check(self, source, target_type, **options):
                return ConversionInfo(
                    True,
                    method_name=self.name,
                    exact=False,
                    target_spec=source.spec,
                    target_class=Normal,
                )

            def execute(self, source, target_type, **options):
                raise AssertionError("the test only checks")

        registry = ConverterRegistry()
        registry.register(_MomentMatching())
        registry.register(FollowingConverter())
        info = registry.check(Bernoulli("b", probs=0.3), Normal)
        assert info.feasible is True
        assert info.method_name == "following"

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            pytest.param(lambda: Gamma("g", concentration=2.0, rate=1.0), Normal, id="float"),
            pytest.param(lambda: Normal("n", loc=0.5, scale=0.1), Bernoulli, id="integer"),
        ],
    )
    def test_the_promise_declares_the_dtype_of_the_fit(self, source, target):
        law = source()
        info = converter_registry.check(law, target, check_support=False)
        result = converter_registry.convert(law, target, check_support=False)
        assert info.target_spec.event_spec.spec.dtype == result.event_spec.spec.dtype

    def test_categorical_from_bernoulli(self):
        b = Bernoulli("b", probs=0.7)
        result = converter_registry.convert(b, Categorical, check_support=False, num_samples=500)
        assert isinstance(result, Categorical)

    def test_negativebinomial_from_poisson(self):
        p = Poisson("p", rate=3.0)
        result = converter_registry.convert(p, NegativeBinomial, check_support=False, total_count=5)
        assert isinstance(result, NegativeBinomial)

    def test_negativebinomial_requires_total_count(self):
        p = Poisson("p", rate=3.0)
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(p, NegativeBinomial, check_support=False)

    def test_dirichlet_from_mvn(self):
        mvn = MultivariateNormal("z", loc=jnp.array([0.3, 0.5, 0.2]), cov=0.01 * jnp.eye(3))
        result = converter_registry.convert(mvn, Dirichlet, check_support=False, num_samples=500)
        assert isinstance(result, Dirichlet)

    def test_multinomial_from_mvn(self):
        mvn = MultivariateNormal("z", loc=jnp.array([3.0, 5.0, 2.0]), cov=jnp.eye(3))
        result = converter_registry.convert(
            mvn, Multinomial, check_support=False, total_count=10, num_samples=500
        )
        assert isinstance(result, Multinomial)

    def test_multinomial_requires_total_count(self):
        mvn = MultivariateNormal("z", loc=jnp.array([3.0, 5.0]), cov=jnp.eye(2))
        with pytest.raises(ValueError, match="total_count"):
            converter_registry.convert(mvn, Multinomial, check_support=False)

    def test_wishart_from_wishart(self):
        w = Wishart("w", df=5.0, scale_tril=jnp.eye(2))
        result = converter_registry.convert(w, Wishart)
        assert result is w  # same-class

    def test_wishart_to_empirical(self):
        w = Wishart("w", df=5.0, scale_tril=jnp.eye(2))
        result = converter_registry.convert(w, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 50

    def test_vonmisesfisher_same_class(self):
        vmf = VonMisesFisher("vmf", mean_direction=jnp.array([1.0, 0.0, 0.0]), concentration=5.0)
        result = converter_registry.convert(vmf, VonMisesFisher)
        assert result is vmf

    def test_mvn_from_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        emp = EmpiricalDistribution(samples, component="x")
        info = converter_registry.check(emp, MultivariateNormal)
        assert info.samples is False  # the empirical law's moments are in closed form
        result = converter_registry.convert(emp, MultivariateNormal)
        assert isinstance(result, MultivariateNormal)
        assert result.loc.shape == (3,)

    def test_mvn_from_mvn(self):
        mvn = MultivariateNormal("z", loc=jnp.zeros(2), cov=jnp.eye(2))
        result = converter_registry.convert(mvn, MultivariateNormal)
        assert result is mvn

    def test_a_vector_family_refuses_a_matrix_event(self):
        w = Wishart("w", df=5.0, scale_tril=jnp.eye(2))
        with pytest.raises(ResolutionError, match="rank 1"):
            converter_registry.convert(w, MultivariateNormal)


# ---------------------------------------------------------------------------
# TFP backend distributions entering ProbPipe
# ---------------------------------------------------------------------------


class TestTFPConverter:
    def test_tfp_normal_to_probpipe(self):
        tfp_n = tfd.Normal(loc=2.0, scale=0.5)
        info = converter_registry.check(tfp_n, Normal)
        assert info.feasible
        assert (info.method_name, info.exact, info.samples) == ("tfp", True, False)

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

    def test_an_event_spec_option_names_the_component(self):
        result = converter_registry.convert(
            tfd.Normal(0.0, 1.0), Normal, event_spec=OutputSpec(mu=None)
        )
        assert list(result.event_spec.components) == ["mu"]

    def test_tfp_to_probpipe_provenance(self):
        result = converter_registry.convert(tfd.Normal(0, 1), Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert"
        assert result.provenance.metadata == {"converter": "tfp", "exact": True}

    def test_an_unknown_tfp_distribution_enters_through_the_backend_adapter(self):
        """A backend distribution without a family is wrapped exactly, not sampled."""
        from probpipe import TFPDistribution

        tfp_dist = tfd.VonMises(loc=0.0, concentration=1.0)
        result = converter_registry.convert(tfp_dist, SupportsLogProb)
        assert type(result) is TFPDistribution
        assert result.raw() is tfp_dist

    def test_unknown_tfp_to_empirical(self):
        """An unknown TFP distribution samples through the backend adapter."""
        tfp_dist = tfd.VonMises(loc=0.0, concentration=1.0)
        result = converter_registry.convert(tfp_dist, EmpiricalDistribution, num_samples=50)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 50

    def test_a_family_exports_its_backend_distribution_through_raw(self):
        """Converting to a backend class is no conversion, since the result carries no declaration."""
        n = Normal("x", loc=3.0, scale=1.0)
        assert isinstance(n.raw(), tfd.Normal)
        np.testing.assert_allclose(float(n.raw().loc), 3.0)
        with pytest.raises(ResolutionError):
            converter_registry.convert(n, tfd.Normal)

    @pytest.mark.parametrize(
        ("law", "backend"),
        [
            (Beta("b", alpha=2.0, beta=5.0), tfd.Beta),
            (Gamma("g", concentration=3.0, rate=1.0), tfd.Gamma),
            (Exponential("e", rate=2.0), tfd.Exponential),
            (Bernoulli("b", probs=0.3), tfd.Bernoulli),
            (Dirichlet("d", concentration=jnp.array([2.0, 3.0, 1.0])), tfd.Dirichlet),
            (
                MultivariateNormal("z", loc=jnp.array([1.0, 2.0]), cov=jnp.eye(2)),
                tfd.MultivariateNormalTriL,
            ),
        ],
        ids=["beta", "gamma", "exponential", "bernoulli", "dirichlet", "mvn"],
    )
    def test_each_family_exports_its_backend_distribution(self, law, backend):
        assert isinstance(law.raw(), backend)

    def test_probpipe_to_tfp_round_trip(self):
        n = Normal("x", loc=5.0, scale=2.0)
        n2 = converter_registry.convert(n.raw(), Normal)
        np.testing.assert_allclose(float(n2._loc), 5.0)
        np.testing.assert_allclose(float(n2._scale), 2.0)

    def test_tfp_cross_family_chain(self):
        """A TFP Laplace moment-matches to a ProbPipe Normal as the Laplace it enters as."""
        tfp_g = tfd.Laplace(loc=9.0, scale=1.0)
        info = converter_registry.check(tfp_g, Normal)
        assert (info.method_name, info.exact) == ("moment_match", False)
        result = converter_registry.convert(tfp_g, Normal, num_samples=5000)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

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
# SciPy frozen distributions entering ProbPipe (optional)
# ---------------------------------------------------------------------------


class TestScipyConverter:
    @pytest.fixture(autouse=True)
    def _check_scipy(self):
        pytest.importorskip("scipy")

    def test_scipy_norm_to_probpipe(self):
        import scipy.stats as ss

        info = converter_registry.check(ss.norm(loc=1.0, scale=2.0), Normal)
        assert (info.method_name, info.exact) == ("scipy", True)
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

    def test_a_conversion_to_a_scipy_class_is_refused(self):
        """A SciPy distribution carries no event declaration, so it is no conversion target."""
        from scipy.stats._distn_infrastructure import rv_frozen

        with pytest.raises(ResolutionError):
            converter_registry.convert(Normal("x", loc=3.0, scale=1.0), rv_frozen)

    def test_scipy_provenance(self):
        import scipy.stats as ss

        result = converter_registry.convert(ss.norm(0, 1), Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert"
        assert result.provenance.metadata == {"converter": "scipy", "exact": True}

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

    def test_unknown_scipy_fallback_to_sampling(self):
        """A SciPy distribution without a family samples through SciPy."""
        import scipy.stats as ss

        result = converter_registry.convert(ss.chi2(df=3), EmpiricalDistribution, num_samples=100)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 100

    def test_scipy_check_unknown_type(self):
        """A SciPy distribution without a family moment-matches from its draws."""
        import scipy.stats as ss

        info = converter_registry.check(ss.chi2(df=3), Normal)
        assert info.feasible
        assert (info.method_name, info.exact, info.samples) == ("moment_match", False, True)
        with workflow_run(seed=0):
            result = converter_registry.convert(ss.chi2(df=3), Normal, num_samples=4000)
        np.testing.assert_allclose(float(result._loc), 3.0, atol=0.2)

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
            @property
            def name(self):
                return "dummy"

            @property
            def exact(self):
                return True

            @property
            def priority(self):
                return 10

            def supported_types(self):
                return ((DummyDist,), (Normal,))

            def check(self, source, target_type, **options):
                return ConversionInfo(
                    feasible=True,
                    method_name=self.name,
                    exact=True,
                    target_spec=Normal("x", loc=0.0, scale=1.0).spec,
                    target_class=Normal,
                )

            def execute(self, source, target_type, **options):
                return Normal("x", loc=source.val, scale=1.0)

        registry = ConverterRegistry()
        registry.register(DummyConverter())
        d = DummyDist(42.0)
        assert registry.is_distribution_type(d)
        result = registry.convert(d, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 42.0)
        assert result.provenance.metadata == {"converter": "dummy", "exact": True}


# ---------------------------------------------------------------------------
# convert delegates to the registry
# ---------------------------------------------------------------------------


class TestConvertDelegation:
    """Verify convert() delegates to the registry."""

    def test_convert_same_class(self):
        n = Normal("x", loc=2.0, scale=0.5)
        result = convert(n, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 2.0)

    def test_convert_cross_family(self):
        t = StudentT("t", df=30.0, loc=9.0, scale=1.0)
        result = convert.with_options(method_options={"num_samples": 5000})(t, Normal)
        assert isinstance(result, Normal)
        np.testing.assert_allclose(float(result._loc), 9.0, atol=0.5)

    def test_convert_support_check(self):
        n = Normal("x", loc=0.5, scale=0.1)
        assert convert.check(n, Beta).route is None
        with pytest.raises(ResolutionError, match="check_support=False"):
            convert(n, Beta)

    def test_convert_check_support_false(self):
        n = Normal("x", loc=0.5, scale=0.1)
        result = converter_registry.convert(n, Beta, check_support=False)
        assert isinstance(result, Beta)

    def test_convert_to_empirical(self):
        n = Normal("x", loc=0.0, scale=1.0)
        emp = convert.with_options(method_options={"num_samples": 50})(n, EmpiricalDistribution)
        assert isinstance(emp, EmpiricalDistribution)
        assert emp.num_atoms == 50

    def test_empirical_to_empirical_returns_under_fresh_identity(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        emp = EmpiricalDistribution(samples, component="orig")
        emp2 = convert(emp, EmpiricalDistribution)
        assert emp2 is not emp
        assert emp2.provenance.operation == "workflow.convert"


# ---------------------------------------------------------------------------
# Provenance and edge cases
# ---------------------------------------------------------------------------


class TestConversionProvenance:
    """A conversion records its source and its converter in provenance."""

    def test_empirical_moments_record_the_source(self):
        """Moment matching an empirical law records the law as its parent."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        emp = EmpiricalDistribution(samples, component="x", label="x")
        result = converter_registry.convert(emp, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert"
        assert [parent.label for parent in result.provenance.parents] == ["x"]

    def test_same_class_records_nothing(self):
        """Same-class conversion returns source directly, no provenance."""
        n = Normal("x", loc=2.0, scale=0.5)
        result = converter_registry.convert(n, Normal)
        assert result is n  # same object, no conversion

    def test_cross_family_provenance_attached(self):
        """Cross-family conversion attaches provenance with source as parent."""
        g = Laplace("x", loc=9.0, scale=1.0, label="g")
        result = converter_registry.convert(g, Normal)
        assert result.provenance is not None
        assert result.provenance.operation == "convert"
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].label == "g"


class TestEdgeCases:
    def test_convert_none_raises(self):
        with pytest.raises(ResolutionError, match="NoneType"):
            converter_registry.convert(None, Normal)

    def test_convert_non_type_target_raises(self):
        with pytest.raises(ResolutionError, match="no method is registered"):
            converter_registry.convert(Normal("x", 0, 1), str)

    def test_check_non_type_target_raises(self):
        with pytest.raises(TypeError, match="class or a protocol"):
            converter_registry.check(Normal("x", 0, 1), "not a type")


# ---------------------------------------------------------------------------
# Capability targets
# ---------------------------------------------------------------------------


class TestProtocolConversion:
    """Tests for converting distributions to satisfy a protocol."""

    def test_already_satisfies_returns_same(self):
        """Distribution that already satisfies the protocol is returned unchanged."""
        n = Normal("x", loc=0.0, scale=1.0)
        result = converter_registry.convert(n, SupportsLogProb)
        assert result is n

    def test_scalar_empirical_to_supports_log_prob(self):
        """Scalar EmpiricalDistribution converts to KDE via SupportsLogProb."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (300,))
        emp = EmpiricalDistribution(samples, component="x")
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
        """An empirical law over vectors converts to a KDE over them."""
        samples = jax.random.normal(jax.random.PRNGKey(1), (300, 4))
        emp = EmpiricalDistribution(samples, component="x")
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
            {
                "mu": jax.random.normal(jax.random.PRNGKey(2), (n,)),
                "log_sigma": jax.random.normal(jax.random.PRNGKey(3), (n,)),
            },
            "row",
            element_spec=NumericRecordSpec(mu=(), log_sigma=()),
            label="r",
        )
        emp = EmpiricalDistribution(rows, label="emp")
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.num_atoms == n
        # log_prob of one record returns a scalar.
        lp = result._log_prob({"mu": 0.0, "log_sigma": 0.0})
        assert lp.shape == ()

    def test_single_field_record_empirical_to_kde_unchanged(self):
        """An empirical law over scalars converts to a KDE whose event is a scalar."""
        samples = jax.random.normal(jax.random.PRNGKey(4), (150,))
        emp = EmpiricalDistribution(samples, component="theta")
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.event_shape == ()

    def test_weighted_single_field_empirical_to_kde_preserves_weights(self):
        """An empirical law with non-uniform weights converts to a KDE with the same weights."""
        n = 80
        samples = jax.random.normal(jax.random.PRNGKey(5), (n,))
        weights = jnp.linspace(0.1, 1.0, n)
        emp = EmpiricalDistribution(samples, weights=weights, component="x")
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        np.testing.assert_allclose(
            np.asarray(result._w.normalized), np.asarray(emp.weights), atol=1e-6
        )
        assert not result._w.is_uniform

    def test_object_array_empirical_to_kde_rejected(self):
        """An empirical law over opaque atoms does not convert to a KDE, which smooths numbers."""
        emp = EmpiricalDistribution(
            OpaqueBatch(
                ["a", "b", "c"],
                "site",
                label="labels",
            ),
            component="emp",
        )
        with pytest.raises(ResolutionError, match="numeric"):
            converter_registry.convert(emp, KDEDistribution)

    def test_check_protocol_already_satisfied(self):
        """check() reports an exact report selecting no converter when the protocol holds."""
        n = Normal("x", loc=0.0, scale=1.0)
        info = converter_registry.check(n, SupportsLogProb)
        assert info.feasible
        assert (info.method_name, info.exact) == (None, True)

    def test_check_protocol_needs_conversion(self):
        """check() reports the kernel density estimate for an empirical law."""
        samples = jax.random.normal(jax.random.PRNGKey(2), (100,))
        emp = EmpiricalDistribution(samples, component="x")
        info = converter_registry.check(emp, SupportsLogProb)
        assert info.feasible
        assert (info.method_name, info.exact, info.samples) == ("kde", False, False)

    def test_a_law_that_samples_converts_to_a_moment_by_its_empirical_law(self):
        """A converter whose result claims the capability serves a request for it."""
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import BijectorTransformedDistribution

        law = BijectorTransformedDistribution("y", Normal("x", 0.0, 1.0), tfb.Exp())
        assert not isinstance(law, SupportsMean)
        info = converter_registry.check(law, SupportsMean)
        assert (info.method_name, info.samples) == ("empirical", True)
        assert isinstance(
            converter_registry.convert(law, SupportsMean, num_samples=20), SupportsMean
        )

    def test_unregistered_protocol_raises(self):
        """A protocol no converter's result claims raises ResolutionError."""
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class SupportsSomethingUnregistered(Protocol):
            def _something_unregistered(self) -> None: ...

        samples = jax.random.normal(jax.random.PRNGKey(3), (50,))
        emp = EmpiricalDistribution(samples, component="x")
        with pytest.raises(ResolutionError):
            converter_registry.convert(emp, SupportsSomethingUnregistered)

    def test_exact_conditioning_is_no_converter_target(self):
        with pytest.raises(ResolutionError):
            converter_registry.convert(Normal("x", 0.0, 1.0), SupportsExactConditioning)

    def test_convert_with_protocol(self):
        """convert() works with protocol targets."""
        samples = jax.random.normal(jax.random.PRNGKey(4), (200,))
        emp = EmpiricalDistribution(samples, component="x")
        result = convert(emp, SupportsLogProb)
        assert isinstance(result, SupportsLogProb)

    def test_protocol_conversion_preserves_provenance(self):
        """Protocol-based conversion attaches provenance."""
        samples = jax.random.normal(jax.random.PRNGKey(5), (200,))
        emp = EmpiricalDistribution(samples, component="x", label="posterior")
        result = converter_registry.convert(emp, SupportsLogProb)
        assert result.provenance is not None
        assert len(result.provenance.parents) == 1
        assert result.provenance.parents[0].label == "posterior"

    def test_multi_field_empirical_preserves_template_through_kde(self):
        """An empirical law over a record converts to a KDE over that record's fields."""
        n = 200
        rows = NumericRecordBatch(
            {
                "intercept": jax.random.normal(jax.random.PRNGKey(0), (n,)),
                "slope": jax.random.normal(jax.random.PRNGKey(1), (n,)),
            },
            "row",
            element_spec=NumericRecordSpec(intercept=(), slope=()),
            label="r",
        )
        emp = EmpiricalDistribution(rows, label="emp")
        result = converter_registry.convert(emp, SupportsLogProb)
        assert isinstance(result, KDEDistribution)
        assert result.event_spec.spec.fields == ("intercept", "slope")

    def test_inference_result_preserves_template_through_kde(self):
        """An inference result converts to a KDE over its target's record."""
        n_draws = 100
        chain_intercept = jax.random.normal(jax.random.PRNGKey(0), (n_draws,))
        chain_slope = jax.random.normal(jax.random.PRNGKey(1), (n_draws,))
        # Stack into per-chain (num_draws, total_dim) layout
        chains = [jnp.stack([chain_intercept, chain_slope], axis=-1)]
        approx = posterior_of(
            chains,
            label="posterior",
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
        kde = KDEDistribution(samples, component="kde")
        assert kde.event_shape == ()
        assert not hasattr(kde, "batch_shape")
        assert kde.num_atoms == 100

    def test_multivariate_construction(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        kde = KDEDistribution(samples, component="kde")
        assert kde.event_shape == (3,)
        assert kde.num_atoms == 100

    def test_log_prob_finite(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        kde = KDEDistribution(samples, component="kde")
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_log_prob_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 2))
        kde = KDEDistribution(samples, component="kde")
        lp = kde._log_prob(jnp.zeros(2))
        assert jnp.isfinite(lp)

    def test_sample_shape_scalar(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution(samples, component="kde")
        s = kde._sample(jax.random.PRNGKey(1), (5,))
        assert s.shape == (5,)

    def test_sample_shape_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        kde = KDEDistribution(samples, component="kde")
        s = kde._sample(jax.random.PRNGKey(1), (5,))
        assert s.shape == (5, 3)

    def test_mean_close_to_sample_mean(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (500,))
        kde = KDEDistribution(samples, component="kde")
        np.testing.assert_allclose(
            float(kde._mean()),
            float(jnp.mean(samples)),
            atol=0.01,
        )

    def test_variance_larger_than_sample_variance(self):
        """KDE variance = sample variance + bandwidth^2, so should be larger."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (500,))
        kde = KDEDistribution(samples, component="kde")
        sample_var = float(jnp.var(samples))
        kde_var = float(kde._variance())
        assert kde_var > sample_var

    def test_weighted_construction(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        weights = jnp.ones(100)
        weights = weights.at[0].set(10.0)
        kde = KDEDistribution(samples, weights=weights, component="kde")
        assert kde.num_atoms == 100
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_custom_bandwidth(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution(samples, bandwidth=0.5, component="kde")
        lp = kde._log_prob(0.0)
        assert jnp.isfinite(lp)

    def test_supports_protocols(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100,))
        kde = KDEDistribution(samples, component="kde")
        assert isinstance(kde, SupportsLogProb)
        assert isinstance(kde, SupportsSampling)
        assert isinstance(kde, SupportsMean)
        assert isinstance(kde, SupportsVariance)
        assert isinstance(kde, SupportsCovariance)

    def test_convert_empirical_to_kde(self):
        """convert(empirical, KDEDistribution) works."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        emp = EmpiricalDistribution(samples, component="x")
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
        n = Normal("x", loc=0.0, scale=1.0)
        kde = converter_registry.convert(n, KDEDistribution, num_samples=500)
        assert isinstance(kde, KDEDistribution)
        assert kde.num_atoms == 500

    def test_cov_scalar(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200,))
        kde = KDEDistribution(samples, component="kde")
        cov = kde._cov()
        assert cov.shape == (1, 1)

    def test_cov_multivariate(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        kde = KDEDistribution(samples, component="kde")
        cov = kde._cov()
        assert cov.shape == (3, 3)

    def test_repr(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (50,))
        kde = KDEDistribution(samples, component="test_kde")
        assert repr(kde) == (
            "KDEDistribution(\n"
            "    component='test_kde',\n"
            "    atoms=array(shape=(50,), dtype=float32),\n"
            "    kernel=GaussianKernel,\n"
            ")"
        )
