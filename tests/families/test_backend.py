"""Contracts of the parametric families (VII.1) and their backend adapter."""

from __future__ import annotations

import asyncio

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe.families as F
from probpipe import (
    Distribution,
    DistributionArray,
    NumericArraySpec,
    NumericDistribution,
    OutputSpec,
    real,
)
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)
from probpipe.families import TFPDistribution
from probpipe.families._backend import _allow_batched_tfp_init
from probpipe.linalg import DiagonalLinOp, LinOp


def _families() -> dict[str, Distribution]:
    return {
        "Normal": F.Normal("a", 0.0, 1.0),
        "Beta": F.Beta("a", 2.0, 3.0),
        "Gamma": F.Gamma("a", 2.0, 1.0),
        "InverseGamma": F.InverseGamma("a", 3.0, 1.0),
        "Exponential": F.Exponential("a", 1.0),
        "LogNormal": F.LogNormal("a", 0.0, 1.0),
        "StudentT": F.StudentT("a", 3.0, 0.0, 1.0),
        "Uniform": F.Uniform("a", 0.0, 1.0),
        "Cauchy": F.Cauchy("a", 0.0, 1.0),
        "Laplace": F.Laplace("a", 0.0, 1.0),
        "HalfNormal": F.HalfNormal("a", 1.0),
        "HalfCauchy": F.HalfCauchy("a", 0.0, 1.0),
        "Pareto": F.Pareto("a", 3.0, 1.0),
        "TruncatedNormal": F.TruncatedNormal("a", 0.0, 1.0, -1.0, 1.0),
        "Bernoulli": F.Bernoulli("a", probs=0.3),
        "Binomial": F.Binomial("a", 5.0, probs=0.3),
        "Poisson": F.Poisson("a", 2.0),
        "Categorical": F.Categorical("a", probs=jnp.array([0.2, 0.8])),
        "NegativeBinomial": F.NegativeBinomial("a", 5.0, probs=0.3),
        "MultivariateNormal": F.MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2)),
        "Dirichlet": F.Dirichlet("a", jnp.ones(3)),
        "Multinomial": F.Multinomial("a", 5, probs=jnp.array([0.2, 0.8])),
        "Wishart": F.Wishart("a", 3.0, scale_tril=jnp.eye(2)),
        "VonMisesFisher": F.VonMisesFisher("a", jnp.array([0.0, 1.0]), 2.0),
    }


_NAMES = list(_families())

_MOMENTS = {SupportsMean, SupportsVariance, SupportsCovariance}

#: The capabilities each family's backend computes beyond sampling and the density.
_EXPECTED = {
    **{
        name: _MOMENTS | {SupportsQuantile}
        for name in (
            "Normal",
            "Beta",
            "Gamma",
            "InverseGamma",
            "Exponential",
            "LogNormal",
            "StudentT",
            "Uniform",
            "Cauchy",
            "Laplace",
            "HalfNormal",
            "HalfCauchy",
            "TruncatedNormal",
            "MultivariateNormal",
        )
    },
    **{
        name: _MOMENTS
        for name in (
            "Pareto",
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Categorical",
            "NegativeBinomial",
            "Dirichlet",
            "Multinomial",
        )
    },
    "Wishart": {SupportsMean, SupportsVariance},
    "VonMisesFisher": {SupportsMean, SupportsCovariance},
}

_PROTOCOLS = {
    "_mean": SupportsMean,
    "_variance": SupportsVariance,
    "_cov": SupportsCovariance,
    "_quantile": SupportsQuantile,
}


class TestTheEventDeclaration:
    def test_event_spec_names_the_component_and_the_label_stays(self):
        prior = F.Normal("prior", 0.0, 1.0, event_spec=OutputSpec(beta=None))
        assert prior.name == "prior"
        assert list(prior.event_spec.components) == ["beta"]
        assert prior.event_spec.spec.shape == ()

    def test_the_component_defaults_to_the_label_captured_at_construction(self):
        law = F.Normal("x", 0.0, 1.0)
        assert list(law.event_spec.components) == ["x"]
        assert list(law.with_name("y").event_spec.components) == ["x"]

    def test_the_event_spec_is_derived_from_the_parameters(self):
        law = F.MultivariateNormal("z", jnp.zeros(3), cov=jnp.eye(3))
        spec = law.event_spec.spec
        assert spec.shape == (3,)
        assert spec.dtype == jnp.zeros(3).dtype
        assert spec.support is not None

    def test_a_declared_type_must_agree_with_the_parameters(self):
        with pytest.raises(ValueError):
            F.Normal("x", 0.0, 1.0, event_spec=OutputSpec(x=NumericArraySpec((3,))))

    @pytest.mark.parametrize("name", _NAMES)
    def test_each_family_is_a_numeric_distribution(self, name):
        assert isinstance(_families()[name], NumericDistribution)


class TestTheAdapter:
    def test_the_adapter_wraps_a_backend_distribution(self):
        law = TFPDistribution("x", tfd.Normal(0.0, 1.0))
        assert law.event_spec.spec.shape == ()
        assert float(law._log_prob(0.0)) == pytest.approx(-0.9189385, rel=1e-6)
        assert not isinstance(law, SupportsMean)

    @pytest.mark.parametrize("name", _NAMES)
    def test_one_draw_matches_the_declaration(self, name):
        law = _families()[name]
        draw = law._sample(jax.random.PRNGKey(0))
        assert tuple(jnp.shape(draw)) == law.event_spec.spec.shape

    @pytest.mark.parametrize("name", _NAMES)
    def test_each_family_claims_exactly_what_its_backend_computes(self, name):
        law = _families()[name]
        claimed = {protocol for protocol in _PROTOCOLS.values() if isinstance(law, protocol)}
        assert claimed == _EXPECTED[name]
        assert type(law)._backend_capabilities == _EXPECTED[name]

    @pytest.mark.parametrize("name", _NAMES)
    @pytest.mark.parametrize("method", ["_mean", "_variance", "_cov"])
    def test_each_claimed_moment_computes(self, name, method):
        law = _families()[name]
        if not isinstance(law, _PROTOCOLS[method]):
            pytest.skip(f"{name} does not claim {method}")
        getattr(law, method)()

    @pytest.mark.parametrize("name", [n for n in _NAMES if SupportsQuantile in _EXPECTED[n]])
    def test_the_quantiles_put_the_levels_first(self, name):
        law = _families()[name]
        quantiles = law._quantile(jnp.array([0.25, 0.5, 0.75]))
        assert quantiles.shape == (3, *law.event_spec.spec.shape)
        assert bool(jnp.all(jnp.diff(quantiles, axis=0) >= 0))

    def test_raw_is_the_backend_distribution(self):
        assert isinstance(F.Normal("x", 0.0, 1.0).raw(), tfd.Normal)

    def test_the_adapter_claims_the_quantile(self):
        law = F.Normal("x", 0.0, 1.0)
        assert isinstance(law, SupportsQuantile)
        assert float(law._quantile(0.5)) == pytest.approx(0.0, abs=1e-6)

    def test_the_multivariate_normal_quantiles_are_those_of_its_marginals(self):
        loc = jnp.array([1.0, -2.0])
        cov = jnp.array([[4.0, 0.6], [0.6, 0.25]])
        law = F.MultivariateNormal("z", loc, cov=cov)
        levels = jnp.array([0.1, 0.9])
        expected = jnp.stack(
            [F.Normal("m", loc[i], jnp.sqrt(cov[i, i]))._quantile(levels) for i in range(2)],
            axis=-1,
        )
        np.testing.assert_allclose(law._quantile(levels), expected, rtol=1e-5)

    def test_the_covariance_is_a_linear_operator(self):
        law = F.MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2))
        assert isinstance(law._cov(), LinOp)

    def test_a_given_covariance_operator_keeps_its_structure(self):
        operator = DiagonalLinOp(jnp.array([1.0, 4.0]))
        law = F.MultivariateNormal("z", jnp.zeros(2), cov=operator)
        assert law._cov() is operator
        np.testing.assert_allclose(law.cov, jnp.diag(jnp.array([1.0, 4.0])), rtol=1e-6)

    def test_an_infinite_support_family_claims_no_expectation(self):
        assert not isinstance(F.Normal("x", 0.0, 1.0), SupportsExpectation)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: F.Bernoulli("b", probs=0.3),
            lambda: F.Binomial("b", 4.0, probs=0.3),
            lambda: F.Categorical("b", probs=jnp.array([0.2, 0.3, 0.5])),
        ],
    )
    def test_a_finite_support_law_over_one_coordinate_has_the_exact_expectation(self, make):
        law = make()
        assert isinstance(law, SupportsExpectation)
        assert float(law._expectation(lambda x: x)) == pytest.approx(float(law._mean()), rel=1e-5)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: F.Bernoulli("b", probs=jnp.array([0.3, 0.6])),
            lambda: F.Binomial("b", jnp.array([4.0, 5.0]), probs=0.3),
            lambda: F.Categorical("b", probs=jnp.array([[0.2, 0.8], [0.5, 0.5]])),
        ],
    )
    def test_a_law_over_several_coordinates_claims_no_exact_expectation(self, make):
        assert not isinstance(make(), SupportsExpectation)


class TestVectorParameters:
    """A scalar family given parameters with axes draws one array of independent coordinates."""

    def test_vector_parameters_draw_a_vector(self):
        law = F.Normal("x", jnp.arange(5.0), 1.0)
        assert law.event_shape == (5,)
        assert law._sample(jax.random.PRNGKey(0)).shape == (5,)
        assert law._sample(jax.random.PRNGKey(0), (3,)).shape == (3, 5)

    def test_the_density_is_the_product_over_the_coordinates(self):
        law = F.Normal("x", jnp.arange(3.0), 2.0)
        value = jnp.array([0.5, 1.5, -1.0])
        expected = sum(float(F.Normal("c", float(i), 2.0)._log_prob(value[i])) for i in range(3))
        assert float(law._log_prob(value)) == pytest.approx(expected, rel=1e-6)
        assert law._log_prob(jnp.zeros((4, 3))).shape == (4,)

    def test_the_coordinates_are_uncorrelated(self):
        law = F.Gamma("g", jnp.array([2.0, 3.0]), 1.0)
        covariance = law._cov()
        assert isinstance(covariance, DiagonalLinOp)
        np.testing.assert_allclose(covariance.to_dense(), jnp.diag(law._variance()), rtol=1e-6)

    def test_parameters_broadcast_to_the_event_shape(self):
        law = F.Beta("b", jnp.ones((2, 3)), 1.0)
        assert law.event_shape == (2, 3)
        assert law._mean().shape == (2, 3)
        assert law._quantile(jnp.array([0.5, 0.9])).shape == (2, 2, 3)

    def test_a_multivariate_family_takes_the_parameters_of_one_law(self):
        d = 3
        with pytest.raises(ValueError, match="DistributionBatch"):
            F.MultivariateNormal(
                "z", jnp.zeros((2, d)), scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d))
            )

    def test_scalar_parameters_draw_a_scalar(self):
        law = F.Normal("x", 0.0, 1.0)
        assert law.event_shape == ()
        assert not hasattr(law, "batch_shape")

    def test_a_batch_of_separate_laws_is_built_from_batched_parameters(self):
        batch = DistributionArray.from_batched_params(
            F.Normal, loc=jnp.zeros(5), scale=1.0, name="x"
        )
        assert batch.batch_shape == (5,)
        assert batch[0].name == "x_0"
        assert batch[0].event_shape == ()


class TestTheSeparateLawsForm:
    """Inside ``_allow_batched_tfp_init`` a family keeps its backend's batch axes as separate laws."""

    def test_the_batch_axes_are_separate_laws_inside_the_block(self):
        with _allow_batched_tfp_init():
            law = F.Normal("x", jnp.zeros(5), 1.0)
        assert law.event_shape == ()
        assert tuple(law.raw().batch_shape) == (5,)
        assert law._log_prob(jnp.zeros(5)).shape == (5,)

    def test_the_form_is_scoped_to_the_block(self):
        with _allow_batched_tfp_init():
            F.Normal("x", jnp.zeros(3), 1.0)
        assert F.Normal("x", jnp.zeros(3), 1.0).event_shape == (3,)

    def test_a_nested_block_restores_the_outer_form(self):
        with _allow_batched_tfp_init():
            with _allow_batched_tfp_init():
                F.Normal("x", jnp.zeros(2), 1.0)
            assert F.Normal("x", jnp.zeros(2), 1.0).event_shape == ()
        assert F.Normal("x", jnp.zeros(2), 1.0).event_shape == (2,)

    def test_the_form_does_not_leak_across_asyncio_tasks(self):
        inside: list[tuple[int, ...]] = []
        outside: list[tuple[int, ...]] = []

        async def within_the_block():
            with _allow_batched_tfp_init():
                # Yield so the sibling task runs while this block is active.
                await asyncio.sleep(0)
                inside.append(F.Normal("x", jnp.zeros(2), 1.0).event_shape)

        async def outside_the_block():
            await asyncio.sleep(0)
            outside.append(F.Normal("y", jnp.zeros(2), 1.0).event_shape)

        async def main():
            await asyncio.gather(within_the_block(), outside_the_block())

        asyncio.run(main())
        assert inside == [()]
        assert outside == [(2,)]


class TestALateBackend:
    """A subclass that builds its backend after construction passes its own declaration."""

    def test_a_late_backend_needs_its_own_declaration(self):
        class _LateBackend(TFPDistribution):
            def __init__(self, name):
                super().__init__(name, None)
                self._tfp_dist = tfd.Normal(0.0, 1.0)

            def _event_support(self):
                return real

        with pytest.raises(TypeError, match=r"_LateBackend builds its backend after .* event_spec"):
            _LateBackend("late")
