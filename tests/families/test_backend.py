"""Contracts of the parametric families (VII.1), checked where the families are defined today."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe.distributions as D
from probpipe import NumericArraySpec, NumericDistribution, OutputSpec
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)
from probpipe.linalg import LinOp


def _families() -> dict[str, D.Distribution]:
    return {
        "Normal": D.Normal("a", 0.0, 1.0),
        "Beta": D.Beta("a", 2.0, 3.0),
        "Gamma": D.Gamma("a", 2.0, 1.0),
        "InverseGamma": D.InverseGamma("a", 3.0, 1.0),
        "Exponential": D.Exponential("a", 1.0),
        "LogNormal": D.LogNormal("a", 0.0, 1.0),
        "StudentT": D.StudentT("a", 3.0, 0.0, 1.0),
        "Uniform": D.Uniform("a", 0.0, 1.0),
        "Cauchy": D.Cauchy("a", 0.0, 1.0),
        "Laplace": D.Laplace("a", 0.0, 1.0),
        "HalfNormal": D.HalfNormal("a", 1.0),
        "HalfCauchy": D.HalfCauchy("a", 0.0, 1.0),
        "Pareto": D.Pareto("a", 3.0, 1.0),
        "TruncatedNormal": D.TruncatedNormal("a", 0.0, 1.0, -1.0, 1.0),
        "Bernoulli": D.Bernoulli("a", probs=0.3),
        "Binomial": D.Binomial("a", 5.0, probs=0.3),
        "Poisson": D.Poisson("a", 2.0),
        "Categorical": D.Categorical("a", probs=jnp.array([0.2, 0.8])),
        "NegativeBinomial": D.NegativeBinomial("a", 5.0, probs=0.3),
        "MultivariateNormal": D.MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2)),
        "Dirichlet": D.Dirichlet("a", jnp.ones(3)),
        "Multinomial": D.Multinomial("a", 5, probs=jnp.array([0.2, 0.8])),
        "Wishart": D.Wishart("a", 3.0, scale_tril=jnp.eye(2)),
        "VonMisesFisher": D.VonMisesFisher("a", jnp.array([0.0, 1.0]), 2.0),
    }


_NAMES = list(_families())

#: Capabilities a family claims whose backend does not compute them.
_UNCOMPUTED = {
    ("Wishart", "_cov"): "the Wishart family claims a covariance its backend lacks",
    (
        "VonMisesFisher",
        "_variance",
    ): "the von Mises-Fisher family claims a variance its backend lacks",
}


def _capability_cases() -> list:
    cases = []
    for name in _NAMES:
        for method in ("_mean", "_variance", "_cov"):
            marks = ()
            if (name, method) in _UNCOMPUTED:
                marks = pytest.mark.pending(reason=_UNCOMPUTED[(name, method)])
            cases.append(pytest.param(name, method, id=f"{name}-{method}", marks=marks))
    return cases


class TestTheEventDeclaration:
    def test_event_spec_names_the_component_and_the_label_stays(self):
        prior = D.Normal("prior", 0.0, 1.0, event_spec=OutputSpec(beta=None))
        assert prior.name == "prior"
        assert list(prior.event_spec.components) == ["beta"]
        assert prior.event_spec.spec.shape == ()

    def test_the_component_defaults_to_the_label_captured_at_construction(self):
        law = D.Normal("x", 0.0, 1.0)
        assert list(law.event_spec.components) == ["x"]
        assert list(law.with_name("y").event_spec.components) == ["x"]

    def test_the_event_spec_is_derived_from_the_parameters(self):
        law = D.MultivariateNormal("z", jnp.zeros(3), cov=jnp.eye(3))
        spec = law.event_spec.spec
        assert spec.shape == (3,)
        assert spec.dtype == jnp.zeros(3).dtype
        assert spec.support is not None

    def test_a_declared_type_must_agree_with_the_parameters(self):
        with pytest.raises(ValueError):
            D.Normal("x", 0.0, 1.0, event_spec=OutputSpec(x=NumericArraySpec((3,))))

    @pytest.mark.parametrize("name", _NAMES)
    def test_each_family_is_a_numeric_distribution(self, name):
        assert isinstance(_families()[name], NumericDistribution)


class TestTheAdapter:
    @pytest.mark.parametrize("name", _NAMES)
    def test_one_draw_matches_the_declaration(self, name):
        law = _families()[name]
        draw = law._sample(jax.random.PRNGKey(0))
        assert tuple(jnp.shape(draw)) == law.event_spec.spec.shape

    @pytest.mark.parametrize(("name", "method"), _capability_cases())
    def test_each_claimed_moment_computes(self, name, method):
        law = _families()[name]
        protocol = {
            "_mean": SupportsMean,
            "_variance": SupportsVariance,
            "_cov": SupportsCovariance,
        }
        if not isinstance(law, protocol[method]):
            pytest.skip(f"{name} does not claim {method}")
        getattr(law, method)()

    @pytest.mark.pending(
        reason="the adapter's raw() is the wrapped backend distribution", raises=AttributeError
    )
    def test_raw_is_the_backend_distribution(self):
        assert isinstance(D.Normal("x", 0.0, 1.0).raw(), tfd.Normal)

    @pytest.mark.pending(
        reason="the adapter implements the closed-form quantile", raises=AssertionError
    )
    def test_the_adapter_claims_the_quantile(self):
        law = D.Normal("x", 0.0, 1.0)
        assert isinstance(law, SupportsQuantile)
        assert float(law._quantile(0.5)) == pytest.approx(0.0, abs=1e-6)

    def test_the_covariance_is_a_linear_operator(self):
        law = D.MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2))
        assert isinstance(law._cov(), LinOp)

    def test_an_infinite_support_family_claims_no_expectation(self):
        assert not isinstance(D.Normal("x", 0.0, 1.0), SupportsExpectation)
