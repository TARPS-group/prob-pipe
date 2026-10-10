"""Every family's law is immutable, and its annotations store stays writable (II.4)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

import probpipe.families as F
from probpipe import Distribution
from probpipe.families import (
    BernoulliFamily,
    BijectorTransformedDistribution,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    FactoredMultivariateGaussian,
    GaussianProcess,
    KDEDistribution,
    LinearBasisFunction,
    MixtureDistribution,
    RandomFunction,
    glm_likelihood,
)


class _ConstantRandomFunction(RandomFunction):
    """A random function whose value at every point is one law."""

    def __call__(self, x: jnp.ndarray) -> Distribution:
        return F.Normal("y", 0.0, 1.0)


def _rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))


def _polynomial_basis(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.ones_like(x), x, x**2], axis=-1)


#: A representative law of each family class, built as the family's own tests build it.
_LAWS: dict[str, Callable[[], object]] = {
    "Normal": lambda: F.Normal("a", 0.0, 1.0),
    "Beta": lambda: F.Beta("a", 2.0, 3.0),
    "Gamma": lambda: F.Gamma("a", 2.0, 1.0),
    "InverseGamma": lambda: F.InverseGamma("a", 3.0, 1.0),
    "Exponential": lambda: F.Exponential("a", 1.0),
    "LogNormal": lambda: F.LogNormal("a", 0.0, 1.0),
    "StudentT": lambda: F.StudentT("a", 3.0, 0.0, 1.0),
    "Uniform": lambda: F.Uniform("a", 0.0, 1.0),
    "Cauchy": lambda: F.Cauchy("a", 0.0, 1.0),
    "Laplace": lambda: F.Laplace("a", 0.0, 1.0),
    "HalfNormal": lambda: F.HalfNormal("a", 1.0),
    "HalfCauchy": lambda: F.HalfCauchy("a", 0.0, 1.0),
    "Pareto": lambda: F.Pareto("a", 3.0, 1.0),
    "TruncatedNormal": lambda: F.TruncatedNormal("a", 0.0, 1.0, -1.0, 1.0),
    "Bernoulli": lambda: F.Bernoulli("a", probs=0.3),
    "Binomial": lambda: F.Binomial("a", 5.0, probs=0.3),
    "Poisson": lambda: F.Poisson("a", 2.0),
    "Categorical": lambda: F.Categorical("a", probs=jnp.array([0.2, 0.8])),
    "NegativeBinomial": lambda: F.NegativeBinomial("a", 5.0, probs=0.3),
    "MultivariateNormal": lambda: F.MultivariateNormal("a", jnp.zeros(2), cov=jnp.eye(2)),
    "Dirichlet": lambda: F.Dirichlet("a", jnp.ones(3)),
    "Multinomial": lambda: F.Multinomial("a", 5, probs=jnp.array([0.2, 0.8])),
    "Wishart": lambda: F.Wishart("a", 3.0, scale_tril=jnp.eye(2)),
    "VonMisesFisher": lambda: F.VonMisesFisher("a", jnp.array([0.0, 1.0]), 2.0),
    "BijectorTransformedDistribution": lambda: BijectorTransformedDistribution(
        "y", F.Normal("x", 0.0, 1.0), tfb.Exp()
    ),
    "MixtureDistribution": lambda: MixtureDistribution(
        [F.Normal("a", 0.0, 1.0), F.Normal("a", 1.0, 1.0)], jnp.array([0.5, 0.5]), label="m"
    ),
    "BootstrapDistribution": lambda: BootstrapDistribution("B", F.Normal("x", 0.0, 1.0), 7),
    "BootstrapReplicateDistribution": lambda: BootstrapReplicateDistribution(
        "b", F.Normal("x", 0.0, 1.0), replicate_size=5
    ),
    "KDEDistribution": lambda: KDEDistribution(
        jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]),
        bandwidth=jnp.array([0.4, 0.6]),
        component="k",
    ),
    "FactoredMultivariateGaussian": lambda: FactoredMultivariateGaussian(
        [F.Normal("a", 0.0, 1.0), F.Normal("g", 2.0, 1.0)],
        label="j",
    ),
    "GaussianProcess": lambda: GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _rbf_kernel),
    "LinearBasisFunction": lambda: LinearBasisFunction(
        "f",
        _polynomial_basis,
        F.MultivariateNormal("weights", jnp.array([1.0, 0.5, 0.1]), cov=0.01 * jnp.eye(3)),
    ),
    "RandomFunction": lambda: _ConstantRandomFunction("rf"),
    "ConditionalDistribution": lambda: glm_likelihood("y", BernoulliFamily()),
}


@pytest.fixture(params=list(_LAWS))
def law(request):
    return _LAWS[request.param]()


class TestALawIsImmutable:
    def test_assigning_a_new_attribute_raises(self, law):
        with pytest.raises(AttributeError, match=f"{type(law).__name__} is immutable"):
            law.fitted = True

    def test_assigning_an_existing_attribute_raises(self, law):
        with pytest.raises(AttributeError, match=f"{type(law).__name__} is immutable"):
            law._label = "renamed"
        assert law.label != "renamed"

    def test_deleting_an_attribute_raises(self, law):
        with pytest.raises(AttributeError, match=f"{type(law).__name__} is immutable"):
            del law._label
        assert law.label

    def test_with_label_returns_a_new_law_and_leaves_the_original(self, law):
        relabeled = law.with_label("renamed")
        assert relabeled is not law
        assert relabeled.label == "renamed"
        assert law.label != "renamed"


class TestTheAnnotationsStoreStaysWritable:
    def test_a_writer_adds_an_entry_to_the_store_in_place(self, law):
        object.__setattr__(law, "_annotations", {"diagnostics": "first"})
        law.annotations["validation"] = "second"
        assert law.annotations == {"diagnostics": "first", "validation": "second"}
