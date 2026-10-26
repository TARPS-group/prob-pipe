"""One law of each catalog family whose default label is its class name.

Each entry builds the law over the output ``x`` and passes its keyword
arguments to the family's constructor, so a test builds the law under its
default label or under a ``label=`` it chooses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

import probpipe
from probpipe.families import KDEDistribution

#: The laws by family name, each built by a call that takes the constructor's keywords.
FAMILIES: dict[str, Callable[..., Any]] = {
    "Normal": lambda **kw: probpipe.Normal("x", 0.0, 1.0, **kw),
    "Beta": lambda **kw: probpipe.Beta("x", 2.0, 3.0, **kw),
    "Gamma": lambda **kw: probpipe.Gamma("x", 2.0, 1.0, **kw),
    "InverseGamma": lambda **kw: probpipe.InverseGamma("x", 2.0, 1.0, **kw),
    "Exponential": lambda **kw: probpipe.Exponential("x", 1.0, **kw),
    "LogNormal": lambda **kw: probpipe.LogNormal("x", 0.0, 1.0, **kw),
    "StudentT": lambda **kw: probpipe.StudentT("x", 3.0, 0.0, 1.0, **kw),
    "Uniform": lambda **kw: probpipe.Uniform("x", -1.0, 2.0, **kw),
    "Cauchy": lambda **kw: probpipe.Cauchy("x", 0.0, 1.0, **kw),
    "Laplace": lambda **kw: probpipe.Laplace("x", 0.0, 1.0, **kw),
    "HalfNormal": lambda **kw: probpipe.HalfNormal("x", 1.0, **kw),
    "HalfCauchy": lambda **kw: probpipe.HalfCauchy("x", 0.5, 1.0, **kw),
    "Pareto": lambda **kw: probpipe.Pareto("x", 2.0, 1.5, **kw),
    "TruncatedNormal": lambda **kw: probpipe.TruncatedNormal("x", 0.0, 1.0, -1.0, 1.0, **kw),
    "Bernoulli": lambda **kw: probpipe.Bernoulli("x", probs=0.3, **kw),
    "Binomial": lambda **kw: probpipe.Binomial("x", 5, probs=0.3, **kw),
    "Poisson": lambda **kw: probpipe.Poisson("x", 2.0, **kw),
    "Categorical": lambda **kw: probpipe.Categorical("x", probs=[0.2, 0.3, 0.5], **kw),
    "NegativeBinomial": lambda **kw: probpipe.NegativeBinomial("x", 5.0, probs=0.3, **kw),
    "MultivariateNormal": lambda **kw: probpipe.MultivariateNormal(
        "x", jnp.zeros(2), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]]), **kw
    ),
    "Dirichlet": lambda **kw: probpipe.Dirichlet("x", jnp.ones(3), **kw),
    "Multinomial": lambda **kw: probpipe.Multinomial(
        "x", 4.0, probs=jnp.array([0.2, 0.3, 0.5]), **kw
    ),
    "Wishart": lambda **kw: probpipe.Wishart("x", 4.0, scale_tril=jnp.eye(2), **kw),
    "VonMisesFisher": lambda **kw: probpipe.VonMisesFisher("x", jnp.array([0.0, 1.0]), 2.0, **kw),
    "KDEDistribution": lambda **kw: KDEDistribution(
        jnp.array([0.0, 1.0, 3.0]), 0.5, output_name="x", **kw
    ),
    "KDEDistribution under a rule": lambda **kw: KDEDistribution(
        jnp.array([0.0, 1.0, 3.0]), weights=jnp.array([0.5, 0.25, 0.25]), output_name="x", **kw
    ),
}
