"""A certified regression simulator for the tests of workflow-owned randomness.

A predictive check owns a call's randomness only for a simulator whose class
carries a workflow generative-provider certificate. The library's GLM kernel is
a ``ConditionalDistribution``, which samples through the distribution ABI, so
these tests certify a simulator of their own: ``y = X @ beta``, with an intercept
unless ``fit_intercept=False``, plus unit Gaussian noise or a Poisson count, for
one parameter vector or a batch of them.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from probpipe.core.protocols import _WorkflowGenerativeProviderCertificate


class CertifiedRegression:
    """A regression simulator certified for workflow-owned randomness."""

    def __init__(self, family: str = "normal", x: Any = None, *, fit_intercept: bool = True):
        self._family = family
        self._x = None if x is None else jnp.asarray(x)
        self._fit_intercept = fit_intercept

    def generate_data(self, params: Any, num_observations: int, *, key: Any = None) -> Any:
        """``num_observations`` responses at each parameter vector of *params*."""
        if key is None:
            key = jax.random.PRNGKey(0)
        beta = jnp.asarray(params)
        rows = self._x[:num_observations]
        if self._fit_intercept:
            eta = beta[..., 0:1] + beta[..., 1:] @ rows.T
        else:
            eta = beta @ rows.T
        if self._family == "poisson":
            return jax.random.poisson(key, jnp.exp(eta)).astype(jnp.float32)
        return eta + jax.random.normal(key, eta.shape)


def _preflight(provider: CertifiedRegression, operation: str) -> None:
    """Refuse a simulator without its design before any randomness is committed."""
    if provider._x is None:
        raise ValueError(
            f"{operation} requires CertifiedRegression to have a stored design matrix "
            "before requesting workflow-owned randomness"
        )


CertifiedRegression._workflow_generative_provider_certificate = (
    _WorkflowGenerativeProviderCertificate(
        provider_type=CertifiedRegression,
        generate_data=CertifiedRegression.generate_data,
        provider_abi="tests.CertifiedRegression.generate_data/v1",
        preflight=_preflight,
    )
)
