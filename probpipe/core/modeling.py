"""
Modeling components for ProbPipe.

Provides abstract interfaces for likelihoods, and an iterative
forecasting module.

MCMC sampling has moved to :mod:`probpipe.inference` and
probabilistic models to :mod:`probpipe.modeling`.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC
from typing import Any

import jax.numpy as jnp

from ..custom_types import ArrayLike
from .distribution import ArrayDistribution, EmpiricalDistribution
from .node import AbstractModule, Module, WorkflowFunction, abstractwf, wf

logger = logging.getLogger(__name__)

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "IterativeForecaster",
    # Backward-compat re-exports (deprecated)
    "MCMCDiagnostics",
    "ApproximatePosterior",
    "MCMCSampler",
    "RWMH",
]


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class Likelihood(AbstractModule):
    """Abstract module for computing log-likelihood of data given parameters."""

    @abstractwf
    def log_likelihood(self, params: ArrayLike, data: ArrayLike) -> float:
        ...


class GenerativeLikelihood(AbstractModule):
    """Abstract module for generating synthetic data given parameters."""

    @abstractwf
    def generate_data(self, params: ArrayLike, n_samples: int) -> ArrayLike:
        ...


# ---------------------------------------------------------------------------
# Approximate posterior base (kept for backward compatibility)
# ---------------------------------------------------------------------------


class ApproximatePosterior(WorkflowFunction, ABC):
    """Abstract base for posterior approximation methods.

    .. deprecated::
        Use :class:`~probpipe.modeling.SimpleModel` with
        :func:`~probpipe.core.ops.condition_on` instead.
    """

    def __init__(
        self,
        *,
        workflow_kind: str | None = None,
        name: str = "compute_posterior",
        **bind: Any,
    ):
        super().__init__(
            func=self._compute_posterior,
            workflow_kind=workflow_kind,
            name=name,
            bind=bind,
        )

    @abstractwf
    def _compute_posterior(
        self,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> EmpiricalDistribution:
        ...


# ---------------------------------------------------------------------------
# IterativeForecaster
# ---------------------------------------------------------------------------


class IterativeForecaster(Module):
    """Iteratively update posterior given new data batches."""

    def __init__(
        self,
        *,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        generative_likelihood: GenerativeLikelihood,
        approx_post: ApproximatePosterior,
        workflow_kind: str | None = None,
    ):
        self._curr_posterior: ArrayDistribution = prior
        self._generative_likelihood = generative_likelihood

        super().__init__(
            likelihood=likelihood,
            generative_likelihood=generative_likelihood,
            approx_post=approx_post,
            prior=prior,
            workflow_kind=workflow_kind,
        )

    @property
    def curr_posterior(self) -> ArrayDistribution:
        return self._curr_posterior

    @wf
    def update(
        self,
        approx_post: ApproximatePosterior,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> ArrayDistribution:
        post_dist = approx_post(
            prior=self._curr_posterior, likelihood=likelihood, data=data
        )
        self._curr_posterior = post_dist
        return post_dist


# ---------------------------------------------------------------------------
# Backward-compat re-exports from new locations
# ---------------------------------------------------------------------------

from ..inference._diagnostics import MCMCDiagnostics  # noqa: E402, F401


def __getattr__(name: str):
    if name == "MCMCSampler":
        warnings.warn(
            "MCMCSampler is deprecated. Use probpipe.modeling.SimpleModel "
            "with condition_on() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import the legacy class from a lazy location
        from ._modeling_legacy import MCMCSampler

        return MCMCSampler
    if name == "RWMH":
        warnings.warn(
            "RWMH is deprecated. Use probpipe.inference.rwmh instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ._modeling_legacy import RWMH

        return RWMH
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
