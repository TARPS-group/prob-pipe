"""Likelihood interfaces and iterative forecasting.

Provides abstract interfaces for likelihoods and an iterative
forecasting module.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

from ..custom_types import ArrayLike
from ..core.distribution import ArrayDistribution, EmpiricalDistribution
from ..core.node import AbstractModule, Module, WorkflowFunction, abstractwf, wf

logger = logging.getLogger(__name__)

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "IterativeForecaster",
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
# IterativeForecaster
# ---------------------------------------------------------------------------


class IterativeForecaster(Module):
    """Iteratively update posterior given new data batches.

    Parameters
    ----------
    prior : ArrayDistribution
        Initial prior distribution.
    likelihood : Likelihood
        Likelihood module.
    generative_likelihood : GenerativeLikelihood
        Module for generating synthetic data from parameters.
    approx_post : WorkflowFunction
        A workflow function that takes ``(prior, likelihood, data)``
        and returns an approximate posterior distribution.
    workflow_kind : str or None
        Prefect orchestration mode.
    """

    def __init__(
        self,
        *,
        prior: ArrayDistribution,
        likelihood: Likelihood,
        generative_likelihood: GenerativeLikelihood,
        approx_post: WorkflowFunction,
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
        approx_post: WorkflowFunction,
        likelihood: Likelihood,
        data: ArrayLike,
    ) -> ArrayDistribution:
        post_dist = approx_post(
            prior=self._curr_posterior, likelihood=likelihood, data=data
        )
        self._curr_posterior = post_dist
        return post_dist
