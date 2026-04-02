"""Likelihood protocols and incremental conditioning.

Provides protocol interfaces for likelihoods and an incremental
conditioner for iteratively updating posteriors with new data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..core.distribution import Distribution
from ..core.node import Module, wf

logger = logging.getLogger(__name__)

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "IncrementalConditioner",
]


# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class Likelihood[P, D](Protocol):
    """Protocol for computing log-likelihood of data given parameters.

    Generic in ``P`` (parameter type) and ``D`` (data type).
    Any class that defines ``log_likelihood(params, data) -> float``
    satisfies this protocol.
    """

    def log_likelihood(self, params: P, data: D) -> float: ...


@runtime_checkable
class GenerativeLikelihood[P, D](Protocol):
    """Protocol for generating synthetic data given parameters.

    Generic in ``P`` (parameter type) and ``D`` (data type).
    Any class that defines ``generate_data(params, n_samples) -> D``
    satisfies this protocol.
    """

    def generate_data(self, params: P, n_samples: int) -> D: ...


# ---------------------------------------------------------------------------
# IncrementalConditioner
# ---------------------------------------------------------------------------


class IncrementalConditioner[P, D](Module):
    """Iteratively update a posterior by conditioning on successive data batches.

    Takes a prior ``Distribution[P]`` and ``Likelihood[P, D]``, builds a
    :class:`~probpipe.modeling.SimpleModel` internally, and uses
    ``condition_on`` (or a custom conditioning callable) to update the
    posterior each time new data arrives.  The current posterior becomes
    the prior for the next update.

    Parameters
    ----------
    prior : Distribution[P]
        Initial prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Likelihood object.
    condition_fn : callable or None
        A callable with signature ``(model, data) -> Distribution[P]``
        that conditions the model on observed data.  Defaults to the
        global ``condition_on`` operation.
    """

    def __init__(
        self,
        prior: Distribution[P],
        likelihood: Likelihood[P, D],
        *,
        condition_fn: Callable | None = None,
    ):
        self._prior = prior
        self._likelihood = likelihood
        self._curr_posterior: Distribution[P] = prior
        if condition_fn is None:
            from ..core.ops import condition_on
            self._condition_fn = condition_on
        else:
            self._condition_fn = condition_fn

    @property
    def curr_posterior(self) -> Distribution[P]:
        """The current posterior (initially the prior)."""
        return self._curr_posterior

    @wf
    def update(self, data: D) -> Distribution[P]:
        """Condition on new data, updating the current posterior.

        Constructs a :class:`~probpipe.modeling.SimpleModel` using the
        current posterior as prior, then conditions on the provided data.

        Parameters
        ----------
        data : D
            New observed data to condition on.

        Returns
        -------
        Distribution[P]
            The updated posterior distribution.
        """
        from ._simple import SimpleModel

        model = SimpleModel(
            prior=self._curr_posterior,
            likelihood=self._likelihood,
        )
        posterior = self._condition_fn(model, data)
        self._curr_posterior = posterior
        return posterior
