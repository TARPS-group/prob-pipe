"""Likelihood protocols and incremental conditioning.

Provides protocol interfaces for likelihoods, a private conditioning
step function, and a stateful module for sequential Bayesian updating.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from ..core.distribution import Distribution
from ..core.node import Module, WorkflowFunction
from ..core.transition import iterate

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
# _ConditioningStep — private WorkflowFunction for IncrementalConditioner
# ---------------------------------------------------------------------------


class _ConditioningStep(WorkflowFunction):
    """One step of incremental Bayesian conditioning.

    Builds a :class:`~probpipe.modeling.SimpleModel` from the current
    distribution and a likelihood, then conditions on observed data
    using ProbPipe's standard ``condition_on`` dispatch.

    If the current distribution does not support
    :class:`~probpipe.core.protocols.SupportsLogProb`, it is
    automatically converted (e.g., MCMC samples -> KDE) via ProbPipe's
    converter registry before building the model.

    Parameters
    ----------
    likelihood : Likelihood[P, D]
        Likelihood object.
    condition_fn : callable or None
        ``(model, data, **kw) -> Distribution[P]``.  Defaults to
        the global ``condition_on`` operation, which dispatches through
        the inference method registry.  Override to use a custom
        conditioning strategy.
    **condition_kwargs
        Extra keyword arguments forwarded to *condition_fn* on every
        call (e.g., ``method="tfp_nuts"``, ``num_results=2000``).
    """

    def __init__(
        self,
        likelihood: Likelihood,
        *,
        condition_fn: Callable | None = None,
        workflow_kind: str | None = None,
        **condition_kwargs: Any,
    ):
        if condition_fn is None:
            from ..core.ops import condition_on

            condition_fn = condition_on

        self._likelihood = likelihood
        self._condition_fn = condition_fn
        self._condition_kwargs = condition_kwargs

        super().__init__(
            func=self._step_impl,
            name="conditioning_step",
            workflow_kind=workflow_kind,
        )

    def _step_impl(
        self,
        dist: Distribution,
        data: Any,
    ) -> Distribution:
        from ._simple import SimpleModel
        from ..core.protocols import SupportsLogProb

        prior = dist
        if not isinstance(prior, SupportsLogProb):
            from ..converters import converter_registry

            prior = converter_registry.convert(prior, SupportsLogProb)

        model = SimpleModel(prior=prior, likelihood=self._likelihood)
        return self._condition_fn(model, data, **self._condition_kwargs)


# ---------------------------------------------------------------------------
# IncrementalConditioner — stateful convenience Module
# ---------------------------------------------------------------------------


class IncrementalConditioner[P, D](Module):
    """Iteratively update a posterior by conditioning on data batches.

    Maintains a *current posterior* (initially the prior) and provides
    :meth:`update` for single-batch conditioning and
    :meth:`update_all` for multi-batch iteration.  Both methods update
    the internal state so that subsequent calls continue from the
    latest posterior.

    The :attr:`step` property exposes the underlying step function
    for direct use with :func:`~probpipe.core.transition.iterate`
    and combinators.

    Parameters
    ----------
    prior : Distribution[P]
        Initial prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Likelihood object.
    condition_fn : callable or None
        Conditioning callable; defaults to the global ``condition_on``
        operation (which dispatches through the inference method
        registry).
    **condition_kwargs
        Extra keyword arguments forwarded to *condition_fn* on every
        call (e.g., ``method="tfp_nuts"``, ``num_results=2000``).

    Examples
    --------
    ::

        conditioner = IncrementalConditioner(prior, likelihood)

        # Single-batch update (stateful):
        posterior1 = conditioner.update(data=batch1)
        posterior2 = conditioner.update(data=batch2)
        conditioner.curr_posterior  # is posterior2

        # Multi-batch update (stateful, returns sequence):
        dists = conditioner.update_all(data_batches=[batch3, batch4])
        conditioner.curr_posterior  # is dists[-1]

        # Functional escape hatch (for combinators):
        dists = iterate(conditioner.step, prior, all_batches)
    """

    def __init__(
        self,
        prior: Distribution[P],
        likelihood: Likelihood[P, D],
        *,
        condition_fn: Callable | None = None,
        **condition_kwargs: Any,
    ):
        self._prior = prior
        self._likelihood = likelihood
        self._curr_posterior: Distribution[P] = prior
        self._step = _ConditioningStep(
            likelihood,
            condition_fn=condition_fn,
            **condition_kwargs,
        )

    @property
    def curr_posterior(self) -> Distribution[P]:
        """The current posterior (initially the prior)."""
        return self._curr_posterior

    @property
    def step(self) -> _ConditioningStep:
        """The underlying step function, for use with ``iterate``."""
        return self._step

    def update(self, data: D) -> Distribution[P]:
        """Condition on new data, updating the current posterior.

        Parameters
        ----------
        data : D
            New observed data to condition on.

        Returns
        -------
        Distribution[P]
            The updated posterior distribution.
        """
        posterior = self._step(self._curr_posterior, data)
        self._curr_posterior = posterior
        return posterior

    def update_all(self, data_batches: Iterable[D]) -> list[Distribution[P]]:
        """Condition on multiple data batches sequentially.

        Calls ``iterate(self.step, self.curr_posterior, data_batches)``
        and updates the internal state to the final posterior.

        Parameters
        ----------
        data_batches : Iterable[D]
            Sequence of data batches to condition on.

        Returns
        -------
        list[Distribution[P]]
            Sequence ``[starting_posterior, post_1, post_2, ...]``.
        """
        dists = iterate(self._step, self._curr_posterior, data_batches)
        self._curr_posterior = dists[-1]
        return dists
