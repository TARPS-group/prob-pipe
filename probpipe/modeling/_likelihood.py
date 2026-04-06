"""Likelihood protocols, conditioning step, and incremental conditioner.

Provides protocol interfaces for likelihoods, a reusable conditioning
step function for iterative distribution transformations, and a
convenience module for sequential Bayesian updating.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from ..core.distribution import Distribution
from ..core.node import Module, WorkflowFunction, workflow_method
from ..core.transition import StepResult, TransitionTrace, iterate

logger = logging.getLogger(__name__)

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "ConditioningStep",
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
# ConditioningStep — a WorkflowFunction subclass for iterate
# ---------------------------------------------------------------------------


class ConditioningStep(WorkflowFunction):
    """One step of incremental Bayesian conditioning.

    Builds a :class:`~probpipe.modeling.SimpleModel` from the current
    distribution and a likelihood, then conditions on observed data.
    Directly callable as a step function for use with
    :func:`~probpipe.core.transition.iterate`.

    If the current distribution does not support
    :class:`~probpipe.core.protocols.SupportsLogProb`, it is
    automatically converted (e.g., MCMC samples → KDE) via the
    converter registry.

    Parameters
    ----------
    likelihood : Likelihood[P, D]
        Likelihood object.
    condition_fn : callable or None
        ``(model, data, **kw) -> Distribution[P]``.  Defaults to
        the global ``condition_on`` operation.
    **condition_kwargs
        Extra keyword arguments forwarded to *condition_fn* on every
        call (e.g., ``method="tfp_nuts"``, ``num_results=2000``).

    Examples
    --------
    ::

        step = ConditioningStep(likelihood, method="tfp_nuts")
        trace = iterate(step, prior, data_batches)
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
    ) -> StepResult:
        from ._simple import SimpleModel
        from ..core.protocols import SupportsLogProb

        prior = dist
        # Auto-convert the prior if it doesn't support log_prob
        # (e.g., MCMCApproximateDistribution → KDEDistribution).
        if not isinstance(prior, SupportsLogProb):
            from ..converters import converter_registry

            prior = converter_registry.convert(prior, SupportsLogProb)

        model = SimpleModel(prior=prior, likelihood=self._likelihood)
        posterior = self._condition_fn(model, data, **self._condition_kwargs)
        return StepResult(distribution=posterior)


# ---------------------------------------------------------------------------
# IncrementalConditioner — convenience Module
# ---------------------------------------------------------------------------


class IncrementalConditioner[P, D](Module):
    """Sequential Bayesian updating: condition on data batches.

    Convenience wrapper around :class:`ConditioningStep` and
    :func:`~probpipe.core.transition.iterate`.  For more control
    (e.g., composing with :func:`~probpipe.core.transition.with_approximation`),
    use ``ConditioningStep`` and ``iterate`` directly.

    Parameters
    ----------
    prior : Distribution[P]
        Initial prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Likelihood object.
    condition_fn : callable or None
        A callable with signature ``(model, data, **kw) -> Distribution[P]``
        that conditions the model on observed data.  Defaults to the
        global ``condition_on`` operation.
    **condition_kwargs
        Extra keyword arguments forwarded to *condition_fn* on every
        call (e.g., ``method="tfp_nuts"``, ``num_results=2000``).

    Examples
    --------
    ::

        conditioner = IncrementalConditioner(prior, likelihood)
        trace = conditioner.update(data_batches=[batch1, batch2, batch3])
        trace.final          # final posterior
        trace.distributions  # full trajectory
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
        self._step = ConditioningStep(
            likelihood,
            condition_fn=condition_fn,
            **condition_kwargs,
        )

    @workflow_method
    def update(self, data_batches: Iterable[D]) -> TransitionTrace[P]:
        """Condition on all data batches, returning the full trajectory.

        Parameters
        ----------
        data_batches : Iterable[D]
            Sequence of data batches to condition on.

        Returns
        -------
        TransitionTrace[P]
            Trajectory including the initial prior and each posterior.
        """
        return iterate(self._step, self._prior, data_batches)
