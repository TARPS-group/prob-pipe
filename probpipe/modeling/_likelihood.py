"""Likelihood protocols, conditioning step, and incremental conditioner.

Provides protocol interfaces for likelihoods, a reusable conditioning
step function for iterative distribution transformations, and a
convenience module for sequential Bayesian updating.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..core.distribution import Distribution
from ..core.node import Module, WorkflowFunction, workflow_method
from ..core.transition import StepResult

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
    distribution and a likelihood, then conditions on observed data
    using ProbPipe's standard ``condition_on`` dispatch (which selects
    NUTS, HMC, or another backend via the inference method registry).

    If the current distribution does not support
    :class:`~probpipe.core.protocols.SupportsLogProb`, it is
    automatically converted (e.g., MCMC samples → KDE) via ProbPipe's
    converter registry before building the model.

    Directly callable as a step function for use with
    :func:`~probpipe.core.transition.iterate`.

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

    Examples
    --------
    ::

        # Uses condition_on dispatch (NUTS by default):
        step = ConditioningStep(likelihood)
        trace = iterate(step, prior, data_batches)

        # Force a specific inference method:
        step = ConditioningStep(likelihood, method="tfp_nuts", num_results=2000)
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
    """Convenience wrapper for single-batch Bayesian conditioning.

    Pairs a prior with a :class:`ConditioningStep` and exposes
    :meth:`update` for conditioning on one data batch at a time.

    For **multi-batch** sequential updating, use the :attr:`step`
    property with :func:`~probpipe.core.transition.iterate`::

        trace = iterate(conditioner.step, prior, data_batches)

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

        # Single-batch update:
        result = conditioner.update(data=batch1)
        result.distribution  # the posterior

        # Multi-batch via iterate:
        trace = iterate(conditioner.step, prior, [batch1, batch2, batch3])
        trace.final          # final posterior
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

    @property
    def step(self) -> ConditioningStep:
        """The underlying step function, usable with ``iterate``."""
        return self._step

    @workflow_method
    def update(self, data: D) -> StepResult[P]:
        """Condition the prior on a single data batch.

        Parameters
        ----------
        data : D
            Observed data to condition on.

        Returns
        -------
        StepResult[P]
            The posterior distribution and any auxiliary info.
        """
        return self._step(self._prior, data)
