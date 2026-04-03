"""SimpleModel: construct a model from a prior and likelihood."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ..custom_types import Array
from ..inference._mcmc_distribution import MCMCApproximateDistribution
from ._base import ProbabilisticModel
from ._likelihood import Likelihood

__all__ = ["SimpleModel"]


class SimpleModel[P, D](ProbabilisticModel[tuple[P, D]], SupportsLogProb):
    """Probabilistic model as a joint distribution over (parameters, data).

    A ``SimpleModel[P, D]`` is a ``Distribution[tuple[P, D]]`` — the joint
    distribution $p(\\theta, y) = p(\\theta) \\, p(y \\mid \\theta)$.
    The prior must support :class:`SupportsLogProb` so that the joint
    log-density is always computable.

    **Named components:** ``"parameters"`` (the prior) and ``"data"``
    (the likelihood).  Only ``"data"`` is conditionable.

    **Log-prob:** ``_log_prob((params, data))`` returns the joint
    log-density: ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``.

    **Conditioning:** ``_condition_on(data)`` delegates to the inference
    method registry, which auto-selects NUTS (if JAX-traceable) or RWMH
    (gradient-free fallback).

    Parameters
    ----------
    prior : Distribution[P] that supports SupportsLogProb
        Prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Must have a ``log_likelihood(params, data)`` method.
    name : str or None
        Model name for provenance.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        prior: Distribution[P],
        likelihood: Likelihood[P, D],
        *,
        name: str | None = None,
    ):
        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"SimpleModel requires a prior that supports SupportsLogProb, "
                f"got {type(prior).__name__}"
            )
        self._prior = prior
        self._likelihood = likelihood
        self._name_str = name

    # -- Distribution interface ---------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name_str

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return ("parameters", "data")

    def __getitem__(self, key: str) -> Any:
        if key == "parameters":
            return self._prior
        if key == "data":
            return self._likelihood
        raise KeyError(
            f"Unknown component: {key!r}; "
            f"available: {self.component_names}"
        )

    # -- SupportsConditionableComponents interface --------------------------

    @property
    def conditionable_components(self) -> dict[str, bool]:
        return {"parameters": False, "data": True}

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("parameters",)

    # -- SupportsLogProb interface -----------------------------------------

    def _log_prob(self, value: tuple[P, D]) -> Array:
        """Joint log-density: prior log-prob + log-likelihood.

        Parameters
        ----------
        value : tuple[P, D]
            A ``(params, data)`` pair.
        """
        params, data = value
        lp = self._prior._log_prob(params)
        ll = self._likelihood.log_likelihood(params=params, data=data)
        return lp + ll

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed: D, /, **kwargs: Any) -> MCMCApproximateDistribution:
        """Condition on observed values, returning posterior samples.

        Delegates to the inference method registry, which auto-selects
        the best method (NUTS if JAX-traceable, RWMH otherwise).
        Pass ``method="tfp_hmc"`` etc. to override.
        """
        from ..inference import inference_method_registry
        return inference_method_registry.execute(self, observed, **kwargs)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        prior_name = type(self._prior).__name__
        lik_name = type(self._likelihood).__name__
        return f"SimpleModel(prior={prior_name}, likelihood={lik_name})"
