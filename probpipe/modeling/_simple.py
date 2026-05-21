"""SimpleModel: construct a model from a prior and likelihood."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ..core.record import Record, RecordTemplate
from ..custom_types import Array
from ._base import ProbabilisticModel
from ._likelihood import Likelihood

__all__ = ["SimpleModel"]


class SimpleModel[P, D](ProbabilisticModel[tuple[P, D]], SupportsLogProb):
    """Probabilistic model as a joint distribution over (parameters, data).

    A ``SimpleModel[P, D]`` is a ``Distribution[tuple[P, D]]`` — the joint
    distribution $p(\\theta, y) = p(\\theta) \\, p(y \\mid \\theta)$.
    The prior must support :class:`SupportsLogProb` so that the joint
    log-density is always computable.

    **Named components:** merged from the prior's ``record_template``
    and the likelihood's ``data_template`` when both are available.
    For example, a GLM model might have
    ``fields == ("X", "intercept", "slope", "y")``.
    Falls back to ``("parameters", "data")`` when templates are absent.

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
        prior: SupportsLogProb[P],
        likelihood: Likelihood[P, D],
        *,
        name: str | None = None,
    ):
        # Type-annotated as ``SupportsLogProb[P]`` so static type
        # checkers catch a wrong-type prior at the call site. The
        # runtime checks remain as a backstop for callers who bypass
        # the type system: the prior must be both ``SupportsLogProb``
        # (so the joint log-density is computable) and a
        # ``RecordDistribution`` (so its ``record_template`` is a
        # required, non-``None`` ``RecordTemplate``).
        from ..core.distribution import RecordDistribution
        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"SimpleModel requires a prior that supports SupportsLogProb, "
                f"got {type(prior).__name__}"
            )
        if not isinstance(prior, RecordDistribution):
            raise TypeError(
                f"SimpleModel requires a prior that is a "
                f"RecordDistribution (has named fields via "
                f"record_template); got {type(prior).__name__}."
            )
        self._prior = prior
        self._likelihood = likelihood
        # ``Distribution`` metaclass requires a non-empty name; default
        # to the class name when the caller doesn't supply one.
        self._name = name if name else "SimpleModel"

        # Build merged record_template: prior params + likelihood data fields.
        # This makes fields include both parameter and data names,
        # so condition_on can use component names as the sole signal for
        # splitting data kwargs from inference kwargs.
        prior_tpl = prior.record_template
        data_tpl = getattr(likelihood, 'data_template', None)
        # Convert legacy Record templates to RecordTemplate
        if isinstance(data_tpl, Record) and not isinstance(data_tpl, RecordTemplate):
            data_tpl = RecordTemplate.from_record(data_tpl)
        if prior_tpl is not None and data_tpl is not None:
            overlap = set(prior_tpl.fields) & set(data_tpl.fields)
            if overlap:
                raise ValueError(
                    f"Parameter and data field names overlap: {overlap}"
                )
            merged = {}
            for f in prior_tpl.fields:
                merged[f] = prior_tpl[f]
            for f in data_tpl.fields:
                merged[f] = data_tpl[f]
            self._record_template = RecordTemplate(merged)
        elif prior_tpl is not None:
            self._record_template = prior_tpl

    # -- Distribution interface ---------------------------------------------

    @property
    def prior(self) -> SupportsLogProb[P]:
        """The prior distribution over parameters."""
        return self._prior

    @property
    def likelihood(self) -> Likelihood[P, D]:
        """The likelihood function ``log p(D | params)``."""
        return self._likelihood

    # -- Named components interface ------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        tpl = self.record_template
        if tpl is not None:
            return tpl.fields
        return ("parameters", "data")

    @property
    def _prior_fields(self) -> tuple[str, ...]:
        """Prior field names in template (insertion) order."""
        tpl = self._prior.record_template
        return tpl.fields if tpl is not None else ()

    @property
    def _data_fields(self) -> tuple[str, ...]:
        """Likelihood data field names in template (insertion) order."""
        tpl = getattr(self._likelihood, 'data_template', None)
        return tpl.fields if tpl is not None else ()

    def __getitem__(self, key: str) -> Distribution | Likelihood:
        if key in self._data_fields:
            return self._likelihood
        if key in self._prior_fields:
            return self._prior
        # Fallback for unstructured models
        if key == "data":
            return self._likelihood
        if key == "parameters":
            return self._prior
        raise KeyError(
            f"Unknown component: {key!r}; "
            f"available: {self.fields}"
        )

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self._prior_fields:
            return tuple(self._prior_fields)
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

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        prior_name = type(self._prior).__name__
        lik_name = type(self._likelihood).__name__
        return f"SimpleModel(prior={prior_name}, likelihood={lik_name})"
