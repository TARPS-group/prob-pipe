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
    ``component_names == ("X", "intercept", "slope", "y")``.
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

        # Build merged record_template: prior params + likelihood data fields.
        # This makes component_names include both parameter and data names,
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
    def name(self) -> str | None:
        return self._name_str

    # -- Named components interface ------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        tpl = self.record_template
        if tpl is not None:
            return tpl.fields
        return ("parameters", "data")

    @property
    def _prior_fields(self) -> frozenset[str]:
        tpl = self._prior.record_template
        return frozenset(tpl.fields) if tpl is not None else frozenset()

    @property
    def _data_fields(self) -> frozenset[str]:
        tpl = getattr(self._likelihood, 'data_template', None)
        return frozenset(tpl.fields) if tpl is not None else frozenset()

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
            f"available: {self.component_names}"
        )

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self._prior_fields:
            return tuple(sorted(self._prior_fields))
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
