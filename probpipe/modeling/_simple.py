"""SimpleModel: construct a model from a prior and likelihood."""

from __future__ import annotations

from typing import Any

from ..core.distribution import Distribution
from ..core.event_template import EventTemplate
from ..core.protocols import SupportsLogProb
from ..core.record import Record
from ..core.tracked import auto_name
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

    **Named components:** merged from the prior's ``event_template``
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
        # ``RecordDistribution`` (so its ``event_template`` is a
        # required, non-``None`` ``EventTemplate``).
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
                f"event_template); got {type(prior).__name__}."
            )
        self._prior = prior
        self._likelihood = likelihood
        # Default to the class name when the caller does not supply one;
        # the default is an auto-derived name.
        name, name_is_auto = auto_name(name or None, "SimpleModel")
        self._init_tracked(name, name_is_auto=name_is_auto)

        # Build merged event_template: prior params + likelihood data fields.
        # This makes fields include both parameter and data names,
        # so condition_on can use component names as the sole signal for
        # splitting data kwargs from inference kwargs.
        #
        # ``prior_tpl`` is contractually non-``None`` (the
        # ``isinstance(prior, RecordDistribution)`` guard above implies
        # the metaclass invariant); ``data_tpl`` may be ``None`` for
        # likelihoods that don't declare a data template.
        prior_tpl: EventTemplate = prior.event_template
        data_tpl = getattr(likelihood, "data_template", None)
        # Convert legacy ``Record``-typed data templates to
        # ``EventTemplate``. ``Record`` and ``EventTemplate`` are
        # unrelated types, so the ``Record`` check is sufficient on
        # its own.
        if isinstance(data_tpl, Record):
            data_tpl = EventTemplate.infer_from(data_tpl)
        if data_tpl is not None:
            overlap = set(prior_tpl.fields) & set(data_tpl.fields)
            if overlap:
                raise ValueError(f"Parameter and data field names overlap: {overlap}")
            # Combine at the one-level (``children``) view so a nested prior/data
            # subtree is carried over whole rather than indexed by a top-level
            # subtree name (which leaf-keyed ``[]`` would reject).
            merged: dict[str, Any] = {**dict(prior_tpl.children), **dict(data_tpl.children)}
            self._event_template: EventTemplate = EventTemplate(merged)
        else:
            self._event_template = prior_tpl

    # -- Distribution interface ---------------------------------------------

    @property
    def prior(self) -> SupportsLogProb[P]:
        """The prior distribution over parameters."""
        return self._prior

    @property
    def likelihood(self) -> Likelihood[P, D]:
        """The likelihood function ``log p(D | params)``."""
        return self._likelihood

    @property
    def event_template(self) -> EventTemplate:
        """Merged ``EventTemplate`` over prior fields + likelihood data fields.

        ``SimpleModel`` is not itself a :class:`RecordDistribution`, but
        it carries a template so :attr:`fields`, conditioning, and
        inference kwarg splitting can address parameters and data
        uniformly. The template is always set — the prior's template
        is guaranteed non-``None`` by the ``RecordDistribution``
        invariant, and the prior's fields are the floor.
        """
        return self._event_template

    # -- Named components interface ------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return self.event_template.fields

    @property
    def _prior_fields(self) -> tuple[str, ...]:
        """Prior field names in template (insertion) order."""
        return self._prior.event_template.fields

    @property
    def _data_fields(self) -> tuple[str, ...]:
        """Likelihood data field names in template (insertion) order."""
        tpl = getattr(self._likelihood, "data_template", None)
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
        raise KeyError(f"Unknown component: {key!r}; available: {self.fields}")

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self._prior_fields:
            return tuple(self._prior_fields)
        return ("parameters",)

    # -- SupportsLogProb interface -----------------------------------------

    def _log_prob(self, value: Record | tuple[P, D]) -> Array:
        """Joint log-density: prior log-prob + log-likelihood.

        Accepts either form:

        * **Record** — a single record carrying all of :attr:`fields`
          (the prior's parameter fields plus the likelihood's named data
          fields). This is what the keyword API
          (``log_prob(model, intercept=..., y=...)``) produces. It is
          split into a parameter value — repacked via the prior's own
          :meth:`~probpipe.core._distribution_base.Distribution._pack_value`
          so a single-field prior receives a bare array and a multi-field
          prior a ``Record`` — and a data sub-record built from the
          likelihood's ``data_template`` fields.
        * **(params, data) pair** — the explicit joint form. Required when
          the likelihood's data has no named template (bare arrays), and
          retained for backward compatibility.

        Parameters
        ----------
        value : Record or tuple[P, D]
            All model fields as a record, or an explicit ``(params, data)``
            pair.
        """
        params, data = self._split_log_prob_value(value)
        lp = self._prior._log_prob(params)
        ll = self._likelihood.log_likelihood(params=params, data=data)
        return lp + ll

    def _split_log_prob_value(self, value: Record | tuple[P, D]) -> tuple[Any, Any]:
        """Resolve ``value`` (Record or (params, data) pair) into
        ``(params, data)`` in the forms the prior and likelihood expect."""
        # A Record is not a tuple, so the tuple check is unambiguous.
        if isinstance(value, tuple) and len(value) == 2:
            return value
        if isinstance(value, Record):
            data_fields = self._data_fields
            if not data_fields:
                # The likelihood has no named data fields (no data_template),
                # so the Record / keyword form cannot supply the data. Reject
                # loudly rather than silently passing data=None downstream.
                raise TypeError(
                    f"{type(self).__name__}: the keyword/Record form is "
                    f"unavailable because the likelihood "
                    f"({type(self._likelihood).__name__}) has no named data "
                    f"fields (no data_template); pass a (params, data) pair "
                    f"positionally instead."
                )
            params = self._prior._pack_value(**{f: value[f] for f in self._prior_fields})
            data = Record(**{f: value[f] for f in data_fields})
            return params, data
        raise TypeError(
            f"SimpleModel._log_prob expects a Record over {self.fields} or a "
            f"(params, data) pair; got {type(value).__name__}."
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        prior_name = type(self._prior).__name__
        lik_name = type(self._likelihood).__name__
        return f"SimpleModel(prior={prior_name}, likelihood={lik_name})"
