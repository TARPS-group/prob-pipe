"""SimpleModel: construct a model from a prior and likelihood."""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec, RecordSpec, _components_record
from ..core.protocols import SupportsLogProb
from ..core.record import Record
from ..core.tracked import auto_name
from ..custom_types import Array
from ..distributions._distribution import Distribution
from ._base import ProbabilisticModel
from ._likelihood import Likelihood

__all__ = ["SimpleModel"]


class SimpleModel[P, D](ProbabilisticModel, SupportsLogProb):
    """Probabilistic model as a joint distribution over (parameters, data).

    A ``SimpleModel`` is the joint distribution
    $p(\\theta, y) = p(\\theta) \\, p(y \\mid \\theta)$ over parameters and data.
    The prior must support :class:`SupportsLogProb` so that the joint
    log-density is always computable.

    **Named components:** merged from the prior's declared components
    and the likelihood's ``data_template`` when it has one.
    For example, a GLM model might have
    ``fields == ("X", "intercept", "slope", "y")``.
    Falls back to ``("parameters", "data")`` when templates are absent.

    Parameters
    ----------
    prior : Distribution that supports SupportsLogProb
        Prior distribution over model parameters.
    likelihood : Likelihood[P, D]
        Must have a ``log_likelihood(params, data)`` method.
    name : str or None
        Model name for provenance.
        Keyword-only, as an interim detail (see :class:`~probpipe.Distribution`).
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        prior: SupportsLogProb,
        likelihood: Likelihood[P, D],
        *,
        name: str | None = None,
    ):
        # Type-annotated as ``SupportsLogProb`` so static type
        # checkers catch a wrong-type prior at the call site. The
        # runtime checks remain as a backstop for callers who bypass
        # the type system: the prior must be both ``SupportsLogProb``
        # (so the joint log-density is computable) and a
        # ``RecordDistribution``, whose declared components name the
        # parameters.
        from ..core._record_distribution import RecordDistribution

        if not isinstance(prior, SupportsLogProb):
            raise TypeError(
                f"SimpleModel requires a prior that supports SupportsLogProb, "
                f"got {type(prior).__name__}"
            )
        if not isinstance(prior, RecordDistribution):
            raise TypeError(
                f"SimpleModel requires a prior that is a "
                f"RecordDistribution, whose declared components name the "
                f"parameters; got {type(prior).__name__}."
            )
        self._prior = prior
        self._likelihood = likelihood
        # Default to the class name when the caller does not supply one;
        # the default is an auto-derived name.
        name = auto_name(name or None, "SimpleModel")
        self._init_tracked(name)

        # The merged record: the prior's parameters and the likelihood's data
        # fields. This makes fields include both parameter and data names,
        # so condition_on can use component names as the sole signal for
        # splitting data kwargs from inference kwargs.
        #
        # ``prior_tpl`` is the record the prior's declared components form;
        # ``data_tpl`` may be ``None`` for likelihoods that don't declare a
        # data template.
        prior_tpl: RecordSpec = _components_record(prior.event_spec)
        data_tpl = getattr(likelihood, "data_template", None)
        # Convert legacy ``Record``-typed data templates to
        # ``RecordSpec``. ``Record`` and ``RecordSpec`` are
        # unrelated types, so the ``Record`` check is sufficient on
        # its own.
        if isinstance(data_tpl, Record):
            data_tpl = RecordSpec.infer_from(data_tpl)
        if data_tpl is not None:
            overlap = set(prior_tpl.fields) & set(data_tpl.fields)
            if overlap:
                raise ValueError(f"Parameter and data field names overlap: {overlap}")
            # Combine at the one-level (``children``) view so a nested prior/data
            # subtree is carried over whole rather than indexed by a top-level
            # subtree name (which leaf-keyed ``[]`` would reject).
            merged: dict[str, Any] = {**dict(prior_tpl.children), **dict(data_tpl.children)}
            self._event_template: RecordSpec = RecordSpec(merged)
        else:
            self._event_template = prior_tpl
        # The model is a law over its parameters and data, the merged record.
        self._init_declaration(OutputSpec(self._event_template))

    # -- Distribution interface ---------------------------------------------

    @property
    def prior(self) -> SupportsLogProb:
        """The prior distribution over parameters."""
        return self._prior

    @property
    def likelihood(self) -> Likelihood[P, D]:
        """The likelihood function ``log p(D | params)``."""
        return self._likelihood

    @property
    def event_template(self) -> RecordSpec:
        """Merged ``RecordSpec`` over prior fields + likelihood data fields.

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
        return tuple(self.event_spec.components)

    @property
    def _prior_fields(self) -> tuple[str, ...]:
        """Prior field names, the prior's declared components in order."""
        return tuple(self._prior.event_spec.components)

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
          :meth:`~probpipe.Distribution._pack_value`
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
            data = Record("data", {f: value[f] for f in data_fields})
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
