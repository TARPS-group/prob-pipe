"""The distribution base class, its term spec, and minimal helpers.

Provides:
  - ``Distribution`` – Abstract base for all ProbPipe distributions.
  - ``DistributionSpec`` – The term spec of the distribution kind.
  - Global defaults for expectation sampling.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core._distribution_array import DistributionArray
    from ..diagnostics.views import DiagnosticsView

from ..core._record_spec import RecordSpec, _check_kind_of, _schema_carried_by
from ..core._spec_base import TermSpec, _unify_specs
from ..core.provenance import Provenance
from ..core.tracked import Annotated, TrackedTerm

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_EVALUATIONS: int = 1024
"""Default number of function evaluations for sample-based expectations."""

RETURN_APPROX_DIST: bool = True
"""When True, approximate expectations return a BootstrapDistribution
capturing MC error instead of a plain array."""


def set_default_num_evaluations(n: int) -> None:
    """Set the global default for ``expectation()`` on infinite-support distributions."""
    global DEFAULT_NUM_EVALUATIONS
    if n < 1:
        raise ValueError("num_evaluations must be at least 1")
    DEFAULT_NUM_EVALUATIONS = n


def set_return_approx_dist(value: bool) -> None:
    """Set whether approximate expectations return error-tracking distributions."""
    global RETURN_APPROX_DIST
    RETURN_APPROX_DIST = bool(value)


# ---------------------------------------------------------------------------
# Distribution — the base class
# ---------------------------------------------------------------------------


class Distribution(TrackedTerm, Annotated, ABC):
    """
    Abstract base for all ProbPipe distributions.

    Every distribution is a tracked term: it is
    :class:`~probpipe.core.tracked.TrackedTerm` (a :attr:`~TrackedTerm.name` and a write-once
    :attr:`~TrackedTerm.provenance`) and
    :class:`~probpipe.core.tracked.Annotated` (free-form
    :attr:`~Annotated.annotations`).  Leaf distributions (Normal, Gamma,
    etc.) require an explicit ``name=`` argument; composite distributions
    (ProductDistribution, EmpiricalDistribution, etc.) auto-derive a
    name from their components when one is not provided. Every transform
    preserves the name; only ``with_name`` replaces it.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.core.protocols.SupportsSampling` protocol.

    Parameters
    ----------
    name : str
        Non-empty name for this distribution.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string.
    """

    # -- Immutability: deferred for this layer ------------------------------

    def __delattr__(self, name: str) -> None:
        """Permit deletion, for the reason :meth:`__setattr__` gives.

        Goes with that method: removing the exemption means removing both.
        """
        object.__delattr__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Permit assignment, which :class:`TrackedTerm` otherwise refuses.

        Interim, and the only exemption from the rule that a tracked term is
        immutable. It stands because the contract for a *fitted* mapping is not
        settled: the documented way to build an emulator is to subclass a random
        function and train it in place, and until fitting has a contract that
        produces a new term instead, enforcing immutability here would break that
        pattern without offering a replacement.

        Deleting this method **and** :meth:`__delattr__` turns the guard on for
        the whole distribution layer. Both, or the layer keeps half an
        exemption: a trainer that clears what it fitted would still raise.
        """
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        name: str,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{type(self).__name__} requires a non-empty name= argument")
        # ``_provenance`` and ``_annotations`` carry state a reconstruction
        # already holds and that construction cannot otherwise reach: provenance
        # is write-once, and annotations are written after construction, so a
        # rebuilt distribution would come back without either. Private, and the
        # reconstruction paths are the only callers.
        self._init_tracked(name, provenance=_provenance)
        self._init_annotations(_annotations)

    # -- keyword-form value construction ------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Any:
        """Build a single draw of this distribution from
        named field kwargs — the adapter behind the keyword form of the
        log_prob-family ops (``log_prob(dist, field=value, ...)``).

        Delegates field validation and ``Record`` construction to the general
        :func:`~probpipe.core.record._pack_fields` and layers this
        distribution's value-type convention on top:

        * **single field** → the bare field value, an array, so a scalar
          distribution's ``_log_prob`` still receives a raw array.
        * **multiple fields** → the :class:`~probpipe.core.record.Record`
          built from the named fields.

        Distributions whose ``_log_prob`` consumes a Record but splits it
        internally (e.g. ``SimpleModel`` → ``(params, data)``) keep this
        default and do the split in ``_log_prob``. Override only when the
        value type is neither a bare array nor a flat Record (e.g.
        ``StanModel``'s single ``parameters=`` flat array).

        Builds exactly one draw (``sample_shape == ()``). Batched
        evaluation does not go through kwargs — pass the batch positionally
        and let ``Function`` broadcasting handle it.

        Raises
        ------
        TypeError
            If the distribution has no named fields, or the kwargs do not
            match its fields exactly (missing or unexpected names).
        """
        from ..core.record import _pack_fields

        fields = getattr(self, "fields", None)
        if not fields:
            raise TypeError(
                f"{type(self).__name__} does not support the keyword form of "
                f"the log_prob-family ops (it has no named fields); pass a "
                f"positional value."
            )
        rec = _pack_fields(fields, field_kwargs, owner=type(self).__name__)
        return field_kwargs[fields[0]] if len(fields) == 1 else rec

    # -- approximation tracking ---------------------------------------------

    @property
    def is_approximate(self) -> bool:
        """Whether this distribution is an approximation.

        Approximate distributions are typically produced by sampling,
        variational inference, MCMC, bootstrap procedures, or other numerical
        approximations.
        """
        return getattr(self, "_approximate", False)

    # -- annotations ---------------------------------------------------------
    #
    # ``annotations`` (the general post-construction metadata store) is
    # provided by the :class:`~probpipe.core.tracked.Annotated` mixin.
    # On a fitted posterior the conventional layout is an ``xarray.DataTree``
    # with ``arviz/`` and ``diagnostics/`` subtrees; see :attr:`diagnostics`.

    @property
    def diagnostics(self) -> DiagnosticsView | None:
        """Structured view over diagnostic results stored in :attr:`annotations`.

        Returns ``None`` if no diagnostics have been computed yet.

        Inference backends and the diagnostics subsystem store posterior
        metadata in an ``xarray.DataTree`` attached to the distribution as
        its :attr:`~probpipe.core.tracked.Annotated.annotations`. The
        expected layout is::

            posterior._annotations
            ├── arviz/          # ArviZ-compatible data and raw inputs
            │   ├── posterior
            │   ├── sample_stats
            │   ├── observed_data
            │   ├── posterior_predictive
            │   └── log_likelihood
            └── diagnostics/    # ProbPipe-computed results and metadata
                ├── mcmc        # rhat, ess_bulk, ess_tail, mcse, ...
                └── runs        # on-demand diagnostics such as ppc, loo, spc
                    ├── ppc
                    ├── loo
                    └── spc

        The ``/arviz/`` subtree is intended to be passed to ArviZ functions
        and to hold raw diagnostic ingredients such as sampler statistics,
        posterior predictive samples, and pointwise log likelihoods. In ArviZ
        1.0+, this is an ArviZ-compatible ``DataTree`` rather than the older
        ``InferenceData`` representation.

        This property returns a structured Python accessor over the
        ``/diagnostics/`` subtree only. It is distinct from the ArviZ-compatible
        data used for plotting or ArviZ computations.

        In other words::

            posterior.diagnostics
                # structured ProbPipe view over posterior.annotations["diagnostics"]

            posterior.arviz_data
                # ArviZ-compatible xarray DataTree subtree, typically
                # posterior.annotations["arviz"]

            posterior.inference_data
                # backward-compatible alias for posterior.arviz_data

        Examples
        --------
        ::

            posterior = condition_on(model, data)

            # MCMC diagnostics mutate posterior._annotations in place and return None.
            add_mcmc_diagnostics(posterior)

            posterior.diagnostics.rhat
            # {"intercept": 1.001, "slope": 1.002}

            posterior.diagnostics.warnings
            # []

            posterior.diagnostics.runs
            # []

            # Posterior predictive checks are stored under diagnostics/runs/ppc.
            add_ppc(
                posterior,
                test_fns=[...],
                observed_data=y,
                generative_likelihood=lik,
            )

            posterior.diagnostics.ppc.result
            # {"var_mean_ratio": {"p_value": 0.43, "observed": 3.2}}

            posterior.diagnostics.runs[0].result
            # {"p_value": {"var_mean_ratio": 0.43}, ...}

            posterior.diagnostics.runs[0].plot_fn
            # "" unless the run wrote ArviZ-compatible plotting inputs

        Notes
        -----
        The diagnostics accessor is read-only. Diagnostic functions such as
        ``add_mcmc_diagnostics`` and ``add_ppc`` are responsible for writing
        diagnostic results into ``posterior._annotations``.
        """
        aux = self.annotations
        if aux is None:
            return None
        children = aux.children if hasattr(aux, "children") else {}
        if "diagnostics" not in children:
            return None
        from ..diagnostics.views import DiagnosticsView

        return DiagnosticsView(aux["diagnostics"])

    # -- batched-construction alias ----------------------------------------

    @classmethod
    def from_batched_params(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params,
    ) -> DistributionArray:
        """Class-method alias for :meth:`DistributionArray.from_batched_params`.

        Lets users write the ergonomic per-class form::

            Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")

        instead of the universal entry point::

            DistributionArray.from_batched_params(
                Normal, loc=jnp.zeros(5), scale=1.0, name="x",
            )

        Both produce the same ``DistributionArray`` — the alias is a
        thin classmethod that calls the universal factory with
        ``cls`` bound. Subclasses inherit the alias automatically;
        no per-family override is needed.

        See :meth:`DistributionArray.from_batched_params` for the full
        contract (dispatch on
        :class:`~probpipe.core.protocols.SupportsArrayBackend`,
        ``batch_shape`` inference, per-cell name suffixing).
        """
        # Local import: ``DistributionArray`` inherits from ``Distribution``,
        # so importing it at module top would create a cycle.
        from ..core._distribution_array import DistributionArray

        return DistributionArray.from_batched_params(
            cls,
            name=name,
            batch_shape=batch_shape,
            **batched_params,
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# DistributionSpec — the term spec of the distribution kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class DistributionSpec(TermSpec):
    """A distribution-kind spec whose current draw schema is a RecordSpec.

    Parameters
    ----------
    event_spec : RecordSpec
        The record schema describing one draw.

    Raises
    ------
    TypeError
        If the draw schema is not a RecordSpec.

    Notes
    -----
    ``is_valid`` requires a Distribution whose ``event_template`` equals
    ``event_spec``, including field order, dtype, and support metadata. A missing
    template, or a getter raising AttributeError or TypeError because the schema
    is unavailable, gives False; other getter errors propagate.

    ``bind_dims_from_value`` validates a concrete declaration by that same rule.
    For a symbolic declaration it instead learns sizes from the distribution's
    schema, without sampling. An unavailable schema raises ValueError.
    ``bind_dims_from_spec`` reads another DistributionSpec using declaration
    compatibility: matching structure, compatible declared dtypes, and consistent
    sizes; field order and support metadata need not be equal.

    Binding returns a new spec retaining the declared metadata. Repeated symbols
    share one scope, including surrounding records or input slots; conflicting
    sizes raise ValueError. ``with_dim_sizes`` may leave unsupplied dimensions symbolic.

    Examples
    --------
    >>> from probpipe import DistributionSpec, RecordSpec
    >>> declared = DistributionSpec(RecordSpec(x=("n",)))
    >>> bound = declared.bind_dims_from_spec(DistributionSpec(RecordSpec(x=(3,))))
    >>> bound.event_spec["x"].shape
    (3,)
    """

    event_spec: RecordSpec

    def __init__(self, event_spec: RecordSpec) -> None:
        if not isinstance(event_spec, RecordSpec):
            raise TypeError(
                f"DistributionSpec.event_spec must be a RecordSpec, got {type(event_spec).__name__}"
            )
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the draw this declares."""
        return self.event_spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> DistributionSpec:
        """This spec around a substituted event declaration."""
        return DistributionSpec(self.event_spec._substitute_dims(bindings))

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Validate a concrete draw schema, or bind a symbolic one from *value*."""
        if not self.free_dims:
            super()._bind_dims_from_value(value, bindings, path)
            return
        actual = _schema_carried_by(value, self, path)
        _check_kind_of(DistributionSpec(actual), value, self, path)
        _unify_specs(self.event_spec, actual, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the declared draw schema against *actual*'s own."""
        if not isinstance(actual, DistributionSpec):
            return False
        _unify_specs(self.event_spec, actual.event_spec, bindings, path)
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` matching this event declaration.

        *value* must be a :class:`~probpipe.Distribution` whose own
        ``event_template`` equals the declared record template. A distribution
        that is not one, or that legitimately exposes no template — no
        ``event_template`` attribute, or a template that cannot yet be
        derived — does not satisfy the spec and returns ``False``. These are
        the only two "schema unavailable" conditions treated as a non-match;
        any *other* error raised while reading ``event_template`` signals a
        malfunctioning distribution and is left to propagate rather than being
        masked as invalid.
        """
        if not isinstance(value, Distribution):
            return False
        try:
            template = value.event_template
        except (AttributeError, TypeError):
            # The two documented "schema unavailable" signals: no
            # ``event_template`` attribute (AttributeError) or a template that
            # cannot be derived (TypeError — e.g. an un-named auto-deriving
            # distribution). Both mean the value can't be certified. A
            # narrower catch than ``Exception`` on purpose: an unexpected
            # error is a bug to surface, not a silent "invalid".
            return False
        # Normalised at construction, so the declaration is always a RecordSpec.
        return template == self.event_spec
