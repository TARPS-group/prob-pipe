"""Generic distribution base class and minimal helpers.

Provides:
  - ``Distribution[T]`` – Abstract base for all ProbPipe distributions.
  - Global defaults for expectation sampling.
  - ``_auto_key()`` helper for convenience PRNG key generation.
"""

from __future__ import annotations

import copy as _copy
from abc import ABC, abstractmethod
# ``_ProtocolMeta`` is technically private (leading underscore in
# ``typing``), but it's the only way to compose a custom metaclass with
# ``@runtime_checkable`` protocols without a metaclass conflict.  The
# name has been stable since Python 3.7 and is widely used in the
# ecosystem (Pydantic, attrs, etc.). If a future Python release renames
# it, the metaclass would need to switch to whatever new base ``typing``
# exposes; the conflict-avoidance constraint itself doesn't change.
from typing import TYPE_CHECKING, Any, _ProtocolMeta  # noqa: PLC2701

if TYPE_CHECKING:
    from xarray import DataTree

    from ._distribution_array import DistributionArray
    from .record import RecordTemplate

import jax
import jax.numpy as jnp

from ..custom_types import PRNGKey
from .provenance import Provenance

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
# Distribution[T] — generic base class
# ---------------------------------------------------------------------------


class _DistributionMeta(_ProtocolMeta):
    """Metaclass enforcing that every Distribution instance has a
    non-empty ``name`` set by the time construction returns.

    The check runs after ``__init__`` so it catches both subclasses
    that call ``super().__init__(name=...)`` and subclasses that
    bypass it and set ``self._name`` directly. The only failure case
    is a subclass that finishes ``__init__`` without setting
    ``_name`` to a non-empty string — then construction raises
    ``TypeError``.

    Extends ``typing._ProtocolMeta`` (rather than the more obvious
    ``ABCMeta``) so subclasses can still mix in ``@runtime_checkable``
    protocols (``SupportsSampling``, ``SupportsLogProb``, …) without
    a metaclass conflict. ``_ProtocolMeta`` is itself an
    ``ABCMeta`` subclass.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        name = getattr(instance, "_name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{cls.__name__}.__init__ must set a non-empty name "
                f"(via super().__init__(name=...) or by assigning "
                f"self._name to a non-empty string) before returning."
            )
        return instance


class Distribution[T](ABC, metaclass=_DistributionMeta):
    """
    Abstract base for all ProbPipe distributions, parameterized by
    value type ``T``.

    Every distribution has a ``name``.  Leaf distributions (Normal, Gamma,
    etc.) require an explicit ``name=`` argument; composite distributions
    (ProductDistribution, EmpiricalDistribution, etc.) auto-generate a
    name from their components when one is not provided.

    Provides naming, provenance, conversion, and approximation tracking.
    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.core.protocols.SupportsSampling` protocol.
    """

    def __init__(self, *, name: str):
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{type(self).__name__} requires a non-empty name= argument"
            )
        self._name = name

    # -- keyword-form value construction ------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Any:
        """Build a single draw of this distribution's value type ``T`` from
        named field kwargs — the adapter behind the keyword form of the
        log_prob-family ops (``log_prob(dist, field=value, ...)``).

        Delegates field validation and ``Record`` construction to the general
        :func:`~probpipe.core.record._pack_fields` (also exposed object-style
        as :meth:`~probpipe.core.record.RecordTemplate.pack`) and layers this
        distribution's value-type convention on top:

        * **single field** → the bare field value (``T = Array``), so a
          scalar distribution's ``_log_prob`` still receives a raw array.
        * **multiple fields** → the :class:`~probpipe.core.record.Record`
          built from the named fields (``T = Record``).

        Distributions whose ``_log_prob`` consumes a Record but splits it
        internally (e.g. ``SimpleModel`` → ``(params, data)``) keep this
        default and do the split in ``_log_prob``. Override only when the
        value type is neither a bare array nor a flat Record (e.g.
        ``StanModel``'s single ``parameters=`` flat array).

        Builds exactly one draw (``sample_shape == ()``). Batched
        evaluation does not go through kwargs — pass the batch positionally
        and let ``WorkflowFunction`` broadcasting handle it.

        Raises
        ------
        TypeError
            If the distribution has no named fields, or the kwargs do not
            match its fields exactly (missing or unexpected names).
        """
        from .record import _pack_fields
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

    # -- auxiliary information ----------------------------------------------

    @property
    def auxiliary(self) -> DataTree | None:
        """Auxiliary metadata attached to this distribution.

        Returns an ``xarray.DataTree`` of auxiliary information, or ``None``
        if no auxiliary information has been attached.

        ``_auxiliary`` is ProbPipe's general-purpose post-construction
        metadata store. It may contain ArviZ-compatible xarray DataTree data,
        diagnostic summaries, validation results, provenance-like metadata,
        or other append-only metadata produced after inference.

        The expected diagnostics-related layout is::

            posterior._auxiliary
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

        The ``/diagnostics/`` subtree contains ProbPipe-owned computed
        summaries, results, warning metadata, and run metadata. It is exposed
        through the structured ``posterior.diagnostics`` Python accessor.

        Documented exception to distribution immutability
        -------------------------------------------------
        Unlike parameter-like state on a :class:`Distribution`, ``_auxiliary``
        is designed to be mutated in place by inference backends, validators,
        and diagnostic operations after construction.

        Diagnostic functions such as ``add_mcmc_diagnostics`` and ``add_ppc``
        write into ``_auxiliary`` and return ``None``. This preserves posterior
        identity/source tracking while allowing diagnostics to annotate an
        already-fitted posterior.

        Treat this channel as append-only: new diagnostic operations should
        write under their own named group and should not overwrite or mutate
        parameter-like distribution state.
        """
        return getattr(self, "_auxiliary", None)

    @property
    def diagnostics(self) -> "DiagnosticsView | None":
        """Structured view over diagnostic results stored in ``_auxiliary``.

        Returns ``None`` if no diagnostics have been computed yet.

        ProbPipe stores auxiliary posterior metadata in a general-purpose
        ``xarray.DataTree`` attached to the distribution. The expected layout is::

            posterior._auxiliary
            ├── arviz/          # ArviZ-compatible data and raw inputs
            └── diagnostics/    # ProbPipe-computed results and metadata

        This property returns a structured Python accessor over the
        ``/diagnostics/`` subtree only. It is distinct from the ArviZ-compatible
        data used for plotting or ArviZ computations.

        In other words::

            posterior.diagnostics
                # structured ProbPipe view over posterior._auxiliary["diagnostics"]

            posterior.arviz_data
                # ArviZ-compatible xarray DataTree subtree, typically
                # posterior._auxiliary["arviz"]

            posterior.inference_data
                # backward-compatible alias for posterior.arviz_data

        Examples
        --------
        ::

            posterior = condition_on(model, data)

            # MCMC diagnostics mutate posterior._auxiliary in place and return None.
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
            # "az.plot_ppc"

        Notes
        -----
        The diagnostics accessor is read-only. Diagnostic functions such as
        ``add_mcmc_diagnostics`` and ``add_ppc`` are responsible for writing
        diagnostic results into ``posterior._auxiliary``.
        """
        aux = self.auxiliary
        if aux is None:
            return None
        children = aux.children if hasattr(aux, "children") else {}
        if "diagnostics" not in children:
            return None
        from ..diagnostics.views import DiagnosticsView
        return DiagnosticsView(aux["diagnostics"])

    # -- naming & provenance ------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this distribution."""
        return self._name

    @property
    def source(self) -> Provenance | None:
        """Provenance describing how this distribution was created, if any."""
        return getattr(self, "_source", None)

    def with_source(self, source: Provenance) -> Distribution:
        """Attach provenance to this distribution (write-once)."""
        if getattr(self, "_source", None) is not None:
            raise RuntimeError(
                f"Source already set on {self!r}. "
                "Provenance is write-once; create a new distribution instead."
            )
        self._source = source
        return self

    def renamed(self, new_name: str) -> Distribution:
        """Return a shallow copy with a different name.

        The copy shares all internal state but has a new ``name``.
        Provenance is tracked: the copy's ``source`` records the rename
        operation and points to the original as parent.

        ``RecordDistribution`` overrides this to also reset its cached
        ``_record_template`` so the auto-build path regenerates with
        the new name.
        """
        clone = _copy.copy(self)
        object.__setattr__(clone, "_name", new_name)
        # Bypass write-once guard so rename provenance can be attached
        object.__setattr__(clone, "_source", None)
        clone.with_source(
            Provenance(
                "renamed",
                parents=(self,),
                metadata={"old_name": self.name, "new_name": new_name},
            )
        )
        return clone

    # -- batched-construction alias ----------------------------------------

    @classmethod
    def from_batched_params(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params,
    ) -> "DistributionArray":
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
        # Local import: ``DistributionArray`` lives in the same
        # subpackage and importing at module top would create a cycle
        # (DistributionArray inherits from Distribution).
        from ._distribution_array import DistributionArray
        return DistributionArray.from_batched_params(
            cls, name=name, batch_shape=batch_shape, **batched_params,
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"
