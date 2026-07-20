"""Generic distribution base class and minimal helpers.

Provides:
  - ``Distribution[T]`` – Abstract base for all ProbPipe distributions.
  - Global defaults for expectation sampling.
  - ``_auto_key()`` helper for convenience PRNG key generation.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..diagnostics.views import DiagnosticsView
    from ._distribution_array import DistributionArray

from .tracked import Annotated, Tracked

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


class Distribution[T](Tracked, Annotated, ABC):
    """
    Abstract base for all ProbPipe distributions, parameterized by
    value type ``T``.

    Every distribution is a tracked term: it is
    :class:`~probpipe.core.tracked.Tracked` (a :attr:`~Tracked.name`, a
    :attr:`~Tracked.name_is_auto` flag, and a write-once
    :attr:`~Tracked.provenance`) and
    :class:`~probpipe.core.tracked.Annotated` (free-form
    :attr:`~Annotated.annotations`).  Leaf distributions (Normal, Gamma,
    etc.) require an explicit ``name=`` argument; composite distributions
    (ProductDistribution, EmpiricalDistribution, etc.) auto-derive a
    name from their components when one is not provided, and mark it with
    ``name_is_auto=True``.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.core.protocols.SupportsSampling` protocol.

    Parameters
    ----------
    name : str
        Non-empty name for this distribution.
    name_is_auto : bool, optional
        ``True`` when *name* was auto-derived by the caller (a subclass
        constructor or an operation) rather than supplied by the user.
        Defaults to ``False``.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string.
    """

    def __init__(self, *, name: str, name_is_auto: bool = False):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{type(self).__name__} requires a non-empty name= argument")
        self._init_tracked(name, name_is_auto=name_is_auto)

    # -- keyword-form value construction ------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Any:
        """Build a single draw of this distribution's value type ``T`` from
        named field kwargs — the adapter behind the keyword form of the
        log_prob-family ops (``log_prob(dist, field=value, ...)``).

        Delegates field validation and ``Record`` construction to the general
        :func:`~probpipe.core.record._pack_fields` and layers this
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
        and let ``Function`` broadcasting handle it.

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
        # Local import: ``DistributionArray`` lives in the same
        # subpackage and importing at module top would create a cycle
        # (DistributionArray inherits from Distribution).
        from ._distribution_array import DistributionArray

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
