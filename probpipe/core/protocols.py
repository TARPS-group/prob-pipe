"""Protocols that are not distribution capabilities.

The capability protocols of the distribution kinds are defined in
:mod:`probpipe.distributions._capabilities`. This module holds the rest:

- ``SupportsArrayBackend``, which a ``Distribution`` subclass implements to
  give ``DistributionArray`` a fused storage backend.
- ``GenerativeLikelihood``, the simulator protocol that
  :func:`~probpipe.validation.predictive_check` takes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from ..custom_types import PRNGKey

if TYPE_CHECKING:
    from ..distributions._distribution import Distribution
    from ._spec_base import TermSpec


# ---------------------------------------------------------------------------
# Array backend (fused storage for DistributionArray)
# ---------------------------------------------------------------------------


@runtime_checkable
class _DistributionArrayBackend(Protocol):
    """Internal storage backend that ``DistributionArray`` consumes.

    A backend owns the *batched* parameters of a homogeneous
    ``DistributionArray`` and delivers vectorised ops directly — TFP's
    native batch axis, a single empirical law whose atoms carry a leading
    batch dim, etc. It carries no ``name`` / ``provenance``
    and lives only as the contract between a distribution class's
    :meth:`SupportsArrayBackend._make_array_backend` and the array
    consumer.

    Backends are private to the library. User code never imports or
    constructs them; they exist solely so a
    :class:`~probpipe.DistributionArray` can fuse storage instead of
    materialising one ``Distribution`` per cell.

    Required surface
    ----------------
    Every backend exposes ``batch_shape``, ``event_shape``, ``cell_spec``, ``cell``,
    and the ``_sample``/``_log_prob``/``_mean``/``_variance``/``_cov``
    methods that mirror whichever moment / density protocols the
    underlying distribution class supports. ``DistributionArray``
    introspects via ``isinstance`` and forwards to whichever ones are
    present.

    ``cell(index)`` materialises a fresh **scalar** ``Distribution``
    (i.e. ``batch_shape == ()``) for the cell at ``index``. Used by
    ``DistributionArray.__getitem__`` and by the WF sweep when
    cell-level dispatch is needed.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    @property
    def cell_spec(self) -> TermSpec:
        """The term every cell draws, which an empty batch reports too."""
        ...

    def cell(self, index: int | tuple[int, ...]) -> Distribution:
        """Fabricate a scalar ``Distribution`` for the cell at ``index``."""
        ...


@runtime_checkable
class SupportsArrayBackend(Protocol):
    """Distribution class that supports efficient batched construction.

    Used by :meth:`DistributionArray.from_batched_params` to fuse
    storage when the caller's components are homogeneous instances of
    the same class. Implementations construct an internal
    :class:`_DistributionArrayBackend` that owns the batched parameters
    and the vectorised ops; ``DistributionArray`` becomes a thin
    consumer.

    Distribution classes that don't implement this protocol still work
    in a ``DistributionArray`` via the literal-array fallback (one
    ``Distribution`` instance per cell) — slower but correct.

    The protocol attaches to the **class**, not to instances. The
    runtime check is ``isinstance(MyDistribution, SupportsArrayBackend)``
    (i.e. the class itself implements ``_make_array_backend``).
    ``isinstance(an_instance, SupportsArrayBackend)`` returns
    ``True`` too — instances inherit class attributes, and
    ``runtime_checkable`` just looks for the named attribute — but
    the result is misleading because the contract is at class
    scope.

    The protocol is internal to the library; user code never calls
    ``_make_array_backend`` directly. ``DistributionArray`` is the
    sole consumer.

    Examples
    --------
    A distribution class declares the capability by implementing the
    classmethod::

        class MyDistribution(Distribution):
            @classmethod
            def _make_array_backend(
                cls,
                *,
                name: str,
                batch_shape: tuple[int, ...],
                **batched_params,
            ) -> _DistributionArrayBackend:
                return _MyArrayBackend(
                    cls=cls, name=name, batch_shape=batch_shape,
                    **batched_params,
                )
    """

    @classmethod
    def _make_array_backend(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> _DistributionArrayBackend:
        """Construct an array backend for this class.

        Parameters
        ----------
        name : str
            Base name; per-cell distributions auto-suffix as
            ``f"{name}_{i}"``.
        batch_shape : tuple of int
            The leading shape of the batched parameters. The backend
            stores parameters with this shape prepended to each
            ``cls(**kwargs)``-style argument.
        **batched_params
            Same keys as ``cls(**kwargs)`` would take, but with
            ``batch_shape`` prepended to each array argument.

        Returns
        -------
        _DistributionArrayBackend
            Backend instance owning the batched parameters and
            delivering vectorised ops.
        """
        ...


# ---------------------------------------------------------------------------
# Likelihoods and generative simulators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkflowGenerativeProviderCertificate:
    """Private exact-provider authority for workflow-owned generation."""

    provider_type: type[Any]
    generate_data: Callable[..., Any]
    provider_abi: str
    preflight: Callable[[Any, str], None]


@runtime_checkable
class GenerativeLikelihood[P, D](Protocol):
    """Protocol for generating synthetic data given parameters.

    Generic in ``P`` (parameter type) and ``D`` (data type).
    Any class that defines
    ``generate_data(params, num_observations, *, key) -> D``
    satisfies this protocol.
    """

    def generate_data(
        self,
        params: P,
        num_observations: int,
        *,
        key: PRNGKey | None = None,
    ) -> D:
        """Generate ``num_observations`` synthetic data points from ``params``.

        Parameters
        ----------
        params : P
            Model parameters.
        num_observations : int
            Number of data points to generate.
        key : PRNGKey or None
            JAX PRNG key for reproducible generation.
        """
        ...


__all__ = [
    "GenerativeLikelihood",
    "SupportsArrayBackend",
]
