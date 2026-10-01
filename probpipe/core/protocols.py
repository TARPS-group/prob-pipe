"""Protocols that are not distribution capabilities.

The capability protocols of the distribution kinds are defined in
:mod:`probpipe.distributions._capabilities`. This module holds the rest:

- ``SupportsArrayBackend``, which a distribution class implements to store
  its laws at batched parameters in one fused backend.
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
# Array backend: the fused storage of laws of one class at batched parameters
# ---------------------------------------------------------------------------


@runtime_checkable
class _DistributionArrayBackend(Protocol):
    """The fused storage of laws of one distribution class at batched parameters.

    A backend owns the batched parameters of the laws and computes their
    capabilities vectorized, through the backend's native batch axis, without
    one ``Distribution`` per position. It carries no ``name`` or
    ``provenance``, and it is the contract between a class's
    :meth:`SupportsArrayBackend._make_array_backend` and the code that stores
    the laws. Backends are private to the library.

    Every backend exposes ``batch_shape``, ``event_shape``, ``cell_spec``,
    ``cell``, and whichever of ``_sample``, ``_log_prob``, ``_mean``,
    ``_variance``, and ``_cov`` the class's laws support. ``cell(index)`` builds
    the law at ``index``, a ``Distribution`` of the class at that position's
    parameters.
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
    """A distribution class that stores its laws at batched parameters in one backend.

    Implementations construct an internal :class:`_DistributionArrayBackend`
    that owns the batched parameters and computes the laws' capabilities
    vectorized. A class that does not implement the protocol stores one
    ``Distribution`` per position instead.

    The protocol attaches to the **class**, not to its instances: the runtime
    check is ``isinstance(MyDistribution, SupportsArrayBackend)``, which holds
    when the class implements ``_make_array_backend``. An instance passes the
    check too, since it inherits the class's attributes, but the contract is
    the class's. The protocol is internal to the library, and user code never
    calls ``_make_array_backend``.

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
