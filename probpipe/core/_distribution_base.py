"""Generic distribution base class and minimal helpers.

Provides:
  - ``Distribution[T]`` – Abstract base for all ProbPipe distributions.
  - Global defaults for expectation sampling.
  - ``_auto_key()`` helper for convenience PRNG key generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

    from .record import Record

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

class Distribution[T](ABC):
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

    # -- approximation tracking ---------------------------------------------

    @property
    def is_approximate(self) -> bool:
        """Whether this distribution is an approximation (e.g., from sampling or MCMC)."""
        return getattr(self, "_approximate", False)

    # -- validation results ---------------------------------------------------

    @property
    def validation_results(self) -> list[dict]:
        """Results from :func:`~probpipe.validation.predictive_check` runs.

        Each entry is a dict with at least ``"replicated_statistics"``
        and ``"test_fn_name"``.  Posterior checks also include
        ``"observed_statistic"`` and ``"p_value"``.
        """
        if not hasattr(self, "_validation_results"):
            self._validation_results: list[dict] = []
        return self._validation_results

    # -- values template ----------------------------------------------------

    @property
    def record_template(self) -> Record | None:
        """A :class:`~probpipe.core.record.Record` describing the named
        structure of samples from this distribution, or ``None``.

        Set automatically from the ``name`` kwarg for named distributions
        and from component structure for joint distributions.
        """
        return getattr(self, "_record_template", None)

    # -- auxiliary information ----------------------------------------------

    @property
    def auxiliary(self) -> DataTree | None:
        """An xarray ``DataTree`` of auxiliary information (diagnostics,
        sample statistics, algorithm metadata), or ``None``.

        Populated by inference methods.  Follows ArviZ group conventions
        (``posterior``, ``sample_stats``, ``warmup``, etc.) with metadata
        stored as DataTree attributes.
        """
        return getattr(self, "_auxiliary", None)

    # -- naming & provenance ------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Provenance | None:
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

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"
