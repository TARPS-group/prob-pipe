"""Base class for probabilistic models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..core.distribution import Distribution
from ..core.protocols import SupportsNamedComponents

__all__ = ["ProbabilisticModel"]


class ProbabilisticModel[T](Distribution[T], SupportsNamedComponents):
    """Abstract base for probabilistic programming models.

    A ``ProbabilisticModel`` is a first-class :class:`Distribution`
    that also supports named components.  Subclasses declare their
    parameter and data components and optionally provide ``_log_prob``,
    ``_sample``, etc.

    Conditioning is handled by the inference method registry — call
    ``condition_on(model, data)`` rather than ``model._condition_on()``.
    """

    # -- SupportsNamedComponents interface ----------------------------------

    @property
    @abstractmethod
    def component_names(self) -> tuple[str, ...]:
        """Names of all model components (parameters + data)."""
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Access a component by name."""
        ...

    # -- Convenience --------------------------------------------------------

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the model's parameters (latent variables)."""
        ...
