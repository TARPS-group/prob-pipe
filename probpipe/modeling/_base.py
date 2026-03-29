"""Base class for probabilistic models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..core.distribution import Distribution
from ..core.protocols import SupportsConditionableComponents

__all__ = ["ProbabilisticModel"]


class ProbabilisticModel(Distribution[Any], SupportsConditionableComponents):
    """Abstract base for probabilistic programming models.

    A ``ProbabilisticModel`` is a first-class :class:`Distribution`
    that also supports named components and conditioning.  Subclasses
    declare their parameter and data components, implement
    ``_condition_on``, and optionally provide ``_log_prob``,
    ``_sample``, etc.

    Inherits :class:`SupportsConditionableComponents`, which provides
    :class:`SupportsNamedComponents` and :class:`SupportsConditioning`
    for free.
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

    # -- SupportsConditionableComponents interface --------------------------

    @property
    @abstractmethod
    def conditionable_components(self) -> dict[str, bool]:
        """Map component name -> whether conditioning is required (True) or optional (False)."""
        ...

    @property
    def required_observations(self) -> tuple[str, ...]:
        """Component names that must be conditioned on."""
        return tuple(
            k for k, required in self.conditionable_components.items() if required
        )

    # -- SupportsConditioning interface -------------------------------------

    @abstractmethod
    def _condition_on(self, observed: Any, /, **kwargs: Any) -> Any:
        """Condition the model on observed data, returning a posterior distribution."""
        ...

    # -- Convenience --------------------------------------------------------

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the model's parameters (latent variables)."""
        ...
