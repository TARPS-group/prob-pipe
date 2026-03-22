"""Converter protocol for distribution type conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._registry import ConversionInfo


class Converter(ABC):
    """Base class for distribution converters.

    Subclasses declare which types they can convert between via
    ``source_types()`` and ``target_types()``, provide a cheap
    ``check()`` probe, and implement ``convert()`` for the actual work.
    """

    @abstractmethod
    def source_types(self) -> tuple[type, ...]:
        """Types this converter can convert FROM."""
        ...

    @abstractmethod
    def target_types(self) -> tuple[type, ...]:
        """Types this converter can convert TO."""
        ...

    @abstractmethod
    def check(self, source: Any, target_type: type) -> ConversionInfo:
        """Inspect feasibility and cost without performing conversion.

        Must be cheap (no sampling, no heavy computation).
        """
        ...

    @abstractmethod
    def convert(self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any) -> Any:
        """Perform the actual conversion.

        Returns an instance of *target_type* (or a compatible subclass).
        """
        ...

    @property
    def priority(self) -> int:
        """Higher priority converters are tried first. Default 0."""
        return 0
