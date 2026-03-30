"""TransportMap base class for deterministic maps between spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.distribution import Distribution
    from ..custom_types import PRNGKey

T = TypeVar("T")
S = TypeVar("S")


class TransportMap(Generic[T, S], ABC):
    """Deterministic map f: T -> S.

    Point-level evaluation via :meth:`forward` (or ``__call__``).
    Distribution-level via :meth:`pushforward`, which delegates to the
    :class:`PushforwardRegistry` for dispatch.

    Parameters
    ----------
    T : type
        Input space type.
    S : type
        Output space type.

    Examples
    --------
    >>> class SquareMap(TransportMap):
    ...     def forward(self, x):
    ...         return x ** 2
    >>> f = SquareMap()
    >>> f(3.0)
    9.0
    >>> f.pushforward(Normal(0, 1))  # EmpiricalDistribution via sampling
    """

    @abstractmethod
    def forward(self, value: T) -> S:
        """Apply the map to a single point.

        Parameters
        ----------
        value : T
            Input value.

        Returns
        -------
        S
            Transformed value.
        """
        ...

    def pushforward(
        self,
        dist: Distribution[T],
        *,
        strategy: str | None = None,
        key: PRNGKey | None = None,
        num_samples: int | None = None,
        return_joint: bool = False,
    ) -> Distribution[S]:
        """Push a distribution through this map.

        Delegates to :data:`pushforward_registry` for dispatch.  The
        registry tries rules in priority order:

        1. Closed-form rule matching ``(type(self), type(dist))``
        2. Change-of-variables (if ``self`` is a :class:`Bijector`)
        3. Sampling fallback (draws samples, applies ``forward``,
           returns :class:`EmpiricalDistribution`)

        Parameters
        ----------
        dist : Distribution[T]
            Input distribution.
        strategy : str, optional
            Force a specific dispatch strategy:
            ``"closed_form"``, ``"change_of_variables"``, or
            ``"sampling"``.  ``None`` (default) tries all in priority
            order.
        key : PRNGKey, optional
            JAX PRNG key for sampling fallback.
        num_samples : int, optional
            Number of samples for sampling fallback.
        return_joint : bool
            If ``True``, return a :class:`BroadcastDistribution` that
            stores both the exact output marginal (when available) and
            paired input–output samples as a joint representation.

        Returns
        -------
        Distribution[S]
            The pushforward distribution.

        Raises
        ------
        ValueError
            If *strategy* is ``"closed_form"`` and no closed-form rule
            exists for this ``(map, dist)`` pair.
        TypeError
            If *strategy* is ``"change_of_variables"`` and this map is
            not a :class:`Bijector`.
        """
        from .registry import pushforward_registry

        return pushforward_registry.apply(
            self,
            dist,
            strategy=strategy,
            key=key,
            num_samples=num_samples,
            return_joint=return_joint,
        )

    def __call__(self, value: T) -> S:
        """Point-level application.  Sugar for ``self.forward(value)``."""
        return self.forward(value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
