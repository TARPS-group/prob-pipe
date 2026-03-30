"""Bijector classes: invertible transport maps with log-det-jacobian.

Provides both a pure-probpipe abstract :class:`Bijector` and a
:class:`TFPBijector` wrapper for TensorFlow Probability bijectors.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar, TYPE_CHECKING

import tensorflow_probability.substrates.jax.bijectors as tfb

from .transport_map import TransportMap

if TYPE_CHECKING:
    from ..core.distribution import Constraint
    from ..custom_types import Array

T = TypeVar("T")

# Map TFP bijector class names -> output Constraint they produce when
# applied to a base distribution with ``real`` support.
_BIJECTOR_SUPPORT_MAP: dict[str, str] = {
    "Exp": "positive",
    "Softplus": "positive",
    "Sigmoid": "unit_interval",
    "Square": "positive",
}


def _resolve_constraint(name: str):
    """Lazily resolve a constraint name to the singleton object."""
    from ..core.distribution import (
        positive,
        unit_interval,
        real,
    )
    return {"positive": positive, "unit_interval": unit_interval, "real": real}.get(
        name
    )


class Bijector(TransportMap[T, T]):
    """Invertible transport map with log-determinant-jacobian.

    Subclasses must implement :meth:`forward`, :meth:`inverse`, and
    :meth:`forward_log_det_jacobian`.  The base class provides a
    default :meth:`inverse_log_det_jacobian` derived from the other
    two.

    Type parameter ``T`` is preserved: ``Bijector[T]`` maps ``T -> T``.
    For array-valued bijectors ``T = Array``.
    """

    @abstractmethod
    def forward(self, value: T) -> T:
        ...

    @abstractmethod
    def inverse(self, value: T) -> T:
        ...

    @abstractmethod
    def forward_log_det_jacobian(self, value: T, event_ndims: int = 0) -> Array:
        """log |det J_f(value)|.

        Parameters
        ----------
        value : T
            Point at which to evaluate.
        event_ndims : int
            Number of rightmost dimensions that are part of the event.
            For scalar bijectors applied element-wise, use 0.
        """
        ...

    def inverse_log_det_jacobian(self, value: T, event_ndims: int = 0) -> Array:
        """log |det J_{f^{-1}}(value)|.

        Default implementation: ``-forward_log_det_jacobian(inverse(value))``.
        Override for efficiency when a direct formula is available.
        """
        return -self.forward_log_det_jacobian(self.inverse(value), event_ndims)

    @property
    def output_constraint(self) -> Constraint | None:
        """Constraint describing the image of the map (e.g. positive).

        Returns ``None`` if unknown.  Used by
        :class:`BijectorTransformedDistribution` to set the ``support``
        property of the resulting distribution.
        """
        return None


class TFPBijector(Bijector):
    """Wraps a ``tfb.Bijector`` as a probpipe :class:`Bijector`.

    Mirrors the :class:`TFPDistribution` pattern: stores the TFP
    object and delegates ``forward`` / ``inverse`` /
    ``log_det_jacobian`` to it.

    Parameters
    ----------
    tfp_bijector : tfb.Bijector
        The TensorFlow Probability bijector to wrap.

    Examples
    --------
    >>> bij = TFPBijector(tfb.Exp())
    >>> bij(0.0)
    Array(1., dtype=float32)
    >>> bij.inverse(1.0)
    Array(0., dtype=float32)
    """

    def __init__(self, tfp_bijector: tfb.Bijector) -> None:
        self._tfp_bijector = tfp_bijector

    # -- delegation ---------------------------------------------------------

    def forward(self, value):
        return self._tfp_bijector.forward(value)

    def inverse(self, value):
        return self._tfp_bijector.inverse(value)

    def forward_log_det_jacobian(self, value, event_ndims: int = 0):
        return self._tfp_bijector.forward_log_det_jacobian(
            value, event_ndims=event_ndims
        )

    def inverse_log_det_jacobian(self, value, event_ndims: int = 0):
        return self._tfp_bijector.inverse_log_det_jacobian(
            value, event_ndims=event_ndims
        )

    # -- constraint ---------------------------------------------------------

    @property
    def output_constraint(self):
        """Derive constraint from the TFP bijector class name."""
        name = type(self._tfp_bijector).__name__
        constraint_name = _BIJECTOR_SUPPORT_MAP.get(name)
        if constraint_name is not None:
            return _resolve_constraint(constraint_name)
        # For Chain bijectors, check outermost (bijectors[0] is applied last)
        if name == "Chain" and hasattr(self._tfp_bijector, "bijectors"):
            outermost = type(self._tfp_bijector.bijectors[0]).__name__
            constraint_name = _BIJECTOR_SUPPORT_MAP.get(outermost)
            if constraint_name is not None:
                return _resolve_constraint(constraint_name)
        return None

    # -- accessors ----------------------------------------------------------

    @property
    def tfp_bijector(self) -> tfb.Bijector:
        """The wrapped TFP bijector object."""
        return self._tfp_bijector

    def __repr__(self) -> str:
        return f"TFPBijector({type(self._tfp_bijector).__name__})"
