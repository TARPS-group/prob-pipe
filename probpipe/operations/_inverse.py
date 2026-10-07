"""The inverse and log_det_jacobian operations: a map's inverse structure.

``inverse(f)`` returns the inverse map as a ``Function``, read through
``is_invertible``, and ``log_det_jacobian(f, x)`` returns the log-determinant
of the Jacobian of ``f`` at ``x`` through ``SupportsLogDetJacobian``. Each has a
capability route and no other.
"""

from __future__ import annotations

from typing import Any

from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec
from ..values import FunctionSpec
from ._operation import BoundCall, RouteSource, _CheckedRoute, operation

__all__ = ["inverse", "log_det_jacobian"]


def _inverse_result(f: Any) -> OutputSpec | None:
    """A map whose output declaration comes from *f*'s input slots.

    One slot is returned whole under that slot's name, and several form an
    exposed record.
    """
    return None


def _log_det_jacobian_result(f: Any, x: Any) -> OutputSpec:
    """A scalar under the component ``log_det_jacobian``."""
    return OutputSpec(log_det_jacobian=NumericArraySpec(()))


@operation(result=_inverse_result, roles={"f": (FunctionSpec,)})
def inverse(f: Any):
    """The inverse of the map *f*, itself invertible with *f* as its inverse.

    Parameters
    ----------
    f : Function
        A map that claims ``SupportsInverse`` and whose guard admits the
        inverse, as ``is_invertible`` reads them.

    Returns
    -------
    Function
        The map ``y ↦ f⁻¹(y)``, whose name derives from *f*'s.

    Raises
    ------
    ResolutionError
        If the inverse is unavailable.
    MathematicalDomainError
        If *f* is known to be noninvertible.
    """


@operation(result=_log_det_jacobian_result, roles={"f": (FunctionSpec,)})
def log_det_jacobian(f: Any, x: Any):
    """The log-determinant of the Jacobian of the map *f* at *x*.

    Parameters
    ----------
    f : Function
        The map, whose ``_log_det_jacobian`` computes the result.
    x : Any
        A point of the map's domain.

    Returns
    -------
    NumericArray
        The scalar ``log |det J_f(x)|``.

    Raises
    ------
    ResolutionError
        If *f* does not claim ``SupportsLogDetJacobian``.
    """


def _invertible(call: BoundCall, result: OutputSpec | None) -> Any:
    """The map claims SupportsInverse and its guard admits it, as is_invertible reads them."""
    raise NotImplementedError("inverse.exact")


def _jacobian(call: BoundCall, result: OutputSpec | None) -> Any:
    """The map claims SupportsLogDetJacobian and its guard admits the call."""
    raise NotImplementedError("log_det_jacobian.exact")


inverse.register_route(
    _CheckedRoute(
        "exact", source=RouteSource.CAPABILITY, check=_invertible, execute=_invertible, exact=True
    )
)
log_det_jacobian.register_route(
    _CheckedRoute(
        "exact", source=RouteSource.CAPABILITY, check=_jacobian, execute=_jacobian, exact=True
    )
)
