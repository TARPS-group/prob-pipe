"""The marginal and factor operations: detached parts of a structured or factored law.

``marginal(d, field)`` returns the detached marginal of a field or field
group, a standalone law with no reference back to ``d``, unlike the
correlation-preserving view ``d[field]``. ``factor(d, component_name)`` returns
the building-block factor of a joint that produces the named component.
"""

from __future__ import annotations

from typing import Any

from ..core._record_spec import RecordSpec
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsMarginals, SupportsSampling, _capability_guard
from ..distributions._conditional import ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import SupportsFactors
from ..distributions._views import _node_at
from ._operation import ApplicabilityError, BoundCall, operation

__all__ = ["factor", "marginal"]

_PATH_SEP = "/"


def _node(d: DistributionSpec, path: str) -> Any:
    """The term spec at *path* in the law's event declaration.

    Raises
    ------
    ApplicabilityError
        If *path* is not an event path of the law.
    """
    try:
        return _node_at(d.event_spec, path)
    except (KeyError, TypeError):
        raise ApplicabilityError(f"marginal: {path!r} is not an event path of the law") from None


def _marginal_result(d: DistributionSpec, field: Any) -> OutputSpec:
    """A law over the node at the path, a whole term under the path's final segment.

    A tuple of paths selects several nodes, returned as an exposed record of
    them.

    Raises
    ------
    ApplicabilityError
        If a path is not an event path, or two selected paths end in the same
        segment.
    """
    if isinstance(field, tuple):
        nodes = {path.rsplit(_PATH_SEP, 1)[-1]: _node(d, path) for path in field}
        if len(nodes) != len(field):
            raise ApplicabilityError(f"marginal: the paths {field!r} end in the same segment")
        return OutputSpec(marginal=DistributionSpec(OutputSpec(RecordSpec(nodes))))
    if not isinstance(field, str):
        raise ApplicabilityError(f"marginal: a field is a path or a tuple of paths; got {field!r}")
    component = field.rsplit(_PATH_SEP, 1)[-1]
    return OutputSpec(marginal=DistributionSpec(OutputSpec(**{component: _node(d, field)})))


@operation(result=_marginal_result)
def marginal(d: Distribution, field: str):
    """The detached marginal of *d* at *field*, a standalone law with no reference back to *d*.

    Parameters
    ----------
    d : Distribution
        The law.
    field : str or tuple of str
        An event path, whose interior node selects the group of fields under
        it, or a tuple of paths.

    Returns
    -------
    Distribution
        The marginal.

    Raises
    ------
    ApplicabilityError
        If *field* is not an event path of *d*.
    ResolutionError
        If *d* has no exact marginal at *field* and does not sample.
    """


def _can_marginalize_path(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law's marginal guard admits the requested path, where its class defines one."""
    return _capability_guard(call.operands["d"], "_marginal", call.operands["field"])


def _can_sample(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law samples, so projected draws form an empirical marginal."""
    d = call.operands["d"]
    return isinstance(d, SupportsSampling) and _capability_guard(d, "_sample")


def _empirical_marginal(call: BoundCall, result: OutputSpec | None) -> Any:
    """The empirical law of draws projected onto the field, declaring the node's event."""
    raise NotImplementedError("marginal.monte_carlo")


marginal.capability_route(
    "exact",
    operand="d",
    protocol=SupportsMarginals,
    method="_marginal",
    check=_can_marginalize_path,
    exact=True,
)
marginal.fallback_route("monte_carlo", check=_can_sample, execute=_empirical_marginal, exact=False)


def _names_a_component(d: Any, component_name: str) -> bool:
    """The name is an output component of the law."""
    return component_name in d.event_spec.components


def _factor_result(d: Any, component_name: str) -> OutputSpec:
    """A law, or a kernel for a dependent edge, whose declaration the returned factor carries."""
    return OutputSpec(factor=None)


@operation(
    result=_factor_result,
    conditions=(_names_a_component,),
    roles={"d": (DistributionSpec, ConditionalDistributionSpec)},
)
def factor(d: Distribution, component_name: str):
    """The complete factor of the joint *d* that produces the component *component_name*.

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        A factored joint.
    component_name : str
        One of the joint's output components.

    Returns
    -------
    Distribution or ConditionalDistribution
        The factor, a kernel when it conditions on another factor's output.

    Raises
    ------
    ApplicabilityError
        If *component_name* is not an output component of *d*.
    ResolutionError
        If *d* exposes no factors.
    """


def _can_find_factor(call: BoundCall, result: OutputSpec | None) -> Any:
    """One of the joint's factors produces the named component."""
    name = call.operands["component_name"]
    return any(name in part.event_spec.components for part in call.operands["d"].factors)


def _factor_producing(call: BoundCall, result: OutputSpec | None) -> Any:
    """The factor whose event declaration has the named component."""
    name = call.operands["component_name"]
    return next(part for part in call.operands["d"].factors if name in part.event_spec.components)


factor.capability_route(
    "exact",
    operand="d",
    protocol=SupportsFactors,
    method="factors",
    check=_can_find_factor,
    execute=_factor_producing,
    exact=True,
)
