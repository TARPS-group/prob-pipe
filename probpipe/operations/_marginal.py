"""The marginal and factor operations: detached parts of a structured or factored law.

``marginal(d, field)`` returns the detached marginal of a field or field
group, a standalone law with no reference back to ``d``, unlike the
correlation-preserving view ``d[field]``. ``factor(d, component_name)`` returns
the building-block factor of a joint that produces the named component.
"""

from __future__ import annotations

from typing import Any

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsMarginals, _capability_guard
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import SupportsFactors, _joined_label
from ..distributions._views import _node_at
from ..functions._call import ApplicabilityError
from ._operation import BoundCall, operation

__all__ = ["factor", "marginal"]

_PATH_SEP = "/"


def _node(d: DistributionSpec, path: str) -> Any:
    """The term spec at *path* in the law's event declaration.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration holds the node.
    path : str
        A path that starts with a component and may name an interior node.

    Returns
    -------
    TermSpec
        A leaf's spec, or the record spec of the fields under an interior node.

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
    """A law over the node at the path, which exposes its event: a whole term under the path's final segment.

    A tuple of paths selects several nodes, the law's event an exposed record
    of them.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration holds the selected nodes.
    field : str or tuple of str
        An event path, or a tuple of paths whose nodes the marginal draws
        jointly.

    Returns
    -------
    OutputSpec
        The declaration of the marginal law as one whole term, whose spec is a
        ``DistributionSpec`` over the selected nodes.

    Raises
    ------
    ApplicabilityError
        If *field* is neither a str nor a tuple, a path is not an event path, or
        two selected paths end in the same segment.
    """
    if isinstance(field, tuple):
        nodes = {path.rsplit(_PATH_SEP, 1)[-1]: _node(d, path) for path in field}
        if len(nodes) != len(field):
            raise ApplicabilityError(f"marginal: the paths {field!r} end in the same segment")
        return OutputSpec(DistributionSpec(OutputSpec(RecordSpec(nodes))))
    if not isinstance(field, str):
        raise ApplicabilityError(f"marginal: a field is a path or a tuple of paths; got {field!r}")
    component = field.rsplit(_PATH_SEP, 1)[-1]
    return OutputSpec(DistributionSpec(OutputSpec(**{component: _node(d, field)})))


def _marginal_label(d: Any, field: Any) -> str:
    """The joined labels of the factors the marginal is, and *d*'s label for any other marginal.

    A marginal over the whole events of some factors of a joint, none of which
    conditions on a component outside them, is the product of those factors,
    so ``marginal(location * scale, "tau")`` is ``scale`` and takes its label.
    Any other marginal integrates a factor out, as the prior predictive does,
    and keeps the joint's label.
    """
    paths = field if isinstance(field, tuple) else (field,)
    parts = _closed_factors(d, paths)
    return d.label if parts is None else _joined_label(part.label for part in parts)


def _closed_factors(d: Any, components: tuple[Any, ...]) -> list[Any] | None:
    """The factors of *d* whose events are *components* together, if none conditions outside them.

    Returns None when *d* has no factors, a path is not a whole component, the
    components split a factor's event, or a factor conditions on a component
    outside them.
    """
    parts = getattr(d, "factors", None)
    wanted = set(components)
    if not parts or not all(isinstance(path, str) and _PATH_SEP not in path for path in wanted):
        return None
    selected = [part for part in parts if wanted & set(part.event_spec.components)]
    produced = {component for part in selected for component in part.event_spec.components}
    if produced != wanted:
        return None
    components_of_d = set(d.event_spec.components)
    for part in selected:
        if not isinstance(part, ConditionalDistribution):
            continue
        # An optional slot that no factor of d produces takes its default, so it
        # conditions on nothing.
        given = part.given_spec
        conditioned = [
            slot for slot in given if slot in components_of_d or slot not in given.optional
        ]
        if any(slot not in wanted for slot in conditioned):
            return None
    return selected


@operation(result=_marginal_result, label=_marginal_label)
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
    """The empirical marginal of projected draws is not implemented, so no call selects the route.

    The route is to apply when the law samples, so that projected draws form
    an empirical marginal; until its execution exists, ``check`` and the call
    both report it infeasible.
    """
    return Feasibility(False, "the empirical marginal of projected draws is not implemented")


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


def _factor_result(d: Any, component_name: str) -> None:
    """None: the returned factor, a law or a kernel for a dependent edge, carries its declaration."""
    return None


def _factor_of(d: Any, component_name: str) -> Any:
    """The factor of *d* whose event declaration has the component, or None."""
    return next(
        (
            part
            for part in getattr(d, "factors", ())
            if component_name in part.event_spec.components
        ),
        None,
    )


def _factor_label(d: Any, component_name: str) -> str:
    """The factor's own label, since the result is the factor itself; the joint's without one."""
    part = _factor_of(d, component_name)
    return d.label if part is None else part.label


@operation(
    result=_factor_result,
    conditions=(_names_a_component,),
    roles={"d": (DistributionSpec, ConditionalDistributionSpec)},
    label=_factor_label,
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
        The factor under its own label, a kernel when it conditions on another
        factor's output.

    Raises
    ------
    ApplicabilityError
        If *component_name* is not an output component of *d*.
    ResolutionError
        If *d* exposes no factors.
    """


def _can_find_factor(call: BoundCall, result: OutputSpec | None) -> Any:
    """One of the joint's factors produces the named component."""
    return _factor_of(call.operands["d"], call.operands["component_name"]) is not None


def _factor_producing(call: BoundCall, result: OutputSpec | None) -> Any:
    """The factor whose event declaration has the named component."""
    return _factor_of(call.operands["d"], call.operands["component_name"])


factor.capability_route(
    "exact",
    operand="d",
    protocol=SupportsFactors,
    method="factors",
    check=_can_find_factor,
    execute=_factor_producing,
    exact=True,
)
