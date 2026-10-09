"""The marginal and factor operations: detached parts of a structured or factored law.

``marginal(d, field)`` returns the detached marginal of a field or field
group, a standalone law with no reference back to ``d``, unlike the
correlation-preserving view ``d[field]``. ``factor(d, component_name)`` returns
the detached building-block factor of a joint that produces the named component.
"""

from __future__ import annotations

from typing import Any

from ..core._dispatch import Feasibility
from ..core._expression import Expression, embedded
from ..core._record_spec import RecordSpec
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsMarginals, _capability_guard
from ..distributions._conditional import ConditionalDistributionSpec
from ..distributions._distribution import (
    Distribution,
    DistributionSpec,
    _detached_term,
    _shared_final_names,
)
from ..distributions._factored import SupportsFactors
from ..distributions._views import _marginal_expression_at, _node_at
from ..functions._call import ApplicabilityError
from ._operation import BoundCall, _install_expression_rule, operation

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
        raise ApplicabilityError(
            f"marginal: {path!r} is not an event path of the law; its fields: "
            f"{list(d.event_spec.components)}"
        ) from None


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
            raise ApplicabilityError(f"marginal: {_shared_final_names(field)}")
        return OutputSpec(DistributionSpec(OutputSpec(RecordSpec(nodes))))
    if not isinstance(field, str):
        raise ApplicabilityError(
            f"marginal: field must be a path string or a tuple of path strings; got {field!r}"
        )
    component = field.rsplit(_PATH_SEP, 1)[-1]
    return OutputSpec(DistributionSpec(OutputSpec(**{component: _node(d, field)})))


def _marginal_expression(d: Any, field: Any) -> Expression:
    """The expression of the marginal of *d* at *field*: the factors it is, or *d* selected at *field*.

    A marginal over the whole events of some factors of a joint, none of which
    conditions on a component outside them, is the product of those factors,
    so ``marginal(location * scale, "tau")`` is ``scale`` and takes its label.
    Several such factors form a product without a label, joined in the order
    the paths name them where a product in that order declares the fields in
    the order of the paths, as ``marginal(model, ("b", "a"))`` is ``b·a``. The
    factors of a field view are its parent's. Any other marginal integrates a
    factor out, as the prior predictive does, so it is *d* selected at
    *field* and keeps *d*'s label. The view ``d[field]`` carries the same
    expression.
    """
    return _marginal_expression_at(d, field)


@operation(result=_marginal_result)
def marginal(d: Distribution, field: str):
    """The detached marginal of *d* at *field*, a standalone law with no reference back to *d*.

    A lift draws the marginal independently of *d* and of the laws *d* is
    built from (V.5).

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
        The marginal, which holds the paths *d* holds fixed. A marginal over
        the whole events of factors of a joint that condition on nothing
        outside them is those factors: one factor keeps its own label, and
        several form a product without a label, as
        ``marginal(model, ("a", "b"))`` displays as ``a(a)·b(b)``. Any other
        marginal keeps *d*'s label, as ``marginal(model, "y")`` displays as
        ``model(y)``. The view ``d[field]`` displays as the marginal does.

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


def _detached_marginal(call: BoundCall, result: OutputSpec | None) -> Distribution:
    """The law's exact marginal at the path, detached as a law's raw form is.

    A marginal can be a factor of a factored joint, and a factor can record a
    batch it was an element of or a law it renames, which a lift reads to draw
    it with that law. The detached marginal records neither, so a lift draws it
    independently of the joint (V.5). The result boundary then gives it the
    marginal's expression, which holds the paths the law holds fixed, and
    records the call's provenance on it.
    """
    return _detached_term(call.operands["d"]._marginal(call.operands["field"]))


def _can_sample(call: BoundCall, result: OutputSpec | None) -> Any:
    """The empirical marginal of projected draws is not implemented, so no call selects the route.

    The route is to apply when the law samples, so that projected draws form
    an empirical marginal; until its execution exists, ``check`` and the call
    both report it infeasible.
    """
    return Feasibility(False, "the empirical marginal of projected draws is not implemented yet")


def _empirical_marginal(call: BoundCall, result: OutputSpec | None) -> Any:
    """The empirical law of draws projected onto the field, declaring the node's event."""
    raise NotImplementedError("the empirical marginal of projected draws is not implemented yet")


marginal.capability_route(
    "exact",
    operand="d",
    protocol=SupportsMarginals,
    method="_marginal",
    check=_can_marginalize_path,
    execute=_detached_marginal,
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


def _factor_expression(d: Any, component_name: str) -> Expression:
    """The factor's own expression, which the detached factor keeps; the joint's without one."""
    part = _factor_of(d, component_name)
    return embedded(d if part is None else part)


@operation(
    result=_factor_result,
    conditions=(_names_a_component,),
    roles={"d": (DistributionSpec, ConditionalDistributionSpec)},
)
def factor(d: Distribution, component_name: str):
    """The complete factor of the joint *d* that produces the component *component_name*.

    The factor is detached from *d*, so a lift draws a factor that is a law
    independently of *d* and of the laws *d* is built from (V.5, VI.8).

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


def _detached_factor(call: BoundCall, result: OutputSpec | None) -> Any:
    """The factor whose event declaration has the named component, detached as a raw form is.

    A factor can record a batch it was an element of or a law it renames, which
    a lift reads to draw it with that law. The detached factor, a law or a
    kernel, records neither. The result boundary then records the call's
    provenance on it.
    """
    return _detached_term(_factor_of(call.operands["d"], call.operands["component_name"]))


factor.capability_route(
    "exact",
    operand="d",
    protocol=SupportsFactors,
    method="factors",
    check=_can_find_factor,
    execute=_detached_factor,
    exact=True,
)


_install_expression_rule(marginal, _marginal_expression)
_install_expression_rule(factor, _factor_expression)
