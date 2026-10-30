"""The marginal and factor operations: detached parts of a structured or factored law.

``marginal(d, paths)`` returns the standalone marginal at a path or a list of
paths, a standalone law with no reference back to ``d``, unlike the
correlation-preserving view ``d[paths]``. ``factor(d, output_name)`` returns
the detached building-block factor of a joint that produces the named output.
"""

from __future__ import annotations

from typing import Any, cast

from ..core._batch import BatchSpec, _flat_batch_spec
from ..core._expression import Expression
from ..core._record_batch import _batch_over
from ..core._shapes import PathsLike
from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsMarginals, _capability_guard
from ..distributions._conditional import ConditionalDistributionSpec
from ..distributions._distribution import (
    Distribution,
    DistributionSpec,
    _detached_term,
    _read_event_paths,
)
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import SupportsFactors, _raw_record
from ..distributions._views import (
    _drawn_form,
    _marginal_expression_at,
    _projector,
    _view_declaration,
)
from ..functions._call import ApplicabilityError
from ..functions._result import SAMPLE_LEVEL
from ._operation import (
    BoundCall,
    _can_sample,
    _install_expression_rule,
    _monte_carlo_draws,
    operation,
)

__all__ = ["factor", "marginal"]


def _read_marginal_paths(event_spec: OutputSpec, label: str, paths: Any) -> str | tuple[str, ...]:
    """The *paths* of a call of ``marginal``, read: one event path, or a list of them in the law's order.

    Parameters
    ----------
    event_spec : OutputSpec
        The law's event spec.
    label : str
        The law's label, which the error messages name.
    paths : Any
        The paths as the caller gave them.

    Returns
    -------
    str or tuple of str
        The one path, or the list of paths in the canonical order of *event_spec*.

    Raises
    ------
    ApplicabilityError
        If the law is over a record and a path is not one of its paths.
    TypeError
        If *paths* is neither a string nor a list or tuple of strings, or the
        law is over a value with no paths, such as an array.
    ValueError
        If a path is empty or has an empty name, or a list of paths is empty,
        names a path twice, or holds a path inside another.
    """
    try:
        return _read_event_paths(
            event_spec, label, paths, action=f"take the marginal of {label!r} at"
        )
    except KeyError as error:
        raise ApplicabilityError(f"marginal: {error.args[0]}") from None


def _marginal_result(d: DistributionSpec, paths: Any) -> OutputSpec:
    """A law over the node at the path, or over the nodes at the list of paths.

    One path gives the node whole under the path's last name, and a list of
    paths a law over the record that keeps each node at its own path, in the
    law's order.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event spec holds the selected nodes.
    paths : str, or list or tuple of str
        An event path, or a list of paths whose nodes the marginal draws
        jointly.

    Returns
    -------
    OutputSpec
        The output spec of the marginal law, in the positional form over a
        ``DistributionSpec`` of the selected nodes.

    Raises
    ------
    ApplicabilityError
        If a path is not an event path.
    TypeError
        If *paths* is neither a string nor a list or tuple of strings.
    ValueError
        If a path is malformed, or a list of paths is empty, names a path
        twice, or holds a path inside another.
    """
    read = _read_marginal_paths(d.event_spec, "the law", paths)
    return OutputSpec(DistributionSpec(_view_declaration(d.event_spec, read)))


def _marginal_expression(d: Any, paths: Any) -> Expression:
    """The expression of the marginal of *d* at *paths*: the factors it is, or *d* selected at *paths*.

    A marginal over the whole events of some factors of a joint, none of which
    conditions on an output outside them, is the product of those factors,
    so ``marginal(location * scale, "tau")`` is ``scale`` and takes its label.
    Several such factors form a product without a label, in the joint's
    order, as ``marginal(model, ["b", "a"])`` is ``a·b``. The factors of a
    field view are its parent's. Any other marginal integrates a factor out,
    as the prior predictive does, so it is *d* selected at *paths* and keeps
    *d*'s label. The view ``d[paths]`` carries the same expression.
    """
    return _marginal_expression_at(d, _read_marginal_paths(d.event_spec, d.label, paths))


@operation(result=_marginal_result)
def marginal(d: Distribution, paths: PathsLike):
    """The standalone marginal of *d* at *paths*, a standalone law with no reference back to *d*.

    ``marginal(d, paths)`` is the law of the view ``d[paths]``, detached: one
    path gives the node there whole under the path's last name, and a list or
    a tuple of paths a law over the record that keeps each node at its own
    path, in *d*'s order, so ``marginal(grouped, ["a", "g/c"])`` draws records
    ``{a, g: {c}}``. A lift draws the marginal independently of *d* and of
    the laws *d* is built from (V.5).

    The law's exact marginal capability is preferred. Otherwise, the ``monte_carlo`` route
    draws ``num_samples`` joint samples once and returns their projected
    empirical law. Its atoms have uniform weights and remain fixed when the
    result is sampled again. Set the count with
    ``marginal.using(num_samples=4096)(d, paths)``. The call records its
    approximation and sample count in provenance.

    With ``given=``, the kernel's exact conditional marginal is tried first,
    followed by exact conditioning and the conditioned law's exact marginal.
    Monte Carlo through the conditional sampler follows when neither exact
    route applies. A check does not evaluate the kernel, so it reports the
    conditional route unresolved when that choice needs the bound law.
    An explicit ``method="monte_carlo"`` selects sampling directly.

    The atoms' leading level is ``sample``. For a batch-valued event,
    underscores are appended until that name differs from the event's levels.

    Parameters
    ----------
    d : Distribution
        The law.
    paths : str, or list or tuple of str
        An event path, whose group selects the fields under it, or a list of
        event paths.

    Returns
    -------
    Distribution
        The marginal, which holds the paths *d* holds fixed. A marginal over
        the whole events of factors of a joint that condition on nothing
        outside them is those factors: one factor keeps its own label, and
        several form a product without a label in the joint's order, as
        ``marginal(model, ["a", "b"])`` displays as ``a(a)·b(b)``. Any other
        marginal keeps *d*'s label, as ``marginal(model, "y")`` displays as
        ``model(y)``. The view ``d[paths]`` displays as the marginal does.

    Raises
    ------
    ApplicabilityError
        If *d* is over a record and a path is not one of its paths.
    TypeError
        If *paths* is neither a string nor a list or tuple of strings, or *d*
        is over a value with no paths, such as an array.
    ValueError
        If a path is empty or has an empty name, or a list of paths is empty,
        names a path twice, or holds a path inside another.
    ResolutionError
        If *d* has no exact marginal at *paths* and does not sample.
    """


def _call_paths(call: BoundCall) -> str | tuple[str, ...]:
    """The call's paths, read against its law."""
    law = call.operands["d"]
    return _read_marginal_paths(law.event_spec, law.label, call.operands["paths"])


def _can_marginalize_path(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law's marginal guard admits the requested paths, where its class defines one."""
    return _capability_guard(call.operands["d"], "_marginal", _call_paths(call))


def _detached_marginal(call: BoundCall, result: OutputSpec | None) -> Distribution:
    """The law's exact marginal at the paths, detached as a law's raw form is.

    A marginal can be a factor of a factored joint, and a factor can record a
    batch it was an element of or a law it renames, which a lift reads to draw
    it with that law. The standalone marginal records neither, so a lift draws it
    independently of the joint (V.5). The result boundary then gives it the
    marginal's expression, which holds the paths the law holds fixed, and
    records the call's provenance on it.
    """
    return _detached_term(call.operands["d"]._marginal(_call_paths(call)))


def _empirical_marginal(call: BoundCall, result: OutputSpec | None) -> EmpiricalDistribution:
    """The empirical law of draws projected onto the field, declaring the node's event."""
    law = call.operands["d"]
    count = call.controls["num_samples"]
    draws = _monte_carlo_draws(call, "marginal")
    event_spec = cast(DistributionSpec, cast(OutputSpec, result).term_spec).event_spec
    event_term_spec = cast(TermSpec, event_spec.term_spec)
    projected = _projector(law.event_spec, _call_paths(call))(draws)
    raw_atoms = _drawn_form(_raw_record(projected), event_term_spec, (count,))
    level = SAMPLE_LEVEL
    if isinstance(event_term_spec, BatchSpec):
        while level in _flat_batch_spec(event_term_spec).level_names:
            level += "_"
    atoms = _batch_over(raw_atoms, BatchSpec(event_spec, {level: count}), law.label)
    return EmpiricalDistribution(atoms, event_spec=event_spec, label=law.label)


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


def _names_an_output(d: Any, output_name: str) -> bool:
    """The name is an output of the law."""
    return output_name in d.event_spec.outputs


def _factor_result(d: Any, output_name: str) -> None:
    """None: the returned factor, a law or a kernel for a dependent edge, carries its own spec."""
    return None


def _factor_of(d: Any, output_name: str) -> Any:
    """The factor of *d* whose event spec has the output, or None."""
    return next(
        (part for part in getattr(d, "factors", ()) if output_name in part.event_spec.outputs),
        None,
    )


def _factor_expression(d: Any, output_name: str) -> Expression:
    """The factor's own expression, which the detached factor keeps; the joint's without one."""
    part = _factor_of(d, output_name)
    return (d if part is None else part)._embedded_expression()


@operation(
    result=_factor_result,
    applicability_checks=(_names_an_output,),
    roles={"d": (DistributionSpec, ConditionalDistributionSpec)},
)
def factor(d: Distribution, output_name: str):
    """The complete factor of the joint *d* that produces the output *output_name*.

    The factor is detached from *d*, so a lift draws a factor that is a law
    independently of *d* and of the laws *d* is built from (V.5, VI.8).

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        A factored joint.
    output_name : str
        One of the joint's outputs.

    Returns
    -------
    Distribution or ConditionalDistribution
        The factor under its own label, a kernel when it conditions on another
        factor's output.

    Raises
    ------
    ApplicabilityError
        If *output_name* is not an output of *d*.
    ResolutionError
        If *d* exposes no factors.
    """


def _can_find_factor(call: BoundCall, result: OutputSpec | None) -> Any:
    """One of the joint's factors produces the named output."""
    return _factor_of(call.operands["d"], call.operands["output_name"]) is not None


def _detached_factor(call: BoundCall, result: OutputSpec | None) -> Any:
    """The factor whose event spec has the named output, detached as a raw form is.

    A factor can record a batch it was an element of or a law it renames, which
    a lift reads to draw it with that law. The detached factor, a law or a
    kernel, records neither. The result boundary then records the call's
    provenance on it.
    """
    return _detached_term(_factor_of(call.operands["d"], call.operands["output_name"]))


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
