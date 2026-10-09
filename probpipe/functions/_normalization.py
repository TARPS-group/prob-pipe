"""The conversion step of normalization: distribution arguments to the class their parameter names.

This private module handles only distribution-valued workflow inputs.
It is not a general normalization layer for all values entering a
``Function`` call.

A distribution argument whose parameter names another distribution class or a
capability protocol converts to it through the converter registry, which
returns a law that already satisfies the target as it is. A backend
distribution at a parameter that names no distribution enters ProbPipe as its
law, so the lift samples it as any law. A union annotation names what its arms
other than ``None`` name, so ``Normal | None`` converts as ``Normal`` does,
and a union of several distribution classes converts nothing.

Each parameter's entry of the ``conversions`` control,
``{"method": ..., "exact_only": ..., **options}``, selects the converter,
restricts it to exact ones, and passes the remaining entries to the converter
as its options.

Keeping those conversions here lets broadcast planning remain a pure
classification step over already-normalized values.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import UnionType
from typing import Any, Union, get_args, get_origin

from ..core._dispatch import ResolutionError
from ..core._repr import type_name
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from ..distributions._conversion import ConversionInfo, converter_registry
from ..distributions._distribution import Distribution, NumericDistribution
from ..values import _binding

DISTRIBUTION_HINT_PROTOCOLS: tuple[type, ...] = (
    SupportsExpectation,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsExactConditioning,
    SupportsApproximateConditioning,
    SupportsMarginals,
)


def is_distribution_hint(expected: Any) -> bool:
    """Return whether a type hint asks for a distribution object.

    A union asks for one when any of its arms does, so an optional
    distribution annotation consumes the distribution as the plain one does.
    """
    if get_origin(expected) in (Union, UnionType):
        return any(is_distribution_hint(arm) for arm in get_args(expected))
    origin = getattr(expected, "__origin__", None)
    expected_type = origin if isinstance(origin, type) else expected
    try:
        if (
            expected_type is not None
            and isinstance(expected_type, type)
            and issubclass(expected_type, Distribution)
        ):
            return True
    except TypeError:
        pass
    return expected in DISTRIBUTION_HINT_PROTOCOLS


def normalize_distribution_values(
    *,
    values: dict[str, Any],
    signature_info: _binding.FunctionSignatureInfo,
    conversions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert each distribution argument to the class or capability its parameter names.

    Non-distribution values are copied through unchanged. A distribution value
    converts to the target its parameter's annotation names, and a backend
    distribution at a parameter that names no distribution enters ProbPipe as
    its law, so the distribution-broadcast path samples it uniformly.

    Parameters
    ----------
    values : dict
        The bound arguments, keyed by parameter.
    signature_info : FunctionSignatureInfo
        The signature and the annotations of the function.
    conversions : Mapping, optional
        Each parameter's entry of the ``conversions`` control.

    Returns
    -------
    dict
        The arguments, the converted ones replaced by their converted laws.

    Raises
    ------
    ApplicabilityError
        If a distribution argument matches none of several named classes.
    ResolutionError
        If a conversion has no feasible converter under its entry's controls.
    TypeError
        If an entry's ``method`` is not a string or its ``exact_only`` is not a
        bool, or the selected converter refuses an option.
    """
    out = dict(values)
    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(out, ref)
        expected = _binding.input_ref_hint(signature_info, ref)
        target = _conversion_target(value, expected, label=ref.label)
        if target is None:
            continue
        entry = _entry(conversions, ref.parameter_name)
        out = _binding.replace_input_ref(
            out, ref, _convert_hinted_distribution(value, target, entry, label=ref.label)
        )
    return out


def plan_distribution_values(
    *,
    values: dict[str, Any],
    signature_info: _binding.FunctionSignatureInfo,
    conversions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, ConversionInfo], tuple[str, ...]]:
    """Plan the conversions :func:`normalize_distribution_values` executes, executing none.

    Parameters
    ----------
    values : dict
        The bound arguments, keyed by parameter.
    signature_info : FunctionSignatureInfo
        The signature and the annotations of the function.
    conversions : Mapping, optional
        Each parameter's entry of the ``conversions`` control.

    Returns
    -------
    tuple
        The values, unconverted; the converter registry's report of each planned
        conversion, by the argument's label, which omits an argument that
        satisfies its target as it is; and, for each backend object a
        conversion brings into ProbPipe, a sentence saying that the call's lift
        waits on the law the conversion constructs.

    Raises
    ------
    ApplicabilityError
        If a distribution argument matches none of several named classes.
    ResolutionError
        If a planned conversion is infeasible, as the call would raise.
    TypeError
        If an entry's ``method`` is not a string or its ``exact_only`` is not a
        bool.
    """
    reports: dict[str, ConversionInfo] = {}
    waiting: list[str] = []
    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(values, ref)
        expected = _binding.input_ref_hint(signature_info, ref)
        target = _conversion_target(value, expected, label=ref.label)
        if target is None:
            continue
        method, exact_only, options = _entry(conversions, ref.parameter_name)
        info = converter_registry.check(
            value, target, method=method, exact_only=exact_only, **options
        )
        if info.feasible is False:
            raise _conversion_failure(ref.label, target, info.description)
        if info.feasible is True and info.method_name is None:
            continue
        reports[ref.label] = info
        if not isinstance(value, Distribution):
            waiting.append(
                f"{ref.label!r}: the call lifts the {_name(target)} its conversion constructs"
            )
    return dict(values), reports, tuple(waiting)


def _name(target: type) -> str:
    return getattr(target, "__name__", repr(target))


def _conversion_failure(label: str, target: type, detail: Any) -> ResolutionError:
    """The error for an argument at *label* that no converter takes to *target*."""
    return ResolutionError(f"cannot convert parameter {label!r} to {_name(target)}: {detail}")


def _entry(
    conversions: Mapping[str, Mapping[str, Any]] | None, parameter: str
) -> tuple[str | None, bool, dict[str, Any]]:
    """The controls and the options of *parameter*'s ``conversions`` entry.

    Parameters
    ----------
    conversions : Mapping or None
        The ``conversions`` control, keyed by parameter name.
    parameter : str
        The parameter whose entry is read.

    Returns
    -------
    method : str or None
        The converter the entry names, or ``None``, which lets the registry
        select one.
    exact_only : bool
        Whether the conversion admits only exact converters, ``False`` by default.
    options : dict of str to Any
        The entry's other settings, which the converter receives as its options.

    Raises
    ------
    TypeError
        If ``method`` is not a string or ``None``, or ``exact_only`` is not a bool.
    """
    options = dict((conversions or {}).get(parameter, {}))
    method = options.pop("method", None)
    exact_only = options.pop("exact_only", False)
    if method is not None and not isinstance(method, str):
        raise TypeError(f"conversions[{parameter!r}]['method'] must be a string; got {method!r}")
    if type(exact_only) is not bool:
        raise TypeError(
            f"conversions[{parameter!r}]['exact_only'] must be a bool; got {exact_only!r}"
        )
    return method, exact_only, options


def _arms(expected: Any) -> tuple[Any, ...]:
    """The arms of a union annotation other than ``None``, or the annotation itself."""
    if get_origin(expected) in (Union, UnionType):
        return tuple(arm for arm in get_args(expected) if arm is not type(None))
    return (expected,)


def _hint_class(arm: Any) -> Any:
    """The class a parametrized annotation names, or the annotation itself."""
    origin = getattr(arm, "__origin__", None)
    return origin if isinstance(origin, type) else arm


def _convert_hinted_distribution(
    value: Any,
    target: type,
    entry: tuple[str | None, bool, dict[str, Any]],
    *,
    label: str,
) -> Any:
    """*value* converted to *target* by the converter registry, under the parameter's entry.

    The registry returns a law that already satisfies *target* as it is.

    Parameters
    ----------
    value : Any
        The distribution argument: a law or a backend distribution object.
    target : type
        The class or capability protocol the parameter names, as
        :func:`_conversion_target` returns it.
    entry : tuple of (str or None, bool, dict)
        The parameter's converter, ``exact_only`` setting, and options, as
        :func:`_entry` returns them.
    label : str
        The label of the argument's reference, which the message names.

    Returns
    -------
    Distribution
        The converted law, which replaces *value* among the arguments.

    Raises
    ------
    ResolutionError
        If no converter is feasible under the entry's controls, naming the
        parameter.
    """
    method, exact_only, options = entry
    try:
        return converter_registry.convert(
            value, target, method=method, exact_only=exact_only, **options
        )
    except ResolutionError as error:
        raise _conversion_failure(label, target, error) from error


def _conversion_target(value: Any, expected: Any, *, label: str) -> type | None:
    """The class or capability the argument converts to at its parameter, or ``None``.

    A distribution of a class *expected* names converts to nothing, and any
    other converts to the single class or capability named; the registry
    returns a law that already satisfies a capability as it is. A backend
    object converts to the class named, or enters ProbPipe as its law where the
    named class admits every numeric law. A backend object at a parameter that
    names no distribution enters ProbPipe as its law.

    Parameters
    ----------
    value : Any
        The argument, which converts only when the converter registry
        recognizes it as a distribution.
    expected : Any
        The annotation that governs lifting at the argument's parameter.
    label : str
        The label of the argument's reference, which the message names.

    Returns
    -------
    type or None
        ``None`` when the argument converts to nothing, and ``Distribution``
        when a backend object enters ProbPipe as its law.

    Raises
    ------
    ApplicabilityError
        If *value* is a distribution of none of several named classes or
        capabilities, which leaves no single conversion target.
    """
    if not converter_registry.is_distribution_type(value):
        return None
    arms = tuple(arm for arm in _arms(expected) if is_distribution_hint(arm))
    if not arms:
        return None if isinstance(value, Distribution) else Distribution
    if any(
        isinstance(value, _hint_class(arm)) for arm in arms if _is_concrete_distribution_hint(arm)
    ):
        return None
    if len(arms) > 1:
        if any(isinstance(value, arm) for arm in arms):
            return None
        from ._call import ApplicabilityError

        accepted = " | ".join(getattr(_hint_class(arm), "__name__", repr(arm)) for arm in arms)
        raise ApplicabilityError(
            f"parameter {label!r} accepts {accepted}, but got {type_name(value)}. A distribution "
            f"converts only to a single annotated class; pass one of {accepted}, or annotate "
            f"{label!r} with one class"
        )
    (arm,) = arms
    if not _is_concrete_distribution_hint(arm):
        return arm
    target = _hint_class(arm)
    if not isinstance(value, Distribution) and issubclass(NumericDistribution, target):
        # A backend object enters ProbPipe as the law the registry converts it
        # to, which is an instance of the class the parameter names.
        return Distribution
    return target


def _is_concrete_distribution_hint(expected: Any) -> bool:
    try:
        expected_class = _hint_class(expected)
        return isinstance(expected_class, type) and issubclass(expected_class, Distribution)
    except TypeError:
        return False
