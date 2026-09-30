"""Resolve workflow controls, bind arguments, and admit them for the call engine.

Steps 2 and 3 of the call stack: the arguments bind to the wrapped function's
signature by Python's rules, and each bound argument is admitted against what
its parameter accepts. A violation of the call contract raises
:class:`ApplicabilityError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin

from ..distributions._capabilities import (
    SupportsConditionalCovariance,
    SupportsConditionalExpectation,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalQuantile,
    SupportsConditionalRandomLogProb,
    SupportsConditionalRandomUnnormalizedLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
)
from ..distributions._conditional import ConditionalDistribution
from ..values import _binding
from ..values._binding import WorkflowSignatureInfo, resolve_workflow_values
from . import _normalization

#: The conditional capabilities, each of which declares that a parameter consumes a kernel.
_KERNEL_HINT_PROTOCOLS: tuple[type, ...] = (
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalLogProb,
    SupportsConditionalRandomUnnormalizedLogProb,
    SupportsConditionalRandomLogProb,
    SupportsConditionalMean,
    SupportsConditionalVariance,
    SupportsConditionalCovariance,
    SupportsConditionalQuantile,
    SupportsConditionalExpectation,
    SupportsConditionalMarginals,
)


class ApplicabilityError(TypeError):
    """The arguments or declarations of a call violate its call contract.

    Raised when an argument's kind is not one its parameter accepts, when swept
    levels do not align, and when the declarations of a call conflict. The
    message names the parameter, what it accepts, and what arrived.
    """


@dataclass(frozen=True)
class WorkflowCallOptions:
    """Optional call-time workflow controls outside user kwargs."""

    n_broadcast_samples: int | None = None
    include_inputs: bool | None = None


@dataclass(frozen=True)
class WorkflowCallOverrides:
    """Resolved call-time workflow settings consumed by ``Function``."""

    n_broadcast_samples: int
    include_inputs: bool


@dataclass(frozen=True)
class ResolvedWorkflowCall:
    """Fully resolved signature-shaped values plus workflow overrides."""

    values: dict[str, Any]
    overrides: WorkflowCallOverrides


def bind_call_inputs(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> tuple[dict[str, Any], WorkflowCallOverrides]:
    """Bind user inputs and resolve workflow controls.

    Call inputs bind exactly like the wrapped Python function. Workflow
    controls come only from explicit ``options`` or construction defaults.
    """
    explicit_options = options if options is not None else WorkflowCallOptions()

    def resolve_option(name: str, default: Any = None) -> Any:
        explicit_value = getattr(explicit_options, name)
        if explicit_value is not None:
            return explicit_value

        return default

    overrides = WorkflowCallOverrides(
        n_broadcast_samples=resolve_option(
            "n_broadcast_samples",
            default_n_broadcast_samples,
        ),
        include_inputs=resolve_option(
            "include_inputs",
            default_include_inputs,
        ),
    )

    bound = info.signature.bind_partial(*args, **call_inputs)
    return dict(bound.arguments), overrides


def resolve_workflow_call(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    workflow_name: str,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> ResolvedWorkflowCall:
    """Resolve one ``Function`` call into values plus overrides."""
    bound_inputs, overrides = bind_call_inputs(
        info,
        args,
        call_inputs,
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
        options=options,
    )
    values = resolve_workflow_values(
        info,
        bound_inputs,
        bind=bind,
        module=module,
        dependency_type=dependency_type,
        workflow_name=workflow_name,
    )
    return ResolvedWorkflowCall(values=values, overrides=overrides)


def _consumes_kernel(expected: Any) -> bool:
    """Whether an annotation declares that its parameter consumes a kernel itself."""
    if get_origin(expected) in (Union, UnionType):
        return any(_consumes_kernel(arm) for arm in get_args(expected))
    try:
        if isinstance(expected, type) and issubclass(expected, ConditionalDistribution):
            return True
    except TypeError:
        pass
    return expected in _KERNEL_HINT_PROTOCOLS


def _expects_value(expected: Any) -> bool:
    """Whether a parameter with annotation *expected* expects a value.

    A parameter expects a value when it is unannotated or annotated with a
    value type. An annotation that names a distribution, a kernel, or one of
    their capabilities declares that the parameter consumes that object, and
    ``Any`` admits every kind.
    """
    if expected is Any:
        return False
    return not (_normalization.is_distribution_hint(expected) or _consumes_kernel(expected))


def admit_arguments(info: WorkflowSignatureInfo, values: Mapping[str, Any]) -> None:
    """Admit each bound argument against what its parameter accepts.

    Each argument is admitted against its parameter's own annotation, so an
    argument that a variadic parameter annotated ``Any`` collects is admitted
    whatever its kind. A conditional distribution at a parameter that expects a
    value is refused, since a kernel has no marginal law to lift over.

    Parameters
    ----------
    info : WorkflowSignatureInfo
        The wrapped function's signature and resolved annotations.
    values : Mapping of str to Any
        The bound arguments, shaped by the signature.

    Raises
    ------
    ApplicabilityError
        If an argument's kind is not one its parameter accepts.
    """
    for ref in _binding.iter_input_refs(info, values):
        value = _binding.input_ref_value(values, ref)
        if isinstance(value, ConditionalDistribution) and _expects_value(
            info.hints.get(ref.parameter_name)
        ):
            raise ApplicabilityError(
                f"parameter {ref.label!r} expects a value, and a value parameter accepts no "
                f"ConditionalDistribution: got {type(value).__name__} {value.name!r}, which is "
                f"a kernel with no marginal law to lift over. Condition it on a given value "
                f"first, or annotate the parameter ConditionalDistribution to consume the "
                f"kernel itself"
            )
