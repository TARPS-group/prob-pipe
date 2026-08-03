"""Private automatic-key broker for workflow-owned stochastic effects."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, Protocol

from ..custom_types import PRNGKey
from . import _workflow_context

_OccurrenceKind = Literal["invocation", "operation"]
_StructuralRngId = tuple[str | int, ...]

_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_PROBPIPE_DISTRIBUTION_PROVIDER_ABI = "probpipe.distribution/v1"


class _RandomEventPlan(Protocol):
    """Structural event fields accepted by the broker."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


@dataclass(frozen=True)
class _DirectRandomEventPlan:
    """Singleton source/unit event used outside lifting."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


@dataclass(frozen=True)
class StochasticEffectPlan:
    """Immutable plan anchor for one automatic-key request."""

    operation_kind: str
    execution_mode: str
    event: _RandomEventPlan
    sample_shape: tuple[int, ...] | None
    sampling_abi: str
    provider_abi: str


def _singleton_effect_plan(
    *,
    operation_kind: str,
    execution_mode: str,
    sample_shape: tuple[int, ...] | None,
    source_index: int = 0,
    sampling_abi: str = _DISTRIBUTION_SAMPLING_ABI,
    provider_abi: str = _PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
) -> StochasticEffectPlan:
    """Build the standard singleton event plan for a direct operation."""
    return StochasticEffectPlan(
        operation_kind=operation_kind,
        execution_mode=execution_mode,
        event=_DirectRandomEventPlan(
            ("source-group", source_index),
            ("singleton",),
        ),
        sample_shape=sample_shape,
        sampling_abi=sampling_abi,
        provider_abi=provider_abi,
    )


@dataclass
class _AutomaticKeyBroker:
    """Lazily commit and serve keys for one stochastic occurrence."""

    occurrence_kind: _OccurrenceKind
    _invocation: _workflow_context._WorkflowInvocation | None = None
    _lock: Any = field(default_factory=Lock, repr=False)

    def key_for(self, plan: StochasticEffectPlan) -> PRNGKey:
        """Return the workflow-owned key for one planned effect."""
        if not isinstance(plan, StochasticEffectPlan):
            raise TypeError("automatic key requests require a StochasticEffectPlan")
        with self._lock:
            if self._invocation is None:
                self._invocation = _workflow_context._commit_stochastic_invocation(
                    self.occurrence_kind
                )
            invocation = self._invocation
        return invocation.key_for(
            stochastic_source_id=plan.event.stochastic_source_id,
            logical_unit_id=plan.event.logical_unit_id,
        )


_ACTIVE_AUTOMATIC_KEY_BROKER: ContextVar[_AutomaticKeyBroker | None] = ContextVar(
    "probpipe_active_automatic_key_broker",
    default=None,
)


@contextmanager
def _function_stochastic_scope() -> Iterator[_AutomaticKeyBroker]:
    """Install a lazy broker for one public Function invocation."""
    _workflow_context._assert_workflow_admission()
    broker = _AutomaticKeyBroker("invocation")
    token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
    try:
        yield broker
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


@contextmanager
def _managed_stochastic_scope() -> Iterator[_AutomaticKeyBroker]:
    """Reuse an active broker or install one managed-operation broker."""
    _workflow_context._assert_workflow_admission()
    active = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if active is not None:
        yield active
        return

    with _workflow_context._ephemeral_workflow_run():
        broker = _AutomaticKeyBroker("operation")
        token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
        try:
            yield broker
        finally:
            _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


def _resolve_automatic_key(
    explicit_key: PRNGKey | None,
    plan: StochasticEffectPlan | None,
) -> PRNGKey:
    """Preserve an explicit key or resolve one through the active broker."""
    if explicit_key is not None:
        return explicit_key
    if plan is None:
        raise TypeError("an omitted key requires a stochastic effect plan")

    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        return broker.key_for(plan)
    with _managed_stochastic_scope() as managed_broker:
        return managed_broker.key_for(plan)
