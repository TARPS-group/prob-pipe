"""Workflow execution contexts for structurally derived random keys."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from os import urandom as _os_urandom
from threading import Lock, get_ident
from types import TracebackType
from typing import Any, Literal

from ..custom_types import PRNGKey
from ._workflow_errors import UnmanagedConcurrentWorkflowEntryError
from ._workflow_rng import (
    RandomEventIdentity,
    derive_event_key_words,
    encode_random_event,
    jax_key_from_words,
    seed_to_root_words,
)

_WorkflowContextKind = Literal[
    "seeded",
    "anonymous",
    "ephemeral",
    "nested",
    "managed",
]


@dataclass(frozen=True)
class _WorkflowOwner:
    """Operational owner of one workflow frame."""

    thread_id: int
    task_id: int | None


@dataclass(frozen=True)
class _WorkflowFrame:
    """Immutable context binding for one explicit or provisional run."""

    kind: _WorkflowContextKind
    seed_words: tuple[int, int] | None
    parent: _WorkflowFrame | None
    owner: _WorkflowOwner
    ledger: _StochasticLedger = field(default_factory=lambda: _StochasticLedger())
    state: _WorkflowFrameState = field(default_factory=lambda: _WorkflowFrameState())


@dataclass
class _StochasticLedger:
    """Thread-safe allocator for one frame's stochastic child positions."""

    next_ordinal: int = 0
    lock: Any = field(default_factory=Lock, repr=False)

    def commit(self) -> int:
        """Commit and return the next stochastic child ordinal."""
        with self.lock:
            ordinal = self.next_ordinal
            self.next_ordinal += 1
            return ordinal


@dataclass
class _WorkflowFrameState:
    """Lazy materialization state kept behind an immutable frame binding."""

    path_prefix: tuple[Any, ...] | None = None
    managed_unit_segment: tuple[Any, ...] | None = None
    root_words: tuple[int, int] | None = None
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass
class _EventClaims:
    """Idempotent raw-key claims within one stochastic invocation."""

    words_by_identity: dict[bytes, tuple[int, int]] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(frozen=True)
class _WorkflowInvocation:
    """One committed stochastic occurrence in an active workflow frame."""

    frame: _WorkflowFrame
    occurrence_path: tuple[Any, ...]
    claims: _EventClaims = field(default_factory=_EventClaims)

    def key_for(
        self,
        *,
        stochastic_source_id: Any,
        logical_unit_id: Any,
    ) -> PRNGKey:
        """Claim the context-derived key for one source and logical unit."""
        identity = RandomEventIdentity(
            occurrence_path=self.occurrence_path,
            stochastic_source_id=stochastic_source_id,
            logical_unit_id=logical_unit_id,
        )
        encoded = encode_random_event(identity)
        with self.claims.lock:
            words = self.claims.words_by_identity.get(encoded)
            if words is None:
                words = derive_event_key_words(_resolve_root_words(self.frame), identity)
                self.claims.words_by_identity[encoded] = words
        return jax_key_from_words(words)


_ACTIVE_WORKFLOW_FRAME: ContextVar[_WorkflowFrame | None] = ContextVar(
    "probpipe_active_workflow_frame",
    default=None,
)


@dataclass
class _StochasticProbeState:
    """Observable stochastic effects reached during one route probe."""

    effect_observed: bool = False


_STOCHASTIC_PROBE_STATE: ContextVar[_StochasticProbeState | None] = ContextVar(
    "probpipe_stochastic_probe_state",
    default=None,
)


class _StochasticProbeSignal(RuntimeError):
    """Signal that JAX probing reached workflow-owned randomness."""


class _WorkflowRunScope:
    """Synchronous context-manager implementation for ``workflow_run``."""

    def __init__(
        self,
        seed: int | None,
        *,
        root_kind: Literal["anonymous", "ephemeral"] = "anonymous",
    ):
        self._seed = seed
        self._root_kind = root_kind
        self._frame: _WorkflowFrame | None = None
        self._token: Token[_WorkflowFrame | None] | None = None

    def __enter__(self) -> None:
        if self._token is not None:
            raise RuntimeError("workflow_run context is already active")
        parent = _ACTIVE_WORKFLOW_FRAME.get()
        _assert_workflow_admission(parent)
        seed_words = None if self._seed is None else seed_to_root_words(self._seed)
        if parent is None:
            kind: _WorkflowContextKind = "seeded" if seed_words is not None else self._root_kind
        else:
            kind = "nested"
        frame = _WorkflowFrame(
            kind=kind,
            seed_words=seed_words,
            parent=parent,
            owner=_current_workflow_owner(),
            state=_WorkflowFrameState(path_prefix=() if parent is None else None),
        )
        self._frame = frame
        self._token = _ACTIVE_WORKFLOW_FRAME.set(frame)
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._token is None or self._frame is None:
            raise RuntimeError("workflow_run context is not active")
        _assert_workflow_admission(self._frame)
        if _ACTIVE_WORKFLOW_FRAME.get() is not self._frame:
            raise RuntimeError("workflow_run contexts must exit in nesting order")
        _ACTIVE_WORKFLOW_FRAME.reset(self._token)
        self._frame = None
        self._token = None


def workflow_run(seed: int | None = None) -> _WorkflowRunScope:
    """Create a workflow-owned random execution scope.

    Parameters
    ----------
    seed : int or None
        Unsigned 64-bit root seed. ``None`` requests a lazy anonymous root.

    Returns
    -------
    context manager
        A synchronous workflow execution scope.

    Notes
    -----
    Sequential Function lifting uses this scope now. Other omitted-key paths
    will migrate to the same workflow broker before this feature is complete.
    """
    return _WorkflowRunScope(seed)


@contextmanager
def _ephemeral_workflow_run() -> Iterator[None]:
    """Install one lazy ephemeral root unless a workflow frame is active."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    if frame is not None:
        _assert_workflow_admission(frame)
        yield
        return
    with _WorkflowRunScope(None, root_kind="ephemeral"):
        yield


@contextmanager
def _workflow_probe() -> Iterator[None]:
    """Prevent a route probe from committing workflow stochastic state."""
    state = _StochasticProbeState()
    token = _STOCHASTIC_PROBE_STATE.set(state)
    try:
        yield
    except _StochasticProbeSignal:
        raise
    else:
        if state.effect_observed:
            raise _StochasticProbeSignal(
                "JAX route probing reached a workflow-owned stochastic operation"
            )
    finally:
        _STOCHASTIC_PROBE_STATE.reset(token)


def _commit_stochastic_invocation(
    occurrence_kind: Literal["invocation", "operation"] = "invocation",
) -> _WorkflowInvocation:
    """Commit one stochastic invocation in the active workflow frame."""
    probe_state = _STOCHASTIC_PROBE_STATE.get()
    if probe_state is not None:
        probe_state.effect_observed = True
        raise _StochasticProbeSignal(
            "JAX route probing reached a workflow-owned stochastic operation"
        )
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    if frame is None:
        raise RuntimeError("a stochastic invocation requires an active workflow context")
    _assert_workflow_admission(frame)
    return _commit_stochastic_invocation_in_frame(frame, occurrence_kind)


def _commit_stochastic_invocation_in_frame(
    frame: _WorkflowFrame,
    occurrence_kind: Literal["invocation", "operation"],
) -> _WorkflowInvocation:
    """Commit against a previously admitted frame for a managed child."""
    path_prefix = _materialize_path(frame)
    ordinal = frame.ledger.commit()
    return _WorkflowInvocation(
        frame=frame,
        occurrence_path=(*path_prefix, (occurrence_kind, ordinal)),
    )


def _materialize_path(frame: _WorkflowFrame) -> tuple[Any, ...]:
    path_prefix = frame.state.path_prefix
    if path_prefix is not None:
        return path_prefix
    if frame.parent is None:  # pragma: no cover - guarded by frame construction
        raise RuntimeError("root workflow frame has no occurrence path")

    with frame.state.lock:
        if frame.state.path_prefix is None:
            parent_path = _materialize_path(frame.parent)
            if frame.state.managed_unit_segment is not None:
                frame.state.path_prefix = (
                    *parent_path,
                    frame.state.managed_unit_segment,
                )
            else:
                scope_ordinal = frame.parent.ledger.commit()
                frame.state.path_prefix = (*parent_path, ("scope", scope_ordinal))
        return frame.state.path_prefix


def _current_workflow_owner() -> _WorkflowOwner:
    """Return the current thread/task identity for admission checks."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return _WorkflowOwner(
        thread_id=get_ident(),
        task_id=None if task is None else id(task),
    )


def _assert_workflow_admission(frame: _WorkflowFrame | None = None) -> None:
    """Reject passive context copies entering from another thread or task."""
    active = _ACTIVE_WORKFLOW_FRAME.get() if frame is None else frame
    if active is None or active.owner == _current_workflow_owner():
        return
    raise UnmanagedConcurrentWorkflowEntryError(
        "The active workflow context belongs to another thread or asyncio task. "
        "Use a ProbPipe-managed execution route, or start a new workflow_run in "
        "the concurrent worker instead of copying the parent context."
    )


def _capture_active_workflow_frame() -> _WorkflowFrame | None:
    """Capture an admitted parent frame for managed work-item transport."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    _assert_workflow_admission(frame)
    return frame


@contextmanager
def _managed_work_item_scope(
    parent: _WorkflowFrame,
    unit_segment: tuple[Any, ...],
) -> Iterator[None]:
    """Install a thread/task-owned child frame for one managed work item."""
    frame = _WorkflowFrame(
        kind="managed",
        seed_words=None,
        parent=parent,
        owner=_current_workflow_owner(),
        state=_WorkflowFrameState(managed_unit_segment=unit_segment),
    )
    token = _ACTIVE_WORKFLOW_FRAME.set(frame)
    try:
        yield
    finally:
        if _ACTIVE_WORKFLOW_FRAME.get() is not frame:
            raise RuntimeError("managed workflow frames must exit in nesting order")
        _ACTIVE_WORKFLOW_FRAME.reset(token)


@contextmanager
def _transported_workflow_frame(
    root_words: tuple[int, int] | None,
) -> Iterator[None]:
    """Install a standalone worker frame from serializable parent authority."""
    frame = _WorkflowFrame(
        kind="managed",
        seed_words=root_words,
        parent=None,
        owner=_current_workflow_owner(),
        state=_WorkflowFrameState(
            path_prefix=(),
            root_words=root_words,
        ),
    )
    token = _ACTIVE_WORKFLOW_FRAME.set(frame)
    try:
        yield
    finally:
        if _ACTIVE_WORKFLOW_FRAME.get() is not frame:
            raise RuntimeError("transported workflow frames must exit in nesting order")
        _ACTIVE_WORKFLOW_FRAME.reset(token)


def _resolve_root_words(frame: _WorkflowFrame) -> tuple[int, int]:
    root_words = frame.state.root_words
    if root_words is not None:
        return root_words

    with frame.state.lock:
        if frame.state.root_words is None:
            if frame.seed_words is not None:
                frame.state.root_words = frame.seed_words
            elif frame.parent is not None:
                frame.state.root_words = _resolve_root_words(frame.parent)
            else:
                entropy = _os_urandom(8)
                if len(entropy) != 8:  # pragma: no cover - OS contract guard
                    raise RuntimeError("OS entropy provider did not return 8 bytes")
                frame.state.root_words = (
                    int.from_bytes(entropy[:4], "big"),
                    int.from_bytes(entropy[4:], "big"),
                )
        return frame.state.root_words
