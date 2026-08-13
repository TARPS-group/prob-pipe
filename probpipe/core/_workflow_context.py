"""Workflow execution contexts for structurally derived random keys."""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from os import urandom as _os_urandom
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from ..custom_types import PRNGKey
from ._workflow_errors import UnmanagedConcurrentWorkflowEntryError
from ._workflow_rng import (
    RandomEventIdentity,
    _RandomEventPath,
    _RandomEventValue,
    derive_event_key_words_from_encoded,
    encode_random_event,
    jax_key_from_words,
    seed_to_root_words,
)

if TYPE_CHECKING:
    from ._workflow_broker import _RemoteCoordinationObservation

_WorkflowContextKind = Literal[
    "seeded",
    "anonymous",
    "ephemeral",
    "nested",
    "managed",
    "replay",
]


@dataclass(frozen=True, slots=True)
class _WorkflowOwner:
    """Operational owner of one workflow frame."""

    process_id: int
    thread_ref: weakref.ReferenceType[threading.Thread]
    task_ref: weakref.ReferenceType[asyncio.Task[Any]] | None


@dataclass(frozen=True, slots=True)
class _WorkflowFrame:
    """Immutable context binding for one explicit or provisional run."""

    kind: _WorkflowContextKind
    seed_words: tuple[int, int] | None
    parent: _WorkflowFrame | None
    owner: _WorkflowOwner
    ledger: _StochasticLedger = field(default_factory=lambda: _StochasticLedger())
    state: _WorkflowFrameState = field(default_factory=lambda: _WorkflowFrameState())


@dataclass(slots=True)
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


@dataclass(slots=True)
class _WorkflowFrameState:
    """Lazy materialization state kept behind an immutable frame binding."""

    path_prefix: _RandomEventPath | None = None
    managed_unit_segment: _RandomEventPath | None = None
    root_words: tuple[int, int] | None = None
    remote_coordination_observation: _RemoteCoordinationObservation | None = None
    closed: bool = False
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(slots=True)
class _EventClaims:
    """Idempotent raw-key claims within one stochastic invocation."""

    words_by_identity: dict[bytes, tuple[int, int]] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(frozen=True, slots=True)
class _WorkflowInvocation:
    """One committed stochastic occurrence in an active workflow frame."""

    frame: _WorkflowFrame
    occurrence_path: _RandomEventPath
    claims: _EventClaims = field(default_factory=_EventClaims)

    def key_for(
        self,
        *,
        stochastic_source_id: _RandomEventValue,
        logical_unit_id: _RandomEventValue,
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
                words = derive_event_key_words_from_encoded(
                    _resolve_root_words(self.frame),
                    encoded,
                )
                self.claims.words_by_identity[encoded] = words
        return jax_key_from_words(words)


_ACTIVE_WORKFLOW_FRAME: ContextVar[_WorkflowFrame | None] = ContextVar(
    "probpipe_active_workflow_frame",
    default=None,
)


@dataclass(slots=True)
class _StochasticProbeState:
    """Observable stochastic effects reached during one route probe."""

    effect_observed: bool = False


_STOCHASTIC_PROBE_STATE: ContextVar[_StochasticProbeState | None] = ContextVar(
    "probpipe_stochastic_probe_state",
    default=None,
)

_JAX_RUNTIME_GUARD: ContextVar[bool] = ContextVar(
    "probpipe_jax_runtime_guard",
    default=False,
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
        from . import _workflow_replay

        if _workflow_replay._replay_is_active():
            from ._workflow_errors import ReplayCompatibilityError

            raise ReplayCompatibilityError("workflow_run cannot be nested inside replay_run")
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
        _close_workflow_frame(self._frame)
        self._frame = None
        self._token = None


def workflow_run(seed: int | None = None) -> _WorkflowRunScope:
    """Create a workflow-owned random execution scope.

    Parameters
    ----------
    seed : int or None
        Non-boolean unsigned 64-bit root seed. Re-entering the same seeded
        scope with the same stochastic structure reproduces its workflow-owned
        random events. ``None`` requests a lazy anonymous root.

    Returns
    -------
    context manager
        A synchronous workflow execution scope.

    Raises
    ------
    TypeError
        If ``seed`` is not an integer or is a boolean.
    ValueError
        If ``seed`` is outside ``[0, 2**64 - 1]``.
    UnmanagedConcurrentWorkflowEntryError
        If an active workflow context was copied into a foreign thread or
        asyncio task without a ProbPipe-managed work-item frame.

    Notes
    -----
    ProbPipe operations that omit a key share this scope's structural random
    event stream. Explicit caller-owned keys do not consume that stream.
    Anonymous and nested scopes remain unmaterialized until their first
    workflow-owned random event. A bare omitted-key operation uses its own
    equivalent ephemeral scope.
    """
    return _WorkflowRunScope(seed)


@contextmanager
def _ephemeral_workflow_run() -> Generator[None, None, None]:
    """Install one lazy ephemeral root unless a workflow frame is active."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    if frame is not None:
        _assert_workflow_admission(frame)
        yield
        return
    with _WorkflowRunScope(None, root_kind="ephemeral"):
        yield


@contextmanager
def _workflow_probe() -> Generator[None, None, None]:
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


@contextmanager
def _workflow_jax_runtime_guard() -> Generator[None, None, None]:
    """Forbid omitted-key effects and managed submission inside actual JAX execution."""
    token = _JAX_RUNTIME_GUARD.set(True)
    try:
        yield
    finally:
        _JAX_RUNTIME_GUARD.reset(token)


def _workflow_side_effects_forbidden() -> bool:
    """Return whether execution is a side-effect-free probe or JAX body."""
    return _STOCHASTIC_PROBE_STATE.get() is not None or _JAX_RUNTIME_GUARD.get()


def _guard_managed_submission() -> None:
    """Reject managed transport from a probe or an actual JAX body."""
    probe_state = _STOCHASTIC_PROBE_STATE.get()
    if probe_state is not None:
        probe_state.effect_observed = True
        raise _StochasticProbeSignal("JAX route probing reached managed submission")
    if _JAX_RUNTIME_GUARD.get():
        raise TypeError(
            "JAX workflow execution cannot perform managed submission. Use "
            "dispatch='auto', 'sequential', or a caller-owned key outside the JAX body."
        )


def _guard_automatic_key_request() -> None:
    """Reject omitted-key randomness during actual JAX execution."""
    if _JAX_RUNTIME_GUARD.get():
        raise TypeError(
            "JAX workflow execution cannot request workflow-owned randomness with "
            "key=None. Pass an explicit key, or use dispatch='auto', 'sequential', "
            "or 'thread'."
        )


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


def _materialize_path(frame: _WorkflowFrame) -> _RandomEventPath:
    _assert_workflow_frame_open(frame)
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


def _materialize_descendant_path(
    frame: _WorkflowFrame,
    ancestor: _WorkflowFrame,
) -> _RandomEventPath:
    """Return a lazily materialized path relative to one managed ancestor."""
    cursor = frame
    while cursor is not ancestor:
        if cursor.parent is None:
            raise RuntimeError("workflow frame is not below its managed work-item frame")
        cursor = cursor.parent

    ancestor_path = _materialize_path(ancestor)
    descendant_path = _materialize_path(frame)
    if descendant_path[: len(ancestor_path)] != ancestor_path:
        raise RuntimeError("workflow descendant path does not extend its managed ancestor")
    return descendant_path[len(ancestor_path) :]


def _current_workflow_owner() -> _WorkflowOwner:
    """Return the current thread/task identity for admission checks."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return _WorkflowOwner(
        process_id=os.getpid(),
        thread_ref=weakref.ref(threading.current_thread()),
        task_ref=None if task is None else weakref.ref(task),
    )


def _assert_workflow_admission(frame: _WorkflowFrame | None = None) -> None:
    """Reject passive context copies entering from another thread or task."""
    active = _ACTIVE_WORKFLOW_FRAME.get() if frame is None else frame
    if active is None:
        return
    owner = active.owner
    current = _current_workflow_owner()
    same_owner = (
        owner.process_id == current.process_id
        and owner.thread_ref() is current.thread_ref()
        and (
            (owner.task_ref is None and current.task_ref is None)
            or (
                owner.task_ref is not None
                and current.task_ref is not None
                and owner.task_ref() is current.task_ref()
            )
        )
    )
    if not same_owner:
        raise UnmanagedConcurrentWorkflowEntryError(
            "The active workflow context belongs to another process, thread, or asyncio task. "
            "Use a ProbPipe-managed execution route, or start a new workflow_run in "
            "the concurrent worker instead of copying the parent context."
        )
    _assert_workflow_frame_open(active)


def _assert_workflow_frame_open(frame: _WorkflowFrame) -> None:
    """Reject use of a workflow frame after its installing scope has exited."""
    with frame.state.lock:
        if frame.state.closed:
            raise UnmanagedConcurrentWorkflowEntryError(
                "The active workflow context has already closed. Do not re-enter a "
                "saved Context from an exited workflow_run."
            )


def _close_workflow_frame(frame: _WorkflowFrame) -> None:
    """Mark a workflow frame unavailable to passive saved-context copies."""
    with frame.state.lock:
        frame.state.closed = True


def _capture_active_workflow_frame() -> _WorkflowFrame | None:
    """Capture an admitted parent frame for managed work-item transport."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    _assert_workflow_admission(frame)
    return frame


def _find_remote_coordination_observation(
    frame: _WorkflowFrame | None,
) -> _RemoteCoordinationObservation | None:
    """Find an attempt observation carried by this managed frame chain."""
    cursor = frame
    while cursor is not None:
        with cursor.state.lock:
            observation = cursor.state.remote_coordination_observation
        if observation is not None:
            return observation
        cursor = cursor.parent
    return None


def _assert_transported_frame_consistency(
    frame: _WorkflowFrame,
    root_words: tuple[int, int],
) -> None:
    """Check that an installed worker frame matches its transport envelope.

    This is an internal wiring invariant, not payload authentication.
    """
    _assert_workflow_admission(frame)
    with frame.state.lock:
        matches = (
            frame.kind == "managed"
            and frame.parent is None
            and frame.seed_words is None
            and frame.state.path_prefix == ()
            and frame.state.root_words == root_words
            and frame.state.managed_unit_segment is None
        )
    if not matches:
        raise RuntimeError("remote managed workflow frame does not match its transport envelope")


@contextmanager
def _managed_work_item_scope(
    parent: _WorkflowFrame,
    unit_segment: _RandomEventPath,
) -> Generator[None, None, None]:
    """Install a thread/task-owned child frame for one managed work item."""
    _assert_workflow_frame_open(parent)
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
        _close_workflow_frame(frame)


@contextmanager
def _transported_workflow_frame(
    root_words: tuple[int, int] | None,
) -> Generator[None, None, None]:
    """Install a standalone worker frame from serializable parent authority."""
    frame = _WorkflowFrame(
        kind="managed",
        seed_words=None,
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
        _close_workflow_frame(frame)


@contextmanager
def _replay_workflow_frame(root_words: tuple[int, int]) -> Generator[None, None, None]:
    """Install one standalone root restored from replay provenance."""
    frame = _WorkflowFrame(
        kind="replay",
        seed_words=None,
        parent=None,
        owner=_current_workflow_owner(),
        state=_WorkflowFrameState(path_prefix=(), root_words=root_words),
    )
    token = _ACTIVE_WORKFLOW_FRAME.set(frame)
    try:
        yield
    finally:
        if _ACTIVE_WORKFLOW_FRAME.get() is not frame:
            raise RuntimeError("replay workflow frames must exit in nesting order")
        _ACTIVE_WORKFLOW_FRAME.reset(token)
        _close_workflow_frame(frame)


def _resolve_root_words(frame: _WorkflowFrame) -> tuple[int, int]:
    _assert_workflow_frame_open(frame)
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
                if frame.kind == "managed":
                    observation = frame.state.remote_coordination_observation
                    if observation is not None:
                        observation.observe_effect()
                    raise RuntimeError(
                        "rootless managed workflow frame requires transported parent RNG authority"
                    )
                entropy = _os_urandom(8)
                if len(entropy) != 8:  # pragma: no cover - OS contract guard
                    raise RuntimeError("OS entropy provider did not return 8 bytes")
                frame.state.root_words = (
                    int.from_bytes(entropy[:4], "big"),
                    int.from_bytes(entropy[4:], "big"),
                )
        return frame.state.root_words


def _describe_rng_origin(frame: _WorkflowFrame) -> dict[str, str | int | None]:
    """Describe the effective root source without runtime ownership state."""
    cursor = frame
    while cursor.seed_words is None and cursor.parent is not None:
        cursor = cursor.parent
    if cursor.seed_words is not None:
        supplied_seed = cursor.seed_words[0] << 32 | cursor.seed_words[1]
        return {
            "context_kind": "seeded_run",
            "root_source": "user_seed",
            "supplied_seed": supplied_seed,
        }
    if cursor.kind == "anonymous":
        return {
            "context_kind": "anonymous_run",
            "root_source": "os_entropy",
            "supplied_seed": None,
        }
    if cursor.kind == "ephemeral":
        return {
            "context_kind": "ephemeral_bare_call",
            "root_source": "os_entropy",
            "supplied_seed": None,
        }
    if cursor.kind == "replay":
        return {
            "context_kind": "replay_run",
            "root_source": "replay_recipe",
            "supplied_seed": None,
        }
    return {
        "context_kind": "transported_run",
        "root_source": "transported_authority",
        "supplied_seed": None,
    }
