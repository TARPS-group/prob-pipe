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
from typing import Any, Literal

from ..core._call_context import _jax_transformation_active, current_call_context
from ..core.config import ProvenanceMode, _register_open_run_provenance_mode_reader, config
from ..custom_types import PRNGKey
from ._errors import UnmanagedConcurrentWorkflowEntryError
from ._rng import (
    RandomEventIdentity,
    _RandomEventPath,
    _RandomEventValue,
    derive_event_key_words_from_encoded,
    encode_random_event,
    jax_key_from_words,
    seed_to_root_words,
)

_WorkflowContextKind = Literal[
    "seeded",
    "anonymous",
    "ephemeral",
    "nested",
    "evaluation_point",
    "replay",
]


@dataclass(frozen=True)
class _WorkflowOwner:
    """Operational owner of one workflow frame."""

    process_id: int
    thread_ref: weakref.ReferenceType[threading.Thread]
    task_ref: weakref.ReferenceType[asyncio.Task[Any]] | None


@dataclass(frozen=True)
class _WorkflowFrame:
    """Immutable context binding for one explicit or provisional run."""

    kind: _WorkflowContextKind
    seed_words: tuple[int, int] | None
    provenance_mode: ProvenanceMode
    parent: _WorkflowFrame | None
    owner: _WorkflowOwner
    default_state: _DefaultWorkflowState | None = None
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
    point_segment: _RandomEventPath | None = None
    root_words: tuple[int, int] | None = None
    closed: bool = False
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(slots=True)
class _DefaultWorkflowState:
    """The root and positions retained between calls in one owner's default scope."""

    owner: _WorkflowOwner
    seed_words: tuple[int, int] | None
    root_words: tuple[int, int] | None = None
    ledger: _StochasticLedger = field(default_factory=_StochasticLedger)
    lock: Any = field(default_factory=Lock, repr=False)

    def resolve_root(self) -> tuple[int, int]:
        """Materialize the default root once, on its first random event."""
        with self.lock:
            if self.root_words is None:
                self.root_words = (
                    self.seed_words if self.seed_words is not None else _entropy_root_words()
                )
            return self.root_words


_DEFAULT_WORKFLOW_STATE: ContextVar[_DefaultWorkflowState | None] = ContextVar(
    "probpipe_default_workflow_state", default=None
)


@dataclass(slots=True)
class _EventClaims:
    """Idempotent raw-key claims within one stochastic invocation."""

    words_by_identity: dict[bytes, tuple[int, int]] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(frozen=True)
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


def _active_provenance_mode() -> ProvenanceMode:
    """Return the provenance mode fixed for the active workflow run."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    return config.provenance.mode if frame is None else frame.provenance_mode


def _open_run_provenance_mode() -> ProvenanceMode | None:
    """The provenance mode fixed for the workflow run open in this thread, or ``None`` outside every run."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    return None if frame is None else frame.provenance_mode


_register_open_run_provenance_mode_reader(_open_run_provenance_mode)


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


#: The refusal of a keyless draw inside a body that JAX maps.
JAX_KEYLESS_DRAW_MESSAGE = (
    "a function that JAX maps cannot draw random numbers without an explicit key. Pass "
    "key=... to each draw, or claim traceable=False so the function runs in Python"
)


class _WorkflowRunScope:
    """Synchronous context-manager implementation for ``workflow_run``."""

    def __init__(
        self,
        seed: int | None,
        *,
        root_kind: Literal["anonymous", "ephemeral"] = "anonymous",
    ):
        self._seed = seed
        self._root_kind: Literal["anonymous", "ephemeral"] = root_kind
        self._frame: _WorkflowFrame | None = None
        self._token: Token[_WorkflowFrame | None] | None = None

    def __enter__(self) -> None:
        if current_call_context().forbids_workflow_randomness and self._root_kind != "ephemeral":
            raise TypeError(JAX_KEYLESS_DRAW_MESSAGE)
        if self._token is not None:
            raise RuntimeError("workflow_run context is already active")
        from . import _replay

        if _replay._replay_is_active():
            from ._errors import ReplayCompatibilityError

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
            provenance_mode=(config.provenance.mode if parent is None else parent.provenance_mode),
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


class _WorkflowRun:
    """Create explicit workflow scopes and manage the current context's default scope.

    ``workflow_run(seed)`` creates an explicit context manager. Its seed and
    nesting rules are stated by ``__call__``. ``set_default`` starts a stream
    for successive bare calls, and ``clear_default`` removes that stream.
    """

    def __call__(self, seed: int | None = None) -> _WorkflowRunScope:
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
            asyncio task without a ProbPipe evaluation point frame.

        Notes
        -----
        The outermost scope fixes the run's provenance mode, the
        ``config.provenance.mode`` in effect when it is entered, and nested scopes
        keep it. Inside the run, a ``config.provenance.using`` block, an
        assignment, or a ``reset`` that would set another mode raises
        ``RuntimeError``, naming the run's mode, so set the mode before the run
        opens.

        ProbPipe operations that omit a key share this scope's structural random
        event stream. Explicit caller-owned keys do not consume that stream.
        Anonymous and nested scopes remain unmaterialized until their first
        workflow-owned random event. A bare omitted-key operation uses the configured
        default scope, or its own ephemeral scope when no default is configured.
        """
        return _WorkflowRunScope(seed)

    def set_default(self, *, seed: int | None = None) -> None:
        """Create or replace the current context's continuing default random scope.

        Each bare top-level call continues the same root and random positions.
        Setting the default again resets those positions. Invalid seeds leave
        the previous default intact. An explicit workflow scope or replay uses
        its own stream, and the default resumes after it closes. A failed call
        keeps the positions it consumed.

        Each top-level call fixes its provenance mode when it opens. The mode
        can change between calls without changing the random stream.
        ``config.reset()`` leaves the default scope intact.

        The default belongs to the process, thread, and asyncio task that sets
        it. In an IPython notebook, it continues across cells, including cells
        with top-level ``await``. Each cell's execution task takes ownership
        while user-created tasks remain isolated. Using an inherited default
        in another owner raises
        ``UnmanagedConcurrentWorkflowEntryError``; set or clear a default in
        that context first. ProbPipe evaluation points use the calling
        workflow's assigned random events. A new thread without a copied context has
        no default scope.

        Parameters
        ----------
        seed : int or None
            Non-boolean unsigned 64-bit root seed. ``None`` creates an anonymous
            continuing stream, reading system entropy on its first random event.

        Raises
        ------
        TypeError
            If ``seed`` is neither an integer nor ``None``, or is a boolean.
        ValueError
            If ``seed`` is outside ``[0, 2**64 - 1]``.
        RuntimeError
            Inside a Function call, workflow scope, replay, function check, or caller-owned
            JAX transformation. Set the default outside these contexts.
        """
        _guard_default_scope_change()
        seed_words = None if seed is None else seed_to_root_words(seed)
        _set_default_workflow_state(_DefaultWorkflowState(_current_workflow_owner(), seed_words))

    def clear_default(self) -> None:
        """Remove the current context's default scope, if any.

        Subsequent bare calls receive independent anonymous scopes. Clearing
        an inherited default removes it only from the current context.
        In a notebook cell, clearing also removes the default for later cells.

        Raises
        ------
        RuntimeError
            Inside a Function call, workflow scope, replay, function check, or caller-owned
            JAX transformation. Clear the default outside these contexts.
        """
        _guard_default_scope_change()
        _set_default_workflow_state(None)


workflow_run = _WorkflowRun()
"""Open explicit workflow scopes and manage the current context's default scope.

# `workflow_run(seed=None)` {#workflow-run-call}

Return a synchronous context manager for an explicit workflow scope.
An integer seed reproduces its random events when the stochastic structure
is unchanged. At the outermost scope, `None` uses system entropy on the first
random event. A nested scope inherits the root unless it supplies a seed,
and extends its parent's structural positions even when it supplies a seed.
An explicit scope entered between bare calls does not consume the default
stream. The scope fixes provenance mode for its whole block.

**Parameter:** `seed` is `None` or a non-boolean integer in `[0, 2**64 - 1]`.
Validation occurs when the context manager is entered. An invalid type raises
`TypeError`, and an out-of-range integer raises `ValueError`.
Using an explicit scope copied into another process, thread, or asyncio task
raises `UnmanagedConcurrentWorkflowEntryError` unless ProbPipe dispatched
that work through its worker mechanism. Changing provenance mode within an
active scope raises `RuntimeError`.

# `workflow_run.set_default(*, seed=None)` {#workflow-run-set-default}

Create or replace the default stream for successive bare calls. Return `None`.
Setting the default restarts its random positions, including when the seed
is unchanged. Calls with the same stochastic structure use the same keys as
successive calls in one explicit scope with the same seed. Explicit scopes
entered between calls and replay use their own streams. A failed call keeps
the random positions it consumed. Explicit-key numerical functions consume
none of the workflow's random stream.

**Parameter:** `seed` is `None` or a non-boolean integer in `[0, 2**64 - 1]`.
`None` creates a continuing anonymous stream, reading system entropy on its
first random event. An invalid type raises `TypeError`, and an out-of-range
integer raises `ValueError`. Validation precedes replacement, so an invalid
seed leaves the existing default intact.

Each top-level call fixes its provenance mode. The mode can change between
calls without changing the random stream. Changing it within an active call
raises `RuntimeError`. `config.reset()` leaves the default intact.

A default belongs to its configuring process, thread, and asyncio task. A new
thread without a copied context has no default. In an IPython notebook, the
default continues across cells, including top-level `await`, with ownership
passing to the task executing each cell. Tasks the user creates remain
isolated. Using an inherited default from another owner raises
`UnmanagedConcurrentWorkflowEntryError`. That owner can set or clear its own
default, or open an explicit scope. A child task's setting does not change
the notebook's default. ProbPipe workers use the random events assigned by
their calling workflow.

**Restrictions:** setting the default raises `RuntimeError` during a Function
call, an explicit workflow scope, or replay. It also raises during a function
check or a caller-owned JAX transformation, such as `jit`, `vmap`, or `grad`.

# `workflow_run.clear_default()` {#workflow-run-clear-default}

Remove the current context's default stream and return `None`. Subsequent
bare calls receive independent anonymous scopes. If no default exists, this
method does nothing. Clearing an inherited default affects only the current
context. In a notebook cell, clearing also removes the default for later cells.

**Restrictions:** the same contexts that forbid `set_default` also forbid
`clear_default`, raising `RuntimeError` before changing the default.
"""


def _set_default_workflow_state(state: _DefaultWorkflowState | None) -> None:
    """Install a default and retain a notebook cell's setting for subsequent cells."""
    from ._notebook_scope import _remember_notebook_default

    _DEFAULT_WORKFLOW_STATE.set(state)
    _remember_notebook_default(state)


def _guard_default_scope_change() -> None:
    """Refuse a default-scope mutation during a running or traced computation."""
    context = current_call_context()
    if (
        _ACTIVE_WORKFLOW_FRAME.get() is not None
        or context.call_depth > 0
        or context.in_evaluation_point
        or context.forbids_workflow_randomness
        or _workflow_side_effects_forbidden()
        or _jax_transformation_active()
    ):
        raise RuntimeError(
            "cannot change the default workflow scope during a call, workflow_run, "
            "replay_run, function check, or JAX transformation. Set or clear the default "
            "outside the computation."
        )


@contextmanager
def _ephemeral_workflow_run() -> Generator[None, None, None]:
    """Reuse an active frame, or open a call under the default or a fresh root."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    if frame is not None:
        _assert_workflow_admission(frame)
        yield
        return
    default = _DEFAULT_WORKFLOW_STATE.get()
    if default is None:
        with _WorkflowRunScope(None, root_kind="ephemeral"):
            yield
        return
    if not _is_current_workflow_owner(default.owner):
        raise UnmanagedConcurrentWorkflowEntryError(
            "the default workflow scope belongs to another thread, process, or asyncio task. "
            "Call workflow_run.set_default() or workflow_run.clear_default() in this context "
            "before making a bare call."
        )
    frame = _WorkflowFrame(
        kind="seeded" if default.seed_words is not None else "anonymous",
        seed_words=default.seed_words,
        provenance_mode=config.provenance.mode,
        parent=None,
        owner=default.owner,
        default_state=default,
        ledger=default.ledger,
        state=_WorkflowFrameState(path_prefix=()),
    )
    token = _ACTIVE_WORKFLOW_FRAME.set(frame)
    try:
        yield
    finally:
        current = _ACTIVE_WORKFLOW_FRAME.get()
        _ACTIVE_WORKFLOW_FRAME.reset(token)
        _close_workflow_frame(frame)
        if current is not frame:
            raise RuntimeError("workflow_run contexts must exit in nesting order")


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
    """Forbid omitted-key effects inside actual JAX execution."""
    token = _JAX_RUNTIME_GUARD.set(True)
    try:
        yield
    finally:
        _JAX_RUNTIME_GUARD.reset(token)


def _workflow_side_effects_forbidden() -> bool:
    """Return whether execution is a side-effect-free probe or JAX body."""
    return _STOCHASTIC_PROBE_STATE.get() is not None or _JAX_RUNTIME_GUARD.get()


def _guard_automatic_key_request() -> None:
    """Reject omitted-key randomness during actual JAX execution."""
    if _JAX_RUNTIME_GUARD.get() or current_call_context().forbids_workflow_randomness:
        raise TypeError(JAX_KEYLESS_DRAW_MESSAGE)


def _guard_caller_jax_trace(operation_kind: str) -> None:
    """Reject a workflow-owned random event claimed inside the caller's JAX trace.

    The engine traces a call only under a route probe or the JAX runtime
    guard, and each of those settles a claim itself, so an active trace
    without either is one the caller opened.

    Parameters
    ----------
    operation_kind : str
        The operation that claims the event, named in the error.

    Raises
    ------
    RuntimeError
        If a JAX transformation the caller opened is tracing the call.
    """
    if _workflow_side_effects_forbidden() or not _jax_transformation_active():
        return
    raise RuntimeError(
        f"the {operation_kind!r} operation cannot draw random values "
        "inside a JAX transformation opened by the caller. Call the operation outside the "
        "transformation, or transform only its deterministic part."
    )


def _commit_stochastic_invocation(
    occurrence_kind: Literal["invocation", "operation"] = "invocation",
) -> _WorkflowInvocation:
    """Check admission and commit an invocation in the active workflow frame.

    Signal workflow-owned randomness during JAX probing before committing any
    state. Otherwise require the active frame to be open and owned by the
    current process, thread, and asyncio task, then delegate the commit to
    ``_commit_stochastic_invocation_in_frame``.
    """
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
    """Commit in an explicit frame whose admission the caller already established.

    Used after active-frame admission and by evaluation point execution when a worker
    lazily commits its parent broker's invocation. That parent frame can belong
    to another thread, so this helper does not repeat owner or JAX-probe checks.
    Path materialization still requires the frame to be open. Callers entering
    through the active context must use ``_commit_stochastic_invocation``.
    """
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
            if frame.state.point_segment is not None:
                frame.state.path_prefix = (
                    *parent_path,
                    frame.state.point_segment,
                )
            else:
                scope_ordinal = frame.parent.ledger.commit()
                frame.state.path_prefix = (*parent_path, ("scope", scope_ordinal))
        return frame.state.path_prefix


def _materialize_descendant_path(
    frame: _WorkflowFrame,
    ancestor: _WorkflowFrame,
) -> _RandomEventPath:
    """Return a lazily materialized path relative to one evaluation point ancestor."""
    cursor = frame
    while cursor is not ancestor:
        if cursor.parent is None:
            raise RuntimeError("workflow frame is not below its evaluation point frame")
        cursor = cursor.parent

    ancestor_path = _materialize_path(ancestor)
    descendant_path = _materialize_path(frame)
    if descendant_path[: len(ancestor_path)] != ancestor_path:
        raise RuntimeError("workflow descendant path does not extend its evaluation point ancestor")
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


def _is_current_workflow_owner(owner: _WorkflowOwner) -> bool:
    """Whether *owner* is the current process, thread, and asyncio task."""
    current = _current_workflow_owner()
    return (
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


def _assert_workflow_admission(frame: _WorkflowFrame | None = None) -> None:
    """Reject passive context copies entering from another thread or task."""
    active = _ACTIVE_WORKFLOW_FRAME.get() if frame is None else frame
    if active is None:
        return
    if not _is_current_workflow_owner(active.owner):
        raise UnmanagedConcurrentWorkflowEntryError(
            "this workflow_run belongs to another thread, process, or asyncio task. Run "
            "concurrent work through a Function (e.g. executor='thread'), or start a separate "
            "workflow_run inside the worker."
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
    """Capture an admitted parent frame for evaluation point transport."""
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    _assert_workflow_admission(frame)
    return frame


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
            frame.kind == "evaluation_point"
            and frame.parent is None
            and frame.seed_words is None
            and frame.state.path_prefix == ()
            and frame.state.root_words == root_words
            and frame.state.point_segment is None
        )
    if not matches:
        raise RuntimeError(
            "remote evaluation point workflow frame does not match its transport envelope"
        )


@contextmanager
def _evaluation_point_frame_scope(
    parent: _WorkflowFrame,
    point_segment: _RandomEventPath,
) -> Generator[None, None, None]:
    """Install a thread/task-owned child frame for one evaluation point."""
    _assert_workflow_frame_open(parent)
    frame = _WorkflowFrame(
        kind="evaluation_point",
        seed_words=None,
        provenance_mode=parent.provenance_mode,
        parent=parent,
        owner=_current_workflow_owner(),
        state=_WorkflowFrameState(point_segment=point_segment),
    )
    token = _ACTIVE_WORKFLOW_FRAME.set(frame)
    try:
        yield
    finally:
        if _ACTIVE_WORKFLOW_FRAME.get() is not frame:
            raise RuntimeError("evaluation point workflow frames must exit in nesting order")
        _ACTIVE_WORKFLOW_FRAME.reset(token)
        _close_workflow_frame(frame)


@contextmanager
def _transported_workflow_frame(
    root_words: tuple[int, int] | None,
) -> Generator[None, None, None]:
    """Install a standalone worker frame from serializable parent authority.

    The frame takes the run's provenance mode from the call context the work
    carries, which is installed first: a block's mode, which cannot differ
    from the run's, or else the session's mode when the root call opened.
    """
    frame = _WorkflowFrame(
        kind="evaluation_point",
        seed_words=None,
        provenance_mode=config.provenance.mode,
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
        provenance_mode=config.provenance.mode,
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
                if frame.kind == "evaluation_point":
                    raise RuntimeError(
                        "rootless evaluation point workflow frame requires transported parent RNG authority"
                    )
                frame.state.root_words = (
                    frame.default_state.resolve_root()
                    if frame.default_state is not None
                    else _entropy_root_words()
                )
        return frame.state.root_words


def _entropy_root_words() -> tuple[int, int]:
    """Read one 64-bit root from the operating system."""
    entropy = _os_urandom(8)
    if len(entropy) != 8:  # pragma: no cover - OS contract guard
        raise RuntimeError("OS entropy provider did not return 8 bytes")
    return int.from_bytes(entropy[:4], "big"), int.from_bytes(entropy[4:], "big")


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
