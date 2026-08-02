"""Workflow execution contexts for structurally derived random keys."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from types import TracebackType
from typing import Any, Literal

from ..custom_types import PRNGKey
from ._workflow_rng import (
    RandomEventIdentity,
    derive_event_key_words,
    encode_random_event,
    jax_key_from_words,
    seed_to_root_words,
)

_WorkflowContextKind = Literal["seeded", "anonymous", "ephemeral", "nested"]


@dataclass(frozen=True)
class _WorkflowFrame:
    """Immutable context binding for one explicit or provisional run."""

    kind: _WorkflowContextKind
    seed_words: tuple[int, int] | None
    parent: _WorkflowFrame | None
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
_STOCHASTIC_PROBE_ACTIVE: ContextVar[bool] = ContextVar(
    "probpipe_stochastic_probe_active",
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
        seed_words = None if self._seed is None else seed_to_root_words(self._seed)
        parent = _ACTIVE_WORKFLOW_FRAME.get()
        if parent is None:
            kind: _WorkflowContextKind = "seeded" if seed_words is not None else self._root_kind
        else:
            kind = "nested"
        frame = _WorkflowFrame(
            kind=kind,
            seed_words=seed_words,
            parent=parent,
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
    """
    return _WorkflowRunScope(seed)


@contextmanager
def _ephemeral_workflow_run() -> Iterator[None]:
    """Install one lazy ephemeral root unless a workflow frame is active."""
    if _ACTIVE_WORKFLOW_FRAME.get() is not None:
        yield
        return
    with _WorkflowRunScope(None, root_kind="ephemeral"):
        yield


@contextmanager
def _workflow_probe() -> Iterator[None]:
    """Prevent a route probe from committing workflow stochastic state."""
    token = _STOCHASTIC_PROBE_ACTIVE.set(True)
    try:
        yield
    finally:
        _STOCHASTIC_PROBE_ACTIVE.reset(token)


def _has_active_workflow_frame() -> bool:
    """Return whether the current task has a workflow frame installed."""
    return _ACTIVE_WORKFLOW_FRAME.get() is not None


def _commit_stochastic_invocation() -> _WorkflowInvocation:
    """Commit one stochastic invocation in the active workflow frame."""
    if _STOCHASTIC_PROBE_ACTIVE.get():
        raise _StochasticProbeSignal(
            "JAX route probing reached a workflow-owned stochastic operation"
        )
    frame = _ACTIVE_WORKFLOW_FRAME.get()
    if frame is None:
        raise RuntimeError("a stochastic invocation requires an active workflow context")
    path_prefix = _materialize_path(frame)
    ordinal = frame.ledger.commit()
    return _WorkflowInvocation(
        frame=frame,
        occurrence_path=(*path_prefix, ("invocation", ordinal)),
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
            scope_ordinal = frame.parent.ledger.commit()
            frame.state.path_prefix = (*parent_path, ("scope", scope_ordinal))
        return frame.state.path_prefix


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
                entropy = os.urandom(8)
                if len(entropy) != 8:  # pragma: no cover - OS contract guard
                    raise RuntimeError("OS entropy provider did not return 8 bytes")
                frame.state.root_words = (
                    int.from_bytes(entropy[:4], "big"),
                    int.from_bytes(entropy[4:], "big"),
                )
        return frame.state.root_words
