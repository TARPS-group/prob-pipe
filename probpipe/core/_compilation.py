"""Compiled JAX programs: the one rule for falling back to eager execution, and the program store.

ProbPipe compiles the computations it calls repeatedly, such as a family's
sampler or a joint's mapped conditional factor, so that a later call with
inputs of the same structure reuses the compiled JAX program. Some
computations do not trace, such as one that converts an input to a Python
number, and they run eagerly.

Provides:
  - :func:`compiled_or_eager`, which traces a JAX program before it first
    runs, runs the eager form when tracing fails, and lets an error raised
    while the JAX program compiles or runs propagate;
  - :class:`_ProgramStore` and :func:`_program_store_of`, the compiled JAX
    programs of one owner that run user or route code, each compiled under
    the static settings of the call that runs it;
  - :class:`_StaticSettings` and :func:`_static_settings_in_effect`, those
    settings.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np

from ._call_context import (
    _TRACED_CALL_TOKEN,
    _EvaluationLog,
    _EvaluationSetRecord,
    _jax_transformation_active,
    _merge_noted_into,
    _NumSamplesUsed,
    _open_call_context,
    _UnclassifiedSettingError,
    _under_call_context,
    current_call_context,
)
from ._immutable import transient_memo
from .config import _static_config_values_in_effect

__all__ = ["compiled_or_eager"]

logger = logging.getLogger(__name__)

#: The resolved controls of a call that a compiled JAX program can read while
#: it is traced, through ``_num_samples_in_effect`` and the planning of a lift.
_STATIC_CONTROL_NAMES = ("max_evaluations", "num_samples")

#: No static keyword arguments.
_NO_STATIC_KEYWORDS: Mapping[str, Hashable] = MappingProxyType({})


class _EmbeddedCallTracingError(TypeError):
    """A boundary declaration or a captured token forbids this staged call."""


_embedded_call_inspector: Callable[..., bool] | None = None


def _install_embedded_call_inspector(inspector: Callable[..., bool]) -> None:
    """Install the function engine's recognition of its own callbacks in jaxprs."""
    global _embedded_call_inspector
    _embedded_call_inspector = inspector


def _contains_embedded_calls(jaxpr: Any, *, validate_tokens: bool = True) -> bool:
    """Inspect the jaxpr through the function engine's callback recognition hook."""
    return (
        _embedded_call_inspector(jaxpr, validate_tokens=validate_tokens)
        if _embedded_call_inspector is not None
        else False
    )


def compiled_or_eager(
    decision_record: MutableMapping[Hashable, Any],
    decision_key: Hashable,
    program: Any,
    eager: Callable[..., Any],
    *arguments: Any,
    trace_cache_key: Hashable | None,
    static_keywords: Mapping[str, Hashable] = _NO_STATIC_KEYWORDS,
) -> Any:
    """*program* at *arguments*, or *eager* at them when *program* does not trace.

    Before *program* first runs under a new *trace_cache_key*, it is traced.
    When that fails, as for a body that converts a traced value to a Python
    number, *eager* runs instead. If *eager* succeeds, *decision_record*
    records *decision_key*, and later calls under it run *eager* without
    tracing *program*. If *eager* raises too, its error propagates and nothing
    is recorded, so a call that fails eagerly, such as one with an invalid
    argument, does not disable the program.

    An error raised while the traced JAX program compiles or runs
    propagates, and *eager* does not run: the eager form computes what the
    JAX program computes, so running it again would repeat the error or hide
    it.

    Parameters
    ----------
    decision_record : MutableMapping
        Where the outcome of each trace is kept, such as the transient memo
        of the object that owns *program*. It gains one entry for each
        decision key that runs eagerly, and one for each trace cache key and
        set of *static_keywords* that traced.
    decision_key : Hashable
        What decides whether *program* runs: the method, and the structure of
        the arguments when it can change the outcome.
    program : jax.jit object
        The compiled computation, a ``jax.jit`` object, which is traced with
        its ``trace`` method.
    eager : Callable
        The same computation, uncompiled, called as ``eager(*arguments)``.
    *arguments : Any
        The arguments of both.
    trace_cache_key : Hashable or None
        What JAX traces *program* again for: :func:`_trace_cache_key` of the
        arguments that can differ between calls under *decision_record*. An
        argument fixed for its owner, such as a law's backend, may be left
        out. ``None`` traces *program* before every run.
    static_keywords : Mapping of str to Hashable, optional
        Keyword arguments of *program* alone, each named in its
        ``static_argnames``; none by default.

    Returns
    -------
    Any
        The result of *program*, or of *eager* when *program* does not trace.

    Raises
    ------
    Exception
        Whatever *eager* raises after tracing *program* failed, and whatever
        the traced JAX program raises while it compiles or runs.
    RuntimeError
        If a trace of a program store's JAX program notes other values on the
        running call than an earlier trace of it, which propagates rather than
        selecting *eager*.
    """
    if (_RUNS_EAGERLY, decision_key) in decision_record:
        return eager(*arguments)
    traced_key = (_TRACED, decision_key, trace_cache_key, tuple(sorted(static_keywords.items())))
    if trace_cache_key is None or traced_key not in decision_record:
        try:
            program.trace(*arguments, **static_keywords)
        except (_VaryingTraceNotesError, _EmbeddedCallTracingError, _UnclassifiedSettingError):
            raise
        except Exception as tracing_error:
            eager_result = eager(*arguments)
            decision_record[(_RUNS_EAGERLY, decision_key)] = True
            logger.debug("%r runs uncompiled: tracing it raised %r", decision_key, tracing_error)
            return eager_result
        if trace_cache_key is not None:
            decision_record[traced_key] = True
    return program(*arguments, **static_keywords)


class _ProgramForCall:
    """A stored JAX program whose dynamic first argument is the running call's token.

    Detection traces with zero as data, never as a static constant. Only a
    program containing this engine's callbacks registers the running call.
    """

    def __init__(self, program: Any, entry: _StoreEntry) -> None:
        self._program = program
        self._entry = entry

    def trace(self, *arguments: Any, **static_keywords: Any) -> Any:
        """Trace and inspect the jaxpr before its first execution."""
        traced = self._program.trace(np.uint32(0), *arguments, **static_keywords)
        self._entry.has_embedded_calls = _contains_embedded_calls(traced.jaxpr)
        return traced

    def __call__(self, *arguments: Any, **static_keywords: Any) -> Any:
        """Run with the current call's token and preserve any host exception's type."""
        state = current_call_context().embedded_calls
        call_token = (
            state.token() if self._entry.has_embedded_calls and state is not None else np.uint32(0)
        )
        try:
            result = self._program(call_token, *arguments, **static_keywords)
            if state is not None and self._entry.has_embedded_calls:
                state.note_result(result)
                jax.block_until_ready(result)
                state.raise_if_failed()
            return result
        except Exception:
            if state is not None:
                state.raise_if_failed()
            raise


class _ProgramStore:
    """The compiled JAX programs that one owner opens to run user or route code.

    An owner, such as a kernel that a joint maps over the draws of its
    inputs, keeps one store in its transient memo, so a copy of the owner with
    other bound values builds its own (:func:`_program_store_of`). The store
    keeps one ``jax.jit`` object for each body it runs, compiled under the
    *static settings* of the call that runs it (:class:`_StaticSettings`),
    which the body receives as a static keyword argument. They are therefore
    part of JAX's cache key, and a call under another ``num_samples`` compiles
    its own JAX program rather than reusing one traced under the first.

    Each JAX program is traced before it first runs, and falls back
    to the eager form by the rule of :func:`compiled_or_eager`. A dynamic
    first argument carries the running call's token into embedded callbacks;
    callback-free programs use zero and register no context. A call made
    inside a JAX transformation runs the body inline and opens no program.

    Code that the body runs can note things on the running call while it is
    traced, such as the number of draws of a Monte Carlo estimate. A trace
    happens only on a cold call, or when JAX traces again on its own, so the
    notes are kept with the store entry the trace belongs to and applied to
    the running call after each run of its JAX program, cold or warm:

    - **Capture.** While the body is traced, it notes into accumulators of
      its own, never into the running call's. A call that the body opens
      keeps its own notes, whatever the running call collects.
    - **One entry per trace.** An entry is keyed by the body's name, the
      static options, the static settings, and the trace cache key of the
      arguments, which is what JAX traces the body again for. A trace reads
      the same key from the arguments it receives, so its notes reach the
      entry of every call that reuses it, whether the store or JAX started
      the trace.
    - **One set of notes per entry.** Every trace of an entry captures the
      same notes, since a note may depend only on the entry's key, and a
      trace that captures other notes raises ``RuntimeError``.
    - **Apply.** After each run that succeeds, the entry's notes merge into
      the running call's accumulators once. A run that raises applies none,
      and an eager run notes into the running call directly. Inside a trace
      that ProbPipe does not open, such as a user's own ``jax.jit``, the JAX
      program is staged into that trace, and its notes apply when it is
      staged.

    The store keeps every entry and every ``jax.jit`` object for the owner's
    lifetime, with no bound.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jitted_bodies: dict[Hashable, Any] = {}
        self._entries: dict[Hashable, _StoreEntry] = {}
        self._decision_record: dict[Hashable, Any] = {}

    def run(
        self,
        body_name: Hashable,
        body: Callable[..., Any],
        eager: Callable[..., Any],
        *arguments: Any,
        static_options: Hashable = (),
    ) -> Any:
        """The body named *body_name* at *arguments*, compiled under the static settings in effect.

        Parameters
        ----------
        body_name : Hashable
            The name of the body within the store. The store keeps the first
            *body* given under a name, so the body must depend only on the
            owner and the name.
        body : Callable
            The computation, called as ``body(static_options, *arguments)``.
        eager : Callable
            The same computation, uncompiled, called as ``eager(*arguments)``.
        *arguments : Any
            The arguments, which the JAX program takes as data.
        static_options : Hashable, optional
            The options that change the JAX program, which it takes as a
            static argument, ``()`` by default.

        Returns
        -------
        Any
            The body's result.

        Raises
        ------
        Exception
            As :func:`compiled_or_eager` raises.
        RuntimeError
            If a trace of the JAX program notes other values on the running
            call than an earlier trace of it, which is a bug in the traced code.
        """
        if _jax_transformation_active():
            return body(static_options, *arguments)
        if current_call_context().embedded_calls is None:
            # A route may run as an evaluation point in another thread or
            # process. Its carried context deliberately has no live host state;
            # the local program owns one for the duration of this invocation.
            with _open_call_context(current_call_context(), entry_method_name="apply"):
                return self.run(body_name, body, eager, *arguments, static_options=static_options)
        static_settings = _static_settings_in_effect()
        trace_cache_key = _trace_cache_key(arguments)
        store_entry = self._entry((body_name, static_options, static_settings, trace_cache_key))
        ran_eagerly = False

        def run_eagerly(*eager_arguments: Any) -> Any:
            nonlocal ran_eagerly
            ran_eagerly = True
            return eager(*eager_arguments)

        body_result = compiled_or_eager(
            self._decision_record,
            (body_name, static_options, jax.tree.structure(arguments)),
            _ProgramForCall(self._jitted_body(body_name, body), store_entry),
            run_eagerly,
            *arguments,
            trace_cache_key=trace_cache_key,
            static_keywords={"static_options": static_options, "static_settings": static_settings},
        )
        if not ran_eagerly:
            self._entry(
                (body_name, static_options, static_settings, trace_cache_key)
            ).apply_trace_notes()
        return body_result

    def runs_eagerly(self, body_name: Hashable) -> bool:
        """Whether a trace of the body named *body_name* failed, so it runs uncompiled."""
        return any(
            decision_key[0] == body_name
            for decision_key in _eager_decision_keys(self._decision_record)
        )

    def _jitted_body(self, body_name: Hashable, body: Callable[..., Any]) -> Any:
        """The ``jax.jit`` object of the body named *body_name*, built from *body* the first time."""
        with self._lock:
            jitted_body = self._jitted_bodies.get(body_name)
            if jitted_body is None:
                jitted_body = self._jitted_bodies[body_name] = jax.jit(
                    self._body_in_static_settings(body_name, body),
                    static_argnames=("static_options", "static_settings"),
                )
            return jitted_body

    def _body_in_static_settings(
        self, body_name: Hashable, body: Callable[..., Any]
    ) -> Callable[..., Any]:
        """*body*, traced under the static settings it receives, keeping what it notes on its entry."""

        def run_body_in_static_settings(
            call_token: Any,
            *arguments: Any,
            static_options: Hashable,
            static_settings: _StaticSettings,
        ) -> Any:
            traced_context = current_call_context().with_static_settings(static_settings)
            token_scope = _TRACED_CALL_TOKEN.set(call_token)
            try:
                with (
                    _under_call_context(traced_context),
                    warnings.catch_warnings(record=True) as captured_warnings,
                ):
                    warnings.simplefilter("always")
                    body_output = body(static_options, *arguments)
            finally:
                _TRACED_CALL_TOKEN.reset(token_scope)
            self._entry(
                (body_name, static_options, static_settings, _trace_cache_key(arguments))
            ).add_trace_notes(
                _TraceNotes(
                    cast(_NumSamplesUsed, traced_context.num_samples_used).num_samples_values,
                    cast(_EvaluationLog, traced_context.evaluation_log).set_records,
                    tuple(
                        (warning.category, str(warning.message)) for warning in captured_warnings
                    ),
                )
            )
            return body_output

        return run_body_in_static_settings

    def _entry(self, entry_key: Hashable) -> _StoreEntry:
        """The entry under *entry_key*, created the first time."""
        with self._lock:
            store_entry = self._entries.get(entry_key)
            if store_entry is None:
                store_entry = self._entries[entry_key] = _StoreEntry()
            return store_entry


def _trace_cache_key(arguments: Any, *, ints_by_value: bool = False) -> Hashable | None:
    """What JAX traces a ``jax.jit`` object again for, read from its positional *arguments*.

    It is the tree structure of *arguments* and, for each leaf, the abstract
    value JAX traces it as, its shape, dtype, and weak type, as
    ``jax.typeof`` gives them. A Python number, a NumPy array, a JAX array,
    and a tracer of the same abstract value therefore give one key, as they
    give one trace. A leaf that is not an array, such as a string, is keyed
    by its type and value.

    Parameters
    ----------
    arguments : Any
        The arguments, concrete or traced.
    ints_by_value : bool, optional
        Whether an integer scalar is keyed by its value, as JAX keys a static
        argument; ``False`` by default, which keys it as data.

    Returns
    -------
    Hashable or None
        The key, or ``None`` when a leaf is neither an array nor hashable.
    """
    leaves, treedef = jax.tree.flatten(arguments)
    leaf_keys: list[Hashable] = []
    for leaf in leaves:
        if ints_by_value and isinstance(leaf, int | np.integer):
            leaf_keys.append((type(leaf), leaf))
            continue
        try:
            leaf_keys.append(jax.typeof(leaf))
        except TypeError:
            try:
                hash(leaf)
            except TypeError:
                return None
            leaf_keys.append((type(leaf), leaf))
    return treedef, tuple(leaf_keys)


def _eager_decision_keys(decision_record: Iterable[Hashable]) -> tuple[Hashable, ...]:
    """The decision keys that *decision_record*, kept by :func:`compiled_or_eager`, runs eagerly."""
    return tuple(
        recorded_key[1]
        for recorded_key in decision_record
        if isinstance(recorded_key, tuple) and recorded_key[0] == _RUNS_EAGERLY
    )


def _program_store_of(owner: Any) -> _ProgramStore:
    """The program store that *owner* keeps in its transient memo, created the first time.

    A transient memo is not carried across a copy or a pickle, so a copy of
    *owner* builds its own store.
    """
    memo = transient_memo(owner)
    program_store = memo.get(_PROGRAM_STORE)
    if program_store is None:
        program_store = memo.setdefault(_PROGRAM_STORE, _ProgramStore())
    return cast(_ProgramStore, program_store)


@dataclass(frozen=True)
class _StaticSettings:
    """The settings that a compiled JAX program ProbPipe opens is traced under, part of its cache key.

    They are what the code JAX traces may read and use to build the JAX
    program, so two calls that differ in them do not share one. A program
    store passes them to the traced body as a static keyword argument, and
    :meth:`~probpipe.core._call_context._CallContext.with_static_settings`
    installs them for the extent of the trace. Every part is a tuple of
    pairs, so the value is hashable.

    Attributes
    ----------
    resolved_controls : tuple of (str, Any)
        The running call's resolved ``max_evaluations`` and ``num_samples``,
        by name, empty outside every call.
    passed_controls : tuple of (str, Any) or None
        The controls an enclosing derived route passes to the calls its
        formula makes, by name, or ``None`` where none are passed.
    config_values : tuple of (str, tuple of (str, Any))
        The value in effect of each static ``config`` setting, by section name
        and then setting name.
    """

    resolved_controls: tuple[tuple[str, Any], ...]
    passed_controls: tuple[tuple[str, Any], ...] | None
    config_values: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]


def _static_settings_in_effect() -> _StaticSettings:
    """The static settings of the current call context."""
    call_context = current_call_context()
    passed_controls = call_context.passed_controls
    return _StaticSettings(
        resolved_controls=tuple(
            (control_name, call_context.resolved_controls[control_name])
            for control_name in _STATIC_CONTROL_NAMES
            if control_name in call_context.resolved_controls
        ),
        passed_controls=None if passed_controls is None else tuple(sorted(passed_controls.items())),
        config_values=_static_config_values_in_effect(),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

#: The tags of the two kinds of entry that :func:`compiled_or_eager` records.
_RUNS_EAGERLY = "runs eagerly"
_TRACED = "traced"

#: The key of an owner's program store in its transient memo.
_PROGRAM_STORE = "program store"


@dataclass(frozen=True)
class _TraceNotes:
    """What one trace of a JAX program noted on the running call.

    Attributes
    ----------
    num_samples_values : frozenset of int
        The number of draws of each Monte Carlo estimate noted.
    set_records : tuple of _EvaluationSetRecord
        The sets of evaluation points recorded, in the order they were
        recorded.
    warnings : tuple of (warning class, str), optional
        Warnings emitted while tracing, replayed once after every successful run.
    """

    num_samples_values: frozenset[int]
    set_records: tuple[_EvaluationSetRecord, ...]
    warnings: tuple[tuple[type[Warning], str], ...] = ()

    def is_empty(self) -> bool:
        """Whether the trace noted nothing."""
        return not self.num_samples_values and not self.set_records and not self.warnings


class _VaryingTraceNotesError(RuntimeError):
    """A trace of a stored JAX program noted other values on the running call than an earlier trace of it."""


class _StoreEntry:
    """One JAX program of a program store, with the notes its traces made on the running call.

    The first trace to finish sets the notes, and every later trace, one the
    store starts or one JAX starts on its own, must capture the same notes,
    in any order.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trace_notes: _TraceNotes | None = None
        self.has_embedded_calls = False

    def add_trace_notes(self, trace_notes: _TraceNotes) -> None:
        """Keep the notes of one trace, raising if they differ from an earlier trace's."""
        with self._lock:
            if self._trace_notes is None:
                self._trace_notes = trace_notes
                return
            kept_notes = self._trace_notes
        if trace_notes.num_samples_values != kept_notes.num_samples_values:
            raise _VaryingTraceNotesError(
                _varying_note_message(
                    "the numbers of Monte Carlo draws",
                    sorted(kept_notes.num_samples_values),
                    sorted(trace_notes.num_samples_values),
                )
            )
        if not _same_in_any_order(trace_notes.set_records, kept_notes.set_records):
            raise _VaryingTraceNotesError(
                _varying_note_message(
                    "the sets of evaluation points of",
                    sorted(set_record.origin for set_record in kept_notes.set_records),
                    sorted(set_record.origin for set_record in trace_notes.set_records),
                )
            )
        # Nested JAX caches can suppress a warning on a later trace. Keep the
        # union for replay rather than treating cache-dependent emissions as
        # a change to the program's semantic notes.
        with self._lock:
            assert self._trace_notes is not None
            self._trace_notes = replace(
                self._trace_notes,
                warnings=tuple(dict.fromkeys((*self._trace_notes.warnings, *trace_notes.warnings))),
            )

    def apply_trace_notes(self) -> None:
        """Merge the notes of the entry's traces into the current call context's accumulators once."""
        trace_notes = self._trace_notes
        if trace_notes is None or trace_notes.is_empty():
            return
        _merge_noted_into(
            current_call_context(), trace_notes.num_samples_values, trace_notes.set_records
        )
        for category, message in trace_notes.warnings:
            warnings.warn(message, category, stacklevel=3)


def _same_in_any_order(
    first_records: Iterable[_EvaluationSetRecord], second_records: Iterable[_EvaluationSetRecord]
) -> bool:
    """Whether two sequences of set records hold equal records, in any order."""
    unmatched_records = list(second_records)
    for set_record in first_records:
        if set_record not in unmatched_records:
            return False
        unmatched_records.remove(set_record)
    return not unmatched_records


def _varying_note_message(note_description: str, kept_value: Any, traced_value: Any) -> str:
    """The internal error of a trace that noted other values than an earlier trace of its JAX program."""
    return (
        f"internal error: a compiled JAX program noted {note_description} {traced_value!r} "
        f"on one trace but {kept_value!r} on an earlier one. Please report this"
    )
