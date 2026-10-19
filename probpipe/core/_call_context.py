"""The call context: the per-call state that the code a call runs reads, wherever it runs.

A ProbPipe call runs code in the calling thread and in the workers it
dispatches to. That code reads the same per-call state, such as the active
``config.using`` blocks and the controls a derived operation's identity passes
down, from one frozen value, the call context, held by one context variable.

Provides:
  - :class:`_CallContext`, the call context, and :func:`current_call_context`,
    which returns the one in effect;
  - :func:`_under_call_context`, the one function that installs a call context,
    and :func:`_open_call_context`, which runs one call in its own context;
  - :class:`_NumSamplesUsed`, the numbers of Monte Carlo draws a call's
    estimates took, with :func:`_collecting_num_samples_used` and
    :func:`_note_num_samples_used`;
  - :class:`_ContextBoundFunction` and :func:`_carrying_call_context`, which
    carry the call context into a worker.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Literal, cast

#: The entry methods of a ``Function`` that open a call context.
_EntryMethodName = Literal["__call__", "apply", "check"]

#: How a scope's numbers of Monte Carlo draws pass to the enclosing context's.
_MergeRule = Literal["when_parent_collects", "always"]

_NO_CONFIG_BLOCKS: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
_NO_CONTROLS: Mapping[str, Any] = MappingProxyType({})

#: The passed controls below a call made under a passed ``exact_only``, which
#: keep the ``exact_only`` alone.
_EXACT_ONLY_BELOW: Mapping[str, Any] = MappingProxyType(
    {"exact_only": True, "max_evaluations": None, "num_samples": None}
)


@dataclass(frozen=True)
class _CallContext:
    """The per-call state that the code a call runs reads, wherever it runs.

    One value of this class is in effect in each thread, held by the context
    variable ``_CALL_CONTEXT``; :func:`current_call_context` returns it.
    Outside every call and ``config`` block it is the default context, with
    every field at its default.

    Every piece of per-call state that crosses into a worker has exactly one
    field here as its home, and no other variable keeps a copy that code reads.

    Lifecycle:

    - Every call of a ``Function`` (``__call__``, ``apply``, ``check``) derives a
      child with :meth:`child_for_call`, and :func:`_open_call_context` installs
      it for the extent of the call and restores the previous context when the
      call ends, an error included.
    - A ``config`` block derives a changed copy with :meth:`with_config_block`,
      and the scope of a derived operation's identity one with
      :meth:`with_passed_controls`; neither opens a call.
    - A worker runs each piece of work it is given under the
      :meth:`evaluation_context` of the call that dispatched it, with a
      ``num_samples_used`` of the work's own, merged into the call's when the
      work succeeds.

    Transport:

    - Within a thread, the context variable carries it.
    - Into worker threads and Prefect tasks, the work function carries
      :meth:`evaluation_context` as data, in a :class:`_ContextBoundFunction`,
      which installs it before the work runs.
    - Into a worker process, the same carrier is pickled. Its read-only
      mappings travel as plain dicts, nested ones included, and are read-only
      again when unpickled, and its ``num_samples_used`` arrives new and empty.

    Invariants:

    - The dataclass is frozen, and its mappings are read-only. Its one mutable
      part, ``num_samples_used``, is a separate object that guards its contents
      with its own lock, so deriving or copying a context never copies it.
    - Installing a context and exiting its scope always restore the previous
      context, an error included; a scope that exits while another scope it
      encloses is still open raises.

    Attributes
    ----------
    config_blocks : Mapping[str, Mapping[str, Any]]
        The values of the active ``config.using`` blocks, by section name and
        then setting name, the innermost block's value winning.
    passed_controls : Mapping[str, Any] or None
        The controls that an enclosing derived operation's identity passes to
        the calls it makes directly: ``exact_only``, and ``max_evaluations`` and
        ``num_samples``, each ``None`` when it is not passed. ``None`` where no
        controls are passed. A child call keeps only a passed ``exact_only``.
    call_depth : int
        The number of calls open in this context: 0 outside every call, and 1
        in a root call.
    function_label : str or None
        The label of the running call's ``Function``, for messages; ``None``
        outside every call.
    resolved_controls : Mapping[str, Any]
        The running call's controls, resolved when it opened from its own
        controls, the passed controls, the ``config`` blocks, and the session.
        Code inside a call reads its controls here, never from the
        ``Function``. Empty outside every call.
    num_samples_used : _NumSamplesUsed or None
        The numbers of Monte Carlo draws the running call's estimates took,
        which its provenance reports as ``num_samples``. A ``__call__`` has its
        own, an ``apply`` shares its enclosing call's, and a ``check`` has none.
        ``None`` outside every call.
    """

    config_blocks: Mapping[str, Mapping[str, Any]] = _NO_CONFIG_BLOCKS
    passed_controls: Mapping[str, Any] | None = None
    call_depth: int = 0
    function_label: str | None = None
    resolved_controls: Mapping[str, Any] = _NO_CONTROLS
    num_samples_used: _NumSamplesUsed | None = None

    def with_config_block(self, config_values: Mapping[str, Mapping[str, Any]]) -> _CallContext:
        """This context under one more ``config`` block, which sets *config_values*.

        Parameters
        ----------
        config_values : Mapping of str to Mapping of str to Any
            The block's checked values, by section name and then setting name.

        Returns
        -------
        _CallContext
            A copy whose ``config_blocks`` holds the enclosing blocks' values
            merged with *config_values*, a value of the new block winning.
        """
        merged_blocks = dict(self.config_blocks)
        for section_name, section_values in config_values.items():
            merged_blocks[section_name] = MappingProxyType(
                {**self.config_blocks.get(section_name, {}), **section_values}
            )
        return replace(self, config_blocks=MappingProxyType(merged_blocks))

    def with_passed_controls(self, passed_controls: Mapping[str, Any] | None) -> _CallContext:
        """This context with *passed_controls* as the controls the calls made in it take.

        Parameters
        ----------
        passed_controls : Mapping of str to Any or None
            The passed ``exact_only``, ``max_evaluations``, and ``num_samples``,
            each count ``None`` when it is not passed, or ``None`` for none.

        Returns
        -------
        _CallContext
            A copy whose ``passed_controls`` is a read-only copy of
            *passed_controls*.
        """
        if passed_controls is not None:
            passed_controls = MappingProxyType(dict(passed_controls))
        return replace(self, passed_controls=passed_controls)

    def with_fresh_num_samples_used(self) -> _CallContext:
        """This context with a new, empty ``num_samples_used``.

        Returns
        -------
        _CallContext
            A copy whose ``num_samples_used`` is a new :class:`_NumSamplesUsed`.
        """
        return replace(self, num_samples_used=_NumSamplesUsed())

    def child_for_call(
        self,
        *,
        entry_method_name: _EntryMethodName,
        function_label: str,
        resolved_controls: Mapping[str, Any],
    ) -> _CallContext:
        """The context of a call opened in this context, which it derives and does not install.

        The child keeps this context's ``config`` blocks, keeps a passed
        ``exact_only`` and drops the passed counts, so the counts apply only to
        the calls an identity makes directly, and is one call deeper. A
        ``__call__`` and an ``apply`` keep this context's ``num_samples_used``,
        which :func:`_open_call_context` replaces with a new one for a
        ``__call__``, and a ``check`` has none, since it draws nothing.

        Parameters
        ----------
        entry_method_name : {"__call__", "apply", "check"}
            The ``Function`` method that opens the call.
        function_label : str
            The label of the called ``Function``.
        resolved_controls : Mapping of str to Any
            The call's controls, resolved from its own controls, the passed
            controls, the ``config`` blocks, and the session.

        Returns
        -------
        _CallContext
            The child context.
        """
        keeps_exact_only = self.passed_controls is not None and self.passed_controls["exact_only"]
        return replace(
            self,
            passed_controls=_EXACT_ONLY_BELOW if keeps_exact_only else None,
            call_depth=self.call_depth + 1,
            function_label=function_label,
            resolved_controls=MappingProxyType(dict(resolved_controls)),
            num_samples_used=None if entry_method_name == "check" else self.num_samples_used,
        )

    def evaluation_context(self) -> _CallContext:
        """The context that the work a call dispatches to a worker runs under.

        It is the call's context: every field it holds applies in the worker.

        Returns
        -------
        _CallContext
            The evaluation context.
        """
        return self

    def is_root_call(self) -> bool:
        """Whether this is the context of a call opened outside every other call."""
        return self.call_depth == 1

    def is_default(self) -> bool:
        """Whether every field holds its default, as outside every call and ``config`` block."""
        return self == _DEFAULT_CALL_CONTEXT

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the context with its read-only mappings as plain dicts.

        The mappings are read-only again when the context is unpickled, and its
        ``num_samples_used`` arrives new and empty, as that class pickles.
        """
        field_values = {
            context_field.name: _picklable(getattr(self, context_field.name))
            for context_field in fields(self)
        }
        return (_unpickled_call_context, (field_values,))


#: The context in effect outside every call and ``config`` block.
_DEFAULT_CALL_CONTEXT = _CallContext()

#: The call context in effect in the running thread.
_CALL_CONTEXT: ContextVar[_CallContext] = ContextVar(
    "probpipe_call_context", default=_DEFAULT_CALL_CONTEXT
)


def current_call_context() -> _CallContext:
    """The call context in effect in the running thread: the default context outside every call and block."""
    return _CALL_CONTEXT.get()


@contextmanager
def _under_call_context(call_context: _CallContext) -> Iterator[_CallContext]:
    """A scope in which *call_context* is the call context in effect.

    It is the one function that installs a call context. On exit, an error
    included, the context in effect before the scope applies again.

    Parameters
    ----------
    call_context : _CallContext
        The context to install.

    Yields
    ------
    _CallContext
        *call_context*.

    Raises
    ------
    RuntimeError
        On exit, if the context in effect is not *call_context*: a scope opened
        inside this one, such as a ``config`` block that a generator holds open,
        is still open. The scope then restores nothing, so it never discards
        the context of the scope still open.
    """
    reset_token = _CALL_CONTEXT.set(call_context)
    try:
        yield call_context
    finally:
        context_in_effect = _CALL_CONTEXT.get()
        if context_in_effect is not call_context:
            raise RuntimeError(_out_of_order_exit_message(call_context, context_in_effect))
        _CALL_CONTEXT.reset(reset_token)


@contextmanager
def _open_call_context(
    child_context: _CallContext, *, entry_method_name: _EntryMethodName
) -> Iterator[None]:
    """A scope that runs one call under *child_context*, which :meth:`_CallContext.child_for_call` derived.

    A ``__call__`` notes its Monte Carlo draws in a ``num_samples_used`` of its
    own, which passes to the enclosing call's when the call succeeds and the
    enclosing call collects calls, as a call realized by a derived operation's
    identity does. An ``apply`` notes its draws in the enclosing call's, and a
    ``check`` notes none.

    Parameters
    ----------
    child_context : _CallContext
        The call's context.
    entry_method_name : {"__call__", "apply", "check"}
        The ``Function`` method that opens the call.
    """
    collecting = (
        _collecting_num_samples_used("when_parent_collects")
        if entry_method_name == "__call__"
        else nullcontext()
    )
    with _under_call_context(child_context), collecting:
        yield


class _NumSamplesUsed:
    """The numbers of Monte Carlo draws that the estimates of one call took, which its provenance reports as ``num_samples``.

    A route that estimates a value from draws notes their number here, through
    :func:`_note_num_samples_used`. It differs from the ``num_samples``
    control, which says how many draws an estimate should take: a call that ran
    no Monte Carlo estimate notes none, and the calls an identity makes may use
    their own counts.

    Only :meth:`_CallContext.with_fresh_num_samples_used` creates one, and only
    :func:`_collecting_num_samples_used` merges one into another, when its
    scope ends without an error. The numbers are guarded by a lock, so worker
    threads may note into one at once. A pickled one arrives new and empty, so
    what a worker process notes does not reach the call.

    Attributes
    ----------
    collects_calls : bool
        Whether the calls made directly within the call merge their numbers
        into this one, which holds for a call realized by a derived operation's
        identity, so its provenance reports the draws of the calls it made. The
        engine sets it when it selects the call's route, and a scope inside the
        call takes the call's value when it opens.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._num_samples_values: set[int] = set()
        self.collects_calls = False

    @property
    def num_samples_values(self) -> frozenset[int]:
        """The number of draws of each Monte Carlo estimate noted so far."""
        with self._lock:
            return frozenset(self._num_samples_values)

    @property
    def provenance_num_samples(self) -> int | tuple[int, ...] | None:
        """The ``num_samples`` that the call's provenance reports.

        It is the one number noted, the sorted numbers when several were noted,
        and ``None`` when no Monte Carlo estimate was noted.
        """
        num_samples_values = self.num_samples_values
        if not num_samples_values:
            return None
        if len(num_samples_values) == 1:
            return next(iter(num_samples_values))
        return tuple(sorted(num_samples_values))

    def note(self, num_samples: int) -> None:
        """Note that one Monte Carlo estimate took *num_samples* draws."""
        with self._lock:
            self._num_samples_values.add(num_samples)

    def merge_from(self, scope_num_samples_used: _NumSamplesUsed) -> None:
        """Add the numbers that *scope_num_samples_used* noted to this one's."""
        noted_in_scope = scope_num_samples_used.num_samples_values
        with self._lock:
            self._num_samples_values |= noted_in_scope

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a new, empty value, with a lock of its own."""
        return (_NumSamplesUsed, ())


@contextmanager
def _collecting_num_samples_used(merge_rule: _MergeRule) -> Iterator[_NumSamplesUsed]:
    """A scope whose Monte Carlo estimates note their draws in a new ``num_samples_used`` of its own.

    When the scope ends without an error, its numbers merge into the
    ``num_samples_used`` of the context it was opened in, by *merge_rule*:

    - ``"when_parent_collects"``, for a call: only when the enclosing call
      collects calls;
    - ``"always"``, for a scope inside a call, such as the work one worker runs
      or one block of a summary computed component by component. Such a scope
      collects the calls made directly within it when its call does, so their
      numbers reach the call through the scope.

    Parameters
    ----------
    merge_rule : {"when_parent_collects", "always"}
        When the scope's numbers merge into the enclosing ones.

    Yields
    ------
    _NumSamplesUsed
        The scope's ``num_samples_used``, empty at the start.
    """
    enclosing_context = current_call_context()
    collecting_context = enclosing_context.with_fresh_num_samples_used()
    scope_num_samples_used = cast(_NumSamplesUsed, collecting_context.num_samples_used)
    enclosing_num_samples_used = enclosing_context.num_samples_used
    if merge_rule == "always" and enclosing_num_samples_used is not None:
        scope_num_samples_used.collects_calls = enclosing_num_samples_used.collects_calls
    with _under_call_context(collecting_context):
        yield scope_num_samples_used
    if enclosing_num_samples_used is not None and (
        merge_rule == "always" or enclosing_num_samples_used.collects_calls
    ):
        enclosing_num_samples_used.merge_from(scope_num_samples_used)


def _note_num_samples_used(num_samples: int) -> None:
    """Note that one Monte Carlo estimate of the running call took *num_samples* draws.

    Outside every call, and inside a ``check``, nothing is noted.
    """
    num_samples_used = current_call_context().num_samples_used
    if num_samples_used is not None:
        num_samples_used.note(num_samples)


class _ContextBoundFunction:
    """A work function bound to the call context it runs under, in whatever worker runs it.

    Each call of the wrapper runs one piece of work: it installs the bound
    context with a new ``num_samples_used`` for that work alone, and merges
    the work's numbers into the context's ``num_samples_used`` when the work
    succeeds, so a failed piece of work leaves no numbers behind. The wrapper
    pickles wherever the work function does, for a worker process.

    Parameters
    ----------
    work_function : callable
        The work function.
    call_context : _CallContext
        The evaluation context of the call that dispatches *work_function*.
    """

    def __init__(self, work_function: Callable[..., Any], call_context: _CallContext) -> None:
        self.work_function = work_function
        self.call_context = call_context

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        """*work_function*'s result, computed under the bound call context.

        ``self`` is positional-only, so a work function's own parameter named
        ``self`` binds as a keyword argument.
        """
        with _under_call_context(self.call_context), _collecting_num_samples_used("always"):
            return self.work_function(*args, **kwargs)


def _carrying_call_context(work_function: Callable[..., Any]) -> Callable[..., Any]:
    """*work_function*, run under the call context in effect, in whatever worker runs it.

    A worker thread starts without the calling thread's context, and a worker
    process without the calling process's, so without the carrier a body run in
    a worker would read the session's settings in place of an active ``config``
    block, lose an ``exact_only`` an identity passes, and note its Monte Carlo
    draws nowhere. In the default context *work_function* is returned as it is.
    """
    current_context = current_call_context()
    if current_context.is_default():
        return work_function
    return _ContextBoundFunction(work_function, current_context.evaluation_context())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _ReadOnlyMappingContents(dict):
    """The contents of a read-only mapping, as a pickled call context holds them."""


def _picklable(value: Any) -> Any:
    """*value* with every read-only mapping in it, nested ones included, as plain dict contents."""
    if isinstance(value, MappingProxyType):
        return _ReadOnlyMappingContents({key: _picklable(entry) for key, entry in value.items()})
    if type(value) is dict:
        return {key: _picklable(entry) for key, entry in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_picklable(entry) for entry in value)
    return value


def _restored_from_picklable(value: Any) -> Any:
    """*value*, which :func:`_picklable` made, with its read-only mappings restored."""
    if isinstance(value, _ReadOnlyMappingContents):
        return MappingProxyType(
            {key: _restored_from_picklable(entry) for key, entry in value.items()}
        )
    if type(value) is dict:
        return {key: _restored_from_picklable(entry) for key, entry in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_restored_from_picklable(entry) for entry in value)
    return value


def _unpickled_call_context(field_values: Mapping[str, Any]) -> _CallContext:
    """The call context that :meth:`_CallContext.__reduce__` pickled as *field_values*."""
    return _CallContext(
        **{name: _restored_from_picklable(value) for name, value in field_values.items()}
    )


def _described_scope(call_context: _CallContext) -> str:
    """*call_context* in a user's words, for an error message."""
    sections = ", ".join(f"config.{name}" for name in call_context.config_blocks)
    if call_context.call_depth == 0:
        return f"a block of {sections}" if sections else "the settings outside every call"
    return f"the call of {call_context.function_label!r}"


def _out_of_order_exit_message(
    exiting_context: _CallContext, context_in_effect: _CallContext
) -> str:
    """The message for a scope that exits while a scope it encloses is still open."""
    return (
        f"cannot close {_described_scope(exiting_context)} while "
        f"{_described_scope(context_in_effect)}, opened inside it, is still open. Close each "
        "with block, and each generator that holds one open, before the scope around it ends"
    )
