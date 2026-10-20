"""The call context: the per-call state that the code a call runs reads, wherever it runs.

A ProbPipe call runs code in the calling thread and in the workers it
dispatches to. That code reads the same per-call state, such as the active
``config.using`` blocks and the controls a derived route passes down, from
one frozen value, the call context, held by one context variable.

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
_NO_SESSION_CONFIG_VALUES: Mapping[str, Mapping[str, Any]] = MappingProxyType({})

#: The passed controls below a call made under a passed ``exact_only``, which
#: keep the ``exact_only`` alone.
_EXACT_ONLY_BELOW: Mapping[str, Any] = MappingProxyType(
    {"exact_only": True, "max_evaluations": None, "num_samples": None}
)


@dataclass(frozen=True)
class _CallContext:
    """The per-call state that the code a call runs reads, wherever it runs.

    One value of this class is in effect in each thread, held by the innermost
    open scope that installed it, which the context variable
    ``_INSTALLED_SCOPE`` holds; :func:`current_call_context` returns it.
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
      and the scope of a derived route one with
      :meth:`with_passed_controls`; neither opens a call.
    - A worker runs each piece of work it is given under the
      :meth:`evaluation_context` of the call that dispatched it, with a
      ``num_samples_used`` of the work's own, merged into the call's when the
      work succeeds. There a setting that no block sets reads
      ``session_config_values``, the session's values when the root call
      opened, so a session assignment made while the call runs reaches none
      of its work.

    Transport:

    - Within a thread, the context variable carries it.
    - Into worker threads and Prefect tasks, the work function carries
      :meth:`evaluation_context` as data, in a :class:`_ContextBoundFunction`,
      which installs it before the work runs.
    - Into a worker process, the same carrier is pickled. Its read-only
      mappings travel as plain dicts, nested ones included, and are read-only
      again when unpickled, so ``session_config_values`` travels with it, and
      its ``num_samples_used`` arrives new and empty.

    Invariants:

    - The dataclass is frozen, and its mappings are read-only. Its one mutable
      part, ``num_samples_used``, is a separate object that guards its contents
      with its own lock, so deriving or copying a context never copies it.
    - Exiting a scope always restores the context of the innermost scope still
      open below it, an error included, so no closed scope's state stays
      behind; a scope that exits while a scope opened inside it is still open
      raises after the context is restored (:func:`_under_call_context`).

    Attributes
    ----------
    config_blocks : Mapping[str, Mapping[str, Any]]
        The values of the active ``config.using`` blocks, by section name and
        then setting name, the innermost block's value winning.
    passed_controls : Mapping[str, Any] or None
        The controls that an enclosing derived route passes to the calls
        its formula makes directly: ``exact_only``, and ``max_evaluations`` and
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
    session_config_values : Mapping[str, Mapping[str, Any]]
        The session's values of every ``config`` section but ``prefect``, by
        section name and then setting name, as they were when the root call
        opened. A child call keeps its parent's. Inside an evaluation point a
        setting that no block sets reads its value here, in place of the
        session's current value. Empty outside every call.
    in_evaluation_point : bool
        Whether this is the context that a call's dispatched work runs under,
        :meth:`evaluation_context`, or a context derived from one.
    """

    config_blocks: Mapping[str, Mapping[str, Any]] = _NO_CONFIG_BLOCKS
    passed_controls: Mapping[str, Any] | None = None
    call_depth: int = 0
    function_label: str | None = None
    resolved_controls: Mapping[str, Any] = _NO_CONTROLS
    num_samples_used: _NumSamplesUsed | None = None
    session_config_values: Mapping[str, Mapping[str, Any]] = _NO_SESSION_CONFIG_VALUES
    in_evaluation_point: bool = False

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
        session_config_values: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> _CallContext:
        """The context of a call opened in this context, which it derives and does not install.

        The child keeps this context's ``config`` blocks, keeps a passed
        ``exact_only`` and drops the passed counts, so the counts apply only to
        the calls a formula makes directly, and is one call deeper. A
        ``__call__`` and an ``apply`` keep this context's ``num_samples_used``,
        which :func:`_open_call_context` replaces with a new one for a
        ``__call__``, and a ``check`` has none, since it draws nothing. A root
        call takes *session_config_values*, and every other call keeps this
        context's.

        Parameters
        ----------
        entry_method_name : {"__call__", "apply", "check"}
            The ``Function`` method that opens the call.
        function_label : str
            The label of the called ``Function``.
        resolved_controls : Mapping of str to Any
            The call's controls, resolved from its own controls, the passed
            controls, the ``config`` blocks, and the session.
        session_config_values : Mapping of str to Mapping of str to Any, optional
            The session's values of every section but ``prefect``, by section
            name and then setting name, given only for a root call.

        Returns
        -------
        _CallContext
            The child context.
        """
        keeps_exact_only = self.passed_controls is not None and self.passed_controls["exact_only"]
        if session_config_values is not None:
            session_config_values = MappingProxyType(
                {
                    section_name: MappingProxyType(dict(section_values))
                    for section_name, section_values in session_config_values.items()
                }
            )
        return replace(
            self,
            passed_controls=_EXACT_ONLY_BELOW if keeps_exact_only else None,
            call_depth=self.call_depth + 1,
            function_label=function_label,
            resolved_controls=MappingProxyType(dict(resolved_controls)),
            num_samples_used=None if entry_method_name == "check" else self.num_samples_used,
            session_config_values=(
                self.session_config_values
                if session_config_values is None
                else session_config_values
            ),
        )

    def evaluation_context(self) -> _CallContext:
        """The context that the work a call dispatches to a worker runs under.

        It is the call's context marked ``in_evaluation_point``, so a setting
        that no block sets reads ``session_config_values`` there, whatever
        worker runs the work and whenever it runs.

        Returns
        -------
        _CallContext
            The evaluation context.
        """
        return replace(self, in_evaluation_point=True)

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

#: The innermost open scope that installed a call context in the running
#: thread, or ``None`` outside every scope.
_INSTALLED_SCOPE: ContextVar[_InstalledScope | None] = ContextVar(
    "probpipe_call_context", default=None
)


def current_call_context() -> _CallContext:
    """The call context in effect in the running thread: the default context outside every call and block."""
    return _context_of(_innermost_open_scope(_INSTALLED_SCOPE.get()))


@contextmanager
def _under_call_context(call_context: _CallContext) -> Iterator[_CallContext]:
    """A scope in which *call_context* is the call context in effect.

    It is the one function that installs a call context. On exit, an error
    included, the context of the innermost scope that is still open below it
    applies again, so no state of a closed scope stays behind.

    A scope that exits while a scope opened inside it is still open, as when a
    generator holds a ``config`` block open, exits out of order. The context is
    restored first, by one of two rules, and then the exit raises:

    - when a call is still open inside the scope, the code that closes the
      scope runs inside that call, so the call keeps its context, the values
      of the closed scope included, and when the call ends the context of the
      innermost scope still open below the closed one applies;
    - otherwise the scopes still open inside it are closed with it, and the
      context of the innermost scope still open below it applies. A scope
      closed this way later exits without effect.

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
        On exit, after the context is restored, if a scope opened inside this
        one is still open, naming both scopes.
    """
    installed_scope = _InstalledScope(call_context, _innermost_open_scope(_INSTALLED_SCOPE.get()))
    _INSTALLED_SCOPE.set(installed_scope)
    try:
        yield call_context
    finally:
        _close_installed_scope(installed_scope)


@contextmanager
def _open_call_context(
    child_context: _CallContext, *, entry_method_name: _EntryMethodName
) -> Iterator[None]:
    """A scope that runs one call under *child_context*, which :meth:`_CallContext.child_for_call` derived.

    A ``__call__`` notes its Monte Carlo draws in a ``num_samples_used`` of its
    own, which passes to the enclosing call's when the call succeeds and the
    enclosing call collects calls, as a derived route's call does. An
    ``apply`` notes its draws in the enclosing call's, and a ``check`` notes
    none.

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
    no Monte Carlo estimate notes none, and the calls a formula makes may use
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
        into this one, which holds for a derived route's call, so its
        provenance reports the draws of the calls it made. The
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
    block, lose an ``exact_only`` a derived route passes, and note its Monte Carlo
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


def _picklable(context_field_value: Any) -> Any:
    """*context_field_value* with every read-only mapping in it, nested ones included, as plain dict contents."""
    if isinstance(context_field_value, MappingProxyType):
        return _ReadOnlyMappingContents(
            {key: _picklable(nested_value) for key, nested_value in context_field_value.items()}
        )
    if type(context_field_value) is dict:
        return {key: _picklable(nested_value) for key, nested_value in context_field_value.items()}
    if type(context_field_value) in (list, tuple):
        return type(context_field_value)(
            _picklable(nested_value) for nested_value in context_field_value
        )
    return context_field_value


def _restored_from_picklable(context_field_value: Any) -> Any:
    """*context_field_value*, which :func:`_picklable` made, with its read-only mappings restored."""
    if isinstance(context_field_value, _ReadOnlyMappingContents):
        return MappingProxyType(
            {
                key: _restored_from_picklable(nested_value)
                for key, nested_value in context_field_value.items()
            }
        )
    if type(context_field_value) is dict:
        return {
            key: _restored_from_picklable(nested_value)
            for key, nested_value in context_field_value.items()
        }
    if type(context_field_value) in (list, tuple):
        return type(context_field_value)(
            _restored_from_picklable(nested_value) for nested_value in context_field_value
        )
    return context_field_value


def _unpickled_call_context(field_values: Mapping[str, Any]) -> _CallContext:
    """The call context that :meth:`_CallContext.__reduce__` pickled as *field_values*."""
    return _CallContext(
        **{
            field_name: _restored_from_picklable(context_field_value)
            for field_name, context_field_value in field_values.items()
        }
    )


class _InstalledScope:
    """One open scope that installed a call context, linked to the scope it was opened in.

    The scopes of a thread form a chain from the innermost scope, which
    ``_INSTALLED_SCOPE`` holds, to the outermost. A scope that is closed out of
    order is marked closed, and reading or restoring the context skips it, in
    every context that holds the chain, a copied one included.

    Parameters
    ----------
    call_context : _CallContext
        The context the scope installed.
    enclosing_scope : _InstalledScope or None
        The scope in effect when this one opened, or ``None`` outside every
        scope.
    """

    __slots__ = ("call_context", "enclosing_scope", "is_closed")

    def __init__(self, call_context: _CallContext, enclosing_scope: _InstalledScope | None) -> None:
        self.call_context = call_context
        self.enclosing_scope = enclosing_scope
        self.is_closed = False

    def opens_call(self) -> bool:
        """Whether the scope is the scope of a call: its context is one call deeper."""
        return self.call_context.call_depth > _context_of(self.enclosing_scope).call_depth


def _context_of(installed_scope: _InstalledScope | None) -> _CallContext:
    """The context *installed_scope* installed, or the default context for ``None``."""
    return _DEFAULT_CALL_CONTEXT if installed_scope is None else installed_scope.call_context


def _innermost_open_scope(installed_scope: _InstalledScope | None) -> _InstalledScope | None:
    """*installed_scope*, or the innermost scope below it that is still open."""
    while installed_scope is not None and installed_scope.is_closed:
        installed_scope = installed_scope.enclosing_scope
    return installed_scope


def _close_installed_scope(closing_scope: _InstalledScope) -> None:
    """Close *closing_scope* and restore the context, raising after it if the scope closed out of order."""
    if closing_scope.is_closed:
        # An enclosing scope closed it out of order already, and raised then.
        return
    closing_scope.is_closed = True
    innermost_scope = _INSTALLED_SCOPE.get()
    scopes_still_open_inside: list[_InstalledScope] = []
    scope = innermost_scope
    while scope is not None and scope is not closing_scope:
        if not scope.is_closed:
            scopes_still_open_inside.append(scope)
        scope = scope.enclosing_scope
    if scope is None:
        # The scope is not in this thread's chain, which a carrier's copy of a
        # context can cause; nothing of this thread's chain is its to restore.
        return
    if not scopes_still_open_inside:
        _INSTALLED_SCOPE.set(_innermost_open_scope(closing_scope.enclosing_scope))
        return
    open_calls_inside = [scope for scope in scopes_still_open_inside if scope.opens_call()]
    if open_calls_inside:
        # The scope is closed from inside a call opened within it, which keeps
        # its context until it ends.
        raise RuntimeError(
            _closed_around_an_open_call_message(closing_scope, open_calls_inside[-1])
        )
    for scope in scopes_still_open_inside:
        scope.is_closed = True
    _INSTALLED_SCOPE.set(_innermost_open_scope(closing_scope.enclosing_scope))
    raise RuntimeError(
        _closed_with_open_scopes_message(closing_scope, scopes_still_open_inside[-1])
    )


def _described_scope(installed_scope: _InstalledScope) -> str:
    """The scope *installed_scope* in a user's words, for an error message."""
    call_context = installed_scope.call_context
    enclosing_context = _context_of(installed_scope.enclosing_scope)
    # A block derives a new mapping of the blocks, and every other scope keeps it.
    if (
        installed_scope.opens_call()
        or call_context.config_blocks is enclosing_context.config_blocks
    ):
        if call_context.function_label is None:
            return "a scope of the settings"
        return f"the call of {call_context.function_label!r}"
    added_sections = [
        f"config.{section_name}"
        for section_name, section_values in call_context.config_blocks.items()
        if enclosing_context.config_blocks.get(section_name) is not section_values
    ]
    return f"a block of {', '.join(added_sections)}"


def _closed_around_an_open_call_message(
    closing_scope: _InstalledScope, open_call_scope: _InstalledScope
) -> str:
    """The message for a scope closed while a call opened inside it still runs."""
    return (
        f"cannot close {_described_scope(closing_scope)} while "
        f"{_described_scope(open_call_scope)}, opened inside it, is still running, so its values "
        "apply until that call ends. Close each with block, and each generator that holds one "
        "open, in the reverse order of opening"
    )


def _closed_with_open_scopes_message(
    closing_scope: _InstalledScope, open_scope: _InstalledScope
) -> str:
    """The message for a scope that ends while a scope opened inside it is still open."""
    return (
        f"{_described_scope(closing_scope)} ended while {_described_scope(open_scope)}, opened "
        "inside it, was still open, so that scope was closed with it. Close each with block, and "
        "each generator that holds one open, before the scope around it ends"
    )
