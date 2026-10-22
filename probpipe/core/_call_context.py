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
  - :class:`_ExecutionChoice`, the executor a call chose and its options;
  - :class:`_NumSamplesUsed` and :class:`_EvaluationLog`, the call's two
    accumulators: the numbers of Monte Carlo draws its estimates took, and the
    sets of evaluation points it ran, with :func:`_collecting_call_accumulators`
    and :func:`_note_num_samples_used`;
  - :func:`_jax_settings_snapshot`, the JAX settings a root call holds fixed;
  - :meth:`_CallContext.with_static_settings`, the context a compiled JAX
    program's body is traced under.

An evaluation point carries the call context into the worker that runs it,
as its chunk, an ``EvaluationChunk``, holds its call's evaluation context.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import cloudpickle
import jax

if TYPE_CHECKING:
    from ._compilation import _StaticSettings

#: The entry methods of a ``Function`` that open a call context.
_EntryMethodName = Literal["__call__", "apply", "check"]

#: How a scope's accumulators pass to the enclosing context's.
_MergeRule = Literal["when_parent_collects", "always"]

#: Where a call's executor choice came from: named for the callable, passed down
#: by an enclosing call, or chosen by automatic selection.
_ExecutionChoiceSource = Literal["own", "inherited", "automatic"]

_NO_CONFIG_BLOCKS: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
_NO_CONTROLS: Mapping[str, Any] = MappingProxyType({})
_NO_SESSION_CONFIG_VALUES: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
_NO_JAX_SETTINGS: Mapping[str, Any] = MappingProxyType({})

#: The JAX settings that change what a computation returns, which a root call
#: holds fixed and a worker process applies before it unpickles any work.
_SHIPPED_JAX_SETTING_NAMES = (
    "jax_enable_x64",
    "jax_default_prng_impl",
    "jax_threefry_partitionable",
    "jax_default_matmul_precision",
)

#: The ``config`` section that is never sent to a worker process, since a call
#: there runs without Prefect orchestration and a started task runner does not
#: pickle.
_UNSHIPPED_CONFIG_SECTION = "prefect"

#: The passed controls below a call made under a passed ``exact_only``, which
#: keep the ``exact_only`` alone.
_EXACT_ONLY_BELOW: Mapping[str, Any] = MappingProxyType(
    {"exact_only": True, "max_evaluations": None, "num_samples": None}
)


@dataclass(frozen=True)
class _ExecutionChoice:
    """The executor a call chose for the sets of evaluation points it runs, and its options.

    A call resolves it once, when it opens, by the rules of V.9: the
    ``executor`` and ``executor_options`` the called ``Function`` sets itself,
    taken as one unit; otherwise the choice of the enclosing call, marked
    inherited, when that names an executor; otherwise automatic selection.
    An evaluation point runs under automatic selection, so a call made inside
    one chooses afresh.

    Attributes
    ----------
    executor_name : str or None
        The registered name of the chosen executor; ``None`` for automatic
        selection, which then receives *executor_options*.
    executor_options : Mapping[str, Any]
        The options that belong to the chosen executor, or to the one
        automatic selection picks when *executor_name* is ``None``.
    source : {"own", "inherited", "automatic"}
        ``"own"`` when the choice was named for the callable, on its own
        ``Function`` or on the call whose own callable it is; ``"inherited"``
        when an enclosing call passed it down; ``"automatic"`` when nobody
        named one.
    applies_only_to : object or None
        The ``Function`` the choice was named for, on its decorator or its
        ``using``, or ``None`` when the choice was named on an operation and
        covers every opaque callable of its call. A set of evaluation points
        that a ``Function``'s call builds is of that ``Function``'s callable,
        so an executor that cannot run it raises rather than falling back; a
        callback that finds the choice through a call token compares the
        ``Function`` it runs with this one.
    named_on : str or None
        The label of the call that named the executor, which a warning about
        an inherited executor names; ``None`` for automatic selection.
    """

    executor_name: str | None = None
    executor_options: Mapping[str, Any] = _NO_CONTROLS
    source: _ExecutionChoiceSource = "automatic"
    applies_only_to: Any = None
    named_on: str | None = None

    def names_executor(self) -> bool:
        """Whether the choice names an executor rather than leaving it to automatic selection."""
        return self.executor_name is not None

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the choice with its options as a plain dict, read-only again when unpickled."""
        return (
            _unpickled_execution_choice,
            (
                self.executor_name,
                _picklable(self.executor_options),
                self.source,
                self.applies_only_to,
                self.named_on,
            ),
        )


#: The choice of a call that names no executor and no options.
_AUTOMATIC_SELECTION = _ExecutionChoice()


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
    - An evaluation point, a piece of work a call runs, runs under the
      :meth:`evaluation_context` of its call, with accumulators of its own,
      merged into the call's when the evaluation point succeeds. There the
      executor choice is automatic selection, so a call made inside the
      evaluation point chooses afresh, and a setting that no block sets reads
      ``session_config_values``, the session's values when the root call
      opened, so a session assignment made while the call runs reaches none
      of its evaluation points.

    Transport:

    - Within a thread, the context variable carries it.
    - Into worker threads and Prefect tasks, the work carries
      :meth:`evaluation_context` as data and installs it before it runs.
    - Into a worker process, the work is pickled with it. Its read-only
      mappings travel as plain dicts, nested ones included, and are read-only
      again when unpickled; its accumulators arrive new and empty; the
      ``prefect`` section of its ``config`` blocks stays behind; and a
      ``config`` value that does not pickle travels as a placeholder that
      raises ``UnshippableSettingError`` only when the worker reads it.

    Reading settings while JAX traces: the body of a compiled JAX program that
    a program store opens is traced under :meth:`with_static_settings`, so it
    reads the static settings that are part of the JAX program's cache key,
    and an inert ``config`` setting, which changes no computed value, as it
    reads it anywhere. What it notes goes into accumulators of the trace's
    own, which the store applies to the running call after each run.

    Invariants:

    - The dataclass is frozen, and its mappings are read-only. Its mutable
      parts, ``num_samples_used`` and ``evaluation_log``, are separate objects
      that guard their contents with their own locks, so deriving or copying a
      context never copies them.
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
        ``Function``. It holds neither ``executor`` nor ``executor_options``,
        whose home is ``execution_choice``. Empty outside every call.
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
        Whether this is the context that a call's evaluation points run under,
        :meth:`evaluation_context`, or a context derived from one.
    execution_choice : _ExecutionChoice
        The executor the running call chose and its options, the only place
        code reads them. Automatic selection outside every call and inside an
        evaluation point.
    evaluation_log : _EvaluationLog or None
        The sets of evaluation points the running call ran, for its
        provenance. Shared with ``num_samples_used`` by the rules of
        :func:`_collecting_call_accumulators`; ``None`` outside every call and
        for a ``check``.
    jax_settings : Mapping[str, Any]
        The JAX settings that change what a computation returns, by name, as
        they were when the root call opened; a child call keeps its parent's,
        and a worker process applies them before it unpickles any work. Empty
        outside every call.
    """

    config_blocks: Mapping[str, Mapping[str, Any]] = _NO_CONFIG_BLOCKS
    passed_controls: Mapping[str, Any] | None = None
    call_depth: int = 0
    function_label: str | None = None
    resolved_controls: Mapping[str, Any] = _NO_CONTROLS
    num_samples_used: _NumSamplesUsed | None = None
    session_config_values: Mapping[str, Mapping[str, Any]] = _NO_SESSION_CONFIG_VALUES
    in_evaluation_point: bool = False
    execution_choice: _ExecutionChoice = _AUTOMATIC_SELECTION
    evaluation_log: _EvaluationLog | None = None
    jax_settings: Mapping[str, Any] = _NO_JAX_SETTINGS

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

    def with_fresh_call_accumulators(self, *, collects_calls: bool = False) -> _CallContext:
        """This context with a new, empty ``num_samples_used`` and ``evaluation_log``.

        Parameters
        ----------
        collects_calls : bool, optional
            Whether the calls made directly within the new scope merge into
            its accumulators; ``False`` by default.

        Returns
        -------
        _CallContext
            A copy whose accumulators are a new :class:`_NumSamplesUsed` and a
            new :class:`_EvaluationLog`.
        """
        return replace(
            self,
            num_samples_used=_NumSamplesUsed(collects_calls=collects_calls),
            evaluation_log=_EvaluationLog(),
        )

    def with_static_settings(self, static_settings: _StaticSettings) -> _CallContext:
        """The context that the body of a compiled JAX program is traced under, with *static_settings*.

        The static ``config`` values apply as the innermost block, so a block
        that the traced code opens applies over them. The passed controls and
        the running call's ``max_evaluations`` and ``num_samples`` are those of
        *static_settings*. The executor choice is automatic selection, since the
        call that runs the JAX program decides it when it runs. The accumulators
        are new and empty, so what the trace notes stays out of the running
        call's until the program store applies it. They collect no calls, so
        a call that the traced code opens keeps its own notes, whichever call
        runs the JAX program.

        Parameters
        ----------
        static_settings : _StaticSettings
            The settings the JAX program is compiled under.

        Returns
        -------
        _CallContext
            The context of the trace.
        """
        traced_context = self.with_config_block(
            {
                section_name: dict(section_values)
                for section_name, section_values in static_settings.config_values
            }
        ).with_passed_controls(
            None
            if static_settings.passed_controls is None
            else dict(static_settings.passed_controls)
        )
        return replace(
            traced_context,
            resolved_controls=MappingProxyType(
                {**self.resolved_controls, **dict(static_settings.resolved_controls)}
            ),
            execution_choice=_AUTOMATIC_SELECTION,
        ).with_fresh_call_accumulators()

    def collects_calls(self) -> bool:
        """Whether the calls made directly within this context merge into its accumulators."""
        return self.num_samples_used is not None and self.num_samples_used.collects_calls

    def child_for_call(
        self,
        *,
        entry_method_name: _EntryMethodName,
        function_label: str,
        resolved_controls: Mapping[str, Any],
        execution_choice: _ExecutionChoice = _AUTOMATIC_SELECTION,
        session_config_values: Mapping[str, Mapping[str, Any]] | None = None,
        jax_settings: Mapping[str, Any] | None = None,
    ) -> _CallContext:
        """The context of a call opened in this context, which it derives and does not install.

        The child keeps this context's ``config`` blocks, keeps a passed
        ``exact_only`` and drops the passed counts, so the counts apply only to
        the calls a formula makes directly, and is one call deeper. A
        ``__call__`` and an ``apply`` keep this context's accumulators, which
        :func:`_open_call_context` replaces with new ones for a ``__call__``,
        and a ``check`` has none, since it draws and runs nothing. A root call
        takes *session_config_values* and *jax_settings*, and every other call
        keeps this context's.

        Parameters
        ----------
        entry_method_name : {"__call__", "apply", "check"}
            The ``Function`` method that opens the call.
        function_label : str
            The label of the called ``Function``.
        resolved_controls : Mapping of str to Any
            The call's controls, resolved from its own controls, the passed
            controls, the ``config`` blocks, and the session, without
            ``executor`` and ``executor_options``.
        execution_choice : _ExecutionChoice, optional
            The executor the call chose and its options; automatic selection by
            default.
        session_config_values : Mapping of str to Mapping of str to Any, optional
            The session's values of every section but ``prefect``, by section
            name and then setting name, given only for a root call.
        jax_settings : Mapping of str to Any, optional
            The JAX settings :func:`_jax_settings_snapshot` took, given only for
            a root call.

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
        runs_nothing = entry_method_name == "check"
        return replace(
            self,
            passed_controls=_EXACT_ONLY_BELOW if keeps_exact_only else None,
            call_depth=self.call_depth + 1,
            function_label=function_label,
            resolved_controls=MappingProxyType(dict(resolved_controls)),
            num_samples_used=None if runs_nothing else self.num_samples_used,
            evaluation_log=None if runs_nothing else self.evaluation_log,
            execution_choice=execution_choice,
            session_config_values=(
                self.session_config_values
                if session_config_values is None
                else session_config_values
            ),
            jax_settings=(
                self.jax_settings if jax_settings is None else MappingProxyType(dict(jax_settings))
            ),
        )

    def evaluation_context(self) -> _CallContext:
        """The context that the call's evaluation points run under.

        It is the call's context marked ``in_evaluation_point``, with automatic
        selection in place of the call's executor choice, so a call made inside
        an evaluation point chooses its executor afresh, and a setting that no
        block sets reads ``session_config_values`` there, whatever worker runs
        the evaluation point and whenever it runs.

        Returns
        -------
        _CallContext
            The evaluation context.
        """
        return replace(self, in_evaluation_point=True, execution_choice=_AUTOMATIC_SELECTION)

    def is_root_call(self) -> bool:
        """Whether this is the context of a call opened outside every other call."""
        return self.call_depth == 1

    def is_default(self) -> bool:
        """Whether every field holds its default, as outside every call and ``config`` block."""
        return self == _DEFAULT_CALL_CONTEXT

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the context for a worker process, with its read-only mappings as plain dicts.

        The mappings are read-only again when the context is unpickled, and its
        accumulators arrive new and empty, as their classes pickle. The
        ``prefect`` section of its ``config`` blocks stays behind, and a
        ``config`` value that does not pickle travels as an
        :class:`_UnshippableSetting`, which raises only when it is read.
        """
        shipped_context = replace(
            self,
            config_blocks=_shippable_config_values(self.config_blocks),
            session_config_values=_shippable_config_values(self.session_config_values),
        )
        field_values = {
            context_field.name: _picklable(getattr(shipped_context, context_field.name))
            for context_field in fields(shipped_context)
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

    A ``__call__`` notes its Monte Carlo draws and the sets of evaluation
    points it runs in accumulators of its own, which pass to the enclosing
    call's when the call succeeds and the enclosing call collects calls, as a
    derived route's call does. An ``apply`` notes into the enclosing call's,
    and a ``check`` notes nothing.

    Parameters
    ----------
    child_context : _CallContext
        The call's context.
    entry_method_name : {"__call__", "apply", "check"}
        The ``Function`` method that opens the call.
    """
    collecting = (
        _collecting_call_accumulators("when_parent_collects")
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

    Only :meth:`_CallContext.with_fresh_call_accumulators` creates one, and
    only :func:`_merge_noted_into` adds what one scope noted to another's,
    when the scope ends without an error. The
    numbers are guarded by a lock, so worker threads may note into one at once.
    A pickled one arrives empty, with a lock of its own and the same
    ``collects_calls``, and what a worker process notes into it comes back
    with the outcomes of its evaluation points.

    Parameters
    ----------
    collects_calls : bool, optional
        Its initial ``collects_calls``; ``False`` by default.

    Attributes
    ----------
    collects_calls : bool
        Whether the calls made directly within the call merge their numbers
        into this one, which holds for a derived route's call, so its
        provenance reports the draws of the calls it made. The
        engine sets it when it selects the call's route, and a scope inside the
        call takes the call's value when it opens.
    """

    def __init__(self, *, collects_calls: bool = False) -> None:
        self._lock = threading.Lock()
        self._num_samples_values: set[int] = set()
        self.collects_calls = collects_calls

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

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a new, empty value with the same ``collects_calls`` and a lock of its own."""
        return (_empty_num_samples_used, (self.collects_calls,))


@dataclass(frozen=True)
class _EvaluationSetRecord:
    """One set of evaluation points that a call ran, as its provenance reports it.

    Attributes
    ----------
    origin : str
        The callable or the site the evaluation points belong to.
    set_ordinal : int
        Which of the call's sets of evaluation points of *origin* it is,
        counting from 0.
    executor_name : str
        The executor that ran it.
    executor_options : Mapping[str, Any]
        The options that executor received.
    num_evaluation_points : int
        The number of its evaluation points.
    failed_point_indices : tuple of int
        The point indices of the evaluation points that failed, in order.
    """

    origin: str
    set_ordinal: int
    executor_name: str
    executor_options: Mapping[str, Any]
    num_evaluation_points: int
    failed_point_indices: tuple[int, ...]


class _EvaluationLog:
    """The sets of evaluation points that one call ran, which its provenance reports.

    The executor dispatcher records each set it runs here, and numbers the
    sets of each callable in the order the call asks for them, so an
    evaluation point's identity can name its set. It also remembers which of
    the call's warnings were given, so each is given once per call.

    Only :meth:`_CallContext.with_fresh_call_accumulators` creates one, and it
    merges as ``num_samples_used`` does (:func:`_merge_noted_into`).
    Its contents are guarded by a lock. A pickled one arrives new and empty,
    and what a worker process records into it comes back with the outcomes of
    its evaluation points.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set_records: list[_EvaluationSetRecord] = []
        self._set_counts_by_origin: dict[str, int] = {}
        self._set_count = 0
        self._warnings_given: set[Any] = set()

    @property
    def set_records(self) -> tuple[_EvaluationSetRecord, ...]:
        """The sets recorded so far, in the order they were recorded."""
        with self._lock:
            return tuple(self._set_records)

    def next_set_numbers(self, origin: str) -> tuple[int, int]:
        """The numbers of the next set of evaluation points of *origin* that the call asks for.

        Parameters
        ----------
        origin : str
            The callable or the site the set belongs to.

        Returns
        -------
        tuple of (int, int)
            The set's ordinal among the call's sets of *origin*, which its
            evaluation points' identities name, and its position among all the
            call's sets, which keeps its evaluation points' workflow scopes
            apart from those of the call's other sets; both count from 0.
        """
        with self._lock:
            set_ordinal = self._set_counts_by_origin.get(origin, 0)
            self._set_counts_by_origin[origin] = set_ordinal + 1
            set_position = self._set_count
            self._set_count += 1
            return set_ordinal, set_position

    def record(self, set_record: _EvaluationSetRecord) -> None:
        """Record that the call ran the set *set_record* describes."""
        with self._lock:
            self._set_records.append(set_record)

    def is_first_warning(self, warning_key: Any) -> bool:
        """Whether the call has not given the warning *warning_key* yet, which this then records."""
        with self._lock:
            if warning_key in self._warnings_given:
                return False
            self._warnings_given.add(warning_key)
            return True

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a new, empty log, with a lock of its own."""
        return (_EvaluationLog, ())


@contextmanager
def _collecting_call_accumulators(merge_rule: _MergeRule) -> Iterator[_CallContext]:
    """A scope that notes its Monte Carlo draws and its sets of evaluation points in accumulators of its own.

    When the scope ends without an error, its accumulators merge into those of
    the context it was opened in, by *merge_rule*:

    - ``"when_parent_collects"``, for a call: only when the enclosing call
      collects calls;
    - ``"always"``, for a scope inside a call, such as one block of a summary
      computed output by output. Such a scope collects the calls made directly
      within it when its call does, so their accumulators reach the call
      through the scope.

    Parameters
    ----------
    merge_rule : {"when_parent_collects", "always"}
        When the scope's accumulators merge into the enclosing ones.

    Yields
    ------
    _CallContext
        The scope's context, whose ``num_samples_used`` and ``evaluation_log``
        are empty at the start.
    """
    enclosing_context = current_call_context()
    enclosing_num_samples_used = enclosing_context.num_samples_used
    collecting_context = enclosing_context.with_fresh_call_accumulators(
        collects_calls=merge_rule == "always" and enclosing_context.collects_calls()
    )
    with _under_call_context(collecting_context):
        yield collecting_context
    if enclosing_num_samples_used is not None and (
        merge_rule == "always" or enclosing_num_samples_used.collects_calls
    ):
        _merge_noted_into(
            enclosing_context,
            cast(_NumSamplesUsed, collecting_context.num_samples_used).num_samples_values,
            cast(_EvaluationLog, collecting_context.evaluation_log).set_records,
        )


def _merge_noted_into(
    call_context: _CallContext,
    num_samples_values: Iterable[int],
    set_records: Iterable[_EvaluationSetRecord],
) -> None:
    """Add the numbers of draws and the sets of evaluation points a scope noted to *call_context*'s accumulators.

    A context without accumulators, as a ``check``'s, takes none.
    """
    if call_context.num_samples_used is not None:
        for num_samples in num_samples_values:
            call_context.num_samples_used.note(num_samples)
    if call_context.evaluation_log is not None:
        for set_record in set_records:
            call_context.evaluation_log.record(set_record)


def _note_num_samples_used(num_samples: int) -> None:
    """Note that one Monte Carlo estimate of the running call took *num_samples* draws.

    Outside every call, and inside a ``check``, nothing is noted.
    """
    num_samples_used = current_call_context().num_samples_used
    if num_samples_used is not None:
        num_samples_used.note(num_samples)


def _jax_settings_snapshot() -> Mapping[str, Any]:
    """The JAX settings that change what a computation returns, as they read now, by name.

    A root call takes this snapshot when it opens, and a worker process applies
    it before it unpickles any work, since a spawned interpreter starts with
    JAX's defaults.
    """
    return MappingProxyType(
        {
            setting_name: getattr(jax.config, setting_name)
            for setting_name in _SHIPPED_JAX_SETTING_NAMES
        }
    )


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


def _unpickled_execution_choice(
    executor_name: str | None,
    executor_options: Any,
    source: _ExecutionChoiceSource,
    applies_only_to: Any,
    named_on: str | None,
) -> _ExecutionChoice:
    """The executor choice that :meth:`_ExecutionChoice.__reduce__` pickled."""
    return _ExecutionChoice(
        executor_name,
        _restored_from_picklable(executor_options),
        source,
        applies_only_to,
        named_on,
    )


def _empty_num_samples_used(collects_calls: bool) -> _NumSamplesUsed:
    """A new, empty :class:`_NumSamplesUsed` that collects calls when *collects_calls* holds."""
    return _NumSamplesUsed(collects_calls=collects_calls)


@dataclass(frozen=True)
class _UnshippableSetting:
    """What a worker process receives in place of a ``config`` value that does not pickle.

    ``config`` raises ``UnshippableSettingError`` when code in the worker reads
    it, and only then, so a value the worker never reads costs nothing.

    Attributes
    ----------
    section_name : str
        The ``config`` section of the setting.
    setting_name : str
        The setting's name.
    value_type_name : str
        The class name of the value that did not pickle.
    pickling_error : str
        The error pickling raised, as its class name and message.
    """

    section_name: str
    setting_name: str
    value_type_name: str
    pickling_error: str


def _shippable_config_values(
    config_values: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    """*config_values*, by section and setting, as a worker process receives them.

    The ``prefect`` section is left out, and a value that does not pickle is
    replaced by an :class:`_UnshippableSetting`.
    """
    shipped_values: dict[str, Mapping[str, Any]] = {}
    for section_name, section_values in config_values.items():
        if section_name == _UNSHIPPED_CONFIG_SECTION:
            continue
        shipped_values[section_name] = MappingProxyType(
            {
                setting_name: _shippable_setting(section_name, setting_name, setting_value)
                for setting_name, setting_value in section_values.items()
            }
        )
    return MappingProxyType(shipped_values)


def _shippable_setting(section_name: str, setting_name: str, setting_value: Any) -> Any:
    """*setting_value*, or an :class:`_UnshippableSetting` in its place when it does not pickle."""
    try:
        cloudpickle.dumps(setting_value)
    except Exception as pickling_error:
        return _UnshippableSetting(
            section_name,
            setting_name,
            type(setting_value).__name__,
            f"{type(pickling_error).__name__}: {pickling_error}",
        )
    return setting_value


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
