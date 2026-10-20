"""The global settings of ProbPipe, grouped in the sections of ``config``.

``config`` is the one ``Config``, and its sections are:

- ``config.sampling``, a ``SamplingConfig``: the default evaluation and draw
  counts of a call;
- ``config.notation``, a ``NotationConfig``: how many nested levels a label or
  a notation shows;
- ``config.provenance``, a ``ProvenanceConfig``: how much lineage history a
  provenance record keeps, a ``ProvenanceMode``;
- ``config.prefect``, a ``PrefectConfig``: how a ``Function`` dispatches its
  work, a ``WorkflowKind``, and the Prefect task runner.

Users import from the top-level package::

    import probpipe
    from probpipe import WorkflowKind, ProvenanceMode

    probpipe.config.prefect.workflow_kind = WorkflowKind.TASK
    probpipe.config.provenance.mode = ProvenanceMode.FULL
    probpipe.config.notation.max_depth = 12
    with probpipe.config.sampling.using(num_samples=4096):
        ...
"""

from __future__ import annotations

import os

__all__ = [
    "Config",
    "NotationConfig",
    "PrefectConfig",
    "ProvenanceConfig",
    "ProvenanceMode",
    "SamplingConfig",
    "WorkflowKind",
    "config",
]
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Self

from .._messages import unknown_names
from ._call_context import _under_call_context, current_call_context
from ._expression import _STORED_DEPTH

# ---------------------------------------------------------------------------
# Sections and their blocks
# ---------------------------------------------------------------------------


class _Section:
    """A section of ``config``: named settings, each with a session value and a check.

    Reading a setting returns the value that the innermost active ``using``
    block sets for it, and otherwise the session's value: inside the work a
    call dispatches, the session's value when the outermost call opened.
    Assigning a setting checks the value and sets the session's value, which
    an active block that sets the setting overrides until it exits.
    """

    #: The section's name in ``config``, which the blocks are keyed by.
    _name: ClassVar[str]

    #: Whether the work a call dispatches reads the section's session values
    #: as they were when the outermost call opened, rather than as they are.
    _is_in_session_snapshot: ClassVar[bool] = True

    #: The check of each setting, by name: it takes the value and the setting's
    #: name, and returns the value or raises as an assignment raises.
    _checks: ClassVar[Mapping[str, Callable[[Any, str], Any]]]

    def _value_in_effect(self, setting_name: str) -> Any:
        """The value of *setting_name* that the innermost block sets, or else the session's value.

        Inside an evaluation point the session's value is the one in the
        snapshot that the root call took, when it holds the section.
        """
        call_context = current_call_context()
        section_block = call_context.config_blocks.get(self._name)
        if section_block is not None and setting_name in section_block:
            return section_block[setting_name]
        if call_context.in_evaluation_point:
            section_snapshot = call_context.session_config_values.get(self._name)
            if section_snapshot is not None:
                return section_snapshot[setting_name]
        return getattr(self, f"_{setting_name}")

    def _session_values(self) -> dict[str, Any]:
        """The session's value of each setting of the section, by setting name."""
        return {setting_name: getattr(self, f"_{setting_name}") for setting_name in self._checks}

    def _checked_block_values(
        self, block_settings: Mapping[str, Any], caller: str
    ) -> dict[str, Any]:
        """*block_settings* checked as an assignment checks them, for the block *caller* opens.

        Parameters
        ----------
        block_settings : Mapping of str to Any
            The temporary values, keyed by setting name.
        caller : str
            The call that opens the block, which the error messages open with.

        Returns
        -------
        dict of str to Any
            The checked values, keyed by setting name.

        Raises
        ------
        TypeError
            If a name is not a setting of the section, naming the settings, or
            as the setting's assignment raises.
        ValueError
            As the setting's assignment raises.
        """
        unknown_setting_names = [
            setting_name for setting_name in block_settings if setting_name not in self._checks
        ]
        if unknown_setting_names:
            raise TypeError(
                f"{caller}: {unknown_names('setting', unknown_setting_names, self._checks)}"
            )
        return {
            setting_name: self._checks[setting_name](setting_value, setting_name)
            for setting_name, setting_value in block_settings.items()
        }

    def using(self, **settings: Any) -> AbstractContextManager[Self]:
        """A block in which *settings* replace the section's values, as ``with section.using(...):``.

        The block's values apply to every read inside it, by the calls a
        ``Function`` dispatches to worker threads or Prefect workers included,
        and the previous values apply again when it exits, an error included.
        Inside a workflow run, a block cannot change the provenance mode.
        Blocks nest, and an inner block's value wins. Blocks close in the
        reverse order of opening, and :class:`Config` states what happens when
        a generator that holds a block open breaks that order. A view's own control,
        which ``f.using(...)`` sets, wins over a block, which wins over the
        session's value. ``f.using(...)`` returns a reusable configured copy of ``f``,
        while ``using`` here returns a context manager.

        Parameters
        ----------
        **settings : Any
            The temporary values, keyed by the section's setting names.

        Returns
        -------
        AbstractContextManager
            The block, which yields the section.

        Raises
        ------
        TypeError
            If a name is not a setting of the section, naming the settings, or
            a value has the wrong type, as an assignment raises.
        ValueError
            If a value is inadmissible, as an assignment raises.
        RuntimeError
            On entry, if the block sets a provenance mode other than that of the
            workflow run open in this thread, naming the run's mode. On exit,
            if the block closes out of order, as :class:`Config` states.
        """
        caller = f"config.{self._name}.using()"
        checked_block_values = self._checked_block_values(settings, caller)
        return _config_block({self._name: checked_block_values}, caller=caller, yields=self)


# ---------------------------------------------------------------------------
# WorkflowKind enum
# ---------------------------------------------------------------------------


class WorkflowKind(Enum):
    """Orchestration mode for ``Function`` instances.

    Members
    -------
    DEFAULT
        Inherit from global config; the shipped global default is
        ``OFF`` unless overridden via ``PROBPIPE_WORKFLOW_KIND`` or
        explicit assignment to ``config.prefect.workflow_kind``. At
        the per-instance level, ``DEFAULT`` means "inherit from
        global config".
    OFF
        No Prefect orchestration.  Plain Python execution.
    TASK
        Wrap execution in a Prefect task (via ``task.map()``).
        Raises ``ImportError`` if Prefect is not installed.
    FLOW
        Wrap execution in a Prefect flow.
        Raises ``ImportError`` if Prefect is not installed.
    """

    DEFAULT = "default"
    OFF = "off"
    TASK = "task"
    FLOW = "flow"


# ---------------------------------------------------------------------------
# Environment-variable override
# ---------------------------------------------------------------------------

_WORKFLOW_KIND_ENV_VAR = "PROBPIPE_WORKFLOW_KIND"


def _initial_workflow_kind() -> WorkflowKind:
    """Resolve the initial ``workflow_kind`` from the environment.

    Reads ``PROBPIPE_WORKFLOW_KIND`` (case-insensitive). Unset →
    ``OFF``. Unknown values raise ``ValueError`` so deployment-config
    typos surface loudly rather than silently falling back to ``OFF``.
    """
    raw = os.environ.get(_WORKFLOW_KIND_ENV_VAR)
    if raw is None:
        return WorkflowKind.OFF
    try:
        return WorkflowKind(raw.lower())
    except ValueError as e:
        valid = ", ".join(repr(k.value) for k in WorkflowKind)
        raise ValueError(
            f"{_WORKFLOW_KIND_ENV_VAR}={raw!r} is not a valid WorkflowKind. "
            f"Expected one of: {valid}."
        ) from e


# ---------------------------------------------------------------------------
# Task-runner auto-detection
# ---------------------------------------------------------------------------


def _auto_detect_task_runner() -> Any:
    """Return a task runner based on installed packages, or ``None``.

    Probe order: Ray > Dask > ``None`` (Prefect built-in default).
    """
    try:
        from prefect_ray import RayTaskRunner

        return RayTaskRunner()
    except ImportError:
        pass
    try:
        from prefect_dask import DaskTaskRunner

        return DaskTaskRunner()
    except ImportError:
        pass
    return None


def _checked_workflow_kind(value: Any, source: str) -> WorkflowKind:
    """*value* as a ``WorkflowKind`` member, which the setting *source* takes."""
    if not isinstance(value, WorkflowKind):
        raise TypeError(f"{source} must be a WorkflowKind enum member, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# PrefectConfig: config.prefect
# ---------------------------------------------------------------------------


class PrefectConfig(_Section):
    """Global Prefect orchestration settings.

    Attributes
    ----------
    workflow_kind : WorkflowKind
        Default orchestration mode for all ``Function`` instances
        that do not override their own.  Initial value is ``OFF`` unless
        the ``PROBPIPE_WORKFLOW_KIND`` environment variable is set, in
        which case its value (``off`` / ``task`` / ``flow`` / ``default``,
        case-insensitive) is used. Production callers wanting Prefect
        orchestration opt in explicitly::

            import probpipe
            probpipe.config.prefect.workflow_kind = probpipe.WorkflowKind.TASK
    task_runner : object or None
        Prefect task runner instance (e.g., ``RayTaskRunner()``).
        ``None`` means auto-detect: use ``RayTaskRunner`` if
        ``prefect-ray`` is installed, then ``DaskTaskRunner`` if
        ``prefect-dask`` is installed, otherwise Prefect's built-in
        default.
    """

    _name = "prefect"
    # A task runner need not pickle, so the work a call dispatches reads the
    # section's current values rather than a snapshot that travels with it.
    _is_in_session_snapshot = False
    _checks = MappingProxyType(
        {"workflow_kind": _checked_workflow_kind, "task_runner": lambda value, source: value}
    )

    def __init__(self) -> None:
        self.reset()

    # -- Public API ---------------------------------------------------------

    def reset(self) -> None:
        """Restore the session's settings to defaults (re-reading the env var).

        An active ``using`` block that sets a setting still applies until it exits.
        """
        self._workflow_kind: WorkflowKind = _initial_workflow_kind()
        self._task_runner: Any = None

    @property
    def workflow_kind(self) -> WorkflowKind:
        """Current global orchestration mode."""
        return self._value_in_effect("workflow_kind")

    @workflow_kind.setter
    def workflow_kind(self, value: WorkflowKind) -> None:
        self._workflow_kind = _checked_workflow_kind(value, "workflow_kind")

    @property
    def task_runner(self) -> Any:
        """Explicit task runner, or ``None`` for auto-detection."""
        return self._value_in_effect("task_runner")

    @task_runner.setter
    def task_runner(self, value: Any) -> None:
        self._task_runner = value

    def resolve_task_runner(self) -> Any:
        """Return the effective task runner (explicit or auto-detected).

        Returns
        -------
        object or None
            A Prefect task runner instance, or ``None`` to use Prefect's
            built-in default.
        """
        runner = self.task_runner
        if runner is not None:
            return runner
        return _auto_detect_task_runner()


# ---------------------------------------------------------------------------
# ProvenanceMode enum
# ---------------------------------------------------------------------------


class ProvenanceMode(Enum):
    """Controls how much history is retained in provenance chains.

    Members
    -------
    FULL
        Store live references to parent Distribution / Record /
        RecordBatch objects.  The entire ancestry chain stays in memory
        as long as the final result is alive.  Good for debugging and
        small test workflows where full graph traversal is useful.
    LIGHTWEIGHT
        Store only lightweight :class:`~probpipe.core.provenance.ParentInfo`
        descriptors — type name, label, and an optional
        fingerprint plus its strength classification. Parent objects are free
        to be garbage-collected once a workflow step completes. This is the
        default and scales to larger workflows.
    OFF
        Attach no provenance at all.  Minimises overhead when lineage
        tracking is not needed.
    """

    FULL = "full"
    LIGHTWEIGHT = "lightweight"
    OFF = "off"


# ---------------------------------------------------------------------------
# Environment-variable override for ProvenanceMode
# ---------------------------------------------------------------------------

_PROVENANCE_MODE_ENV_VAR = "PROBPIPE_PROVENANCE_MODE"


def _initial_provenance_mode() -> ProvenanceMode:
    """Resolve the initial ``mode`` from the environment.

    Reads ``PROBPIPE_PROVENANCE_MODE`` (case-insensitive).  Unset →
    ``LIGHTWEIGHT``.  Unknown values raise ``ValueError`` so deployment-config
    typos surface loudly rather than silently falling back to ``LIGHTWEIGHT``.
    """
    raw = os.environ.get(_PROVENANCE_MODE_ENV_VAR)
    if raw is None:
        return ProvenanceMode.LIGHTWEIGHT
    try:
        return ProvenanceMode(raw.lower())
    except ValueError as e:
        valid = ", ".join(repr(m.value) for m in ProvenanceMode)
        raise ValueError(
            f"{_PROVENANCE_MODE_ENV_VAR}={raw!r} is not a valid ProvenanceMode. "
            f"Expected one of: {valid}."
        ) from e


def _checked_provenance_mode(value: Any, source: str) -> ProvenanceMode:
    """*value* as a ``ProvenanceMode`` member, which the setting *source* takes."""
    if not isinstance(value, ProvenanceMode):
        raise TypeError(
            f"{source} must be a ProvenanceMode enum member, got {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# ProvenanceConfig: config.provenance
# ---------------------------------------------------------------------------


class ProvenanceConfig(_Section):
    """Global provenance tracking settings.

    Controls how much lineage history ``Function`` retains when
    assembling provenance for each result.  Set once at application startup::

        import probpipe
        from probpipe import ProvenanceMode

        probpipe.config.provenance.mode = ProvenanceMode.FULL  # for debugging

    The initial mode can also be set via the ``PROBPIPE_PROVENANCE_MODE``
    environment variable (``full``, ``lightweight``, or ``off``,
    case-insensitive).

    The mode is fixed for a workflow run. The outermost ``workflow_run`` takes
    the mode in effect when it is entered, and a call outside every
    ``workflow_run`` takes it when the call opens. Inside the run, a ``using``
    block, an assignment, or a ``reset`` that would set another mode raises
    ``RuntimeError``, naming the run's mode; one that sets the run's own mode
    is accepted::

        with probpipe.workflow_run(seed=0):
            with probpipe.config.provenance.using(mode=ProvenanceMode.OFF):  # raises
                ...

    Set the mode before the run opens instead.
    """

    _name = "provenance"
    _checks = MappingProxyType({"mode": _checked_provenance_mode})

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore the session's mode to its default (re-reading the env var).

        An active ``using`` block that sets the mode still applies until it exits.

        Raises
        ------
        RuntimeError
            Inside a workflow run, if the default differs from the run's mode.
        """
        initial_mode = _initial_provenance_mode()
        _refuse_changing_the_run_provenance_mode(initial_mode, "config.provenance.reset()")
        self._mode: ProvenanceMode = initial_mode

    @property
    def mode(self) -> ProvenanceMode:
        """Current global provenance tracking mode.

        Raises
        ------
        TypeError
            On assignment of a value that is not a ``ProvenanceMode``.
        RuntimeError
            On assignment inside a workflow run of a mode other than the run's.
        """
        return self._value_in_effect("mode")

    @mode.setter
    def mode(self, value: ProvenanceMode) -> None:
        checked_mode = _checked_provenance_mode(value, "mode")
        _refuse_changing_the_run_provenance_mode(checked_mode, "config.provenance.mode")
        self._mode = checked_mode


# ---------------------------------------------------------------------------
# NotationConfig: config.notation
# ---------------------------------------------------------------------------

_NOTATION_MAX_DEPTH_ENV_VAR = "PROBPIPE_NOTATION_MAX_DEPTH"

#: The number of nested levels a label or a notation shows by default.
_DEFAULT_MAX_DEPTH = 8


def _invalid_max_depth(source: str, got: str) -> str:
    """The message that the depth *source* names is not a valid one, having got *got*."""
    return f"{source} must be an integer from 1 to {_STORED_DEPTH}, got {got}"


def _checked_max_depth(value: Any, source: str) -> int:
    """*value* as a number of nested levels, an integer from 1 to ``_STORED_DEPTH``.

    A stored expression keeps at most ``_STORED_DEPTH`` levels, so a rendering
    of more levels would show no more.

    Parameters
    ----------
    value : Any
        The value assigned.
    source : str
        The name of the setting, which the error messages open with.

    Returns
    -------
    int
        *value*.

    Raises
    ------
    TypeError
        If *value* is not an integer, a bool included.
    ValueError
        If *value* is less than 1 or greater than ``_STORED_DEPTH``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(_invalid_max_depth(source, type(value).__name__))
    if not 1 <= value <= _STORED_DEPTH:
        raise ValueError(_invalid_max_depth(source, repr(value)))
    return value


def _initial_max_depth() -> int:
    """Resolve the initial ``max_depth`` from the environment.

    Reads ``PROBPIPE_NOTATION_MAX_DEPTH``. Unset gives 8, and a value that is
    not an integer from 1 to ``_STORED_DEPTH`` raises ``ValueError``, so a typo
    in a deployment's configuration surfaces rather than falling back to the
    default.
    """
    raw = os.environ.get(_NOTATION_MAX_DEPTH_ENV_VAR)
    if raw is None:
        return _DEFAULT_MAX_DEPTH
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(_invalid_max_depth(_NOTATION_MAX_DEPTH_ENV_VAR, repr(raw))) from None
    return _checked_max_depth(value, _NOTATION_MAX_DEPTH_ENV_VAR)


class NotationConfig(_Section):
    """Global settings of how labels and notations render.

    A tracked term's label and the notation of a law, a kernel, or a function
    are renderings of the expression the term carries, which nests one level
    for each value or law it was computed from, as
    ``𝔼[f(beta ~ model; y)]`` nests four. A rendering shows at most
    :attr:`max_depth` levels::

        import probpipe

        probpipe.config.notation.max_depth = 12

    The setting is global to the process. A term's label is rendered once,
    when the term is built, at the setting in effect then.

    The initial depth can also be set by the ``PROBPIPE_NOTATION_MAX_DEPTH``
    environment variable. The depth is an integer from 1 to 64, the number of
    levels a stored expression keeps, since a rendering of more levels would
    show no more.

    Raises
    ------
    ValueError
        On construction, which reads ``PROBPIPE_NOTATION_MAX_DEPTH``, if the
        variable is set to anything but an integer from 1 to 64. The section
        ``config.notation`` is constructed when ``probpipe`` is imported, so the
        import raises.
    """

    _name = "notation"
    _checks = MappingProxyType(
        {"max_depth": lambda value, source: _checked_max_depth(value, source)}
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore the session's settings to their defaults, reading ``PROBPIPE_NOTATION_MAX_DEPTH`` again.

        An active ``using`` block that sets ``max_depth`` still applies until it
        exits.

        Raises
        ------
        ValueError
            If ``PROBPIPE_NOTATION_MAX_DEPTH`` is set to anything but an integer
            from 1 to 64.
        """
        self._max_depth: int = _initial_max_depth()

    @property
    def max_depth(self) -> int:
        """The number of nested levels a label or a notation shows, 8 by default.

        A part nested deeper renders as its label, the name of a law or a
        function, or as ``…`` for a value, and ``str()``, ``repr()``, and
        ``notation`` warn with a ``UserWarning`` when what they show has such
        a part. Every label the library derives from one law and a
        few operators on it nests at most 5 levels, as
        ``𝔼[f(beta ~ model; y)][sample=0] + 1`` does, so the default shows
        each of them in full, while a label derived through a long chain of
        operations, such as a loop that adds to a value, stays bounded.

        A term's label is rendered once, when the term is built, at the setting
        then, and the term keeps it. The notation of a law, a kernel, or a
        function is rendered each time it is shown, at the current setting. A
        stored expression keeps at most 64 levels. A term built over an operand
        that nests 64 levels stores the whole operand as its label or ``…``,
        which shows so at any setting, with a ``UserWarning`` that says no
        setting shows it. A loop that adds to a value therefore gives, every 63
        steps, a label that shows the last step alone, as ``… + 1``.

        Raises
        ------
        TypeError
            On assignment of a value that is not an integer, a bool included.
        ValueError
            On assignment of an integer less than 1 or greater than 64, the
            number of levels a stored expression keeps.
        """
        return self._value_in_effect("max_depth")

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._max_depth = _checked_max_depth(value, "max_depth")


# ---------------------------------------------------------------------------
# SamplingConfig: config.sampling
# ---------------------------------------------------------------------------

_MAX_EVALUATIONS_ENV_VAR = "PROBPIPE_SAMPLING_MAX_EVALUATIONS"
_NUM_SAMPLES_ENV_VAR = "PROBPIPE_SAMPLING_NUM_SAMPLES"

#: The maximum number of evaluations of a lifted function by default.
_DEFAULT_MAX_EVALUATIONS = 256

#: The number of Monte Carlo draws of a summary or an approximate conversion by default.
_DEFAULT_NUM_SAMPLES = 1024


def _checked_count(value: Any, source: str) -> int:
    """*value* as a count of evaluations or draws, a positive integer.

    Every check of the controls ``max_evaluations`` and ``num_samples``, and of
    their session defaults, words its message here.

    Parameters
    ----------
    value : Any
        The candidate count.
    source : str
        The name of the control or the setting, which the error messages open with.

    Returns
    -------
    int
        *value*.

    Raises
    ------
    TypeError
        If *value* is not an integer, a bool included.
    ValueError
        If *value* is less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{source} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    return value


def _initial_count(env_var: str, default: int) -> int:
    """Resolve the initial value of a count from the environment variable *env_var*.

    Unset gives *default*, and a value that is not a positive integer raises
    ``ValueError``, so a typo in a deployment's configuration raises at import.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{env_var} must be a positive integer, got {raw!r}") from None
    return _checked_count(value, env_var)


class SamplingConfig(_Section):
    """The session's defaults of the evaluation count of a lift and the draw count of an estimate.

    Every ``Function`` and operation has two count controls, which
    ``using`` sets for one view:

    1. ``max_evaluations``: the maximum number of times a function lifted over
       laws is evaluated, 256 by default.
    2. ``num_samples``: the number of Monte Carlo draws of a summary, such as a
       ``mean`` with no closed form, or of an approximate conversion, 1024 by
       default.

    A view that sets neither reads the defaults here when it runs, so a
    setting governs the functions constructed before it::

        import probpipe

        probpipe.config.sampling.max_evaluations = 1000
        probpipe.config.sampling.num_samples = 4096

    The settings are global to the process. Their initial values can also be
    set by the environment variables ``PROBPIPE_SAMPLING_MAX_EVALUATIONS`` and
    ``PROBPIPE_SAMPLING_NUM_SAMPLES``.

    Raises
    ------
    ValueError
        On construction, which reads the two environment variables, if either is
        set to anything but a positive integer. The singleton
        ``probpipe.config.sampling`` is constructed when ``probpipe`` is
        imported, so the import raises.
    """

    _name = "sampling"
    _checks = MappingProxyType({"max_evaluations": _checked_count, "num_samples": _checked_count})

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore both session settings to their defaults, reading their environment variables again.

        An active ``using`` block that sets a setting still applies until it exits.

        Raises
        ------
        ValueError
            If ``PROBPIPE_SAMPLING_MAX_EVALUATIONS`` or
            ``PROBPIPE_SAMPLING_NUM_SAMPLES`` is set to anything but a positive
            integer.
        """
        self._max_evaluations: int = _initial_count(
            _MAX_EVALUATIONS_ENV_VAR, _DEFAULT_MAX_EVALUATIONS
        )
        self._num_samples: int = _initial_count(_NUM_SAMPLES_ENV_VAR, _DEFAULT_NUM_SAMPLES)

    @property
    def max_evaluations(self) -> int:
        """The default of the ``max_evaluations`` control, 256 unless the environment sets another.

        :class:`~probpipe.Function` states how a lift spends the count.

        Raises
        ------
        TypeError
            On assignment of a value that is not an integer, a bool included.
        ValueError
            On assignment of an integer less than 1.
        """
        return self._value_in_effect("max_evaluations")

    @max_evaluations.setter
    def max_evaluations(self, value: int) -> None:
        self._max_evaluations = _checked_count(value, "max_evaluations")

    @property
    def num_samples(self) -> int:
        """The default of the ``num_samples`` control, 1024 unless the environment sets another.

        A converter that draws, such as moment matching, takes the converting
        call's ``num_samples`` unless its own ``num_samples`` option is given.

        Raises
        ------
        TypeError
            On assignment of a value that is not an integer, a bool included.
        ValueError
            On assignment of an integer less than 1.
        """
        return self._value_in_effect("num_samples")

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        self._num_samples = _checked_count(value, "num_samples")


class Config:
    """The global settings of ProbPipe, grouped in sections.

    The sections are:

    1. :attr:`sampling`, a :class:`SamplingConfig`: the default evaluation and
       draw counts of a call;
    2. :attr:`notation`, a :class:`NotationConfig`: how many nested levels a
       label or a notation shows;
    3. :attr:`provenance`, a :class:`ProvenanceConfig`: how much lineage
       history a provenance record keeps;
    4. :attr:`prefect`, a :class:`PrefectConfig`: the orchestration of a
       ``Function`` and the Prefect task runner.

    Assigning a setting sets the session's value, which is global to the
    process. A ``using`` block sets temporary values, for one section with
    ``config.<section>.using(...)`` or for several with :meth:`using`::

        import probpipe

        probpipe.config.sampling.num_samples = 4096
        with probpipe.config.using(sampling={"num_samples": 256}, notation={"max_depth": 12}):
            ...
        probpipe.config.reset()

    A setting resolves from the session's value, then the innermost active
    block, then a view's own control: ``f.using(num_samples=...)`` wins over a
    block, which wins over the session. ``f.using(...)`` returns a reusable
    configured copy of ``f``, while ``config.using(...)`` is a context manager.
    A block applies in the thread that opens it and in the workers that
    ``Function`` calls inside it dispatch to, whether threads or Prefect
    workers, and not in another thread. The work a call dispatches reads the
    session's values as they were when the outermost call opened, except
    those of ``prefect``, so a session assignment made while the call runs
    reaches none of that work. Inside a workflow run, a block or an
    assignment cannot change the provenance mode, as :class:`ProvenanceConfig`
    states. Assigning a setting inside a block sets
    the session's value, which applies once the block exits, and
    :meth:`reset` inside a block restores the session's values while the block
    still applies. Blocks close in the reverse order of opening, and a block
    held open by a generator can break that order in two ways, each of which
    raises ``RuntimeError``:

    - a block closed while a call opened inside it still runs, as when that
      call closes the generator, keeps applying until the call ends;
    - a call or a block that ends while a block opened inside it is still
      open, as when the generator outlives it, closes that block with it.

    Either way no closed block applies afterwards. A call resolves its settings
    when it opens. ProbPipe code
    inside a user's own ``jax.jit`` keeps the values in effect while JAX traced
    it.
    """

    def __init__(self) -> None:
        self._sections: Mapping[str, _Section] = MappingProxyType(
            {
                "sampling": SamplingConfig(),
                "notation": NotationConfig(),
                "provenance": ProvenanceConfig(),
                "prefect": PrefectConfig(),
            }
        )

    @property
    def sampling(self) -> SamplingConfig:
        """The default evaluation count of a lift and the default draw count of an estimate."""
        return self._sections["sampling"]  # type: ignore[return-value]

    @property
    def notation(self) -> NotationConfig:
        """How many nested levels a label or a notation shows."""
        return self._sections["notation"]  # type: ignore[return-value]

    @property
    def provenance(self) -> ProvenanceConfig:
        """How much lineage history a provenance record keeps."""
        return self._sections["provenance"]  # type: ignore[return-value]

    @property
    def prefect(self) -> PrefectConfig:
        """The orchestration of a ``Function`` and the Prefect task runner."""
        return self._sections["prefect"]  # type: ignore[return-value]

    def using(self, **sections: Mapping[str, Any]) -> AbstractContextManager[Config]:
        """A block in which the given settings of several sections replace their values.

        ``config.using(sampling={"num_samples": 4096}, notation={"max_depth": 12})``
        is the block of ``config.sampling.using(num_samples=4096)`` and
        ``config.notation.using(max_depth=12)`` together. The class docstring
        states how a block applies.

        Parameters
        ----------
        **sections : Mapping of str to Any
            The temporary values of each section, keyed by section name and
            then by setting name.

        Returns
        -------
        AbstractContextManager
            The block, which yields ``config``.

        Raises
        ------
        TypeError
            If a name is not a section, naming the sections, if a section is
            given anything but a mapping, or as a section's ``using`` raises.
        ValueError
            As a section's ``using`` raises.
        RuntimeError
            On entry, if the block sets a provenance mode other than that of the
            workflow run open in this thread. On exit, if the block closes out
            of order, as the class docstring states.
        """
        unknown = [name for name in sections if name not in self._sections]
        if unknown:
            raise TypeError(f"config.using(): {unknown_names('section', unknown, self._sections)}")
        checked_config_values = {}
        for name, settings in sections.items():
            if not isinstance(settings, Mapping):
                raise TypeError(
                    f"config.using(): {name}= must be a mapping of settings, "
                    f"got {type(settings).__name__}"
                )
            checked_config_values[name] = self._sections[name]._checked_block_values(
                settings, f"config.using({name}=...)"
            )
        return _config_block(checked_config_values, caller="config.using()", yields=self)

    def reset(self) -> None:
        """Restore every section's session settings to their defaults, reading the environment variables again.

        An active ``using`` block still applies until it exits.

        Raises
        ------
        ValueError
            If an environment variable that a section reads is set to an
            inadmissible value, as each section's ``reset`` states.
        RuntimeError
            Inside a workflow run, if the default provenance mode differs from
            the run's mode. No section is reset then.
        """
        _refuse_changing_the_run_provenance_mode(_initial_provenance_mode(), "config.reset()")
        for section in self._sections.values():
            section.reset()

    def _session_values_snapshot(self) -> dict[str, dict[str, Any]]:
        """The session's values of every section but ``prefect``, by section name and then setting name.

        A root call takes this snapshot when it opens, and the work it
        dispatches reads it in place of the session's current values.
        """
        return {
            section_name: section._session_values()
            for section_name, section in self._sections.items()
            if section._is_in_session_snapshot
        }


def _num_samples_in_effect() -> int:
    """The ``num_samples`` of the running call, or the session's value in effect outside every call."""
    call_context = current_call_context()
    if call_context.call_depth == 0:
        return config.sampling.num_samples
    return call_context.resolved_controls["num_samples"]


@contextmanager
def _config_block(
    config_values: Mapping[str, Mapping[str, Any]], *, caller: str, yields: Any
) -> Iterator[Any]:
    """A block in which the checked *config_values* of each named section apply, yielding *yields*.

    The block installs the call context in effect with one more ``config``
    block, so its values merge over those of the enclosing blocks, and the
    enclosing values apply again when it exits, an error included. A block
    that would change the provenance mode of the open workflow run raises
    when it is entered, before it installs anything, with a message that opens
    with *caller*, the ``using`` call that opened the block.
    """
    block_provenance_mode = config_values.get("provenance", {}).get("mode")
    if block_provenance_mode is not None:
        _refuse_changing_the_run_provenance_mode(block_provenance_mode, caller)
    with _under_call_context(current_call_context().with_config_block(config_values)):
        yield yields


def _no_open_run_provenance_mode() -> ProvenanceMode | None:
    """No workflow run is open, as before the workflow scopes register their reader."""
    return None


#: Reads the provenance mode of the workflow run open in the running thread,
#: or ``None`` outside every run. The workflow scopes are defined above
#: ``core``, so they register the reader when they are imported.
_open_run_provenance_mode: Callable[[], ProvenanceMode | None] = _no_open_run_provenance_mode


def _register_open_run_provenance_mode_reader(
    open_run_provenance_mode: Callable[[], ProvenanceMode | None],
) -> None:
    """Register the reader of the open workflow run's provenance mode, which the workflow scopes define."""
    global _open_run_provenance_mode
    _open_run_provenance_mode = open_run_provenance_mode


def _refuse_changing_the_run_provenance_mode(requested_mode: ProvenanceMode, caller: str) -> None:
    """Raise if *caller* would set *requested_mode* inside a workflow run fixed to another mode.

    Parameters
    ----------
    requested_mode : ProvenanceMode
        The mode that a block, an assignment, or a reset would set.
    caller : str
        The block, the assignment, or the reset, which the message opens with.

    Raises
    ------
    RuntimeError
        If a workflow run is open in the running thread and its provenance mode
        is not *requested_mode*.
    """
    run_provenance_mode = _open_run_provenance_mode()
    if run_provenance_mode is None or run_provenance_mode is requested_mode:
        return
    raise RuntimeError(
        f"{caller}: cannot set the provenance mode to ProvenanceMode.{requested_mode.name} "
        f"inside a workflow run under ProvenanceMode.{run_provenance_mode.name}. Set the mode "
        "before the workflow_run opens, or before the call if no workflow_run is open"
    )


# Module-level singleton, built after the helpers its sections call.
config = Config()
"""The global settings of ProbPipe, an instance of ``Config``."""
