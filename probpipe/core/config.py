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
from contextvars import ContextVar
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Self

from .._messages import unknown_names
from ._expression import _STORED_DEPTH

# ---------------------------------------------------------------------------
# Sections and temporary settings
# ---------------------------------------------------------------------------

#: The values the active ``using`` blocks set, by section name and setting,
#: the innermost block's last. Reading a setting returns its value here, and
#: otherwise the session's value.
_TEMPORARY: ContextVar[Mapping[str, Mapping[str, Any]]] = ContextVar(
    "probpipe_temporary_settings", default=MappingProxyType({})
)


@contextmanager
def _temporarily(values: Mapping[str, Mapping[str, Any]], scope: Any) -> Iterator[Any]:
    """A block in which the checked *values* of each named section apply, yielding *scope*.

    The block's values are merged over those of the enclosing blocks, and the
    enclosing values apply again when it exits, an error included.
    """
    enclosing = _TEMPORARY.get()
    merged = dict(enclosing)
    for name, fields in values.items():
        merged[name] = MappingProxyType({**enclosing.get(name, {}), **fields})
    token = _TEMPORARY.set(MappingProxyType(merged))
    try:
        yield scope
    finally:
        _TEMPORARY.reset(token)


def _temporary_settings() -> dict[str, dict[str, Any]]:
    """The values of the active ``using`` blocks, as plain mappings that a worker can take."""
    return {name: dict(fields) for name, fields in _TEMPORARY.get().items()}


@contextmanager
def _applying(settings: Mapping[str, Mapping[str, Any]]) -> Iterator[None]:
    """A scope in which *settings*, which :func:`_temporary_settings` took, apply in place of any block."""
    token = _TEMPORARY.set(
        MappingProxyType({name: MappingProxyType(dict(f)) for name, f in settings.items()})
    )
    try:
        yield
    finally:
        _TEMPORARY.reset(token)


class _Section:
    """A section of ``config``: named settings, each with a session value and a check.

    Reading a setting returns the value that the innermost active ``using``
    block sets for it, and otherwise the session's value. Assigning a setting
    checks the value and sets the session's value, which an active block that
    sets the setting overrides until it exits.
    """

    #: The section's name in ``config``, which the blocks are keyed by.
    _name: ClassVar[str]

    #: The check of each setting, by name: it takes the value and the setting's
    #: name, and returns the value or raises as an assignment raises.
    _checks: ClassVar[Mapping[str, Callable[[Any, str], Any]]]

    def _current(self, setting: str) -> Any:
        """The value of *setting* that the innermost block sets, or else the session's value."""
        block = _TEMPORARY.get().get(self._name)
        if block is not None and setting in block:
            return block[setting]
        return getattr(self, f"_{setting}")

    def _checked(self, settings: Mapping[str, Any], caller: str) -> dict[str, Any]:
        """*settings* checked as an assignment checks them, for the block *caller* opens.

        Parameters
        ----------
        settings : Mapping of str to Any
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
        unknown = [name for name in settings if name not in self._checks]
        if unknown:
            raise TypeError(f"{caller}: {unknown_names('setting', unknown, self._checks)}")
        return {name: self._checks[name](value, name) for name, value in settings.items()}

    def using(self, **settings: Any) -> AbstractContextManager[Self]:
        """A block in which *settings* replace the section's values, as ``with section.using(...):``.

        The block's values apply to every read inside it, by the calls a
        ``Function`` dispatches to worker threads or Prefect workers included,
        and the previous values apply again when it exits, an error included.
        Blocks nest, and an inner block's value wins. A view's own control,
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
        """
        return _temporarily(
            {self._name: self._checked(settings, f"config.{self._name}.using()")}, self
        )


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
        return self._current("workflow_kind")

    @workflow_kind.setter
    def workflow_kind(self, value: WorkflowKind) -> None:
        self._workflow_kind = _checked_workflow_kind(value, "workflow_kind")

    @property
    def task_runner(self) -> Any:
        """Explicit task runner, or ``None`` for auto-detection."""
        return self._current("task_runner")

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
    case-insensitive). A top-level workflow takes the mode in effect when it
    is entered.
    """

    _name = "provenance"
    _checks = MappingProxyType({"mode": _checked_provenance_mode})

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore the session's mode to its default (re-reading the env var).

        An active ``using`` block that sets the mode still applies until it exits.
        """
        self._mode: ProvenanceMode = _initial_provenance_mode()

    @property
    def mode(self) -> ProvenanceMode:
        """Current global provenance tracking mode."""
        return self._current("mode")

    @mode.setter
    def mode(self, value: ProvenanceMode) -> None:
        self._mode = _checked_provenance_mode(value, "mode")


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
        return self._current("max_depth")

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
        return self._current("max_evaluations")

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
        return self._current("num_samples")

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
    A block is stored in a context variable, as ``workflow_run`` stores its
    scope, so it applies in the thread that opens it and in the workers that
    ``Function`` calls inside it dispatch to, whether threads or Prefect
    workers, and not in another thread. Assigning a setting inside a block sets
    the session's value, which applies once the block exits, and
    :meth:`reset` inside a block restores the session's values while the block
    still applies. A setting read while JAX traces a function, as under
    ``jax.jit``, is the value at trace time, and the compiled function keeps it.
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
        """
        unknown = [name for name in sections if name not in self._sections]
        if unknown:
            raise TypeError(f"config.using(): {unknown_names('section', unknown, self._sections)}")
        checked = {}
        for name, settings in sections.items():
            if not isinstance(settings, Mapping):
                raise TypeError(
                    f"config.using(): {name}= must be a mapping of settings, "
                    f"got {type(settings).__name__}"
                )
            checked[name] = self._sections[name]._checked(settings, f"config.using({name}=...)")
        return _temporarily(checked, self)

    def reset(self) -> None:
        """Restore every section's session settings to their defaults, reading the environment variables again.

        An active ``using`` block still applies until it exits.

        Raises
        ------
        ValueError
            If an environment variable that a section reads is set to an
            inadmissible value, as each section's ``reset`` states.
        """
        for section in self._sections.values():
            section.reset()


# Module-level singleton
config = Config()
"""The global settings of ProbPipe, an instance of ``Config``."""
