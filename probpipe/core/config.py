"""Global configuration for ProbPipe orchestration and provenance.

Provides:
- ``WorkflowKind`` enum and ``PrefectConfig`` singleton (``prefect_config``)
  controlling how ``Function`` instances dispatch work.
- ``ProvenanceMode`` enum and ``ProvenanceConfig`` singleton
  (``provenance_config``) controlling how much lineage history is retained.
- ``NotationConfig`` singleton (``notation_config``) setting how many nested
  levels a label or a notation shows.

Users import from the top-level package::

    import probpipe
    from probpipe import WorkflowKind, ProvenanceMode

    probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
    probpipe.provenance_config.mode = ProvenanceMode.FULL
    probpipe.notation_config.max_depth = 12
"""

from __future__ import annotations

import os

__all__ = [
    "NotationConfig",
    "PrefectConfig",
    "ProvenanceConfig",
    "ProvenanceMode",
    "WorkflowKind",
    "notation_config",
    "prefect_config",
    "provenance_config",
]
from enum import Enum
from typing import Any

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
        explicit assignment to ``prefect_config.workflow_kind``. At
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


# ---------------------------------------------------------------------------
# PrefectConfig singleton
# ---------------------------------------------------------------------------


class PrefectConfig:
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
            probpipe.prefect_config.workflow_kind = probpipe.WorkflowKind.TASK
    task_runner : object or None
        Prefect task runner instance (e.g., ``RayTaskRunner()``).
        ``None`` means auto-detect: use ``RayTaskRunner`` if
        ``prefect-ray`` is installed, then ``DaskTaskRunner`` if
        ``prefect-dask`` is installed, otherwise Prefect's built-in
        default.
    """

    def __init__(self) -> None:
        self.reset()

    # -- Public API ---------------------------------------------------------

    def reset(self) -> None:
        """Restore all settings to defaults (re-reading the env var)."""
        self._workflow_kind: WorkflowKind = _initial_workflow_kind()
        self._task_runner: Any = None

    @property
    def workflow_kind(self) -> WorkflowKind:
        """Current global orchestration mode."""
        return self._workflow_kind

    @workflow_kind.setter
    def workflow_kind(self, value: WorkflowKind) -> None:
        if not isinstance(value, WorkflowKind):
            raise TypeError(
                f"workflow_kind must be a WorkflowKind enum member, got {type(value).__name__}"
            )
        self._workflow_kind = value

    @property
    def task_runner(self) -> Any:
        """Explicit task runner, or ``None`` for auto-detection."""
        return self._task_runner

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
        if self._task_runner is not None:
            return self._task_runner
        return _auto_detect_task_runner()


# Module-level singleton
prefect_config = PrefectConfig()
"""The global Prefect orchestration settings, an instance of ``PrefectConfig``."""


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


# ---------------------------------------------------------------------------
# ProvenanceConfig singleton
# ---------------------------------------------------------------------------


class ProvenanceConfig:
    """Global provenance tracking settings.

    Controls how much lineage history ``Function`` retains when
    assembling provenance for each result.  Set once at application startup::

        import probpipe
        from probpipe import ProvenanceMode

        probpipe.provenance_config.mode = ProvenanceMode.FULL  # for debugging

    The initial mode can also be set via the ``PROBPIPE_PROVENANCE_MODE``
    environment variable (``full``, ``lightweight``, or ``off``,
    case-insensitive).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore all settings to defaults (re-reading the env var)."""
        self._mode: ProvenanceMode = _initial_provenance_mode()

    @property
    def mode(self) -> ProvenanceMode:
        """Current global provenance tracking mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ProvenanceMode) -> None:
        if not isinstance(value, ProvenanceMode):
            raise TypeError(
                f"mode must be a ProvenanceMode enum member, got {type(value).__name__}"
            )
        self._mode = value


# Module-level singleton
provenance_config = ProvenanceConfig()
"""The global provenance tracking settings, an instance of ``ProvenanceConfig``."""


# ---------------------------------------------------------------------------
# NotationConfig singleton
# ---------------------------------------------------------------------------

_NOTATION_MAX_DEPTH_ENV_VAR = "PROBPIPE_NOTATION_MAX_DEPTH"

#: The number of nested levels a label or a notation shows by default.
_DEFAULT_MAX_DEPTH = 8


def _checked_max_depth(value: Any, source: str) -> int:
    """*value* as a number of nested levels, which must be a positive integer.

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
        If *value* is less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{source} must be a positive integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{source} must be a positive integer, got {value}")
    return value


def _initial_max_depth() -> int:
    """Resolve the initial ``max_depth`` from the environment.

    Reads ``PROBPIPE_NOTATION_MAX_DEPTH``. Unset gives 8, and a value that is
    not a positive integer raises ``ValueError``, so a typo in a deployment's
    configuration surfaces rather than falling back to the default.
    """
    raw = os.environ.get(_NOTATION_MAX_DEPTH_ENV_VAR)
    if raw is None:
        return _DEFAULT_MAX_DEPTH
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(
            f"{_NOTATION_MAX_DEPTH_ENV_VAR}={raw!r} is not a valid depth; it must be a "
            f"positive integer"
        )
    return value


class NotationConfig:
    """Global settings of how labels and notations render.

    A tracked term's label and the notation of a law, a kernel, or a function
    are renderings of the expression the term carries, which nests one level
    for each value or law it was computed from, as
    ``E[f(beta ~ model; y)]`` nests four. A rendering shows at most
    :attr:`max_depth` levels::

        import probpipe

        probpipe.notation_config.max_depth = 12

    The initial depth can also be set by the ``PROBPIPE_NOTATION_MAX_DEPTH``
    environment variable, a positive integer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore all settings to defaults (re-reading the env var)."""
        self._max_depth: int = _initial_max_depth()

    @property
    def max_depth(self) -> int:
        """The number of nested levels a label or a notation shows, 8 by default.

        A part nested deeper renders as its label, the name of a law or a
        function, or as ``…`` for a value, and the rendering warns with a
        ``UserWarning``. Every label the library derives from one law and a
        few operators on it nests at most 5 levels, as
        ``E[f(beta ~ model; y)][sample=0] + 1`` does, so the default shows
        each of them in full, while a label derived through a long chain of
        operations, such as a loop that adds to a value, stays bounded.
        Setting it changes the renderings made afterwards, and a term keeps
        the label it was given.

        Raises
        ------
        TypeError
            On assignment of a value that is not an integer, a bool included.
        ValueError
            On assignment of an integer less than 1.
        """
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._max_depth = _checked_max_depth(value, "max_depth")


# Module-level singleton
notation_config = NotationConfig()
"""The global notation settings, an instance of ``NotationConfig``."""
