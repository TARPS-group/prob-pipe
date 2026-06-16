"""Global configuration for ProbPipe orchestration and provenance.

Provides:
- ``WorkflowKind`` enum and ``PrefectConfig`` singleton (``prefect_config``)
  controlling how ``WorkflowFunction`` instances dispatch work.
- ``ProvenanceMode`` enum and ``ProvenanceConfig`` singleton
  (``provenance_config``) controlling how much lineage history is retained.

Users import from the top-level package::

    import probpipe
    from probpipe import WorkflowKind, ProvenanceMode

    probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
    probpipe.provenance_config.mode = ProvenanceMode.FULL
"""

from __future__ import annotations

import os

__all__ = [
    "WorkflowKind",
    "PrefectConfig",
    "prefect_config",
    "ProvenanceMode",
    "ProvenanceConfig",
    "provenance_config",
]
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# WorkflowKind enum
# ---------------------------------------------------------------------------

class WorkflowKind(Enum):
    """Orchestration mode for ``WorkflowFunction`` instances.

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

    Parameters
    ----------
    workflow_kind : WorkflowKind
        Default orchestration mode for all ``WorkflowFunction`` instances
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
                f"workflow_kind must be a WorkflowKind enum member, "
                f"got {type(value).__name__}"
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


# ---------------------------------------------------------------------------
# ProvenanceMode enum
# ---------------------------------------------------------------------------

class ProvenanceMode(Enum):
    """Controls how much history is retained in provenance chains.

    Members
    -------
    FULL
        Store live references to parent Distribution / Record /
        RecordArray objects.  The entire ancestry chain stays in memory
        as long as the final result is alive.  Good for debugging and
        small test workflows where full graph traversal is useful.
    LIGHTWEIGHT
        Store only lightweight :class:`~probpipe.core.provenance.ParentInfo`
        descriptors — type name, distribution name, and an optional
        fingerprint.  Parent objects are free to be garbage-collected once
        a workflow step completes.  This is the default and scales to
        larger workflows.
    OFF
        Attach no provenance at all.  Minimises overhead when lineage
        tracking is not needed.
    """

    FULL = "full"
    LIGHTWEIGHT = "lightweight"
    OFF = "off"


# ---------------------------------------------------------------------------
# ProvenanceConfig singleton
# ---------------------------------------------------------------------------

class ProvenanceConfig:
    """Global provenance tracking settings.

    Controls how much lineage history ``WorkflowFunction`` retains when
    assembling provenance for each result.  Set once at application startup::

        import probpipe
        from probpipe import ProvenanceMode

        probpipe.provenance_config.mode = ProvenanceMode.FULL  # for debugging
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore all settings to defaults."""
        self._mode: ProvenanceMode = ProvenanceMode.LIGHTWEIGHT

    @property
    def mode(self) -> ProvenanceMode:
        """Current global provenance tracking mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ProvenanceMode) -> None:
        if not isinstance(value, ProvenanceMode):
            raise TypeError(
                f"mode must be a ProvenanceMode enum member, "
                f"got {type(value).__name__}"
            )
        self._mode = value


# Module-level singleton
provenance_config = ProvenanceConfig()
