"""Global Prefect orchestration configuration for ProbPipe.

Provides a ``WorkflowKind`` enum and a ``PrefectConfig`` singleton
(``prefect_config``) that controls how ``WorkflowFunction`` instances
dispatch work.  Users import from the top-level package::

    import probpipe
    from probpipe import WorkflowKind

    probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
"""

from __future__ import annotations

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
        Auto-detect: use ``TASK`` if Prefect is installed, otherwise
        ``OFF``.  At the per-instance level, ``DEFAULT`` means "inherit
        from global config".
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
        that do not override their own.  Default: ``WorkflowKind.DEFAULT``
        (use Prefect tasks if available, otherwise off).
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
        """Restore all settings to defaults."""
        self._workflow_kind: WorkflowKind = WorkflowKind.DEFAULT
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
