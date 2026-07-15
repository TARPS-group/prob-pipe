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
    "CacheMode",
    "PrefectConfig",
    "ProvenanceConfig",
    "ProvenanceMode",
    "WorkflowKind",
    "prefect_config",
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
# CacheMode enum
# ---------------------------------------------------------------------------


class CacheMode(Enum):
    """Controls whether Prefect task results are cached across runs.

    Caching is opt-in and only takes effect when Prefect orchestration is
    active (``WorkflowKind.TASK`` / ``FLOW``).

    Members
    -------
    OFF
        No caching.  Tasks always execute; behaviour is identical to not
        configuring caching at all.  This is the default.
    FINGERPRINT
        Cache task results keyed by a stable content fingerprint of the user
        function, its inputs, and the runtime environment.  A re-run with the
        same function, inputs, and environment reuses the persisted result
        instead of re-executing the task.
    """

    OFF = "off"
    FINGERPRINT = "fingerprint"


# ---------------------------------------------------------------------------
# Environment-variable override for CacheMode
# ---------------------------------------------------------------------------

_CACHE_MODE_ENV_VAR = "PROBPIPE_CACHE_MODE"


def _initial_cache_mode() -> CacheMode:
    """Resolve the initial ``cache_mode`` from the environment.

    Reads ``PROBPIPE_CACHE_MODE`` (case-insensitive).  Unset → ``OFF``.
    Unknown values raise ``ValueError`` so deployment-config typos surface
    loudly rather than silently disabling caching.
    """
    raw = os.environ.get(_CACHE_MODE_ENV_VAR)
    if raw is None:
        return CacheMode.OFF
    try:
        return CacheMode(raw.lower())
    except ValueError as e:
        valid = ", ".join(repr(m.value) for m in CacheMode)
        raise ValueError(
            f"{_CACHE_MODE_ENV_VAR}={raw!r} is not a valid CacheMode. Expected one of: {valid}."
        ) from e


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
    cache_mode : CacheMode
        Cross-run task caching mode.  ``OFF`` (default) disables caching;
        ``FINGERPRINT`` caches task results keyed by a content fingerprint.
        Initial value comes from ``PROBPIPE_CACHE_MODE`` if set.  Only takes
        effect under ``WorkflowKind.TASK`` / ``FLOW``.
    cache_result_storage : object or None
        Where cached results and cache-key records are stored.  ``None``
        (default) uses Prefect's built-in local storage (single-machine only);
        set a shared Prefect storage target to cache across workers/machines.
    """

    def __init__(self) -> None:
        self.reset()

    # -- Public API ---------------------------------------------------------

    def reset(self) -> None:
        """Restore all settings to defaults (re-reading the env vars)."""
        self._workflow_kind: WorkflowKind = _initial_workflow_kind()
        self._task_runner: Any = None
        self._cache_mode: CacheMode = _initial_cache_mode()
        self._cache_result_storage: Any = None

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

    # -- Caching ------------------------------------------------------------

    @property
    def cache_mode(self) -> CacheMode:
        """Current cross-run task caching mode (default ``OFF``)."""
        return self._cache_mode

    @cache_mode.setter
    def cache_mode(self, value: CacheMode) -> None:
        if not isinstance(value, CacheMode):
            raise TypeError(
                f"cache_mode must be a CacheMode enum member, got {type(value).__name__}"
            )
        self._cache_mode = value

    @property
    def caching_enabled(self) -> bool:
        """Whether caching is active (``cache_mode`` is not ``OFF``)."""
        return self._cache_mode is not CacheMode.OFF

    @property
    def cache_result_storage(self) -> Any:
        """Storage target for persisted cache results and cache-key records.

        ``None`` (default) uses Prefect's built-in local storage, which is
        single-machine only.  Set a shared Prefect storage target to cache
        across workers/machines; the same target is used for both the task's
        ``result_storage`` and the cache policy's ``key_storage``.
        """
        return self._cache_result_storage

    @cache_result_storage.setter
    def cache_result_storage(self, value: Any) -> None:
        self._cache_result_storage = value

    def resolve_cache_storage(self) -> Any:
        """Return the configured cache storage target, or ``None`` for local.

        The value is passed to both Prefect's ``result_storage`` and the cache
        policy's ``key_storage``.  ``None`` means use Prefect's local default.
        """
        return self._cache_result_storage


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

    Controls how much lineage history ``WorkflowFunction`` retains when
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
