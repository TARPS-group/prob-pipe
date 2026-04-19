# Plan: Global Prefect Configuration for ProbPipe

## Motivation

ProbPipe's `WorkflowFunction` already supports Prefect orchestration via
`workflow_kind="task"|"flow"`, but the integration has several gaps that
prevent real-world use:

1. **No global defaults.** Every `WorkflowFunction` must be individually
   configured. High-level ops like `condition_on` — which are pre-built
   `WorkflowFunction` instances — have no way to receive orchestration
   settings at all. This means the bagged posterior workflow
   (`condition_on` with `BootstrapReplicateDistribution`) cannot use
   Prefect without modifying library internals.

2. **No task runner support.** Prefect's default task runner executes
   tasks sequentially in-process. For actual parallelism you need a
   runner like `RayTaskRunner`, but there is no way to pass one — neither
   per-instance nor globally.

3. **No global off switch.** If Prefect is installed (e.g., as part of a
   larger environment), there is no way to disable orchestration
   system-wide without removing the package.

4. **Brittle fallback.** When Prefect is not installed, setting
   `workflow_kind` on *any* `WorkflowFunction` raises an `ImportError`
   at call time — even if the user intended it as a soft preference via
   global config.

These gaps surfaced while building a scalability example notebook
(issue: "validate scalability when using Prefect"). The notebook could
not actually demonstrate Prefect-distributed bagged posteriors because
the infrastructure did not support it.

## Philosophy

The guiding principle is **set once, apply everywhere** with smart
defaults. ProbPipe should do the right thing out of the box:

- If Prefect is installed → use it (as tasks, by default)
- If Prefect + Ray are installed → use Ray as the task runner
- If nothing is installed → run plain Python, no errors

This is captured in the `WorkflowKind.DEFAULT` mode, which is the
default for both the global config and every `WorkflowFunction`. The
user never *has* to configure anything — but they *can*:

```python
import probpipe

# Explicit: force all WorkflowFunctions to use Prefect tasks
probpipe.prefect_config.workflow_kind = WorkflowKind.TASK

# Or: turn off Prefect entirely
probpipe.prefect_config.workflow_kind = WorkflowKind.OFF

# Or: override the auto-detected task runner
probpipe.prefect_config.task_runner = RayTaskRunner(address="ray://cluster:10001")
```

Per-instance overrides remain available for fine-grained control:

```python
# This specific function uses a flow, regardless of global config
my_wf = WorkflowFunction(func=f, workflow_kind=WorkflowKind.FLOW)

# This one opts out of Prefect entirely
my_other_wf = WorkflowFunction(func=g, workflow_kind=WorkflowKind.OFF)
```

## Ultimate Goal

Enable the following end-to-end workflow with **zero changes** to
existing ProbPipe modeling code:

```python
# If prefect and ray are installed, this Just Works — no config needed.
# DEFAULT mode auto-detects Prefect + Ray and uses them.

bootstrap_data = BootstrapReplicateDistribution(
    EmpiricalDistribution(jnp.asarray(y_observed))
)
bagged_posterior = condition_on(
    model, bootstrap_data,
    num_results=500, num_warmup=300, random_seed=42,
    n_broadcast_samples=48,
)
```

Each of the 48 bootstrap MCMC fits runs as a Prefect task on Ray workers
— automatically, because `DEFAULT` detected both packages. If only
Prefect is installed (no Ray), it uses Prefect's default runner. If
neither is installed, it falls back to plain sequential/threaded
execution.

For explicit control over a remote cluster:

```python
import probpipe
from probpipe import WorkflowKind
from prefect_ray import RayTaskRunner

probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
probpipe.prefect_config.task_runner = RayTaskRunner(address="ray://cluster:10001")
```

The notebook demonstrating this should be simple, beautiful, and use the
existing bagged posterior example from the ProbPipe tutorial — not a
custom-built scenario.

## Design

### `WorkflowKind` enum

Replace string literals and `None` with an explicit enum. Every state
is named — no ambiguity about what `None` means:

```python
from enum import Enum

class WorkflowKind(Enum):
    """Orchestration mode for WorkflowFunction instances.

    Members
    -------
    DEFAULT
        Auto-detect: use ``TASK`` if Prefect is installed, otherwise
        ``OFF``. At the per-instance level, ``DEFAULT`` means "inherit
        from global config".
    OFF
        No Prefect orchestration. Plain Python execution.
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
```

**Key semantics of `DEFAULT`:**
- **At the global config level:** "use `TASK` if Prefect is installed,
  otherwise `OFF`." This is the zero-configuration happy path.
- **At the per-instance level:** "inherit from global config." A
  `WorkflowFunction` with `workflow_kind=DEFAULT` defers to whatever
  the global config resolves to.

### New module: `probpipe/core/config.py`

A lightweight configuration singleton:

```python
class PrefectConfig:
    """Global Prefect orchestration settings.

    Parameters
    ----------
    workflow_kind : WorkflowKind
        Default orchestration mode for all ``WorkflowFunction`` instances
        that do not override their own. Default: ``WorkflowKind.DEFAULT``
        (use Prefect tasks if available, otherwise off).
    task_runner : object or None
        Prefect task runner instance (e.g., ``RayTaskRunner()``) passed
        to internally-created flows. ``None`` means auto-detect: use
        ``RayTaskRunner`` if ``prefect-ray`` is installed, otherwise
        Prefect's built-in default.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Restore all settings to defaults."""
        self._workflow_kind = WorkflowKind.DEFAULT
        self._task_runner = None   # None = auto-detect

    @property
    def workflow_kind(self) -> WorkflowKind:
        return self._workflow_kind

    @workflow_kind.setter
    def workflow_kind(self, value: WorkflowKind):
        if not isinstance(value, WorkflowKind):
            raise TypeError(
                f"workflow_kind must be a WorkflowKind enum member, "
                f"got {type(value).__name__}"
            )
        self._workflow_kind = value

    # Similar property for task_runner...

    def resolve_task_runner(self):
        """Return the effective task runner (explicit or auto-detected)."""
        if self._task_runner is not None:
            return self._task_runner
        return _auto_detect_task_runner()
```

### Task runner auto-detection

When `task_runner` is `None` (the default), ProbPipe probes for
installed runner packages and picks the best available:

```python
def _auto_detect_task_runner():
    """Return a task runner based on installed packages, or None."""
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
    return None  # Use Prefect's built-in default (sequential)
```

Priority order: **Ray > Dask > Prefect default**. The user can always
override by setting `prefect_config.task_runner` explicitly.

### Resolution property: `effective_workflow_kind`

Lazy resolution at call time (not construction time), so config changes
after construction take effect:

```python
@property
def effective_workflow_kind(self) -> WorkflowKind:
    raw = self._workflow_kind_raw

    # 1. Per-instance explicit (non-DEFAULT) override
    if raw is not WorkflowKind.DEFAULT:
        kind = raw
    else:
        # 2. Resolve global config's DEFAULT
        global_kind = prefect_config.workflow_kind
        if global_kind is WorkflowKind.DEFAULT:
            # DEFAULT at global level = auto-detect
            kind = WorkflowKind.TASK if task is not None else WorkflowKind.OFF
        else:
            kind = global_kind

    # 3. Validate availability
    if kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and task is None:
        # Was this an explicit user request or auto-detected?
        if raw not in (WorkflowKind.DEFAULT,):
            raise ImportError(
                f"Prefect is required for workflow_kind={kind!r}: "
                f"pip install probpipe[prefect]"
            )
        # Auto-detected but unavailable → fall back silently
        return WorkflowKind.OFF

    return kind
```

**Resolution summary:**

| Per-instance | Global config | Prefect installed? | Result |
|---|---|---|---|
| `DEFAULT` | `DEFAULT` | Yes | `TASK` (auto-detect) |
| `DEFAULT` | `DEFAULT` | No | `OFF` (graceful fallback) |
| `DEFAULT` | `TASK` | Yes | `TASK` |
| `DEFAULT` | `OFF` | Yes | `OFF` |
| `TASK` | (any) | Yes | `TASK` |
| `TASK` | (any) | No | `ImportError` (explicit request) |
| `FLOW` | (any) | Yes | `FLOW` |
| `OFF` | (any) | (any) | `OFF` |

### Task runner threading

The resolved task runner is passed to the Prefect `@flow` decorator in
`_execute_many_prefect_task` and `_execute_many_prefect_flow`:

```python
def _execute_many_prefect_task(self, call_value_list):
    runner = prefect_config.resolve_task_runner()

    @flow(name=f"{self._name}_map",
          **({"task_runner": runner} if runner is not None else {}))
    def _task_map_flow():
        return self._map_task(call_value_list)

    return _task_map_flow()
```

Similarly for `_broadcast_jax` when wrapping vmap in a flow.

### Module class

The `Module` class also defaults to `WorkflowKind.DEFAULT`, so child
`@workflow_method` instances inherit the global config unless the module
or method explicitly overrides.

## Action Plan

### Phase 1: Infrastructure (this PR)

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1 | Create `WorkflowKind` enum | `probpipe/core/config.py` | `DEFAULT`, `OFF`, `TASK`, `FLOW` |
| 2 | Create `PrefectConfig` class and `prefect_config` singleton | `probpipe/core/config.py`  | Properties with validation; `reset()` for testing; `resolve_task_runner()` with auto-detection |
| 3 | Add `_auto_detect_task_runner()` helper | `probpipe/core/config.py` | Ray > Dask > None probe chain |
| 4 | Change `WorkflowFunction.__init__` default to `WorkflowKind.DEFAULT` | `probpipe/core/node.py` | Store as `_workflow_kind_raw`; accept both enum and legacy strings (convert internally) |
| 5 | Add `effective_workflow_kind` property | `probpipe/core/node.py` | Resolution logic: per-instance → global → auto-detect → availability check |
| 6 | Replace all internal reads of `self._workflow_kind` with `self.effective_workflow_kind` | `probpipe/core/node.py` | `_execute_many`, `_broadcast_jax`, provenance metadata |
| 7 | Thread `prefect_config.resolve_task_runner()` into `@flow` decorators | `probpipe/core/node.py` | `_execute_many_prefect_task`, `_execute_many_prefect_flow`, and `_broadcast_jax` |
| 8 | Update `Module.__init__` to default to `WorkflowKind.DEFAULT` | `probpipe/core/node.py` | Propagation to child `@workflow_method` preserves DEFAULT |
| 9 | Export `prefect_config` and `WorkflowKind` from package | `probpipe/__init__.py` | Public API additions |
| 10 | Write tests for config behavior | `tests/test_prefect_config.py` | See test plan below |
| 11 | Verify existing tests pass unchanged | `tests/test_prefect_orchestration.py` | Backward compatibility check; may need to update string literals to enum values |

### Phase 2: Example notebook (follow-up PR)

Rewrite `docs/examples/10_prefect_scalability.ipynb` using the bagged
posterior model from the tutorial. The notebook should:

- Show that `DEFAULT` mode auto-detects Prefect + Ray with no config
- Show explicit config for remote Ray clusters
- Run the same bagged posterior code from the tutorial, now distributed
- Compare sequential vs. threaded vs. Prefect timing
- Show `WorkflowKind.OFF` and per-instance overrides
- Be simple and beautiful — no custom models or synthetic data generation

## Test Plan

### New tests (`tests/test_prefect_config.py`)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_default_values` | `prefect_config` starts with `workflow_kind=DEFAULT`, `task_runner=None` |
| 2 | `test_workflow_kind_enum_values` | Enum has exactly `DEFAULT`, `OFF`, `TASK`, `FLOW` |
| 3 | `test_global_workflow_kind_applies` | WF with `DEFAULT` inherits global `TASK` setting |
| 4 | `test_per_instance_overrides_global` | Explicit `FLOW` beats global `TASK` |
| 5 | `test_explicit_off_overrides_global` | Explicit `OFF` disables Prefect even if global is `TASK` |
| 6 | `test_default_auto_detects_prefect` | `DEFAULT` resolves to `TASK` when Prefect is installed |
| 7 | `test_default_falls_back_without_prefect` | `DEFAULT` resolves to `OFF` when Prefect is not installed |
| 8 | `test_explicit_task_raises_without_prefect` | Per-instance `TASK` + Prefect missing → `ImportError` |
| 9 | `test_auto_detect_ray_runner` | `resolve_task_runner()` returns `RayTaskRunner` when `prefect-ray` is installed |
| 10 | `test_auto_detect_dask_runner` | `resolve_task_runner()` returns `DaskTaskRunner` when only `prefect-dask` is installed |
| 11 | `test_auto_detect_no_runner` | `resolve_task_runner()` returns `None` when no runner packages installed |
| 12 | `test_explicit_runner_overrides_auto` | Setting `task_runner` explicitly bypasses auto-detection |
| 13 | `test_task_runner_threaded_to_flow` | `resolve_task_runner()` result is passed to `@flow(task_runner=...)` |
| 14 | `test_reset` | `reset()` restores all defaults |
| 15 | `test_validation_rejects_invalid` | Non-enum `workflow_kind` value raises `TypeError` |
| 16 | `test_config_change_after_construction` | Config change after WF creation takes effect (lazy resolution) |
| 17 | `test_module_inherits_global` | `Module()` with `DEFAULT` → children use global config |
| 18 | `test_legacy_string_conversion` | `WorkflowFunction(workflow_kind="task")` auto-converts to `WorkflowKind.TASK` |

### Existing tests

All tests in `test_prefect_orchestration.py` should pass. Tests that
pass `workflow_kind="task"` as a string will need updating to use
`WorkflowKind.TASK`, or the constructor must accept legacy strings and
convert them (see task #4).

## Backward Compatibility

**No breaking changes** for the default case. The migration is:

| Before | After | Behavior |
|--------|-------|----------|
| `WorkflowFunction(func=f)` | `workflow_kind=DEFAULT` → auto-detect | Same if Prefect not installed (`OFF`). **New**: auto-uses Prefect if installed. |
| `WorkflowFunction(func=f, workflow_kind="task")` | Legacy string auto-converted to `WorkflowKind.TASK` | Same |
| `WorkflowFunction(func=f, workflow_kind=None)` | Legacy `None` auto-converted to `WorkflowKind.OFF` | Same |
| `Module(workflow_kind="task")` | Legacy string auto-converted | Same |
| `Module()` | `workflow_kind=DEFAULT` → auto-detect | Same if Prefect not installed |

**Important behavior change:** If Prefect *is* installed, the default
behavior shifts from "no orchestration" to "use Prefect tasks". This is
intentional — `workflow_kind` should
default to task. Users who want the old behavior can set
`prefect_config.workflow_kind = WorkflowKind.OFF`.

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `probpipe/core/config.py` | **New** | `WorkflowKind` enum, `PrefectConfig` class, `prefect_config` singleton, `_auto_detect_task_runner()` |
| `probpipe/core/node.py` | Modified | `DEFAULT` enum default, `effective_workflow_kind` property, task runner threading, legacy string conversion |
| `probpipe/__init__.py` | Modified | Export `prefect_config` and `WorkflowKind` |
| `tests/test_prefect_config.py` | **New** | Config + enum behavior tests |

## Open Questions

1. **Should `effective_workflow_kind` supersede `parallel`?**
   Currently `parallel=True` (threading) is orthogonal to Prefect. If
   the effective kind resolves to `TASK` or `FLOW`, should it
   automatically replace threading? Proposed: yes — if orchestration is
   active, it handles distribution; `parallel` is ignored.

2. **Context manager API?** e.g., `with prefect_config.override(workflow_kind=OFF):`.
   Proposed: defer to a follow-up.

3. **Environment variable support?** e.g., `PROBPIPE_WORKFLOW_KIND=off`.
   Proposed: defer to a follow-up unless reviewers feel it is essential.
