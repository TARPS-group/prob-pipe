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

The guiding principle is **set once, apply everywhere**:

```python
import probpipe

probpipe.prefect_config.workflow_kind = "task"
probpipe.prefect_config.task_runner = RayTaskRunner()
```

After these two lines, every `WorkflowFunction` in the system — including
the one inside `condition_on` — automatically dispatches work as Prefect
tasks on a Ray cluster. No function signatures change. No new parameters
thread through call chains. The user opts in at the top of their script
and the entire pipeline benefits.

Per-instance overrides remain available for fine-grained control:

```python
# This specific function uses a flow, regardless of global config
my_wf = WorkflowFunction(func=f, workflow_kind="flow")

# This one opts out of Prefect entirely
my_other_wf = WorkflowFunction(func=g, workflow_kind=None)
```

When Prefect is not installed, global config settings degrade gracefully
(silent fallback to plain execution), while explicit per-instance
settings raise a clear error (the user asked for something specific and
should know it is unavailable).

## Ultimate Goal

Enable the following end-to-end workflow with **zero changes** to
existing ProbPipe modeling code:

```python
import probpipe
from prefect_ray import RayTaskRunner

# Configure once
probpipe.prefect_config.workflow_kind = "task"
probpipe.prefect_config.task_runner = RayTaskRunner()

# Existing bagged posterior code — now automatically distributed
bootstrap_data = BootstrapReplicateDistribution(
    EmpiricalDistribution(jnp.asarray(y_observed))
)
bagged_posterior = condition_on(
    model, bootstrap_data,
    num_results=500, num_warmup=300, random_seed=42,
    n_broadcast_samples=48,
)
```

Each of the 48 bootstrap MCMC fits runs as a Prefect task on Ray workers.
The notebook demonstrating this should be simple, beautiful, and use the
existing bagged posterior example from the ProbPipe tutorial — not a
custom-built scenario.

## Design

### New module: `probpipe/core/config.py`

A lightweight configuration singleton:

```python
class PrefectConfig:
    """Global Prefect orchestration settings.

    Parameters
    ----------
    workflow_kind : ``"task"`` | ``"flow"`` | None
        Default orchestration mode for all ``WorkflowFunction`` instances
        that do not specify their own. Default: ``None`` (no orchestration).
    task_runner : object or None
        Prefect task runner instance (e.g., ``RayTaskRunner()``) passed
        to internally-created flows. Default: ``None`` (Prefect default).
    enabled : bool
        Global kill switch. When ``False``, all orchestration is disabled
        regardless of other settings. Default: ``True``.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Restore all settings to defaults."""
        self._workflow_kind: str | None = None
        self._task_runner: Any = None
        self._enabled: bool = True

    # Properties with validation on workflow_kind, etc.
```

Exposed at package level:

```python
# probpipe/__init__.py
from .core.config import prefect_config
```

### Sentinel pattern in `WorkflowFunction`

A private `_UNSET` sentinel distinguishes "user didn't specify" from
"user explicitly passed `None`":

```python
_UNSET = object()

class WorkflowFunction:
    def __init__(self, ..., workflow_kind=_UNSET, ...):
        self._workflow_kind_raw = workflow_kind
```

### Resolution property: `effective_workflow_kind`

Lazy resolution at call time (not construction time), so config changes
after construction take effect:

```python
@property
def effective_workflow_kind(self) -> str | None:
    # 1. Global kill switch
    if not prefect_config.enabled:
        return None

    # 2. Per-instance explicit override
    if self._workflow_kind_raw is not _UNSET:
        kind = self._workflow_kind_raw
        # Explicit setting + Prefect missing = error
        if kind is not None and task is None:
            raise ImportError(
                "Prefect is required for workflow_kind={kind!r}: "
                "pip install probpipe[prefect]"
            )
        return kind

    # 3. Global default
    kind = prefect_config.workflow_kind
    # Global setting + Prefect missing = graceful fallback
    if kind is not None and task is None:
        return None
    return kind
```

### Task runner threading

The `task_runner` is passed to the Prefect `@flow` decorator in
`_execute_many_prefect_task` and `_execute_many_prefect_flow`:

```python
def _execute_many_prefect_task(self, call_value_list):
    runner = prefect_config.task_runner

    @flow(name=f"{self._name}_map",
          **({"task_runner": runner} if runner is not None else {}))
    def _task_map_flow():
        return self._map_task(call_value_list)

    return _task_map_flow()
```

Similarly for `_broadcast_jax` when wrapping vmap in a flow.

### Module class

The `Module` class also uses `_UNSET` as its default, so child
`@workflow_method` instances inherit the global config unless the module
or method explicitly overrides.

## Action Plan

### Phase 1: Infrastructure (this PR)

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1 | Create `PrefectConfig` class and `prefect_config` singleton | `probpipe/core/config.py` (new) | Properties with validation; `reset()` method for testing |
| 2 | Add `_UNSET` sentinel to `node.py` | `probpipe/core/node.py` | Module-level private constant |
| 3 | Change `WorkflowFunction.__init__` default from `None` to `_UNSET` | `probpipe/core/node.py` | Store as `_workflow_kind_raw` |
| 4 | Add `effective_workflow_kind` property | `probpipe/core/node.py` | Resolution logic: enabled → per-instance → global → graceful fallback |
| 5 | Replace all internal reads of `self._workflow_kind` with `self.effective_workflow_kind` | `probpipe/core/node.py` | `_execute_many`, `_broadcast_jax`, provenance metadata |
| 6 | Thread `prefect_config.task_runner` into `@flow` decorators | `probpipe/core/node.py` | `_execute_many_prefect_task`, `_execute_many_prefect_flow`, and `_broadcast_jax` |
| 7 | Update `Module.__init__` to use `_UNSET` default | `probpipe/core/node.py` | Propagation to child `@workflow_method` preserves _UNSET |
| 8 | Export `prefect_config` from package | `probpipe/__init__.py` | Public API addition |
| 9 | Write tests for config behavior | `tests/test_prefect_config.py` (new) | See test plan below |
| 10 | Verify existing tests pass unchanged | `tests/test_prefect_orchestration.py` | Backward compatibility check |

### Phase 2: Example notebook (follow-up PR)

Rewrite `docs/examples/10_prefect_scalability.ipynb` using the bagged
posterior model from the tutorial. The notebook should:

- Use `probpipe.prefect_config` to enable orchestration globally
- Run the same bagged posterior code from the tutorial, now distributed
- Compare sequential vs. threaded vs. Prefect timing
- Show the global off switch and per-instance override
- Be simple and beautiful — no custom models or synthetic data generation

## Test Plan

### New tests (`tests/test_prefect_config.py`)

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_default_values` | `prefect_config` starts with `workflow_kind=None`, `task_runner=None`, `enabled=True` |
| 2 | `test_global_workflow_kind_applies` | WF without explicit `workflow_kind` uses global config value |
| 3 | `test_per_instance_overrides_global` | Explicit `workflow_kind="flow"` beats global `"task"` |
| 4 | `test_explicit_none_overrides_global` | Explicit `workflow_kind=None` disables Prefect even if global is `"task"` |
| 5 | `test_enabled_false_disables_all` | `enabled=False` → no Prefect even with `workflow_kind="task"` globally |
| 6 | `test_graceful_fallback_global` | Prefect not installed + global `workflow_kind="task"` → silent fallback to `None` |
| 7 | `test_explicit_raises_without_prefect` | Prefect not installed + explicit per-instance `workflow_kind="task"` → `ImportError` |
| 8 | `test_task_runner_threaded_to_flow` | `prefect_config.task_runner` is passed to `@flow(task_runner=...)` |
| 9 | `test_reset` | `reset()` restores all defaults |
| 10 | `test_validation` | Invalid `workflow_kind` value raises `ValueError` |
| 11 | `test_config_change_after_construction` | Config change after WF creation takes effect (lazy resolution) |
| 12 | `test_module_inherits_global` | `Module()` without explicit `workflow_kind` → children use global config |

### Existing tests

All tests in `test_prefect_orchestration.py` pass unchanged — they use
explicit `workflow_kind=` per instance, which is preserved as a
per-instance override.

## Backward Compatibility

**No breaking changes.** The migration is invisible:

| Before | After | Behavior |
|--------|-------|----------|
| `WorkflowFunction(func=f)` | `workflow_kind=_UNSET` → resolves to global (default `None`) | Same: no Prefect |
| `WorkflowFunction(func=f, workflow_kind="task")` | Per-instance override | Same: Prefect task |
| `Module(workflow_kind="task")` | Per-instance override propagated to children | Same |
| `Module()` | `workflow_kind=_UNSET` → children resolve to global | Same: no Prefect |
| `@workflow_function` decorated ops | `workflow_kind=_UNSET` → resolves to global | Same: no Prefect (unless user sets global config) |

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `probpipe/core/config.py` | **New** | `PrefectConfig` class + `prefect_config` singleton |
| `probpipe/core/node.py` | Modified | Sentinel, `effective_workflow_kind`, task runner threading |
| `probpipe/__init__.py` | Modified | Export `prefect_config` |
| `tests/test_prefect_config.py` | **New** | Config behavior tests |

## Open Questions

1. **Should `prefect_config` also support a `parallel` default?**
   Currently `parallel=True` (threading) is orthogonal to Prefect. If
   `workflow_kind` is set, should it automatically supersede `parallel`?
   Proposed: yes — if `effective_workflow_kind` is not `None`, ignore
   `parallel` (Prefect handles distribution).

2. **Context manager API?** Should we support scoped configuration via
   context manager (e.g., `with prefect_config.override(enabled=False):
   ...`)? Proposed: defer to a follow-up — the property-based API is
   sufficient for now.

3. **Environment variable support?** e.g., `PROBPIPE_PREFECT_ENABLED=0`.
   Proposed: defer to a follow-up unless reviewers feel it is essential.
