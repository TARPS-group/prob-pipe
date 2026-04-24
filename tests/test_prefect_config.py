"""Tests for global Prefect configuration (WorkflowKind + PrefectConfig).

Exercises:
- WorkflowKind enum values and validation
- PrefectConfig defaults, setters, reset
- effective_workflow_kind resolution (per-instance vs global vs auto-detect)
- Graceful fallback when Prefect is not installed
- Task runner auto-detection and explicit override
- Legacy string / None conversion
- Module inherits global config
"""

import pytest

from probpipe.core.config import (
    WorkflowKind,
    PrefectConfig,
    prefect_config,
    _auto_detect_task_runner,
)


# ---------------------------------------------------------------------------
# WorkflowKind enum
# ---------------------------------------------------------------------------

class TestWorkflowKindEnum:
    """Verify enum has the expected members and values."""

    def test_members(self):
        assert set(WorkflowKind) == {
            WorkflowKind.DEFAULT,
            WorkflowKind.OFF,
            WorkflowKind.TASK,
            WorkflowKind.FLOW,
        }

    def test_values(self):
        assert WorkflowKind.DEFAULT.value == "default"
        assert WorkflowKind.OFF.value == "off"
        assert WorkflowKind.TASK.value == "task"
        assert WorkflowKind.FLOW.value == "flow"

    def test_construct_from_string(self):
        assert WorkflowKind("task") is WorkflowKind.TASK
        assert WorkflowKind("flow") is WorkflowKind.FLOW
        assert WorkflowKind("off") is WorkflowKind.OFF
        assert WorkflowKind("default") is WorkflowKind.DEFAULT

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            WorkflowKind("banana")


# ---------------------------------------------------------------------------
# PrefectConfig singleton
# ---------------------------------------------------------------------------

class TestPrefectConfigDefaults:
    """Verify default values and reset behavior."""

    def test_default_workflow_kind(self):
        pc = PrefectConfig()
        assert pc.workflow_kind is WorkflowKind.DEFAULT

    def test_default_task_runner(self):
        pc = PrefectConfig()
        assert pc.task_runner is None

    def test_reset_restores_defaults(self):
        pc = PrefectConfig()
        pc.workflow_kind = WorkflowKind.TASK
        pc.task_runner = "something"
        pc.reset()
        assert pc.workflow_kind is WorkflowKind.DEFAULT
        assert pc.task_runner is None


class TestPrefectConfigValidation:
    """Verify setters reject invalid values."""

    def test_workflow_kind_rejects_string(self):
        pc = PrefectConfig()
        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            pc.workflow_kind = "task"

    def test_workflow_kind_rejects_none(self):
        pc = PrefectConfig()
        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            pc.workflow_kind = None

    def test_workflow_kind_rejects_int(self):
        pc = PrefectConfig()
        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            pc.workflow_kind = 42

    def test_workflow_kind_accepts_enum(self):
        pc = PrefectConfig()
        for kind in WorkflowKind:
            pc.workflow_kind = kind
            assert pc.workflow_kind is kind


class TestTaskRunnerAutoDetection:
    """Verify auto-detection probes for installed runner packages."""

    def test_returns_none_when_nothing_installed(self, monkeypatch):
        # Block both runner imports
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("prefect_ray", "prefect_dask"):
                raise ImportError(f"mocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert _auto_detect_task_runner() is None

    def test_explicit_runner_overrides_auto(self):
        pc = PrefectConfig()
        sentinel = object()
        pc.task_runner = sentinel
        assert pc.resolve_task_runner() is sentinel

    def test_resolve_with_no_explicit_runner(self, monkeypatch):
        # When no explicit runner and no runner packages, resolve returns None
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("prefect_ray", "prefect_dask"):
                raise ImportError(f"mocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        pc = PrefectConfig()
        assert pc.resolve_task_runner() is None


# ---------------------------------------------------------------------------
# effective_workflow_kind resolution
# ---------------------------------------------------------------------------

class TestEffectiveWorkflowKind:
    """Test the resolution logic on WorkflowFunction instances."""

    @pytest.fixture(autouse=True)
    def _reset_config(self):
        """Reset global config before and after each test."""
        prefect_config.reset()
        yield
        prefect_config.reset()

    def test_default_resolves_to_task_when_prefect_installed(self):
        """DEFAULT + Prefect installed → TASK."""
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, vectorize="loop", seed=0)
        # Prefect is installed in test env (importorskip at module level
        # is not used here, but the fixture guarantees config is DEFAULT)
        kind = wf.effective_workflow_kind
        # If prefect is importable, should be TASK; if not, OFF
        import probpipe.core.node as node_mod
        if node_mod.task is not None:
            assert kind is WorkflowKind.TASK
        else:
            assert kind is WorkflowKind.OFF

    def test_default_resolves_to_off_when_prefect_missing(self, monkeypatch):
        """DEFAULT + Prefect not installed → OFF (graceful)."""
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, vectorize="loop", seed=0)
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_explicit_task_overrides_global(self):
        """Per-instance TASK beats global OFF."""
        prefect_config.workflow_kind = WorkflowKind.OFF

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(
            func=noop, workflow_kind=WorkflowKind.TASK, vectorize="loop", seed=0,
        )
        import probpipe.core.node as node_mod
        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.TASK

    def test_explicit_off_overrides_global_task(self):
        """Per-instance OFF beats global TASK."""
        prefect_config.workflow_kind = WorkflowKind.TASK

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(
            func=noop, workflow_kind=WorkflowKind.OFF, vectorize="loop", seed=0,
        )
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_explicit_task_warns_without_prefect(self, monkeypatch):
        """Per-instance TASK + Prefect missing → warning + OFF."""
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(
            func=noop, workflow_kind=WorkflowKind.TASK, vectorize="loop", seed=0,
        )
        with pytest.warns(UserWarning, match="Prefect is not installed"):
            kind = wf.effective_workflow_kind
        assert kind is WorkflowKind.OFF

    def test_global_task_falls_back_without_prefect(self, monkeypatch):
        """Global TASK + Prefect missing → OFF (graceful)."""
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        prefect_config.workflow_kind = WorkflowKind.TASK

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, vectorize="loop", seed=0)
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_global_flow_applies_to_default_instance(self):
        """Global FLOW → DEFAULT instance resolves to FLOW."""
        prefect_config.workflow_kind = WorkflowKind.FLOW

        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, vectorize="loop", seed=0)
        import probpipe.core.node as node_mod
        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.FLOW

    def test_config_change_after_construction(self):
        """Config change after WF creation takes effect (lazy resolution)."""
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, vectorize="loop", seed=0)
        prefect_config.workflow_kind = WorkflowKind.OFF
        assert wf.effective_workflow_kind is WorkflowKind.OFF

        prefect_config.workflow_kind = WorkflowKind.FLOW
        import probpipe.core.node as node_mod
        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.FLOW


# ---------------------------------------------------------------------------
# Legacy string / None conversion
# ---------------------------------------------------------------------------

class TestLegacyConversion:
    """Verify old-style workflow_kind values are auto-converted."""

    def test_string_task_converts(self):
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, workflow_kind="task", vectorize="loop", seed=0)
        assert wf._workflow_kind_raw is WorkflowKind.TASK

    def test_string_flow_converts(self):
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, workflow_kind="flow", vectorize="loop", seed=0)
        assert wf._workflow_kind_raw is WorkflowKind.FLOW

    def test_none_converts_to_off(self):
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        wf = WorkflowFunction(func=noop, workflow_kind=None, vectorize="loop", seed=0)
        assert wf._workflow_kind_raw is WorkflowKind.OFF

    def test_invalid_string_raises(self):
        from probpipe.core.node import WorkflowFunction

        def noop(x):
            return x

        with pytest.raises(ValueError):
            WorkflowFunction(func=noop, workflow_kind="banana", vectorize="loop", seed=0)


# ---------------------------------------------------------------------------
# Module inherits global config
# ---------------------------------------------------------------------------

class TestModuleInheritsConfig:
    """Module with DEFAULT propagates global config to children."""

    @pytest.fixture(autouse=True)
    def _reset_config(self):
        prefect_config.reset()
        yield
        prefect_config.reset()

    def test_module_default_passes_default_to_children(self):
        from probpipe.core.node import Module, workflow_method

        class MyModule(Module):
            @workflow_method
            def step(self, x):
                return x + 1

        mod = MyModule()
        # The child WorkflowFunction should have DEFAULT as raw value
        assert mod.step._workflow_kind_raw is WorkflowKind.DEFAULT

    def test_module_explicit_off_passes_off_to_children(self):
        from probpipe.core.node import Module, workflow_method

        class MyModule(Module):
            @workflow_method
            def step(self, x):
                return x + 1

        mod = MyModule(workflow_kind=WorkflowKind.OFF)
        assert mod.step._workflow_kind_raw is WorkflowKind.OFF

    def test_module_legacy_string_converts(self):
        from probpipe.core.node import Module, workflow_method

        class MyModule(Module):
            @workflow_method
            def step(self, x):
                return x + 1

        mod = MyModule(workflow_kind="task")
        assert mod.step._workflow_kind_raw is WorkflowKind.TASK
