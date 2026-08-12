"""Tests for global Prefect configuration (WorkflowKind + PrefectConfig).

Exercises:
- WorkflowKind enum values and validation
- PrefectConfig defaults, setters, reset
- effective_workflow_kind resolution (per-instance vs global)
- Graceful fallback when Prefect is not installed
- Task runner auto-detection and explicit override
- Strict Function / Module workflow_kind validation
- Module inherits global config
- PROBPIPE_WORKFLOW_KIND environment-variable override
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from probpipe.core.config import (
    _WORKFLOW_KIND_ENV_VAR,
    PrefectConfig,
    WorkflowKind,
    _auto_detect_task_runner,
    prefect_config,
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

    def test_default_workflow_kind(self, monkeypatch):
        monkeypatch.delenv(_WORKFLOW_KIND_ENV_VAR, raising=False)
        pc = PrefectConfig()
        assert pc.workflow_kind is WorkflowKind.OFF

    def test_default_task_runner(self):
        pc = PrefectConfig()
        assert pc.task_runner is None

    def test_reset_restores_defaults(self, monkeypatch):
        monkeypatch.delenv(_WORKFLOW_KIND_ENV_VAR, raising=False)
        pc = PrefectConfig()
        pc.workflow_kind = WorkflowKind.TASK
        pc.task_runner = "something"
        pc.reset()
        assert pc.workflow_kind is WorkflowKind.OFF
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

        monkeypatch.delitem(sys.modules, "prefect_ray", raising=False)
        monkeypatch.delitem(sys.modules, "prefect_dask", raising=False)

        def mock_import(name, *args, **kwargs):
            if name in ("prefect_ray", "prefect_dask"):
                raise ImportError(f"mocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert _auto_detect_task_runner() is None

    def test_returns_ray_runner_when_prefect_ray_installed(self, monkeypatch):
        class FakeRayTaskRunner:
            pass

        fake_ray = types.ModuleType("prefect_ray")
        fake_ray.RayTaskRunner = FakeRayTaskRunner
        monkeypatch.setitem(sys.modules, "prefect_ray", fake_ray)
        monkeypatch.delitem(sys.modules, "prefect_dask", raising=False)

        assert isinstance(_auto_detect_task_runner(), FakeRayTaskRunner)

    def test_prefers_ray_runner_over_dask_runner(self, monkeypatch):
        class FakeRayTaskRunner:
            pass

        class FakeDaskTaskRunner:
            pass

        fake_ray = types.ModuleType("prefect_ray")
        fake_ray.RayTaskRunner = FakeRayTaskRunner
        fake_dask = types.ModuleType("prefect_dask")
        fake_dask.DaskTaskRunner = FakeDaskTaskRunner

        monkeypatch.setitem(sys.modules, "prefect_ray", fake_ray)
        monkeypatch.setitem(sys.modules, "prefect_dask", fake_dask)

        runner = _auto_detect_task_runner()
        assert isinstance(runner, FakeRayTaskRunner)
        assert not isinstance(runner, FakeDaskTaskRunner)

    def test_explicit_runner_overrides_auto(self):
        pc = PrefectConfig()
        sentinel = object()
        pc.task_runner = sentinel
        assert pc.resolve_task_runner() is sentinel

    def test_resolve_with_no_explicit_runner(self, monkeypatch):
        # When no explicit runner and no runner packages, resolve returns None
        import builtins

        real_import = builtins.__import__

        monkeypatch.delitem(sys.modules, "prefect_ray", raising=False)
        monkeypatch.delitem(sys.modules, "prefect_dask", raising=False)

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
    """Test the resolution logic on Function instances."""

    @pytest.fixture(autouse=True)
    def _reset_config(self, monkeypatch):
        """Reset global config before and after each test.

        The env-var override is unset so ``reset()`` lands at the
        shipped default of ``OFF``; individual tests that need a
        different starting state assign to ``prefect_config.workflow_kind``
        explicitly.
        """
        monkeypatch.delenv(_WORKFLOW_KIND_ENV_VAR, raising=False)
        prefect_config.reset()
        yield
        prefect_config.reset()

    def test_default_resolves_to_off(self):
        """DEFAULT global → OFF (Prefect is opt-in, not auto-detected).

        The shipped default is OFF regardless of Prefect importability,
        so this case subsumes the prior `prefect missing` variant.
        """
        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(func=noop, dispatch="sequential", seed=0)
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_explicit_task_overrides_global(self):
        """Per-instance TASK beats global OFF."""
        prefect_config.workflow_kind = WorkflowKind.OFF

        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(
            func=noop,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            seed=0,
        )
        import probpipe.core.node as node_mod

        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.TASK

    def test_explicit_off_overrides_global_task(self):
        """Per-instance OFF beats global TASK."""
        prefect_config.workflow_kind = WorkflowKind.TASK

        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(
            func=noop,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
            seed=0,
        )
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_explicit_task_warns_without_prefect(self, monkeypatch):
        """Per-instance TASK + Prefect missing → warning + OFF."""
        import probpipe.core.node as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(
            func=noop,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            seed=0,
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

        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(func=noop, dispatch="sequential", seed=0)
        assert wf.effective_workflow_kind is WorkflowKind.OFF

    def test_global_flow_applies_to_default_instance(self):
        """Global FLOW → DEFAULT instance resolves to FLOW."""
        prefect_config.workflow_kind = WorkflowKind.FLOW

        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(func=noop, dispatch="sequential", seed=0)
        import probpipe.core.node as node_mod

        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.FLOW

    def test_config_change_after_construction(self):
        """Config change after WF creation takes effect (lazy resolution)."""
        from probpipe.core.node import Function

        def noop(x):
            return x

        wf = Function(func=noop, dispatch="sequential", seed=0)
        prefect_config.workflow_kind = WorkflowKind.OFF
        assert wf.effective_workflow_kind is WorkflowKind.OFF

        prefect_config.workflow_kind = WorkflowKind.FLOW
        import probpipe.core.node as node_mod

        if node_mod.task is not None:
            assert wf.effective_workflow_kind is WorkflowKind.FLOW


# ---------------------------------------------------------------------------
# Strict constructor workflow_kind validation
# ---------------------------------------------------------------------------


class TestWorkflowKindConstructorValidation:
    """Verify constructors reject old-style workflow_kind values."""

    def test_function_rejects_string(self):
        from probpipe.core.node import Function

        def noop(x):
            return x

        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            Function(
                func=noop,
                workflow_kind="task",
                dispatch="sequential",
                seed=0,
            )

    def test_function_rejects_none(self):
        from probpipe.core.node import Function

        def noop(x):
            return x

        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            Function(
                func=noop,
                workflow_kind=None,
                dispatch="sequential",
                seed=0,
            )

    def test_module_rejects_string(self):
        from probpipe.core.node import Module

        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            Module(workflow_kind="task")

    def test_module_rejects_none(self):
        from probpipe.core.node import Module

        with pytest.raises(TypeError, match="WorkflowKind enum member"):
            Module(workflow_kind=None)


# ---------------------------------------------------------------------------
# Module inherits global config
# ---------------------------------------------------------------------------


class TestModuleInheritsConfig:
    """Module with DEFAULT propagates global config to children."""

    @pytest.fixture(autouse=True)
    def _reset_config(self, monkeypatch):
        monkeypatch.delenv(_WORKFLOW_KIND_ENV_VAR, raising=False)
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
        # The child Function should have DEFAULT as raw value
        assert mod.step._workflow_kind_raw is WorkflowKind.DEFAULT

    def test_module_explicit_off_passes_off_to_children(self):
        from probpipe.core.node import Module, workflow_method

        class MyModule(Module):
            @workflow_method
            def step(self, x):
                return x + 1

        mod = MyModule(workflow_kind=WorkflowKind.OFF)
        assert mod.step._workflow_kind_raw is WorkflowKind.OFF

    def test_module_explicit_task_passes_task_to_children(self):
        from probpipe.core.node import Module, workflow_method

        class MyModule(Module):
            @workflow_method
            def step(self, x):
                return x + 1

        mod = MyModule(workflow_kind=WorkflowKind.TASK)
        assert mod.step._workflow_kind_raw is WorkflowKind.TASK


# ---------------------------------------------------------------------------
# PROBPIPE_WORKFLOW_KIND environment variable
# ---------------------------------------------------------------------------


class TestEnvVarOverride:
    """The ``PROBPIPE_WORKFLOW_KIND`` env var sets the initial workflow_kind."""

    def test_unset_falls_back_to_off(self, monkeypatch):
        monkeypatch.delenv(_WORKFLOW_KIND_ENV_VAR, raising=False)
        pc = PrefectConfig()
        assert pc.workflow_kind is WorkflowKind.OFF

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("off", WorkflowKind.OFF),
            ("task", WorkflowKind.TASK),
            ("flow", WorkflowKind.FLOW),
            ("default", WorkflowKind.DEFAULT),
        ],
    )
    def test_valid_value_sets_initial_kind(self, monkeypatch, value, expected):
        monkeypatch.setenv(_WORKFLOW_KIND_ENV_VAR, value)
        pc = PrefectConfig()
        assert pc.workflow_kind is expected

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(_WORKFLOW_KIND_ENV_VAR, "TASK")
        pc = PrefectConfig()
        assert pc.workflow_kind is WorkflowKind.TASK

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv(_WORKFLOW_KIND_ENV_VAR, "banana")
        with pytest.raises(ValueError, match="banana"):
            PrefectConfig()

    def test_invalid_value_fails_at_import(self):
        """A bad env var should fail loudly at ``import probpipe`` time.

        Uses a subprocess so the module-level singleton instantiation
        runs under the bad env var (the in-process singleton was already
        constructed with the current env var when this test suite loaded).
        """
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", "import probpipe"],
            env={**os.environ, _WORKFLOW_KIND_ENV_VAR: "banana"},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "banana" in result.stderr
        assert "PROBPIPE_WORKFLOW_KIND" in result.stderr

    def test_explicit_assignment_overrides_env(self, monkeypatch):
        monkeypatch.setenv(_WORKFLOW_KIND_ENV_VAR, "task")
        pc = PrefectConfig()
        assert pc.workflow_kind is WorkflowKind.TASK
        pc.workflow_kind = WorkflowKind.OFF
        assert pc.workflow_kind is WorkflowKind.OFF

    def test_reset_re_reads_env(self, monkeypatch):
        monkeypatch.setenv(_WORKFLOW_KIND_ENV_VAR, "task")
        pc = PrefectConfig()
        pc.workflow_kind = WorkflowKind.OFF
        pc.reset()
        assert pc.workflow_kind is WorkflowKind.TASK
