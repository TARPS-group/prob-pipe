"""Tests for the Prefect fingerprint-caching layer.

Covers:
- Threading: the WorkflowFunction's content fingerprint and the shared cache
  storage are attached to ``WorkflowExecutionConfig`` only on the Prefect paths
  and only when caching is enabled.
- Cache key: ``map_task`` builds a ``cache_key_fn`` from the function
  fingerprint, the environment salt, and a content fingerprint of the inputs,
  so a change to any of them is a miss; the key is order-independent and stable
  across processes; storage is wired to both the task and the cache policy.
- End-to-end: a real Prefect run confirms a re-run hits the cache (the user
  function is not re-executed) while a changed input re-executes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import probpipe
from probpipe import CacheMode
from probpipe.core._fingerprint import fingerprint
from probpipe.core._workflow_execution import (
    CacheKeyFnPolicy,
    WorkflowExecutionConfig,
    _cache_task_options,
)
from probpipe.core.node import WorkflowFunction

prefect_required = pytest.mark.skipif(CacheKeyFnPolicy is None, reason="requires prefect")


@pytest.fixture
def caching_on():
    """Enable FINGERPRINT caching for the test (autouse conftest resets after)."""
    probpipe.prefect_config.cache_mode = CacheMode.FINGERPRINT
    yield


def _wf(func):
    return WorkflowFunction(func=func, dispatch="sequential", seed=0)


class TestExecutionConfigDefault:
    def test_func_fingerprint_defaults_none(self):
        cfg = WorkflowExecutionConfig(mode="sequential")
        assert cfg.func_fingerprint is None


class TestFuncFingerprintThreading:
    def test_none_when_caching_off(self):
        # Default cache_mode is OFF, so even a Prefect-mode config carries no
        # fingerprint and the non-caching path is unchanged.
        wf = _wf(lambda x: x)
        cfg = wf._make_execution_config(mode="prefect_task")
        assert cfg.func_fingerprint is None

    def test_set_when_caching_on_and_prefect(self, caching_on):
        def f(x):
            return x + 1

        wf = _wf(f)
        cfg = wf._make_execution_config(mode="prefect_task")
        assert cfg.func_fingerprint == fingerprint(wf)
        assert isinstance(cfg.func_fingerprint, str)
        assert len(cfg.func_fingerprint) == 16

    def test_set_on_prefect_flow_mode(self, caching_on):
        wf = _wf(lambda x: x)
        cfg = wf._make_execution_config(mode="prefect_flow")
        assert cfg.func_fingerprint == fingerprint(wf)

    def test_none_for_non_prefect_mode_even_when_caching_on(self, caching_on):
        # Caching only applies to Prefect execution; local dispatch never caches.
        wf = _wf(lambda x: x)
        assert wf._make_execution_config(mode="sequential").func_fingerprint is None
        assert wf._make_execution_config(mode="thread").func_fingerprint is None

    def test_distinguishes_function_bodies(self, caching_on):
        # Two functions differing only in a literal constant must key differently,
        # or a changed function body would silently reuse a stale cached result.
        def f(x):
            return x + 1

        def g(x):
            return x + 2

        cfg_f = _wf(f)._make_execution_config(mode="prefect_task")
        cfg_g = _wf(g)._make_execution_config(mode="prefect_task")
        assert cfg_f.func_fingerprint != cfg_g.func_fingerprint

    def test_memoized_stable_across_calls(self, caching_on):
        wf = _wf(lambda x: x)
        first = wf._make_execution_config(mode="prefect_task").func_fingerprint
        second = wf._make_execution_config(mode="prefect_task").func_fingerprint
        assert first is not None
        assert first == second


# ===========================================================================
# Step 4: cache_key_fn + task options built in map_task
# ===========================================================================


def _cfg(func_fingerprint, storage=None):
    return WorkflowExecutionConfig(
        mode="prefect_task", func_fingerprint=func_fingerprint, cache_result_storage=storage
    )


def _key(func_fingerprint, parameters, storage=None):
    """Extract and invoke the cache_key_fn that map_task would attach."""
    options = _cache_task_options(_cfg(func_fingerprint, storage))
    return options["cache_policy"].cache_key_fn(None, parameters)


@prefect_required
class TestCacheTaskOptions:
    def test_empty_when_caching_off(self):
        # func_fingerprint None (caching off) → no task options, task unchanged.
        assert _cache_task_options(_cfg(None)) == {}

    def test_options_present_when_caching_on(self):
        options = _cache_task_options(_cfg("abc123abc123abc1"))
        assert isinstance(options["cache_policy"], CacheKeyFnPolicy)
        assert options["persist_result"] is True

    def test_no_result_storage_when_storage_none(self):
        options = _cache_task_options(_cfg("abc123abc123abc1", storage=None))
        assert "result_storage" not in options
        assert options["cache_policy"].key_storage is None

    def test_storage_wired_to_both_surfaces(self):
        # One config field must feed both result_storage (the task) and
        # key_storage (the policy) or other workers miss on a shared store.
        options = _cache_task_options(_cfg("abc123abc123abc1", storage="/tmp/cache_store"))
        assert options["result_storage"] == "/tmp/cache_store"
        assert options["cache_policy"].key_storage == "/tmp/cache_store"


# Prefect binds a **kwargs task's arguments as {"kwargs": {...}}.
PARAMS = {"kwargs": {"x": 3, "y": 4}}


@prefect_required
class TestCacheKey:
    def test_deterministic_for_same_inputs(self):
        assert _key("fp0000000000000a", PARAMS) == _key("fp0000000000000a", PARAMS)

    def test_changes_with_inputs(self):
        # A changed input must be a cache miss.
        other = {"kwargs": {"x": 3, "y": 5}}
        assert _key("fp0000000000000a", PARAMS) != _key("fp0000000000000a", other)

    def test_order_independent_over_kwargs(self):
        # Prefect returns kwargs unordered; the key must not depend on order.
        a = {"kwargs": {"x": 3, "y": 4}}
        b = {"kwargs": {"y": 4, "x": 3}}
        assert _key("fp0000000000000a", a) == _key("fp0000000000000a", b)

    def test_changes_with_func_fingerprint(self):
        # A changed function body (different func fingerprint) must be a miss.
        assert _key("fp0000000000000a", PARAMS) != _key("fp0000000000000b", PARAMS)

    def test_changes_with_environment_salt(self, monkeypatch):
        # An environment change (version bump / x64 toggle) must be a miss.
        import probpipe.core._workflow_execution as we

        monkeypatch.setattr(we, "environment_salt", lambda: "env=A")
        key_a = _key("fp0000000000000a", PARAMS)
        monkeypatch.setattr(we, "environment_salt", lambda: "env=B")
        key_b = _key("fp0000000000000a", PARAMS)
        assert key_a != key_b

    def test_cross_process_deterministic(self):
        # The key must be identical in a fresh process (no PYTHONHASHSEED / id
        # leakage), or cross-machine caching never hits.
        code = textwrap.dedent(
            """
            from probpipe.core._workflow_execution import WorkflowExecutionConfig, _cache_task_options
            cfg = WorkflowExecutionConfig(mode="prefect_task", func_fingerprint="fp0000000000000a")
            fn = _cache_task_options(cfg)["cache_policy"].cache_key_fn
            print(fn(None, {"kwargs": {"x": 3, "y": 4}}))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout.strip()
        assert out == _key("fp0000000000000a", PARAMS)


# ===========================================================================
# Step 4: end-to-end cache behavior through a real Prefect run (slow)
# ===========================================================================


@pytest.mark.prefect
class TestCacheEndToEnd:
    @pytest.fixture(autouse=True)
    def _prefect_harness(self):
        # In-process temporary Prefect server so caching persists to a clean,
        # isolated store (mirrors tests/core/test_prefect_orchestration.py).
        prefect_testing = pytest.importorskip("prefect.testing.utilities")
        harness = prefect_testing.prefect_test_harness(server_startup_timeout=60)
        try:
            harness.__enter__()
        except Exception as e:
            pytest.skip(f"Prefect server unavailable: {e}")
        try:
            yield
        finally:
            harness.__exit__(None, None, None)

    def test_rerun_hits_and_changed_input_misses(self, tmp_path):
        counter = tmp_path / "exec_log.txt"
        counter.write_text("")

        def add_one(x):
            with open(counter, "a") as fh:
                fh.write("run\n")
            return x + 1

        from probpipe import WorkflowKind

        wf = WorkflowFunction(
            func=add_one, workflow_kind=WorkflowKind.TASK, dispatch="sequential", seed=0
        )
        probpipe.prefect_config.cache_mode = CacheMode.FINGERPRINT

        def n_runs():
            with open(counter) as fh:
                return sum(1 for ln in fh if ln.strip())

        wf(x=3)  # miss → runs
        assert n_runs() == 1
        wf(x=3)  # hit → skipped
        assert n_runs() == 1
        wf(x=4)  # different input → miss → runs
        assert n_runs() == 2
        wf(x=3)  # hit again → skipped
        assert n_runs() == 2
