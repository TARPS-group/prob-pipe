"""Tests for Prefect orchestration in Function.

Exercises all Prefect dispatch paths:
- workflow_kind=WorkflowKind.TASK with sequential, thread-option, and JAX dispatch
- workflow_kind=WorkflowKind.FLOW with sequential and JAX dispatch
- Import guard when Prefect is unavailable
- Provenance metadata includes orchestration info

Requires ``prefect>=3`` (installed via ``pip install probpipe[prefect]``).
Uses ``prefect_test_harness()`` for an in-process temporary server.
"""

from __future__ import annotations

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe.core._workflow_broker as broker_mod
from probpipe import Normal, WorkflowKind, sample, workflow_run
from probpipe.core.node import Function

prefect_testing = pytest.importorskip("prefect.testing.utilities")
prefect_test_harness = prefect_testing.prefect_test_harness
prefect_settings = pytest.importorskip("prefect.settings")


@pytest.fixture(scope="module", autouse=True)
def _prefect_harness():
    """Start a temporary in-process Prefect server for the entire module.

    The default ephemeral server startup timeout is only 20 s, which is
    often too short on resource-constrained CI runners.  We raise it to
    60 s.  If the server still can't start, skip the module gracefully
    rather than failing the entire run with 17 ERRORs.
    """
    harness = prefect_test_harness(server_startup_timeout=60)
    try:
        harness.__enter__()
    except Exception as e:
        pytest.skip(f"Prefect server unavailable: {e}")
    try:
        yield
    finally:
        harness.__exit__(None, None, None)


@pytest.fixture
def normal_dist():
    return Normal(loc=1.0, scale=0.5, name="x")


# ---------------------------------------------------------------------------
# Helper functions for workflows
# ---------------------------------------------------------------------------


def add_one(x: jnp.ndarray) -> jnp.ndarray:
    return x + 1.0


def double_it(x: jnp.ndarray) -> jnp.ndarray:
    return x * 2.0


def sum_xy(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return x + y


def _draw_standard_normal():
    return sample(Normal(loc=0.0, scale=1.0, name="x"))


_THREADED_DRAW = Function(
    func=_draw_standard_normal,
    dispatch="thread",
    max_workers=1,
    name="threaded_draw",
)


def _call_threaded_draw():
    return _THREADED_DRAW()


def _draw_under_nested_seed():
    with workflow_run(seed=91):
        return _draw_standard_normal()


_PREFECT_RETRY_KEY_WORDS: list[tuple[int, int]] = []


def _claim_key_and_fail_once():
    key = broker_mod._resolve_automatic_key(
        None,
        broker_mod._singleton_effect_plan(
            operation_kind="prefect-retry-conformance",
            execution_mode="sampled",
            sample_shape=(),
        ),
    )
    words = tuple(int(word) for word in jax.random.key_data(key))
    _PREFECT_RETRY_KEY_WORDS.append(words)
    if len(_PREFECT_RETRY_KEY_WORDS) == 1:
        raise RuntimeError("retry conformance failure")
    return words


# ---------------------------------------------------------------------------
# Cross-route RNG conformance
# ---------------------------------------------------------------------------


class TestPrefectRngConformance:
    def test_lifted_samples_match_local_thread_and_real_prefect(self, normal_dist):
        workflows = (
            (
                Function(
                    func=add_one,
                    workflow_kind=WorkflowKind.OFF,
                    dispatch="sequential",
                    n_broadcast_samples=6,
                ),
                False,
            ),
            (
                Function(
                    func=add_one,
                    workflow_kind=WorkflowKind.OFF,
                    dispatch="thread",
                    max_workers=2,
                    n_broadcast_samples=6,
                ),
                False,
            ),
            (
                Function(
                    func=add_one,
                    workflow_kind=WorkflowKind.TASK,
                    dispatch="sequential",
                    n_broadcast_samples=6,
                ),
                False,
            ),
            (
                Function(
                    func=add_one,
                    workflow_kind=WorkflowKind.TASK,
                    dispatch="thread",
                    max_workers=2,
                    n_broadcast_samples=6,
                ),
                True,
            ),
            (
                Function(
                    func=add_one,
                    workflow_kind=WorkflowKind.FLOW,
                    dispatch="sequential",
                    n_broadcast_samples=6,
                ),
                False,
            ),
        )

        samples = []
        for workflow, warns_about_local_controls in workflows:
            warning_context = (
                pytest.warns(UserWarning, match="do not control Prefect scheduling")
                if warns_about_local_controls
                else nullcontext()
            )
            with warning_context, workflow_run(seed=17):
                result = workflow(x=normal_dist)
            samples.append(np.asarray(result.samples))

        for sample_values in samples[1:]:
            np.testing.assert_array_equal(sample_values, samples[0])

    def test_nested_seed_matches_local_and_real_prefect_for_any_outer_seed(self):
        local = Function(
            func=_draw_under_nested_seed,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
        )
        remote = Function(
            func=_draw_under_nested_seed,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )

        results = []
        for workflow, outer_seed in (
            (local, 1),
            (local, 2),
            (remote, 1),
            (remote, 2),
        ):
            with workflow_run(seed=outer_seed):
                results.append(np.asarray(workflow()))

        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])

    def test_rootless_task_coordinates_nested_managed_thread(self):
        local = Function(
            func=_call_threaded_draw,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
            name="nested_thread_owner",
        )
        remote = Function(
            func=_call_threaded_draw,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            name="nested_thread_owner",
        )

        with workflow_run(seed=17):
            local_result = local()
        with workflow_run(seed=17):
            remote_result = remote()

        np.testing.assert_array_equal(
            remote_result["sample"],
            local_result["sample"],
        )
        assert remote_result.provenance is not None
        randomness = remote_result.provenance.controls["randomness"]
        replay = remote_result.provenance.controls["replay"]
        rng_origin = remote_result.provenance.diagnostics["rng_origin"]
        assert randomness["root_words"] == [0, 17]
        assert replay["standalone"] == {
            "eligibility": "nested_workflow_rng_execution",
            "restriction": "nested_automatic_function",
        }
        assert rng_origin == {
            "context_kind": "seeded_run",
            "root_source": "user_seed",
            "supplied_seed": 17,
        }

    def test_real_prefect_retry_reuses_key_and_commits_one_effect(self):
        _PREFECT_RETRY_KEY_WORDS.clear()
        workflow = Function(
            func=_claim_key_and_fail_once,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )

        with (
            prefect_settings.temporary_settings(
                updates={
                    prefect_settings.PREFECT_TASKS_DEFAULT_RETRIES: 1,
                    prefect_settings.PREFECT_TASKS_DEFAULT_RETRY_DELAY_SECONDS: 0,
                }
            ),
            workflow_run(seed=17),
        ):
            result = workflow()

        assert len(_PREFECT_RETRY_KEY_WORDS) == 2
        assert _PREFECT_RETRY_KEY_WORDS[0] == _PREFECT_RETRY_KEY_WORDS[1]
        randomness = result.provenance.controls["randomness"]
        assert randomness["expected_event_count"] == 1
        assert len(randomness["events"]) == 1


# ---------------------------------------------------------------------------
# workflow_kind=WorkflowKind.TASK with sequential dispatch
# ---------------------------------------------------------------------------


class TestPrefectTaskRowWise:
    """Exercises Prefect task execution via row-wise dispatch."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            n_broadcast_samples=30,
        )
        with workflow_run(seed=0):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.num_atoms == 30

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            n_broadcast_samples=200,
        )
        with workflow_run(seed=1):
            result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 + 1.0)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)),
            2.0,
            atol=0.15,
        )

    def test_multiple_broadcast_args(self, normal_dist):
        wf = Function(
            func=sum_xy,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            n_broadcast_samples=30,
        )
        d2 = Normal(loc=2.0, scale=0.3, name="y")
        with workflow_run(seed=2):
            result = wf(x=normal_dist, y=d2)
        assert hasattr(result, "samples")
        assert result.num_atoms == 30


# ---------------------------------------------------------------------------
# workflow_kind=WorkflowKind.FLOW with sequential dispatch
# ---------------------------------------------------------------------------


class TestPrefectFlowRowWise:
    """Exercises Prefect flow execution via row-wise dispatch."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            func=double_it,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
            n_broadcast_samples=25,
        )
        with workflow_run(seed=10):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.num_atoms == 25

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            func=double_it,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
            n_broadcast_samples=200,
        )
        with workflow_run(seed=11):
            result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 * 2)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)),
            2.0,
            atol=0.15,
        )


# ---------------------------------------------------------------------------
# workflow_kind=WorkflowKind.TASK with JAX dispatch
# ---------------------------------------------------------------------------


class TestPrefectTaskJax:
    """Exercises the Prefect-wrapped distribution-broadcast JAX path."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="jax",
            n_broadcast_samples=30,
        )
        with workflow_run(seed=20):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.num_atoms == 30

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="jax",
            n_broadcast_samples=200,
        )
        with workflow_run(seed=21):
            result = wf(x=normal_dist)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)),
            2.0,
            atol=0.15,
        )


# ---------------------------------------------------------------------------
# workflow_kind=WorkflowKind.FLOW with JAX dispatch
# ---------------------------------------------------------------------------


class TestPrefectFlowJax:
    """Exercises Prefect flow-wrapped jax.vmap path."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            func=double_it,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="jax",
            n_broadcast_samples=25,
        )
        with workflow_run(seed=30):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.num_atoms == 25

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            func=double_it,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="jax",
            n_broadcast_samples=200,
        )
        with workflow_run(seed=31):
            result = wf(x=normal_dist)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)),
            2.0,
            atol=0.15,
        )


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------


class TestPrefectProvenance:
    """Verify provenance includes orchestration info."""

    def test_task_provenance(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            n_broadcast_samples=20,
        )
        with workflow_run(seed=40):
            result = wf(x=normal_dist)
        assert result.provenance is not None
        assert result.provenance.operation == "broadcast"
        assert result.provenance.metadata["orchestrate"] == "task"
        assert result.provenance.metadata["n_samples"] == 20

    def test_flow_provenance(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
            n_broadcast_samples=20,
        )
        with workflow_run(seed=41):
            result = wf(x=normal_dist)
        assert result.provenance is not None
        assert result.provenance.metadata["orchestrate"] == "flow"

    def test_no_orchestration_provenance(self, normal_dist):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
            n_broadcast_samples=20,
        )
        with workflow_run(seed=42):
            result = wf(x=normal_dist)
        assert result.provenance is not None
        assert result.provenance.metadata["orchestrate"] == "off"


# ---------------------------------------------------------------------------
# Non-broadcast calls with workflow_kind
# ---------------------------------------------------------------------------


class TestPrefectNonBroadcast:
    """When concrete args are passed, Prefect wrapping still applies."""

    def test_task_no_broadcast(self):
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )
        # Pass concrete value, not a distribution — no broadcasting
        result = wf(x=jnp.array(5.0))
        np.testing.assert_allclose(float(result), 6.0)

    def test_flow_no_broadcast(self):
        wf = Function(
            func=double_it,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
        )
        result = wf(x=jnp.array(3.0))
        np.testing.assert_allclose(float(result), 6.0)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestPrefectImportGuard:
    """When Prefect is not installed, workflow_kind should warn and fall back to OFF."""

    def test_task_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
            n_broadcast_samples=10,
        )
        with workflow_run(seed=60), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")

    def test_flow_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
            n_broadcast_samples=10,
        )
        with workflow_run(seed=61), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")

    def test_jax_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="jax",
            n_broadcast_samples=10,
        )
        with workflow_run(seed=62), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert hasattr(result, "samples")
