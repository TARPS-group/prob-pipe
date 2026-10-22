"""Tests for Prefect orchestration in Function.

Exercises every way Prefect runs a call's evaluation points:
- workflow_kind=WorkflowKind.TASK, one task per evaluation point of a function
  that claims traceable=False, a named executor that has no effect, and one
  task that maps a traceable function with JAX
- workflow_kind=WorkflowKind.FLOW, one task per evaluation point and one flow
  that maps a traceable function
- Import guard when Prefect is unavailable
- Provenance metadata includes orchestration info

Requires ``prefect>=3`` (installed via ``pip install probpipe[prefect]``).
Uses ``prefect_test_harness()`` for an in-process temporary server.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe.functions._evaluation_keys as evaluation_keys_mod
from probpipe import (
    Distribution,
    DistributionBatch,
    EmpiricalDistribution,
    EvaluationError,
    Function,
    Normal,
    NumericArrayBatch,
    OutputSpec,
    ProvenanceMode,
    ResolutionError,
    WorkflowKind,
    config,
    mean,
    sample,
    workflow_run,
)
from probpipe.functions._context import _active_provenance_mode
from probpipe.operations import OperationRegistry, operation

from ..operations._laws import Sampler

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
    return Normal("x", loc=1.0, scale=0.5)


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
    return sample(Normal("x", loc=0.0, scale=1.0))


_THREADED_DRAW = Function(
    _draw_standard_normal,
    traceable=False,
    executor="thread",
    executor_options={"max_workers": 1},
    label="threaded_draw",
)


def _call_threaded_draw():
    return _THREADED_DRAW()


def _draw_under_nested_seed():
    with workflow_run(seed=91):
        return _draw_standard_normal()


_PREFECT_RETRY_KEY_WORDS: list[tuple[int, int]] = []


def _claim_key_and_fail_once():
    key = evaluation_keys_mod._resolve_automatic_key(
        None,
        evaluation_keys_mod._singleton_effect_plan(
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
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.OFF,
                    traceable=False,
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.OFF,
                    executor="jax",
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.OFF,
                    traceable=False,
                    executor="thread",
                    executor_options={"max_workers": 2},
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.TASK,
                    traceable=False,
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.TASK,
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.TASK,
                    traceable=False,
                    executor="thread",
                    executor_options={"max_workers": 2},
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                True,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.FLOW,
                    traceable=False,
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
            (
                Function(
                    add_one,
                    label="add_one",
                    workflow_kind=WorkflowKind.FLOW,
                    max_evaluations=6,
                    output_spec=OutputSpec(add_one=None),
                ),
                False,
            ),
        )

        samples = []
        for workflow, warns_about_local_controls in workflows:
            warning_context = (
                pytest.warns(UserWarning, match="has no effect: workflow_kind=WorkflowKind.TASK")
                if warns_about_local_controls
                else nullcontext()
            )
            with warning_context, workflow_run(seed=17):
                result = workflow(x=normal_dist)
            samples.append(np.asarray(result.atoms))

        for sample_values in samples[1:]:
            np.testing.assert_array_equal(sample_values, samples[0])

    def test_nested_seed_matches_local_and_real_prefect_for_any_outer_seed(self):
        local = Function(
            _draw_under_nested_seed,
            label="_draw_under_nested_seed",
            workflow_kind=WorkflowKind.OFF,
            traceable=False,
        )
        remote = Function(
            _draw_under_nested_seed,
            label="_draw_under_nested_seed",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
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

    def test_a_task_whose_body_calls_a_threaded_draw_draws_as_it_does_locally(self):
        local = Function(
            _call_threaded_draw,
            workflow_kind=WorkflowKind.OFF,
            traceable=False,
            label="nested_thread_owner",
        )
        remote = Function(
            _call_threaded_draw,
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            label="nested_thread_owner",
        )

        with workflow_run(seed=17):
            local_result = local()
        with workflow_run(seed=17):
            remote_result = remote()

        np.testing.assert_array_equal(
            remote_result,
            local_result,
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
            _claim_key_and_fail_once,
            label="_claim_key_and_fail_once",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
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
# workflow_kind=WorkflowKind.TASK, one task per evaluation point
# ---------------------------------------------------------------------------


class TestPrefectTaskRowWise:
    """Exercises Prefect running one task per evaluation point of an opaque function."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            max_evaluations=30,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=0):
            result = wf(x=normal_dist)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 30

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            max_evaluations=200,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=1):
            result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 + 1.0)
        np.testing.assert_allclose(
            float(jnp.mean(np.asarray(result.atoms))),
            2.0,
            atol=0.15,
        )

    def test_multiple_broadcast_args(self, normal_dist):
        wf = Function(
            sum_xy,
            label="sum_xy",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            max_evaluations=30,
            output_spec=OutputSpec(sum_xy=None),
        )
        d2 = Normal("y", loc=2.0, scale=0.3)
        with workflow_run(seed=2):
            result = wf(x=normal_dist, y=d2)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 30


# ---------------------------------------------------------------------------
# workflow_kind=WorkflowKind.FLOW, one task per evaluation point
# ---------------------------------------------------------------------------


class TestPrefectFlowRowWise:
    """Exercises Prefect running one flow with a task per evaluation point of an opaque function."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = Function(
            double_it,
            label="double_it",
            workflow_kind=WorkflowKind.FLOW,
            traceable=False,
            max_evaluations=25,
            output_spec=OutputSpec(double_it=None),
        )
        with workflow_run(seed=10):
            result = wf(x=normal_dist)
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 25

    def test_output_values_correct(self, normal_dist):
        wf = Function(
            double_it,
            label="double_it",
            workflow_kind=WorkflowKind.FLOW,
            traceable=False,
            max_evaluations=200,
            output_spec=OutputSpec(double_it=None),
        )
        with workflow_run(seed=11):
            result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 * 2)
        np.testing.assert_allclose(
            float(jnp.mean(np.asarray(result.atoms))),
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
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            max_evaluations=20,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=40):
            result = wf(x=normal_dist)
        assert result.provenance is not None
        assert result.provenance.operation == "broadcast"
        assert result.provenance.metadata["orchestrate"] == "task"
        assert result.provenance.metadata["max_evaluations"] == 20

    def test_flow_provenance(self, normal_dist):
        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.FLOW,
            traceable=False,
            max_evaluations=20,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=41):
            result = wf(x=normal_dist)
        assert result.provenance is not None
        assert result.provenance.metadata["orchestrate"] == "flow"

    def test_no_orchestration_provenance(self, normal_dist):
        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.OFF,
            traceable=False,
            max_evaluations=20,
            output_spec=OutputSpec(add_one=None),
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
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            output_spec=OutputSpec(add_one=None),
        )
        # Pass concrete value, not a distribution — no broadcasting
        result = wf(x=jnp.array(5.0))
        np.testing.assert_allclose(float(result), 6.0)

    def test_flow_no_broadcast(self):
        wf = Function(
            double_it,
            label="double_it",
            workflow_kind=WorkflowKind.FLOW,
            traceable=False,
            output_spec=OutputSpec(double_it=None),
        )
        result = wf(x=jnp.array(3.0))
        np.testing.assert_allclose(float(result), 6.0)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestPrefectImportGuard:
    """When Prefect is not installed, workflow_kind should warn and fall back to OFF."""

    def test_task_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.functions._function as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            traceable=False,
            max_evaluations=10,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=60), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert isinstance(result, EmpiricalDistribution)

    def test_flow_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.functions._function as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.FLOW,
            traceable=False,
            max_evaluations=10,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=61), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert isinstance(result, EmpiricalDistribution)

    def test_jax_warns_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.functions._function as node_mod

        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = Function(
            add_one,
            label="add_one",
            workflow_kind=WorkflowKind.TASK,
            max_evaluations=10,
            output_spec=OutputSpec(add_one=None),
        )
        with workflow_run(seed=62), pytest.warns(UserWarning, match="Prefect is not installed"):
            result = wf(x=normal_dist)
        assert isinstance(result, EmpiricalDistribution)


# ---------------------------------------------------------------------------
# Controls a derived route passes to a Prefect constituent
# ---------------------------------------------------------------------------


def _mean_of_law(law: Distribution) -> Any:
    """The raw mean of *law*, a constituent that runs in a Prefect task."""
    return mean.using(raw=True)(law)


class TestInheritedControlsInPrefectTasks:
    """Prefect's in-process task runner keeps the controls a derived route passes down.

    A worker that receives the work function serialized is covered in
    ``tests/operations/test_operation.py``.
    """

    def test_exact_only_holds_in_a_prefect_task_constituent(self):
        inner = Function(_mean_of_law, workflow_kind=WorkflowKind.TASK, traceable=False).using(
            raw=True
        )

        @operation(
            result=lambda d: None, registry=OperationRegistry(), derived_route_check=lambda d: True
        )
        def summed_means(d: Distribution):
            """The sum of the means of two copies of *d*."""
            laws = DistributionBatch([d, d], "laws", label="laws")
            return jnp.sum(jnp.asarray(inner(laws)))

        law = Sampler("s")
        with (
            workflow_run(seed=63),
            pytest.raises(
                EvaluationError, match="mean: no route applies with exact_only"
            ) as raised,
        ):
            summed_means.using(exact_only=True)(law)
        assert isinstance(raised.value.__cause__, ResolutionError)
        assert law.shapes == []


class TestConfigBlocksInPrefectTasks:
    """A ``config.using`` block applies to the work a call inside it runs as Prefect tasks."""

    def test_a_block_sets_the_draws_of_a_summary_in_a_prefect_task(self):
        inner = Function(_mean_of_law, workflow_kind=WorkflowKind.TASK, traceable=False).using(
            raw=True
        )
        first, second = Sampler("s", 1.0), Sampler("s", 2.0)
        with workflow_run(seed=64), config.sampling.using(num_samples=37):
            inner(DistributionBatch([first, second], "laws", label="laws"))
        assert first.shapes == second.shapes == [(37,)]


def _provenance_mode_code(x: jnp.ndarray) -> jnp.ndarray:
    """The position in ``ProvenanceMode`` of the mode the running work is recorded under, plus 0 * x."""
    return x * 0.0 + list(ProvenanceMode).index(_active_provenance_mode())


class TestProvenanceModeInPrefectTasks:
    """A Prefect task runs under the provenance mode of the workflow run that started it.

    A worker that receives the payload pickled is covered in
    ``tests/functions/test_parallel.py``.
    """

    def test_a_prefect_task_runs_under_the_runs_mode(self):
        reporter = Function(
            _provenance_mode_code, workflow_kind=WorkflowKind.TASK, traceable=False
        ).using(raw=True)
        config.provenance.mode = ProvenanceMode.OFF
        with workflow_run(seed=65):
            codes = reporter(NumericArrayBatch(jnp.zeros(2), "x", label="xs"))
        assert np.asarray(codes).tolist() == [list(ProvenanceMode).index(ProvenanceMode.OFF)] * 2
