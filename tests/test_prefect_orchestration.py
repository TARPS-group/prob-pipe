"""Tests for Prefect orchestration in WorkflowFunction.

Exercises all Prefect dispatch paths:
- workflow_kind="task" with loop and JAX vectorization
- workflow_kind="flow" with loop and JAX vectorization
- Import guard when Prefect is unavailable
- Provenance metadata includes orchestration info

Requires ``prefect>=3`` (installed via ``pip install probpipe[prefect]``).
Uses ``prefect_test_harness()`` for an in-process temporary server.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

prefect = pytest.importorskip("prefect")
from prefect.testing.utilities import prefect_test_harness

from probpipe import Normal
from probpipe.core.node import WorkflowFunction


@pytest.fixture(scope="module", autouse=True)
def _prefect_harness():
    """Start a temporary in-process Prefect server for the entire module."""
    with prefect_test_harness():
        yield


@pytest.fixture
def normal_dist():
    return Normal(loc=1.0, scale=0.5)


# ---------------------------------------------------------------------------
# Helper functions for workflows
# ---------------------------------------------------------------------------

def add_one(x: jnp.ndarray) -> jnp.ndarray:
    return x + 1.0


def double_it(x: jnp.ndarray) -> jnp.ndarray:
    return x * 2.0


def sum_xy(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return x + y


# ---------------------------------------------------------------------------
# workflow_kind="task" with loop vectorization
# ---------------------------------------------------------------------------

class TestPrefectTaskLoop:
    """Exercises _execute_many_prefect_task via loop vectorization."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="loop",
            n_broadcast_samples=30,
            seed=0,
        )
        result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.n == 30

    def test_output_values_correct(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="loop",
            n_broadcast_samples=200,
            seed=1,
        )
        result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 + 1.0)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)), 2.0, atol=0.15,
        )

    def test_multiple_broadcast_args(self, normal_dist):
        wf = WorkflowFunction(
            func=sum_xy,
            workflow_kind="task",
            vectorize="loop",
            n_broadcast_samples=30,
            seed=2,
        )
        d2 = Normal(loc=2.0, scale=0.3)
        result = wf(x=normal_dist, y=d2)
        assert hasattr(result, "samples")
        assert result.n == 30


# ---------------------------------------------------------------------------
# workflow_kind="flow" with loop vectorization
# ---------------------------------------------------------------------------

class TestPrefectFlowLoop:
    """Exercises _execute_many_prefect_flow via loop vectorization."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = WorkflowFunction(
            func=double_it,
            workflow_kind="flow",
            vectorize="loop",
            n_broadcast_samples=25,
            seed=10,
        )
        result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.n == 25

    def test_output_values_correct(self, normal_dist):
        wf = WorkflowFunction(
            func=double_it,
            workflow_kind="flow",
            vectorize="loop",
            n_broadcast_samples=200,
            seed=11,
        )
        result = wf(x=normal_dist)
        # Mean should be ~2.0 (1.0 * 2)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)), 2.0, atol=0.15,
        )


# ---------------------------------------------------------------------------
# workflow_kind="task" with JAX vectorization
# ---------------------------------------------------------------------------

class TestPrefectTaskJax:
    """Exercises Prefect-wrapped jax.vmap path in _broadcast_jax."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="jax",
            n_broadcast_samples=30,
            seed=20,
        )
        result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.n == 30

    def test_output_values_correct(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="jax",
            n_broadcast_samples=200,
            seed=21,
        )
        result = wf(x=normal_dist)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)), 2.0, atol=0.15,
        )


# ---------------------------------------------------------------------------
# workflow_kind="flow" with JAX vectorization
# ---------------------------------------------------------------------------

class TestPrefectFlowJax:
    """Exercises Prefect flow-wrapped jax.vmap path."""

    def test_returns_empirical_distribution(self, normal_dist):
        wf = WorkflowFunction(
            func=double_it,
            workflow_kind="flow",
            vectorize="jax",
            n_broadcast_samples=25,
            seed=30,
        )
        result = wf(x=normal_dist)
        assert hasattr(result, "samples")
        assert result.n == 25

    def test_output_values_correct(self, normal_dist):
        wf = WorkflowFunction(
            func=double_it,
            workflow_kind="flow",
            vectorize="jax",
            n_broadcast_samples=200,
            seed=31,
        )
        result = wf(x=normal_dist)
        np.testing.assert_allclose(
            float(jnp.mean(result.samples)), 2.0, atol=0.15,
        )


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

class TestPrefectProvenance:
    """Verify provenance includes orchestration info."""

    def test_task_provenance(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="loop",
            n_broadcast_samples=20,
            seed=40,
        )
        result = wf(x=normal_dist)
        assert result.source is not None
        assert result.source.operation == "broadcast"
        assert result.source.metadata["orchestrate"] == "task"
        assert result.source.metadata["n_samples"] == 20

    def test_flow_provenance(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="flow",
            vectorize="loop",
            n_broadcast_samples=20,
            seed=41,
        )
        result = wf(x=normal_dist)
        assert result.source is not None
        assert result.source.metadata["orchestrate"] == "flow"

    def test_no_orchestration_provenance(self, normal_dist):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind=None,
            vectorize="loop",
            n_broadcast_samples=20,
            seed=42,
        )
        result = wf(x=normal_dist)
        assert result.source is not None
        assert result.source.metadata["orchestrate"] == "none"


# ---------------------------------------------------------------------------
# Non-broadcast calls with workflow_kind
# ---------------------------------------------------------------------------

class TestPrefectNonBroadcast:
    """When concrete args are passed, Prefect wrapping still applies."""

    def test_task_no_broadcast(self):
        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="loop",
            seed=50,
        )
        # Pass concrete value, not a distribution — no broadcasting
        result = wf(x=jnp.array(5.0))
        np.testing.assert_allclose(float(result), 6.0)

    def test_flow_no_broadcast(self):
        wf = WorkflowFunction(
            func=double_it,
            workflow_kind="flow",
            vectorize="loop",
            seed=51,
        )
        result = wf(x=jnp.array(3.0))
        np.testing.assert_allclose(float(result), 6.0)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

class TestPrefectImportGuard:
    """When Prefect is not installed, workflow_kind should raise ImportError."""

    def test_task_raises_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="loop",
            n_broadcast_samples=10,
            seed=60,
        )
        with pytest.raises(ImportError, match="Prefect is required"):
            wf(x=normal_dist)

    def test_flow_raises_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="flow",
            vectorize="loop",
            n_broadcast_samples=10,
            seed=61,
        )
        with pytest.raises(ImportError, match="Prefect is required"):
            wf(x=normal_dist)

    def test_jax_raises_without_prefect(self, normal_dist, monkeypatch):
        import probpipe.core.node as node_mod
        monkeypatch.setattr(node_mod, "task", None)
        monkeypatch.setattr(node_mod, "flow", None)

        wf = WorkflowFunction(
            func=add_one,
            workflow_kind="task",
            vectorize="jax",
            n_broadcast_samples=10,
            seed=62,
        )
        with pytest.raises(ImportError, match="Prefect is required"):
            wf(x=normal_dist)
