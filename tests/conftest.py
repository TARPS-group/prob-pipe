import pytest
import jax
import jax.numpy as jnp
import numpy as np

from probpipe import EmpiricalDistribution, WorkflowKind, prefect_config


@pytest.fixture(autouse=True)
def _disable_prefect_for_tests():
    """Force ``WorkflowKind.OFF`` globally for each test.

    The default ``WorkflowKind.DEFAULT`` auto-promotes to ``TASK`` when
    Prefect is importable (see ``WorkflowFunction.effective_workflow_kind``).
    That auto-promotion picks up whatever ``PREFECT_API_URL`` the user
    has in ``~/.prefect/profiles.toml`` — if the configured server isn't
    running, every ``WorkflowFunction`` call raises
    ``RuntimeError: Failed to reach API at ...``. CI happens to escape
    because it has no profile and falls back to ephemeral mode, but
    contributors with an existing Prefect setup hit it immediately.

    Function-scoped (not session-scoped) so the OFF state is restored
    around every test — otherwise other test classes that flip the
    global (``test_prefect_config.py`` resets to ``DEFAULT`` between
    its tests) leak that state into later tests.

    The Prefect-orchestration tests in ``test_prefect_orchestration.py``
    don't depend on the global config — they pass ``workflow_kind="task"``
    or ``"flow"`` explicitly, which is a per-instance override that
    bypasses the global, so this fixture doesn't interfere with them.
    The ``test_prefect_config.py`` tests have their own function-scoped
    autouse fixture that resets to ``DEFAULT``; pytest runs the
    conftest fixture first and the test-class fixture second, so those
    tests still see ``DEFAULT`` when they expect to.
    """
    saved = prefect_config.workflow_kind
    prefect_config.workflow_kind = WorkflowKind.OFF
    try:
        yield
    finally:
        prefect_config.workflow_kind = saved


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng():
    """Legacy numpy RNG for tests that still need it."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_samples():
    return jnp.array([[1.0], [2.0], [3.0]])


@pytest.fixture
def empirical(simple_samples, key):
    return EmpiricalDistribution(simple_samples)


@pytest.fixture
def simple_weights():
    return jnp.array([0.2, 0.3, 0.5])


@pytest.fixture
def dim():
    return 3


@pytest.fixture
def loc(dim):
    # Use JAX's default float dtype so the fixture stays consistent with
    # the cov_matrix fixture (which also uses defaults) under x64 mode.
    return jnp.arange(dim, dtype=float)


@pytest.fixture
def cov_matrix(dim):
    A = jnp.eye(dim) * 2.0
    A = A.at[0, 1].set(0.3)
    A = A.at[1, 0].set(0.3)
    return A
