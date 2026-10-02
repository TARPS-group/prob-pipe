import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import EmpiricalDistribution, ProvenanceMode


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pending(reason, raises=NotImplementedError): a documented contract whose "
        "implementation has not merged. The test xfails strictly, and only on *raises*, "
        "so it fails once the implementation passes it or when it fails for another reason.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in item.iter_markers("pending"):
            reason = marker.kwargs.get("reason") or (marker.args[0] if marker.args else "")
            item.add_marker(
                pytest.mark.xfail(
                    raises=marker.kwargs.get("raises", NotImplementedError),
                    strict=True,
                    reason=f"pending: {reason}",
                )
            )


@pytest.fixture(autouse=True)
def _reset_provenance_config():
    """Always restore provenance_config to defaults after each test.

    Under pytest-xdist, a test that sets the mode and raises before its own
    cleanup would otherwise leak the mode into subsequent tests on that worker.
    """
    yield
    probpipe.provenance_config.reset()


@pytest.fixture
def full_provenance_mode():
    """Switch to FULL provenance mode for the duration of a test."""
    probpipe.provenance_config.mode = ProvenanceMode.FULL
    yield


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
    return EmpiricalDistribution("empirical", simple_samples)


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


@pytest.fixture(scope="session")
def _stanc():
    """Skip unless BridgeStan's stanc compiler is here, which a StanModel reads its program with."""
    from tests._stanc import require_stanc

    require_stanc()


@pytest.fixture(scope="module")
def _stan_toolchain(tmp_path_factory):
    """Skip Stan integration tests unless BridgeStan can compile here.

    Compiling a trivial, data-free probe separates a missing C++ toolchain
    (a legitimate skip) from a real construction failure (a bug that must
    surface, not skip). Shared by the BridgeStan-backed tests in
    tests/modeling/ and tests/inference/.
    """
    bridgestan = pytest.importorskip("bridgestan")
    from probpipe.families._programs import _BRIDGESTAN_MAKE_ARGS

    probe = tmp_path_factory.mktemp("stan_probe") / "probe.stan"
    probe.write_text("parameters { real x; } model { x ~ normal(0, 1); }")
    try:
        # Built as the adapter builds model libraries, without TBB's malloc proxy,
        # which would break JAX's allocations for the rest of the process.
        bridgestan.StanModel(str(probe), make_args=list(_BRIDGESTAN_MAKE_ARGS))
    except Exception as exc:
        pytest.skip(f"Stan compilation unavailable: {exc}")
