import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import EmpiricalDistribution, ProvenanceMode

#: The fixtures that skip a test unless BridgeStan is installed; a test requesting one is marked
#: ``stan``, the marker CI's stan job selects.
_STAN_FIXTURES = frozenset({"_stanc", "_stan_toolchain"})


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pending(reason, raises=NotImplementedError, strict=True): a documented contract "
        "whose implementation has not merged. The test xfails strictly, and only on *raises*, "
        "so it fails once the implementation passes it or when it fails for another reason. "
        "strict=False marks a failure that depends on the platform's numerics, such as a "
        "sampler's run at a fixed budget, which passes on some platforms.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _STAN_FIXTURES.intersection(item.fixturenames):
            item.add_marker(pytest.mark.stan)
        for marker in item.iter_markers("pending"):
            reason = marker.kwargs.get("reason") or (marker.args[0] if marker.args else "")
            item.add_marker(
                pytest.mark.xfail(
                    raises=marker.kwargs.get("raises", NotImplementedError),
                    strict=marker.kwargs.get("strict", True),
                    reason=f"pending: {reason}",
                )
            )


@pytest.fixture(autouse=True)
def _reset_provenance_config():
    """Always restore provenance_config and notation_config to defaults after each test.

    Under pytest-xdist, a test that sets a setting and raises before its own
    cleanup would otherwise leak it into subsequent tests on that worker.
    """
    yield
    probpipe.provenance_config.reset()
    probpipe.notation_config.reset()


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop JAX's compiled executables and traces after each test module.

    JAX keeps every executable a process compiles, so an xdist worker's memory
    grows with the number of tests it runs. Over the full suite the two workers
    of a CI runner then exhaust its 16 GB, and the runner shuts the job down.
    Clearing at each module boundary bounds a worker's memory by its largest
    module.
    """
    yield
    jax.clear_caches()


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
