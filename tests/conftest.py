import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import EmpiricalDistribution, ProvenanceMode


@pytest.fixture
def full_provenance_mode():
    """Switch to FULL provenance mode and restore the default after the test."""
    probpipe.provenance_config.mode = ProvenanceMode.FULL
    yield
    probpipe.provenance_config.reset()


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


@pytest.fixture(scope="module")
def _stan_toolchain(tmp_path_factory):
    """Skip Stan integration tests unless BridgeStan can compile here.

    Compiling a trivial, data-free probe separates a missing C++ toolchain
    (a legitimate skip) from a real construction failure (a bug that must
    surface, not skip). Shared by the BridgeStan-backed tests in
    tests/modeling/ and tests/inference/.
    """
    bridgestan = pytest.importorskip("bridgestan")
    probe = tmp_path_factory.mktemp("stan_probe") / "probe.stan"
    probe.write_text("parameters { real x; } model { x ~ normal(0, 1); }")
    try:
        bridgestan.StanModel(str(probe))
    except Exception as exc:
        pytest.skip(f"Stan compilation unavailable: {exc}")
