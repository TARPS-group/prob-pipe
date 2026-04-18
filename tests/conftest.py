import pytest
import jax
import jax.numpy as jnp
import numpy as np

from probpipe import EmpiricalDistribution


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
    return jnp.arange(dim, dtype=jnp.float32)


@pytest.fixture
def cov_matrix(dim):
    A = jnp.eye(dim) * 2.0
    A = A.at[0, 1].set(0.3)
    A = A.at[1, 0].set(0.3)
    return A
