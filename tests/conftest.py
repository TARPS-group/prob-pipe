
import pytest
import numpy as np
from probpipe.core.distributions import EmpiricalDistribution

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def simple_samples():
    return np.array([[1.0], [2.0], [3.0]])

@pytest.fixture
def empirical(simple_samples, rng):
    return EmpiricalDistribution(simple_samples, rng=rng)

@pytest.fixture
def simple_weights():
    return np.array([0.2, 0.3, 0.5])

@pytest.fixture
def dim():
    return 3

@pytest.fixture
def mean(dim):
    return np.arange(dim, dtype=float)  # [0,1,2]

@pytest.fixture
def cov_matrix(dim):
    A = np.eye(dim) * 2.0
    A[0,1] = A[1,0] = 0.3
    return A
