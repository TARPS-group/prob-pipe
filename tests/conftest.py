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